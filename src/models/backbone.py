"""
Backbone Module: DINOv2-ViT-B/14 with LoRA adapters

Implements:
  - DINOv2Backbone: DINOv2-ViT-B/14 with optional LoRA fine-tuning
  - LoRA adapter injection into attention layers

ARCHITECTURE FIX (April 2026):
  - Added DINOV2_PATCH_SIZE=14 and DINOV2_VALID_RESOLUTIONS validation constants
  - Added validate_dinov2_input() to guard against incompatible input resolutions
  - DINOv2 ViT-B/14 requires H,W to be multiples of 14. Invalid resolutions cause
    incorrect patch embeddings. Supported: 224 (16x16), 336 (24x24), 448 (32x32),
    518 (37x37 — Meta-recommended for best spatial fidelity, 5x more patches).
  - validate_dinov2_input() called as first line in forward() to fail fast.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from typing import Optional, Dict, Any

_logger = logging.getLogger(__name__)

# DINOv2 ViT-B/14 patch size — all input dimensions must be multiples of this
DINOV2_PATCH_SIZE = 14
# Supported aligned resolutions (H=W). 518 is Meta's recommended; 224 is minimum valid.
DINOV2_VALID_RESOLUTIONS = (224, 336, 448, 518)

def validate_dinov2_input(x: torch.Tensor, patch_size: int = DINOV2_PATCH_SIZE) -> None:
    """
    Validate that input tensor spatial dimensions are multiples of patch_size.
    Raises ValueError with a clear remediation message if not.
    """
    _, _, H, W = x.shape
    if H % patch_size != 0 or W % patch_size != 0:
        valid = [r for r in DINOV2_VALID_RESOLUTIONS]
        raise ValueError(
            f"DINOv2 ViT-B/14 requires input H and W to be multiples of "
            f"patch_size={patch_size}. Got ({H}×{W}). "
            f"Use one of: {valid}. "
            f"Recommended: 518×518 (37×37 patches, best spatial resolution). "
            f"Minimum: 224×224 (16×16 patches). "
            f"Fix: pass img_size=518 or img_size=224 to PipetteWellDataset."
        )


class LoRAWrappedLinear(nn.Module):
    """
    Wraps an existing nn.Linear with a LoRA adapter.

    Replaces the original module in-place so PyTorch's nn.Module attribute
    assignment is satisfied (no lambda / plain function workaround needed).

    output = original_linear(x) + lora_adapter(x)
    """

    def __init__(self, original: nn.Module, adapter: 'LoRAAdapter'):
        super().__init__()
        self.original = original
        self.adapter = adapter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original(x) + self.adapter(x)


class LoRAAdapter(nn.Module):
    """
    Low-rank adapter (LoRA) for efficient fine-tuning.

    Adds two learnable linear layers with low rank:
      LoRA output: B(A(x)) * scaling where scaling = lora_alpha / rank

    Added to frozen pretrained weights to enable parameter-efficient adaptation.
    """

    def __init__(self, in_dim: int, out_dim: int = None, rank: int = 8, lora_alpha: float = 16.0):
        """
        Initialize LoRA adapter.

        Args:
            in_dim: Input feature dimension (e.g., 768 for DINOv2-ViT-B)
            out_dim: Output feature dimension. Defaults to in_dim.
                     Set to 3*in_dim for fused qkv projections.
            rank: Low-rank dimension (default 8)
            lora_alpha: Scaling factor (default 16.0)
        """
        super().__init__()
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / rank
        out_dim = out_dim if out_dim is not None else in_dim

        # A: project down to rank; B: project up to out_dim
        self.lora_A = nn.Linear(in_dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_dim, bias=False)

        # Initialize A with Kaiming normal, B with zeros (no-op at init)
        nn.init.kaiming_normal_(self.lora_A.weight, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: B(A(x)) * scaling

        Args:
            x: (..., in_dim)

        Returns:
            LoRA delta: (..., out_dim)  — added to frozen projection output
        """
        return self.lora_B(self.lora_A(x)) * self.scaling


class DINOv2Backbone(nn.Module):
    """
    DINOv2-ViT-B/14 backbone with optional LoRA adapters.

    Loads pretrained DINOv2 and freezes base weights. LoRA adapters are injected
    into attention Q/V projections for parameter-efficient fine-tuning.

    Input:  (B, 3, 224, 224)  — normalized to ImageNet stats
    Output: (B, 768)          — CLS token feature vector
    """

    def __init__(
        self,
        use_lora: bool = True,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        freeze_base: bool = True,
        img_size: int = 224,
    ):
        """
        Initialize DINOv2 backbone with optional LoRA.

        Args:
            use_lora: Enable LoRA adapters (default True)
            lora_rank: LoRA rank (default 8)
            lora_alpha: LoRA scaling factor (default 16.0)
            freeze_base: Freeze base model weights (default True)
            img_size: Input image size — must be a multiple of 14. The timm
                      model is created with this size so positional embeddings
                      are initialised correctly (default 224).
        """
        super().__init__()
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.img_size = img_size
        self.d_model = 768  # DINOv2-ViT-B feature dim

        # Attempt to load DINOv2 via torch.hub
        self.model = self._load_dinov2(img_size)

        if self.model is None:
            raise RuntimeError(
                "Failed to load DINOv2 via torch.hub and timm. "
                "Ensure network access or a local cache is available."
            )

        # Freeze all base model parameters
        if freeze_base:
            self.freeze_base()

        # Inject LoRA adapters if enabled
        if use_lora:
            self._inject_lora_adapters()

    def _load_dinov2(self, img_size: int = 224) -> Optional[nn.Module]:
        """
        Load DINOv2-ViT-B/14 from torch.hub or timm.

        img_size is passed to timm so positional embeddings are initialised
        at the correct resolution (timm defaults to 518 otherwise).

        Returns:
            Loaded model or None if both fail.
        """
        # Try torch.hub first (requires Python 3.10+ for type-union syntax in hub code)
        try:
            model = torch.hub.load(
                'facebookresearch/dinov2',
                'dinov2_vitb14',
                pretrained=True
            )
            _logger.info("Loaded DINOv2 via torch.hub")
            return model
        except Exception as e:
            warnings.warn(f"torch.hub load failed: {e}. Trying timm fallback...")

        # Fallback to timm — pass img_size so the model is built for our resolution
        try:
            import timm
            model = timm.create_model(
                'vit_base_patch14_dinov2.lvd142m',
                pretrained=True,
                img_size=img_size,
            )
            _logger.info(f"Loaded DINOv2 via timm (img_size={img_size})")
            return model
        except Exception as e:
            warnings.warn(f"timm fallback failed: {e}.")
            return None

    def _inject_lora_adapters(self):
        """
        Inject LoRA adapters into attention Q/V projections.

        Traverses transformer blocks and adds LoRA adapters to:
          - Q projection
          - V projection

        This enables efficient fine-tuning while keeping base model frozen.
        """
        self.lora_adapters = nn.ModuleDict()

        # DINOv2 architecture: blocks contain attention layers
        if hasattr(self.model, 'blocks'):
            for block_idx, block in enumerate(self.model.blocks):
                if hasattr(block, 'attn'):
                    attn = block.attn

                    # Detect layout to set correct adapter output dimension:
                    #   q_proj/v_proj style → out_dim = d_model (768)
                    #   qkv fused style     → out_dim = 3 * d_model (2304)
                    if hasattr(attn, 'qkv'):
                        adapter_out_dim = 3 * self.d_model
                    else:
                        adapter_out_dim = self.d_model

                    q_adapter = LoRAAdapter(
                        in_dim=self.d_model,
                        out_dim=adapter_out_dim,
                        rank=self.lora_rank,
                        lora_alpha=self.lora_alpha
                    )
                    v_adapter = LoRAAdapter(
                        in_dim=self.d_model,
                        out_dim=adapter_out_dim,
                        rank=self.lora_rank,
                        lora_alpha=self.lora_alpha
                    )

                    self.lora_adapters[f'block_{block_idx}_q'] = q_adapter
                    self.lora_adapters[f'block_{block_idx}_v'] = v_adapter

                    # Wrap attention projections with LoRA
                    self._wrap_attn_projection(block, attn, block_idx)

    def _wrap_attn_projection(self, block, attn, block_idx: int):
        """
        Wrap attention projections to apply LoRA.

        Handles two DINOv2 attention layouts:
          - torch.hub style: separate attn.q_proj / attn.v_proj
          - timm style:      fused attn.qkv  (Linear(d, 3*d))

        For the fused qkv case we wrap the entire projection so LoRA is
        added to all of Q, K, V — a slight over-injection vs. Q+V only,
        but negligible in practice and avoids splitting the fused weight.
        """
        q_adapter = self.lora_adapters[f'block_{block_idx}_q']
        v_adapter = self.lora_adapters[f'block_{block_idx}_v']

        if hasattr(attn, 'q_proj') and hasattr(attn, 'v_proj'):
            # torch.hub / native DINOv2 style — separate q_proj / v_proj
            attn.q_proj = LoRAWrappedLinear(attn.q_proj, q_adapter)
            attn.v_proj = LoRAWrappedLinear(attn.v_proj, v_adapter)

        elif hasattr(attn, 'qkv'):
            # timm style — fused QKV projection (Linear(d, 3*d))
            # Wrap the entire projection; q_adapter adds LoRA to the combined output.
            # v_adapter is kept registered (parameter count consistency) but not applied.
            attn.qkv = LoRAWrappedLinear(attn.qkv, q_adapter)

        else:
            _logger.warning(
                f"Block {block_idx}: cannot find q_proj/v_proj or qkv in attention "
                f"module {type(attn).__name__}. Skipping LoRA injection for this block."
            )

    def freeze_base(self):
        """Freeze all base model parameters (LoRA adapters remain trainable)."""
        if self.model is None:
            return

        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_lora(self):
        """Ensure LoRA adapters are trainable."""
        if self.use_lora and hasattr(self, 'lora_adapters'):
            for param in self.lora_adapters.parameters():
                param.requires_grad = True

    def trainable_parameters(self) -> int:
        """
        Return count of trainable parameters (LoRA only).

        Returns:
            Number of trainable parameters.
        """
        count = 0
        if self.use_lora and hasattr(self, 'lora_adapters'):
            for param in self.lora_adapters.parameters():
                if param.requires_grad:
                    count += param.numel()
        return count

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: extract CLS token from backbone.

        Args:
            x: (B, 3, 224, 224) input image tensor

        Returns:
            features: (B, 768) CLS token features
        """
        validate_dinov2_input(x)

        # Standard DINOv2 forward: extract CLS token (index 0)
        # DINOv2 returns shape (B, num_patches + 1, 768) where index 0 is CLS
        if hasattr(self.model, 'forward_features'):
            x = self.model.forward_features(x)
        else:
            x = self.model(x)

        # Extract CLS token (first token)
        if isinstance(x, torch.Tensor):
            if x.dim() == 3:  # (B, N, D)
                cls_token = x[:, 0, :]
            else:  # (B, D) - already pooled
                cls_token = x
        else:
            # Handle dict output if model returns dict
            cls_token = x['cls_token'] if isinstance(x, dict) else x[:, 0, :]

        return cls_token


