"""
Backbone Module: DINOv2-ViT-B/14 with LoRA adapters + ResNet-18 fallback

Implements:
  - DINOv2Backbone: DINOv2-ViT-B/14 with optional LoRA fine-tuning
  - LegacyResNet18Backbone: Fallback ResNet-18 backbone
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


class LoRAAdapter(nn.Module):
    """
    Low-rank adapter (LoRA) for efficient fine-tuning.

    Adds two learnable linear layers with low rank:
      LoRA output: B(A(x)) * scaling where scaling = lora_alpha / rank

    Added to frozen pretrained weights to enable parameter-efficient adaptation.
    """

    def __init__(self, d_model: int, rank: int = 8, lora_alpha: float = 16.0):
        """
        Initialize LoRA adapter.

        Args:
            d_model: Feature dimension (e.g., 768 for DINOv2-ViT-B)
            rank: Low-rank dimension (default 8)
            lora_alpha: Scaling factor (default 16.0)
        """
        super().__init__()
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / rank

        # A: project down to rank
        self.lora_A = nn.Linear(d_model, rank, bias=False)
        # B: project back to d_model
        self.lora_B = nn.Linear(rank, d_model, bias=False)

        # Initialize A with Kaiming normal, B with zeros
        nn.init.kaiming_normal_(self.lora_A.weight, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: x + B(A(x)) * scaling

        Args:
            x: (... , d_model)

        Returns:
            LoRA output: (... , d_model)
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
    ):
        """
        Initialize DINOv2 backbone with optional LoRA.

        Args:
            use_lora: Enable LoRA adapters (default True)
            lora_rank: LoRA rank (default 8)
            lora_alpha: LoRA scaling factor (default 16.0)
            freeze_base: Freeze base model weights (default True)
        """
        super().__init__()
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.d_model = 768  # DINOv2-ViT-B feature dim

        # Attempt to load DINOv2 via torch.hub
        self.model = self._load_dinov2()

        if self.model is None:
            warnings.warn(
                "Failed to load DINOv2 via torch.hub. Using LegacyResNet18Backbone fallback. "
                "This may impact model performance."
            )
            self.use_fallback = True
            self.model = None  # Will be handled by forward()
        else:
            self.use_fallback = False

            # Freeze all base model parameters
            if freeze_base:
                self.freeze_base()

            # Inject LoRA adapters if enabled
            if use_lora:
                self._inject_lora_adapters()

    def _load_dinov2(self) -> Optional[nn.Module]:
        """
        Load DINOv2-ViT-B/14 from torch.hub or timm.

        Returns:
            Loaded model or None if both fail.
        """
        # Try torch.hub first
        try:
            model = torch.hub.load(
                'facebookresearch/dinov2',
                'dinov2_vitb14',
                pretrained=True
            )
            return model
        except Exception as e:
            warnings.warn(f"torch.hub load failed: {e}. Trying timm fallback...")

        # Fallback to timm
        try:
            import timm
            model = timm.create_model(
                'vit_base_patch14_dinov2.lvd142m',
                pretrained=False
            )
            return model
        except Exception as e:
            warnings.warn(f"timm fallback failed: {e}. Will use ResNet18 fallback.")
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

                    # Create LoRA adapters for Q and V
                    q_adapter = LoRAAdapter(
                        self.d_model,
                        rank=self.lora_rank,
                        lora_alpha=self.lora_alpha
                    )
                    v_adapter = LoRAAdapter(
                        self.d_model,
                        rank=self.lora_rank,
                        lora_alpha=self.lora_alpha
                    )

                    self.lora_adapters[f'block_{block_idx}_q'] = q_adapter
                    self.lora_adapters[f'block_{block_idx}_v'] = v_adapter

                    # Wrap original q_proj and v_proj to add LoRA output
                    self._wrap_attn_projection(block, attn, block_idx)

    def _wrap_attn_projection(self, block, attn, block_idx: int):
        """
        Wrap attention Q/V projections to apply LoRA.

        Args:
            block: Transformer block
            attn: Attention module
            block_idx: Block index (for adapter lookup)
        """
        # Store original projections
        original_q_proj = attn.q_proj
        original_v_proj = attn.v_proj

        q_adapter = self.lora_adapters[f'block_{block_idx}_q']
        v_adapter = self.lora_adapters[f'block_{block_idx}_v']

        def q_proj_with_lora(x):
            return original_q_proj(x) + q_adapter(x)

        def v_proj_with_lora(x):
            return original_v_proj(x) + v_adapter(x)

        attn.q_proj = q_proj_with_lora
        attn.v_proj = v_proj_with_lora

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
        # Validate DINOv2 input resolution alignment (must be multiples of 14)
        if not self.use_fallback:
            validate_dinov2_input(x)

        if self.use_fallback:
            # Use ResNet18 fallback
            from torchvision.models import resnet18
            if not hasattr(self, '_fallback_resnet'):
                self._fallback_resnet = resnet18(pretrained=True)
                # Remove classification head
                self._fallback_resnet = nn.Sequential(
                    *list(self._fallback_resnet.children())[:-1]
                )
                self._fallback_resnet.eval()
                for param in self._fallback_resnet.parameters():
                    param.requires_grad = False
                self._fallback_resnet = self._fallback_resnet.to(x.device)

            # Global average pooling output
            with torch.no_grad():
                features = self._fallback_resnet(x)
            return features.view(features.size(0), -1)

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


class LegacyResNet18Backbone(nn.Module):
    """
    ResNet-18 backbone for spatial feature extraction (legacy/fallback).

    Takes (B, 3, 224, 224) input and outputs (B, 512) features.
    Used as fallback when DINOv2 fails to load.
    """

    def __init__(self, pretrained: bool = True, freeze_early: bool = True):
        """
        Initialize ResNet-18 backbone.

        Args:
            pretrained: Load ImageNet pretrained weights (default True)
            freeze_early: Freeze layer1, layer2 for training stability (default True)
        """
        super().__init__()

        from torchvision.models import resnet18, ResNet18_Weights

        # Load ResNet-18 — gracefully fall back to random weights if download blocked
        if pretrained:
            try:
                self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                _logger.info("Loaded ImageNet pretrained ResNet-18 weights")
            except Exception as e:
                _logger.warning(
                    f"Could not download pretrained weights ({e}). "
                    "Training from random initialisation."
                )
                self.model = resnet18(weights=None)
        else:
            self.model = resnet18(weights=None)

        # Remove classification head (replace with identity)
        self.model.fc = nn.Identity()

        # Optionally freeze early layers
        if freeze_early:
            self._freeze_layers(max_layer=2)

    def _freeze_layers(self, max_layer: int = 2):
        """
        Freeze layers up to and including max_layer.

        Args:
            max_layer: Maximum layer number to freeze (1, 2, 3, or 4)
        """
        # Freeze conv1 and bn1
        for param in self.model.conv1.parameters():
            param.requires_grad = False
        for param in self.model.bn1.parameters():
            param.requires_grad = False

        # Freeze specified layers
        for layer_num in range(1, max_layer + 1):
            layer = getattr(self.model, f'layer{layer_num}')
            for param in layer.parameters():
                param.requires_grad = False

    def unfreeze_layers(self, start_layer: int = 2):
        """
        Unfreeze layers from start_layer onwards for fine-tuning.

        Args:
            start_layer: Layer number to start unfreezing (1, 2, 3, or 4)
        """
        for layer_num in range(start_layer, 5):
            layer = getattr(self.model, f'layer{layer_num}')
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ResNet-18.

        Args:
            x: (B, 3, 224, 224) input tensor

        Returns:
            features: (B, 512) feature vector
        """
        # Pass through conv1, bn1, relu, maxpool
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        # Pass through residual layers
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        # Global average pooling
        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)

        return x
