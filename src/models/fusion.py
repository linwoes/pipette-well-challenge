"""
Fusion Module: Temporal attention + late fusion + factorized output heads

Implements:
  - TemporalAttention: Transformer over ordered frames per view
  - DualViewFusion: Complete model with shared backbone and dual-view fusion
  - WellDetectionLoss: Combined focal loss for row and column heads

ARCHITECTURE FIX (April 2026):
  - Added img_size parameter to DualViewFusion.__init__()
  - Validates DINOv2 resolution alignment (multiples of 14) in __init__
  - Auto-snaps invalid sizes and logs warning. Supported: 224, 336, 448, 518.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Tuple, Optional
from .backbone import DINOv2Backbone

_logger = logging.getLogger(__name__)


class TemporalAttention(nn.Module):
    """
    Temporal Transformer over N ordered frames from a single view.

    Learns temporal relationships across frame sequence using multi-head
    self-attention with learnable positional embeddings.

    Input:  (B, N, D)  — B batch, N frames, D features (768 for DINOv2)
    Output: (B, D)     — temporally pooled representation
    """

    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_frames: int = 8,
    ):
        """
        Initialize temporal attention module.

        Args:
            d_model: Feature dimension (default 768)
            nhead: Number of attention heads (default 8)
            num_layers: Number of transformer layers (default 2)
            dim_feedforward: Feedforward dimension (default 2048)
            dropout: Dropout rate (default 0.1)
            max_frames: Maximum sequence length for position embeddings (default 8)
        """
        super().__init__()
        self.d_model = d_model
        self.max_frames = max_frames

        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, max_frames, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: apply temporal attention and pool.

        Args:
            x: (B, N, D) frame features

        Returns:
            pooled: (B, D) temporally aggregated features
        """
        B, N, D = x.shape

        # Add positional embeddings
        pos = self.pos_embed[:, :N, :]  # (1, N, D)
        x = x + pos  # (B, N, D)

        # Apply transformer
        x = self.transformer(x)  # (B, N, D)

        # Mean pooling over time dimension
        pooled = x.mean(dim=1)  # (B, D)

        return pooled


class DualViewFusion(nn.Module):
    """
    Full pipette well detection model with dual-view late fusion.

    Pipeline:
      FPV frames    [B, N, 3, 224, 224]  → per-frame DINOv2 → [B, N, 768]
                                          → TemporalAttention → [B, 768]
                                                                      ↓
                                                               Concat → [B, 1536]
                                                                      ↓
                                                            Fusion MLP → [B, 256]
                                                                      ↓
                                               ┌──────────────────────┴──────────────────────┐
                                               ↓                                             ↓
                                      Row head: FC(256→8)                       Col head: FC(256→12)

      TopView frames [B, N, 3, 224, 224] → same pipeline →  [B, 768]

    Returns raw logits (no sigmoid) — sigmoid applied in loss / postprocessing
    """

    def __init__(
        self,
        num_rows: int = 8,
        num_columns: int = 12,
        shared_backbone: bool = True,
        use_lora: bool = True,
        lora_rank: int = 8,
        temporal_dim: int = 768,
        temporal_nhead: int = 8,
        temporal_layers: int = 2,
        fusion_hidden_dim: int = 512,
        output_dim: int = 256,
        dropout: float = 0.3,
        max_frames: int = 8,
        img_size: int = 224,
        use_dinov2: bool = True,
    ):
        """
        Initialize DualViewFusion model.

        Args:
            num_rows: Number of rows (default 8)
            num_columns: Number of columns (default 12)
            shared_backbone: Share backbone weights between views (default True)
            use_lora: Enable LoRA fine-tuning (default True)
            lora_rank: LoRA rank (default 8)
            temporal_dim: Temporal attention feature dimension (default 768)
            temporal_nhead: Temporal attention heads (default 8)
            temporal_layers: Temporal transformer layers (default 2)
            fusion_hidden_dim: Fusion MLP hidden dimension (default 512)
            output_dim: Output dimension before heads (default 256)
            dropout: Dropout rate (default 0.3)
            max_frames: Maximum frame sequence length (default 8)
            img_size: Input image size (default 224). For DINOv2, must be multiple of 14.
            use_dinov2: Use DINOv2 backbone if True, ResNet18 if False (default True)
        """
        super().__init__()

        self.num_rows = num_rows
        self.num_columns = num_columns
        self.shared_backbone = shared_backbone
        self.temporal_dim = temporal_dim
        self.use_dinov2 = use_dinov2

        # Validate DINOv2 resolution alignment
        if use_dinov2:
            from .backbone import DINOV2_PATCH_SIZE
            from src.preprocessing.video_loader import snap_to_dinov2_resolution
            snapped = snap_to_dinov2_resolution(img_size)
            if snapped != img_size:
                _logger.warning(
                    f"img_size={img_size} adjusted to {snapped} for DINOv2 patch alignment."
                )
            self.img_size = snapped
        else:
            self.img_size = img_size

        # Initialize backbones
        if use_dinov2:
            self.backbone_fpv = DINOv2Backbone(
                use_lora=use_lora,
                lora_rank=lora_rank,
                freeze_base=True,
                img_size=self.img_size,
            )
            if not shared_backbone:
                self.backbone_topview = DINOv2Backbone(
                    use_lora=use_lora,
                    lora_rank=lora_rank,
                    freeze_base=True,
                    img_size=self.img_size,
                )
            else:
                self.backbone_topview = None
        else:
            # Use ResNet18 fallback for CPU training
            from .backbone import LegacyResNet18Backbone
            self.backbone_fpv = LegacyResNet18Backbone(pretrained=True, freeze_early=True)
            # ResNet18 outputs 512 dims, need to adjust temporal_dim
            self.temporal_dim = 512
            temporal_dim = 512
            # ResNet18 with 512 dims needs nhead divisible by 8
            temporal_nhead = 8  # 512 / 8 = 64

            if not shared_backbone:
                self.backbone_topview = LegacyResNet18Backbone(pretrained=True, freeze_early=True)
            else:
                self.backbone_topview = None

        # L-2: Dimension guard — verify backbone output dim matches temporal_dim
        # before constructing the Temporal Attention modules.
        # A mismatch (e.g. loading a DINOv2 checkpoint into ResNet18 or vice-versa)
        # causes a silent shape error that is hard to debug at inference time.
        if use_dinov2:
            backbone_output_dim = 768  # DINOv2-ViT-B/14 CLS token
        else:
            backbone_output_dim = 512  # ResNet-18 global avg pool
        assert backbone_output_dim == temporal_dim, (
            f"Backbone output dim ({backbone_output_dim}) != temporal_dim ({temporal_dim}). "
            f"Use temporal_dim={backbone_output_dim} when use_dinov2={use_dinov2}."
        )

        # Temporal attention for each view
        self.temporal_fpv = TemporalAttention(
            d_model=temporal_dim,
            nhead=temporal_nhead,
            num_layers=temporal_layers,
            dropout=dropout,
            max_frames=max_frames
        )

        self.temporal_topview = TemporalAttention(
            d_model=temporal_dim,
            nhead=temporal_nhead,
            num_layers=temporal_layers,
            dropout=dropout,
            max_frames=max_frames
        )

        # Fusion MLP: concatenated features → shared representation
        # Input: 2 * temporal_dim (FPV + TopView concatenated)
        fusion_input_dim = 2 * temporal_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Factorized output heads (no activation — raw logits)
        self.row_head = nn.Linear(output_dim, num_rows)
        self.col_head = nn.Linear(output_dim, num_columns)

    def forward(
        self,
        fpv_frames: torch.Tensor,
        topview_frames: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through dual-view fusion model.

        Args:
            fpv_frames: (B, N, 3, 224, 224) FPV video frames
            topview_frames: (B, N, 3, 224, 224) top-view video frames

        Returns:
            row_logits: (B, 8) raw row predictions
            col_logits: (B, 12) raw column predictions
        """
        B, N, C, H, W = fpv_frames.shape

        # Process FPV frames through backbone
        # LoRA adapters are trainable — do NOT wrap in no_grad so they receive gradients.
        # The frozen base weights remain non-differentiable via requires_grad=False.
        fpv_flat = fpv_frames.reshape(B * N, C, H, W)  # (B*N, 3, H, W)
        fpv_features = self.backbone_fpv(fpv_flat)      # (B*N, 768)
        fpv_features = fpv_features.reshape(B, N, -1)  # (B, N, 768)

        # Process TopView frames through backbone
        topview_flat = topview_frames.reshape(B * N, C, H, W)  # (B*N, 3, H, W)
        if self.shared_backbone:
            topview_features = self.backbone_fpv(topview_flat)      # (B*N, 768)
        else:
            topview_features = self.backbone_topview(topview_flat)  # (B*N, 768)
        topview_features = topview_features.reshape(B, N, -1)  # (B, N, 768)

        # Apply temporal attention to each view
        fpv_temporal = self.temporal_fpv(fpv_features)  # (B, 768)
        topview_temporal = self.temporal_topview(topview_features)  # (B, 768)

        # Late fusion: concatenate temporal outputs
        fused = torch.cat([fpv_temporal, topview_temporal], dim=1)  # (B, 1536)

        # Apply fusion MLP
        shared_repr = self.fusion_mlp(fused)  # (B, 256)

        # Factorized heads: output raw logits
        row_logits = self.row_head(shared_repr)  # (B, 8)
        col_logits = self.col_head(shared_repr)  # (B, 12)

        return row_logits, col_logits


class WellDetectionLoss(nn.Module):
    """
    Combined focal loss for row and column well detection heads.

    Focal loss downweights easy examples and focuses on hard negatives:
      FL(p) = -alpha * (1 - p)^gamma * log(p) for positive class
      FL(p) = -(1 - alpha) * p^gamma * log(1 - p) for negative class

    Default: gamma=2.0, alpha=0.75 (tuned for 96-well plate class imbalance ~96% negatives)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.75,
        row_weight: float = 1.0,
        col_weight: float = 1.0,
        well_consistency_weight: float = 0.5,
    ):
        """
        Initialize focal loss with optional well-level consistency term.

        Args:
            gamma: Focusing parameter (default 2.0)
            alpha: Weighting factor for positive class (default 0.75). Set > 0.5 to
                upweight the rare positive wells in a 96-well plate (~96% negatives).
            row_weight: Weight for row loss (default 1.0)
            col_weight: Weight for column loss (default 1.0)
            well_consistency_weight: Weight for the outer-product well-level
                BCE loss that bridges the row/col factor gap (default 0.5).
                Set to 0.0 to disable.
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.row_weight = row_weight
        self.col_weight = col_weight
        self.well_consistency_weight = well_consistency_weight

    def forward(
        self,
        row_logits: torch.Tensor,
        col_logits: torch.Tensor,
        row_targets: torch.Tensor,
        col_targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss for row and column predictions.

        The loss has three components:
          1. Row focal loss — per-row binary classification
          2. Column focal loss — per-column binary classification
          3. Well consistency loss — BCE on the 8×12 outer product of
             row/col probabilities vs the outer product of row/col targets.
             This provides a direct gradient signal toward the Cartesian
             product that exact_match and Jaccard actually measure.

        Args:
            row_logits: (B, 8) raw row predictions
            col_logits: (B, 12) raw column predictions
            row_targets: (B, 8) binary row targets
            col_targets: (B, 12) binary column targets

        Returns:
            loss: Scalar combined loss
        """
        # Compute focal loss for rows
        row_loss = self._focal_loss(row_logits, row_targets)

        # Compute focal loss for columns
        col_loss = self._focal_loss(col_logits, col_targets)

        # Weighted combination of factored losses
        total_loss = self.row_weight * row_loss + self.col_weight * col_loss

        # Well-level consistency loss: outer product of probabilities.
        # This loss is only valid when the ground truth well set is exactly
        # the Cartesian product of its row and column marginals (rectangular
        # patterns: single wells, full rows, full columns). For scatter
        # patterns (e.g., {A1, C3, F7}), the outer product generates phantom
        # wells not in ground truth, producing incorrect supervision.
        # Per-sample mask ensures the loss is skipped for non-rectangular GT.
        if self.well_consistency_weight > 0:
            row_probs = torch.sigmoid(row_logits)   # (B, 8)
            col_probs = torch.sigmoid(col_logits)   # (B, 12)

            # Predicted well probabilities: P(well_ij) = P(row_i) * P(col_j)
            pred_wells = torch.bmm(
                row_probs.unsqueeze(2),   # (B, 8, 1)
                col_probs.unsqueeze(1),   # (B, 1, 12)
            )  # (B, 8, 12)

            # Ground truth well grid: GT(well_ij) = GT(row_i) * GT(col_j)
            gt_cart = torch.bmm(
                row_targets.unsqueeze(2).float(),  # (B, 8, 1)
                col_targets.unsqueeze(1).float(),  # (B, 1, 12)
            )  # (B, 8, 12) — Cartesian product of row/col marginals

            # Per-sample rectangularity mask: 1 where GT == Cartesian product.
            # If a sample's well pattern is not rectangular, its consistency
            # loss contribution is zeroed out to avoid incorrect supervision.
            #
            # For a rectangular pattern (GT_well_ij == row_i AND col_j for all
            # active wells) the Cartesian product exactly equals the true GT
            # and the mask is 1. For scatter patterns (e.g. {A1,C3}) the
            # Cartesian product adds phantom wells (A3, C1) and the mask is 0.
            is_rectangular = (gt_cart == gt_cart).all(dim=(1, 2))  # placeholder shape
            # Compute actual GT well grid from labels (if passed directly, use gt_cart)
            # Check: does the Cartesian product match an independent well-level GT?
            # Since we only have row/col targets (not per-well GT), we detect
            # non-rectangular patterns by checking if |active rows| * |active cols|
            # equals |active wells via OR-marginals| (which equals total active Cartesian cells).
            active_rows = row_targets.sum(dim=1)      # (B,) — number of active rows
            active_cols = col_targets.sum(dim=1)      # (B,) — number of active columns
            cart_count = active_rows * active_cols    # (B,) — cells in Cartesian product

            # Heuristic: if cart_count > 1 and neither a full row nor full col
            # pattern, we cannot verify rectangularity without per-well GT.
            # Conservative rule: apply loss only to patterns that are provably
            # rectangular — i.e., single row, single column, or single well.
            # For multi-row × multi-col patterns, skip (set weight to 0).
            is_safe = (active_rows <= 1) | (active_cols <= 1)  # (B,) bool

            if is_safe.any():
                # Only compute loss over safe (rectangular) samples
                safe_mask = is_safe.float()  # (B,)
                well_loss_per_sample = F.binary_cross_entropy(
                    pred_wells, gt_cart, reduction='none'
                ).mean(dim=(1, 2))  # (B,)
                well_loss = (safe_mask * well_loss_per_sample).sum() / (safe_mask.sum() + 1e-8)
                total_loss = total_loss + self.well_consistency_weight * well_loss

        return total_loss

    def _focal_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss for a single task (row or column).

        Standard focal loss: -alpha_t * (1 - p_t)^gamma * log(p_t)
        where alpha_t = alpha for positive examples, (1 - alpha) for negatives.

        Args:
            logits: (B, C) raw logits
            targets: (B, C) binary targets [0, 1]

        Returns:
            loss: Scalar focal loss
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(logits)  # (B, C)

        # Focal weighting: (1 - p_t)^gamma
        # For positive samples: (1 - p)^gamma  — hard positives get upweighted
        # For negative samples: p^gamma         — hard negatives get upweighted
        focal_weight = torch.where(
            targets > 0.5,
            (1 - probs) ** self.gamma,  # Positive: (1-p)^gamma
            probs ** self.gamma  # Negative: p^gamma
        )

        # Alpha balancing: alpha for positives, (1 - alpha) for negatives.
        # With alpha=0.75 this gives 3× weight to rare positive wells,
        # counteracting the ~96% negative class imbalance in 96-well plates.
        alpha_weight = torch.where(
            targets > 0.5,
            torch.full_like(targets, self.alpha),       # positives: alpha
            torch.full_like(targets, 1.0 - self.alpha), # negatives: 1-alpha
        )

        # Binary cross-entropy (numerically stable, from logits)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets.float(),
            reduction='none'
        )  # (B, C)

        # Combine: alpha_t * focal_weight * BCE
        focal_loss = alpha_weight * focal_weight * bce_loss  # (B, C)

        # Return mean loss
        return focal_loss.mean()
