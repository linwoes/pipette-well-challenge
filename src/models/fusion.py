"""
Fusion Module: Temporal attention + late fusion + factorized output heads

Implements:
  - TemporalAttention: Transformer over ordered frames per view
  - DualViewFusion: Complete model with shared backbone and dual-view fusion
  - WellDetectionLoss: Combined focal loss for row and column heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .backbone import DINOv2Backbone


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
            use_dinov2: Use DINOv2 backbone if True, ResNet18 if False (default True)
        """
        super().__init__()

        self.num_rows = num_rows
        self.num_columns = num_columns
        self.shared_backbone = shared_backbone
        self.temporal_dim = temporal_dim
        self.use_dinov2 = use_dinov2

        # Initialize backbones
        if use_dinov2:
            self.backbone_fpv = DINOv2Backbone(
                use_lora=use_lora,
                lora_rank=lora_rank,
                freeze_base=True
            )
            if not shared_backbone:
                self.backbone_topview = DINOv2Backbone(
                    use_lora=use_lora,
                    lora_rank=lora_rank,
                    freeze_base=True
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
        fpv_flat = fpv_frames.reshape(B * N, C, H, W)  # (B*N, 3, 224, 224)
        with torch.no_grad():
            fpv_features = self.backbone_fpv(fpv_flat)  # (B*N, 768)
        fpv_features = fpv_features.reshape(B, N, -1)  # (B, N, 768)

        # Process TopView frames through backbone
        topview_flat = topview_frames.reshape(B * N, C, H, W)  # (B*N, 3, 224, 224)
        if self.shared_backbone:
            with torch.no_grad():
                topview_features = self.backbone_fpv(topview_flat)  # (B*N, 768)
        else:
            with torch.no_grad():
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

    Default: gamma=2.0, alpha=0.25 (from Focal Loss paper)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        row_weight: float = 1.0,
        col_weight: float = 1.0,
    ):
        """
        Initialize focal loss.

        Args:
            gamma: Focusing parameter (default 2.0)
            alpha: Weighting factor for positive class (default 0.25)
            row_weight: Weight for row loss (default 1.0)
            col_weight: Weight for column loss (default 1.0)
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.row_weight = row_weight
        self.col_weight = col_weight

    def forward(
        self,
        row_logits: torch.Tensor,
        col_logits: torch.Tensor,
        row_targets: torch.Tensor,
        col_targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss for row and column predictions.

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

        # Weighted combination
        total_loss = self.row_weight * row_loss + self.col_weight * col_loss

        return total_loss

    def _focal_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss for a single task (row or column).

        Args:
            logits: (B, C) raw logits
            targets: (B, C) binary targets [0, 1]

        Returns:
            loss: Scalar focal loss
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(logits)  # (B, C)

        # Focal weighting
        # For positive samples: (1 - p)^gamma
        # For negative samples: p^gamma
        focal_weight = torch.where(
            targets > 0.5,
            (1 - probs) ** self.gamma,  # Positive: (1-p)^gamma
            probs ** self.gamma  # Negative: p^gamma
        )

        # Binary cross-entropy with focal weighting
        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets.float(),
            reduction='none'
        )  # (B, C)

        # Apply focal weighting
        focal_loss = self.alpha * focal_weight * bce_loss  # (B, C)

        # Return mean loss
        return focal_loss.mean()
