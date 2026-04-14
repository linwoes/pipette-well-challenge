"""
Fusion Module: Dual-view feature fusion and output heads

TODO:
  - Implement DualViewFusion class
  - Support multiple fusion strategies: concatenation, cross-attention, gating
  - Implement factorized output heads (row + column)
  - Optional: cardinality head for explicit well count prediction
"""

import torch
import torch.nn as nn


class DualViewFusion(nn.Module):
    """
    Late-fusion architecture for FPV + top-view dual-view inputs.

    Takes independent feature vectors from both views and produces
    row/column logits via factorized heads.

    Architecture:
        FPV backbone:      (B, 3, 224, 224) → (B, 512)
        TopView backbone:  (B, 3, 224, 224) → (B, 512)
                                                    ↓
                                              Concat → (B, 1024)
                                                    ↓
                                              FC → (B, 512)
                                                    ↓
                            ┌───────────────────────┴───────────────────────┐
                            ↓                                               ↓
                    Row Head: FC → (B, 8)              Column Head: FC → (B, 12)
                            ↓                                               ↓
                        Sigmoid                                        Sigmoid
    """

    def __init__(self, feature_dim: int = 512, num_rows: int = 8, num_columns: int = 12,
                 fusion_type: str = "concatenation", dropout: float = 0.3):
        """
        Initialize fusion module.

        Args:
            feature_dim: Dimension of backbone features (512 for ResNet-18)
            num_rows: Number of rows on plate (8 for 96-well)
            num_columns: Number of columns on plate (12 for 96-well)
            fusion_type: 'concatenation', 'cross_attention', or 'gating'
            dropout: Dropout rate for FC layers

        TODO:
            1. Implement concatenation-based fusion (simplest)
            2. Implement cross-attention fusion (more sophisticated)
            3. Implement gating fusion (learned weighting)
            4. Create factorized output heads
        """
        super().__init__()

        raise NotImplementedError("DualViewFusion not yet implemented")

    def forward(self, fpv_features: torch.Tensor, topview_features: torch.Tensor):
        """
        Forward pass.

        Args:
            fpv_features: (B, 512) features from FPV backbone
            topview_features: (B, 512) features from top-view backbone

        Returns:
            row_logits: (B, 8) row predictions (one per row A-H)
            col_logits: (B, 12) column predictions (one per column 1-12)

        TODO:
            1. Fuse FPV and top-view features
            2. Pass through shared FC layers
            3. Split to row and column heads
            4. Apply sigmoid activation
            5. Return both output tensors
        """
        raise NotImplementedError("forward() not yet implemented")

    def _concat_fusion(self, fpv_features, topview_features):
        """Concatenate-based fusion (simplest)."""
        raise NotImplementedError("_concat_fusion() not yet implemented")

    def _cross_attention_fusion(self, fpv_features, topview_features):
        """Cross-attention fusion (more sophisticated)."""
        raise NotImplementedError("_cross_attention_fusion() not yet implemented")

    def _gating_fusion(self, fpv_features, topview_features):
        """Gating fusion (learned weighting)."""
        raise NotImplementedError("_gating_fusion() not yet implemented")


class MultiTaskHead(nn.Module):
    """
    Multi-task output head with row, column, and optional cardinality prediction.

    Used within DualViewFusion or as standalone module.

    TODO:
        1. Implement row head (8-class classifier)
        2. Implement column head (12-class classifier)
        3. Optional: cardinality head (1-class, 8-class, 12-class classifier)
    """

    def __init__(self, input_dim: int = 512, num_rows: int = 8, num_columns: int = 12,
                 include_cardinality: bool = False, dropout: float = 0.3):
        """
        Initialize multi-task head.

        Args:
            input_dim: Input feature dimension
            num_rows: Number of rows to classify
            num_columns: Number of columns to classify
            include_cardinality: Add cardinality (1/8/12) prediction head
            dropout: Dropout rate

        TODO:
            1. Create row FC layer: input_dim → num_rows
            2. Create column FC layer: input_dim → num_columns
            3. Optional: create cardinality FC layer: input_dim → 3 (for softmax)
        """
        super().__init__()

        raise NotImplementedError("MultiTaskHead not yet implemented")

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (B, input_dim) features

        Returns:
            row_logits: (B, num_rows)
            col_logits: (B, num_columns)
            cardinality_logits: (B, 3) if include_cardinality, else None

        TODO:
            1. Pass x through row head with sigmoid
            2. Pass x through col head with sigmoid
            3. Optional: pass x through cardinality head with softmax
            4. Return logits (NOT probabilities; no activation before loss)
        """
        raise NotImplementedError("forward() not yet implemented")
