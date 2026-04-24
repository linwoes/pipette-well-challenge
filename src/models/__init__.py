"""
Models Module: Neural network architecture for well detection

Components:
  - backbone: DINOv2-ViT-B/14 with LoRA adapters
  - fusion: Temporal attention + late dual-view fusion
  - loss: Focal loss for multi-label well classification
"""

from .backbone import DINOv2Backbone, LoRAAdapter
from .fusion import TemporalAttention, DualViewFusion, WellDetectionLoss

__all__ = [
    'DINOv2Backbone',
    'LoRAAdapter',
    'TemporalAttention',
    'DualViewFusion',
    'WellDetectionLoss',
]
