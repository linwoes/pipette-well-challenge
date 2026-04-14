"""
Backbone Module: ResNet-18 feature extractor

TODO:
  - Load torchvision.models.resnet18(pretrained=True)
  - Freeze early layers (layer1, layer2) initially
  - Replace final layer for 512-D feature output
  - Add option to unfreeze for fine-tuning
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNet18Backbone(nn.Module):
    """
    ResNet-18 backbone for spatial feature extraction.

    Takes (B, 3, 224, 224) input and outputs (B, 512) features.

    TODO:
        1. Load pretrained ResNet-18
        2. Remove classification head (replace with identity)
        3. Add methods to freeze/unfreeze layers
        4. Implement forward pass
    """

    def __init__(self, pretrained: bool = True, freeze_early: bool = True):
        """
        Initialize backbone.

        Args:
            pretrained: Load ImageNet pre-trained weights
            freeze_early: Freeze layer1, layer2 for initial training stability
        """
        super().__init__()

        # TODO: Load ResNet-18
        raise NotImplementedError("ResNet18Backbone not yet implemented")

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (B, 3, 224, 224) input tensor

        Returns:
            features: (B, 512) feature vector

        TODO:
            1. Pass through conv1 + layer1-4
            2. Apply global average pooling
            3. Return (B, 512) features
        """
        raise NotImplementedError("forward() not yet implemented")

    def unfreeze_layers(self, start_layer: int = 2):
        """
        Unfreeze layers from start_layer onwards for fine-tuning.

        Args:
            start_layer: Layer number to start unfreezing (1, 2, 3, or 4)

        TODO:
            1. Iterate over model parameters
            2. Unfreeze layers >= start_layer
        """
        raise NotImplementedError("unfreeze_layers() not yet implemented")
