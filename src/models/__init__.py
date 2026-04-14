"""
Models Module: Neural network architecture for well detection

Components:
  - backbone: ResNet-18 backbone for spatial feature extraction
  - fusion: Dual-view feature fusion (concatenation, cross-attention, gating)
  - heads: Output heads (factorized row/column classification)

TODO:
  - Implement backbone.py with ResNet-18 + pretrained weights
  - Implement fusion.py with multiple fusion strategies
  - Implement multi-task head (row + column + cardinality)
"""
