#!/usr/bin/env python3
"""
Pipette Well Detection - Inference CLI
====================================

Predicts which well(s) of a 96-well plate are targeted during automated
liquid dispensing, given synchronized dual-view video (FPV + top-view).

Usage:
  python inference.py --fpv path/to/fpv.mp4 --topview path/to/topview.mp4 --output result.json
  python inference.py --fpv fpv.mp4 --topview top.mp4  # writes to stdout if no --output
  python inference.py --fpv fpv.mp4 --topview top.mp4 --model checkpoints/best.pt --threshold 0.4

Author: ML Team
Date: April 2026
"""

import argparse
import json
import logging
import os
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

from src.models.backbone import DINOv2Backbone
from src.models.fusion import DualViewFusion
from src.preprocessing.video_loader import load_video, align_clips, preprocess_frame
from src.postprocessing.output_formatter import logits_to_wells, validate_output, format_json_output

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipetteWellModel(nn.Module):
    """Unified model combining backbone and fusion."""

    def __init__(self, backbone, fusion):
        super().__init__()
        self.backbone = backbone
        self.fusion = fusion

    def forward(self, fpv_frames: torch.Tensor, topview_frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            fpv_frames: (B, N, 3, 224, 224) or (B, 3, 224, 224)
            topview_frames: (B, N, 3, 224, 224) or (B, 3, 224, 224)

        Returns:
            row_logits: (B, 8)
            col_logits: (B, 12)
        """
        # Handle both 4D and 5D inputs
        if fpv_frames.dim() == 5:
            B, N, C, H, W = fpv_frames.shape
            fpv_frames = fpv_frames.view(B * N, C, H, W)
            topview_frames = topview_frames.view(B * N, C, H, W)
            extract_cls = True
        else:
            B = fpv_frames.shape[0]
            extract_cls = False

        # Extract features from both views
        fpv_features = self.backbone(fpv_frames)  # (B*N, 768) or (B, 768)
        topview_features = self.backbone(topview_frames)

        # Pool over frames if needed
        if extract_cls:
            fpv_features = fpv_features.view(B, N, -1).mean(dim=1)  # (B, 768)
            topview_features = topview_features.view(B, N, -1).mean(dim=1)

        # Fuse and predict
        row_logits, col_logits = self.fusion(fpv_features, topview_features)
        return row_logits, col_logits


class PipetteWellDetector:
    """Main inference class for well detection."""

    def __init__(self, config_path: Optional[str] = None, model_checkpoint: Optional[str] = None,
                 threshold: float = 0.5, device: Optional[str] = None):
        """
        Initialize the detector.

        Args:
            config_path: Path to YAML configuration file
            model_checkpoint: Path to model checkpoint (overrides config)
            threshold: Confidence threshold for predictions
            device: Device to run on ('cuda' or 'cpu')
        """
        self.config_path = config_path or "configs/default.yaml"
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Initializing PipetteWellDetector")
        logger.info(f"Device: {self.device}")
        logger.info(f"Threshold: {threshold}")

        # Load configuration
        self.config = self._load_config(self.config_path)

        # Initialize model
        self.model = self._load_model(model_checkpoint)
        self.model.eval()

    def _load_config(self, config_path: str) -> Dict:
        """Load YAML configuration."""
        if not os.path.exists(config_path):
            logger.warning(f"Config not found: {config_path}. Using defaults.")
            return self._default_config()

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded config from {config_path}")
        return config

    def _default_config(self) -> Dict:
        """Return default configuration."""
        return {
            'model': {
                'backbone': 'dinov2',
                'input_size': [224, 224],
                'num_rows': 8,
                'num_columns': 12,
                'fusion_type': 'concatenation',
            },
            'checkpoint': {
                'path': 'checkpoints/best.pt',
                'device': self.device,
            },
            'inference': {
                'batch_size': 1,
                'confidence_threshold': self.threshold,
                'max_inference_time_seconds': 120,
            },
            'video': {
                'target_fps': 30,
                'frame_resize': [224, 224],
            }
        }

    def _load_model(self, model_checkpoint: Optional[str] = None) -> PipetteWellModel:
        """Load trained model from checkpoint."""
        checkpoint_path = model_checkpoint or self.config.get('checkpoint', {}).get('path', 'checkpoints/best.pt')

        # Initialize backbone and fusion
        backbone = DINOv2Backbone(use_lora=True, freeze_base=False)
        fusion = DualViewFusion(
            feature_dim=768,
            num_rows=8,
            num_columns=12,
            fusion_type=self.config.get('model', {}).get('fusion_type', 'concatenation'),
        )

        model = PipetteWellModel(backbone, fusion)
        model = model.to(self.device)

        # Try to load checkpoint
        if os.path.exists(checkpoint_path):
            try:
                state = torch.load(checkpoint_path, map_location=self.device,
                                       weights_only=True)
                model.load_state_dict(state['model_state_dict'])
                logger.info(f"Loaded checkpoint from {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint from {checkpoint_path}: {e}")
                logger.warning("Running with uninitialized weights (inference will be random)")
        else:
            logger.warning(f"No checkpoint found at {checkpoint_path}")
            logger.warning("Running with uninitialized weights (inference will be random)")

        return model

    def load_and_preprocess_videos(self, fpv_path: str, topview_path: str, num_frames: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and preprocess dual-view videos.

        Args:
            fpv_path: Path to FPV video
            topview_path: Path to top-view video
            num_frames: Number of frames to sample

        Returns:
            (fpv_tensor, topview_tensor): Both shape (1, N, 3, 224, 224)
        """
        logger.info(f"Loading videos: FPV={fpv_path}, TopView={topview_path}")

        # Validate input files
        if not Path(fpv_path).exists():
            raise FileNotFoundError(f"FPV video not found: {fpv_path}")
        if not Path(topview_path).exists():
            raise FileNotFoundError(f"Top-view video not found: {topview_path}")

        # Load frames from both videos
        fpv_frames = load_video(fpv_path, max_frames=num_frames)  # (N, H, W, 3)
        topview_frames = load_video(topview_path, max_frames=num_frames)

        # Align to same length
        fpv_frames, topview_frames = align_clips(fpv_frames, topview_frames)
        logger.info(f"Loaded {len(fpv_frames)} frames from each video")

        # Preprocess each frame
        fpv_processed = np.array([preprocess_frame(f) for f in fpv_frames])  # (N, 224, 224, 3)
        topview_processed = np.array([preprocess_frame(f) for f in topview_frames])

        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        fpv_processed = (fpv_processed - mean[None, None, :]) / std[None, None, :]
        topview_processed = (topview_processed - mean[None, None, :]) / std[None, None, :]

        # Convert to tensors: (N, 224, 224, 3) -> (N, 3, 224, 224) -> (1, N, 3, 224, 224)
        fpv_tensor = torch.from_numpy(fpv_processed.transpose(0, 3, 1, 2)).float()
        topview_tensor = torch.from_numpy(topview_processed.transpose(0, 3, 1, 2)).float()

        # Add batch dimension
        fpv_tensor = fpv_tensor.unsqueeze(0).to(self.device)  # (1, N, 3, 224, 224)
        topview_tensor = topview_tensor.unsqueeze(0).to(self.device)

        logger.info(f"Preprocessed tensors: FPV {fpv_tensor.shape}, TopView {topview_tensor.shape}")

        return fpv_tensor, topview_tensor

    def infer(self, fpv_tensor: torch.Tensor, topview_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run model inference."""
        logger.info("Running inference...")

        start_time = time.time()

        with torch.no_grad():
            row_logits, col_logits = self.model(fpv_tensor, topview_tensor)

        elapsed = time.time() - start_time

        if elapsed > 90:
            logger.warning(f"Inference took {elapsed:.2f}s (approaching 2-min SLA buffer)")

        logger.info(f"Inference completed in {elapsed:.3f}s")
        logger.info(f"Row logits shape: {row_logits.shape}, Col logits shape: {col_logits.shape}")

        return row_logits, col_logits, elapsed

    def postprocess_predictions(self, row_logits: torch.Tensor, col_logits: torch.Tensor,
                               fpv_path: str, topview_path: str, inference_time: float) -> Dict:
        """
        Convert logits to well predictions with uncertainty handling.

        Args:
            row_logits: (B, 8) tensor
            col_logits: (B, 12) tensor
            fpv_path: FPV video path
            topview_path: Topview video path
            inference_time: Inference time in seconds

        Returns:
            JSON output dictionary
        """
        logger.info("Post-processing predictions...")

        # Apply sigmoid
        row_probs = torch.sigmoid(row_logits).squeeze(0).cpu().numpy()  # (8,)
        col_probs = torch.sigmoid(col_logits).squeeze(0).cpu().numpy()  # (12,)

        logger.info(f"Row max prob: {row_probs.max():.4f}, Col max prob: {col_probs.max():.4f}")

        # Check for uncertain output
        if row_probs.max() < self.threshold or col_probs.max() < self.threshold:
            logger.warning(f"Low confidence detected. Max row: {row_probs.max():.4f}, Max col: {col_probs.max():.4f}")
            output = format_json_output(fpv_path, topview_path, [])
            output['metadata']['uncertain'] = True
            output['metadata']['reason'] = 'low_confidence'
            output['metadata']['max_row_prob'] = float(row_probs.max())
            output['metadata']['max_col_prob'] = float(col_probs.max())
        else:
            # Convert logits to wells using Cartesian product
            wells = logits_to_wells(row_probs, col_probs, threshold=self.threshold)
            output = format_json_output(fpv_path, topview_path, wells)

        # Add metadata
        output['metadata']['inference_time_s'] = round(inference_time, 3)
        output['metadata']['model'] = 'DINOv2-ViT-B/14+LoRA'
        output['metadata']['confident'] = row_probs.max() >= self.threshold and col_probs.max() >= self.threshold
        output['metadata']['threshold'] = self.threshold

        logger.info(f"Predicted {len(output.get('wells', []))} wells")

        return output

    def infer_and_predict(self, fpv_path: str, topview_path: str) -> Dict:
        """End-to-end inference pipeline."""
        start_time = time.time()

        try:
            # Load and preprocess videos
            fpv_tensor, topview_tensor = self.load_and_preprocess_videos(fpv_path, topview_path)

            # Run inference
            row_logits, col_logits, inference_time = self.infer(fpv_tensor, topview_tensor)

            # Post-process
            output = self.postprocess_predictions(row_logits, col_logits, fpv_path, topview_path, inference_time)

            # Validate schema
            if 'wells' in output:
                if not validate_output(output['wells']):
                    logger.error("Output failed schema validation")
                    output['wells'] = []

            total_time = time.time() - start_time
            logger.info(f"Total pipeline time: {total_time:.3f}s")

            return output

        except Exception as e:
            logger.error(f"Inference failed: {e}", exc_info=True)
            raise


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='Pipette Well Detection - Predict well coordinates from dual-view video'
    )

    parser.add_argument('--fpv', type=str, required=True, help='Path to FPV video file')
    parser.add_argument('--topview', type=str, required=True, help='Path to top-view video file')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file (default: stdout)')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Config YAML file')
    parser.add_argument('--model', type=str, default=None, help='Model checkpoint path')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info("=" * 70)
    logger.info("Pipette Well Detection Inference")
    logger.info("=" * 70)
    logger.info(f"FPV video: {args.fpv}")
    logger.info(f"Top-view video: {args.topview}")
    logger.info(f"Output: {args.output or 'stdout'}")
    logger.info(f"Config: {args.config}")
    if args.model:
        logger.info(f"Model: {args.model}")
    logger.info(f"Threshold: {args.threshold}")
    logger.info("=" * 70)

    try:
        # Initialize detector
        detector = PipetteWellDetector(
            config_path=args.config,
            model_checkpoint=args.model,
            threshold=args.threshold
        )

        # Run inference
        result = detector.infer_and_predict(args.fpv, args.topview)

        # Write output
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Results written to: {output_path}")

        # Always print to stdout
        print(json.dumps(result, indent=2))

        return 0

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
