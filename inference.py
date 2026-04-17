#!/usr/bin/env python3
"""
Pipette Well Detection - Inference CLI

Predicts which well(s) of a 96-well plate are targeted during automated
liquid dispensing, given synchronized dual-view video (FPV + top-view).

Usage:
  python inference.py --fpv path/to/fpv.mp4 --topview path/to/topview.mp4 --output result.json
  python inference.py --fpv fpv.mp4 --topview top.mp4  # writes to stdout if no --output
  python inference.py --fpv fpv.mp4 --topview top.mp4 --model checkpoints/best.pt --threshold 0.4
  python inference.py --fpv fpv.mp4 --topview top.mp4 --img_size 448

The model architecture is DualViewFusion, which internally manages the DINOv2
backbone, temporal attention, and factorised row/column heads. Checkpoints saved
by train.py load directly into DualViewFusion without any wrapper class.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.models.fusion import DualViewFusion
from src.preprocessing.video_loader import load_video, align_clips, preprocess_frame
from src.postprocessing.output_formatter import (
    logits_to_wells, logits_to_wells_adaptive,
    validate_output, format_json_output,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipetteWellDetector:
    """
    Main inference class for well detection.

    Uses DualViewFusion directly — the same nn.Module that train.py saves
    checkpoints for. No intermediate wrapper class needed.
    """

    def __init__(
        self,
        model_checkpoint: Optional[str] = None,
        threshold: float = 0.3,
        device: Optional[str] = None,
        img_size: int = 448,
        use_adaptive: bool = True,
    ):
        """
        Initialize the detector.

        Args:
            model_checkpoint: Path to model checkpoint (.pt file)
            threshold: Confidence threshold for fixed-threshold predictions
            device: Device to run on ('cuda' or 'cpu')
            img_size: Input image size (must be multiple of 14 for DINOv2)
            use_adaptive: Use adaptive post-processing (recommended)
        """
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        self.use_adaptive = use_adaptive

        logger.info("Initializing PipetteWellDetector")
        logger.info(f"Device: {self.device}")
        logger.info(f"Threshold: {threshold}")
        logger.info(f"Image size: {img_size}")
        logger.info(f"Adaptive post-processing: {use_adaptive}")

        # Initialize model
        self.model = self._load_model(model_checkpoint)
        self.model.eval()

    def _load_model(self, model_checkpoint: Optional[str] = None) -> DualViewFusion:
        """
        Load trained DualViewFusion model from checkpoint.

        The checkpoint was saved by train.py which uses DualViewFusion directly,
        so we create the same architecture and load state_dict into it.

        Backbone type (DINOv2 vs ResNet18) is auto-detected from checkpoint keys:
        - If checkpoint contains 'backbone_fpv.model.conv1.weight', it was trained
          with LegacyResNet18Backbone (use_dinov2=False).
        - Otherwise, DINOv2 backbone is assumed (use_dinov2=True).
        """
        checkpoint_path = model_checkpoint or 'checkpoints/best.pt'

        # Auto-detect backbone type from checkpoint before building model
        use_dinov2 = True  # default
        if os.path.exists(checkpoint_path):
            try:
                state = torch.load(
                    checkpoint_path,
                    map_location='cpu',
                    weights_only=True,
                )
                ckpt_keys = set(state.get('model_state_dict', state).keys())
                if any('conv1.weight' in k for k in ckpt_keys):
                    use_dinov2 = False
                    logger.info(
                        "Checkpoint uses LegacyResNet18Backbone (detected conv1.weight). "
                        "Initializing with use_dinov2=False."
                    )
                else:
                    logger.info("Checkpoint uses DINOv2 backbone. Initializing with use_dinov2=True.")
            except Exception as e:
                logger.warning(f"Could not pre-inspect checkpoint keys: {e}. Defaulting to use_dinov2=True.")

        # Create model matching the detected architecture
        model = DualViewFusion(
            num_rows=8,
            num_columns=12,
            shared_backbone=True,
            use_lora=True,
            lora_rank=8,
            use_dinov2=use_dinov2,
            img_size=self.img_size,
        )
        model = model.to(self.device)

        # Load checkpoint weights
        if os.path.exists(checkpoint_path):
            try:
                state = torch.load(
                    checkpoint_path,
                    map_location=self.device,
                    weights_only=True,
                )
                model_state = state.get('model_state_dict', state)
                model.load_state_dict(model_state)
                epoch = state.get('epoch', '?')
                val_loss = state.get('val_loss', '?')
                logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {epoch}, val_loss={val_loss})")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint from {checkpoint_path}: {e}")
                logger.warning("Running with uninitialized weights (inference will be random)")
        else:
            logger.warning(f"No checkpoint found at {checkpoint_path}")
            logger.warning("Running with uninitialized weights (inference will be random)")

        return model

    def load_and_preprocess_videos(
        self,
        fpv_path: str,
        topview_path: str,
        num_frames: int = 8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and preprocess dual-view videos.

        Args:
            fpv_path: Path to FPV video
            topview_path: Path to top-view video
            num_frames: Number of frames to sample

        Returns:
            (fpv_tensor, topview_tensor): Both shape (1, N, 3, img_size, img_size)
        """
        logger.info(f"Loading videos: FPV={fpv_path}, TopView={topview_path}")

        # Validate input files
        if not Path(fpv_path).exists():
            raise FileNotFoundError(f"FPV video not found: {fpv_path}")
        if not Path(topview_path).exists():
            raise FileNotFoundError(f"Top-view video not found: {topview_path}")

        # Load frames from both videos
        fpv_frames = load_video(fpv_path, max_frames=num_frames)
        topview_frames = load_video(topview_path, max_frames=num_frames)

        # Align to same length
        fpv_frames, topview_frames = align_clips(fpv_frames, topview_frames)
        logger.info(f"Loaded {len(fpv_frames)} frames from each video")

        # Preprocess each frame (resize to img_size × img_size, normalise to [0, 1])
        fpv_processed = np.array([
            preprocess_frame(f, size=(self.img_size, self.img_size)) for f in fpv_frames
        ])
        topview_processed = np.array([
            preprocess_frame(f, size=(self.img_size, self.img_size)) for f in topview_frames
        ])

        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        fpv_processed = (fpv_processed - mean[None, None, :]) / std[None, None, :]
        topview_processed = (topview_processed - mean[None, None, :]) / std[None, None, :]

        # Convert to tensors: (N, H, W, 3) → (N, 3, H, W) → (1, N, 3, H, W)
        fpv_tensor = torch.from_numpy(fpv_processed.transpose(0, 3, 1, 2)).float()
        topview_tensor = torch.from_numpy(topview_processed.transpose(0, 3, 1, 2)).float()

        fpv_tensor = fpv_tensor.unsqueeze(0).to(self.device)
        topview_tensor = topview_tensor.unsqueeze(0).to(self.device)

        logger.info(f"Preprocessed tensors: FPV {fpv_tensor.shape}, TopView {topview_tensor.shape}")

        return fpv_tensor, topview_tensor

    def infer(
        self,
        fpv_tensor: torch.Tensor,
        topview_tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Run model inference.

        Args:
            fpv_tensor: (1, N, 3, H, W) preprocessed FPV frames
            topview_tensor: (1, N, 3, H, W) preprocessed top-view frames

        Returns:
            row_logits: (1, 8) raw row predictions
            col_logits: (1, 12) raw column predictions
            elapsed: Inference time in seconds
        """
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

    def postprocess_predictions(
        self,
        row_logits: torch.Tensor,
        col_logits: torch.Tensor,
        fpv_path: str,
        topview_path: str,
        inference_time: float,
    ) -> Dict:
        """
        Convert logits to well predictions with uncertainty handling.

        Args:
            row_logits: (1, 8) tensor
            col_logits: (1, 12) tensor
            fpv_path: FPV video path (used as clip_id)
            topview_path: Topview video path (used as clip_id)
            inference_time: Inference time in seconds

        Returns:
            JSON output dictionary matching challenge spec
        """
        logger.info("Post-processing predictions...")

        # Convert to numpy
        row_arr = row_logits.squeeze(0).cpu().numpy()
        col_arr = col_logits.squeeze(0).cpu().numpy()

        # Apply sigmoid for probability reporting
        row_probs = 1.0 / (1.0 + np.exp(-np.clip(row_arr, -500, 500)))
        col_probs = 1.0 / (1.0 + np.exp(-np.clip(col_arr, -500, 500)))

        logger.info(f"Row max prob: {row_probs.max():.4f}, Col max prob: {col_probs.max():.4f}")

        # Generate well predictions
        if self.use_adaptive:
            wells = logits_to_wells_adaptive(row_arr, col_arr)
            confident = True  # adaptive always produces predictions
        elif row_probs.max() < self.threshold or col_probs.max() < self.threshold:
            logger.warning(
                f"Low confidence — Max row: {row_probs.max():.4f}, Max col: {col_probs.max():.4f}"
            )
            wells = []
            confident = False
        else:
            wells = logits_to_wells(row_arr, col_arr, threshold=self.threshold)
            confident = True

        # Format output matching challenge spec
        output = format_json_output(
            clip_id_fpv=fpv_path,
            clip_id_topview=topview_path,
            wells=wells,
            inference_time_s=inference_time,
            confident=confident,
        )

        # Add extra metadata
        output['metadata']['threshold'] = self.threshold
        output['metadata']['adaptive'] = self.use_adaptive
        output['metadata']['max_row_prob'] = float(row_probs.max())
        output['metadata']['max_col_prob'] = float(col_probs.max())

        logger.info(f"Predicted {len(wells)} well(s)")

        return output

    def infer_and_predict(self, fpv_path: str, topview_path: str) -> Dict:
        """End-to-end inference pipeline: load videos → model → well predictions."""
        start_time = time.time()

        try:
            # Load and preprocess videos
            fpv_tensor, topview_tensor = self.load_and_preprocess_videos(fpv_path, topview_path)

            # Run inference
            row_logits, col_logits, inference_time = self.infer(fpv_tensor, topview_tensor)

            # Post-process
            output = self.postprocess_predictions(
                row_logits, col_logits, fpv_path, topview_path, inference_time
            )

            # Validate schema
            if output.get('wells_prediction'):
                if not validate_output(output['wells_prediction']):
                    logger.error("Output failed schema validation — clearing predictions")
                    output['wells_prediction'] = []

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
    parser.add_argument('--model', type=str, default=None, help='Model checkpoint path')
    parser.add_argument('--threshold', type=float, default=0.3, help='Confidence threshold (default 0.3)')
    parser.add_argument('--img_size', type=int, default=448,
                        help='Input image size (must be multiple of 14; default 448)')
    parser.add_argument('--no-adaptive', action='store_true',
                        help='Disable adaptive post-processing (use fixed threshold)')
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
    logger.info(f"Image size: {args.img_size}")
    if args.model:
        logger.info(f"Model: {args.model}")
    logger.info(f"Threshold: {args.threshold}")
    logger.info("=" * 70)

    try:
        detector = PipetteWellDetector(
            model_checkpoint=args.model,
            threshold=args.threshold,
            img_size=args.img_size,
            use_adaptive=not args.no_adaptive,
        )

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
