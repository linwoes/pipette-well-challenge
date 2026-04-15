"""
Video Loader Module: Extract and synchronize frames from dual-view videos

Implements:
  - load_video(): Load video frames using cv2.VideoCapture
  - align_clips(): Align FPV and top-view frames by truncation
  - preprocess_frame(): Resize, normalize, convert BGR→RGB
  - snap_to_dinov2_resolution(): Round up to nearest DINOv2-compatible resolution

ARCHITECTURE FIX (April 2026):
  - Added snap_to_dinov2_resolution() to enforce patch-size alignment
  - DINOv2 ViT-B/14 requires img_size to be multiple of 14 (patch_size).
  - Automatically snaps invalid sizes to valid multiples with a warning.
  - Preferred: 224, 336, 448, 518. Use 518×518 for best spatial resolution.
"""

import cv2
import numpy as np
import logging
import math
from typing import Tuple


def load_video(path: str, max_frames: int = 8) -> np.ndarray:
    """
    Load video frames and sample evenly across video duration.

    Args:
        path: Path to MP4 or video file
        max_frames: Maximum number of frames to extract (default 8)

    Returns:
        frames: (N, H, W, 3) numpy array in RGB format

    Raises:
        FileNotFoundError: If video file not found
        IOError: If video cannot be opened or is corrupted
    """
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        raise IOError(f"Failed to open video: {path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count == 0:
        raise IOError(f"Video has no frames: {path}")

    # Calculate frame indices to sample evenly
    frame_indices = np.linspace(0, frame_count - 1, max_frames, dtype=int)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret or frame is None:
            continue

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if not frames:
        raise IOError(f"No valid frames extracted from video: {path}")

    # Stack frames into array (N, H, W, 3)
    return np.array(frames, dtype=np.uint8)


def align_clips(fpv_frames: np.ndarray, topview_frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align FPV and top-view frames by truncating to shorter length.

    Args:
        fpv_frames: (N, H, W, 3) array of FPV frames
        topview_frames: (M, H, W, 3) array of top-view frames

    Returns:
        (aligned_fpv, aligned_topview): Both arrays with same length N=min(N, M)
    """
    min_len = min(len(fpv_frames), len(topview_frames))
    return fpv_frames[:min_len], topview_frames[:min_len]


def snap_to_dinov2_resolution(size: int, patch_size: int = 14) -> int:
    """
    Round size UP to the nearest multiple of patch_size.

    DINOv2 ViT-B/14 requires all spatial dimensions to be multiples of 14.
    Preferred values: 224 (16×14), 336 (24×14), 448 (32×14), 518 (37×14).
    518 is Meta's recommended resolution for best spatial fidelity.

    Args:
        size: Requested image size
        patch_size: DINOv2 patch size (default 14)

    Returns:
        Nearest valid size >= requested size

    Examples:
        snap_to_dinov2_resolution(224) -> 224  (already valid)
        snap_to_dinov2_resolution(256) -> 266  (next multiple of 14)
        snap_to_dinov2_resolution(518) -> 518  (already valid)
    """
    snapped = math.ceil(size / patch_size) * patch_size
    if snapped != size:
        logging.getLogger(__name__).warning(
            f"img_size={size} is not a multiple of DINOv2 patch_size={patch_size}. "
            f"Snapping to {snapped}. Use 224 or 518 to avoid this adjustment."
        )
    return snapped


def preprocess_frame(frame: np.ndarray, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Preprocess frame: resize, normalize to [0, 1], ensure RGB.

    Note: When using DINOv2 backbone, both dimensions must be multiples of 14
    (the ViT-B/14 patch size). Valid sizes: 224 (16×14), 336, 448, 518 (37×14).
    518×518 is recommended for best spatial resolution (37×37=1369 patches vs
    224×224's 16×16=256 patches). Use snap_to_dinov2_resolution() to auto-correct.

    Args:
        frame: Input frame (H, W, 3) in RGB format
        size: Target size (height, width)

    Returns:
        Preprocessed frame (size[0], size[1], 3) normalized to [0, 1]
    """
    # Resize frame
    resized = cv2.resize(frame, (size[1], size[0]))  # cv2.resize takes (width, height)

    # Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0

    return normalized
