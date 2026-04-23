#!/usr/bin/env python3
"""
Pipette Well Detection - Training Script
========================================

Trains DINOv2-ViT-B/14 backbone with LoRA adapters on pipette well detection task.

Usage:
  python train.py --data_dir ./data --labels labels.json --epochs 50 --output checkpoints/
  python train.py --data_dir ./data --labels labels.json --val_split 0.2 --batch_size 4 --device cuda:0

ARCHITECTURE FIX (April 2026):
  - Added img_size validation in PipetteWellDataset.__init__()
  - Auto-snaps invalid sizes to DINOv2-compatible resolutions (multiples of 14)
  - Supported: 224 (16×14, minimum), 336 (24×14), 448 (32×14), 518 (37×14, recommended).
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Try to import torch with CUDA library path handling
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
except ImportError as e:
    # If CUDA library errors, set environment to suppress them
    if 'libcuda' in str(e).lower() or 'libc++' in str(e).lower():
        os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu:' + os.environ.get('LD_LIBRARY_PATH', '')
        # Retry import after setting LD_LIBRARY_PATH
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import Dataset, DataLoader
    else:
        raise

import yaml
from tqdm import tqdm

# Handle albumentations gracefully
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    logging.warning("albumentations not available, using basic transforms")

from src.models.backbone import DINOv2Backbone
from src.models.fusion import DualViewFusion, WellDetectionLoss, TemporalAttention
from src.preprocessing.video_loader import load_video, align_clips, preprocess_frame
from src.utils.metrics import exact_match, jaccard_similarity, cardinality_accuracy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipetteWellDataset(Dataset):
    """
    Loads video pairs and labels for training.

    Expected data_dir structure:
      data_dir/
        clip_001_FPV.mp4
        clip_001_Topview.mp4
        clip_002_FPV.mp4
        clip_002_Topview.mp4
        ...

    labels.json format:
      [
        {
          "clip_id_FPV": "clip_001_FPV",
          "clip_id_Topview": "clip_001_Topview",
          "wells_ground_truth": [{"well_row": "A", "well_column": 1}, ...]
        },
        ...
      ]
    """

    def __init__(self, data_dir: str, labels_path: str, num_frames: int = 8, img_size: int = 224, augment: bool = False):
        """
        Initialize dataset.

        Args:
            data_dir: Path to directory containing videos
            labels_path: Path to JSON labels file
            num_frames: Number of frames to sample per video (default 8)
            img_size: Target image size (default 224)
            augment: Enable augmentation (default False)
        """
        self.data_dir = Path(data_dir)
        self.num_frames = num_frames
        self.img_size = img_size
        self.augment = augment and HAS_ALBUMENTATIONS

        # Validate DINOv2 patch alignment
        from src.preprocessing.video_loader import snap_to_dinov2_resolution
        snapped = snap_to_dinov2_resolution(img_size)
        if snapped != img_size:
            logger.warning(f"img_size={img_size} → snapped to {snapped} for DINOv2 alignment")
            self.img_size = snapped

        # Load labels
        with open(labels_path, 'r') as f:
            self.labels = json.load(f)

        logger.info(f"Loaded {len(self.labels)} labels")

        # T-3: Validate label schema at load time. Fail fast rather than
        # discovering malformed entries as cryptic errors mid-training.
        self._validate_labels(self.labels, labels_path)

        # Build augmentation pipeline (only if albumentations available)
        if self.augment:
            self.transform = A.Compose([
                A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.3),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, p=0.5),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.GaussNoise(p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], is_check_shapes=False)
        elif HAS_ALBUMENTATIONS:
            self.transform = A.Compose([
                A.Resize(height=img_size, width=img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], is_check_shapes=False)
        else:
            self.transform = None  # Will use simple normalization

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            fpv_frames: (N, 3, H, W) tensor
            topview_frames: (N, 3, H, W) tensor
            row_labels: (8,) multi-hot binary vector
            col_labels: (12,) multi-hot binary vector
        """
        label_entry = self.labels[idx]

        # Build video paths - clip IDs already include full filename
        fpv_stem = label_entry['clip_id_FPV']
        topview_stem = label_entry['clip_id_Topview']

        fpv_path = self.data_dir / f"{fpv_stem}.mp4"
        topview_path = self.data_dir / f"{topview_stem}.mp4"

        # Check if files exist
        if not fpv_path.exists():
            logger.warning(f"FPV file not found: {fpv_path}")
            raise FileNotFoundError(f"FPV file not found: {fpv_path}")
        if not topview_path.exists():
            logger.warning(f"Topview file not found: {topview_path}")
            raise FileNotFoundError(f"Topview file not found: {topview_path}")

        # Load and preprocess videos
        try:
            fpv_frames = load_video(str(fpv_path), max_frames=self.num_frames)  # (N, H, W, 3)
            topview_frames = load_video(str(topview_path), max_frames=self.num_frames)

            # Align to same length
            fpv_frames, topview_frames = align_clips(fpv_frames, topview_frames)

            # Preprocess each frame
            fpv_processed = np.array([preprocess_frame(f, size=(self.img_size, self.img_size)) for f in fpv_frames])
            topview_processed = np.array([preprocess_frame(f, size=(self.img_size, self.img_size)) for f in topview_frames])

            # Apply transforms per frame
            fpv_list = []
            for f in fpv_processed:
                if self.transform is not None:
                    # Convert from [0, 1] to [0, 255] for albumentations
                    f_uint8 = (f * 255).astype(np.uint8)
                    augmented = self.transform(image=f_uint8)
                    fpv_list.append(augmented['image'])
                else:
                    # Simple normalization without albumentations
                    fpv_tensor = torch.from_numpy(f).permute(2, 0, 1).float()
                    # ImageNet normalization
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    fpv_tensor = (fpv_tensor - mean) / std
                    fpv_list.append(fpv_tensor)

            topview_list = []
            for f in topview_processed:
                if self.transform is not None:
                    f_uint8 = (f * 255).astype(np.uint8)
                    augmented = self.transform(image=f_uint8)
                    topview_list.append(augmented['image'])
                else:
                    # Simple normalization
                    topview_tensor = torch.from_numpy(f).permute(2, 0, 1).float()
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    topview_tensor = (topview_tensor - mean) / std
                    topview_list.append(topview_tensor)

            fpv_tensor = torch.stack(fpv_list)  # (N, 3, H, W)
            topview_tensor = torch.stack(topview_list)  # (N, 3, H, W)

        except Exception as e:
            logger.error(f"Error loading {fpv_path} or {topview_path}: {e}")
            raise

        # Encode well labels to multi-hot vectors
        wells = label_entry.get('wells_ground_truth', [])
        row_labels, col_labels = self._encode_wells(wells)

        return fpv_tensor, topview_tensor, row_labels, col_labels

    @staticmethod
    def _encode_wells(wells: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert list of {well_row, well_column} to multi-hot row and col tensors.

        Args:
            wells: List of well dictionaries with 'well_row' and 'well_column'

        Returns:
            (row_labels, col_labels): Both shape (8,) and (12,) respectively
        """
        row_labels = torch.zeros(8)
        col_labels = torch.zeros(12)

        for w in wells:
            row_idx = ord(w['well_row']) - ord('A')
            col_idx = int(w['well_column']) - 1
            row_labels[row_idx] = 1.0
            col_labels[col_idx] = 1.0

        return row_labels, col_labels

    @staticmethod
    def _validate_labels(labels: List[Dict], labels_path: str) -> None:
        """
        T-3: Validate label schema at dataset load time.

        Checks:
          - Top-level list (not dict)
          - Each entry has required keys: clip_id_FPV, clip_id_Topview, wells_ground_truth
          - Each well has well_row (single uppercase A–H) and well_column (1–12)
          - well_column must be int-castable
          - Duplicate (clip_id_FPV, clip_id_Topview) pairs are flagged

        Raises:
            ValueError: On any schema violation, with the entry index and message.

        Logs a warning per entry for non-fatal issues (e.g., empty well list).
        """
        REQUIRED_ENTRY_KEYS = {'clip_id_FPV', 'clip_id_Topview', 'wells_ground_truth'}
        VALID_ROWS = set('ABCDEFGH')

        if not isinstance(labels, list):
            raise ValueError(
                f"labels.json must be a JSON array at the top level. "
                f"Got {type(labels).__name__}. File: {labels_path}"
            )

        seen_pairs: set = set()
        errors: List[str] = []

        for i, entry in enumerate(labels):
            prefix = f"labels[{i}]"

            # Required keys
            missing = REQUIRED_ENTRY_KEYS - set(entry.keys())
            if missing:
                errors.append(f"{prefix}: missing required keys {sorted(missing)}")
                continue  # skip further checks for this entry

            # Duplicate clip pairs
            pair = (entry['clip_id_FPV'], entry['clip_id_Topview'])
            if pair in seen_pairs:
                errors.append(f"{prefix}: duplicate clip pair {pair}")
            seen_pairs.add(pair)

            # Empty well list is valid (no-well clip) but worth logging
            wells = entry['wells_ground_truth']
            if not isinstance(wells, list):
                errors.append(f"{prefix}: wells_ground_truth must be a list, got {type(wells).__name__}")
                continue

            if len(wells) == 0:
                logger.warning(f"{prefix} ({pair[0]}): empty wells_ground_truth — is this intentional?")

            for j, well in enumerate(wells):
                wprefix = f"{prefix}.wells[{j}]"

                if not isinstance(well, dict):
                    errors.append(f"{wprefix}: expected dict, got {type(well).__name__}")
                    continue

                # well_row validation
                row = well.get('well_row')
                if row is None:
                    errors.append(f"{wprefix}: missing well_row")
                elif not isinstance(row, str) or row.upper() not in VALID_ROWS:
                    errors.append(f"{wprefix}: well_row={row!r} must be one of A–H")

                # well_column validation
                col = well.get('well_column')
                if col is None:
                    errors.append(f"{wprefix}: missing well_column")
                else:
                    try:
                        col_int = int(col)
                        if not (1 <= col_int <= 12):
                            errors.append(f"{wprefix}: well_column={col!r} out of range 1–12")
                    except (TypeError, ValueError):
                        errors.append(f"{wprefix}: well_column={col!r} is not int-castable")

        if errors:
            summary = "\n  ".join(errors)
            raise ValueError(
                f"Label validation failed for {labels_path} "
                f"({len(errors)} error(s)):\n  {summary}"
            )

        logger.info(f"Label validation passed: {len(labels)} entries, {len(seen_pairs)} unique clips")


class Trainer:
    """Training orchestrator."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str,
        output_dir: str,
        epochs: int = 50,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
        patience: int = 20,
        grad_clip: float = 1.0,
        focal_alpha: float = 0.75,
        focal_gamma: float = 0.0,
        col_weight: float = 2.0,
        well_consistency_weight: float = 0.2,
    ):
        """Initialize trainer."""
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.patience = patience
        self.grad_clip = grad_clip
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.col_weight = col_weight
        self.well_consistency_weight = well_consistency_weight

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer: only trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

        self.optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

        # Loss function: weighted BCE (gamma=0) or focal loss (gamma>0) + well-level consistency loss
        # col_weight=2.0: diagnostic shows column max sigmoid (~0.32) lags row (~0.54); upweight to close gap
        self.criterion = WellDetectionLoss(
            gamma=focal_gamma, alpha=focal_alpha,
            col_weight=col_weight,
            well_consistency_weight=well_consistency_weight,
        )

        # LR scheduler with warmup
        self.scheduler = self._build_scheduler(epochs, warmup_epochs)

        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if device.startswith('cuda') else None

        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def _build_scheduler(self, total_epochs: int, warmup_epochs: int):
        """Build learning rate scheduler with warmup."""
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Linear warmup
                return (epoch + 1) / warmup_epochs
            else:
                # Cosine annealing
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))

        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        for fpv, topview, row_labels, col_labels in tqdm(self.train_loader, desc='Training'):
            fpv = fpv.to(self.device)
            topview = topview.to(self.device)
            row_labels = row_labels.to(self.device)
            col_labels = col_labels.to(self.device)

            self.optimizer.zero_grad()

            # Mixed precision forward pass
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    row_logits, col_logits = self.model(fpv, topview)
                    loss = self.criterion(row_logits, col_logits, row_labels, col_labels)

                # Backward with scale
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                row_logits, col_logits = self.model(fpv, topview)
                loss = self.criterion(row_logits, col_logits, row_labels, col_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self) -> Tuple[float, Dict]:
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0

        all_row_preds = []
        all_col_preds = []
        all_row_targets = []
        all_col_targets = []

        with torch.no_grad():
            for fpv, topview, row_labels, col_labels in tqdm(self.val_loader, desc='Validation'):
                fpv = fpv.to(self.device)
                topview = topview.to(self.device)
                row_labels = row_labels.to(self.device)
                col_labels = col_labels.to(self.device)

                row_logits, col_logits = self.model(fpv, topview)
                loss = self.criterion(row_logits, col_logits, row_labels, col_labels)

                total_loss += loss.item()

                # Convert to predictions
                row_probs = torch.sigmoid(row_logits).cpu().numpy()
                col_probs = torch.sigmoid(col_logits).cpu().numpy()

                all_row_preds.append(row_probs)
                all_col_preds.append(col_probs)
                all_row_targets.append(row_labels.cpu().numpy())
                all_col_targets.append(col_labels.cpu().numpy())

        # Compute metrics.
        # v6 post-mortem: threshold=0.3 showed 0% exact match throughout 50 epochs,
        # but diagnostic sweep revealed threshold=0.4 gives 60% on the same checkpoint.
        # The model was learning correctly — the evaluation threshold was wrong.
        # v7 fix: use 0.4. This matches the model's learned output scale (active
        # row/col sigmoids cluster around 0.5–0.9; threshold=0.3 admitted too many
        # false positives while threshold=0.4 correctly excludes inactive rows/cols).
        row_preds_all = np.vstack(all_row_preds)
        col_preds_all = np.vstack(all_col_preds)
        row_targets_all = np.vstack(all_row_targets)
        col_targets_all = np.vstack(all_col_targets)

        val_threshold = 0.4
        row_preds_binary = (row_preds_all > val_threshold).astype(int)
        col_preds_binary = (col_preds_all > val_threshold).astype(int)

        exact_match_scores = []
        jaccard_scores = []
        cardinality_scores = []
        row_letters = 'ABCDEFGH'
        for i in range(len(row_preds_binary)):
            row_pred_idx = np.where(row_preds_binary[i])[0]
            col_pred_idx = np.where(col_preds_binary[i])[0]
            row_target_idx = np.where(row_targets_all[i])[0]
            col_target_idx = np.where(col_targets_all[i])[0]

            pred_wells = [{'well_row': row_letters[r], 'well_column': int(c + 1)}
                          for r in row_pred_idx for c in col_pred_idx]
            target_wells = [{'well_row': row_letters[r], 'well_column': int(c + 1)}
                            for r in row_target_idx for c in col_target_idx]

            exact_match_scores.append(exact_match(pred_wells, target_wells))
            jaccard_scores.append(jaccard_similarity(pred_wells, target_wells))
            cardinality_scores.append(cardinality_accuracy(pred_wells, target_wells))

        metrics = {
            'val_loss':        total_loss / len(self.val_loader),
            'exact_match':     np.mean(exact_match_scores),
            'jaccard':         np.mean(jaccard_scores),
            'cardinality_acc': np.mean(cardinality_scores),
        }

        return metrics['val_loss'], metrics

    def train(self, start_epoch: int = 0):
        """Main training loop."""
        logger.info(f"Starting training: epochs {start_epoch+1}–{self.epochs}")

        for epoch in range(start_epoch, self.epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.epochs}")

            # Train
            train_loss = self.train_epoch()
            logger.info(f"Train loss: {train_loss:.4f}")

            # Validate
            val_loss, metrics = self.validate()
            logger.info(f"Val loss: {val_loss:.4f}")
            logger.info(f"Exact match: {metrics['exact_match']:.4f}")
            logger.info(f"Jaccard: {metrics['jaccard']:.4f}")
            logger.info(f"Cardinality acc: {metrics['cardinality_acc']:.4f}")

            # LR scheduler step
            self.scheduler.step()

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint(epoch, metrics)
                logger.info(f"Saved best checkpoint at epoch {epoch + 1}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

    def _save_checkpoint(self, epoch: int, metrics: Dict):
        """Save model checkpoint compatible with weights_only=True.

        All metric values are explicitly cast to native Python float/int so
        that no numpy scalar types end up in the pickle stream.  This makes
        the checkpoint loadable with ``torch.load(..., weights_only=True)``
        (the safe default in PyTorch >= 2.6) without any allowlist.
        """
        checkpoint = {
            'epoch': int(epoch),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            # Cast every metric to a plain Python float — keeps numpy out of pickle
            'val_loss':        float(metrics['val_loss']),
            'exact_match':     float(metrics['exact_match']),
            'jaccard':         float(metrics['jaccard']),
            'cardinality_acc': float(metrics['cardinality_acc']),
            # Model config — required to reconstruct the architecture for inference/diagnostics
            'model_config': getattr(self, '_model_config', {}),
        }

        path = self.output_dir / 'best.pt'
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train pipette well detection model')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory with videos')
    parser.add_argument('--labels', type=str, required=True, help='Labels JSON file')
    parser.add_argument('--output', type=str, default='checkpoints/', help='Output directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num_frames', type=int, default=8, help='Frames per video')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size. Must be a multiple of 14 for DINOv2 (e.g. 224, 336, 448, 518)')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    parser.add_argument('--backbone', type=str, default='dinov2', choices=['dinov2', 'resnet18'],
                        help='Backbone architecture (dinov2 or resnet18 for CPU training)')
    parser.add_argument('--focal_alpha', type=float, default=0.75, help='Focal loss alpha — weight for positive class (default 0.75; was 0.25)')
    parser.add_argument('--focal_gamma', type=float, default=0.0, help='Focal loss gamma — focusing parameter (default 0.0 = plain weighted BCE; use 2.0 for focal loss)')
    parser.add_argument('--col_weight', type=float, default=2.0,
                        help='Loss weight for column head (default 2.0 — upweights column discrimination which lags row)')
    parser.add_argument('--lora_rank', type=int, default=4,
                        help='LoRA adapter rank (default 4; was 8 — reduced to limit overfitting on 80 samples)')
    parser.add_argument('--temporal_layers', type=int, default=1,
                        help='Temporal attention layers (default 1; was 2 — reduced to cut trainable params)')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience in epochs (default 20; was 10 in v4 — too aggressive)')
    parser.add_argument('--well_consistency_weight', type=float, default=0.2,
                        help='Weight for outer-product well-level consistency loss (default 0.2; was 0.5 in v4)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')

    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Load dataset
    logger.info("Loading dataset...")
    dataset = PipetteWellDataset(
        args.data_dir,
        args.labels,
        num_frames=args.num_frames,
        img_size=args.img_size,
        augment=True
    )

    # Split into train/val
    n_val = int(len(dataset) * args.val_split)
    n_train = len(dataset) - n_val

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )

    logger.info(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    # Build model
    logger.info(f"Building model with {args.backbone} backbone...")
    use_dinov2 = (args.backbone == 'dinov2')
    model = DualViewFusion(
        num_rows=8,
        num_columns=12,
        shared_backbone=True,
        use_lora=True,
        lora_rank=args.lora_rank,
        temporal_layers=args.temporal_layers,
        use_dinov2=use_dinov2,
        img_size=args.img_size,
    )
    model = model.to(device)

    model_config = {
        'num_rows': 8,
        'num_columns': 12,
        'shared_backbone': True,
        'use_lora': True,
        'lora_rank': args.lora_rank,
        'temporal_layers': args.temporal_layers,
        'use_dinov2': use_dinov2,
        'img_size': args.img_size,
    }

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=args.output,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        col_weight=args.col_weight,
        patience=args.patience,
        well_consistency_weight=args.well_consistency_weight,
    )
    trainer._model_config = model_config

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        logger.info(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        trainer.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        trainer.best_val_loss = ckpt['val_loss']
        start_epoch = ckpt['epoch'] + 1
        logger.info(f"Resumed at epoch {start_epoch} | best val_loss={ckpt['val_loss']:.4f} | exact_match={ckpt['exact_match']*100:.1f}%")

    # Train (from start_epoch if resuming)
    trainer.train(start_epoch=start_epoch)

    logger.info("Training complete!")


if __name__ == '__main__':
    sys.exit(main())
