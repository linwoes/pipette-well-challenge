#!/usr/bin/env python3
"""
Pipette Well Detection - Training Script
========================================

Trains DINOv2-ViT-B/14 backbone with LoRA adapters on pipette well detection task.

Usage:
  python train.py --data_dir ./data --labels labels.json --epochs 50 --output checkpoints/
  python train.py --data_dir ./data --labels labels.json --val_split 0.2 --batch_size 4 --device cuda:0
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import yaml
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

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
        self.augment = augment

        # Load labels
        with open(labels_path, 'r') as f:
            self.labels = json.load(f)

        logger.info(f"Loaded {len(self.labels)} labels")

        # Build augmentation pipeline
        if augment:
            self.transform = A.Compose([
                A.RandomResizedCrop(img_size, img_size, scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.3),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, p=0.5),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.GaussNoise(var_limit=(5, 20), p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], is_check_shapes=False)
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], is_check_shapes=False)

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

        # Build video paths
        fpv_stem = label_entry['clip_id_FPV']
        topview_stem = label_entry['clip_id_Topview']

        fpv_path = self.data_dir / f"{fpv_stem}.mp4"
        topview_path = self.data_dir / f"{topview_stem}.mp4"

        # Load and preprocess videos
        try:
            fpv_frames = load_video(str(fpv_path), max_frames=self.num_frames)  # (N, H, W, 3)
            topview_frames = load_video(str(topview_path), max_frames=self.num_frames)

            # Align to same length
            fpv_frames, topview_frames = align_clips(fpv_frames, topview_frames)

            # Preprocess each frame
            fpv_processed = np.array([preprocess_frame(f, size=(self.img_size, self.img_size)) for f in fpv_frames])
            topview_processed = np.array([preprocess_frame(f, size=(self.img_size, self.img_size)) for f in topview_frames])

            # Apply augmentation per frame
            fpv_list = []
            for f in fpv_processed:
                # Convert from [0, 1] to [0, 255] for albumentations
                f_uint8 = (f * 255).astype(np.uint8)
                augmented = self.transform(image=f_uint8)
                fpv_list.append(augmented['image'])

            topview_list = []
            for f in topview_processed:
                f_uint8 = (f * 255).astype(np.uint8)
                augmented = self.transform(image=f_uint8)
                topview_list.append(augmented['image'])

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
        patience: int = 10,
        grad_clip: float = 1.0,
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

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer: only trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

        self.optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

        # Loss function
        self.criterion = WellDetectionLoss(gamma=2.0, alpha=0.25)

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

        # Compute metrics
        row_preds_all = np.vstack(all_row_preds)
        col_preds_all = np.vstack(all_col_preds)
        row_targets_all = np.vstack(all_row_targets)
        col_targets_all = np.vstack(all_col_targets)

        # Threshold at 0.5
        row_preds_binary = (row_preds_all > 0.5).astype(int)
        col_preds_binary = (col_preds_all > 0.5).astype(int)

        # Compute well predictions
        exact_match_scores = []
        jaccard_scores = []
        for i in range(len(row_preds_binary)):
            row_pred_idx = np.where(row_preds_binary[i])[0]
            col_pred_idx = np.where(col_preds_binary[i])[0]
            row_target_idx = np.where(row_targets_all[i])[0]
            col_target_idx = np.where(col_targets_all[i])[0]

            pred_wells = [{'well_row': chr(ord('A') + r), 'well_column': int(c + 1)}
                         for r in row_pred_idx for c in col_pred_idx]
            target_wells = [{'well_row': chr(ord('A') + r), 'well_column': int(c + 1)}
                           for r in row_target_idx for c in col_target_idx]

            exact_match_scores.append(exact_match(pred_wells, target_wells))
            jaccard_scores.append(jaccard_similarity(pred_wells, target_wells))

        metrics = {
            'val_loss': total_loss / len(self.val_loader),
            'exact_match': np.mean(exact_match_scores),
            'jaccard': np.mean(jaccard_scores),
            'cardinality_acc': np.mean([cardinality_accuracy(
                [{'well_row': chr(ord('A') + r), 'well_column': int(c + 1)}
                 for r in np.where(row_preds_binary[i])[0] for c in np.where(col_preds_binary[i])[0]],
                [{'well_row': chr(ord('A') + r), 'well_column': int(c + 1)}
                 for r in np.where(row_targets_all[i])[0] for c in np.where(col_targets_all[i])[0]]
            ) for i in range(len(row_preds_binary))])
        }

        return metrics['val_loss'], metrics

    def train(self):
        """Main training loop."""
        logger.info(f"Starting training for {self.epochs} epochs")

        for epoch in range(self.epochs):
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
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': metrics['val_loss'],
            'exact_match': metrics['exact_match'],
            'jaccard': metrics['jaccard'],
            'cardinality_acc': metrics['cardinality_acc'],
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
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

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

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Build model
    logger.info("Building model...")
    model = DualViewFusion(
        num_rows=8,
        num_columns=12,
        shared_backbone=True,
        use_lora=True,
        lora_rank=8,
    )
    model = model.to(device)

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
    )

    # Train
    trainer.train()

    logger.info("Training complete!")


if __name__ == '__main__':
    sys.exit(main())
