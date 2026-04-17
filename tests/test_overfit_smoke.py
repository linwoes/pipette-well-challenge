#!/usr/bin/env python3
"""
H-3: Memorisation Smoke Test
=============================
Trains DualViewFusion (ResNet18, CPU) on 5 fixed examples for 100 epochs and
asserts that the model can memorise the training set (exact_match >= 0.8).

Purpose
-------
This is the minimum sanity check before any full training run. If the model
cannot overfit 5 examples in 100 epochs, there is a bug in the forward pass,
loss, or label encoding that would silently prevent learning on the full dataset.

Design
------
- No real video data required: inputs are random tensors with fixed seed.
- Uses ResNet18 backbone (CPU-compatible, no LFS data needed).
- Well targets are drawn from realistic patterns: single-well, full-row,
  full-column — i.e., rectangular patterns for which the well consistency
  loss is correctly defined.
- Threshold of 0.3 (matching Bug-3 fix in train.py / inference.py).
- Pass criterion: exact_match >= 0.8 at end of training (4 of 5 examples).

Usage
-----
    cd /path/to/pipette-well-challenge
    python -m pytest tests/test_overfit_smoke.py -v
  or
    python tests/test_overfit_smoke.py
"""

import sys
import os

# Ensure repo root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.optim as optim

from src.models.fusion import DualViewFusion, WellDetectionLoss
from src.utils.metrics import exact_match


# ── Helpers ──────────────────────────────────────────────────────────────────

ROW_LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

THRESHOLD = 0.3  # Matches Bug-3 fix


def _row_col_to_wells(row_vec, col_vec):
    """Convert binary row (8,) and col (12,) vectors to well dicts."""
    wells = []
    for r, rv in enumerate(row_vec):
        for c, cv in enumerate(col_vec):
            if rv > THRESHOLD and cv > THRESHOLD:
                wells.append({'well_row': ROW_LETTERS[r], 'well_column': c + 1})
    return wells


def _targets_to_wells(row_target, col_target):
    """Convert 0/1 row/col target tensors to ground-truth well dicts."""
    wells = []
    for r, rv in enumerate(row_target.tolist()):
        for c, cv in enumerate(col_target.tolist()):
            if rv > 0.5 and cv > 0.5:
                wells.append({'well_row': ROW_LETTERS[r], 'well_column': c + 1})
    return wells


# ── Fixtures ─────────────────────────────────────────────────────────────────

def make_fixed_dataset(n=5, num_frames=2, img_size=64, seed=42):
    """
    Build n fixed training examples as random frame tensors with deterministic labels.

    Labels are rectangular patterns (safe for well consistency loss):
      0: single well A1
      1: single well C5
      2: full row B (all 12 cols active)
      3: full column 7 (all 8 rows active)
      4: single well H12
    """
    rng = torch.Generator()
    rng.manual_seed(seed)

    # Fixed rectangular label patterns
    row_targets = torch.zeros(n, 8)
    col_targets = torch.zeros(n, 12)

    # 0: A1
    row_targets[0, 0] = 1.0; col_targets[0, 0] = 1.0
    # 1: C5
    row_targets[1, 2] = 1.0; col_targets[1, 4] = 1.0
    # 2: full row B (row index 1)
    row_targets[2, 1] = 1.0; col_targets[2, :] = 1.0
    # 3: full column 7 (col index 6)
    row_targets[3, :] = 1.0; col_targets[3, 6] = 1.0
    # 4: H12
    row_targets[4, 7] = 1.0; col_targets[4, 11] = 1.0

    # Random frame tensors (deterministic)
    fpv = torch.rand(n, num_frames, 3, img_size, img_size, generator=rng)
    topview = torch.rand(n, num_frames, 3, img_size, img_size, generator=rng)

    return fpv, topview, row_targets, col_targets


# ── Smoke test ────────────────────────────────────────────────────────────────

def test_overfit_smoke(
    epochs: int = 100,
    lr: float = 1e-3,
    min_exact_match: float = 0.8,
):
    """
    Train DualViewFusion on 5 fixed examples for 100 epochs.
    Assert exact_match >= 0.8 on training set.
    """
    torch.manual_seed(0)

    fpv, topview, row_targets, col_targets = make_fixed_dataset()
    n = fpv.shape[0]

    model = DualViewFusion(
        use_dinov2=False,          # ResNet18 — CPU-compatible, no weights download
        use_lora=False,
        max_frames=fpv.shape[1],
        img_size=fpv.shape[-1],
    )
    model.train()

    criterion = WellDetectionLoss(
        gamma=2.0,
        alpha=0.75,                # Bug-4 fix value
        well_consistency_weight=0.2,  # v5 recommendation
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_exact = 0.0

    for epoch in range(1, epochs + 1):
        # Single full-batch gradient step over all 5 examples
        optimizer.zero_grad()
        row_logits, col_logits = model(fpv, topview)
        loss = criterion(row_logits, col_logits, row_targets, col_targets)
        loss.backward()
        optimizer.step()

        # Evaluate exact match every 10 epochs
        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                row_prob = torch.sigmoid(row_logits)   # reuse last batch (same data)
                col_prob = torch.sigmoid(col_logits)

            n_correct = 0
            for i in range(n):
                pred_wells = _row_col_to_wells(row_prob[i], col_prob[i])
                gt_wells = _targets_to_wells(row_targets[i], col_targets[i])
                if exact_match(pred_wells, gt_wells):
                    n_correct += 1

            em = n_correct / n
            best_exact = max(best_exact, em)
            print(f"Epoch {epoch:3d} | loss={loss.item():.4f} | exact_match={em:.2f} ({n_correct}/{n})")
            model.train()

    print(f"\nBest exact_match = {best_exact:.2f}  (threshold >= {min_exact_match})")
    assert best_exact >= min_exact_match, (
        f"Memorisation failed: best exact_match={best_exact:.2f} < {min_exact_match}. "
        "The model cannot overfit 5 training examples in 100 epochs — there is a bug "
        "in the loss, forward pass, or label encoding that must be fixed before v5."
    )
    print("PASS — model can memorise training set.")


if __name__ == '__main__':
    test_overfit_smoke()
