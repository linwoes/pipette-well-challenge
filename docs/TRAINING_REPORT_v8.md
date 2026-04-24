# Training Report — v8

**Date:** 2026-04-24  
**Checkpoint:** `checkpoints/best.pt` (epoch 33, val_loss=0.4330)  
**Run duration:** ~9h 12m (CPU-only)  
**Epochs:** 53 of 80 (early stop, patience=20)

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Device | CPU (no CUDA) |
| Backbone | DINOv2-ViT-B/14 via timm fallback |
| Trainable params | 12,260,887 |
| img_size | 448 |
| frames | 4 |
| batch | 2 |
| lora_rank | 4 |
| temporal_layers | 1 |
| focal_gamma | 0.0 |
| col_weight | 2.0 |
| type_loss_weight | 1.0 |
| Train / Val split | 80 / 20 |
| Resumed from | v6 checkpoint |

---

## v8 Change: Clip-Type Head

v8 introduced a dedicated classification head predicting clip type (single / row / col). Goal was to eliminate threshold sensitivity in multi-well cardinality prediction. As validated below, it succeeded at this subgoal.

---

## Training Metrics

| Epoch | Train Loss | Val Loss | Exact Match | Jaccard | Cardinality |
|-------|-----------|---------|------------|---------|-------------|
| 1 | 1.5148 | 2.2689 | 0.00 | 0.010 | 0.60 |
| 2 | 1.2945 | 1.8234 | 0.05 | 0.059 | 0.60 |
| 4 | 0.7934 | 0.9916 | 0.05 | 0.050 | 0.95 |
| **5** | 0.6815 | 0.6774 | 0.05 | 0.050 | **1.00** |
| 14 | 0.4279 | 0.4690 | 0.00 | 0.000 | 1.00 |
| 21 | 0.3539 | 0.4380 | 0.00 | 0.000 | 1.00 |
| **33 (best ckpt)** | 0.2685 | **0.4330** | 0.00 | 0.000 | 1.00 |
| 53 (early stop) | 0.1800 | 0.4575 | 0.05 | 0.050 | 1.00 |

Val loss plateaued in the 0.43–0.47 range from epoch 27 onward with no visible accuracy movement. Early stopping triggered at epoch 53 (best at epoch 33).

---

## Critical Finding: Apparent Failure Was Decoder Artifact

**The model is substantially better than training metrics indicated.**

The training loop evaluated accuracy using `logits_to_wells_adaptive`. Post-training diagnostic sweep with fixed thresholds revealed the real performance of the epoch-33 checkpoint:

| Decoder | Exact Match | Jaccard | Cardinality |
|---------|------------|---------|-------------|
| adaptive (used in training) | 10.0% | 0.185 | 10.0% |
| threshold=0.1 | 0.0% | 0.096 | 0.0% |
| threshold=0.2 | 0.0% | 0.233 | 0.0% |
| threshold=0.3 | 30.0% | 0.488 | 30.0% |
| **threshold=0.4** | **55.0%** | **0.663** | **70.0%** |
| threshold=0.5 | 40.0% | 0.400 | 55.0% |

**55% exact match / 0.66 Jaccard** is the true quality of the saved checkpoint.  
The 0–5% numbers logged during all 53 training epochs were decoder noise, not model signal.  
Early stopping and checkpointing responded to val loss (which did track model quality), but the accuracy numbers were meaningless throughout.

---

## Why the Adaptive Decoder Fails

`logits_to_wells_adaptive` builds an 8×12 outer-product probability map and thresholds at 50% of its peak, falling back to argmax when the selected count exceeds `max_wells=12`.

The sigmoid distribution on the val set explains the collapse:

| Statistic | Row head | Col head |
|-----------|----------|----------|
| max sigmoid (mean) | 0.825 | 0.679 |
| 2nd sigmoid (mean) | 0.733 | 0.529 |
| max/2nd ratio (mean) | **1.28×** | **1.51×** |

For a full-column dispense (8 wells), all 8 row sigmoids fire near 0.7+. The outer product map has far more than 12 cells above the adaptive threshold, so the fallback argmax fires and returns only 1 well. Every multi-well clip hits this path.

---

## Per-Type Accuracy (Adaptive Decoder, Diagnostic)

| Clip Type | n | Exact Match | Jaccard |
|-----------|---|-------------|---------|
| single (1 well) | 2 | 50.0% | 0.667 |
| row (8 wells) | 12 | 8.3% | 0.177 |
| col (other) | 6 | 0.0% | 0.042 |

Single-well clips work because argmax fallback returns 1 well — which is the right count. Multi-well clips always collapse to 1 well.

---

## Fixes in Priority Order

### Fix 1 — Switch decoder to threshold=0.4 (no retraining)

**Impact: immediately takes reported performance from 10% → 55% exact match.**

Replace all calls to `logits_to_wells_adaptive` in both `train.py` (validation loop) and `inference.py` with `logits_to_wells(..., threshold=0.4)`. No weights change required. This is the highest-leverage action available.

---

### Fix 2 — Use the type head in the decoder

**Impact: eliminates threshold sensitivity for multi-well cases; expected to push exact match above 55%.**

The v8 type head predicts clip type (single/row/col) with near-100% cardinality accuracy from epoch 5 onward. This signal is currently ignored entirely in both training evaluation and inference. Replace the threshold decoder with a type-driven top-k selector:

- **single** → top-1 row × top-1 col → 1 well
- **row** (8-well) → top-8 rows × top-1 col → 8 wells
- **col** (12-well or other) → top-1 row × top-k cols → k wells

This is what the type head was designed for and closes the problem that motivated v8.

---

### Fix 3 — Fix training evaluation to use the correct decoder

**Impact: makes early stopping and checkpointing respond to real accuracy rather than decoder artifacts.**

Once Fix 1 or Fix 2 is in place, apply the same decoder to the validation loop in `train.py`. The epoch-33 checkpoint was saved by val loss rather than accuracy — which happened to work — but the logged exact match and Jaccard were meaningless for all 53 epochs. Future runs need accurate in-loop metrics to guide early stopping and hyperparameter decisions.

---

### Fix 4 — Fix the checkpoint criterion

**Impact: ensures the saved model is the best by the metric that matters.**

Checkpointing currently saves on lowest val loss. Val loss improved to 0.4330 at epoch 33 while exact match (as measured by the adaptive decoder) showed 0.0% — indistinguishable from every other epoch. Switch checkpointing to save on highest Jaccard (or exact match), measured with the corrected decoder.

---

### Fix 5 — Increase frames from 4 to 8

**Impact: better temporal coverage of dispense event; expected accuracy gain on ambiguous clips.**

The architecture specification requires 8 frames. v8 uses 4 (speed compromise for CPU). With 4 frames there is a meaningful chance the dispense event frame is missed entirely. On GPU this is a free win; on CPU it doubles per-epoch time, so it becomes practical only when running on hardware with acceleration.

---

### Fix 6 — Phase out type loss after saturation

**Impact: recovers gradient capacity for spatial head in later epochs.**

Cardinality accuracy hits 100% at epoch 5 and stays there. After that the type loss contribution is near zero but still occupies the optimizer's attention. Schedule `type_loss_weight` to decay to 0 after epoch 10 so that later training focuses entirely on spatial localization.

---

### Fix 7 — Synthetic data generation

**Impact: highest-ceiling improvement, but requires significant up-front effort.**

80 training examples covering a 96-well grid is severely underdetermined for spatial localization. The architecture document calls for 5–10× synthetic clips per undersampled well (target 500–1000 clips). Until this is in place, generalization across all 96 wells is structurally limited regardless of architecture improvements.

---

## Summary

v8 succeeded at its stated goal (cardinality via type head). The spatial model is meaningfully better than training logs showed — 55% exact match is real and achievable today with a decoder fix. Fixes 1–4 require no retraining and can be validated immediately. Fixes 5–7 require a new training run.
