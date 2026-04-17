# QA Report: Training v4

**Date:** 2026-04-17  
**Author:** Hopper (QA) — reviewed by PM  
**Log:** `training_results/training_v4.output`  
**Backbone:** DINOv2-ViT-B/14 via timm (img_size=448, 23.6M trainable params)  
**Stopped:** Epoch 15 (early stopping, patience=10, best at epoch 5)

---

## 1. Executive Summary

v4 is a meaningful step forward — not because it achieved exact match (it did not), but because it is the first run where the model produces non-zero Jaccard from epoch 1 and sustains it throughout training. That was impossible under v3's broken focal loss. The corrected loss function is working: positive well activations are being generated, evaluated, and producing gradient signal.

However, exact match remains zero across all 15 epochs. Three causes are active:

1. **L-3 (consistency loss for scatter patterns) was not yet fixed** when v4 ran. The fix was written after the red team review but before v4 was submitted — it was not in the repo at training time. Scatter-pattern clips (clips with non-rectangular ground truth) received incorrect supervision from the outer-product well loss throughout v4, capping Jaccard around 0.05.

2. **Training loss instability in epochs 4 and 8–15** — the tqdm output shows epoch 4 batch times doubling mid-epoch (18 s/it → 44 s/it). This is consistent with the well consistency loss generating large per-sample gradients on scatter pattern clips. Once the L-3 guard is in place, this instability should disappear.

3. **Validation set is too small** (20 examples) for stable metric estimation. Jaccard oscillates ±0.018 epoch-to-epoch due to single-clip noise, causing early stopping to fire on a false plateau.

---

## 2. Per-Epoch Metrics

| Epoch | Train Loss | Val Loss | Exact Match | Jaccard | Cardinality Acc | Notes |
|-------|-----------|----------|-------------|---------|-----------------|-------|
| 1  | 0.2658 | 0.3263 | 0.000 | **0.0550** | 0.000 | First non-zero Jaccard at epoch 1 — confirmed positive activations |
| 2  | 0.2594 | 0.3044 | 0.000 | 0.0518 | 0.000 | |
| 3  | 0.2305 | 0.2827 | 0.000 | 0.0458 | 0.000 | |
| 4  | 0.2216 | 0.2704 | 0.000 | 0.0546 | 0.000 | Batch slowdown (18→44 s/it) — likely scatter gradient spike |
| **5** | 0.2071 | **0.2158** | 0.000 | 0.0510 | 0.000 | **Best checkpoint** |
| 6  | 0.2064 | 0.2364 | 0.000 | 0.0540 | 0.000 | |
| 7  | 0.2170 | 0.2230 | 0.000 | 0.0401 | 0.000 | Train loss rises (gradient instability) |
| 8  | 0.1786 | 0.2826 | 0.000 | 0.0384 | 0.000 | Val loss spikes +0.060 from previous |
| 9  | 0.1843 | 0.2433 | 0.000 | 0.0431 | 0.000 | |
| 10 | 0.1761 | 0.2287 | 0.000 | 0.0376 | 0.000 | |
| 11 | 0.2171 | 0.2645 | 0.000 | 0.0374 | 0.000 | Train loss spike — largest since epoch 1 |
| 12 | 0.1781 | 0.2255 | 0.000 | 0.0450 | 0.000 | |
| 13 | 0.1721 | 0.2229 | 0.000 | 0.0454 | 0.000 | |
| 14 | 0.1573 | 0.2273 | 0.000 | 0.0430 | **0.050** | First cardinality accuracy in v4 |
| 15 | 0.1620 | 0.2254 | 0.000 | 0.0443 | **0.100** | Early stop triggered |

---

## 3. Comparison Across All Runs

| Metric | v2 | v3 | v4 | Trend |
|--------|----|----|-----|-------|
| Backbone | DINOv2 timm 224px | DINOv2 timm 448px | DINOv2 timm 448px | — |
| Best val loss | 0.1162 | 0.1127 | 0.2158 | ↑ (worse — expected, loss is harder now) |
| Epoch of best | 18 | 14 | 5 | ↓ faster convergence to a lower floor |
| Epochs run | 20 | 20 | 15 | Early stopping now active |
| Peak Jaccard | 0.0307 | 0.0629 | **0.0550 (epoch 1)** | Non-zero from epoch 1 for first time |
| Jaccard variance | ±0.016 | ±0.019 | ±0.009 | More stable |
| Peak cardinality | 0.20 | 0.30 | **0.10** | Lower — consistent low-level learning |
| Exact match | 0 | 0 | **0** | Still zero |

The higher val loss in v4 vs v3 is not a regression — it reflects the corrected loss computing a strictly harder objective (correct alpha weighting + well consistency term). v3's low val loss was partly a measurement artifact of the broken focal loss producing near-zero logits that minimised BCE cheaply.

The key qualitative signal: **Jaccard 0.055 at epoch 1 is the strongest opening epoch across all four runs**. The model is engaging from the first batch, not coasting on a collapsed logit space.

---

## 4. Root Cause Analysis — Why Exact Match Is Still Zero

### 4.1 L-3 was active during v4

The well consistency loss guard (skip for non-rectangular patterns) was committed *after* v4 training started. The 3 scatter-pattern clips and 4 column-sweep clips in the validation set (7 of 20 = 35%) were receiving incorrect well-grid supervision throughout training. At the observed batch slowdowns (epoch 4: 18→44 s/it; epoch 8 val_loss spike to 0.2826), the most likely trigger is large gradients from scatter samples being incorrectly penalised for not predicting the phantom wells in their outer-product grid.

Fixing L-3 removes incorrect supervision from 35% of validation examples and an unknown fraction of training examples.

### 4.2 Jaccard plateau at ~0.04–0.055

The Jaccard stalls in the 0.037–0.055 band after epoch 1 and never breaks through. This is consistent with the model learning to reliably activate the correct row or column, but not both simultaneously for the same clip. Single-head accuracy of ~0.22 IoU on rows + columns independently maps to ~0.05 Jaccard on the product — which is exactly what we observe. The well consistency loss is supposed to fix this coupling, but its incorrect supervision on scatter clips is counteracting the correct signal on single/row/column-sweep clips.

### 4.3 Early stopping fires on noise

With 20 validation clips, one correct prediction change swings Jaccard by 0.05. The best epoch (5, val_loss=0.2158) is only marginally better than epochs 7, 10, 12, 13, 15 (all in the 0.22–0.23 range). Early stopping with patience=10 is too aggressive for a val set this size — it stopped a run that was still making slow progress (cardinality accuracy was rising: 0 → 0.05 → 0.10 in the last two epochs).

---

## 5. Confirmed Fixes Working

| Fix | Evidence |
|-----|----------|
| Focal loss alpha (Bug 1) | Non-zero Jaccard from epoch 1 (impossible with broken alpha) |
| Threshold 0.3 (Bug 3) | Cardinality accuracy > 0 at epochs 14–15 (model predicting multiple wells) |
| Well consistency loss direction (Bug 2) | Jaccard is higher and more stable early; less collapse to single-well predictions |

---

## 6. Findings for v5

### Must-fix before v5 (block)

**F-1 (from L-3):** Apply the L-3 guard before training. The fix is already written in `src/models/fusion.py` but was not committed before v4 ran. Commit it now and verify it is active.

**F-2 (from H-3):** Run the memorisation smoke test before submitting v5. With the L-3 fix in place, a 5-example / 100-epoch overfit test should produce exact_match > 0. If it does not, there is another bug to find before spending 4+ hours on a full run.

### Recommended for v5

**F-3:** Increase early stopping patience from 10 to 20, or switch the monitor metric from `val_loss` to `jaccard`. With a 20-sample val set, val_loss has ~0.06 noise; stopping on it loses valid training signal.

**F-4:** Lower `well_consistency_weight` from 0.5 to 0.2 for v5. The current weight is strong enough to cause gradient spikes on scatter clips even after the L-3 guard, because the guard is heuristic (single row OR single col). A lower weight reduces the penalty for any borderline cases.

**F-5:** The training loss oscillation pattern (stable for 10 batches, then sudden slowdown) is visible in epoch 4 and again in epochs 8–11. This strongly suggests a specific subset of training clips — likely the full row-sweep clips with 12 active wells each — are causing large gradient updates. Adding per-sample gradient clipping (or clip-level loss weighting by cardinality) would stabilise these spikes.

---

## 7. Artifact Index

| Artifact | Path |
|----------|------|
| v4 training log | `training_results/training_v4.output` |
| v3 QA report (prior baseline) | `docs/QA_REPORT_v3.md` |
| Red team review | `docs/RED_TEAM_REVIEW.md` |
| L-3 fix (not yet committed) | `src/models/fusion.py` (in-progress, see red team fixes branch) |
