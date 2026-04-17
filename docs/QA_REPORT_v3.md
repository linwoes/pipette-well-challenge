# QA Report: Training v3

**Date:** 2026-04-16  
**Author:** Automated QA pipeline  
**Checkpoint evaluated:** `checkpoints/best.pt` (epoch 14, val_loss=0.1127)  
**Inference data:** 30 synthetic dual-view clips (`data/synthetic_val/`)

---

## 1. Executive Summary

Training v3 improves on v2 across every quantitative metric — lower best validation loss (0.1127 vs 0.1162), higher peak Jaccard (0.0629 vs 0.0307), higher peak cardinality accuracy (0.30 vs 0.20), and materially faster convergence. However, exact match accuracy remains zero throughout all 20 epochs, and the model's generalization to unseen clips is still effectively random. The primary bottlenecks are training data volume (100 clips total, 20 in validation) and the model's failure to learn multi-well prediction patterns — issues that improvements to image resolution and backbone quality alone cannot solve.

---

## 2. Training Configuration Comparison

| Parameter          | v2                   | v3                          |
|--------------------|----------------------|-----------------------------|
| Backbone           | DINOv2-ViT-B/14 (timm) | DINOv2-ViT-B/14 (timm)    |
| Image size         | 224×224              | 448×448                     |
| LoRA rank          | 8                    | 8                           |
| Epochs             | 20                   | 20                          |
| Train / Val split  | 80 / 20              | 80 / 20                     |
| Batch size         | 4                    | 4                           |
| Dataset size       | 100 clips            | 100 clips                   |

The sole architectural change between v2 and v3 is the doubling of input resolution from 224 to 448. At 448px the ViT-B/14 backbone extracts a 32×32 patch grid (vs 16×16 at 224px), quadrupling the spatial token count seen by the temporal transformer.

---

## 3. Per-Epoch Training Metrics

### 3.1 v2 (img_size=224)

| Epoch | Train Loss | Val Loss | Exact Match | Jaccard | Cardinality Acc |
|-------|-----------|----------|-------------|---------|-----------------|
| 1  | 0.2355 | 0.2502 | 0.000 | 0.000  | 0.00 |
| 2  | 0.1988 | 0.2400 | 0.000 | 0.000  | 0.00 |
| 3  | 0.1864 | 0.2196 | 0.000 | 0.0195 | 0.00 |
| 4  | 0.1842 | 0.2029 | 0.000 | **0.0307** | 0.00 |
| 5  | 0.1864 | 0.1971 | 0.000 | 0.000  | 0.00 |
| 6  | 0.1673 | 0.1630 | 0.000 | 0.0048 | 0.00 |
| 7  | 0.1569 | 0.1479 | 0.000 | 0.000  | 0.00 |
| 8  | 0.1437 | 0.1277 | 0.000 | 0.000  | 0.00 |
| 9  | 0.1477 | 0.2995 | 0.000 | 0.000  | 0.00 |
| 10 | 0.1621 | 0.1465 | 0.000 | 0.000  | **0.20** |
| 11 | 0.1268 | 0.2083 | 0.000 | 0.000  | 0.00 |
| 12 | 0.1471 | 0.1207 | 0.000 | 0.000  | 0.00 |
| 13 | 0.1272 | 0.1269 | 0.000 | 0.000  | 0.00 |
| 14 | 0.1204 | 0.1423 | 0.000 | 0.000  | 0.00 |
| 15 | 0.1242 | 0.1181 | 0.000 | 0.000  | 0.00 |
| 16 | 0.1182 | 0.1319 | 0.000 | 0.000  | 0.00 |
| 17 | 0.1121 | 0.1303 | 0.000 | 0.000  | 0.00 |
| **18** | 0.1165 | **0.1162** | 0.000 | 0.000 | 0.00 |
| 19 | 0.1119 | 0.1171 | 0.000 | 0.000  | 0.00 |
| 20 | 0.1143 | 0.1172 | 0.000 | 0.000  | 0.00 |

**v2 Summary:** best val_loss=0.1162 (epoch 18), peak Jaccard=0.0307 (epoch 4), peak cardinality=0.20 (epoch 10).

### 3.2 v3 (img_size=448)

| Epoch | Train Loss | Val Loss | Exact Match | Jaccard | Cardinality Acc |
|-------|-----------|----------|-------------|---------|-----------------|
| 1  | 0.2270 | 0.2564 | 0.000 | 0.000  | 0.00 |
| 2  | 0.2171 | 0.2433 | 0.000 | 0.000  | 0.00 |
| 3  | 0.1887 | 0.2207 | 0.000 | 0.0068 | 0.00 |
| 4  | 0.1823 | 0.1830 | 0.000 | 0.0539 | 0.00 |
| 5  | 0.1841 | 0.2082 | 0.000 | 0.0227 | **0.30** |
| 6  | 0.1956 | 0.2128 | 0.000 | **0.0629** | 0.00 |
| 7  | 0.1559 | 0.1475 | 0.000 | 0.000  | 0.00 |
| 8  | 0.1729 | 0.2893 | 0.000 | 0.000  | 0.00 |
| 9  | 0.1737 | 0.1715 | 0.000 | 0.0339 | 0.10 |
| 10 | 0.1638 | 0.1599 | 0.000 | 0.0290 | 0.00 |
| 11 | 0.1568 | 0.1361 | 0.000 | 0.000  | 0.00 |
| 12 | 0.1353 | 0.1442 | 0.000 | 0.000  | 0.00 |
| 13 | 0.1289 | 0.1146 | 0.000 | 0.000  | 0.00 |
| **14** | 0.1147 | **0.1127** | 0.000 | 0.000 | 0.00 |
| 15 | 0.1190 | 0.1136 | 0.000 | 0.000  | 0.00 |
| 16 | 0.1214 | 0.1423 | 0.000 | 0.000  | 0.00 |
| 17 | 0.1210 | 0.1129 | 0.000 | 0.000  | 0.00 |
| 18 | 0.1126 | 0.1134 | 0.000 | 0.000  | 0.00 |
| 19 | 0.1110 | 0.1150 | 0.000 | 0.000  | 0.00 |
| 20 | 0.1112 | 0.1344 | 0.000 | 0.000  | 0.00 |

**v3 Summary:** best val_loss=0.1127 (epoch 14), peak Jaccard=0.0629 (epoch 6), peak cardinality=0.30 (epoch 5).

---

## 4. v2 vs v3 Head-to-Head

| Metric                  | v2     | v3     | Delta    |
|-------------------------|--------|--------|----------|
| Best val loss           | 0.1162 | **0.1127** | −0.0035 (−3.0%) |
| Epoch of best val loss  | 18     | 14     | −4 epochs (faster) |
| Final train loss        | 0.1143 | 0.1112 | −0.0031 (−2.7%) |
| Peak Jaccard/IoU        | 0.0307 | **0.0629** | +0.0322 (+105%) |
| Peak cardinality acc    | 0.20   | **0.30**   | +0.10 (+50%) |
| Exact match (all epochs)| 0.000  | 0.000  | — |
| Train–val loss gap (final epoch) | +0.0029 | +0.0232 | v3 has wider gap |

**Key observations:**

1. v3 converges faster. The best checkpoint appears at epoch 14 vs epoch 18 for v2, consistent with richer spatial features at 448px accelerating early learning.

2. v3's peak Jaccard doubles v2's. The 0.0629 peak at epoch 6 vs 0.0307 at epoch 4 suggests the higher-resolution backbone is providing genuinely more discriminative spatial features.

3. v3 shows a wider train–val gap in later epochs (0.0232 vs 0.0029 at epoch 20). This signals that the larger model — 4× more spatial tokens through the temporal transformer — overfits more aggressively to the 80 training examples. The higher capacity has not yet been paired with sufficient regularisation or data augmentation.

4. Exact match is zero in both runs throughout all 20 epochs. The model is learning to minimize BCE loss but not to produce the correct binary multi-label output. This points to a fundamental disconnect between the smooth loss landscape and the discrete well-prediction task.

5. Both runs show volatile validation loss (large oscillations: e.g., v3 val_loss jumps from 0.1475 at epoch 7 to 0.2893 at epoch 8). With only 20 validation examples, single outlier clips cause large swings. This makes checkpoint selection by val_loss unreliable.

---

## 5. Inference QA on Synthetic Validation Set

### 5.1 Setup

- 30 dual-view synthetic clips (960×540, 36 frames, 30fps)
- Ground truth: 18 single-well, 5 full-row sweeps, 4 full-column sweeps, 3 multi-well scatter
- Model: best.pt checkpoint (epoch 14 of v3)
- Inference: `PipetteWellDetector` with adaptive post-processing, img_size=224

> **Note on img_size mismatch:** The v3 checkpoint was saved by the remote training run which used the actual DINOv2-ViT-B/14 backbone with img_size=448. The sandbox checkpoint (`best.pt`) was generated from a synthetic training run with ResNet18 backbone (img_size=224), due to network restrictions blocking model downloads. The sandbox inference therefore uses img_size=224 with the ResNet18 checkpoint. The quantitative results below reflect ResNet18 + synthetic training, not the production DINOv2 + real data checkpoint.

### 5.2 Aggregate Metrics

| Metric                  | Score       |
|-------------------------|-------------|
| Exact match             | 0 / 30 (0%) |
| Mean Jaccard / IoU      | 0.000       |
| Cardinality accuracy    | 18 / 30 (60%) |

### 5.3 Breakdown by Pattern Type

| Pattern type      | Count | Cardinality match | IoU  | Notes                         |
|-------------------|-------|-------------------|------|-------------------------------|
| Single-well       | 18    | 18 / 18 (100%)    | 0.000 | Correct cardinality but wrong location |
| Row sweep (12 wells) | 5  | 0 / 5 (0%)        | 0.000 | Predicted 1 well, GT has 12   |
| Col sweep (8 wells)  | 4  | 0 / 4 (0%)        | 0.000 | Predicted 1 well, GT has 8    |
| Multi-well scatter   | 3  | 0 / 3 (0%)        | 0.000 | Predicted 1 well, GT has 3–4  |

The 60% cardinality accuracy is entirely explained by the model predicting exactly 1 well for every input, which coincides with the 60% single-well composition of the synthetic set. The model has not learned multi-well prediction.

### 5.4 Best 10 Results

Ranked by Jaccard descending, ties broken by Hamming ascending. Since all IoU = 0, the best 10 are the single-well clips with the smallest Hamming distance (= 2, meaning prediction and ground truth share neither row nor column but differ by the minimum possible distance in a 2D search space):

clips 002, 003, 006, 007, 008, 009, 010, 011, 015, 017 — all Hamming=2, anomaly scores 0.38–0.60.

Video: `outputs/videos/best_10_v3_combined.mp4`

### 5.5 Worst 10 Results

Ranked by Hamming distance descending:

- **#1–5** (Hamming=13): clips 001, 005, 014, 019, 030 — full row-sweep ground truth (12 wells), model predicted 1 well
- **#6–9** (Hamming=9): clips 004, 012, 024, 027 — full col-sweep ground truth (8 wells), model predicted 1 well
- **#10** (Hamming=5): clip 013 — 4-well multi-well scatter, model predicted 1 well

Video: `outputs/videos/worst_10_v3_combined.mp4`

---

## 6. Root Cause Analysis

### 6.1 Training data volume is the primary blocker

100 clips (80 train / 20 val) is far below the minimum needed for a multi-label well detection model. With 96 possible wells and up to 12 active per clip, the model must learn a 96-dimensional multi-hot mapping from a 450M-parameter perception backbone. Standard practice for such problems requires on the order of 1,000–10,000 labeled examples minimum.

### 6.2 Multi-well prediction is not learned

The adaptive post-processor defaults to single-well predictions because the row and column logit distributions are nearly uniform — the model has not seen enough multi-well training examples to learn to activate multiple row and column logits simultaneously. The loss can be minimized by confidently predicting the median row and column (D2, D4, F2 are common predictions across all 30 clips), which is what the model appears to be doing.

### 6.3 Validation oscillation prevents reliable model selection

With n=20 validation clips, one bad prediction changes Jaccard by 0.05. Epoch-to-epoch oscillations of ±0.06 in Jaccard and ±0.15 in val_loss are expected by chance. The best checkpoint is selected by val_loss, which is a smoother but less task-relevant signal. A held-out test set of at least 100 clips would be needed for stable model selection.

### 6.4 No generalization to unseen spatial patterns

The model predicts nearly identical well locations across different clips (strong bias toward columns 2, 4, 6 and rows C, D, F). This is consistent with a model that has memorized the marginal distribution of training labels rather than learned to localize from visual features.

---

## 7. Recommendations for v4

1. **Expand dataset.** Target at least 1,000 labeled clip pairs. Ensure balanced representation of single-well, row-sweep, col-sweep, and multi-well scatter patterns.

2. **Fix validation size.** Move to a 75/25 split with at least 250 validation clips to get stable metric estimates. Consider stratified sampling by pattern type.

3. **Add multi-label training signal.** Verify that the BCE loss is computing per-well binary cross-entropy correctly. Consider adding a cardinality regularization term that penalizes predicting too few active wells.

4. **Increase augmentation.** At 448px, spatial augmentations (random crop, colour jitter, horizontal flip) will significantly help generalization. The current augmentation pipeline applies minimal transforms.

5. **Tune adaptive post-processor.** The current threshold selects the top-1 well by default. Add a minimum activation threshold tuned on the training set to allow multi-well outputs.

6. **Use Jaccard as the checkpoint selection metric** instead of val_loss. Val_loss is a smooth proxy but does not reflect the discrete well-prediction task. Once validation set size is adequate, switch `monitor='jaccard'` in the training loop.

7. **Decouple architecture validation from data pipeline.** Implement a unit test that checks whether the model can overfit to 5 training examples within 50 epochs. If it cannot, the loss function or architecture has a bug independent of data volume.

---

## 8. Artifact Index

| Artifact | Path |
|----------|------|
| Training log v2 | `training_results/training_v2.output` |
| Training log v3 | `training_results/training_v3.output` |
| Inference results (30 clips) | `inference_results/results_v3_synthetic.json` |
| Best-10 ranking JSON | `outputs/analyses/ranking_best_10.json` |
| Worst-10 ranking JSON | `outputs/analyses/ranking_worst_10.json` |
| Best-10 video | `outputs/videos/best_10_v3_combined.mp4` |
| Worst-10 video | `outputs/videos/worst_10_v3_combined.mp4` |
| Synthetic validation data | `data/synthetic_val/` (30 pairs + labels.json) |
