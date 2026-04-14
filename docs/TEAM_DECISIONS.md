# Team Decisions Log: Transfyr AI Pipette Well Challenge

**Date:** April 14, 2026  
**Compiled by:** Cross-functional team (Data Scientist, Architect, ML Scientist, QA Engineer)  
**Purpose:** Document key architectural decisions, drivers, rationale, and open questions

---

## Decision 1: Primary Architecture Selection (Deep Learning vs. Classical CV)

**Decision:** Use **Deep Learning End-to-End** (Architecture 2) as primary approach.

**Driver:** Architect (with input from ML Scientist)

**Rationale:**
1. **Best accuracy-to-effort ratio:** With 100 labeled samples available, DL should achieve 85–95% accuracy on held-out set (vs. 80–90% for classical CV)
2. **Inference speed:** ~500ms per sample on GPU vs. 1.6–3.1 sec for classical CV (100–10× speedup)
3. **Robustness to variation:** Neural networks inherently robust to lighting, color, minor occlusion via learned feature representations
4. **Sample efficiency via transfer learning:** ImageNet pre-trained ResNet-18 provides strong spatial initialization
5. **Implementation maturity:** Standard PyTorch pipeline with well-tested techniques (ResNet, multi-label classification, augmentation)

**Alternative Considered:** Classical CV (geometric pipeline)
- Pros: Interpretable, no training required, generalizable to different plate formats
- Cons: Brittle calibration, cascading failures, lighting-sensitive color thresholding
- Reason Not Chosen: Camera calibration errors compound through pipeline; single point of failure

**Fallback Strategy:** If DL accuracy stalls <85% on validation, immediately pivot to **Hybrid approach** (geometric priors + learned features)

**Cross-Reference:** `docs/ARCHITECTURE.md`, Section "Architect's Recommendation"

---

## Decision 2: Backbone Architecture (ResNet-18)

**Decision:** Use **ResNet-18** (torchvision.models.resnet18, ImageNet pre-trained)

**Driver:** ML Scientist (with input from Data Scientist)

**Rationale:**
1. **Parameter efficiency:** 11M parameters is lightweight for 100-sample dataset (prevents catastrophic overfitting)
2. **Transfer learning proven:** ImageNet pre-training gives strong spatial priors directly applicable to microplate imaging
3. **Computational efficiency:** ~20–50ms inference on GPU; comfortable within 2-min budget
4. **Simplicity:** Fewer hyperparameters to tune than ViT or EfficientNet (reduces pathological overfitting risk)

**Alternative Considered:** MobileNetV3 (~5M params)
- Data Scientist recommended for parameter efficiency
- ML Scientist decision: ResNet-18 preferred because:
  - 100 samples not large enough to benefit from MobileNetV3 compression (inference not bottlenecked)
  - ResNet-18 has broader transfer-learning support and more pre-trained checkpoints
  - MobileNetV3's depthwise separable convolutions can reduce feature quality for fine-grained spatial tasks

**Training Strategy:** Freeze layer1, layer2 for first epoch; unfreeze after 3 epochs of stable validation loss

**Cross-Reference:** `docs/ML_STACK.md`, Section "1.2 Backbone Model"

---

## Decision 3: Output Head Architecture (Factorized Row + Column)

**Decision:** Use **two independent classification heads**: 8-class row + 12-class column, each with sigmoid activation.

**Driver:** ML Scientist (with input from Data Scientist on cardinality handling)

**Rationale:**
1. **Reduces overfitting:** Predicts 20 independent outputs (8+12) instead of 96 combinations; critical with only 100 samples
2. **Enforces geometric structure:** Multi-well outputs must respect plate geometry (all selected wells in same row OR same column)
3. **Multi-label semantics:** Sigmoid activation allows simultaneous activation of multiple rows/columns (supports 1-well, 8-well, 12-channel operations naturally)
4. **Interpretability:** Row and column scores independently inspectable; easier debugging

**Alternative Considered:** Direct 96-class multi-label with per-well sigmoid
- Pros: Maximally expressive; no geometric assumptions
- Cons: 96 independent parameters from ~100 samples (severe overfitting); requires custom constraint-decoding post-processing
- Reason Not Chosen: Factorized approach reduces parameter explosion while maintaining expressiveness

**Output Format:**
```
Row logits: [8] (one score per row A-H)
Column logits: [12] (one score per column 1-12)
Final prediction: All (row_i, col_j) pairs where row_logits[i] > 0.5 AND col_logits[j] > 0.5
```

**Cross-Reference:** `docs/ML_STACK.md`, Section "1.6 Output Head Architecture"

---

## Decision 4: Loss Function (Focal Loss)

**Decision:** Use **Focal Loss** (γ=2.0) with weighted BCE per output head.

**Driver:** ML Scientist (with input from Data Scientist on class imbalance)

**Rationale:**
1. **Addresses severe class imbalance:** 5–15 wells with zero training samples; 10:1 to 50:1 imbalance ratio worst case
2. **Down-weights easy negatives:** Focal loss down-weights high-confidence correct predictions; focuses learning on hard (rare) examples
3. **Dynamic weighting:** Adapts during training (vs. static class weights which are prone to manual miscalibration)
4. **Proven on imbalanced data:** Superior to standard BCE + fixed class weights on microplate detection tasks

**Formula:**
```
Focal Loss: FL(p_t) = -α(1 - p_t)^γ log(p_t)
where p_t = p if y=1, else 1-p

Applied per head:
loss_row = focal_loss(row_logits, row_labels, gamma=2.0)
loss_col = focal_loss(col_logits, col_labels, gamma=2.0)
total_loss = loss_row + loss_col (or weighted average)
```

**Alternative Considered:** Standard BCE with class-weighted sampling
- Pros: Simpler implementation; widely available
- Cons: Requires manual weight computation (prone to errors); doesn't adapt dynamically
- Reason Not Chosen: Focal loss proven more effective on this type of long-tail problem

**Implementation:** Use `torchvision.ops.sigmoid_focal_loss` with α=0.25, γ=2.0

**Cross-Reference:** `docs/ML_STACK.md`, Section "1.7 Loss Function"

---

## Decision 5: Data Augmentation Strategy

**Decision:** Implement **mandatory three-tier augmentation** (temporal >> geometric > photometric)

**Driver:** Data Scientist (with ML Scientist validation)

**Rationale:**
Given only 100 samples, augmentation is non-optional:
1. **Temporal augmentation (highest priority):**
   - Frame offset: ±3 frames from ground-truth
   - Speed jitter: 0.9–1.1× playback speed
   - Frame interpolation for slow-motion clips
   - **Expected boost:** 3–5× effective dataset size

2. **Geometric augmentation:**
   - Plate rotation: ±8°
   - Crop/zoom: 90–110% of well ROI
   - Affine transforms: shear ±5°, scale ±5%
   - **Expected boost:** 2–3× additional diversity

3. **Photometric augmentation:**
   - Brightness jitter: ±15% intensity
   - Contrast jitter: 1.0–1.3×
   - Gaussian blur: 1–3 pixel kernel
   - Hue shift: ±10° in HSV

**Implementation:** Use `albumentations` library (1.3+) with Compose API

**Expected Outcome:** Effective training set size of 500–1000 samples vs. 100 raw

**Cross-Reference:** `docs/DATA_ANALYSIS.md`, Section "B.1 Data Augmentation Pipeline"; `docs/ML_STACK.md` Section "1.8"

---

## Decision 6: Dual-View Fusion Strategy

**Decision:** Use **late concatenation fusion** (process FPV and top-view independently, fuse after feature extraction)

**Driver:** Data Scientist + Architect (with ML Scientist implementation)

**Rationale:**
1. **Different coordinate systems:** FPV is perspective-distorted; top-view is orthogonal. Early fusion would confuse representations
2. **View complementarity:** FPV strong on temporal motion; top-view strong on spatial localization
3. **Symmetric treatment:** Both views processed through identical ResNet backbones; features fused before output heads

**Architecture:**
```
FPV stream:    (B,3,224,224) ──> ResNet-18 ──> (B,512)
                                                    │
Top-view:      (B,3,224,224) ──> ResNet-18 ──> (B,512)  ──> Concat ──> (B,1024) ──> FC ──> Heads
```

**Alternative Considered:** Early fusion (concatenate frames before backbone)
- Pros: Simpler pipeline; single backbone
- Cons: Confuses coordinate systems; requires explicit geometric alignment
- Reason Not Chosen: Data Scientist analysis showed top-view should be primary signal for well ID

**Synchronization:** Explicit temporal alignment required before inference (use cross-correlation to find optimal frame offset)

**Cross-Reference:** `docs/DATA_ANALYSIS.md`, Section "3. Dual-View Signal Analysis"; `docs/ML_STACK.md` Section "1.9"

---

## Decision 7: Validation & Evaluation Metrics

**Decision:** Use **multi-faceted metric suite** (not just accuracy):

**Primary Metrics:**
1. **Per-well accuracy:** Recall and precision for each well (identify which wells model struggles with)
2. **Cardinality-aware F1:** Separate F1 per cardinality class (1-channel, 8-channel, 12-channel)
3. **Exact-match accuracy:** All predicted wells match ground truth exactly
4. **Localization MAE:** Mean absolute error in pixel space (for grid alignment validation)

**Secondary Metrics:**
1. **Jaccard index:** Intersection / union of predicted and ground-truth well sets (multi-label evaluation)
2. **Confidence calibration:** Compare model confidence vs. actual per-well accuracy
3. **Generalization to unseen wells:** Track accuracy on wells with 0, 1, or 2 training examples separately

**Driver:** QA Engineer (with Data Scientist & ML Scientist input)

**Rationale:**
- Single accuracy metric is misleading for imbalanced multi-label problems
- Per-well metrics expose systematic biases (e.g., "model always confuses A5 and A6")
- Cardinality-aware metrics catch multi-label failure modes
- Confidence analysis enables safe deployment thresholding

**Test Set Protocol:**
- Never touch test set during development
- Report three numbers on final hold-out evaluation:
  1. Exact-match accuracy (%)
  2. Cardinality-wise accuracy (separate for 1/8/12-well ops)
  3. Per-well coverage (% of wells in test set correctly identified)

**Cross-Reference:** `docs/DATA_ANALYSIS.md`, Section "D.1 Metrics"; `docs/QA_STRATEGY.md`, Section throughout

---

## Decision 8: Training/Validation/Test Split

**Decision:** Use **70/20/10 split** with stratification by cardinality.

**Driver:** Data Scientist (with QA Engineer validation)

**Rationale:**
- **70 training samples:** Enough to fine-tune ResNet-18 with regularization
- **20 validation samples:** Sufficient for hyperparameter selection and early stopping
- **10 held-out test samples:** Final evaluation (tight, but reflects real evaluation window)
- **Stratification:** Ensure train/val/test each contain single-channel, 8-channel, 12-channel operations proportionally

**Alternative Considered:** Leave-one-sample-out (LOOCV) on validation set
- Pros: Maximizes validation data utilization
- Cons: Computationally expensive (101 training runs)
- Reason Not Chosen: 70/20/10 provides reasonable validation budget without excessive computation

**Important Caveat:** With stratification, effective training size drops to 50–60 samples. **Severe overfitting risk** — regularization (dropout, weight decay, early stopping) is essential.

**Cross-Reference:** `docs/DATA_ANALYSIS.md`, Section "6. Statistical Concerns"; `docs/ML_STACK.md` Training section

---

## Decision 9: GPU vs. CPU Inference

**Decision:** **GPU required for inference** (CPU fallback not supported).

**Driver:** ML Scientist (with input from Architect on latency)

**Rationale:**
- **GPU inference:** ~500ms per sample (ResNet-18 on NVIDIA GPU)
- **CPU inference:** ~2–3 seconds per sample (unacceptable for 2-min budget across 10 samples)
- **Deployment assumption:** Testing performed on GPU instance (AWS g4dn, GCP A100, or similar)

**Optimization Considerations:**
- Use mixed precision (AMP) during inference to reduce memory footprint
- Batch processing: Process all 10 samples together if possible (further speedup via vectorization)
- Model quantization: Optional post-training quantization (int8) for additional speed, negligible accuracy loss

**Target Hardware:** NVIDIA A100 or V100 (80GB or 32GB VRAM)

**Cross-Reference:** `docs/ARCHITECTURE.md`, Section "1. Latency & Performance"; `docs/ML_STACK.md` Section "1.5"

---

## Decision 10: Handling Unseen Wells (Zero-Shot Challenge)

**Decision:** **Attempt prediction with confidence gating** (don't silently output low-confidence guesses).

**Driver:** QA Engineer (with Data Scientist risk analysis)

**Rationale:**
Data Scientist analysis shows 5–15 wells will have zero training samples. Standard ML approach (predict anyway) is inadequate.

**Recommended Protocol:**
1. **Attempt prediction** using learned feature representations (transfer learning from seen wells)
2. **Check confidence:** If max confidence <0.4 for an unseen well, **do NOT output it**
3. **Output valid wells only:** Return only wells with confidence ≥0.4
4. **Log rationale:** Write to stderr: `"WARNING: Well A5 was unseen in training. Skipped due to confidence 0.35."`
5. **Accept partial credit:** Predicting 8 correct wells is better than 10 predictions with 2 spurious guesses

**Acceptance Criteria:**
- **Seen wells** (≥2 training examples): Target ≥85% accuracy
- **Rare wells** (1 training example): Target ≥70% accuracy
- **Unseen wells**: Treat as generalization risk; don't penalize if accuracy 50%+

**Coverage Report (required before hold-out):**
- Data Scientist generates heatmap of well coverage in training data
- QA Engineer requests hold-out metadata: which wells present, any unseen?
- Sets appropriate baseline expectations

**Cross-Reference:** `docs/QA_STRATEGY.md`, Section "2. Hold-Out Set Analysis"; Section "2.4 What the Model Should Do"

---

## Open Questions & Risks Flagged by QA

### Risk 1: Generalization Gap (Flagged by Data Scientist)
**Issue:** Expected 30 percentage point gap between training accuracy and held-out test accuracy.  
**Why:** Severe overfitting risk with 100 samples and 11M parameter backbone.  
**Mitigation:** Aggressive regularization (dropout 0.5, weight decay 1e-5, early stopping), data augmentation, lightweight architecture.  
**Acceptance:** Conservative estimate: 70–80% test accuracy; optimistic: 85–92%.

### Risk 2: Temporal Synchronization (Flagged by Data Scientist)
**Issue:** FPV and top-view recorded asynchronously; frame offset ±1–2 frames typical.  
**Why:** Mismatch between ground-truth frame in FPV and aligned frame in top-view.  
**Mitigation:** Use cross-correlation of optical flow to find peak alignment; include frame offset as model input.  
**Testing:** Unit test to validate frame alignment on validation set samples.

### Risk 3: Multi-Label Cardinality Misclassification (Flagged by QA)
**Issue:** Model may predict 7 wells when ground truth is 8-channel (row operation).  
**Why:** Insufficient training examples of 8-channel and 12-channel operations.  
**Mitigation:** Explicit cardinality prediction head; post-processing constraint: if cardinality=8, enforce 8 wells in single row.  
**Testing:** Separate cardinality accuracy metrics (1, 8, 12-channel separately).

### Risk 4: Domain Shift on Hold-Out (Flagged by QA)
**Issue:** Hold-out set may contain lighting, plate handling, or equipment condition shifts not in training.  
**Why:** Testing conducted at different time/location than training; natural environmental variation.  
**Mitigation:** Augmentation covers lighting (brightness/contrast), plate rotation (±8°), blur (motion artifacts).  
**Contingency:** If hold-out accuracy <60%, diagnostic steps: per-view ablation, temporal alignment validation, failure case review.

### Risk 5: Class Bias / Majority Well Overprediction (Flagged by QA)
**Issue:** Model always predicts well A1 (most frequent in training) regardless of input.  
**Why:** Severe class imbalance; cross-entropy loss dominated by majority class.  
**Mitigation:** Focal loss down-weights easy negatives; class weighting for rare wells; validation metric per well.  
**Testing:** Confusion matrix per well; flag if any well represents >30% of predictions.

---

## Acceptance Criteria (Final Hold-Out Evaluation)

**All of the following must be satisfied:**

### Pre-Evaluation Checklist
- [ ] Validation accuracy ≥85% on 20 held-out validation samples
- [ ] Per-well cardinality accuracy ≥80% (1-channel, 8-channel, 12-channel separately)
- [ ] Inference latency <1 second per sample on target GPU
- [ ] All edge case unit tests passing
- [ ] Data Scientist coverage report completed (well coverage heatmap, unseen wells identified)

### On Hold-Out Evaluation (10 unknown samples)
- [ ] All 10 samples produce valid JSON output
- [ ] JSON schema validated: wells array with row (A-H) and column (1-12) keys
- [ ] Exact-match accuracy ≥80% (or ≥70% if hold-out contains many unseen wells)
- [ ] Cardinality-wise accuracy ≥75% (separate scoring for 1/8/12-well operations)
- [ ] Total runtime ≤20 minutes (2 minutes per sample average)
- [ ] No runtime errors, exceptions, or timeouts
- [ ] Predictions consistent with visual inspection of videos

### Success Metrics
- [ ] ≥90% accuracy on well coordinates (ideal case)
- [ ] <10 second inference per sample (ideally <1 second)
- [ ] Reproducible results (git-tracked, seeded random states)
- [ ] Generalizes to well row/column combinations not seen in training

---

## Contingency Plans

### If DL Accuracy Stalls <85% on Validation
1. **Week 1:** Implement Hybrid approach (geometric priors + learned features)
2. Estimate: 5–10% accuracy recovery via geometric constraints
3. Target: ≥95% on validation, ≥92% on hold-out

### If Temporal Synchronization Fails
1. Re-evaluate frame offset estimation algorithm
2. Try multiple offset candidates; ensemble predictions
3. Fallback: Use FPV clip only (if top-view unavailable)

### If Hold-Out Contains Unseen Wells & Accuracy <70%
1. This is expected; not an automatic failure
2. Evaluate per-well: which wells misclassified?
3. If well coverage issues: report as data limitation, not model limitation
4. If model systematically confuses adjacent wells: add contrastive loss in retraining

---

## Appendix: Decision Timeline

| Date | Decision | Owner | Status |
|------|----------|-------|--------|
| Apr 14 | Architecture selection (DL primary) | Architect | FINAL |
| Apr 14 | Backbone architecture (ResNet-18) | ML Scientist | FINAL |
| Apr 14 | Output heads (factorized row/col) | ML Scientist | FINAL |
| Apr 14 | Loss function (focal loss) | ML Scientist | FINAL |
| Apr 14 | Augmentation strategy | Data Scientist | FINAL |
| Apr 14 | Dual-view fusion (late concat) | Data Scientist | FINAL |
| Apr 14 | Evaluation metrics (multi-faceted) | QA Engineer | FINAL |
| Apr 14 | Train/val/test split (70/20/10) | Data Scientist | FINAL |
| Apr 14 | GPU requirement (no CPU fallback) | ML Scientist | FINAL |
| Apr 14 | Unseen well protocol (confidence gating) | QA Engineer | FINAL |

---

**Document Status:** FINAL  
**Last Updated:** April 14, 2026  
**Next Review:** After hold-out evaluation completion
