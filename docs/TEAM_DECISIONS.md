# Team Decisions Log: Transfyr AI Pipette Well Challenge

**Date:** April 14, 2026 (Final)  
**Updated:** April 15, 2026 (Version Divergence Fix — DINOv2 is Primary)  
**Compiled by:** Cross-functional team (Data Scientist, Architect, ML Scientist, QA Engineer)  
**Purpose:** Document architectural decisions and implementation rationale

---

## Decision 11: [FINAL] Primary Backbone — DINOv2-ViT-B/14 + LoRA Fine-Tuning (April 15, 2026)

**Date:** 2026-04-15  
**Driver:** ML Scientist + Architect  
**Status:** FINAL — **implemented in code**; correcting documentation to match reality

**Decision:** The **primary production backbone is DINOv2-ViT-B/14 with LoRA adapters (r=8, α=16).**

This is **NOT ResNet-18.** This is **NOT VideoMAE.** This is **DINOv2.**

**Rationale:**

1. **"Deeper spatial intuition":** DINOv2 pre-trained on 142M images with spatial self-supervision directly learns coordinate geometry — precisely what well localisation requires
2. **LoRA efficiency:** only ~33K trainable parameters vs. 11M for full ResNet-18 fine-tuning — prevents catastrophic overfitting on N=100 samples
3. **Patch structure:** 14×14 patches preserve spatial layout; ResNet's global pooling destroys it (7×7 final features cannot distinguish 96 wells)
4. **Few-shot superiority:** DINOv2 ViT-B outperforms ResNet-50 by +8 percentage points at 10-shot (published benchmarks, Oquab et al. 2023)
5. **Self-supervised pre-training:** DINO contrastive learning (not ImageNet supervised) learns semantic geometry without label bias

**Backbone status (April 2026):** DINOv2 weights are downloaded from Hugging Face Hub at training start. The earlier `LegacyResNet18Backbone` random-init fallback was kept while sandbox proxy restrictions blocked weight downloads; once those constraints lifted, the fallback was removed entirely (commit d5b8f04). DINOv2 is now the only backbone path — there is no `use_dinov2` flag and no `--backbone` CLI argument.

**Architecture:**
```
FPV Video (8 frames) → DINOv2-ViT-B/14 (frozen) + LoRA r=4 → patch features
                    → Temporal Transformer (1 layer) → pooled
                                                      ↓ Late Fusion (MLP)
Top-view Video (8 frames) → DINOv2-ViT-B/14 (frozen) + LoRA r=4 → patch features
                          → Temporal Transformer (1 layer) → pooled
                                                             ↓
                                       MLP fusion → 256-dim
                                                             ↓
                       Row head (8) + Col head (12) + Type head (3-class: single/row/col)
```

**Code locations:**
- `src/models/backbone.py`: `DINOv2Backbone` class — no fallback, raises if weights can't load
- `src/models/fusion.py`: `DualViewFusion` (returns row, col, type logits) + `WellDetectionLoss`
- `train.py`: training loop, hybrid checkpoint criterion, type-conditioned val decoder

**VideoMAE Status:** Discussed as a **future/alternative** for stronger temporal modeling, but NOT in the codebase and NOT the primary path.

**ResNet-18 Status:** Deprecated (2015) and **removed** from the codebase in commit d5b8f04 along with the `use_dinov2` flag and `LegacyResNet18Backbone` class. Retained in `docs/ML_STACK.md` only as comparative justification.

---

## Empirical Data Decisions (April 15, 2026)

Based on real dataset analysis (`DATA_ANALYSIS_EMPIRICAL.md`), two new decisions are added to address data-specific implementation:

---

### Decision R-1: Train/Val Split Strategy (Plate-Based)

**Driver:** Data Scientist (empirical analysis)

**Date:** 2026-04-15

**Decision:** Use **plate-based stratification** for train/val split to eliminate data leakage:
- **Train:** Plates 1, 2, 3, 4, 9 (72 clips)
- **Validation:** Plates 5, 10 (28 clips)

**Rationale:**

1. **No plate leakage:** Clips from the same physical plate never appear in both train and val. Plates are independent experimental units (different lighting, camera setup, operator handling).

2. **Real-world simulation:** Mirrors production scenario: model trained on past plates, validated on new plates.

3. **Eliminates instrument bias:** Different plates have different lighting, well geometry, camera alignment. Plate-level split prevents learning plate-specific artifacts rather than true well-detection capability.

4. **Natural operation stratification:** Split preserves operation type distribution:
   - Train: 6 row-sweep, 12 column-sweep, 54 single-well
   - Val: 7 row-sweep, 0 column-sweep, 21 single-well
   (Note: Column-sweeps absent from val; mitigate with focal loss or class weighting)

5. **Simplicity:** Whole-plate split is easier to implement than stratified random split and avoids subtle leakage bugs.

**Supersedes:** Random val_split=0.2 currently in train.py (still present as default; plate split requires explicit indices passed to DataLoader)

**Implementation:**
```python
# In train.py
train_plate_indices = [plates[i] for i in [1, 2, 3, 4, 9]]
val_plate_indices = [plates[i] for i in [5, 10]]
train_dataset = create_dataset(indices=train_plate_indices)
val_dataset = create_dataset(indices=val_plate_indices)
```

**Testing:** Verify no overlap: `set(train_indices) & set(val_indices) == {}`

---

### Decision R-2: well_column String Type Handling

**Driver:** Data Scientist (empirical analysis)

**Date:** 2026-04-15

**Decision:** All code must treat `well_column` as convertible string/int via `int()` conversion:
- **Labels (source):** Stored as STRING ("1", "2", ..., "12")
- **Model output:** Integers (1-12)
- **Validation:** `output_formatter.validate_output()` must handle both string and int inputs

**Rationale:**

1. **Schema mismatch reality:** JSON labels use string values ("1"), but model outputs integer logits (1-12). Conversion must happen somewhere.

2. **Code locations affected:**
   - `train.py` `_encode_wells()`: Parse labels, convert `well_column` to int for indexing
   - `output_formatter.py` `logits_to_wells()`: Already uses `int(w['well_column']) - 1`; ensure backward compatible
   - `output_formatter.py` `validate_output()`: Must accept both string and int columns when comparing ground truth to predictions

3. **Robustness:** Explicit type coercion avoids silent failures from type mismatches.

**Implementation Examples:**

```python
# Safe parsing (handles both "1" and 1)
col_value = label['well_column']  # Could be "1" or 1
col_int = int(col_value)  # Converts both "1" and 1 to integer 1
col_index = col_int - 1   # Converts to 0-based index (0-11)

# In validation
def validate_output(prediction, ground_truth):
    pred_col = int(prediction['well_column'])
    gt_col = int(ground_truth['well_column'])
    return pred_col == gt_col
```

**Testing:** Add assertion in data loading:
```python
assert isinstance(col, str) and col.isdigit(), \
    f"well_column must be numeric string; got {col}"
```

---

## Executive Summary: Key Architectural Decisions

The team identified critical gaps in the original ResNet-18 single-frame approach and implemented comprehensive architectural revisions to address them.

**Critical Decisions Made:**
- Architecture Primary: DINOv2-ViT-B/14 + LoRA + Temporal Transformer (not ResNet-18)
- Fusion Architecture: Late Fusion (explicit mandate for proper geometric composition)
- Synthetic Data Strategy: 3D Blender + VideoMAE fine-tuning (10K+ samples)
- Primary Metric: ECE < 0.10 (calibration and reproducibility over raw accuracy)
- Confident Refusal Gate: Model must abstain on uncertain inputs
- Temporal Modeling Mandatory: Temporal Transformer required for event-level understanding
- Branch Preservation: ResNet-18 baseline on `baseline/resnet-cv-pipeline`
- 3DGS as Research Track: Added for glare-heavy scenarios

---

## Decision 1: [REVISED] Architecture Primary — DINOv2-ViT-B/14 + LoRA + Temporal Transformer

**Previous Decision:** Use ResNet-18 backbone with single-frame classification

**Current Decision (CORRECTED):** Use **DINOv2-ViT-B/14 (frozen) + LoRA adapters (r=8) + Temporal Transformer** as primary architecture

**CRITICAL CLARIFICATION:** Earlier versions of this document incorrectly stated "VideoMAE" as the primary. **The code implements DINOv2, not VideoMAE.** This decision log is now corrected to match the implementation.

**Driver:** ML Scientist + Architect

**Architectural Analysis:**

- ResNet-18 (2015) and baseline approaches are insufficient for Physical AI pipetting
- Transfyr expects Vision-Language-Action (VLA) approach where motion trajectory is first-class
- Single-frame models cannot distinguish pipette entering vs. leaving (temporal blindness)
- 11M-parameter ResNet on 100 samples will memorize background/lighting, not learn geometry

**Revised Rationale (CORRECTED):**

1. **DINOv2 Foundation Model (2023 SOTA, self-supervised):**
   - DINOv2-ViT-B/14 pre-trained on 142M unlabeled images via self-supervised DINO contrastive learning
   - Learns spatial coordinate geometry directly (not object classification like ImageNet)
   - Explicit patch embeddings: 14×14 grid (196 tokens, 768-dim each) preserves spatial structure
   - LoRA fine-tuning: frozen backbone + low-rank adapters (~33K trainable params) prevent overfitting on N=100
   - Few-shot validated: +8% over ResNet-50 on 10-shot benchmarks
   
2. **Temporal Understanding:**
   - Temporal Transformer with 2–4 attention blocks over ordered frames
   - Replaces frame max-pooling (which destroys temporal order)
   - Models dispense as event: approach → hold → release (not state)
   - Aligns with Transfyr's "Tacit Knowledge" mission: motion intent is causal
   
3. **Data Efficiency via LoRA Fine-tuning:**
   - DINOv2 pre-training on self-supervised coordinate geometry stabilizes learning on 100 real samples
   - LoRA adapters learn task-specific alignment without destroying pre-trained spatial structure
   - 10,000+ synthetic samples = 100× data scale-up (mitigates overfitting crisis)

**Architecture at a Glance:**
```
Input: (T=4-8 ordered frames per view, 224×224×3)
    ↓
Temporal Backbone: DINOv2-ViT-B/14 encoder → (T, 768) feature sequence
    ↓
Temporal Transformer: Cross-attention over frames → Event localization
    ↓
Late Fusion: FPV features ⊗ Top-view features (cross-modal attention)
    ↓
Output Heads: 8-class row + 12-class column with calibrated sigmoid
```

**Fallback Hierarchy:**
1. **Primary:** DINOv2-ViT-B/14 + LoRA + Temporal Transformer
2. **Sandbox fallback:** ResNet-18 (proxy constraints only; not recommended for production)
3. **Future alternative:** VideoMAE (not yet implemented)

---

## Decision 2: [REVISED] Synthetic Data Strategy — 3D Blender + VideoMAE Fine-Tuning

**Previous Decision:** Augmentation only; 100 real samples sufficient with regularization

**Revised Decision:** Generate **10,000+ synthetic dispense events** combining:
- **Option A:** 3D Blender photorealistic simulation (5,000 samples)
- **Option B:** VideoMAE fine-tuning generative sampling (5,000 samples)

**Driver:** Data Scientist

**Architectural Analysis:**

- 100 samples alone represent "a failure of scale" for a Physical AI company
- ResNet-18 (11M params) will memorize lighting/background of those specific 100 videos
- No synthetic data strategy = relying entirely on brittle real data

**Revised Rationale:**

1. **Option A: 3D Blender Simulation (Recommended for coverage)**
   - Model 96-well plate with photorealistic Cycles rendering
   - Render all 96 wells × multiple camera angles (FPV, top-view) × lighting conditions × pipette trajectories
   - ~96 wells × 12 directions × 5 trajectories × 3 lightings = 17,280 synthetic frames
   - Perfect ground-truth coordinates; reproduces geometric extremes (unusual lighting, edge cases)
   - Batch rendering: 100 frames/minute on CPU
   
2. **Option B: VideoMAE Fine-Tuning (Recommended for realism)**
   - Fine-tune VideoMAE-Base on 100 real dispense videos (masked autoencoder objective)
   - Use latent space to generate 100–200 synthetic videos per well (targeting 10,000 total)
   - Generated videos include realistic blur, refraction, liquid dynamics
   - Condition on: well location, pipette angle, lighting, plate orientation
   - Validates domain gap via Fréchet Video Distance (FVD < 20)

3. **Combined Dataset (Mandatory):**
   - 50% Blender synthetic (hard geometric coverage) + 50% VideoMAE-fine-tuned (realistic trajectories) + 100% real
   - Balanced stratification: train on synthetic + real; validate on real only
   - Total training: 10,100 samples (10K synthetic + 100 real) → 100× scale-up

**Training Pipeline:**
```
Real data (100) ──┐
                  ├──> Fine-tune VideoMAE ──> Generate 5K synthetic videos
                  │
Blender render    ├──> 5K photorealistic hard negatives
                  │
Augmentation      └──> 2–3× further diversity (crop, rotate, color jitter)
                       ↓
                  Total: ~10,100 samples for training
                       ↓
                  Validation: Real samples only (measure true generalization)
```

**Validation Protocol:**
- **Stage 1 (Synthetic validation):** Hold-out 10% of synthetic data; measure accuracy (target >90%), ECE (target <0.05)
- **Stage 2 (Real validation):** Hold-out real samples; measure accuracy (60–80% expected), ECE, cross-view agreement
- **Stage 3 (Domain gap):** Compute FVD between synthetic and real; target FVD < 20 (acceptable for synthetic data)

---

## Decision 3: [NEW] Fusion Architecture — Late Fusion (Explicit Mandate)

**Previous Decision:** "Late concatenation fusion" (ambiguous; caused confusion)

**New Decision:** **Late Fusion is MANDATORY and explicitly defined:**

Late Fusion = Process FPV and Top-view independently through spatiotemporal backbones → Fuse at feature level (after spatial extraction) → Fuse via cross-attention or gating

**Driver:** Architect + ML Scientist

**Architectural Analysis:**

- Earlier documentation identified critical contradiction: ML_STACK specified early fusion; TEAM_DECISIONS specified late fusion
- Indicates breakdown in technical alignment on fundamental architectural choice
- In lab environment with precise coordinate requirements, fusion strategy is not "coding detail" but **coordinate system design decision**

**Why Late Fusion is Architecturally Correct:**

1. **Geometric Incompatibility:**
   - FPV uses perspective projection: depth matters (object at z=10cm ≠ z=20cm)
   - Top-view uses orthographic projection: all objects at different z project to same (x, y)
   - Early fusion (feature-map level, before GAP) conflates these incompatible coordinate systems

2. **Late Fusion Respects Projection Differences:**
   - FPV extracts features in perspective space; GAP → (B, 768) perspective-aware features
   - Top-view extracts features in orthographic space; GAP → (B, 768) orthographic-aware features
   - Fusion via cross-attention: FPV attends to top-view features while preserving native coordinate frames
   - No geometric interference early in pipeline

3. **Interpretability & Uncertainty:**
   - FPV confidence and top-view confidence tracked independently
   - Disagreement signals hard cases (wells where views conflict)
   - Enables per-view ablation for failure diagnostics

**Late Fusion Architecture (Preferred):**
```
FPV Video → DINOv2 + Temporal Transformer → (768,) ──┐
                                                       ├──> Cross-attention ──> (768,) ──> Heads
Top-view Video → DINOv2 + Temporal Transformer → (768,) ──┘
```

**Alternative: Gating (if compute-constrained):**
```
FPV features (768,) ──┬──> Dense(768 → 256) ──┬──> Gating ──> α·FPV + (1-α)·TopView
                      │                        │  (learned weight α)
Top-view (768,) ──────> Dense(768 → 256) ───┘
```

**Explicit Commitment:** Early fusion is **FORBIDDEN** for this project. Any design proposing early fusion (concatenate before spatial extraction) must be explicitly overridden.

---

## Decision 4: [REVISED] Primary Metric — Expected Calibration Error (ECE < 0.10)

**Previous Decision:** Maximize exact-match accuracy (% of samples with correct row & column)

**Revised Decision:** **Primary metric is Expected Calibration Error (ECE < 0.10), not raw accuracy**

**Driver:** QA Engineer + ML Scientist

**Architectural Analysis:**

- In science, Reproducibility > Accuracy
- Model can achieve 85% accuracy while being poorly calibrated
- "I don't know with 90% certainty" is more valuable than "overly confident guess"
- Lab environments demand reliable confidence scores over raw performance

**Revised Rationale:**

1. **Calibration is Critical for Lab Deployment:**
   - Operator trusts model's confidence scores more than raw accuracy
   - Poorly calibrated model (high confidence on wrong predictions) causes systematic errors
   - Well-calibrated model enables safe thresholding (defer to human when uncertain)

2. **ECE Definition:**
   ```
   ECE = Σ |accuracy_in_bin - confidence_in_bin| × (# samples in bin / total)
   ```
   - Divides predictions into confidence bins (e.g., 0–0.1, 0.1–0.2, ..., 0.9–1.0)
   - For each bin: measures gap between claimed confidence and empirical accuracy
   - **ECE = 0:** perfectly calibrated (predictions match reality)
   - **ECE = 0.1:** average 10% mismatch between claimed and actual confidence
   - **Target: ECE < 0.10** on held-out real validation set

3. **Reliability Diagrams (Visualization):**
   - Plot predicted confidence (x-axis) vs. actual accuracy (y-axis) for validation set
   - Well-calibrated model is diagonal (perfect agreement)
   - Helps visualize under/overconfidence patterns

**Acceptance Criteria (Revised):**

| Metric | Target | Why |
|--------|--------|-----|
| **ECE (primary)** | < 0.10 | Predictions match reality |
| **Exact-match accuracy (all)** | 60–80% on real validation | Honest generalization measure |
| **Exact-match accuracy (high-conf)** | > 85% at p_max > 0.85 | Minimize false positives |
| **"Confident refusal" rate** | 5–15% | System knows when to defer |
| **Cross-view agreement** | > 70% | Views mostly align |
| **Temporal coherence** | Correlation > 0.6 | Motion signals align with accuracy |

**Metrics Hierarchy:**
1. **First:** ECE < 0.10 (calibration, reproducibility)
2. **Second:** Exact-match accuracy > 85% on high-confidence predictions
3. **Third:** False positive rate < 10% (confident wrong predictions rare)
4. **Fourth:** Exact-match accuracy ~70% on all samples (raw metric for reference)

---

## Decision 5: [NEW] Confident Refusal Gate (Uncertainty Abstention)

**Previous Decision:** Attempt prediction for all samples; threshold confidence at 0.5

**New Decision:** Model must output `{"uncertain": true}` when max confidence < 0.70

**Driver:** QA Engineer

**Rationale:**

1. **Confidence Thresholding on Tiny Validation Set:**
   - Validation set: ~10 real samples (N=100 train, 70/20/10 split)
   - Threshold tuning on 10 samples is unstable (±5% accuracy swing from threshold ±0.05)
   - Conservative threshold (0.70) reduces false positives; builds operator trust

2. **Confident Refusal Logic:**
   ```json
   If max(row_confidence, col_confidence) < 0.70:
   {
     "uncertain": true,
     "predicted_wells": null,
     "candidates": [
       {"well": "A4", "confidence": 0.68},
       {"well": "B5", "confidence": 0.65}
     ],
     "action": "defer to human operator for visual inspection"
   }
   ```

3. **Benefits:**
   - Avoids cascading errors from uncertain predictions
   - Operators handle hard cases manually (cheaper than low-confidence errors)
   - Reduces pressure on model to achieve perfect accuracy (acceptable to abstain)

**Evaluation Metrics for Confident Refusal:**

| Metric | Target | Definition |
|--------|--------|-----------|
| **"Confident refusal" rate** | 5–15% | % of samples system defers to human |
| **Accuracy on deferred cases** | N/A (human decides) | Track which cases needed human intervention |
| **Accuracy on confident cases** | > 85% | When system predicts, be right 85%+ |
| **False positive rate @ p_max > 0.90** | < 10% | Wrong high-confidence predictions are rare |

**Implementation:**
- Separate threshold for row and column? No; use global max across both heads
- Threshold tuning: Sweep {0.5, 0.6, 0.7, 0.8} on validation; select threshold that achieves 5–15% deferral rate

---

## Decision 6: [NEW] Temporal Modeling Mandatory (Temporal Transformer Required)

**Previous Decision:** Frame max-pooling over 2 frames before classification heads

**New Decision:** Temporal Transformer (2–4 attention blocks) is MANDATORY; single-frame max-pooling is PROHIBITED

**Driver:** Architect + ML Scientist

**Architectural Analysis:**

- Max-pooling is order-agnostic: max(frame1, frame2) = max(frame2, frame1)
- Cannot distinguish pipette entering well vs. exiting (temporal order destroyed)
- "Dispense" is an event (trajectory), not a state (position)
- Transfyr's "Tacit Knowledge" mission centers on motion semantics

**Revised Rationale:**

1. **Temporal Transformer Architecture:**
   ```
   Input: T=4-8 ordered frames per view
       ↓
   Temporal Backbone: DINOv2 encoder → (T, 768) feature sequence
       ↓
   Temporal Transformer: 2–4 attention layers with causal masking
       ├─ Frame-wise temporal attention (learns frame relationships)
       └─ Event localization (identifies dispense frame)
       ↓
   Temporal Pooling: Attend to "dispense event" frame → (768,)
       ↓
   Classification Heads: row logits, column logits
   ```

2. **Why Temporal Attention Matters:**
   - Motion trajectory encodes expert intent (approach, dispense, withdrawal)
   - Temporal attention learns to identify dispense phase (frame where tip pauses in well)
   - Solves "entering vs. leaving" ambiguity via motion sequence

3. **Tacit Knowledge Connection:**
   - Expert lab technicians execute pipetting via learned motor patterns
   - Pipette trajectory is first-class signal, not ancillary information
   - Temporal Transformer respects this: motion is primary semantic feature

**Explicit Prohibition:** Single-frame max-pooling (current approach) is NO LONGER VIABLE. Any design using max-pooling across frames must be explicitly overridden. Temporal Transformer is non-negotiable.

---

## Decision 7: [NEW] Branch Preservation — Baseline ResNet-18 on `baseline/resnet-cv-pipeline`

**New Decision:** Preserve original ResNet-18 single-frame approach on dedicated branch `baseline/resnet-cv-pipeline`

**Driver:** Engineer (for historical comparison and legacy support)

**Rationale:**

1. **Historical Validation:**
   - Original approach represents pre-revision strategy
   - Enables before/after performance comparison
   - Useful for understanding impact of DINOv2 + temporal modeling changes

2. **Legacy System Compatibility:**
   - Some deployment pipelines may require single-frame inference
   - Fallback if DINOv2 training fails
   - Reference for edge-case analysis

3. **Proof-of-Concept Preservation:**
   - Documents evolution of thinking
   - Useful for future re-evaluation if temporal approach underperforms

**Git Management:**
```bash
git branch baseline/resnet-cv-pipeline  # Branched from current state
# Keep in sync with main for critical bug fixes
# Never merge back to main (divergent architecture)
```

**Documentation:** README now includes section "Baseline Implementation (Preserved for Reference)" explaining where to find original code.

---

## Decision 8: [NEW] 3D Gaussian Splatting as Research Track

**New Decision:** Add **3D Gaussian Splatting (3DGS)** as Architecture 4 for glare-heavy scenarios

**Driver:** Architect

**Architectural Analysis:**

- Polystyrene wells refract light; translucent pipette tips add refraction effects
- Specular reflection (glare) shifts visual center of well
- 2D CNN cannot model depth and refraction; "visual center" ≠ "true well center"

**Revised Rationale:**

1. **3DGS Approach (Future Research):**
   - Reconstruct 3D scene (well geometry + liquid level) from FPV + top-view using 3DGS
   - Ray-trace from camera through reconstructed 3D scene
   - Back-project to true well center accounting for refraction via Snell's law
   - Use 3D coordinates (millimeters) instead of 2D pixels

2. **Why 3DGS Handles Refraction:**
   - Explicit 3D geometry captures depth (well depth, liquid level)
   - Rendering computes refraction naturally via ray tracing
   - True well center (in 3D space) decoupled from pixel position

3. **Expected Improvements:**
   - Pixel-space accuracy (2D pixels): ~75% (refraction causes 5–10 pixel error)
   - 3D-space accuracy (via 3DGS): ~85–90%
   - Inference time: 5–10 sec per sample (expensive; research-only)

**Implementation Timeline:**
- **Phase 1 (Current):** Temporal Transformer on 2D pixels (baseline)
- **Phase 2 (Future):** Add 3DGS as fallback for edge cases (glare-heavy wells)
- **Phase 3 (Research):** Full 3D end-to-end pipeline

**Near-term Mitigation:** Use Temporal Transformer to detect glare events (low confidence on bright frames); flag uncertain pixels for manual inspection.

---

## Decision 9: [ARCHIVED] GPU vs. CPU Inference (No Change)

**Status:** UNCHANGED from original Decision 9

**Decision:** GPU required for inference (CPU fallback not supported).

**Rationale:** Temporal Transformer inference on CPU ~3–5× slower than GPU; incompatible with 2-min budget.

---

## Decision 10: [ARCHIVED] Handling Unseen Wells (No Change)

**Status:** UNCHANGED from original Decision 10

**Decision:** Confidence gating on unseen wells; do not output low-confidence guesses.

**Enhancement:** Now integrated with Confident Refusal gate (Decision 5): if confidence < 0.70 (which includes many unseen wells), output `{"uncertain": true}`.

---

## Open Questions & Risks

### Risk 1: Synthetic Data Domain Gap
**Issue:** Blender-rendered videos may not match real lab footage; domain gap causes distribution shift.  
**Mitigation:** Validate via Fréchet Video Distance (FVD < 20); use VideoMAE fine-tuning (realistic trajectories) + augmentation (domain randomization).  
**Testing:** Hold-out validation on real samples only; measure synthetic-to-real transfer gap (target < 15%).

### Risk 2: Temporal Alignment Complexity
**Issue:** Temporal Transformer over 4–8 frames requires precise FPV/top-view synchronization.  
**Mitigation:** Use cross-correlation of optical flow magnitude to find frame offset; validate on validation set.  
**Testing:** Unit test on synthetic data (known offset); integration test on real samples.

### Risk 3: Calibration Overfitting
**Issue:** ECE computed on validation set (N~10) is unstable; threshold tuning prone to spurious optimization.  
**Mitigation:** Use conservative threshold (0.70) to avoid fine-tuning on tiny set; accept "confident refusal" rate of 5–15% as safety margin.  
**Testing:** Hold-out evaluation must report ECE separately (not tuned on hold-out).

### Risk 4: VideoMAE Fine-Tuning Divergence
**Issue:** Fine-tuning VideoMAE on 100 videos may cause catastrophic forgetting of robotics pre-training.  
**Mitigation:** Use low learning rate (1e-5), early stopping on validation loss, validate FVD before using synthetic data.  
**Testing:** Measure FVD at each epoch; halt if FVD increases >10%.

### Risk 5: Temporal Model Compute Overhead
**Issue:** Temporal Transformer slower than single-frame models; may exceed 20-min batch inference budget.  
**Mitigation:** Target <2 min per sample average (20 min for ~10 samples); batch processing may provide additional speedup.  
**Testing:** Profile inference on target hardware (GPU); if batch >20 min, fallback to fewer frames (T=4 instead of 8).

---

## Acceptance Criteria (Final)

**All of the following must be satisfied before hold-out evaluation:**

### Pre-Evaluation Checklist
- [ ] Synthetic data generated: 5K Blender + 5K VideoMAE-fine-tuned videos
- [ ] VideoMAE fine-tuning validated: FVD < 20 on synthetic vs. real
- [ ] Temporal Transformer implemented: processes T=4-8 frames, replaces max-pooling
- [ ] Late Fusion committed: FPV and top-view trained separately, fused at feature level
- [ ] ECE computed and tracked on validation set: target ECE < 0.10
- [ ] Confident refusal gate working: 5–15% of samples deferred at p_max < 0.70
- [ ] Cross-view agreement > 70% on held-out samples (FPV and top-view mostly align)
- [ ] Inference latency validated: <30 sec per sample on target GPU (within 2-min budget for 10 samples)
- [ ] Documentation updated: architecture diagrams, training logs, decision rationale
- [ ] Baseline branch created: `baseline/resnet-cv-pipeline` preserves original implementation
- [ ] Focal loss configured with α=0.75 for class imbalance

### On Hold-Out Evaluation (10 unknown samples)
- [ ] All 10 samples produce valid JSON output
- [ ] JSON schema validated (wells array, well_row A-H, well_column 1-12)
- [ ] **Primary metric:** ECE < 0.12 on hold-out (calibration holds on unseen data)
- [ ] **Secondary metric:** Exact-match accuracy ≥ 70% (accounting for unseen wells, calibration-first approach)
- [ ] Cardinality-wise accuracy ≥ 75% (separate scoring for 1/8/12-well operations)
- [ ] Cross-view agreement > 70% on hold-out samples
- [ ] False positive rate < 10% (confident wrong predictions are rare)
- [ ] Total runtime ≤ 20 minutes (<2 min per dual-view sample for ~10 samples)
- [ ] No runtime errors, exceptions, or timeouts
- [ ] Predictions consistent with visual inspection of videos

---

## Contingency Plans

### If VideoMAE Fine-Tuning Fails (FVD > 30)
1. Increase fine-tuning epochs; reduce learning rate to 1e-6
2. Add stronger augmentation to synthetic data (domain randomization)
3. If still failing: fallback to DINOv2 + Temporal Transformer (without generative sampling)
4. Use Blender-only synthetic data (5K samples) instead of VideoMAE-fine-tuned

### If Temporal Transformer Accuracy Drops Below 70% on Validation
1. Reduce temporal depth: use T=4 frames instead of 8
2. Switch to SlowFast backbone instead of DINOv2 (faster inference, different temporal modeling)
3. Increase synthetic data ratio (85% synthetic, 15% real) to boost training signal
4. If still failing: revert to single-frame baseline with architecture constraints

### If Hold-Out Contains Unseen Wells & ECE > 0.15
1. This indicates model uncertainty is not well-calibrated
2. Diagnostic: Per-well confusion matrix; identify systematic biases
3. Retraining: Add contrastive loss (pull apart confused wells) or hard negative mining
4. If time permits: Temperature scaling post-hoc (scale logits before softmax)

### If Inference Time Exceeds 2 Minutes for 10 Samples
1. Reduce temporal depth (T=4 instead of 8)
2. Use model quantization (int8 post-training) for 20–30% speedup
3. Batch process all 10 samples together (GPU vectorization)
4. Fallback: Single-frame model (5× faster, accept accuracy trade-off)

---

## Decision 12: Batch Size Reduced to 2 for T4 Compatibility (Apr 27, 2026)

**Decision:** Set `BATCH_SIZE=2` (down from 8) for all Kaggle T4 training runs.

**Rationale:** At `img_size=448`, `N=8` frames, and `B=8`, the effective backbone input is `B×N=64` sequences of 1024 tokens through 12 ViT transformer layers. The backward pass (saved activations) exhausts T4 VRAM (~14.6 GB) mid-first-epoch. The smoke test ran in `eval()+no_grad()` and did not catch this. `B=2` gives `B×N=16` backbone calls per step and fits comfortably.

**Tradeoff:** Effective batch is smaller (gradient noise higher). Gradient accumulation would recover effective batch size but is not currently implemented.

---

## Decision 13: Fix AMP-Incompatible BCE in Well-Consistency Loss (Apr 27, 2026)

**Decision:** Wrap the well-consistency loss block in `torch.cuda.amp.autocast(enabled=False)` and cast logits to `float32` before computing `F.binary_cross_entropy`.

**Rationale:** The well-consistency loss computes an outer product of two sigmoid outputs — a probability, not a logit — so `binary_cross_entropy_with_logits` cannot be used. PyTorch's autocast (AMP) forbids `F.binary_cross_entropy` on half-precision tensors. The fix computes only this sub-block in fp32; the rest of the forward pass stays in mixed precision.

---

## Decision 14: v11 Overfitting Diagnosis — Synthetic→Real Gap (Apr 28, 2026)

**Decision:** v11 training (640 clips: 80 real + 560 synthetic, `WEIGHT_DECAY=1e-4`) exhibited severe overfitting: train Jaccard reached 83% by epoch 18 while val Jaccard remained at ~0% throughout. Val loss improved through epoch 6 then steadily worsened. Best checkpoint: epoch 6, val_loss=0.4745.

**Root cause hypothesis:** 87.5% of training samples are synthetic but all 20 val samples are real. The model memorises the synthetic distribution and fails to generalise to real clips.

**Action taken (v12):**
1. `USE_SYNTHETIC=0` — real-only training (80 train / 20 val) to isolate whether the model can learn from real data at all before reintroducing synthetics.
2. `WEIGHT_DECAY=1e-3` — 10× increase (was `1e-4`) to slow memorisation of the small training set.

**Expected outcome:** If val Jaccard rises under real-only training, the synthetic generation quality needs improvement before being reintroduced. If it remains at 0%, the problem is in the model or data pipeline.

---

## Appendix: Decision Timeline

| Date | Decision | Owner | Status |
|------|----------|-------|--------|
| Apr 14 | [REVISED] Architecture: DINOv2 + Temporal Transformer | ML Scientist + Architect | FINAL |
| Apr 14 | [NEW] Synthetic Data: Blender + VideoMAE fine-tuning | Data Scientist | FINAL |
| Apr 14 | [NEW] Late Fusion mandate | Architect + ML Scientist | FINAL |
| Apr 14 | [REVISED] Primary metric: ECE < 0.10 | QA Engineer + ML Scientist | FINAL |
| Apr 14 | [NEW] Confident Refusal gate (p_max < 0.70) | QA Engineer | FINAL |
| Apr 14 | [NEW] Temporal Transformer mandatory | Architect + ML Scientist | FINAL |
| Apr 14 | [NEW] Branch preservation: baseline/resnet-cv-pipeline | Engineer | FINAL |
| Apr 14 | [NEW] 3DGS research track | Architect | FINAL |
| Apr 27 | [NEW] Batch size 8→2 for T4 VRAM (Decision 12) | Engineer | FINAL |
| Apr 27 | [NEW] AMP-safe BCE fix in well-consistency loss (Decision 13) | Engineer | FINAL |
| Apr 28 | [NEW] v11 overfitting diagnosis; v12 real-only + weight decay 10× (Decision 14) | ML Scientist | FINAL |

---

**Document Status:** FINAL  
**Last Updated:** April 28, 2026  
**Next Review:** After v12 real-only validation results

**Key References:**
- `docs/ML_STACK.md` — DINOv2 + temporal architecture, synthetic data strategy, training protocols
- `docs/ARCHITECTURE.md` — 4 architectural proposals, physical AI design principles
- `docs/QA_REPORT.md` — Status and quality assurance strategy
- `README.md` — Project overview and implementation status
