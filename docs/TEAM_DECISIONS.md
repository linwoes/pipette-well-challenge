# Team Decisions Log: Transfyr AI Pipette Well Challenge (Post-Red Team Review)

**Date:** April 14, 2026 (Revised Post-Red Team)  
**Compiled by:** Cross-functional team (Data Scientist, Architect, ML Scientist, QA Engineer)  
**Purpose:** Document post-red-team architectural decisions, addressing critical gaps in 100-sample VideoMAE+synthetic-data strategy

---

## Executive Summary: Response to Red Team Review

The red team identified 5 critical gaps in the original ResNet-18 single-frame approach. This document captures decisions made to address each gap. **All original decisions (1–10) remain archived; new decisions [REVISED] and [NEW] below supersede where noted.**

**Critical Decisions Made:**
- [REVISED] Architecture Primary: VideoMAE + Temporal Transformer (not ResNet-18)
- [NEW] Fusion Architecture: Late Fusion (explicit mandate from red team)
- [NEW] Synthetic Data Strategy: 3D Blender + VideoMAE fine-tuning (10K+ samples)
- [REVISED] Primary Metric: ECE < 0.10 (not raw accuracy)
- [NEW] Confident Refusal Gate: Model must abstain on uncertain inputs
- [NEW] Temporal Modeling Mandatory: Temporal Transformer required
- [NEW] Branch Preservation: ResNet-18 baseline on `baseline/resnet-cv-pipeline`
- [NEW] 3DGS as Research Track: Added for glare-heavy scenarios

---

## Decision 1: [REVISED] Architecture Primary — VideoMAE + Temporal Transformer

**Previous Decision:** Use ResNet-18 backbone with single-frame classification

**Revised Decision:** Use **VideoMAE-Base (Open X-Embodiment pre-trained) + Temporal Transformer** as primary architecture

**Driver:** ML Scientist + Architect (responding to Red Team 2.2 & 3.2)

**Red Team Finding:**
- ResNet-18 (2015) and Focal Loss (2017) are "commodity models"; insufficient for Physical AI
- Transfyr expects Vision-Language-Action (VLA) approach where motion trajectory is first-class
- Single-frame models cannot distinguish pipette entering vs. leaving (temporal blindness)
- 11M-parameter ResNet on 100 samples will memorize background/lighting, not learn geometry

**Revised Rationale:**

1. **Foundation Model Pre-training (2026 SOTA):**
   - VideoMAE-Base pre-trained on Open X-Embodiment robotics data (1M+ manipulation videos)
   - Self-supervised learning learns coordinate geometry without supervision (better than ImageNet for well-center localization)
   - Transfer to pipette-liquid interaction is direct: end-effector alignment ≈ well-center alignment
   
2. **Temporal Understanding (Addressing Red Team 3.2):**
   - Temporal Transformer with 2–4 attention blocks over ordered frames
   - Replaces frame max-pooling (which destroys temporal order)
   - Models dispense as event: approach → hold → release (not state)
   - Aligns with Transfyr's "Tacit Knowledge" mission: motion intent is causal
   
3. **Data Efficiency via Fine-tuning:**
   - Pre-training on robotics data stabilizes learning on 100 real samples
   - Fine-tuned VideoMAE encoder used to generate synthetic dispense trajectories
   - 10,000+ synthetic samples = 100× data scale-up (mitigates overfitting crisis)

**Architecture at a Glance:**
```
Input: (T=4-8 ordered frames per view, 224×224×3)
    ↓
Temporal Backbone: VideoMAE encoder → (T, 768) feature sequence
    ↓
Temporal Transformer: Cross-attention over frames → Event localization
    ↓
Late Fusion: FPV features ⊗ Top-view features (cross-modal attention)
    ↓
Output Heads: 8-class row + 12-class column with calibrated sigmoid
```

**Fallback Hierarchy:**
1. **Primary:** VideoMAE-Base + Temporal Transformer
2. **Alternative:** DINO-v2 (ViT-B/14) as frozen feature extractor + Temporal Transformer
3. **Legacy:** ResNet-18 single-frame (preserved on `baseline/resnet-cv-pipeline` for reference)

**Cross-Reference:** `docs/ML_STACK.md` Part 2–3; `docs/ARCHITECTURE.md` Architecture 2 (revised); `docs/RED_TEAM_REVIEW.md` Section 2.2, 3.2

---

## Decision 2: [REVISED] Synthetic Data Strategy — 3D Blender + VideoMAE Fine-Tuning

**Previous Decision:** Augmentation only; 100 real samples sufficient with regularization

**Revised Decision:** Generate **10,000+ synthetic dispense events** combining:
- **Option A:** 3D Blender photorealistic simulation (5,000 samples)
- **Option B:** VideoMAE fine-tuning generative sampling (5,000 samples)

**Driver:** Data Scientist (responding to Red Team 2.1: "N=100 is failure of scale")

**Red Team Finding:**
- 100 samples is "a failure of scale" for Physical AI company
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

**Cross-Reference:** `docs/ML_STACK.md` Part 1; `docs/RED_TEAM_REVIEW.md` Section 2.1

---

## Decision 3: [NEW] Fusion Architecture — Late Fusion (Explicit Mandate)

**Previous Decision:** "Late concatenation fusion" (ambiguous; caused confusion)

**New Decision:** **Late Fusion is MANDATORY and explicitly defined:**

Late Fusion = Process FPV and Top-view independently through spatiotemporal backbones → Fuse at feature level (after spatial extraction) → Fuse via cross-attention or gating

**Driver:** Architect + ML Scientist (responding to Red Team 2.3 + QA_REPORT conflict)

**Red Team Finding:**
- QA_REPORT identified critical contradiction: ML_STACK specified early fusion; TEAM_DECISIONS specified late fusion
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
FPV Video → VideoMAE + Temporal Transformer → (768,) ──┐
                                                       ├──> Cross-attention ──> (768,) ──> Heads
Top-view Video → VideoMAE + Temporal Transformer → (768,) ──┘
```

**Alternative: Gating (if compute-constrained):**
```
FPV features (768,) ──┬──> Dense(768 → 256) ──┬──> Gating ──> α·FPV + (1-α)·TopView
                      │                        │  (learned weight α)
Top-view (768,) ──────> Dense(768 → 256) ───┘
```

**Explicit Commitment:** Early fusion is **FORBIDDEN** for this project. Any design proposing early fusion (concatenate before spatial extraction) must be explicitly overridden.

**Cross-Reference:** `docs/ARCHITECTURE.md` Section 2.3; `docs/QA_REPORT.md` Issue 1.3; `docs/ML_STACK.md` Part 4

---

## Decision 4: [REVISED] Primary Metric — Expected Calibration Error (ECE < 0.10)

**Previous Decision:** Maximize exact-match accuracy (% of samples with correct row & column)

**Revised Decision:** **Primary metric is Expected Calibration Error (ECE < 0.10), not raw accuracy**

**Driver:** QA Engineer + ML Scientist (responding to Red Team 5)

**Red Team Finding:**
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

**Cross-Reference:** `docs/ML_STACK.md` Part 5; `docs/QA_STRATEGY.md` Section 5; `docs/RED_TEAM_REVIEW.md` Section 5

---

## Decision 5: [NEW] Confident Refusal Gate (Uncertainty Abstention)

**Previous Decision:** Attempt prediction for all samples; threshold confidence at 0.5

**New Decision:** Model must output `{"uncertain": true}` when max confidence < 0.70

**Driver:** QA Engineer (responding to Red Team 5 + calibration-first approach)

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

**Cross-Reference:** `docs/ML_STACK.md` Section 5.4; `docs/QA_STRATEGY.md` Section 5

---

## Decision 6: [NEW] Temporal Modeling Mandatory (Temporal Transformer Required)

**Previous Decision:** Frame max-pooling over 2 frames before classification heads

**New Decision:** Temporal Transformer (2–4 attention blocks) is MANDATORY; single-frame max-pooling is PROHIBITED

**Driver:** Architect + ML Scientist (responding to Red Team 3.2)

**Red Team Finding:**
- Max-pooling is order-agnostic: max(frame1, frame2) = max(frame2, frame1)
- Cannot distinguish pipette entering well vs. exiting (temporal order destroyed)
- "Dispense" is an event (trajectory), not a state (position)
- Transfyr's "Tacit Knowledge" mission centers on motion semantics

**Revised Rationale:**

1. **Temporal Transformer Architecture:**
   ```
   Input: T=4-8 ordered frames per view
       ↓
   Temporal Backbone: VideoMAE or DINO encoder → (T, 768) feature sequence
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

**Cross-Reference:** `docs/ML_STACK.md` Part 3; `docs/ARCHITECTURE.md` Section 2.1–2.2; `docs/RED_TEAM_REVIEW.md` Section 3.2

---

## Decision 7: [NEW] Branch Preservation — Baseline ResNet-18 on `baseline/resnet-cv-pipeline`

**New Decision:** Preserve original ResNet-18 single-frame approach on dedicated branch `baseline/resnet-cv-pipeline`

**Driver:** Engineer (for historical comparison and legacy support)

**Rationale:**

1. **Historical Validation:**
   - Original approach represents pre-red-team strategy
   - Enables before/after performance comparison
   - Useful for understanding impact of VideoMAE + temporal modeling changes

2. **Legacy System Compatibility:**
   - Some deployment pipelines may require single-frame inference
   - Fallback if VideoMAE training fails
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

**Driver:** Architect (responding to Red Team 3.1: "Transparency Risk")

**Red Team Finding:**
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

**Cross-Reference:** `docs/ARCHITECTURE.md` Appendix; `docs/ML_STACK.md` Part 9.4; `docs/RED_TEAM_REVIEW.md` Section 3.1

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

## Open Questions & Risks Flagged by Red Team

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

## Acceptance Criteria (Revised Post-Red Team)

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

### On Hold-Out Evaluation (10 unknown samples)
- [ ] All 10 samples produce valid JSON output
- [ ] JSON schema validated (wells array, well_row A-H, well_column 1-12)
- [ ] **Primary metric:** ECE < 0.12 on hold-out (calibration holds on unseen data)
- [ ] **Secondary metric:** Exact-match accuracy ≥ 70% (accounting for unseen wells, calibration-first approach)
- [ ] Cardinality-wise accuracy ≥ 75% (separate scoring for 1/8/12-well operations)
- [ ] Cross-view agreement > 70% on hold-out samples
- [ ] False positive rate < 10% (confident wrong predictions are rare)
- [ ] Total runtime ≤ 20 minutes (batch inference for ~10 samples)
- [ ] No runtime errors, exceptions, or timeouts
- [ ] Predictions consistent with visual inspection of videos

---

## Contingency Plans

### If VideoMAE Fine-Tuning Fails (FVD > 30)
1. Increase fine-tuning epochs; reduce learning rate to 1e-6
2. Add stronger augmentation to synthetic data (domain randomization)
3. If still failing: fallback to DINO-v2 + Temporal Transformer (without generative sampling)
4. Use Blender-only synthetic data (5K samples) instead of VideoMAE-fine-tuned

### If Temporal Transformer Accuracy Drops Below 70% on Validation
1. Reduce temporal depth: use T=4 frames instead of 8
2. Switch to SlowFast backbone instead of VideoMAE (faster inference, different temporal modeling)
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

## Appendix: Decision Timeline (Post-Red Team)

| Date | Decision | Owner | Red Team Trigger | Status |
|------|----------|-------|------------------|--------|
| Apr 14 | [REVISED] Architecture: VideoMAE + Temporal Transformer | ML Scientist + Architect | 2.2, 3.2 | FINAL |
| Apr 14 | [NEW] Synthetic Data: Blender + VideoMAE fine-tuning | Data Scientist | 2.1 | FINAL |
| Apr 14 | [NEW] Late Fusion mandate | Architect + ML Scientist | 2.3 | FINAL |
| Apr 14 | [REVISED] Primary metric: ECE < 0.10 | QA Engineer + ML Scientist | 5 | FINAL |
| Apr 14 | [NEW] Confident Refusal gate (p_max < 0.70) | QA Engineer | 5 | FINAL |
| Apr 14 | [NEW] Temporal Transformer mandatory | Architect + ML Scientist | 3.2 | FINAL |
| Apr 14 | [NEW] Branch preservation: baseline/resnet-cv-pipeline | Engineer | Historical | FINAL |
| Apr 14 | [NEW] 3DGS research track | Architect | 3.1 | FINAL |

---

**Document Status:** FINAL (Post-Red Team Response)  
**Last Updated:** April 14, 2026  
**Next Review:** After hold-out evaluation completion

**Key References:**
- `docs/RED_TEAM_REVIEW.md` — Red team findings (5 findings × 8 sections each)
- `docs/ML_STACK.md` — VideoMAE + temporal architecture, synthetic data strategy, training protocols
- `docs/ARCHITECTURE.md` — 4 architectural proposals, physical AI design principles
- `docs/QA_REPORT.md` — ORANGE status (31 issues); red team response section
- `README.md` — Updated with post-red-team strategy, response to 5 findings
