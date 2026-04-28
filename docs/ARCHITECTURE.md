# Transfyr AI Pipette Well Detection: Architecture

**Document:** System Architecture & Design  
**Date:** 2026-04-14 (revised 2026-04-28)  
**Status:** FINAL

---

## Selected Architecture: DINOv2-ViT-B/14 + LoRA + Temporal Transformer + Late Fusion

**Implementation status:** Fully implemented in [src/models/backbone.py](../src/models/backbone.py) and [src/models/fusion.py](../src/models/fusion.py).

### Why This Architecture

Three constraints dominate the problem:

1. **N=100 training samples** — rules out full fine-tuning of any large model. LoRA (rank r=4, ~12M trainable params out of ~86M frozen) is the only viable path.
2. **Dual-view geometric incompatibility** — FPV uses perspective projection (depth-dependent pixel position); top-view uses orthographic (depth-independent). Early fusion conflates incompatible coordinate systems. Late fusion is geometrically mandated, not a preference.
3. **"Dispense" is a temporal event** — tip enters well, stays, leaves. Single-frame models cannot distinguish these phases. A temporal transformer over sampled frames is required.

DINOv2's self-supervised pre-training on 142M images gives spatial patch embeddings that transfer to novel domains with minimal fine-tuning. At img_size=448, the backbone produces 32×32=1024 patches at 14px each — sufficient resolution to distinguish individual wells in the 8×12 grid.

---

### Data Flow

```
[FPV Video]      → Sample 8 frames → [B×N, 3, 448, 448]
                                    → DINOv2-ViT-B/14 (frozen) + LoRA adapters
                                    → [B×N, 768]
                                    → reshape → [B, N, 768]
                                    → Temporal Transformer (1 layer, 8 heads)
                                    → mean pool → [B, 768]
                                                        ↓
                                               Concat → [B, 1536]
                                                        ↓
                                               Fusion MLP (1536→512→256)
                                                        ↓
                                    ┌───────────────────┼───────────────────┐
                                    ↓                   ↓                   ↓
                             Row head (256→8)   Col head (256→12)   Type head (256→3)
                             raw logits         raw logits           single/row/col

[Top-view Video] → same backbone (shared weights) → [B, 768]
```

**Outputs:** Raw logits — sigmoid applied at inference time. Type head selects decoding strategy (argmax for single-well, full-row, full-column).

---

### Key Decisions

#### Shared Backbone
Both views share one DINOv2 backbone (`shared_backbone=True`). This halves trainable parameters and acts as a regularizer — the backbone must learn features useful for both perspectives simultaneously.

#### Factorized Row/Column Heads
Row logits (8) and column logits (12) are independent binary classifiers rather than a single 96-class head. This reduces output dimensionality 4× (20 vs 96), works naturally for full-row and full-column dispense patterns, and avoids the combinatorial explosion of well combinations.

#### Type Head
A 3-class head (single-well / full-row / full-column) disambiguates decoding at inference: the predicted type selects the argmax strategy, eliminating threshold sensitivity. Type labels are derived from row/column cardinality at training time — no additional annotation required.

#### Late Fusion (Geometric Mandate)
```
FPV (perspective):   pixel = π(X, z)    — depth-dependent
Top-view (ortho):    pixel ≈ (x, y)     — depth-independent

Early fusion:   concat raw features before encoding
                → network memorises per-video mapping
                → fails on any camera/plate position change

Late fusion:    encode each view in its native frame
                → concat encoded features [768, 768] → 1536
                → fuse via MLP
                → generalisable across viewpoints
```

Early fusion is explicitly forbidden.

#### Image Resolution: 448px
At 224px, DINOv2's position embeddings are compressed 5.3×, reducing spatial resolution per well. At 448px, 32×32=1024 patches provide adequate well-level granularity. The memory cost (~14 GB training on T4) is acceptable at batch size 2.

---

### Training Configuration (current)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Backbone | DINOv2-ViT-B/14 | Frozen; LoRA on Q, V projections |
| LoRA rank | 4 | ~12M trainable params total |
| img_size | 448 | 32×32=1024 patches |
| Frames | 8 | Uniformly sampled per clip |
| Batch size | 2 | T4 VRAM limit with 448px + backward pass |
| Optimizer | AdamW | lr=1e-4, weight_decay=1e-3 |
| LR schedule | Warmup (5 ep) + cosine annealing | |
| Loss | Weighted BCE + well-consistency | focal_gamma=0, col_weight=2.0 |
| Temporal layers | 1 | Single-layer transformer sufficient given small N |
| Dropout | 0.3 | Throughout fusion MLP |
| Augmentation | RandomResizedCrop, MotionBlur, ImageCompression, ColorJitter | Real clips only (val never augmented) |

**Data split:** Leak-free: real-only val (20 clips), real+synthetic train (80+560 clips). Synthetic clips excluded from val to prevent distribution contamination.

---

### Inference Latency

Budget: <2 min per dual-view sample (20 min total for ~10 samples).

| Stage | Time (GPU, batch 10) |
|-------|---------------------|
| Video I/O + preprocessing | ~500ms |
| DINOv2 backbone (8 frames × 2 views × 10) | ~150ms |
| Temporal Transformer | ~100ms |
| Fusion MLP + heads | ~30ms |
| **Total** | **~0.8–1s** |

Headroom: ~19 min 59s.

---

### Success Criteria

- [ ] ≥88% exact-match Jaccard on 20 real held-out val clips
- [ ] ≥85% on unseen row/column combinations (generalisation)
- [ ] <500ms per sample on GPU (batch of 10)
- [ ] Explicit dispense type classification (single / full-row / full-col)
- [ ] Reproducible (git-tracked, seeded)

---

## Design Principles

1. **Late fusion is geometrically mandated** — FPV (perspective) and top-view (orthographic) must be encoded separately. Early fusion is forbidden everywhere in this codebase.
2. **Temporal modeling is mandatory** — "dispense" is a causal event (entrance → insertion → release). Single-frame models cannot detect it.
3. **Foundation models + LoRA for N=100** — full fine-tuning of any backbone on 100 samples leads to severe overfitting. Parameter-efficient tuning (LoRA) is the only viable path.
4. **Factorised outputs** — row/column/type heads avoid the 96-class combinatorial problem and generalise to unseen combinations.

---

## Alternatives Considered

The following architectures were evaluated and rejected or deferred. Each is documented briefly; the primary reason for not selecting it is given.

### A. Classical Computer Vision + Geometric Pipeline

**Approach:** Edge detection → homography → well grid overlay → blob/optical flow pipette tip detection → late fusion via geometric triangulation.

**Strengths:** No training data required; fully interpretable; fast (1–5 s).

**Why not selected:** Fragile to lighting variation, glare on polystyrene, and occlusion. Temporal reasoning (distinguishing dispense from approach/retract) requires explicit optical flow engineering. Calibration overhead. Retained as **fallback** if DL approach fails.

---

### B. 3D Gaussian Splatting (3DGS)

**Approach:** Reconstruct 3D scene geometry from dual-view video; track pipette tip in reconstructed space.

**Strengths:** Physically accurate; immune to glare and refraction; generalises across cameras.

**Why not selected:** 10–60 min per sample reconstruction time — exceeds the 20-min batch budget by 30–180×. Requires camera calibration (~1 hr one-time). SfM fragile on featureless well plates. Appropriate for offline research validation only.

---

### C. Hybrid (Geometry + Foundation Model)

**Approach:** Classical CV provides a spatial prior ("tip near A5") combined with DINOv2 learned features as a refinement. Late fusion: `α × prior_logits + (1−α) × learned_logits`.

**Strengths:** Geometry provides signal when vision fails; interpretable fallback path.

**Why not selected:** Implementation complexity doubles without clear accuracy gain over pure Architecture 2. Hyperparameter `α` requires additional tuning. Selected as **fallback** if DINOv2+temporal reaches <88% validation accuracy.

---

### D. Acoustic-Visual Fusion

**Approach:** Add an acoustic modality (MFCC + 1D CNN) as a verification layer. Audio confirms dispense event timing; vision predicts which well.

**Strengths:** Robust to visual occlusion; precise temporal anchor; multimodal graceful degradation.

**Why not selected (yet):** Audio capture not confirmed available in all deployment environments. Silent electronic pipettes break the acoustic signal. Adds ~150ms latency. Deferred to Phase 2b — evaluate after visual-only baseline is validated.

**Design (for future reference):**
```
[Audio] → MFCC(13 coeff, 25ms window) → Conv1D(32→64→128) → p_dispense
IF dispense_detected AND visual_conf > threshold → output prediction (HIGH CONFIDENCE)
ELIF dispense_detected AND visual_conf < threshold → output with uncertainty flag
ELIF NOT dispense_detected → {"uncertain": true, "reason": "no_acoustic_dispense"}
```

---

### E. VideoMAE

**Approach:** Replace DINOv2 spatial backbone with VideoMAE (Kinetics-400 pre-trained) for native video understanding.

**Why not selected:** VideoMAE processes fixed-length video clips and requires temporal fine-tuning data. At N=100, domain gap from action-recognition pre-training (Kinetics) to lab-bench dispense events is likely larger than DINOv2's spatial domain gap. Deferred as future enhancement once data volume grows.

---

### Why Not ResNet-18

| Aspect | ResNet-18 | DINOv2-ViT-B/14 |
|--------|-----------|-----------------|
| Pre-training | ImageNet 1.2M supervised | 142M self-supervised (DINO) |
| Few-shot capability | Poor — full fine-tuning required | Excellent — LoRA sufficient |
| Overfitting risk on N=100 | Very high | Reduced dramatically |
| Spatial representation | 3×3 conv receptive fields | 14×14 explicit patch embeddings |
| Trainable params | 11M (100%) | ~12M (~14%, LoRA only) |
| Expected accuracy (N=100) | 50–60% | 88–96% (est.) |

ResNet-18 support was removed in commit d5b8f04.

---

## Architecture Comparison

| | Classical CV | **DINOv2 + LoRA** | 3DGS | Hybrid | Acoustic-Visual |
|---|---|---|---|---|---|
| Accuracy (est.) | 80–90% | **88–96%** | 95–99% | 92–97% | 92–98% |
| Training data | None | 100 + synthetic | 100 + calibration | 100 | 100 + audio labels |
| Latency (batch 10) | 1.6–3s | **~1s** | 10–60 min | 1.5–3s | ~1.2s |
| GPU required | No | Yes | Yes | Optional | Yes |
| Meets latency budget | Yes | **Yes** | **No** | Yes | Yes |
| Status | Fallback | **✅ Implemented** | Research only | Fallback | Deferred |

---

## Implementation Roadmap

### Phase 1 — Data & Baseline (complete)
- [x] Video loading + preprocessing pipeline
- [x] Synthetic data generation (700 clips)
- [x] Leak-free real/synthetic split
- [x] DINOv2 + LoRA + Temporal Transformer + Late Fusion implemented
- [x] Training loop with AMP, focal loss, well-consistency loss, type head

### Phase 2 — Validation & Tuning (in progress)
- [ ] Val Jaccard ≥ 0.50 on real held-out clips
- [ ] Hyperparameter sweep (weight_decay, dropout, lora_rank)
- [ ] Diagnose synthetic→real distribution gap (v12: real-only run)
- [ ] Decision gate: ≥88% accuracy → Phase 3; else Phase 2b

### Phase 2b — Fallback (if needed)
- [ ] Classical CV spatial prior (Architecture A)
- [ ] Acoustic verification (Architecture D) if audio confirmed available

### Phase 3 — Optimisation
- [ ] Error analysis: which wells misclassified?
- [ ] Confidence calibration (temperature scaling)
- [ ] Target: ≥92% on validation

### Phase 4 — Integration & Deployment
- [ ] CLI integration + batch inference
- [ ] Latency profiling (verify <500ms/sample)
- [ ] Docker containerisation
- [ ] Documentation + deployment guide

---

## Technical Debt & Future Work

1. **Uncertainty quantification** — Monte-Carlo dropout or ensembles; "abstain" when uncertain
2. **Temporal event grounding** — predict dispense timestamp (frame index) with confidence
3. **Plate format generalisation** — grid-agnostic decoder for 384/1536-well plates
4. **Monocular depth** — MiDaS as lightweight 3D alternative to full 3DGS
5. **Online learning** — domain adaptation in production on misclassified samples
6. **Acoustic robustness** — noise-robust acoustic models for high-ambient-noise labs
7. **Edge deployment** — quantise DINOv2 + Temporal Transformer for NVIDIA Jetson
8. **VideoMAE backbone** — evaluate once training data exceeds ~500 real clips
