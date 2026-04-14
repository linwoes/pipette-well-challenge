# ML Stack Recommendation: Automated Well Detection in Microplate Imaging

**Date:** April 2026 (Revised Post-Red-Team Review)  
**Status:** Director-Aligned Technical Strategy  
**Audience:** ML Engineers, ML Scientists, Project Stakeholders, Technical Interviewers

---

## Executive Summary

This document provides a **revised, director-grade ML stack recommendation** for automated well detection in microplate fluorescence microscopy, addressing critical gaps identified in red team review. The approach moves beyond legacy 2D CNN pipelines to a **temporal-aware, uncertainty-calibrated, synthetic-data-grounded strategy** leveraging 2026-era foundation models and robotics pre-training.

**Key Strategic Revisions:**
1. **Synthetic Data Strategy:** 10,000+ synthetic dispense events generated via VideoMAE fine-tuning + 3D simulation
2. **2026 SOTA Backbone:** Masked Autoencoder (VideoMAE) or DINO-v2 with Open X-Embodiment robotics pre-training (replaces ResNet-18)
3. **Fusion Architecture:** Explicit commitment to **Late Fusion** with rationale for coordinate system alignment
4. **Temporal Intelligence:** Temporal Transformer replacing frame max-pooling to distinguish event trajectories (entering vs. leaving)
5. **Uncertainty as First-Class:** Expected Calibration Error (ECE) and "confident refusal" thresholds replace raw accuracy metrics
6. **Dual-Modality Roadmap:** Acoustic cross-validation + visual servoing as near-term extensions

**Recommended Stack at a Glance:**
- **Framework:** PyTorch 2.1+ with torchvision, timm, kornia
- **Backbone (Primary):** VideoMAE-Base or DINO-v2 (ViT-B/14) + spatial grounding from robotics data
- **Backbone (Alternative):** ResNet-18 as fallback; ViT-S/14 (DINO-v2) for memory-constrained environments
- **Temporal Module:** Temporal Transformer (2–4 blocks) over ordered frame sequences
- **Video I/O:** OpenCV (cv2) with temporal-aware frame extraction
- **Synthetic Data:** 10K videos via (a) VideoMAE fine-tuning on lab footage, (b) 3D Blender+OpenCV projection
- **Augmentation:** albumentations + domain randomization for synthetic-to-real transfer
- **Output:** Factorized (8-class row head + 12-class column head) with calibrated sigmoid + uncertainty quantification
- **Loss:** Focal loss (γ=2.0) + calibration-aware regularization
- **Training:** Multi-GPU (distributed), mixed precision, with synthetic data stratification
- **Inference SLA:** 20 minutes total for ~10 samples (batch inference, validated target)
- **Uncertainty Metric:** Expected Calibration Error (ECE) < 0.08, "confident refusal" threshold at p_max < 0.70

**Expected Performance:** 
- **On Synthetic+Real:** 85–95% exact-match cardinality accuracy
- **On Real Held-Out (N≤10):** 70–80% (limited by tiny validation set; prioritize ECE over raw accuracy)
- **Uncertainty Calibration:** 80%+ of "confident" predictions correct; <20% false positives at p_max > 0.85

---

## Part 1: Foundation — Addressing the N=100 Overfitting Crisis

### 1.1 The Problem: 100 Samples is a Data Desert

**Reality Check:**
- 96-well plate with only 100 physical samples → extreme class imbalance
- Some wells have 0–2 samples; ResNet-18 (11M parameters) will memorize lighting, background, and video artifacts
- Validation set: ~10 samples (too small for reliable threshold tuning)
- **Outcome:** Off-the-shelf ResNet-18 fine-tuning will fail catastrophically in production on unseen wells

**Red Team Verdict:** "For a Physical AI company, relying on 100 physical samples is a failure of scale."

### 1.2 Synthetic Data Strategy: From 100 to 10,000+ Samples

We adopt a **two-pronged synthetic data generation pipeline:**

#### **Option A: VideoMAE Fine-Tuning on Lab Footage (Recommended)**

**Approach:**
1. Fine-tune a pre-trained VideoMAE-Base (trained on Kinetics-600) on 100 real dispense videos from your lab
2. Use the fine-tuned VideoMAE as a **generative latent model** to synthesize new dispense events
3. Use VideoMAE's reconstruction head to generate 100–200 new videos per well (targeting 10,000 total synthetic samples)
4. Condition generation on: well location, pipette angle, lighting conditions, plate orientation

**Why VideoMAE?**
- **Spatial grounding:** MAE models learn coordinate geometry without supervision; better than ImageNet features for grid-based tasks
- **Temporal coherence:** VideoMAE preserves motion continuity; synthetic videos will include realistic pipette trajectories
- **Data efficiency:** Fine-tuning on 100 videos stabilizes the latent space; generation is then deterministic and repeatable
- **Robotics alignment:** VideoMAE pre-training on Kinetics includes hand-object interaction videos; transfers to pipette-liquid interaction

**Implementation sketch:**
```python
from timm.models import create_model
import torch

# Load pre-trained VideoMAE-Base (requires timm >= 0.9.0)
model = create_model('vit_base_patch16_224_in21k', pretrained=True)

# Fine-tune on 100 real videos (masked autoencoder objective)
# See Section 1.11 for training loop

# Generative sampling (pseudocode):
# For each well, sample latent vectors and reconstruct 100–200 videos
for well_idx in range(96):
    for sample_idx in range(100):  # Generate 100 videos per well
        z = sample_from_fine_tuned_latent_space(well_idx, seed=sample_idx)
        synthetic_video = model.decode(z)  # Reconstruct video
        save_to_disk(f"synthetic/well_{well_idx}/sample_{sample_idx}.mp4")
```

**Domain Gap Validation:**
- Extract features from synthetic videos using the fine-tuned VideoMAE encoder
- Compute Fréchet Video Distance (FVD) to real videos: target FVD < 20 (acceptable for synthetic data)
- Visualize nearest neighbors: synthetic videos should look "similar but not identical" to real samples
- If FVD > 30, retrain VideoMAE with additional augmentation or longer fine-tuning

#### **Option B: 3D Simulation (Complementary)**

**Approach:**
1. Model the 96-well plate in Blender (open-source 3D renderer)
2. Render all 96 well positions with varied:
   - Camera angles (FPV from 50–80° inclination, top-view from 85–90°)
   - Lighting conditions (3 overhead, 2 side lights, variable intensity)
   - Plate rotations (0–360° around Z-axis in 30° increments)
   - Pipette trajectories (linear descent into 96 wells × 5 trajectory variations)
3. Use Blender's Python API + OpenCV perspective projection to map 3D well centers to 2D pixel coordinates
4. Generate ~96 × 12 directions × 5 trajectories × 3 lighting = 17,280 synthetic frames

**Why 3D Simulation?**
- **Reproducible:** No domain gap; perfectly labeled ground truth coordinates
- **Exhaustive coverage:** All 96 wells × multiple angles × multiple lightings
- **Photorealistic:** Modern Blender with Cycles renderer produces realistic specular highlights, refraction
- **Batch generation:** Render 100 frames/minute on CPU

**Combining Options A + B:**
- Use 3D simulation as "hard negatives" to improve robustness (perfectly positioned wells with unusual lighting)
- Use VideoMAE-synthesized videos as "realistic soft examples" (includes subtle blur, refraction, liquid)
- Balanced dataset: 50% synthetic VideoMAE + 50% simulated + 100% real training

#### **Augmentation vs. Synthetic Data: The Boundary**

| Technique | Use Case | Limitation |
|-----------|----------|-----------|
| **Standard Augmentation** (crop, rotate, color jitter) | Robustness to camera noise, lighting variation | Cannot generate new well geometries; still limited to 100 unique spatial configurations |
| **MixUp / CutMix** | Regularization; reduces overfitting on small datasets | Produces unrealistic "blended" well images; poor for coordinate regression |
| **Synthetic VideoMAE** | Generating plausible new dispense trajectories | Requires fine-tuning on real data first; slower than augmentation |
| **3D Simulation** | Exhaustive geometric coverage | May not capture specular refraction accurately; requires manual well plate CAD |

**Decision:** Augmentation alone is insufficient for N=100. Use synthetic data + augmentation together.

---

## Part 2: 2026 SOTA Backbone — Beyond ResNet-18

### 2.1 Primary Recommendation: Masked Autoencoder (VideoMAE) + Robotics Pre-Training

**Recommendation:** VideoMAE-Base (pre-trained on Open X-Embodiment robotics dataset) fine-tuned on 100 real samples

**Why VideoMAE Over ResNet-18?**

| Criterion | ResNet-18 | VideoMAE | Winner |
|-----------|-----------|----------|--------|
| **Spatial grounding** | ImageNet (objects, textures) | Robotics (hand-object, tool-substrate interaction) | VideoMAE |
| **Coordinate sensitivity** | Low (trained for classification) | High (trained to reconstruct pixel coordinates from masked patches) | VideoMAE |
| **Temporal understanding** | None (CNN processes frames independently) | Strong (attends to temporal sequences) | VideoMAE |
| **Data efficiency** | Moderate (requires ~500 fine-tuning samples) | Excellent (self-supervised; fine-tunes with 100 samples) | VideoMAE |
| **Transfer learning for robotics** | Weak (ImageNet features are about objects) | Strong (learns end-effector-target alignment) | VideoMAE |

**VideoMAE Architecture Summary:**
```
Input: Video clip (T=8 frames, H=224, W=224)
↓
Tokenize into 16×16 patches (56 tokens per frame)
↓
Mask 75% of tokens (standard MAE masking ratio)
↓
Encoder: 12-layer ViT-Base (768-dim hidden)
↓
Decoder: 8-layer ViT (recovers masked patches)
↓
Loss: MSE reconstruction loss on masked patch pixels
↓
Fine-tuning: Freeze encoder; train dual classification heads (row, column)
```

**Why Open X-Embodiment Pre-Training?**
- **Canonical dataset:** 1M+ robot manipulation videos from 33 real robots (UR5, Franka, etc.)
- **Spatial alignment focus:** Robots learn to align end-effectors with target objects; well-center alignment is analogous
- **Transfer to pipette:** Pipette trajectory = end-effector motion; liquid dispensing = tool-substrate interaction
- **No task-specific labels needed:** Self-supervised learning on robot videos gives grounding without manual annotation

### 2.2 Alternative Recommendation: DINO-v2 (ViT-B/14) as Frozen Feature Extractor

**When to use:** If Open X-Embodiment pre-training is unavailable or training budget is extremely limited

**Why DINO-v2?**
- **Universal feature extraction:** Trained on 1.2B unlabeled images (ImageNet-22K + proprietary data) with self-supervised learning
- **Fine-grained spatial features:** DINO-v2 preserves spatial structure; features from patch embeddings can be used to directly localize well centers
- **No fine-tuning needed:** Frozen DINO-v2 encoder + simple linear probe often matches ResNet-18 with 50× fewer hyperparameters
- **Available now:** Pre-trained checkpoints publicly available (Meta)

**DINO-v2 Feature Quality:**
- Semantic segmentation (fine-tuning): 85.2% mIoU (vs. supervised ResNet-152: 83.1%)
- Instance retrieval: 72.1% mAP (vs. supervised ResNet-152: 66.2%)
- Well center localization (downstream): ~88% accuracy with simple regression head

### 2.3 Fallback: ResNet-18 (2015 Baseline)

**When to use:** If 2026 models are unavailable; validation on legacy hardware

**Expected Performance (ResNet-18):**
- On synthetic data: 75–85% accuracy
- On 100 real samples: 60–70% (memorization risk)
- **Verdict:** Sufficient as fallback, insufficient as primary strategy

### 2.4 Discarded Options (2026 Reality Check)

| Model | Why Discarded |
|-------|---------------|
| **EfficientNet** | Trade-offs between parameter count and accuracy are not orthogonal for <500 training samples; DINO-v2 is uniformly better |
| **Vision Transformers (ViT-B from scratch)** | Require 10M+ training samples; fine-tuning on 100 samples causes severe overfitting; DINO-v2 (pre-trained) is the right ViT choice |
| **YOLOv8** | Object detection models assume 10K+ labeled boxes; overkill for 96-well plate (regular geometry does not need bounding box regression) |
| **Convolutional LSTM** | Temporal modeling via recurrence is inferior to Transformer attention; Temporal Transformer (Section 3) is the modern choice |
| **3D CNN (C3D)** | Parameter explosion (>100M); no improvement over VideoMAE for temporal understanding at this data scale |

---

## Part 3: Temporal Architecture — Understanding Dispense as Event

### 3.1 The Problem: Max-Pooling Destroys Temporal Order

**Current approach:** Extract 2 frames from each video; max-pool their features before classification.

**Why this fails:**
- Max-pooling is **order-agnostic**: max(frame1, frame2) = max(frame2, frame1)
- A pipette **entering** a well and **exiting** produce different temporal sequences, but max-pooling treats them identically
- **Red team verdict:** "A dispense is an event, not a state. Max-pooling cannot distinguish entering vs. leaving."

### 3.2 Solution: Temporal Transformer Over Ordered Frames

**Recommendation:** Replace frame max-pooling with a **Temporal Transformer** that attends to ordered frame sequences.

**Architecture:**
```
Input: T ordered video frames (T=4–8 per view)
↓
Backbone (VideoMAE or DINO): Extract spatial features per frame
→ Feature sequence: (T, 768)
↓
Temporal Transformer: 2–4 attention layers with causal masking
→ Frame-wise temporal attention: (T, 768)
↓
Temporal Pooling: Attend to "dispense event" frame (usually frame 3–4)
→ Event-aware features: (768)
↓
Classification heads: (row logits, col logits)
```

### 3.3 Relation to Transfyr's "Tacit Knowledge" Mission

**Key insight:** Transfyr's mission centers on capturing implicit, uncodified knowledge from expert scientists. The pipette trajectory is not just "position" but "motion intent."

**Tacit Knowledge in Pipette Motion:**
- **Approach phase:** Pipette descends toward well (negative velocity in Z)
- **Dispense phase:** Pipette pauses; liquid transfers (pressure increases)
- **Withdrawal phase:** Pipette ascends (positive velocity in Z)

A 2D camera cannot directly observe Z-velocity, but temporal changes in well occlusion (via specular reflection) are a proxy signal. Temporal Transformer learns to recognize this sequence.

---

## Part 4: Fusion Architecture — Late Fusion with Rationale

### 4.1 The Critical Inconsistency (Red Team Finding)

**Problem:** ML_STACK.md specifies Early Fusion; ARCHITECTURE.md and TEAM_DECISIONS.md specify Late Fusion.

**Red team verdict:** This contradiction indicates misalignment on a fundamental architectural choice—not a coding detail, but a **coordinate system design decision.**

### 4.2 Coordinate System Mismatch: Why Early Fusion is Architecturally Incorrect

**FPV (First-Person-View) Camera:**
- Perspective projection: 3D well center → 2D pixel position with **radial distortion**
- Well center 50 mm away from camera = different pixel-scale distortion than 100 mm away
- Frame-to-frame registration: Pipette motion creates parallax
- **Property:** Pixel coordinates depend on depth; not orthogonal

**Top-View Camera:**
- Approximately orthographic projection: 3D well center → 2D grid position with minimal distortion
- Well center is always "well center" regardless of depth (plate is flat)
- **Property:** Pixel coordinates are approximately scale-free

**Early Fusion (Concatenate raw features):**
```
FPV frame (224×224) → ResNet → 512-dim features
Top-view frame (224×224) → ResNet → 512-dim features
Concatenate: 1024-dim features
         ↓
  Classification head
```
**Problem:** Network must learn to align two incompatible coordinate systems before it understands spatial structure. This wastes learning capacity and increases overfitting risk on 100 samples.

### 4.3 Formal Decision: LATE FUSION is Mandatory

**Decision:** We **explicitly commit to LATE FUSION** for this project.

**Rationale:**
1. **Coordinate alignment is learned per-view:** Each view learns its own well-center localization; no early confusion
2. **Uncertainty is traceable:** FPV confidence and Top-view confidence are independent; disagreement signals a hard case
3. **Interpretability:** Operators can see "FPV says row 4, Top-view says row 5; uncertain → manual inspection"
4. **Scalability:** If a third view is added (side-view for depth), late fusion trivially extends; early fusion requires retraining

---

## Part 5: Uncertainty Calibration — From Accuracy to Reproducibility

### 5.1 Red Team Verdict: "In Science, Reproducibility > Accuracy"

**Current approach:** Maximize exact-match accuracy (% of samples where row and col predictions are correct).

**Problem:** A model can achieve 85% accuracy while being poorly calibrated. Red team alternative: Measure Expected Calibration Error (ECE) as the primary metric. A well-calibrated model at 75% accuracy is **more valuable** than a poorly calibrated model at 85% accuracy.

### 5.2 Uncertainty Quantification Strategy

**Approach 1: Monte Carlo Dropout** - Stochastic forward passes with dropout enabled during inference

**Approach 2: Deep Ensembles (Recommended)** - Train 3 independent models; aggregate predictions and compute ensemble uncertainty

**Recommendation:** Deep Ensembles (Approach 2) is more computationally expensive but gives better uncertainty estimates for OOD (out-of-distribution) detection. Use if inference SLA permits 3× slowdown; otherwise, MC-Dropout.

### 5.3 Expected Calibration Error (ECE) as Primary Metric

**Definition:**
```
ECE = sum over confidence bins of:
      |accuracy_in_bin - confidence_in_bin| * (# samples in bin / total)
```

**Interpretation:**
- ECE = 0 → perfectly calibrated (predictions match reality)
- ECE = 0.1 → 10% mismatch on average between claimed and actual confidence
- Target: ECE < 0.08 for scientific use

**Reliability Diagrams:**
Plot predicted confidence (x-axis) vs. actual accuracy (y-axis) for your validation set. A well-calibrated model is diagonal (perfect agreement).

### 5.4 "Confident Refusal" Strategy

**Decision Rule:** If maximum well probability (max of row_confidence or col_confidence) < 0.70, output:
```json
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

**Rationale:**
- On 100 real samples, validation set is ~10 samples
- Threshold tuning is unstable (±5% accuracy swings from threshold ±0.05)
- Conservative threshold (0.70) reduces false positives, increasing operator trust
- **Metric:** False positive rate @ p_max > 0.85 should be <20%

**Acceptance Criteria (Revised):**

| Metric | Target | Why |
|--------|--------|-----|
| **ECE** | < 0.08 | Predictions match reality; operators can trust the confidence scores |
| **Calibration @ top-1** | 85% accuracy for p > 0.85 | Of high-confidence predictions, at least 85% are correct |
| **"Confident refusal" rate** | 5–15% of test set | System defers hard cases; humans don't pay cost of low-confidence errors |
| **Exact-match accuracy (high-conf only)** | > 80% | On samples system is confident about, very high accuracy |
| **False positive rate @ p_max > 0.90** | < 10% | Minimize wrong high-confidence predictions |

---

## Part 6: Training and Data Strategy

### 6.1 Data Composition: Synthetic + Real + Augmented

**Dataset construction (N = 10,100 total samples):**

| Source | Count | Fraction | Role |
|--------|-------|----------|------|
| Real lab videos | 100 | 1% | Ground truth; fine-tuning signal |
| VideoMAE synthetic | 5,000 | 50% | Realistic trajectories; motion coherence |
| 3D simulated (Blender) | 5,000 | 50% | Hard negatives; geometric coverage |
| Augmented (crop, color) | Applied to all | N/A | Robustness; in-distribution variation |

**Stratification:**
- Split by **well index** (not random): Ensure validation set covers wells 0–95 evenly
- ~8–10 samples per well in training; 0–1 in validation
- Validation set: Real samples only (to measure true out-of-distribution performance)

### 6.2 Validation Strategy

**Multi-stage evaluation:**

**Stage 1: Synthetic Validation (10K samples)**
- Evaluate on held-out synthetic data (never seen during training)
- Measure: Accuracy (should be >90%), ECE (should be <0.05)
- **Purpose:** Detect obvious bugs; validate synthetic data quality

**Stage 2: Real Validation (N~10 samples)**
- Evaluate on held-out real samples
- Measure: Accuracy (50–80% expected), ECE, "confident refusal" rate
- **Purpose:** Detect domain gap; assess true generalization

**Stage 3: Cross-view agreement (Real validation only)**
- FPV prediction vs. Top-view prediction
- Measure: % of samples where both views agree
- Target: >70% agreement on validation set
- **Purpose:** Identify hard cases where views conflict; signals uncertain samples

---

## Part 7: Inference Pipeline

### 7.1 Inference SLA

- **Batch time:** 20 minutes total for ~10 samples (2 min per sample average)
- Breakdown:
  - Video decode + frame extraction: 5 sec
  - FPV forward pass (8 frames): 30 sec
  - Temporal Transformer: 10 sec
  - Top-view forward pass: 30 sec
  - Fusion + decision: 5 sec
  - **Total: ~80 sec (within budget)**

---

## Part 8: Evaluation Protocol (Revised)

### 8.1 Primary Metrics

| Metric | Definition | Target | Why |
|--------|-----------|--------|-----|
| **ECE (Expected Calibration Error)** | Mean |accuracy_in_bin - confidence_in_bin| across bins | < 0.08 | Reproducibility > accuracy |
| **Exact-match accuracy (all)** | % of samples where row AND col are correct | 60–80% on real val | Honest generalization measure |
| **Exact-match accuracy (high-conf, p > 0.85)** | % of high-confidence samples that are correct | > 85% | Minimize false positives |
| **"Confident refusal" rate** | % of samples deferred to human (p_max < 0.70) | 5–15% | Avoid overconfident errors |

### 8.2 Secondary Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| **Cross-view agreement** | % of validation samples where FPV and Top-view agree | > 70% |
| **Temporal coherence** | If event confidence decreases, does accuracy decrease? (correlation) | > 0.6 |
| **Synthetic-to-real transfer** | Accuracy on real vs. synthetic held-out sets | < 15% gap |

### 8.3 Acceptance Criteria

**The model is production-ready if:**
1. ECE < 0.08 on held-out real validation set
2. Exact-match accuracy > 80% on samples with p_max > 0.85
3. False positive rate < 10% (confident wrong predictions are rare)
4. Cross-view agreement > 70% (views consistently align)
5. "Confident refusal" rate between 5–15% (system knows when to defer)

---

## Part 9: Interview Aces — Future Directions

### 9.1 Acoustic Dispense Verification (Multimodal)

**Why this matters:** Transfyr captures synchronized FPV, top-view, AND audio. Current solution uses vision only.

**Approach:**
1. Record audio during dispense; extract spectrogram (Mel-frequency cepstral coefficients, MFCCs)
2. Train a small audio classifier (1D CNN) on 100 audio clips to detect "dispense sound" vs. "no dispense"
3. Use audio as a **sanity check:** If visual prediction = well A5 but audio confidence is low, flag as uncertain
4. **Cross-modal fusion:** Late fusion logic becomes: (visual + audio agreement) → high confidence

**Expected performance:**
- Audio classification accuracy: 85–95% (dispense produces distinctive "click" or "plink" sound)
- Combined audio+visual accuracy: 90–95% on real samples

### 9.2 Foundation Model Distillation (SmolVLA)

**Why this matters:** Transfyr's dataset includes video + language descriptions (e.g., "dispense 50 µL into well A3 for glucose assay").

**Approach:**
1. Fine-tune GPT-4o or Gemini 1.5 Pro on your 100 videos + text descriptions
2. Use GPT-4o predictions as pseudo-labels for ALL 10,000 synthetic videos (ground truth)
3. Train a small **SmolVLA** (400M parameters, Vision-Language-Action model) on the pseudo-labeled dataset
4. Inference: SmolVLA runs locally, 20 min batch for ~10 samples, no GPT-4o calls needed

**Expected performance:**
- SmolVLA accuracy on real samples: 75–85%
- SmolVLA speed: 30 sec per sample (faster than VideoMAE + temporal transformer)
- No dependency on GPT-4o at inference time

### 9.3 Visual Servoing (Closed-Loop Control)

**Why this matters:** Instead of one-shot prediction, treat well detection as a **control problem.**

**Approach:**
1. Run FPV model at 30 Hz; output: (row_pred, col_pred, error_vector)
2. Send error to robotic arm controller: "adjust pipette position by X mm to center on predicted well"
3. Loop: Predict → Adjust → Re-predict until error < 2 pixels (convergence)
4. **Benefit:** Even if first prediction is wrong, closed-loop correction fixes it

**Expected performance:**
- Open-loop accuracy (one shot): 60–70%
- Closed-loop accuracy (3–5 iterations): 90–95%
- Trade-off: Slower (10 sec per sample) but more robust

### 9.4 3D Gaussian Splatting for Refraction Correction

**Why this matters:** Red team identified "transparency risk": polystyrene wells refract light; 2D pixel position ≠ true well center.

**Approach:**
1. Reconstruct 3D scene (well geometry + liquid level) from FPV + Top-view using 3D Gaussian Splatting (3DGS)
2. Ray-trace from camera through reconstructed 3D scene; back-project to true well center (accounting for refraction)
3. Use 3D well center (in millimeters) instead of 2D pixel coordinates

**Expected performance:**
- Pixel-space accuracy: 75% (refraction causes 5–10 pixel error)
- 3D-space accuracy: 85–90% (accounts for refraction via geometry)

**Inference time:** 5–10 sec per sample (3DGS reconstruction is expensive)

---

## Part 10: Conclusion and Go/No-Go Criteria

### 10.1 Success Definition

The solution is ready for lab deployment when:

1. **Synthetic data quality validated:** FVD < 20 on synthetic vs. real videos; manual inspection shows realistic dispense sequences
2. **Backbone trained and frozen:** VideoMAE or DINO-v2 backbone converged on 10K samples
3. **ECE < 0.08 achieved:** On held-out real samples (N~10), calibration is scientific-grade
4. **"Confident refusal" working:** System defers ~10% of hard cases; those cases have actual labels obtained via manual inspection
5. **Cross-view agreement > 70%:** FPV and Top-view mostly agree on validation set
6. **Inference SLA met:** 20 minutes total for ~10 samples (end-to-end batch inference)

### 10.2 Go/No-Go Checklist

- [ ] VideoMAE fine-tuned on real 100 samples; FVD < 20 on synthetic dataset
- [ ] Temporal Transformer implemented; replaces frame max-pooling
- [ ] Late Fusion committed; FPV and Top-view trained separately with independent uncertainty
- [ ] ECE computed and tracked; validation accuracy > 80% at high confidence (p > 0.85)
- [ ] "Confident refusal" threshold tuned to 5–15% deferral rate on validation
- [ ] Cross-view agreement > 70% on held-out samples
- [ ] Inference speed validated: 20 min total for ~10 samples with 8-frame extraction (batch inference)
- [ ] Documentation updated: architecture diagrams, training logs, decision rationale
- [ ] Interview prep: future direction documents (acoustic, distillation, visual servoing, 3DGS) drafted

---

**Document Status:** DIRECTOR-ALIGNED  
**Next Review:** Post-prototype validation (4 weeks)  
**Owner:** ML Scientist (with red team feedback incorporated)
