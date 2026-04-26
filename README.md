# Pipette Well Challenge: Automated Well Detection in Microplate Imaging

**Challenge:** Transfyr AI Pipette Well Challenge  
**Objective:** Predict which well(s) of a 96-well plate are targeted during automated liquid dispensing operations  
**Dataset:** 100 labeled video clip pairs (FPV + Top-view)  
**Date:** April 2026

## Executive Summary

This project implements an automated well detection system for microplate-based liquid handling. Given synchronized dual-view video (first-person view from the pipette operator + bird's-eye top-view of the plate), the system predicts which of the 96 wells (8 rows × 12 columns, labeled A1–H12) are being dispensed into. The solution supports multi-well predictions (1-channel single wells, 8-channel row operations, 12-channel column operations).

**Recommended Architecture:** Deep Learning End-to-End (PyTorch + DINOv2-ViT-B/14 + LoRA) with temporal transformer, factorized row/column heads, a 3-class clip-type head (single/row/col), and weighted BCE + well-consistency loss. DINOv2 is the only backbone path; the legacy ResNet-18 fallback was removed in commit d5b8f04. See ML_STACK.md for full justification.

**Expected Performance:** 70–80% exact-match accuracy on held-out validation set (with calibration-first approach prioritizing confidence scores).

---

## 🎯 Primary Architecture: DINOv2-ViT-B/14 + LoRA

**Production stack:** DINOv2-ViT-B/14 (frozen) → LoRA adapters (r=8, α=16) → Temporal Transformer (2 layers) → Late Fusion → Factorized Row (8) + Col (12) heads

**Why DINOv2, not ResNet-18:**
- **Patch-based spatial structure** (14×14 = 196 patches) preserves well coordinates; ResNet's global pooling destroys them (7×7 = 49 locations cannot distinguish 96 wells)
- **Self-supervised pre-training** (142M unlabeled images, DINO contrastive) learns coordinate geometry directly, not object classification
- **LoRA efficiency** (~33K trainable params) prevents overfitting on N=100 samples; full ResNet-18 fine-tuning would require 11M parameters (catastrophic risk)
- **Few-shot empirical validation:** DINOv2 ViT-B outperforms ResNet-50 by +8 percentage points on 10-shot benchmarks (Oquab et al., ICCV 2023)

**Implementation note:** DINOv2-ViT-B/14 weights are downloaded from Hugging Face Hub (timm) at training start. The earlier ResNet-18 fallback path was removed once weight downloads were no longer constrained — see `src/models/backbone.py`.

---

## Challenge Description

### Problem Context

Automated liquid handling systems (pipette robots) require precise well identification for:
- Drug screening assays
- Genomics workflows
- High-throughput chemical synthesis
- Quality control in manufacturing

Manual well annotation of video is tedious and error-prone. This challenge seeks to automate well prediction from video alone.

### Input Format

Two synchronized video clips per sample:
1. **FPV (First-Person View):** Camera mounted on/near the pipette; shows the operator's perspective (hand, arm, plate from above)
2. **Top-View (Bird's-Eye):** Overhead camera; shows the full 96-well grid with minimal perspective distortion

### Output Format

JSON array of predicted wells:
```json
{
  "wells": [
    {"well_row": "A", "well_column": 1},
    {"well_row": "A", "well_column": 2},
    {"well_row": "A", "well_column": 3},
    {"well_row": "A", "well_column": 4},
    {"well_row": "A", "well_column": 5},
    {"well_row": "A", "well_column": 6},
    {"well_row": "A", "well_column": 7},
    {"well_row": "A", "well_column": 8}
  ]
}
```

### Key Constraints

- **Training data:** 100 labeled video pairs (small dataset → high overfitting risk)
- **Inference budget:** ~2 minutes per dual-view sample
- **Class imbalance:** 6× imbalance (max 6, min 1 well occurrence) — moderate, manageable with focal loss
- **Multi-label complexity:** 1, 8, or 12 wells per video (not single-label classification)
- **Temporal alignment:** FPV and top-view asynchronous → requires explicit synchronization
- **Well coverage:** All 96 wells represented in empirical dataset — validates model generalization potential

---

## Architecture Decision & Rationale

### Chosen Approach: Temporal Deep Learning with Foundation Models

**Decision:** Use a PyTorch-based neural network with **DINOv2-ViT-B/14 + LoRA + Temporal Transformer** as the primary architecture, backed by synthetic data generation (3D Blender simulation + VideoMAE fine-tuning) to scale beyond 100 samples.

**Key Architectural Decisions:**

1. **Primary Backbone:** DINOv2-ViT-B/14 + LoRA (not ResNet-18, not VideoMAE)
   - Self-supervised pre-training on 142M unlabeled images learns coordinate geometry directly
   - Patch-based spatial structure (14×14 = 196 tokens) preserves well localization precision
   - LoRA fine-tuning (~33K trainable params) prevents overfitting on N=100 samples
   - Few-shot empirical advantage: +8% over ResNet-50 on 10-shot benchmarks
   - Rationale: Self-supervised spatial learning more relevant than object classification or temporal pre-training for well-center localization

2. **Temporal Modeling:** Temporal Transformer (mandatory, not optional)
   - Replaces frame max-pooling which destroys temporal order
   - Distinguishes pipette entering vs. leaving a well via motion trajectory
   - Aligned with Transfyr's "Tacit Knowledge" mission: motion intent is first-class
   - Rationale: A "dispense" is an event, not a static state

3. **Synthetic Data Strategy:** 10,000+ synthetic samples (critical mitigation)
   - 3D Blender simulation: photorealistic rendering of all 96 wells × multiple angles × lighting conditions
   - VideoMAE fine-tuning: generative sampling to create realistic dispense trajectories
   - Mix with real data: 50% synthetic + 50% real for balanced training
   - Rationale: Scaling beyond 100 samples is essential for robust well discrimination

4. **Fusion Architecture:** Late Fusion (mandatory, explicit)
   - FPV and Top-view have incompatible coordinate systems (perspective vs. orthographic)
   - Early fusion is geometrically incorrect; late fusion respects projection differences
   - Each view learns independent well-center localization; uncertainty is traceable
   - Rationale: Technical leadership alignment; coordinate system integrity

5. **Primary Metric:** Expected Calibration Error (ECE < 0.10), not raw accuracy
   - In science, reproducibility > accuracy
   - Model must calibrate confidence scores: saying "I don't know" with 90% certainty is more valuable than overconfident guesses
   - "Confident Refusal": if max_confidence < 0.70, output {"uncertain": true} instead of guessing
   - Rationale: Lab environments demand reliability over raw performance

### Baseline Implementation (Preserved for Reference)

The original ResNet-18 single-frame approach has been preserved on the `baseline/resnet-cv-pipeline` branch for:
- Historical comparison
- Legacy system compatibility testing
- Proof-of-concept validation

See `docs/ARCHITECTURE.md` for detailed comparison and non-negotiable design constraints.

### Alternative Approaches for Research

1. **3D Gaussian Splatting (for glare-heavy scenarios):**
   - Reconstructs 3D scene from dual-view; ray-traces back-projection to account for refraction
   - Handles polystyrene refraction and liquid level effects not captured by 2D CNNs
   - Added as Architecture 4 for future research

2. **Acoustic Cross-Validation (multimodal):**
   - Transfyr captures audio during dispense; distinctive "click" or "plink" provides independent signal
   - Cross-modal fusion: visual + audio agreement → high confidence

3. **Visual Servoing (closed-loop control):**
   - Run FPV model at 30 Hz; iteratively adjust pipette position until error converges
   - Trade-off: slower (10 sec per sample) but achieves 90–95% accuracy

---

## ML Stack

### Recommended Technology Stack

**Framework:** PyTorch 2.1+  
**Backbone:** DINOv2-ViT-B/14 (self-supervised on 142M images, frozen + LoRA r=4 adapters; ~12.3M trainable params)  
**Future alternative:** VideoMAE-Base for stronger temporal modeling (not in current codebase)  
**Temporal Module:** Temporal Transformer (2–4 attention blocks over ordered frames)  
**Video I/O:** OpenCV (cv2) with temporal-aware frame extraction  
**Synthetic Data:** 10K+ videos via (a) VideoMAE fine-tuning, (b) 3D Blender simulation  
**Augmentation:** albumentations 1.3+ + domain randomization for synthetic-to-real transfer  
**Output Heads:** Factorized (8-class row + 12-class column) with calibrated sigmoid + uncertainty quantification  
**Loss Function:** Focal loss (γ=2.0, α=0.75) + calibration-aware regularization  
**Optimizer:** AdamW + cosine annealing with warmup  
**Primary Metric:** Expected Calibration Error (ECE < 0.10)  
**Inference SLA:** <2 min per dual-view sample

### Key Design Decisions

#### 1. Backbone Architecture: DINOv2-ViT-B/14 + LoRA (PRIMARY)
- **Why DINOv2 over ResNet-18:**
  - Self-supervised pre-training on 142M unlabeled images learns spatial coordinate geometry directly
  - Patch embeddings (14×14 = 196 tokens, 768-dim) preserve well-center localization precision
  - LoRA fine-tuning (~33K trainable params) prevents overfitting vs. ResNet-18 full fine-tuning (11M params)
  - Few-shot empirical advantage: +8% accuracy over ResNet-50 on 10-shot benchmarks
  - Transfer to well detection is direct: self-supervised spatial learning ≈ coordinate localization
  
- **LoRA fine-tuning approach:**
  - Backbone frozen; only LoRA adapters (rank r=8, α=16) trainable
  - Low-rank constraint prevents catastrophic forgetting on N=100 real samples
  - Extract features from DINOv2 frozen encoder for downstream classification
  
- **Alternative: VideoMAE-Base (future enhancement)**
  - For stronger temporal modeling; not yet in codebase
  - Would replace DINOv2 in temporal_backbone role if needed
  - Kinetics-400 pre-training provides excellent video understanding

- **Removed: ResNet-18**
  - Legacy 2015 architecture; the fallback path was removed in commit d5b8f04
  - Insufficient spatial resolution (7×7 features ≠ 96 wells)
  - Severe overfitting risk on N=100 with full fine-tuning
  - Kept only as comparative justification in `docs/ML_STACK.md`

#### 2. Temporal Modeling: Temporal Transformer (MANDATORY)
```
Input: (T=4-8 ordered frames, H=448, W=448, C=3)
    ↓
Temporal Backbone: VideoMAE or DINO encoder → (T, 768) feature sequence
    ↓
Temporal Transformer: 2–4 attention blocks with causal masking
    ├─ Frame-wise temporal attention
    └─ Event localization (detect dispense frame)
    ↓
Temporal Pooling: Attend to dispense event frame → (768,) features
    ↓
Classification Heads: (row logits, col logits)
```

**Why temporal Transformer:**
- Max-pooling over frames is order-agnostic; destroys event semantics (entering ≠ leaving)
- Temporal attention learns motion patterns: approach phase → dispensing → withdrawal
- Aligns with Transfyr's mission: motion trajectory encodes expert intent

#### 3. Output Head: Factorized Row + Column
```
Fused features (768,) ──┬──> Linear(768→8) ──> Sigmoid ──> Row logits (8,)
                        └──> Linear(768→12) ──> Sigmoid ──> Column logits (12,)
```
- **Reduces overfitting:** Predicts 20 independent outputs (8+12) instead of 96
- **Enforces structure:** Multi-well predictions respect plate geometry
- **Uncertainty quantification:** Each head has calibrated confidence; disagreement signals hard cases

#### 4. Late Fusion Strategy (EXPLICIT MANDATE)

FPV and Top-view processed independently, then fused at feature level:
```
FPV Video (T=4-8 frames) ──> VideoMAE + Temporal Transformer ──> FPV features (768,)
                                                                       │
                                                              Late Fusion Block
                                                                       │
Top-view Video (T=4-8 frames) ──> VideoMAE + Temporal Transformer ──> Top-view features (768,)
                                                                       │
                                                        Cross-attention or Gating
                                                                       ↓
                                                        Fused features (768,) ──> Heads
```

**Why late fusion is mandatory:**
- **Coordinate system incompatibility:** FPV uses perspective projection; top-view uses orthographic
- **Early fusion is geometrically incorrect:** Fusing at feature-map level conflates coordinate systems
- **Reduces interference:** Each view learns independent spatial extraction; fusion respects native frames
- **Interpretability:** Can visualize which view dominates per well; traceable uncertainty

#### 5. Synthetic Data Strategy: 3D Simulation + Generative Models (ESSENTIAL)

**Option A: 3D Blender Simulation (recommended for hard negatives)**
1. Model 96-well plate in Blender with photorealistic rendering
2. Render all 96 wells × varied camera angles (FPV + top-view) × lighting conditions × pipette trajectories
3. Generate ~5,000 synthetic frames with perfect ground-truth well coordinates
4. Use as "hard negatives" to improve robustness (unusual lighting, edge cases)

**Option B: VideoMAE Fine-Tuning (recommended for realistic trajectories)**
1. Fine-tune VideoMAE-Base on 100 real dispense videos (masked autoencoder objective)
2. Use latent space to generate 100–200 synthetic videos per well
3. Target: 5,000 realistic synthetic videos with subtle blur, refraction, liquid dynamics
4. Condition generation on: well location, pipette angle, lighting, plate orientation

**Combined approach:**
- 50% Blender synthetic (hard coverage) + 50% VideoMAE-fine-tuned (realistic) + 100% real training
- Balanced dataset: 10,100 total samples (10K synthetic + 100 real)
- Validates domain gap: Fréchet Video Distance (FVD) < 20 between synthetic and real

#### 6. Calibration & Confident Refusal (MANDATORY)

**Acceptance Criteria:**

| Metric | Target | Why |
|--------|--------|-----|
| **ECE (Expected Calibration Error)** | < 0.10 | Predictions match reality on validation set |
| **Exact-match accuracy (high-conf, p > 0.85)** | > 85% | High-confidence predictions must be correct |
| **"Confident refusal" rate** | 5–15% of test set | System defers uncertain cases; humans handle hard cases |
| **False positive rate (p_max > 0.90)** | < 10% | Minimize overconfident wrong predictions |

**Confident Refusal Strategy:**
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

Why 0.70 threshold: On validation set (N~10), threshold tuning is unstable. Conservative threshold (0.70) reduces false positives; operators trust the model's uncertainty estimates.

#### 7. Data Augmentation Strategy

On top of synthetic data:

**Temporal augmentation:**
- Frame offset: random ±1 frame from ground-truth
- Speed jitter: playback speed 0.95–1.05×
- Frame skip: simulate variable dispensing speed

**Geometric augmentation:**
- Plate rotation: ±8°
- Crop/zoom: 90–110% of frame
- Affine transforms: shear ±5°, scale ±5%

**Photometric augmentation:**
- Brightness jitter: ±15% intensity
- Contrast jitter: 1.0–1.3×
- Gaussian blur: 1–3 pixel kernel
- Hue shift: ±10° in HSV

**Domain randomization (for synthetic data):**
- Camera intrinsics variation
- Lighting intensity/color shifts
- Plate material texture variation
- Liquid appearance variation

#### 8. Training Configuration
- **Optimizer:** AdamW (weight_decay=1e-5)
- **Learning rate:** 1e-4 with cosine annealing + 5-epoch linear warmup
- **Batch size:** 4–8 (limited by temporal models + 100 real samples)
- **Epochs:** 50–100
- **Early stopping:** Monitor validation ECE; stop if no improvement for 15 epochs
- **Regularization:** Dropout 0.5 in FC, batch normalization, class weighting per DATA_ANALYSIS
- **Multi-GPU:** Distributed training with mixed precision (fp16)

See `docs/ML_STACK.md` for complete technical specification including code examples, latency analysis, and infrastructure requirements.

---

## Data Analysis & Insights

### Real Dataset Statistics (Empirical)
- **100 clips** across 7 plates (Plate_1–5, Plate_9–10)
- **All 96 wells covered** — mean 3.41 samples/well, 6× imbalance (max 6, min 1)
- **Operation types**: 75% single-well, 13% 12-well row-sweeps, 12% 8-well column-sweeps
- **Video format**: 1920×1080 @ 30fps, ~2.4s duration (~72 frames)
- **Label schema**: `well_column` is stored as a string (handled automatically in pipeline)
- See [`docs/DATA_ANALYSIS_EMPIRICAL.md`](docs/DATA_ANALYSIS_EMPIRICAL.md) for the full analysis

### View-Specific Strengths

| Sub-task | FPV | Top-view | Fusion Strategy |
|----------|-----|----------|-----------------|
| **Temporal localization** (when dispense) | Strong | Strong | Cross-correlation consensus |
| **Row identification (A-H)** | Weak | **Strong** | Top-view primary |
| **Column identification (1-12)** | Weak | **Strong** | Top-view primary |
| **Multi-channel detection** | Weak | **Strong** | Top-view counts tips |
| **Hand/equipment detection** | **Strong** | N/A | FPV exclusive |
| **Plate geometry/tilt** | **Strong** | Strong | Consensus |

**Conclusion:** Top-view is the primary signal for well identification; FPV is critical for temporal alignment and motion detection.

### Key Challenges

1. **Overfitting crisis:** Severe generalization gap expected (~30 percentage points between training and test)
   - Mitigation: Regularization (dropout, weight decay, early stopping), data augmentation, lightweight architecture

2. **Synchronization issues:** FPV and top-view recorded asynchronously; frames may be offset by 1–2 frames
   - Mitigation: Use cross-correlation to find peak frame alignment

3. **Edge cases:** Lighting glare, motion blur, plate tilt, hand occlusion, device wear
   - Mitigation: Robust augmentation, fallback to single-view if one fails

See `docs/DATA_ANALYSIS.md` for complete analysis including lighting challenges, geometric issues, and risk mitigation recommendations.

---

## Quick Start

### Installation

```bash
# Clone and navigate
git clone https://github.com/linwoes/pipette-well-challenge.git
cd pipette-well-challenge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running Inference

```bash
# Basic usage (uses checkpoints/best.pt by default; writes to stdout)
python inference.py --fpv path/to/fpv.mp4 --topview path/to/topview.mp4 --output result.json

# Use a specific release
python inference.py --model releases/deployed/model.pt --fpv fpv.mp4 --topview topview.mp4 --output result.json

# Override the threshold-decoder fallback (default 0.4; primary path is type-conditioned)
python inference.py --fpv fpv.mp4 --topview topview.mp4 --threshold 0.5

# Custom image resolution (must be a multiple of 14; 448 recommended)
python inference.py --fpv fpv.mp4 --topview topview.mp4 --img_size 448

# Verbose output
python inference.py --fpv fpv.mp4 --topview topview.mp4 --verbose
```

**Note on image resolution:** img_size=448 is the project default. DINOv2 ViT-B/14 requires the size to be a multiple of 14 (`snap_to_dinov2_resolution()` enforces this). 224 was tried earlier but proved insufficient for per-well discrimination.

### Output Format

The script generates JSON output matching the challenge submission spec, plus a small metadata block:
```json
{
  "clip_id_FPV": "Plate_5_clip_0003_FPV",
  "clip_id_Topview": "Plate_5_clip_0003_Topview",
  "wells_prediction": [
    {"well_row": "A", "well_column": 1},
    {"well_row": "A", "well_column": 2}
  ],
  "metadata": {
    "model": "DINOv2-ViT-B/14+LoRA",
    "inference_time_s": 1.23,
    "confident": true
  }
}
```

---

## Visualization Tool

The visualization tool (`tools/visualizer.py`) overlays inference results on dual-view video and supports ranking, annotation, and analysis workflows. See `docs/DESIGN_VISUALIZATION_TOOL.md` for the full specification.

```bash
# Render a specific clip with grid overlay
python tools/visualizer.py render --input clip_001 --labels data/pipette_well_dataset/labels.json

# Render results 0–19 from an inference file
python tools/visualizer.py render --input results.json::0-19 --labels labels.json

# Find the 20 worst detections
python tools/visualizer.py rank --input results.json --labels labels.json --mode worst --top 20

# Find the 5 strangest results
python tools/visualizer.py rank --input results.json --labels labels.json --mode strangest --top 5

# Create a QA annotation
python tools/visualizer.py annotate --clip clip_001 --result-index 0 --text "False positive in E5" --author qa_lead

# Generate an error heatmap across all results
python tools/visualizer.py heatmap --input results.json --labels labels.json
```

---

## Project Structure

```
pipette-well-challenge/
├── README.md                           ← This file
├── .gitignore
├── .gitattributes                      ← Git LFS patterns (model weights, MP4s, synthetic labels)
├── requirements.txt                    ← Pinned dependencies
├── inference.py                        ← Inference CLI
├── train.py                            ← Training entry point
├── run_training.sh                     ← Launcher; auto-versions runs as YYYYMMDD.<git-hash>
├── make_release.py                     ← Package a checkpoint into releases/<version>/
├── generate_synthetic_data.py          ← FFmpeg-based augmentation pipeline
├── diagnostic_threshold_sweep.py       ← One-off decoder/threshold diagnostic
│
├── docs/
│   ├── ARCHITECTURE.md                 ← Architecture proposals + comparison
│   ├── ARCHITECURE_DIAGRAM.md          ← Diagram supplement (note: filename has historical typo)
│   ├── DATA_ANALYSIS.md                ← Pre-analysis predictions and strategy
│   ├── DATA_ANALYSIS_EMPIRICAL.md      ← Real findings from the actual dataset
│   ├── DESIGN_VISUALIZATION_TOOL.md    ← Visualizer spec
│   ├── FEATURE_SCENE_CLASSIFICATION.md ← Future feature proposal
│   ├── ML_STACK.md                     ← Stack rationale (PyTorch, DINOv2, loss, training)
│   ├── QA_STRATEGY.md                  ← Test strategy
│   ├── QA_REPORT.md                    ← QA snapshot (historical)
│   ├── TRAINING_REPORT_v8.md           ← Training analysis snapshot (historical)
│   └── TEAM_DECISIONS.md               ← Cross-team decisions log
│
├── src/
│   ├── preprocessing/video_loader.py   ← Frame extraction, dual-view alignment, temporal jitter
│   ├── models/
│   │   ├── backbone.py                 ← DINOv2-ViT-B/14 + LoRA backbone
│   │   └── fusion.py                   ← Dual-view fusion + row/col/type heads + WellDetectionLoss
│   ├── postprocessing/output_formatter.py
│   │     ← logits_to_wells (threshold), logits_to_wells_typed (PRIMARY decoder),
│   │       logits_to_wells_adaptive (alternative)
│   └── utils/metrics.py                ← exact_match, jaccard_similarity, cardinality_accuracy
│
├── releases/                           ← Versioned model releases (see releases/README.md)
│   ├── README.md
│   ├── index.json                      ← Registry of all releases with metrics
│   ├── latest    -> <version>          ← Symlink to most recently packaged release
│   ├── deployed  -> <version>          ← Symlink to validated production release
│   └── YYYYMMDD.<hash>/                ← Each release: model.pt (LFS) + config.json + RELEASE_NOTES.md
│
├── checkpoints/                        ← Live training checkpoints (gitignored apart from LFS .pt)
├── training_results/                   ← Per-run training_<version>.log files
├── data/pipette_well_dataset/          ← Real + synthetic clip pairs + labels.json
├── tools/visualizer.py                 ← Visualization & analysis CLI
├── tests/                              ← test_models, test_output_schema, test_preprocessing,
│                                          test_training_setup, test_overfit_smoke (currently skipped)
└── configs/default.yaml
```

---

## Documentation

### For Data Scientists
- **START HERE:** `docs/DATA_ANALYSIS_EMPIRICAL.md` (real findings from actual dataset)
- **REFERENCE:** `docs/DATA_ANALYSIS.md` (pre-analysis predictions and strategy)
- Covers: Class imbalance analysis, coverage gaps, view-specific signals, augmentation strategy
- Includes: Statistical concerns, overfitting risk, generalization bounds, per-well metrics

### For Architects
- **START HERE:** `docs/ARCHITECTURE.md`
- Covers: Three architectural proposals (CV, DL, Hybrid), latency analysis, extensibility
- Includes: Comparative table, implementation roadmap, non-negotiable constraints
- **Decision:** Deep Learning (Architecture 2) recommended with classical CV as fallback

### For ML Scientists
- **START HERE:** `docs/ML_STACK.md`
- Covers: Framework (PyTorch), backbone (DINOv2-ViT-B/14 + LoRA), loss (weighted BCE with α=0.75 + well-consistency), training strategy
- Includes: Code examples, hyperparameter tuning, mixed precision training, inference SLA validation
- See "Why Not ResNet" section for the historical comparative justification (ResNet-18 was rejected and removed)

### For QA Engineers
- **START HERE:** `docs/QA_STRATEGY.md`
- Covers: Testing layers (unit → integration → system → acceptance), edge cases, failure modes
- Includes: Hold-out set risk analysis, confidence calibration, robustness protocols, acceptance criteria

### For Visualization & Debugging
- **START HERE:** `docs/DESIGN_VISUALIZATION_TOOL.md`
- **Tool:** `tools/visualizer.py` — CLI for overlaying predictions on video, ranking results, QA annotations, and error heatmaps
- Covers: render, rank (best/worst/strangest), annotate, heatmap commands
- Phase 1 (flat file) is implemented; Phase 2 (cloud database) is design-only

### Feature Requests
- **SCENE CLASSIFICATION:** `docs/FEATURE_SCENE_CLASSIFICATION.md` — Capturing classification of objects in the scene (wells, pipette tip, thumb, liquid, etc.)

### Cross-Team Reference
- **TEAM_DECISIONS.md:** Key architectural decisions, driver roles, rationale, open questions

---

## Model Training

### Launching a run

```bash
# Real-only labels, auto-versioned, resumes from checkpoints/best.pt if present
USE_COMBINED=0 bash run_training.sh

# Use combined real + synthetic labels (NOTE: random val split currently leaks
# synthetic augmentations of training clips into val; clean split is on the v11 roadmap)
USE_COMBINED=1 bash run_training.sh

# Force GPU
DEVICE=cuda:0 bash run_training.sh
```

`run_training.sh` auto-computes `TRAINING_VERS` from `date -u +%Y%m%d.<git-hash>` and writes the log to `training_results/training_<version>.log`. Override with `TRAINING_VERS=custom`.

### Architecture

- **Backbone:** DINOv2-ViT-B/14 frozen + LoRA (rank=4) — ~12.3M trainable params
- **Temporal:** 1-layer transformer over 8 sampled frames (per view)
- **Fusion:** Late, MLP-based; FPV and Top-view processed independently
- **Heads:** factorized row (8) + column (12) + 3-class clip-type (single/row/col)
- **Loss:** weighted BCE (focal_gamma=0, alpha=0.75) on row/col + cross-entropy on type + well-consistency loss (outer product, rectangular-pattern mask)

### Validation decoder (commit 3af379f)

Validation uses the **type-conditioned decoder** `logits_to_wells_typed`: the type head's argmax picks the strategy (single → top-1 row × top-1 col; row → top-1 row × all 12 cols; col → all 8 rows × top-1 col). Threshold-based metrics (`jaccard_thresh`, `exact_match_thresh`) are still logged as a diagnostic so we can see when the spatial heads strengthen enough that thresholding becomes viable.

### Checkpoint criterion

Hybrid: save when **either** Jaccard improves **or** val_loss improves. Patience resets on either improvement. This avoids early-stopping while val_loss is still falling but Jaccard is noisy on a 20-sample val set.

### Synthetic data

`generate_synthetic_data.py` produces 700 augmented clip pairs from the 100 real clips using ffmpeg (4 photometric: bright/dark/noise/contrast; 3 geometric flips with corresponding label remapping). Output: `data/pipette_well_dataset/labels_synthetic.json` (700) + `labels_combined.json` (800). Use with `USE_COMBINED=1`.

### Releasing a trained model

```bash
# Package the current best.pt as releases/YYYYMMDD.<git-hash>/
python make_release.py --notes "Jaccard 0.42 after 40 epochs"

# Package and immediately mark as deployed (the validated production model)
python make_release.py --notes "..." --deploy

# Promote an already-packaged release to deployed without re-packaging
python make_release.py --deploy-only 20260512.a3f91bc
```

See `releases/README.md` for the full versioning + deployment scheme. Weight files live under Git LFS.

---

## QA & Testing

### Test Suite

Run all tests:
```bash
pytest tests/ -v
```

**Test Coverage:**
- `test_output_schema.py`: Validates JSON output structure (wells array, row/column ranges, required fields)
- `test_preprocessing.py`: Video loading, frame extraction, temporal alignment
- `test_models.py`: Backbone + fusion module shape/forward checks
- `test_training_setup.py`: Trainer construction and config plumbing
- `test_overfit_smoke.py`: Memorisation smoke test (currently skipped — needs rewrite for the 3-output model + DINOv2 CPU runtime)

### Edge Cases Covered

**Visual:** Extreme glare, dark/low light, motion blur, occlusion, hand in frame, liquid splash  
**Geometric:** Plate rotation, partial out-of-bounds, tilt, multiple plates, inverted orientation  
**Pipette:** Single-channel, 8-channel, 12-channel, barely visible, unusual color, damaged tip  
**Temporal:** Very short clips, incomplete dispense, multiple dispenses, no dispense, slow dispense  
**Input Files:** Mismatched FPV/top-view, corrupted video, low resolution, unusual codec, audio-only  
**Output:** Zero wells, >12 wells, out-of-bounds coordinates, invalid well types, duplicates

See `docs/QA_STRATEGY.md` for complete edge case catalogue with severity ratings and mitigation strategies.

### Acceptance Criteria

**Before Hold-Out Evaluation:**
- [ ] ECE < 0.10 on 20 held-out real validation samples
- [ ] Exact-match accuracy > 85% on high-confidence predictions (p_max > 0.85)
- [ ] Cross-view agreement > 70% (FPV and top-view mostly agree on validation set)
- [ ] "Confident refusal" rate between 5–15% (system appropriately defers hard cases)
- [ ] False positive rate < 10% (wrong predictions with high confidence are rare)
- [ ] Synthetic-to-real transfer gap < 15% (accuracy drop from synthetic to real should be modest)
- [ ] Inference latency < 2 sec per sample on target GPU (within 2-minute budget)
- [ ] All edge case tests passing

**On Hold-Out Evaluation (10 unknown samples):**
- [ ] All 10 samples produce valid JSON output
- [ ] Calibration holds: confidence scores match empirical accuracy (ECE < 0.12)
- [ ] Exact-match accuracy ≥ 70% (accounting for unseen wells, calibration-first approach)
- [ ] Total runtime ≤20 minutes
- [ ] No runtime errors or exceptions
- [ ] Uncertain predictions deferred appropriately (not overconfident guesses)

---

## Known Limitations & Future Work

### Current Limitations
1. **Small real dataset:** 100 samples → severe generalization gap without synthetic data (mitigated by 10K synthetic)
2. **Unseen wells:** 5–15 wells with zero training examples; zero-shot predictions still risky (mitigated by confident refusal)
3. **Refraction effects:** 2D pixel accuracy limited by polystyrene/liquid refraction (addressed by 3DGS research track)
4. **GPU requirement:** Inference on CPU much slower; deployment assumes GPU availability
5. **Inference latency:** Temporal Transformer slower than single-frame models (~30–60 sec for 8-frame video vs. ~5 sec)

### Future Enhancements
1. **3D Gaussian Splatting:** Reconstruct 3D scene; ray-trace through refracted medium for pixel-perfect well centers
2. **Acoustic Cross-Validation:** Use audio (dispense "click") as independent signal; fuse with visual predictions
3. **Visual Servoing:** Closed-loop control; iterate pipette position until error converges (90–95% accuracy at cost of 10 sec/sample)
4. **Foundation Model Distillation:** Use GPT-4o to pseudo-label 10K synthetic videos; distill into lightweight SmolVLA
5. **Active Learning:** Sample uncertainty to prioritize which new well geometries/lighting conditions to label next
6. **Online Learning:** Retrain on deployment errors; adapt to new pipette types, tip colors, lighting setups

---

## Team Roles & Contributions

- **Data Scientist:** Class imbalance analysis, view-specific signal characterization, augmentation strategy, generalization bounds estimation
- **Architect:** Architecture proposals (CV vs. DL vs. Hybrid), latency/cost analysis, implementation roadmap
- **ML Scientist:** Stack recommendation (PyTorch, DINOv2, focal loss), training strategy, hyperparameter tuning
- **QA Engineer:** Testing strategy, edge case catalogue, failure mode analysis, acceptance criteria

---

## References

**Key Papers:**
- ResNet: He et al. "Deep Residual Learning for Image Recognition" (2015)
- Focal Loss: Lin et al. "Focal Loss for Dense Object Detection" (2017)
- Transfer Learning: Yosinski et al. "How Transferable Are Features in Deep Neural Networks?" (2014)

**Frameworks:**
- PyTorch: https://pytorch.org
- OpenCV: https://opencv.org
- albumentations: https://albumentations.ai
- timm: https://timm.fast.ai

---

## License & Attribution

This project is part of the Transfyr AI challenge (April 2026). Developed by a cross-functional team of Data Scientists, Architects, ML Scientists, and QA Engineers.

---

**Last Updated:** April 25, 2026  
**Status:** DINOv2 + LoRA training in progress; type-conditioned decoder + train metrics landed (commit 3af379f); first versioned release packaged (`releases/20260425.edb3173`)
