# Pipette Well Challenge: Automated Well Detection in Microplate Imaging

**Challenge:** Transfyr AI Pipette Well Challenge  
**Objective:** Predict which well(s) of a 96-well plate are targeted during automated liquid dispensing operations  
**Dataset:** 100 labeled video clip pairs (FPV + Top-view)  
**Date:** April 2026

## Executive Summary

This project implements an automated well detection system for microplate-based liquid handling. Given synchronized dual-view video (first-person view from the pipette operator + bird's-eye top-view of the plate), the system predicts which of the 96 wells (8 rows × 12 columns, labeled A1–H12) are being dispensed into. The solution supports multi-well predictions (1-channel single wells, 8-channel row operations, 12-channel column operations).

**Recommended Architecture:** Deep Learning End-to-End (PyTorch + ResNet-18) with factorized row/column output heads, transfer learning from ImageNet, and focal loss for handling severe class imbalance.

**Expected Performance:** 80–90% exact-match accuracy on held-out validation set.

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
- **Class imbalance:** 96 wells, 100 samples → extreme long-tail distribution
- **Multi-label complexity:** 1, 8, or 12 wells per video (not single-label classification)
- **Temporal alignment:** FPV and top-view asynchronous → requires explicit synchronization

---

## Architecture Decision & Rationale

### Chosen Approach: Deep Learning End-to-End

**Decision:** Use a PyTorch-based neural network with transfer learning from ImageNet pre-trained ResNet-18 backbone.

**Rationale (from ARCHITECTURE.md):**

1. **Data efficiency:** 100 labeled samples available; transfer learning leverages ImageNet priors to compensate for small training set
2. **Accuracy potential:** 85–95% accuracy achievable (vs. 80–90% for classical CV)
3. **Robustness:** Neural networks inherently robust to lighting variation, minor occlusion, and plate tilt via learned feature representations
4. **Inference speed:** <500ms per sample on GPU (ample headroom within 2-min budget)
5. **Extensibility:** Works for any camera/lighting setup within same plate format
6. **Development maturity:** Standard PyTorch pipeline with well-tested techniques

### Alternative Approaches Evaluated

1. **Classical Computer Vision (Geometric Pipeline):**
   - Pros: Interpretable, no training required, generalizable to different plate formats
   - Cons: Brittle calibration, lighting-sensitive color thresholding, cascading failures
   - Estimated accuracy: 80–90%

2. **Hybrid (Geometric priors + Deep Learning):**
   - Pros: Robust fallback if either approach fails individually
   - Cons: Implementation complexity, requires tuning blend weights
   - Recommended as secondary fallback if DL accuracy <85%

See `docs/ARCHITECTURE.md` for detailed comparison and non-negotiable design constraints.

---

## ML Stack

### Recommended Technology Stack

**Framework:** PyTorch 2.0+  
**Backbone:** ResNet-18 (ImageNet pre-trained, ~11M parameters)  
**Video I/O:** OpenCV (cv2)  
**Augmentation:** albumentations 1.3+  
**Output Heads:** Factorized (8-class row + 12-class column)  
**Loss Function:** Focal loss (γ=2.0) with weighted BCE per head  
**Optimizer:** AdamW + cosine annealing with warmup  
**Inference SLA:** <2 min per dual-view sample

### Key Design Decisions

#### 1. Backbone Architecture (ResNet-18)
- 11M parameters: lightweight enough to avoid catastrophic overfitting on 100 samples
- ImageNet pre-training provides strong spatial priors (edges, textures, shapes)
- Inference speed: ~20–50ms on GPU; acceptable for budget
- Preferred over MobileNetV3 (5M params) due to better transfer learning support

#### 2. Output Head: Factorized Row + Column
```
Input: (B, 3, 224, 224) ──> ResNet-18 ──> (B, 512) ──┬──> Linear(512→8) ──> Row logits (8,)
                                                      └──> Linear(512→12) ──> Column logits (12,)
```
- **Reduces overfitting:** Predicts 20 independent binary outputs (8+12) instead of 96
- **Enforces structure:** Multi-well predictions respect plate geometry (all selected wells in same row/column)
- **Multi-label semantics:** Sigmoid activation allows simultaneous row/column activation (supports 1-well, 8-well, 12-well operations)

#### 3. Focal Loss for Class Imbalance
```
Focal loss: FL(p_t) = -α(1 - p_t)^γ log(p_t), γ=2.0
```
- Addresses severe class imbalance: 5–15 wells with zero training samples
- Down-weights easy negatives; focuses on hard positives
- Per-head application: separate focal losses for row and column predictions
- Optional class weighting: rare wells weighted 2–10× higher than frequent wells

#### 4. Dual-View Fusion Strategy

FPV and Top-view processed independently, then concatenated:
```
FPV stream:    (B,3,224,224) ──> ResNet-18 ──> (B,512)
                                                    │
Top-view:      (B,3,224,224) ──> ResNet-18 ──> (B,512)  ──> Concat ──> (B,1024) ──> FC ──> Heads
```

**Rationale:**
- FPV and top-view have different coordinate systems (perspective vs. orthogonal)
- Late fusion (after spatial feature extraction) preserves view-specific information
- Symmetric treatment of both views

#### 5. Data Augmentation Strategy (mandatory given 100 samples)

**Temporal augmentation (highest priority):**
- Frame offset: random ±3 frames from ground-truth
- Speed jitter: playback speed 0.9–1.1×
- Frame interpolation: linearly interpolate missing frames
- Expected boost: 3–5× effective dataset size

**Geometric augmentation:**
- Plate rotation: ±8°
- Crop/zoom: 90–110% of well ROI
- Affine transforms: shear ±5°, scale ±5%
- Expected boost: 2–3× additional diversity

**Photometric augmentation:**
- Brightness jitter: ±15% intensity
- Contrast jitter: 1.0–1.3×
- Gaussian blur: 1–3 pixel kernel
- Hue shift: ±10° in HSV

#### 6. Training Configuration
- **Optimizer:** AdamW (weight_decay=1e-5)
- **Learning rate:** 1e-4 with cosine annealing + 5-epoch linear warmup
- **Batch size:** 8–16 (limited by 100 samples)
- **Epochs:** 50
- **Early stopping:** Monitor validation loss; stop if no improvement for 10 epochs
- **Regularization:** Dropout 0.3 in FC layers, batch normalization, class weighting

See `docs/ML_STACK.md` for complete technical specification including code examples, latency analysis, and infrastructure requirements.

---

## Data Analysis & Insights

### Dataset Characteristics (100 samples)

**Well coverage distribution:**
- Standard pipetting order (row → column sweep) creates non-uniform sampling
- Multi-channel operations (8-channel: full rows, 12-channel: full columns) cause correlation
- **Estimated coverage:** 40–60 unique wells have training examples
- **Estimated gap:** 5–15 wells have zero training samples

**Class imbalance magnitude:**
- Most frequent wells (corners, edges): 5–10 samples each
- Least frequent wells (isolated interior): 0–2 samples each
- **Imbalance ratio:** 10:1 to 50:1 worst case

**Multi-label structure:**
- Single-channel: ~35 samples (1 well per sample)
- 8-channel rows: ~40 samples (8 wells per sample)
- 12-channel columns: ~20 samples (12 wells per sample)
- Total well-instances: ~610 (but only ~100 unique training samples)

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
# Basic usage
python inference.py --fpv path/to/fpv.mp4 --topview path/to/topview.mp4 --output result.json

# With custom config
python inference.py --fpv fpv.mp4 --topview topview.mp4 --output result.json --config configs/custom.yaml

# Verbose output
python inference.py --fpv fpv.mp4 --topview topview.mp4 --output result.json --verbose
```

### Output Format

The script generates JSON output:
```json
{
  "wells": [
    {"well_row": "A", "well_column": 1},
    {"well_row": "A", "well_column": 2}
  ],
  "metadata": {
    "inference_time_seconds": 1.23,
    "confidence_threshold": 0.5,
    "fpv_frames_analyzed": 150,
    "topview_frames_analyzed": 150
  }
}
```

---

## Project Structure

```
pipette-well-challenge/
├── README.md                           ← This file
├── .gitignore                          ← Python, video files, models
├── requirements.txt                    ← Pinned dependencies
├── inference.py                        ← CLI entrypoint (skeleton with placeholders)
│
├── docs/
│   ├── DATA_ANALYSIS.md                ← Data Scientist's analysis (96 well coverage, class imbalance, view strengths/weaknesses)
│   ├── ARCHITECTURE.md                 ← Architect's proposals (3 approaches: CV, DL, Hybrid)
│   ├── ML_STACK.md                     ← ML Scientist's stack recommendation (PyTorch, ResNet-18, focal loss details)
│   ├── QA_STRATEGY.md                  ← QA strategy (testing, edge cases, failure modes, acceptance criteria)
│   └── TEAM_DECISIONS.md               ← Cross-team decisions log (architecture choice, loss function, evaluation metrics)
│
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── video_loader.py             ← Frame extraction, FPV+top-view synchronization
│   ├── models/
│   │   ├── __init__.py
│   │   ├── backbone.py                 ← ResNet-18 backbone definition
│   │   └── fusion.py                   ← Dual-view feature fusion logic
│   ├── postprocessing/
│   │   ├── __init__.py
│   │   └── output_formatter.py         ← Convert logits to JSON well predictions
│   └── utils/
│       ├── __init__.py
│       └── metrics.py                  ← Evaluation metrics (per-well accuracy, cardinality accuracy, etc.)
│
├── tests/
│   ├── test_output_schema.py           ← JSON schema validation (wells array, row/column ranges, etc.)
│   └── test_preprocessing.py           ← Frame extraction, synchronization tests
│
└── configs/
    └── default.yaml                    ← Default configuration (frame sampling, thresholds, model paths)
```

---

## Documentation

### For Data Scientists
- **START HERE:** `docs/DATA_ANALYSIS.md`
- Covers: Class imbalance analysis, coverage gaps, view-specific signals, augmentation strategy
- Includes: Statistical concerns, overfitting risk, generalization bounds, per-well metrics

### For Architects
- **START HERE:** `docs/ARCHITECTURE.md`
- Covers: Three architectural proposals (CV, DL, Hybrid), latency analysis, extensibility
- Includes: Comparative table, implementation roadmap, non-negotiable constraints
- **Decision:** Deep Learning (Architecture 2) recommended with classical CV as fallback

### For ML Scientists
- **START HERE:** `docs/ML_STACK.md`
- Covers: Framework selection (PyTorch), backbone (ResNet-18), loss function (focal loss), training strategy
- Includes: Code examples, hyperparameter tuning, mixed precision training, inference SLA validation

### For QA Engineers
- **START HERE:** `docs/QA_STRATEGY.md`
- Covers: Testing layers (unit → integration → system → acceptance), edge cases, failure modes
- Includes: Hold-out set risk analysis, confidence calibration, robustness protocols, acceptance criteria

### Cross-Team Reference
- **TEAM_DECISIONS.md:** Key architectural decisions, driver roles, rationale, open questions

---

## Model Training (Skeleton Only)

The `inference.py` file contains a placeholder skeleton for the inference CLI. To implement full training:

1. **Implement `src/models/backbone.py`:**
   - Load ResNet-18 from torchvision
   - Freeze early layers initially
   - Unfreeze layer2+ for fine-tuning

2. **Implement `src/models/fusion.py`:**
   - Concatenate FPV and top-view features
   - Pass through FC layers
   - Output row and column logits

3. **Implement training loop:**
   - Data loading and augmentation (albumentations)
   - Focal loss computation
   - Validation on held-out set
   - Early stopping with best checkpoint tracking

4. **Implement inference pipeline:**
   - Frame extraction from dual videos
   - Temporal synchronization
   - Model forward pass
   - Post-processing (threshold, cardinality constraints)

See `ML_STACK.md` for detailed pseudocode and implementation notes.

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
- [ ] Validation accuracy ≥85% on 20 held-out validation samples
- [ ] Per-well cardinality accuracy ≥80% (1-channel, 8-channel, 12-channel separately)
- [ ] Inference latency <1 second per sample on target GPU
- [ ] All edge case tests passing

**On Hold-Out Evaluation (10 unknown samples):**
- [ ] All 10 samples produce valid JSON output
- [ ] Accuracy ≥80% on well coordinates (accounting for unseen wells)
- [ ] Total runtime ≤20 minutes
- [ ] No runtime errors or exceptions
- [ ] Predictions consistent with visual inspection

---

## Known Limitations & Future Work

### Current Limitations
1. **Small dataset:** 100 samples → severe overfitting risk; estimated 30% generalization gap
2. **Unseen wells:** 5–15 wells with zero training examples; zero-shot predictions on these are guesses
3. **Temporal blindness (single-frame model):** If temporal cues matter, upgrade to 3D CNN
4. **GPU requirement:** Inference on CPU much slower (~2–3s per sample); deployment assumes GPU availability
5. **Lighting sensitivity:** While augmentation helps, extreme lighting shifts may degrade performance

### Future Enhancements
1. **Temporal models:** Implement 3D CNN or LSTM to leverage motion signals (tip approach trajectory)
2. **Confidence calibration:** Use temperature scaling or Platt scaling for reliable confidence scores
3. **Uncertainty quantification:** Monte-Carlo dropout or ensemble methods for prediction intervals
4. **Active learning:** Use uncertainty sampling to prioritize which new samples to label
5. **Online learning:** Retrain on deployment errors to adapt to new pipette types/lighting
6. **Multi-modal fusion:** Incorporate audio (lab noise cues) or auxiliary sensors

---

## Team Roles & Contributions

- **Data Scientist:** Class imbalance analysis, view-specific signal characterization, augmentation strategy, generalization bounds estimation
- **Architect:** Architecture proposals (CV vs. DL vs. Hybrid), latency/cost analysis, implementation roadmap
- **ML Scientist:** Stack recommendation (PyTorch, ResNet-18, focal loss), training strategy, hyperparameter tuning
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

**Last Updated:** April 14, 2026  
**Status:** Scaffold & documentation complete; model training skeleton ready for implementation
