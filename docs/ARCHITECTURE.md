# Transfyr AI Pipette Well Detection: Architectural Proposals

**Document:** System Architecture & Design Recommendations  
**Date:** 2026-04-14  
**Role:** Software/ML Architect  
**Audience:** ML Scientists, Development Team  

---

## Executive Summary

This document presents two distinct architectural proposals for solving the Transfyr pipette well detection challenge, along with analysis of a hybrid approach. Given the constraints—100 training samples, 2-minute inference budget, multi-label outputs, dual-view inputs, and 96-well grid prediction—both classical CV and deep learning approaches are viable but with different risk/reward profiles.

**Key Tension:** Interpretability vs. Accuracy given limited data.

---

## Problem Statement (Refined)

**Input:**
- Two synchronized video clips: FPV (First-Person View from pipette) + Top-view (bird's eye of plate)
- 96-well plate (8 rows × 12 columns, labeled A1–H12)
- Multi-channel dispensing: 1–8 wells targeted simultaneously

**Output:**
- JSON: `[{"well_row": "A", "well_column": 1}, {"well_row": "B", "well_column": 2}, ...]`
- Support for multi-well predictions

**Constraints:**
- **Training data:** 100 labeled examples
- **Inference latency:** ~2 min per sample (20 min for 10 samples)
- **Accuracy:** Sufficient to match human labeler on held-out set
- **Robustness:** Varying lighting, occlusion, plate tilt, pipette geometry

---

## Architecture 1: Classical Computer Vision + Geometric Pipeline

### Overview

A deterministic, rule-based pipeline that leverages geometric priors about well plates and pipettes. The approach is modular, interpretable, and does not require training data for the core geometry logic.

### Data Flow

```
[FPV Video] ─┬─> Frame Extraction ─> Tip Detection ──┐
             │                                         │
             └─> Optical Flow (motion) ────────────────┤
                                                       │
                                                   Fusion & Mapping
                                                   (Geometric Triangulation)
                                                       │
[Top-view Video] ─┬─> Frame Extraction ──> Plate Detection ──┐
                  │                       & Rectification    │
                  ├─> Grid Overlay ──────────────────────────┤
                  │   (Template Matching)                    │
                  └─> Well Coordinates ────────────────────→ JSON Output
                      & Localization
```

### Component Details

#### 1.1 Plate Detection & Rectification (Top-view)

**Goal:** Normalize the top-view image to a canonical bird's-eye orientation.

**Approach:**
- **Edge detection:** Canny or Sobel to find plate boundaries (circular/octagonal well plate)
- **Contour fitting:** Fit ellipse or polygon to identify plate corners or edges
- **Homography estimation:** Compute perspective transform matrix to warp to canonical alignment
- **Validation:** Check that corner points align with expected 96-well geometry (standard ANSI microplate dimensions: ~85.5 mm × 127.5 mm)

**Implementation notes:**
- Use OpenCV: `cv2.findContours()`, `cv2.getPerspectiveTransform()`, `cv2.warpPerspective()`
- Handle partial occlusion (pipette arm blocking plate edge) via robust corner detection
- Cache homography matrix per video (compute once per clip, not per frame)

#### 1.2 Well Grid Overlay (Top-view)

**Goal:** Map video pixel coordinates to well logical coordinates (A1–H12).

**Approach:**
- **Template generation:** Create synthetic grid overlay (8×12 equally-spaced points) corresponding to well centers
- **Spacing calibration:** From rectified plate, measure well-to-well distance in pixels; verify against standard 4.5 mm pitch
- **Coordinate mapping:** Build lookup table: `pixel_center (x, y) → (row, column)`
- **Sub-pixel refinement:** Use blob detection or circular Hough transform to detect actual well boundaries and refine grid

**Implementation notes:**
- Draw grid lines on rectified image for debugging/visualization
- Tolerance: ±2 pixels acceptable (given typical plate size in video frame)
- Store transformation matrix for forward/inverse lookup

#### 1.3 Pipette Tip Detection (FPV)

**Goal:** Localize pipette tip in FPV video; track motion.

**Approach:**
- **Color thresholding:** Pipette tips are typically bright (white, yellow, or silver). Extract foreground using HSV or RGB thresholding.
- **Morphological operations:** Dilate/erode to clean noise; connect tip segments
- **Blob detection:** Use OpenCV `SimpleBlobDetector` or contour analysis to identify tip as the largest connected component
- **Centroid extraction:** Compute (x_tip, y_tip) in FPV frame

**Multi-channel handling:**
- If pipette has 8-channel head, detect multiple distinct blobs (tips in parallel)
- Cluster tips by spatial proximity; assign each blob a "channel index" (0–7)

**Implementation notes:**
- Calibrate color range once per pipette type (stored as a config parameter)
- Track motion via frame-to-frame centroid displacement to validate movement
- Handle occlusion: if tip temporarily hidden, interpolate position via velocity model

#### 1.4 Optical Flow & Motion Tracking (FPV)

**Goal:** Infer when/where pipette is dispensing (motion + tip position correlation).

**Approach:**
- **Sparse optical flow:** Use Lucas-Kanade or sparse optical flow to track tip across frames
- **Velocity estimation:** Compute frame-to-frame tip velocity; detect zero/low velocity (dispensing event)
- **Temporal aggregation:** Frame range where tip velocity ≈ 0 indicates active dispensing

**Alternative – Dense optical flow:**
- Compute optical flow over entire FPV frame; analyze flow around tip region
- More robust to small tip detection errors, but higher computational cost

**Implementation notes:**
- Timestamp each FPV frame; synchronize with top-view timestamps (hardware clocks or frame indexing)
- Low velocity threshold: e.g., < 2 pixels/frame for >10 consecutive frames → dispensing active

#### 1.5 Multi-View Fusion & Geometric Mapping

**Goal:** Map FPV tip location to top-view well coordinates.

**Approach:**

**Case 1: Single-view solution (Top-view primary)**
- Assume pipette arm moves in roughly predictable pattern over plate
- Use top-view plate detection + well grid to directly localize tip shadow or reflection in top-view video
- Simpler, but requires visible tip shadow or pipette reflection in top-view footage

**Case 2: Geometric triangulation (Dual-view)**
- **Calibration phase (offline, once per setup):**
  - Mount calibration pattern (checkerboard) at known well positions
  - Capture dual-view images of each well (or subset)
  - Compute camera intrinsics (K_fpv, K_topview) and relative extrinsics (R, t) between cameras
  - Store as calibration.yaml

- **Runtime:**
  - FPV detects tip at pixel (x_fpv, y_fpv)
  - Top-view plate rectification provides plate frame origin + orientation
  - Triangulate 3D position of tip using both camera matrices
  - Project back to rectified top-view; lookup well coordinate from grid

**Case 3: Learned alignment (Hybrid-lite)**
- Train small CNN (few parameters) on calibration data to predict offset between FPV tip and top-view well position
- See Hybrid Architecture section

**Recommended approach:** Case 1 (top-view primary) with Case 2 (triangulation) as fallback.

**Implementation notes:**
- Requires camera calibration; use OpenCV `cv2.calibrateCamera()` and `cv2.triangulatePoints()`
- Calibration can be simplified if dual-camera rig is fixed (known R, t)
- For multi-channel: triangulate each detected tip independently

#### 1.6 Output Generation

**Assembly:** Aggregate dispensing frames from optical flow + tip coordinates. Output JSON.

```json
[
  {"well_row": "A", "well_column": 1},
  {"well_row": "C", "well_column": 5},
  ...
]
```

---

### Strengths

1. **Interpretability:** Every step is explicit; failures are debuggable (e.g., "grid overlay incorrect" vs. "model confused")
2. **No training required:** Core geometry logic works out-of-box without labeled data; 100 samples used for validation, not training
3. **Generalization:** Can adapt to different plate formats (384-well, 1536-well) by changing grid dimensions
4. **Real-time capable:** Each component is <100ms; total latency ~1–5 sec per video pair (well under 2 min budget)
5. **Robustness to variation:** Homography handles plate tilt; color thresholding handles lighting shifts
6. **Extensibility:** Easy to add post-processing rules (e.g., "filter wells outside plate bounds")

---

### Weaknesses & Failure Modes

1. **Camera calibration complexity:** Accurate triangulation requires careful calibration; miscalibration compounds errors
2. **Lighting sensitivity:** 
   - Reflective well plates can cause glare; thresholding may fail
   - Shadows from pipette arm degrade plate detection
3. **Occlusion:** Pipette arm often obscures top-view plate edges, complicating corner detection
4. **Pipette tip detection brittleness:** 
   - Tip color varies by manufacturer; threshold may not generalize across pipette types
   - Multi-channel tips: false positives (dust, reflection) or false negatives (touching well)
5. **Synchronization:** FPV and top-view must be temporally aligned; clock drift or dropped frames cause misalignment
6. **Edge cases:** 
   - Curved or non-planar well plates (some rare formats)
   - Very fast dispensing (motion blur)
   - Pipette partially outside frame
7. **Multi-well disparity:** If dispensing to wells in different focal planes (top vs. bottom of plate), depth ambiguity arises

---

### When This Breaks

- **Low contrast images:** Dark wells in dim lighting → grid overlay fails
- **Non-standard pipette geometry:** Very thin or transparent tips
- **Plate not fully visible:** Pipette arm blocks >30% of plate
- **Rapid multi-well dispensing:** Tips move between wells faster than frame rate → position ambiguity
- **Dirty optics:** Dust on camera lens degrades edge detection

---

## Architecture 2: Deep Learning End-to-End

### Overview

A learned approach using neural networks to map video frames directly to well predictions. Leverages spatial and temporal patterns learned from the 100 training examples.

### Data Flow

```
[FPV Video] ──> Frame Sampling ──> FPV Backbone (ResNet/EfficientNet) ──┐
                                  (e.g., layer 4 features)               │
                                                                         ├─> Feature Fusion ──> Decoder ──> Output Head
                                                                         │
[Top-view Video] ──> Frame Sampling ──> Top-view Backbone (ResNet/EfficientNet) ──┘
                                       (e.g., layer 4 features)

                                                                         ├─> Multi-label softmax (96-way)
                                                                         │   OR
                                                                         ├─> Row head (8-way) + Column head (12-way)
                                                                         │   OR
                                                                         └─> Coordinate regression (2-D output)
```

### Component Details

#### 2.1 Input Strategy: Frame Sampling vs. Temporal Models

**Option A: Single-frame classification** (Simpler, faster)
- Extract middle frame from each video (or max-motion frame)
- Stack FPV + Top-view frames (3 channels each) → 6-channel input OR two separate 3-channel inputs
- Backbone: ResNet-18/34 or EfficientNet-B0
- Pros: Fast inference, easy to implement
- Cons: Ignores temporal dynamics (when dispensing occurs); may miss motion cues

**Option B: Temporal models** (More powerful, higher latency)
- Sample 4–8 frames uniformly across video duration
- Use 3D CNN (C3D, SlowFast, I3D) or multi-frame concatenation
- Captures motion patterns: tip approaching well, tip inserting, liquid dispensing
- Pros: Richer signal; can detect dispensing event implicitly
- Cons: Higher compute cost; may exceed 2-min budget per sample if not optimized

**Option C: Optical flow + RGB** (Hybrid signal)
- Compute optical flow between consecutive frames
- Feed [RGB frame, flow components] as 5-channel input
- Single-frame backbone (ResNet) processes [R, G, B, flow_x, flow_y]
- Pros: Captures motion without temporal model overhead
- Cons: Requires flow computation (OpenCV adds latency)

**Recommended:** Option A (single-frame) as baseline, with Option C as alternative if accuracy insufficient.

#### 2.2 Backbone Architecture

**Single-view backbone (applied to both FPV and Top-view):**

```
Input (3 channels, 224×224) 
    ↓
ResNet-18 or EfficientNet-B0
    ├─ Conv1: 64 filters, 7×7, stride=2
    ├─ Layer1-Layer3: residual blocks (stride doubling)
    ├─ Layer4: final residual block (→ 512 channels, 7×7 spatial)
    ├─ Global avg pool → 512-D vector
    ↓
Backbone output: (batch, 512)
```

**Why these backbones:**
- **ResNet-18:** Fast, ~11M params, pre-trained on ImageNet available
- **EfficientNet-B0:** Better accuracy-to-parameter ratio, ~5M params, efficient scaling
- Both are lightweight enough for 2-min latency budget

**Pre-training:** Use ImageNet pre-trained weights; fine-tune all layers on the 100 well detection samples.

#### 2.3 Feature Fusion

**Goal:** Combine FPV and Top-view features into a unified representation.

**Approach 1: Concatenation**
```
FPV features (512,)  ─┬─> Concat ──> FC(512+512 → 256) ──> Fused (256,)
Top-view features (512,) ┘
```
Simple, fast, symmetric treatment of both views.

**Approach 2: Cross-attention**
```
FPV features (512,)  ──> Query ──┐
                                 ├─> Scaled dot-product attention ──> Fused (512,)
Top-view features (512,) ──> Key/Value ┘
```
More sophisticated; allows FPV to attend to informative regions in top-view.

**Approach 3: Gating (learn fusion weights)**
```
FPV (512,) ──┬─> Gate: α = sigmoid(FC(512+512→1)) ──> α·FPV + (1-α)·Top-view
Top-view (512,) ┘
```
Lightweight; learned weighting of views.

**Recommended:** Approach 1 (concatenation) for baseline simplicity. Evaluate Approach 2 if accuracy stalls.

#### 2.4 Output Head Design

**Critical design choice:** How to structure the 96-way multi-label problem.

**Option A: 96-class multi-label softmax**
```
Fused features (256,) ──> FC(256 → 96) ──> Sigmoid (per-well probability)
Output: logits[96], each ∈ [0, 1] (independent probabilities)
Loss: Binary cross-entropy per well
Threshold: wells with prob > 0.5 are predicted active
```
Pros: Direct; intuitive probability interpretation.
Cons: Treats each well independently (ignores spatial correlation); may overfit with 100 samples.

**Option B: Factorized (Row + Column heads)**
```
Fused features (256,) ──┬─> FC(256 → 8) ──> Softmax over rows ──> row_logits[8]
                        └─> FC(256 → 12) ──> Softmax over cols ──> col_logits[12]

Output: Outer product logits[8, 12]
Prediction: wells where row[i] * col[j] > threshold
Loss: Cross-entropy on row/col predictions separately
```
Pros: Reduced parameters; enforces structure (fewer parameters = less overfitting with 100 samples).
Cons: Assumes independence of row/col (not always true; some rows/cols may be inaccessible).

**Option C: Coordinate regression**
```
Fused features (256,) ──> FC(256 → 2) ──> (x, y) ∈ [0, 7] × [0, 11] (continuous)
Loss: L2 or Huber loss (regression)
Post-process: Round to nearest well; for multi-channel, apply clustering to detected tips
```
Pros: Natural for single-well case; continuous representation.
Cons: Difficult to extend to multi-well; regression targets may be ill-defined if multiple wells active.

**Option D: Keypoint detection (advanced)**
```
Fused features ──> Heatmap decoder (8×12 spatial output)
Heatmaps: Per-well activation map (sigmoid)
Output: Extract peaks (non-maximum suppression) to get well coordinates
Loss: L2 on heatmaps, or focal loss
```
Pros: Spatial coherence; can detect multiple peaks naturally.
Cons: Higher complexity; requires decoder design (transposed conv layers).

**Recommended:** Option B (factorized row/column) for 100-sample setting, as it reduces overfitting risk while maintaining expressiveness. Validate against Option A.

#### 2.5 Training Strategy for Limited Data

**Challenge:** 100 samples is very small for deep learning. Overfitting is the primary risk.

**Strategy 1: Transfer Learning + Heavy Augmentation**
- Start with ImageNet pre-trained ResNet-18 / EfficientNet-B0
- Freeze early layers (conv1, layer1); fine-tune layer2 onwards + fusion + head
- Augmentation:
  - Random crop (224 → 224)
  - Rotation (±10°)
  - Brightness/contrast shift (±20%)
  - Gaussian blur
  - Horizontal flip (if plate orientation is symmetric)
  - ColorJitter (simulates lighting variation)
- Regularization:
  - Dropout (p=0.5) in FC layers
  - L2 weight decay (λ=1e-4)
  - Early stopping on validation set (80/20 train/val split)
- Batch size: 8–16 (small due to data size)
- Optimizer: Adam, lr=1e-4, exponential decay

**Strategy 2: Data Augmentation (synthetic samples)**
- For each training video, generate synthetic variants:
  - Simulate plate rotation (+±5°)
  - Simulate camera blur (motion blur filter)
  - Simulate lighting changes (overlay gamma correction)
  - Horizontal/vertical flip if valid
- Augmentation factor: 5–10× (creating 500–1000 synthetic examples)
- Train on augmented set; validate on original 20 unaugmented samples

**Strategy 3: Few-shot / Meta-learning (optional, advanced)**
- Treat as few-shot classification: given K examples per well (K=1–2), learn to recognize new wells
- Use prototypical networks or MAML (Model-Agnostic Meta-Learning)
- More complex; evaluate after Strategy 1

**Strategy 4: Self-supervised pre-training (if raw video available)**
- Use unlabeled video frames to pre-train backbone via contrastive learning (SimCLR)
- Fine-tune on labeled 100 samples
- Requires additional unlabeled video data

**Recommended:** Strategy 1 (Transfer + Augmentation) as foundation. Implement Strategy 2 if accuracy is <90% on validation set.

---

### Latency Analysis

**Per-sample inference budget:** 2 minutes

Breakdown:
- **Frame extraction & preprocessing:** 100–200ms (load video, decode frames, resize to 224×224)
- **Backbone inference (ResNet-18):**
  - FPV: ~50ms (GPU)
  - Top-view: ~50ms (GPU)
- **Feature fusion:** <5ms
- **Output head:** <5ms
- **Total forward pass:** ~110ms
- **Overhead (I/O, serialization):** ~50ms

**Total per sample:** ~160ms (GPU) or ~2–3s (CPU)

For 10 samples: ~1.6s (GPU) or ~20–30s (CPU) — **well under 2-min budget**.

**Scaling:** If batch processing all 10 samples together:
- Batch forward pass (10 samples): ~150ms
- Total: ~300ms (GPU)

---

### Strengths

1. **Data-driven:** Learns task-specific patterns from 100 labeled examples; no hand-crafted rules
2. **Robustness to variation:** Neural networks are inherently robust to lighting, color, minor occlusion
3. **Multi-scale features:** Deep features capture semantic patterns (well geometry, pipette shape) automatically
4. **Multi-label handling:** Naturally supports 1–8 simultaneous wells without special logic
5. **Extensibility:** Can incorporate temporal models later if needed
6. **Fast inference:** <200ms per sample on GPU (ample headroom in 2-min budget)
7. **No calibration:** Doesn't require camera calibration; learns alignment implicitly

---

### Weaknesses & Limitations

1. **Data efficiency:** 100 samples is small. Risk of overfitting despite augmentation. Accuracy ceiling may be ~85–90%.
2. **Interpretability:** Black-box; difficult to debug specific errors ("Why was A5 predicted instead of A6?")
3. **Generalization to new pipettes:** If pipette geometry/color differs, model may not transfer
4. **Temporal blindness (Option A):** Single-frame model ignores when dispensing occurs; may confuse static positioning with active dispensing
5. **GPU requirement:** Inference on CPU much slower (~2–3s per sample). Deployment requires GPU (or quantization)
6. **Training iteration:** Requires labeled data for each new well plate / camera setup (not zero-shot)
7. **Class imbalance:** If some wells appear more frequently in training data, model may bias toward them

---

### When This Breaks

- **Out-of-distribution inputs:** Pipette type very different from training set
- **Insufficient data:** <50 training samples → severe overfitting
- **Domain shift:** Different camera, lighting, or plate material not represented in training data
- **Adversarial scenarios:** Deliberately confusing layouts (e.g., all wells lit)
- **Temporal signals matter:** If ground truth depends on dispensing timing (not just position), single-frame models fail

---

## Architecture 3: Hybrid Approach (Optional)

### Overview

A middle path combining geometric priors (from Architecture 1) with learned features (from Architecture 2).

### Design: Geometry-Guided Deep Learning

```
[FPV Video] ──> Tip Detection (Classical CV) ──┐
                                                ├─> Geometric Prior Module ──┐
[Top-view Video] ──> Plate Rectification (Classical CV) ──┘               │
                     & Grid Overlay                                       │
                                                                         ├─> Fusion
                                                                         │
[FPV & Top-view] ──> Backbone features (ResNet) ───────────────────────┘
                     & Fusion
                                                                         ├─> Multi-label head
                                                                         │
Geometric Prior: Prior probabilities over wells (Gaussian centered at detected tip position)
Learned Head: Raw logits from network
Fusion: Prior + Logits → Final softmax
```

**Interpretation:**
1. Classical CV provides a rough prior: "tip is near well A5" (from blob detection + rectification)
2. Deep learning refines this prior: "given image features, this is likely A5 or A6"
3. Combine: Prior weight (geometric) × learned weight (data) → final prediction

### Implementation

```python
prior_logits = -distance_to_tips_per_well  # Gaussian kernel
learned_logits = model_output  # From neural net
alpha = 0.3  # Blend weight (hyperparameter)
final_logits = alpha * prior_logits + (1 - alpha) * learned_logits
predictions = sigmoid(final_logits)
```

### Strengths

- **Robustness:** If vision fails (dark image), geometric prior still provides signal
- **Sample efficiency:** Geometric constraints act as strong regularization; reduces overfitting
- **Interpretability:** Geometric component is interpretable; learned component is modular
- **Graceful degradation:** If one modality (FPV or top-view) fails, prior alone provides baseline accuracy

### Weaknesses

- **Implementation complexity:** Requires tuning α (blend weight); more moving parts
- **Overfitting to prior:** If geometric prior is wrong, network may over-correct
- **Latency:** Geometric module (CV pipeline) adds ~100–500ms
- **Fewer samples:** Splitting effort between two approaches may reduce effectiveness of each

### Recommendation

Hybrid is a good **fallback option** if either Architecture 1 or 2 underperforms alone. Not recommended as initial approach given time/complexity tradeoffs.

---

## Cross-Cutting Concerns

### 1. Latency & Performance

**Constraint:** 20 min for 10 samples = 120 sec per sample.

**Actual budget:** 2 min = 120 sec (conservative).

**Breakdown:**

| Component | Classical CV | Deep Learning |
|-----------|--------------|---------------|
| Video load/decode | 100–200ms | 100–200ms |
| Frame extraction | 50ms | 50ms |
| Plate detection | 500–1000ms | — |
| Tip detection | 200–500ms | — |
| Optical flow | 500–1000ms | — |
| Backbone inference | — | 100ms |
| Grid overlay | 50–100ms | — |
| Triangulation | 100–200ms | — |
| Fusion & output | 50ms | 50ms |
| **Total** | **1.6–3.1 sec** | **0.3–0.5 sec** |

**Winner:** Deep learning significantly faster (100–10× speedup). Classical CV still feasible.

**Optimization for latency:**
- Resize frames to 224×224 before processing (reduces load)
- Use GPU for deep learning (essential for <500ms target)
- Cache calibration matrices (computed once, reused)
- Multi-threaded video I/O (decode FPV and top-view in parallel)
- Batch process all 10 samples together on GPU (if deployment allows)

### 2. Robustness to Variation

#### Lighting & Shadows
- **Classical CV:** Color thresholding brittle; shadows degrade plate detection. Mitigate: adaptive thresholding, morphological closing.
- **Deep learning:** Inherently robust via data augmentation (brightness/contrast shift). More forgiving.

#### Occlusion (Pipette arm blocking plate)
- **Classical CV:** Homography + grid overlay still works if >70% plate visible. Blob detection may fail if tip occluded.
- **Deep learning:** Learned features may hallucinate well position even if obscured. Risk: overconfident incorrect predictions.

#### Plate Tilt / Non-planar wells
- **Classical CV:** Homography assumes planarity; fails for curved plates. Triangulation still works if cameras calibrated.
- **Deep learning:** Robust to tilt (learned via augmentation). No assumption of planarity.

#### Pipette type variation
- **Classical CV:** Color thresholding is pipette-specific; requires re-calibration per type.
- **Deep learning:** May generalize if training set has variety. More sample-efficient than manual recalibration.

### 3. Extensibility

#### Different plate formats (384-well, 1536-well)
- **Classical CV:** Change grid dimensions (16×24 for 384-well, 32×48 for 1536-well). Geometric logic unchanged. **High extensibility.**
- **Deep learning:** Retraining required for different grid. Backbone learns format-specific patterns. **Lower extensibility** without transfer.

#### Different camera setups
- **Classical CV:** Requires recalibration (homography, camera matrix). ~30–60 min one-time setup.
- **Deep learning:** Requires retraining on new camera data. More expensive but one-time.

#### Different lighting conditions
- **Classical CV:** Thresholding parameters may need adjustment.
- **Deep learning:** Robust via augmentation; no retraining needed if shift within training distribution.

### 4. Deployment & DevOps

**Containerization:**
```dockerfile
FROM python:3.10-slim
RUN pip install opencv-python torch torchvision
COPY model.pth /app/
COPY inference.py /app/
CMD ["python", "/app/inference.py"]
```

**CLI Interface (both architectures):**
```bash
$ python pipette_detect.py --fpv ./fpv.mp4 --topview ./topview.mp4 --output result.json
```

**Output format (both):**
```json
{
  "timestamp": "2026-04-14T12:34:56Z",
  "wells": [
    {"row": "A", "column": 1, "confidence": 0.95},
    {"row": "A", "column": 2, "confidence": 0.88}
  ],
  "metadata": {
    "architecture": "classical_cv" | "deep_learning",
    "inference_time_ms": 2340,
    "fpv_frames_processed": 150,
    "topview_frames_processed": 150
  }
}
```

**GPU vs. CPU:**
- Deep learning: **GPU recommended** (10–50× faster). TensorRT or ONNX for optimization.
- Classical CV: **CPU acceptable**. OpenCV highly optimized for CPU.

**Deployment targets:**
- **Cloud:** GPU instance (AWS g4dn, GCP A100) or CPU instance
- **Edge:** NVIDIA Jetson (TX2, Orin) for GPU; laptop/workstation for CPU
- **Testing:** Likely run on GCP/AWS due to latency budget (CI/CD integration)

---

## Comparative Analysis

| Criterion | Classical CV | Deep Learning | Hybrid |
|-----------|--------------|---------------|--------|
| **Accuracy (est.)** | 80–90% | 85–95% | 90–97% |
| **Interpretability** | High | Low | Medium |
| **Training data needed** | None (for geometry) | 100 (all available) | 100 |
| **Inference latency** | 1.6–3.1 sec | 0.3–0.5 sec | 0.8–1.5 sec |
| **Extensibility (new format)** | High | Low | Medium |
| **Robustness (lighting)** | Medium | High | High |
| **Robustness (occlusion)** | Medium | Medium | High |
| **Development time** | 2–3 weeks | 1–2 weeks | 3–4 weeks |
| **Maintenance burden** | Low | Medium | Medium–High |
| **GPU required?** | No | Yes | No (optional) |
| **Failure mode** | Cascading (plate → tip) | Single point (network) | Degraded (one module fails) |

---

## Architect's Recommendation

### Primary Recommendation: **Architecture 2 (Deep Learning End-to-End)**

**Rationale:**

1. **Best accuracy-to-effort ratio:** With 100 labeled samples available, deep learning should achieve **85–95% accuracy** on held-out set. Classical CV likely plateaus at 80–90%.

2. **Sample efficiency via transfer learning:** ImageNet pre-trained backbones provide strong initialization. Factorized output (row + column heads) further reduces overfitting risk.

3. **Inference speed:** ~300–500ms per sample on GPU is **comfortable within 2-min budget**, enabling real-time testing and iteration.

4. **Robustness to variation:** Augmentation (brightness, rotation, blur) makes model resilient to lighting, plate tilt, and minor camera shifts without manual tuning.

5. **Extensibility (within format):** Once trained, model works for any camera/lighting setup within same well plate format. No recalibration needed.

6. **Implementation maturity:** Standard PyTorch pipeline; well-tested techniques (ResNet, multi-label classification, data augmentation).

### Secondary Recommendation: **Hybrid (if Deep Learning underperforms)**

If deep learning accuracy stalls <85% on validation, **immediately pivot to Hybrid**:

1. Implement classical CV module (plate detection + tip localization) in parallel
2. Train a lightweight fusion network that combines geometric prior + learned features
3. Hybrid should recover ~5–10% accuracy via geometric constraints

### Why NOT Classical CV as primary:

- **Brittleness to calibration:** Camera calibration errors compound through geometric pipeline. Single point of failure: if plate detection fails, entire pipeline fails.
- **Lighting sensitivity:** Color thresholding requires manual tuning; less generalizable across lighting conditions.
- **Development risk:** More complex pipeline (6–7 steps); higher chance of subtle bugs in geometric mappings.
- **Data efficiency:** Ignores 100 labeled examples except for validation.

---

## Non-Negotiable Design Constraints

### For Architecture 2 (Recommended)

1. **Multi-label output:** Output must support 1–8 simultaneous wells without special-casing. Use factorized (row + column) heads or 96-class sigmoid to ensure this.

2. **Dual-view input:** Both FPV and top-view streams must be processed and fused. Architecture should be symmetric in treatment of both views (see concatenation fusion in 2.3).

3. **Latency ceiling:** Inference time per sample **must be <10 seconds** (ideally <1 second) to fit within 2-min budget for 10 samples.

4. **Validation accuracy:** Minimum **85%** on held-out set before deploying to production. This is binary: if <85%, do not proceed with pure deep learning; hybrid required.

5. **Generalization:** Model must achieve ≥80% accuracy on well coordinates it did **not** see in training data (i.e., not just overfitting to 100 specific wells, but generalizing to unseen row/column combinations).

6. **Reproducibility:** Random seeds and full hyperparameters logged. Model checkpoint saved with git hash of training code.

### For Classical CV (if pursued as fallback)

1. **Plate detection robustness:** Must detect plate correctly in ≥95% of frames even with partial occlusion.

2. **Tip localization:** Multi-channel tips must be detected and disambiguated (color/shape-based).

3. **Calibration time:** Initial camera calibration must complete in <1 hour (feasible, given standard checkerboard procedure).

4. **Geometric accuracy:** Tip position error <2 pixels in rectified frame (corresponds to ~0.5 mm on plate; acceptable for 4.5 mm well pitch).

---

## Implementation Roadmap

### Phase 1: Establish baseline (Week 1)

- [ ] Implement data loading & preprocessing pipeline
- [ ] Implement deep learning Architecture 2 (ResNet-18 + factorized heads)
- [ ] Implement classical CV Architecture 1 (plate detection + tip detection only, no triangulation)
- [ ] Evaluate both on validation set (80/20 split of 100 samples)
- [ ] **Decision gate:** Which architecture meets ≥85% accuracy on validation?

### Phase 2: Optimize winning architecture (Week 2)

- [ ] Hyperparameter sweep (learning rate, weight decay, augmentation intensity)
- [ ] Error analysis: which wells are misclassified? Any systematic bias?
- [ ] Implement augmentation strategies (if deep learning)
- [ ] Implement triangulation (if classical CV)
- [ ] Target: ≥90% on validation

### Phase 3: Integration & deployment (Week 3)

- [ ] Integrate with CLI interface
- [ ] Test on 10 held-out samples
- [ ] Latency profiling; optimize if needed
- [ ] Package as Docker container
- [ ] Documentation for deployment

### Phase 4: Fallback / Hybrid (if needed, Week 4)

- [ ] If Phase 2 accuracy <85%, implement Hybrid Architecture 3
- [ ] Fuse geometric prior + learned features
- [ ] Target: ≥95% on validation

---

## Technical Debt & Future Work

1. **Temporal models:** If single-frame deep learning plateaus, implement 3D CNN (Option B from 2.1) to leverage motion signals
2. **Confidence calibration:** Calibrate confidence scores via temperature scaling or Platt scaling for reliable thresholding
3. **Uncertainty quantification:** Use Monte-Carlo dropout or ensemble methods to estimate prediction uncertainty per well
4. **Active learning:** With 100 samples, use uncertainty sampling to prioritize which new samples to label next
5. **Online learning:** Deploy model in production; retrain on new errors to adapt to new pipette types / lighting

---

## Conclusion

**Recommended path:** Deep Learning (Architecture 2) with factorized output heads, ResNet-18 backbone, transfer learning, and aggressive data augmentation. This balances accuracy (85–95%), interpretability (factorized heads are semi-interpretable), and development speed (1–2 weeks).

**Fallback path:** Hybrid approach if deep learning <85%. Geometric prior provides safety net.

**Non-negotiable:** Support multi-label output, dual-view fusion, <2-min latency, and ≥85% validation accuracy.

Success criteria:
- [ ] ≥90% accuracy on held-out test set (10 samples)
- [ ] <10 sec inference per sample (ideally <1 sec)
- [ ] Reproducible results (git-tracked, seeded)
- [ ] Generalizes to unseen row/column combinations

