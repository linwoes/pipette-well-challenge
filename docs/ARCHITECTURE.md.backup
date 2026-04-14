# Transfyr AI Pipette Well Detection: Architectural Proposals

**Document:** System Architecture & Design Recommendations  
**Date:** 2026-04-14  
**Role:** Software/ML Architect  
**Audience:** ML Scientists, Development Team  
**Status:** REVISED (Red Team Feedback Integrated)

---

## Executive Summary

This document presents four distinct architectural proposals for solving the Transfyr pipette well detection challenge: Classical CV, Deep Learning (with temporal modeling), Hybrid, and 3D Gaussian Splatting. Given the constraints—100 training samples, 2-minute inference budget, multi-label outputs, dual-view inputs, and 96-well grid prediction—the choice depends on physical accuracy requirements.

**Critical Insight:** The lab bench is a dynamic 3D physical environment. The pipette is a rigid object moving through space, the plate shifts and tilts, and polystyrene/translucent materials introduce specular reflection and refraction. Robust solutions must model physics explicitly, not treat the task as 2D image classification.

**Key Tension:** 2D pixel accuracy vs. 3D scene geometry; interpretability vs. physical plausibility.

---

## Physical AI Design Principles

Before diving into architectures, we establish a foundation: **Physical AI treats robotics/lab tasks as problems in scene geometry, material properties, and causal relationships—not pixel classification.**

### Core Principles

1. **Scene Geometry Over Pixel Classification**
   - The well plate has 3D structure: surface depth, well depth, liquid level
   - The pipette is a 3D object with pose (position + orientation)
   - The task is: "At what 3D coordinate in lab space is the tip?" not "What pixel in image space is bright?"

2. **Camera Projection Incompatibility**
   - FPV (first-person view) uses perspective projection: objects at different depths project differently
   - Top-view uses approximate orthographic projection: pixel position ≈ world position
   - Early fusion of these feature maps is geometrically incorrect; late fusion respects projection differences

3. **Material Properties Matter**
   - Polystyrene (plate) reflects and refracts light specularly
   - Translucent pipette tips transmit and refract liquid
   - Lambertian assumptions (used in 2D grid detection) fail; depth-aware reasoning is required

4. **Temporal Semantics**
   - A "dispense" is an event (tip enters → tip stays → tip leaves)
   - Max-pooling destroys temporal order; motion trajectory encodes expert intent
   - VLA architectures make action sequences first-class objects

### Relationship to Transfyr's Mission

Transfyr's "Tacit Knowledge" research posits that expert lab technicians execute tasks via learned motor patterns. The pipette trajectory (motion sequence) is a first-class signal, not ancillary. Architectures should respect this by modeling temporal dynamics of the action.

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
- **Inference latency:** 20 minutes total for ~10 samples (batch inference)
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
4. **Real-time capable:** Each component is <100ms; total latency ~1–5 sec per video pair (well under 20 min batch budget)
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

## Architecture 2: Deep Learning with Temporal Modeling

### Overview

A learned approach using neural networks to model video sequences as action trajectories, mapping temporal patterns directly to well predictions. Leverages spatial and temporal patterns learned from the 100 training examples, with explicit temporal attention to identify dispense events.

### Data Flow

```
[FPV Video] ──> Frame Sampling ──> Temporal Backbone ──┐
               (4–8 frames)     (TimeSformer/VideoMAE)  │
                                or 3D CNN              │
                                                       ├─> Temporal Fusion ──> Dispense Event ──> Output Head
                                                       │   (Cross-attention)   Decoder
[Top-view Video] ──> Frame Sampling ──> Temporal Backbone ──┘
                    (4–8 frames)    (TimeSformer/VideoMAE)

                                                       ├─> Multi-label softmax (96-way)
                                                       │   OR
                                                       ├─> Row head (8-way) + Column head (12-way)
                                                       │   OR
                                                       └─> Sequence-to-label (frame → well ID at dispense)
```

**Key Difference from v1:** Temporal models (TimeSformer, VideoMAE, I3D) treat frames as an ordered sequence, not a bag of images. Temporal attention identifies the dispense event (entrance signature in FPV motion) rather than relying on frame pooling.

### Component Details

#### 2.1 Temporal Modeling: Core Architecture Decision

**RED TEAM FINDING:** Max-pooling over frames destroys temporal order. A "dispense" is a causal event (tip entrance → well insertion → liquid release), not a state. Single-frame models cannot distinguish tip entering vs. leaving.

**Required Approach: Temporal Attention Models**

**Option A: Single-frame + 3D backbone (DEPRECATED for this task)**
- Extract middle frame; ignore motion
- Pros: Fast, simple
- Cons: **DOES NOT WORK** for this task (cannot distinguish dispense phases)
- Status: REJECTED based on red team critique and Transfyr's "Tacit Knowledge" mission

**Option B: Temporal Models with explicit event detection (RECOMMENDED)**
- Sample 4–8 frames uniformly across video duration
- Use **TimeSformer** (vision transformer with temporal attention) or **VideoMAE** (masked autoencoder on video clips)
- Design: Temporal attention learns motion patterns; output an "event decoder" that identifies frame index where dispense occurs
- Flow: [T, H, W, 3] → Temporal attention → Per-frame logits + frame-level event probability
- Pros: Captures motion semantics; identifies dispense event; generalizes to variable-speed dispensing
- Cons: Higher compute cost (~500ms–2s per sample)

**Option C: 3D CNN backbone (SlowFast, I3D)**
- Stack frames as a temporal dimension; convolve spatiotemporally
- SlowFast: dual pathways (slow=semantic, fast=motion)
- Pros: Proven on action recognition; lower latency than transformers (~500ms)
- Cons: Requires careful sampling strategy; less interpretable than attention

**Option D: Optical flow + Temporal CNN**
- Compute optical flow per frame; stack with RGB
- Feed [R, G, B, flow_x, flow_y] to 3D CNN or 2D CNN with temporal convolution
- Pros: Explicit motion signal
- Cons: Optical flow adds preprocessing cost

**COMMITMENT:** Recommended **Option B (TimeSformer)** with fallback to **Option C (SlowFast)** if compute budget exceeded. Option A (single-frame) is no longer viable for this architecture.

#### 2.2 Temporal Backbone Architecture

**For Option B (TimeSformer):**

```
Input: (T=8 frames, H=224, W=224, C=3)
    ↓
TimeSformer-Base (ImageNet pre-trained, then video pre-trained on Kinetics-400)
    ├─ Patch embedding: (H×W, C) → (196, 768) [14×14 patches]
    ├─ Temporal + spatial attention (interleaved blocks)
    │   Each block: [temporal attention over T] → [spatial attention over HW]
    ├─ Final classification token (CLS) → (T, 768)
    ├─ Temporal aggregation (mean pooling) → 768-D vector
    ↓
Backbone output: (batch, 768)
```

**For Option C (SlowFast):**

```
Input: (T=8 frames, H=224, W=224, C=3)
    ↓
SlowFast Backbone
    ├─ Slow pathway: Sample every 2 frames (T_slow=4) → 3D conv (lower framerate, semantic)
    ├─ Fast pathway: All frames (T_fast=8) → 3D conv with smaller filters (motion detail)
    ├─ Lateral fusion: Connect pathways via fused blocks
    ├─ Global avg pool → Concatenate slow + fast features
    ↓
Backbone output: (batch, 2048)  [slow: 1024 + fast: 1024]
```

**Why these backbones:**
- **TimeSformer:** State-of-the-art on video understanding (Kinetics-400, Something-Something-v2); explicit temporal attention interpretable; pre-training on video motion is directly relevant
- **SlowFast:** Battle-tested on action recognition; dual-pathway design naturally captures semantic + motion; lower latency than transformers
- Both are pre-trainable on large-scale video datasets (Kinetics-400, Kinetics-700)

**Pre-training:** Use Kinetics-400 pre-trained weights; fine-tune all layers on the 100 well detection samples. Kinetics pre-training is more relevant than ImageNet for motion-aware tasks.

#### 2.3 Late Fusion Strategy (ARCHITECTURAL COMMITMENT)

**RED TEAM FINDING:** Early fusion is geometrically incorrect. FPV and Top-view have incompatible projection models.

**Geometric Incompatibility:**
- **FPV (perspective projection):** Depth matters. Object at z=10cm projects differently than z=20cm.
- **Top-view (orthographic):** All objects at different z project to the same (x, y) pixel.
- **Early fusion** (fusing feature maps from both views at layer 2–3) conflates these coordinate systems.

**Solution: Late Fusion with Domain-Specific Encoding**

```
[FPV Video Frames] ──> TimeSformer backbone ──> FPV features (768,)
                                                           │
                                                   Late Fusion Block
                                                           │
[Top-view Video Frames] ──> TimeSformer backbone ──> Top-view features (768,)
```

**Fusion Layer (Approach 1: Learned Gating)**
```
FPV (768,)  ──┬─> Dense(768 → 256) ──┬─> Gating ──> α·FPV_proj + (1-α)·Top_proj
              │                       │  α = sigmoid(Dense(1024 → 1))
Top-view (768,) ──> Dense(768 → 256) ┘
```
Learned weighting respects that each view contributes differently.

**Fusion Layer (Approach 2: Cross-attention, PREFERRED)**
```
FPV (768,)  ──> Query ──┐
                        ├─> Multi-head cross-attention (8 heads) ──> Attended FPV (768,)
Top-view (768,) ──> Key/Value ┘                                       │
                                                                  ├─> Concat (1536,) ──> FC → (256,)
                                                                  │
FPV (768,)  ────────────────────────────────────────────────────┘
```
Allows FPV features to selectively attend to top-view features while preserving their native coordinate frames.

**Why Late Fusion:**
1. Respects projection geometry (each view encoded in native space first)
2. Reduces interference between incompatible coordinate systems
3. More interpretable (can visualize which view dominates per well)
4. Empirically more robust in stereo/multi-view settings

**COMMITMENT:** All future variants use Late Fusion. Early fusion is explicitly forbidden.

#### 2.4 Output Head Design

**Critical design choice:** How to structure the 96-way multi-label problem while maintaining temporal sensitivity.

**Option A: 96-class multi-label with per-frame logits**
```
Fused features (T=8, 256,) ──> FC(256 → 96) per frame ──> Sigmoid (per-frame, per-well probability)
                            ──> Temporal aggregation (mean/max over T) ──> Output logits[96]
Output: wells with prob > 0.5 are predicted active
Loss: Binary cross-entropy per well
```
Pros: Direct; captures temporal dynamics in intermediate features.
Cons: Treats each well independently; temporal aggregation (mean/max) still suboptimal.

**Option B: Sequence-to-Label (Event-focused, RECOMMENDED)**
```
Fused features (T=8, 256,) ──> Temporal attention decoder ──> Event logits (per frame)
                           ──> Dispense event index (frame_idx ∈ [0, T-1]) (argmax)
                           ──> Spatial head at frame_idx ──> FC(256 → 96) ──> well logits
```
Output: (well_id, confidence, dispense_frame_idx)
Loss: Cross-entropy on frame index + cross-entropy on well ID (2-task learning)

This explicitly models "when did the tip enter the well?" and "which well?"
Pros: Temporal event detection built-in; interpretable (can visualize which frame triggered prediction); aligns with VLA paradigm.
Cons: Requires frame-level annotations (can be derived from multi-frame labels via interpolation).

**Option C: Factorized (Row + Column heads) with temporal aggregation**
```
Fused features (T=8, 256,) ──┬─> Temporal pool (mean over T) ──> FC(256 → 8) ──> Softmax ──> row_logits
                             └─> FC(256 → 12) ──> Softmax ──> col_logits
Output: Outer product logits[8, 12]
```
Pros: Reduced parameters; spatial structure enforced; fast.
Cons: Loses temporal information in the pooling step (defeats the purpose of temporal modeling).

**Option D: Per-well heatmap + temporal peak detection**
```
Fused features (T=8, 256,) ──> Decoder ──> Heatmaps (T, 8, 12)  [T temporal frames, 8x12 well grid]
Extract peaks: For each of T frames, find local maxima in the 8x12 grid
Temporal localization: Find frame where each well's activation peaks
Output: (well_id, peak_confidence, peak_frame_idx)
Loss: L2 + temporal localization loss
```
Pros: Spatial + temporal coherence; multi-well naturally detected as multiple peaks.
Cons: Higher complexity; decoder network required.

**RECOMMENDATION:** **Option B (Sequence-to-Label)** is aligned with Transfyr's VLA mission and red team feedback. Temporal event detection is first-class, not afterthought. Fallback to **Option C** (factorized row/col) if compute budget exceeded.

#### 2.5 Training Strategy for Limited Data

**Challenge:** 100 samples is very small for deep learning; extreme class imbalance (some wells appear 0–2 times). Overfitting and memorization are severe risks.

**RED TEAM FINDING:** Relying on 100 physical samples without synthetic data generation is a failure of scale. A 11M-parameter ResNet will memorize background/lighting of specific videos rather than learn geometry.

**Strategy 1: Transfer Learning + Heavy Augmentation (FOUNDATION)**
- Start with **Kinetics-400 pre-trained TimeSformer** or **SlowFast** (not ImageNet)
- Freeze early temporal layers (first 2–3 blocks); fine-tune later layers + fusion + head
- Augmentation (per-frame):
  - Random spatial crop (224 → 224)
  - Temporal jitter (sample ±1 frame offset per clip)
  - Rotation (±10°)
  - Brightness/contrast shift (±20%)
  - Gaussian blur
  - Horizontal flip (if plate symmetric)
  - ColorJitter (lighting variation)
- Temporal augmentation:
  - Reverse frame order (check if temporal symmetry valid)
  - Frame skip (simulate variable dispensing speed)
  - Frame interpolation (synthetic frames between real frames)
- Regularization:
  - Dropout (p=0.5) in FC layers
  - L2 weight decay (λ=1e-4)
  - Early stopping on validation set (80/20 train/val split)
  - Mixup (on feature space, not input space)
- Batch size: 4–8 (small due to data size and temporal models)
- Optimizer: AdamW, lr=1e-4, cosine annealing decay

**Strategy 2: Synthetic Data Generation (ESSENTIAL per red team)**
- Generate synthetic video clips for wells with <3 training samples
- Use Stable Video Diffusion (fine-tuned on lab footage) to create:
  - Pipette approaching well from different angles
  - Tip entering, hovering, exiting
  - Lighting variations (shadow, glare)
- Target: Create 5–10× synthetic variants per well (500–1000 synthetic clips total)
- Mix synthetic + real in training (50/50 split or curriculum learning)

**Strategy 3: Self-supervised pre-training on unlabeled video (IF available)**
- Use any unlabeled lab video to pre-train backbone via contrastive learning
- Methods: MoCo v3 on video, masked autoencoding (VideoMAE)
- Fine-tune on labeled 100 samples
- Provides stronger feature initialization than Kinetics-400 (if domain very different)

**Strategy 4: Few-shot / Meta-learning (optional, advanced)**
- Prototypical networks or MAML: given K examples per well (K=1–2), learn to generalize to new wells
- Evaluate after Strategy 1 if class imbalance remains problematic

**RECOMMENDATION:** **Strategy 1 + Strategy 2 (Transfer + Synthetic Data)** is mandatory. Kinetics-400 pre-training is non-negotiable for temporal models. Synthetic data generation is essential to scale beyond 100 samples.

---

### Latency Analysis

**Batch inference budget:** 20 minutes total for ~10 samples

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

## Architecture 3: 3D Gaussian Splatting (3DGS) Scene Reconstruction

### Overview

A physically-grounded approach that reconstructs the 3D geometry of the lab scene from dual-view video, enabling depth-aware well localization and handling of specular reflections/refractions. This is the most robust long-term solution but computationally expensive.

### Motivation: Why 3D Geometry is Required

**RED TEAM FINDING:** Polystyrene plates and translucent pipette tips introduce specular reflection and refraction that 2D CNN solutions cannot handle.

**Physical Phenomena:**
1. **Glare shift:** Specular reflection shifts the "visual center" of a well in the image plane, but the well itself remains fixed in 3D space
2. **Refraction:** Liquid in the well refracts light; the apparent well position shifts with viewing angle
3. **Depth ambiguity:** Without depth, cannot distinguish between "tip at surface of well" vs. "tip at bottom of well"

**2D CNN Failure Mode:** Trains on pixels, not 3D coordinates. Will memorize that "glare blob at pixel (x, y) means well A5" instead of learning "the actual well center is at 3D position (x_world, y_world, z_world)".

**3D Scene Reconstruction Solution:** By fitting Gaussian primitives to the 3D scene, we:
- Explicitly model depth and lighting
- Reason about well positions in world coordinates, not pixel space
- Handle glare/refraction via explicit depth estimation

### Data Flow

```
[FPV Video] ──┐
              ├─> Camera Calibration (intrinsics + extrinsics) ──┐
[Top-view Video] ──┘                                            │
                                                                ├─> Structure-from-Motion / SLAM
                                                                │   (sparse 3D point cloud)
                                                                │
                                                                ├─> 3D Gaussian Splatting Optimizer
                                                                │   (fit Gaussians to scene)
                                                                │
                                                                ├─> Depth Estimation (render depth maps)
                                                                │
                                                                ├─> Well Localization
                                                                │   (3D scene search for well-like Gaussian clusters)
                                                                │
                                                                └─> JSON Output
```

### Component Details

#### 3.1 Camera Calibration

**Goal:** Estimate camera intrinsics (K_fpv, K_topview) and relative extrinsics (R, t) between FPV and top-view.

**Approach:**
- Mount a checkerboard calibration pattern in the lab
- Capture synchronized images of the checkerboard from both cameras
- Use OpenCV `cv2.calibrateCamera()` and `cv2.stereoCalibrate()` to compute:
  - Focal length, principal point (intrinsics)
  - Rotation matrix R and translation vector t (extrinsics between cameras)
- Store as calibration.yaml (one-time setup, ~1 hour)

**Implementation notes:**
- Checkerboard must cover multiple depths (near and far from cameras) for robust calibration
- Validation: Reprojection error <1 pixel for all calibration points

#### 3.2 Structure-from-Motion (SfM) & Sparse 3D Reconstruction

**Goal:** Estimate sparse 3D positions of key points (well centers, pipette tips, plate edges) from video.

**Approach:**
- Extract SIFT/ORB features from FPV and top-view frames
- Compute feature matches across frames (within same view) and across views
- Estimate essential matrix F; decompose into R, t
- Triangulate matched features using calibrated cameras to get 3D coordinates
- Filter outliers via reprojection error threshold

**Alternative (faster):** Use COLMAP or ORB-SLAM if available; these handle temporal consistency better.

**Output:** Sparse point cloud (~1000–10000 points) representing scene geometry.

#### 3.3 3D Gaussian Splatting (Core Innovation)

**What is 3DGS?**
3D Gaussian Splatting represents a scene as a set of 3D Gaussians (parameterized by position, covariance, opacity, color). The advantage: these Gaussians can be differentiably rendered to match input images, enabling optimization without voxels or implicit functions.

**Process:**

1. **Initialization:**
   - Start with sparse SfM point cloud
   - Fit a Gaussian to each point cloud cluster (e.g., 10 Gaussians per well plate region)
   - Initialize Gaussian parameters: center = point position, covariance = small isotropic, color = average image color at point

2. **Optimization:**
   ```
   Objective: Minimize ||I_rendered - I_observed||^2 + regularization
   Variables: Position, covariance, opacity, color per Gaussian
   Renderer: Splatting (project 3D Gaussians to 2D, blend by opacity)
   ```
   - Alternately optimize FPV view and top-view view
   - Use gradient descent (Adam) to refine Gaussian parameters
   - Regularization: sparsity (penalize non-zero opacity), smoothness (penalize covariance explosion)

3. **Refinement:**
   - Iteratively split large Gaussians and prune small ones
   - Run for 5000–10000 iterations (typically hours for large scenes, seconds for plate-sized scenes)

**Rendering:**
For any novel viewpoint (including the original FPV/top-view), render a 2D image by:
- Project 3D Gaussians to 2D
- Sort by depth
- Alpha-blend colors and opacities (front to back)

#### 3.4 Depth-Aware Well Localization

**Goal:** Find well positions by analyzing the 3D scene (not image pixels).

**Approach 1: Cluster Gaussians (Geometry-based)**
- Look for clusters of high-opacity Gaussians at roughly the same z-depth (surface of plate)
- Cluster in 2D (x, y) space after filtering by z ≈ z_plate
- Cluster centers approximate well positions
- Count clusters; validate that #clusters ≈ 96

**Approach 2: Render Depth Maps (Physics-based)**
- Use the optimized Gaussians to render a depth map from a virtual top-view camera
- Apply blob detection to the depth map to find well centers (wells appear as circular depressions)
- Refine with sub-pixel accuracy

**Approach 3: Learned Decoder on Rendered Features**
- Render features (not RGB) from the Gaussian splat: e.g., per-Gaussian class scores (well vs. plate vs. pipette)
- Attach a small CNN decoder to detect well positions from rendered feature maps
- Hybrid: combines geometry (3D scene) with learned refinement

**Recommended:** Approach 2 (depth map rendering) for interpretability; fallback to Approach 3 if accuracy insufficient.

#### 3.5 Pipette Tip Localization

**Goal:** Localize the pipette tip in 3D space.

**Approach:**
- Fit a separate set of Gaussians to represent the pipette (high color variance, thin/needle-like structure)
- Search for Gaussian clusters that match pipette properties (color, opacity pattern, elongation)
- Tip center = highest point in the pipette Gaussian cluster
- Confidence: weighted by opacity and reprojection error

#### 3.6 Output Generation

**Assembly:** Match dispense event (from temporal model or manual frame annotation) to tip and well positions.

```json
{
  "wells": [
    {"row": "A", "column": 1, "confidence": 0.98, "world_position": [10.5, 20.3, 5.0]},
    {"row": "A", "column": 2, "confidence": 0.95, "world_position": [14.5, 20.3, 5.0]}
  ],
  "pipette_tip_position": [12.5, 20.3, 0.5],
  "depth_confidence": 0.92,
  "scene_reconstruction_rmse": 0.8
}
```

---

### Strengths

1. **Physically correct:** Reasons in 3D space, not pixel space. Handles specular reflections and refraction naturally.
2. **Depth-aware:** Explicit depth estimation; can distinguish well surface vs. bottom.
3. **Robust to glare:** Glare is a 2D artifact; 3D scene geometry is unaffected.
4. **Interpretable:** Gaussian positions and opacities are human-readable; can visualize scene.
5. **Novel view synthesis:** Can render scene from arbitrary camera angles (useful for validation).
6. **Generalization:** Once scene is reconstructed, can apply classical well detection (grid overlay) in 3D.

---

### Weaknesses & Computational Cost

1. **Slow inference:** SfM + 3DGS optimization can take **10–60 minutes per sample** (depending on scene complexity and GPU), exceeds 2-min latency budget significantly.
2. **Calibration required:** Accurate camera calibration is essential; miscalibration propagates to 3D positions.
3. **Sparse features:** If plate is featureless (solid color), feature matching may fail; SfM becomes impossible.
4. **Gaussian explosion:** Naive optimization can produce thousands of Gaussians; pruning heuristics may be needed.
5. **Temporal inconsistency:** Independent optimization per frame pair; no temporal coherence across video (can be addressed with temporal regularization).

---

### When to Use Architecture 4

- **Glare failures dominate error analysis:** If 2D models fail specifically on high-glare wells, 3DGS solves the problem.
- **High-precision requirement:** If accuracy must exceed 98%, 3D reasoning helps.
- **Limited to <1 sample per day:** Latency is prohibitive for high-throughput labs, but acceptable for careful validation.
- **Research/publication:** Demonstrates Physical AI rigor (excellent for CV conferences).

### When NOT to Use Architecture 4

- **Real-time requirements:** 2-min latency budget is non-negotiable.
- **Featureless plates:** Sparse features break SfM.
- **Production deployment:** Overhead not justified if Architectures 2/3 meet accuracy targets.

---

## Architecture 4b: Lightweight 3D (Optional Compromise)

If 3DGS is too slow but depth modeling is desired:

**Lightweight approach:** Use a monocular depth estimator (MiDaS v3.1) to predict depth from each view, then:
- Back-project FPV depth to 3D using camera intrinsics
- Use top-view as orthographic depth (all points at plate surface)
- Triangulate FPV tip position using predicted depth + top-view well coordinates

**Latency:** ~500ms per sample (depth estimation is fast)
**Accuracy:** Lower than full 3DGS, but avoids SfM calibration overhead
**Status:** Fallback for time-constrained projects

---

## Architecture 5: Hybrid Approach (Optional)

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

**Constraint:** 20 minutes total for ~10 samples (batch inference).

**Actual budget:** 20 minutes total (2 min per sample average, conservative).

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

| Criterion | Classical CV | DL w/ Temporal | 3DGS | Hybrid (DL+CV) |
|-----------|--------------|-----------------|------|---|
| **Accuracy (est.)** | 80–90% | 88–96% | 95–99% | 92–97% |
| **Interpretability** | High | Medium | Very High | Medium–High |
| **Training data needed** | None (for geometry) | 100 (+ synthetic) | 100 + calibration | 100 |
| **Inference latency** | 1.6–3.1 sec | 0.8–2 sec | 10–60 min | 1.5–3 sec |
| **Physical realism** | Low (2D grid) | Medium (temporal) | **Very High** (3D scene) | Medium |
| **Handles glare/refraction?** | No | Partially (learned) | **Yes** (explicit depth) | Partially |
| **Extensibility (new format)** | High | Low | High (scene-agnostic) | Medium |
| **Robustness (lighting)** | Medium | High | High | High |
| **Robustness (occlusion)** | Medium | Medium | High | High |
| **Robustness (plate tilt)** | Low | Medium | High | Medium–High |
| **Development time** | 2–3 weeks | 2–3 weeks | 4–6 weeks | 3–4 weeks |
| **Maintenance burden** | Low | Medium | High | Medium |
| **GPU required?** | No | Yes | Yes (strong) | Yes (optional) |
| **Failure mode** | Cascading (plate → tip) | Overconfident misclassification | Complex (SfM failure) | Degraded (one module) |
| **2-min budget?** | Yes | **Yes** | No | Yes |
| **Recommended for production?** | If accuracy <80% | **Yes (primary)** | No (too slow) | If accuracy <88% |

---

## Architect's Recommendation

### Primary Recommendation: **Architecture 2 (Deep Learning with Temporal Modeling)**

**Rationale:**

1. **Temporal signals are first-class:** Single-frame models cannot distinguish dispense phases (entering vs. leaving). Temporal models (TimeSformer, SlowFast) identify the dispense event explicitly, aligning with Transfyr's "Tacit Knowledge" mission.

2. **Best accuracy-to-effort ratio within 2-min budget:** With 100 labeled samples + synthetic data generation, temporal DL should achieve **88–96% accuracy** on held-out set.

3. **Inference speed:** ~0.8–2 sec per sample on GPU is **well within 2-min budget**, enabling iteration.

4. **Robustness to variation:** Augmentation (brightness, rotation, temporal jitter) makes model resilient to lighting, plate tilt, variable dispensing speed without manual tuning.

5. **Implementable with Kinetics-400 pre-training:** TimeSformer/SlowFast are trained on large-scale video; transfer learning is strong.

6. **Synthetic data generation mandatory:** Red team critique on N=100 overfitting requires generating synthetic dispense events (5–10× augmentation). Strategy 2 in Section 2.5 is non-negotiable.

### Secondary Recommendation: **Hybrid (if DL accuracy stalls <88%)**

If temporal deep learning accuracy plateaus <88% on validation:

1. Implement classical CV module (plate detection + tip localization) in parallel
2. Train a lightweight fusion network that combines geometric prior + temporal features
3. Hybrid should recover ~5–10% accuracy via geometric constraints while preserving temporal awareness

### Tertiary Recommendation: **Architecture 3 (3DGS) for High-Precision / Glare-dominated Failure**

If error analysis reveals glare/refraction as primary failure mode (>50% of misclassifications):

1. Implement 3D Gaussian Splatting pipeline (4–6 weeks)
2. Trade 2-min latency for near-perfect accuracy (95–99%)
3. Use for offline validation or high-stakes assays
4. **Not** for production throughput; latency is prohibitive (10–60 min per sample)

### Why NOT Classical CV as primary:

- **Temporal blindness:** Cannot distinguish dispense phases; violates Transfyr's core mission.
- **Brittleness to calibration:** Camera calibration errors compound through geometric pipeline. Single point of failure: if plate detection fails, entire pipeline fails.
- **Lighting sensitivity:** Color thresholding requires manual tuning per setup.
- **Development risk:** More complex pipeline (6–7 steps); higher chance of subtle bugs in geometric mappings.
- **Data waste:** Ignores 100 labeled examples except for validation (no temporal signals extracted).

### Why NOT 3DGS as primary:

- **Latency prohibitive:** 10–60 min per sample far exceeds 2-min budget.
- **Calibration overhead:** Requires careful camera calibration (1 hour one-time setup).
- **SfM fragility:** Feature matching can fail on featureless plates.
- **Overkill for current task:** If Architectures 2/3 meet 88%+ accuracy, 3D reasoning unnecessary.
- **Status:** Excellent for research validation; production-unviable.

---

## Non-Negotiable Design Constraints (Revised)

### For Architecture 2 (Recommended)

1. **Temporal-aware backbone:** Must use TimeSformer or SlowFast, NOT single-frame models. Temporal modeling is mandatory for this task.

2. **Late Fusion (geometrically correct):** FPV and top-view must be encoded separately (respecting their projection models), then fused via cross-attention. Early fusion explicitly forbidden.

3. **Synthetic data generation:** Must create 5–10× synthetic dispense events to mitigate N=100 overfitting. Strategy 2 (Section 2.5) is non-negotiable.

4. **Kinetics-400 pre-training:** Transfer learning from video (Kinetics) is non-negotiable, not ImageNet.

5. **Dispense event detection:** Output head must explicitly model "when did tip enter well?" (Option B in Section 2.4, Sequence-to-Label). Temporal aggregation (mean/max) is insufficient.

6. **Latency ceiling:** Inference time per sample **must be <10 seconds** (ideally <2 sec) to fit within 2-min budget for 10 samples.

7. **Validation accuracy:** Minimum **88%** on held-out set before production deployment. If <88%, hybrid or 3DGS required.

8. **Generalization:** Model must achieve ≥85% accuracy on well coordinates it did **not** see in training data. Cannot overfit to training wells.

### For Classical CV (if pursued as fallback)

1. **Plate detection robustness:** Must detect plate correctly in ≥95% of frames even with partial occlusion.

2. **Temporal motion tracking:** Optical flow + velocity estimation (Section 1.4) is mandatory for identifying dispense events.

3. **Calibration time:** Initial camera calibration must complete in <1 hour.

4. **Geometric accuracy:** Tip position error <2 pixels in rectified frame (corresponds to ~0.5 mm on plate; acceptable for 4.5 mm well pitch).

### For 3DGS (if used for validation/research)

1. **Scene reconstruction RMSE:** 3D point positions must be accurate to <1 mm (critical for sub-well-pitch accuracy).

2. **Calibration validation:** Reprojection error <1 pixel on all calibration images.

3. **Inference time tolerance:** <60 min per sample acceptable for offline research (not production).

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

## Implementation Roadmap (Revised)

### Phase 1: Data Preparation & Synthetic Generation (Week 1)

- [ ] Implement data loading & preprocessing pipeline
- [ ] Analyze class balance: which wells appear <3 times?
- [ ] **Implement synthetic data generation (Stable Video Diffusion or template-based):** Create 5–10× synthetic clips for undersampled wells. Target: 500–1000 synthetic clips total.
- [ ] Split: 80/20 train/val on (100 real + 500–1000 synthetic) = 600–1100 samples
- [ ] **Decision gate:** Synthetic data quality acceptable (visual inspection + FID score)?

### Phase 2: Baseline Temporal Model (Week 2)

- [ ] Implement Architecture 2 (TimeSformer backbone + temporal fusion + Sequence-to-Label head)
- [ ] Kinetics-400 pre-trained initialization (no ImageNet)
- [ ] Implement training loop: transfer learning + aggressive augmentation (Strategy 1, Section 2.5)
- [ ] Evaluate on validation set (20 real + 100 synthetic samples)
- [ ] **Decision gate:** Accuracy ≥88% on validation?
  - **YES:** Proceed to Phase 3
  - **NO:** Proceed to Phase 2b (hybrid)

### Phase 2b (Optional): Hybrid if DL <88%

- [ ] Implement classical CV module (plate detection + optical flow) in parallel
- [ ] Implement geometric prior fusion (Architecture 5)
- [ ] Re-evaluate: hybrid accuracy ≥88%?

### Phase 3: Optimization & Error Analysis (Week 3)

- [ ] Hyperparameter sweep: learning rate, weight decay, augmentation intensity, fusion strategy
- [ ] Error analysis: which wells are misclassified? Any systematic bias (rows/columns)?
- [ ] Temporal event detection validation: does model learn to identify dispense frame?
- [ ] Implement confidence calibration (temperature scaling)
- [ ] Target: ≥92% on validation

### Phase 4: Integration & Testing (Week 4)

- [ ] Integrate with CLI interface
- [ ] Test on 10 held-out real samples (no synthetic data)
- [ ] Latency profiling; verify <2 sec per sample (GPU)
- [ ] Package as Docker container
- [ ] Final documentation + deployment guide

### Phase 5: Fallback / Research (If needed, Week 5+)

- [ ] If Phase 4 test accuracy <88%, implement Hybrid (Architecture 5)
- [ ] If error dominated by glare/refraction, explore 3DGS (Architecture 3) for research validation
- [ ] Publish findings on Physical AI for lab automation

---

## Technical Debt & Future Work

1. **Uncertainty quantification:** Use Monte-Carlo dropout or ensemble methods to estimate per-well confidence; "abstain" when uncertain (red team critique on reproducibility over accuracy).

2. **Active learning:** With 100 samples, use uncertainty sampling to prioritize which new samples to label; iteratively improve dataset coverage.

3. **Temporal event grounding:** Extend output head to predict dispense timestamp (frame index) with confidence; useful for video verification and debugging.

4. **Robustness to plate variation:** Test generalization to different plate formats (384-well, 1536-well) without retraining. Use architectural changes (grid-agnostic decoder) if needed.

5. **Calibration-free 3D:** Explore monocular depth estimation (MiDaS) as lightweight alternative to full 3DGS when glare failures emerge.

6. **Online learning / domain adaptation:** Deploy in production; retrain on misclassified samples to adapt to new pipette types, lighting, or plate materials.

7. **Multimodal fusion:** Integrate acoustic signals (Transfyr captures audio) to cross-validate dispense events. Audio "plink" is higher-fidelity than blurred video.

8. **Edge deployment:** Quantize temporal backbone for NVIDIA Jetson; enable real-time lab deployment instead of cloud inference.

---

## Appendix: Early Fusion is Geometrically Wrong (Rationale for Late Fusion)

**Why early fusion (concatenating FPV and top-view at layer 2) is incorrect:**

```
FPV image (perspective projection):
  - Object at depth z=10cm projects to pixel p1
  - Same object at z=20cm projects to pixel p2 (further from camera)
  - Depth directly affects pixel position

Top-view image (orthographic projection):
  - All objects at any depth z project to same pixel p_ortho ≈ (x_world, y_world)
  - Pixel position ≈ world position (depth-independent)

Early Fusion (pixel-level concat):
  [FPV_features(p1)] + [TopView_features(p_ortho)]
  
  The network tries to align (p1 ≠ p_ortho) from different projection models.
  This forces the network to learn a non-linear, ad-hoc mapping between
  perspective and orthographic coordinates—a brittle, low-generalization solution.

Late Fusion (after encoding):
  FPV_features = Encoder_FPV([FPV_image]) → (f1, f2, ..., f_d)  [in FPV coordinate frame]
  TopView_features = Encoder_TopView([TopView_image]) → (g1, g2, ..., g_d)  [in world frame]
  
  Fused_features = Attention(query=FPV_features, key_value=TopView_features)
  
  Each view is encoded in its native coordinate frame; fusion respects geometry.
```

**Empirical consequence:**
- Early fusion models memorize specific FPV-TopView pixel alignments from training data
- Generalization to new camera angles or plate positions fails
- Late fusion models learn coordinate-frame-agnostic concepts; generalization improves

**Reference:** Multi-view geometry literature (Hartley & Zisserman, 2003) establishes that feature-space fusion respects projection model differences better than pixel-space fusion.

---

## Conclusion

**Recommended path:** Architecture 2 (Temporal Deep Learning) with Kinetics-400 pre-training, late fusion, synthetic data generation, and explicit dispense event detection. This balances accuracy (88–96%), temporal semantics (first-class), generalization (unseen wells), and latency (batch: <20 min for ~10 samples).

**Physical AI commitment:** All architectures explicitly model 3D geometry (projection models, depth, temporal causality) rather than treating the task as 2D image classification.

**Research path (high-precision):** Architecture 3 (3DGS) for offline validation. Demonstrates scientific rigor but unsuitable for production throughput.

**Fallback path:** Hybrid (Architecture 5) if deep learning <88%. Geometric prior provides safety net while preserving temporal awareness.

**Non-negotiable constraints:**
- Support multi-label output, dual-view fusion, <2-min latency, ≥88% validation accuracy
- Temporal modeling (no single-frame baselines)
- Late fusion (geometrically correct)
- Synthetic data generation (scale beyond N=100)
- Uncertainty quantification (reproducibility over overconfidence)

Success criteria:
- [ ] ≥88% accuracy on held-out test set (10 real samples)
- [ ] <2 sec inference per sample on GPU (ideally <1 sec)
- [ ] ≥85% accuracy on unseen row/column combinations (generalization test)
- [ ] Explicit dispense event detection (temporal localization)
- [ ] Reproducible results (git-tracked, seeded, calibrated confidence scores)
- [ ] Failure mode analysis: glare/occlusion robustness characterized

