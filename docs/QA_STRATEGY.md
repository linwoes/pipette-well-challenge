# QA Strategy: Transfyr AI Pipette Well Challenge

**Document Version:** 1.0  
**Date:** 2026-04-14  
**Author:** QA Engineer  
**Audience:** Data Scientist, Architect, ML Scientist, Interview Panel

---

## Executive Summary

This QA Strategy defines a rigorous testing and validation approach for the Transfyr AI Pipette Well Challenge solution. Given the small dataset (100 training samples), severe class imbalance, multi-label complexity, and tight evaluation window (20 minutes for 10 hold-out samples), this document establishes:

1. A layered QA approach spanning unit → integration → system → acceptance testing
2. Probabilistic analysis of hold-out risk given class imbalance
3. Comprehensive edge case catalogue with severity ratings
4. Failure mode analysis for both Deep Learning and Classical CV approaches
5. Robustness testing protocols
6. Risk mitigation recommendations for all roles
7. Clear acceptance criteria for the hold-out evaluation

---

## 1. QA Strategy Overview

### 1.1 Testing Philosophy

This project requires a **defense-in-depth** testing strategy due to:
- **Small dataset** (100 samples) → high generalization risk
- **Class imbalance** (5-15 wells with zero training samples) → prediction blindness for held-out wells
- **Multi-label complexity** (1, 8, or 12 wells per sample) → output cardinality misclassification risk
- **High-stakes evaluation** (10 samples in 20 minutes) → no second chances

**Core Principle:** Fail fast, fail safe. Every component must validate inputs, signal failures clearly, and gracefully degrade rather than silently produce wrong predictions.

### 1.2 Layered QA Approach

#### Layer 1: Unit Testing
**Scope:** Individual functions (video loading, frame extraction, well position classification, JSON serialization)  
**Owner:** ML Scientist + Architect  
**Checklist:**
- [ ] Video file I/O handles missing files, corrupted frames, missing audio
- [ ] Frame extraction produces correct timestamp and pixel data
- [ ] Well grid alignment algorithms return valid coordinates (A-H, 1-12 only)
- [ ] JSON serialization matches expected schema exactly
- [ ] Confidence thresholding logic is implemented correctly
- [ ] Multi-label cardinality post-processing works for 1, 8, 12-channel scenarios

#### Layer 2: Integration Testing
**Scope:** FPV + Top-view fusion, dual-view feature concatenation, end-to-end pipeline  
**Owner:** ML Scientist + Architect  
**Checklist:**
- [ ] FPV and Top-view clips are synchronized (frame counts, timestamps match)
- [ ] Feature fusion produces correct tensor dimensions
- [ ] Model inference produces output for both single-well and multi-well cases
- [ ] Pipeline executes in <2 min per sample on evaluation hardware
- [ ] Error in one view doesn't crash the pipeline (graceful fallback to single view)

#### Layer 3: System Testing
**Scope:** CLI invocation, file I/O, JSON output validation, timeout enforcement  
**Owner:** QA Engineer  
**Checklist:**
- [ ] CLI accepts exactly 2 video file paths
- [ ] CLI outputs JSON to stdout with no extraneous debug output
- [ ] JSON schema validation: must include `wells` array with `row` and `column` keys
- [ ] Total execution time for 10 samples ≤ 20 minutes
- [ ] Timeout handling: if single sample exceeds 3 minutes, terminate gracefully
- [ ] Memory usage stays <4GB throughout pipeline

#### Layer 4: Acceptance Testing
**Scope:** Hold-out evaluation on 10 unknown samples  
**Owner:** Interview Panel (observed by QA Engineer)  
**Checklist:**
- [ ] All 10 samples produce valid JSON output
- [ ] JSON output matches expected schema for all samples
- [ ] Accuracy meets minimum threshold (see Section 7)
- [ ] Latency satisfies 20-minute total constraint
- [ ] No runtime errors or exceptions
- [ ] Predictions are consistent with visual inspection of videos

---

## 2. Hold-Out Set Analysis

### 2.1 Class Imbalance Risk: Hold-Out Containing Unseen Wells

**Given:**
- 96 total wells on plate
- 100 training samples
- Each sample may cover 1–12 wells
- Assume uniform distribution: ~100–1200 well instances in training

**Probability Calculation:**

Conservative estimate: Assume each training sample covers **2 wells on average** (accounting for some multi-channel operations).
- Training well instances: ~200
- Unique wells covered: Statistical model suggests 40–60 wells have at least one training instance
- **Unseen wells in training:** 36–56 wells (37.5–58% of plate)

**Hold-out set risk (10 samples):**
- If each hold-out sample covers 2 wells: ~20 well predictions
- **Expected unseen wells in hold-out:** 8–12 wells with zero training examples

**Critical Implication:** A trained model will have **zero learned parameters** for these wells. Any prediction for an unseen well is a generalization guess.

### 2.2 Multi-Label Cardinality Distribution Risk

**Data Scientist Finding:** Training data may contain mostly single-channel operations (1-well) with fewer 8-channel or 12-channel examples.

**Hold-out risk:** 
- If training has 70% single-channel, 20% 8-channel, 10% 12-channel
- Hold-out with random 10 samples may contain 0–2 multi-channel operations
- Model confidence in multi-channel predictions may be low/unreliable

### 2.3 Fairness and Representativeness Strategy

**Strategy 1: Analysis Pre-Evaluation**
1. **Data Scientist** generates coverage report:
   - Heatmap of wells seen in training data (mark coverage gaps)
   - Distribution of cardinalities (1, 8, 12-well operations)
   - Identify "risky" wells (low training frequency)
2. **QA Engineer** requests hold-out set metadata **before** visual inspection:
   - Which wells are present in the 10 hold-out samples?
   - How many single vs. multi-channel operations?
   - Are there held-out wells with zero training examples?

**Strategy 2: Adaptive Baseline**
- Establish a **coverage-weighted acceptance criterion:**
  - "Seen wells" (≥2 training examples): Target ≥85% accuracy
  - "Rare wells" (1 training example): Target ≥70% accuracy
  - "Unseen wells": Treat as generalization risk; flag but don't penalize heavily if accuracy is 50%+
- This acknowledges the inherent difficulty of predicting unseen classes

**Strategy 3: Graceful Degradation for Unseen Wells**
- If model confidence on an unseen well is <0.3, output explicit "LOW_CONFIDENCE_PREDICTION" flag
- Never silently output high-confidence predictions for zero-shot wells
- Allow human review of these flagged predictions

### 2.4 What the Model Should Do: Unseen Well Protocol

**Recommended Behavior:**
1. **Attempt prediction** using learned feature representations (transfer learning from seen wells)
2. **Check confidence:** If max confidence <0.4 for an unseen well, **do NOT output it**
3. **Output valid wells only:** Return only wells with confidence ≥0.4
4. **Log rationale:** Write to stderr: `"WARNING: Well A5 was unseen in training. Skipped due to confidence 0.35."`
5. **Accept partial credit:** Predicting 10 correct wells is better than 12 predictions with 2 spurious guesses

---

## 3. Edge Cases Catalogue

### 3.1 Visual Edge Cases

| Edge Case | Description | Severity | Detection | Mitigation |
|-----------|-------------|----------|-----------|-----------|
| Extreme glare | Light reflections wash out plate grid | **Critical** | Histogram clipping detection; if >40% pixels saturated, flag | Use grayscale/edge detection; apply local contrast enhancement |
| Dark/low light | Insufficient illumination makes wells indistinguishable | **Critical** | Mean pixel intensity <50 (0–255 scale); log warning | Gamma correction; switch to edge-based detection if grayscale fails |
| Motion blur | Fast pipette movement blurs tip and target well | **High** | Compute Laplacian variance of frames; if <threshold, flag | Extract sharp middle frames only; skip first/last 5 frames |
| Occluded plate | Hand, tube, or debris covers well region | **High** | Foreground segmentation; if >20% of plate masked, reject frame | Attempt reconstruction from unoccluded frames; if fail, output "OCCLUSION_DETECTED" |
| Hand in frame | Operator hand obscures wells or pipette tip | **High** | Skin color detection; if present over well region, flag | Use frames before hand enters; document limitation |
| Liquid splash | Droplets or aerosol obscure view | **Medium** | Morphological analysis of sudden intensity changes | Switch to temporal consistency: use frames from before splash; flag uncertainty |

### 3.2 Geometric Edge Cases

| Edge Case | Description | Severity | Detection | Mitigation |
|-----------|-------------|----------|-----------|-----------|
| Plate rotated | Well grid is rotated 5–45° from camera | **Critical** | Hough line detection; if grid angle > ±10°, flag | Implement rotation-invariant grid detection; apply affine correction |
| Plate partially OOB | Wells at edge of frame are cut off | **High** | Check if well boundary markers visible; if <85% of grid in frame, flag | Pad inference region; mark edge wells with lower confidence; accept "partial results" |
| Tilted plate | Plate is tilted (3D perspective) | **High** | Vanishing point analysis; if perspective distortion > threshold, flag | Apply perspective correction (homography estimation) |
| Two plates on table | Multiple plates visible simultaneously | **Critical** | Connected component analysis; if 2+ separated plate regions, flag | Detect bounding boxes of each plate; process largest only; warn of ambiguity |
| Plate upside down | Well labels (A1, H12) reversed | **Medium** | OCR plate labels if visible; check orientation consistency | Infer orientation from pipette trajectory; auto-flip if inverted |

### 3.3 Pipette Edge Cases

| Edge Case | Description | Severity | Detection | Mitigation |
|-----------|-------------|----------|-----------|-----------|
| Single-channel 1-channel pipette | Solitary tip; standard case | **N/A** | Baseline | Standard processing |
| 8-channel pipette | Eight parallel tips in row | **High** | Count tip-like objects; if 8 aligned, flag 8-channel mode | Implement multi-tip detection; predict 8 wells simultaneously; validate cardinality post-processing |
| 12-channel pipette | Twelve parallel tips in row | **High** | Count tip-like objects; if 12 aligned, flag 12-channel mode | Implement 12-tip detection; predict 12 wells; validate cardinality |
| Pipette barely visible | Tip at edge of frame or very small | **High** | Tip bounding box < 20 pixels; tip center near frame edge | Expand search region; use FPV clip as primary (more reliable); flag uncertainty |
| Unusual pipette color | Non-standard tip color (e.g., red, clear) | **Medium** | Color-based tip detection may fail; validate with shape heuristics | Use grayscale + edge detection instead of color thresholding |
| Damaged/misshapen tip | Tip is bent or chipped | **Medium** | Shape deformation detection via contour analysis | Fallback to centroid-based positioning; lower confidence but still usable |

### 3.4 Temporal Edge Cases

| Edge Case | Description | Severity | Detection | Mitigation |
|-----------|-------------|----------|-----------|-----------|
| Very short clip | Video < 1 second duration | **High** | Check FPS × duration; if <30 frames total, flag | Require minimum 1 sec = 30 frames; reject or interpolate |
| Clip ends before dispense completes | Video cuts off mid-dispense | **High** | Analyze liquid meniscus motion; if stops mid-trajectory, flag | Use FPV temporal cues; infer final position from trajectory; mark "INCOMPLETE_DISPENSE" |
| Multiple dispenses in one clip | Pipette dispenses into 2+ wells sequentially | **Critical** | Detect multiple tip descents; if >1 detected, flag | Segment clip by dispense event; process each separately; merge outputs |
| No dispense (hovering only) | Pipette hovers over well but doesn't dispense | **Medium** | Check for liquid motion or Z-axis descent; if zero, flag | Validate: tip must touch/enter well; if no contact, output empty wells array |
| Slow/gradual dispense | Liquid released over 5+ seconds | **Low** | Analyze meniscus change rate; if slow, mark as low-urgency | Standard processing; may have lower confidence due to motion artifacts |

### 3.5 Input File Edge Cases

| Edge Case | Description | Severity | Detection | Mitigation |
|-----------|-------------|----------|-----------|-----------|
| Mismatched FPV/Top-view | Two clips from different samples/times | **Critical** | Timestamp and content mismatch; FPV/top-view show different wells | Require explicit validation: cross-check well predictions match between views; if divergence > 20%, reject |
| Corrupted video file | File is truncated, unreadable, or damaged | **Critical** | FFmpeg read failure; frame extraction throws exception | Catch exception early; output error JSON: `{"error": "CORRUPTED_VIDEO", "file": "..."}` |
| Extremely low resolution | Video < 240p | **High** | Check frame dimensions; if <240×240, flag | Upscale with cubic interpolation; note reduced confidence; continue |
| Unusual codec | Format not H.264/H.265 (e.g., old MJPEG) | **Medium** | Ffmpeg codec detection | Attempt transcode; if fail, reject with clear error message |
| Audio-only file | No video stream present | **Critical** | ffprobe detects 0 video streams | Reject immediately: `{"error": "NO_VIDEO_STREAM"}` |

### 3.6 Output Edge Cases

| Edge Case | Description | Severity | Detection | Mitigation |
|-----------|-------------|----------|-----------|-----------|
| Zero wells predicted | Model outputs empty wells array | **Medium** | Check len(wells) == 0 | Validate acceptable: only if all model confidences < threshold; log reason; report to stdout |
| >12 wells predicted | Model outputs >12 wells (impossible) | **Critical** | Check len(wells) > 12 | Reject: clip top-12 by confidence or implement cardinality constraint; flag as MODEL_ERROR |
| Well out of bounds | Predicted well outside A-H, 1-12 range | **Critical** | Validate row in [A-H], col in [1-12] | Clamp coordinates or reject; log invalid prediction; never output out-of-bounds |
| Conflicting row/column | E.g., row="I", column="13" | **Critical** | Schema validation before JSON output | Implement strict output validation; raise exception if invalid; output error JSON |
| Non-integer column | E.g., column=5.7 | **High** | Type check on JSON output | Round to nearest integer; validate before serialization |
| Duplicate wells | Same well listed twice | **Medium** | Check set(wells) == len(wells) | Deduplicate; sort in canonical order (A1, A2, ..., H12) |
| Non-uppercase rows | E.g., row="a" instead of "A" | **Medium** | Validate row.isupper() and in [A-H] | Uppercase all rows before output |

### 3.7 Transparency & Specular Reflection Edge Cases (NEW)

Red team identified polystyrene plates + translucent tips as high-risk. These edge cases require special handling.

| Edge Case | Description | Severity | Detection Method | Expected Failure Mode | Mitigation |
|-----------|-------------|----------|------------------|----------------------|-----------|
| **Glare on well plate** | Full-frame specular highlight (white saturation) obscures multiple wells, creates false well-like artifacts | **Critical** | Histogram analysis: if >30% pixels saturated (>240/255), flag glare. Compute local contrast: if std_dev > 100 in bright regions, flag specular | Model confuses bright artifacts for wells; predicts false wells outside plate area | Use edge detection + morphology to isolate plate boundary; mask out oversaturated regions; apply local histogram equalization |
| **Translucent tip invisible** | Pipette tip blends with background (clear/translucent plastic in bright light); tip position indistinguishable from plate background | **High** | Tip detection fails: color-based thresholding returns empty mask; shape detection finds no elongated objects in expected region. Check frame by frame: if tip detectable in <50% of frames, flag | Model cannot localize tip; predicts well at wrong spatial location; off-by-1 or off-by-multiple wells | Switch to FPV view (more tip contrast); use edge/contour detection; fallback to temporal consistency (track tip motion across frames with Kalman filter) |
| **Liquid meniscus distortion** | Liquid surface at well acts as lens; refracts light and distorts apparent well position; appears offset from true well center | **High** | Analyze well boundary sharpness: meniscus creates soft gradient at well edge instead of sharp boundary. If edge sharpness < threshold, flag refraction | Model detects well center at distorted (refracted) position; prediction offset by 1–2 pixel clusters from true center | Account for refraction in post-processing: well center refinement using template matching on un-distorted well regions; use multiple frames to triangulate true position |
| **Multi-channel tip array (8 tips)** | 8 translucent tips in parallel row, partially occluded by each other or plate shadows; tips merge visually into continuous line | **Critical** | Multi-tip detection: count distinct tip objects in frame. If count ≠ expected (1, 8, or 12), flag ambiguity. Use HOG or edge profiles to separate individual tips | Model predicts fewer wells than expected (8 tips → 4 predictions if tips merge); cardinality wrong; wells detected at averaged positions rather than individual tip positions | Explicit multi-tip segmentation: apply morphological operations (erosion/dilation) to separate fused tips; use blob detection with expected size constraints; validate 8 well detections align in row; if separation fails, flag "MULTI_TIP_DETECTION_FAILED" |

### 3.8 Temporal Edge Cases (NEW)

Red team identified temporal blindness as a gap. These edge cases address temporal misunderstanding.

| Edge Case | Description | Severity | Detection Method | Expected Behavior | Test Case Design |
|-----------|-------------|----------|------------------|-------------------|------------------|
| **Hovering without dispensing** | Pipette positioned directly over well, tip approaches but does NOT enter well or dispense liquid; no dispense event occurs | **High** | Analyze pipette trajectory: if tip_z_position stays constant (hovering) with no liquid motion, flag. Check for meniscus changes: if liquid surface unchanged, no dispense occurred | Model must NOT output a well prediction; refusal (empty wells array) is correct response. If confidence high, it's a false positive | Construct synthetic video: frame 1–30: pipette approaches well A1 to <1mm distance; frames 31–60: pipette hovers stationary over A1; frames 61–90: pipette withdraws. Ground truth: empty wells array (no dispense) |
| **Multiple dispenses in one clip** | Single 5–10 second video contains TWO separate dispense events into different wells (e.g., dispense to A1, withdraw, move, dispense to B2) | **Critical** | Temporal analysis: detect multiple tip descents and ascents. If descents > 1, flag. Analyze liquid motion: if >1 temporal peak in meniscus change, multiple dispenses | Model must segment events and predict the PRIMARY (first or most visible) dispense only. If cardinality expects multiple outputs, post-processing must handle: output PRIMARY well OR output multiple wells with timestamps (if specification allows) | Frame 1–30: dispense into A1; frames 31–60: withdraw and move; frames 61–90: dispense into B2. Ground truth: either {"wells": [A1]} (primary) OR {"wells": [A1, B2]} with event metadata. **Document which is required** |
| **Partial dispense (early withdrawal)** | Pipette begins dispensing (tip in well, liquid motion starts) but is withdrawn before completion; dispense incomplete | **High** | Analyze liquid motion timeline: if meniscus change (volume decrease) stops mid-trajectory, flag. Incomplete dispense: final liquid amount < expected dispense volume | Model should output the well where dispense began, even if incomplete. Partial credit: marking as "partial dispense" in metadata is acceptable; should NOT predict wrong well due to timing | Frame 1–20: tip descends into A1; frames 21–40: liquid begins to flow (meniscus drops); frames 41–50: tip withdrawn abruptly while liquid still flowing. Ground truth: {"wells": [A1], "event_type": "partial_dispense"} |
| **Enter vs Exit confusion** | Model must distinguish between TWO temporal phases: (1) pipette ENTERING well (tip descends, not yet dispensed) vs. (2) pipette EXITING well (tip leaves after dispense). Simple visual similarity between frames at entry and exit can confuse model | **High** | Temporal order validation: check frame timestamps. If pipette is descending (Z increasing frame-by-frame), entry phase; if ascending (Z decreasing), exit phase. Use optical flow to determine direction | Model must NOT predict same well twice (once on entry, once on exit). Correct behavior: predict well ONLY once, during/after dispense. Entry-only or exit-only predictions are errors | Frame 1–30: pipette descends into A1 (entry); frames 31–60: dispense occurs (static position); frames 61–90: pipette ascends away from A1 (exit). Ground truth: {"wells": [A1]} (one prediction, not two) |

---

## 4. Failure Mode Analysis

### 4.1 Deep Learning Architecture Failures

#### 4.1.1 Silent Failure: Wrong Prediction with High Confidence

**Symptom:** Model outputs well C5 with confidence 0.92, but video clearly shows A1.

**Root Cause:**
- Feature representations for A1 and C5 are similar in high-dimensional space
- Model overfitted to confusable wells in training data
- Lack of contrastive learning between visually similar wells

**Detection:**
- [ ] Cross-validation: train/val accuracy diverges significantly
- [ ] Confidence calibration: compare model confidence vs. actual accuracy per well
- [ ] Human review of top-confidence mistakes

**Mitigation:**
1. Use confidence score as a **validity check**, not just decision threshold
2. Implement **uncertainty quantification:** output confidence intervals, not point estimates
3. Flag predictions with confidence 0.8–0.95 as "borderline"; recommend human review
4. Add **contrastive loss** during training to maximize inter-well separation

---

#### 4.1.2 Class Bias Failure: Always Predicts Common Wells

**Symptom:** Model predicts well A1 (most frequent in training) for every sample.

**Root Cause:**
- Severe class imbalance (some wells 10x more frequent than others)
- Cross-entropy loss dominated by majority class
- Model learns majority-class features and ignores minority wells

**Detection:**
- [ ] Validation accuracy per well: create confusion matrix per well
- [ ] Output distribution: track histogram of predicted wells across validation set
- [ ] If any well represents >30% of predictions, flag bias

**Mitigation:**
1. **Weighted cross-entropy loss:** upweight rare wells (weight ∝ 1 / class_frequency)
2. **Focal loss:** down-weight easy examples, focus on hard (rare) ones
3. **Class-balanced sampling:** oversample rare wells or undersample majority
4. **Post-processing constraint:** if model predicts >3 samples of the same well in a row, flag anomaly

---

#### 4.1.3 Multi-Label Failure: Wrong Cardinality Prediction

**Symptom:** 8-channel operation (8 wells expected) but model predicts 3 wells.

**Root Cause:**
- Insufficient training examples of 8-channel operations
- No explicit cardinality loss (model learns to predict wells independently)
- Confidence thresholding too aggressive for rare cardinality classes

**Detection:**
- [ ] Cardinality accuracy: track fraction of samples with correct well count
- [ ] Separate metrics for 1-well vs. 8-well vs. 12-well operations
- [ ] If cardinality accuracy <70% on validation, abort training run

**Mitigation:**
1. **Explicit cardinality prediction head:** add auxiliary classifier (1 vs. 8 vs. 12)
2. **Loss weighting:** `loss = 0.7 * well_loss + 0.3 * cardinality_loss`
3. **Post-processing rule:** 
   - Predict cardinality first (classify as 1, 8, or 12)
   - Then predict exactly that many wells (sort by confidence, take top-k)
4. **Data augmentation:** synthetize minority cardinality examples if needed

---

#### 4.1.4 Domain Shift Failure: Works in Training, Fails on Hold-Out

**Symptom:** 87% validation accuracy but 45% hold-out accuracy.

**Root Cause:**
- Lighting condition change (hold-out recorded under different lamp)
- Camera angle shift (slightly different viewpoint)
- Plate handling different (more reflections, more shadows)
- Temporal distribution shift (holds-out recorded later, with equipment wear)

**Detection:**
- [ ] Validation vs. test accuracy divergence > 20 percentage points
- [ ] Hold-out samples have different brightness/saturation histogram than training
- [ ] Visual inspection: hold-out videos "look different" from training videos

**Mitigation:**
1. **Domain adaptation training:** 
   - Use unlabeled hold-out data (if available pre-eval) to fine-tune
   - Apply test-time augmentation: predict on brightness-adjusted versions, average outputs
2. **Robust feature learning:**
   - Train with augmentation: random brightness, contrast, saturation shifts
   - Use domain-adversarial training (if time permits)
3. **Conservative thresholding:** lower confidence threshold for hold-out to accept more predictions
4. **Fallback to Classical CV:** if DL accuracy < 50% on first 3 hold-out samples, switch to plate detection + grid alignment

---

### 4.2 Classical CV Architecture Failures

#### 4.2.1 Plate Detection Cascade Failure

**Symptom:** Plate boundary detection fails; well grid not aligned to actual plate.

**Root Cause:**
- Canny edge detection over-/under-sensitive to glare, shadows
- Hough line fitting biased by hand, spurious edges
- Perspective correction assumes flat plate; fails on tilted plate

**Detection:**
- [ ] Validate detected corners against expected aspect ratio (should be ~1:1 square)
- [ ] Check if any corner outside image bounds
- [ ] Verify grid center roughly aligns with image center

**Mitigation:**
1. **Multi-scale detection:** run edge detection at 2–3 image scales; fuse results
2. **Morphological preprocessing:** close small holes, open thin artifacts
3. **RANSAC plate fitting:** robust to outlier edges
4. **Validation check:** if detected plate corners outside [0, image_width] × [0, image_height], reject

---

#### 4.2.2 Grid Alignment Drift: Small Error Becomes Large Misclassification

**Symptom:** Well grid off by 5 pixels; predicts A1 when truth is A2.

**Root Cause:**
- Plate corners detected accurately but grid divisions calculated with accumulated floating-point error
- Wells on plate edges most affected (non-linear magnification)
- Assumption of rectilinear grid fails on tilted or curved plate

**Detection:**
- [ ] Cross-validate detected well positions: overlap detected wells with manually labeled ground truth
- [ ] If > 3% of wells misaligned by >5 pixels, flag drift
- [ ] Visualize detected grid overlaid on frame; manual inspection

**Mitigation:**
1. **Sub-pixel refinement:** use template matching (NCC) to refine well center positions post-grid-generation
2. **Iterative refinement:** initialize coarse grid, then refine well boundaries using local edge detection
3. **Robustness check:** predict well only if confidence (NCC score) > 0.7; else output "LOW_CONFIDENCE"
4. **Fallback to DL:** if grid alignment drift detected, switch to learned feature-based well classification

---

#### 4.2.3 Tip Color Thresholding Breakdown

**Symptom:** Pipette tip not detected; model outputs no wells.

**Root Cause:**
- Tip color changes (different batch, lighting)
- HSV thresholding calibrated for one lighting condition; fails under different illumination
- Blue well plate and blue tip visually indistinguishable in some frames

**Detection:**
- [ ] Validate: tip must be detected in ≥80% of frames in a clip
- [ ] If tip detection fails, flag "TIP_NOT_DETECTED" and fallback to alternative method

**Mitigation:**
1. **Adaptive color thresholding:**
   - Estimate illumination (using white balance reference if available)
   - Adjust HSV thresholds dynamically per frame
2. **Multi-feature detection:** combine color + shape + texture
3. **Temporal consistency:** tip should move smoothly; use Kalman filter to track tip across frames
4. **Fallback:** if color fails, use shape-based detection (tip is elongated, pointed object)

---

### 4.3 System-Level Failures

#### 4.3.1 Timeout Failure: Inference >20 min for 10 Samples

**Symptom:** Hold-out evaluation times out after 20 minutes; only 6 of 10 samples evaluated.

**Root Cause:**
- Video decoding slow (unusual codec, no GPU acceleration)
- Deep learning inference on CPU instead of GPU
- Memory swapping (running out of RAM; using disk for virtual memory)
- Frame extraction naive loop (redundant operations)

**Detection:**
- [ ] Profile each stage: video loading, frame extraction, inference, JSON serialization
- [ ] Per-sample timing: log time for each sample; if any sample > 3 min, investigate
- [ ] Memory profiling: peak memory usage during evaluation

**Mitigation:**
1. **Optimize frame extraction:**
   - Load entire video into memory if < 500MB
   - Use hardware-accelerated decoder (NVIDIA NVDEC or FFmpeg hwaccel)
2. **Batch inference:** process multiple frames simultaneously if possible
3. **Model pruning:** use quantized (INT8) model if available; reduces memory and latency
4. **Timeout enforcement:**
   - Wrap inference in timeout context: `signal.alarm(3 * 60)` (3 min per sample)
   - If timeout triggered, skip sample and move to next
5. **Test on evaluation hardware:** run timing test on actual evaluation machine (not laptop)

---

#### 4.3.2 JSON Malformation: Output Doesn't Match Schema

**Symptom:** Model outputs `{"wells": [[1, 2], [3, 4]]}` (nested list) instead of `{"wells": [{"row": "A", "column": 1}, ...]}`

**Root Cause:**
- JSON serialization inconsistency between development and production
- Type mismatch: rows encoded as integers instead of strings
- Missing fields in output

**Detection:**
- [ ] Schema validation: use jsonschema library; validate output against schema before writing to stdout
- [ ] Unit test: run JSON serializer on dummy input; verify schema compliance

**Mitigation:**
1. **Explicit schema definition:**
   ```json
   {
     "type": "object",
     "properties": {
       "wells": {
         "type": "array",
         "items": {
           "type": "object",
           "properties": {
             "row": {"type": "string", "pattern": "^[A-H]$"},
             "column": {"type": "integer", "minimum": 1, "maximum": 12}
           },
           "required": ["row", "column"]
         }
       }
     },
     "required": ["wells"]
   }
   ```
2. **Pre-output validation:** run `jsonschema.validate(output, schema)` before printing
3. **Error handling:** if validation fails, catch exception, output error JSON:
   ```json
   {"error": "INVALID_JSON_SCHEMA", "reason": "..."}
   ```

---

#### 4.3.3 File Not Found: Video Path Invalid

**Symptom:** Script invoked with wrong video file path; throws "FileNotFoundError".

**Root Cause:**
- Relative path used instead of absolute path
- Symlink broken
- File deleted between submission and evaluation
- Typo in file path argument

**Detection:**
- [ ] Check file existence before inference: `os.path.exists(fpv_path) and os.path.exists(topview_path)`

**Mitigation:**
1. **Early validation:**
   ```python
   if not os.path.exists(fpv_path):
       print(json.dumps({"error": "FILE_NOT_FOUND", "file": fpv_path}))
       sys.exit(1)
   ```
2. **Absolute path requirement:** document that CLI requires absolute paths
3. **Informative error message:** include attempted path in error JSON

---

#### 4.3.4 Memory Overflow: Large Video Files OOM

**Symptom:** 4GB RAM insufficient; process killed by OS.

**Root Cause:**
- Video files larger than expected (>1GB)
- Naive frame extraction: load all frames into memory at once
- No streaming/chunked processing

**Detection:**
- [ ] Test with large video file (500MB–1GB); monitor peak memory

**Mitigation:**
1. **Streaming frame extraction:**
   - Process frames in chunks of 100–300 (not all at once)
   - Extract features on-the-fly; discard frame after processing
2. **Memory limit:** set max memory to 3GB; if exceeded, terminate gracefully
3. **Adaptive processing:** if input video > 500MB, use lower frame rate (e.g., every 2nd frame)

---

## 5. Acceptance Criteria & Calibration Framework

### 5.1 Revised Acceptance Criteria: Reproducibility Over Accuracy

**Red Team Finding:** "Criteria focus on exact-match accuracy. In science, Reproducibility > Accuracy. We should be measuring Uncertainty Calibration."

This section reframes acceptance criteria to prioritize well-calibrated uncertainty over raw accuracy.

#### 5.1.1 Evaluation Hierarchy (Reordered)

**Priority 1: Calibration Quality (Critical)**
- Expected Calibration Error (ECE) < 0.10 (well-calibrated)
- Reliability diagram: visualize confidence vs. empirical accuracy
- Interpretation: model's confidence scores should match reality

**Priority 2: Accuracy on Confident Predictions (Important)**
- Accuracy on predictions with confidence ≥ 0.7: target ≥ 90%
- Accuracy on predictions with confidence < 0.7: target ≥ 60% (intentionally lower, model is uncertain)
- This decouples accuracy from confidence, revealing miscalibration

**Priority 3: Coverage & Refusal Rate (Secondary)**
- Confident Refusal Rate: % of degraded/unseen inputs correctly rejected (confidence < 0.4)
- Target: ≥ 80% of intentionally difficult inputs should trigger refusal
- Coverage: % of well predictions vs. abstentions (target: 90% coverage with <10% uncertainty-triggered abstentions on clean data)

#### 5.1.2 Expected Calibration Error (ECE) Metric

Measure ECE on validation set:

```python
def compute_ece(predictions, confidences, ground_truth, n_bins=10):
    """
    predictions: array of predicted well IDs
    confidences: array of confidence scores [0, 1]
    ground_truth: array of true well IDs
    Returns: ECE in [0, 1]
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i+1])
        if mask.sum() > 0:
            bin_accuracy = (predictions[mask] == ground_truth[mask]).mean()
            bin_confidence = confidences[mask].mean()
            ece += abs(bin_accuracy - bin_confidence) * mask.sum() / len(predictions)
    return ece
```

**Acceptance Gate:** ECE < 0.10 (well-calibrated)

#### 5.1.3 Reliability Diagram Validation

Generate reliability diagram on hold-out validation split:
- X-axis: binned predicted confidences (10 bins: [0–0.1], [0.1–0.2], ..., [0.9–1.0])
- Y-axis: empirical accuracy within each bin
- Plot line y=x (perfect calibration)
- **Acceptance:** Curve should stay within confidence band of ±0.05 around y=x

#### 5.1.4 Calibration on Hold-Out Evaluation

The 10-sample hold-out set must include at least **2–3 deliberately challenging samples designed to trigger uncertainty:**

1. **Unseen Well Test:** At least one sample with a well that has zero training examples
   - Expected: model outputs low confidence or refusal (confidence < 0.4)
   - Fail if: model outputs high-confidence (>0.8) prediction for truly unseen well

2. **Degraded Video Quality Test:** At least one sample with:
   - Heavy motion blur, reduced contrast, or simulated glare
   - Expected: lower average confidence across predictions
   - Fail if: confidence stays high despite quality degradation

3. **Novel Context Test:** At least one sample with:
   - Different lighting, different pipette color, different plate orientation
   - Expected: model expresses uncertainty (lower average confidence)
   - Fail if: confidence unchanged from standard conditions

**Scoring:** Accept partial credit if model correctly outputs {"uncertain": true} or empty wells array rather than high-confidence wrong prediction.

#### 5.1.5 Confident Refusal Rate (Formal Protocol)

**Definition:** When model encounters adversarial/degraded input, does it correctly abstain?

**Test Setup:**
1. Create 5 "intentionally hard" validation samples:
   - Extreme blur (Gaussian kernel size 9)
   - 50% brightness reduction
   - Heavy shadow (dark half of plate)
   - Horizontal rotation 30°
   - Unseen well position (trained on A1–D6, test on E7–H12)

2. Run inference on these adversarial samples
3. Count how many trigger refusal (confidence < 0.4 OR output empty wells)

**Pass Criteria:**
- ≥ 80% of adversarial samples trigger refusal (≥4 out of 5)
- When model does predict on adversarial input, accuracy < 50% is acceptable (model uncertain)
- **Fail Criteria:** Model outputs high-confidence (>0.8) prediction on adversarial input

**Implementation:**
```python
def confident_refusal_rate(model, adversarial_samples, confidence_threshold=0.4):
    refusals = 0
    for sample in adversarial_samples:
        output = model.infer(sample)
        max_confidence = max([w['confidence'] for w in output['wells']] or [0.0])
        if max_confidence < confidence_threshold or len(output['wells']) == 0:
            refusals += 1
    return refusals / len(adversarial_samples)
```

### 5.2 Acceptance Criteria Summary Table

| Metric | Target | Interpretation | Red Team Alignment |
|--------|--------|-----------------|-------------------|
| **ECE** | < 0.10 | Model confidence matches empirical accuracy | Calibration focus |
| **Accuracy (confident)** | ≥ 90% on conf ≥ 0.7 | When confident, be right | Reproducibility |
| **Accuracy (uncertain)** | ≥ 60% on conf < 0.7 | Lower bar for uncertain preds | Appropriate uncertainty |
| **Confident Refusal Rate** | ≥ 80% on adversarial | Say "I don't know" on hard inputs | Uncertainty quantification |
| **Coverage** | 85–95% | Predictions on clean data; slight abstention on edge cases | Balance precision/coverage |
| **Exact-Match Accuracy** | ≥ 85% seen wells, ≥ 70% unseen | Overall accuracy; lower bar for unseen | Secondary metric |

---

## 6. Synthetic Data Quality Tests

The red team criticized the project's N=100 overfitting risk and recommended synthetic data generation. This section defines QA tests for synthetic data pipeline.

### 6.1 Domain Gap Measurement

**Objective:** Ensure synthetic data doesn't introduce visual artifacts that break real-world inference.

**Test:** Fréchet Inception Distance (FID) score between real and synthetic feature distributions.

```python
def compute_fid_score(real_samples, synthetic_samples, model_backbone):
    """
    Measure domain gap between real and synthetic video frames.
    Uses ResNet-18 intermediate features (before classification).
    """
    real_features = extract_features(real_samples, model_backbone)  # (N, 512, 7, 7)
    synthetic_features = extract_features(synthetic_samples, model_backbone)
    
    # Compute statistics per spatial location and channel
    mu_real = real_features.mean(axis=(0, 2, 3))  # (512,)
    mu_synthetic = synthetic_features.mean(axis=(0, 2, 3))
    
    sigma_real = np.cov(real_features.reshape(real_features.shape[0], -1).T)
    sigma_synthetic = np.cov(synthetic_features.reshape(synthetic_features.shape[0], -1).T)
    
    # FID formula
    fid = np.linalg.norm(mu_real - mu_synthetic) ** 2
    fid += np.trace(sigma_real + sigma_synthetic - 2 * scipy.linalg.sqrtm(sigma_real @ sigma_synthetic))
    
    return fid
```

**Acceptance Criteria:**
- FID < 15 (tight domain alignment; FID 0 = perfect, >50 = poor)
- Interpretation: synthetic features should cluster near real features in embedding space

### 6.2 Label Accuracy Verification for Synthetic Samples

**Objective:** Synthetic data with ground-truth dispensing events must have accurate labels.

**Test:**
1. Generate 100 synthetic samples with known ground-truth well positions
2. Run well detection pipeline on synthetic frames
3. Compare detected well positions to ground truth

**Acceptance Criteria:**
- Detected well positions match synthetic ground truth within ±1 pixel: ≥ 98% (synthetic labels should be exact)
- If accuracy < 98%, synthetic data generation has systematic bias

### 6.3 Model Performance: Real Only vs. Real+Synthetic

**Objective:** Ensure synthetic data improves validation accuracy (doesn't hurt).

**Test:** Train two models:
- Model A: ResNet-18 on 100 real samples only
- Model B: ResNet-18 on 100 real + 900 synthetic samples

**Validation Split:** Same 30 real samples for both (fair comparison)

**Acceptance Criteria:**
- Model B validation accuracy ≥ Model A accuracy (synthetic should help or be neutral)
- Model B validation accuracy - Model A ≥ 0 (no regression allowed)
- Interpretation: synthetic data does not introduce harmful domain shift

### 6.4 Overfitting Detection: Training vs. Validation Gap

**Objective:** Synthetic data should help generalization; gap between train/val should narrow.

**Test:**
```python
training_accuracy = compute_accuracy(model, train_set)
validation_accuracy = compute_accuracy(model, val_set)
generalization_gap = training_accuracy - validation_accuracy
```

**Acceptance Criteria:**
- Generalization gap < 15 percentage points (red team's baseline concern was 30–40 point gap with N=100)
- If gap > 15%, model is still overfitting; need more regularization or synthetic data

---

## 7. Robustness Testing Plan

### 7.1 Confident Refusal Testing Protocol (Formal Acceptance Gate)

**Purpose:** Validate that model outputs uncertainty appropriately. This is a NEW acceptance gate that must PASS before hold-out evaluation begins.

#### 7.1.1 Adversarial Test Case Construction

Create 5 validation samples with properties designed to trigger uncertainty:

**Test Case 1: Heavily Blurred Video**
- Apply Gaussian blur (kernel size 7–9) to all frames
- Well boundaries become indistinguishable
- Expected: model confidence < 0.4 OR empty wells array
- Fail if: model outputs high-confidence (>0.8) predictions

**Test Case 2: Extreme Glare**
- Brighten 40–50% of frame to saturation (pixel values > 240)
- Introduce specular highlights on well plate
- Expected: model outputs low confidence OR refusal
- Fail if: model confidently predicts wells in glare region

**Test Case 3: Unseen Well Position**
- Use well position from training data, but place pipette over well NOT in training set
- Example: if training had wells A1–D6, test with E7, F8, H12
- Expected: model recognizes unseen well and outputs {"uncertain": true} or low confidence
- Fail if: model outputs high-confidence prediction for unseen well

**Test Case 4: Severe Underexposure**
- Reduce brightness 40–50%; image becomes dark, well boundaries disappear
- Expected: model outputs low confidence
- Fail if: model maintains high confidence despite darkness

**Test Case 5: Severe Rotation**
- Rotate frame 25–30°; well grid no longer aligned to image axes
- Expected: model recognizes out-of-distribution rotation and outputs low confidence
- Fail if: model predicts rotated grid as if it were standard orientation

#### 7.1.2 Pass Criteria

**Quantitative:**
- At least 4 out of 5 adversarial test cases must trigger refusal (confidence < 0.4 OR empty wells output)
- When model does predict on adversarial input: accuracy on adversarial data < 50% is acceptable (model is appropriately uncertain)
- When model refusals correctly, no penalty

**Qualitative (Manual Review):**
- Review top-3 adversarial cases where model outputs predictions
- Inspect confidence scores: should be visibly lower than on clean test data
- Confidence trajectory over frames should show hesitation (confidence rising/falling, not stable high)

#### 7.1.3 Fail Criteria

**Hard Fail (STOP, do not proceed to hold-out):**
- Fewer than 4/5 adversarial cases trigger refusal
- Model outputs high confidence (>0.8) on more than 1 adversarial case
- Any adversarial case predicts >3 wells (spurious predictions due to noise/artifacts)

#### 7.1.4 Implementation Pseudo-Code

```python
def test_confident_refusal(model, adversarial_samples, confidence_threshold=0.4):
    """
    Validate model's uncertainty behavior on adversarial inputs.
    Returns: (refusal_rate, fail_reasons)
    """
    refusals = 0
    failures = []
    
    for idx, (sample, description) in enumerate(adversarial_samples):
        output = model.infer(sample)
        wells = output.get('wells', [])
        
        # Check refusal
        if len(wells) == 0:
            refusals += 1
            print(f"✓ Case {idx} ({description}): Refusal (empty wells)")
            continue
        
        max_confidence = max([w['confidence'] for w in wells])
        
        if max_confidence < confidence_threshold:
            refusals += 1
            print(f"✓ Case {idx} ({description}): Low confidence refusal ({max_confidence:.3f})")
        else:
            # High confidence on adversarial input: potential failure
            failures.append({
                'case': idx,
                'description': description,
                'max_confidence': max_confidence,
                'well_count': len(wells)
            })
            print(f"✗ Case {idx} ({description}): High confidence ({max_confidence:.3f}) on adversarial input")
    
    refusal_rate = refusals / len(adversarial_samples)
    print(f"\nRefusal Rate: {refusal_rate:.2%} ({refusals}/{len(adversarial_samples)})")
    
    # HARD FAIL if < 80% refusal rate
    if refusal_rate < 0.80:
        return False, f"Refusal rate {refusal_rate:.2%} < 80% threshold"
    
    # WARN if high confidences detected (but allow <20% fail rate)
    if failures:
        print(f"WARNING: {len(failures)} adversarial cases with high confidence:")
        for f in failures:
            print(f"  - Case {f['case']}: {f['description']} (conf={f['max_confidence']:.3f})")
        return True, f"Passed with {len(failures)} warnings"
    
    return True, "All adversarial cases correctly triggered refusal"

# GATE: Must pass before hold-out evaluation
success, reason = test_confident_refusal(model, adversarial_test_cases)
if not success:
    print("ERROR: Confident Refusal test FAILED. Cannot proceed to hold-out evaluation.")
    sys.exit(1)
```

---

### 7.2 Pre-Submission Testing Checklist

Before submitting final solution for hold-out evaluation, QA Engineer must verify:

#### Required Implementation & Validation Steps

- [ ] **Unit Tests Pass:** All frame extraction, well classification, JSON serialization tests pass
- [ ] **Integration Tests Pass:** Dual-view fusion works; no crashes on 10 sample videos
- [ ] **System Tests Pass:** CLI produces valid JSON for all 10 local test samples in <20 min total
- [ ] **Synthetic Data Validation (NEW):**
  - [ ] FID score between real and synthetic feature distributions < 15
  - [ ] Synthetic label accuracy (detected wells match ground truth) ≥ 98%
  - [ ] Real+Synthetic model accuracy ≥ Real-only model accuracy (no regression)
  - [ ] Generalization gap (train - val) < 15 percentage points
- [ ] **Calibration Metrics (NEW):**
  - [ ] Compute ECE on validation set: ECE < 0.10 (well-calibrated)
  - [ ] Generate and inspect reliability diagram: curve within ±0.05 of y=x
  - [ ] Accuracy on confident predictions (conf ≥ 0.7): ≥ 90%
  - [ ] Accuracy on uncertain predictions (conf < 0.7): ≥ 60%
- [ ] **Confident Refusal Gate (NEW - MANDATORY):**
  - [ ] Construct 5 adversarial test cases (blur, glare, unseen well, dark, rotation)
  - [ ] Run `test_confident_refusal()` protocol (Section 7.1)
  - [ ] Achieve ≥ 80% refusal rate on adversarial inputs
  - [ ] **STOP and fix if this fails; do not proceed to hold-out**
- [ ] **Edge Case Sanity Checks (Updated):**
  - [ ] Rotated plate test (manually rotate one test frame 15°; verify grid detection recovers)
  - [ ] Glare test (brighten test frame to saturation; verify detections still valid OR refusal triggered)
  - [ ] Dark test (darken test frame; verify fallback methods work OR confidence < 0.4)
  - [ ] Corrupted frame test (insert noise into one frame; verify pipeline doesn't crash)
  - [ ] **NEW: Transparency/Specular Reflection test** (Section 3.7 edge cases)
    - [ ] Test on frame with simulated glare (specular highlight covering 30%+ of plate)
    - [ ] Verify detections exclude glare region or confidence drops
  - [ ] **NEW: Temporal Edge Case tests** (Section 3.8)
    - [ ] Hovering without dispense: model outputs empty wells ✓
    - [ ] Multiple dispenses: model segments correctly OR documents which dispense (primary) ✓
    - [ ] Partial dispense: model outputs correct well despite incomplete action ✓
    - [ ] Enter vs exit: model predicts well only once, not twice ✓
- [ ] **Schema Validation:** `python -m jsonschema validate output.json schema.json` passes for all outputs
- [ ] **Timeout Regression:** wall-clock time for 10 samples ≤ 18 min (2 min buffer)
- [ ] **Memory Profiling:** peak memory during evaluation < 3GB
- [ ] **Coverage Analysis (NEW):**
  - [ ] Data scientist provides coverage heatmap: which wells appear in training, which are unseen
  - [ ] QA identifies 2–3 hold-out samples with unseen wells for uncertainty calibration verification

### 5.2 Synthetic Perturbation Tests (Design, Not Execution)

These tests validate robustness without requiring actual implementation before evaluation. **Describe the plan; execute during development iteration if time permits.**

#### Brightness Perturbation
**Test:** Generate synthetic versions of 5 test frames with brightness adjustments (-50%, -25%, +25%, +50%).
**Validation:** Model predictions should be stable (±10% confidence variance) across brightness range.
**Acceptance:** Predicted wells remain constant across brightness variants.

#### Rotation Perturbation
**Test:** Rotate test frames by -10°, -5°, +5°, +10°.
**Validation:** Well predictions should rotate accordingly (e.g., A1 → A12 when rotated 90°).
**Acceptance:** Grid alignment post-processing correctly de-rotates; final predictions match original frame.

#### Noise Perturbation
**Test:** Add Gaussian noise (σ = 10, 20, 30 pixels) to test frames.
**Validation:** Model accuracy degrades gracefully; confidence scores decrease proportionally to noise level.
**Acceptance:** Accuracy loss < 5% per 10-unit noise increase.

#### Scale Perturbation
**Test:** Downscale test frames (0.5x, 0.75x) then upscale back to original.
**Validation:** Well detection should still work with slight confidence reduction.
**Acceptance:** Accuracy loss < 10% for 0.5x downscale.

#### Blur Perturbation
**Test:** Apply Gaussian blur (kernel sizes 3, 5, 7) to test frames.
**Validation:** Pipette tip detection should still work; well predictions may have lower confidence.
**Acceptance:** Accuracy >= 80% at kernel size 3; >= 70% at kernel size 5.

#### Geometric Perturbation
**Test:** Apply small affine transformations (shear, perspective) to simulate tilted plate.
**Validation:** Grid alignment + perspective correction should handle ±5° tilt.
**Acceptance:** Predicted wells consistent with ground truth within ±1 well tolerance.

### 5.3 JSON Output Schema Validation

**Automated check before each inference:**

```python
import json
import jsonschema

SCHEMA = {
    "type": "object",
    "properties": {
        "wells": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "row": {
                        "type": "string",
                        "enum": ["A", "B", "C", "D", "E", "F", "G", "H"]
                    },
                    "column": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 12
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                },
                "required": ["row", "column"],
                "additionalProperties": False
            }
        }
    },
    "required": ["wells"],
    "additionalProperties": False
}

def validate_output(output_dict):
    try:
        jsonschema.validate(output_dict, SCHEMA)
        return True, None
    except jsonschema.ValidationError as e:
        return False, str(e)

# Before printing output:
valid, error = validate_output(output_dict)
if not valid:
    print(json.dumps({"error": "SCHEMA_VALIDATION_FAILED", "reason": error}), file=sys.stdout)
    sys.exit(1)
else:
    print(json.dumps(output_dict), file=sys.stdout)
```

### 5.4 Timeout Regression Tests

**Test:** Measure wall-clock time for inference on 10 samples; verify ≤ 18 min.

**Implementation:**
```python
import time

start_time = time.time()
results = []
for i, (fpv_path, topview_path) in enumerate(sample_pairs):
    sample_start = time.time()
    result = infer_sample(fpv_path, topview_path)
    sample_time = time.time() - sample_start
    results.append((i, sample_time, result))
    print(f"Sample {i}: {sample_time:.2f}s", file=sys.stderr)
    
    if sample_time > 180:  # 3 minute timeout
        print(f"WARNING: Sample {i} exceeded 3-minute timeout. Skipping.", file=sys.stderr)
        results[i] = (i, sample_time, {"error": "TIMEOUT"})

total_time = time.time() - start_time
print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} min)", file=sys.stderr)
assert total_time < 1080, f"Total time {total_time}s exceeds 18-minute limit"
```

**Acceptance Criteria:**
- Total time for 10 samples ≤ 18 minutes
- All per-sample times ≤ 3 minutes
- No samples timeout-killed

---

## 6. Feedback to Other Roles

### 6.1 To the Data Scientist

#### 6.1.1 Data Analysis Recommendations

**Action:** Conduct detailed coverage analysis **before** hold-out evaluation.

**Specific Questions to Answer:**
1. **Well Coverage Heatmap:**
   - For each of 96 wells, count training samples containing that well
   - Visualize as 8×12 grid; mark "covered" (≥2 samples), "rare" (1 sample), "unseen" (0 samples)
   - Quantify: "45 wells have 0 training examples; 20 wells have exactly 1"

2. **Cardinality Distribution:**
   - What % of training samples are 1-channel? 8-channel? 12-channel?
   - Hypothesis: if <20% are multi-channel, model will struggle predicting 8/12-well operations
   - Recommendation: if severe imbalance (e.g., 90% single-channel), use weighted loss or synthetic augmentation

3. **Difficult Well Pairs:**
   - Which pairs of wells are visually most similar in training data?
   - Example: wells A1 and B1 may have similar row orientation (tip trajectory nearly vertical)
   - Recommendation: use contrastive learning or data augmentation to separate confusion-prone pairs

4. **Temporal Analysis:**
   - Analyze video clip durations: min, max, median
   - If any clips < 20 frames at 30fps, flag as "very short"
   - Recommendation: require minimum 0.5–1 second clip duration

5. **View-Specific Signals:**
   - Which views (FPV vs. top-view) are more predictive for single-channel vs. multi-channel?
   - Hypothesis: FPV better for temporal cues; top-view better for spatial precision
   - Recommendation: use view-specific confidence weighting in fusion

#### 6.1.2 Visualization Deliverables

**Request the following visualizations to assess hold-out risk:**

1. **Coverage heatmap** (8×12 grid showing well frequency in training)
2. **Cardinality histogram** (distribution of 1, 8, 12-well operations)
3. **Per-well accuracy** from cross-validation (which wells are hardest to predict?)
4. **Confusion matrix** (which well pairs are confused most often?)
5. **Temporal signal quality** (histogram of video durations, frame counts)

---

### 6.2 To the Architect

#### 6.2.1 Defensive Programming Requirements

**Critical:** The solution must not crash under any circumstance. Implement robust error handling:

1. **Input Validation Layer:**
   ```python
   def validate_inputs(fpv_path, topview_path):
       errors = []
       if not os.path.exists(fpv_path):
           errors.append(f"FPV file not found: {fpv_path}")
       if not os.path.exists(topview_path):
           errors.append(f"Top-view file not found: {topview_path}")
       if not is_valid_video(fpv_path):
           errors.append(f"FPV is not a valid video file")
       if not is_valid_video(topview_path):
           errors.append(f"Top-view is not a valid video file")
       if errors:
           return False, errors
       return True, None
   ```

2. **Exception Wrapper:**
   - Wrap all main inference steps in try/except
   - Never let unhandled exceptions propagate; always output error JSON
   - Log full stack trace to stderr for debugging

3. **Graceful Degradation:**
   - If DL model fails to load, fallback to Classical CV
   - If top-view preprocessing fails, use FPV alone
   - If both fail, output empty wells array with warning

4. **Validation Checkpoints:**
   - After video loading: verify frame count, resolution, codec
   - After frame extraction: verify pixel data valid (no NaN, no all-black)
   - After well detection: verify coordinates in bounds [A-H, 1-12]
   - After confidence thresholding: verify output cardinality valid (1, 8, or 12)

#### 6.2.2 Fallback Behavior Strategy

**Recommendation:** Implement a three-tier fallback:

**Tier 1: Primary Model (Deep Learning ResNet-18)**
- Use if loaded and inference succeeds
- Output confidence scores with each well

**Tier 2: Fallback (Classical CV)**
- Triggered if: DL model fails to load, inference crashes, or confidence all <0.3
- Use plate detection → grid alignment → tip detection → well matching
- Output lower-confidence predictions; flag as fallback

**Tier 3: Emergency (Heuristic)**
- Triggered if: both DL and CV fail
- Use frame with highest motion (likely dispense moment)
- Use simple centroid of tip + nearest grid cell
- Output single well with very low confidence; log "EMERGENCY_MODE"

**Never output empty wells without logging reason:**
```python
if len(output_wells) == 0:
    logger.warning(f"No wells detected. DL confidence max={dl_max_conf:.3f}, CV failed={cv_failed}")
```

---

### 6.3 To the ML Scientist

#### 6.3.1 Post-Processing Rules

**Implement the following post-processing layer after model inference:**

1. **Well Coordinate Validation:**
   ```python
   VALID_ROWS = set('ABCDEFGH')
   VALID_COLS = set(range(1, 13))
   
   filtered_wells = []
   for well in predicted_wells:
       row, col = well['row'], well['column']
       if row in VALID_ROWS and col in VALID_COLS:
           filtered_wells.append(well)
       else:
           logger.warning(f"Invalid well coordinates: {row}{col}. Skipping.")
   
   return filtered_wells
   ```

2. **Duplicate Removal & Sorting:**
   ```python
   # Remove duplicates
   unique_wells = {}
   for well in filtered_wells:
       key = (well['row'], well['column'])
       if key not in unique_wells or well['confidence'] > unique_wells[key]['confidence']:
           unique_wells[key] = well
   
   # Sort in canonical order
   output_wells = sorted(
       unique_wells.values(),
       key=lambda w: (w['row'], w['column'])
   )
   return output_wells
   ```

3. **Cardinality Enforcement:**
   ```python
   def enforce_cardinality(wells, predicted_cardinality):
       """Enforce exact cardinality: 1, 8, or 12 wells."""
       target_count = {1: 1, 8: 8, 12: 12}.get(predicted_cardinality, 1)
       sorted_wells = sorted(wells, key=lambda w: w['confidence'], reverse=True)
       return sorted_wells[:target_count]
   ```

4. **Confidence Post-Processing:**
   - Clip all confidences to [0, 1]
   - Round to 3 decimal places
   - Flag wells with 0.3 < confidence < 0.5 as "LOW_CONFIDENCE"

#### 6.3.2 Confidence Threshold Strategy

**Recommendation: Implement adaptive thresholding based on dataset characteristics.**

1. **Single-Channel Default:**
   ```python
   if predicted_cardinality == 1:
       # For single-well, be more conservative
       confidence_threshold = 0.4
       output_top_1 = wells[0] if wells[0]['confidence'] >= threshold else None
   ```

2. **Multi-Channel Default:**
   ```python
   if predicted_cardinality == 8:
       # For 8-channel, accept lower confidence (rare in training)
       confidence_threshold = 0.2
       # Enforce exactly 8 wells; sort by confidence and take top-8
   ```

3. **Unseen Well Strategy:**
   ```python
   if is_unseen_well(row, col):
       # For wells not in training set, lower threshold further
       confidence_threshold = 0.15  # Very generous
       # Or: skip output entirely (safer, avoids spurious guesses)
       logger.warning(f"Well {row}{col} was unseen in training. Confidence {conf:.3f}")
       if conf < 0.3:
           continue  # Skip output
   ```

4. **Validation:**
   - Log threshold applied for each well
   - A/B test thresholds on validation data: compare F1 scores
   - **Never silently change threshold; must be explicit in code**

#### 6.3.3 Handling Zero Wells Prediction (All Confidences Below Threshold)

**Scenario:** All model confidences < threshold; output would be empty.

**Recommended Behavior (in priority order):**

1. **Option A: Output highest-confidence well (fallback):**
   ```python
   if len(output_wells) == 0 and len(all_wells) > 0:
       best_well = max(all_wells, key=lambda w: w['confidence'])
       if best_well['confidence'] > 0.05:  # Sanity check
           output_wells.append(best_well)
           logger.warning(f"All confidences below threshold. Outputting best guess: {best_well}")
   ```

2. **Option B: Output empty array with warning (conservative):**
   ```python
   if len(output_wells) == 0:
       logger.warning("No wells exceeded confidence threshold. Outputting empty array.")
       output_wells = []
   ```

3. **Option C: Trigger fallback to Classical CV:**
   ```python
   if len(output_wells) == 0:
       logger.warning("DL model produced no confident predictions. Switching to CV fallback.")
       output_wells = classical_cv_infer(fpv_path, topview_path)
   ```

**Recommendation:** Use **Option B (conservative)** for hold-out evaluation. Better to output nothing than output a random guess. Allows human review.

---

## 7. Acceptance Criteria

### 7.1 Minimum Accuracy Thresholds

**Hold-out evaluation will be scored on a per-sample basis. Define passing criteria:**

#### 7.1.1 Per-Sample Accuracy

| Metric | Definition | Threshold | Rationale |
|--------|-----------|-----------|-----------|
| **Exact Match** | All predicted wells match ground truth exactly (no false positives, no false negatives) | ≥ 80% of samples | Core requirement: at least 8 of 10 samples correct |
| **Partial Match (Cardinality)** | Predicted cardinality matches (e.g., 8 wells predicted when 8 needed) even if some wells differ | ≥ 90% of samples | Cardinality is the primary challenge; accuracy here validates architecture choice |
| **Overlap Match** | At least 70% of predicted wells are correct (allows 1–2 errors per sample) | ≥ 90% of samples | Tolerates rare misclassifications |
| **Any Correct Well** | At least 1 predicted well is correct | ≥ 95% of samples | Baseline: model shouldn't be completely wrong |

#### 7.1.2 Coverage-Weighted Accuracy

**Acknowledge that unseen wells are inherently harder. Define tiered targets:**

| Well Category | Training Frequency | Target Accuracy | Reasoning |
|---------------|-------------------|-----------------|-----------|
| Common wells | ≥ 3 training examples | ≥ 90% | Model has learned these well |
| Rare wells | 1–2 training examples | ≥ 75% | Some overfitting risk, but learnable |
| Unseen wells | 0 training examples | ≥ 50% | Pure generalization; 50% = better than random |

**Weighted Score:**
```
accuracy_weighted = (
    0.60 * accuracy_common +
    0.25 * accuracy_rare +
    0.15 * accuracy_unseen
)
```

**Pass threshold: accuracy_weighted ≥ 0.75 (75%)**

---

### 7.2 Latency Requirement

| Constraint | Threshold | Notes |
|-----------|-----------|-------|
| **Total time for 10 samples** | ≤ 20 minutes | Hard deadline per problem statement |
| **Per-sample timeout** | ≤ 3 minutes | Any sample exceeding 3 min auto-skipped |
| **Average per-sample** | ≤ 120 seconds | ~2 min per sample (20 min / 10) |
| **Peak memory usage** | ≤ 3.5 GB | Evaluation hardware typically has 4–8 GB RAM |

**Latency Pass Criteria:**
- [ ] No sample times out
- [ ] Total execution time ≤ 19 minutes (1 min safety buffer)
- [ ] Peak memory < 3.5 GB

---

### 7.3 Output Format Requirements

**All JSON output must satisfy:**

1. **Schema Compliance:**
   - [ ] Valid JSON (parseable by `json.loads()`)
   - [ ] Top-level object with key `"wells"`
   - [ ] `"wells"` is an array of objects
   - [ ] Each object has keys `"row"` (string A-H) and `"column"` (int 1-12)
   - [ ] Optional `"confidence"` (float 0-1)
   - [ ] No extraneous keys or comments

2. **Coordinate Validation:**
   - [ ] All rows in [A-H]
   - [ ] All columns in [1-12]
   - [ ] No duplicates (same well listed twice)
   - [ ] Rows uppercase (A, not a)
   - [ ] Columns integers (1, not 1.0)

3. **Cardinality Validation:**
   - [ ] Output length in {1, 8, 12} or 0 (empty)
   - [ ] If training data shows all operations are 1-channel, avoid predicting 8 or 12

4. **No Extraneous Output:**
   - [ ] Exactly one line to stdout: the JSON object
   - [ ] All debug/logging to stderr only
   - [ ] No print statements, assertions, or stack traces to stdout

**Example Valid Output:**
```json
{
  "wells": [
    {"row": "A", "column": 1, "confidence": 0.92},
    {"row": "A", "column": 2, "confidence": 0.88}
  ]
}
```

**Example Invalid Output:**
```json
{
  "wells": [[0, 1], [0, 2]],
  "debug": "..."
}
```

---

### 7.4 Overall Pass/Fail Decision (REVISED: Reproducibility-First)

**GATE 0 (Pre-Hold-Out, Non-Negotiable):** Confident Refusal test (Section 7.1) must PASS.
- If Confident Refusal test fails (refusal_rate < 80%), **STOP. Do not proceed to hold-out evaluation.**
- This ensures model doesn't output high-confidence guesses on adversarial inputs.

**Solution passes hold-out evaluation if and only if ALL of the following are met:**

#### Tier 1: Calibration (Critical — Red Team Priority)

| Criterion | Pass Condition | Rationale |
|-----------|---|---|
| **Calibration Quality** | ECE < 0.10 (well-calibrated) | Red team: reproducibility > accuracy. Model confidence must match reality. |
| **Confidence-Accuracy Alignment** | Accuracy(conf ≥ 0.7) ≥ 90% AND Accuracy(conf < 0.7) ≥ 60% | When model is confident, be right. When uncertain, lower expectations. |
| **Refusal on Uncertainty** | Confident Refusal Rate ≥ 80% on hold-out's hard cases | Model must say "I don't know" on unclear inputs, not guess. |

#### Tier 2: Accuracy (Important, but Calibration First)

| Criterion | Pass Condition | Rationale |
|-----------|---|---|
| **Exact-Match Accuracy** | ≥ 85% on seen wells, ≥ 70% on unseen wells, weighted average ≥ 80% | Seen wells: tight tolerance. Unseen wells: generalization is harder; lower bar acceptable. |
| **Cardinality Accuracy** | Separate for 1-well, 8-well, 12-well; ≥ 75% overall cardinality correct | Multi-channel operations must predict correct well count. |
| **Per-Well Consistency** | Predictions align with visual inspection (human sanity check on 3–5 worst cases) | Edge case validation: does model's worst-case behavior make sense? |

#### Tier 3: Operational (Important but Not Blocking)

| Criterion | Pass Condition | Rationale |
|-----------|---|---|
| **Latency** | Total time ≤ 20 minutes AND all per-sample times ≤ 3 minutes | Operational requirement; if calibration is perfect but slow, still worth noting |
| **Output Format** | 100% of outputs valid JSON AND comply with schema | Non-negotiable for infrastructure |
| **No Crashes** | 0 runtime exceptions; graceful error handling on all 10 samples | Robustness requirement |

**Pass Decision Logic:**
```
IF (Calibration Tier ALL pass) AND (Accuracy Tier at least 2/3 pass) AND (Operational Tier ALL pass):
    PASS
ELIF (Calibration Tier PARTIALLY pass) AND (Accuracy Tier 3/3 pass) AND (Operational Tier ALL pass):
    CONDITIONAL PASS (note: model trades some calibration for raw accuracy; acceptable if ECE < 0.15)
ELSE:
    FAIL
```

**Interpretation:**
- **PASS:** Model is reproducible (well-calibrated) and accurate. Green light.
- **CONDITIONAL PASS:** Model is accurate but slightly miscalibrated (ECE 0.10–0.15). Yellow light; recommend uncertainty quantification improvement.
- **FAIL:** Model fails calibration or crashes. Red light; do not use in production.

**Partial Credit:** If latency exceeds 20 min but all other criteria pass, solution receives a "PASS with latency warning." Document for future optimization.

---

## 8. Risk Mitigation Summary

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| Hold-out contains unseen wells (5–15 of 96) | Very High | High | Obtain coverage heatmap pre-eval; set low confidence threshold for unseen wells; fallback to CV |
| Multi-label cardinality misprediction (8-well predicted as 1-well) | High | High | Implement explicit cardinality head; use weighted loss; post-processing constraint |
| Domain shift in lighting/camera angle | Medium | High | Train with augmentation; test-time augmentation; fallback to CV if DL accuracy <50% on first 3 samples |
| Timeout (inference >20 min) | Medium | Critical | Profile on evaluation hardware; use GPU acceleration; implement timeout enforcement; optimize video loading |
| Class bias (always predict common wells) | High | High | Use weighted cross-entropy or focal loss; monitor per-well accuracy; validate output well distribution |
| JSON schema mismatch | Low | Critical | Implement jsonschema validation before output; unit test JSON serialization |
| Plate detection failure (rotation, tilt, occlusion) | Medium | High | Use multi-scale edge detection; implement perspective correction; fallback to DL |
| Silent failure (wrong prediction with high confidence) | Medium | High | Implement uncertainty quantification; flag borderline predictions (0.8–0.95 confidence); contrastive learning |
| Memory overflow (OOM) | Low | Critical | Stream frame processing; set memory limits; adaptive downsampling for large videos |
| No wells detected at all | Low | Medium | Graceful fallback; output empty array with warning; allow human review |

---

## 9. Pre-Evaluation Checklist

**QA Engineer: Complete the following before hold-out evaluation begins.**

### Code Readiness
- [ ] All edge case handling implemented (see Section 3)
- [ ] Exception handling in place for all I/O operations
- [ ] JSON schema validation integrated into inference pipeline
- [ ] Timeout enforcement implemented (3 min per sample)
- [ ] Memory profiling completed; peak memory < 3.5 GB

### Testing Completeness
- [ ] Unit tests pass (100%)
- [ ] Integration tests pass (100%)
- [ ] System tests pass on 10 local samples
- [ ] Schema validation tests pass
- [ ] Timing tests confirm <20 min for 10 samples
- [ ] Edge case tests (rotation, glare, dark) manually validated

### Data Analysis Deliverables (from Data Scientist)
- [ ] Well coverage heatmap received and analyzed
- [ ] Cardinality distribution report received
- [ ] Per-well accuracy from CV received
- [ ] High-risk wells identified (unseen, rare, confusion-prone)

### Model & Architecture Finalization
- [ ] Final model weights frozen (no changes during eval)
- [ ] Architecture choice confirmed (DL vs. fallback CV)
- [ ] Confidence thresholds tuned on validation data
- [ ] Post-processing rules documented and tested
- [ ] Fallback triggers and behavior defined

### Documentation
- [ ] CLI usage documented (input format, output format, examples)
- [ ] Error codes documented (what each error JSON means)
- [ ] Known limitations documented (e.g., "Model may struggle with unseen wells")
- [ ] Fallback behavior documented

### Hardware & Environment
- [ ] Code tested on evaluation hardware (or equivalent spec)
- [ ] GPU availability confirmed (if DL model requires)
- [ ] Video codec support verified (FFmpeg hwaccel if available)
- [ ] Python dependencies frozen (requirements.txt)

### Final Dry Run
- [ ] End-to-end test on 10 local samples: all produce JSON in <20 min
- [ ] Manual inspection of output: no errors, format correct
- [ ] Stderr logs reviewed: all warnings/errors logged appropriately
- [ ] Git commit made with final version tag

---

## 10. Post-Evaluation Debrief

**After hold-out evaluation, QA Engineer will document:**

1. **Accuracy Results:**
   - Per-sample breakdown (which samples passed, which failed)
   - Coverage analysis (which wells were missed; were they seen/unseen?)
   - Cardinality accuracy (how often was correct number of wells predicted?)

2. **Failure Analysis:**
   - Root cause analysis for any failed samples (visual inspection of videos)
   - If any domain shift suspected (lighting different, etc.)
   - If any edge cases observed (plate tilted, hand in frame, etc.)

3. **Lessons Learned:**
   - What assumptions held vs. didn't hold about the data?
   - What edge cases were relevant/irrelevant?
   - How effective was each layer of QA (unit/integration/system)?

4. **Recommendations for Future Iterations:**
   - What data collection would reduce risk? (more multi-channel examples, harder lighting conditions)
   - What model improvements would help? (uncertainty quantification, domain adaptation)
   - What process improvements? (more granular timing profiling, confidence calibration)

---

## Appendix A: Risk Matrix

**Severity vs. Likelihood Assessment:**

```
         High Likelihood    Medium Likelihood    Low Likelihood
         
High     Unseen wells      Domain shift         Silent failure
Impact   in hold-out       (camera/lighting)    (wrong prediction,
                                               high confidence)
                           Multi-label card     
                           misprediction        JSON malformation
                           
Medium   Class bias        Plate detection      No wells detected
Impact   (always common)   failure
                           
         Per-well CV       Timeout              Corrupted video
         confusion
         
Low      Output sorting    (handled by          Memory OOM
Impact   inconsistency    architecture)        
```

**High-risk items (top-left quadrant):** Prioritize mitigation before eval.

---

## Appendix B: Edge Case Priority Matrix

| Edge Case | Severity | Frequency in Real Use | Recommended Action |
|-----------|----------|----------------------|-------------------|
| Extreme glare | Critical | Low | Implement histogram-based glare detection + edge detection fallback |
| Plate rotated | Critical | Medium | RANSAC grid fitting + rotation correction |
| Multi-channel (8/12-well) | Critical | Medium | Explicit cardinality prediction head |
| Mismatched views | Critical | Low | Cross-validate well predictions between FPV and top-view |
| Domain shift (lighting) | High | Medium | Test-time augmentation + fallback to CV |
| Motion blur | High | Medium | Use sharp middle frames; skip edge frames |
| Unseen wells | High | Very High | Probabilistic coverage analysis; graceful "SKIP" for <threshold confidence |
| Timeout | High | Low | Profile on target hardware; optimize video loading |
| Occluded plate | Medium | Low | Morphological preprocessing; attempt reconstruction |
| Tilted plate | Medium | Medium | Perspective correction (homography) |
| Dark/low light | Medium | Medium | Gamma correction; edge-based detection |
| Two plates on table | Medium | Low | Connected components; use largest plate |
| Pipette barely visible | Medium | Low | Expand search region; use FPV as primary |
| Multi-dispense in clip | Medium | Medium | Temporal segmentation by dispense event |
| Zero wells predicted | Medium | High | Allow empty output; log reason; validate acceptable |
| Out-of-bounds well | Critical | Low | Clamp coordinates; reject invalid predictions |

---

## Appendix C: Glossary

| Term | Definition |
|------|-----------|
| **FPV** | First-Person View; mounted camera showing pipette-tip-centric perspective |
| **Top-view** | Overhead camera showing full 96-well plate from above |
| **Well** | Single compartment in 96-well plate; identified by row (A-H) and column (1-12) |
| **Cardinality** | Number of wells dispensed into; typically 1, 8 (row), or 12 (column) |
| **Class Imbalance** | Severe skew in training data: some wells appear 10x more often than others |
| **Domain Shift** | Mismatch between training data distribution and evaluation data (e.g., different lighting) |
| **Unseen Well** | Well position with zero training examples; model must generalize |
| **Confidence** | Model's predicted probability; higher = model more certain |
| **JSON Schema** | Formal specification of valid JSON structure (e.g., "row must be string A-H") |
| **Fallback** | Alternative method triggered if primary approach fails (e.g., CV if DL crashes) |
| **Timeout** | Maximum allowed execution time; if exceeded, terminate gracefully |

---

**End of QA Strategy Document**

*For questions, contact QA Engineer.*
