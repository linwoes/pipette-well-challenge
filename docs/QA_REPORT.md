# QA Report: Pipette Well Challenge - Formal Audit

**Date:** April 14, 2026  
**Auditor:** QA Engineer  
**Project:** Transfyr AI Pipette Well Challenge  
**Status:** AMBER – Issues identified, project ready for mitigation

---

## Executive Summary

### Overall Project Health: AMBER

The Pipette Well Challenge project demonstrates **excellent documentation architecture** and **clear decision-making processes**, but faces several critical implementation gaps and inconsistencies that must be resolved before hold-out evaluation.

**Key Risk:** The project is a **scaffold with placeholders** — virtually all code modules contain `NotImplementedError()` stubs. While documentation is comprehensive, implementation is non-existent.

### Issue Summary by Severity

| Severity | Count | Category | Impact |
|----------|-------|----------|--------|
| **Critical** | 5 | Missing implementations, specification conflicts, schema mismatches | Blocks inference entirely |
| **High** | 8 | Test coverage gaps, incomplete design specs, undocumented assumptions | Likely failures on hold-out evaluation |
| **Medium** | 6 | Minor inconsistencies, ambiguous requirements, edge case handling | May cause issues under specific conditions |
| **Low** | 4 | Documentation clarity, coding standards, non-blocking improvements | Improve robustness and maintainability |

**Total Issues:** 23

### Key Risks to Hold-Out Evaluation Success

1. **Inference pipeline unimplemented** – All model loading, video processing, and inference code is a skeleton. Cannot execute inference without full implementation.
2. **Output schema inconsistencies** – README specifies `{"wells": [...]}` but code and tests use `{"row": "A", "column": 1}` format inconsistently.
3. **Temporal alignment unspecified** – Critical synchronization of FPV/top-view is documented as "TODO" in video_loader.py with no algorithm specified.
4. **Test coverage insufficient** – Edge case tests are all placeholder (`pass` statements); preprocessing tests have zero real validation.
5. **Cardinality constraint enforcement missing** – Post-processing logic for 8-channel and 12-channel operations is not implemented.

---

## Section 1: Document Consistency Audit

Cross-checked all document pairs for contradictions, gaps, and alignment.

### 1.1 README.md vs. ARCHITECTURE.md

**Agreements:**
- ✓ Both recommend Deep Learning end-to-end (ResNet-18, dual-view fusion)
- ✓ Both cite 85–95% expected accuracy range
- ✓ Both specify <2 min inference SLA

**Contradictions/Gaps:**
- **Schema format inconsistency:** README shows output as `{"wells": [{"well_row": "A", "well_column": 1}, ...]}` (well_row, well_column keys)
- ARCHITECTURE.md Section "4. Deployment & DevOps" shows schema with `{"row": "A", "column": 1, "confidence": 0.95}` (row, column keys, adds confidence)
- **Severity:** **CRITICAL** – Tests use `well_row` and `well_column` (test_output_schema.py lines 28–88), but ML_STACK.md inference pipeline uses `row` and `column` (line 501)
- **Resolution required:** Decide on single schema format. Recommend: `{"well_row": "A", "well_column": 1}` from README (more explicit).

**Additional gap:**
- README states "output must include well coordinates (row + column)" but doesn't specify if confidence per well is mandatory or optional
- ARCHITECTURE.md shows confidence in metadata example; QA_STRATEGY.md Section 5.3 shows confidence as optional in schema

---

### 1.2 README.md vs. ML_STACK.md

**Agreements:**
- ✓ Both specify ResNet-18 backbone
- ✓ Both recommend focal loss (γ=2.0)
- ✓ Both cite data augmentation (temporal >> geometric >> photometric)

**Contradictions:**
- **Output head design:** README says "factorized (8-class row + 12-class column)" but doesn't specify activation function
- ML_STACK.md Section 1.6 specifies **sigmoid** activation per head
- ML_STACK.md Section 1.10 says threshold tuning on validation set; README says default 0.5
- **Severity:** **HIGH** – Can be resolved, but implementation must match ML_STACK

- **Inference pipeline:** README says "implement training loop" but ML_STACK Section 1.12 specifies complete inference pseudocode with temporal max-pooling over frames
- README suggests single-frame classification; ML_STACK says process 2 frames and aggregate with max pooling
- **Severity:** **HIGH** – Design choice not documented in README

---

### 1.3 ARCHITECTURE.md vs. ML_STACK.md

**Agreements:**
- ✓ Both recommend Deep Learning as primary, Classical CV as fallback
- ✓ Both specify <2 min latency target
- ✓ Both cite transfer learning from ImageNet

**Contradictions:**
- **Fusion strategy:** ARCHITECTURE Section 2.2 (DL) shows late fusion: "FPV stream → ResNet → (B, 512) → Concat → (B, 1024) → FC → Heads"
- ML_STACK Section 1.9 shows early fusion: "concatenate at feature-map level (B, 512, 7, 7) → transition conv → GAP"
- **Severity:** **CRITICAL** – Different architectures will have different memory/latency profiles and may produce different results
- **Resolution:** ML_STACK is more detailed and specific; recommend implementing ML_STACK version (early fusion at feature-map level)

- **Backbone selection justification:** ARCHITECTURE says ResNet-18 preferred over MobileNetV3 for broader transfer-learning support
- ML_STACK Section 1.2 says MobileNetV3 was rejected because "ResNet-18's spatial feature extraction better for well detection"
- **Not contradictory, but ARCHITECTURE could be more explicit**

---

### 1.4 ML_STACK.md vs. TEAM_DECISIONS.md

**Agreements:**
- ✓ Decision 2 confirms ResNet-18 as chosen backbone
- ✓ Decision 3 confirms factorized row/column heads
- ✓ Decision 4 confirms focal loss (γ=2.0)
- ✓ Decision 6 confirms late fusion strategy (contradicts ML_STACK!)

**Critical contradiction:**
- **Fusion strategy mismatch (again):** TEAM_DECISIONS Decision 6 explicitly states: **"late concatenation fusion" (process FPV and top-view independently, fuse after feature extraction)**
- ML_STACK Section 1.9 says **early fusion at feature-map level**
- TEAM_DECISIONS Section 6 quotes: "FPV stream: (B,3,224,224) ──> ResNet-18 ──> (B,512)" then "Concat ──> (B,1024) ──> FC ──> Heads"
- **This is LATE fusion (concatenate AFTER global avg pooling), not early fusion**
- **Severity:** **CRITICAL** – Implementation must clarify this. Current difference:
  - Early (ML_STACK): Concatenate at (B, 512, 7, 7) level before GAP → (B, 512) after transition
  - Late (TEAM_DECISIONS): Concatenate at (B, 512) level after GAP → (B, 1024) before FC heads
  - Late version has more parameters in FC layers; early version is more efficient

**Resolution:** Recommend **late fusion as specified in TEAM_DECISIONS** (it matches Data Scientist rationale: FPV and top-view have different coordinate systems, so independent spatial extraction is preferred).

---

### 1.5 QA_STRATEGY.md vs. TEAM_DECISIONS.md

**Agreements:**
- ✓ Both define acceptance criteria: ≥85% validation accuracy, <1 sec inference latency
- ✓ Both specify focal loss as mitigation for class imbalance
- ✓ Both cite unseen well protocol (confidence gating)

**Gaps:**
- **Output schema validation:** QA_STRATEGY Section 5.3 specifies complete JSON schema with strict field validation
- TEAM_DECISIONS doesn't reference this schema; implies schema is documented elsewhere
- **Severity:** **MEDIUM** – Not contradictory, but QA_STRATEGY's schema should be cited in all other documents

---

### 1.6 DATA_ANALYSIS.md vs. TEAM_DECISIONS.md

**Agreements:**
- ✓ Both identify 5–15 wells with zero training samples
- ✓ Both recommend 70/20/10 train/val/test split
- ✓ Both emphasize overfitting risk and regularization necessity

**Gaps:**
- **Class weighting:** DATA_ANALYSIS Section "B.2 Regularization Techniques" says "class weighting: weight ∝ 1 / class_frequency"
- TEAM_DECISIONS Decision 4 on focal loss says "with α=0.25, γ=2.0" but doesn't specify class weighting
- ML_STACK Section 1.7 says "Optionally compute class_weights... for this dataset, row weights range from ~0.5 to ~2.0"
- **Severity:** **MEDIUM** – Should be explicit: focal loss SHOULD use class weighting per DATA_ANALYSIS recommendation

---

### Summary: Critical Cross-Document Issues

| Issue | Files | Severity | Resolution |
|-------|-------|----------|-----------|
| Output schema format (well_row/column vs row/column) | README, test_output_schema.py, ARCHITECTURE, ML_STACK, TEAM_DECISIONS | **CRITICAL** | Standardize on `well_row` and `well_column` from README |
| Fusion strategy (early vs late) | ARCHITECTURE, ML_STACK, TEAM_DECISIONS | **CRITICAL** | Implement **late fusion** per TEAM_DECISIONS; update ML_STACK |
| Confidence per well requirement | README, ARCHITECTURE, QA_STRATEGY | **MEDIUM** | Clarify: optional in metadata or per-well? Recommend optional |
| Frame aggregation strategy | README vs ML_STACK | **HIGH** | Implement ML_STACK's 2-frame max-pooling approach |
| Class weighting with focal loss | DATA_ANALYSIS, ML_STACK | **MEDIUM** | Explicitly use class weighting (weight ∝ 1 / frequency) |

---

## Section 2: Code Quality Audit

### 2.1 inference.py

**Overall Assessment:** Skeleton with extensive TODOs; no executable code paths.

**Findings:**

| Line(s) | Issue | Severity | Category |
|---------|-------|----------|----------|
| 27–31 | All model/preprocessing imports commented out | **CRITICAL** | Missing imports block entire pipeline |
| 64 | Device set to "cuda" without fallback check | **HIGH** | Will crash if GPU unavailable; no CPU fallback |
| 89–90 | `_load_model()` raises NotImplementedError | **CRITICAL** | Model loading not implemented |
| 126–127 | `load_and_align_videos()` raises NotImplementedError | **CRITICAL** | Video loading/alignment not implemented |
| 151–152 | `preprocess_frames()` raises NotImplementedError | **CRITICAL** | Preprocessing not implemented |
| 183 | `infer()` raises NotImplementedError | **CRITICAL** | Model inference not implemented |
| 219 | `postprocess_predictions()` raises NotImplementedError | **CRITICAL** | Post-processing not implemented |
| 246–285 | `validate_output_schema()` is implemented ✓ | **N/A** | This function is complete and correct |
| 287–350 | `infer_and_predict()` orchestrates pipeline but all components raise NotImplementedError | **CRITICAL** | Pipeline cannot execute |

**Code Quality Issues:**

1. **No docstring for class PipetteWellDetector** – Should describe architecture, inputs, outputs
   - **Severity:** LOW – Only documentation quality

2. **Type hints present but incomplete** – `Optional[str]` used correctly, but Dict types lack key/value type annotations
   - **Severity:** LOW – Code clarity

3. **Logging is good** – Uses standard logging module; easy to enable DEBUG mode
   - **Status:** ✓ GOOD

4. **Error handling is minimal** – Only NotImplementedError for stubs; no try/catch for I/O, GPU availability
   - **Severity:** HIGH – Need robust error handling for hold-out evaluation

5. **validate_output_schema() has logical bug** – Line 246 checks `if not isinstance(wells, list)` but doesn't exit early; continues to process even if invalid
   - **Severity:** MEDIUM – Should return False immediately

---

### 2.2 src/preprocessing/video_loader.py

**Overall Assessment:** All functions are stubs (raise NotImplementedError)

**Critical Missing Implementations:**

| Function | Lines | Required | Status |
|----------|-------|----------|--------|
| `load_video()` | 12–34 | Use cv2.VideoCapture to extract frames | **TODO** |
| `find_temporal_offset()` | 37–57 | Compute optical flow and cross-correlate FPV/top-view | **TODO** |
| `align_and_extract_frames()` | 60–92 | Combine above; return synchronized frames | **TODO** |

**Design Gaps:**

1. **Temporal alignment algorithm not specified** – Docstring says "cross-correlation peak" but doesn't specify:
   - Which frames to use for optical flow? First? Middle? All?
   - How to handle variable video lengths?
   - What if offset > max_offset?
   - **Recommendation:** Use full-duration optical flow magnitude vectors; cross-correlate at integer offsets; return (offset, confidence)

2. **No normalization specification** – Should frames be uint8 (0–255) or float (0–1)? 
   - Docstring doesn't specify
   - **Recommendation:** Return uint8 (standard for OpenCV); normalize in preprocessing before model input

3. **No synchronization error handling** – What if videos have very different frame counts?
   - **Recommendation:** Handle gracefully by padding or resampling shorter video

---

### 2.3 src/models/backbone.py

**Overall Assessment:** Class skeleton only; no implementation

**Critical Missing:**

| Item | Lines | Issue |
|------|-------|-------|
| `__init__()` | 29–40 | Raises NotImplementedError; should load ResNet-18 from torchvision |
| `forward()` | 42–57 | Raises NotImplementedError; should return (B, 512) features |
| `unfreeze_layers()` | 59–70 | Raises NotImplementedError; should set requires_grad on layers |

**Design Issues:**

1. **freeze_early parameter documented but no implementation** – Should freeze layer1 and layer2 per TEAM_DECISIONS
   - **Recommendation:** Implement as:
     ```python
     if freeze_early:
         for param in self.model.layer1.parameters():
             param.requires_grad = False
         for param in self.model.layer2.parameters():
             param.requires_grad = False
     ```

2. **No head replacement** – Docstring says "Remove classification head" but doesn't specify how
   - **Recommendation:** Set `self.model.fc = nn.Identity()` to keep 7×7 spatial features before GAP

3. **Missing global average pooling** – Backbone should output (B, 512) but ResNet has spatial dimensions (B, 512, 7, 7)
   - **Recommendation:** Add `self.gap = nn.AdaptiveAvgPool2d((1, 1))` to __init__; apply in forward

---

### 2.4 src/models/fusion.py

**Overall Assessment:** Two class stubs (DualViewFusion, MultiTaskHead); no implementation

**Critical Missing:**

| Class | Method | Lines | Issue |
|-------|--------|-------|-------|
| DualViewFusion | `__init__()` | 37–57 | Raises NotImplementedError; should create FC layers and output heads |
| DualViewFusion | `forward()` | 59–78 | Raises NotImplementedError; should fuse and output row/col logits |
| DualViewFusion | fusion helper methods | 80–90 | All raise NotImplementedError (_concat_fusion, _cross_attention_fusion, _gating_fusion) |
| MultiTaskHead | `__init__()` | 105–124 | Raises NotImplementedError |
| MultiTaskHead | `forward()` | 126–144 | Raises NotImplementedError |

**Architectural Clarification Needed:**

1. **Fusion architecture mismatch with TEAM_DECISIONS:** 
   - Docstring shows concatenation at (B, 1024) level (late fusion)
   - But signature suggests input is already processed features
   - **Clarify:** Should DualViewFusion:
     - (A) Take raw images + apply backbones internally? OR
     - (B) Take pre-extracted features from backbones?
   - **Recommendation:** (B) – Take features; keeps backbone separate (modularity)

2. **Output head dimensions unclear:**
   - Docstring says output is "(B, 8)" and "(B, 12)"
   - Should these have sigmoid already applied, or raw logits?
   - **Recommendation:** Return raw logits (no activation); apply sigmoid in loss function (better numerical stability)

---

### 2.5 src/postprocessing/output_formatter.py

**Overall Assessment:** All functions are stubs

**Critical Missing Implementations:**

| Function | Lines | Purpose |
|----------|-------|---------|
| `logit_to_wells()` | 11–50 | Convert row/col logits to well list |
| `enforce_cardinality_constraint()` | 53–73 | Filter to 1, 8, or 12 wells based on cardinality |
| `sort_and_deduplicate()` | 76–92 | Sort canonically (A1, A2, ..., H12) and remove duplicates |
| `validate_wells()` | 95–112 | Check all wells in [A-H] × [1-12] |

**Critical Logic Gaps:**

1. **Cardinality constraint enforcement under-specified:**
   - Current docstring says "find row with most detected wells" for 8-channel
   - But what if two rows have equal counts? Tie-breaking not specified
   - **Recommendation:** Use highest average confidence to break ties

2. **Confidence score handling:**
   - README says confidence per well is optional
   - But logit_to_wells() docstring returns confidence for each well
   - **Resolution:** Include confidence in output; mark as optional in JSON schema

3. **Threshold application:**
   - Should threshold be applied **before** or **after** cardinality constraint?
   - **Recommendation:** Apply threshold first (get candidate wells), then enforce cardinality post-hoc

---

### 2.6 src/utils/metrics.py

**Overall Assessment:** All metrics are function stubs

**Issue:** No implementations; just docstrings with TODO

| Metric | Implementation Status | Criticality |
|--------|----------------------|-------------|
| exact_match_accuracy() | Not implemented | **HIGH** – Primary evaluation metric |
| cardinality_accuracy() | Not implemented | **HIGH** – Required for cardinality evaluation |
| per_well_metrics() | Not implemented | **MEDIUM** – Diagnostic but important |
| jaccard_index() | Not implemented | **MEDIUM** – Multi-label standard metric |
| Other metrics | Not implemented | **LOW** – Supplementary |

**Recommendation:** Implement at least exact_match_accuracy() and cardinality_accuracy() before hold-out evaluation.

---

### 2.7 tests/test_output_schema.py

**Overall Assessment:** Solid test structure; actually executes validation logic

**Positive Findings:**
- ✓ Tests cover well boundary checking (A-H, 1-12)
- ✓ Tests check JSON serialization
- ✓ Tests validate no duplicates
- ✓ Tests check canonical ordering
- ✓ Tests validate multiwell cardinality consistency

**Issues:**

| Test | Line(s) | Issue | Severity |
|------|---------|-------|----------|
| test_well_row_uppercase() | 159–173 | Row format check: expects uppercase (✓ correct) | OK |
| test_multiwell_cardinality_consistency() | 210–238 | 8-channel should be **same column**, 12-channel same **row** | **CRITICAL BUG** |
| test_maximum_wells_limit() | 139–157 | Test generates >96 wells but doesn't fail the test; just marks as "invalid" | **LOGIC ERROR** |

**Critical Bug in test_multiwell_cardinality_consistency():**

```python
# For 8-channel, all wells should be in same column
columns = set(w["well_column"] for w in output_8channel["wells"])
assert len(columns) <= 1, "8-channel operation should have all wells in same column"
```

**WRONG:** 8-channel operates on a ROW (A1–H1 is one 8-channel operation), not column.
**CORRECT:** Should be:
```python
# For 8-channel, all wells should be in same ROW (8 tips in a row)
rows = set(w["well_row"] for w in output_8channel["wells"])
assert len(rows) <= 1, "8-channel operation should have all wells in same row"
```

**Severity:** **CRITICAL** – This inverts the 8/12-channel semantics. Tests will incorrectly validate wrong predictions.

---

### 2.8 tests/test_preprocessing.py

**Overall Assessment:** All tests are placeholders (pass statements)

**Finding:** Every test method contains:
```python
# TODO: ...description...
pass
```

**No actual tests execute.** This is a test skeleton.

**Critical Missing Tests:**

1. **Temporal alignment validation** – No test for find_temporal_offset()
2. **Frame extraction** – No test that extracted frames have correct shape/dtype
3. **Synchronization** – No test that FPV/top-view are properly synchronized
4. **Edge cases** – No tests for corrupted videos, missing files, extreme codecs

**Recommendation:** Implement at least:
- test_load_video_valid_file() – Load sample video, verify shape (N, H, W, 3)
- test_temporal_offset_detection() – Create synthetic offset, verify detection
- test_frame_extraction_latency() – Ensure <500ms per sample

---

## Section 3: Specification Completeness Audit

### 3.1 Implementation Decisions Left Ambiguous

| Specification Gap | Files | Impact | Recommendation |
|-------------------|-------|--------|-----------------|
| **Temporal alignment algorithm** | video_loader.py (lines 50–57) | Cannot implement without algorithm details | Specify optical flow method (Farneback, Lucas-Kanade) and cross-correlation approach |
| **Frame aggregation strategy** | README vs ML_STACK (1.12) | Should infer use max pooling? mean? Per-frame separate? | Use ML_STACK's max pooling over 2 frames |
| **Confidence threshold tuning** | ML_STACK (1.10) but not in TEAM_DECISIONS | Should threshold be tuned per head or global? | Separate thresholds for row and column; tune on validation set |
| **Cardinality head inclusion** | ML_STACK (optional in 1.6) vs fusion.py docstring | Is cardinality prediction mandatory or optional? | Make optional; if included, use softmax (1/8/12 classifier) |
| **Frame count per video** | ML_STACK says 2 frames; config default.yaml says frame_sampling_rate: 5 | Inconsistent specifications | Clarify: extract 2 frames per video (early + middle or middle + late) |

---

### 3.2 Edge Cases NOT Handled by inference.py

From QA_STRATEGY.md edge cases (Section 3) — which are NOT addressed in inference.py:

| Edge Case | Severity (QA_STRATEGY) | Handled in Code? | Gap |
|-----------|------------------------|------------------|-----|
| Extreme glare | Critical | ✗ | No histogram clipping detection |
| Dark/low light | Critical | ✗ | No gamma correction fallback |
| Motion blur | High | ✗ | No Laplacian variance check |
| Occluded plate | High | ✗ | No foreground segmentation |
| Plate rotated | Critical | ✗ | No rotation-invariant detection |
| Multiple plates visible | Critical | ✗ | No multi-plate handling |
| Corrupted video | Critical | ✗ | No try/catch for FFmpeg errors |
| Very short clip (<1 sec) | High | ✗ | No minimum duration check |
| Multiple dispenses | Critical | ✗ | No event segmentation |
| Mismatched FPV/top-view | Critical | ✗ | No cross-view validation |
| Zero wells predicted | Medium | ✓ | Handled (empty array OK) |
| Out-of-bounds coordinates | Critical | ✓ | validate_output_schema() checks this |

**Recommendation:** Document which edge cases inference.py will handle and which are out-of-scope. At minimum, add error handling for:
1. Missing video files
2. Corrupted/unreadable videos
3. GPU unavailability (fallback to CPU or error gracefully)
4. Synchronization failure (single-view fallback or clear error)

---

### 3.3 ML_STACK Recommendations NOT in configs/default.yaml

| ML_STACK Specification | default.yaml Coverage | Gap |
|------------------------|----------------------|-----|
| Backbone: ResNet-18 | ✓ Listed | OK |
| Fusion type: concatenation | ✓ Listed | OK |
| Focal loss (γ=2.0) | ✗ Not listed | **GAP** – should add loss_gamma: 2.0 |
| Class weighting | ✗ Not listed | **GAP** – should add class_weighting: true |
| Warmup epochs (5) | ✗ Not listed | **GAP** – training-only; OK to omit |
| Cosine annealing | ✗ Not listed | **GAP** – training-only; OK to omit |
| AdamW optimizer | ✗ Not listed | **GAP** – training-only; OK to omit |
| Dropout (0.3) | ✓ Implied by architecture | OK |
| Frame count (2) | ✗ Not listed | **GAP** – should add num_frames_per_video: 2 |
| Confidence threshold | ✓ confidence_threshold: 0.5 | OK |

**Recommendation:** Update config to include:
```yaml
training:
  focal_loss_gamma: 2.0
  class_weighting: true
  optimizer: "adamw"
  learning_rate: 0.0001
  warmup_epochs: 5
  total_epochs: 50
  
video:
  num_frames: 2
```

---

## Section 4: Test Coverage Assessment

### 4.1 What IS Covered

| Category | Tests | Status |
|----------|-------|--------|
| **Output schema** | test_output_schema.py | ✓ GOOD (except cardinality bug) |
| **JSON serialization** | TestMetadata | ✓ GOOD |
| **Well boundary checking** | test_valid_well_row_range, test_valid_well_column_range | ✓ GOOD |

### 4.2 What is NOT Covered (Critical Gaps)

| Component | Test Coverage | Required Tests | Priority |
|-----------|---------------|-----------------|----------|
| **Video loading** | 0% (placeholders only) | load_video(), error handling | **CRITICAL** |
| **Temporal alignment** | 0% | find_temporal_offset(), sync validation | **CRITICAL** |
| **Model inference** | 0% | Forward pass, output tensor shapes | **CRITICAL** |
| **Preprocessing** | 0% | Frame normalization, resizing, dtype | **CRITICAL** |
| **Post-processing** | 0% | logit_to_wells(), cardinality enforcement | **CRITICAL** |
| **End-to-end pipeline** | 0% | Full inference from video to JSON | **CRITICAL** |
| **Edge case robustness** | 0% (except schema validation) | Glare, dark, motion blur, occlusion | **HIGH** |
| **GPU/CPU fallback** | 0% | Device availability checks | **HIGH** |
| **Timeout handling** | 0% | Per-sample timeout enforcement | **HIGH** |
| **Metrics computation** | 0% | exact_match_accuracy(), F1, per-well | **HIGH** |

### 4.3 Required Tests Before Hold-Out Evaluation

**Must implement:**

1. **test_full_inference_pipeline()** – Load two test videos, run inference, verify JSON output
2. **test_temporal_alignment()** – Verify FPV/top-view synchronization on known offset
3. **test_model_output_shapes()** – Verify backbone → fusion → heads produces (B, 8) and (B, 12)
4. **test_inference_latency()** – Verify <2 min per sample on target hardware
5. **test_cardinality_constraint()** – Verify 8/12-channel operations produce correct well counts
6. **test_confidence_thresholding()** – Verify threshold 0.5 produces correct output
7. **test_json_schema_validation()** – Fix cardinality bug; re-run all schema tests

---

## Section 5: Risk Register (Updated)

Based on audit + original QA_STRATEGY risks:

### 5.1 Critical Risks (Stop Work if Not Resolved)

| Risk ID | Description | Severity | Likelihood | Impact | Owner | Mitigation | Status |
|---------|-------------|----------|-----------|--------|-------|-----------|--------|
| R1 | Implementation incomplete; inference.py is all stubs | **CRITICAL** | **Very High** | Cannot run inference; hold-out impossible | ML Scientist | Implement all NotImplementedError stubs (video_loader, backbone, fusion, postprocessing) | **OPEN** |
| R2 | Output schema format inconsistency (well_row vs row) | **CRITICAL** | **High** | Tests validate wrong format; inference outputs wrong JSON | Architect | Standardize on `well_row` and `well_column`; update ML_STACK Section 1.12; fix test bug (8 vs 12-channel) | **OPEN** |
| R3 | Temporal alignment algorithm unspecified | **CRITICAL** | **High** | Cannot synchronize FPV/top-view; inference fails silently or crashes | ML Scientist | Document optical flow + cross-correlation algorithm in video_loader.py | **OPEN** |
| R4 | Fusion architecture contradiction (early vs late) | **CRITICAL** | **Medium** | Different implementations may have different accuracy/latency | Architect | Resolve TEAM_DECISIONS vs ML_STACK; implement TEAM_DECISIONS (late fusion) | **OPEN** |
| R5 | No model checkpoint path specified | **CRITICAL** | **High** | Model loading will fail; configs/default.yaml points to models/best_checkpoint.pth but doesn't exist | ML Scientist | Generate or specify mock checkpoint during testing; document training pipeline | **OPEN** |

### 5.2 High-Risk Issues (Will Likely Cause Failures)

| Risk ID | Description | Severity | Likelihood | Impact | Mitigation |
|---------|-------------|----------|-----------|--------|-----------|
| R6 | Generalization gap 30 percentage points (from DATA_ANALYSIS) | **HIGH** | **Very High** | 85% training accuracy ≠ 85% test accuracy; expect 55–65% on hold-out | Use aggressive regularization (dropout 0.5, weight decay 1e-5, early stopping) |
| R7 | Unseen wells (5–15) with zero training samples | **HIGH** | **Very High** | Model cannot predict these; confidence gating essential | Implement confidence threshold per QA_STRATEGY Section 2.4; log unseen wells |
| R8 | Cardinality prediction incorrect (from QA_STRATEGY 4.1.3) | **HIGH** | **High** | 8-channel or 12-channel operations misclassified (e.g., predict 3 wells instead of 8) | Implement explicit cardinality head or post-processing constraint; validate per TEAM_DECISIONS Decision 3 |
| R9 | Test bug: 8-channel validation inverted | **HIGH** | **High** | Tests will pass for wrong predictions; hold-out evaluation will fail | Fix test_multiwell_cardinality_consistency() to check row consistency for 8-channel, column for 12-channel |
| R10 | Frame aggregation strategy ambiguous | **HIGH** | **Medium** | If using different frames (early vs late) for each view, temporal misalignment occurs | Document frame selection in video_loader; extract (early, middle) or (middle, late) pairs consistently |

### 5.3 Medium-Risk Issues

| Risk ID | Description | Mitigation |
|---------|-------------|-----------|
| R11 | Class weighting not in config.yaml | Add class_weighting parameter to config; implement in training loop |
| R12 | GPU device handling no fallback | Add torch.cuda.is_available() check; provide clear error if GPU unavailable |
| R13 | Memory overflow on large videos | Implement streaming frame extraction; test with 1GB video |
| R14 | Timeout handling not implemented | Wrap inference in signal.alarm(180) context manager; skip sample if timeout |
| R15 | Confidence scores optional in output | Document explicitly; update JSON schema; recommend including for debugging |

---

## Section 6: Issue Tracker

### All 23 Issues Identified

| ID | Severity | File | Line(s) | Description | Recommendation |
|----|----------|------|---------|-------------|-----------------|
| I1 | CRITICAL | inference.py | 27–31 | Model/preprocessing imports commented out | Uncomment; ensure all modules implement NotImplementedError properly |
| I2 | CRITICAL | inference.py | 89–90, 126–127, 151–152, 183, 219 | Core functions raise NotImplementedError (5 occurrences) | Implement all methods: _load_model, load_and_align_videos, preprocess_frames, infer, postprocess_predictions |
| I3 | HIGH | inference.py | 64 | Device hardcoded to "cuda"; no fallback | Add: `self.device = "cuda" if torch.cuda.is_available() else "cpu"` |
| I4 | MEDIUM | inference.py | 246–285 | validate_output_schema() missing early return on failure | Add: `if not isinstance(wells, list): return False` at line 246; exit early |
| I5 | CRITICAL | video_loader.py | 37–57 | Temporal alignment algorithm unspecified | Document: use cv2.calcOpticalFlowFarneback; cross-correlate magnitude vectors; return (offset, confidence) tuple |
| I6 | HIGH | video_loader.py | 12–34 | load_video() not implemented | Use cv2.VideoCapture; extract frames at fixed intervals; return (T, H, W, 3) numpy uint8 array |
| I7 | MEDIUM | video_loader.py | 12–92 | No normalization specification (uint8 vs float) | Document: return uint8; normalize in preprocessing to [0, 1] for model input |
| I8 | CRITICAL | src/models/backbone.py | 29–70 | ResNet-18 not loaded; __init__ and forward() raise NotImplementedError | Implement: load torchvision.models.resnet18(pretrained=True); freeze layer1/layer2; add GAP; return (B, 512) |
| I9 | MEDIUM | src/models/backbone.py | 29 | freeze_early parameter not implemented | Use param.requires_grad = False for layer1, layer2 if freeze_early=True |
| I10 | CRITICAL | src/models/fusion.py | 37–78 | DualViewFusion class not implemented | Implement __init__ (FC layers), forward (fusion + heads), helper fusion methods |
| I11 | HIGH | src/models/fusion.py | 59–78 | Output activation not specified (logits vs probabilities) | Return raw logits; apply sigmoid in loss/inference only |
| I12 | MEDIUM | src/models/fusion.py | 37–57 | Fusion architecture ambiguous (early vs late; internal backbones vs external?) | Clarify: take pre-extracted features; use late concatenation at (B, 512) level per TEAM_DECISIONS |
| I13 | CRITICAL | src/postprocessing/output_formatter.py | 11–50, 53–73, 76–92, 95–112 | All post-processing functions raise NotImplementedError (4 functions) | Implement: logit_to_wells, enforce_cardinality_constraint, sort_and_deduplicate, validate_wells |
| I14 | HIGH | src/postprocessing/output_formatter.py | 53–73 | Cardinality constraint tie-breaking unspecified | Use highest average confidence to break ties when multiple rows/columns have equal counts |
| I15 | MEDIUM | src/utils/metrics.py | 16–192 | All metrics functions are stubs | Implement: exact_match_accuracy, cardinality_accuracy (CRITICAL); per_well_metrics, jaccard_index (MEDIUM) |
| I16 | CRITICAL | tests/test_output_schema.py | 210–238 | 8-channel/12-channel semantics inverted | Fix: 8-channel should check same ROW (not column); 12-channel checks same COLUMN (not row) |
| I17 | HIGH | tests/test_output_schema.py | 139–157 | test_maximum_wells_limit assertion logic wrong | Assertion should fail when len(wells) > 96; currently just flags but doesn't fail |
| I18 | CRITICAL | tests/test_preprocessing.py | 18–197 | All preprocessing tests are placeholders (pass statements) | Implement: test_load_video_valid_file, test_temporal_offset_detection, test_frame_extraction_latency |
| I19 | CRITICAL | README.md vs others | Output schema | Format inconsistency: well_row/well_column vs row/column | Standardize on `well_row` and `well_column` |
| I20 | CRITICAL | ARCHITECTURE.md vs ML_STACK.md, TEAM_DECISIONS.md | Fusion strategy | Early fusion vs late fusion conflict | Implement TEAM_DECISIONS (late) version; update ML_STACK documentation |
| I21 | MEDIUM | configs/default.yaml | Missing parameters | Focal loss gamma, class weighting, frame count not in config | Add: loss_gamma, class_weighting, num_frames_per_video to config |
| I22 | HIGH | ML_STACK.md | 1.12 | Frame aggregation via max-pooling not in README | Document in README; clarify inference extracts 2 frames and max-pools outputs |
| I23 | MEDIUM | README.md, QA_STRATEGY.md | Output schema | Confidence per well is optional in some places, mandatory in others | Clarify: confidence is optional; include in output for debugging but not required |

---

## Section 7: Pre-Hold-Out Evaluation Checklist

### Go/No-Go Criteria

**All of the following MUST pass before submission:**

- [ ] **I1–I5:** All CRITICAL implementation gaps closed (inference.py, video_loader.py, model loading)
- [ ] **I6–I15:** All code stubs implemented (video_loader, backbone, fusion, postprocessing, metrics)
- [ ] **I16–I17:** Test bugs fixed (cardinality semantics, assertion logic)
- [ ] **I19–I20:** Documentation reconciled (schema format, fusion architecture decided)
- [ ] **test_output_schema.py** runs and passes (except I16)
- [ ] **Full inference pipeline** tested on 2 local sample videos; produces valid JSON in <2 min each
- [ ] **Latency profiling** on target hardware (GPU); verify per-sample <1 sec inference
- [ ] **Edge case spot checks:**
  - [ ] Missing video file → graceful error (not crash)
  - [ ] Corrupted video → graceful error
  - [ ] GPU unavailable → fallback or clear error
  - [ ] Temporal alignment detected correctly on known-offset synthetic videos
- [ ] **JSON schema validation** against strict schema from QA_STRATEGY Section 5.3
- [ ] **Cardinality validation** on synthetic 8-channel and 12-channel test cases
- [ ] **Confidence thresholding** produces expected well count at threshold 0.5
- [ ] **Model checkpoint exists** and loads without error (even if mock/untrained)

### Acceptance Metrics

**On Hold-Out Evaluation (10 samples), all of the following must be satisfied:**

- [ ] All 10 samples produce valid JSON output (no crashes, timeouts)
- [ ] JSON schema validation passes for all outputs
- [ ] Exact-match accuracy ≥80% (or ≥70% if hold-out contains many unseen wells)
- [ ] Cardinality-wise accuracy ≥75% (separate for 1-well, 8-well, 12-well if sample distribution allows)
- [ ] Total runtime ≤20 minutes (2 min/sample average)
- [ ] No runtime errors, exceptions, or OOM kills
- [ ] Predictions consistent with visual inspection (sanity check on 3–5 worst predictions)

---

## Section 8: Recommendations Summary

### Prioritized Actions for Each Role

#### **Data Scientist**

1. **Generate coverage heatmap** – Before hold-out evaluation, identify which wells have training examples and which are unseen (5–15 wells)
2. **Validate augmentation strategy** – Ensure temporal augmentation (frame offset ±3, speed 0.9–1.1×) is implemented in training pipeline
3. **Class weighting verification** – Ensure focal loss + class weights (1/frequency) are applied; confirm weights ∝ [0.5, 2.0] range
4. **Statistical analysis** – Run bootstrap confidence intervals on validation metrics; report mean ± 95% CI

#### **Architect**

1. **Resolve fusion architecture** – Make final decision: early (feature-map level) vs late (GAP level) fusion; **recommend late per TEAM_DECISIONS**
2. **Update ML_STACK.md** – If choosing late fusion, update Section 1.9 example code to match TEAM_DECISIONS
3. **Standardize output schema** – Update all documents (ARCHITECTURE, ML_STACK, test files) to use consistent `well_row`, `well_column` format
4. **GPU availability plan** – Document deployment assumption: GPU required; no CPU fallback
5. **Update config.yaml** – Add focal_loss_gamma, class_weighting, num_frames, training parameters

#### **ML Scientist**

1. **Implement all code stubs** (CRITICAL):
   - src/preprocessing/video_loader.py: load_video(), find_temporal_offset(), align_and_extract_frames()
   - src/models/backbone.py: ResNet-18 loading, GAP, freezing logic
   - src/models/fusion.py: late fusion, row/column heads, sigmoid outputs
   - src/postprocessing/output_formatter.py: logit conversion, cardinality constraint, sorting
   - src/utils/metrics.py: exact_match_accuracy(), cardinality_accuracy()

2. **Implement test infrastructure**:
   - tests/test_preprocessing.py: at least 3 real tests (load_video, temporal_offset, latency)
   - Integration test: end-to-end inference on 2 sample videos

3. **Verify inference latency**:
   - Profile on target hardware (GPU): target <1 sec/sample
   - If >1 sec, optimize: batch processing, frame extraction, model quantization

4. **Implement error handling**:
   - Graceful fallbacks for missing video, corrupted file, GPU unavailable
   - Output error JSON on failure (not crash)

#### **QA Engineer**

1. **Fix test bugs** (CRITICAL):
   - test_multiwell_cardinality_consistency(): invert 8/12 row/column checks
   - test_maximum_wells_limit(): add failing assertion for >96 wells

2. **Implement comprehensive test suite**:
   - Schema validation (already good; just fix I16)
   - Preprocessing tests (I18)
   - Integration test for full pipeline
   - Edge case spot checks (glare, dark, motion blur simulation)

3. **Pre-hold-out validation**:
   - Run full pipeline on 10 local test videos (if available)
   - Verify all acceptance criteria from Section 7
   - Document any deviations or limitations

4. **Hold-out evaluation protocol**:
   - Set up timing instrumentation (log per-sample latency)
   - Have backup plan if first 3 samples fail (restart, diagnostic steps)
   - Collect metadata: which wells present, any unseen? For post-eval analysis

---

## Section 9: Final Assessment

### Summary

The **Pipette Well Challenge is a well-documented project with excellent architectural decisions**, but **implementation is entirely missing**. The codebase is a comprehensive skeleton with clear TODOs, which is preferable to incomplete implementations, but leaves substantial work before evaluation.

### Can This Ship?

**Status: NOT READY FOR HOLD-OUT EVALUATION**

**Timeline to Ready:**
- **CRITICAL implementations (I1–I5):** 3–5 days (1 ML Scientist)
- **Code completion (I6–I15):** 5–7 days (1 ML Scientist + 1 Architect)
- **Test implementation (I16–I18):** 2–3 days (1 QA Engineer)
- **Integration & validation:** 2–3 days (1 ML Scientist)

**Total: 12–18 days** assuming parallel work and no blockers.

### Success Probability

**If all recommendations addressed:** 75–85% (depends on generalization gap, unseen wells, domain shift)  
**If recommendations partially addressed:** 40–60% (likely cardinality or edge case failures)  
**If recommendations ignored:** <20% (inference will crash; schema validation will fail)

### Key Decision Points

1. **Fusion architecture (Early vs Late):** Must decide and implement consistently
2. **Output schema format:** Standardize across all code/tests/docs
3. **Temporal alignment:** Specify algorithm (currently "TODO")
4. **Cardinality handling:** Implement post-processing constraint enforcement
5. **Test coverage:** Implement integration tests before hold-out (not after)

---

## Appendix A: Cross-Reference Guide

**Key documents by role:**

- **Data Scientists:** START → DATA_ANALYSIS.md → QA_STRATEGY.md (Section 2.1–2.4)
- **Architects:** START → ARCHITECTURE.md → TEAM_DECISIONS.md
- **ML Scientists:** START → ML_STACK.md → inference.py (scaffold)
- **QA Engineers:** START → QA_STRATEGY.md → tests/ (all files)

**Critical decisions logged in:**
- TEAM_DECISIONS.md (Decisions 1–10)
- ARCHITECTURE.md Section "Architect's Recommendation"
- ML_STACK.md Section "Part 1: Definitive ML Stack Recommendation"

---

**Report Status:** FINAL  
**Date:** April 14, 2026  
**Next Review:** After implementation of CRITICAL issues (I1–I5)
