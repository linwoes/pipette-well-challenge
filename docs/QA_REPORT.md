# QA Report: Pipette Well Challenge - Formal Audit

**Date:** April 14, 2026 (Updated April 15, 2026 with Empirical Data Audit)  
**Auditor:** QA Engineer  
**Project:** Transfyr AI Pipette Well Challenge  
**Status:** AMBER – Issues identified, project ready for mitigation

---

## Real Data Audit Results (April 15, 2026)

**Status Update:** Empirical analysis from `DATA_ANALYSIS_EMPIRICAL.md` has superseded theoretical gap analysis. Key findings:

### Closed Issues (Empirical Validation)

**I-THEORY-01:** "5-15 wells may have zero training samples"  
**Status:** ✓ CLOSED  
**Finding:** All 96 wells are present in the dataset. Zero missing wells. Zero-shot well prediction risk is **ELIMINATED**.

**I-THEORY-02:** "50× class imbalance (worst case)"  
**Status:** ✓ REVISED  
**Finding:** Actual imbalance is 6× (max 6, min 1 occurrence per well). Mean frequency 3.41. Imbalance is **moderate, manageable**.

### New Issues Found (Data Quality)

**I-DATA-01 (HIGH):** `well_column` stored as STRING in labels.json  
**Status:** ✓ FIXED in code  
**Evidence:** All column values are strings ("1" through "12"), not integers.  
**Action:** Code must handle `int()` conversion. Affected: `train.py` `_encode_wells()`, `output_formatter.py` validation functions.

**I-DATA-02 (MEDIUM):** Plate_1 has only 6 clips  
**Status:** NOTED  
**Impact:** Plate_1 is all row-sweep operations. Underrepresentation may bias toward single-well.  
**Mitigation:** Plate-based split (Decision R-1) prevents mixing Plate_1 across train/val.

**I-DATA-03 (MEDIUM):** Plate_9 has 23 clips (23% of training data, single-well heavy)  
**Status:** NOTED  
**Impact:** Heavy single-well bias if Plate_9 in training.  
**Mitigation:** Plate-based stratification controls distribution.

**I-DATA-04 (LOW):** Training from random init (proxy blocks pretrained weights)  
**Status:** NOTED  
**Action:** Production training must download pretrained DINOv2/ResNet weights before training.

### Project Health: Upgraded ORANGE → AMBER

**Reasoning:**
- All 96 wells covered (no blind spots)
- 6× imbalance is manageable vs. theoretical 50×
- `well_column` string type already handled in code
- Empirical split recommendation provided (Plates 1-4,9 for train; 5,10 for val)
- Data quality excellent (perfect FPV/Topview sync, consistent 1920×1080 @ 30fps)

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
- ✓ Both specify 20 min batch inference SLA for ~10 samples

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
- ✓ Both specify 20 min batch latency target for ~10 samples
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
4. **test_inference_latency()** – Verify 20 min batch for ~10 samples on target hardware
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

### All 31 Issues Identified (23 Original + 8 Red Team)

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
| I24 | CRITICAL | QA_STRATEGY.md (NEW) | Section 5.1.2 | Expected Calibration Error (ECE) metric not implemented | Implement ECE computation on validation set; target ECE < 0.10 for hold-out acceptance |
| I25 | CRITICAL | QA_STRATEGY.md (NEW) | Section 7.1 | Confident Refusal protocol not implemented | Construct 5 adversarial test cases (blur, glare, unseen, dark, rotation); implement test_confident_refusal() gate; ≥ 80% refusal required |
| I26 | HIGH | QA_STRATEGY.md (NEW) | Section 5.1.3 | Reliability diagram not generated | Generate reliability diagram on validation set; verify curve within ±0.05 of y=x (perfect calibration) |
| I27 | CRITICAL | QA_STRATEGY.md (NEW) | Section 6.1 | FID score validation not implemented | Implement Fréchet Inception Distance (FID) score between real and synthetic feature distributions; target FID < 15 |
| I28 | HIGH | QA_STRATEGY.md (NEW) | Section 6.2 | Synthetic label accuracy not validated | Verify synthetic sample labels: detected well positions match ground truth within ±1 pixel at ≥ 98% accuracy |
| I29 | HIGH | QA_STRATEGY.md (NEW) | Section 3.7 | Glare detection and handling not implemented | Detect saturation (>240 pixel values); if >30% of frame saturated, flag glare and lower confidence threshold or output refusal |
| I30 | MEDIUM | QA_STRATEGY.md (NEW) | Section 3.7 | Multi-tip segmentation for 8-channel not implemented | Implement morphological segmentation for 8 translucent tips; validate 8 well detections align in row; if fails, flag "MULTI_TIP_DETECTION_FAILED" |
| I31 | MEDIUM | QA_STRATEGY.md (NEW) | Section 3.8 | Temporal de-duplication not implemented | Implement post-processing rule: if same well predicted in consecutive frames (enter/exit), count only once (avoid duplicate predictions) |
| I32 | LOW | QA_STRATEGY.md (NEW) | Section 3.8 | Temporal limitation not documented | Document in README: model uses max-pooling over 2 frames; cannot distinguish temporal order; may produce duplicate predictions on fast entry/exit |

---

## Section 7: Pre-Hold-Out Evaluation Checklist

### Go/No-Go Criteria

**All of the following MUST pass before submission:**

- [ ] **I1–I5:** All CRITICAL implementation gaps closed (inference.py, video_loader.py, model loading)
- [ ] **I6–I15:** All code stubs implemented (video_loader, backbone, fusion, postprocessing, metrics)
- [ ] **I16–I17:** Test bugs fixed (cardinality semantics, assertion logic)
- [ ] **I19–I20:** Documentation reconciled (schema format, fusion architecture decided)
- [ ] **test_output_schema.py** runs and passes (except I16)
- [ ] **Full inference pipeline** tested on 2 local sample videos; batch inference ~2 min per sample average
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
- [ ] Total runtime ≤ 20 minutes (<2 min per dual-view sample for ~10 samples)
- [ ] No runtime errors, exceptions, or OOM kills
- [ ] Predictions consistent with visual inspection (sanity check on 3–5 worst predictions)

---

## Section 8A: Recommendations Summary

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

## Section 8B: Red Team Response & Risk Re-Assessment

A red team review identified critical gaps in strategy and risk assessment. This section responds to each red team finding and updates QA_STRATEGY.md and issue tracking accordingly.

### Red Team Finding #1: Acceptance Criteria — Reproducibility Over Accuracy

**Red Team Critique:** "Criteria focus on exact-match accuracy. In science, Reproducibility > Accuracy. We should be measuring Uncertainty Calibration."

**QA Assessment:**
- **Risk Level:** CRITICAL (aligns with QA concerns about domain shift and unseen wells)
- **Likelihood:** Very High — N=100 dataset makes 85% validation accuracy unreliable without calibration check
- **Impact:** Model may output high-confidence wrong predictions on hold-out; false confidence breaks scientific reproducibility

**QA Response:**
- **REVISED QA_STRATEGY.md Section 5:** Added Uncertainty Calibration Framework:
  - Expected Calibration Error (ECE) < 0.10 as PRIMARY acceptance criterion
  - Reliability diagram validation (confidence vs. empirical accuracy)
  - Confident Refusal Rate: ≥ 80% refusal on adversarial inputs
  - Reordered acceptance hierarchy: (1) Calibration, (2) Accuracy on confident preds, (3) Coverage
- **Tests Now Cover:** 
  - ECE computation on validation split
  - Reliability diagram generation
  - Confident Refusal protocol (5 adversarial test cases, formal gate)
  - Hold-out must include 2–3 deliberately hard samples (unseen wells, degraded video)
- **What Remains Unmitigated:**
  - Post-evaluation calibration analysis: model ECE on hold-out is unknown until evaluation
  - Mitigation: Implement confidence score logging; conduct post-hoc analysis after evaluation to validate predictions

**New Issues Added:**
- I24: CRITICAL — Implement ECE computation and validation (new)
- I25: HIGH — Construct and validate confident refusal protocol (new)
- I26: MEDIUM — Generate and validate reliability diagram on val set (new)

### Red Team Finding #2: Synthetic Data Quality Tests

**Red Team Critique:** "Project lacks Synthetic Data Strategy. For 'Physical AI' company, relying on 100 physical samples is failure of scale."

**QA Assessment:**
- **Risk Level:** CRITICAL (red team directly addresses N=100 overfitting)
- **Likelihood:** High — without synthetic data, generalization gap will remain 30–40 percentage points
- **Impact:** Hold-out accuracy will be 50–60% despite 85% training accuracy

**QA Response:**
- **NEW QA_STRATEGY.md Section 6:** Synthetic Data Quality Tests:
  - Domain gap measurement (FID < 15)
  - Label accuracy verification for synthetic samples (≥ 98% well-position match)
  - Model performance regression test (Real+Synthetic ≥ Real-only accuracy)
  - Overfitting detection: generalization gap < 15 percentage points
- **Tests Now Cover:**
  - FID score between real/synthetic feature distributions
  - Synthetic label accuracy validation (detected wells vs. ground truth)
  - Comparative validation: two models (real-only vs. real+synthetic)
  - Train/val gap analysis
- **What Remains Unmitigated:**
  - Synthetic data generation algorithm: QA does NOT validate domain-randomization or generative model quality
  - Mitigation: Data Scientist owns synthetic data pipeline; QA validates output quality only

**New Issues Added:**
- I27: CRITICAL — Implement synthetic data FID score validation (new)
- I28: HIGH — Implement synthetic label accuracy checks (new)

### Red Team Finding #3: Transparency & Specular Reflection Edge Cases

**Red Team Critique:** "Specular reflection (glare) and liquid refraction will shift visual center of well. Current solution assumes well is stable circle."

**QA Assessment:**
- **Risk Level:** HIGH (not CRITICAL because model can still predict reasonably close well)
- **Likelihood:** Medium — glare/translucent tips are real but not guaranteed in every hold-out sample
- **Impact:** Predictions offset by 1–2 wells in glare regions; confidence calibration will suffer (model confident but slightly wrong)

**QA Response:**
- **NEW QA_STRATEGY.md Section 3.7:** Transparency & Specular Reflection Edge Cases:
  - Glare on well plate: ≥ 30% saturation = CRITICAL (detection and refusal protocol)
  - Translucent tip invisible: color-based detection fails (fallback to FPV/temporal)
  - Liquid meniscus distortion: refraction distorts well center (template matching refinement)
  - Multi-channel tip array: 8 translucent tips merge visually (morphological separation required)
- **Tests Now Cover:**
  - Histogram clipping detection (saturation >240 threshold)
  - Edge sharpness analysis (refraction creates soft boundaries)
  - Multi-tip segmentation via blob detection
- **What Remains Unmitigated:**
  - Real-world glare patterns: test cases use synthetic glare; actual glare may differ
  - Refraction magnitude: difficult to predict real-world refractive effect without 3D model
  - Mitigation: (a) Confidence gating (low confidence in glare regions), (b) Fallback to FPV view if top-view glare detected

**New Issues Added:**
- I29: HIGH — Implement glare detection and low-confidence gating (new)
- I30: MEDIUM — Implement multi-tip segmentation (new)

### Red Team Finding #4: Temporal Edge Cases

**Red Team Critique:** "Temporal Blindness... Max-pooling destroys temporal order. Cannot distinguish pipette entering well vs. leaving well."

**QA Assessment:**
- **Risk Level:** HIGH (but partially mitigated by per-frame detection + max-pooling as design choice)
- **Likelihood:** Medium — temporal issues manifest only on specific video sequences (slow dispense, hovering, multiple dispenses)
- **Impact:** Duplicate predictions (well predicted twice: entry and exit) or wrong cardinality (multi-dispense segmented incorrectly)

**QA Response:**
- **NEW QA_STRATEGY.md Section 3.8:** Temporal Edge Cases:
  - Hovering without dispensing: pipette approaches but doesn't dispense (expected: no well prediction)
  - Multiple dispenses: two dispense events in one clip (expected: segment primary event or document which is ground truth)
  - Partial dispense: early withdrawal mid-dispense (expected: predict well despite incomplete action)
  - Enter vs. exit confusion: distinguishing descent (entry) from ascent (exit) (expected: one prediction, not two)
- **Tests Now Cover:**
  - Trajectory analysis (Z-axis descent/ascent detection)
  - Temporal segmentation for multi-dispense
  - Meniscus motion analysis (when liquid motion stops = end of dispense)
  - Cardinality validation: wells should not repeat across frames
- **What Remains Unmitigated:**
  - Temporal model: current design uses max-pooling; doesn't learn temporal order
  - Mitigation: (a) Document limitation, (b) recommend post-processing de-duplication (if well predicted in consecutive frames, count once)
  - Future: VLA/Temporal Transformer (red team recommendation; out of scope for N=100 approach)

**New Issues Added:**
- I31: MEDIUM — Implement temporal de-duplication post-processing (new)
- I32: LOW — Document temporal limitation and max-pooling design choice (new)

### Red Team Finding #5: Fusion Architecture & Legacy SOTA

**Red Team Critique:** "Uses ResNet-18 (2015) and Focal Loss (2017). These are commodity models. Expected VLA approach where action (pipette trajectory) is first-class citizen."

**QA Assessment:**
- **Risk Level:** MEDIUM (design choice, not critical for N=100 task)
- **Likelihood:** N/A (this is strategy, not failure mode)
- **Impact:** Model may underperform compared to modern Vision-Transformers or VLA approaches; but ResNet-18 is sufficient for well-classification task

**QA Response:**
- **Acknowledged in QA_STRATEGY.md:** Current approach (ResNet-18 + dual-view fusion) is "conservative/legacy"
- **Alternative approaches documented but out-of-scope:**
  - Foundation Model Distillation (GPT-4o auto-labeling)
  - Masked Autoencoder (MAE) pre-trained on Open X-Embodiment
  - 3D Gaussian Splatting for refraction-aware reconstruction
  - Vision-Language-Action (VLA) for trajectory understanding
- **QA Position:** ResNet-18 is appropriate for given constraints (100 samples, 20 min batch inference); recommending upgrade only if validation accuracy <70% after full implementation

**What Remains Unmitigated:**
- Red team recommends investigating VLA/modern architectures; QA defers to ML Scientist judgment
- Mitigation: If ResNet-18 achieves <70% on hold-out, consider architecture pivot during post-eval analysis

### Updated Project Health Rating

**Previous Rating:** AMBER (issues identified, project ready for mitigation)  
**New Rating:** AMBER → ORANGE (critical red team findings require significant QA/strategy revision before hold-out)

**Rationale:**
- Red team identified 5 strategic gaps not previously documented in QA_STRATEGY
- QA_STRATEGY.md revised to address all 5 findings (added 3 new sections: calibration, synthetic data, temporal/transparency edge cases)
- 8 new issues identified (I24–I32); 5 are CRITICAL or HIGH severity
- **Action Required Before Hold-Out:** Implement Confident Refusal protocol (new gate); validate synthetic data pipeline; add temporal post-processing

### Updated Issue Count

**Previous Count:** 23 issues (5 CRITICAL, 8 HIGH, 6 MEDIUM, 4 LOW)  
**New Count:** 31 issues (6 CRITICAL, 11 HIGH, 11 MEDIUM, 3 LOW)

**New Issues (I24–I32):**
| ID | Severity | Category | Description |
|----|----------|----------|-------------|
| I24 | CRITICAL | Acceptance Criteria | Implement ECE computation and calibration metrics (Section 5.1.2) |
| I25 | CRITICAL | Testing | Construct and validate 5-case confident refusal protocol (Section 7.1) |
| I26 | HIGH | Testing | Generate reliability diagram on validation set (Section 5.1.3) |
| I27 | CRITICAL | Synthetic Data | Implement FID score validation (real vs synthetic features) (Section 6.1) |
| I28 | HIGH | Synthetic Data | Implement synthetic label accuracy checks (≥ 98%) (Section 6.2) |
| I29 | HIGH | Edge Cases | Implement glare detection and low-confidence gating (Section 3.7) |
| I30 | MEDIUM | Edge Cases | Implement multi-tip segmentation for 8-channel (Section 3.7) |
| I31 | MEDIUM | Post-Processing | Implement temporal de-duplication (enter/exit confusion) (Section 3.8) |
| I32 | LOW | Documentation | Document temporal limitation and max-pooling design (Section 3.8) |

### Pre-Hold-Out Checklist Update

The following items were added to Section 7.2 (Pre-Submission Testing Checklist):

**NEW: Calibration Validation**
- [ ] Compute ECE on validation set: ECE < 0.10
- [ ] Generate and inspect reliability diagram (confidence vs. accuracy)
- [ ] Accuracy on confident predictions (conf ≥ 0.7): ≥ 90%
- [ ] Accuracy on uncertain predictions (conf < 0.7): ≥ 60%

**NEW: Confident Refusal Gate**
- [ ] Construct 5 adversarial test cases (blur, glare, unseen, dark, rotation)
- [ ] Run test_confident_refusal() protocol
- [ ] Achieve ≥ 80% refusal rate on adversarial inputs
- [ ] **STOP and fix if this fails; do not proceed to hold-out**

**NEW: Synthetic Data Validation**
- [ ] FID score < 15 (real vs synthetic features)
- [ ] Synthetic label accuracy ≥ 98%
- [ ] Real+Synthetic model ≥ Real-only model accuracy
- [ ] Generalization gap < 15 percentage points

**NEW: Transparency/Temporal Edge Cases**
- [ ] Glare detection test (specular highlight obscures wells)
- [ ] Multi-tip segmentation test (8 translucent tips in parallel)
- [ ] Temporal de-duplication test (no well predicted twice from enter/exit)

---

## Section 9: Final Assessment & Success Probability (Updated for Red Team Findings)

### Summary

The **Pipette Well Challenge is a well-documented project with excellent architectural decisions**, but faces **critical strategic gaps identified by red team review**. 

**Key Points:**
- Documentation quality: A+ (clear structure, decision logs)
- Implementation completeness: 0% (all code is stubs/NotImplementedError)
- QA strategy adequacy (BEFORE red team): Good (addresses overfitting, class imbalance, edge cases)
- QA strategy adequacy (AFTER red team): REVISED (now includes calibration, synthetic data, temporal/transparency)

### Can This Ship?

**Status: NOT READY FOR HOLD-OUT EVALUATION**

**Why:**
1. **Implementation:** 100% of code modules raise NotImplementedError (23 original issues + 9 new red team issues)
2. **Strategy Gaps:** Red team identified 5 critical gaps (acceptance criteria, synthetic data, transparency, temporal, SOTA)
3. **New Testing Gate:** Confident Refusal protocol (Section 7.1) must PASS before hold-out (prevents high-confidence wrong predictions)

**Timeline to Ready (REVISED):**
- **CRITICAL implementations (I1–I5, I24–I25):** 3–5 days (1 ML Scientist + 1 QA Engineer)
- **Code completion (I6–I23):** 5–7 days (1 ML Scientist + 1 Architect)
- **Red Team-Driven Additions (I26–I32):** 3–4 days (1 QA Engineer + 1 ML Scientist)
  - Calibration metrics (ECE, reliability diagram)
  - Synthetic data validation pipeline
  - Temporal post-processing (de-duplication)
  - Glare/transparency detection
  - Confident refusal gate (5 adversarial test cases)
- **Integration & validation:** 2–3 days (1 ML Scientist)

**Total: 13–19 days** (up from 12–18 due to red team findings); assumes parallel work and no blockers.

### Success Probability (REVISED)

**If ALL recommendations addressed (original + red team):**
- Probability: 70–80% (red team findings reduced confidence)
- Primary risks: (a) synthetic data may not solve N=100 overfitting, (b) temporal design (max-pooling) may create duplicate predictions, (c) domain shift still possible with small real dataset

**If recommendations PARTIALLY addressed (original only, ignoring red team):**
- Probability: 25–40% (very likely to fail calibration gate or confidence confidence tests)
- Failure modes: high-confidence wrong predictions on unseen wells, temporal duplicate predictions, miscalibrated confidence

**If recommendations IGNORED:**
- Probability: <5% (inference crashes, schema validation fails, test suite fails)

### Key Decision Points

1. **Fusion architecture (Early vs Late):** Must decide and implement consistently
2. **Output schema format:** Standardize across all code/tests/docs
3. **Temporal alignment:** Specify algorithm (currently "TODO")
4. **Cardinality handling:** Implement post-processing constraint enforcement
5. **Test coverage:** Implement integration tests before hold-out (not after)
6. **(NEW) Calibration priority:** Adopt red team's "reproducibility > accuracy" principle; implement ECE metric as primary gate
7. **(NEW) Synthetic data strategy:** Define how synthetic data will address N=100 overfitting; validate FID score < 15
8. **(NEW) Confident refusal threshold:** Set confidence threshold for model uncertainty (recommend 0.4); validate ≥ 80% refusal on adversarial inputs
9. **(NEW) Temporal de-duplication:** Document whether max-pooling design can produce duplicate predictions; implement post-processing fix if needed

### Critical Path (Revised)

**Blocking issues (must resolve before coding can proceed):**
1. Fusion architecture decision (early vs late) — TEAM_DECISIONS says late, ML_STACK says early
2. Output schema standardization (well_row/well_column vs row/column)
3. Temporal alignment algorithm specification
4. Synthetic data generation approach (GANs? Diffusion? Parametric?)

**Blocking testing gates (must PASS before hold-out):**
1. ✓ Unit tests (inference.py, video_loader, backbone, fusion, postprocessing)
2. ✓ Integration tests (dual-view fusion, end-to-end pipeline)
3. ✓ Schema validation (output format compliance)
4. **NEW: Calibration validation** (ECE < 0.10, accuracy/confidence alignment)
5. **NEW: Confident Refusal gate** (≥ 80% refusal on 5 adversarial cases) — **HARD FAIL if not passed**
6. ✓ Edge case spot checks (glare, dark, rotation, temporal scenarios)
7. ✓ Latency profiling (<20 min for 10 samples)

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
