# Feature Request: Scene Object Classification for Dual-View Pipette Analysis

**Document:** Feature Request & Implementation Specification  
**Date:** 2026-04-16  
**Role:** ML Product Team / Engineering  
**Audience:** ML Scientists, Data Scientists, Software Engineers, QA Engineers, Stakeholders  
**Status:** DRAFT FOR REVIEW  
**Project:** Transfyr AI Pipette Well Challenge

---

## Executive Summary

The Transfyr Pipette Well Challenge currently predicts *which wells* a pipette dispenses into using synchronized dual-view video (FPV + top-view) and a DINOv2-ViT-B/14 + LoRA + Temporal Transformer + Late Fusion architecture. This is a *spatial localization task*: the model answers **"Where is the tip?"**

This feature request proposes adding **scene object classification** to answer a complementary question: **"What objects are present and in what state?"** The system should classify and characterize important objects visible in the frames—wells, pipette tips, operator hands, liquid, plate components, and more—to create rich semantic annotations of laboratory scenes.

This enriched data is valuable for multiple downstream consumers: other Transfyr teams may use scene classification for task understanding, data scientists can mine patterns about operator behavior and technique, QA teams can flag contamination risks or procedural deviations, future models can leverage scene context for improved robustness, and internal analytics can build dashboards showing lab efficiency metrics.

Scene classification transforms the current system from a single-task spatial detector into a multi-modal scene understanding system. This aligns with Transfyr's mission of capturing tacit knowledge embedded in expert lab operations and making it accessible, analyzable, and actionable.

---

## Motivation & Business Value

### Why Scene Classification Matters Beyond Well Detection

Well coordinate prediction is essential but incomplete. A robust physical AI system for lab automation requires understanding not just *where* operations occur, but also *how* they occur and *what context* surrounds them. Scene classification provides this context.

**Gaps in Current System:**
- Well detection alone cannot distinguish between a successful dispense and a failed attempt
- No visibility into equipment state (is the pipette tip intact? clean?)
- Operator behavior remains opaque—was the hand in the frame? How tight was the grip?
- Liquid dynamics unseen—was liquid transferred? Did a droplet splash?
- Quality signals hidden—are there signs of contamination, condensation, or misalignment?

**Scene classification bridges these gaps.** By tagging objects and their states across frames, downstream consumers gain interpretable signals that individual well coordinates cannot provide.

### Who Consumes This Data

1. **Downstream Transfyr Teams**
   - **Task Understanding & Analytics:** Correlate scene objects with task success/failure rates. Example: "Dispenses with visible condensation have 15% higher error rates."
   - **Operator Profiling:** Identify expert vs. novice technique by analyzing hand position, grip, and timing relative to scene state.
   - **Equipment Monitoring:** Flag worn pipette tips, contaminated plates, or misaligned racks via automated scene analysis.

2. **Data Scientists & Researchers**
   - Train models on curated subsets: e.g., "all scenes with liquid visible" for liquid dynamics research
   - Analyze failure modes: "which object combinations lead to mispredictions?"
   - Build unsupervised clustering of procedural variants across operators/labs

3. **QA & Compliance Teams**
   - **Protocol Validation:** Automated checks that standard procedures are followed (e.g., "was glove present?" "was plate lid removed before dispense?")
   - **Contamination Detection:** Flag scenes with thumb over well or spilled liquid
   - **Compliance Audits:** Demonstrate that lab operations followed documented procedures

4. **Future Model Development**
   - **Multi-task Learning:** Scene classification as auxiliary task improves well detection robustness via shared representations
   - **Context-Aware Prediction:** Models can condition well prediction on scene state ("given pipette tip is visible and liquid-filled, predict dispense accuracy")
   - **Domain Adaptation:** Scene features generalize better than raw pixels across different labs, lighting, equipment

5. **Analytics & Dashboards**
   - Summarize operational patterns: "80% of dispenses in Lab A use single-channel tips, 90% in Lab B use multi-channel"
   - Track equipment health: "Pipette brand X retired after scene analysis showed >5% tip breakage rate"
   - Operator metrics: "Operator C achieves 98% success rate with gloved hands, 94% ungloved"

### Strategic Alignment with Transfyr's Physical AI Mission

Transfyr's core mission is to **capture tacit knowledge from expert physical operations**. Well coordinate prediction captures *where* the expert acts; scene classification captures *what the expert perceives and how the environment responds*. Together, they form a richer model of expertise.

Scene classification enables:
- **Knowledge Transfer:** Trained models + interpretable scene tags allow knowledge transfer to robotic systems ("move tip to this region, only when scene contains empty well without condensation")
- **Continuous Learning:** Scene annotations create feedback loops for domain adaptation as labs introduce new equipment or procedures
- **Transparent Operations:** Stakeholders can audit whether systems are "seeing" the lab the way humans do
- **Generalization:** Scene-based features generalize better than pixel-based features to new lab setups, lighting conditions, and equipment brands

---

## Object Taxonomy

Successful scene classification requires a carefully curated taxonomy of objects that matter for pipette dispense prediction and lab automation. The taxonomy should balance comprehensiveness (capture important objects) with practicality (avoidable explosion of labels).

### Design Principles

1. **Physical Significance:** Objects included because they carry information about task success, equipment state, operator technique, or safety risks
2. **Detection Feasibility:** Objects should be visible in at least one view (FPV or top-view) and distinguishable with computer vision
3. **Annotation Efficiency:** Avoid micro-categories that are rarely observed or hard to distinguish; merge similar classes
4. **Future-Proofing:** Structure allows adding new objects/attributes without schema redesign

### Object Categories

#### 1. **Well / Well Plate**

**Description:** The 96-well plate itself and individual wells.

**Attributes:**
| Attribute | Type | Values | Notes |
|-----------|------|--------|-------|
| Presence | Binary | present / absent | Always present during dispense; relevant for scenes with partial plate visibility |
| Well State | Multi-class | empty, liquid_filled, partially_filled, foam, precipitate, covered_by_tip, air_bubble | Per-well classification (high cardinality; may group into simplified classes) |
| Plate Format | Categorical | 96_well, 384_well, 1536_well | Currently always 96; future-proofs for format generalization |
| Plate Orientation | Categorical | standard, tilted, rotated | Detect if plate not level; affects well prediction confidence |
| Plate Boundary Visible | Binary | yes / no | Whether edge of plate visible in frame; useful for localization |
| Barcode / Label Visible | Binary | yes / no | Whether plate ID barcode or sample label visible |

**Classification Type:** Multi-label per well (hard to scale); simpler approach: binary presence + coarse state (empty / non-empty / partially_filled)

**Priority:** P0 (critical for dispense understanding)

**Best View:** Top-view (orthographic provides clear well array visualization)

---

#### 2. **Pipette Tip**

**Description:** The disposable tip inserted into pipette barrel; critical indicator of dispense capability.

**Attributes:**
| Attribute | Type | Values | Notes |
|-----------|------|--------|-------|
| Presence | Binary | present / absent | If absent, no dispense possible; aids in failure diagnostics |
| Channel Type | Categorical | single_channel, multi_channel_8, multi_channel_12, multi_channel_96 | Determines max parallelism |
| Tip State | Categorical | clean, contaminated, cracked, liquid_filled, empty, detached | Liquid-filled indicates readiness; cracked/detached indicate failure |
| Liquid Presence | Binary | yes / no | Is tip visibly filled with liquid (meniscus, droplet)? |
| Tip Position | Categorical | in_pipette, approaching_well, in_well, ejected, on_surface | Temporal progression; aids event detection |

**Classification Type:** Object detection (bounding box) + multi-label state attributes

**Priority:** P0 (essential for dispense verification)

**Best View:** FPV (tip visible in first-person video; harder in orthographic top-view due to parallax)

---

#### 3. **Liquid**

**Description:** Visible liquid in scene—in pipette tip, on well surface, airborne, or spilled.

**Attributes:**
| Attribute | Type | Values | Notes |
|-----------|------|--------|-------|
| Presence | Binary | yes / no | Is any liquid visible? |
| Location | Multi-class | in_tip, in_well, on_surface, airborne, spill | Where is liquid? |
| Volume Estimate | Categorical | trace, small, medium, large | Visual estimate of amount |
| Color | Categorical | clear, yellow, red, blue, green, other | Reagent type indicator |
| Meniscus Visible | Binary | yes / no | Shows if tip is sealed and liquid is contained |

**Classification Type:** Semantic segmentation (pixel-level); alternately, bounding box detection

**Priority:** P1 (high value for understanding dispense accuracy; harder to annotate reliably)

**Best View:** FPV (lighting often better for fluid visibility; top-view obscured by plate wells)

---

#### 4. **Operator Hand / Thumb**

**Description:** Visible human hand or fingers in frame, indicating operator presence and grip/positioning.

**Attributes:**
| Attribute | Type | Values | Notes |
|-----------|------|--------|-------|
| Presence | Binary | yes / no | Is any hand/fingers visible? |
| Body Part | Categorical | thumb, fingers, palm, wrist, full_hand | Which parts visible? |
| Glove State | Categorical | bare_hand, latex_glove, nitrile_glove, cloth_glove | Safety / contamination indicator |
| Grip Position | Categorical | above_well, over_well, supporting_plate, steering | Relative to plate/well |
| Hand Proximity to Well | Categorical | distant (>5cm), near (1-5cm), touching | Contamination risk |

**Classification Type:** Object detection (hand bounding box) + pose/state attributes

**Priority:** P1 (procedural compliance; operator behavior analysis)

**Best View:** FPV (hands most visible in first-person perspective)

---

#### 5. **Pipette Barrel / Body**

**Description:** The main pipette instrument body (not the tip).

**Attributes:**
| Attribute | Type | Values | Notes |
|-----------|------|--------|-------|
| Presence | Binary | yes / no | Is pipette body in frame? |
| Pipette Type | Categorical | single_channel, 8_channel, 12_channel, 96_channel | Indicates parallelism capability |
| Brand | Categorical | gilson, eppendorf, rainin, tecan, other | Different brands have different handling; useful for failure analysis |
| Plunger State | Categorical | rest, pressed, halfway, released | Indicates if aspiration/dispense in progress |

**Classification Type:** Object detection + categorical attributes

**Priority:** P2 (contextual; useful for equipment tracking but not critical for task)

**Best View:** FPV (barrel mostly off-screen in top-view during dispense)

---

#### 6. **Plate Lid / Cover**

**Description:** Plate sealing cover (if applicable).

**Attributes:**
| Attribute | Type | Values | Notes |
|-----------|------|--------|-------|
| Presence | Binary | yes / no | Is plate covered? |
| Lid State | Categorical | closed, partially_open, fully_open, removed, fallen_off | Procedural step indicator |
| Lid Material | Categorical | foil, plastic_film, hard_plastic_cover, other | Different materials have different visibility properties |

**Classification Type:** Object detection + state

**Priority:** P2 (protocol compliance; lid must be removed before dispense)

**Best View:** Top-view (lid visible when closed; sides visible in FPV when opened)

---

#### 7. **Gloves**

**Description:** Operator's gloves (separate from hand, for explicit tracking).

**Attributes:**
| Attribute | Type | Values | Notes |
|-----------|------|--------|-------|
| Presence | Binary | yes / no | Are gloves worn? |
| Glove Type | Categorical | latex, nitrile, cloth, vinyl, other | Material matters for contamination risk and tactile feedback |
| Glove Condition | Categorical | clean, dusty, contaminated, torn | Integrity indicator |

**Classification Type:** Binary presence + categorical attributes

**Priority:** P1 (compliance; safety and contamination control)

**Best View:** FPV (gloves visible when hands are in frame)

---

#### 8. **Rack / Well Plate Holder**

**Description:** The mechanical rack or fixture holding the plate.

**Attributes:**
| Attribute | Type | Values | Notes |
|-----------|------|--------|-------|
| Presence | Binary | yes / no | Is rack visible? |
| Rack Type | Categorical | fixed_bench, motorized_carousel, magnetic_block, tube_holder, other | Indicates automation level; affects procedural constraints |
| Plate Stability | Categorical | stable, tilted, moving, fallen | Safety and accuracy indicator |

**Classification Type:** Object detection

**Priority:** P2 (contextual; supports equipment profiling)

**Best View:** Top-view (full rack visible in orthographic view)

---

#### 9. **Reagent Bottles / Tubes Nearby**

**Description:** Containers holding reagents or samples near the plate.

**Attributes:**
| Attribute | Type | Values | Notes |
|-----------|------|--------|-------|
| Presence | Binary | yes / no | Are bottles/tubes in frame? |
| Container Type | Categorical | reagent_bottle, test_tube, 96_well_source_plate, reservoir, other | Source/destination plate type |
| Fill Level | Categorical | full, three_quarter, half, quarter, nearly_empty | Inventory indicator |
| Lid State | Categorical | capped, open, loosely_capped | Contamination / spill risk |

**Classification Type:** Object detection + attributes

**Priority:** P2 (context for multi-step workflows)

**Best View:** Top-view (full scene includes surrounding workspace)

---

#### 10. **Lab Bench Surface**

**Description:** The work surface containing plates and equipment.

**Attributes:**
| Attribute | Type | Values | Notes |
|-----------|------|--------|-------|
| Presence | Binary | yes / no | Is bench visible? |
| Surface Material | Categorical | plastic_laminate, stainless_steel, wooden, tile, other | Affects cleanup/contamination |
| Surface Cleanliness | Categorical | clean, dusty, wet, stained, cluttered | Hygiene indicator |
| Spill Detection | Binary | yes / no | Is there visible liquid spill on bench? |

**Classification Type:** Semantic segmentation of bench area, or binary presence

**Priority:** P3 (nice-to-have for safety/QA dashboards)

**Best View:** Top-view (full bench layout visible)

---

#### 11. **Shadows & Reflections (Negative Class)**

**Description:** Visual artifacts that may be mistaken for objects but are not physical entities.

**Attributes:**
| Attribute | Type | Values | Notes |
|-----------|------|--------|-------|
| Shadow Presence | Binary | yes / no | Are shadows visible that might confuse tip detection? |
| Reflection Presence | Binary | yes / no | Are specular reflections present (glare on wells, tips)? |
| Glare Intensity | Categorical | none, mild, moderate, severe | Optical interference level |

**Classification Type:** Binary presence; helps models avoid false positives

**Priority:** P2 (robustness; especially important given specular polystyrene wells)

**Best View:** Both (shadows in FPV and top-view; reflections context-dependent)

---

#### 12. **Ejection Mechanism / Tip Ejector**

**Description:** Visible components of the tip ejection system.

**Attributes:**
| Attribute | Type | Values | Notes |
|-----------|------|--------|-------|
| Presence | Binary | yes / no | Is ejector mechanism visible? |
| Ejector State | Categorical | retracted, engaged, ejecting, tip_released | Indicates phase of ejection |

**Classification Type:** Categorical state detection

**Priority:** P3 (temporal marker; useful for event synchronization)

**Best View:** FPV (mechanical movement visible in first-person view)

---

### Taxonomy Summary & Priorities

| Object | P0 (Critical) | P1 (High) | P2 (Medium) | P3 (Nice) | Primary View |
|--------|:---:|:---:|:---:|:---:|---|
| Well / Plate | ✓ | | | | Top-view |
| Pipette Tip | ✓ | | | | FPV |
| Liquid | | ✓ | | | FPV |
| Hand / Operator | | ✓ | | | FPV |
| Pipette Barrel | | | ✓ | | FPV |
| Plate Lid | | ✓ | | | Top-view |
| Gloves | | ✓ | | | FPV |
| Rack / Holder | | | ✓ | | Top-view |
| Reagent Bottles | | | ✓ | | Top-view |
| Bench Surface | | | ✓ | | Top-view |
| Shadows / Reflections | | | ✓ | | Both |
| Ejector Mechanism | | | | ✓ | FPV |

### Phased Rollout Strategy

**Phase 1 (MVP):** P0 + P1 objects (wells, tips, liquid, hands, gloves, lids). Enables immediate value for QA and operator profiling.

**Phase 2 (Extended):** Add P2 objects (barrel, rack, bottles, bench, shadows). Completes equipment profiling and robustness signals.

**Phase 3 (Future):** P3 objects (ejector) and advanced attributes (hand pose estimation, liquid volume quantification via physics simulation).

---

## Data Schema

Scene classification annotations must integrate seamlessly with the existing well prediction output while remaining extensible for future object types and attributes.

### JSON Schema Structure

The current `inference.py` output format (from `output_formatter.py`):

```json
{
  "clip_id_FPV": "fpv_001",
  "clip_id_Topview": "topview_001",
  "wells_prediction": [
    {"well_row": "A", "well_column": 1},
    {"well_row": "A", "well_column": 2}
  ],
  "metadata": {
    "model": "DINOv2-ViT-B/14+LoRA",
    "inference_time_s": 0.350,
    "confident": true
  }
}
```

**Proposed extended format (backward-compatible):**

```json
{
  "clip_id_FPV": "fpv_001",
  "clip_id_Topview": "topview_001",
  "wells_prediction": [
    {"well_row": "A", "well_column": 1},
    {"well_row": "A", "well_column": 2}
  ],
  "scene_classification": {
    "timestamp_frame_index": 4,
    "frame_timestamp_s": 0.133,
    "detected_objects": [
      {
        "object_id": "well_A1",
        "object_type": "well",
        "confidence": 0.98,
        "attributes": {
          "state": "empty",
          "plate_format": "96_well",
          "orientation": "standard",
          "boundary_visible": true,
          "barcode_visible": false
        },
        "bbox_fpv": null,
        "bbox_topview": [145, 82, 172, 109],
        "semantic_mask_topview": "base64_encoded_png",
        "view": "top_view"
      },
      {
        "object_id": "tip_001",
        "object_type": "pipette_tip",
        "confidence": 0.96,
        "attributes": {
          "presence": true,
          "channel_type": "single_channel",
          "state": "clean",
          "liquid_filled": true,
          "position": "approaching_well"
        },
        "bbox_fpv": [210, 145, 245, 190],
        "bbox_topview": null,
        "semantic_mask_fpv": null,
        "view": "fpv"
      },
      {
        "object_id": "hand_001",
        "object_type": "operator_hand",
        "confidence": 0.92,
        "attributes": {
          "presence": true,
          "body_part": "thumb",
          "glove_state": "nitrile_glove",
          "grip_position": "above_well",
          "proximity_to_well": "near"
        },
        "bbox_fpv": [50, 200, 120, 280],
        "bbox_topview": null,
        "semantic_mask_fpv": null,
        "view": "fpv"
      },
      {
        "object_id": "liquid_001",
        "object_type": "liquid",
        "confidence": 0.87,
        "attributes": {
          "presence": true,
          "location": "in_tip",
          "volume_estimate": "small",
          "color": "clear",
          "meniscus_visible": true
        },
        "bbox_fpv": [215, 155, 240, 175],
        "bbox_topview": null,
        "semantic_mask_fpv": null,
        "view": "fpv"
      }
    ],
    "scene_summary": {
      "overall_scene_quality": "good",
      "flags": ["none"],
      "procedural_compliance": {
        "gloves_worn": true,
        "plate_lid_removed": true,
        "tip_clean": true,
        "hand_contamination_risk": false,
        "spill_detected": false
      }
    }
  },
  "metadata": {
    "model": "DINOv2-ViT-B/14+LoRA",
    "scene_model": "SceneClassifier-v1",
    "inference_time_s": 0.350,
    "scene_classification_time_s": 0.120,
    "confident": true
  }
}
```

### Field Descriptions

**`scene_classification` (object):** Top-level container for all scene understanding data.

- **`timestamp_frame_index` (integer):** Index of the frame in the clip where scene classification was performed (0-indexed). Useful for multi-frame annotation.
- **`frame_timestamp_s` (float):** Timestamp in seconds relative to clip start. Enables temporal alignment with audio or external logs.
- **`detected_objects` (array of objects):** List of detected objects, each with:
  - **`object_id` (string):** Unique identifier within this clip (e.g., "well_A1", "tip_001"). Allows tracking across frames.
  - **`object_type` (string):** One of: "well", "pipette_tip", "liquid", "operator_hand", "pipette_barrel", "plate_lid", "glove", "rack", "reagent_bottle", "bench_surface", "shadow_reflection", "ejector".
  - **`confidence` (float, 0–1):** Model confidence in detecting this object.
  - **`attributes` (object):** Key-value pairs for object-specific state. Schema depends on `object_type`. See taxonomy section for per-type details.
  - **`bbox_fpv` (array [x0, y0, x1, y1] or null):** Bounding box in FPV image coordinates (if visible in FPV).
  - **`bbox_topview` (array [x0, y0, x1, y1] or null):** Bounding box in top-view coordinates (if visible in top-view).
  - **`semantic_mask_fpv` (base64 string or null):** Optional per-pixel segmentation mask for FPV (encoded as PNG for space efficiency).
  - **`semantic_mask_topview` (base64 string or null):** Optional per-pixel segmentation mask for top-view.
  - **`view` (string):** Which view this detection came from: "fpv", "top_view", or "both" if detected in both views.

**`scene_summary` (object):** High-level summary useful for QA/compliance workflows.

- **`overall_scene_quality` (string):** "good", "fair", "poor" based on visibility, glare, occlusion.
- **`flags` (array of strings):** Warnings or alerts, e.g., ["spill_detected", "glare_moderate", "plate_tilted", "hand_contamination_risk"].
- **`procedural_compliance` (object):** Derived boolean checks for compliance:
  - `gloves_worn`: Operator is wearing gloves (boolean).
  - `plate_lid_removed`: Plate lid is open/removed (boolean).
  - `tip_clean`: Pipette tip shows no visible contamination (boolean).
  - `hand_contamination_risk`: Hand is too close to well (boolean).
  - `spill_detected`: Liquid spill on bench (boolean).

**`metadata.scene_model` (string):** Name and version of the scene classification model (e.g., "SceneClassifier-v1").

**`metadata.scene_classification_time_s` (float):** Latency for scene classification step (separate from well detection).

### Schema Extensibility

The schema is designed for forward compatibility:

1. **New object types:** Add to `object_type` enum and define corresponding `attributes` schema.
2. **New attributes:** Extend the `attributes` object for each type without breaking existing consumers.
3. **Per-frame annotations:** If multi-frame classification desired, repeat `scene_classification` structure per frame, indexed by `timestamp_frame_index`.
4. **Batch output:** Wrap list of single-frame results in a parent array if needed.

### Integration with Existing Code

**Output Formatter (`src/postprocessing/output_formatter.py`):**
- Add `format_scene_classification_output()` function mirroring `format_json_output()`
- Merge well predictions + scene classification in main output
- Ensure backward compatibility: if `scene_model` is None, omit `scene_classification` key

**Inference Pipeline (`inference.py`):**
- After DualViewFusion forward pass, conditionally invoke SceneClassifier
- Merge outputs before JSON serialization
- Add `--with-scene-classification` CLI flag (default True for consistency)

**Logging & Analytics:**
- Index output JSON by well predictions + scene objects
- Enable queries like: "all scenes with glove=true and tip_state=liquid_filled"

---

## Labeling Strategy

High-quality scene classifications require high-quality ground truth annotations. With N=100 real samples, manual annotation is feasible but must be strategic to maximize coverage and consistency.

### Phase 1: Annotation Infrastructure & Taxonomy Validation (1–2 weeks)

**Goals:** Establish labeling workflow, validate taxonomy usability, create annotation guidelines.

**Steps:**

1. **Select Annotation Tool:** Choose between:
   - **CVAT (Computer Vision Annotation Tool):** Open-source, supports bounding boxes, semantic segmentation, classification. Good for distributed team.
   - **Label Studio:** Web-based, flexible schemas, easy team management. Good for rapid iteration.
   - **Custom Streamlit App:** Lightweight, integrates directly with project code, fast iteration. Best for expert annotation.

   *Recommendation:* Start with **Streamlit app** for rapid prototyping (1–2 days to build). Transition to CVAT/Label Studio if scaling to external annotators.

2. **Create Annotation Guideline Document:**
   - Per-object examples with images (do/don't cases)
   - Ambiguity resolution (e.g., "Is a partial tip visible? How much must be visible to count?")
   - Consistency checklist
   - Target: <5% disagreement between two independent annotators (inter-rater reliability)

3. **Pilot Annotation:** Expert annotates 5–10 clips using tool and guidelines.
   - Identify taxonomy gaps or overly granular categories
   - Refine attribute definitions
   - Measure time per clip (~5–10 min per clip for full annotation is target)

4. **Taxonomy Refinement:** Based on pilot, merge/remove redundant classes. Validate with team review.

### Phase 2: Ground Truth Annotation (2–3 weeks)

**Goals:** Annotate all N=100 real samples with scene objects.

**Workflow:**

1. **Stratified Sampling:** Ensure coverage across:
   - Well positions (all 96 wells represented at least once, ideally 3+ times)
   - Pipette types (single-channel, multi-channel if available)
   - Operators (different hands, techniques)
   - Plate conditions (empty, filled, foam)
   - Lighting conditions (bright, dim, glare)

2. **Annotation Distribution:**
   - **Tier 1 (Expert Annotation):** 20 clips (~20% of 100) annotated by project expert (team member familiar with lab).
     - Purpose: High-quality reference set; establish consistency baseline
     - Time: ~2–3 hours total
   - **Tier 2 (Distributed Annotation):** 60 clips (~60%) annotated by 2–3 team members in parallel.
     - Purpose: Coverage; consensus via majority vote where disagreement
     - Time: ~12–15 hours total (5–10 min per clip)
   - **Tier 3 (Spot Check):** 20 clips (~20%) annotated by 1 annotator.
     - Purpose: Redundancy; catch outliers
     - Time: ~2–3 hours

3. **Quality Control:**
   - **Agreement Scoring:** For Tier 1 (all by expert) use as gold standard. For Tier 2, measure pairwise agreement (Jaccard IoU for boxes, F1 for class labels).
   - **Threshold:** Require ≥80% agreement between annotators on shared clips. If lower, review disagreement and refine guidelines.
   - **Outlier Detection:** Flag clips with unusual object combinations or ambiguous scenes for expert review.

4. **Iterative Refinement:** After ~20% of annotations complete, review and update guidelines based on difficult cases.

### Phase 3: Active Learning & Pseudo-Labeling (2–3 weeks, parallel with Phase 2)

**Goals:** Reduce manual annotation burden; increase training data diversity.

**Approaches:**

1. **Uncertainty Sampling with SceneClassifier (Early Iteration):**
   - Train initial SceneClassifier on Phase 2 annotations as they accumulate
   - On remaining unannotated clips, run model inference and rank by prediction uncertainty (entropy over attribute values)
   - Prioritize high-uncertainty clips for manual annotation
   - Expected result: Annotate most uncertain 20–30 clips, skip easy ones

2. **Pseudo-Labeling from Auxiliary Models:**
   - **YOLOv8 for Object Detection:** Pre-trained YOLO can detect hands, bottles, generic objects. Transfer YOLO detections as weak supervision for well, rack, bottle objects.
   - **Liquid Detection via Color Thresholding:** Simple heuristic (HSV-based) can detect liquid in tips; use as weak label.
   - **Glove Detection via CNN:** Train lightweight 2-class CNN (glove / no-glove) on FPV frames; use predictions as initialization for refinement.
   - **Quality Caveat:** Pseudo-labels introduce noise; use confidence thresholding and require human review for model-sensitive attributes.

3. **Synthetic Scene Generation (Optional):**
   - If specific object combinations are rare (e.g., "liquid_filled + multi_channel_tip"), use Stable Diffusion or physics-based simulators to generate synthetic variations
   - Augment real annotations with synthetic scenes (but mark source clearly in metadata)
   - Risk: Synthetic scenes may not match real distribution; use judiciously

### Phase 4: Multi-Frame & Temporal Annotation (Weeks 4–5, if needed)

**Goals:** Capture temporal dynamics (e.g., tip transition from "clean" to "liquid_filled").

**Approach:**
- For each of 20 representative clips, annotate scene state every 5 frames
- Build temporal sequences: {frame_0: well_state=empty, frame_5: well_state=empty, frame_10: well_state=liquid_filled, ...}
- Train temporal models to predict state transitions
- Expected benefit: Better dispense event detection

**Decision Gate:** Only pursue Phase 4 if Phase 2–3 analysis shows temporal patterns are important for well prediction. May not be necessary if single-frame classification sufficient.

### Phase 5: Validation & Deployment (1 week)

**Goals:** Ensure annotation quality meets production standards.

**Validation:**
- Hold out 10 clips (10% of 100) as fully-annotated test set (double-annotated for agreement verification)
- Train SceneClassifier on remaining 90 clips
- Evaluate on held-out 10: target ≥85% F1 for P0 objects (wells, tips), ≥75% for P1 (liquid, hands)
- If below threshold, perform error analysis and refine annotations for misclassified objects

**Failure Modes to Watch:**
- **Class imbalance:** If 95% of clips have gloves, model may always predict gloves=true. Use weighted loss or resampling.
- **Annotation drift:** If multiple annotators, ensure consistency over time. Periodic spot-checks.
- **View bias:** Some objects only visible in FPV or top-view. Ensure model doesn't overfit to one view.

---

## Model Architecture Considerations

Scene classification must integrate elegantly with the existing DINOv2-ViT-B/14 + Temporal Transformer well detection pipeline. Three architectural approaches are viable, each with tradeoffs.

### Option 1: Multi-Task Learning with Shared Backbone (Recommended)

**Architecture:**
```
[FPV frames] → DINOv2-ViT-B/14 + LoRA → TemporalAttention(FPV)
                                               ↓
                                         [FPV features, 768D]
                                                ↓
[Top-view frames] → DINOv2-ViT-B/14 + LoRA → TemporalAttention(TopView)
                                               ↓
                                         [TopView features, 768D]
                                                ↓
                                         Late Fusion (concat) → [1536D]
                                                ↓
                                    Fusion MLP → [256D shared repr]
                                                ↓
                   ┌───────────────┬──────────────┬──────────────┐
                   ↓               ↓              ↓              ↓
            Row Head         Col Head      Scene Classifier    (existing)
            (8 logits)       (12 logits)    (object detection
                                           + classification)
```

**Scene Classifier Sub-Module (Added to DualViewFusion):**

```python
class SceneClassificationHead(nn.Module):
    """Multi-task head for object detection + classification."""
    
    def __init__(self, input_dim=256, num_classes=12):
        super().__init__()
        # Object presence detection (sigmoid output)
        self.object_detector = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)  # logits for each object type
        )
        
        # Per-object attribute classifiers (object-specific)
        self.attribute_heads = nn.ModuleDict({
            'well': WellAttributeHead(input_dim),           # state: empty/filled/partial
            'pipette_tip': PipetteTipAttributeHead(input_dim),  # state: clean/liquid/cracked
            'liquid': LiquidAttributeHead(input_dim),       # location: in_tip/in_well/spill
            'operator_hand': HandAttributeHead(input_dim),  # glove_state, proximity
            # ... etc for each object type
        })
    
    def forward(self, shared_repr, fpv_features, topview_features):
        """
        Args:
            shared_repr: (B, 256) fused representation from well detection
            fpv_features: (B, 768) temporal FPV features
            topview_features: (B, 768) temporal TopView features
        
        Returns:
            object_logits: (B, num_objects) presence logits
            attribute_outputs: dict[object_type] -> attributes
        """
        # Detect object presence
        object_logits = self.object_detector(shared_repr)  # (B, num_objects)
        
        # Classify attributes for detected objects
        attribute_outputs = {}
        for obj_type, attr_head in self.attribute_heads.items():
            # Input attributes classifiers with relevant features
            # Use shared_repr for global context, view-specific features for view-dependent attrs
            attrs = attr_head(shared_repr, fpv_features, topview_features)
            attribute_outputs[obj_type] = attrs
        
        return object_logits, attribute_outputs
```

**Advantages:**
- **Shared backbone:** Reuses DINOv2 pre-training; reduces overfitting on N=100
- **Efficient:** Single forward pass through backbone serves both tasks
- **Interpretability:** Can visualize attention patterns for scene vs. well prediction
- **Multi-task regularization:** Scene classification acts as auxiliary task; improves well robustness
- **Low latency:** ~100ms additional overhead vs. well detection alone

**Disadvantages:**
- **Architecture complexity:** Requires new sub-modules within DualViewFusion
- **Entanglement:** Errors in scene classification may degrade well prediction if loss balancing poor
- **Limited to frame-level:** Current architecture single-frame per clip; temporal scene dynamics harder to model

**Implementation Effort:** ~2–3 weeks (define attribute heads, loss functions, training pipeline)

---

### Option 2: Separate Lightweight Detector (YOLO-style)

**Architecture:**
```
[FPV frames] → YOLOv8-nano (small, fast model, pre-trained on COCO)
                   ↓
              [object detection: bboxes + object class confidence]
              
[Top-view frames] → YOLOv8-nano
                   ↓
              [object detection: bboxes + object class confidence]
              
Attribute Classification (separate):
  Crop detected regions → Small CNN classifier → attributes (glove_state, liquid_filled, etc.)
  
Well Detection: DINOv2 + LoRA + TemporalTransformer + LateFusion (unchanged)
```

**Advantages:**
- **Decoupled:** Scene classification independent from well detection; errors don't cascade
- **Production-ready:** YOLOv8 is mature, well-documented
- **Parallelizable:** Can run scene detector on GPU while well detector runs on CPU if needed
- **Simpler training:** Leverage pre-trained YOLO weights; fine-tune only on new classes

**Disadvantages:**
- **Duplicated backbone:** Both models run separate forward passes; latency increases ~150–200ms
- **Memory overhead:** Two models in memory (DINOv2 ~86M params + YOLOv8-nano ~3M params)
- **Less integration:** Scene classification doesn't inform well prediction; missed opportunity for multi-task regularization
- **Annotation sync:** Must annotate bboxes for all objects (vs. just present/absent in multi-task)

**Implementation Effort:** ~1.5 weeks (YOLO fine-tuning is standard; attribute classifiers simpler than multi-task heads)

---

### Option 3: Frame-Level Scene Classifier (Lightweight CNN)

**Architecture:**
```
[FPV frame, 448x448] → ResNet-18 (small, 11M params)
                         ↓
                    [512D features]
                         ↓
                    Classification heads:
                      - object_presence (12 logits, sigmoid → 12 binary outputs)
                      - object_attributes (per-object classifiers)

[Top-view frame] → ResNet-18
                      ↓
                   [512D features] → concat with FPV features → [1024D]
                      ↓
                   Fusion MLP → [256D]
                      ↓
                   Scene logits (object classes + attributes)
```

**Advantages:**
- **Simplest to implement:** Straightforward CNN + classification heads; ~1 week effort
- **Lightweight:** Minimal memory/latency overhead
- **No DINOv2 overhead:** Separate model; can use CPU for scene classification if needed
- **Fast prototyping:** Easy to iterate on attribute definitions

**Disadvantages:**
- **Redundant features:** ResNet-18 features not aligned with DINOv2 features from well detection
- **Poor generalization:** ResNet-18 has weaker few-shot capability than DINOv2 on limited data
- **No multi-task benefit:** Loses regularization signal from auxiliary task
- **Maintenance burden:** Another model to version, train, and deploy

**Implementation Effort:** ~1 week

---

### Recommendation: Option 1 (Multi-Task Learning)

**Chosen:** Multi-task learning with shared DINOv2 backbone.

**Justification:**
- Aligns with Transfyr's philosophy of efficient, foundation-model-based systems
- Leverages existing DINOv2 pre-training for both tasks
- Minimal latency penalty (~100ms for scene classification)
- Multi-task regularization improves robustness
- Enables future work (e.g., uncertainty quantification across both tasks)

**Implementation Plan:**
1. Extend `DualViewFusion.forward()` to return scene classifications alongside well logits
2. Define per-object attribute head classes (WellAttributeHead, PipetteTipAttributeHead, etc.)
3. Implement multi-task loss: `loss_total = λ_well * loss_well + λ_scene * loss_scene`
4. Fine-tune λ weights via validation set (typically λ_well ≈ 1.0, λ_scene ≈ 0.5–0.8)

---

## Integration with Existing Pipeline

Scene classification must integrate cleanly into the training and inference pipelines without breaking existing workflows.

### Training Pipeline (`train.py`)

**Current Flow:**
```python
# Current train.py
for batch in dataloader:
    fpv_frames, topview_frames, well_targets = batch
    row_logits, col_logits = model(fpv_frames, topview_frames)
    loss = criterion(row_logits, col_logits, row_targets, col_targets)
    loss.backward()
    optimizer.step()
```

**Extended Flow (Multi-Task):**
```python
# Extended train.py
for batch in dataloader:
    fpv_frames, topview_frames, well_targets, scene_targets = batch
    
    # Forward pass now returns both well + scene predictions
    row_logits, col_logits, object_logits, attribute_logits = model(
        fpv_frames, topview_frames
    )
    
    # Multi-task loss
    loss_well = well_criterion(row_logits, col_logits, row_targets, col_targets)
    loss_scene = scene_criterion(object_logits, attribute_logits, scene_targets)
    loss_total = lambda_well * loss_well + lambda_scene * loss_scene
    
    loss_total.backward()
    optimizer.step()
    
    # Logging
    log_metrics({
        'loss_well': loss_well.item(),
        'loss_scene': loss_scene.item(),
        'loss_total': loss_total.item(),
    })
```

**Dataset Extension (`src/preprocessing/video_loader.py`):**
- Add `load_scene_annotations(clip_id, frame_index)` function
- Returns scene targets as dict: `{'object_logits': [0, 1, 1, ...], 'attributes': {...}}`
- Handles JSON loading of annotations created during labeling phase
- Error handling: if scene annotation missing, return None; multi-task loss skips scene term (graceful degradation)

**Training Configuration:**
- Add YAML config for scene classification hyperparameters:
  ```yaml
  scene_classification:
    enabled: true
    lambda_well: 1.0
    lambda_scene: 0.7
    loss_type: "focal"  # or "bce" for simple binary cross-entropy
    attribute_head_config:
      well:
        num_states: 3  # empty, filled, partial
      pipette_tip:
        num_states: 5  # clean, liquid_filled, cracked, detached, contaminated
      # ... etc
  ```

### Inference Pipeline (`inference.py`)

**Current Flow:**
```python
# Current inference.py
detector = PipetteWellDetector(model_checkpoint, ...)
output = detector.infer(fpv_video_path, topview_video_path)
# Returns: {"wells_prediction": [...], "metadata": {...}}
```

**Extended Flow:**
```python
# Extended inference.py with scene classification
detector = PipetteWellDetector(
    model_checkpoint, 
    with_scene_classification=True,
    scene_confidence_threshold=0.5
)
output = detector.infer(fpv_video_path, topview_video_path)
# Returns: {
#   "wells_prediction": [...],
#   "scene_classification": {
#     "detected_objects": [...],
#     "scene_summary": {...}
#   },
#   "metadata": {...}
# }
```

**Code Changes:**
1. Modify `PipetteWellDetector._load_model()` to handle multi-task model (already has row/col heads; add scene heads)
2. Add `PipetteWellDetector.infer_scene()` method:
   ```python
   def infer_scene(self, fpv_features, topview_features):
       """Infer scene objects from fusion features."""
       object_logits, attribute_logits = self.model.scene_head(
           shared_repr, fpv_features, topview_features
       )
       # Post-process logits → scene output JSON
       return self._postprocess_scene(object_logits, attribute_logits)
   ```
3. Modify `PipetteWellDetector.infer()` to invoke scene inference if enabled:
   ```python
   def infer(self, fpv_path, topview_path):
       # ... existing well detection code ...
       if self.with_scene_classification:
           scene_output = self.infer_scene(fpv_features, topview_features)
           output['scene_classification'] = scene_output
       return output
   ```
4. Extend `output_formatter.py` with `format_scene_output()` function

### Output Format Compatibility

**Backward Compatibility:**
- Existing JSON schema unchanged if `scene_classification` omitted
- Consumers expecting only well predictions continue to work
- Consumers looking for scene data check for `"scene_classification"` key presence

**Version Tracking:**
- Add `"scene_model_version"` field in metadata (e.g., "SceneClassifier-v1")
- Allows consumers to handle schema versions gracefully

### CLI Integration

**New Arguments for `inference.py`:**
```
python inference.py \
  --fpv fpv.mp4 \
  --topview topview.mp4 \
  --with-scene-classification \  # new flag
  --scene-threshold 0.5 \         # new flag
  --output result.json
```

**Default Behavior:**
- `--with-scene-classification` defaults to True (enabled by default for consistency)
- Can be disabled with `--no-scene-classification` for backward compatibility with downstream consumers expecting old format

---

## Use Cases

Scene classification enables rich downstream applications across multiple teams and workflows.

### 1. Procedural Compliance & QA

**Persona:** QA Engineer

**Use Case:** Automated protocol verification. Verify that dispense operations follow documented lab procedures.

**Workflow:**
```
Input: 100 clips from production lab
Process:
  1. Run inference with scene classification enabled
  2. For each clip, check procedural_compliance flags:
     - gloves_worn == True (required)
     - plate_lid_removed == True (required before dispense)
     - tip_clean == True (required)
     - hand_contamination_risk == False (must be safe)
  3. Generate compliance report: "98/100 clips passed; 2 flag violations (no gloves in clips 15, 47)"
Output: Compliance dashboard showing per-clip, per-operator metrics
Value: Ensures reproducibility; documents adherence to protocols for audits
```

**Success Metric:** <5% false positive compliance violations; >95% recall of actual violations

---

### 2. Operator Profiling & Technique Analysis

**Persona:** Data Scientist

**Use Case:** Understand operator behavior and technique variations. Identify expert vs. novice operators.

**Workflow:**
```
Input: 1000 clips from 10 operators over 3 months
Process:
  1. Extract scene features per clip:
     - hand_attributes: {glove_state, grip_position, proximity_to_well}
     - tip_attributes: {state, liquid_filled}
     - timing: frame count between plate_lid_open and first_dispense
  2. Cluster operators by scene patterns
  3. Correlate patterns with dispense accuracy:
     - Operator A: gloved, tight_grip, mean_accuracy=98%
     - Operator B: ungloved, loose_grip, mean_accuracy=92%
  4. Build "expert profile" from top performers
  5. Create training recommendations for lower performers
Output: Operator profiling dashboard; automated training alerts
Value: Enables targeted training; highlights expert techniques for knowledge transfer
```

**Success Metric:** Identified patterns correlate with >80% of accuracy variance

---

### 3. Equipment Health Monitoring

**Persona:** Lab Manager

**Use Case:** Track pipette tip health and equipment status over time.

**Workflow:**
```
Input: Daily inference on all liquid handling operations (1000s of clips/month)
Process:
  1. Aggregate tip_state labels over time:
     - Track "cracked" or "contaminated" counts by pipette brand
     - Flag pipettes with >5% crack rate
  2. Detect plate lid failures:
     - If plate_lid_state == "fallen_off" (anomaly flag)
     - Alert maintenance to inspect mechanism
  3. Monitor glassware/plasticware degradation:
     - If bench_surface_cleanliness trends from "clean" → "stained"
     - Suggest surface cleaning protocol
Output: Equipment health dashboard; proactive maintenance alerts
Value: Reduces equipment failures; extends equipment lifetime; improves reproducibility
```

**Success Metric:** Predicted equipment issues with >85% precision (low false alarm rate)

---

### 4. Contamination Risk Detection

**Persona:** QA / Lab Safety Officer

**Use Case:** Automated contamination risk detection based on scene analysis.

**Workflow:**
```
Input: Real-time inference on liquid handling operations
Process:
  1. Per-frame scene flags:
     - hand_contamination_risk = (hand_proximity == "touching" and glove_state == "bare_hand")
     - spill_risk = (spill_detected or liquid_location == "on_surface")
     - tip_contamination_risk = (tip_state == "contaminated" or liquid_color != "clear")
  2. Alert operator if risk detected:
     - "High contamination risk: hand touching well without gloves. Recommend abort and restart."
  3. Log risky operations for post-hoc analysis
Output: Real-time alerts + audit log of risky operations
Value: Prevents contamination events; improves product quality
```

**Success Metric:** Detected high-risk scenes correlate with >80% of contamination failures (retrospectively validated)

---

### 5. Training & Onboarding

**Persona:** Lab Training Coordinator

**Use Case:** Create training materials showing expert vs. novice technique.

**Workflow:**
```
Input: Scene classification from expert (Operator A: 99% success rate) and trainee (Operator B: 85%)
Process:
  1. Extract key frame sequences from high-confidence dispenses:
     - Operator A: [plate_lid → opened, gloves → on, hand → positioning, tip → inserting, liquid → dispensing, tip → ejecting]
     - Operator B: [same sequence, but hand_position → over_well instead of above_well]
  2. Generate side-by-side video clips highlighting differences
  3. Annotate with scene labels: "Expert: hand above well (distance=2cm). Novice: hand over well (distance=0cm, contamination risk)."
Output: Annotated training videos
Value: Accelerates onboarding; documents best practices
```

**Success Metric:** Trainees trained with materials show >10% accuracy improvement vs. control

---

### 6. Data Scientist Research & Feature Mining

**Persona:** ML Research Scientist

**Use Case:** Leverage scene annotations as features for downstream models.

**Workflow:**
```
Input: 1000 clips with well predictions + scene classifications
Process:
  1. Curate subsets for specialized training:
     - "only clips with liquid_visible": train liquid-dynamics model
     - "only clips with hand_visible": train hand-pipette-interaction model
     - "only clips with no_gloves": compare with gloved samples
  2. Build meta-classifier: "given scene context, predict dispense accuracy"
     - Features: {well_state, tip_state, hand_proximity, liquid_color, plate_orientation}
     - Target: dispense_success (inferred from well_state change before/after)
  3. Improve main well detector via multi-task auxiliary learning
     - Scene classification as auxiliary task; improves well detector generalization
Output: New datasets for specialized models; improved backbone features
Value: Unlocks new research directions; improves accuracy of primary task
```

**Success Metric:** Auxiliary task improves well detection accuracy by >2% on held-out test set

---

### 7. Analytics & Reporting

**Persona:** Operations Manager

**Use Case:** Dashboard summarizing lab operations, trends, and efficiency metrics.

**Workflow:**
```
Input: All scene classifications from past month (1000s of clips)
Process:
  1. Aggregate object prevalence:
     - "Multi-channel tips used in 75% of dispenses (up from 60% last month)"
     - "Gloves worn in 98% of operations (up from 95%, good compliance trend)"
  2. Track temporal patterns:
     - "Dispenses take 8–12 seconds on average; no trend over month"
     - "Contamination flags spike at 5pm (end of shift fatigue? → recommend scheduling review)"
  3. Benchmark cross-lab:
     - "Lab A: 99% glove compliance, Lab B: 94%, Lab C: 96%"
     - "Lab A achieves 98% accuracy; Lab B: 91%; Lab C: 95% (correlation: glove compliance)"
Output: Monthly ops dashboard; trend analysis
Value: Data-driven insights for process improvement; motivates teams with metrics
```

**Success Metric:** Dashboard used by >80% of stakeholders for weekly operations review

---

### 8. Anomaly Detection

**Persona:** Infrastructure / System Monitoring

**Use Case:** Detect unusual scenes that may indicate equipment malfunction or operator error.

**Workflow:**
```
Input: Real-time inference with scene classification enabled
Process:
  1. Define baseline distribution of scene attributes:
     - Normal: {gloves=true, tip_clean=true, plate_orientation=standard}
  2. For each new clip, compute anomaly score via isolation forest or VAE:
     - If scene significantly deviates from baseline, flag as anomaly
     - Example anomalies: "plate_tilted=true" (equipment misalignment), "all_tips_liquid_filled=true" (pre-dispense setup unusual)
  3. Alert ops team:
     - "Anomaly detected in clip 42: plate tilted 15°. Suggest equipment check."
Output: Anomaly alert stream
Value: Early detection of equipment issues before they cause failures
```

**Success Metric:** Detected anomalies correlate with >70% of subsequent equipment failures

---

## Acceptance Criteria

"Done" for the scene classification feature requires meeting the following criteria across model quality, integration, and deployment dimensions.

### Model Quality (P0 Acceptance Criteria)

1. **Object Detection Accuracy (Primary Task)**
   - **Well Plate:** ≥95% precision, ≥92% recall (must reliably detect plate)
   - **Pipette Tip:** ≥92% precision, ≥88% recall (critical for dispense verification)
   - **Target:** F1 ≥0.90 for both well and tip detection

2. **Attribute Classification Accuracy (Stratified by Object)**
   - **Well state (empty/filled/partial):** ≥85% F1 per class
   - **Tip state (clean/liquid/cracked):** ≥80% F1 per class
   - **Hand presence/glove state:** ≥90% F1 (binary classification easier)
   - **Liquid presence:** ≥80% F1 (harder; shadows confuse; acceptable)
   - **Overall attribute F1 (macro-averaged):** ≥85%

3. **Validation Metrics (Held-Out Test Set)**
   - Evaluate on 10 held-out clips (never seen during training)
   - Per-object and per-attribute metrics as above
   - No overfitting: train/test gap <3% on primary metrics

4. **Multi-Task Learning Constraint**
   - Well detection accuracy (row/col F1) must NOT degrade when scene classification added
   - Target: Maintain well F1 ≥0.95 (same as without scene classification)
   - Scene classification auxiliary task must improve or maintain well robustness

### Data Quality (P0 Acceptance Criteria)

1. **Annotation Completeness**
   - All 100 real samples annotated with P0 objects (well, tip)
   - ≥80% of samples annotated with P1 objects (liquid, hand, lid)
   - Inter-rater agreement (on overlap samples): ≥80% IoU for bboxes, ≥85% F1 for attributes

2. **Schema Consistency**
   - All output JSON valid against schema (lintable via `jsonschema` library)
   - No missing required fields; consistent attribute enums across samples
   - Field types match specification (string, float, array, object)

3. **Ground Truth Quality**
   - Random spot-check: 10 clips reviewed by expert; agreement ≥90%
   - Consistency over time: annotations created early and late in phase should align (no annotator drift)

### Integration & Code Quality (P1 Acceptance Criteria)

1. **Training Pipeline**
   - `train.py` extended to support multi-task learning
   - Config YAML specifies scene classification hyperparameters
   - Training runs without errors; loss converges within 100 epochs
   - Checkpoints save/load correctly with scene weights

2. **Inference Pipeline**
   - `inference.py` supports `--with-scene-classification` flag
   - Scene output correctly formatted as JSON per schema
   - Backward compatible: old code without scene flag still works
   - Latency impact <150ms per sample (scene inference adds <100ms to well detection)

3. **Output Format**
   - JSON output matches `FEATURE_SCENE_CLASSIFICATION.md` schema
   - All bbox coordinates in correct image coordinate system
   - Confidence scores between 0–1; valid probabilities
   - No NaN or inf values in output

4. **Testing & Documentation**
   - Unit tests: SceneClassificationHead forward pass, postprocessing, schema validation
   - Integration tests: full pipeline on synthetic data (10 clips)
   - Code documentation: docstrings for all new functions; architecture diagram in ARCHITECTURE.md
   - README updated with scene classification instructions

### Deployment & Operational (P2 Acceptance Criteria)

1. **Performance at Scale**
   - Batch inference (10 samples): <5 sec total (existing SLA maintained)
   - GPU memory: <6GB for model + batch (leaves headroom for ops)
   - CPU inference (fallback): <2 min per sample acceptable (scene on GPU preferred)

2. **Robustness & Failure Modes**
   - Model handles edge cases: no plate visible, no tip visible, severe glare
   - Graceful degradation: if scene confidence <0.3, output conservative labels or "uncertain"
   - No crashes on malformed input (video, frames)

3. **Monitoring & Alerting**
   - Log scene confidence scores per clip; alert if <0.5 (low-quality scene)
   - Track error rates per object type; alert if >20% error rate on any object
   - Dashboard shows scene classification success over past week/month

4. **Stakeholder Sign-Off**
   - QA Engineer: Scene compliance flags useful, <5% false positive rate acceptable
   - Data Scientist: Can export annotated dataset for research; schema extensible
   - Lab Manager: Equipment health signals correlate with known issues
   - ML Scientist: Multi-task training improves well accuracy or maintains it

### Documentation & Handoff (P2 Acceptance Criteria)

1. **Feature Documentation**
   - This feature request document (FEATURE_SCENE_CLASSIFICATION.md) finalized
   - ARCHITECTURE.md updated with scene classification details
   - Per-object annotation guidelines published (for QA/future annotators)

2. **Code Handoff**
   - Code reviewed by >1 team member; comments addressed
   - Runnable example: `python inference.py --fpv test_fpv.mp4 --topview test_topview.mp4 --with-scene-classification`
   - Deployment checklist provided (dependencies, checkpoints, environment setup)

---

## Risks & Open Questions

### Technical Risks

**Risk 1: Annotation Quality & Consistency**
- **Issue:** Manual annotation is labor-intensive and subjective. Inconsistent ground truth degrades model quality.
- **Severity:** High (affects all downstream uses)
- **Mitigation:**
  - Develop detailed annotation guidelines with do/don't examples
  - Measure inter-rater agreement; retrain annotators if <80% agreement
  - Use expert (team member) as gold standard for validation set
  - Implement active learning to prioritize ambiguous samples
- **Contingency:** If inter-rater agreement <75% after mitigation, reduce taxonomy scope (P0 only: wells, tips)

**Risk 2: Multi-Task Learning Interference**
- **Issue:** Scene classification as auxiliary task may compete with well detection, degrading primary task accuracy.
- **Severity:** Medium (primary task must not degrade)
- **Mitigation:**
  - Monitor well detection loss and scene classification loss separately during training
  - Use loss weighting (λ_well, λ_scene) to balance tasks; validate on held-out set
  - If well accuracy drops >1%, increase λ_well or reduce λ_scene
- **Contingency:** Fall back to Option 2 (separate YOLO detector) if multi-task approach degrades well accuracy

**Risk 3: Limited Training Data**
- **Issue:** N=100 real samples with scene annotations may overfit; scene model may not generalize to new operators/equipment.
- **Severity:** Medium (affects deployment robustness)
- **Mitigation:**
  - Aggressive augmentation: crop, rotate, flip, ColorJitter during training
  - Synthetic data generation: use Stable Diffusion or physics simulators to augment scene diversity
  - Target: 500–1000 total clips (100 real + 400–900 synthetic)
  - Cross-validation: test on held-out natural clips; measure domain gap
- **Contingency:** If validation accuracy <75%, reduce object taxonomy and focus on P0 (wells, tips only)

**Risk 4: View Dependency**
- **Issue:** Objects may be visible in only one view (FPV or top-view), causing class imbalance or view-specific overfitting.
- **Severity:** Low-Medium (affects some objects like liquid)
- **Mitigation:**
  - Design classifier to handle missing view gracefully (use available view only)
  - Ensure training data represents both view-dependent and view-independent objects
  - Evaluate separately by view: FPV accuracy, TopView accuracy, combined
- **Contingency:** For view-specific objects (liquid, hand), train separate models per view if combined model underperforms

### Data & Labeling Risks

**Risk 5: Labeling Cost & Schedule Slippage**
- **Issue:** Manual annotation of 100 clips at 5–10 min/clip = 8–16 hours. If annotators unavailable or scope creep, schedule slips.
- **Severity:** Medium (delays feature release)
- **Mitigation:**
  - Allocate 2–3 annotators in parallel; target completion in 2 weeks
  - Prioritize P0 objects first; P1/P2 can follow in Phase 2
  - Use pseudo-labeling from YOLO + confidence thresholding to reduce manual burden
  - Build streamlined Streamlit annotation tool (1–2 days) to minimize friction
- **Contingency:** Launch with P0 only (wells, tips) if schedule tight; add P1/P2 in Phase 2

**Risk 6: Annotation Ambiguity**
- **Issue:** Some scenes are ambiguous. Example: "Is a partial tip visible? How much must be visible to count as present?"
- **Severity:** Medium (reduces inter-rater agreement)
- **Mitigation:**
  - Create detailed annotation guidelines with images (do/don't cases)
  - Pilot annotation on 5 clips; identify ambiguities before full annotation
  - Refine taxonomy based on pilot feedback
  - Establish tie-breaking procedure (expert annotator resolves disagreement)
- **Contingency:** Accept lower agreement on ambiguous attributes (e.g., 75% vs. 85%); document in metadata

### Model & Performance Risks

**Risk 7: Class Imbalance**
- **Issue:** Some objects rare (e.g., "cracked_tip"), some common (e.g., "glove_present"). Imbalanced classes cause poor recall on rare classes.
- **Severity:** Medium (affects P1/P2 objects)
- **Mitigation:**
  - Use class-weighted loss: weight rare classes higher
  - Stratified sampling: ensure training batches contain diverse classes
  - Synthetic data generation: oversample rare classes (e.g., generate synthetic "cracked_tip" scenes)
  - Evaluate with F1 (not accuracy), which accounts for imbalance
- **Contingency:** Merge rare classes; e.g., combine "cracked", "detached", "contaminated" → "tip_damaged"

**Risk 8: Temporal Dynamics Unsupported**
- **Issue:** Current architecture is frame-level. Scene states change over clip (tip goes from clean → liquid_filled), but single-frame model misses dynamics.
- **Severity:** Low (frame-level often sufficient; temporal can be Phase 2)
- **Mitigation:**
  - For Phase 1, classify only on frame with dispense event (use temporal attention to identify frame)
  - Document that temporal scene tracking is future work
  - If temporal patterns important, extend to multi-frame classification in Phase 2
- **Contingency:** Not a blocker; implement temporal in Phase 2 if downstream uses demand it

### Operational & Stakeholder Risks

**Risk 9: Adoption & Buy-In**
- **Issue:** Teams may not use scene classification outputs if utility unclear or outputs noisy.
- **Severity:** Low-Medium (affects ROI)
- **Mitigation:**
  - Demonstrate on real examples: "This scene flag caught contamination in Lab A"
  - Iterate with stakeholders (QA, Data Scientists) during development
  - Build simple dashboard showing scene insights (operator metrics, equipment health)
  - Provide clear documentation on how to consume scene output JSON
- **Contingency:** Make scene classification opt-in (flag in CLI); don't force adoption

**Risk 10: Regulatory & Documentation**
- **Issue:** If scene classification used for compliance/QA, may have regulatory requirements (audit trails, validation).
- **Severity:** Low (depends on lab regulatory environment)
- **Mitigation:**
  - Document model validation: accuracy on test set, inter-rater agreement, failure modes
  - Log all predictions with timestamps and model version
  - Ensure scene output JSON is queryable/searchable for audits
  - Coordinate with compliance team if using for regulatory reporting
- **Contingency:** Mark scene classification as research/non-regulatory until validation complete

### Open Questions

1. **Phased Rollout:** Should P0 (wells, tips) launch first, then P1 (liquid, hands, lids), or all together?
   - Recommendation: Stagger—P0 + simple hand detection in Phase 1 (2 weeks). Add P1 in Phase 2 (2 more weeks).

2. **Synthetic Data Quality:** How much synthetic data is needed? What quality bar?
   - Recommendation: Start with 500 synthetic clips; target FID <50 (visually plausible). Validate on held-out real set.

3. **Attribute Granularity:** Should "tip_state" be 3 classes (clean/liquid/cracked) or 5 (clean/liquid/cracked/detached/contaminated)?
   - Recommendation: Start with 3 (MVP); expand to 5 in Phase 2 if data supports it.

4. **Real-Time vs. Batch:** Should scene classification run per-frame or per-clip?
   - Recommendation: Per-clip (frame with highest dispense confidence). Temporal tracking in Phase 2.

5. **Multi-View Fusion:** For objects visible in both FPV and top-view (e.g., well state), should fuse predictions or use best view only?
   - Recommendation: Fuse via attention (late fusion already present). Weight FPV lower for objects where top-view more reliable.

6. **False Positive Tolerance:** QA use case requires low false positives (don't want to flag safe procedures as risky). What's the acceptable FP rate?
   - Recommendation: <5% false positive rate on compliance flags (e.g., "hand contamination risk" when no risk). Okay with higher FP rate on minor attributes (e.g., "spill detected").

7. **Privacy & Data Retention:** Should scene annotations (which include hand/operator identity) be retained long-term? What's the privacy policy?
   - Recommendation: Anonymize operator identity in annotations; retain object states but not hand-specific info. Coordinate with compliance.

---

## Implementation Phases

Scene classification will be rolled out in phases to manage risk, enable early feedback, and allow for parallel work.

### Phase 1: MVP (Weeks 1–2)

**Objectives:**
- Build annotation infrastructure and taxonomy validation
- Annotate all 100 real samples with P0 + key P1 objects
- Implement multi-task learning integration into DINOv2 backbone
- Achieve ≥90% F1 on well and tip detection; ≥85% on hand/glove presence

**Deliverables:**
1. **Annotation Guideline Document** (Day 1)
   - Object definitions, attribute enums, edge cases, examples
   - Target: <5% disagreement between annotators

2. **Streamlit Annotation Tool** (Days 2–3)
   - Load video clips and FPV/top-view frames
   - Click-and-drag bounding boxes for objects
   - Dropdown menus for attributes
   - Save annotations to JSON

3. **Phase 1 Annotation** (Days 4–8)
   - Expert annotates 20 clips (reference set)
   - 2 other team members annotate remaining 80 in parallel
   - Agreement checks; refine guidelines if needed
   - Target: 80 clips fully annotated by end of week 1

4. **SceneClassificationHead Implementation** (Days 3–10)
   - Add scene classification branch to DualViewFusion
   - Define WellAttributeHead, TipAttributeHead, HandAttributeHead classes
   - Multi-task loss: focal_loss_well + focal_loss_scene
   - Extend train.py to support multi-task training

5. **Training & Validation** (Days 11–14)
   - Train on 80 annotated real clips + synthetic augmentation
   - Validate on 20 held-out real clips
   - Target: well F1 ≥0.90, tip F1 ≥0.88, hand F1 ≥0.90
   - If targets met, proceed to Phase 2; else debug and iterate

**Team Allocation:**
- 1 ML Engineer: Implement SceneClassificationHead + training
- 2 Data Annotators: Annotation in parallel
- 1 ML Scientist: Oversee, validate results
- (Optional) 1 Software Engineer: Streamlit tool (can be weeks 1–2)

**Success Metrics:**
- ✓ All 100 samples have P0 annotations (well, tip bounding boxes)
- ✓ 80% of samples have P1 annotations (hand, glove, lid presence)
- ✓ Well F1 ≥0.90 on held-out test (not degraded from baseline ~0.95; acceptable loss <2%)
- ✓ Scene F1 ≥0.85 macro-averaged

---

### Phase 2: Extended Taxonomy & Robustness (Weeks 3–4)

**Objectives:**
- Add P2 objects (barrel, rack, bottles, bench, shadows)
- Implement active learning to reduce annotation burden
- Validate multi-task learning generalization on unseen row/column combinations
- Integrate into inference.py with `--with-scene-classification` flag

**Deliverables:**
1. **P2 Annotation** (Days 1–5)
   - Annotate 60 clips with P2 objects (rack, reagent bottles, shadows)
   - Use active learning: rank unannotated clips by uncertainty, annotate top 30–40
   - Reduce manual burden via pseudo-labeling (YOLO pre-labels bboxes; humans refine)

2. **Pseudo-Labeling Pipeline** (Days 2–4)
   - Fine-tune YOLOv8-nano on P1 annotations (hand detection as reference)
   - Generate weak labels for P2 objects (rack, bottles)
   - Human review: filter low-confidence predictions
   - Target: 50% of P2 annotations from pseudo-labels (validated by human)

3. **Inference Integration** (Days 5–8)
   - Extend inference.py: `--with-scene-classification` flag
   - Modify PipetteWellDetector.infer() to invoke scene classifier
   - Format output JSON per schema (FEATURE_SCENE_CLASSIFICATION.md)
   - Backward compatibility: old code without flag still works

4. **Generalization Testing** (Days 9–10)
   - Create test set: unseen row/column combinations (e.g., if training on rows A–D, cols 1–6, test on rows E–H, cols 7–12)
   - Measure well F1 on unseen combinations; target ≥85%
   - Scene F1 on unseen combinations; target ≥80%
   - If targets not met, perform error analysis and retrain with augmentation

5. **Documentation & CLI Updates** (Days 11–14)
   - Update ARCHITECTURE.md with scene classification details
   - Update README with example: `python inference.py --fpv ... --topview ... --with-scene-classification`
   - Write scene classification postprocessing code (logits → scene output)

**Team Allocation:**
- 1 ML Engineer: Inference integration + YOLO fine-tuning
- 1 Data Annotator: Annotation + pseudo-label review
- 1 ML Scientist: Validation + error analysis

**Success Metrics:**
- ✓ All 100 samples annotated with P0 + P1
- ✓ 60 samples annotated with P2
- ✓ Well F1 ≥0.90 on held-out test (maintained from Phase 1)
- ✓ Well F1 ≥0.85 on unseen row/column combinations (generalization check)
- ✓ Scene inference runs in <150ms per sample (latency acceptable)
- ✓ inference.py produces valid JSON per schema

---

### Phase 3: Temporal Modeling & Advanced Features (Weeks 5–6)

**Objectives:**
- Extend scene classification to multi-frame temporal analysis
- Implement dispense event detection (when does dispense occur?)
- Validate scene classification improves well prediction robustness
- Prepare for production deployment

**Deliverables:**
1. **Temporal Scene Analysis** (Days 1–5)
   - For 20 representative clips, annotate scene state every 5 frames
   - Train temporal model (LSTM or Transformer) to predict state transitions
   - Example: [frame_0: well_empty, frame_5: well_empty, frame_10: well_liquid_filled] → model learns transition dynamics
   - Evaluate: predict next frame state; target ≥80% accuracy

2. **Dispense Event Detection** (Days 3–7)
   - Define dispense event as first frame where well_state transitions from empty → liquid_filled
   - Train detector to predict dispense_frame_idx in clip
   - Validate: measure error in predicted vs. ground truth frame index; target <3 frames error
   - Use dispense frame to improve well detection (select highest-confidence frame for well prediction)

3. **Robustness Analysis** (Days 8–10)
   - Test scene classification on edge cases:
     - Severe glare / shadows
     - Partial plate visibility (tip obscures well)
     - Multiple tips in frame (multi-channel)
   - Measure accuracy degradation; document failure modes
   - Propose mitigations for Phase 4+

4. **Deployment Checklist** (Days 11–12)
   - Model checkpoints saved with version tags (e.g., "SceneClassifier-v1")
   - Dependencies documented (PyTorch, DINOv2, etc.)
   - Docker image for containerized inference (optional but recommended)
   - Runnable integration test: `python test_scene_classification.py` (runs on synthetic data)

**Team Allocation:**
- 1 ML Scientist: Temporal modeling + event detection
- 1 ML Engineer: Robustness testing + deployment prep

**Success Metrics:**
- ✓ Temporal model predicts state transitions with ≥80% accuracy
- ✓ Dispense event detector <3 frames error on average
- ✓ Scene classification accuracy stable across edge cases (within 5% of baseline)
- ✓ Deployment checklist completed; ready for production

---

### Phase 4: Production Deployment & Monitoring (Weeks 7–8)

**Objectives:**
- Deploy scene classification to production
- Monitor performance and gather stakeholder feedback
- Iterate on low-confidence predictions
- Build analytics dashboards

**Deliverables:**
1. **Production Deployment** (Days 1–3)
   - Deploy checkpoint and inference code to production environment
   - Enable `--with-scene-classification` for all new inferences
   - Monitor inference latency and error rates
   - Set up alerting: if scene_confidence <0.5 on >10% of clips, alert

2. **Analytics Dashboard** (Days 4–7)
   - Build dashboard (e.g., via Grafana or custom web app) showing:
     - Per-object detection rates (% of clips with well detected, tip detected, etc.)
     - Per-operator metrics (glove compliance, hand_contamination_risk flags)
     - Equipment health trends (tip_cracked rate per brand)
     - Temporal trends (are glove compliance rates improving?)
   - Target: <1s query latency; refreshed daily

3. **Stakeholder Feedback & Iteration** (Days 5–8)
   - QA Team: Review compliance flag accuracy; if >5% FP, refine thresholds
   - Data Scientists: Validate scene features useful for research; request additional attributes if needed
   - Lab Managers: Confirm equipment health signals correlate with known issues
   - Iterate: refine model or post-processing based on feedback

4. **Documentation & Handoff** (Days 8–14)
   - Write deployment guide: how to run inference with scene classification
   - Create runbook for troubleshooting: "If scene_confidence low, check X"
   - Document failure modes: "Scene classification unreliable when plate tilted >15°"
   - Schedule knowledge transfer session with stakeholders

**Team Allocation:**
- 1 ML Engineer: Deployment + monitoring
- 1 Data Scientist: Analytics dashboard + stakeholder feedback
- 1 ML Scientist: Validation + troubleshooting

**Success Metrics:**
- ✓ Scene classification running on 100% of new inferences (no errors)
- ✓ Analytics dashboard available to stakeholders; >50% adoption in first week
- ✓ Stakeholder feedback collected; ≥3 actionable insights identified
- ✓ Scene classification accuracy stable in production (no data drift)

---

### Phase 5: Research & Future Work (Weeks 9+, if needed)

**Potential Extensions** (not in initial scope; for future consideration):

1. **3D Reconstruction of Scene (Research):**
   - Extend 3D Gaussian Splatting (from ARCHITECTURE.md) to reconstruct full scene (not just well location)
   - Use to improve robustness to glare/reflections (especially for liquid detection)

2. **Uncertainty Quantification:**
   - Implement MC Dropout or Deep Ensembles for per-prediction uncertainty
   - Output confidence intervals: "hand_proximity within 1–3cm with 90% confidence"
   - Use for risk-aware decision-making in QA workflows

3. **Domain Adaptation:**
   - Fine-tune on new lab setups with minimal labeled data
   - Use transfer learning from Phase 1–4 as starting point
   - Measure domain shift via feature distribution divergence

4. **Multi-Label Well State:**
   - Extend well classification from single-state to multi-state (e.g., partially_filled with foam)
   - Requires more granular annotation; defer to Phase 5

---

## Success Criteria Summary

By end of Phase 2 (end of Week 4):

1. ✓ All 100 samples annotated with P0 + P1 objects
2. ✓ Multi-task model achieves ≥90% F1 on well detection, ≥88% on tip detection, ≥85% on hand/glove detection
3. ✓ Well detection accuracy NOT degraded (well F1 ≥0.93, acceptable loss <2% from baseline ~0.95)
4. ✓ Scene classification JSON schema validated; all outputs parse without error
5. ✓ inference.py extended with `--with-scene-classification` flag; backward compatible
6. ✓ Documentation updated (this feature request + ARCHITECTURE.md updated)
7. ✓ Stakeholder sign-off: QA Engineer, Data Scientist, ML Scientist all confirm utility and readiness

By end of Phase 4 (end of Week 8):

8. ✓ Production deployment stable; 0 critical errors over 1 week
9. ✓ Analytics dashboard active; ≥50% adoption among stakeholders
10. ✓ Stakeholder feedback incorporated; ≥3 improvements made
11. ✓ Failure modes documented; runbook created for troubleshooting

---

## Conclusion

Scene object classification transforms the Pipette Well Challenge from a single-task spatial detector into a multi-modal scene understanding system. By classifying wells, pipette tips, liquid, hands, and supporting equipment, the system captures rich semantic information about lab operations that downstream teams can use for compliance, analytics, research, and continuous improvement.

This feature request provides:
1. Clear motivation for why scene classification matters
2. Comprehensive object taxonomy (P0–P3 priorities)
3. Data schema for extensible, team-friendly output
4. Realistic labeling strategy (manual + active learning + pseudo-labeling)
5. Multi-task learning architecture integrated into existing pipeline
6. Concrete use cases showing downstream value
7. Acceptance criteria for code quality and model performance
8. Risk mitigation and open questions
9. Phase-wise implementation roadmap (Phases 1–5 over 8 weeks)

**Next Steps:**
1. **Review & Feedback** (This Week): Team reviews feature request; collect feedback
2. **Annotation Tool Setup** (Week 1): Build Streamlit annotation tool
3. **Phase 1 Kick-Off** (Week 1): Annotation begins; multi-task head implementation starts
4. **Phase 1 Completion** (Week 2): MVP scene classification trained and validated
5. **Proceed to Phase 2** (Week 3): Extended taxonomy, inference integration, deployment prep

This feature aligns with Transfyr's mission of capturing tacit knowledge from expert physical operations and is expected to unlock significant value for QA, analytics, and future model development.

---

**Document Version:** 1.0  
**Last Updated:** 2026-04-16  
**Next Review Date:** 2026-05-01 (after Phase 1 completion)
