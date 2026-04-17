# Design: Pipette Well Detection Visualization Tool

**Document:** Visualization Tool Specification & Architecture  
**Date:** 2026-04-16  
**Role:** Software/ML Architect  
**Audience:** ML Scientists, Data Scientists, QA Team, Development Team  
**Status:** DESIGN SPECIFICATION  
**Scope:** Phase 1 (Flat File) + Phase 2 (Cloud Service) Design

---

## 1. Executive Summary

The Pipette Well Detection Visualization Tool enables the team to inspect, annotate, and debug model predictions on dual-view video. It ingests inference results (JSON format) and ground-truth labels, overlays predictions on synchronized FPV and top-view videos, and exports annotated video clips with analysis metadata.

### Core Goals

1. **Reduce debugging friction** — User specifies clip ID or result index; tool auto-matches videos and renders output
2. **Enable systematic analysis** — Filter by accuracy metric (best, worst, strangest) and generate ranked video compilations
3. **Capture team insights** — Persistent QA annotations linked to results for regression testing and institutional learning
4. **Support multiple workflows** — CLI-first for Phase 1; web UI for Phase 2
5. **Maintain traceability** — Every output includes date, clip_id, result index, and persistent reference ID

### Deliverables

- **Phase 1 (Implement Now):** Flat-file JSON storage, directory-based organization, Python CLI tool (`tools/visualizer.py`)
- **Phase 2 (Design Only):** PostgreSQL schema, REST API, S3/GCS storage, web UI mockups

---

## 2. Use Cases

### 2.1 User-Initiated Queries

**UC-1: Visualize Specific Clip**  
*User:* "Show me the predictions for clip_001_FPV"  
*Tool:* Locates clip_001_FPV.mp4 + clip_001_Topview.mp4, loads inference result and ground truth, renders overlay, outputs video.

**UC-2: Visualize Result Index**  
*User:* "Show me result 19 from my inference run"  
*Tool:* Loads results.json, extracts result[19], auto-matches to videos by clip_id, renders, outputs video.

**UC-3: Free-Form Annotation**  
*User:* "Annotate result 42 as 'false positive—pipette was not dispensing'"  
*Tool:* Creates annotation record linked to result 42 with timestamp, stores in annotations.json for future reference.

### 2.2 Data Scientist Workflows

**UC-4: Worst Detections**  
*User:* "Generate a ranked compilation of the 20 poorest detections from results.json"  
*Tool:* Computes error metric (e.g., Hamming distance or IoU) for each prediction vs. ground truth, sorts ascending, renders top 20 clips, concatenates into single output video with ranking index.

**UC-5: Strangest Results**  
*User:* "Show me the 5 most ambiguous predictions—high confidence on both row and column but low correlation"  
*Tool:* Computes anomaly score (e.g., max_row_prob × max_col_prob normalized against correlation), sorts by uncertainty, renders top 5.

**UC-6: Best Detections**  
*User:* "Generate a highlight reel of the 10 best detections"  
*Tool:* Sorts by accuracy (Hamming or IoU), renders top 10, applies green highlight to predicted wells.

**UC-7: Inference Latency Profiling**  
*User (ML Scientist):* "Overlay per-frame inference latency on the video timeline"  
*Tool:* Reads inference metadata (per-frame timing), draws latency bar on side of video, highlights frames exceeding SLA (e.g., >100ms).

**UC-8: Confusion Heatmap**  
*User (Data Scientist):* "Generate a heatmap overlay showing where the model most often confuses rows vs. columns across all clips"  
*Tool:* Aggregates errors across results, computes per-well error frequency, renders as color overlay on 96-well grid visible in top-view, saves as image + video.

### 2.3 QA & Regression Testing

**UC-9: QA Annotation Suite**  
*User (QA Engineer):* "Create a free-form text note on result 15: 'Pipette nozzle reflection caused false positive in E5 and E6—design will address with new geometry'"  
*Tool:* Stores annotation with result_id, timestamp, author tag, searchable by result_id or date range.

**UC-10: Regression Test Suite**  
*User (QA Lead):* "Build a regression test suite: compile 30 annotated clips, mark expected predictions, enable automated comparison on each new model version"  
*Tool:* Reads all annotations, exports a structured JSON file with result_ids, clip metadata, expected wells, and QA notes. New inference runs can compare against this ground truth.

**UC-11: False Positive/Negative Analysis**  
*User (ML Scientist):* "Find all false positives where predicted ≠ ground_truth; group by prediction type (extra well, missing well, correct well)"  
*Tool:* Filters results by error type, renders compilation video, generates summary statistics (count, examples by error category).

---

## 3. Phase 1: Flat File Architecture

### 3.1 Directory Structure

```
project_root/
├── data/
│   ├── pipette_well_dataset/
│   │   ├── labels.json                    # Ground truth labels (provided)
│   │   ├── clip_001_FPV.mp4               # Videos (provided)
│   │   ├── clip_001_Topview.mp4
│   │   ├── clip_002_FPV.mp4
│   │   └── ... (more clips)
│   └── (other datasets if added)
│
├── tools/
│   └── visualizer.py                      # Main CLI tool (to implement)
│
├── outputs/
│   ├── visualizations/
│   │   ├── 2026-04-16_results_run1/        # One directory per inference run
│   │   │   ├── manifest.json               # Metadata for this run
│   │   │   ├── result_00_clip_001.mp4      # Rendered clips
│   │   │   ├── result_00_clip_001.json     # Metadata for each clip
│   │   │   ├── result_05_clip_003.mp4
│   │   │   └── ... (more rendered clips)
│   │   │
│   │   └── 2026-04-16_results_run2/
│   │       └── ... (similar structure)
│   │
│   ├── annotations/
│   │   ├── annotations.json                # All QA annotations
│   │   └── annotation_index.json           # Index by result_id for fast lookup
│   │
│   ├── analyses/
│   │   ├── heatmap_errors.json             # Aggregated error heatmap data
│   │   ├── heatmap_errors.png              # Visualization
│   │   ├── latency_profile_2026-04-16.json # Per-frame latency analysis
│   │   └── strangest_results.json          # Anomaly scoring results
│   │
│   └── regression_suites/
│       ├── suite_v1.0.json                 # Regression test suite
│       └── suite_v1.0_log.json             # Comparison results from new runs
│
└── inference_results/
    ├── results_2026-04-16_run1.json        # Raw inference output (provided by inference.py)
    ├── results_2026-04-16_run2.json
    └── ... (more result files)
```

### 3.2 JSON Schema Reference

#### 3.2.1 Inference Results (Input)
Source: `inference.py` output or user-provided results file.

```json
{
  "clip_id_FPV": "clip_001_FPV",
  "clip_id_Topview": "clip_001_Topview",
  "wells_prediction": [
    {"well_row": "A", "well_column": 1},
    {"well_row": "A", "well_column": 2}
  ],
  "metadata": {
    "inference_time_s": 0.342,
    "confident": true,
    "threshold": 0.5,
    "adaptive": true,
    "max_row_prob": 0.98,
    "max_col_prob": 0.95
  }
}
```

#### 3.2.2 Ground Truth Labels (Input)
Source: `data/pipette_well_dataset/labels.json`

```json
[
  {
    "clip_id_FPV": "clip_001_FPV",
    "clip_id_Topview": "clip_001_Topview",
    "wells_ground_truth": [
      {"well_row": "A", "well_column": 1}
    ]
  },
  ...
]
```

#### 3.2.3 Rendered Clip Metadata (Output)
Stored alongside each rendered video as `result_XX_clip_YYY.json`.

```json
{
  "result_index": 0,
  "clip_id": "clip_001",
  "video_output_path": "outputs/visualizations/2026-04-16_results_run1/result_00_clip_001.mp4",
  "render_timestamp": "2026-04-16T14:32:05Z",
  "render_duration_s": 12.3,
  "fps": 30,
  "resolution": "1920x1080",
  "prediction": {
    "wells": [{"well_row": "A", "well_column": 1}],
    "metadata": {
      "inference_time_s": 0.342,
      "confident": true,
      "max_row_prob": 0.98,
      "max_col_prob": 0.95
    }
  },
  "ground_truth": {
    "wells": [{"well_row": "A", "well_column": 1}]
  },
  "accuracy": {
    "metric": "hamming_distance",
    "value": 0,
    "interpretation": "Exact match"
  },
  "visual_config": {
    "overlay_style": "dual_grid",
    "well_radius_px": 18,
    "color_correct": "#00FF00",
    "color_fp": "#FF0000",
    "color_fn": "#0000FF"
  },
  "persistent_id": "2026-04-16:clip_001:result_0:20260416T143205Z"
}
```

#### 3.2.4 Annotations (Output)
Stored in `outputs/annotations/annotations.json` as a list; index in `annotation_index.json`.

```json
{
  "annotation_id": "ann_00001",
  "result_id": "2026-04-16:clip_001:result_0:20260416T143205Z",
  "clip_id": "clip_001",
  "result_index": 0,
  "author": "qa_lead",
  "timestamp": "2026-04-16T15:10:22Z",
  "category": "false_positive",
  "text": "Pipette nozzle reflection caused false positive in E5 and E6—design will address with new geometry in v2.0",
  "tags": ["reflection_artifact", "optical_issue", "design_feedback"],
  "linked_to_run": "2026-04-16_results_run1",
  "severity": "medium",
  "actionable": true
}
```

Annotation Index (for fast lookup by result_id):
```json
{
  "2026-04-16:clip_001:result_0:20260416T143205Z": ["ann_00001"],
  "2026-04-16:clip_002:result_1:20260416T143206Z": ["ann_00002", "ann_00003"]
}
```

#### 3.2.5 Run Manifest (Output)
Stored in `outputs/visualizations/<run_dir>/manifest.json`.

```json
{
  "run_id": "2026-04-16_results_run1",
  "created": "2026-04-16T14:25:00Z",
  "inference_results_file": "inference_results/results_2026-04-16_run1.json",
  "total_results": 100,
  "total_rendered": 100,
  "output_directory": "outputs/visualizations/2026-04-16_results_run1/",
  "render_config": {
    "overlay_style": "dual_grid",
    "well_radius_px": 18,
    "include_metadata_text": true,
    "include_fps_counter": true
  },
  "summary_statistics": {
    "accuracy_mean": 0.92,
    "accuracy_std": 0.15,
    "false_positive_count": 8,
    "false_negative_count": 6,
    "total_render_time_s": 1200.5
  },
  "clips_rendered": [
    {
      "result_index": 0,
      "clip_id": "clip_001",
      "metadata_file": "result_00_clip_001.json"
    },
    ...
  ]
}
```

#### 3.2.6 Regression Suite (Output)
Stored in `outputs/regression_suites/suite_vX.json`.

```json
{
  "suite_id": "suite_v1.0",
  "created": "2026-04-16T16:00:00Z",
  "description": "Regression test suite: 30 annotated clips with QA sign-off",
  "total_clips": 30,
  "criteria": {
    "only_annotated": true,
    "annotation_tags": ["regression_tested"],
    "min_severity": "low"
  },
  "clips": [
    {
      "clip_id": "clip_001",
      "result_index": 0,
      "persistent_id": "2026-04-16:clip_001:result_0:20260416T143205Z",
      "expected_wells": [
        {"well_row": "A", "well_column": 1}
      ],
      "annotation": "QA approved—exact match with ground truth"
    },
    ...
  ]
}
```

#### 3.2.7 Latency Profile (Output)
Stored in `outputs/analyses/latency_profile_YYYYMMDD.json`.

```json
{
  "profile_date": "2026-04-16",
  "total_clips": 100,
  "frames_total": 7200,
  "frames_by_latency_bucket": {
    "0_to_50ms": 6800,
    "50_to_100ms": 350,
    "100_to_200ms": 40,
    "over_200ms": 10
  },
  "percentiles": {
    "p50": 8.5,
    "p95": 52.3,
    "p99": 125.6
  },
  "sla": "100ms",
  "sla_violations": 50,
  "sla_compliance_percent": 99.3,
  "per_clip": [
    {
      "clip_id": "clip_001",
      "mean_frame_latency_ms": 10.2,
      "max_frame_latency_ms": 87.5,
      "sla_violations": 0
    },
    ...
  ]
}
```

#### 3.2.8 Strangest Results (Output)
Stored in `outputs/analyses/strangest_results.json`.

```json
{
  "analysis_date": "2026-04-16",
  "total_clips_analyzed": 100,
  "anomaly_metric": "confidence_correlation_delta",
  "top_k": 5,
  "results": [
    {
      "rank": 1,
      "clip_id": "clip_042",
      "result_index": 41,
      "anomaly_score": 0.87,
      "interpretation": "High row confidence (0.98) but low column confidence (0.32) — row detection reliable, column detection uncertain",
      "max_row_prob": 0.98,
      "max_col_prob": 0.32,
      "prediction": [{"well_row": "C", "well_column": 4}],
      "ground_truth": [{"well_row": "C", "well_column": 4}],
      "correct": true,
      "persistent_id": "2026-04-16:clip_042:result_41:..."
    },
    ...
  ]
}
```

---

## 4. Phase 1: CLI Reference

### 4.1 Tool Location & Invocation

**Path:** `tools/visualizer.py`

**Invocation:**
```bash
python tools/visualizer.py <command> [options]
```

All commands support `--help` for detailed usage:
```bash
python tools/visualizer.py --help
python tools/visualizer.py render --help
python tools/visualizer.py annotate --help
```

### 4.2 Commands

> **Implementation Status (Phase 1):**
> - `render` — **Implemented** in `tools/visualizer.py`
> - `rank` — **Implemented** in `tools/visualizer.py`
> - `annotate` — **Implemented** in `tools/visualizer.py`
> - `heatmap` — **Implemented** in `tools/visualizer.py`
> - `latency` — *Design only (Milestone 2)*
> - `regression` — *Design only (Milestone 3)*
> - `match` — *Design only (Milestone 2)*

#### 4.2.1 `render` — Render Predictions on Videos

**Purpose:** Visualize inference results by overlaying predictions on dual-view video.

**Signature:**
```bash
python tools/visualizer.py render \
  --input <INPUT_SPEC> \
  --labels <LABELS_JSON> \
  --output-dir <OUTPUT_DIR> \
  [--well-radius <PIXELS>] \
  [--overlay-style <STYLE>] \
  [--include-metadata] \
  [--include-latency] \
  [--video-pattern <PATTERN>] \
  [--exclude-indices <INDICES>] \
  [--workers <N>]
```

**Input Specification (`--input`):**

Can be one of:
1. **Inference results JSON file:**
   ```bash
   --input inference_results/results_2026-04-16_run1.json
   ```
   Renders all results in the file.

2. **Inference results with specific indices:**
   ```bash
   --input inference_results/results_2026-04-16_run1.json::0,5,10-15,20
   ```
   Renders results at indices 0, 5, 10 through 15, and 20.

3. **Single video clip (by clip_id):**
   ```bash
   --input clip_001
   ```
   Searches for clip_001_FPV.mp4 and clip_001_Topview.mp4 in the default video directory.

4. **List of clips:**
   ```bash
   --input clip_001,clip_002,clip_005
   ```

5. **Video list file:**
   ```bash
   --input @video_list.txt
   ```
   File contains one clip_id per line.

**Labels and Video Search:**

- `--labels` (required): Path to labels.json for ground truth.
- `--video-pattern` (optional): Glob pattern for video search. Default: `data/pipette_well_dataset/{clip_id}_*.mp4`
- Videos are auto-discovered by matching clip_id from inference results or input spec.

**Rendering Options:**

- `--output-dir` (default: `outputs/visualizations/TIMESTAMP_auto/`): Where to save rendered clips.
- `--well-radius` (default: 18): Circle radius in pixels for well overlay.
- `--overlay-style` (default: `dual_grid`):
  - `dual_grid`: 96-well grid overlay on both FPV and top-view.
  - `topview_only`: Grid overlay only on top-view.
  - `outline_only`: Outline predicted wells, no grid.
- `--include-metadata`: Overlay text: clip_id, result_index, result_index, persistent_id, timestamp.
- `--include-latency`: Overlay per-frame latency bar (requires metadata in results).
- `--exclude-indices <INDICES>`: Skip specified result indices (format: `0,5,10-15`).
- `--workers` (default: 1): Parallel rendering threads.

**Output:**

For each rendered clip:
- `result_XX_clip_YYY.mp4` — Video file.
- `result_XX_clip_YYY.json` — Metadata (see schema 3.2.3).

Plus:
- `manifest.json` — Run summary.
- `STDOUT`: Line-by-line progress.

**Example:**
```bash
# Render all results from a run
python tools/visualizer.py render \
  --input inference_results/results_2026-04-16_run1.json \
  --labels data/pipette_well_dataset/labels.json \
  --output-dir outputs/visualizations/2026-04-16_results_run1/ \
  --include-metadata \
  --workers 4

# Render specific indices
python tools/visualizer.py render \
  --input inference_results/results_2026-04-16_run1.json::0-19 \
  --labels data/pipette_well_dataset/labels.json \
  --output-dir outputs/visualizations/top_20/

# Render single clip with custom styling
python tools/visualizer.py render \
  --input clip_001 \
  --labels data/pipette_well_dataset/labels.json \
  --output-dir outputs/visualizations/debug_clip_001/ \
  --overlay-style outline_only \
  --well-radius 24
```

---

#### 4.2.2 `rank` — Rank Results by Accuracy Metric

**Purpose:** Sort results by accuracy (best, worst, strangest) and render top-k clips.

**Signature:**
```bash
python tools/visualizer.py rank \
  --input <RESULTS_JSON> \
  --labels <LABELS_JSON> \
  --metric <METRIC> \
  --top-k <K> \
  --output-dir <OUTPUT_DIR> \
  [--metric-config <CONFIG>] \
  [--render-only] \
  [--output-ranking-json] \
  [--concat-video] \
  [--workers <N>]
```

**Metrics:**

- `accuracy` (default): Exact match (all predicted wells match ground truth). Higher is better.
- `hamming_distance`: Symmetric difference (false positives + false negatives). Lower is better.
- `iou`: Intersection over Union. Higher is better.
- `confidence_mean`: Mean (max_row_prob, max_col_prob). Higher is better.
- `anomaly_score`: Confidence correlation delta (see section 3.3.4). Higher = more anomalous.

**Metric Config (`--metric-config`):**

JSON dict with metric-specific parameters. Examples:

```bash
# For anomaly_score: weighting of row/col correlation
--metric-config '{"weight_correlation": 0.5, "weight_uncertainty": 0.5}'

# For hamming_distance: cost weights for FP vs. FN
--metric-config '{"cost_fp": 1.0, "cost_fn": 1.5}'
```

**Ranking Behavior:**

- `best`: Sort by metric descending (e.g., high accuracy, high confidence).
- `worst`: Sort by metric ascending (e.g., low accuracy, high error).
- `strangest`: Sort by anomaly_score descending.

Inferred from context or explicit via:
```bash
python tools/visualizer.py rank --input ... --labels labels.json --mode best --top 10
```

**Options:**

- `--render-only`: Generate ranking JSON but don't render videos (faster, ~1–2s per 100 results).
- `--output-ranking-json`: Save ranking results as JSON (default: always saved to `outputs/analyses/`).
- `--concat-video`: After rendering top-k clips, concatenate into a single output video (with chapter markers, result index, timestamps).
- `--workers` (default: 1): Parallel rendering threads.

**Output:**

- `ranking_<metric>_<top_k>_TIMESTAMP.json` — Ranking results (see schema 3.2.8).
- Individual rendered videos (if not `--render-only`).
- Concatenated video `<metric>_top_<k>_concat.mp4` (if `--concat-video`).

**Example:**
```bash
# Render best 10 detections
python tools/visualizer.py rank \
  --input inference_results/results_2026-04-16_run1.json \
  --labels data/pipette_well_dataset/labels.json \
  --metric accuracy \
  --ranking best \
  --top-k 10 \
  --output-dir outputs/analyses/best_10/ \
  --concat-video

# Find 5 strangest results (no rendering, just ranking)
python tools/visualizer.py rank \
  --input inference_results/results_2026-04-16_run1.json \
  --labels data/pipette_well_dataset/labels.json \
  --metric anomaly_score \
  --top-k 5 \
  --render-only

# Worst 20 by Hamming distance, save ranking
python tools/visualizer.py rank \
  --input inference_results/results_2026-04-16_run1.json \
  --labels data/pipette_well_dataset/labels.json \
  --metric hamming_distance \
  --ranking worst \
  --top-k 20 \
  --output-dir outputs/analyses/worst_20/ \
  --output-ranking-json
```

---

#### 4.2.3 `annotate` — Add QA Notes to Results

**Purpose:** Create, update, or list QA annotations linked to specific results.

**Signature:**
```bash
python tools/visualizer.py annotate \
  --action <ACTION> \
  [--persistent-id <ID>] \
  [--clip-id <CLIP_ID>] \
  [--result-index <INDEX>] \
  [--run <RUN_ID>] \
  [--text <TEXT>] \
  [--category <CATEGORY>] \
  [--author <AUTHOR>] \
  [--tags <TAGS>] \
  [--severity <SEVERITY>] \
  [--query <QUERY>]
```

**Actions:**

1. **`create`** — Add a new annotation.
   ```bash
   python tools/visualizer.py annotate --action create \
     --persistent-id "2026-04-16:clip_001:result_0:20260416T143205Z" \
     --text "Pipette reflection caused false positive in E5" \
     --category "false_positive" \
     --author "qa_lead" \
     --tags "reflection_artifact,optical_issue" \
     --severity "medium"
   ```

   Or use clip_id + result_index (tool looks up persistent_id):
   ```bash
   python tools/visualizer.py annotate --action create \
     --clip-id "clip_001" \
     --result-index 0 \
     --run "2026-04-16_results_run1" \
     --text "Exact match, approved for regression suite" \
     --tags "regression_tested" \
     --severity "info"
   ```

2. **`update`** — Modify an existing annotation.
   ```bash
   python tools/visualizer.py annotate --action update \
     --persistent-id "2026-04-16:clip_001:result_0:20260416T143205Z" \
     --text "Updated: reflection issue will be fixed in v2.0 design" \
     --severity "resolved"
   ```

3. **`list`** — Query annotations.
   ```bash
   # List all annotations for a result
   python tools/visualizer.py annotate --action list \
     --persistent-id "2026-04-16:clip_001:result_0:20260416T143205Z"

   # List all annotations for a clip
   python tools/visualizer.py annotate --action list \
     --clip-id "clip_001" \
     --run "2026-04-16_results_run1"

   # List all annotations with a specific tag
   python tools/visualizer.py annotate --action list \
     --query "tags:reflection_artifact"

   # List recent annotations (last 7 days)
   python tools/visualizer.py annotate --action list \
     --query "created:>2026-04-09"
   ```

4. **`delete`** — Remove an annotation (admin only).
   ```bash
   python tools/visualizer.py annotate --action delete \
     --persistent-id "2026-04-16:clip_001:result_0:20260416T143205Z"
   ```

**Categories (Predefined):**

- `false_positive` — Prediction includes extra wells not in ground truth.
- `false_negative` — Prediction missing wells in ground truth.
- `correct` — Exact match.
- `optical_issue` — Light reflection, glare, or optical artifact.
- `calibration_issue` — Plate alignment or camera calibration issue.
- `temporal_issue` — Frame timing or synchronization issue.
- `edge_case` — Unusual dispensing pattern or edge case scenario.
- `design_feedback` — Suggestion for model/optical design improvement.

**Tags (Free-form):**

User-defined comma-separated list. Examples:
- `regression_tested`, `qa_approved`
- `reflection_artifact`, `glare_issue`
- `design_v2.0`, `optical_improvement`

**Severity:**

- `info` — Informational note.
- `low`, `medium`, `high` — Issue severity.
- `resolved`, `wontfix`, `duplicate` — Status.

**Output:**

For `create`, `update`:
```
✓ Annotation created: ann_00042
  Result ID: 2026-04-16:clip_001:result_0:20260416T143205Z
  Clip ID: clip_001
  Author: qa_lead
  Timestamp: 2026-04-16T15:10:22Z
```

For `list`:
```
Found 3 annotations for result 2026-04-16:clip_001:result_0:...:
  [ann_00001] 2026-04-16 15:10:22 | qa_lead  | false_positive | "Pipette nozzle reflection..."
  [ann_00002] 2026-04-16 15:12:15 | qa_engineer | optical_issue | "Glare from ring light..."
  [ann_00003] 2026-04-16 15:14:30 | engineer   | design_feedback | "New geometry will reduce reflections..."
```

---

#### 4.2.4 `heatmap` — Generate Error Heatmap

**Purpose:** Aggregate errors across all clips and visualize as overlay on 96-well grid.

**Signature:**
```bash
python tools/visualizer.py heatmap \
  --input <RESULTS_JSON> \
  --labels <LABELS_JSON> \
  --metric <METRIC> \
  --output-dir <OUTPUT_DIR> \
  [--normalize] \
  [--colormap <COLORMAP>] \
  [--render-as-video]
```

**Metrics:**

- `error_count` (default): Number of errors per well.
- `false_positive_rate`: FP / (TP + FP) per well.
- `false_negative_rate`: FN / (TP + FN) per well.
- `confusion_rows`: Which rows are confused with which (heatmap of row-pair errors).
- `confusion_cols`: Which columns are confused with which.

**Normalize:**

- `none` (default): Raw counts.
- `log`: Log-scale (useful if error counts are highly skewed).
- `percent`: Percentage of total errors.

**Colormap:**

- `viridis` (default), `plasma`, `hot`, `coolwarm`, `RdYlGn_r` (red = high error)

**Output:**

- `heatmap_<metric>.json` — Aggregated data (grid of values, 8×12 for well layout, 8×8 for confusion matrices).
- `heatmap_<metric>.png` — Image file suitable for static analysis.
- `heatmap_<metric>_video.mp4` (if `--render-as-video`): Video showing wells being animated with error counts, useful for presentation.

**Example:**
```bash
# Error count heatmap
python tools/visualizer.py heatmap \
  --input inference_results/results_2026-04-16_run1.json \
  --labels data/pipette_well_dataset/labels.json \
  --metric error_count \
  --output-dir outputs/analyses/heatmaps/ \
  --normalize log \
  --colormap hot

# False positive rate heatmap
python tools/visualizer.py heatmap \
  --input inference_results/results_2026-04-16_run1.json \
  --labels data/pipette_well_dataset/labels.json \
  --metric false_positive_rate \
  --output-dir outputs/analyses/heatmaps/ \
  --normalize percent

# Row confusion matrix
python tools/visualizer.py heatmap \
  --input inference_results/results_2026-04-16_run1.json \
  --labels data/pipette_well_dataset/labels.json \
  --metric confusion_rows \
  --output-dir outputs/analyses/heatmaps/
```

---

#### 4.2.5 `latency` — Profile Per-Frame Inference Latency

**Purpose:** Extract latency from inference metadata and generate latency profile.

**Signature:**
```bash
python tools/visualizer.py latency \
  --input <RESULTS_JSON> \
  --sla <SLA_MS> \
  --output-dir <OUTPUT_DIR> \
  [--render-timeline-video] \
  [--clips <CLIP_IDS>]
```

**SLA (Service-Level Agreement):**

Default: 100ms per frame. Violations are frames exceeding this threshold.

**Output:**

- `latency_profile_YYYYMMDD.json` — Aggregated stats (see schema 3.2.7).
- `latency_per_clip_YYYYMMDD.json` — Per-clip breakdown.
- `latency_timeline_<clip_id>.mp4` (if `--render-timeline-video`): Video with latency timeline bar overlay.

**Example:**
```bash
python tools/visualizer.py latency \
  --input inference_results/results_2026-04-16_run1.json \
  --sla 100 \
  --output-dir outputs/analyses/latency/ \
  --render-timeline-video \
  --clips clip_001,clip_002,clip_003
```

---

#### 4.2.6 `regression` — Build or Run Regression Test Suite

**Purpose:** Create a curated set of annotated clips for regression testing across model versions.

**Signature:**
```bash
python tools/visualizer.py regression \
  --action <ACTION> \
  [--suite-name <NAME>] \
  [--clip-ids <LIST>] \
  [--query <QUERY>] \
  [--new-results <JSON>] \
  [--output-dir <OUTPUT_DIR>]
```

**Actions:**

1. **`create`** — Build regression suite from annotated clips.
   ```bash
   # Create suite from all annotated clips with tag "regression_tested"
   python tools/visualizer.py regression --action create \
     --suite-name "regression_suite_v1.0" \
     --query "tags:regression_tested"

   # Create from explicit list of clips
   python tools/visualizer.py regression --action create \
     --suite-name "regression_suite_v1.0" \
     --clip-ids clip_001,clip_002,clip_005,clip_010 \
     --output-dir outputs/regression_suites/
   ```

2. **`run`** — Compare new inference results against regression suite.
   ```bash
   python tools/visualizer.py regression --action run \
     --suite-name "regression_suite_v1.0" \
     --new-results inference_results/results_2026-04-16_new_model.json \
     --output-dir outputs/regression_suites/
   ```
   Output: `regression_suite_v1.0_log_TIMESTAMP.json` with per-clip pass/fail.

3. **`report`** — Generate summary report of suite performance.
   ```bash
   python tools/visualizer.py regression --action report \
     --suite-name "regression_suite_v1.0" \
     --output-dir outputs/regression_suites/
   ```

**Output (for `run`):**

```json
{
  "suite_id": "regression_suite_v1.0",
  "run_timestamp": "2026-04-16T17:00:00Z",
  "new_results_file": "inference_results/results_2026-04-16_new_model.json",
  "total_clips": 30,
  "passed": 29,
  "failed": 1,
  "pass_rate_percent": 96.7,
  "failures": [
    {
      "clip_id": "clip_042",
      "expected": [{"well_row": "C", "well_column": 4}],
      "got": [{"well_row": "C", "well_column": 4}, {"well_row": "C", "well_column": 5}],
      "error_type": "false_positive"
    }
  ]
}
```

---

#### 4.2.7 `match` — Auto-Match Results to Videos (Utility)

**Purpose:** Verify that inference results can be matched to videos; diagnose missing videos.

**Signature:**
```bash
python tools/visualizer.py match \
  --input <RESULTS_JSON> \
  --video-pattern <PATTERN> \
  [--verbose]
```

**Output:**

```
✓ Matched 100/100 results
  - clip_001: ✓ found FPV, ✓ found Topview
  - clip_002: ✓ found FPV, ✓ found Topview
  ...
  - clip_050: ✗ MISSING Topview (searched: data/pipette_well_dataset/clip_050_Topview.mp4)
```

---

### 4.3 Global Options

All commands support:

- `--help`: Print command-specific help.
- `--verbose` / `-v`: Verbose logging (DEBUG level).
- `--quiet` / `-q`: Suppress non-critical output.
- `--config <CONFIG_JSON>`: Load default parameters from JSON file (overridable by CLI args).
- `--dry-run`: Simulate operations, don't write files.
- `--log-file <PATH>`: Write logs to file in addition to STDOUT.

**Global Config File (`visualizer.config.json`):**

```json
{
  "data_dir": "data/pipette_well_dataset/",
  "output_dir": "outputs/",
  "labels_path": "data/pipette_well_dataset/labels.json",
  "video_pattern": "data/pipette_well_dataset/{clip_id}_*.mp4",
  "render": {
    "well_radius_px": 18,
    "overlay_style": "dual_grid",
    "include_metadata": true,
    "fps": 30
  },
  "latency_sla_ms": 100,
  "workers": 4
}
```

Use with:
```bash
python tools/visualizer.py render --config visualizer.config.json --input ...
```

---

## 5. Phase 1: Rendering Pipeline

### 5.1 Overview

The rendering pipeline takes inference results, ground truth labels, video frames, and rendering config, then outputs annotated MP4 files with well grid overlay.

**High-level flow:**

```
Input: results.json + labels.json + videos
  ↓
[Match] results → videos by clip_id
  ↓
[Load] ground truth labels for each clip
  ↓
For each result:
  - Load video frames (FPV + Topview)
  - Decode prediction wells from result JSON
  - Compute color codes: green (correct), red (FP), blue (FN)
  - For each frame:
    * Resize/normalize frame
    * Draw 96-well grid (if `overlay_style` enables)
    * Draw predicted wells with assigned colors
    * Draw ground truth wells (outline or secondary color)
    * Optionally overlay: clip_id, result_index, persistent_id, timestamp, latency, inference metadata
  - Encode output video (MP4, same fps as input)
  - Write metadata JSON
  ↓
Output: result_XX_clip_YYY.mp4 + result_XX_clip_YYY.json
```

### 5.2 Grid Overlay Geometry

**96-Well Plate Layout:**

- 8 rows (A–H)
- 12 columns (1–12)
- Standard well spacing: ~9mm between well centers
- Standard plate dimensions: ~127.5mm × ~85.5mm
- Well diameter: ~6.5mm (radius ~3.25mm)

**Top-View Overlay:**

1. **Plate Boundary Detection** (if not manually calibrated):
   - Detect white edge of well plate in frame
   - Estimate plate bounding box (homography from known plate geometry)
   - Fallback: User-provided calibration JSON with plate corners

2. **Grid Generation:**
   - 8 × 12 grid of well centers, evenly spaced across plate bounding box
   - Each well center is a point in frame coordinates

3. **Well Rendering:**
   - For each well (row, column):
     - Compute frame position from grid
     - Determine color:
       - **Green (#00FF00):** Predicted AND in ground truth (True Positive)
       - **Red (#FF0000):** Predicted NOT in ground truth (False Positive)
       - **Blue (#0000FF):** NOT predicted but in ground truth (False Negative)
       - **Gray (#CCCCCC):** Neither predicted nor in ground truth (True Negative, optional)
     - Draw circle at well center with radius `--well-radius` pixels
     - Optional: Draw well label ("A1", "B2", etc.)

**FPV Overlay:**

- FPV is pipette-centric perspective view
- Well grid overlay on FPV is **approximate** because:
  - Pipette tip is foreground; well plate is background
  - Perspective distortion makes grid non-uniform
- Options:
  1. Disable grid on FPV; show only predicted well overlay (region of uncertainty)
  2. Draw grid as reference (less accurate but contextual)
  3. Use optical flow to track well positions frame-by-frame

**Recommendation:** For Phase 1, disable grid on FPV; draw grid on top-view only. Annotate FPV with a marker indicating where the model predicts the pipette will dispense (estimated from predicted wells + plate geometry).

### 5.3 Overlay Styles

#### 5.3.1 `dual_grid` (Default)

- **Top-view:** Full 96-well grid with color-coded wells (TP/FP/FN).
- **FPV:** Estimated dispense location marked (e.g., red dot or circle).
- **Appearance:** Transparent grid overlay (~30% opacity) so video is still visible.

```
Top-view:                         FPV:
[well grid with colors]           [pipette video + red circle]
```

#### 5.3.2 `topview_only`

- **Top-view:** Grid (as above).
- **FPV:** Raw video, no overlay.

#### 5.3.3 `outline_only`

- **Top-view:** No grid; only outlines of predicted/ground truth wells.
- **FPV:** Single marker for dispense location.

---

### 5.4 Metadata Text Overlay

If `--include-metadata`, render text on video:

```
Top-left corner:
[Clip ID: clip_001]
[Result: 0 / 100]
[Persistent ID: 2026-04-16:clip_001:result_0:20260416T143205Z]
[Timestamp: 2026-04-16T14:32:05Z]

Bottom-left corner:
[Inference: 0.342s | Confident: ✓ | Threshold: 0.5]
[Row Prob: 0.98 | Col Prob: 0.95]

Bottom-right corner:
[Accuracy: Exact Match]
[Ground Truth: [A1]  |  Prediction: [A1]]
```

### 5.5 Implementation Considerations

**Frame-by-frame Processing:**

- Load video with `cv2.VideoCapture` or `ffmpeg-python`
- For each frame:
  1. Decode frame
  2. Resize to standard resolution (e.g., 1920×1080)
  3. Draw overlays using `cv2.circle`, `cv2.putText`, etc.
  4. Encode to MP4 with ffmpeg-python (or OpenCV encoder)

**Performance:**

- Typical video: 72 frames @ 1920×1080 @ 30fps = ~2.4s
- Rendering 100 clips sequentially: ~2 min
- With `--workers 4`: ~30s

**Video Encoding:**

- Output codec: H.264 (MP4)
- Bitrate: 10 Mbps (quality/size tradeoff)
- Frame rate: Match input (typically 30fps)

**Grid Calibration:**

- Store calibration data (plate corners in frame coordinates) in a JSON file per video
- If no calibration provided, auto-detect using edge detection and Hough transform
- Cache calibration for repeated renders

---

## 6. Phase 1: Annotation & Notes System

### 6.1 Annotation Storage

**Primary File:** `outputs/annotations/annotations.json`

- Append-only JSON array
- One annotation object per QA note
- Immutable (annotations are never overwritten, only deleted or superseded)

**Index File:** `outputs/annotations/annotation_index.json`

- Maps `persistent_id` → list of annotation IDs
- Enables O(1) lookup of annotations for a given result
- Rebuilt on every `annotate --action create` or `delete`

### 6.2 Annotation Lifecycle

**Create:**
```bash
python tools/visualizer.py annotate --action create \
  --persistent-id "2026-04-16:clip_001:result_0:20260416T143205Z" \
  --text "Exact match, approved for regression" \
  --tags "regression_tested,qa_approved"
```

**Update:** Only the `text` and `severity` fields can be updated (for ammendments/corrections).

```bash
python tools/visualizer.py annotate --action update \
  --persistent-id "2026-04-16:clip_001:result_0:20260416T143205Z" \
  --text "Updated: Discovered reflection artifact in frame 45, investigating..."
```

**Query by Result ID:**
```bash
python tools/visualizer.py annotate --action list \
  --persistent-id "2026-04-16:clip_001:result_0:20260416T143205Z"
```

Output: All annotations for that result, with author, timestamp, and full text.

### 6.3 Integration with Regression Testing

**Workflow:**

1. Data Scientist generates inference results + renders clips.
2. QA team reviews rendered clips and annotates with approvals/issues.
3. QA lead compiles approved clips into regression suite:
   ```bash
   python tools/visualizer.py regression --action create \
     --suite-name "regression_suite_v1.0" \
     --query "tags:qa_approved,regression_tested"
   ```
4. New model version runs inference on same clips.
5. Compare against suite:
   ```bash
   python tools/visualizer.py regression --action run \
     --suite-name "regression_suite_v1.0" \
     --new-results inference_results/new_model_results.json
   ```
6. Output: Pass/fail report; failures highlighted for review.

---

## 7. Phase 1: Ranking & Filtering

### 7.1 Accuracy Metrics

#### 7.1.1 Exact Match

```
match(pred, gt) = 1 if set(pred) == set(gt) else 0
```

Range: [0, 1]. Discrete; doesn't penalize magnitude of error.

#### 7.1.2 Hamming Distance

```
hamming(pred, gt) = |pred| + |gt| - 2|pred ∩ gt|
                  = |pred \ gt| + |gt \ pred|
                  = (false positives) + (false negatives)
```

Range: [0, ∞). Lower is better. Counts total error wells.

#### 7.1.3 Intersection over Union (IoU)

```
iou(pred, gt) = |pred ∩ gt| / |pred ∪ gt|
              = TP / (TP + FP + FN)
```

Range: [0, 1]. Higher is better. Standard object detection metric.

#### 7.1.4 Confidence Mean

```
confidence_mean(result) = mean(max_row_prob, max_col_prob)
```

Range: [0, 1]. Higher = more confident. Independent of ground truth; measures model self-confidence.

#### 7.1.5 Anomaly Score (Strangest Results)

**Phase 1 (Implemented):** Confidence-only formula (no ground truth required):

```
anomaly_score = |max_row_prob - max_col_prob| × 2.0
              + (1 - max_row_prob × max_col_prob) × 0.5
```

This allows anomaly scoring before labels are available and is used by the `rank --mode strangest` command.

**Phase 2 (Full Formula):** Incorporates ground truth for richer signal:

```
anomaly_score = |max_row_prob - max_col_prob|
             × (1 - correlation(row_probs, col_probs))
             × (1 - iou(pred, gt))
```

Intuition:
- High row confidence but low column confidence → anomalous
- Row and column predictions uncorrelated → anomalous
- Prediction doesn't match ground truth → anomalous

### 7.2 Ranking Workflow

**User Request:** "Generate top 20 worst detections"

**Steps:**

1. Load `results.json` (100 results)
2. Load `labels.json` (100 ground truths)
3. Match each result to ground truth by clip_id
4. Compute accuracy metric for each result
5. Sort ascending (for "worst")
6. Select top 20
7. Render videos (parallel)
8. Save ranking JSON with sorted results + metric values
9. Optionally concatenate videos

**Ranking JSON Output:**

```json
{
  "analysis_date": "2026-04-16",
  "total_clips_analyzed": 100,
  "metric": "hamming_distance",
  "ranking": "worst",
  "top_k": 20,
  "results": [
    {
      "rank": 1,
      "clip_id": "clip_042",
      "result_index": 41,
      "metric_value": 6,
      "prediction": [{"well_row": "A", "well_column": 1}, ...],
      "ground_truth": [{"well_row": "A", "well_column": 1}],
      "persistent_id": "2026-04-16:clip_042:result_41:..."
    },
    ...
  ]
}
```

---

## 8. Phase 2: Cloud Service Architecture (Design Only)

**Note:** This section is design-only. DO NOT implement in Phase 1.

### 8.1 Overview

Phase 2 replaces flat-file storage with a cloud-native stack:

- **Database:** PostgreSQL for results, annotations, metadata (queryable)
- **Storage:** S3/GCS for videos, rendered outputs
- **Compute:** GPU-enabled inference service (optional; can use Phase 1 offline)
- **API:** REST API for queries, filtering, annotation CRUD
- **Web UI:** Dashboard for browsing results, creating annotations, viewing heatmaps

### 8.2 Database Schema (PostgreSQL)

#### 8.2.1 `inference_runs`

```sql
CREATE TABLE inference_runs (
  run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name VARCHAR(255),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  model_checkpoint VARCHAR(255),
  model_version VARCHAR(50),
  threshold FLOAT,
  adaptive BOOLEAN,
  total_results INT,
  notes TEXT
);
```

#### 8.2.2 `results`

```sql
CREATE TABLE results (
  result_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  run_id UUID NOT NULL REFERENCES inference_runs(run_id),
  result_index INT NOT NULL,
  clip_id VARCHAR(100) NOT NULL,
  clip_id_fpv VARCHAR(100) NOT NULL,
  clip_id_topview VARCHAR(100) NOT NULL,
  wells_prediction JSONB NOT NULL,  -- Array of {well_row, well_column}
  wells_ground_truth JSONB NOT NULL,
  inference_time_s FLOAT,
  confidence BOOLEAN,
  max_row_prob FLOAT,
  max_col_prob FLOAT,
  accuracy_metric FLOAT,  -- Hamming, IoU, etc.
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE (run_id, result_index)
);

CREATE INDEX idx_results_clip_id ON results(clip_id);
CREATE INDEX idx_results_accuracy ON results(accuracy_metric);
CREATE INDEX idx_results_run_id ON results(run_id);
```

#### 8.2.3 `annotations`

```sql
CREATE TABLE annotations (
  annotation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  result_id UUID NOT NULL REFERENCES results(result_id),
  author VARCHAR(100),
  category VARCHAR(50),  -- false_positive, false_negative, etc.
  text TEXT NOT NULL,
  tags JSONB,  -- Array of strings
  severity VARCHAR(20),  -- info, low, medium, high, resolved, etc.
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  linked_to_run UUID REFERENCES inference_runs(run_id)
);

CREATE INDEX idx_annotations_result_id ON annotations(result_id);
CREATE INDEX idx_annotations_author ON annotations(author);
CREATE INDEX idx_annotations_category ON annotations(category);
```

#### 8.2.4 `regression_suites`

```sql
CREATE TABLE regression_suites (
  suite_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name VARCHAR(255) UNIQUE,
  description TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  criteria JSONB,  -- {only_annotated: bool, tags: [string], min_severity: string}
  total_clips INT
);

CREATE TABLE regression_suite_clips (
  suite_clip_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  suite_id UUID NOT NULL REFERENCES regression_suites(suite_id),
  result_id UUID NOT NULL REFERENCES results(result_id),
  expected_wells JSONB NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_suite_clips_suite_id ON regression_suite_clips(suite_id);
```

#### 8.2.5 `regression_runs`

```sql
CREATE TABLE regression_runs (
  run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  suite_id UUID NOT NULL REFERENCES regression_suites(suite_id),
  new_inference_run_id UUID NOT NULL REFERENCES inference_runs(run_id),
  executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  total_clips INT,
  passed INT,
  failed INT,
  pass_rate_percent FLOAT,
  failures JSONB  -- Array of {result_id, expected, got, error_type}
);

CREATE INDEX idx_regression_runs_suite_id ON regression_runs(suite_id);
```

### 8.3 S3/GCS Object Structure

```
s3://pipette-well-bucket/
├── videos/
│   ├── raw/
│   │   ├── clip_001_FPV.mp4
│   │   ├── clip_001_Topview.mp4
│   │   └── ...
│   └── rendered/
│       ├── {run_id}/
│       │   ├── result_00_clip_001.mp4
│       │   ├── result_05_clip_003.mp4
│       │   └── ...
│       └── {run_id}/
│
├── metadata/
│   ├── results/
│   │   ├── {run_id}_results.json
│   │   └── ...
│   ├── annotations/
│   │   ├── annotations_export_TIMESTAMP.json
│   │   └── ...
│   └── analyses/
│       ├── heatmap_error_count_{date}.json
│       ├── latency_profile_{date}.json
│       └── ...
│
└── labels/
    └── labels.json
```

### 8.4 REST API Design

**Base URL:** `https://api.pipette-well.internal/v1/`

#### 8.4.1 Inference Runs

```
GET /runs                              # List all inference runs
  ?limit=50&offset=0
  ?model_version=v2.1
  ?created_after=2026-04-10

GET /runs/{run_id}                     # Get run details

POST /runs                             # Register new inference run
  {
    "name": "2026-04-16_model_v2.1",
    "model_version": "v2.1",
    "threshold": 0.5,
    ...
  }
```

#### 8.4.2 Results

```
GET /runs/{run_id}/results             # List results for a run
  ?limit=50&offset=0
  ?accuracy_min=0.8
  ?accuracy_max=1.0
  ?metric=iou

GET /results/{result_id}               # Get single result

GET /clips/{clip_id}/results           # Find all results for a clip
  ?run_id={run_id}
```

#### 8.4.3 Annotations

```
GET /results/{result_id}/annotations   # List annotations for a result

POST /results/{result_id}/annotations  # Create annotation
  {
    "author": "qa_lead",
    "category": "false_positive",
    "text": "...",
    "tags": ["reflection_artifact"],
    "severity": "medium"
  }

PATCH /annotations/{annotation_id}     # Update annotation

DELETE /annotations/{annotation_id}    # Delete annotation
```

#### 8.4.4 Heatmap & Analysis

```
GET /analyses/heatmap                  # Get error heatmap
  ?metric=error_count
  ?run_id={run_id}
  ?normalize=log

GET /analyses/latency                  # Get latency profile
  ?run_id={run_id}
  ?sla_ms=100

GET /analyses/strangest               # Get anomaly-scored results
  ?run_id={run_id}
  ?top_k=5
```

#### 8.4.5 Regression Suites

```
GET /regression-suites                 # List all suites

POST /regression-suites                # Create new suite
  {
    "name": "regression_v1.0",
    "description": "...",
    "criteria": {
      "only_annotated": true,
      "tags": ["regression_tested"]
    }
  }

GET /regression-suites/{suite_id}/clips   # List clips in suite

POST /regression-suites/{suite_id}/run    # Execute regression test
  {
    "new_inference_run_id": "{run_id}"
  }

GET /regression-suites/{suite_id}/runs    # List test runs for suite
```

### 8.5 Web UI (Mockup)

**Home Page:**
- List recent inference runs with summary stats (accuracy, FP/FN counts)
- Link to "Browse Results", "Create Annotation", "View Heatmaps"

**Browse Results:**
- Filterable table: clip_id, accuracy, result_index, created_at
- Click row → view rendered video (player) + metadata
- Inline annotation creation
- Quick actions: "Mark for Regression", "Tag as False Positive"

**Heatmap View:**
- Interactive 96-well grid showing error counts
- Hover over well → see example errors, list of clips with that error
- Download as PNG/SVG

**Regression Suite Manager:**
- List suites
- View clips in suite (thumbnail grid)
- Run new inference against suite
- View pass/fail results with diff highlighting

**Annotation Dashboard:**
- Filter annotations by: category, author, tags, date range, severity
- Export as CSV/JSON for external analysis

---

## 9. Integration Points

### 9.1 With `inference.py`

**Current Flow:**

```
inference.py --fpv X.mp4 --topview Y.mp4 --output results.json
```

**Integration:**

The visualizer reads `results.json` directly. No changes to `inference.py` needed, but consider:

1. **Include latency metadata** (already done in Phase 1 results schema)
2. **Per-frame latency** (optional; would require modifying `inference.py` to emit frame-level timing)
3. **Raw logits** (optional; would help compute alternative metrics downstream)

**No Breaking Changes Required** for Phase 1.

### 9.2 With `train.py`

**No direct integration** for Phase 1. Visualizer consumes inference results, not training pipeline.

For Phase 2: Could expose model checkpoint endpoints to compute per-layer latency or intermediate representations.

### 9.3 With Dataset / Labels

**Dependency:** Visualizer requires `labels.json` and video files to be in known locations.

**Current Assumption:**
- Labels: `data/pipette_well_dataset/labels.json`
- Videos: `data/pipette_well_dataset/clip_XXX_{FPV,Topview}.mp4`

**Flexibility:**
- Use `--video-pattern` to override search path
- Use `--labels` to specify custom labels file

---

## 10. Testing Strategy

### 10.1 Unit Tests

**Test Rendering:**
- Load dummy inference result + label
- Render single frame
- Check that well circles are drawn at expected positions
- Check color assignment (TP/FP/FN)

**Test Accuracy Metrics:**
- Test each metric (exact match, Hamming, IoU, confidence, anomaly) with known inputs
- Verify output range and ordering

**Test Annotation Storage:**
- Create, update, delete annotations
- Verify JSON file integrity
- Test index rebuild

**Test Ranking:**
- Create list of results with known metrics
- Rank ascending/descending
- Verify sorted order

### 10.2 Integration Tests

**Test Render Pipeline End-to-End:**
- Create fake inference result JSON
- Run `python tools/visualizer.py render --input test_results.json --labels test_labels.json`
- Verify output video files exist and have expected properties (fps, resolution, duration)

**Test Annotation Workflow:**
- Create annotation via CLI
- List annotations
- Update annotation
- Verify file consistency

**Test Ranking + Render:**
- Run `rank` command with `--concat-video`
- Verify concatenated output has expected structure (chapters, timestamps)

### 10.3 Manual QA

**Sanity Checks:**
- Render 1–2 clips manually
- Visually inspect overlay: grid position, colors, metadata text alignment
- Verify latency bars align with frame timeline
- Check that persistent_id is stable across re-renders

**Edge Cases:**
- Clip with no predicted wells (empty prediction)
- Clip with all wells predicted (high false positive rate)
- Clip with mismatched video length (FPV 72 frames, Topview 70 frames)
- Missing ground truth label for a clip

---

## 11. Implementation Roadmap (Phase 1)

### Milestone 1: Core Rendering (Week 1)

- [ ] Implement `render` command
  - Load inference results JSON
  - Match to videos by clip_id
  - Load labels
  - Compute accuracy metrics
  - Draw grid overlay on top-view
  - Encode output MP4
  - Write metadata JSON

- [ ] Create directory structure and JSON schemas
- [ ] Unit tests for rendering and metrics

### Milestone 2: Ranking & Analysis (Week 2)

- [ ] Implement `rank` command
- [ ] Implement `heatmap` command
- [ ] Implement `latency` command
- [ ] Integration tests for end-to-end workflows

### Milestone 3: Annotations & Regression (Week 2–3)

- [ ] Implement `annotate` command
- [ ] Implement `regression` command
- [ ] Test annotation storage and querying

### Milestone 4: Documentation & Polish (Week 3)

- [ ] CLI help text for all commands
- [ ] User guide with examples
- [ ] Example configuration file
- [ ] Test on real inference outputs

---

## 12. Success Criteria

### Phase 1 Success

- [ ] All CLI commands work as specified
- [ ] User can render a clip in <10 seconds
- [ ] User can rank 100 results and render top-20 in <2 minutes
- [ ] Annotation creation/querying is fast (<100ms)
- [ ] Metadata JSON is valid and parseable
- [ ] Rendered videos are identical across multiple runs (deterministic)
- [ ] Grid overlay is geometrically accurate (verified manually on 5 sample clips)

### Phase 2 Success

- [ ] Database schema is normalized and indexed
- [ ] REST API has <100ms response time for typical queries
- [ ] Web UI is intuitive and responsive
- [ ] Regression suite can be executed in <5 minutes for 30 clips

---

## 13. Appendices

### A. Example Workflows

#### A1. User Visualizes Clip by ID

```bash
python tools/visualizer.py render \
  --input clip_001 \
  --labels data/pipette_well_dataset/labels.json \
  --output-dir outputs/visualizations/clip_001_debug/ \
  --include-metadata

# Output:
# outputs/visualizations/clip_001_debug/
#   ├── result_00_clip_001.mp4
#   ├── result_00_clip_001.json
#   └── manifest.json
```

#### A2. Data Scientist Generates Worst 20 Detections with Concat Video

```bash
python tools/visualizer.py rank \
  --input inference_results/results_2026-04-16_run1.json \
  --labels data/pipette_well_dataset/labels.json \
  --metric hamming_distance \
  --ranking worst \
  --top-k 20 \
  --output-dir outputs/analyses/worst_20/ \
  --concat-video

# Output:
# outputs/analyses/worst_20/
#   ├── ranking_hamming_distance_20_20260416T143205Z.json
#   ├── result_00_clip_042.mp4
#   ├── result_00_clip_042.json
#   ├── result_01_clip_015.mp4
#   ├── result_01_clip_015.json
#   ...
#   └── hamming_distance_top_20_concat.mp4  # Single video with 20 clips
```

#### A3. QA Team Creates Regression Suite

```bash
# First, annotate approved clips
python tools/visualizer.py annotate --action create \
  --clip-id "clip_001" --result-index 0 --run "2026-04-16_results_run1" \
  --text "Exact match, QA approved" \
  --tags "regression_tested,qa_approved"

# ... (repeat for 30 clips)

# Then build suite
python tools/visualizer.py regression --action create \
  --suite-name "regression_suite_v1.0" \
  --query "tags:qa_approved,regression_tested" \
  --output-dir outputs/regression_suites/

# Output: suite_v1.0.json (list of 30 clips with expected wells)

# New model version runs inference
python inference.py --fpv clip_001_FPV.mp4 --topview clip_001_Topview.mp4 > results_new.json

# ... (run inference on all 30 clips)

# Run regression test
python tools/visualizer.py regression --action run \
  --suite-name "regression_suite_v1.0" \
  --new-results inference_results/results_new_model.json \
  --output-dir outputs/regression_suites/

# Output:
# outputs/regression_suites/
#   └── regression_suite_v1.0_log_20260416T170000Z.json
#       {
#         "pass_rate_percent": 96.7,
#         "passed": 29,
#         "failed": 1,
#         "failures": [...]
#       }
```

#### A4. ML Scientist Profiles Latency

```bash
python tools/visualizer.py latency \
  --input inference_results/results_2026-04-16_run1.json \
  --sla 100 \
  --output-dir outputs/analyses/latency/ \
  --render-timeline-video

# Output:
# outputs/analyses/latency/
#   ├── latency_profile_20260416.json
#   │   {
#   │     "p99": 125.6,
#   │     "sla_compliance_percent": 99.3,
#   │     ...
#   │   }
#   ├── latency_timeline_clip_001.mp4  # Video with latency bars
#   ├── latency_timeline_clip_002.mp4
#   └── ...
```

#### A5. Data Scientist Finds Anomalous Results

```bash
python tools/visualizer.py rank \
  --input inference_results/results_2026-04-16_run1.json \
  --labels data/pipette_well_dataset/labels.json \
  --metric anomaly_score \
  --top-k 5 \
  --render-only

# Output: ranking_anomaly_score_5_TIMESTAMP.json (no videos rendered, just scores)

# Then render only the strangest ones
python tools/visualizer.py render \
  --input inference_results/results_2026-04-16_run1.json::41,42,43,44,45 \
  --labels data/pipette_well_dataset/labels.json \
  --output-dir outputs/analyses/strangest_5/
```

### B. Persistent ID Format

**Format:** `YYYYMMDD:clip_id:result_index:YYYYmmddTHHmmssZ`

**Example:** `2026-04-16:clip_001:result_0:20260416T143205Z`

**Purpose:**
- Stable reference across re-renders and database migrations
- Human-readable date and clip_id
- Unique key for annotation lookups

---

## 14. Questions for Refinement

1. **Plate Calibration:** Should the tool auto-detect well positions from image processing, or expect pre-calibrated coordinates? Recommendation: Auto-detect with optional calibration override.

2. **Multi-Well Displays:** Should rendering show both FPV and Topview side-by-side, or as separate files? Recommendation: Side-by-side for Phase 1.

3. **Performance SLA:** Should rendering of 100 clips complete in <5 minutes? Recommendation: Target <5 min with `--workers 4` on modern GPU.

4. **Regression Suite Scope:** Should regression suites include only exact-match clips, or allow tolerance (e.g., ±1 well)? Recommendation: Exact match for Phase 1; tolerance config option for Phase 2.

5. **QA Note Visibility:** Should annotations appear overlaid on rendered videos, or only in metadata? Recommendation: Metadata only for Phase 1; video overlay (text annotation) for Phase 2.

---

**End of Design Document**

---

## Glossary

- **Clip ID:** Unique identifier for a pair of synchronized videos (e.g., `clip_001`).
- **Persistent ID:** Stable, unique reference to a single result, format: `YYYYMMDD:clip_id:result_index:timestamp`.
- **FPV:** First-person view; pipette-mounted camera showing approach and dispensing.
- **Top-view:** Overhead camera showing the 96-well plate and pipette position.
- **Inference Run:** Set of inference results from a single model checkpoint on a batch of clips.
- **Ground Truth:** Human-labeled correct wells for a clip (from `labels.json`).
- **Regression Suite:** Curated set of clips with expected predictions, used to validate model changes.
- **Annotation:** Free-form text note linked to a result, created by QA team for documentation.
- **Metric:** Computed measure of prediction accuracy (Hamming, IoU, exact match, etc.).

