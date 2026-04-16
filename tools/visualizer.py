#!/usr/bin/env python3
"""
Pipette Well Detection — Visualization Tool (Phase 1: Flat Files)

Overlays inference results on dual-view video, supports ranking, annotation,
and analysis workflows. See docs/DESIGN_VISUALIZATION_TOOL.md for full spec.

Usage:
  python tools/visualizer.py render  --input results.json --labels labels.json
  python tools/visualizer.py render  --input results.json::0-19 --labels labels.json
  python tools/visualizer.py render  --input clip_001 --labels labels.json
  python tools/visualizer.py rank    --input results.json --labels labels.json --mode worst --top 20
  python tools/visualizer.py annotate --result-id <ID> --text "note" --author qa_lead
  python tools/visualizer.py annotate --query --clip clip_001
  python tools/visualizer.py heatmap --input results.json --labels labels.json
"""

import argparse
import copy
import csv
import glob
import json
import logging
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

# ── Project-root imports (add project root to path) ──────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.metrics import exact_match, jaccard_similarity, cardinality_accuracy

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("visualizer")

# ── Constants ────────────────────────────────────────────────────────────────
ROWS = list("ABCDEFGH")
COLS = list(range(1, 13))
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_VIZ_DIR = DEFAULT_OUTPUT_ROOT / "visualizations"
DEFAULT_ANNOTATION_DIR = DEFAULT_OUTPUT_ROOT / "annotations"
DEFAULT_ANALYSIS_DIR = DEFAULT_OUTPUT_ROOT / "analyses"

# Colours (BGR for OpenCV)
COL_CORRECT = (0, 255, 0)       # Green — true positive
COL_FP = (0, 0, 255)            # Red — false positive
COL_FN = (255, 0, 0)            # Blue — false negative
COL_GRID = (200, 200, 200)      # Light grey — grid lines
COL_TEXT_BG = (0, 0, 0)         # Black — text background
COL_TEXT_FG = (255, 255, 255)   # White — text foreground
COL_HEADER_BG = (40, 40, 40)    # Dark grey — header bar

WELL_RADIUS_DEFAULT = 18
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.45
FONT_THICKNESS = 1


# ═════════════════════════════════════════════════════════════════════════════
# Utility helpers
# ═════════════════════════════════════════════════════════════════════════════

def _well_set(wells: List[Dict]) -> Set[Tuple[str, int]]:
    """Convert a list of well dicts to a set of (row, col) tuples."""
    return {(w["well_row"], int(w["well_column"])) for w in wells}


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _persistent_id(clip_id: str, result_index: int) -> str:
    """Create a stable, human-readable persistent ID."""
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    ts = now.strftime("%Y%m%dT%H%M%SZ")
    return f"{date_str}:{clip_id}:result_{result_index}:{ts}"


def _extract_clip_id(fpv_id: str) -> str:
    """Extract base clip_id from 'clip_001_FPV' → 'clip_001'."""
    return re.sub(r"_(FPV|Topview)$", "", fpv_id, flags=re.IGNORECASE)


def _find_videos(clip_id: str, search_dirs: List[Path]) -> Tuple[Optional[Path], Optional[Path]]:
    """Locate FPV and Topview MP4s for a clip_id across search directories."""
    fpv, topview = None, None
    for d in search_dirs:
        for f in d.glob(f"{clip_id}_FPV.*"):
            fpv = f
        for f in d.glob(f"{clip_id}_Topview.*"):
            topview = f
        # Also try case-insensitive
        if fpv is None:
            for f in d.iterdir():
                if f.stem.lower() == f"{clip_id}_fpv".lower() and f.suffix.lower() in (".mp4", ".avi", ".mov"):
                    fpv = f
        if topview is None:
            for f in d.iterdir():
                if f.stem.lower() == f"{clip_id}_topview".lower() and f.suffix.lower() in (".mp4", ".avi", ".mov"):
                    topview = f
        if fpv and topview:
            break
    return fpv, topview


def _load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _save_json(data: Any, path: Path, indent: int = 2):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)
    logger.info(f"Saved: {path}")


def _parse_index_spec(spec: str) -> List[int]:
    """Parse '0,5,10-15,20' into [0, 5, 10, 11, 12, 13, 14, 15, 20]."""
    indices = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            indices.extend(range(int(lo), int(hi) + 1))
        else:
            indices.append(int(part))
    return sorted(set(indices))


def _parse_input_spec(spec: str) -> dict:
    """
    Parse --input value into a structured spec.
    Returns dict with keys: type, path, indices, clip_ids.
    """
    # @file_list.txt
    if spec.startswith("@"):
        list_path = Path(spec[1:])
        if not list_path.exists():
            raise FileNotFoundError(f"Video list file not found: {list_path}")
        clip_ids = [line.strip() for line in list_path.read_text().splitlines() if line.strip()]
        return {"type": "clip_ids", "clip_ids": clip_ids}

    # results.json::0-19
    if "::" in spec:
        path_str, idx_str = spec.split("::", 1)
        return {
            "type": "results_indexed",
            "path": Path(path_str),
            "indices": _parse_index_spec(idx_str),
        }

    path = Path(spec)
    # If it's a JSON file
    if path.suffix.lower() == ".json" and path.exists():
        return {"type": "results_file", "path": path}

    # Comma-separated clip IDs
    if "," in spec:
        return {"type": "clip_ids", "clip_ids": [c.strip() for c in spec.split(",")]}

    # Single clip ID
    return {"type": "clip_ids", "clip_ids": [spec]}


# ═════════════════════════════════════════════════════════════════════════════
# Scoring / Ranking
# ═════════════════════════════════════════════════════════════════════════════

def _hamming_distance(pred_wells: List[Dict], gt_wells: List[Dict]) -> int:
    """Count of wells in symmetric difference (FP + FN)."""
    pred = _well_set(pred_wells)
    gt = _well_set(gt_wells)
    return len(pred.symmetric_difference(gt))


def _anomaly_score(result: Dict) -> float:
    """
    Compute 'strangeness' — high when row and column confidences diverge,
    or when the model is highly confident but wrong.

    Uses confidence-only signal (no ground truth required) so the score
    can be computed before labels are available.  The design doc defines
    a richer formula that also incorporates IoU; that variant is reserved
    for Phase 2 where ground truth is always at hand via the database.
    """
    meta = result.get("metadata", {})
    rp = meta.get("max_row_prob", 0.5)
    cp = meta.get("max_col_prob", 0.5)
    # Divergence between row and column confidence
    divergence = abs(rp - cp)
    # Product represents joint confidence; high divergence + high product = strange
    joint = rp * cp
    return divergence * 2.0 + (1.0 - joint) * 0.5


def _score_result(result: Dict, gt_entry: Optional[Dict]) -> Dict:
    """Compute all accuracy metrics for a single result."""
    pred = result.get("wells_prediction", [])
    gt = gt_entry.get("wells_ground_truth", []) if gt_entry else []
    return {
        "exact_match": exact_match(pred, gt),
        "jaccard": jaccard_similarity(pred, gt),
        "cardinality_match": cardinality_accuracy(pred, gt),
        "hamming_distance": _hamming_distance(pred, gt),
        "anomaly_score": _anomaly_score(result),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Ground-truth matching
# ═════════════════════════════════════════════════════════════════════════════

def _build_gt_index(labels: List[Dict]) -> Dict[str, Dict]:
    """Index ground-truth labels by clip_id for fast lookup."""
    idx = {}
    for entry in labels:
        clip_id = _extract_clip_id(entry.get("clip_id_FPV", ""))
        idx[clip_id] = entry
        # Also index by the raw FPV id
        idx[entry.get("clip_id_FPV", "")] = entry
    return idx


# ═════════════════════════════════════════════════════════════════════════════
# Rendering engine
# ═════════════════════════════════════════════════════════════════════════════

class WellGridOverlay:
    """Draws a 96-well plate grid overlay on a video frame."""

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        well_radius: int = WELL_RADIUS_DEFAULT,
    ):
        self.fw = frame_width
        self.fh = frame_height
        self.well_radius = well_radius

        # Grid geometry — place the 8×12 grid in the centre-right of frame
        # Leave margins for text overlay
        margin_top = 60
        margin_bottom = 30
        margin_left = 30
        margin_right = 30

        usable_w = self.fw - margin_left - margin_right
        usable_h = self.fh - margin_top - margin_bottom

        # Well spacing
        self.col_spacing = usable_w / (len(COLS) + 1)
        self.row_spacing = usable_h / (len(ROWS) + 1)

        self.origin_x = margin_left + self.col_spacing
        self.origin_y = margin_top + self.row_spacing

    def well_centre(self, row: str, col: int) -> Tuple[int, int]:
        """Get pixel centre for a well given row letter and column number."""
        ri = ROWS.index(row)
        ci = col - 1
        x = int(self.origin_x + ci * self.col_spacing)
        y = int(self.origin_y + ri * self.row_spacing)
        return x, y

    def draw_grid(self, frame: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        """Draw the base 96-well grid with row/column labels."""
        overlay = frame.copy()

        # Draw all wells as empty circles
        for row in ROWS:
            for col in COLS:
                cx, cy = self.well_centre(row, col)
                cv2.circle(overlay, (cx, cy), self.well_radius, COL_GRID, 1, cv2.LINE_AA)

        # Row labels (A-H) on left
        for i, row in enumerate(ROWS):
            cx, cy = self.well_centre(row, 1)
            cv2.putText(
                overlay, row,
                (cx - self.well_radius - 20, cy + 5),
                FONT, FONT_SCALE, COL_GRID, FONT_THICKNESS, cv2.LINE_AA,
            )

        # Column labels (1-12) on top
        for col in COLS:
            cx, cy = self.well_centre(ROWS[0], col)
            label = str(col)
            cv2.putText(
                overlay, label,
                (cx - 5, cy - self.well_radius - 8),
                FONT, FONT_SCALE, COL_GRID, FONT_THICKNESS, cv2.LINE_AA,
            )

        # Blend
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    def highlight_wells(
        self,
        frame: np.ndarray,
        wells: Set[Tuple[str, int]],
        colour: Tuple[int, int, int],
        filled: bool = True,
    ) -> np.ndarray:
        """Highlight a set of wells with circles."""
        for row, col in wells:
            if row in ROWS and 1 <= col <= 12:
                cx, cy = self.well_centre(row, col)
                thickness = -1 if filled else 2
                cv2.circle(frame, (cx, cy), self.well_radius, colour, thickness, cv2.LINE_AA)
        return frame


def _draw_header_bar(
    frame: np.ndarray,
    clip_id: str,
    result_index: int,
    timestamp: str,
    persistent_id: str,
    score_text: str = "",
) -> np.ndarray:
    """Draw a dark header bar with metadata text at the top of the frame."""
    bar_h = 50
    cv2.rectangle(frame, (0, 0), (frame.shape[1], bar_h), COL_HEADER_BG, -1)

    line1 = f"Clip: {clip_id}  |  Result #{result_index}  |  {timestamp}"
    line2 = f"ID: {persistent_id}"
    if score_text:
        line2 += f"  |  {score_text}"

    cv2.putText(frame, line1, (10, 18), FONT, 0.42, COL_TEXT_FG, 1, cv2.LINE_AA)
    cv2.putText(frame, line2, (10, 38), FONT, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
    return frame


def _draw_legend(frame: np.ndarray) -> np.ndarray:
    """Draw colour legend at bottom of frame."""
    h = frame.shape[0]
    y = h - 18
    items = [
        ("Correct", COL_CORRECT),
        ("False Positive", COL_FP),
        ("False Negative", COL_FN),
    ]
    x = 10
    for label, colour in items:
        cv2.circle(frame, (x + 6, y - 4), 6, colour, -1, cv2.LINE_AA)
        cv2.putText(frame, label, (x + 16, y), FONT, 0.38, COL_TEXT_FG, 1, cv2.LINE_AA)
        x += 140
    return frame


def render_clip(
    fpv_path: Path,
    topview_path: Path,
    pred_wells: List[Dict],
    gt_wells: List[Dict],
    clip_id: str,
    result_index: int,
    output_path: Path,
    well_radius: int = WELL_RADIUS_DEFAULT,
    include_metadata: bool = True,
) -> Dict:
    """
    Render a single clip with prediction overlay.

    Opens both FPV and Topview, draws side-by-side with grid overlays,
    highlights correct/FP/FN wells, writes to output MP4.

    Returns metadata dict for the rendered clip.
    """
    cap_fpv = cv2.VideoCapture(str(fpv_path))
    cap_top = cv2.VideoCapture(str(topview_path))

    if not cap_fpv.isOpened():
        raise IOError(f"Cannot open FPV video: {fpv_path}")
    if not cap_top.isOpened():
        raise IOError(f"Cannot open Topview video: {topview_path}")

    fps = cap_top.get(cv2.CAP_PROP_FPS) or 30.0
    fw = int(cap_top.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap_top.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_top = int(cap_top.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames_fpv = int(cap_fpv.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = min(total_frames_top, total_frames_fpv)

    # We'll render as side-by-side: [FPV | Topview] each scaled to half width
    out_w = fw  # same width as topview (we'll resize FPV to match)
    out_h = fh
    half_w = out_w // 2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))

    # Compute well sets
    pred_set = _well_set(pred_wells)
    gt_set = _well_set(gt_wells)
    tp = pred_set & gt_set
    fp = pred_set - gt_set
    fn = gt_set - pred_set

    # Grid overlay for the topview half
    grid = WellGridOverlay(half_w, out_h, well_radius)

    timestamp = _now_iso()
    pid = _persistent_id(clip_id, result_index)

    score = _hamming_distance(pred_wells, gt_wells)
    jacc = jaccard_similarity(pred_wells, gt_wells)
    score_text = f"Hamming={score}  IoU={jacc:.2f}"

    frame_idx = 0
    while frame_idx < total_frames:
        ret_fpv, f_fpv = cap_fpv.read()
        ret_top, f_top = cap_top.read()
        if not ret_fpv or not ret_top:
            break

        # Resize both to half-width
        f_fpv_r = cv2.resize(f_fpv, (half_w, out_h))
        f_top_r = cv2.resize(f_top, (half_w, out_h))

        # Draw grid on topview
        f_top_r = grid.draw_grid(f_top_r, alpha=0.35)

        # Highlight wells
        f_top_r = grid.highlight_wells(f_top_r, tp, COL_CORRECT, filled=True)
        f_top_r = grid.highlight_wells(f_top_r, fp, COL_FP, filled=True)
        f_top_r = grid.highlight_wells(f_top_r, fn, COL_FN, filled=False)

        # Combine side-by-side
        canvas = np.hstack([f_fpv_r, f_top_r])

        # View labels
        cv2.putText(canvas, "FPV", (10, out_h - 10), FONT, 0.5, COL_TEXT_FG, 1, cv2.LINE_AA)
        cv2.putText(canvas, "Top-View", (half_w + 10, out_h - 10), FONT, 0.5, COL_TEXT_FG, 1, cv2.LINE_AA)

        # Metadata header
        if include_metadata:
            canvas = _draw_header_bar(canvas, clip_id, result_index, timestamp, pid, score_text)
            canvas = _draw_legend(canvas)

        writer.write(canvas)
        frame_idx += 1

    cap_fpv.release()
    cap_top.release()
    writer.release()

    meta = {
        "result_index": result_index,
        "clip_id": clip_id,
        "video_output_path": str(output_path),
        "render_timestamp": timestamp,
        "fps": fps,
        "total_frames": frame_idx,
        "resolution": f"{out_w}x{out_h}",
        "prediction": {"wells": pred_wells},
        "ground_truth": {"wells": gt_wells},
        "accuracy": {
            "exact_match": bool(exact_match(pred_wells, gt_wells)),
            "jaccard": float(jacc),
            "hamming_distance": int(score),
            "cardinality_match": bool(cardinality_accuracy(pred_wells, gt_wells)),
        },
        "well_breakdown": {
            "true_positives": [{"well_row": r, "well_column": c} for r, c in sorted(tp)],
            "false_positives": [{"well_row": r, "well_column": c} for r, c in sorted(fp)],
            "false_negatives": [{"well_row": r, "well_column": c} for r, c in sorted(fn)],
        },
        "persistent_id": pid,
    }

    # Save sidecar metadata
    meta_path = output_path.with_suffix(".json")
    _save_json(meta, meta_path)

    logger.info(
        f"Rendered: {clip_id} (result #{result_index}) — "
        f"{frame_idx} frames, Hamming={score}, IoU={jacc:.2f} → {output_path.name}"
    )
    return meta


# ═════════════════════════════════════════════════════════════════════════════
# Commands
# ═════════════════════════════════════════════════════════════════════════════

def cmd_render(args):
    """Render predictions overlaid on video clips."""
    spec = _parse_input_spec(args.input)
    labels = _load_json(Path(args.labels)) if args.labels else []
    gt_index = _build_gt_index(labels)

    # Determine video search directories
    video_dirs = [Path(d) for d in args.video_dirs] if args.video_dirs else []
    # Default: data/pipette_well_dataset/
    default_data = PROJECT_ROOT / "data" / "pipette_well_dataset"
    if default_data.exists():
        video_dirs.append(default_data)

    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_VIZ_DIR / f"{_now_compact()}_render"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results based on spec type
    results_to_render = []  # List of (index, result_dict, clip_id)

    if spec["type"] in ("results_file", "results_indexed"):
        raw = _load_json(spec["path"])
        # Handle both single result and list of results
        all_results = raw if isinstance(raw, list) else [raw]
        indices = spec.get("indices", list(range(len(all_results))))
        for idx in indices:
            if 0 <= idx < len(all_results):
                r = all_results[idx]
                cid = _extract_clip_id(r.get("clip_id_FPV", r.get("clip_id", f"unknown_{idx}")))
                results_to_render.append((idx, r, cid))

    elif spec["type"] == "clip_ids":
        # User specified clip IDs — find matching results if labels exist
        for cid in spec["clip_ids"]:
            gt = gt_index.get(cid, {})
            # Create a stub result with ground truth only (no predictions)
            stub = {
                "clip_id_FPV": f"{cid}_FPV",
                "clip_id_Topview": f"{cid}_Topview",
                "wells_prediction": [],
                "metadata": {},
            }
            results_to_render.append((0, stub, cid))

    if not results_to_render:
        logger.error("No results to render. Check --input specification.")
        return 1

    manifest_clips = []
    for idx, result, clip_id in results_to_render:
        fpv_path, top_path = _find_videos(clip_id, video_dirs)
        if not fpv_path or not top_path:
            logger.warning(f"Videos not found for {clip_id} — skipping")
            continue

        pred = result.get("wells_prediction", [])
        gt_entry = gt_index.get(clip_id, {})
        gt = gt_entry.get("wells_ground_truth", [])

        out_name = f"result_{idx:03d}_{clip_id}.mp4"
        out_path = output_dir / out_name

        try:
            meta = render_clip(
                fpv_path, top_path, pred, gt, clip_id, idx, out_path,
                well_radius=args.well_radius,
                include_metadata=args.include_metadata,
            )
            manifest_clips.append({
                "result_index": idx,
                "clip_id": clip_id,
                "metadata_file": out_path.with_suffix(".json").name,
                "video_file": out_name,
            })
        except Exception as e:
            logger.error(f"Failed to render {clip_id}: {e}")

    # Write manifest
    manifest = {
        "run_id": output_dir.name,
        "created": _now_iso(),
        "total_rendered": len(manifest_clips),
        "output_directory": str(output_dir),
        "clips_rendered": manifest_clips,
    }
    _save_json(manifest, output_dir / "manifest.json")
    logger.info(f"Render complete: {len(manifest_clips)} clips → {output_dir}")
    return 0


def cmd_rank(args):
    """Rank results by accuracy metric and optionally render top-K."""
    results_raw = _load_json(Path(args.input))
    all_results = results_raw if isinstance(results_raw, list) else [results_raw]
    labels = _load_json(Path(args.labels)) if args.labels else []
    gt_index = _build_gt_index(labels)

    scored = []
    for idx, r in enumerate(all_results):
        cid = _extract_clip_id(r.get("clip_id_FPV", f"unknown_{idx}"))
        gt_entry = gt_index.get(cid)
        scores = _score_result(r, gt_entry)
        scored.append({
            "index": idx,
            "clip_id": cid,
            "scores": scores,
            "result": r,
            "gt": gt_entry,
        })

    # Sort based on mode
    mode = args.mode
    if mode == "worst":
        scored.sort(key=lambda s: s["scores"]["hamming_distance"], reverse=True)
    elif mode == "best":
        scored.sort(key=lambda s: s["scores"]["jaccard"], reverse=True)
    elif mode == "strangest":
        scored.sort(key=lambda s: s["scores"]["anomaly_score"], reverse=True)
    else:
        logger.error(f"Unknown rank mode: {mode}")
        return 1

    top_k = scored[: args.top]

    # Print ranking
    print(f"\n{'='*70}")
    print(f"  Ranking: {mode.upper()} {args.top} of {len(scored)} results")
    print(f"{'='*70}")
    for rank, entry in enumerate(top_k, 1):
        s = entry["scores"]
        em = "✓" if s["exact_match"] else "✗"
        print(
            f"  #{rank:2d}  idx={entry['index']:3d}  clip={entry['clip_id']:<16s}  "
            f"exact={em}  hamming={s['hamming_distance']:2d}  "
            f"IoU={s['jaccard']:.3f}  anomaly={s['anomaly_score']:.3f}"
        )
    print(f"{'='*70}\n")

    # Save ranking to analysis dir
    analysis_dir = Path(args.output_dir) if args.output_dir else DEFAULT_ANALYSIS_DIR
    analysis_dir.mkdir(parents=True, exist_ok=True)
    ranking_out = analysis_dir / f"ranking_{mode}_{args.top}.json"
    ranking_data = {
        "analysis_date": _now_iso(),
        "mode": mode,
        "top_k": args.top,
        "total_results": len(scored),
        "results": [
            {
                "rank": rank,
                "result_index": e["index"],
                "clip_id": e["clip_id"],
                **e["scores"],
            }
            for rank, e in enumerate(top_k, 1)
        ],
    }
    _save_json(ranking_data, ranking_out)

    # If --render requested, render the top-K clips
    if args.render:
        logger.info(f"Rendering top {len(top_k)} clips...")
        viz_dir = (
            Path(args.viz_output_dir) if args.viz_output_dir
            else DEFAULT_VIZ_DIR / f"{_now_compact()}_{mode}_{args.top}"
        )
        viz_dir.mkdir(parents=True, exist_ok=True)

        video_dirs = [Path(d) for d in args.video_dirs] if args.video_dirs else []
        default_data = PROJECT_ROOT / "data" / "pipette_well_dataset"
        if default_data.exists():
            video_dirs.append(default_data)

        for rank, entry in enumerate(top_k, 1):
            cid = entry["clip_id"]
            fpv, top = _find_videos(cid, video_dirs)
            if not fpv or not top:
                logger.warning(f"Videos not found for {cid} — skipping render")
                continue
            pred = entry["result"].get("wells_prediction", [])
            gt = entry["gt"].get("wells_ground_truth", []) if entry["gt"] else []
            out_path = viz_dir / f"rank_{rank:02d}_result_{entry['index']:03d}_{cid}.mp4"
            try:
                render_clip(fpv, top, pred, gt, cid, entry["index"], out_path)
            except Exception as e:
                logger.error(f"Render failed for {cid}: {e}")

        logger.info(f"Rendered {len(top_k)} clips → {viz_dir}")

    return 0


def cmd_annotate(args):
    """Create, update, or query QA annotations."""
    ann_dir = DEFAULT_ANNOTATION_DIR
    ann_dir.mkdir(parents=True, exist_ok=True)
    ann_file = ann_dir / "annotations.json"
    idx_file = ann_dir / "annotation_index.json"

    # Load existing
    annotations = _load_json(ann_file) if ann_file.exists() else []
    index = _load_json(idx_file) if idx_file.exists() else {}

    if args.query:
        # Query mode
        matches = annotations
        if args.clip:
            matches = [a for a in matches if a.get("clip_id") == args.clip]
        if args.result_index is not None:
            matches = [a for a in matches if a.get("result_index") == args.result_index]
        if args.tag:
            matches = [a for a in matches if args.tag in a.get("tags", [])]
        if args.category:
            matches = [a for a in matches if a.get("category") == args.category]

        if not matches:
            print("No annotations found matching query.")
        else:
            print(f"\nFound {len(matches)} annotation(s):\n")
            for ann in matches:
                print(f"  [{ann['annotation_id']}] {ann['timestamp']}")
                print(f"    Clip: {ann.get('clip_id', 'N/A')}  Result: {ann.get('result_index', 'N/A')}")
                print(f"    Category: {ann.get('category', 'general')}  Severity: {ann.get('severity', 'info')}")
                print(f"    Author: {ann.get('author', 'unknown')}")
                print(f"    Tags: {', '.join(ann.get('tags', []))}")
                print(f"    Note: {ann['text']}")
                print()
        return 0

    # Create mode
    if not args.text:
        logger.error("--text is required when creating an annotation")
        return 1

    # Build annotation
    ann_id = f"ann_{len(annotations):05d}"
    clip_id = args.clip or ""
    result_idx = args.result_index if args.result_index is not None else -1
    result_id = args.result_id or ""

    # Parse tags from comma-separated
    tags = [t.strip() for t in args.tags.split(",")] if args.tags else []

    annotation = {
        "annotation_id": ann_id,
        "result_id": result_id,
        "clip_id": clip_id,
        "result_index": result_idx,
        "author": args.author or "unknown",
        "timestamp": _now_iso(),
        "category": args.category or "general",
        "text": args.text,
        "tags": tags,
        "severity": args.severity or "info",
    }

    annotations.append(annotation)

    # Update index
    if result_id:
        index.setdefault(result_id, []).append(ann_id)
    if clip_id:
        index.setdefault(f"clip:{clip_id}", []).append(ann_id)

    _save_json(annotations, ann_file)
    _save_json(index, idx_file)

    print(f"Created annotation {ann_id}: {args.text[:80]}...")
    return 0


def cmd_heatmap(args):
    """Generate an error heatmap across all results."""
    results_raw = _load_json(Path(args.input))
    all_results = results_raw if isinstance(results_raw, list) else [results_raw]
    labels = _load_json(Path(args.labels)) if args.labels else []
    gt_index = _build_gt_index(labels)

    # Accumulate per-well error counts
    fp_counts = np.zeros((8, 12), dtype=int)  # False positive frequency
    fn_counts = np.zeros((8, 12), dtype=int)  # False negative frequency
    tp_counts = np.zeros((8, 12), dtype=int)  # True positive frequency

    for idx, r in enumerate(all_results):
        cid = _extract_clip_id(r.get("clip_id_FPV", f"unknown_{idx}"))
        gt_entry = gt_index.get(cid, {})
        pred = _well_set(r.get("wells_prediction", []))
        gt = _well_set(gt_entry.get("wells_ground_truth", []))

        for row, col in pred & gt:
            ri = ROWS.index(row)
            tp_counts[ri][col - 1] += 1
        for row, col in pred - gt:
            ri = ROWS.index(row)
            fp_counts[ri][col - 1] += 1
        for row, col in gt - pred:
            ri = ROWS.index(row)
            fn_counts[ri][col - 1] += 1

    # Render heatmap as image
    cell_w, cell_h = 60, 60
    margin = 40
    img_w = margin + 12 * cell_w + margin
    img_h = margin + 8 * cell_h + margin + 30  # extra for title
    img = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255

    # Title
    cv2.putText(img, "Error Heatmap: FP (red) + FN (blue) per well", (margin, 25),
                FONT, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    total_errors = fp_counts + fn_counts
    max_err = max(total_errors.max(), 1)

    for ri, row in enumerate(ROWS):
        for ci, col in enumerate(COLS):
            x = margin + ci * cell_w
            y = margin + 30 + ri * cell_h

            fp = fp_counts[ri][ci]
            fn = fn_counts[ri][ci]
            total = fp + fn

            # Colour: blend red (FP) and blue (FN) proportionally
            intensity = total / max_err
            r_val = int(min(255, fp / max(total, 1) * 255 * intensity))
            b_val = int(min(255, fn / max(total, 1) * 255 * intensity))
            g_val = int(max(0, 255 - 255 * intensity))  # Green decreases with errors
            colour = (b_val, g_val, r_val)

            cv2.rectangle(img, (x + 2, y + 2), (x + cell_w - 2, y + cell_h - 2), colour, -1)
            cv2.rectangle(img, (x + 2, y + 2), (x + cell_w - 2, y + cell_h - 2), (100, 100, 100), 1)

            # Well label
            label = f"{row}{col}"
            cv2.putText(img, label, (x + 8, y + 22), FONT, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
            # Error count
            if total > 0:
                cv2.putText(img, f"FP:{fp} FN:{fn}", (x + 4, y + 40), FONT, 0.28, (0, 0, 0), 1, cv2.LINE_AA)

        # Row label
        cv2.putText(img, row, (10, margin + 30 + ri * cell_h + 35), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Column labels
    for ci, col in enumerate(COLS):
        cv2.putText(img, str(col), (margin + ci * cell_w + 22, margin + 22), FONT, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    # Save
    analysis_dir = Path(args.output_dir) if args.output_dir else DEFAULT_ANALYSIS_DIR
    analysis_dir.mkdir(parents=True, exist_ok=True)
    img_path = analysis_dir / "heatmap_errors.png"
    cv2.imwrite(str(img_path), img)

    # Save data
    data_path = analysis_dir / "heatmap_errors.json"
    _save_json({
        "analysis_date": _now_iso(),
        "total_results": len(all_results),
        "fp_counts": fp_counts.tolist(),
        "fn_counts": fn_counts.tolist(),
        "tp_counts": tp_counts.tolist(),
        "max_error_per_well": int(max_err),
    }, data_path)

    logger.info(f"Heatmap saved: {img_path}")
    print(f"Heatmap image: {img_path}")
    print(f"Heatmap data:  {data_path}")
    return 0


# ═════════════════════════════════════════════════════════════════════════════
# CLI argument parsing
# ═════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visualizer",
        description="Pipette Well Detection — Visualization & Analysis Tool (Phase 1)",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # ── render ────────────────────────────────────────────────────────────
    p_render = sub.add_parser("render", help="Overlay predictions on video clips")
    p_render.add_argument("--input", required=True,
                          help="Input spec: results.json, results.json::0-19, clip_001, @list.txt")
    p_render.add_argument("--labels", default=None,
                          help="Path to labels.json (ground truth)")
    p_render.add_argument("--output-dir", default=None,
                          help="Output directory (default: outputs/visualizations/TIMESTAMP/)")
    p_render.add_argument("--video-dirs", nargs="*", default=None,
                          help="Directories to search for video files")
    p_render.add_argument("--well-radius", type=int, default=WELL_RADIUS_DEFAULT,
                          help=f"Well circle radius in pixels (default: {WELL_RADIUS_DEFAULT})")
    p_render.add_argument("--include-metadata", action="store_true", default=True,
                          help="Include metadata header bar (default: True)")
    p_render.add_argument("--no-metadata", dest="include_metadata", action="store_false",
                          help="Disable metadata header bar")

    # ── rank ──────────────────────────────────────────────────────────────
    p_rank = sub.add_parser("rank", help="Rank results by accuracy metric")
    p_rank.add_argument("--input", required=True, help="Inference results JSON file")
    p_rank.add_argument("--labels", required=True, help="Path to labels.json")
    p_rank.add_argument("--mode", choices=["best", "worst", "strangest"], default="worst",
                        help="Ranking mode (default: worst)")
    p_rank.add_argument("--top", type=int, default=10, help="Number of results to show (default: 10)")
    p_rank.add_argument("--render", action="store_true", help="Also render the top-K clips as video")
    p_rank.add_argument("--output-dir", default=None, help="Analysis output directory")
    p_rank.add_argument("--viz-output-dir", default=None, help="Video render output directory (if --render)")
    p_rank.add_argument("--video-dirs", nargs="*", default=None,
                        help="Directories to search for video files")

    # ── annotate ──────────────────────────────────────────────────────────
    p_ann = sub.add_parser("annotate", help="Create or query QA annotations")
    p_ann.add_argument("--query", action="store_true", help="Query mode (search annotations)")
    p_ann.add_argument("--text", default=None, help="Annotation text (create mode)")
    p_ann.add_argument("--clip", default=None, help="Clip ID to annotate or filter by")
    p_ann.add_argument("--result-index", type=int, default=None, help="Result index to annotate")
    p_ann.add_argument("--result-id", default=None, help="Persistent result ID")
    p_ann.add_argument("--author", default=None, help="Author name/tag")
    p_ann.add_argument("--category", default=None,
                        help="Category: false_positive, false_negative, optical_issue, design_feedback, general")
    p_ann.add_argument("--tags", default=None, help="Comma-separated tags")
    p_ann.add_argument("--severity", default=None,
                        help="Severity: info, low, medium, high, critical")
    p_ann.add_argument("--tag", default=None, help="Filter by tag (query mode)")

    # ── heatmap ───────────────────────────────────────────────────────────
    p_heat = sub.add_parser("heatmap", help="Generate per-well error heatmap")
    p_heat.add_argument("--input", required=True, help="Inference results JSON file")
    p_heat.add_argument("--labels", required=True, help="Path to labels.json")
    p_heat.add_argument("--output-dir", default=None, help="Output directory for heatmap")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    dispatch = {
        "render": cmd_render,
        "rank": cmd_rank,
        "annotate": cmd_annotate,
        "heatmap": cmd_heatmap,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
