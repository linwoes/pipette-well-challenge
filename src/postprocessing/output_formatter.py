"""
Output Formatter Module: Convert model logits to well predictions

Implements:
  - logits_to_wells(): Convert row/col logits to well predictions with Cartesian product
  - validate_output(): Check well validity
  - format_json_output(): Create JSON structure matching challenge spec
"""

from typing import List, Dict, Tuple
import numpy as np


def logits_to_wells(row_logits: np.ndarray, col_logits: np.ndarray, threshold: float = 0.5) -> List[Dict]:
    """
    Convert factorized row/column logits to well predictions.

    Args:
        row_logits: (8,) array of logits for rows A-H
        col_logits: (12,) array of logits for columns 1-12
        threshold: Confidence threshold (default 0.5)

    Returns:
        List of well dictionaries with keys: well_row, well_column
    """
    # Apply sigmoid if not already probabilities
    row_probs = _sigmoid(row_logits)
    col_probs = _sigmoid(col_logits)

    # Threshold
    row_indices = np.where(row_probs >= threshold)[0]
    col_indices = np.where(col_probs >= threshold)[0]

    # Cartesian product
    wells = []
    row_letters = 'ABCDEFGH'

    for row_idx in row_indices:
        for col_idx in col_indices:
            wells.append({
                'well_row': row_letters[row_idx],
                'well_column': int(col_idx + 1)
            })

    # Deduplicate and sort
    wells = _deduplicate_wells(wells)
    wells = _sort_wells(wells)

    return wells


def validate_output(wells: List[Dict]) -> bool:
    """
    Validate that all wells have valid row and column values.
    Handles both string and int well_column values.

    Args:
        wells: List of well dictionaries

    Returns:
        True if all wells are valid, False otherwise
    """
    if not isinstance(wells, list):
        return False

    if len(wells) == 0:
        return False  # Must predict at least one well

    valid_rows = set('ABCDEFGH')
    valid_cols = set(range(1, 13))

    for well in wells:
        if not isinstance(well, dict):
            return False
        if 'well_row' not in well or 'well_column' not in well:
            return False
        if well['well_row'] not in valid_rows:
            return False
        # Handle both string and int for well_column
        try:
            col = int(well['well_column'])
            if col not in valid_cols:
                return False
        except (ValueError, TypeError):
            return False

    return True


def format_json_output(
    clip_id_fpv: str,
    clip_id_topview: str,
    wells: List[Dict],
    inference_time_s: float = 0.0,
    confident: bool = True,
) -> Dict:
    """
    Format well predictions as JSON output matching challenge spec exactly.

    Challenge spec output format:
        {
          "clip_id_FPV": "...",
          "clip_id_Topview": "...",
          "wells_prediction": [{"well_row": "A", "well_column": 1}, ...]
        }

    Args:
        clip_id_fpv: FPV clip identifier
        clip_id_topview: Top-view clip identifier
        wells: List of well dictionaries with keys well_row, well_column
        inference_time_s: Wall-clock inference time in seconds
        confident: Whether model was confident (max prob >= threshold)

    Returns:
        Dictionary matching the challenge submission format
    """
    return {
        'clip_id_FPV': clip_id_fpv,
        'clip_id_Topview': clip_id_topview,
        'wells_prediction': wells,
        'metadata': {
            'model': 'DINOv2-ViT-B/14+LoRA',
            'inference_time_s': round(inference_time_s, 3),
            'confident': confident,
        }
    }


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    """Apply sigmoid activation."""
    return 1.0 / (1.0 + np.exp(-logits))


def _deduplicate_wells(wells: List[Dict]) -> List[Dict]:
    """Remove duplicate wells, keeping first occurrence."""
    seen = set()
    deduped = []
    for well in wells:
        key = (well['well_row'], well['well_column'])
        if key not in seen:
            seen.add(key)
            deduped.append(well)
    return deduped


def _sort_wells(wells: List[Dict]) -> List[Dict]:
    """Sort wells in canonical order (A1, A2, ..., H12)."""
    return sorted(wells, key=lambda w: (ord(w['well_row']) - ord('A'), w['well_column']))


def logits_to_wells_adaptive(
    row_logits: np.ndarray,
    col_logits: np.ndarray,
    max_wells: int = 12,
) -> List[Dict]:
    """
    Adaptive well prediction using outer-product probability map.

    Instead of a fixed sigmoid threshold (which collapses to 0 or 96 predictions
    depending on model confidence), this uses a relative threshold:
      - Compute the full 8×12 outer-product probability map
      - Threshold at 50% of the map's maximum value
      - If that yields > max_wells, fall back to argmax (top-1)
      - Guarantees at least 1 prediction; caps at max_wells

    This resolves the column-head collapse failure mode where fixed threshold=0.5
    produces 0 or 70+ predictions instead of 1, 8, or 12.

    Args:
        row_logits: (8,) logits for rows A-H
        col_logits: (12,) logits for columns 1-12
        max_wells: cap on predictions before falling back to top-1 (default 12)

    Returns:
        List of well dicts with well_row and well_column
    """
    row_probs = _sigmoid(row_logits)
    col_probs = _sigmoid(col_logits)

    # Outer product: (8, 12) joint probability map
    well_map = np.outer(row_probs, col_probs)

    # Adaptive threshold: 50% of peak
    adaptive_thresh = well_map.max() * 0.5
    mask = well_map >= max(adaptive_thresh, 0.05)  # floor at 5% to avoid no-op

    if mask.sum() == 0 or mask.sum() > max_wells:
        # Fallback: top-1 argmax
        idx = np.unravel_index(well_map.argmax(), well_map.shape)
        wells = [{'well_row': 'ABCDEFGH'[idx[0]], 'well_column': int(idx[1] + 1)}]
    else:
        rows, cols = np.where(mask)
        wells = [{'well_row': 'ABCDEFGH'[r], 'well_column': int(c + 1)}
                 for r, c in sorted(zip(rows, cols))]

    return _sort_wells(wells)
