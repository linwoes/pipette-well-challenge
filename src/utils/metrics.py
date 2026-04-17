"""
Metrics Module: Evaluation metrics for multi-label well detection

Implements:
  - exact_match(): Check if predicted well set exactly matches ground truth
  - jaccard_similarity(): Intersection over union metric
  - cardinality_accuracy(): Check if predicted count matches ground truth count
"""

from typing import List, Dict


def exact_match(pred_wells: List[Dict], gt_wells: List[Dict]) -> bool:
    """
    Check if predicted well set exactly equals ground truth set (order-insensitive).

    Args:
        pred_wells: List of predicted well dictionaries
        gt_wells: List of ground truth well dictionaries

    Returns:
        True if sets are identical, False otherwise
    """
    pred_set = set((w['well_row'], w['well_column']) for w in pred_wells)
    gt_set = set((w['well_row'], w['well_column']) for w in gt_wells)
    return pred_set == gt_set


def jaccard_similarity(pred_wells: List[Dict], gt_wells: List[Dict]) -> float:
    """
    Compute Jaccard similarity (intersection over union) between two well sets.

    Args:
        pred_wells: List of predicted well dictionaries
        gt_wells: List of ground truth well dictionaries

    Returns:
        Jaccard index (0-1), or 0 if both sets empty
    """
    pred_set = set((w['well_row'], w['well_column']) for w in pred_wells)
    gt_set = set((w['well_row'], w['well_column']) for w in gt_wells)

    intersection = len(pred_set & gt_set)
    union = len(pred_set | gt_set)

    if union == 0:
        # Both sets empty — define similarity as 0.0 (no correct prediction to reward).
        # Returning 1.0 here would inflate validation Jaccard whenever a clip has no
        # active wells and the model correctly predicts none, masking real failures.
        return 0.0

    return intersection / union


def cardinality_accuracy(pred_wells: List[Dict], gt_wells: List[Dict]) -> bool:
    """
    Check if predicted well count matches ground truth well count.

    Args:
        pred_wells: List of predicted well dictionaries
        gt_wells: List of ground truth well dictionaries

    Returns:
        True if len(pred) == len(gt), False otherwise
    """
    return len(pred_wells) == len(gt_wells)
