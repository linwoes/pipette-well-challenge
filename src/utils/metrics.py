"""
Metrics Module: Evaluation metrics for multi-label well detection

Metrics:
  - Per-well accuracy, precision, recall
  - Cardinality-aware F1
  - Jaccard index (for multi-label evaluation)
  - Localization error (MAE in pixel space)
  - Confidence calibration (reliability diagram)

TODO:
  - Implement all metrics below
"""


def exact_match_accuracy(predictions, ground_truth):
    """
    Exact-match accuracy: fraction of samples where predicted wells match exactly.

    Args:
        predictions: List of well lists (one per sample)
        ground_truth: List of ground truth well lists

    Returns:
        Accuracy (0-1)

    TODO:
        1. For each sample, check if predicted well set == ground truth well set
        2. Count matches
        3. Return matches / total
    """
    raise NotImplementedError("exact_match_accuracy() not yet implemented")


def cardinality_accuracy(predictions, ground_truth):
    """
    Cardinality accuracy: fraction of samples with correct well count.

    Args:
        predictions: List of well lists
        ground_truth: List of ground truth well lists

    Returns:
        Accuracy (0-1)

    TODO:
        1. For each sample, check if len(predicted wells) == len(ground truth wells)
        2. Count matches
        3. Return matches / total
    """
    raise NotImplementedError("cardinality_accuracy() not yet implemented")


def per_well_metrics(predictions, ground_truth):
    """
    Per-well accuracy, precision, and recall.

    Args:
        predictions: List of well lists (96-well format, e.g., [(A, 1), (A, 2), ...])
        ground_truth: List of ground truth well lists

    Returns:
        {
            'accuracy': Dict[well, float],  # Fraction of samples where well correctly predicted
            'precision': Dict[well, float],  # Of predicted positives, how many true
            'recall': Dict[well, float],  # Of true positives, how many predicted
            'f1': Dict[well, float],  # Harmonic mean of precision and recall
        }

    TODO:
        1. For each well (A1-H12):
           - Count TP (predicted positive, ground truth positive)
           - Count FP (predicted positive, ground truth negative)
           - Count FN (predicted negative, ground truth positive)
           - Compute precision = TP / (TP + FP)
           - Compute recall = TP / (TP + FN)
           - Compute F1 = 2 * (precision * recall) / (precision + recall)
        2. Return per-well metrics
    """
    raise NotImplementedError("per_well_metrics() not yet implemented")


def jaccard_index(predictions, ground_truth):
    """
    Jaccard index (Intersection over Union) for multi-label evaluation.

    Args:
        predictions: List of well sets (one per sample)
        ground_truth: List of ground truth well sets

    Returns:
        List of Jaccard indices (0-1, one per sample)

    Formula:
        J(A, B) = |A ∩ B| / |A ∪ B|

    TODO:
        1. For each sample:
           - Compute intersection size
           - Compute union size
           - Return |intersection| / |union|
        2. Return list of indices
    """
    raise NotImplementedError("jaccard_index() not yet implemented")


def confusion_matrix_per_well(predictions, ground_truth, num_rows=8, num_cols=12):
    """
    Confusion matrix per well: (num_rows × num_cols) × (num_rows × num_cols) matrix.

    Args:
        predictions: List of well lists
        ground_truth: List of ground truth well lists
        num_rows: Number of rows on plate (8)
        num_cols: Number of columns on plate (12)

    Returns:
        (96, 96) confusion matrix where entry [i, j] = count of samples
        where well i was ground truth and well j was predicted

    TODO:
        1. Flatten well coordinates to single index (0-95)
        2. For each sample, update confusion matrix
        3. Return (96, 96) matrix
    """
    raise NotImplementedError("confusion_matrix_per_well() not yet implemented")


def cardinality_aware_f1(predictions, ground_truth):
    """
    Cardinality-aware F1: separate F1 scores for 1-channel, 8-channel, 12-channel operations.

    Args:
        predictions: List of well lists
        ground_truth: List of ground truth well lists

    Returns:
        {
            'f1_1channel': float,  # F1 for samples with 1 well
            'f1_8channel': float,  # F1 for samples with 8 wells
            'f1_12channel': float,  # F1 for samples with 12 wells
        }

    TODO:
        1. Partition samples by ground truth cardinality
        2. Compute F1 (via Jaccard index) for each partition
        3. Return per-cardinality F1 scores
    """
    raise NotImplementedError("cardinality_aware_f1() not yet implemented")


def localization_error(predictions, ground_truth, pixel_coords):
    """
    Localization error: MAE between predicted and ground truth pixel coordinates.

    Args:
        predictions: List of (row, col) tuples (well coordinates)
        ground_truth: List of (row, col) tuples
        pixel_coords: Dict mapping (row, col) to pixel center (x, y)

    Returns:
        MAE in pixel space (float)

    TODO:
        1. For each sample, convert well coordinates to pixel coordinates
        2. Compute L2 distance for each well
        3. Return mean absolute error
    """
    raise NotImplementedError("localization_error() not yet implemented")


def confidence_calibration(predictions_with_confidence, ground_truth):
    """
    Confidence calibration: compare model confidence vs. actual accuracy.

    Args:
        predictions_with_confidence: List of wells with confidence scores
        ground_truth: List of ground truth well lists

    Returns:
        {
            'reliability_bins': Dict[bin, (avg_confidence, actual_accuracy)],
            'calibration_error': float,  # Expected Calibration Error (ECE)
        }

    TODO:
        1. Bin predictions by confidence (e.g., [0-0.1], [0.1-0.2], ..., [0.9-1.0])
        2. For each bin, compute average confidence and actual accuracy
        3. Compute ECE = sum(|confidence - accuracy| * bin_size)
        4. Return bins and ECE
    """
    raise NotImplementedError("confidence_calibration() not yet implemented")
