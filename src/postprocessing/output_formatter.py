"""
Output Formatter Module: Convert model logits to well predictions

TODO:
  - Implement logit_to_wells() function
  - Implement cardinality constraint enforcement
  - Implement output validation and sorting
"""


def logit_to_wells(row_logits, col_logits, confidence_threshold=0.5, cardinality_head_output=None):
    """
    Convert factorized row/column logits to well predictions.

    Args:
        row_logits: (8,) array of row probabilities after sigmoid (A-H)
        col_logits: (12,) array of column probabilities after sigmoid (1-12)
        confidence_threshold: Minimum confidence to include a well
        cardinality_head_output: Optional (3,) softmax output for cardinality (1/8/12)

    Returns:
        List of well dictionaries:
        [
            {"well_row": "A", "well_column": 1, "confidence": 0.95},
            {"well_row": "A", "well_column": 2, "confidence": 0.88},
            ...
        ]

    Logic:
        1. Threshold row_logits and col_logits at confidence_threshold
        2. Take Cartesian product: all (row_i, col_j) pairs that both pass threshold
        3. If cardinality_head provided:
           - Predict cardinality as argmax(cardinality_head)
           - If predicted cardinality = 8, enforce exactly 8 wells in single row
           - If predicted cardinality = 12, enforce exactly 12 wells in single column
        4. Sort in canonical order (A1, A2, ..., H12)
        5. Deduplicate
        6. Validate all wells in [A-H] × [1-12]
        7. Return with confidence scores

    TODO:
        1. Apply sigmoid to logits if not already done
        2. Implement thresholding logic
        3. Implement Cartesian product
        4. Implement cardinality constraint enforcement
        5. Implement sorting and deduplication
        6. Implement validation
        7. Return well list
    """
    raise NotImplementedError("logit_to_wells() not yet implemented")


def enforce_cardinality_constraint(wells, cardinality):
    """
    Enforce cardinality constraint: 1, 8 (full row), or 12 (full column) wells.

    Args:
        wells: List of (row, col) tuples with confidence scores
        cardinality: Predicted cardinality (1, 8, or 12)

    Returns:
        Filtered well list respecting cardinality constraint

    TODO:
        1. If cardinality = 1: keep well with highest confidence; drop rest
        2. If cardinality = 8:
           - Find row with most detected wells
           - Keep those 8 wells; drop others
        3. If cardinality = 12:
           - Find column with most detected wells
           - Keep those 12 wells; drop others
    """
    raise NotImplementedError("enforce_cardinality_constraint() not yet implemented")


def sort_and_deduplicate(wells):
    """
    Sort wells in canonical order (A1, A2, ..., H12) and remove duplicates.

    Args:
        wells: List of well dictionaries

    Returns:
        Sorted, deduplicated well list

    TODO:
        1. Create (row, col) tuples
        2. Remove duplicates (keep first occurrence)
        3. Sort by (row_index, col_index)
        4. Return sorted well list
    """
    raise NotImplementedError("sort_and_deduplicate() not yet implemented")


def validate_wells(wells):
    """
    Validate that all wells have valid row (A-H) and column (1-12) values.

    Args:
        wells: List of well dictionaries

    Returns:
        (is_valid: bool, error_messages: List[str])

    TODO:
        1. Check each well has 'well_row' and 'well_column' keys
        2. Validate row in ['A', 'B', ..., 'H']
        3. Validate column in [1, 2, ..., 12]
        4. Collect error messages for invalid wells
        5. Return validation result and error list
    """
    raise NotImplementedError("validate_wells() not yet implemented")
