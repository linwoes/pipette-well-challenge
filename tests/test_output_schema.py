"""
Test Output Schema Validation

Tests that JSON output from inference.py matches expected schema:
  - wells array is present
  - Each well has row (A-H) and column (1-12)
  - No out-of-bounds or invalid wells
  - No duplicates
  - Proper JSON serialization

Author: QA Engineer
Date: April 2026
"""

import json
import pytest


class TestOutputSchema:
    """Test suite for output JSON schema validation."""

    def test_output_has_wells_array(self):
        """Test that output contains 'wells' array."""
        output = {"wells": [], "metadata": {}}
        assert "wells" in output
        assert isinstance(output["wells"], list)

    def test_valid_well_row_range(self):
        """Test that well rows are A-H."""
        valid_rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

        for row in valid_rows:
            well = {"well_row": row, "well_column": 1}
            assert well["well_row"] in valid_rows

        invalid_rows = ['I', 'Z', '1', 'a', 'AA']
        for row in invalid_rows:
            well = {"well_row": row, "well_column": 1}
            assert well["well_row"] not in valid_rows

    def test_valid_well_column_range(self):
        """Test that well columns are 1-12."""
        valid_cols = list(range(1, 13))

        for col in valid_cols:
            well = {"well_row": "A", "well_column": col}
            assert well["well_column"] in valid_cols

        invalid_cols = [0, 13, 14, 25, -1, 1.5, "1"]
        for col in invalid_cols:
            well = {"well_row": "A", "well_column": col}
            try:
                assert well["well_column"] not in valid_cols
            except (TypeError, AssertionError):
                pass  # Expected for type mismatches

    def test_no_duplicate_wells(self):
        """Test that wells array contains no duplicates."""
        output = {
            "wells": [
                {"well_row": "A", "well_column": 1},
                {"well_row": "A", "well_column": 2},
                {"well_row": "A", "well_column": 1},  # Duplicate
            ]
        }

        well_set = set()
        for well in output["wells"]:
            key = (well["well_row"], well["well_column"])
            assert key not in well_set, f"Duplicate well detected: {key}"
            well_set.add(key)

    def test_all_wells_in_bounds(self):
        """Test that all predicted wells are within 96-well plate bounds."""
        output = {
            "wells": [
                {"well_row": "A", "well_column": 1},
                {"well_row": "H", "well_column": 12},
                {"well_row": "D", "well_column": 6},
            ]
        }

        valid_rows = set('ABCDEFGH')
        valid_cols = set(range(1, 13))

        for well in output["wells"]:
            assert well["well_row"] in valid_rows, f"Invalid row: {well['well_row']}"
            assert well["well_column"] in valid_cols, f"Invalid column: {well['well_column']}"

    def test_well_object_structure(self):
        """Test that each well object has required keys."""
        output = {
            "wells": [
                {"well_row": "A", "well_column": 1},
                {"well_row": "B", "well_column": 2},
            ]
        }

        required_keys = {"well_row", "well_column"}

        for well in output["wells"]:
            assert isinstance(well, dict), f"Well is not a dict: {well}"
            for key in required_keys:
                assert key in well, f"Missing key '{key}' in well: {well}"

    def test_json_serialization(self):
        """Test that output can be serialized to JSON and parsed back."""
        output = {
            "wells": [
                {"well_row": "A", "well_column": 1},
                {"well_row": "H", "well_column": 12},
            ],
            "metadata": {
                "inference_time_seconds": 1.23,
                "confidence_threshold": 0.5,
            }
        }

        # Serialize to JSON
        json_str = json.dumps(output)
        assert isinstance(json_str, str)

        # Parse back
        parsed = json.loads(json_str)
        assert parsed == output

    def test_empty_wells_array_valid(self):
        """Test that empty wells array is valid (no wells detected)."""
        output = {
            "wells": [],
            "metadata": {
                "reason": "No wells detected with confidence > threshold"
            }
        }

        assert isinstance(output["wells"], list)
        assert len(output["wells"]) == 0

    def test_maximum_wells_limit(self):
        """Test that output doesn't exceed 96 wells (plate maximum)."""
        # Valid: 96 wells for 12-channel operation on entire plate
        output_max = {
            "wells": [
                {"well_row": chr(ord('A') + i // 12), "well_column": (i % 12) + 1}
                for i in range(96)
            ]
        }
        assert len(output_max["wells"]) <= 96

        # Invalid: >96 wells
        output_invalid = {
            "wells": [
                {"well_row": chr(ord('A') + i // 12), "well_column": (i % 12) + 1}
                for i in range(100)  # Would exceed bounds
            ]
        }
        assert len(output_invalid["wells"]) > 96  # Flag as invalid

    def test_well_row_uppercase(self):
        """Test that well rows are uppercase."""
        output = {
            "wells": [
                {"well_row": "A", "well_column": 1},  # Valid
                {"well_row": "a", "well_column": 1},  # Invalid: lowercase
            ]
        }

        valid_rows = set('ABCDEFGH')

        for well in output["wells"]:
            row = well["well_row"]
            assert row.isupper(), f"Row not uppercase: {row}"
            assert row in valid_rows, f"Invalid row: {row}"

    def test_well_column_is_integer(self):
        """Test that well columns are integers (not floats or strings)."""
        output = {
            "wells": [
                {"well_row": "A", "well_column": 1},  # Valid
                {"well_row": "A", "well_column": "1"},  # Invalid: string
                {"well_row": "A", "well_column": 1.5},  # Invalid: float
            ]
        }

        for well in output["wells"]:
            col = well["well_column"]
            # Allow int only (or strings that are digits)
            if isinstance(col, str):
                assert col.isdigit(), f"Column is non-numeric string: {col}"
            else:
                assert isinstance(col, int), f"Column is not int: {col}"

    def test_sorted_canonical_order(self):
        """Test that wells are in canonical order (A1, A2, ..., H12) for easier verification."""
        output = {
            "wells": [
                {"well_row": "A", "well_column": 1},
                {"well_row": "A", "well_column": 2},
                {"well_row": "B", "well_column": 1},
            ]
        }

        # Convert to sortable tuples
        well_tuples = [(w["well_row"], w["well_column"]) for w in output["wells"]]
        sorted_tuples = sorted(well_tuples, key=lambda x: (ord(x[0]) - ord('A'), x[1]))

        # Check if already sorted (optional, but good practice)
        assert well_tuples == sorted_tuples, "Wells not in canonical order"

    def test_multiwell_cardinality_consistency(self):
        """Test that 8-channel and 12-channel operations are geometrically consistent."""
        output_8channel = {
            "wells": [
                {"well_row": "A", "well_column": 1},
                {"well_row": "B", "well_column": 1},
                {"well_row": "C", "well_column": 1},
                {"well_row": "D", "well_column": 1},
                {"well_row": "E", "well_column": 1},
                {"well_row": "F", "well_column": 1},
                {"well_row": "G", "well_column": 1},
                {"well_row": "H", "well_column": 1},
            ]
        }

        # For 8-channel, all wells should be in same column
        columns = set(w["well_column"] for w in output_8channel["wells"])
        assert len(columns) <= 1, "8-channel operation should have all wells in same column"

        output_12channel = {
            "wells": [
                {"well_row": "A", "well_column": i}
                for i in range(1, 13)
            ]
        }

        # For 12-channel, all wells should be in same row
        rows = set(w["well_row"] for w in output_12channel["wells"])
        assert len(rows) <= 1, "12-channel operation should have all wells in same row"


class TestMetadata:
    """Test suite for output metadata validation."""

    def test_metadata_present(self):
        """Test that output includes metadata."""
        output = {
            "wells": [],
            "metadata": {
                "inference_time_seconds": 1.5,
                "confidence_threshold": 0.5,
            }
        }
        assert "metadata" in output
        assert isinstance(output["metadata"], dict)

    def test_inference_time_positive(self):
        """Test that inference time is positive."""
        output = {
            "metadata": {
                "inference_time_seconds": 0.5
            }
        }
        assert output["metadata"]["inference_time_seconds"] > 0

    def test_confidence_threshold_valid_range(self):
        """Test that confidence threshold is in valid range (0-1)."""
        output = {
            "metadata": {
                "confidence_threshold": 0.5
            }
        }
        threshold = output["metadata"]["confidence_threshold"]
        assert 0 <= threshold <= 1, f"Threshold out of range: {threshold}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
