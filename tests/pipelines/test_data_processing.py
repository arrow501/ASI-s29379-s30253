"""Test data preparation nodes and edge cases from Sprint 1."""

import numpy as np
import pandas as pd
import pytest

from asi_project.data import DataContractError
from asi_project.pipelines.data_processing.nodes import (
    filter_data,
    normalize_data,
    validate_data,
)


@pytest.fixture
def params(project):
    return {"dataset_id": project.contract.dataset_id}


@pytest.fixture
def raw_data(project):
    """Create valid rows with duplicate, unsorted index labels."""
    frame = project.sample.iloc[[0] * 6].copy()
    frame.index = pd.Index([8, 2, 8, 1, 9, 4], name="source_row")
    frame["y"] = ["yes", "no", "yes", "no", "yes", "no"]
    frame["pdays"] = [-1, 0, 5, -1, 10, 20]
    return frame


def test_validation_preserves_values_columns_and_order(raw_data, params):
    original = raw_data.copy(deep=True)

    result = validate_data(raw_data, params)

    pd.testing.assert_frame_equal(result, original)
    pd.testing.assert_frame_equal(raw_data, original)


@pytest.mark.parametrize("column", ["y", "age", "pdays"])
def test_validation_rejects_missing_required_columns(raw_data, params, column):
    invalid = raw_data.drop(columns=column)

    with pytest.raises(DataContractError, match=column):
        validate_data(invalid, params)


@pytest.mark.parametrize(
    "labels",
    [
        ["no"] * 6,
        ["yes", "no", "maybe", "no", "yes", "no"],
        [None] * 6,
    ],
    ids=["single-class", "third-class", "all-targets-missing"],
)
def test_validation_rejects_invalid_target(raw_data, params, labels):
    invalid = raw_data.assign(y=labels)

    with pytest.raises(DataContractError):
        validate_data(invalid, params)


def test_validation_rejects_empty_dataset(raw_data, params):
    with pytest.raises(DataContractError):
        validate_data(raw_data.iloc[:0], params)


@pytest.mark.parametrize("missing", [None, np.nan, pd.NA])
def test_validation_allows_missing_target_when_both_classes_remain(raw_data, params, missing):
    raw_data["y"] = ["yes", "no", missing, "no", "yes", "no"]

    result = validate_data(raw_data, params)

    pd.testing.assert_frame_equal(result, raw_data)


@pytest.mark.parametrize(
    "values",
    [
        [-1, 0, 5, None, -1, 20],
        [" -1 ", "0", "5", "", "-1", "20"],
        pd.array([-1, 0, 5, pd.NA, -1, 20], dtype="Int64"),
    ],
    ids=["numbers", "text", "nullable-integers"],
)
def test_normalization_handles_pdays_without_changing_other_columns(raw_data, values):
    raw_data["pdays"] = values
    original = raw_data.copy(deep=True)
    expected = raw_data.copy(deep=True)
    expected["pdays"] = pd.array([pd.NA, 0, 5, pd.NA, pd.NA, 20], dtype="Int64")

    result = normalize_data(raw_data)

    pd.testing.assert_frame_equal(result, expected)
    pd.testing.assert_frame_equal(raw_data, original)


def test_normalization_preserves_valid_categories_and_negative_balances(raw_data):
    raw_data["job"] = [" admin. ", "", "   ", None, "unknown", "technician"]
    raw_data["balance"] = [-100, -1, 0, 10, 100, 200]

    result = normalize_data(raw_data)

    assert result["job"].iloc[0] == "admin."
    assert result["job"].iloc[1:4].isna().all()
    assert result["job"].iloc[4] == "unknown"
    pd.testing.assert_series_equal(result["balance"], raw_data["balance"])
    pd.testing.assert_series_equal(result["y"], raw_data["y"])
    pd.testing.assert_index_equal(result.index, raw_data.index)


def test_normalization_is_idempotent_and_independent_of_other_rows(raw_data):
    normalized = normalize_data(raw_data)

    pd.testing.assert_frame_equal(normalize_data(normalized), normalized)

    for position in range(len(raw_data)):
        pd.testing.assert_frame_equal(
            normalize_data(raw_data.iloc[[position]]),
            normalized.iloc[[position]],
        )


@pytest.mark.parametrize("missing", [None, np.nan, pd.NA])
def test_filter_removes_only_missing_targets_and_preserves_order(raw_data, params, missing):
    # Rows 0 and 2 share an index label. Only row 2 should be removed.
    raw_data["y"] = ["yes", "no", missing, "no", "yes", "no"]
    original = raw_data.copy(deep=True)
    expected = raw_data.iloc[[0, 1, 3, 4, 5]].copy()

    result = filter_data(raw_data, params)

    pd.testing.assert_frame_equal(result, expected)
    pd.testing.assert_frame_equal(raw_data, original)


def test_filter_preserves_feature_missing_values_and_duplicate_records(raw_data, params):
    raw_data["age"] = np.nan
    raw_data["job"] = None
    normalized = normalize_data(raw_data)
    # Include an exact duplicate; filtering must not perform deduplication.
    normalized = pd.concat([normalized, normalized.iloc[[0]]])
    original = normalized.copy(deep=True)

    result = filter_data(normalized, params)

    pd.testing.assert_frame_equal(result, original)
    pd.testing.assert_frame_equal(normalized, original)


@pytest.mark.parametrize("empty_input", [False, True])
def test_filter_handles_empty_result(raw_data, params, empty_input):
    frame = raw_data.iloc[:0].copy() if empty_input else raw_data.assign(y=None)
    expected = frame.iloc[:0].copy()

    result = filter_data(frame, params)

    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("blank_target", ["", "   "])
def test_normalization_then_filter_removes_blank_targets(raw_data, params, blank_target):
    raw_data["y"] = ["yes", "no", blank_target, "no", "yes", "no"]

    normalized = normalize_data(raw_data)
    result = filter_data(normalized, params)

    assert pd.isna(normalized["y"].iloc[2])
    pd.testing.assert_frame_equal(
        result,
        normalized.iloc[[0, 1, 3, 4, 5]],
    )
