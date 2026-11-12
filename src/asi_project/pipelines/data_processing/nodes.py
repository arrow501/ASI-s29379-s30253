"""Prepare data using Sprint 1 functions and Kedro-supplied settings."""

from typing import Any

import pandas as pd

from asi_project.config import load_registry
from asi_project.data import assert_valid, filter_training_rows, normalize_input


def validate_data(raw_data: pd.DataFrame, project: dict[str, Any]) -> pd.DataFrame:
    """Validate the dataset contract and return the unchanged DataFrame."""
    contract = load_registry()[project["dataset_id"]]
    assert_valid(raw_data, contract)
    return raw_data


def normalize_data(validated_data: pd.DataFrame) -> pd.DataFrame:
    """Normalize input values using Sprint 1 rules while preserving row order."""
    return normalize_input(validated_data)


def filter_data(normalized_data: pd.DataFrame, project: dict[str, Any]) -> pd.DataFrame:
    """Remove rows with missing targets while preserving the remaining row order."""
    contract = load_registry()[project["dataset_id"]]
    return filter_training_rows(normalized_data, contract)
