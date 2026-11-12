"""Test the real data processing pipeline and its prepared_data boundary."""

import pandas as pd
import pytest
from kedro.io import DataCatalog, MemoryDataset
from kedro.runner import SequentialRunner

from asi_project.data import DataContractError
from asi_project.pipelines.data_processing.pipeline import create_pipeline


def run_processing(raw_data, dataset_id):
    """Run the actual pipeline with an explicit in-memory output."""
    catalog = DataCatalog(
        {
            "raw_data": MemoryDataset(data=raw_data),
            "params:project": MemoryDataset(
                data={"dataset_id": dataset_id},
            ),
            "prepared_data": MemoryDataset(),
        }
    )

    SequentialRunner().run(create_pipeline(), catalog)

    return catalog.load("prepared_data")


def test_pipeline_exposes_the_agreed_inputs_and_output():
    pipeline = create_pipeline()

    assert pipeline.inputs() == {"raw_data", "params:project"}
    assert pipeline.outputs() == {"prepared_data"}
    assert len(pipeline.nodes) == 3


def test_pipeline_prepares_ci_sample_without_losing_records(project):
    raw = project.sample.copy(deep=True)
    original = raw.copy(deep=True)
    expected = raw.copy(deep=True)
    expected["pdays"] = expected["pdays"].astype("Int64")
    expected.loc[expected["pdays"] == -1, "pdays"] = pd.NA

    result = run_processing(raw, project.contract.dataset_id)

    pd.testing.assert_frame_equal(result, expected)
    pd.testing.assert_frame_equal(raw, original)
    assert result["y"].notna().all()
    assert set(result["y"]) == {"yes", "no"}


def test_pipeline_preserves_order_duplicates_and_feature_missing_values(project):
    raw = project.sample.iloc[[0] * 6].copy()
    raw.index = pd.Index([8, 2, 8, 1, 9, 4], name="source_row")
    raw["y"] = ["yes", "no", None, "yes", "no", "yes"]
    raw["pdays"] = [-1, 0, 5, -1, 10, 20]
    raw["job"] = [" admin. ", "", "technician", "unknown", None, "retired"]
    raw["balance"] = [-100, -1, 0, 10, 100, 200]
    raw = pd.concat([raw, raw.iloc[[0]]])
    original = raw.copy(deep=True)

    expected = raw.iloc[[0, 1, 3, 4, 5, 6]].copy()
    expected["pdays"] = pd.array([pd.NA, 0, pd.NA, 10, 20, pd.NA], dtype="Int64")
    expected["job"] = ["admin.", float("nan"), "unknown", None, "retired", "admin."]

    result = run_processing(raw, project.contract.dataset_id)

    pd.testing.assert_frame_equal(result, expected)
    pd.testing.assert_frame_equal(raw, original)


def test_pipeline_stops_on_invalid_input(project):
    invalid = project.sample.drop(columns="y")

    with pytest.raises(DataContractError, match="y"):
        run_processing(invalid, project.contract.dataset_id)
