"""Executable handoff contracts shared by the two Sprint 2 pipeline owners."""

import pandas as pd

from asi_project.config import ConfigError, DatasetContract
from asi_project.data import (
    DataContractError,
    assert_valid,
    filter_training_rows,
    normalize_column_names,
    normalize_input,
)


def assert_pipeline_wiring(pipelines) -> None:
    """Fail explicitly while preparation is absent or the two halves do not connect."""
    processing = pipelines["data_processing"]
    science = pipelines["data_science"]
    default = pipelines["__default__"]
    if not processing.nodes:
        raise ConfigError("S2-2 pending: data_processing has no nodes. Implement issue #11 first.")
    if not science.nodes:
        raise ConfigError("data_science has no nodes.")
    produced = {name for item in processing.nodes for name in item.outputs}
    if "prepared_data" not in produced or "prepared_data" not in science.inputs():
        raise ConfigError(
            "Pipeline handoff must connect data_processing → prepared_data → data_science."
        )
    if set(default.nodes) != set(processing.nodes + science.nodes):
        raise ConfigError("The default pipeline must contain both complete pipeline factories.")
    external = {name for name in default.inputs() if not name.startswith("params:")}
    if external != {"raw_data"}:
        raise ConfigError(
            f"Default pipeline must need only raw_data plus parameters; got {sorted(external)}."
        )


def assert_prepared_data(
    raw: pd.DataFrame, prepared: pd.DataFrame, contract: DatasetContract
) -> None:
    """Check S1-equivalent preparation, including source index/order and target values.

    Excluded/extra source columns may be retained or dropped. All contract-required
    columns must survive unchanged except for existing S1 normalization/filtering.
    """
    if not isinstance(raw, pd.DataFrame) or not isinstance(prepared, pd.DataFrame):
        raise DataContractError("raw_data and prepared_data must be pandas DataFrames.")
    canonical_raw = normalize_column_names(raw)
    assert_valid(canonical_raw, contract)
    assert_valid(prepared, contract)
    expected = filter_training_rows(normalize_input(canonical_raw), contract)
    columns = list(contract.required_columns)
    try:
        pd.testing.assert_frame_equal(prepared.loc[:, columns], expected.loc[:, columns])
    except AssertionError as error:
        raise DataContractError(
            "prepared_data must retain S1-normalized values, dtypes, target and source index/order. "
            "Do not shuffle, reset the index, encode the target or fit preprocessing here.\n"
            + str(error)
        ) from error
