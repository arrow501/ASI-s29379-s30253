"""Accept the real S2-2 + S2-3 plug-in; exit 2 while pipeline wiring is incomplete."""

import argparse
import sys

import pandas as pd

from asi_project.config import ConfigError, find_repo_root, get_contract, load_project_config
from asi_project.data import DataContractError, load_dataset, normalize_column_names
from asi_project.metrics import compute_metrics, validate_metrics_record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=["local", "ci"], default="ci")
    args = parser.parse_args(argv)

    from kedro.framework.session import KedroSession
    from kedro.framework.startup import bootstrap_project
    from kedro.io import MemoryDataset
    from kedro.runner import SequentialRunner

    from asi_project.pipeline_registry import register_pipelines
    from asi_project.pipelines.contracts import assert_pipeline_wiring, assert_prepared_data
    from asi_project.pipelines.data_science.nodes import split_data

    root = find_repo_root()
    config = load_project_config(root)
    contract = get_contract(config)
    bootstrap_project(root)
    pipelines = register_pipelines()
    try:
        assert_pipeline_wiring(pipelines)
    except ConfigError as error:
        print(f"Integration not ready: {error}", file=sys.stderr)
        return 2

    mode = "sample" if args.env == "ci" else "full"
    with KedroSession.create(project_path=root, env=args.env) as session:
        context = session.load_context()
        if context.params["reporting"]["data_mode"] != mode:
            raise ConfigError(f"Environment {args.env} must report data_mode={mode!r}.")
        catalog = context.catalog
        if "raw_data" not in catalog:
            raise ConfigError("Add raw_data in catalog_data_processing.yml for base and ci.")
        raw = catalog.load("raw_data")
        if not isinstance(raw, pd.DataFrame):
            raise DataContractError("raw_data must be a pandas DataFrame; use pandas.CSVDataset.")
        try:
            pd.testing.assert_frame_equal(
                normalize_column_names(raw), load_dataset(config, contract, mode=mode)
            )
        except AssertionError as error:
            raise DataContractError(
                f"raw_data does not match the {mode} source. Check its catalog path and CSV separator."
            ) from error

        # First inspect the preparation boundary; then run the actual combined DAG.
        if "prepared_data" not in catalog:
            catalog["prepared_data"] = MemoryDataset()
        runner = SequentialRunner()
        runner.run(pipelines["data_processing"], catalog)
        prepared = catalog.load("prepared_data")
        assert_prepared_data(raw, prepared, contract)
        runner.run(pipelines["__default__"], catalog)
        record = catalog.load("metrics")
        model = catalog.load("trained_model")
        _, X_test, _, y_test = split_data(prepared, context.params["project"])
        if validate_metrics_record(record, contract):
            raise DataContractError("Integrated metrics fail the existing schema.")
        if record["data_mode"] != mode:
            raise DataContractError("Persisted data_mode does not match the selected environment.")
        if (
            compute_metrics(y_test, model.predict(X_test))
            != record["models"]["logistic_regression"]
        ):
            raise DataContractError("Persisted model predictions do not match integrated metrics.")
    print(
        f"OK: {args.env} raw input, preparation contract, default pipeline, model reload and metrics."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
