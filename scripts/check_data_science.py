"""Verify S2-3 persistence and baseline parity while S2-2 integration is pending."""

import argparse

import numpy as np

from asi_project.config import find_repo_root, get_contract, load_project_config
from asi_project.data import filter_training_rows, load_dataset, normalize_input
from asi_project.metrics import compute_metrics, load_metrics, validate_metrics_record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-mode", choices=["sample", "full"], default="sample")
    parser.add_argument("--compare-baseline", action="store_true")
    args = parser.parse_args(argv)
    if args.compare_baseline and args.data_mode != "full":
        parser.error(
            "--compare-baseline requires --data-mode full (the tracked baseline uses full data)"
        )

    # Load S2 dependencies only when this optional component check is invoked.
    from kedro.framework.session import KedroSession
    from kedro.framework.startup import bootstrap_project
    from kedro.io import MemoryDataset
    from kedro.runner import SequentialRunner

    from asi_project.pipelines.data_science.nodes import split_data
    from asi_project.pipelines.data_science.pipeline import create_pipeline

    root = find_repo_root()
    config = load_project_config(root)
    contract = get_contract(config)
    raw = load_dataset(config, contract, mode=args.data_mode)
    prepared = filter_training_rows(normalize_input(raw), contract)
    bootstrap_project(root)
    with KedroSession.create(
        project_path=root, env="ci" if args.data_mode == "sample" else "local"
    ) as session:
        context = session.load_context()
        catalog = context.catalog
        catalog["prepared_data"] = MemoryDataset(data=prepared)
        SequentialRunner().run(create_pipeline(), catalog)
        record = catalog.load("metrics")
        model = catalog.load("trained_model")
        _, X_test, _, y_test = split_data(prepared, context.params["project"])
        probabilities = model.predict_proba(X_test)
        if (
            probabilities.shape != (len(X_test), 2)
            or not np.isfinite(probabilities).all()
            or not np.allclose(probabilities.sum(axis=1), 1)
        ):
            raise ValueError("Reloaded model did not produce valid binary probabilities.")
        if validate_metrics_record(record, contract):
            raise ValueError("Persisted metrics do not satisfy the dataset contract.")
        if (
            compute_metrics(y_test, model.predict(X_test))
            != record["models"]["logistic_regression"]
        ):
            raise ValueError("Reloaded model predictions do not reproduce persisted metrics.")
        if args.compare_baseline:
            if record != load_metrics(config.metrics_path):
                raise ValueError("Full-data output differs from the tracked Sprint 1 baseline.")
            print("Full metric record exactly matches Sprint 1.")
        print(f"OK: {args.data_mode} component run, persisted model reload and metric parity.")
        print("prepared_data supplied by S1 helpers; this is not the integrated raw-data CLI run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
