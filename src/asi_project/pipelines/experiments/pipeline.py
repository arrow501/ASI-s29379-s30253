"""Explicit prepare → train → freeze decision → optional final test graphs."""

from kedro.pipeline import Pipeline, node

from .cross_validation import train_cross_validated_models
from .evaluation import evaluate_test, freeze_selection
from .nodes import train_holdout_models
from .splits import prepare_bagged_folds, prepare_chronological_folds, split_experiment_data


def create_pipeline(protocol="holdout", include_test=False):
    shared = {
        "raw_data": "raw_data",
        "project": "params:project",
        "training": "params:training",
        "experiments": "params:experiments",
        "data_mode": "params:reporting.data_mode",
        "store": "experiment_store",
    }
    if protocol == "holdout":
        prepare = node(
            split_experiment_data,
            {"prepared_data": "prepared_data", "project": "params:project"},
            [
                "experiment_X_train",
                "experiment_X_validation",
                "experiment_y_train",
                "experiment_y_validation",
                "experiment_split",
            ],
            name="split_holdout",
        )
        train = node(
            train_holdout_models,
            {
                **shared,
                "X_train": "experiment_X_train",
                "X_validation": "experiment_X_validation",
                "y_train": "experiment_y_train",
                "y_validation": "experiment_y_validation",
                "split_summary": "experiment_split",
            },
            "experiment_report",
            name="train_holdout_models",
        )
    elif protocol in {"bagged", "chronological"}:
        prepare = node(
            prepare_bagged_folds if protocol == "bagged" else prepare_chronological_folds,
            {
                "prepared_data": "prepared_data",
                "project": "params:project",
                "protocols": "params:validation_protocols",
            },
            ["experiment_X_development", "experiment_y_development", "experiment_fold_plan"],
            name=f"prepare_{protocol}_folds",
        )
        train = node(
            train_cross_validated_models,
            {
                **shared,
                "X": "experiment_X_development",
                "y": "experiment_y_development",
                "plan": "experiment_fold_plan",
                "protocols": "params:validation_protocols",
            },
            "experiment_report",
            name=f"train_{protocol}_models",
        )
    else:
        raise ValueError(f"Unknown experiment protocol: {protocol!r}")
    pipeline = Pipeline(
        [
            prepare,
            train,
            node(
                freeze_selection,
                {"report": "experiment_report"},
                "experiment_selection",
                name="freeze_validation_decision",
            ),
        ]
    )
    if include_test:
        pipeline += Pipeline(
            [
                node(
                    evaluate_test,
                    {
                        "prepared_data": "prepared_data",
                        "raw_data": "raw_data",
                        "project": "params:project",
                        "selection": "experiment_selection",
                    },
                    "experiment_test_report",
                    name="evaluate_frozen_test",
                )
            ]
        )
    return pipeline
