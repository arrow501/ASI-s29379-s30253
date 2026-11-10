"""Baseline pipeline consuming the prepared_data boundary from S2-2."""

from kedro.pipeline import Pipeline, node

from .nodes import evaluate_models, split_data, train_models


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                split_data,
                ["prepared_data", "params:project"],
                ["X_train", "X_test", "y_train", "y_test"],
                name="split_data",
            ),
            node(
                train_models,
                ["X_train", "y_train", "params:project", "params:training"],
                ["trained_model", "dummy_model"],
                name="train_models",
            ),
            node(
                evaluate_models,
                [
                    "trained_model",
                    "dummy_model",
                    "X_test",
                    "y_test",
                    "y_train",
                    "params:project",
                    "params:reporting.data_mode",
                ],
                "metrics",
                name="evaluate_models",
            ),
        ]
    )
