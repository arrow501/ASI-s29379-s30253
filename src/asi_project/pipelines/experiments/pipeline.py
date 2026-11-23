"""Train and validate locally; the final test is not an input to training."""

from kedro.pipeline import Pipeline, node

from .nodes import evaluate_frozen_experiments, run_experiments, split_experiment_data


def create_pipeline(include_test=False):
    pipeline = Pipeline(
        [
            node(
                split_experiment_data,
                ["prepared_data", "params:project"],
                [
                    "experiment_X_train",
                    "experiment_X_validation",
                    "experiment_y_train",
                    "experiment_y_validation",
                    "experiment_split",
                ],
                name="split_experiment_data",
            ),
            node(
                run_experiments,
                [
                    "experiment_X_train",
                    "experiment_X_validation",
                    "experiment_y_train",
                    "experiment_y_validation",
                    "experiment_split",
                    "raw_data",
                    "params:project",
                    "params:training",
                    "params:experiments",
                    "params:reporting.data_mode",
                    "experiment_store",
                ],
                "experiment_report",
                name="run_local_experiments",
            ),
        ]
    )
    if include_test:
        pipeline += Pipeline(
            [
                node(
                    evaluate_frozen_experiments,
                    ["prepared_data", "raw_data", "params:project", "experiment_report"],
                    "experiment_test_report",
                    name="evaluate_frozen_experiments",
                )
            ]
        )
    return pipeline
