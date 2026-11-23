"""Register both Sprint 2 pipelines and their default composition."""

from kedro.pipeline import Pipeline, node

from asi_project.pipelines.data_processing.pipeline import create_pipeline as data_processing
from asi_project.pipelines.data_science.pipeline import create_pipeline as data_science
from asi_project.pipelines.experiments.cross_validation import create_pipeline as cross_validation
from asi_project.pipelines.experiments.pipeline import create_pipeline as experiments
from asi_project.tracking import log_baseline


def register_pipelines() -> dict[str, Pipeline]:
    processing = data_processing()
    science = data_science()
    return {
        "data_processing": processing,
        "data_science": science,
        "__default__": processing + science,
        "experiments": processing + experiments(),
        "experiments_test": processing + experiments(include_test=True),
        "experiments_bagged": processing + cross_validation("bagged"),
        "experiments_bagged_test": processing + cross_validation("bagged", include_test=True),
        "experiments_crossfold": processing + cross_validation("chronological"),
        "experiments_crossfold_test": processing
        + cross_validation("chronological", include_test=True),
        "baseline_tracking": processing
        + science
        + Pipeline(
            [
                node(
                    log_baseline,
                    ["raw_data", "metrics", "params:project", "params:training", "params:tracking"],
                    None,
                    name="log_baseline",
                )
            ]
        ),
    }
