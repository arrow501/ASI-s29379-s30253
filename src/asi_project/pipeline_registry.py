"""Register the S2 baseline and three opt-in local experiment protocols."""

from kedro.pipeline import Pipeline, node

from asi_project.pipelines.data_processing.pipeline import create_pipeline as data_processing
from asi_project.pipelines.data_science.pipeline import create_pipeline as data_science
from asi_project.pipelines.experiments.pipeline import create_pipeline as experiments
from asi_project.tracking import track_baseline


def register_pipelines() -> dict[str, Pipeline]:
    processing = data_processing()
    science = data_science()
    return {
        "data_processing": processing,
        "data_science": science,
        "__default__": processing + science,
        "baseline_tracked": processing
        + science
        + Pipeline(
            [
                node(
                    track_baseline,
                    {
                        "metrics": "metrics",
                        "raw_data": "raw_data",
                        "project": "params:project",
                        "training": "params:training",
                        "store": "experiment_store",
                        "tracking": "params:tracking",
                    },
                    "baseline_tracking",
                    name="track_s2_baseline",
                )
            ]
        ),
        "experiments": processing + experiments(),
        "experiments_test": processing + experiments(include_test=True),
        "experiments_bagged": processing + experiments("bagged"),
        "experiments_bagged_test": processing + experiments("bagged", include_test=True),
        "experiments_crossfold": processing + experiments("chronological"),
        "experiments_crossfold_test": processing + experiments("chronological", include_test=True),
    }
