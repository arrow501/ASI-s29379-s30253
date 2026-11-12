"""Register both Sprint 2 pipelines and their default composition."""

from kedro.pipeline import Pipeline

from asi_project.pipelines.data_processing.pipeline import create_pipeline as data_processing
from asi_project.pipelines.data_science.pipeline import create_pipeline as data_science


def register_pipelines() -> dict[str, Pipeline]:
    processing = data_processing()
    science = data_science()
    return {
        "data_processing": processing,
        "data_science": science,
        "__default__": processing + science,
    }
