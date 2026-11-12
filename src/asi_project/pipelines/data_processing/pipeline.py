"""Connect data validation, normalization and filtering into a Kedro pipeline."""

from kedro.pipeline import Pipeline, node

from .nodes import filter_data, normalize_data, validate_data


def create_pipeline(**kwargs) -> Pipeline:
    """Create the validation, normalization and filtering pipeline."""
    return Pipeline(
        [
            node(
                func=validate_data,
                inputs={"raw_data": "raw_data", "project": "params:project"},
                outputs="validated_data",
                name="validate_data",
            ),
            node(
                func=normalize_data,
                inputs="validated_data",
                outputs="normalized_data",
                name="normalize_data",
            ),
            node(
                func=filter_data,
                inputs={"normalized_data": "normalized_data", "project": "params:project"},
                outputs="prepared_data",
                name="filter_data",
            ),
        ]
    )
