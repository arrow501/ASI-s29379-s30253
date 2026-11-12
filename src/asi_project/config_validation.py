"""Shared validation for project settings and resolved Kedro parameters."""

from typing import Any

from asi_project.config import ConfigError, DatasetContract, ProjectConfig
from asi_project.data import feature_list_problems


def project_parameters(config: ProjectConfig, contract: DatasetContract) -> dict[str, Any]:
    """Expose team settings and immutable contract fields to pipeline nodes."""
    return {
        "dataset_id": config.dataset_id,
        "numeric_features": list(config.numeric_features),
        "categorical_features": list(config.categorical_features),
        "test_size": config.test_size,
        "random_state": config.random_state,
        "target": contract.target,
        "positive_class": contract.positive_class,
        "split_strategy": contract.split_strategy,
    }


def validate_project_parameters(params: object, contract: DatasetContract) -> None:
    """Reject invalid effective settings before data loading or node execution."""
    if not isinstance(params, dict):
        raise ConfigError("project parameters must be a mapping.")
    problems = []
    for name, expected in (
        ("dataset_id", contract.dataset_id),
        ("target", contract.target),
        ("positive_class", contract.positive_class),
        ("split_strategy", contract.split_strategy),
    ):
        if params.get(name) != expected:
            problems.append(f"project.{name} must match the dataset contract: {expected!r}.")

    features = {}
    for name in ("numeric_features", "categorical_features"):
        names = params.get(name)
        if not isinstance(names, list) or any(
            not isinstance(value, str) or not value.strip() for value in names
        ):
            problems.append(f"project.{name} must be a list of nonempty column names.")
        else:
            features[name] = names
    if len(features) == 2:
        numeric, categorical = features["numeric_features"], features["categorical_features"]
        problems.extend(feature_list_problems(contract, numeric + categorical))
        overlap = set(numeric) & set(categorical)
        if overlap:
            problems.append(f"Features cannot be both numeric and categorical: {sorted(overlap)}.")

    test_size = params.get("test_size")
    if (
        isinstance(test_size, bool)
        or not isinstance(test_size, int | float)
        or not 0 < test_size < 1
    ):
        problems.append("project.test_size must be a number strictly between 0 and 1.")
    seed = params.get("random_state")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        problems.append("project.random_state must be an integer in 0..2**32-1.")
    if problems:
        raise ConfigError("\n".join(problems))
