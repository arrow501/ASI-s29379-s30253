"""Positional chronological splits; final test rows never reach fit nodes."""

import math

import numpy as np
from sklearn.model_selection import StratifiedKFold

from asi_project.config import ConfigError
from asi_project.data import DataContractError
from asi_project.pipelines.data_science.nodes import split_data

from .settings import validate_protocol_settings


def prepare_bagged_folds(prepared_data, project, protocols):
    return prepare_cross_validation(prepared_data, project, protocols, protocol="bagged")


def prepare_chronological_folds(prepared_data, project, protocols):
    return prepare_cross_validation(prepared_data, project, protocols, protocol="chronological")


def prepare_cross_validation(prepared_data, project, protocols, *, protocol):
    validate_protocol_settings(protocols)
    if protocol not in {"bagged", "chronological"} or project["split_strategy"] != "time_ordered":
        raise ConfigError("Expected bagged/chronological protocol with a chronological outer test.")
    X, _, y, y_test = split_data(prepared_data, project)
    if not X.index.is_unique:
        raise DataContractError("Development indices must be unique.")
    if protocol == "chronological":
        folds = chronological_folds(
            len(y), project["validation_size"], protocols["chronological_folds"]
        )
        source = "chronological forward-fold predictions"
    else:
        if y.value_counts().min() < protocols["bagging_folds"]:
            raise DataContractError("Each class needs at least bagging_folds development rows.")
        folds = list(
            StratifiedKFold(
                protocols["bagging_folds"], shuffle=True, random_state=project["random_state"]
            ).split(X, y)
        )
        source = "random-fold OOF predictions"
    for train, valid in folds:
        if y.iloc[train].nunique() != 2 or y.iloc[valid].nunique() != 2:
            raise DataContractError("Each training and validation fold must contain both classes.")
    plan = {
        "protocol": protocol,
        "evaluation_set": source,
        "n_development": len(y),
        "n_test": len(y_test),
        "test_positions": [len(y), len(prepared_data)],
        "folds": [(train.tolist(), valid.tolist()) for train, valid in folds],
    }
    return X, y, plan


def split_experiment_data(prepared_data, project):
    if project["split_strategy"] != "time_ordered":
        raise ConfigError("S3 experiments require the agreed chronological split.")
    X_development, X_test, y_development, y_test = split_data(prepared_data, project)
    fraction = project["validation_size"]
    cut = len(X_development) - math.ceil(len(X_development) * fraction)
    X_train, X_validation = X_development.iloc[:cut], X_development.iloc[cut:]
    y_train, y_validation = y_development.iloc[:cut], y_development.iloc[cut:]
    for name, target in (("train", y_train), ("validation", y_validation)):
        if target.empty or target.nunique() != 2:
            raise DataContractError(f"{name} must be nonempty and contain both classes.")
    summary = {
        "strategy": "time_ordered",
        "test_size": project["test_size"],
        "validation_size_of_development": fraction,
        "n_train": len(y_train),
        "n_validation": len(y_validation),
        "n_test": len(y_test),
        "class_counts": {
            "train": {str(label): int(count) for label, count in y_train.value_counts().items()},
            "validation": {
                str(label): int(count) for label, count in y_validation.value_counts().items()
            },
        },
        "positions": {
            "train": [0, cut],
            "validation": [cut, len(X_development)],
            "test": [len(X_development), len(prepared_data)],
        },
    }
    # No holdout rows reach the fit/evaluation node. Reconstruct them for final S3-2 evaluation.
    return X_train, X_validation, y_train, y_validation, summary


def chronological_folds(n_rows, validation_fraction, n_folds):
    """Expanding prefixes followed by disjoint, equally sized validation windows."""
    if isinstance(n_rows, bool) or not isinstance(n_rows, int) or n_rows < 1:
        raise ConfigError("n_rows must be a positive integer.")
    if isinstance(n_folds, bool) or not isinstance(n_folds, int) or n_folds < 2:
        raise ConfigError("chronological_folds must be an integer >= 2.")
    if (
        isinstance(validation_fraction, bool)
        or not isinstance(validation_fraction, int | float)
        or not 0 < validation_fraction < 1
    ):
        raise ConfigError("validation fraction must be in (0, 1).")
    size = math.ceil(n_rows * validation_fraction)
    first = n_rows - n_folds * size
    if first < 2:
        raise ConfigError("Not enough development rows for the requested chronological folds.")
    return [(np.arange(stop), np.arange(stop, stop + size)) for stop in range(first, n_rows, size)]
