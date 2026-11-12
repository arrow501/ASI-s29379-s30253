"""Split, fit and evaluate the Sprint 1 baseline using Kedro-supplied settings."""

import warnings
from dataclasses import dataclass

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from asi_project.config import ConfigError, load_registry
from asi_project.config_validation import (
    validate_project_parameters,
    validate_reporting_parameters,
    validate_training_parameters,
)
from asi_project.data import DataContractError, assert_valid, make_xy, split_train_test
from asi_project.metrics import build_metrics_record, compute_metrics, validate_metrics_record
from asi_project.model import build_baseline_pipeline, make_dummy


def _contract(project):
    registry = load_registry()
    dataset_id = project.get("dataset_id") if isinstance(project, dict) else None
    if not isinstance(dataset_id, str) or dataset_id not in registry:
        raise ConfigError("project.dataset_id must select a registered dataset.")
    contract = registry[dataset_id]
    validate_project_parameters(project, contract)
    return contract


def _validate_xy(X, y, features, *, part, require_two_classes=False):
    """sklearn matches rows positionally; reject accidental label/index misalignment."""
    if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
        raise DataContractError(f"{part}: expected a feature DataFrame and target Series.")
    if X.empty or not X.index.equals(y.index):
        raise DataContractError(
            f"{part}: feature rows and target labels must be nonempty and aligned."
        )
    if list(X.columns) != features:
        raise DataContractError(f"{part}: features must match the configured names and order.")
    if y.isna().any() or not y.isin([0, 1]).all():
        raise DataContractError(f"{part}: target must be encoded as 0/1 without missing values.")
    if require_two_classes and y.nunique() != 2:
        raise DataContractError(f"{part}: training requires both target classes.")


@dataclass(frozen=True)
class _SplitSettings:
    test_size: float
    random_state: int


def split_data(prepared_data, project):
    """Select allowed features and preserve the contract's chronological split."""
    contract = _contract(project)
    assert_valid(prepared_data, contract)
    features = project["numeric_features"] + project["categorical_features"]
    X, y = make_xy(prepared_data, contract, features)
    return split_train_test(X, y, contract, project["test_size"], project["random_state"])


def train_models(X_train, y_train, project, training):
    """Fit all learned preprocessing on training rows only; retain a dummy comparator."""
    _contract(project)
    validate_training_parameters(training)
    features = project["numeric_features"] + project["categorical_features"]
    _validate_xy(X_train, y_train, features, part="Training", require_two_classes=True)
    max_iter = training["max_iter"]
    model = build_baseline_pipeline(
        project["numeric_features"],
        project["categorical_features"],
        max_iter=max_iter,
        random_state=project["random_state"],
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(X_train, y_train)
    except ConvergenceWarning as error:
        raise ValueError(
            f"LogisticRegression did not converge within training.max_iter={max_iter}; "
            "increase the iteration limit before saving a model."
        ) from error
    dummy = make_dummy().fit(X_train, y_train)
    return model, dummy


def evaluate_models(trained_model, dummy_model, X_test, y_test, y_train, project, data_mode):
    """Return the existing metrics schema with effective runtime split values."""
    contract = _contract(project)
    validate_reporting_parameters({"data_mode": data_mode})
    features = project["numeric_features"] + project["categorical_features"]
    _validate_xy(X_test, y_test, features, part="Evaluation")
    if list(trained_model.feature_names_in_) != features:
        raise DataContractError("Trained model features do not match evaluation metadata.")
    record = build_metrics_record(
        contract=contract,
        config=_SplitSettings(project["test_size"], project["random_state"]),
        features=features,
        n_train=len(y_train),
        n_test=len(y_test),
        data_mode=data_mode,
        results={
            "dummy": compute_metrics(y_test, dummy_model.predict(X_test)),
            "logistic_regression": compute_metrics(y_test, trained_model.predict(X_test)),
        },
    )
    problems = validate_metrics_record(record, contract)
    if problems:
        raise ValueError("Invalid evaluation record:\n" + "\n".join(problems))
    return record
