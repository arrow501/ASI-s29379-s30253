"""Fit the explicit holdout protocol; orchestration lives in pipeline.py."""

import logging
import time

from asi_project.data import DataContractError
from asi_project.pipelines.data_science.nodes import _validate_xy, train_models

from .artifacts import create_run, write_json
from .models import fit_holdout_autogluon, save_sklearn, verify_reload
from .scoring import validation_result
from .settings import validate_experiment_settings

LOGGER = logging.getLogger(__name__)


def train_holdout_models(
    X_train,
    X_validation,
    y_train,
    y_validation,
    split_summary,
    raw_data,
    project,
    training,
    experiments,
    data_mode,
    store,
):
    validate_experiment_settings(experiments)
    features = project["numeric_features"] + project["categorical_features"]
    for part, X, y in (("Training", X_train, y_train), ("Validation", X_validation, y_validation)):
        _validate_xy(X, y, features, part=part, require_two_classes=True)
    if (
        not X_train.index.is_unique
        or not X_validation.index.is_unique
        or not X_train.index.intersection(X_validation.index).empty
    ):
        raise DataContractError("Training and validation indices must be unique and disjoint.")
    root, report = create_run(
        store,
        raw_data,
        project,
        training,
        experiments,
        data_mode,
        protocol="holdout",
        evaluation_set="validation",
    )
    try:
        write_json(root / "split.json", split_summary)
        start = time.perf_counter()
        model, dummy = train_models(X_train, y_train, project, training)
        seconds = time.perf_counter() - start
        model_path = root / "baseline.pkl"
        save_sklearn(model, model_path)
        restored = verify_reload(model, model_path, "sklearn", X_validation)
        report["results"]["baseline"] = {
            **validation_result(
                restored, X_validation, y_validation, experiments["threshold_mode"]
            ),
            "training_seconds": seconds,
            "model_path": str(model_path),
            "model_kind": "sklearn",
        }
        report["results"]["dummy"] = validation_result(dummy, X_validation, y_validation)
        dummy_path = root / "dummy.pkl"
        save_sklearn(dummy, dummy_path)
        report["results"]["dummy"].update(model_path=str(dummy_path), model_kind="sklearn")
        write_json(root / "metrics.json", report)
        if experiments["autogluon_enabled"]:
            for candidate in experiments["candidates"]:
                name = candidate["name"]
                LOGGER.info("Local experiment %s (limit=%ss)", name, candidate["time_limit"])
                start = time.perf_counter()
                predictor = fit_holdout_autogluon(
                    X_train,
                    X_validation,
                    y_train,
                    y_validation,
                    project,
                    candidate,
                    root / name,
                    experiments["num_cpus"],
                    experiments["stopping_metric"],
                )
                seconds = time.perf_counter() - start
                restored = verify_reload(predictor, root / name, "autogluon", X_validation)
                report["results"][name] = {
                    **validation_result(
                        restored, X_validation, y_validation, experiments["threshold_mode"]
                    ),
                    "training_seconds": seconds,
                    "model_path": str(root / name),
                    "model_kind": "autogluon",
                    "best_model": restored.model_best,
                    "models": restored.model_names(),
                }
                write_json(root / "metrics.json", report)
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(root / "metrics.json", report)
    LOGGER.info("Local results: %s", root)
    return report
