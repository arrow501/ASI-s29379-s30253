"""Chronological local experiments without online services or a production package."""

import json
import logging
import math
import pickle
import platform
import re
import time
import uuid
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, log_loss, precision_recall_curve, roc_auc_score

from asi_project.config import ConfigError
from asi_project.data import DataContractError
from asi_project.metrics import compute_metrics
from asi_project.pipelines.data_science.nodes import _validate_xy, split_data, train_models
from asi_project.tracking import data_fingerprint, git_provenance

LOGGER = logging.getLogger(__name__)


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


def validate_experiment_settings(settings):
    if not isinstance(settings, dict) or set(settings) != {
        "autogluon_enabled",
        "num_cpus",
        "candidates",
        "stopping_metric",
        "threshold_mode",
    }:
        raise ConfigError(
            "experiments requires autogluon_enabled, num_cpus, candidates, stopping_metric and threshold_mode."
        )
    if settings["stopping_metric"] not in {"log_loss", "roc_auc", "average_precision"}:
        raise ConfigError(
            "stopping_metric must use probabilities: log_loss, roc_auc or average_precision."
        )
    if settings["threshold_mode"] not in {"fixed", "validation_f1"}:
        raise ConfigError("threshold_mode must be fixed or validation_f1.")
    if not isinstance(settings["autogluon_enabled"], bool):
        raise ConfigError("experiments.autogluon_enabled must be boolean.")
    cpus = settings["num_cpus"]
    if isinstance(cpus, bool) or not isinstance(cpus, int) or cpus < 1:
        raise ConfigError("experiments.num_cpus must be a positive integer.")
    candidates = settings["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise ConfigError("experiments.candidates must be a nonempty list.")
    names = {"baseline"}
    for candidate in candidates:
        if not isinstance(candidate, dict) or set(candidate) != {
            "name",
            "time_limit",
            "hyperparameters",
        }:
            raise ConfigError("Each candidate requires name, time_limit and hyperparameters.")
        name = candidate["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name in names:
            raise ConfigError(
                "Candidate names must be unique safe identifiers (baseline is reserved)."
            )
        names.add(name)
        limit = candidate["time_limit"]
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int | float)
            or not 0 < limit < math.inf
        ):
            raise ConfigError("Each candidate time_limit must be finite and positive.")
        hyperparameters = candidate["hyperparameters"]
        if (
            not isinstance(hyperparameters, dict)
            or not hyperparameters
            or not set(hyperparameters) <= {"GBM", "CAT", "RF", "XT"}
        ):
            raise ConfigError("Supported local backends: GBM, CAT, RF and XT.")
        if any(not isinstance(params, dict) for params in hyperparameters.values()):
            raise ConfigError("Each backend configuration must be a mapping.")


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def positive_probabilities(model, X):
    probabilities = model.predict_proba(X)
    if hasattr(probabilities, "loc"):
        probabilities = probabilities.loc[:, [0, 1]].to_numpy()
    else:
        probabilities = probabilities[:, np.asarray(model.classes_).argsort()]
    if (
        probabilities.shape != (len(X), 2)
        or not np.isfinite(probabilities).all()
        or not ((probabilities >= 0) & (probabilities <= 1)).all()
        or not np.allclose(probabilities.sum(axis=1), 1)
    ):
        raise ValueError("Invalid binary probabilities.")
    return probabilities[:, 1]


def validation_threshold(y, probability):
    """Maximize validation F1; ties choose the highest threshold."""
    precision, recall, thresholds = precision_recall_curve(y, probability)
    denominator = precision[:-1] + recall[:-1]
    scores = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )
    return float(thresholds[np.flatnonzero(scores == scores.max())[-1]])


def _validation_result(model, X, y, threshold_mode="fixed"):
    start = time.perf_counter()
    probability = positive_probabilities(model, X)
    seconds = time.perf_counter() - start
    threshold = validation_threshold(y, probability) if threshold_mode == "validation_f1" else 0.5
    prediction = (probability >= threshold).astype(int)
    return {
        "validation": compute_metrics(y, prediction),
        "validation_at_0_5": compute_metrics(y, (probability >= 0.5).astype(int)),
        "validation_diagnostics": {
            "positive_rate": float(y.mean()),
            "predicted_positive_count": int(prediction.sum()),
            "max_probability": float(probability.max()),
            "roc_auc": float(roc_auc_score(y, probability)),
            "average_precision": float(average_precision_score(y, probability)),
            "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        },
        "prediction_seconds": seconds,
        "prediction_rows": len(X),
        "decision_threshold": threshold,
        "threshold_source": "validation" if threshold_mode == "validation_f1" else "fixed",
    }


def _fit_autogluon(
    X_train,
    X_validation,
    y_train,
    y_validation,
    project,
    candidate,
    path,
    cpus,
    stopping_metric="log_loss",
):
    # This optional import never runs in baseline-only CI.
    from autogluon.features.generators import AutoMLPipelineFeatureGenerator
    from autogluon.tabular import TabularPredictor

    # Nullable pandas Int64 (pdays) is not recognized by AG's NumPy dtype inference.
    numeric_types = {name: "float64" for name in project["numeric_features"]}
    X_train = X_train.astype(numeric_types)
    X_validation = X_validation.astype(numeric_types)

    # A pre-fitted generator prevents AutoGluon's default fit on train + tuning_data.
    generator = AutoMLPipelineFeatureGenerator(
        enable_text_ngram_features=False,
        enable_text_special_features=False,
        enable_raw_text_features=False,
        verbosity=0,
    )
    generator.fit_transform(X_train, y_train)
    predictor = TabularPredictor(
        label=project["target"],
        problem_type="binary",
        positive_class=1,
        # Inner model/ensemble selection must also avoid the flat F1-at-0.5 plateau.
        # The project still compares candidates using validation F1 after threshold selection.
        eval_metric=stopping_metric,
        path=str(path),
        verbosity=2,
        log_to_file=True,
        learner_kwargs={"random_state": project["random_state"]},
    )
    predictor.fit(
        train_data=X_train.assign(**{project["target"]: y_train}),
        tuning_data=X_validation.assign(**{project["target"]: y_validation}),
        feature_generator=generator,
        hyperparameters=candidate["hyperparameters"],
        time_limit=candidate["time_limit"],
        num_cpus=cpus,
        num_gpus=0,
        num_bag_folds=0,
        num_stack_levels=0,
        dynamic_stacking=False,
        calibrate_decision_threshold=False,
        refit_full=False,
        fit_strategy="sequential",
        raise_on_model_failure=True,
        ag_args_fit={"stopping_metric": stopping_metric},
    )
    return predictor


def run_experiments(
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
    root = Path(store) / (datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8])
    root.mkdir(parents=True, exist_ok=False)
    _write_json(
        root / "config.json",
        {
            "project": project,
            "training": training,
            "experiments": experiments,
            "data_mode": data_mode,
        },
    )
    packages = ["numpy", "pandas", "scikit-learn"]
    if experiments["autogluon_enabled"]:
        packages += ["autogluon.tabular", "lightgbm", "catboost"]
    _write_json(
        root / "provenance.json",
        {
            **git_provenance(),
            "data_fingerprint": data_fingerprint(raw_data),
            "fingerprint_method": "loaded-data-csv-v1",
            "python": platform.python_version(),
            "versions": {p: version(p) for p in packages},
        },
    )
    _write_json(root / "split.json", split_summary)
    report = {
        "run_dir": str(root),
        "status": "running",
        "evaluation_set": "validation",
        "test_evaluated": False,
        "results": {},
    }
    _write_json(root / "metrics.json", report)
    try:
        start = time.perf_counter()
        model, dummy = train_models(X_train, y_train, project, training)
        seconds = time.perf_counter() - start
        model_path = root / "baseline.pkl"
        model_path.write_bytes(pickle.dumps(model))
        restored = pickle.loads(model_path.read_bytes())
        np.testing.assert_allclose(
            model.predict_proba(X_validation), restored.predict_proba(X_validation)
        )
        report["results"]["baseline"] = {
            **_validation_result(
                restored, X_validation, y_validation, experiments["threshold_mode"]
            ),
            "training_seconds": seconds,
            "model_path": str(model_path),
        }
        report["results"]["dummy"] = _validation_result(dummy, X_validation, y_validation)
        dummy_path = root / "dummy.pkl"
        dummy_path.write_bytes(pickle.dumps(dummy))
        report["results"]["dummy"]["model_path"] = str(dummy_path)
        _write_json(root / "metrics.json", report)
        if experiments["autogluon_enabled"]:
            from autogluon.tabular import TabularPredictor

            for candidate in experiments["candidates"]:
                name = candidate["name"]
                LOGGER.info("Local experiment %s (limit=%ss)", name, candidate["time_limit"])
                start = time.perf_counter()
                predictor = _fit_autogluon(
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
                restored = TabularPredictor.load(str(root / name), verbosity=0)
                np.testing.assert_allclose(
                    predictor.predict_proba(X_validation), restored.predict_proba(X_validation)
                )
                report["results"][name] = {
                    **_validation_result(
                        restored, X_validation, y_validation, experiments["threshold_mode"]
                    ),
                    "training_seconds": seconds,
                    "model_path": str(root / name),
                    "best_model": restored.model_best,
                    "models": restored.model_names(),
                }
                _write_json(root / "metrics.json", report)
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        _write_json(root / "metrics.json", report)
    LOGGER.info("Local results: %s", root)
    return report


def evaluate_frozen_experiments(prepared_data, raw_data, project, report):
    """Freeze validation-based selection before evaluating any final test predictions."""
    if report["status"] != "completed":
        raise ValueError("Test evaluation requires a completed experiment suite.")
    root = Path(report["run_dir"])
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    if config["project"] != project or provenance["data_fingerprint"] != data_fingerprint(raw_data):
        raise ValueError("Test inputs must match the experiment configuration and fingerprint.")
    eligible = {name: result for name, result in report["results"].items() if name != "dummy"}
    selected = min(
        eligible,
        key=lambda name: (
            -eligible[name].get("selection_score", eligible[name]["validation"]["f1"]),
            name,
        ),
    )
    selection = {
        "selected_candidate": selected,
        "criterion": report.get("selection_criterion", "validation F1")
        + "; ties resolved alphabetically",
        "selection_score": eligible[selected].get(
            "selection_score", eligible[selected]["validation"]["f1"]
        ),
        "validation_f1": eligible[selected]["validation"]["f1"],
        "decision_threshold": eligible[selected]["decision_threshold"],
        "threshold_source": eligible[selected]["threshold_source"],
        "test_used_for_selection": False,
    }
    # Exclusive creation prevents silently rewriting a decision after inspecting the test.
    with (root / "selection.json").open("x", encoding="utf-8") as handle:
        json.dump(selection, handle, indent=2, allow_nan=False)
        handle.write("\n")
    _, X_test, _, y_test = split_data(prepared_data, project)
    test_report = {**selection, "n_test": len(y_test), "status": "running", "results": {}}
    try:
        for name, result in report["results"].items():
            if name in {"baseline", "dummy"}:
                model = pickle.loads(Path(result["model_path"]).read_bytes())
            else:
                from autogluon.tabular import TabularPredictor

                model = TabularPredictor.load(result["model_path"], verbosity=0)
            probability = positive_probabilities(model, X_test)
            threshold = result["decision_threshold"]
            test_report["results"][name] = {
                "test": compute_metrics(y_test, (probability >= threshold).astype(int)),
                "test_at_0_5": compute_metrics(y_test, (probability >= 0.5).astype(int)),
                "decision_threshold": threshold,
                "threshold_source": result["threshold_source"],
            }
            _write_json(root / "test_metrics.json", test_report)
        test_report["status"] = "completed"
    except Exception as error:
        test_report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        _write_json(root / "test_metrics.json", test_report)
        _write_json(
            root / "metrics.json", {**report, "test_evaluated": bool(test_report["results"])}
        )
    LOGGER.info("Final test report: %s", root / "test_metrics.json")
    return test_report
