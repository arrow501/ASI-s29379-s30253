"""Random bagging and expanding chronological folds inside the development partition."""

import logging
import math
import pickle
import platform
import time
import uuid
from datetime import UTC, datetime
from functools import partial
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from asi_project.config import ConfigError
from asi_project.data import DataContractError
from asi_project.metrics import compute_metrics
from asi_project.pipelines.data_science.nodes import split_data, train_models
from asi_project.tracking import data_fingerprint, git_provenance

from .nodes import (
    _fit_autogluon,
    _write_json,
    positive_probabilities,
    validate_experiment_settings,
    validation_threshold,
)

LOGGER = logging.getLogger(__name__)


def chronological_folds(n_rows, validation_fraction, n_folds):
    """Expanding prefixes followed by disjoint, equally sized validation windows."""
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


def _check_protocols(settings):
    if not isinstance(settings, dict) or set(settings) != {"bagging_folds", "chronological_folds"}:
        raise ConfigError("validation_protocols requires bagging_folds and chronological_folds.")
    for key, value in settings.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ConfigError(f"validation_protocols.{key} must be an integer >= 2.")


def _fit_bagged(X, y, project, candidate, path, experiments, n_folds):
    from autogluon.features.generators import AutoMLPipelineFeatureGenerator
    from autogluon.tabular import TabularPredictor

    X = X.astype({name: "float64" for name in project["numeric_features"]})
    predictor = TabularPredictor(
        label=project["target"],
        problem_type="binary",
        positive_class=1,
        eval_metric=experiments["stopping_metric"],
        path=str(path),
        verbosity=2,
        learner_kwargs={"random_state": project["random_state"]},
    )
    predictor.fit(
        X.assign(**{project["target"]: y}),
        feature_generator=AutoMLPipelineFeatureGenerator(
            enable_text_ngram_features=False,
            enable_text_special_features=False,
            enable_raw_text_features=False,
            verbosity=0,
        ),
        hyperparameters=candidate["hyperparameters"],
        time_limit=candidate["time_limit"],
        num_cpus=experiments["num_cpus"],
        num_gpus=0,
        num_bag_folds=n_folds,
        num_bag_sets=1,
        num_stack_levels=0,
        dynamic_stacking=False,
        refit_full=False,
        calibrate_decision_threshold=False,
        fit_strategy="sequential",
        raise_on_model_failure=True,
        ag_args_fit={"stopping_metric": experiments["stopping_metric"]},
        ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
    )
    oof = predictor.predict_proba_oof()
    if not oof.index.equals(y.index) or len(oof) != len(y):
        raise DataContractError("AutoGluon OOF rows must match development rows exactly.")
    return predictor, oof[1].to_numpy()


def _result(y, probability, experiments, source, fold_slices):
    if not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
        raise ValueError("Invalid out-of-fold probabilities.")
    threshold = (
        validation_threshold(y, probability)
        if experiments["threshold_mode"] == "validation_f1"
        else 0.5
    )
    fold_scores = [
        compute_metrics(y.iloc[a:b], probability[a:b] >= threshold)["f1"] for a, b in fold_slices
    ]
    return {
        "validation": compute_metrics(y, probability >= threshold),
        "validation_at_0_5": compute_metrics(y, probability >= 0.5),
        "selection_score": float(np.mean(fold_scores)),
        "fold_f1": fold_scores,
        "fold_f1_min": float(min(fold_scores)),
        "fold_f1_std": float(np.std(fold_scores)),
        "decision_threshold": threshold,
        "threshold_source": source if experiments["threshold_mode"] == "validation_f1" else "fixed",
    }


def run_cross_validation(
    prepared_data,
    raw_data,
    project,
    training,
    experiments,
    protocols,
    data_mode,
    store,
    *,
    protocol,
):
    validate_experiment_settings(experiments)
    _check_protocols(protocols)
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
    root = Path(store) / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + protocol + "-" + uuid.uuid4().hex[:8]
    )
    root.mkdir(parents=True, exist_ok=False)
    _write_json(
        root / "config.json",
        {
            "project": project,
            "training": training,
            "experiments": experiments,
            "validation_protocols": protocols,
            "protocol": protocol,
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
    fold_summary = [
        {"train_positions": train.tolist(), "validation_positions": valid.tolist()}
        for train, valid in folds
    ]
    _write_json(
        root / "folds.json",
        {
            "n_development": len(y),
            "n_test": len(y_test),
            "test_positions": [len(y), len(prepared_data)],
            "folds": fold_summary,
        },
    )
    report = {
        "run_dir": str(root),
        "status": "running",
        "protocol": protocol,
        "evaluation_set": source,
        "selection_criterion": "mean fold F1; pooled OOF F1 for bagging",
        "test_evaluated": False,
        "results": {},
    }
    if protocol == "bagged":
        report["limitation"] = (
            "Random folds mix development periods; AutoGluon shares feature generation across folds. RF/XT may use native OOB."
        )
    _write_json(root / "metrics.json", report)
    candidates = [{"name": "baseline"}]
    if experiments["autogluon_enabled"]:
        candidates += experiments["candidates"]
    try:
        for candidate in candidates:
            name = candidate["name"]
            LOGGER.info("%s: %s", protocol, name)
            start = time.perf_counter()
            if name == "baseline" or protocol == "chronological":
                probabilities, labels, fold_slices = [], [], []
                offset = 0
                for number, (train, valid) in enumerate(folds):
                    if name == "baseline":
                        model, _ = train_models(X.iloc[train], y.iloc[train], project, training)
                    else:
                        model = _fit_autogluon(
                            X.iloc[train],
                            X.iloc[valid],
                            y.iloc[train],
                            y.iloc[valid],
                            project,
                            candidate,
                            root / name / f"fold_{number + 1}",
                            experiments["num_cpus"],
                            experiments["stopping_metric"],
                        )
                    probabilities.append(positive_probabilities(model, X.iloc[valid]))
                    labels.append(y.iloc[valid])
                    fold_slices.append((offset, offset + len(valid)))
                    offset += len(valid)
                import pandas as pd

                probability = np.concatenate(probabilities)
                target = pd.concat(labels)
                if name == "baseline":
                    model, _ = train_models(X, y, project, training)
                    path = root / "baseline.pkl"
                    path.write_bytes(pickle.dumps(model))
                else:
                    # Last chronological fold's train+validation is exactly the development 80%.
                    # Refit its selected model and dependencies; no random holdout or test rows.
                    model.refit_full(
                        model="best",
                        set_best_to_refit_full=True,
                        num_cpus=experiments["num_cpus"],
                        num_gpus=0,
                        fit_strategy="sequential",
                    )
                    path = Path(model.path)
            else:
                model, probability = _fit_bagged(
                    X, y, project, candidate, root / name, experiments, protocols["bagging_folds"]
                )
                target = y
                fold_slices = [(0, len(y))]
                path = root / name
            # Bagging compares pooled random OOF scores; chronological CV compares mean window F1.
            if protocol == "bagged":
                fold_slices = [(0, len(target))]
            result = _result(target, probability, experiments, source, fold_slices)
            result.update(
                model_path=str(path),
                training_seconds=time.perf_counter() - start,
                final_training_rows=len(y),
            )
            if name != "baseline":
                from autogluon.tabular import TabularPredictor

                restored = TabularPredictor.load(str(path), verbosity=0)
                result.update(best_model=restored.model_best, models=restored.model_names())
            else:
                restored = pickle.loads(path.read_bytes())
            np.testing.assert_allclose(
                positive_probabilities(model, X.iloc[-5:]),
                positive_probabilities(restored, X.iloc[-5:]),
            )
            report["results"][name] = result
            _write_json(
                root / f"{name}_oof.json",
                {
                    "indices": target.index.tolist(),
                    "target": target.tolist(),
                    "probability": probability.tolist(),
                },
            )
            _write_json(root / "metrics.json", report)
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        _write_json(root / "metrics.json", report)
    return report


def create_pipeline(protocol, include_test=False):
    from kedro.pipeline import Pipeline, node

    from .nodes import evaluate_frozen_experiments

    pipeline = Pipeline(
        [
            node(
                partial(run_cross_validation, protocol=protocol),
                [
                    "prepared_data",
                    "raw_data",
                    "params:project",
                    "params:training",
                    "params:experiments",
                    "params:validation_protocols",
                    "params:reporting.data_mode",
                    "experiment_store",
                ],
                "experiment_report",
                name=f"run_{protocol}_experiments",
            )
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
