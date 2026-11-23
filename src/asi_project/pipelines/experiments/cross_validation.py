"""Train on a prepared fold plan and refit on development; no final test inputs."""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from asi_project.pipelines.data_science.nodes import train_models

from .artifacts import create_run, write_json
from .models import fit_bagged_autogluon, fit_holdout_autogluon, save_sklearn, verify_reload
from .scoring import fold_result, positive_probabilities
from .settings import validate_experiment_settings, validate_protocol_settings

LOGGER = logging.getLogger(__name__)


def train_cross_validated_models(
    X,
    y,
    plan,
    raw_data,
    project,
    training,
    experiments,
    protocols,
    data_mode,
    store,
):
    validate_experiment_settings(experiments)
    validate_protocol_settings(protocols)
    protocol = plan["protocol"]
    folds = plan["folds"]
    source = plan["evaluation_set"]
    root, report = create_run(
        store,
        raw_data,
        project,
        training,
        experiments,
        data_mode,
        protocol=protocol,
        evaluation_set=source,
        protocols=protocols,
    )
    report["selection_criterion"] = (
        "mean forward-window F1" if protocol == "chronological" else "pooled random OOF F1"
    )
    if protocol == "bagged":
        report["limitation"] = (
            "Random folds mix development periods; AG shares feature generation across folds. RF/XT may use native OOB."
        )
    candidates = [{"name": "baseline"}]
    if experiments["autogluon_enabled"]:
        candidates += experiments["candidates"]
    try:
        write_json(root / "folds.json", plan)
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
                        model = fit_holdout_autogluon(
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
                probability = np.concatenate(probabilities)
                target = pd.concat(labels)
                if name == "baseline":
                    model, _ = train_models(X, y, project, training)
                    path = root / "baseline.pkl"
                    save_sklearn(model, path)
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
                model, probability = fit_bagged_autogluon(
                    X, y, project, candidate, root / name, experiments, protocols["bagging_folds"]
                )
                target = y
                fold_slices = [(0, len(y))]
                path = root / name
            # Bagging compares pooled random OOF scores; chronological CV compares mean window F1.
            if protocol == "bagged":
                fold_slices = [(0, len(target))]
            result = fold_result(target, probability, experiments, source, fold_slices)
            result.update(
                model_path=str(path),
                training_seconds=time.perf_counter() - start,
                final_training_rows=len(y),
            )
            kind = "sklearn" if name == "baseline" else "autogluon"
            restored = verify_reload(model, path, kind, X.iloc[-5:])
            result["model_kind"] = kind
            if kind == "autogluon":
                result.update(best_model=restored.model_best, models=restored.model_names())
            report["results"][name] = result
            write_json(
                root / f"{name}_oof.json",
                {
                    "indices": target.index.tolist(),
                    "target": target.tolist(),
                    "probability": probability.tolist(),
                },
            )
            write_json(root / "metrics.json", report)
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(root / "metrics.json", report)
    return report
