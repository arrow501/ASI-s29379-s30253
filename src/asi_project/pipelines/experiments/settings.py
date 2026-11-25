"""Validate the supported local experiment configuration."""

import math
import re

from asi_project.config import ConfigError


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
    if not isinstance(settings["stopping_metric"], str) or settings["stopping_metric"] not in {
        "log_loss",
        "roc_auc",
        "average_precision",
    }:
        raise ConfigError(
            "stopping_metric must use probabilities: log_loss, roc_auc or average_precision."
        )
    if not isinstance(settings["threshold_mode"], str) or settings["threshold_mode"] not in {
        "fixed",
        "validation_f1",
    }:
        raise ConfigError("threshold_mode must be fixed or validation_f1.")
    if not isinstance(settings["autogluon_enabled"], bool):
        raise ConfigError("experiments.autogluon_enabled must be boolean.")
    cpus = settings["num_cpus"]
    if isinstance(cpus, bool) or not isinstance(cpus, int) or cpus < 1:
        raise ConfigError("experiments.num_cpus must be a positive integer.")
    candidates = settings["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise ConfigError("experiments.candidates must be a nonempty list.")
    names = {"baseline", "dummy"}
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
                "Candidate names must be unique safe identifiers (baseline and dummy are reserved)."
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


def validate_protocol_settings(settings):
    if not isinstance(settings, dict) or set(settings) != {"bagging_folds", "chronological_folds"}:
        raise ConfigError("validation_protocols requires bagging_folds and chronological_folds.")
    for key, value in settings.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ConfigError(f"validation_protocols.{key} must be an integer >= 2.")
