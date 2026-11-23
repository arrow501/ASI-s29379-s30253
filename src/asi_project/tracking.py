"""Opt-in W&B tracking for the unchanged Sprint 2 baseline."""

import hashlib
import logging
import os
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

from asi_project.config import ConfigError
from asi_project.metrics import validate_metrics_record

LOGGER = logging.getLogger(__name__)


def tracking_settings(tracking):
    """WANDB_MODE can disable an online configuration, including in CI."""
    if not isinstance(tracking, dict):
        raise ConfigError("tracking must be a mapping.")
    unknown = set(tracking) - {"mode", "entity", "project", "name"}
    if unknown:
        raise ConfigError(f"Unknown tracking parameters: {sorted(unknown)}")
    settings = {**tracking, "mode": os.environ.get("WANDB_MODE", tracking.get("mode"))}
    if settings["mode"] not in {"disabled", "offline", "online"}:
        raise ConfigError("tracking.mode / WANDB_MODE must be disabled, offline or online.")
    for key in ("entity", "project", "name"):
        value = settings.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ConfigError(f"tracking.{key} must be a nonempty string or null.")
    if settings["mode"] == "online" and not all(settings.get(key) for key in ("entity", "project")):
        raise ConfigError("Online tracking requires explicit tracking.entity and tracking.project.")
    return settings


def data_fingerprint(raw_data):
    """Hash loaded values, headers, row order and index, rather than a stale sidecar."""
    payload = raw_data.to_csv(index=True, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_provenance():
    """Anchor Git to this checkout, independent of the process working directory."""
    root = Path(__file__).resolve().parents[2]

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()

    return {"git_sha": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain"))}


def log_baseline(raw_data, metrics, project, training, tracking):
    """Log aggregate results only; failures never silently become successful online runs."""
    settings = tracking_settings(tracking)
    if settings["mode"] == "disabled":
        LOGGER.info("W&B disabled; no SDK import or network access.")
        return
    problems = validate_metrics_record(metrics)
    if problems:
        raise ValueError("Cannot track invalid metrics: " + "; ".join(problems))
    try:
        import wandb
    except ImportError as error:
        raise ConfigError("Install wandb==0.30.0 using constraints.txt before tracking.") from error

    config = {
        "project": project,
        "training": training,
        "data_mode": metrics["data_mode"],
        "split": metrics["split"],
        "data_fingerprint": data_fingerprint(raw_data),
        "fingerprint_method": "loaded-data-csv-v1",
        **git_provenance(),
        "versions": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("pandas", "scikit-learn", "wandb")},
        },
    }
    if settings["mode"] == "online" and not wandb.login(prompt=False, verify=True, timeout=15):
        raise ConfigError("W&B credentials missing. Run wandb login in your terminal first.")

    with wandb.init(
        mode=settings["mode"],
        entity=settings.get("entity"),
        project=settings.get("project") or "asi-baseline-local",
        name=settings.get("name"),
        job_type="baseline",
        tags=["sprint-3", "baseline", metrics["data_mode"]],
        config=config,
        settings=wandb.Settings(
            init_timeout=30,
            finish_timeout=30,
            finish_timeout_raises=True,
            disable_git=True,
            disable_code=True,
            disable_job_creation=True,
        ),
    ) as run:
        values = {}
        for model, scores in metrics["models"].items():
            for name, value in scores.items():
                if isinstance(value, dict):
                    values.update({f"{model}/{name}/{key}": count for key, count in value.items()})
                else:
                    values[f"{model}/{name}"] = value
        run.log(values)
        LOGGER.info("W&B run: %s (mode=%s)", run.url or run.id, settings["mode"])
