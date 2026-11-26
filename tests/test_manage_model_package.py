"""Komendy obsługi pakietów, rzeczywisty eksport lokalny i kontrolowane operacje zdalne."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from asi_project.config_validation import project_parameters
from asi_project.data import normalize_input
from asi_project.model_package import load_model_package, predict_model_package
from asi_project.pipelines.experiments.evaluation import evaluate_test, freeze_selection
from asi_project.pipelines.experiments.nodes import train_holdout_models
from asi_project.pipelines.experiments.splits import split_experiment_data


def load_command(repo_root):
    spec = importlib.util.spec_from_file_location(
        "manage_model_package_command", repo_root / "scripts/manage_model_package.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def export_source(project, tmp_path):
    raw = project.sample.iloc[:100].copy()
    raw["y"] = np.where(np.arange(len(raw)) % 2, "yes", "no")
    parameters = project_parameters(project.config, project.contract)
    settings = {
        "autogluon_enabled": False,
        "num_cpus": 1,
        "stopping_metric": "log_loss",
        "threshold_mode": "validation_f1",
        "candidates": [{"name": "gbm", "time_limit": 10, "hyperparameters": {"GBM": {}}}],
    }
    report = train_holdout_models(
        *split_experiment_data(normalize_input(raw), parameters),
        raw,
        parameters,
        {"max_iter": 1000},
        settings,
        "sample",
        tmp_path,
    )
    selection = freeze_selection(report)
    evaluate_test(normalize_input(raw), raw, parameters, selection)
    return Path(report["run_dir"]), raw


@pytest.mark.parametrize("include_test,separator", [(False, ","), (True, ";")])
def test_export_command_uses_existing_artifacts_without_fit(
    export_source, tmp_path, repo_root, include_test, separator
):
    run_dir, raw = export_source
    source = tmp_path / "raw.csv"
    raw.to_csv(source, index=False, sep=separator)
    output = tmp_path / "exported"
    script = repo_root / "scripts/manage_model_package.py"
    arguments = [
        "export",
        "--run",
        str(run_dir),
        "--candidate",
        "baseline",
        "--data",
        str(source),
        "--sep",
        separator,
        "--output",
        str(output),
    ]
    if include_test:
        arguments.append("--include-test")
    code = """
import runpy, sys
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
def forbidden(*args, **kwargs):
    raise AssertionError("export called fit")
for component in (Pipeline, SimpleImputer, OneHotEncoder, StandardScaler):
    component.fit = forbidden
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
"""
    process = subprocess.run(
        [sys.executable, "-B", "-c", code, str(script), *arguments],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(repo_root / "src"), "WANDB_MODE": "disabled"},
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(process.stdout)
    assert Path(result["package_dir"]) == output
    model, metadata = load_model_package(output)
    assert metadata["candidate_name"] == "baseline"
    assert ("test" in metadata["metrics"]) is include_test
    assert len(predict_model_package(model, metadata, raw.iloc[:5])) == 5


@pytest.mark.parametrize("production", [False, True])
def test_publish_command_delegates_the_explicit_production_choice(
    repo_root, tmp_path, monkeypatch, capsys, production
):
    command = load_command(repo_root)
    tracking = {"mode": "online", "entity": "team", "project": "project"}
    captured = {}

    def settings(env):
        captured["env"] = env
        return tracking

    def publish(package_dir, resolved, *, artifact_name, production):
        captured.update(
            package=package_dir, tracking=resolved, name=artifact_name, production=production
        )
        return {"artifact_ref": "team/project/model:v7"}

    monkeypatch.setattr(command, "_tracking_parameters", settings)
    monkeypatch.setattr(command, "publish_model_package", publish)
    arguments = ["publish", "--package", str(tmp_path), "--name", "model", "--env", "ci"]
    if production:
        arguments.append("--production")
    assert command.main(arguments) == 0
    assert captured == {
        "env": "ci",
        "package": tmp_path,
        "tracking": tracking,
        "name": "model",
        "production": production,
    }
    assert json.loads(capsys.readouterr().out)["artifact_ref"] == "team/project/model:v7"


def test_download_command_preserves_the_resolved_artifact_version(
    repo_root, tmp_path, monkeypatch, capsys
):
    command = load_command(repo_root)
    tracking = {"mode": "online", "entity": "team", "project": "project"}
    captured = {}
    monkeypatch.setattr(command, "_tracking_parameters", lambda env: tracking)

    def download(reference, destination, resolved):
        captured.update(reference=reference, destination=destination, tracking=resolved)
        return destination, {"artifact_ref": "team/project/model:v7"}

    monkeypatch.setattr(command, "download_model_package", download)
    assert (
        command.main(
            [
                "download",
                "--artifact",
                "team/project/model:production",
                "--output",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert captured["reference"] == "team/project/model:production"
    assert captured["destination"] == tmp_path
    assert captured["tracking"] == tracking
    assert json.loads(capsys.readouterr().out)["artifact_ref"] == "team/project/model:v7"


def test_publish_command_does_not_access_wandb_when_disabled(repo_root, tmp_path, monkeypatch):
    command = load_command(repo_root)
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setitem(sys.modules, "wandb", None)
    monkeypatch.setattr(
        command,
        "_tracking_parameters",
        lambda env: {"mode": "disabled", "entity": None, "project": None},
    )
    with pytest.raises(ValueError, match="require online tracking"):
        command.main(["publish", "--package", str(tmp_path), "--name", "model"])


def test_tracking_settings_are_loaded_from_the_selected_kedro_checkout(repo_root, monkeypatch):
    command = load_command(repo_root)
    monkeypatch.setenv("WANDB_MODE", "disabled")
    settings = command._tracking_parameters("ci")
    assert set(settings) == {"mode", "entity", "project"}
    assert settings["entity"] and settings["project"]
