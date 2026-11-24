"""Track real local reports with controlled SDK IO; disabled needs no SDK or credentials."""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from asi_project import tracking
from asi_project.config_validation import project_parameters
from asi_project.data import normalize_input
from asi_project.pipelines.experiments.artifacts import read_json, write_json
from asi_project.pipelines.experiments.evaluation import evaluate_test, freeze_selection
from asi_project.pipelines.experiments.nodes import train_holdout_models
from asi_project.pipelines.experiments.splits import split_experiment_data
from asi_project.provenance import data_fingerprint

SETTINGS = {"mode": "online", "entity": "team", "project": "project"}


@pytest.fixture(autouse=True)
def isolated_tracking_env(monkeypatch):
    for key in ("WANDB_MODE", "WANDB_ENTITY", "WANDB_PROJECT"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def frozen_run(project, tmp_path):
    raw = project.sample.iloc[:100].copy()
    raw["y"] = np.where(np.arange(100) % 2, "yes", "no")
    prepared = normalize_input(raw)
    parameters = project_parameters(project.config, project.contract)
    settings = {
        "autogluon_enabled": False,
        "num_cpus": 1,
        "stopping_metric": "log_loss",
        "threshold_mode": "validation_f1",
        "candidates": [{"name": "gbm", "time_limit": 10, "hyperparameters": {"GBM": {}}}],
    }
    report = train_holdout_models(
        *split_experiment_data(prepared, parameters),
        raw,
        parameters,
        {"max_iter": 1000},
        settings,
        "sample",
        tmp_path,
    )
    selected = freeze_selection(report)
    tested = evaluate_test(prepared, raw, parameters, selected)
    return Path(report["run_dir"]), raw, selected, tested


@pytest.fixture
def sdk(monkeypatch):
    calls = []

    class Artifact:
        def __init__(self, name, type):
            self.type, self.files = type, {}

        def add_file(self, path, name):
            self.files[name] = read_json(path)

    class Run:
        def __init__(self, kwargs):
            self.kwargs, self.summary, self.logs, self.artifacts = kwargs, {}, [], []
            self.url = "https://wandb.example/team/project/runs/" + kwargs["id"]
            self.exit_error = None

        def __enter__(self):
            return self

        def __exit__(self, cls, error, tb):
            self.exit_error = cls

        def log(self, payload):
            self.logs.append(payload)

        def log_artifact(self, artifact):
            self.artifacts.append(artifact)

    def init(**kwargs):
        run = Run(kwargs)
        calls.append(run)
        return run

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(init=init, Artifact=Artifact, Settings=lambda **kwargs: kwargs),
    )
    return calls


def test_disabled_ignores_sdk_and_missing_credentials(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    monkeypatch.setenv("WANDB_MODE", "disabled")
    assert tracking.track_experiment({}, SETTINGS) == {"status": "disabled", "runs": {}}
    assert tracking.track_baseline(None, None, None, None, None, SETTINGS)["status"] == "disabled"


@pytest.mark.parametrize(
    "override",
    [{"mode": "typo"}, {"project": None}, {"entity": "team/project"}, {"api_key": "forbidden"}],
)
def test_invalid_target_or_secret_config_is_rejected(override):
    with pytest.raises(ValueError):
        tracking.tracking_settings({**SETTINGS, **override})


def test_environment_target_overrides_config(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_ENTITY", "another-team")
    monkeypatch.setenv("WANDB_PROJECT", "another-project")
    assert tracking.tracking_settings(SETTINGS) == {
        "mode": "offline",
        "entity": "another-team",
        "project": "another-project",
    }


def test_nested_confusion_counts_and_nonfinite_metrics():
    assert tracking.flatten_metrics(
        {
            "test": {"f1": 2 / 3, "confusion_matrix": {"tp": 3}},
            "model_path": "/tmp/model",
            "is_selected": True,
        }
    ) == {"test/f1": 2 / 3, "test/confusion_matrix/tp": 3}
    with pytest.raises(ValueError, match="Non-finite"):
        tracking.flatten_metrics({"validation": {"f1": float("nan")}})


def test_actual_payload_contains_frozen_metrics_config_and_data_provenance(
    frozen_run, sdk, monkeypatch
):
    root, raw, selected, tested = frozen_run
    monkeypatch.setenv("WANDB_API_KEY", "sentinel-never-upload-this")
    receipt = tracking.track_experiment(selected, SETTINGS, tested)
    assert receipt["status"] == "completed" and set(receipt["runs"]) == {"baseline", "dummy"}
    assert len(sdk) == 2
    assert read_json(root / "wandb.json") == receipt
    for run in sdk:
        config = run.kwargs["config"]
        name = config["candidate_name"]
        assert config["provenance"]["data_fingerprint"] == data_fingerprint(raw)
        assert len(config["provenance"]["git_sha"]) == 40
        assert config["project"]["positive_class"] == "yes"
        assert config["project"]["test_size"] == 0.2
        assert config["training"]["max_iter"] == 1000
        assert run.logs[0]["test/f1"] == tested["results"][name]["test"]["f1"]
        assert (
            run.logs[0]["test/confusion_matrix/tp"]
            == tested["results"][name]["test"]["confusion_matrix"]["tp"]
        )
        assert (
            run.logs[0]["decision_threshold"] == selected["candidates"][name]["decision_threshold"]
        )
        assert run.kwargs["resume"] == "allow" and run.exit_error is None
        artifact = run.artifacts[0]
        assert artifact.type == "experiment-report"
        assert set(artifact.files) == {
            "config.json",
            "provenance.json",
            "metrics.json",
            "selection.json",
            "split.json",
            "test_metrics.json",
        }
        assert "sentinel-never-upload-this" not in json.dumps([config, run.logs, artifact.files])
    first_ids = [run.kwargs["id"] for run in sdk]
    tracking.track_experiment(selected, SETTINGS, tested)
    assert [run.kwargs["id"] for run in sdk[2:]] == first_ids


def test_validation_only_does_not_publish_existing_test(frozen_run, sdk):
    _, _, selected, _ = frozen_run
    tracking.track_experiment(selected, SETTINGS)
    assert all(not any(k.startswith("test/") for k in run.logs[0]) for run in sdk)
    assert all("test_metrics.json" not in run.artifacts[0].files for run in sdk)


@pytest.mark.parametrize(
    "change", ["selection", "threshold", "missing_model", "test", "fingerprint", "nan"]
)
def test_invalid_source_is_rejected_before_remote_init(frozen_run, sdk, change):
    root, _, selected, tested = frozen_run
    if change == "selection":
        selected["selected_candidate"] = "dummy"
    elif change == "test":
        tested["results"]["baseline"]["decision_threshold"] = 1
    elif change == "fingerprint":
        path = root / "provenance.json"
        provenance = read_json(path)
        provenance["data_fingerprint"] = "changed"
        write_json(path, provenance)
    else:
        path = root / "metrics.json"
        report = read_json(path)
        if change == "threshold":
            report["results"]["baseline"]["decision_threshold"] = 1
        elif change == "missing_model":
            report["results"].pop("dummy")
        else:
            report["results"]["dummy"]["validation"]["f1"] = float("nan")
        path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        tracking.track_experiment(selected, SETTINGS, tested)
    assert sdk == []
    assert not (root / "wandb.json").exists()


def test_sdk_failure_preserves_local_results_and_marks_receipt(frozen_run, sdk, monkeypatch):
    root, _, selected, tested = frozen_run
    expected = (root / "test_metrics.json").read_bytes()
    fake_sdk = sys.modules["wandb"]
    actual_init = fake_sdk.init

    def failing_init(**kwargs):
        run = actual_init(**kwargs)

        def fail(values):
            raise RuntimeError("do-not-store-a-secret-from-sdk-error")

        run.log = fail
        return run

    monkeypatch.setattr(fake_sdk, "init", failing_init)
    with pytest.raises(RuntimeError):
        tracking.track_experiment(selected, SETTINGS, tested)
    assert (root / "test_metrics.json").read_bytes() == expected
    receipt = read_json(root / "wandb.json")
    assert receipt["status"] == "failed" and receipt["error_type"] == "RuntimeError"
    assert "do-not-store" not in json.dumps(receipt)
    assert sdk[0].exit_error is RuntimeError


def test_s2_reference_preserves_exact_metrics_and_threshold(project, tmp_path, sdk):
    from asi_project.pipelines.data_science.nodes import evaluate_models, split_data, train_models

    raw = project.sample
    params = project_parameters(project.config, project.contract)
    X_train, X_test, y_train, y_test = split_data(normalize_input(raw), params)
    model, dummy = train_models(X_train, y_train, params, {"max_iter": 1000})
    metrics = evaluate_models(model, dummy, X_test, y_test, y_train, params, "sample")
    expected = copy.deepcopy(metrics)
    receipt = tracking.track_baseline(metrics, raw, params, {"max_iter": 1000}, tmp_path, SETTINGS)
    assert metrics == expected
    assert len(sdk) == 1 and set(receipt["runs"]) == {"s2_baseline"}
    run = sdk[0]
    assert run.kwargs["config"]["decision_threshold"] == 0.5
    assert run.kwargs["config"]["validation_used"] is False
    assert (
        run.logs[0]["test/logistic_regression/f1"] == metrics["models"]["logistic_regression"]["f1"]
    )
    assert run.artifacts[0].files["s2_metrics.json"] == expected


def test_real_offline_sdk_persists_binary_run_without_network(frozen_run, monkeypatch):
    pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_MODE", "offline")
    root, _, selected, tested = frozen_run
    receipt = tracking.track_experiment(selected, SETTINGS, tested)
    assert receipt["status"] == "completed"
    assert all(run["url"] is None for run in receipt["runs"].values())
    assert len(list(root.glob("wandb/offline-run-*/run-*.wandb"))) == 2


def test_missing_sdk_records_tracking_failure_without_changing_training(frozen_run, monkeypatch):
    root, _, selected, tested = frozen_run
    monkeypatch.setitem(sys.modules, "wandb", None)
    with pytest.raises(ModuleNotFoundError):
        tracking.track_experiment(selected, SETTINGS, tested)
    assert read_json(root / "wandb.json")["status"] == "failed"
    assert read_json(root / "metrics.json")["status"] == "completed"


def test_tracking_config_is_checked_before_training(project):
    from asi_project.config_validation import validate_run_parameters

    with pytest.raises(ValueError, match="tracking"):
        validate_run_parameters(
            {
                "project": project_parameters(project.config, project.contract),
                "training": {"max_iter": 1000},
                "reporting": {"data_mode": "sample"},
                "tracking": {**SETTINGS, "api_key": "do-not-put-keys-in-params"},
            },
            project.contract,
        )
