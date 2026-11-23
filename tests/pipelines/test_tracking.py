"""Tracking must preserve provenance and distinguish local runs from online success."""

import hashlib
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pandas as pd
import pytest

from asi_project.config import ConfigError
from asi_project.tracking import data_fingerprint, log_baseline, tracking_settings


@pytest.fixture
def record():
    scores = {
        "accuracy": 0.5,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "confusion_matrix": {"tn": 1, "fp": 1, "fn": 1, "tp": 1},
    }
    return {
        "schema_version": 1,
        "dataset": "bank_marketing",
        "target": "y",
        "positive_class": "yes",
        "data_mode": "sample",
        "split": {"strategy": "time_ordered", "n_train": 16, "n_test": 4},
        "features": ["age"],
        "models": {"dummy": scores, "logistic_regression": scores},
    }


def test_fingerprint_has_known_serialization_and_detects_value_order_and_target():
    raw = pd.DataFrame({"x": [1, 2], "y": ["no", "yes"]})
    expected = hashlib.sha256(b",x,y\n0,1,no\n1,2,yes\n").hexdigest()
    assert data_fingerprint(raw) == expected
    assert data_fingerprint(raw.copy()) == expected
    for changed in (raw.iloc[::-1], raw.assign(y=["yes", "no"]), raw.assign(x=[3, 2])):
        assert data_fingerprint(changed) != expected


def test_disabled_needs_neither_sdk_nor_metadata(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setitem(sys.modules, "wandb", None)
    log_baseline(None, None, None, None, {"mode": "online"})


@pytest.mark.parametrize(
    "settings",
    [
        {"mode": "onlien"},
        {"mode": "online"},
        {"mode": "offline", "project": ""},
        {"mode": "offline", "projet": "typo"},
    ],
)
def test_invalid_or_ambiguous_destination_is_rejected(settings, monkeypatch):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    with pytest.raises(ConfigError):
        tracking_settings(settings)


def test_logged_payload_contains_effective_settings_and_known_results(record, monkeypatch):
    from asi_project import tracking

    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setattr(
        tracking, "git_provenance", lambda: {"git_sha": "abc123", "git_dirty": True}
    )
    monkeypatch.setattr(tracking, "version", lambda name: "test-version")
    captured = {}

    @contextmanager
    def init(**kwargs):
        captured.update(kwargs)
        yield SimpleNamespace(
            url="https://wandb.ai/team/project/runs/test", log=lambda x: captured.update(values=x)
        )

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init, Settings=lambda **kw: kw))
    project = {"positive_class": "yes", "test_size": 0.2, "random_state": 42}
    log_baseline(
        pd.DataFrame({"age": [20]}), record, project, {"max_iter": 1000}, {"mode": "offline"}
    )
    assert captured["config"]["project"] == project
    assert captured["config"]["training"] == {"max_iter": 1000}
    assert captured["config"]["git_sha"] == "abc123"
    assert captured["config"]["git_dirty"] is True
    assert captured["config"]["data_mode"] == "sample"
    assert len(captured["config"]["data_fingerprint"]) == 64
    assert captured["values"]["logistic_regression/f1"] == 0.5
    assert captured["values"]["logistic_regression/confusion_matrix/tp"] == 1
    assert captured["settings"]["disable_code"] is True


def test_online_missing_credentials_fails_without_creating_run(record, monkeypatch):
    from asi_project import tracking

    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setattr(
        tracking, "git_provenance", lambda: {"git_sha": "abc123", "git_dirty": False}
    )
    monkeypatch.setattr(tracking, "version", lambda name: "test-version")

    def login(**kwargs):
        assert kwargs == {"prompt": False, "verify": True, "timeout": 15}
        return False

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(login=login))
    with pytest.raises(ConfigError, match="credentials missing"):
        log_baseline(
            pd.DataFrame({"age": [20]}),
            record,
            {},
            {},
            {"mode": "online", "entity": "team", "project": "project"},
        )


def test_online_failure_is_not_swallowed(record, monkeypatch):
    from asi_project import tracking

    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setattr(
        tracking, "git_provenance", lambda: {"git_sha": "abc123", "git_dirty": False}
    )
    monkeypatch.setattr(tracking, "version", lambda name: "test-version")

    def init(**kwargs):
        raise RuntimeError("permission denied")

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(login=lambda **kw: True, init=init, Settings=lambda **kw: kw),
    )
    with pytest.raises(RuntimeError, match="permission denied"):
        log_baseline(
            pd.DataFrame({"age": [20]}),
            record,
            {},
            {},
            {"mode": "online", "entity": "team", "project": "project"},
        )


def test_tracking_pipeline_is_opt_in():
    from asi_project.pipeline_registry import register_pipelines

    pipelines = register_pipelines()
    assert "log_baseline" not in {node.name for node in pipelines["__default__"].nodes}
    assert "log_baseline" in {node.name for node in pipelines["baseline_tracking"].nodes}
    assert pipelines["baseline_tracking"].inputs() - pipelines["__default__"].inputs() == {
        "params:tracking"
    }
