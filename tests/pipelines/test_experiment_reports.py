"""Real reports, immutable decisions, partial failures and native loading in a new process."""

import copy
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from kedro.io import DatasetError

from asi_project.config_validation import project_parameters
from asi_project.data import DataContractError, filter_training_rows, normalize_input
from asi_project.experiment_store import ExperimentStore
from asi_project.pipelines.experiments import artifacts, evaluation
from asi_project.pipelines.experiments.artifacts import read_json, write_json
from asi_project.pipelines.experiments.evaluation import evaluate_test, freeze_selection
from asi_project.pipelines.experiments.nodes import train_holdout_models
from asi_project.pipelines.experiments.splits import split_experiment_data


@pytest.fixture
def completed_run(project, tmp_path):
    data = normalize_input(project.sample.iloc[:100].copy())
    data["y"] = np.where(np.arange(100) % 2, "yes", "no")
    parameters = project_parameters(project.config, project.contract)
    settings = {
        "autogluon_enabled": False,
        "num_cpus": 2,
        "stopping_metric": "log_loss",
        "threshold_mode": "validation_f1",
        "candidates": [{"name": "gbm", "time_limit": 10, "hyperparameters": {"GBM": {}}}],
    }
    parts = split_experiment_data(data, parameters)
    report = train_holdout_models(
        *parts, data, parameters, {"max_iter": 1000}, settings, "sample", tmp_path
    )
    return data, parameters, settings, parts, report


def test_serialization_failure_does_not_truncate_valid_report(tmp_path):
    path = tmp_path / "report.json"
    write_json(path, {"status": "completed"})
    with pytest.raises(ValueError):
        write_json(path, {"score": float("nan")})
    assert read_json(path) == {"status": "completed"}
    assert list(tmp_path.iterdir()) == [path]


def test_failed_replace_keeps_previous_report_and_cleans_temporary_file(tmp_path, monkeypatch):
    path = tmp_path / "report.json"
    write_json(path, {"version": 1})

    def fail(*args):
        raise OSError("disk unavailable")

    monkeypatch.setattr(artifacts.os, "replace", fail)
    with pytest.raises(OSError, match="disk unavailable"):
        write_json(path, {"version": 2})
    assert read_json(path) == {"version": 1}
    assert list(tmp_path.iterdir()) == [path]


def test_metadata_failure_creates_failed_run_not_a_completed_or_abandoned_one(
    project, tmp_path, monkeypatch
):
    def fail():
        raise RuntimeError("git unavailable")

    monkeypatch.setattr(artifacts, "git_provenance", fail)
    with pytest.raises(RuntimeError, match="git unavailable"):
        artifacts.create_run(
            tmp_path,
            project.sample,
            {},
            {},
            {"autogluon_enabled": False},
            "sample",
            protocol="holdout",
            evaluation_set="validation",
        )
    report = read_json(next(tmp_path.glob("*/metrics.json")))
    assert report["status"] == "failed" and report["results"] == {}


def test_frozen_snapshot_contains_every_candidate_and_ignores_later_report_mutation(completed_run):
    data, parameters, _, _, report = completed_run
    frozen = freeze_selection(report)
    expected = copy.deepcopy(frozen["candidates"])
    assert set(expected) == {"baseline", "dummy"}
    report["results"]["baseline"]["decision_threshold"] = 1
    report["results"]["dummy"]["model_path"] = "/does/not/exist"
    tested = evaluate_test(data, data, parameters, frozen)
    for name, result in tested["results"].items():
        assert result["decision_threshold"] == expected[name]["decision_threshold"]
    assert read_json(Path(frozen["run_dir"]) / "selection.json") == frozen


def test_modified_decision_is_rejected_before_loading_any_model(completed_run, monkeypatch):
    data, parameters, _, _, report = completed_run
    frozen = freeze_selection(report)
    frozen["candidates"]["dummy"]["decision_threshold"] = 0
    monkeypatch.setattr(
        evaluation, "load_native_model", lambda *args: pytest.fail("loaded model before validation")
    )
    with pytest.raises(ValueError, match="frozen"):
        evaluate_test(data, data, parameters, frozen)


@pytest.mark.parametrize("change", ["data", "parameters"])
def test_wrong_dataset_or_project_cannot_be_evaluated(completed_run, change):
    data, parameters, _, _, report = completed_run
    frozen = freeze_selection(report)
    if change == "data":
        data = data.assign(balance=data.balance + 1)
    else:
        parameters = {**parameters, "random_state": 24}
    with pytest.raises(ValueError, match="fingerprint"):
        evaluate_test(data, data, parameters, frozen)
    assert not (Path(report["run_dir"]) / "test_metrics.json").exists()


@pytest.mark.parametrize("change", ["target", "features", "order"])
def test_prepared_changes_are_rejected_even_when_raw_fingerprint_matches(
    completed_run, change, monkeypatch
):
    raw, parameters, _, _, report = completed_run
    frozen = freeze_selection(report)
    prepared = raw.copy()
    if change == "target":
        prepared.loc[prepared.index[-20:], "y"] = prepared.iloc[-20:].y.map(
            {"yes": "no", "no": "yes"}
        )
    elif change == "features":
        prepared.loc[prepared.index[-20:], "balance"] += 100
    else:
        prepared = prepared.iloc[::-1]
    monkeypatch.setattr(
        evaluation,
        "load_native_model",
        lambda *args: pytest.fail("loaded a model before validating prepared_data"),
    )
    with pytest.raises(DataContractError, match="prepared_data"):
        evaluate_test(prepared, raw, parameters, frozen)
    root = Path(report["run_dir"])
    assert not (root / "test_metrics.json").exists()
    assert read_json(root / "metrics.json")["test_evaluated"] is False
    assert read_json(root / "selection.json") == frozen


@pytest.mark.parametrize("drop_excluded", [False, True])
def test_valid_normalization_and_filtering_can_be_evaluated(project, tmp_path, drop_excluded):
    raw = project.sample.iloc[:100].copy()
    raw["y"] = np.where(np.arange(100) % 2, "yes", "no")
    raw.loc[raw.index[5], "y"] = None
    raw.loc[raw.index[0], "job"] = "  admin.  "
    raw.loc[raw.index[0], "pdays"] = -1
    prepared = filter_training_rows(normalize_input(raw), project.contract)
    if drop_excluded:
        prepared = prepared.drop(columns="duration")
    parameters = project_parameters(project.config, project.contract)
    settings = {
        "autogluon_enabled": False,
        "num_cpus": 2,
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
    tested = evaluate_test(prepared, raw, parameters, freeze_selection(report))
    assert tested["status"] == "completed" and tested["n_test"] == 20


@pytest.mark.parametrize("value", [-0.1, 1.1, True, float("inf")])
def test_invalid_threshold_cannot_be_frozen(completed_run, value):
    *_, report = completed_run
    report["results"]["baseline"]["decision_threshold"] = value
    path = Path(report["run_dir"]) / "metrics.json"
    # Deliberately malformed source artifact; our writer correctly forbids infinity.
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="threshold"):
        freeze_selection(report)
    assert not path.with_name("selection.json").exists()


def test_missing_candidate_and_failed_suite_cannot_be_selected(completed_run):
    *_, report = completed_run
    report["status"] = "failed"
    with pytest.raises(ValueError, match="completed"):
        freeze_selection(report)
    report["status"] = "completed"
    report["results"].pop("dummy")
    write_json(Path(report["run_dir"]) / "metrics.json", report)
    with pytest.raises(ValueError, match="all configured"):
        freeze_selection(report)


def test_nonfinite_snapshot_does_not_leave_a_partial_decision_file(completed_run):
    *_, report = completed_run
    report["results"]["baseline"]["selection_score"] = 0.5
    report["results"]["baseline"]["validation"]["f1"] = float("inf")
    root = Path(report["run_dir"])
    (root / "metrics.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        freeze_selection(report)
    assert not (root / "selection.json").exists()


def test_test_failure_preserves_already_evaluated_results(completed_run, monkeypatch):
    data, parameters, _, _, report = completed_run
    frozen = freeze_selection(report)
    real_load = evaluation.load_native_model

    def load(path, kind):
        if Path(path).name == "dummy.pkl":
            raise RuntimeError("damaged dummy")
        return real_load(path, kind)

    monkeypatch.setattr(evaluation, "load_native_model", load)
    with pytest.raises(RuntimeError, match="damaged dummy"):
        evaluate_test(data, data, parameters, frozen)
    root = Path(report["run_dir"])
    tested = read_json(root / "test_metrics.json")
    assert tested["status"] == "failed" and set(tested["results"]) == {"baseline"}
    assert read_json(root / "metrics.json")["test_evaluated"] is True
    with pytest.raises(FileExistsError):
        evaluate_test(data, data, parameters, frozen)


def test_native_baseline_loads_in_new_process_without_fit(completed_run, repo_root, tmp_path):
    _, _, _, parts, report = completed_run
    path = report["results"]["baseline"]["model_path"]
    frame_path = tmp_path / "validation.pkl"
    parts[1].to_pickle(frame_path)
    expected = pickle.loads(Path(path).read_bytes()).predict_proba(parts[1])[:, 1]
    code = """
import json, sys
import pandas as pd
from sklearn.pipeline import Pipeline
from asi_project.pipelines.experiments.models import load_native_model
from asi_project.pipelines.experiments.scoring import positive_probabilities
def forbidden(*args, **kwargs):
    raise AssertionError("inference called fit")
Pipeline.fit = forbidden
model = load_native_model(sys.argv[1], "sklearn")
print(json.dumps(positive_probabilities(model, pd.read_pickle(sys.argv[2])).tolist()))
"""
    process = subprocess.run(
        [sys.executable, "-c", code, path, str(frame_path)],
        env={**os.environ, "PYTHONPATH": str(repo_root / "src")},
        capture_output=True,
        text=True,
        check=True,
    )
    np.testing.assert_allclose(json.loads(process.stdout), expected)


def test_catalog_location_is_read_only_and_does_not_create_artifacts(tmp_path):
    path = tmp_path / "not_created"
    store = ExperimentStore(str(path))
    assert store.load() == path and not path.exists()
    with pytest.raises(DatasetError, match="read-only"):
        store.save({})
    assert not path.exists()
