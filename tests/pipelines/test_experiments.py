"""Independent split boundaries and real local persistence for Sprint 3."""

import copy
import json
import pickle
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from asi_project.config import ConfigError
from asi_project.config_validation import project_parameters
from asi_project.data import DataContractError, filter_training_rows, normalize_input
from asi_project.pipelines.experiments.evaluation import evaluate_test, freeze_selection
from asi_project.pipelines.experiments.nodes import train_holdout_models
from asi_project.pipelines.experiments.scoring import validation_result, validation_threshold
from asi_project.pipelines.experiments.settings import validate_experiment_settings
from asi_project.pipelines.experiments.splits import split_experiment_data


@pytest.fixture
def parameters(project):
    return project_parameters(project.config, project.contract)


@pytest.fixture
def frame(project):
    data = project.sample.iloc[:101].copy()
    # Known labels ensure both classes around every boundary, independent of the sample.
    data["y"] = np.where(np.arange(len(data)) % 2, "yes", "no")
    data.index = pd.Index(np.arange(len(data))[::-1] * 7)
    return filter_training_rows(normalize_input(data), project.contract)


@pytest.fixture
def settings():
    return {
        "autogluon_enabled": False,
        "num_cpus": 2,
        "stopping_metric": "log_loss",
        "threshold_mode": "validation_f1",
        "candidates": [{"name": "gbm", "time_limit": 10, "hyperparameters": {"GBM": {}}}],
    }


@pytest.mark.parametrize("n,counts", [(100, (64, 16, 20)), (101, (64, 16, 21))])
def test_split_has_independent_expected_boundaries_and_preserves_indices(
    frame, parameters, n, counts
):
    data = frame.iloc[:n]
    original = data.copy(deep=True)
    X_train, X_val, y_train, y_val, summary = split_experiment_data(data, parameters)
    assert (summary["n_train"], summary["n_validation"], summary["n_test"]) == counts
    pd.testing.assert_index_equal(X_train.index, data.index[:64])
    pd.testing.assert_index_equal(X_val.index, data.index[64:80])
    pd.testing.assert_index_equal(X_train.index, y_train.index)
    pd.testing.assert_index_equal(X_val.index, y_val.index)
    assert summary["positions"]["test"] == [80, n]
    assert "duration" not in X_train and "y" not in X_train
    assert y_val.tolist() == [0, 1] * 8
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize("fraction", [0, 1, -0.2, True, "0.2", float("nan")])
def test_invalid_validation_fraction_is_rejected(frame, parameters, fraction):
    with pytest.raises(ConfigError, match="validation_size"):
        split_experiment_data(frame, {**parameters, "validation_size": fraction})


def test_empty_or_single_class_validation_is_rejected(frame, parameters):
    data = frame.copy()
    data.iloc[64:80, data.columns.get_loc("y")] = "no"
    with pytest.raises(DataContractError, match="validation"):
        split_experiment_data(data, parameters)
    with pytest.raises(DataContractError, match="train"):
        split_experiment_data(frame, {**parameters, "validation_size": 0.999})


def test_local_baseline_is_persisted_reloadable_and_runs_without_optional_sdks(
    frame, parameters, settings, tmp_path, monkeypatch
):
    monkeypatch.setitem(sys.modules, "wandb", None)
    monkeypatch.setitem(sys.modules, "autogluon.tabular", None)
    split = split_experiment_data(frame, parameters)
    reports = [
        train_holdout_models(
            *split, frame, parameters, {"max_iter": 1000}, settings, "sample", tmp_path
        )
        for _ in range(2)
    ]
    assert reports[0]["run_dir"] != reports[1]["run_dir"]
    for report in reports:
        from pathlib import Path

        root = Path(report["run_dir"])
        saved = json.loads((root / "metrics.json").read_text())
        assert saved == report
        assert saved["status"] == "completed" and saved["test_evaluated"] is False
        assert set(saved["results"]) == {"baseline", "dummy"}
        assert json.loads((root / "config.json").read_text())["project"] == parameters
        assert len(json.loads((root / "provenance.json").read_text())["data_fingerprint"]) == 64
        model = pickle.loads((root / "baseline.pkl").read_bytes())
        numeric = model.named_steps["preprocessing"].named_transformers_["numeric"]
        expected = np.array(
            [
                split[0][name].dropna().median() if split[0][name].notna().any() else 0
                for name in parameters["numeric_features"]
            ],
            dtype=float,
        )
        np.testing.assert_allclose(numeric.named_steps["imputer"].statistics_, expected)
        assert len(model.predict(split[1])) == 16
        assert saved["results"]["dummy"]["validation"]["f1"] == 0


def test_overlapping_data_are_rejected_before_writing(frame, parameters, settings, tmp_path):
    X_train, _, y_train, _, summary = split_experiment_data(frame, parameters)
    with pytest.raises(DataContractError, match="disjoint"):
        train_holdout_models(
            X_train,
            X_train,
            y_train,
            y_train,
            summary,
            frame,
            parameters,
            {"max_iter": 1000},
            settings,
            "sample",
            tmp_path,
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "change",
    [
        {"num_cpus": True},
        {"num_cpus": 0},
        {"autogluon_enabled": "false"},
        {"candidates": []},
        {"num_cpu": 2},
    ],
)
def test_bad_settings_are_rejected(settings, change):
    with pytest.raises(ConfigError):
        validate_experiment_settings({**settings, **change})


@pytest.mark.parametrize(
    "change",
    [
        {"name": "../escape"},
        {"name": "baseline"},
        {"time_limit": float("inf")},
        {"time_limit": 0},
        {"hyperparameters": {"NN_TORCH": {}}},
    ],
)
def test_bad_candidates_are_rejected(settings, change):
    candidate = {**settings["candidates"][0], **change}
    with pytest.raises(ConfigError):
        validate_experiment_settings({**settings, "candidates": [candidate]})


def test_failure_preserves_partial_results_and_is_not_a_success(
    frame, parameters, settings, tmp_path, monkeypatch
):
    from asi_project.pipelines.experiments import nodes

    def fail(*args, **kwargs):
        raise RuntimeError("deliberate failure")

    monkeypatch.setattr(nodes, "fit_holdout_autogluon", fail)
    enabled = {**copy.deepcopy(settings), "autogluon_enabled": True}
    # No SDK required to test the failure contract.
    monkeypatch.setitem(
        sys.modules, "autogluon.tabular", type("Fake", (), {"TabularPredictor": object})
    )
    from asi_project.pipelines.experiments import artifacts

    monkeypatch.setattr(artifacts, "version", lambda name: "test-version")
    with pytest.raises(RuntimeError, match="deliberate failure"):
        train_holdout_models(
            *split_experiment_data(frame, parameters),
            frame,
            parameters,
            {"max_iter": 1000},
            enabled,
            "sample",
            tmp_path,
        )
    saved = json.loads(next(tmp_path.glob("*/metrics.json")).read_text())
    assert saved["status"] == "failed"
    assert saved["results"]["baseline"]["validation"]
    assert "deliberate failure" in saved["error"]


def test_autogluon_prefits_features_only_on_train_and_gets_explicit_validation(
    frame, parameters, settings, tmp_path, monkeypatch
):
    from asi_project.pipelines.experiments.models import fit_holdout_autogluon

    captured = {}

    class Generator:
        def __init__(self, **kwargs):
            pass

        def fit_transform(self, X, y):
            captured["feature_fit_index"] = X.index.tolist()
            captured["feature_fit_target"] = y.tolist()

    class Predictor:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def fit(self, **kwargs):
            captured["fit"] = kwargs

    features = ModuleType("autogluon.features.generators")
    features.AutoMLPipelineFeatureGenerator = Generator
    tabular = ModuleType("autogluon.tabular")
    tabular.TabularPredictor = Predictor
    monkeypatch.setitem(sys.modules, "autogluon.features.generators", features)
    monkeypatch.setitem(sys.modules, "autogluon.tabular", tabular)
    X, X_val, y, y_val, _ = split_experiment_data(frame, parameters)
    fit_holdout_autogluon(X, X_val, y, y_val, parameters, settings["candidates"][0], tmp_path, 2)
    assert captured["feature_fit_index"] == frame.index[:64].tolist()
    assert captured["feature_fit_target"] == [0, 1] * 32
    pd.testing.assert_frame_equal(
        captured["fit"]["train_data"].drop(columns="y"), X, check_dtype=False
    )
    pd.testing.assert_frame_equal(
        captured["fit"]["tuning_data"].drop(columns="y"), X_val, check_dtype=False
    )
    assert captured["fit"]["tuning_data"].index.tolist() == frame.index[64:80].tolist()
    assert captured["init"]["positive_class"] == 1
    assert captured["init"]["eval_metric"] == "log_loss"
    assert captured["init"]["learner_kwargs"]["random_state"] == 42
    for name in ("num_bag_folds", "num_stack_levels", "num_gpus"):
        assert captured["fit"][name] == 0
    for name in ("refit_full", "dynamic_stacking", "calibrate_decision_threshold"):
        assert captured["fit"][name] is False
    assert captured["fit"]["time_limit"] == 10
    assert captured["fit"]["num_cpus"] == 2
    assert captured["fit"]["raise_on_model_failure"] is True
    assert captured["fit"]["ag_args_fit"] == {"stopping_metric": "log_loss"}
    assert captured["fit"]["train_data"]["pdays"].dtype == np.dtype("float64")


def test_validation_threshold_recovers_known_signal_below_half():
    probability = np.array([0.01, 0.12, 0.08, 0.2])
    target = pd.Series([0, 1, 0, 1])
    model = SimpleNamespace(
        classes_=np.array([0, 1]),
        predict_proba=lambda X: np.column_stack([1 - probability, probability]),
    )
    result = validation_result(model, target, target, "validation_f1")
    assert result["decision_threshold"] == 0.12
    assert result["validation"]["f1"] == 1
    assert result["validation_at_0_5"]["f1"] == 0
    assert result["validation_diagnostics"]["roc_auc"] == 1
    assert result["validation_diagnostics"]["predicted_positive_count"] == 2


def test_threshold_matches_independent_exhaustive_search_including_ties():
    target = np.array([0, 1, 0, 1, 1, 0])
    probability = np.array([0.1, 0.1, 0.2, 0.3, 0.3, 0.4])

    def f1(threshold):
        prediction = probability >= threshold
        tp = int((prediction & (target == 1)).sum())
        fp = int((prediction & (target == 0)).sum())
        fn = int((~prediction & (target == 1)).sum())
        return 2 * tp / (2 * tp + fp + fn)

    expected = max(set(probability), key=lambda t: (f1(t), t))
    assert validation_threshold(target, probability) == expected


def test_test_stage_freezes_validation_selection_and_does_not_recalibrate(
    frame, parameters, settings, tmp_path
):
    selections = []
    for changed_test in (False, True):
        data = frame.copy()
        if changed_test:
            data.iloc[80:, data.columns.get_loc("y")] = ["yes", "no"] * 10 + ["yes"]
        report = train_holdout_models(
            *split_experiment_data(data, parameters),
            data,
            parameters,
            {"max_iter": 1000},
            settings,
            "sample",
            tmp_path,
        )
        result = evaluate_test(data, data, parameters, freeze_selection(report))
        assert result["status"] == "completed"
        assert result["selected_candidate"] == "baseline"
        assert result["test_used_for_selection"] is False
        assert result["decision_threshold"] == report["results"]["baseline"]["decision_threshold"]
        assert result["n_test"] == 21
        matrix = result["results"]["dummy"]["test"]["confusion_matrix"]
        assert matrix == {
            "tn": 10 if changed_test else 11,
            "fp": 0,
            "fn": 11 if changed_test else 10,
            "tp": 0,
        }
        selections.append((result["selected_candidate"], result["decision_threshold"]))
        with pytest.raises(FileExistsError):
            evaluate_test(data, data, parameters, freeze_selection(report))
    assert selections[0] == selections[1]
