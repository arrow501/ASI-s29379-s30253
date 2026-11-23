"""Independent temporal fold boundaries and persisted, test-independent decisions."""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from asi_project.config import ConfigError
from asi_project.config_validation import project_parameters
from asi_project.data import DataContractError, normalize_input
from asi_project.pipelines.experiments.cross_validation import (
    _check_protocols,
    chronological_folds,
    run_cross_validation,
)
from asi_project.pipelines.experiments.nodes import evaluate_frozen_experiments


def test_forward_folds_have_known_boundaries_and_never_train_on_future():
    folds = chronological_folds(80, 0.2, 3)
    assert [(len(train), len(valid)) for train, valid in folds] == [(32, 16), (48, 16), (64, 16)]
    for (train, valid), start in zip(folds, [32, 48, 64], strict=True):
        assert train.tolist() == list(range(start))
        assert valid.tolist() == list(range(start, start + 16))
        assert train.max() < valid.min()
    assert np.concatenate([v for _, v in folds]).tolist() == list(range(32, 80))


@pytest.mark.parametrize(
    "rows,fraction,folds",
    [(10, 0.4, 3), (80, 0, 3), (80, True, 3), (80, 0.2, 1), (80, 0.2, True), (80, 0.2, 2.5)],
)
def test_impossible_or_malformed_folds_fail_clearly(rows, fraction, folds):
    with pytest.raises(ConfigError):
        chronological_folds(rows, fraction, folds)


@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"bagging_folds": True, "chronological_folds": 3},
        {"bagging_folds": 5, "chronological_folds": 0},
    ],
)
def test_protocol_settings_reject_unknown_or_invalid_counts(settings):
    with pytest.raises(ConfigError):
        _check_protocols(settings)


@pytest.mark.parametrize("protocol", ["bagged", "chronological"])
def test_real_baseline_protocol_ignores_test_labels_and_refits_only_development(
    project, tmp_path, protocol
):
    data = normalize_input(project.sample.iloc[:100].copy())
    data["y"] = np.where(np.arange(100) % 2, "yes", "no")
    data.loc[data.index[80:], "job"] = "only_in_test"
    parameters = project_parameters(project.config, project.contract)
    settings = {
        "autogluon_enabled": False,
        "num_cpus": 2,
        "stopping_metric": "log_loss",
        "threshold_mode": "validation_f1",
        "candidates": [{"name": "gbm", "time_limit": 10, "hyperparameters": {"GBM": {}}}],
    }
    decisions = []
    for flipped in (False, True):
        raw = data.copy()
        if flipped:
            raw.loc[raw.index[80:], "y"] = ["yes", "no"] * 10
        report = run_cross_validation(
            raw,
            raw,
            parameters,
            {"max_iter": 1000},
            settings,
            {"bagging_folds": 5, "chronological_folds": 3},
            "sample",
            tmp_path,
            protocol=protocol,
        )
        result = report["results"]["baseline"]
        assert result["final_training_rows"] == 80
        model = pickle.loads(Path(result["model_path"]).read_bytes())
        categorical = model.named_steps["preprocessing"].named_transformers_["categorical"]
        # A known category only in test must not appear in any fitted encoder.
        assert "only_in_test" not in str(categorical.named_steps["encoder"].categories_)
        numeric = model.named_steps["preprocessing"].named_transformers_["numeric"]
        expected = raw.iloc[:80][parameters["numeric_features"]].median().fillna(0).to_numpy(float)
        np.testing.assert_allclose(numeric.named_steps["imputer"].statistics_, expected)
        assert len(result["fold_f1"]) == (3 if protocol == "chronological" else 1)
        assert result["selection_score"] == np.mean(result["fold_f1"])
        oof = json.loads((Path(report["run_dir"]) / "baseline_oof.json").read_text())
        expected_positions = list(range(32, 80)) if protocol == "chronological" else list(range(80))
        assert sorted(oof["indices"]) == sorted(raw.index[expected_positions].tolist())
        tested = evaluate_frozen_experiments(raw, raw, parameters, report)
        assert tested["n_test"] == 20 and tested["test_used_for_selection"] is False
        decisions.append(
            (tested["selected_candidate"], tested["decision_threshold"], result["fold_f1"])
        )
    assert decisions[0] == decisions[1]


def test_fold_with_missing_class_fails_before_creating_outputs(project, tmp_path):
    data = normalize_input(project.sample.iloc[:100].copy())
    data["y"] = ["no"] * 32 + ["yes", "no"] * 34
    settings = {
        "autogluon_enabled": False,
        "num_cpus": 2,
        "stopping_metric": "log_loss",
        "threshold_mode": "fixed",
        "candidates": [{"name": "gbm", "time_limit": 10, "hyperparameters": {"GBM": {}}}],
    }
    with pytest.raises(DataContractError, match="both classes"):
        run_cross_validation(
            data,
            data,
            project_parameters(project.config, project.contract),
            {"max_iter": 1000},
            settings,
            {"bagging_folds": 5, "chronological_folds": 3},
            "sample",
            tmp_path,
            protocol="chronological",
        )
    assert list(tmp_path.iterdir()) == []
