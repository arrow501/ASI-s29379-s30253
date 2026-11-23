"""SDK call contracts independently of optional AutoGluon availability in core CI."""

import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from asi_project.config_validation import project_parameters
from asi_project.data import DataContractError, normalize_input
from asi_project.pipelines.experiments import artifacts, cross_validation, models
from asi_project.pipelines.experiments.splits import prepare_cross_validation


@pytest.mark.parametrize("bad_oof", [False, True])
def test_bagging_receives_development_only_and_checks_oof_alignment(
    project, tmp_path, monkeypatch, bad_oof
):
    captured = {}
    data = normalize_input(project.sample.iloc[:100].copy())
    parameters = project_parameters(project.config, project.contract)
    X = data[parameters["numeric_features"] + parameters["categorical_features"]].iloc[:80]
    y = pd.Series([0, 1] * 40, index=X.index, name="y")

    class Generator:
        def __init__(self, **kwargs):
            captured["generator"] = kwargs

    class Predictor:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def fit(self, train_data, **kwargs):
            captured["data"] = train_data
            captured["fit"] = kwargs

        def predict_proba_oof(self):
            index = X.index[::-1] if bad_oof else X.index
            return pd.DataFrame({0: 0.6, 1: 0.4}, index=index)

    feature_module = ModuleType("autogluon.features.generators")
    feature_module.AutoMLPipelineFeatureGenerator = Generator
    tabular_module = ModuleType("autogluon.tabular")
    tabular_module.TabularPredictor = Predictor
    monkeypatch.setitem(sys.modules, "autogluon.features.generators", feature_module)
    monkeypatch.setitem(sys.modules, "autogluon.tabular", tabular_module)
    settings = {"num_cpus": 3, "stopping_metric": "log_loss"}
    candidate = {"hyperparameters": {"GBM": {}}, "time_limit": 17}
    if bad_oof:
        with pytest.raises(DataContractError, match="OOF"):
            models.fit_bagged_autogluon(X, y, parameters, candidate, tmp_path, settings, 5)
        return
    _, probability = models.fit_bagged_autogluon(X, y, parameters, candidate, tmp_path, settings, 5)
    assert captured["data"].index.tolist() == list(range(80))
    assert captured["data"]["y"].tolist() == [0, 1] * 40
    assert captured["data"]["pdays"].dtype == np.dtype("float64")
    assert "tuning_data" not in captured["fit"]
    assert captured["fit"]["num_bag_folds"] == 5 and captured["fit"]["num_bag_sets"] == 1
    assert captured["fit"]["num_stack_levels"] == 0
    assert captured["fit"]["time_limit"] == 17 and captured["fit"]["num_cpus"] == 3
    np.testing.assert_array_equal(probability, np.full(80, 0.4))


def test_chronological_adapter_uses_exact_prefixes_then_refits_only_last_fold(
    project, tmp_path, monkeypatch
):
    data = normalize_input(project.sample.iloc[:100].copy())
    data["y"] = np.where(np.arange(100) % 2, "yes", "no")
    parameters = project_parameters(project.config, project.contract)
    protocols = {"bagging_folds": 5, "chronological_folds": 3}
    X, y, plan = prepare_cross_validation(data, parameters, protocols, protocol="chronological")
    calls, refits = [], []

    class Predictor:
        def __init__(self, path):
            self.path = str(path)
            Path(path).mkdir(parents=True)
            self.model_best = "GBM"

        def predict_proba(self, frame):
            return pd.DataFrame({0: 0.7, 1: 0.3}, index=frame.index)

        def refit_full(self, **kwargs):
            refits.append((self.path, kwargs))
            self.model_best = "GBM_FULL"

        def model_names(self):
            return [self.model_best]

    def fit(
        train_X,
        validation_X,
        train_y,
        validation_y,
        project,
        candidate,
        path,
        cpus,
        stopping_metric,
    ):
        calls.append((train_X.index.tolist(), validation_X.index.tolist()))
        return Predictor(path)

    monkeypatch.setattr(cross_validation, "fit_holdout_autogluon", fit)
    monkeypatch.setattr(cross_validation, "verify_reload", lambda model, *args: model)
    monkeypatch.setattr(artifacts, "version", lambda name: "fixture-version")
    settings = {
        "autogluon_enabled": True,
        "num_cpus": 2,
        "stopping_metric": "log_loss",
        "threshold_mode": "validation_f1",
        "candidates": [{"name": "gbm", "time_limit": 10, "hyperparameters": {"GBM": {}}}],
    }
    report = cross_validation.train_cross_validated_models(
        X, y, plan, data, parameters, {"max_iter": 1000}, settings, protocols, "sample", tmp_path
    )
    assert calls == [
        (list(range(32)), list(range(32, 48))),
        (list(range(48)), list(range(48, 64))),
        (list(range(64)), list(range(64, 80))),
    ]
    assert len(refits) == 1 and Path(refits[0][0]).name == "fold_3"
    assert refits[0][1] == {
        "model": "best",
        "set_best_to_refit_full": True,
        "num_cpus": 2,
        "num_gpus": 0,
        "fit_strategy": "sequential",
    }
    assert report["results"]["gbm"]["final_training_rows"] == 80
    assert report["results"]["gbm"]["best_model"] == "GBM_FULL"
