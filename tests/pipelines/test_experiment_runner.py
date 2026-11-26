"""Run the real preparation/training/selection/test graph with native local artifacts."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from kedro.io import DataCatalog, MemoryDataset
from kedro.runner import SequentialRunner

from asi_project.config_validation import project_parameters
from asi_project.data import normalize_input
from asi_project.model_package import (
    load_model_package,
    load_native_model,
    predict_model_package,
)
from asi_project.pipeline_registry import register_pipelines
from asi_project.pipelines.experiments import nodes
from asi_project.pipelines.experiments.artifacts import read_json
from asi_project.pipelines.experiments.evaluation import freeze_selection
from asi_project.pipelines.experiments.scoring import positive_probabilities
from asi_project.pipelines.experiments.splits import split_experiment_data


@pytest.mark.parametrize("name", ["experiments", "experiments_bagged", "experiments_crossfold"])
@pytest.mark.parametrize("include_test", [False, True])
def test_real_kedro_graph_persists_decision_and_optional_test_without_sdks(
    project, tmp_path, monkeypatch, name, include_test
):
    monkeypatch.setitem(sys.modules, "wandb", None)
    monkeypatch.setitem(sys.modules, "autogluon.tabular", None)
    raw = project.sample.iloc[:100].copy()
    raw["y"] = np.where(np.arange(100) % 2, "yes", "no")
    catalog = DataCatalog.from_config(
        {
            "model_package_store": {
                "type": "asi_project.experiment_store.ExperimentStore",
                "filepath": str(tmp_path / "packages"),
            },
            "experiment_store": {
                "type": "asi_project.experiment_store.ExperimentStore",
                "filepath": str(tmp_path / "reports"),
            },
        }
    )
    for key, value in {
        "raw_data": raw,
        "params:project": project_parameters(project.config, project.contract),
        "params:training": {"max_iter": 1000},
        "params:experiments": {
            "autogluon_enabled": False,
            "num_cpus": 2,
            "stopping_metric": "log_loss",
            "threshold_mode": "validation_f1",
            "candidates": [{"name": "gbm", "time_limit": 10, "hyperparameters": {"GBM": {}}}],
        },
        "params:validation_protocols": {"bagging_folds": 5, "chronological_folds": 3},
        "params:reporting.data_mode": "sample",
        "params:tracking": {"mode": "disabled", "entity": None, "project": None},
    }.items():
        catalog[key] = MemoryDataset(data=value)
    pipeline = register_pipelines()[name + ("_test" if include_test else "")]
    outputs = SequentialRunner().run(pipeline, catalog)
    root = next((tmp_path / "reports").iterdir())
    report = json.loads((root / "metrics.json").read_text())
    selection = json.loads((root / "selection.json").read_text())
    assert report["status"] == "completed" and report["test_evaluated"] is include_test
    assert selection["test_used_for_selection"] is False
    assert selection["selected_candidate"] == "baseline"
    expected = {"baseline", "dummy"} if name == "experiments" else {"baseline"}
    assert set(selection["candidates"]) == expected
    assert all(
        Path(candidate["model_path"]).exists() for candidate in selection["candidates"].values()
    )
    packages = outputs["experiment_packages"].load()
    assert set(packages) == expected
    assert json.loads((root / "packages.json").read_text()) == packages
    parameters = project_parameters(project.config, project.contract)
    features = parameters["numeric_features"] + parameters["categorical_features"]
    prediction_input = normalize_input(raw).loc[:, features].iloc[-5:]
    for candidate, package_dir in packages.items():
        model, metadata = load_model_package(package_dir)
        frozen = selection["candidates"][candidate]
        native = load_native_model(frozen["model_path"], frozen["model_kind"])
        predictions = predict_model_package(model, metadata, raw.iloc[-5:])
        probability = positive_probabilities(native, prediction_input)
        np.testing.assert_allclose(predictions["probability_positive"], probability)
        assert predictions["prediction"].tolist() == [
            metadata["positive_class"]
            if value >= frozen["decision_threshold"]
            else metadata["negative_class"]
            for value in probability
        ]
        assert predictions.index.equals(raw.iloc[-5:].index)
        assert metadata["candidate_name"] == candidate
        assert metadata["source_run_id"] == root.name
        assert metadata["decision_threshold"] == frozen["decision_threshold"]
        assert metadata["metrics"]["validation"] == report["results"][candidate]["validation"]
        assert ("test" in metadata["metrics"]) is include_test
        assert Path(package_dir).is_relative_to(tmp_path / "packages")
    if include_test:
        tested = json.loads((root / "test_metrics.json").read_text())
        assert tested["status"] == "completed" and tested["n_test"] == 20
        assert set(tested["results"]) == expected
        for candidate in expected:
            assert (
                tested["results"][candidate]["decision_threshold"]
                == selection["candidates"][candidate]["decision_threshold"]
            )
        assert "experiment_tracking" in outputs
    else:
        assert not (root / "test_metrics.json").exists()
        assert "experiment_tracking" in outputs


@pytest.fixture
def frozen_packaging_run(project, tmp_path):
    """Przygotowuje dwa rzeczywiste modele do sprawdzenia ponawiania eksportu."""
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
    report = nodes.train_holdout_models(
        *split_experiment_data(normalize_input(raw), parameters),
        raw,
        parameters,
        {"max_iter": 1000},
        settings,
        "sample",
        tmp_path / "reports",
    )
    return freeze_selection(report), raw


@pytest.mark.parametrize("failure", ["second_model", "manifest"])
def test_packaging_can_be_retried_after_a_partial_failure(
    frozen_packaging_run, tmp_path, monkeypatch, failure
):
    selection, raw = frozen_packaging_run
    root = Path(selection["run_dir"])
    store = tmp_path / "packages"
    source_bytes = {
        name: Path(candidate["model_path"]).read_bytes()
        for name, candidate in selection["candidates"].items()
    }
    original_save = nodes.save_model_package
    original_write = nodes.write_json
    saved = []
    failed = False

    def save(model, metadata, destination):
        nonlocal failed
        name = metadata["candidate_name"]
        if failure == "second_model" and name == "dummy" and not failed:
            failed = True
            raise OSError("simulated packaging failure")
        result = original_save(model, metadata, destination)
        saved.append(name)
        return result

    def write(path, payload):
        nonlocal failed
        if failure == "manifest" and Path(path) == root / "packages.json" and not failed:
            failed = True
            assert (store / root.name / "baseline").is_dir()
            raise OSError("simulated packaging failure")
        return original_write(path, payload)

    monkeypatch.setattr(nodes, "save_model_package", save)
    monkeypatch.setattr(nodes, "write_json", write)
    with pytest.raises(OSError, match="simulated packaging failure"):
        nodes.package_experiment_models(selection, raw, store)

    assert "baseline" in saved
    assert not (store / root.name).exists()
    assert not list(store.glob(".model-packages-*"))
    assert not (root / "packages.json").exists()
    returned, packages = nodes.package_experiment_models(selection, raw, store)

    assert returned == selection
    assert set(packages) == {"baseline", "dummy"}
    assert read_json(root / "packages.json") == packages
    assert not list(store.glob(".model-packages-*"))
    for name, package in packages.items():
        model, metadata = load_model_package(package)
        assert Path(package) == store / root.name / name
        assert metadata["candidate_name"] == name
        assert len(predict_model_package(model, metadata, raw.iloc[:5])) == 5
        assert Path(selection["candidates"][name]["model_path"]).read_bytes() == source_bytes[name]


def test_packaging_does_not_overwrite_a_completed_run(frozen_packaging_run, tmp_path):
    selection, raw = frozen_packaging_run
    store = tmp_path / "packages"
    nodes.package_experiment_models(selection, raw, store)
    before = {
        path.relative_to(store): path.read_bytes() for path in store.rglob("*") if path.is_file()
    }
    manifest = Path(selection["run_dir"]) / "packages.json"
    manifest_before = manifest.read_bytes()

    with pytest.raises(FileExistsError):
        nodes.package_experiment_models(selection, raw, store)

    after = {
        path.relative_to(store): path.read_bytes() for path in store.rglob("*") if path.is_file()
    }
    assert after == before
    assert manifest.read_bytes() == manifest_before
