"""Zapis i odczyt pakietów, odrzucanie uszkodzeń oraz zamrożone metadane eksperymentów."""

import copy
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from asi_project import model_package
from asi_project.config_validation import project_parameters
from asi_project.data import normalize_input
from asi_project.pipelines.experiments.artifacts import read_json, write_json
from asi_project.pipelines.experiments.evaluation import evaluate_test, freeze_selection
from asi_project.pipelines.experiments.nodes import train_holdout_models
from asi_project.pipelines.experiments.splits import split_experiment_data


def refuse_native_load(*args, **kwargs):
    pytest.fail("An invalid package reached native model loading")


def write_metadata(package_dir, metadata):
    # Celowo pomiń właściwy zapis, aby przygotować uszkodzone pliki wejściowe.
    (package_dir / "metadata.json").write_text(
        json.dumps(metadata, allow_nan=True), encoding="utf-8"
    )


def test_roundtrip_and_relocation_preserve_predictions_and_metadata(fitted_package, tmp_path):
    model, metadata, raw, destination = fitted_package
    original = copy.deepcopy(metadata)
    probability = model.predict_proba(normalize_input(raw))[:, 1]
    saved = model_package.save_model_package(model, metadata, destination)
    assert metadata == original
    assert "model_files" not in metadata
    assert (saved / "model/pipeline.pkl").is_file()
    stored = read_json(saved / "metadata.json")
    assert stored == {**original, "model_files": ["pipeline.pkl"]}
    assert all(not Path(name).is_absolute() for name in stored["model_files"])
    moved = tmp_path / "relocated"
    assert saved.resolve().is_relative_to(tmp_path.resolve())
    assert moved.resolve().is_relative_to(tmp_path.resolve())
    saved.rename(moved)
    assert not saved.exists()
    restored, restored_metadata = model_package.load_model_package(moved)
    result = model_package.predict_model_package(restored, restored_metadata, raw)
    np.testing.assert_allclose(result.probability_positive, probability)
    assert result.prediction.tolist() == np.where(probability >= 0.6, "yes", "no").tolist()
    pd.testing.assert_index_equal(result.index, raw.index)
    assert restored_metadata == stored


@pytest.mark.parametrize(
    "content,message",
    [("", "Cannot read valid"), ("{", "Cannot read valid"), ("[]", "must contain an object")],
)
def test_invalid_json_is_rejected_before_native_loading(
    saved_package, monkeypatch, content, message
):
    package_dir, _ = saved_package
    (package_dir / "metadata.json").write_text(content, encoding="utf-8")
    monkeypatch.setattr(model_package, "load_native_model", refuse_native_load)
    with pytest.raises(ValueError, match=message):
        model_package.load_model_package(package_dir)


@pytest.mark.parametrize(
    "field",
    [
        "model_kind",
        "numeric_features",
        "target",
        "positive_class",
        "metrics",
        "git_sha",
        "data_fingerprint",
        "versions",
        "model_files",
    ],
)
def test_missing_required_metadata_is_rejected(saved_package, monkeypatch, field):
    package_dir, _ = saved_package
    metadata = read_json(package_dir / "metadata.json")
    del metadata[field]
    write_metadata(package_dir, metadata)
    monkeypatch.setattr(model_package, "load_native_model", refuse_native_load)
    with pytest.raises(ValueError, match=f"Missing metadata fields:.*{field}"):
        model_package.load_model_package(package_dir)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("model_kind", "unsupported", "model_kind"),
        ("numeric_features", ["age", "age"], "numeric_features"),
        ("categorical_features", ["age"], "Features must"),
        ("numeric_features", ["y"], "exclude the target"),
        ("positive_class", "no", "classes must differ"),
        ("decision_threshold", True, "decision_threshold"),
        ("decision_threshold", float("nan"), "decision_threshold"),
        ("decision_threshold", 1.1, "decision_threshold"),
        ("git_sha", "not-a-sha", "git_sha"),
        ("data_fingerprint", "changed", "data_fingerprint"),
        ("versions", {"python": "3.11"}, "versions"),
        ("metrics.validation.f1", float("nan"), "metrics.validation.f1"),
        ("metrics.validation.f1", 1.1, "metrics.validation.f1"),
    ],
)
def test_invalid_metadata_values_are_rejected(saved_package, monkeypatch, field, value, message):
    package_dir, _ = saved_package
    metadata = read_json(package_dir / "metadata.json")
    destination = metadata
    parts = field.split(".")
    for key in parts[:-1]:
        destination = destination[key]
    destination[parts[-1]] = value
    write_metadata(package_dir, metadata)
    monkeypatch.setattr(model_package, "load_native_model", refuse_native_load)
    with pytest.raises(ValueError, match=message):
        model_package.load_model_package(package_dir)


@pytest.mark.parametrize(
    "name,message",
    [("metadata.json", "Cannot read valid"), ("model/pipeline.pkl", "Missing model file")],
)
def test_missing_package_files_are_rejected(saved_package, monkeypatch, name, message):
    package_dir, _ = saved_package
    (package_dir / name).unlink()
    monkeypatch.setattr(model_package, "load_native_model", refuse_native_load)
    with pytest.raises(ValueError, match=message):
        model_package.load_model_package(package_dir)


def test_missing_model_directory_is_rejected(saved_package, tmp_path, monkeypatch):
    package_dir, _ = saved_package
    source = package_dir / "model"
    destination = package_dir / "unavailable"
    assert source.resolve().is_relative_to(tmp_path.resolve())
    assert destination.resolve().is_relative_to(tmp_path.resolve())
    source.rename(destination)
    monkeypatch.setattr(model_package, "load_native_model", refuse_native_load)
    with pytest.raises(ValueError, match="Missing model directory"):
        model_package.load_model_package(package_dir)


def test_missing_autogluon_dependency_is_rejected_without_importing_sdk(saved_package, monkeypatch):
    package_dir, _ = saved_package
    metadata = read_json(package_dir / "metadata.json")
    metadata["model_kind"] = "autogluon"
    metadata["versions"]["autogluon.tabular"] = "fixture-version"
    metadata["model_files"] = ["pipeline.pkl", "models/rf/child.pkl"]
    write_metadata(package_dir, metadata)
    monkeypatch.setattr(model_package, "load_native_model", refuse_native_load)
    with pytest.raises(ValueError, match="Missing model file: models/rf/child.pkl"):
        model_package.load_model_package(package_dir)


@pytest.mark.parametrize("absolute", [False, True])
def test_manifest_paths_must_stay_inside_the_package(
    saved_package, tmp_path, monkeypatch, absolute
):
    package_dir, _ = saved_package
    outside = tmp_path / "outside.pkl"
    outside.write_bytes(b"not a model")
    metadata = read_json(package_dir / "metadata.json")
    metadata["model_kind"] = "autogluon"
    metadata["versions"]["autogluon.tabular"] = "fixture-version"
    metadata["model_files"] = [str(outside.resolve()) if absolute else "../outside.pkl"]
    write_metadata(package_dir, metadata)
    monkeypatch.setattr(model_package, "load_native_model", refuse_native_load)
    with pytest.raises(ValueError, match="Invalid model file path"):
        model_package.load_model_package(package_dir)


def test_corrupted_model_is_reported_with_its_underlying_cause(saved_package):
    package_dir, _ = saved_package
    (package_dir / "model/pipeline.pkl").write_bytes(b"broken pickle")
    with pytest.raises(ValueError, match="Cannot load the model") as error:
        model_package.load_model_package(package_dir)
    assert error.value.__cause__ is not None


def test_existing_package_is_not_overwritten(fitted_package):
    model, metadata, _, destination = fitted_package
    saved = model_package.save_model_package(model, metadata, destination)
    before = {
        path.relative_to(saved): path.read_bytes() for path in saved.rglob("*") if path.is_file()
    }
    with pytest.raises(FileExistsError):
        model_package.save_model_package(model, metadata, destination)
    after = {
        path.relative_to(saved): path.read_bytes() for path in saved.rglob("*") if path.is_file()
    }
    assert after == before


@pytest.mark.parametrize("failure", ["metadata", "io"])
def test_failed_save_does_not_leave_a_partial_package(
    fitted_package, tmp_path, monkeypatch, failure
):
    model, metadata, _, destination = fitted_package
    if failure == "metadata":
        metadata["decision_threshold"] = True
        expected_error = ValueError
        message = "decision_threshold"
    else:

        def fail_save(model, path):
            Path(path).write_bytes(b"partial")
            raise OSError("disk unavailable")

        monkeypatch.setattr(model_package, "save_sklearn", fail_save)
        expected_error = OSError
        message = "disk unavailable"
    with pytest.raises(expected_error, match=message):
        model_package.save_model_package(model, metadata, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".model-package-*"))


@pytest.fixture
def experiment_run(project, tmp_path):
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
    tested = evaluate_test(normalize_input(raw), raw, parameters, selection)
    return Path(report["run_dir"]), raw, selection, tested


@pytest.mark.parametrize("include_test", [False, True])
def test_metadata_builder_uses_the_candidate_reports(experiment_run, include_test):
    root, raw, selection, tested = experiment_run
    metadata = model_package.build_model_metadata(root, "baseline", raw, include_test=include_test)
    report = read_json(root / "metrics.json")
    assert metadata["candidate_name"] == "baseline"
    assert metadata["positive_class"] == "yes" and metadata["negative_class"] == "no"
    assert (
        metadata["decision_threshold"] == selection["candidates"]["baseline"]["decision_threshold"]
    )
    assert metadata["metrics"]["validation"] == report["results"]["baseline"]["validation"]
    assert ("test" in metadata["metrics"]) is include_test
    if include_test:
        assert metadata["metrics"]["test"] == tested["results"]["baseline"]["test"]
    assert "model_files" not in metadata
    model_package.validate_metadata(metadata, require_model_files=False)


def test_metadata_builder_preserves_historical_provenance(experiment_run):
    root, raw, _, _ = experiment_run
    provenance = read_json(root / "provenance.json")
    provenance.update(
        git_sha="b" * 40,
        git_dirty=False,
        python="3.11.12",
        versions={"numpy": "2.2.5", "pandas": "2.3.2", "scikit-learn": "1.7.1"},
    )
    write_json(root / "provenance.json", provenance)
    metadata = model_package.build_model_metadata(root, "baseline", raw)
    assert metadata["git_sha"] == provenance["git_sha"]
    assert metadata["git_dirty"] is False
    assert metadata["data_fingerprint"] == provenance["data_fingerprint"]
    assert metadata["versions"] == {**provenance["versions"], "python": "3.11.12"}


def test_metadata_builder_rejects_different_raw_data(experiment_run):
    root, raw, _, _ = experiment_run
    changed = raw.copy()
    changed.loc[changed.index[0], "age"] += 1
    with pytest.raises(ValueError, match="fingerprint"):
        model_package.build_model_metadata(root, "baseline", changed)


@pytest.mark.parametrize("source", ["metrics", "test_metrics"])
def test_metadata_builder_rejects_changed_frozen_thresholds(experiment_run, source):
    root, raw, _, _ = experiment_run
    report = read_json(root / f"{source}.json")
    report["results"]["baseline"]["decision_threshold"] = 0.9
    write_json(root / f"{source}.json", report)
    with pytest.raises(ValueError, match="differs from the frozen selection"):
        model_package.build_model_metadata(
            root, "baseline", raw, include_test=source == "test_metrics"
        )


def test_metadata_builder_can_skip_absent_test_results(experiment_run):
    root, raw, _, _ = experiment_run
    (root / "test_metrics.json").unlink()
    metadata = model_package.build_model_metadata(root, "baseline", raw)
    assert "test" not in metadata["metrics"]
    with pytest.raises(FileNotFoundError):
        model_package.build_model_metadata(root, "baseline", raw, include_test=True)


@pytest.mark.parametrize("kind", ["pipeline", "classifier", "invalid_object"])
def test_save_rejects_unfitted_or_invalid_sklearn_model(fitted_package, monkeypatch, kind):
    model, metadata, _, destination = fitted_package
    invalid = {
        "pipeline": clone(model),
        "classifier": clone(model.named_steps["classifier"]),
        "invalid_object": None,
    }[kind]

    def refuse_save(*args, **kwargs):
        pytest.fail("An invalid estimator reached serialization")

    monkeypatch.setattr(model_package, "save_sklearn", refuse_save)
    with pytest.raises(ValueError, match="must be fitted|must provide predict_proba"):
        model_package.save_model_package(invalid, metadata, destination)
    assert not destination.exists()


@pytest.mark.parametrize("kind", ["pipeline", "classifier", "invalid_object"])
def test_load_rejects_unfitted_or_invalid_sklearn_model(saved_package, fitted_package, kind):
    package_dir, _ = saved_package
    model, _, _, _ = fitted_package
    invalid = {
        "pipeline": clone(model),
        "classifier": clone(model.named_steps["classifier"]),
        "invalid_object": None,
    }[kind]
    # Pomiń funkcję zapisu, aby zasymulować błędny pakiet z zewnętrznego źródła.
    (package_dir / "model/pipeline.pkl").write_bytes(pickle.dumps(invalid))
    with pytest.raises(ValueError, match="must be fitted|must provide predict_proba"):
        model_package.load_model_package(package_dir)


@pytest.mark.parametrize("unused_feature", [False, True])
def test_autogluon_save_uses_deployment_clone(
    fitted_package, tmp_path, monkeypatch, unused_feature
):
    _, metadata, _, destination = fitted_package
    metadata = copy.deepcopy(metadata)
    metadata["model_kind"] = "autogluon"
    metadata["versions"]["autogluon.tabular"] = "test"
    if unused_feature:
        metadata["numeric_features"].append("constant")
    calls = []

    class Predictor:
        path = str(tmp_path / "training")

        def features(self):
            return ["age", "pdays", "job"]

        def clone(self, *args, **kwargs):
            pytest.fail("A full clone would include cached training data")

        def clone_for_deployment(self, path, *, model):
            calls.append(model)
            Path(path).mkdir()
            (Path(path) / "predictor.pkl").write_bytes(b"deployment")

    monkeypatch.setattr(model_package, "load_native_model", lambda *args: Predictor())
    model_package.save_model_package(Predictor(), metadata, destination)
    assert calls == ["best"]
    assert read_json(destination / "metadata.json")["model_files"] == ["predictor.pkl"]


@pytest.mark.parametrize("operation", ["save", "load"])
@pytest.mark.parametrize("change", ["missing", "unexpected", "order"])
def test_sklearn_package_rejects_incompatible_feature_schema(
    fitted_package, monkeypatch, operation, change
):
    model, metadata, _, destination = fitted_package
    metadata = copy.deepcopy(metadata)
    if operation == "load":
        model_package.save_model_package(model, metadata, destination)
        metadata = read_json(destination / "metadata.json")

    if change == "missing":
        metadata["numeric_features"].remove("pdays")
    elif change == "unexpected":
        metadata["numeric_features"].append("unknown_feature")
    else:
        metadata["numeric_features"].reverse()

    if operation == "save":

        def refuse_save(*args, **kwargs):
            pytest.fail("An incompatible schema reached model serialization")

        monkeypatch.setattr(model_package, "save_sklearn", refuse_save)
        with pytest.raises(ValueError, match="Model input features do not match metadata"):
            model_package.save_model_package(model, metadata, destination)
        assert not destination.exists()
    else:
        # Zasymuluj zmianę metadanych niezależną od dopasowanego modelu.
        write_metadata(destination, metadata)
        with pytest.raises(ValueError, match="Model input features do not match metadata"):
            model_package.load_model_package(destination)
