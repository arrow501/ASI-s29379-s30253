"""Opcjonalny test zapisu i odczytu AutoGluona; podstawowe CI nie trenuje tego modelu."""

import copy
import json
import os
import subprocess
import sys
from importlib.metadata import version

import numpy as np
import pandas as pd
import pytest

from asi_project.data import normalize_input
from asi_project.metrics import compute_metrics
from asi_project.model_package import load_model_package, save_model_package
from asi_project.provenance import data_fingerprint


def test_autogluon_package_roundtrip(request, fitted_package, tmp_path, repo_root, monkeypatch):
    if not request.config.getoption("--run-autogluon-package"):
        pytest.skip("Run with --run-autogluon-package to fit and relocate a real AutoGluon model.")
    from autogluon.features.generators import AutoMLPipelineFeatureGenerator
    from autogluon.tabular import TabularPredictor

    rng = np.random.default_rng(42)
    raw = pd.DataFrame(
        {
            "age": rng.integers(20, 80, size=120),
            "pdays": [-1, 2, 5, 10] * 30,
            "job": [" admin. ", "technician", "services", None] * 30,
            "constant": 1,
        },
        index=pd.Index(np.arange(120) * 3 + 10, name="source"),
    )
    target = pd.Series(np.arange(120) % 2, index=raw.index, name="y")
    features = normalize_input(raw).astype({"age": "float64", "pdays": "float64"})
    generator = AutoMLPipelineFeatureGenerator(
        enable_text_ngram_features=False,
        enable_text_special_features=False,
        enable_raw_text_features=False,
        verbosity=0,
    )
    generator.fit_transform(features.iloc[:80], target.iloc[:80])
    training = tmp_path / "training"
    model = TabularPredictor(
        label="y",
        problem_type="binary",
        positive_class=1,
        eval_metric="log_loss",
        path=str(training),
        verbosity=0,
        log_to_file=False,
    )
    model.fit(
        train_data=features.iloc[:80].assign(y=target.iloc[:80]),
        tuning_data=features.iloc[80:100].assign(y=target.iloc[80:100]),
        feature_generator=generator,
        hyperparameters={"RF": {"n_estimators": 5}},
        time_limit=30,
        num_cpus=1,
        num_gpus=0,
        num_bag_folds=0,
        num_stack_levels=0,
        dynamic_stacking=False,
        calibrate_decision_threshold=False,
        refit_full=False,
        fit_strategy="sequential",
        raise_on_model_failure=True,
    )
    ensembles = [name for name in model.model_names() if name.startswith("WeightedEnsemble")]
    assert ensembles, "The fixture must exercise ensemble dependencies."
    model.set_model_best(ensembles[0], save_trainer=True)
    source_models = model.model_names()
    source_cache = {
        path.relative_to(training): path.read_bytes()
        for path in (training / "utils/data").rglob("*")
        if path.is_file()
    }
    assert source_cache, "The fixture must contain cached training data before export."

    _, original_metadata, _, _ = fitted_package
    metadata = copy.deepcopy(original_metadata)
    metadata["model_kind"] = "autogluon"
    metadata["numeric_features"].append("constant")
    metadata["versions"]["autogluon.tabular"] = version("autogluon.tabular")
    metadata["data_fingerprint"] = data_fingerprint(
        raw.iloc[:100].assign(y=target.iloc[:100].map({0: "no", 1: "yes"}))
    )
    valid_probability = model.predict_proba(features.iloc[80:100]).loc[:, 1].to_numpy()
    metadata["metrics"] = {
        "validation": compute_metrics(target.iloc[80:100], valid_probability >= 0.6)
    }
    expected_probability = model.predict_proba(features.iloc[100:]).loc[:, 1].to_numpy()
    expected_classes = np.where(expected_probability >= 0.6, "yes", "no").tolist()
    expected_index = raw.iloc[100:].index.tolist()

    def forbidden(*args, **kwargs):
        raise AssertionError("Package loading or inference called fit")

    monkeypatch.setattr(TabularPredictor, "fit", forbidden)
    invalid_metadata = copy.deepcopy(metadata)
    invalid_metadata["numeric_features"].remove("pdays")
    invalid_destination = tmp_path / "invalid-package"
    with pytest.raises(ValueError, match="Metadata omits model input features.*pdays"):
        save_model_package(model, invalid_metadata, invalid_destination)
    assert not invalid_destination.exists()

    package = save_model_package(model, metadata, tmp_path / "ag-package")
    assert not any(path.is_file() for path in (package / "model/utils/data").rglob("*"))
    assert model.model_names() == source_models
    assert {
        path.relative_to(training): path.read_bytes()
        for path in (training / "utils/data").rglob("*")
        if path.is_file()
    } == source_cache
    moved = tmp_path / "moved"
    unavailable = tmp_path / "training-unavailable"
    for path in (package, moved, training, unavailable):
        assert path.resolve().is_relative_to(tmp_path.resolve())
    package.rename(moved)
    training.rename(unavailable)
    assert not package.exists() and not training.exists()
    inputs = tmp_path / "input.pkl"
    raw.iloc[100:].to_pickle(inputs)
    code = """
import json, sys
import pandas as pd
from autogluon.tabular import TabularPredictor
from asi_project.model_package import load_model_package, predict_model_package
def forbidden(*args, **kwargs):
    raise AssertionError("inference called fit")
TabularPredictor.fit = forbidden
model, metadata = load_model_package(sys.argv[1])
result = predict_model_package(model, metadata, pd.read_pickle(sys.argv[2]))
print(json.dumps({"model_best": model.model_best, "index": result.index.tolist(), "predictions": result.to_dict(orient="records")}))
"""
    process = subprocess.run(
        [sys.executable, "-B", "-c", code, str(moved), str(inputs)],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(repo_root / "src"), "WANDB_MODE": "disabled"},
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(process.stdout)
    assert payload["model_best"] == ensembles[0]
    assert payload["index"] == expected_index
    assert [row["prediction"] for row in payload["predictions"]] == expected_classes
    np.testing.assert_allclose(
        [row["probability_positive"] for row in payload["predictions"]],
        expected_probability,
    )
    assert (moved / "metadata.json").is_file()
    stored = json.loads((moved / "metadata.json").read_text(encoding="utf-8"))
    assert len(stored["model_files"]) > 1
    assert not any(name.startswith("utils/data/") for name in stored["model_files"])

    assert "model_files" not in metadata
    assert stored == {**metadata, "model_files": stored["model_files"]}

    invalid_metadata = copy.deepcopy(stored)
    invalid_metadata["numeric_features"].remove("pdays")
    metadata_path = moved / "metadata.json"
    original = metadata_path.read_bytes()
    try:
        metadata_path.write_text(json.dumps(invalid_metadata), encoding="utf-8")
        with pytest.raises(ValueError, match="Metadata omits model input features.*pdays"):
            load_model_package(moved)
    finally:
        metadata_path.write_bytes(original)
