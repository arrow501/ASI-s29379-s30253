"""Predykcja z pakietu, zachowanie tożsamości wierszy i użycie CLI w nowym procesie."""

import copy
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from asi_project.data import normalize_input
from asi_project.model_package import (
    load_model_package,
    predict_model_package,
)


def forbidden_fit(*args, **kwargs):
    raise AssertionError("Inference must not fit a model or its preprocessing")


def test_real_package_normalizes_without_filtering_or_changing_input(saved_package, monkeypatch):
    package_dir, raw = saved_package
    for component in (Pipeline, SimpleImputer, StandardScaler, OneHotEncoder):
        monkeypatch.setattr(component, "fit", forbidden_fit)
    model, metadata = load_model_package(package_dir)
    rows = raw.iloc[[5, 0, 5]][["job", "pdays", "age"]].copy()
    rows["y"] = [None, "yes", None]
    rows["duration"] = [999, 0, 999]
    original = rows.copy(deep=True)
    result = predict_model_package(model, metadata, rows)
    expected = model.predict_proba(normalize_input(rows[["age", "pdays", "job"]]))[:, 1]
    np.testing.assert_allclose(result["probability_positive"], expected)
    assert result["prediction"].tolist() == np.where(expected >= 0.6, "yes", "no").tolist()
    pd.testing.assert_index_equal(result.index, rows.index)
    pd.testing.assert_frame_equal(rows, original)
    assert list(result.columns) == ["prediction", "probability_positive"]
    assert len(result) == 3


def test_threshold_boundary_and_reversed_encoded_classes(saved_package):
    package_dir, raw = saved_package
    _, metadata = load_model_package(package_dir)
    metadata["decision_threshold"] = 0.5

    class Model:
        classes_ = np.array([1, 0])

        def predict_proba(self, X):
            return np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]])

    result = predict_model_package(Model(), metadata, raw.iloc[:3])
    assert result.prediction.tolist() == ["no", "yes", "yes"]
    np.testing.assert_allclose(result.probability_positive, [0.1, 0.5, 0.9])


def test_autogluon_adapter_receives_float_numeric_features(saved_package):
    package_dir, raw = saved_package
    _, original = load_model_package(package_dir)
    metadata = copy.deepcopy(original)
    metadata["model_kind"] = "autogluon"
    metadata["versions"]["autogluon.tabular"] = "fixture-version"
    captured = {}

    class Predictor:
        def predict_proba(self, X):
            captured["input"] = X.copy()
            return pd.DataFrame({1: [0.2, 0.6, 0.9], 0: [0.8, 0.4, 0.1]}, index=X.index)

    result = predict_model_package(Predictor(), metadata, raw.iloc[:3])
    assert captured["input"].age.dtype == np.dtype("float64")
    assert captured["input"].pdays.dtype == np.dtype("float64")
    assert pd.isna(captured["input"].pdays.iloc[0])
    assert captured["input"].job.iloc[0] == "admin."
    assert result.prediction.tolist() == ["no", "yes", "yes"]
    pd.testing.assert_index_equal(result.index, raw.iloc[:3].index)


@pytest.mark.parametrize(
    "change,message", [("missing", "Missing required"), ("duplicate", "unique")]
)
def test_invalid_feature_columns_are_rejected(saved_package, change, message):
    package_dir, raw = saved_package
    model, metadata = load_model_package(package_dir)
    if change == "missing":
        raw = raw.drop(columns="age")
    else:
        raw = pd.concat([raw, raw[["age"]]], axis=1)
    with pytest.raises(ValueError, match=message):
        predict_model_package(model, metadata, raw)


def test_empty_input_preserves_schema_without_calling_the_model(saved_package, monkeypatch):
    package_dir, raw = saved_package
    model, metadata = load_model_package(package_dir)
    monkeypatch.setattr(model, "predict_proba", forbidden_fit)
    result = predict_model_package(model, metadata, raw.iloc[:0])
    assert result.empty
    assert list(result.columns) == ["prediction", "probability_positive"]
    assert result.probability_positive.dtype == np.dtype("float64")
    pd.testing.assert_index_equal(result.index, raw.iloc[:0].index)


def test_non_dataframe_input_is_rejected(saved_package):
    package_dir, _ = saved_package
    model, metadata = load_model_package(package_dir)
    with pytest.raises(TypeError, match="DataFrame"):
        predict_model_package(model, metadata, [])


def test_cli_uses_a_relocated_package_in_a_fresh_process_without_fit(
    saved_package, tmp_path, repo_root
):
    package_dir, raw = saved_package
    model, metadata = load_model_package(package_dir)
    expected = predict_model_package(model, metadata, raw.iloc[:5])
    moved = tmp_path / "moved"
    assert package_dir.resolve().is_relative_to(tmp_path.resolve())
    assert moved.resolve().is_relative_to(tmp_path.resolve())
    package_dir.rename(moved)
    input_path = tmp_path / "input.csv"
    raw.to_csv(input_path, index=False)
    script = repo_root / "scripts/predict_model_package.py"
    code = """
import runpy, sys
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
def forbidden(*args, **kwargs):
    raise AssertionError("inference called fit")
for component in (Pipeline, SimpleImputer, OneHotEncoder, StandardScaler):
    component.fit = forbidden
script, package, source = sys.argv[1:]
sys.argv = [script, "--package", package, "--input", source, "--rows", "5"]
runpy.run_path(script, run_name="__main__")
"""
    process = subprocess.run(
        [sys.executable, "-B", "-c", code, str(script), str(moved), str(input_path)],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(repo_root / "src"), "WANDB_MODE": "disabled"},
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(process.stdout)
    assert payload["n_rows"] == 5
    assert payload["positive_class"] == "yes"
    assert payload["decision_threshold"] == 0.6
    assert payload["predictions"] == expected.to_dict(orient="records")
