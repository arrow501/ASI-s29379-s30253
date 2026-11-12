"""Regression tests for the prepared_data → persisted baseline boundary."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro.io import DataCatalog, MemoryDataset
from kedro.runner import SequentialRunner

from asi_project.config import ConfigError
from asi_project.config_validation import project_parameters
from asi_project.data import filter_training_rows, normalize_input
from asi_project.metrics import validate_metrics_record
from asi_project.model import build_baseline_pipeline
from asi_project.pipelines.data_science.nodes import evaluate_models, split_data, train_models
from asi_project.pipelines.data_science.pipeline import create_pipeline


@pytest.fixture
def prepared(project):
    return filter_training_rows(normalize_input(project.sample), project.contract)


@pytest.fixture
def params(project):
    return project_parameters(project.config, project.contract)


def test_split_preserves_row_order_and_excludes_leakage(prepared, params):
    original = prepared.copy(deep=True)
    X_train, X_test, y_train, y_test = split_data(prepared, params)
    cut = len(prepared) - int(np.ceil(len(prepared) * params["test_size"]))
    pd.testing.assert_index_equal(X_train.index, prepared.index[:cut])
    pd.testing.assert_index_equal(X_test.index, prepared.index[cut:])
    pd.testing.assert_index_equal(X_train.index, y_train.index)
    pd.testing.assert_index_equal(X_test.index, y_test.index)
    assert not set(X_train.index).intersection(X_test.index)
    assert list(X_train) == params["numeric_features"] + params["categorical_features"]
    assert "duration" not in X_train and "y" not in X_train
    pd.testing.assert_frame_equal(prepared, original)


def test_fit_keeps_preprocessing_training_only_and_predicts_unseen_categories(prepared, params):
    X_train, X_test, y_train, _ = split_data(prepared, params)
    original_train = X_train.copy(deep=True)
    X_test = X_test.copy()
    X_test.loc[:, "age"] = 10**8
    X_test.loc[:, "job"] = "category_not_seen_in_training"
    model, dummy = train_models(X_train, y_train, params, {"max_iter": 1000})
    numeric = model.named_steps["preprocessing"].named_transformers_["numeric"]
    expected = X_train[params["numeric_features"]].median().fillna(0).to_numpy(dtype=float)
    np.testing.assert_allclose(numeric.named_steps["imputer"].statistics_, expected)
    filled_train = (
        X_train[params["numeric_features"]]
        .astype(float)
        .fillna(X_train[params["numeric_features"]].median())
        .fillna(0)
    )
    np.testing.assert_allclose(numeric.named_steps["scaler"].mean_, filled_train.mean())
    np.testing.assert_allclose(numeric.named_steps["scaler"].var_, filled_train.var(ddof=0))
    categorical = model.named_steps["preprocessing"].named_transformers_["categorical"]
    for column, categories in zip(
        params["categorical_features"], categorical.named_steps["encoder"].categories_, strict=True
    ):
        assert set(categories) == set(X_train[column].dropna())
    assert "category_not_seen_in_training" not in categorical.named_steps["encoder"].categories_[0]
    pd.testing.assert_frame_equal(X_train, original_train)
    proba = model.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1)
    assert len(dummy.predict(X_test)) == len(X_test)


def test_effective_training_parameters_reach_model(prepared, params):
    params = {**params, "random_state": 23}
    X_train, _, y_train, _ = split_data(prepared, params)
    model, _ = train_models(X_train, y_train, params, {"max_iter": 1700})
    assert model.named_steps["classifier"].max_iter == 1700
    assert model.named_steps["classifier"].random_state == 23


@pytest.mark.parametrize("invalid", [0, -1, 1.5, True])
def test_invalid_training_parameters_are_rejected(prepared, params, invalid):
    X_train, _, y_train, _ = split_data(prepared, params)
    with pytest.raises(ValueError, match="max_iter"):
        train_models(X_train, y_train, params, {"max_iter": invalid})


@pytest.mark.parametrize(
    "change",
    [{"numeric_features": ["duration"]}, {"split_strategy": "stratified_random"}, {"test_size": 0}],
)
def test_direct_nodes_still_enforce_contract(prepared, params, change):
    with pytest.raises(ValueError):
        split_data(prepared, {**params, **change})


def test_evaluation_uses_effective_metadata_and_existing_schema(prepared, params, project):
    params = {**params, "test_size": 0.3, "random_state": 17}
    X_train, X_test, y_train, y_test = split_data(prepared, params)
    model, dummy = train_models(X_train, y_train, params, {"max_iter": 1000})
    record = evaluate_models(model, dummy, X_test, y_test, y_train, params, "sample")
    assert validate_metrics_record(record, project.contract) == []
    assert record["split"]["test_size"] == 0.3
    assert record["split"]["random_state"] == 17
    assert record["split"]["n_train"] + record["split"]["n_test"] == len(prepared)
    assert record["data_mode"] == "sample"
    assert record["models"]["dummy"]["f1"] == 0
    with pytest.raises(ValueError, match="data_mode"):
        evaluate_models(model, dummy, X_test, y_test, y_train, params, "unknown")


@pytest.mark.parametrize("env", ["local", "ci"])
@pytest.mark.parametrize("absolute_override", [False, True])
def test_real_runner_writes_reloadable_model_and_json(
    prepared, params, repo_root, tmp_path, env, absolute_override
):
    bootstrap_project(repo_root)
    with KedroSession.create(project_path=repo_root, env=env) as session:
        context = session.load_context()
        run_params = context.params
        entries = {
            name: deepcopy(context.config_loader["catalog"][name])
            for name in ("trained_model", "metrics")
        }
    originals = []
    if absolute_override:
        for name, entry in entries.items():
            original = tmp_path / "existing_artifacts" / name / Path(entry["filepath"]).name
            original.parent.mkdir(parents=True)
            original.write_bytes(b"existing artifact must not change")
            entry["filepath"] = str(original)
            originals.append(original)
    for name, entry in entries.items():
        # Absolute paths from local overrides must never escape the test output directory.
        entry["filepath"] = str(tmp_path / name / Path(entry["filepath"]).name)
    catalog = DataCatalog.from_config(entries)
    for name, value in {
        "prepared_data": prepared,
        "params:project": params,
        "params:training": run_params["training"],
        "params:reporting.data_mode": run_params["reporting"]["data_mode"],
    }.items():
        catalog[name] = MemoryDataset(data=value)
    SequentialRunner().run(create_pipeline(), catalog)
    for original in originals:
        assert original.read_bytes() == b"existing artifact must not change"
    # A fresh catalog must load real files, rather than reuse runner state.
    assert Path(entries["trained_model"]["filepath"]).is_file()
    assert Path(entries["metrics"]["filepath"]).is_file()
    fresh_catalog = DataCatalog.from_config(entries)
    restored = fresh_catalog.load("trained_model")
    record = fresh_catalog.load("metrics")
    assert validate_metrics_record(record) == []
    X_train, X_test, y_train, _ = split_data(prepared, params)
    # Reference the S1 factory directly, not the node under test again.
    expected = build_baseline_pipeline(
        params["numeric_features"],
        params["categorical_features"],
        max_iter=run_params["training"]["max_iter"],
        random_state=params["random_state"],
    ).fit(X_train, y_train)
    np.testing.assert_allclose(restored.predict_proba(X_test), expected.predict_proba(X_test))
    assert record["split"]["n_test"] == len(X_test)
    assert record["data_mode"] == ("sample" if env == "ci" else "full")


def test_ci_output_paths_do_not_overwrite_full_outputs(repo_root):
    base = yaml.safe_load((repo_root / "conf/base/catalog.yml").read_text())
    ci = yaml.safe_load((repo_root / "conf/ci/catalog.yml").read_text())
    paths = [
        (repo_root / entries[name]["filepath"]).resolve()
        for entries in (base, ci)
        for name in ("trained_model", "metrics")
    ]
    assert len(set(paths)) == 4
    assert all((repo_root / "data").resolve() in path.parents for path in paths)
    assert (repo_root / "metrics.json").resolve() not in paths


def test_training_rejects_misaligned_target_before_fit(prepared, params):
    X_train, _, y_train, _ = split_data(prepared, params)
    with pytest.raises(ValueError, match="align"):
        train_models(X_train, y_train.iloc[::-1], params, {"max_iter": 1000})


def test_nonconvergence_does_not_return_a_model(prepared, params):
    X_train, _, y_train, _ = split_data(prepared, params)
    with pytest.raises(ValueError, match="converge"):
        train_models(X_train, y_train, params, {"max_iter": 1})


def test_split_does_not_sort_arbitrary_source_indices(prepared, params):
    reordered_index = prepared.copy()
    reordered_index.index = pd.Index(np.arange(len(prepared))[::-1])
    X_train, X_test, _, _ = split_data(reordered_index, params)
    pd.testing.assert_index_equal(X_train.index, reordered_index.index[: len(X_train)])
    pd.testing.assert_index_equal(X_test.index, reordered_index.index[len(X_train) :])


@pytest.mark.parametrize("bad_project", [{}, {"dataset_id": "missing"}, {"dataset_id": []}, None])
def test_unknown_or_malformed_dataset_selection_has_clear_error(prepared, bad_project):
    with pytest.raises(ConfigError, match="dataset_id"):
        split_data(prepared, bad_project)


def test_training_rejects_unencoded_labels(prepared, params):
    X_train, _, y_train, _ = split_data(prepared, params)
    with pytest.raises(ValueError, match="encoded"):
        train_models(X_train, y_train.replace({0: "no", 1: "yes"}), params, {"max_iter": 1000})


def test_evaluation_rejects_misalignment_and_incorrect_model_metadata(prepared, params):
    X_train, X_test, y_train, y_test = split_data(prepared, params)
    model, dummy = train_models(X_train, y_train, params, {"max_iter": 1000})
    with pytest.raises(ValueError, match="align"):
        evaluate_models(model, dummy, X_test, y_test.iloc[::-1], y_train, params, "sample")
    reduced = {**params, "numeric_features": ["age"]}
    features = reduced["numeric_features"] + reduced["categorical_features"]
    with pytest.raises(ValueError, match="metadata"):
        evaluate_models(model, dummy, X_test[features], y_test, y_train, reduced, "sample")


def test_evaluation_numbers_have_an_independent_known_answer(prepared, params):
    X_train, X_test, y_train, _ = split_data(prepared, params)
    X_test = X_test.iloc[:4]
    y_test = pd.Series([0, 0, 1, 1], index=X_test.index)
    model = SimpleNamespace(
        feature_names_in_=np.asarray(X_test.columns), predict=lambda frame: np.array([0, 1, 0, 1])
    )
    dummy = SimpleNamespace(predict=lambda frame: np.zeros(len(frame), dtype=int))
    record = evaluate_models(model, dummy, X_test, y_test, y_train, params, "sample")
    assert record["models"]["logistic_regression"] == {
        "accuracy": 0.5,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "confusion_matrix": {"tn": 1, "fp": 1, "fn": 1, "tp": 1},
    }
    assert record["models"]["dummy"] == {
        "accuracy": 0.5,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "confusion_matrix": {"tn": 2, "fp": 0, "fn": 2, "tp": 0},
    }
