"""Przypadki brzegowe predykcji baseline."""

import numpy as np
import pandas as pd
import pytest

from asi_project.model import build_baseline_pipeline


def test_pipeline_handles_missing_and_unseen_values_without_mutating_input():
    train = pd.DataFrame({"num": [1.0, np.nan, 3.0, 4.0], "cat": ["a", "b", np.nan, "a"]})
    original = train.copy(deep=True)
    pipeline = build_baseline_pipeline(["num"], ["cat"])
    pipeline.fit(train, [0, 1, 0, 1])
    test = pd.DataFrame({"num": [np.nan, 1000.0], "cat": ["new", np.nan]})
    probabilities = pipeline.predict_proba(test)
    assert probabilities.shape == (2, 2)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    pd.testing.assert_frame_equal(train, original)
    # Predicting on new values must not refit the training statistics.
    numeric = pipeline.named_steps["preprocessing"].named_transformers_["numeric"]
    np.testing.assert_allclose(numeric.named_steps["imputer"].statistics_, [3.0])


def test_pipeline_retains_features_missing_throughout_training():
    train = pd.DataFrame({"num": [np.nan] * 4, "cat": [np.nan] * 4})
    pipeline = build_baseline_pipeline(["num"], ["cat"])
    pipeline.fit(train, [0, 1, 0, 1])
    test = pd.DataFrame({"num": [2.0], "cat": ["new"]})
    assert np.isfinite(pipeline.predict_proba(test)).all()


@pytest.mark.parametrize("dtype", ["object", "string", "category"])
@pytest.mark.parametrize("missing", [pd.NA, None, np.nan])
def test_pipeline_accepts_pandas_missing_values_at_fit_and_predict(dtype, missing):
    train = pd.DataFrame(
        {
            "num": pd.Series([1, pd.NA, 3, 4], dtype="Float64"),
            "cat": pd.Series(["a", missing, "b", "a"], dtype=dtype),
        }
    )
    original = train.copy(deep=True)
    pipeline = build_baseline_pipeline(["num"], ["cat"])
    pipeline.fit(train, [0, 1, 0, 1])
    test = pd.DataFrame({"num": [np.nan, 2.0], "cat": pd.Series([missing, "new"], dtype=dtype)})
    before_prediction = test.copy(deep=True)
    probabilities = pipeline.predict_proba(test)
    assert probabilities.shape == (2, 2)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    # All missing representations must receive the same imputation.
    equivalent = test.copy()
    equivalent["cat"] = equivalent["cat"].astype(object).where(test["cat"].notna(), np.nan)
    np.testing.assert_allclose(probabilities, pipeline.predict_proba(equivalent))
    pd.testing.assert_frame_equal(train, original)
    pd.testing.assert_frame_equal(test, before_prediction)
