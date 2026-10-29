"""Regresje normalizacji Bank Marketing, także dla pojedynczego rekordu API."""

import numpy as np
import pandas as pd
import pytest

from asi_project.data import normalize_input
from asi_project.model import build_baseline_pipeline


@pytest.mark.parametrize(
    "values",
    [
        [-1, 5, 0, None],
        [" -1 ", "5", "0", ""],
        pd.Series([-1, 5, 0, pd.NA], dtype="Int64"),
    ],
)
def test_pdays_sentinel_becomes_missing_without_changing_other_features(values):
    frame = pd.DataFrame({"pdays": values})
    frame.index = [9, 2, 7, 4]
    frame["previous"] = [0, 1, 1, 0]
    frame["balance"] = [-100, -1, 0, 100]
    frame["poutcome"] = ["unknown", "success", "failure", "unknown"]
    original = frame.copy(deep=True)

    normalized = normalize_input(frame)

    assert pd.api.types.is_numeric_dtype(normalized["pdays"])
    assert normalized["pdays"].isna().tolist() == [True, False, False, True]
    assert normalized["pdays"].iloc[1:3].tolist() == [5, 0]
    pd.testing.assert_frame_equal(normalized.drop(columns="pdays"), frame.drop(columns="pdays"))
    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_frame_equal(normalize_input(normalized), normalized)
    for position in range(len(frame)):
        pd.testing.assert_frame_equal(
            normalize_input(frame.iloc[[position]]), normalized.iloc[[position]]
        )


def test_normalized_pdays_missing_values_work_at_fit_and_single_record_prediction():
    raw = pd.DataFrame({"pdays": [-1, 5, -1, 10, 20, -1], "previous": [0, 1, 0, 2, 1, 0]})
    train = normalize_input(raw)
    pipeline = build_baseline_pipeline(["pdays", "previous"], [])
    pipeline.fit(train, [0, 1, 0, 1, 0, 1])
    single = normalize_input(pd.DataFrame({"pdays": [-1], "previous": [0]}))
    before_prediction = single.copy(deep=True)

    probabilities = pipeline.predict_proba(single)

    assert probabilities.shape == (1, 2)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    pd.testing.assert_frame_equal(single, before_prediction)
