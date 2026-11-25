"""Independent boundaries and saved-artifact checks for optional split diagnostics."""

import json

import numpy as np
import pandas as pd
import pytest

from asi_project.split_exploration import (
    category_drift,
    full_sequence_view,
    read_oof_catalog,
    split_layout,
    split_summary,
    threshold_counts,
    threshold_curve,
)


def labels(n=100):
    return pd.Series(np.arange(n) % 2, index=np.arange(n) * 7 + 11)


def test_holdout_preserves_source_indices_and_exact_boundaries():
    y = labels()
    parts = split_layout(y)
    assert parts.index.equals(y.index)
    assert parts.tolist() == ["train"] * 64 + ["validation"] * 16 + ["test"] * 20
    X = pd.DataFrame({"previous": [0] * 64 + [1] * 36}, index=y.index)
    stats = split_summary(X, y, parts).set_index("part")
    assert stats.rows.to_dict() == {"train": 64, "validation": 16, "test": 20}
    assert stats.positive.to_dict() == {"train": 32, "validation": 8, "test": 10}
    assert stats.previous_contact.to_dict() == {"train": 0, "validation": 1, "test": 1}
    assert stats.loc["validation", "first_position"] == 65
    assert stats.loc["test", "last_position"] == 100
    with pytest.raises(ValueError, match="identical indices"):
        split_summary(X.iloc[::-1], y, parts)


@pytest.mark.parametrize("fold,train_end,val_end", [(1, 32, 48), (2, 48, 64), (3, 64, 80)])
def test_expanding_folds_never_use_future_development(fold, train_end, val_end):
    parts = split_layout(labels(), "chronological", fold=fold)
    assert parts.tolist() == (
        ["train"] * train_end + ["validation"] * 16 + ["unused"] * (80 - val_end) + ["test"] * 20
    )


def test_random_folds_cover_development_once_and_keep_test_fixed():
    y = labels()
    validation = []
    for fold in range(1, 6):
        parts = split_layout(y, "bagged", fold=fold)
        assert parts.iloc[80:].eq("test").all()
        assert parts.iloc[:80].isin(["train", "validation"]).all()
        assert y[parts.eq("validation")].value_counts().to_dict() == {0: 8, 1: 8}
        validation.extend(parts[parts.eq("validation")].index)
    assert sorted(validation) == sorted(y.iloc[:80].index)
    pd.testing.assert_series_equal(split_layout(y, "bagged"), split_layout(y, "bagged"))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"protocol": "unknown"},
        {"test_size": 1},
        {"validation_fraction": 0},
        {"protocol": "chronological", "fold": True},
        {"protocol": "chronological", "validation_fraction": 0.4},
        {"protocol": "bagged", "fold": 6},
    ],
)
def test_invalid_split_inputs_fail(kwargs):
    with pytest.raises(ValueError):
        split_layout(labels(), **kwargs)


def test_novel_categories_and_unknown_keep_their_support():
    X = pd.DataFrame({"month": ["jan", "unknown", "apr", "apr", None]})
    parts = pd.Series(["train", "train", "validation", "test", "test"])
    table = category_drift(X, parts, "month").set_index(["part", "category"])
    assert table.loc[("train", "unknown"), "share"] == 0.5
    assert table.loc[("validation", "apr"), "unseen_in_train"]
    assert table.loc[("test", "<missing>"), "rows"] == 1
    assert table.groupby(level=0).share.sum().eq(1).all()


def test_zero_f1_at_default_threshold_can_hide_perfect_ranking():
    target, probability = [0, 1, 0, 1], [0.1, 0.4, 0.2, 0.3]
    assert threshold_counts(target, probability, 0.5) == {
        "tp": 0,
        "fp": 0,
        "fn": 2,
        "tn": 2,
        "precision": 0,
        "recall": 0,
        "f1": 0,
    }
    assert threshold_counts(target, probability, 0.3)["f1"] == 1
    curve = threshold_curve(target, probability).set_index("threshold")
    assert curve.loc[0.3, "f1"] == 1
    assert curve.loc[0.1, "f1"] == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    "target,probability",
    [
        ([0, 1], [0.1]),
        ([0, 2], [0.1, 0.2]),
        ([0, 1], [0.1, np.nan]),
        ([0, 1], [0.1, 1.1]),
        ([[0], [1]], [[0.1], [0.2]]),
    ],
)
def test_threshold_functions_reject_misalignment_and_invalid_values(target, probability):
    for function in (threshold_curve, lambda y, p: threshold_counts(y, p, 0.5)):
        with pytest.raises(ValueError):
            function(target, probability)


def write_run(root, run, *, fingerprint="match", indices=None, target=None):
    folder = root / "data/08_reporting/experiments" / run
    folder.mkdir(parents=True)
    records = {
        "provenance": {"data_fingerprint": fingerprint},
        "metrics": {"protocol": "bagged", "status": "completed", "results": {"model": {}}},
        "config": {"project": {"test_size": 0.2}},
        "model_oof": {
            "indices": indices or [11, 18],
            "target": target or [0, 1],
            "probability": [0.1, 0.4],
        },
    }
    for name, record in records.items():
        (folder / f"{name}.json").write_text(json.dumps(record), encoding="utf-8")


def test_catalog_reads_real_json_selects_latest_and_ignores_other_data(tmp_path):
    assert read_oof_catalog(tmp_path, "match", labels()) == {}
    write_run(tmp_path, "001")
    write_run(tmp_path, "002")
    write_run(tmp_path, "003", fingerprint="other-dataset")
    catalog = read_oof_catalog(tmp_path, "match", labels())
    assert len(catalog) == 1
    assert next(iter(catalog.values()))["run"] == "002"
    np.testing.assert_array_equal(next(iter(catalog.values()))["target"], [0, 1])


@pytest.mark.parametrize(
    "indices,target", [([571, 578], [0, 1]), ([11, 18], [1, 0]), ([11, 11], [0, 1])]
)
def test_catalog_rejects_test_rows_changed_labels_and_duplicates(tmp_path, indices, target):
    write_run(tmp_path, "001", indices=indices, target=target)
    with pytest.raises(ValueError, match="do not match development"):
        read_oof_catalog(tmp_path, "match", labels())


def test_sample_full_sequence_is_available_without_download(repo_root):
    X, y, fingerprint = full_sequence_view(repo_root, "sample")
    assert X.index.equals(y.index)
    assert len(X) >= 10
    assert "duration" not in X and "y" not in X
    assert len(fingerprint) == 64
    assert split_layout(y).eq("test").any()
