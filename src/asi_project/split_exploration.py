"""Descriptive split diagnostics, independent of training pipelines and tracking SDKs."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold

from asi_project.config import get_contract, load_project_config
from asi_project.data import (
    assert_valid,
    filter_training_rows,
    load_dataset,
    make_xy,
    normalize_input,
)
from asi_project.exploration import segment_rates

PART_LABELS = {
    "train": "Train",
    "validation": "Validation",
    "test": "Test",
    "unused": "Future development (unused)",
}
PART_COLOURS = {"train": "#2a9d8f", "validation": "#e9a23b", "test": "#d26464", "unused": "#bcc3cc"}


def full_sequence_view(root, mode):
    config = load_project_config(root)
    contract = get_contract(config)
    if contract.split_strategy != "time_ordered":
        raise ValueError("These diagnostics require chronologically ordered data.")
    raw = load_dataset(config, contract, mode=mode)
    assert_valid(raw, contract, config)
    fingerprint = hashlib.sha256(
        raw.to_csv(index=True, lineterminator="\n").encode("utf-8")
    ).hexdigest()
    prepared = filter_training_rows(normalize_input(raw), contract)
    X, y = make_xy(prepared, contract, config.model_features)
    return X, y, fingerprint


def split_layout(y, protocol="holdout", *, validation_fraction=0.2, fold=3, test_size=0.2, seed=42):
    """Return descriptive membership; never shuffle the source table itself."""
    if (
        not isinstance(y, pd.Series)
        or len(y) < 10
        or not y.index.is_unique
        or y.isna().any()
        or not y.isin([0, 1]).all()
    ):
        raise ValueError("Need at least ten aligned binary-labelled rows with a unique index.")
    if (
        isinstance(test_size, bool)
        or not isinstance(test_size, int | float)
        or not 0 < test_size < 1
    ):
        raise ValueError("test_size must be in (0, 1).")
    if (
        isinstance(validation_fraction, bool)
        or not isinstance(validation_fraction, int | float)
        or not 0 < validation_fraction < 1
    ):
        raise ValueError("validation_fraction must be in (0, 1).")
    n = len(y)
    development = n - math.ceil(n * test_size)
    labels = np.full(n, "unused", dtype=object)
    labels[development:] = "test"
    if protocol == "holdout":
        cut = development - math.ceil(development * validation_fraction)
        if cut < 1:
            raise ValueError("The training part would be empty.")
        labels[:cut], labels[cut:development] = "train", "validation"
    elif protocol == "chronological":
        if isinstance(fold, bool) or not isinstance(fold, int) or not 1 <= fold <= 3:
            raise ValueError("Chronological fold must be 1, 2 or 3.")
        window = math.ceil(development * validation_fraction)
        first = development - 3 * window
        if first < 1:
            raise ValueError("Three validation windows leave no initial training rows.")
        cut = first + (fold - 1) * window
        labels[:cut], labels[cut : cut + window] = "train", "validation"
    elif protocol == "bagged":
        if isinstance(fold, bool) or not isinstance(fold, int) or not 1 <= fold <= 5:
            raise ValueError("Random fold must be between 1 and 5.")
        if y.iloc[:development].value_counts().min() < 5 or y.iloc[:development].nunique() != 2:
            raise ValueError(
                "Five stratified folds need at least five development rows of each class."
            )
        folds = list(
            StratifiedKFold(5, shuffle=True, random_state=seed).split(
                np.arange(development), y.iloc[:development]
            )
        )
        train, valid = folds[fold - 1]
        labels[train], labels[valid] = "train", "validation"
    else:
        raise ValueError(f"Unknown protocol: {protocol!r}")
    return pd.Series(labels, index=y.index, name="part")


def split_summary(X, y, membership):
    if not X.index.equals(y.index) or not membership.index.equals(y.index):
        raise ValueError("Features, targets and membership must have identical indices.")
    rates = segment_rates(membership, y).set_index("segment")
    rows = []
    for part in PART_LABELS:
        mask = membership.eq(part)
        if not mask.any():
            continue
        positions = np.flatnonzero(mask)
        row = rates.loc[part]
        rows.append(
            {
                "part": part,
                "rows": int(row.rows),
                "positive": int(row.positive),
                "rate": float(row.rate),
                "low": float(row.low),
                "high": float(row.high),
                "first_position": int(positions.min() + 1),
                "last_position": int(positions.max() + 1),
                "previous_contact": float(X.loc[mask, "previous"].gt(0).mean()),
            }
        )
    return pd.DataFrame(rows)


def category_drift(X, membership, feature):
    if not X.index.equals(membership.index):
        raise ValueError("Membership must preserve the feature index.")
    values = X[feature].astype("string").fillna("<missing>")
    seen = set(values.loc[membership.eq("train")])
    table = pd.crosstab(membership, values).reindex(
        index=["train", "validation", "test"], fill_value=0
    )
    table = table.loc[table.sum(axis=1).gt(0)]
    counts = table.stack().rename("rows").reset_index()
    counts.columns = ["part", "category", "rows"]
    counts["share"] = counts.rows / counts.part.map(table.sum(axis=1))
    counts["unseen_in_train"] = ~counts.category.isin(seen)
    return counts


def read_oof_catalog(root, fingerprint, y, test_size=0.2):
    """Read the latest matching completed run per protocol; reject holdout rows or labels."""
    folder = Path(root) / "data/08_reporting/experiments"
    catalog = {}
    selected_protocols = set()
    development = len(y) - math.ceil(len(y) * test_size)
    allowed = y.iloc[:development]
    for provenance_path in sorted(folder.glob("*/provenance.json"), reverse=True):
        run = provenance_path.parent
        if not (run / "metrics.json").is_file() or not (run / "config.json").is_file():
            continue
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        report = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        protocol = report.get("protocol")
        if (
            provenance.get("data_fingerprint") != fingerprint
            or report.get("status") != "completed"
            or protocol not in {"bagged", "chronological"}
            or protocol in selected_protocols
            or config.get("project", {}).get("test_size") != test_size
        ):
            continue
        records = {}
        for name, scores in report["results"].items():
            path = run / f"{name}_oof.json"
            if not path.is_file():
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            indices = pd.Index(record["indices"])
            probability = np.asarray(record["probability"], dtype=float)
            target = np.asarray(record["target"])
            if (
                not indices.is_unique
                or not indices.isin(allowed.index).all()
                or probability.shape != (len(indices),)
                or target.shape != (len(indices),)
                or not np.isfinite(probability).all()
                or not ((probability >= 0) & (probability <= 1)).all()
                or not np.array_equal(target, allowed.loc[indices].to_numpy())
                or len(np.unique(target)) != 2
            ):
                raise ValueError(f"OOF rows/labels/probabilities do not match development: {path}")
            records[f"{protocol} · {name} · {run.name}"] = {
                "protocol": protocol,
                "model": name,
                "run": run.name,
                "target": target,
                "probability": probability,
                "scores": scores,
            }
        if records:
            catalog.update(records)
            selected_protocols.add(protocol)
    return catalog


def _validated_predictions(target, probability):
    probability = np.asarray(probability, dtype=float)
    target = np.asarray(target)
    if (
        target.ndim != 1
        or probability.shape != target.shape
        or set(target) != {0, 1}
        or not np.isfinite(probability).all()
        or not ((probability >= 0) & (probability <= 1)).all()
    ):
        raise ValueError("Need aligned binary targets and finite probabilities in [0, 1].")
    return target, probability


def threshold_curve(target, probability):
    target, probability = _validated_predictions(target, probability)
    precision, recall, thresholds = precision_recall_curve(target, probability)
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return pd.DataFrame(
        {"threshold": thresholds, "precision": precision[:-1], "recall": recall[:-1], "f1": f1}
    )


def threshold_counts(target, probability, threshold):
    target, probability = _validated_predictions(target, probability)
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be in [0, 1].")
    truth = np.asarray(target) == 1
    positive = np.asarray(probability) >= threshold
    tp, fp = int((positive & truth).sum()), int((positive & ~truth).sum())
    fn, tn = int((~positive & truth).sum()), int((~positive & ~truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }
