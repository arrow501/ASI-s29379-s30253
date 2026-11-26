"""Binary probabilities, validation-only threshold selection and metrics."""

import time

import numpy as np
from sklearn.metrics import average_precision_score, log_loss, precision_recall_curve, roc_auc_score

from asi_project.metrics import compute_metrics


def positive_probabilities(model, X):
    probabilities = model.predict_proba(X)
    if hasattr(probabilities, "loc"):
        if len(probabilities.columns) != 2 or set(probabilities.columns) != {0, 1}:
            raise ValueError("Probability columns must be exactly the encoded classes 0 and 1.")
        if not probabilities.index.equals(X.index):
            raise ValueError("Probability rows must preserve the input index and order.")
        probabilities = probabilities.loc[:, [0, 1]].to_numpy()
    else:
        classes = np.asarray(model.classes_)
        if classes.shape != (2,) or set(classes) != {0, 1}:
            raise ValueError("Model classes must be exactly the encoded classes 0 and 1.")
        probabilities = np.asarray(probabilities)
        if probabilities.ndim != 2 or probabilities.shape[1] != 2:
            raise ValueError("Invalid binary probability shape.")
        probabilities = probabilities[:, classes.argsort()]
    if (
        probabilities.shape != (len(X), 2)
        or not np.isfinite(probabilities).all()
        or not ((probabilities >= 0) & (probabilities <= 1)).all()
        or not np.allclose(probabilities.sum(axis=1), 1)
    ):
        raise ValueError("Invalid binary probabilities.")
    return probabilities[:, 1]


def validation_threshold(y, probability):
    """Maximize validation F1; ties choose the highest threshold."""
    probability = np.asarray(probability)
    target = np.asarray(y)
    if (
        probability.ndim != 1
        or target.ndim != 1
        or len(target) != len(probability)
        or len(target) == 0
        or set(target) != {0, 1}
        or not np.isfinite(probability).all()
        or not ((probability >= 0) & (probability <= 1)).all()
    ):
        raise ValueError(
            "Threshold selection needs aligned binary targets and finite probabilities in [0, 1]."
        )
    precision, recall, thresholds = precision_recall_curve(y, probability)
    denominator = precision[:-1] + recall[:-1]
    scores = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )
    # Błędy zaokrągleń nie powinny zmieniać reguły wyboru wyższego progu przy remisie.
    best_scores = np.isclose(scores, scores.max(), rtol=0, atol=1e-12)
    return float(thresholds[np.flatnonzero(best_scores)[-1]])


def validation_result(model, X, y, threshold_mode="fixed"):
    start = time.perf_counter()
    probability = positive_probabilities(model, X)
    seconds = time.perf_counter() - start
    threshold = validation_threshold(y, probability) if threshold_mode == "validation_f1" else 0.5
    prediction = (probability >= threshold).astype(int)
    return {
        "validation": compute_metrics(y, prediction),
        "validation_at_0_5": compute_metrics(y, (probability >= 0.5).astype(int)),
        "validation_diagnostics": {
            "positive_rate": float(y.mean()),
            "predicted_positive_count": int(prediction.sum()),
            "max_probability": float(probability.max()),
            "roc_auc": float(roc_auc_score(y, probability)),
            "average_precision": float(average_precision_score(y, probability)),
            "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        },
        "prediction_seconds": seconds,
        "prediction_rows": len(X),
        "decision_threshold": threshold,
        "threshold_source": "validation" if threshold_mode == "validation_f1" else "fixed",
    }


def fold_result(y, probability, experiments, source, fold_slices):
    if (
        probability.shape != (len(y),)
        or not np.isfinite(probability).all()
        or not ((probability >= 0) & (probability <= 1)).all()
    ):
        raise ValueError("Invalid out-of-fold probabilities.")
    threshold = (
        validation_threshold(y, probability)
        if experiments["threshold_mode"] == "validation_f1"
        else 0.5
    )
    fold_scores = [
        compute_metrics(y.iloc[a:b], probability[a:b] >= threshold)["f1"] for a, b in fold_slices
    ]
    return {
        "validation": compute_metrics(y, probability >= threshold),
        "validation_at_0_5": compute_metrics(y, probability >= 0.5),
        "selection_score": float(np.mean(fold_scores)),
        "fold_f1": fold_scores,
        "fold_f1_min": float(min(fold_scores)),
        "fold_f1_std": float(np.std(fold_scores)),
        "decision_threshold": threshold,
        "threshold_source": source if experiments["threshold_mode"] == "validation_f1" else "fixed",
    }
