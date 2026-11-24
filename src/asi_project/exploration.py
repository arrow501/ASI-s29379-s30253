"""Training-only data and geometry for the optional visual exploration app."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.preprocessing import RobustScaler

from asi_project.config import get_contract, load_project_config
from asi_project.data import (
    assert_valid,
    filter_training_rows,
    load_dataset,
    make_xy,
    normalize_input,
    split_train_test,
)
from asi_project.model import build_baseline_pipeline


def training_view(root, mode):
    """Keep the chronological holdout out of all downstream exploration."""
    config = load_project_config(root)
    contract = get_contract(config)
    data = load_dataset(config, contract, mode=mode)
    assert_valid(data, contract, config)
    data = filter_training_rows(normalize_input(data), contract)
    X, y = make_xy(data, contract, config.model_features)
    X_train, _, y_train, _ = split_train_test(X, y, contract, config.test_size, config.random_state)
    return config, contract, X_train, y_train


@dataclass
class Geometry:
    sample: pd.DataFrame
    positions: np.ndarray
    encoded_sample: np.ndarray
    scores: np.ndarray
    pca: PCA
    encoded_names: np.ndarray


def make_geometry(X, numeric, categorical, *, scaling="standard", max_rows=1500, seed=42):
    """Fit label-free preprocessing/PCA on training data, then sample uniformly."""
    if not numeric and not categorical:
        raise ValueError("Select at least one feature.")
    if scaling not in {"standard", "robust"}:
        raise ValueError("Unknown scaling method.")
    if max_rows < 3 or len(X) < 3:
        raise ValueError("At least three rows are required.")
    preprocessor = build_baseline_pipeline(numeric, categorical).named_steps["preprocessing"]
    if scaling == "robust":
        preprocessor.set_params(numeric__scaler=RobustScaler())
    encoded = preprocessor.fit_transform(X)
    encoded = encoded.toarray() if hasattr(encoded, "toarray") else np.asarray(encoded)
    if not np.isfinite(encoded).all() or encoded.shape[1] < 2:
        raise ValueError("The selected features must produce at least two finite dimensions.")
    pca = PCA(svd_solver="full").fit(encoded)
    if pca.explained_variance_.sum() <= 0:
        raise ValueError("The selected features have no variance.")
    positions = np.sort(
        np.random.default_rng(seed).choice(len(X), min(max_rows, len(X)), replace=False)
    )
    return Geometry(
        sample=X.iloc[positions].copy(),
        positions=positions,
        encoded_sample=encoded[positions],
        scores=pca.transform(encoded[positions]),
        pca=pca,
        encoded_names=preprocessor.get_feature_names_out(),
    )


def tsne_map(geometry, *, perplexity=30, seed=42, iterations=750):
    """Embed the same sampled rows; evaluate against original encoded distances."""
    if not 0 < perplexity < len(geometry.sample):
        raise ValueError("Perplexity must be positive and smaller than the sample size.")
    reduced = geometry.scores[:, : min(30, geometry.scores.shape[1])]
    model = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        max_iter=iterations,
        init="pca",
        learning_rate="auto",
    )
    coordinates = model.fit_transform(reduced)
    quality = trustworthiness(
        geometry.encoded_sample, coordinates, n_neighbors=min(10, (len(coordinates) - 1) // 2)
    )
    return coordinates, float(quality), float(model.kl_divergence_)


def segment_rates(values, target, minimum=1):
    """Group support, positive rates and bounded 95% Wilson intervals."""
    groups = pd.DataFrame({"segment": values.fillna("<missing>").astype(str), "y": target})
    summary = groups.groupby("segment", observed=True)["y"].agg(positive="sum", rows="count")
    summary = summary.loc[summary.rows >= minimum].copy()
    p, n, z = summary.positive / summary.rows, summary.rows, 1.959963984540054
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    summary["rate"] = p
    summary["low"] = np.minimum(p, np.maximum(0, centre - half))
    summary["high"] = np.maximum(p, np.minimum(1, centre + half))
    return summary.sort_values("rate").reset_index()
