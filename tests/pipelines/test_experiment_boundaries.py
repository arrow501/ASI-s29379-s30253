"""Invalid probabilities, reserved names and graph ordering with independent expectations."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from asi_project.config import ConfigError
from asi_project.pipelines.experiments.pipeline import create_pipeline
from asi_project.pipelines.experiments.scoring import positive_probabilities, validation_threshold
from asi_project.pipelines.experiments.settings import validate_experiment_settings


@pytest.mark.parametrize(
    "probability",
    [
        np.array([0.2, 0.8]),
        np.array([[0.2], [0.8]]),
        np.array([[0.2, 0.9], [0.1, 0.9]]),
        np.array([[np.nan, 0.5], [0.1, 0.9]]),
        np.array([[-0.1, 1.1], [0.1, 0.9]]),
        np.array([[np.inf, 0.5], [0.1, 0.9]]),
        np.array([[0.2, 0.8]]),
    ],
)
def test_invalid_probabilities_are_rejected(probability):
    model = SimpleNamespace(classes_=np.array([0, 1]), predict_proba=lambda X: probability)
    with pytest.raises(ValueError):
        positive_probabilities(model, pd.DataFrame({"x": [1, 2]}))


@pytest.mark.parametrize("classes", [[0], [1, 2], [0, 0]])
def test_wrong_class_labels_are_rejected(classes):
    model = SimpleNamespace(
        classes_=np.array(classes), predict_proba=lambda X: np.array([[0.7, 0.3]])
    )
    with pytest.raises(ValueError, match="classes"):
        positive_probabilities(model, pd.DataFrame({"x": [1]}))


def test_reversed_class_order_is_handled_for_both_model_interfaces():
    X = pd.DataFrame({"x": [1, 2]}, index=[9, 3])
    probability = np.array([[0.2, 0.8], [0.9, 0.1]])
    for values in (probability, pd.DataFrame(probability, columns=[1, 0], index=X.index)):
        model = SimpleNamespace(
            classes_=np.array([1, 0]), predict_proba=lambda X, values=values: values
        )
        np.testing.assert_array_equal(positive_probabilities(model, X), [0.2, 0.9])


def test_dataframe_probability_order_cannot_silently_misalign_targets():
    X = pd.DataFrame({"x": [1, 2]}, index=[9, 3])
    probability = pd.DataFrame([[0.2, 0.8], [0.9, 0.1]], columns=[0, 1], index=[3, 9])
    with pytest.raises(ValueError, match="index"):
        positive_probabilities(SimpleNamespace(predict_proba=lambda X: probability), X)


@pytest.mark.parametrize(
    "target,probability",
    [
        ([], []),
        ([0, 1], [0.2]),
        ([0, 0], [0.2, 0.3]),
        ([0, 2], [0.2, 0.3]),
        ([0, 1], [0.2, np.nan]),
        ([0, 1], [-0.1, 0.9]),
        ([0, 1], [[0.2], [0.9]]),
    ],
)
def test_threshold_rejects_empty_misaligned_or_nonbinary_data(target, probability):
    with pytest.raises(ValueError):
        validation_threshold(target, probability)


@pytest.mark.parametrize("name", ["baseline", "dummy"])
def test_reserved_model_names_cannot_overwrite_references(name):
    with pytest.raises(ConfigError, match="reserved"):
        validate_experiment_settings(
            {
                "autogluon_enabled": True,
                "num_cpus": 2,
                "stopping_metric": "log_loss",
                "threshold_mode": "fixed",
                "candidates": [{"name": name, "time_limit": 10, "hyperparameters": {"GBM": {}}}],
            }
        )


@pytest.mark.parametrize(
    "protocol,prepare,train",
    [
        ("holdout", "split_holdout", "train_holdout_models"),
        ("bagged", "prepare_bagged_folds", "train_bagged_models"),
        ("chronological", "prepare_chronological_folds", "train_chronological_models"),
    ],
)
def test_graph_makes_selection_before_test_explicit(protocol, prepare, train):
    pipeline = create_pipeline(protocol, include_test=True)
    assert [n.name for n in pipeline.nodes] == [
        prepare,
        train,
        "freeze_validation_decision",
        "evaluate_frozen_test",
    ]
    training = pipeline.nodes[1]
    assert "prepared_data" not in training.inputs
    assert "raw_data" in training.inputs  # Fingerprint only; actual fit uses prepared partitions.
    test = pipeline.nodes[-1]
    assert "experiment_selection" in test.inputs and "experiment_report" not in test.inputs
    assert [n.name for n in create_pipeline(protocol).nodes][-1] == "freeze_validation_decision"


def test_removed_online_pipeline_is_not_registered():
    from asi_project.pipeline_registry import register_pipelines

    assert set(register_pipelines()) == {
        "data_processing",
        "data_science",
        "__default__",
        "experiments",
        "experiments_test",
        "experiments_bagged",
        "experiments_bagged_test",
        "experiments_crossfold",
        "experiments_crossfold_test",
    }
