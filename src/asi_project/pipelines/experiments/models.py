"""Trening modeli eksperymentalnych ze wspólnymi funkcjami serializacji."""

import numpy as np

from asi_project.data import DataContractError
from asi_project.model_package import load_native_model

from .scoring import positive_probabilities


def verify_reload(model, path, kind, X):
    restored = load_native_model(path, kind)
    if not np.allclose(positive_probabilities(model, X), positive_probabilities(restored, X)):
        raise ValueError("Native model reload changed predicted probabilities.")
    return restored


def fit_holdout_autogluon(
    X_train,
    X_validation,
    y_train,
    y_validation,
    project,
    candidate,
    path,
    cpus,
    stopping_metric="log_loss",
):
    # This optional import never runs in baseline-only CI.
    from autogluon.features.generators import AutoMLPipelineFeatureGenerator
    from autogluon.tabular import TabularPredictor

    # Nullable pandas Int64 (pdays) is not recognized by AG's NumPy dtype inference.
    numeric_types = {name: "float64" for name in project["numeric_features"]}
    X_train = X_train.astype(numeric_types)
    X_validation = X_validation.astype(numeric_types)

    # A pre-fitted generator prevents AutoGluon's default fit on train + tuning_data.
    generator = AutoMLPipelineFeatureGenerator(
        enable_text_ngram_features=False,
        enable_text_special_features=False,
        enable_raw_text_features=False,
        verbosity=0,
    )
    generator.fit_transform(X_train, y_train)
    predictor = TabularPredictor(
        label=project["target"],
        problem_type="binary",
        positive_class=1,
        # Inner model/ensemble selection must also avoid the flat F1-at-0.5 plateau.
        # The project still compares candidates using validation F1 after threshold selection.
        eval_metric=stopping_metric,
        path=str(path),
        verbosity=2,
        log_to_file=True,
        learner_kwargs={"random_state": project["random_state"]},
    )
    predictor.fit(
        train_data=X_train.assign(**{project["target"]: y_train}),
        tuning_data=X_validation.assign(**{project["target"]: y_validation}),
        feature_generator=generator,
        hyperparameters=candidate["hyperparameters"],
        time_limit=candidate["time_limit"],
        num_cpus=cpus,
        num_gpus=0,
        num_bag_folds=0,
        num_stack_levels=0,
        dynamic_stacking=False,
        calibrate_decision_threshold=False,
        refit_full=False,
        fit_strategy="sequential",
        raise_on_model_failure=True,
        ag_args_fit={"stopping_metric": stopping_metric},
    )
    return predictor


def fit_bagged_autogluon(X, y, project, candidate, path, experiments, n_folds):
    from autogluon.features.generators import AutoMLPipelineFeatureGenerator
    from autogluon.tabular import TabularPredictor

    X = X.astype({name: "float64" for name in project["numeric_features"]})
    predictor = TabularPredictor(
        label=project["target"],
        problem_type="binary",
        positive_class=1,
        eval_metric=experiments["stopping_metric"],
        path=str(path),
        verbosity=2,
        learner_kwargs={"random_state": project["random_state"]},
    )
    predictor.fit(
        X.assign(**{project["target"]: y}),
        feature_generator=AutoMLPipelineFeatureGenerator(
            enable_text_ngram_features=False,
            enable_text_special_features=False,
            enable_raw_text_features=False,
            verbosity=0,
        ),
        hyperparameters=candidate["hyperparameters"],
        time_limit=candidate["time_limit"],
        num_cpus=experiments["num_cpus"],
        num_gpus=0,
        num_bag_folds=n_folds,
        num_bag_sets=1,
        num_stack_levels=0,
        dynamic_stacking=False,
        refit_full=False,
        calibrate_decision_threshold=False,
        fit_strategy="sequential",
        raise_on_model_failure=True,
        ag_args_fit={"stopping_metric": experiments["stopping_metric"]},
        ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
    )
    oof = predictor.predict_proba_oof()
    if not oof.index.equals(y.index) or len(oof) != len(y):
        raise DataContractError("AutoGluon OOF rows must match development rows exactly.")
    return predictor, oof[1].to_numpy()
