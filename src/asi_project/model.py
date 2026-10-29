"""Model preprocessing i modele bazowe (Sprint 1).

Model preprocessing (imputacja, skalowanie, kodowanie — wszystko, co wymaga ``fit``) żyje
wewnątrz sklearn Pipeline. Dzięki temu przy predykcji wykonują się dokładnie te same kroki,
które zostały dopasowane na X_train, a cały model można zapisać jako jeden artefakt.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def _categorical_missing_values(frame: pd.DataFrame) -> pd.DataFrame:
    """Ujednolica braki przed imputacją, także dla pandas StringDtype i category."""
    return frame.astype(object).where(frame.notna(), np.nan)


def make_dummy() -> DummyClassifier:
    """Baseline bez wiedzy o cechach: zawsze przewiduje najczęstszą klasę z X_train."""
    return DummyClassifier(strategy="most_frequent")


def build_baseline_pipeline(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> Pipeline:
    """Pipeline = model preprocessing (ColumnTransformer) + LogisticRegression.

    Wymagania (sprawdzane w tests/test_project.py):
    - Pipeline przyjmuje DataFrame z kolumnami ``numeric_features + categorical_features``,
    - cały model preprocessing jest wewnątrz Pipeline (bez wcześniejszego kodowania ręcznie),
    - braki danych i kategorie niewidziane w treningu nie przerywają predykcji,
    - ostatni krok ma ``predict_proba``,
    - wynik jest deterministyczny (dwa dopasowania na tych samych danych dają te same wyniki).

    Wzorzec znasz z drugich zajęć (notebook 02 onboardingu).
    """
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            (
                "missing_values",
                FunctionTransformer(_categorical_missing_values, feature_names_out="one-to-one"),
            ),
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric, list(numeric_features)),
            ("categorical", categorical, list(categorical_features)),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocessing", preprocessing),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
