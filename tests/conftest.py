"""Wspólne fixture'y testów.

Testy ogólne (rejestr datasetów, logika kontraktu, narzędzia) działają zawsze, także w szablonie.
Testy projektu zespołu (tests/test_project.py) są pomijane z jasnym komunikatem, dopóki zespół
nie wykona kolejnych kroków Sprintu 1. Postęp pokazuje: python scripts/check_dod.py
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from asi_project.config import DatasetContract, ProjectConfig, load_project_config, load_registry
from asi_project.data import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ProjectContext:
    config: ProjectConfig
    contract: DatasetContract
    sample: pd.DataFrame


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def registry() -> dict[str, DatasetContract]:
    return load_registry()


@pytest.fixture
def project_config() -> ProjectConfig:
    return load_project_config(REPO_ROOT)


@pytest.fixture
def project(project_config: ProjectConfig, registry: dict) -> ProjectContext:
    if not project_config.dataset_selected:
        pytest.skip("S1: dataset nie jest jeszcze wybrany w project.yaml.")
    contract = registry.get(project_config.dataset_id)
    if contract is None:
        pytest.fail(f"Nieznany dataset w project.yaml: {project_config.dataset_id}")
    if not project_config.ci_sample_path.is_file():
        pytest.skip("S1: brak próbki CI — uruchom python scripts/make_ci_sample.py.")
    sample = load_dataset(project_config, contract, mode="sample")
    return ProjectContext(project_config, contract, sample)


@pytest.fixture
def toy_contract() -> DatasetContract:
    """Syntetyczny kontrakt do testów logiki niezależnych od wybranego datasetu."""
    return DatasetContract(
        dataset_id="toy",
        title="Toy",
        source_url="https://example.org/toy",
        doi="https://doi.org/10.0000/toy",
        license="CC BY 4.0",
        download_url="https://example.org/toy.zip",
        download_member="toy.csv",
        read_csv_kwargs={},
        target="label",
        positive_class="yes",
        prediction_moment="Przed poznaniem wyniku.",
        split_strategy="stratified_random",
        allowed_features=("num", "cat"),
        excluded_features={"leak": "znane po zdarzeniu"},
    )


@pytest.fixture
def toy_frame() -> pd.DataFrame:
    rows = 40
    return pd.DataFrame(
        {
            "num": [float(i) for i in range(rows)],
            "cat": [" a", "b ", "", "c"] * (rows // 4),
            "leak": [i % 2 for i in range(rows)],
            "label": ["yes" if i % 4 == 0 else "no" for i in range(rows)],
        },
    )


def pytest_addoption(parser):
    parser.addoption(
        "--run-autogluon-package",
        action="store_true",
        default=False,
        help="Fit a tiny AutoGluon model and verify its portable package in a fresh process.",
    )


@pytest.fixture
def fitted_package(tmp_path):
    """Udostępnia dopasowany pipeline, metadane i surowe wiersze bez danych zewnętrznych."""
    import sys
    from importlib.metadata import version

    from asi_project.data import normalize_input
    from asi_project.metrics import compute_metrics
    from asi_project.model import build_baseline_pipeline
    from asi_project.provenance import data_fingerprint

    raw = pd.DataFrame(
        {
            "age": [20, 25, 30, 35, 40, 45, 50, 55],
            "pdays": [-1, 2, -1, 4, 5, -1, 7, 8],
            "job": [" admin. ", "technician", "admin.", None] * 2,
        },
        index=pd.Index([10, 20, 30, 40, 50, 60, 70, 80], name="source"),
    )
    target = pd.Series([0, 1] * 4, index=raw.index)
    X = normalize_input(raw)
    model = build_baseline_pipeline(["age", "pdays"], ["job"]).fit(X, target)
    probability = model.predict_proba(X)[:, 1]
    metadata = {
        "model_kind": "sklearn",
        "dataset_id": "toy",
        "numeric_features": ["age", "pdays"],
        "categorical_features": ["job"],
        "target": "y",
        "negative_class": "no",
        "positive_class": "yes",
        "decision_threshold": 0.6,
        "metrics": {"validation": compute_metrics(target, probability >= 0.6)},
        "git_sha": "a" * 40,
        "git_dirty": False,
        "data_fingerprint": data_fingerprint(raw),
        "versions": {
            "python": sys.version.split()[0],
            **{name: version(name) for name in ("numpy", "pandas", "scikit-learn")},
        },
    }
    return model, metadata, raw, tmp_path / "package"


@pytest.fixture
def saved_package(fitted_package):
    """Zapisuje wspólny mały model sklearn z fixture’a jako rzeczywisty pakiet."""
    from asi_project.model_package import save_model_package

    model, metadata, raw, package_dir = fitted_package
    return save_model_package(model, metadata, package_dir), raw
