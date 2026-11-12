"""Component checks must validate predictions and real, isolated artifacts."""

import importlib.util
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from kedro.framework.session import KedroSession
from kedro.io import DataCatalog, DatasetError, MemoryDataset
from kedro.runner import SequentialRunner
from kedro_datasets.json import JSONDataset
from kedro_datasets.pickle import PickleDataset
from sklearn.pipeline import Pipeline as SklearnPipeline

from asi_project.config import ConfigError
from asi_project.config_validation import project_parameters


@pytest.fixture
def component_check(project, repo_root, tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "check_data_science", repo_root / "scripts/check_data_science.py"
    )
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    paths = {"trained_model": tmp_path / "model.pkl", "metrics": tmp_path / "metrics.json"}
    catalog = DataCatalog(
        {
            "trained_model": PickleDataset(filepath=str(paths["trained_model"])),
            "metrics": JSONDataset(filepath=str(paths["metrics"])),
        }
    )
    params = {
        "project": project_parameters(project.config, project.contract),
        "training": {"max_iter": 1000},
        "reporting": {"data_mode": "sample"},
    }
    for name, value in params.items():
        catalog[f"params:{name}"] = MemoryDataset(data=value)
    catalog["params:reporting.data_mode"] = MemoryDataset(data="sample")
    context = SimpleNamespace(catalog=catalog, params=params)

    @contextmanager
    def isolated_session(**kwargs):
        yield SimpleNamespace(load_context=lambda: context)

    monkeypatch.setattr(KedroSession, "create", isolated_session)
    monkeypatch.setattr("kedro.framework.startup.bootstrap_project", lambda root: None)
    monkeypatch.setattr(checker, "find_repo_root", lambda: repo_root)
    return SimpleNamespace(checker=checker, catalog=catalog, paths=paths, params=params)


@pytest.mark.parametrize(
    "scenario",
    [
        "persistent",
        "memory_metrics",
        "memory_model",
        "memory_both",
        "missing_metrics",
        "missing_model",
    ],
)
def test_component_check_requires_real_artifacts(component_check, monkeypatch, scenario):
    check = component_check
    memory_names = {
        "memory_metrics": ("metrics",),
        "memory_model": ("trained_model",),
        "memory_both": ("trained_model", "metrics"),
    }.get(scenario, ())
    for name in memory_names:
        check.catalog[name] = MemoryDataset()

    removed = []
    if scenario.startswith("missing_"):
        run = SequentialRunner.run
        missing = "metrics" if scenario == "missing_metrics" else "trained_model"

        def run_then_remove(self, *args, **kwargs):
            result = run(self, *args, **kwargs)
            check.paths[missing].unlink()
            removed.append(missing)
            return result

        monkeypatch.setattr(SequentialRunner, "run", run_then_remove)

    if memory_names:
        with pytest.raises(ConfigError, match="persist"):
            check.checker.main(["--data-mode", "sample"])
        assert not any(path.exists() for path in check.paths.values())
    elif scenario.startswith("missing_"):
        with pytest.raises(DatasetError):
            check.checker.main(["--data-mode", "sample"])
        assert removed == [missing]
    else:
        assert check.checker.main(["--data-mode", "sample"]) == 0
        assert all(path.is_file() for path in check.paths.values())


@pytest.mark.parametrize(
    "row,valid",
    [
        pytest.param([-0.2, 1.2], False, id="out_of_range"),
        pytest.param([np.nan, np.nan], False, id="nonfinite"),
        pytest.param([np.inf, -np.inf], False, id="infinite"),
        pytest.param([0.2, 0.3], False, id="wrong_sum"),
        pytest.param([0.2, 0.3, 0.5], False, id="wrong_shape"),
        pytest.param([0.0, 1.0], True, id="boundary_probabilities"),
    ],
)
def test_component_check_validates_reloaded_probabilities(component_check, monkeypatch, row, valid):
    def predict_proba(self, X):
        return np.tile(row, (len(X), 1))

    # Patch model behavior; validation and the rest of the real pipeline stay intact.
    monkeypatch.setattr(SklearnPipeline, "predict_proba", predict_proba)
    if valid:
        assert component_check.checker.main(["--data-mode", "sample"]) == 0
    else:
        with pytest.raises(ValueError, match="probabilities"):
            component_check.checker.main(["--data-mode", "sample"])


def test_component_check_rejects_mislabeled_data_before_writing(component_check):
    check = component_check
    check.params["reporting"]["data_mode"] = "full"
    check.catalog["params:reporting.data_mode"] = MemoryDataset(data="full")
    with pytest.raises(ConfigError, match="data_mode"):
        check.checker.main(["--data-mode", "sample"])
    assert not any(path.exists() for path in check.paths.values())


@pytest.mark.parametrize("output", ["metrics", "trained_model"])
def test_component_check_preserves_baseline(
    component_check, project, tmp_path, monkeypatch, output
):
    check = component_check
    baseline = tmp_path / "s1_metrics.json"
    original = project.config.metrics_path.read_bytes()
    baseline.write_bytes(original)
    dataset_type = JSONDataset if output == "metrics" else PickleDataset
    check.catalog[output] = dataset_type(filepath=str(baseline))
    check.params["reporting"]["data_mode"] = "full"
    check.catalog["params:reporting.data_mode"] = MemoryDataset(data="full")
    monkeypatch.setattr(
        check.checker,
        "load_project_config",
        lambda root: replace(project.config, metrics_path=baseline),
    )
    monkeypatch.setattr(
        check.checker, "load_dataset", lambda *args, **kwargs: project.sample.copy()
    )
    with pytest.raises(ConfigError, match="baseline"):
        check.checker.main(["--data-mode", "full", "--compare-baseline"])
    assert baseline.read_bytes() == original
    assert not any(path.exists() for path in check.paths.values())
