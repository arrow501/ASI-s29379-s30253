"""Component checks must prove persistence without touching configured output paths."""

import importlib.util
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from kedro.framework.session import KedroSession
from kedro.io import DataCatalog, DatasetError, MemoryDataset
from kedro.runner import SequentialRunner

from asi_project.config import ConfigError
from asi_project.config_validation import project_parameters


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
def test_component_check_requires_real_artifacts(
    project, repo_root, tmp_path, monkeypatch, scenario
):
    spec = importlib.util.spec_from_file_location(
        "check_data_science", repo_root / "scripts/check_data_science.py"
    )
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    paths = {"trained_model": tmp_path / "model.pkl", "metrics": tmp_path / "metrics.json"}
    catalog = DataCatalog.from_config(
        {
            "trained_model": {
                "type": "pickle.PickleDataset",
                "filepath": str(paths["trained_model"]),
            },
            "metrics": {"type": "json.JSONDataset", "filepath": str(paths["metrics"])},
        }
    )
    memory_names = {
        "memory_metrics": ("metrics",),
        "memory_model": ("trained_model",),
        "memory_both": ("trained_model", "metrics"),
    }.get(scenario, ())
    for name in memory_names:
        catalog[name] = MemoryDataset()
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
    if scenario.startswith("missing_"):
        run = SequentialRunner.run
        missing = "metrics" if scenario == "missing_metrics" else "trained_model"

        def run_then_remove(self, *args, **kwargs):
            result = run(self, *args, **kwargs)
            paths[missing].unlink()
            return result

        monkeypatch.setattr(SequentialRunner, "run", run_then_remove)

    if memory_names:
        with pytest.raises(ConfigError, match="persist"):
            checker.main(["--data-mode", "sample"])
        assert not any(path.exists() for path in paths.values())
    elif scenario.startswith("missing_"):
        with pytest.raises(DatasetError):
            checker.main(["--data-mode", "sample"])
    else:
        assert checker.main(["--data-mode", "sample"]) == 0
        assert all(path.is_file() for path in paths.values())
