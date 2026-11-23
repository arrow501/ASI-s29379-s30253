"""Run the real preparation/training/selection/test graph with native local artifacts."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from kedro.io import DataCatalog, MemoryDataset
from kedro.runner import SequentialRunner

from asi_project.config_validation import project_parameters
from asi_project.pipeline_registry import register_pipelines


@pytest.mark.parametrize("name", ["experiments", "experiments_bagged", "experiments_crossfold"])
@pytest.mark.parametrize("include_test", [False, True])
def test_real_kedro_graph_persists_decision_and_optional_test_without_sdks(
    project, tmp_path, monkeypatch, name, include_test
):
    monkeypatch.setitem(sys.modules, "wandb", None)
    monkeypatch.setitem(sys.modules, "autogluon.tabular", None)
    raw = project.sample.iloc[:100].copy()
    raw["y"] = np.where(np.arange(100) % 2, "yes", "no")
    catalog = DataCatalog.from_config(
        {
            "experiment_store": {
                "type": "asi_project.experiment_store.ExperimentStore",
                "filepath": str(tmp_path),
            }
        }
    )
    for key, value in {
        "raw_data": raw,
        "params:project": project_parameters(project.config, project.contract),
        "params:training": {"max_iter": 1000},
        "params:experiments": {
            "autogluon_enabled": False,
            "num_cpus": 2,
            "stopping_metric": "log_loss",
            "threshold_mode": "validation_f1",
            "candidates": [{"name": "gbm", "time_limit": 10, "hyperparameters": {"GBM": {}}}],
        },
        "params:validation_protocols": {"bagging_folds": 5, "chronological_folds": 3},
        "params:reporting.data_mode": "sample",
    }.items():
        catalog[key] = MemoryDataset(data=value)
    pipeline = register_pipelines()[name + ("_test" if include_test else "")]
    outputs = SequentialRunner().run(pipeline, catalog)
    root = next(tmp_path.iterdir())
    report = json.loads((root / "metrics.json").read_text())
    selection = json.loads((root / "selection.json").read_text())
    assert report["status"] == "completed" and report["test_evaluated"] is include_test
    assert selection["test_used_for_selection"] is False
    assert selection["selected_candidate"] == "baseline"
    expected = {"baseline", "dummy"} if name == "experiments" else {"baseline"}
    assert set(selection["candidates"]) == expected
    assert all(
        Path(candidate["model_path"]).exists() for candidate in selection["candidates"].values()
    )
    if include_test:
        tested = json.loads((root / "test_metrics.json").read_text())
        assert tested["status"] == "completed" and tested["n_test"] == 20
        assert set(tested["results"]) == expected
        for candidate in expected:
            assert (
                tested["results"][candidate]["decision_threshold"]
                == selection["candidates"][candidate]["decision_threshold"]
            )
        assert "experiment_test_report" in outputs
    else:
        assert not (root / "test_metrics.json").exists()
        assert "experiment_selection" in outputs
