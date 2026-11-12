"""Anton's plug-in contract, tested without implementing his production pipeline."""

import importlib.util

import numpy as np
import pandas as pd
import pytest
from kedro.io import DataCatalog, MemoryDataset
from kedro.pipeline import Pipeline, node
from kedro.runner import SequentialRunner

from asi_project.config import ConfigError, load_registry
from asi_project.config_validation import project_parameters
from asi_project.data import DataContractError, filter_training_rows, normalize_input
from asi_project.pipelines.contracts import (
    assert_baseline_not_output,
    assert_pipeline_wiring,
    assert_prepared_data,
)
from asi_project.pipelines.data_science.pipeline import create_pipeline


def _fixture_preparation(raw, project):
    """Test stand-in only; production data_processing remains owned by S2-2."""
    return filter_training_rows(normalize_input(raw), load_registry()[project["dataset_id"]])


@pytest.mark.parametrize("alias", ["path", "file_uri", "relative", "hardlink"])
def test_output_guard_rejects_baseline_alias_before_writing(tmp_path, monkeypatch, alias):
    baseline = tmp_path / "metrics.json"
    baseline.write_bytes(b"original reference")
    filepath = str(baseline)
    if alias == "file_uri":
        filepath = baseline.as_uri()
    elif alias == "relative":
        monkeypatch.chdir(tmp_path)
        filepath = "metrics.json"
    elif alias == "hardlink":
        linked = tmp_path / "linked.json"
        try:
            linked.hardlink_to(baseline)
        except OSError:
            pytest.skip("Hard links are unavailable on this filesystem.")
        filepath = str(linked)
    with pytest.raises(ConfigError, match="baseline"):
        assert_baseline_not_output({"metrics": {"filepath": filepath}}, baseline)
    assert baseline.read_bytes() == b"original reference"


def _load_checker(repo_root):
    spec = importlib.util.spec_from_file_location(
        "check_pipeline_integration", repo_root / "scripts/check_pipeline_integration.py"
    )
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    return checker


@pytest.fixture
def boundary_data(project):
    raw = project.sample.copy()
    raw.index = pd.Index(np.arange(len(raw)) * 3 + 7)
    raw.loc[raw.index[0], "job"] = "  technician  "
    raw.loc[raw.index[1], project.contract.target] = None
    expected = filter_training_rows(normalize_input(raw), project.contract)
    return raw, expected


def test_boundary_accepts_s1_preparation_and_optional_excluded_columns(boundary_data, project):
    raw, expected = boundary_data
    assert_prepared_data(raw, expected, project.contract)
    assert_prepared_data(raw, expected.drop(columns=["duration"]), project.contract)


@pytest.mark.parametrize(
    "change", ["shuffle", "reset_index", "encode_target", "impute", "drop_row", "scale"]
)
def test_boundary_rejects_hidden_preprocessing_or_order_changes(boundary_data, project, change):
    raw, prepared = boundary_data
    prepared = prepared.copy()
    if change == "shuffle":
        prepared = prepared.iloc[::-1]
    elif change == "reset_index":
        prepared = prepared.reset_index(drop=True)
    elif change == "encode_target":
        prepared["y"] = prepared.y.eq("yes").astype(int)
    elif change == "impute":
        prepared["pdays"] = prepared.pdays.fillna(0)
    elif change == "drop_row":
        prepared = prepared.iloc[1:]
    else:
        prepared["age"] = prepared.age.astype(float) / 100
    with pytest.raises(DataContractError):
        assert_prepared_data(raw, prepared, project.contract)


def test_wiring_reports_pending_preparation_explicitly():
    science = create_pipeline()
    with pytest.raises(ConfigError, match="S2-2 pending"):
        assert_pipeline_wiring(
            {"data_processing": Pipeline([]), "data_science": science, "__default__": science}
        )


def test_wiring_rejects_a_misspelled_handoff():
    processing = Pipeline([node(lambda frame: frame, "raw_data", "prepared_frame", name="fixture")])
    science = create_pipeline()
    with pytest.raises(ConfigError, match="prepared_data"):
        assert_pipeline_wiring(
            {
                "data_processing": processing,
                "data_science": science,
                "__default__": processing + science,
            }
        )


def test_standin_preparation_plugs_into_real_science_factory(project, repo_root, tmp_path):
    processing = Pipeline(
        [
            node(
                _fixture_preparation,
                ["raw_data", "params:project"],
                "prepared_data",
                name="fixture_preparation",
            )
        ]
    )
    science = create_pipeline()
    pipelines = {
        "data_processing": processing,
        "data_science": science,
        "__default__": processing + science,
    }
    assert_pipeline_wiring(pipelines)
    params = project_parameters(project.config, project.contract)
    catalog = DataCatalog.from_config(
        {
            "trained_model": {
                "type": "pickle.PickleDataset",
                "filepath": str(tmp_path / "model.pkl"),
            },
            "metrics": {"type": "json.JSONDataset", "filepath": str(tmp_path / "metrics.json")},
        }
    )
    for name, value in {
        "raw_data": project.sample,
        "params:project": params,
        "params:training": {"max_iter": 1000},
        "params:reporting.data_mode": "sample",
    }.items():
        catalog[name] = MemoryDataset(data=value)
    SequentialRunner().run(pipelines["__default__"], catalog)
    assert catalog.load("metrics")["data_mode"] == "sample"
    assert hasattr(catalog.load("trained_model"), "predict_proba")


@pytest.mark.parametrize(
    "env,csv_separator,memory_outputs",
    [
        ("ci", ",", ()),
        ("ci", ";", ()),
        ("local", ";", ()),
        ("local", ",", ()),
        ("ci", ",", ("trained_model",)),
        ("ci", ",", ("metrics",)),
        ("ci", ",", ("trained_model", "metrics")),
        ("local", ";", ("trained_model",)),
    ],
)
def test_acceptance_command_runs_real_default_registry_with_fixture_preparation(
    project, repo_root, tmp_path, monkeypatch, env, csv_separator, memory_outputs
):
    """Exercise the acceptance harness; the stand-in is not shipped as production preparation."""
    import yaml

    from asi_project import pipeline_registry

    checker = _load_checker(repo_root)
    # bootstrap_project adds to sys.path; restore it when this temporary project is gone.
    monkeypatch.syspath_prepend(str(repo_root / "src"))
    processing = Pipeline(
        [
            node(
                _fixture_preparation,
                ["raw_data", "params:project"],
                "prepared_data",
                name="fixture_preparation",
            )
        ]
    )
    monkeypatch.setattr(pipeline_registry, "data_processing", lambda: processing)
    (tmp_path / "pyproject.toml").write_text(
        (repo_root / "pyproject.toml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "project.yaml").write_text(
        (repo_root / "project.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "conf/base").mkdir(parents=True)
    (tmp_path / "conf/ci").mkdir()
    (tmp_path / "conf/local").mkdir()
    input_path = (
        "data/01_raw/sample.csv" if env == "ci" else "data/01_raw/bank_marketing/bank-full.csv"
    )
    (tmp_path / input_path).parent.mkdir(parents=True)
    source_separator = "," if env == "ci" else ";"
    project.sample.to_csv(tmp_path / input_path, index=False, sep=source_separator)
    for source in ["base/parameters.yml", "ci/parameters.yml"]:
        (tmp_path / "conf" / source).write_text(
            (repo_root / "conf" / source).read_text(encoding="utf-8"), encoding="utf-8"
        )
    entries = {
        "raw_data": {
            "type": "pandas.CSVDataset",
            "filepath": input_path,
            "load_args": {"sep": csv_separator},
        },
        "trained_model": {"type": "pickle.PickleDataset", "filepath": "data/model.pkl"},
        "metrics": {"type": "json.JSONDataset", "filepath": "data/metrics.json"},
    }
    (tmp_path / "conf/base/catalog.yml").write_text(yaml.safe_dump(entries), encoding="utf-8")
    if memory_outputs:
        overrides = {name: {"type": "MemoryDataset"} for name in memory_outputs}
        (tmp_path / "conf" / env / "catalog_overrides.yml").write_text(
            yaml.safe_dump(overrides), encoding="utf-8"
        )
    monkeypatch.setattr(checker, "find_repo_root", lambda: tmp_path)
    if memory_outputs:
        with pytest.raises(ConfigError, match="persist"):
            checker.main(["--env", env])
        assert not (tmp_path / "data/model.pkl").exists()
        assert not (tmp_path / "data/metrics.json").exists()
    elif csv_separator == source_separator:
        assert checker.main(["--env", env]) == 0
        assert (tmp_path / "data/model.pkl").is_file()
        assert (tmp_path / "data/metrics.json").is_file()
    else:
        with pytest.raises(DataContractError, match="separator"):
            checker.main(["--env", env])
        assert not (tmp_path / "data/model.pkl").exists()
        assert not (tmp_path / "data/metrics.json").exists()


def test_acceptance_command_reports_pending_preparation(repo_root, monkeypatch, capsys):
    from asi_project import pipeline_registry

    checker = _load_checker(repo_root)
    monkeypatch.syspath_prepend(str(repo_root / "src"))
    monkeypatch.setattr(pipeline_registry, "data_processing", lambda: Pipeline([]))
    assert checker.main(["--env", "ci"]) == 2
    assert "S2-2 pending" in capsys.readouterr().err


@pytest.mark.parametrize("broken", ["incomplete_default", "extra_input"])
def test_wiring_rejects_incomplete_default_or_unconfigured_input(broken):
    science = create_pipeline()
    processing = Pipeline(
        [
            node(
                lambda raw, project: raw,
                ["raw_data", "params:project"],
                "prepared_data",
                name="fixture",
            )
        ]
    )
    default = processing + science
    expected = "both complete"
    if broken == "incomplete_default":
        default = processing + science.only_nodes("split_data")
    else:
        processing = Pipeline(
            [
                node(
                    lambda raw, other, project: raw,
                    ["raw_data", "unconfigured", "params:project"],
                    "prepared_data",
                    name="fixture",
                )
            ]
        )
        default = processing + science
        expected = "raw_data plus parameters"
    with pytest.raises(ConfigError, match=expected):
        assert_pipeline_wiring(
            {"data_processing": processing, "data_science": science, "__default__": default}
        )
