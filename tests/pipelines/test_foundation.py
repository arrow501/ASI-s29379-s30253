"""Check the configuration boundary used by both pipeline owners."""

import pytest
import yaml
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

from asi_project.config import ConfigError, get_contract, load_project_config
from asi_project.config_validation import project_parameters


@pytest.mark.parametrize("env", ["local", "ci"])
def test_kedro_loads_shared_parameters_in_both_environments(repo_root, env, monkeypatch):
    monkeypatch.delenv("ASI_PROJECT_ROOT", raising=False)
    monkeypatch.chdir(repo_root / "notebooks")
    bootstrap_project(repo_root)
    config = load_project_config(repo_root)
    contract = get_contract(config)
    with KedroSession.create(project_path=repo_root, env=env) as session:
        context = session.load_context()
        assert context.params["project"] == {
            "dataset_id": config.dataset_id,
            "numeric_features": list(config.numeric_features),
            "categorical_features": list(config.categorical_features),
            "test_size": config.test_size,
            "random_state": config.random_state,
            "target": contract.target,
            "positive_class": contract.positive_class,
            "split_strategy": contract.split_strategy,
        }


def test_parameters_follow_project_yaml_without_copying_values(repo_root, tmp_path, monkeypatch):
    bootstrap_project(repo_root)
    raw = yaml.safe_load((repo_root / "project.yaml").read_text())
    raw["split"]["test_size"] = 0.3
    raw["split"]["random_state"] = 17
    raw["features"]["numeric"] = ["age"]
    raw["features"]["categorical"] = ["job"]
    (tmp_path / "project.yaml").write_text(yaml.safe_dump(raw))
    monkeypatch.delenv("ASI_PROJECT_ROOT", raising=False)
    with KedroSession.create(project_path=tmp_path, conf_source=str(repo_root / "conf")) as session:
        params = session.load_context().params["project"]
    assert params["test_size"] == 0.3
    assert params["random_state"] == 17
    assert params["numeric_features"] == ["age"]
    assert params["categorical_features"] == ["job"]


def test_default_pipeline_connects_both_factories(monkeypatch):
    from kedro.pipeline import Pipeline, node

    from asi_project import pipeline_registry

    processing = Pipeline([node(lambda raw: raw, "raw_data", "prepared_data")])
    science = Pipeline([node(lambda prepared: prepared, "prepared_data", "trained_model")])
    monkeypatch.setattr(pipeline_registry, "data_processing", lambda: processing)
    monkeypatch.setattr(pipeline_registry, "data_science", lambda: science)
    registered = pipeline_registry.register_pipelines()
    assert registered["data_processing"] is processing
    assert registered["data_science"] is science
    default = registered["__default__"]
    assert default.inputs() == {"raw_data"}
    assert default.outputs() == {"trained_model"}
    assert len(default.nodes) == 2


@pytest.mark.parametrize("source", ["runtime", "environment"])
@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"numeric_features": ["duration"]}, "EXCLUDED"),
        ({"numeric_features": ["unknown_column"]}, "ALLOWED"),
        ({"numeric_features": ["y"]}, "Target"),
        ({"numeric_features": ["age", "age"]}, "więcej niż raz"),
        ({"numeric_features": ["age"], "categorical_features": ["age"]}, "both numeric"),
        ({"numeric_features": [], "categorical_features": []}, "pusta"),
        ({"numeric_features": "age"}, "numeric_features"),
        ({"categorical_features": [True]}, "categorical_features"),
        ({"split_strategy": "stratified_random"}, "split_strategy"),
        ({"target": "other"}, "target"),
        ({"positive_class": "no"}, "positive_class"),
        ({"dataset_id": "iranian_churn"}, "dataset_id"),
        ({"test_size": 0}, "test_size"),
        ({"test_size": 1}, "test_size"),
        ({"test_size": float("nan")}, "test_size"),
        ({"test_size": "0.2"}, "test_size"),
        ({"test_size": True}, "test_size"),
        ({"random_state": -1}, "random_state"),
        ({"random_state": 2**32}, "random_state"),
        ({"random_state": 42.9}, "random_state"),
        ({"random_state": True}, "random_state"),
    ],
)
def test_kedro_rejects_invalid_effective_settings(repo_root, tmp_path, source, change, message):
    bootstrap_project(repo_root)
    kwargs = {"project_path": repo_root, "env": "ci"}
    if source == "runtime":
        kwargs["runtime_params"] = {"project": change}
    else:
        (tmp_path / "base").mkdir()
        (tmp_path / "ci").mkdir()
        (tmp_path / "base" / "parameters.yml").write_text(
            (repo_root / "conf/base/parameters.yml").read_text()
        )
        (tmp_path / "ci" / "parameters.yml").write_text(yaml.safe_dump({"project": change}))
        kwargs["conf_source"] = str(tmp_path)
    with KedroSession.create(**kwargs) as session:
        with pytest.raises(ConfigError, match=message):
            _ = session.load_context().params


def test_kedro_accepts_valid_runtime_overrides(repo_root):
    bootstrap_project(repo_root)
    change = {"test_size": 0.3, "random_state": 0, "numeric_features": ["age"]}
    with KedroSession.create(project_path=repo_root, runtime_params={"project": change}) as session:
        params = session.load_context().params["project"]
        for key, value in change.items():
            assert params[key] == value
        assert params["split_strategy"] == "time_ordered"


def test_catalog_cannot_supply_invalid_parameters_to_nodes(repo_root):
    bootstrap_project(repo_root)
    with KedroSession.create(
        project_path=repo_root,
        runtime_params={"project": {"split_strategy": "stratified_random"}},
    ) as session:
        with pytest.raises(ConfigError, match="split_strategy"):
            _ = session.load_context().catalog


@pytest.mark.parametrize("env", ["local", "ci"])
@pytest.mark.parametrize("with_runtime_override", [False, True])
def test_partial_overrides_preserve_settings_and_follow_precedence(
    repo_root, tmp_path, env, with_runtime_override
):
    bootstrap_project(repo_root)
    config = load_project_config(repo_root)
    expected = project_parameters(config, get_contract(config))
    original_project_yaml = (repo_root / "project.yaml").read_bytes()
    (tmp_path / "base").mkdir()
    (tmp_path / env).mkdir()
    (tmp_path / "base" / "parameters.yml").write_text(
        (repo_root / "conf/base/parameters.yml").read_text()
    )
    # Override selected fields; lists replace defaults, other fields survive.
    (tmp_path / env / "parameters.yml").write_text(
        yaml.safe_dump(
            {
                "project": {
                    "test_size": 0.3,
                    "random_state": 17,
                    "categorical_features": ["job"],
                },
                "experiment": {"label": "override-test"},
            }
        )
    )
    expected.update(test_size=0.3, random_state=17, categorical_features=["job"])
    runtime_params = None
    if with_runtime_override:
        # CLI wins over the environment, while retaining its other overrides.
        runtime_params = {"project": {"test_size": 0.25}}
        expected["test_size"] = 0.25
    with KedroSession.create(
        project_path=repo_root,
        env=env,
        conf_source=str(tmp_path),
        runtime_params=runtime_params,
    ) as session:
        params = session.load_context().params
        assert params["project"] == expected
        assert params["experiment"] == {"label": "override-test"}
    assert (repo_root / "project.yaml").read_bytes() == original_project_yaml


@pytest.mark.parametrize("working_directory", ["outside", "other_checkout"])
def test_selected_project_wins_over_cwd_and_environment(
    repo_root, tmp_path, monkeypatch, working_directory
):
    bootstrap_project(repo_root)
    selected = tmp_path / "selected project Łódź"
    other = tmp_path / "other checkout"
    selected.mkdir()
    other.mkdir()
    original = yaml.safe_load((repo_root / "project.yaml").read_text(encoding="utf-8"))
    selected_raw = {**original, "split": {"test_size": 0.3, "random_state": 17}}
    other_raw = {**original, "split": {"test_size": 0.4, "random_state": 99}}
    for path, raw in ((selected, selected_raw), (other, other_raw)):
        (path / "project.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    monkeypatch.chdir(tmp_path if working_directory == "outside" else other)
    # An ambient setting must not redirect an explicitly selected Kedro project.
    monkeypatch.setenv("ASI_PROJECT_ROOT", str(other))
    with KedroSession.create(
        project_path=selected, conf_source=str(repo_root / "conf")
    ) as selected_session:
        selected_context = selected_session.load_context()
        with KedroSession.create(
            project_path=other, conf_source=str(repo_root / "conf")
        ) as other_session:
            other_context = other_session.load_context()
            assert selected_context.params["project"]["test_size"] == 0.3
            assert other_context.params["project"]["test_size"] == 0.4
            assert selected_context.params["project"]["random_state"] == 17
            assert other_context.params["project"]["random_state"] == 99
