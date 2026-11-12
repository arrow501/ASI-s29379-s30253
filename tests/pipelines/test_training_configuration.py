"""Błędne ustawienia treningu muszą być odrzucone przed uruchomieniem pipeline’u."""

import pytest
import yaml
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

from asi_project.config import ConfigError


@pytest.mark.parametrize(
    "override",
    [
        {"training": {"max_iter": 0}},
        {"training": {"max_iter": True}},
        {"training": {"max_iters": 1000}},
        {"reporting": {"data_mode": "ci"}},
        {"reporting": {"data_mdoe": "sample"}},
    ],
)
@pytest.mark.parametrize("source", ["runtime", "environment"])
def test_invalid_run_settings_fail_before_catalog_or_training(
    repo_root, tmp_path, override, source
):
    bootstrap_project(repo_root)
    kwargs = {"project_path": repo_root}
    if source == "runtime":
        kwargs["runtime_params"] = override
    else:
        (tmp_path / "base").mkdir()
        (tmp_path / "ci").mkdir()
        (tmp_path / "base/parameters.yml").write_text(
            (repo_root / "conf/base/parameters.yml").read_text(encoding="utf-8"), encoding="utf-8"
        )
        (tmp_path / "ci/parameters.yml").write_text(yaml.safe_dump(override), encoding="utf-8")
        kwargs.update(conf_source=str(tmp_path), env="ci")
    with pytest.raises(ConfigError, match=next(iter(override))):
        with KedroSession.create(**kwargs) as session:
            session.load_context()
