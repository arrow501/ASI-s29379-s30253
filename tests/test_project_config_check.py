"""Walidator konfiguracji działa bez pobierania danych."""

import os
import subprocess
import sys

import pytest
import yaml


@pytest.mark.parametrize(
    ("change", "valid"),
    [
        ({}, True),
        ({"dataset": "missing"}, False),
        ({"features": {"numeric": [], "categorical": []}}, False),
        ({"features": {"numeric": ["duration"], "categorical": []}}, False),
        ({"features": {"numeric": ["age"], "categorical": ["age"]}}, False),
        ({"split": {"test_size": 1, "random_state": 42}}, False),
        ({"split": {"test_size": 0.2, "random_state": -1}}, False),
        ({"split": {"test_size": 0.2, "random_state": 42.9}}, False),
        ({"split": {"test_size": 0.2, "random_state": True}}, False),
        ({"split": {"test_size": "0.2", "random_state": 42}}, False),
        ({"features": {"numeric": {"age": False}, "categorical": []}}, False),
        ({"features": {"numeric": ["age"], "categorical": "job"}}, False),
        ({"features": {"numeric": [True], "categorical": []}}, False),
    ],
)
def test_config_check(tmp_path, repo_root, change, valid):
    config = yaml.safe_load((repo_root / "project.yaml").read_text())
    config.update(change)
    (tmp_path / "project.yaml").write_text(yaml.safe_dump(config))
    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts/check_project_config.py")],
        env={**os.environ, "ASI_PROJECT_ROOT": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == (0 if valid else 1), result.stdout + result.stderr
