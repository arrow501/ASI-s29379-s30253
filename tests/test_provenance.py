"""Real Git metadata and independently specified fingerprint bytes."""

import hashlib
import os
import subprocess

import pandas as pd
import pytest

from asi_project.provenance import data_fingerprint, git_provenance


def test_fingerprint_detects_value_order_target_headers_and_index():
    raw = pd.DataFrame({"x": [1, 2], "y": ["no", "yes"]})
    expected = hashlib.sha256(b",x,y\n0,1,no\n1,2,yes\n").hexdigest()
    assert data_fingerprint(raw) == expected
    for changed in (
        raw.iloc[::-1],
        raw.assign(y=["yes", "no"]),
        raw.assign(x=[3, 2]),
        raw.rename(columns={"x": "z"}),
        raw.set_axis([3, 4]),
    ):
        assert data_fingerprint(changed) != expected


@pytest.fixture(params=["default", "signing", "hooks"])
def external_git_settings(tmp_path, monkeypatch, request):
    """Simulate user settings without touching the real home or Git configuration."""
    config = tmp_path / "user.gitconfig"
    if request.param == "signing":
        config.write_text(
            '[commit]\n gpgsign = true\n[gpg]\n program = "'
            + (tmp_path / "absent-signer").as_posix()
            + '"\n',
            encoding="utf-8",
        )
    elif request.param == "hooks":
        hooks = tmp_path / "user-hooks"
        hooks.mkdir()
        hook = hooks / "pre-commit"
        hook.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
        hook.chmod(0o755)
        config.write_text('[core]\n hooksPath = "' + hooks.as_posix() + '"\n', encoding="utf-8")
    else:
        config.write_text("", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))


@pytest.fixture
def isolated_git_repository(tmp_path, monkeypatch, external_git_settings):
    # GIT_DIR/WORK_TREE/INDEX_FILE and config environment overrides must not redirect
    # the fixture into the caller's repository or reactivate their signing/hooks.
    for name in list(os.environ):
        if name.startswith("GIT_"):
            monkeypatch.delenv(name)
    config = tmp_path / "empty.gitconfig"
    config.write_text("", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    repository = tmp_path / "fixture-repository"
    repository.mkdir()
    return repository


def test_git_metadata_uses_real_commit_and_detects_tracked_and_untracked_changes(
    isolated_git_repository,
):
    repository = isolated_git_repository

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(repository), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init")
    path = repository / "file.txt"
    path.write_text("first")
    git("add", "file.txt")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture")
    sha = git("rev-parse", "HEAD")
    assert git_provenance(repository) == {"git_sha": sha, "git_dirty": False}
    path.write_text("changed")
    assert git_provenance(repository) == {"git_sha": sha, "git_dirty": True}
    path.write_text("first")
    assert git_provenance(repository) == {"git_sha": sha, "git_dirty": False}
    (repository / "untracked.txt").write_text("new")
    assert git_provenance(repository)["git_dirty"] is True


def test_default_provenance_does_not_follow_another_cwd_or_project_root(
    tmp_path, monkeypatch, repo_root
):
    sha = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ASI_PROJECT_ROOT", str(tmp_path))
    assert git_provenance()["git_sha"] == sha
