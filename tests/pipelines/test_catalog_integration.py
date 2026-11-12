"""Run both real pipelines with tracked catalogs in a disposable project root."""

import importlib.util
import shutil

import pytest


@pytest.mark.parametrize("env", ["local", "ci"])
def test_tracked_catalog_connects_csv_preparation_training_and_artifacts(
    project, repo_root, tmp_path, monkeypatch, env
):
    # Copy shared configuration environments; never inherit conf/local overrides.
    for name in ("base", "ci"):
        shutil.copytree(repo_root / "conf" / name, tmp_path / "conf" / name)
    (tmp_path / "conf/local").mkdir()
    for name in ("project.yaml", "pyproject.toml"):
        shutil.copy2(repo_root / name, tmp_path / name)
    (tmp_path / "src").mkdir()

    # Different contents and separators detect selecting the wrong source/environment.
    sample = tmp_path / "data/01_raw/sample.csv"
    full = tmp_path / "data/01_raw/bank_marketing/bank-full.csv"
    full.parent.mkdir(parents=True)
    project.sample.to_csv(sample, index=False)
    project.sample.assign(balance=project.sample["balance"] + 13).to_csv(full, sep=";", index=False)

    spec = importlib.util.spec_from_file_location(
        "check_real_catalog", repo_root / "scripts/check_pipeline_integration.py"
    )
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    monkeypatch.setattr(checker, "find_repo_root", lambda: tmp_path)
    # bootstrap_project changes sys.path; restore it after the disposable project is gone.
    monkeypatch.syspath_prepend(str(repo_root / "src"))

    # Guard the sandbox before invoking a checker that writes configured artifacts.
    from kedro.framework.session import KedroSession
    from kedro.framework.startup import bootstrap_project

    bootstrap_project(tmp_path)
    suffix = "ci/" if env == "ci" else ""
    expected = {
        "raw_data": "data/01_raw/sample.csv"
        if env == "ci"
        else "data/01_raw/bank_marketing/bank-full.csv",
        "trained_model": f"data/06_models/{suffix}baseline.pkl",
        "metrics": f"data/08_reporting/{suffix}metrics.json",
    }
    with KedroSession.create(project_path=tmp_path, env=env) as session:
        context = session.load_context()
        entries = context.config_loader["catalog"]
        for name, path in expected.items():
            assert entries[name]["filepath"] == path
        assert entries["raw_data"]["load_args"]["sep"] == ("," if env == "ci" else ";")
        resolved, _, _, _ = context.catalog.to_config()
        for name, path in expected.items():
            assert resolved[name]["filepath"] == (tmp_path / path).as_posix()

    assert checker.main(["--env", env]) == 0
    assert (tmp_path / expected["trained_model"]).is_file()
    assert (tmp_path / expected["metrics"]).is_file()
    other = "" if env == "ci" else "ci/"
    assert not (tmp_path / f"data/06_models/{other}baseline.pkl").exists()
    assert not (tmp_path / f"data/08_reporting/{other}metrics.json").exists()
