"""Smoke test w nowym interpreterze i kontrakt odpowiedzi bez dostępu do W&B."""

import importlib.util
import json
import os
import shutil

import numpy as np
import pytest

from asi_project.model_package import load_model_package, predict_model_package


@pytest.fixture
def command(repo_root):
    spec = importlib.util.spec_from_file_location(
        "smoke_model_command", repo_root / "scripts/smoke_model.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def smoke_input(saved_package, tmp_path):
    package, raw = saved_package
    path = tmp_path / "rows.csv"
    raw.assign(y="yes").to_csv(path, index=False)
    return package, raw, path


def test_local_smoke_reloads_relocated_package_in_new_process(
    command, smoke_input, tmp_path, monkeypatch
):
    package, raw, source = smoke_input
    model, metadata = load_model_package(package)
    expected = predict_model_package(model, metadata, raw.iloc[:5])
    moved = tmp_path / "relocated"
    assert moved.resolve().is_relative_to(tmp_path.resolve())
    package.rename(moved)
    monkeypatch.setenv("WANDB_MODE", "disabled")

    def no_download(*args, **kwargs):
        pytest.fail("Local CI smoke accessed W&B")

    monkeypatch.setattr(command, "download_model_package", no_download)
    output = tmp_path / "result.json"
    assert (
        command.main(["--package", str(moved), "--input", str(source), "--output", str(output)])
        == 0
    )
    report = json.loads(output.read_text())
    result = report["inference"]
    assert report["status"] == "passed" and report["mode"] == "local"
    assert "artifact_ref" not in report
    assert result["pid"] != os.getpid()
    assert result["rows"] == 5
    assert result["input_index"] == result["prediction_index"] == list(range(5))
    assert result["target_removed"] and result["training_disabled"]
    assert result["wandb_mode"] == "disabled"
    assert result["model"]["positive_class"] == "yes"
    assert result["model"]["decision_threshold"] == metadata["decision_threshold"]
    np.testing.assert_allclose(
        [row["probability_positive"] for row in result["predictions"]],
        expected.probability_positive,
    )
    assert [row["prediction"] for row in result["predictions"]] == expected.prediction.tolist()


@pytest.mark.parametrize(
    "change",
    ["count", "order", "schema", "class", "threshold", "nan", "inf", "range", "text"],
)
def test_prediction_contract_rejects_incorrect_results(command, fitted_package, change):
    model, metadata, raw, _ = fitted_package
    rows = raw.iloc[:5]
    result = predict_model_package(model, {**metadata, "model_files": ["pipeline.pkl"]}, rows)
    if change == "count":
        result = result.iloc[:-1]
    elif change == "order":
        result = result.iloc[::-1]
    elif change == "schema":
        result = result[["probability_positive", "prediction"]]
    elif change == "class":
        result.loc[result.index[0], "prediction"] = "unexpected"
    elif change == "threshold":
        result.loc[result.index[0], "prediction"] = (
            "no" if result.prediction.iloc[0] == "yes" else "yes"
        )
    elif change == "text":
        result["probability_positive"] = "0.5"
    else:
        result.loc[result.index[0], "probability_positive"] = {
            "nan": np.nan,
            "inf": np.inf,
            "range": 1.1,
        }[change]
    with pytest.raises(ValueError):
        command.validate_predictions(result, rows, metadata)


def test_prediction_contract_rejects_target_and_wrong_positive_class(command, fitted_package):
    model, metadata, raw, _ = fitted_package
    rows = raw.iloc[:5]
    result = predict_model_package(model, {**metadata, "model_files": ["pipeline.pkl"]}, rows)
    with pytest.raises(ValueError, match="target"):
        command.validate_predictions(result, rows.assign(y="yes"), metadata)
    with pytest.raises(ValueError, match="no/yes"):
        command.validate_predictions(result, rows, {**metadata, "positive_class": "no"})


@pytest.fixture
def downloaded_package(command, smoke_input, monkeypatch):
    package, _, source = smoke_input
    _, metadata = load_model_package(package)
    calls = []

    def download(reference, destination, settings):
        assert not destination.exists()
        assert settings == {"mode": "online", "entity": "team", "project": "project"}
        calls.append((reference, destination))
        shutil.copytree(package, destination)
        return destination, {
            "requested_ref": reference,
            "artifact_ref": "team/project/model:v7",
            "git_sha": metadata["git_sha"],
            "data_fingerprint": metadata["data_fingerprint"],
        }

    monkeypatch.setattr(command, "download_model_package", download)
    return source, calls


def test_download_path_uses_only_downloaded_package_and_records_concrete_version(
    command, downloaded_package, tmp_path
):
    source, calls = downloaded_package
    output = tmp_path / "online-result.json"
    assert (
        command.main(
            [
                "--artifact",
                "team/project/model:production",
                "--expected-ref",
                "team/project/model:v7",
                "--input",
                str(source),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(output.read_text())
    assert report["mode"] == "online"
    assert report["artifact_ref"] == "team/project/model:v7"
    assert report["requested_ref"] == "team/project/model:production"
    assert report["inference"]["rows"] == 5
    assert len(calls) == 1 and not calls[0][1].exists()


def test_download_failure_has_no_local_fallback_or_success_report(command, tmp_path, monkeypatch):
    def fail_download(*args, **kwargs):
        raise RuntimeError("W&B unavailable")

    def no_inference(*args, **kwargs):
        pytest.fail("Download failure reached prediction")

    monkeypatch.setattr(command, "download_model_package", fail_download)
    monkeypatch.setattr(command, "run_inference", no_inference)
    output = tmp_path / "failed.json"
    with pytest.raises(RuntimeError, match="W&B unavailable"):
        command.main(["--artifact", "team/project/model:production", "--output", str(output)])
    assert not output.exists()


def test_wrong_production_version_is_rejected_before_inference(
    command, downloaded_package, tmp_path, monkeypatch
):
    source, calls = downloaded_package

    def no_inference(*args, **kwargs):
        pytest.fail("A different production version reached prediction")

    monkeypatch.setattr(command, "run_inference", no_inference)
    output = tmp_path / "wrong-version.json"
    with pytest.raises(ValueError, match="different version"):
        command.main(
            [
                "--artifact",
                "team/project/model:production",
                "--expected-ref",
                "team/project/model:v8",
                "--input",
                str(source),
                "--output",
                str(output),
            ]
        )
    assert not output.exists() and not calls[0][1].exists()


def test_existing_report_is_preserved_before_download(command, tmp_path, monkeypatch):
    output = tmp_path / "keep.json"
    output.write_bytes(b"previous acceptance")

    def no_download(*args, **kwargs):
        pytest.fail("Existing report reached W&B")

    monkeypatch.setattr(command, "download_model_package", no_download)
    with pytest.raises(FileExistsError):
        command.main(["--artifact", "team/project/model:production", "--output", str(output)])
    assert output.read_bytes() == b"previous acceptance"


def test_short_input_fails_without_success_report(command, smoke_input, tmp_path):
    package, raw, source = smoke_input
    raw.iloc[:4].to_csv(source, index=False)
    output = tmp_path / "short.json"
    with pytest.raises(RuntimeError, match="exactly five"):
        command.main(["--package", str(package), "--input", str(source), "--output", str(output)])
    assert not output.exists()
