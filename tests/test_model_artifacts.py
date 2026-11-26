"""Ponawianie pobierania pakietu po awarii zapisu potwierdzenia W&B."""

import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from asi_project import model_artifacts
from asi_project.model_package import load_model_package, predict_model_package
from asi_project.pipelines.experiments import artifacts


@pytest.mark.parametrize("existing_receipt", [False, True])
def test_download_can_be_retried_after_receipt_write_failure(
    saved_package, tmp_path, monkeypatch, existing_receipt
):
    source, raw = saved_package
    destination = tmp_path / "downloaded"
    receipt_path = tmp_path / "downloaded.wandb.json"
    original_receipt = b'{"artifact_ref": "team/project/model:v1"}\n'
    if existing_receipt:
        receipt_path.write_bytes(original_receipt)
    downloads = []

    def download(*, root):
        downloads.append(root)
        shutil.copytree(source, root)
        return root

    artifact = SimpleNamespace(
        entity="team", project="project", name="model:v7", version="v7", download=download
    )

    def get_artifact(ref, *, type):
        assert ref == "team/project/model:production" and type == "model"
        return artifact

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(Api=lambda **kwargs: SimpleNamespace(artifact=get_artifact)),
    )
    for key in ("WANDB_MODE", "WANDB_ENTITY", "WANDB_PROJECT"):
        monkeypatch.delenv(key, raising=False)
    settings = {"mode": "online", "entity": "team", "project": "project"}
    original_replace = artifacts.os.replace
    failed = False

    def fail_receipt_replace_once(source_path, target_path):
        nonlocal failed
        if Path(target_path) == receipt_path and not failed:
            failed = True
            assert destination.is_dir()
            raise OSError("receipt write failed")
        return original_replace(source_path, target_path)

    monkeypatch.setattr(artifacts.os, "replace", fail_receipt_replace_once)
    with pytest.raises(OSError, match="receipt write failed"):
        model_artifacts.download_model_package(
            "team/project/model:production", destination, settings
        )

    assert not destination.exists()
    assert not list(tmp_path.glob(".model-download-*"))
    assert not list(tmp_path.glob("downloaded.wandb.json.*.tmp"))
    if existing_receipt:
        assert receipt_path.read_bytes() == original_receipt
    else:
        assert not receipt_path.exists()
    assert source.is_dir()

    package_dir, receipt = model_artifacts.download_model_package(
        "team/project/model:production", destination, settings
    )

    assert package_dir == destination.resolve()
    assert receipt["artifact_ref"] == "team/project/model:v7"
    assert artifacts.read_json(receipt_path) == receipt
    model, metadata = load_model_package(package_dir)
    assert len(predict_model_package(model, metadata, raw)) == len(raw)
    assert len(downloads) == 2
    assert not list(tmp_path.glob(".model-download-*"))
