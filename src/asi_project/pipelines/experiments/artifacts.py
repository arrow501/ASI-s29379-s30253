"""Atomic local reports and provenance shared by all experiment protocols."""

import json
import os
import platform
import tempfile
import uuid
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from asi_project.config_validation import validate_reporting_parameters
from asi_project.provenance import data_fingerprint, git_provenance


def write_json(path, payload):
    """Keep the previous valid report if serialization or writing fails."""
    path = Path(path)
    content = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def create_run(
    store,
    raw_data,
    project,
    training,
    experiments,
    data_mode,
    *,
    protocol,
    evaluation_set,
    protocols=None,
):
    validate_reporting_parameters({"data_mode": data_mode})
    root = Path(store) / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + protocol + "-" + uuid.uuid4().hex[:8]
    )
    root.mkdir(parents=True, exist_ok=False)
    report = {
        "run_dir": str(root),
        "status": "initializing",
        "protocol": protocol,
        "evaluation_set": evaluation_set,
        "test_evaluated": False,
        "results": {},
    }
    write_json(root / "metrics.json", report)
    try:
        config = {
            "project": project,
            "training": training,
            "experiments": experiments,
            "data_mode": data_mode,
            "protocol": protocol,
        }
        if protocols is not None:
            config["validation_protocols"] = protocols
        write_json(root / "config.json", config)
        packages = ["numpy", "pandas", "scikit-learn"]
        if experiments["autogluon_enabled"]:
            packages += ["autogluon.tabular", "lightgbm", "catboost"]
        write_json(
            root / "provenance.json",
            {
                **git_provenance(),
                "data_fingerprint": data_fingerprint(raw_data),
                "fingerprint_method": "loaded-data-csv-v1",
                "python": platform.python_version(),
                "versions": {p: version(p) for p in packages},
            },
        )
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(root / "metrics.json", report)
        raise
    report["status"] = "running"
    write_json(root / "metrics.json", report)
    return root, report
