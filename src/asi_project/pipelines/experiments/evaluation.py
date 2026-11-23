"""Freeze validation decisions before evaluating the final chronological test."""

import copy
import json
import math
from pathlib import Path

from asi_project.config import load_registry
from asi_project.metrics import compute_metrics
from asi_project.pipelines.contracts import assert_prepared_data
from asi_project.pipelines.data_science.nodes import split_data
from asi_project.provenance import data_fingerprint

from .artifacts import read_json, write_json
from .models import load_native_model
from .scoring import positive_probabilities


def freeze_selection(report):
    if report["status"] != "completed":
        raise ValueError("Selection requires a completed experiment suite.")
    root = Path(report["run_dir"])
    if (root / "selection.json").exists():
        raise FileExistsError(root / "selection.json")
    if read_json(root / "metrics.json") != report:
        raise ValueError("In-memory results differ from the persisted report.")
    config = read_json(root / "config.json")
    provenance = read_json(root / "provenance.json")
    expected = {"baseline"}
    if config["protocol"] == "holdout":
        expected.add("dummy")
    if config["experiments"]["autogluon_enabled"]:
        expected.update(candidate["name"] for candidate in config["experiments"]["candidates"])
    if set(report["results"]) != expected:
        raise ValueError("Completed report does not contain all configured candidates.")
    candidates = {}
    for name, result in report["results"].items():
        score = result.get("selection_score", result["validation"]["f1"])
        threshold = result["decision_threshold"]
        for label, value in (("selection score", score), ("decision threshold", threshold)):
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"Invalid {label} for {name}.")
        kind = result["model_kind"]
        if kind not in {"sklearn", "autogluon"}:
            raise ValueError(f"Unknown native experiment model kind: {kind!r}")
        path = Path(result["model_path"]).resolve()
        if not path.is_relative_to(root.resolve()) or not path.exists():
            raise ValueError(f"Model artifact for {name} must exist inside the run directory.")
        if (kind == "sklearn" and not path.is_file()) or (
            kind == "autogluon" and not path.is_dir()
        ):
            raise ValueError(f"Model artifact for {name} does not match its native kind.")
        candidates[name] = {
            "model_path": str(path),
            "model_kind": kind,
            "decision_threshold": threshold,
            "threshold_source": result["threshold_source"],
            "selection_score": score,
        }
    eligible = {name for name in candidates if name != "dummy"}
    selected = min(eligible, key=lambda name: (-candidates[name]["selection_score"], name))
    selection = {
        "run_dir": str(root),
        "project": copy.deepcopy(config["project"]),
        "data_fingerprint": provenance["data_fingerprint"],
        "selected_candidate": selected,
        "criterion": report.get("selection_criterion", "validation F1")
        + "; ties resolved alphabetically",
        "selection_score": candidates[selected]["selection_score"],
        "validation_f1": report["results"][selected]["validation"]["f1"],
        "decision_threshold": candidates[selected]["decision_threshold"],
        "threshold_source": candidates[selected]["threshold_source"],
        "candidates": candidates,
        "test_used_for_selection": False,
    }
    content = json.dumps(selection, indent=2, allow_nan=False) + "\n"
    with (root / "selection.json").open("x", encoding="utf-8") as handle:
        handle.write(content)
    return selection


def evaluate_test(prepared_data, raw_data, project, selection):
    root = Path(selection["run_dir"])
    if read_json(root / "selection.json") != selection:
        raise ValueError("Selection differs from the frozen on-disk decision.")
    if selection["project"] != project or selection["data_fingerprint"] != data_fingerprint(
        raw_data
    ):
        raise ValueError("Test inputs must match the experiment configuration and fingerprint.")
    if (root / "test_metrics.json").exists():
        raise FileExistsError(root / "test_metrics.json")
    assert_prepared_data(raw_data, prepared_data, load_registry()[project["dataset_id"]])
    _, X_test, _, y_test = split_data(prepared_data, project)
    tested = {
        key: selection[key]
        for key in (
            "selected_candidate",
            "criterion",
            "selection_score",
            "decision_threshold",
            "threshold_source",
            "test_used_for_selection",
        )
    }
    tested.update(n_test=len(y_test), status="running", results={})
    try:
        for name, result in selection["candidates"].items():
            model = load_native_model(result["model_path"], result["model_kind"])
            probability = positive_probabilities(model, X_test)
            threshold = result["decision_threshold"]
            tested["results"][name] = {
                "test": compute_metrics(y_test, (probability >= threshold).astype(int)),
                "test_at_0_5": compute_metrics(y_test, (probability >= 0.5).astype(int)),
                "decision_threshold": threshold,
                "threshold_source": result["threshold_source"],
            }
            write_json(root / "test_metrics.json", tested)
        tested["status"] = "completed"
    except Exception as error:
        tested.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(root / "test_metrics.json", tested)
        report = read_json(root / "metrics.json")
        write_json(root / "metrics.json", {**report, "test_evaluated": bool(tested["results"])})
    return tested
