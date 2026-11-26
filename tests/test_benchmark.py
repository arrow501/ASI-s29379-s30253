"""Measurements use real files and new interpreters; arithmetic has independent oracles."""

import copy
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from asi_project import benchmark
from asi_project.config_validation import project_parameters
from asi_project.data import normalize_input
from asi_project.pipelines.experiments.artifacts import read_json, write_json
from asi_project.pipelines.experiments.evaluation import evaluate_test, freeze_selection
from asi_project.pipelines.experiments.nodes import train_holdout_models
from asi_project.pipelines.experiments.splits import split_experiment_data


def test_inventory_counts_nested_bytes_and_detects_equal_size_edits(tmp_path):
    (tmp_path / "a").write_bytes(b"abc")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b").write_bytes(b"12345")
    first = benchmark.artifact_inventory(tmp_path)
    assert first["bytes"] == 8 and first["files"] == 2
    (tmp_path / "a").write_bytes(b"abd")
    assert benchmark.artifact_inventory(tmp_path)["sha256"] != first["sha256"]
    (tmp_path / "a").rename(tmp_path / "c")
    assert benchmark.artifact_inventory(tmp_path)["sha256"] != first["sha256"]


@pytest.mark.parametrize("existing", [False, True])
def test_empty_or_missing_inventory_is_rejected(tmp_path, existing):
    path = tmp_path if existing else tmp_path / "missing"
    with pytest.raises(ValueError, match="Empty or missing"):
        benchmark.artifact_inventory(path)


def test_symlink_inventory_is_rejected(tmp_path):
    (tmp_path / "a").write_bytes(b"abc")
    try:
        (tmp_path / "b").symlink_to(tmp_path / "a")
    except OSError:
        pytest.skip("OS does not allow symlink creation")
    with pytest.raises(ValueError, match="symbolic"):
        benchmark.artifact_inventory(tmp_path)


def test_timing_summary_uses_median_and_real_throughput():
    result = benchmark.timing_summary([0.4, 0.1, 0.2], 100)
    assert result == {
        "samples_seconds": [0.4, 0.1, 0.2],
        "median_seconds": 0.2,
        "min_seconds": 0.1,
        "max_seconds": 0.4,
        "rows": 100,
        "rows_per_second": 500,
    }


@pytest.mark.parametrize(
    "samples,rows",
    [([], 1), ([0], 1), ([-1], 1), ([float("nan")], 1), ([float("inf")], 1), ([1], 0)],
)
def test_invalid_timings_are_rejected(samples, rows):
    with pytest.raises(ValueError):
        benchmark.timing_summary(samples, rows)


@pytest.mark.parametrize(
    "threshold,expected",
    [(0.5, {"tn": 2, "fp": 0, "fn": 0, "tp": 2}), (1, {"tn": 2, "fp": 0, "fn": 2, "tp": 0})],
)
def test_worker_loads_real_model_in_new_process_and_applies_threshold(
    tmp_path, threshold, expected
):
    X = pd.DataFrame({"x": [-10.0, -5.0, 5.0, 10.0]})
    y = pd.Series([0, 0, 1, 1])
    model = LogisticRegression().fit(X, y)
    path = tmp_path / "model.pkl"
    path.write_bytes(pickle.dumps(model))
    result = benchmark.run_worker(
        {
            "model_path": str(path),
            "model_kind": "sklearn",
            "decision_threshold": threshold,
            "threads": 1,
            "warmup": 1,
            "repeats": 3,
        },
        X,
        y,
        timeout=30,
    )
    assert result["pid"] != os.getpid()
    assert result["metrics"]["confusion_matrix"] == expected
    assert result["metrics"]["f1"] == (1 if threshold == 0.5 else 0)
    assert len(result["batch"]["samples_seconds"]) == 3
    assert result["batch"]["rows"] == 4
    assert result["single_row"]["rows"] == 1
    assert (
        result["rss_loaded_delta_bytes"] == result["rss_loaded_bytes"] - result["rss_before_bytes"]
    )


def test_worker_failure_propagates_without_a_fake_measurement(tmp_path):
    path = tmp_path / "bad.pkl"
    path.write_bytes(b"not a pickle")
    with pytest.raises(RuntimeError, match="child failed"):
        benchmark.run_worker(
            {
                "model_path": str(path),
                "model_kind": "sklearn",
                "decision_threshold": 0.5,
                "threads": 1,
            },
            pd.DataFrame({"x": [1]}),
            pd.Series([1]),
            timeout=30,
        )


@pytest.fixture
def frozen_run(project, tmp_path):
    raw = project.sample.iloc[:100].copy()
    raw["y"] = np.where(np.arange(len(raw)) % 2, "yes", "no")
    prepared = normalize_input(raw)
    parameters = project_parameters(project.config, project.contract)
    settings = {
        "autogluon_enabled": False,
        "num_cpus": 1,
        "stopping_metric": "log_loss",
        "threshold_mode": "validation_f1",
        "candidates": [{"name": "gbm", "time_limit": 10, "hyperparameters": {"GBM": {}}}],
    }
    report = train_holdout_models(
        *split_experiment_data(prepared, parameters),
        raw,
        parameters,
        {"max_iter": 1000},
        settings,
        "sample",
        tmp_path,
    )
    selection = freeze_selection(report)
    evaluate_test(prepared, raw, parameters, selection)
    return Path(report["run_dir"]), raw


def test_end_to_end_real_reports_reload_and_mark_native_scope(frozen_run, tmp_path):
    root, raw = frozen_run
    report = benchmark.benchmark_runs([root], raw, warmup=1, repeats=2, threads=1)
    assert report["artifact_scope"] == "native_experiment"
    assert report["n_test"] == 20
    assert {row["candidate"] for row in report["results"]} == {"baseline", "dummy"}
    assert all(row["pid"] != os.getpid() for row in report["results"])
    # Real JSON roundtrip preserves validation, numbers and generated tables.
    path = tmp_path / "report.json"
    write_json(path, report)
    restored = read_json(path)
    benchmark.validate_report(restored)
    text = benchmark.render_tables(restored)
    assert "natywny artefakt" in text and "TN / FP / FN / TP" in text
    assert "| dummy | 0.0000 |" in text
    assert report["sources"][root.name]["wandb_url"] is None
    # All source links, configs and frozen thresholds survive the measurement.
    assert report["sources"][root.name]["provenance"] == read_json(root / "provenance.json")


@pytest.mark.parametrize(
    "change", ["raw", "missing_candidate", "threshold", "failed", "n_test", "outside_path"]
)
def test_source_changes_fail_before_starting_a_worker(frozen_run, monkeypatch, change):
    root, raw = frozen_run
    if change == "raw":
        raw = raw.iloc[::-1]
    elif change in {"missing_candidate", "threshold", "outside_path"}:
        path = root / "selection.json"
        selected = read_json(path)
        if change == "missing_candidate":
            selected["candidates"].pop("dummy")
        elif change == "threshold":
            selected["candidates"]["baseline"]["decision_threshold"] = float("inf")
        else:
            selected["candidates"]["baseline"]["model_path"] = str(root.parent / "other.pkl")
        path.write_text(json.dumps(selected))
    else:
        path = root / "test_metrics.json"
        tested = read_json(path)
        tested["status" if change == "failed" else "n_test"] = "failed" if change == "failed" else 1
        write_json(path, tested)
    monkeypatch.setattr(
        benchmark, "run_worker", lambda *a, **kw: pytest.fail("worker started before validation")
    )
    with pytest.raises(ValueError):
        benchmark.benchmark_runs([root], raw)


@pytest.fixture
def small_report():
    metrics = {
        "accuracy": 0.7,
        "precision": 0.75,
        "recall": 0.6,
        "f1": 2 / 3,
        "confusion_matrix": {"tn": 4, "fp": 1, "fn": 2, "tp": 3},
    }
    return {
        "schema_version": 1,
        "artifact_scope": "native_experiment",
        "n_test": 10,
        "method": {"repeats": 3},
        "sources": {
            "run": {
                "candidates": {"a": {}},
                "config": {"protocol": "holdout"},
                "test_metrics": {"a": {"test": copy.deepcopy(metrics)}},
            }
        },
        "results": [
            {
                "run_id": "run",
                "candidate": "a",
                "metrics": metrics,
                "artifact": {"bytes": 2**20, "files": 1},
                "load_seconds": 0.1,
                "rss_before_bytes": 2**20,
                "rss_loaded_bytes": 3 * 2**20,
                "rss_loaded_delta_bytes": 2 * 2**20,
                "rss_warm_bytes": 4 * 2**20,
                "rss_warm_delta_bytes": 3 * 2**20,
                "batch": benchmark.timing_summary([0.1, 0.2, 0.3], 10),
                "single_row": benchmark.timing_summary([0.01, 0.02, 0.03], 1),
            }
        ],
    }


def test_renderer_units_and_confusion_order(small_report):
    text = benchmark.render_tables(small_report)
    assert "| a | 200.00 | 20.00 | 0.100 | 1.000 | 1.0 | 3.0 / 2.0 | 4.0 / 3.0 |" in text
    assert "4 / 1 / 2 / 3" in text


@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "f1", "rss", "rows", "repeats", "throughput", "nan", "size"]
)
def test_report_completeness_and_independent_arithmetic(small_report, change):
    report = small_report
    row = report["results"][0]
    if change == "missing":
        report["results"] = []
    elif change == "duplicate":
        report["results"].append(copy.deepcopy(row))
    elif change == "f1":
        # The same mistaken value in reference and result must still fail.
        row["metrics"]["f1"] = report["sources"]["run"]["test_metrics"]["a"]["test"]["f1"] = 0.9
    elif change == "rss":
        row["rss_warm_delta_bytes"] = 0
    elif change == "rows":
        report["n_test"] = 20
    elif change == "repeats":
        report["method"]["repeats"] = 4
    elif change == "throughput":
        row["batch"]["rows_per_second"] = 999
    elif change == "nan":
        row["batch"]["samples_seconds"][0] = float("nan")
    else:
        row["artifact"]["bytes"] = 0
    with pytest.raises(ValueError):
        benchmark.validate_report(report)


def test_existing_output_is_preserved_before_loading_data(tmp_path):
    path = tmp_path / "result.json"
    path.write_text("keep")
    with pytest.raises(FileExistsError):
        benchmark.main(["--run", str(tmp_path), "--output", str(path)])
    assert path.read_text() == "keep"


@pytest.fixture
def packaged_run(frozen_run, tmp_path):
    from asi_project import model_package as package_api

    root, raw = frozen_run
    store = tmp_path / "packages"
    selection = read_json(root / "selection.json")
    for name, candidate in selection["candidates"].items():
        metadata = package_api.build_model_metadata(root, name, raw, include_test=True)
        model = package_api.load_native_model(candidate["model_path"], candidate["model_kind"])
        package_api.save_model_package(model, metadata, store / root.name / name)
    return root, raw, store


def test_package_measurement_uses_real_loader_and_raw_input_in_new_process(packaged_run):
    root, raw, store = packaged_run
    report = benchmark.benchmark_runs(
        [root], raw, packages_root=store, warmup=1, repeats=2, threads=1
    )
    assert report["artifact_scope"] == "model_package"
    assert report["n_test"] == 20
    assert {row["candidate"] for row in report["results"]} == {"baseline", "dummy"}
    for row in report["results"]:
        assert row["pid"] != os.getpid()
        assert row["artifact"] == benchmark.artifact_inventory(store / root.name / row["candidate"])
        assert (
            row["metrics"] == report["sources"][root.name]["test_metrics"][row["candidate"]]["test"]
        )
    text = benchmark.render_tables(report)
    assert "Porównanie pakietów" in text and "normalizację wejścia" in text
    assert "To nie są pomiary paczki" not in text


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_run_id", "another-run"),
        ("candidate_name", "dummy"),
        ("negative_class", "unexpected"),
        ("decision_threshold", 0.99),
        ("git_sha", "a" * 40),
        ("data_fingerprint", "a" * 64),
        ("metrics.test.f1", 0.123),
    ],
)
def test_mismatched_package_rejected_before_any_worker(packaged_run, monkeypatch, field, value):
    root, raw, store = packaged_run
    path = store / root.name / "baseline" / "metadata.json"
    metadata = read_json(path)
    destination = metadata
    parts = field.split(".")
    for part in parts[:-1]:
        destination = destination[part]
    destination[parts[-1]] = value
    write_json(path, metadata)
    monkeypatch.setattr(
        benchmark,
        "run_worker",
        lambda *a, **kw: pytest.fail("worker started before package validation"),
    )
    with pytest.raises(ValueError, match="Package"):
        benchmark.benchmark_runs([root], raw, packages_root=store)


def test_changed_package_probabilities_rejected_even_with_unchanged_metadata(packaged_run):
    root, raw, store = packaged_run
    path = store / root.name / "baseline" / "model" / "pipeline.pkl"
    model = pickle.loads(path.read_bytes())
    model.named_steps["classifier"].coef_ *= 0.5
    path.write_bytes(pickle.dumps(model))
    with pytest.raises(RuntimeError, match="Not equal to tolerance"):
        benchmark.benchmark_runs([root], raw, packages_root=store, warmup=1, repeats=1, threads=1)


def test_missing_package_rejected_before_worker(packaged_run, monkeypatch):
    root, raw, store = packaged_run
    (store / root.name / "dummy" / "metadata.json").unlink()
    monkeypatch.setattr(
        benchmark,
        "run_worker",
        lambda *a, **kw: pytest.fail("worker started before all packages were checked"),
    )
    with pytest.raises(FileNotFoundError):
        benchmark.benchmark_runs([root], raw, packages_root=store)


@pytest.mark.parametrize("operation", ["render", "benchmark"])
def test_cli_removes_partial_output_after_write_failure(
    tmp_path, monkeypatch, operation, small_report
):
    output = tmp_path / "report.txt"
    real_open = Path.open
    if operation == "render":
        source = tmp_path / "input.json"
        source.write_text(json.dumps(small_report), encoding="utf-8")
        args = ["--render", str(source), "--output", str(output)]
    else:
        # Isolate output I/O from model measurements; keep the real CLI/config loading.
        monkeypatch.setattr(benchmark, "benchmark_runs", lambda *a, **kw: small_report)
        args = ["--run", str(tmp_path), "--env", "ci", "--output", str(output)]

    class FailingWriter:
        def __enter__(self):
            self.stream = real_open(output, "x", encoding="utf-8")
            return self

        def write(self, content):
            self.stream.write(content[:4])
            self.stream.flush()
            assert output.read_text() == content[:4]
            raise OSError("disk full")

        def __exit__(self, *args):
            self.stream.close()

    def open_path(path, *args, **kwargs):
        if path == output and args and args[0] == "x":
            return FailingWriter()
        return real_open(path, *args, **kwargs)

    # Exercise each real CLI branch with bytes written before the injected failure.
    monkeypatch.setattr(Path, "open", open_path)
    with pytest.raises(OSError, match="disk full"):
        benchmark.main(args)
    assert not output.exists()
    monkeypatch.setattr(Path, "open", real_open)
    assert benchmark.main(args) == 0
    if operation == "render":
        assert output.read_text(encoding="utf-8") == benchmark.render_tables(small_report)
    else:
        assert json.loads(output.read_text(encoding="utf-8")) == small_report


def test_output_writer_does_not_remove_or_replace_existing_result(tmp_path):
    output = tmp_path / "report.json"
    output.write_bytes(b"existing result")
    with pytest.raises(FileExistsError):
        benchmark.write_new_output(output, "replacement")
    assert output.read_bytes() == b"existing result"
