"""Fresh-process measurements of frozen local experiment artifacts (Sprint 3)."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import pickle
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from asi_project.metrics import compute_metrics
from asi_project.pipelines.experiments.artifacts import read_json
from asi_project.provenance import data_fingerprint, git_provenance


def artifact_inventory(path):
    """Logical bytes and content digest of all native files, including dependencies."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Benchmark artifacts must not contain symbolic links.")
    files = [path] if path.is_file() else sorted(path.rglob("*"))
    digest = hashlib.sha256()
    size = count = 0
    for file in files:
        if file.is_symlink():
            raise ValueError("Benchmark artifacts must not contain symbolic links.")
        if not file.is_file():
            continue
        name = file.name if path.is_file() else file.relative_to(path).as_posix()
        digest.update(name.encode() + b"\0")
        content_hash = hashlib.sha256()
        with file.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                size += len(chunk)
                content_hash.update(chunk)
        digest.update(content_hash.digest())
        count += 1
    if count == 0 or size == 0:
        raise ValueError(f"Empty or missing native artifact: {path}")
    return {"bytes": size, "files": count, "sha256": digest.hexdigest()}


def timing_summary(samples, rows):
    if not samples or rows < 1 or any(not math.isfinite(x) or x <= 0 for x in samples):
        raise ValueError("Timings must be finite positive durations for nonempty input.")
    median = statistics.median(samples)
    return {
        "samples_seconds": samples,
        "median_seconds": median,
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "rows": rows,
        "rows_per_second": rows / median,
    }


def measure_native(request, X, y):
    """Called only in a child; no training, threshold selection or package creation."""
    import psutil
    from threadpoolctl import threadpool_limits

    from asi_project.pipelines.experiments.models import load_native_model
    from asi_project.pipelines.experiments.scoring import positive_probabilities

    process = psutil.Process()
    rss_before = process.memory_info().rss
    start = time.perf_counter()
    model = load_native_model(request["model_path"], request["model_kind"])
    load_seconds = time.perf_counter() - start
    rss_loaded = process.memory_info().rss

    def predict(frame):
        return (positive_probabilities(model, frame) >= request["decision_threshold"]).astype(int)

    # Environment limits apply before backend import; threadpoolctl also covers loaded BLAS/OpenMP.
    # Estimator-specific parallelism stays as saved in the artifact.
    with threadpool_limits(limits=request["threads"]):
        for _ in range(request["warmup"]):
            prediction = predict(X)
        rss_warm = process.memory_info().rss
        samples = []
        for _ in range(request["repeats"]):
            start = time.perf_counter()
            prediction = predict(X)
            samples.append(time.perf_counter() - start)
        single = X.iloc[:1]
        for _ in range(request["warmup"]):
            predict(single)
        single_samples = []
        for _ in range(request["repeats"]):
            start = time.perf_counter()
            predict(single)
            single_samples.append(time.perf_counter() - start)
    return {
        "pid": os.getpid(),
        "metrics": compute_metrics(y, prediction),
        "load_seconds": load_seconds,
        "rss_before_bytes": rss_before,
        "rss_loaded_bytes": rss_loaded,
        "rss_loaded_delta_bytes": rss_loaded - rss_before,
        "rss_warm_bytes": rss_warm,
        "rss_warm_delta_bytes": rss_warm - rss_before,
        "batch": timing_summary(samples, len(X)),
        "single_row": timing_summary(single_samples, 1),
    }


def run_worker(request, X, y, *, timeout):
    """Exec a fresh interpreter, never fork a process containing trained estimators."""
    with tempfile.TemporaryDirectory(prefix="asi-benchmark-") as directory:
        root = Path(directory)
        (root / "request.json").write_text(json.dumps(request), encoding="utf-8")
        (root / "input.pkl").write_bytes(pickle.dumps((X, y)))
        env = dict(os.environ)
        for key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            env[key] = str(request["threads"])
        env["WANDB_MODE"] = "disabled"
        env["KEDRO_DISABLE_TELEMETRY"] = "1"
        completed = subprocess.run(
            [sys.executable, "-m", "asi_project.benchmark", "--worker", str(root)],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if completed.returncode:
            raise RuntimeError(f"Benchmark child failed:\n{completed.stderr[-6000:]}")
        return read_json(root / "response.json")


def experiment_inputs(run, raw):
    """Check the frozen source and reconstruct the exact common final test."""
    from asi_project.config import load_registry
    from asi_project.data import filter_training_rows, normalize_input
    from asi_project.pipelines.data_science.nodes import split_data

    config = read_json(run / "config.json")
    provenance = read_json(run / "provenance.json")
    selection = read_json(run / "selection.json")
    tested = read_json(run / "test_metrics.json")
    expected = {"baseline"}
    if config["protocol"] == "holdout":
        expected.add("dummy")
    if config["experiments"]["autogluon_enabled"]:
        expected.update(item["name"] for item in config["experiments"]["candidates"])
    if (
        set(selection["candidates"]) != expected
        or tested["status"] != "completed"
        or selection["test_used_for_selection"] is not False
        or config["project"] != selection["project"]
        or provenance["data_fingerprint"] != selection["data_fingerprint"]
        or data_fingerprint(raw) != selection["data_fingerprint"]
        or set(tested["results"]) != set(selection["candidates"])
    ):
        raise ValueError("Incomplete experiment or inconsistent frozen test/provenance.")
    contract = load_registry()[config["project"]["dataset_id"]]
    prepared = filter_training_rows(normalize_input(raw), contract)
    _, X, _, y = split_data(prepared, config["project"])
    if tested["n_test"] != len(y):
        raise ValueError("Frozen test row count does not match reconstructed input.")
    for name, candidate in selection["candidates"].items():
        threshold = candidate["decision_threshold"]
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, int | float)
            or not math.isfinite(threshold)
            or not 0 <= threshold <= 1
            or tested["results"][name]["decision_threshold"] != threshold
        ):
            raise ValueError("Invalid or changed frozen threshold.")
        path = Path(candidate["model_path"]).resolve()
        if not path.is_relative_to(run.resolve()):
            raise ValueError("Native artifact must be inside its source run directory.")
    return config, provenance, selection, tested, X, y


def validate_report(report):
    """Reject partial rows and inconsistent arithmetic before persisting a comparison."""
    if report["schema_version"] != 1 or report["artifact_scope"] != "native_experiment":
        raise ValueError("Unsupported benchmark schema or scope.")
    expected = {
        (run, name) for run, source in report["sources"].items() for name in source["candidates"]
    }
    actual = [(row["run_id"], row["candidate"]) for row in report["results"]]
    if len(actual) != len(set(actual)) or set(actual) != expected or not actual:
        raise ValueError("Benchmark must contain exactly all source candidates.")
    for row in report["results"]:
        source = report["sources"][row["run_id"]]
        if row["metrics"] != source["test_metrics"][row["candidate"]]["test"]:
            raise ValueError("Reloaded predictions differ from the frozen final test.")
        cm = row["metrics"]["confusion_matrix"]
        if (
            any(type(v) is not int or v < 0 for v in cm.values())
            or sum(cm.values()) != report["n_test"]
        ):
            raise ValueError("Confusion matrix does not cover the common test.")
        tn, fp, fn, tp = (cm[key] for key in ("tn", "fp", "fn", "tp"))
        ratios = {
            "accuracy": (tn + tp, tn + fp + fn + tp),
            "precision": (tp, tp + fp),
            "recall": (tp, tp + fn),
            "f1": (2 * tp, 2 * tp + fp + fn),
        }
        for metric, (numerator, denominator) in ratios.items():
            expected_metric = numerator / denominator if denominator else 0.0
            if not math.isclose(row["metrics"][metric], expected_metric, abs_tol=1e-12):
                raise ValueError("Metrics disagree with confusion-matrix arithmetic.")
        if row["artifact"]["bytes"] <= 0 or row["artifact"]["files"] <= 0:
            raise ValueError("Missing artifact size.")
        if not math.isfinite(row["load_seconds"]) or row["load_seconds"] <= 0:
            raise ValueError("Invalid loading measurement.")
        for stage in ("loaded", "warm"):
            before = row["rss_before_bytes"]
            after = row[f"rss_{stage}_bytes"]
            if before <= 0 or after <= 0 or row[f"rss_{stage}_delta_bytes"] != after - before:
                raise ValueError("Invalid RSS arithmetic.")
        for key, rows in (("batch", report["n_test"]), ("single_row", 1)):
            timing = row[key]
            if len(timing["samples_seconds"]) != report["method"][
                "repeats"
            ] or timing != timing_summary(timing["samples_seconds"], rows):
                raise ValueError("Invalid timing summary or incomplete repetitions.")


def installed_versions():
    versions = {}
    for name in (
        "numpy",
        "pandas",
        "scikit-learn",
        "psutil",
        "threadpoolctl",
        "autogluon.tabular",
        "lightgbm",
        "catboost",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def cpu_description():
    if platform.processor():
        return platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.machine()


def render_tables(report):
    """Render measured values only; MiB = 2**20 bytes, timings are warm medians."""
    validate_report(report)
    lines = [
        "# Porównanie natywnych modeli — Sprint 3",
        "",
        "Rozmiar obejmuje natywny artefakt, a RSS proces wraz z bibliotekami. "
        "To nie są pomiary paczki produkcyjnej z #20.",
        "",
    ]
    for run, source in report["sources"].items():
        lines.extend(
            [
                f"## {source['config']['protocol']} — `{run}`",
                "",
                "| Model | F1 | Precision | Recall | Accuracy | TN / FP / FN / TP |",
                "| --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        rows = [row for row in report["results"] if row["run_id"] == run]
        for row in rows:
            m = row["metrics"]
            counts = " / ".join(str(m["confusion_matrix"][k]) for k in ("tn", "fp", "fn", "tp"))
            lines.append(
                f"| {row['candidate']} | {m['f1']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['accuracy']:.4f} | {counts} |"
            )
        lines.extend(
            [
                "",
                "| Model | Batch [ms] | 1 rekord [ms] | Load [s] | Artefakt [MiB] | RSS przed [MiB] | RSS load / Δ [MiB] | RSS warm / Δ [MiB] |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in rows:
            mib = 2**20
            lines.append(
                f"| {row['candidate']} | {row['batch']['median_seconds'] * 1000:.2f} | {row['single_row']['median_seconds'] * 1000:.2f} | {row['load_seconds']:.3f} | {row['artifact']['bytes'] / mib:.3f} | {row['rss_before_bytes'] / mib:.1f} | {row['rss_loaded_bytes'] / mib:.1f} / {row['rss_loaded_delta_bytes'] / mib:.1f} | {row['rss_warm_bytes'] / mib:.1f} / {row['rss_warm_delta_bytes'] / mib:.1f} |"
            )
        lines.append("")
    return "\n".join(lines)


def benchmark_runs(runs, raw, *, warmup=2, repeats=5, threads=8, timeout=600):
    if any(type(v) is not int or v < 1 for v in (warmup, repeats, threads, timeout)):
        raise ValueError("Warmup, repeats, threads and timeout must be positive integers.")
    runs = [Path(run).resolve() for run in runs]
    if not runs or len({run.name for run in runs}) != len(runs):
        raise ValueError("Supply nonempty, uniquely named source runs.")
    import psutil

    report = {
        "schema_version": 1,
        "artifact_scope": "native_experiment",
        "created_utc": datetime.now(UTC).isoformat(),
        "benchmark_code": {
            **git_provenance(),
            "module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": cpu_description(),
            "logical_cpus": os.cpu_count(),
            "ram_bytes": psutil.virtual_memory().total,
        },
        "versions": installed_versions(),
        "python": platform.python_version(),
        "method": {
            "warmup": warmup,
            "repeats": repeats,
            "threads": threads,
            "timeout_seconds": timeout,
        },
        "sources": {},
        "results": [],
    }
    common = None
    # Validate every source before starting measurements; reject mixed split/config/data.
    inputs = []
    for run in runs:
        config, provenance, selection, tested, X, y = experiment_inputs(run, raw)
        signature = (config["project"], data_fingerprint(X), data_fingerprint(y.to_frame()))
        if common is not None and signature != common:
            raise ValueError("All runs must use the same project and exact ordered test rows.")
        common = signature
        inputs.append((run, selection, X, y))
        report["sources"][run.name] = {
            "config": config,
            "provenance": provenance,
            "selection_sha256": hashlib.sha256((run / "selection.json").read_bytes()).hexdigest(),
            "selected_candidate": selection["selected_candidate"],
            "candidates": {
                name: {k: v for k, v in candidate.items() if k != "model_path"}
                for name, candidate in selection["candidates"].items()
            },
            "test_metrics": tested["results"],
            "wandb_url": None,
        }
    report["n_test"] = len(inputs[0][3])
    report["test_features_fingerprint"] = common[1]
    report["test_target_fingerprint"] = common[2]
    for run, selection, X, y in inputs:
        for name, candidate in selection["candidates"].items():
            print(f"Benchmark: {run.name} / {name}", flush=True, file=sys.stderr)
            artifact = artifact_inventory(candidate["model_path"])
            measurement = run_worker(
                {**candidate, "warmup": warmup, "repeats": repeats, "threads": threads},
                X,
                y,
                timeout=timeout,
            )
            if artifact != artifact_inventory(candidate["model_path"]):
                raise ValueError("Artifact changed during benchmark.")
            report["results"].append(
                {
                    "run_id": run.name,
                    "candidate": name,
                    "protocol": report["sources"][run.name]["config"]["protocol"],
                    "artifact": artifact,
                    **measurement,
                }
            )
    validate_report(report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--render", type=Path, help="Render an existing JSON report as Markdown")
    parser.add_argument("--run", action="append", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--env", default="local")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args(argv)
    if args.worker:
        request = read_json(args.worker / "request.json")
        X, y = pickle.loads((args.worker / "input.pkl").read_bytes())
        result = measure_native(request, X, y)
        (args.worker / "response.json").write_text(
            json.dumps(result, allow_nan=False), encoding="utf-8"
        )
        return 0
    if args.render:
        if args.output is None:
            parser.error("--render requires --output")
        content = render_tables(read_json(args.render))
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(content)
        return 0
    if not args.run or args.output is None:
        parser.error("--run and --output are required")
    if args.output.exists():
        raise FileExistsError(args.output)
    from kedro.framework.session import KedroSession
    from kedro.framework.startup import bootstrap_project

    from asi_project.config import find_repo_root

    root = find_repo_root()
    bootstrap_project(root)
    with KedroSession.create(project_path=root, env=args.env) as session:
        raw = session.load_context().catalog.load("raw_data")
    report = benchmark_runs(
        args.run,
        raw,
        warmup=args.warmup,
        repeats=args.repeats,
        threads=args.threads,
        timeout=args.timeout,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(report, indent=2, allow_nan=False) + "\n"
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
