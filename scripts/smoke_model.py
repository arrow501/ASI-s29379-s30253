"""Sprawdza pakiet w nowym procesie; tryb online pobiera rzeczywisty W&B Artifact."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from asi_project.model_artifacts import download_model_package
from asi_project.model_package import load_model_package, predict_model_package
from asi_project.pipelines.experiments.artifacts import read_json, write_json
from asi_project.provenance import git_provenance

ROWS = 5
REPO_ROOT = Path(__file__).resolve().parents[1]


def validate_predictions(result, rows, metadata):
    """Sprawdza kontrakt pięciu odpowiedzi, ich kolejność, klasy i prawdopodobieństwa."""
    if metadata["negative_class"] != "no" or metadata["positive_class"] != "yes":
        raise ValueError("The Bank Marketing smoke test requires no/yes classes.")
    if metadata["target"] in rows:
        raise ValueError("Smoke input must not contain the target.")
    if len(rows) != ROWS or len(result) != ROWS:
        raise ValueError("The smoke test requires exactly five input and output rows.")
    if list(result.columns) != ["prediction", "probability_positive"]:
        raise ValueError("Unexpected prediction schema.")
    if not result.index.equals(rows.index):
        raise ValueError("Predictions must preserve input index and order.")
    probabilities = result["probability_positive"]
    if not pd.api.types.is_numeric_dtype(probabilities):
        raise ValueError("Probabilities must be numeric.")
    values = probabilities.to_numpy(dtype=float)
    if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise ValueError("Probabilities must be finite and in [0, 1].")
    expected = np.where(values >= metadata["decision_threshold"], "yes", "no")
    if not np.array_equal(result["prediction"].to_numpy(), expected):
        raise ValueError("Predicted classes must follow the stored decision threshold.")


def _forbid_fit(*args, **kwargs):
    """Przerywa smoke test, jeśli odczyt lub predykcja spróbują uruchomić trening."""
    raise RuntimeError("Model loading and smoke inference must not call fit.")


def _disable_training(kind):
    """Blokuje dopasowanie obsługiwanych modeli i komponentów w procesie predykcji."""
    from sklearn.dummy import DummyClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    for component in (
        DummyClassifier,
        LogisticRegression,
        Pipeline,
        SimpleImputer,
        OneHotEncoder,
        StandardScaler,
    ):
        component.fit = _forbid_fit
    if kind == "autogluon":
        from autogluon.tabular import TabularPredictor

        TabularPredictor.fit = _forbid_fit


def _worker(directory):
    """Wczytuje tylko wskazany pakiet; nie ma dostępu do SDK W&B ani obiektu z treningu."""
    request = read_json(directory / "request.json")
    os.environ["WANDB_MODE"] = "disabled"
    sys.modules["wandb"] = None
    metadata = read_json(Path(request["package"]) / "metadata.json")
    _disable_training(metadata["model_kind"])
    model, metadata = load_model_package(request["package"])
    rows = pd.read_csv(request["input"], nrows=ROWS)
    rows = rows.drop(columns=[metadata["target"]], errors="ignore")
    result = predict_model_package(model, metadata, rows)
    validate_predictions(result, rows, metadata)
    write_json(
        directory / "response.json",
        {
            "status": "passed",
            "pid": os.getpid(),
            "rows": len(result),
            "input_index": rows.index.tolist(),
            "prediction_index": result.index.tolist(),
            "columns": list(result.columns),
            "target_removed": metadata["target"] not in rows,
            "training_disabled": True,
            "wandb_mode": "disabled",
            "model": {
                **{
                    key: metadata[key]
                    for key in (
                        "model_kind",
                        "dataset_id",
                        "target",
                        "numeric_features",
                        "categorical_features",
                        "negative_class",
                        "positive_class",
                        "decision_threshold",
                        "git_sha",
                        "git_dirty",
                        "data_fingerprint",
                        "versions",
                    )
                },
                **{
                    key: metadata[key]
                    for key in ("source_run_id", "candidate_name", "protocol", "data_mode")
                    if key in metadata
                },
            },
            "predictions": result.to_dict(orient="records"),
        },
    )


def run_inference(package_dir, input_path, *, timeout):
    """Uruchamia nowy interpreter z wyłączonym W&B i bez przekazywania klucza API."""
    with tempfile.TemporaryDirectory(prefix="asi-model-smoke-") as temporary:
        directory = Path(temporary)
        write_json(
            directory / "request.json",
            {"package": str(Path(package_dir).resolve()), "input": str(input_path.resolve())},
        )
        env = {**os.environ, "WANDB_MODE": "disabled", "KEDRO_DISABLE_TELEMETRY": "1"}
        env.pop("WANDB_API_KEY", None)
        completed = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", str(directory)],
            cwd=directory,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if completed.returncode:
            raise RuntimeError(f"Smoke inference process failed:\n{completed.stderr[-6000:]}")
        result = read_json(directory / "response.json")
        if result["status"] != "passed" or result["pid"] == os.getpid():
            raise ValueError("Smoke inference must succeed in a separate process.")
        return result


def _download_and_predict(artifact_ref, expected_ref, input_path, timeout):
    """Pobiera do pustego katalogu; błąd pobrania kończy próbę bez lokalnego zamiennika."""
    parts = artifact_ref.split("/")
    if len(parts) != 3 or ":" not in parts[2] or not all(parts):
        raise ValueError("Supply the full entity/project/artifact:alias-or-version reference.")
    settings = {"mode": "online", "entity": parts[0], "project": parts[1]}
    with tempfile.TemporaryDirectory(prefix="asi-production-smoke-") as temporary:
        package, receipt = download_model_package(
            artifact_ref, Path(temporary) / "package", settings
        )
        resolved = receipt["artifact_ref"]
        if re.fullmatch(r"[^/]+/[^/]+/[^/:]+:v\d+", resolved) is None:
            raise ValueError("The download receipt must identify a concrete artifact version.")
        if expected_ref is not None and resolved != expected_ref:
            raise ValueError("Production resolves to a different version than the approved model.")
        inference = run_inference(package, input_path, timeout=timeout)
        for key in ("git_sha", "data_fingerprint"):
            if inference["model"][key] != receipt[key]:
                raise ValueError("Downloaded model provenance differs from its receipt.")
        return {"mode": "online", **receipt, "inference": inference}


def main(argv=None):
    """Zapisuje wynik lokalnego lub rzeczywistego smoke testu online bez nadpisywania raportów."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--package", type=Path, help="Validate a local package without W&B")
    source.add_argument("--artifact", help="Download the full W&B production reference")
    parser.add_argument("--expected-ref", help="Require the approved concrete W&B version")
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "data/01_raw/sample.csv")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        _worker(args.worker)
        return 0
    if not (args.package or args.artifact) or args.output is None:
        parser.error("--package or --artifact and --output are required")
    if args.timeout < 1:
        parser.error("--timeout must be a positive integer")
    if args.expected_ref is not None and not args.artifact:
        parser.error("--expected-ref requires --artifact")
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.package:
        result = {
            "mode": "local",
            "package_dir": str(args.package.resolve()),
            "inference": run_inference(args.package, args.input, timeout=args.timeout),
        }
    else:
        result = _download_and_predict(args.artifact, args.expected_ref, args.input, args.timeout)
    result.update(
        status="passed",
        checked_utc=datetime.now(UTC).isoformat(),
        checkout=git_provenance(REPO_ROOT),
        smoke_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
