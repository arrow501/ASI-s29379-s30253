"""Wczytuje pakiet i wykonuje predykcję dla wierszy CSV bez treningu i dostępu do W&B."""

import argparse
import json
from pathlib import Path

import pandas as pd

from asi_project.model_package import load_model_package, predict_model_package


def main(argv=None):
    """Wykonuje lokalną predykcję przez wspólny interfejs odbiorców pakietu."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--sep", default=",")
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.rows < 1:
        parser.error("--rows must be a positive integer")
    data = pd.read_csv(args.input, sep=args.sep, nrows=args.rows)
    if len(data) != args.rows:
        parser.error(f"Input contains only {len(data)} rows; requested {args.rows}")
    model, metadata = load_model_package(args.package)
    predictions = predict_model_package(model, metadata, data)
    payload = {
        "n_rows": len(predictions),
        "positive_class": metadata["positive_class"],
        "decision_threshold": metadata["decision_threshold"],
        "git_sha": metadata["git_sha"],
        "data_fingerprint": metadata["data_fingerprint"],
        "predictions": predictions.to_dict(orient="records"),
    }
    content = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(content, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
