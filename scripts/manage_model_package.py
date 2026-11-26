"""Eksport modeli, publikacja pakietów i pobieranie wersjonowanych artefaktów W&B."""

import argparse
import json
from pathlib import Path

import pandas as pd

from asi_project.model_artifacts import download_model_package, publish_model_package
from asi_project.model_package import build_model_metadata, load_native_model, save_model_package
from asi_project.pipelines.experiments.artifacts import read_json


def _tracking_parameters(env):
    """Odczytuje konfigurację Kedro z bieżącej kopii projektu bez uruchamiania pipeline’u."""
    from kedro.framework.session import KedroSession
    from kedro.framework.startup import bootstrap_project

    root = Path(__file__).resolve().parents[1]
    bootstrap_project(root)
    with KedroSession.create(project_path=root, env=env) as session:
        return session.load_context().params["tracking"]


def main(argv=None):
    """Przekazuje operacje na pakietach do wspólnych funkcji bez trenowania modelu."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="Package a candidate from a completed experiment")
    export.add_argument("--run", required=True, type=Path)
    export.add_argument("--candidate", required=True)
    export.add_argument("--data", required=True, type=Path)
    export.add_argument("--sep", default=",")
    export.add_argument("--output", required=True, type=Path)
    export.add_argument("--include-test", action="store_true")
    publish = commands.add_parser("publish", help="Publish an existing package to W&B")
    publish.add_argument("--package", required=True, type=Path)
    publish.add_argument("--name", required=True)
    publish.add_argument("--production", action="store_true")
    publish.add_argument("--env", default="local")
    download = commands.add_parser("download", help="Download a W&B model package")
    download.add_argument("--artifact", required=True)
    download.add_argument("--output", required=True, type=Path)
    download.add_argument("--env", default="local")
    args = parser.parse_args(argv)
    if args.command == "export":
        raw = pd.read_csv(args.data, sep=args.sep)
        metadata = build_model_metadata(
            args.run, args.candidate, raw, include_test=args.include_test
        )
        selection = read_json(args.run / "selection.json")
        candidate = selection["candidates"][args.candidate]
        model = load_native_model(candidate["model_path"], candidate["model_kind"])
        package_dir = save_model_package(model, metadata, args.output)
        result = {
            "package_dir": str(package_dir),
            "candidate_name": args.candidate,
            "source_run_id": metadata["source_run_id"],
        }
    elif args.command == "publish":
        result = publish_model_package(
            args.package,
            _tracking_parameters(args.env),
            artifact_name=args.name,
            production=args.production,
        )
    else:
        package_dir, receipt = download_model_package(
            args.artifact, args.output, _tracking_parameters(args.env)
        )
        result = {**receipt, "package_dir": str(package_dir)}
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
