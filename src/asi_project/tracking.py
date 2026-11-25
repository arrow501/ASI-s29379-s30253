"""Opt-in W&B publication of completed local results, never a training dependency."""

import hashlib
import math
import os
from pathlib import Path

from asi_project.pipelines.experiments.artifacts import create_run, read_json, write_json

MODES = {"disabled", "offline", "online"}


def tracking_settings(settings):
    if not isinstance(settings, dict) or set(settings) != {"mode", "entity", "project"}:
        raise ValueError(
            "tracking requires exactly mode, entity and project; credentials stay outside config."
        )
    mode = os.environ.get("WANDB_MODE", settings["mode"])
    if mode not in MODES:
        raise ValueError("tracking.mode / WANDB_MODE must be disabled, offline or online.")
    resolved = {"mode": mode}
    for key, env in (("entity", "WANDB_ENTITY"), ("project", "WANDB_PROJECT")):
        value = os.environ.get(env, settings[key])
        if mode != "disabled" and (not isinstance(value, str) or not value.strip() or "/" in value):
            raise ValueError(f"Explicit W&B {key} is required for offline/online tracking.")
        resolved[key] = value
    return resolved


def flatten_metrics(values, prefix=""):
    """Only finite numeric observations; never serialize arbitrary objects or environment."""
    result = {}
    for key, value in values.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten_metrics(value, name))
        elif isinstance(value, int | float) and not isinstance(value, bool):
            if not math.isfinite(value):
                raise ValueError(f"Non-finite tracking metric: {name}")
            result[name] = value
    return result


def publish_results(root, config, provenance, results, tracking, *, report_files):
    """One run per candidate, resumable under a deterministic source/target ID."""
    if tracking["mode"] == "disabled":
        return {"status": "disabled", "runs": {}}
    root = Path(root)
    receipt_path = root / "wandb.json"
    receipt = {"status": "running", "target": tracking, "runs": {}}
    write_json(receipt_path, receipt)
    try:
        import wandb

        for name, values in results.items():
            run_id = hashlib.sha256(f"{root.name}/{name}".encode()).hexdigest()[:20]
            run_config = {
                **config,
                "provenance": provenance,
                "local_run_id": root.name,
                "candidate_name": name,
            }
            # No code upload, console capture, raw rows or models; production packages belong to #20.
            with wandb.init(
                entity=tracking["entity"],
                project=tracking["project"],
                mode=tracking["mode"],
                id=run_id,
                resume="allow" if tracking["mode"] == "online" else None,
                name=f"{config['protocol']}/{name}/{root.name}",
                group=root.name,
                job_type=config["protocol"],
                config=run_config,
                save_code=False,
                dir=str(root),
                settings=wandb.Settings(
                    disable_code=True, disable_git=True, console="off", init_timeout=45
                ),
            ) as run:
                run.log(flatten_metrics(values))
                run.summary["local_status"] = "completed"
                artifact = wandb.Artifact(f"reports-{run_id}", type="experiment-report")
                for filename in report_files:
                    artifact.add_file(str(root / filename), name=filename)
                run.log_artifact(artifact)
                receipt["runs"][name] = {
                    "id": run_id,
                    "url": run.url if tracking["mode"] == "online" else None,
                }
            write_json(receipt_path, receipt)
        receipt["status"] = "completed"
    except Exception as error:
        # Keep local results usable. Record the exception type, not potentially sensitive messages.
        receipt.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        write_json(receipt_path, receipt)
    return receipt


def track_experiment(selection, tracking, tested=None):
    resolved = tracking_settings(tracking)
    if resolved["mode"] == "disabled":
        return {"status": "disabled", "runs": {}}
    root = Path(selection["run_dir"])
    if (
        selection != read_json(root / "selection.json")
        or selection["test_used_for_selection"] is not False
    ):
        raise ValueError("Tracking requires the unchanged frozen validation decision.")
    report = read_json(root / "metrics.json")
    config = read_json(root / "config.json")
    provenance = read_json(root / "provenance.json")
    expected = {"baseline"}
    if config["protocol"] == "holdout":
        expected.add("dummy")
    if config["experiments"]["autogluon_enabled"]:
        expected.update(item["name"] for item in config["experiments"]["candidates"])
    if (
        report["status"] != "completed"
        or set(report["results"]) != expected
        or set(selection["candidates"]) != expected
        or selection["project"] != config["project"]
        or selection["data_fingerprint"] != provenance["data_fingerprint"]
    ):
        raise ValueError("Tracking requires complete matching experiment results and provenance.")
    files = [
        "config.json",
        "provenance.json",
        "metrics.json",
        "selection.json",
        "split.json" if config["protocol"] == "holdout" else "folds.json",
    ]
    if tested is not None:
        if (
            tested != read_json(root / "test_metrics.json")
            or tested["status"] != "completed"
            or set(tested["results"]) != expected
        ):
            raise ValueError("Tracking requires the completed unchanged final test.")
        files.append("test_metrics.json")
    candidates = {item["name"]: item for item in config["experiments"]["candidates"]}
    results = {}
    for name in sorted(expected):
        local = report["results"][name]
        frozen = selection["candidates"][name]
        if local["decision_threshold"] != frozen["decision_threshold"]:
            raise ValueError("Tracking threshold differs from the frozen validation decision.")
        values = {**local, "selected": int(name == selection["selected_candidate"])}
        if name in candidates:
            values["time_limit_seconds"] = candidates[name]["time_limit"]
        if tested is not None:
            if tested["results"][name]["decision_threshold"] != frozen["decision_threshold"]:
                raise ValueError("Test threshold differs from frozen validation decision.")
            values.update(tested["results"][name])
        results[name] = values
    # Validate all payloads and required files before creating any remote runs.
    for values in results.values():
        flatten_metrics(values)
    for filename in files:
        if not (root / filename).is_file():
            raise FileNotFoundError(root / filename)
    return publish_results(root, config, provenance, results, resolved, report_files=files)


def track_baseline(metrics, raw_data, project, training, store, tracking):
    resolved = tracking_settings(tracking)
    if resolved["mode"] == "disabled":
        return {"status": "disabled", "runs": {}}
    from asi_project.config import load_registry
    from asi_project.metrics import validate_metrics_record

    if validate_metrics_record(metrics, load_registry()[project["dataset_id"]]):
        raise ValueError("Invalid S2 baseline metrics.")
    root, report = create_run(
        store,
        raw_data,
        project,
        training,
        {"autogluon_enabled": False},
        metrics["data_mode"],
        protocol="s2_baseline",
        evaluation_set="test",
    )
    write_json(root / "s2_metrics.json", metrics)
    report.update(status="completed", results=metrics["models"], test_evaluated=True)
    write_json(root / "metrics.json", report)
    config = read_json(root / "config.json")
    config.update(split=metrics["split"], decision_threshold=0.5, validation_used=False)
    write_json(root / "config.json", config)
    # Keep the exact S2 baseline and dummy together in the first reference run.
    return publish_results(
        root,
        config,
        read_json(root / "provenance.json"),
        {"s2_baseline": {"test": metrics["models"]}},
        resolved,
        report_files=["config.json", "provenance.json", "s2_metrics.json"],
    )


def main(argv=None):
    """Publish persisted runs after an outage, without retraining or changing selection."""
    import argparse

    from kedro.framework.session import KedroSession
    from kedro.framework.startup import bootstrap_project

    from asi_project.config import find_repo_root

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=Path, required=True)
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--env", default="local")
    args = parser.parse_args(argv)
    root = find_repo_root()
    bootstrap_project(root)
    with KedroSession.create(project_path=root, env=args.env) as session:
        settings = session.load_context().params["tracking"]
    for source in args.run:
        receipt = track_experiment(
            read_json(source / "selection.json"),
            settings,
            read_json(source / "test_metrics.json") if args.include_test else None,
        )
        print(f"{source.name}: {receipt['status']}")
        for candidate, run in receipt["runs"].items():
            print(f"  {candidate}: {run['url'] or run['id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
