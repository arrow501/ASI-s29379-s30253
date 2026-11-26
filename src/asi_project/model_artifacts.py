"""Publikacja i pobieranie wersjonowanych pakietów modeli przez W&B Artifacts."""

import tempfile
from pathlib import Path

from asi_project.model_package import load_model_package
from asi_project.pipelines.experiments.artifacts import write_json
from asi_project.tracking import tracking_settings


def _online_settings(tracking):
    """Odrzuca tryby disabled i offline przed uzyskaniem dostępu do W&B."""
    settings = tracking_settings(tracking)

    if settings["mode"] != "online":
        raise ValueError("Publishing and downloading artifacts require online tracking.")

    return settings


def _version_ref(artifact):
    """Zwraca identyfikator artefaktu wskazujący konkretną wersję."""
    name = artifact.name.rsplit(":", 1)[0]

    return f"{artifact.entity}/{artifact.project}/{name}:{artifact.version}"


def publish_model_package(
    package_dir,
    tracking,
    *,
    artifact_name,
    production=False,
):
    """Publikuje sprawdzony pakiet i zapisuje konkretną wersję artefaktu."""
    settings = _online_settings(tracking)
    package_dir = Path(package_dir).resolve()
    _, metadata = load_model_package(package_dir)

    if not isinstance(production, bool):
        raise ValueError("production must be a boolean.")

    import wandb

    artifact = wandb.Artifact(
        artifact_name,
        type="model",
        metadata={
            key: metadata[key]
            for key in (
                "model_kind",
                "dataset_id",
                "git_sha",
                "data_fingerprint",
                "decision_threshold",
            )
        },
    )
    artifact.add_dir(str(package_dir))

    aliases = ["latest", "production"] if production else ["latest"]

    with wandb.init(
        entity=settings["entity"],
        project=settings["project"],
        mode="online",
        job_type="publish-model",
        dir=str(package_dir.parent),
        save_code=False,
        settings=wandb.Settings(
            disable_code=True,
            disable_git=True,
            console="off",
            init_timeout=45,
        ),
    ) as run:
        uploaded = run.log_artifact(artifact, aliases=aliases)
        uploaded.wait()

        receipt = {
            "artifact_ref": _version_ref(uploaded),
            "aliases": aliases,
            "git_sha": metadata["git_sha"],
            "data_fingerprint": metadata["data_fingerprint"],
        }

    write_json(
        package_dir.parent / f"{package_dir.name}.wandb.json",
        receipt,
    )

    return receipt


def download_model_package(artifact_ref, package_dir, tracking):
    """Pobiera i sprawdza pakiet modelu oraz zapisuje jego konkretną wersję."""
    settings = _online_settings(tracking)
    package_dir = Path(package_dir).resolve()

    if package_dir.exists():
        raise FileExistsError(package_dir)

    import wandb

    api = wandb.Api(
        overrides={
            "entity": settings["entity"],
            "project": settings["project"],
        }
    )
    artifact = api.artifact(artifact_ref, type="model")

    package_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=".model-download-",
        dir=package_dir.parent,
    ) as temporary:
        staging = Path(temporary) / "package"
        artifact.download(root=str(staging))

        _, metadata = load_model_package(staging)

        receipt = {
            "requested_ref": artifact_ref,
            "artifact_ref": _version_ref(artifact),
            "git_sha": metadata["git_sha"],
            "data_fingerprint": metadata["data_fingerprint"],
        }

        staging.rename(package_dir)

        try:
            write_json(
                package_dir.parent / f"{package_dir.name}.wandb.json",
                receipt,
            )
        except OSError:
            # Cofnij przeniesienie, aby awaria zapisu nie blokowała ponownego pobrania.
            package_dir.rename(staging)
            raise

    return package_dir, receipt
