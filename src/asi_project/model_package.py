"""Walidacja i odczyt przenośnych pakietów modeli z metadanymi eksperymentu."""

import copy
import json
import math
import pickle
import re
import tempfile
from pathlib import Path

import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from asi_project.data import label_text, normalize_input, target_classes
from asi_project.pipelines.experiments.artifacts import read_json
from asi_project.pipelines.experiments.scoring import positive_probabilities
from asi_project.provenance import data_fingerprint

REQUIRED_FIELDS = {
    "model_kind",
    "dataset_id",
    "numeric_features",
    "categorical_features",
    "target",
    "negative_class",
    "positive_class",
    "decision_threshold",
    "metrics",
    "git_sha",
    "git_dirty",
    "data_fingerprint",
    "versions",
    "model_files",
}


def _string_list(value, name, *, allow_empty=False):
    """Sprawdza, czy lista zawiera unikalne, niepuste napisy."""
    if (
        not isinstance(value, list)
        or (not allow_empty and not value)
        or not all(isinstance(item, str) and item.strip() for item in value)
        or len(value) != len(set(value))
    ):
        raise ValueError(f"{name}: expected a list of unique non-empty strings.")


def _validate_sklearn_model(model):
    """Wymaga dopasowanego estymatora zwracającego prawdopodobieństwa klas."""
    if not callable(getattr(model, "predict_proba", None)):
        raise ValueError("The sklearn model must provide predict_proba.")

    try:
        check_is_fitted(model)
    except (NotFittedError, TypeError) as error:
        raise ValueError("The sklearn model must be fitted.") from error


def _validate_model_features(model, metadata):
    """Sprawdza zgodność schematu pakietu z cechami wejściowymi zapisanymi w modelu."""
    features = metadata["numeric_features"] + metadata["categorical_features"]
    if metadata["model_kind"] == "sklearn":
        model_features = getattr(model, "feature_names_in_", None)
        if model_features is None:
            raise ValueError("The sklearn model must record input feature names.")
        if list(model_features) != features:
            raise ValueError("Model input features do not match metadata names and order.")
    else:
        # AutoGluon może usuwać stałe lub nieużywane kolumny podczas tworzenia cech.
        missing = sorted(set(model.features()) - set(features))
        if missing:
            raise ValueError(f"Metadata omits model input features: {missing}")


def save_sklearn(model, path):
    """Zapisuje dopasowany model sklearn wraz z przetwarzaniem wstępnym."""
    Path(path).write_bytes(pickle.dumps(model))


def load_native_model(path, kind):
    """Wczytuje model w jego oryginalnym formacie bez ponownego dopasowywania."""
    if kind == "sklearn":
        return pickle.loads(Path(path).read_bytes())

    if kind == "autogluon":
        from autogluon.tabular import TabularPredictor

        return TabularPredictor.load(str(path), verbosity=0)

    raise ValueError(f"Unknown native experiment model kind: {kind!r}")


def validate_metadata(metadata, *, require_model_files=True):
    """Sprawdza metadane przed zapisaniem lub odczytaniem całego pakietu."""
    if not isinstance(metadata, dict):
        raise ValueError("metadata.json must contain an object.")

    required = REQUIRED_FIELDS if require_model_files else REQUIRED_FIELDS - {"model_files"}
    missing = required - metadata.keys()

    if missing:
        raise ValueError(f"Missing metadata fields: {sorted(missing)}")

    if metadata["model_kind"] not in ("sklearn", "autogluon"):
        raise ValueError("model_kind must be sklearn or autogluon.")

    for key in ("dataset_id", "target"):
        if not isinstance(metadata[key], str) or not metadata[key].strip():
            raise ValueError(f"{key} must be a non-empty string.")

    for key in ("numeric_features", "categorical_features"):
        _string_list(metadata[key], key, allow_empty=True)

    features = metadata["numeric_features"] + metadata["categorical_features"]

    if not features or len(features) != len(set(features)) or metadata["target"] in features:
        raise ValueError("Features must be non-empty, disjoint and exclude the target.")

    for key in ("negative_class", "positive_class"):
        value = metadata[key]

        if type(value) not in (str, int) or (isinstance(value, str) and not value.strip()):
            raise ValueError(f"{key} must be a non-empty string or integer.")

    if metadata["negative_class"] == metadata["positive_class"]:
        raise ValueError("Negative and positive classes must differ.")

    threshold = metadata["decision_threshold"]

    if (
        type(threshold) not in (int, float)
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("decision_threshold must be a finite number in [0, 1].")

    metrics = metadata["metrics"]

    if not isinstance(metrics, dict) or not metrics or set(metrics) - {"validation", "test"}:
        raise ValueError("metrics must contain validation and/or test results.")

    for split, scores in metrics.items():
        if not isinstance(scores, dict):
            raise ValueError(f"metrics.{split} must contain metric values.")

        for key in ("accuracy", "precision", "recall", "f1"):
            value = scores.get(key)

            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"metrics.{split}.{key} must be a finite number in [0, 1].")

    for key, length in (("git_sha", 40), ("data_fingerprint", 64)):
        value = metadata[key]

        if not isinstance(value, str) or re.fullmatch(rf"[0-9a-fA-F]{{{length}}}", value) is None:
            raise ValueError(f"{key} must contain {length} hexadecimal characters.")

    if not isinstance(metadata["git_dirty"], bool):
        raise ValueError("git_dirty must be a boolean.")

    versions = metadata["versions"]
    required_versions = {"python", "numpy", "pandas", "scikit-learn"}

    if metadata["model_kind"] == "autogluon":
        required_versions.add("autogluon.tabular")

    if (
        not isinstance(versions, dict)
        or not required_versions.issubset(versions)
        or not all(
            isinstance(key, str) and key.strip() and isinstance(value, str) and value.strip()
            for key, value in versions.items()
        )
    ):
        raise ValueError(
            f"versions must contain non-empty versions for {sorted(required_versions)}."
        )

    if require_model_files or "model_files" in metadata:
        _string_list(metadata["model_files"], "model_files")

        if metadata["model_kind"] == "sklearn" and metadata["model_files"] != ["pipeline.pkl"]:
            raise ValueError("The sklearn package must contain model/pipeline.pkl.")


def load_model_package(package_dir):
    """Sprawdza metadane i wymienione pliki, po czym wczytuje model bez dopasowywania."""
    package_dir = Path(package_dir)

    try:
        metadata = json.loads((package_dir / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("Cannot read valid metadata.json from the package.") from error

    validate_metadata(metadata)

    model_dir = (package_dir / "model").resolve()
    if not model_dir.is_dir():
        raise ValueError("Missing model directory.")

    for name in metadata["model_files"]:
        relative = Path(name)
        path = (model_dir / relative).resolve()

        if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(model_dir):
            raise ValueError(f"Invalid model file path: {name}")

        if not path.is_file():
            raise ValueError(f"Missing model file: {name}")

    model_path = model_dir / "pipeline.pkl" if metadata["model_kind"] == "sklearn" else model_dir

    try:
        model = load_native_model(model_path, metadata["model_kind"])
    except Exception as error:
        raise ValueError("Cannot load the model from the package.") from error

    if metadata["model_kind"] == "sklearn":
        _validate_sklearn_model(model)

    _validate_model_features(model, metadata)

    return model, metadata


def save_model_package(model, metadata, package_dir):
    """Zapisuje samodzielny pakiet bez zmieniania artefaktów treningowych."""
    if not isinstance(metadata, dict):
        raise ValueError("Metadata must be a dictionary.")

    kind = metadata.get("model_kind")
    if kind not in ("sklearn", "autogluon"):
        raise ValueError("model_kind must be sklearn or autogluon.")

    if kind == "sklearn":
        _validate_sklearn_model(model)

    validate_metadata(metadata, require_model_files=False)
    _validate_model_features(model, metadata)

    package_dir = Path(package_dir).resolve()
    if package_dir.exists():
        raise FileExistsError(package_dir)

    if kind == "autogluon":
        source_dir = Path(model.path).resolve()
        if package_dir.is_relative_to(source_dir):
            raise ValueError("The package must be outside the original AutoGluon directory.")

    package_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=".model-package-",
        dir=package_dir.parent,
    ) as temporary:
        staging = Path(temporary) / "package"
        staging.mkdir()
        model_dir = staging / "model"

        if kind == "sklearn":
            model_dir.mkdir()
            save_sklearn(model, model_dir / "pipeline.pkl")
        else:
            model.clone_for_deployment(str(model_dir), model="best")

        packaged_metadata = copy.deepcopy(metadata)
        packaged_metadata["model_files"] = sorted(
            path.relative_to(model_dir).as_posix()
            for path in model_dir.rglob("*")
            if path.is_file()
        )

        validate_metadata(packaged_metadata)

        content = (
            json.dumps(
                packaged_metadata,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )

        (staging / "metadata.json").write_text(
            content,
            encoding="utf-8",
        )

        load_model_package(staging)
        staging.rename(package_dir)

    return package_dir


def build_model_metadata(
    run_dir,
    candidate_name,
    raw_data,
    *,
    include_test=False,
):
    """Buduje metadane pakietu z zapisanych wyników i zgodnych danych surowych."""
    if not isinstance(include_test, bool):
        raise ValueError("include_test must be a boolean.")

    run_dir = Path(run_dir)

    config = read_json(run_dir / "config.json")
    provenance = read_json(run_dir / "provenance.json")
    report = read_json(run_dir / "metrics.json")
    selection = read_json(run_dir / "selection.json")
    project = config["project"]

    if report["status"] != "completed":
        raise ValueError("Metadata requires a completed experiment.")

    if (
        selection["test_used_for_selection"] is not False
        or selection["project"] != project
        or selection["data_fingerprint"] != provenance["data_fingerprint"]
        or report["protocol"] != config["protocol"]
        or set(report["results"]) != set(selection["candidates"])
    ):
        raise ValueError("Experiment reports, selection and provenance do not match.")

    if candidate_name not in selection["candidates"]:
        raise ValueError(f"Unknown experiment candidate: {candidate_name}")

    result = report["results"][candidate_name]
    frozen = selection["candidates"][candidate_name]

    if Path(result["model_path"]).resolve() != Path(frozen["model_path"]).resolve():
        raise ValueError("Candidate model_path differs from the frozen selection.")

    for key in ("model_kind", "decision_threshold", "threshold_source"):
        if result[key] != frozen[key]:
            raise ValueError(f"Candidate {key} differs from the frozen selection.")

    if data_fingerprint(raw_data) != provenance["data_fingerprint"]:
        raise ValueError("Raw data does not match the experiment fingerprint.")

    if project["target"] not in raw_data:
        raise ValueError("Raw data must contain the original target column.")

    classes = [value for value in target_classes(raw_data[project["target"]]) if value]
    positive = label_text(project["positive_class"])

    if len(classes) != 2 or positive not in classes:
        raise ValueError("The target must contain two classes including the positive class.")

    negative = next(value for value in classes if value != positive)
    metrics = {"validation": copy.deepcopy(result["validation"])}

    if include_test:
        tested = read_json(run_dir / "test_metrics.json")

        if (
            tested["status"] != "completed"
            or tested["test_used_for_selection"] is not False
            or tested["selected_candidate"] != selection["selected_candidate"]
            or set(tested["results"]) != set(selection["candidates"])
        ):
            raise ValueError("Test metrics must come from the completed frozen evaluation.")

        test_result = tested["results"][candidate_name]

        for key in ("decision_threshold", "threshold_source"):
            if test_result[key] != frozen[key]:
                raise ValueError(f"Test {key} differs from the frozen selection.")

        metrics["test"] = copy.deepcopy(test_result["test"])

    metadata = {
        "model_kind": frozen["model_kind"],
        "dataset_id": project["dataset_id"],
        "numeric_features": copy.deepcopy(project["numeric_features"]),
        "categorical_features": copy.deepcopy(project["categorical_features"]),
        "target": project["target"],
        "negative_class": negative,
        "positive_class": positive,
        "decision_threshold": frozen["decision_threshold"],
        "metrics": metrics,
        "git_sha": provenance["git_sha"],
        "git_dirty": provenance["git_dirty"],
        "data_fingerprint": provenance["data_fingerprint"],
        "versions": {
            **provenance["versions"],
            "python": provenance["python"],
        },
        "source_run_id": Path(report["run_dir"].replace("\\", "/")).name,
        "candidate_name": candidate_name,
        "protocol": config["protocol"],
        "data_mode": config["data_mode"],
        "threshold_source": frozen["threshold_source"],
    }

    validate_metadata(metadata, require_model_files=False)

    return metadata


def predict_model_package(model, metadata, raw_data):
    """Zwraca oryginalne etykiety klas i prawdopodobieństwa klasy pozytywnej bez treningu."""
    validate_metadata(metadata)
    if not isinstance(raw_data, pd.DataFrame):
        raise TypeError("Prediction input must be a pandas DataFrame.")
    if not raw_data.columns.is_unique:
        raise ValueError("Prediction input must have unique column names.")
    features = metadata["numeric_features"] + metadata["categorical_features"]
    missing = [name for name in features if name not in raw_data.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    if len(raw_data) == 0:
        return pd.DataFrame(
            {
                "prediction": pd.Series(index=raw_data.index, dtype=object),
                "probability_positive": pd.Series(index=raw_data.index, dtype=float),
            }
        )
    X = normalize_input(raw_data.loc[:, features])
    if metadata["model_kind"] == "autogluon":
        X = X.astype({name: "float64" for name in metadata["numeric_features"]})
    probability = positive_probabilities(model, X)
    prediction = [
        metadata["positive_class"]
        if value >= metadata["decision_threshold"]
        else metadata["negative_class"]
        for value in probability
    ]
    return pd.DataFrame(
        {"prediction": prediction, "probability_positive": probability},
        index=raw_data.index.copy(),
    )
