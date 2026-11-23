"""Catalog-owned location for native local experiment outputs."""

from pathlib import Path

from kedro.io import AbstractDataset


class ExperimentStore(AbstractDataset):
    def __init__(self, filepath):
        self._path = Path(filepath)

    def _load(self):
        return self._path

    def _save(self, data):
        raise NotImplementedError("ExperimentStore is an input location, not a model package.")

    def _describe(self):
        return {"filepath": str(self._path)}
