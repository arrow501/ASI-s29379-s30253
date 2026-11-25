"""Catalog-owned location for native local experiment outputs."""

from pathlib import Path

from kedro.io import AbstractDataset, DatasetError


class ExperimentStore(AbstractDataset):
    def __init__(self, filepath):
        self._path = Path(filepath)

    def _load(self):
        return self._path

    def _save(self, data):
        raise DatasetError(
            "ExperimentStore is a read-only output location supplied to training nodes."
        )

    def _describe(self):
        return {"filepath": str(self._path)}
