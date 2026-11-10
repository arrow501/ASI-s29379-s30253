"""Browser-local orthonormal PCA rotations; only selection returns to Python."""

from pathlib import Path

import anywidget
import traitlets


class PCATour(anywidget.AnyWidget):
    _esm = Path(__file__).with_name("pca_tour.mjs")
    _css = Path(__file__).with_name("pca_tour.css")
    scores = traitlets.List().tag(sync=True)
    variance = traitlets.List().tag(sync=True)
    records = traitlets.List().tag(sync=True)
    selected_ids = traitlets.List(traitlets.Int()).tag(sync=True)
