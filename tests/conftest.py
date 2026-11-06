"""Shared test helpers."""

import contextlib

import numpy as np
import pytest
from rasterio.transform import from_origin

from terrain.io import Raster


class _Tab:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeStreamlit:
    """Stands in for `streamlit` when calling an app's params(st) in tests.

    Every widget simply returns its default value, so tests use the same
    starting values as the app. Text output (markdown, caption...) is ignored.
    """

    def __init__(self, overrides=None):
        self.overrides = overrides or {}

    def _value(self, label, default):
        return self.overrides.get(label, default)

    def slider(self, label, min_value=None, max_value=None, value=None, step=None, **kw):
        return self._value(label, value)

    def number_input(self, label, min_value=None, max_value=None, value=None, step=None, **kw):
        return self._value(label, value)

    def select_slider(self, label, options=(), value=None, **kw):
        return self._value(label, value if value is not None else options[0])

    def selectbox(self, label, options, index=0, **kw):
        return self._value(label, list(options)[index])

    def radio(self, label, options, index=0, **kw):
        return self._value(label, list(options)[index])

    def checkbox(self, label, value=False, **kw):
        return self._value(label, value)

    def text_input(self, label, value="", **kw):
        return self._value(label, value)

    def file_uploader(self, label, **kw):
        return self._value(label, None)

    def columns(self, spec, **kw):
        n = spec if isinstance(spec, int) else len(spec)
        return [self] * n

    @contextlib.contextmanager
    def expander(self, *a, **kw):
        yield self

    @contextlib.contextmanager
    def form(self, *a, **kw):
        yield self

    def tabs(self, labels, **kw):
        return [_Tab() for _ in labels]  # each tab is a "with" block that does nothing

    def __getattr__(self, name):  # markdown, caption, write, info, warning, ...
        return lambda *a, **kw: None


@pytest.fixture
def fake_st():
    return FakeStreamlit


CELL = 0.1
LEFT, TOP = 500000.0, 4000000.0


def make_raster(data, cell=CELL):
    """Wrap a 2-D array as a Raster in UTM meters with the given cell size."""
    return Raster(data=np.asarray(data, dtype=float), transform=from_origin(LEFT, TOP, cell, cell), crs="EPSG:32618")


def local_grid(shape, cell=CELL):
    """Meters east (x) and north (y) of the bottom-left corner, for each cell center."""
    rows, cols = shape
    r, c = np.indices(shape)
    return (c + 0.5) * cell, (rows - r - 0.5) * cell


def to_map(x_local, y_local, rows, cell=CELL):
    """Convert local meters (from the bottom-left corner) to map coordinates."""
    return LEFT + x_local, TOP - rows * cell + y_local
