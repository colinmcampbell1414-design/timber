"""Shared test helpers."""

import numpy as np
import pytest
from rasterio.transform import from_origin

from terrain.io import Raster
from terrain.selfcheck import DefaultWidgets


# The stand-in for streamlit now lives in terrain/selfcheck.py (the self-check uses it too).
FakeStreamlit = DefaultWidgets


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
