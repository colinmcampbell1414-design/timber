"""Tests for preparing a survey, its disk cache, and the fast photo read."""

import os
import time

import numpy as np
import pytest
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin

from terrain import io, survey

CRS = "EPSG:32618"
LEFT, TOP = 500000.0, 4000000.0


def write_tif(path, bands, cell, nodata=None, dtype="float32", overviews=None):
    """Save one or more 2-D arrays as a GeoTIFF (optionally with overview copies, like WebODM)."""
    if bands.ndim == 2:  # one band
        bands = bands[None]
    with rasterio.open(path, "w", driver="GTiff", height=bands.shape[1], width=bands.shape[2], count=bands.shape[0],
                       dtype=dtype, crs=CRS, transform=from_origin(LEFT, TOP, cell, cell), nodata=nodata,
                       tiled=True, blockxsize=64, blockysize=64) as dst:
        dst.write(bands.astype(dtype))
        if overviews:
            dst.build_overviews(overviews, Resampling.average)


def make_survey(folder, size=200, cell=0.1):
    """A 20 m x 20 m survey: a tilted ground, a 5 m box on it, and a photo that is
    red in the west half and green in the east half, with no photo in one corner."""
    folder.mkdir()
    r, c = np.indices((size, size))
    ground = 0.02 * c * cell + 0.01 * r * cell
    box = np.where((abs(r - 100) < 20) & (abs(c - 100) < 20), 5.0, 0.0)
    write_tif(folder / "s-dtm.tif", ground, cell, nodata=-9999)
    write_tif(folder / "s-dsm.tif", ground + box, cell, nodata=-9999)
    # The photo is twice as sharp (5 cm), with overviews.
    pr, pc = np.indices((2 * size, 2 * size))
    rgba = np.zeros((4, 2 * size, 2 * size))
    rgba[0] = np.where(pc < size, 200, 20)
    rgba[1] = np.where(pc < size, 30, 180)
    rgba[2] = 40
    rgba[3] = np.where((pr < 60) & (pc < 60), 0, 255)  # top-left corner: no photo
    write_tif(folder / "s-orthophoto.tif", rgba, cell / 2, dtype="uint8", overviews=[2, 4, 8])
    return folder


def test_fast_photo_read_matches_known_colors(tmp_path):
    folder = make_survey(tmp_path / "s")
    grid = io.resample(io.load_raster(folder / "s-dtm.tif"), 0.5)  # 50 cm analysis grid: uses an overview
    rgb = io.read_rgb_on_grid(folder / "s-orthophoto.tif", grid)
    assert rgb.shape == (*grid.shape, 3)
    west, east = rgb[20:, 5], rgb[20:, 35]
    assert np.allclose(west, [200 / 255, 30 / 255, 40 / 255], atol=0.01)
    assert np.allclose(east, [20 / 255, 180 / 255, 40 / 255], atol=0.01)
    assert np.isnan(rgb[0, 0]).all()  # the corner with no photo stays empty


def test_prepare_then_reuse_saved_copy(tmp_path):
    folder = make_survey(tmp_path / "s")
    cache = tmp_path / "cache"
    first = survey.prepare(folder, cell_size=0.2, cache_dir=cache)
    assert not first.from_cache and len(list(cache.glob("*.npz"))) == 1
    t = time.time()
    again = survey.prepare(folder, cell_size=0.2, cache_dir=cache)
    assert again.from_cache
    assert time.time() - t < 2.0
    assert np.array_equal(again.dtm.data, first.dtm.data, equal_nan=True)
    assert np.array_equal(again.dsm.data, first.dsm.data, equal_nan=True)
    assert again.dtm.transform == first.dtm.transform and again.dtm.crs == first.dtm.crs
    assert np.allclose(again.ortho, first.ortho, equal_nan=True, atol=1e-6)
    for key in ("depth", "acc_area_m2", "filled"):
        assert np.allclose(again.flow[key], first.flow[key], equal_nan=True, rtol=1e-5, atol=1e-5)
    assert np.array_equal(again.flow["receiver"], first.flow["receiver"])


def test_new_setting_or_changed_file_prepares_again(tmp_path):
    folder = make_survey(tmp_path / "s")
    cache = tmp_path / "cache"
    survey.prepare(folder, cell_size=0.2, cache_dir=cache)
    assert not survey.prepare(folder, cell_size=0.4, cache_dir=cache).from_cache  # new cell size
    assert not survey.prepare(folder, cell_size=0.2, pit_depth=None, cache_dir=cache).from_cache  # repair off
    dsm = folder / "s-dsm.tif"
    os.utime(dsm, ns=(dsm.stat().st_atime_ns, dsm.stat().st_mtime_ns + 10**9))  # the DSM was replaced
    assert not survey.prepare(folder, cell_size=0.2, cache_dir=cache).from_cache
    assert survey.prepare(folder, cell_size=0.2, cache_dir=cache).from_cache


def test_damaged_saved_copy_is_replaced(tmp_path):
    folder = make_survey(tmp_path / "s")
    cache = tmp_path / "cache"
    survey.prepare(folder, cell_size=0.2, cache_dir=cache)
    saved = next(cache.glob("*.npz"))
    saved.write_bytes(b"not a real file")
    s = survey.prepare(folder, cell_size=0.2, cache_dir=cache)
    assert not s.from_cache and s.dtm.shape == (100, 100)


def test_old_saved_copies_are_cleaned_up(tmp_path, monkeypatch):
    monkeypatch.setattr(survey, "KEEP_CACHED", 2)
    folder = make_survey(tmp_path / "s")
    cache = tmp_path / "cache"
    for cell in (0.2, 0.3, 0.4, 0.5):
        survey.prepare(folder, cell_size=cell, cache_dir=cache)
        time.sleep(0.02)
    assert len(list(cache.glob("*.npz"))) == 2


def test_missing_dtm_gives_a_clear_error(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        survey.prepare(tmp_path / "empty")
