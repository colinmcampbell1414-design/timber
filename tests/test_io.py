"""Tests for loading, nodata masking and resampling, using small synthetic GeoTIFFs."""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from conftest import local_grid, make_raster
from terrain import io

UTM_17N = "EPSG:32617"  # a UTM zone measured in meters


def write_tif(path, data, cell=0.02, nodata=-9999.0, crs=UTM_17N):
    """Save a 2-D array as a GeoTIFF with the given cell size (meters)."""
    transform = from_origin(500000.0, 4000000.0, cell, cell)
    with rasterio.open(
        path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1],
        count=1, dtype="float32", crs=crs, transform=transform, nodata=nodata,
    ) as dst:
        dst.write(data.astype("float32"), 1)


def test_load_reads_pixel_size_and_masks_nodata(tmp_path):
    data = np.full((10, 12), 250.0)
    data[0, 0] = -9999.0
    path = tmp_path / "dtm.tif"
    write_tif(path, data, cell=0.02)

    r = io.load_raster(path)
    assert r.pixel_size == pytest.approx((0.02, 0.02))
    assert r.shape == (10, 12)
    assert np.isnan(r.data[0, 0])  # nodata became NaN, not -9999 or 0
    assert np.nanmin(r.data) == pytest.approx(250.0)
    assert io.unit_warnings(r) == []


def test_load_from_bytes_matches_load_from_file(tmp_path):
    data = np.arange(20.0).reshape(4, 5)
    path = tmp_path / "dtm.tif"
    write_tif(path, data)
    a = io.load_raster(path)
    b = io.load_raster_bytes(path.read_bytes())
    np.testing.assert_array_equal(a.data, b.data)


def test_geographic_crs_gives_warning(tmp_path):
    path = tmp_path / "latlon.tif"
    write_tif(path, np.zeros((4, 4)), cell=0.0001, crs="EPSG:4326")
    assert io.unit_warnings(io.load_raster(path))


def test_resample_averages_blocks_and_ignores_nodata(tmp_path):
    # 4x4 grid of 1 cm cells -> 2x2 grid of 2 cm cells.
    data = np.array(
        [
            [1, 3, 10, 10],
            [1, 3, 10, -9999],
            [0, 0, 5, 5],
            [0, 4, 5, 5],
        ],
        dtype=float,
    )
    path = tmp_path / "dtm.tif"
    write_tif(path, data, cell=0.01)

    r = io.resample(io.load_raster(path), 0.02)
    assert r.pixel_size == pytest.approx((0.02, 0.02))
    expected = [[2.0, 10.0], [1.0, 5.0]]  # the nodata cell is left out of its average
    np.testing.assert_allclose(r.data, expected)


def test_resample_never_makes_cells_smaller(tmp_path):
    path = tmp_path / "dtm.tif"
    write_tif(path, np.ones((5, 5)), cell=0.10)
    r = io.load_raster(path)
    assert io.resample(r, 0.05) is r


def test_relative_heights_start_at_zero():
    rel = io.relative_heights(np.array([[100.0, 101.5], [np.nan, 100.25]]))
    assert np.nanmin(rel) == 0.0
    assert np.nanmax(rel) == pytest.approx(1.5)


def test_latlon_round_trip():
    x, y = 523456.78, 4512345.67
    lon, lat = io.map_to_latlon(x, y, "EPSG:32618")
    x2, y2 = io.latlon_to_map(lon, lat, "EPSG:32618")
    assert (x2[0], y2[0]) == pytest.approx((x, y), abs=1e-4)


def test_load_image_reads_rgba_and_zoomed_window(tmp_path):
    # 100 x 200 pixel RGBA photo, 1 cm pixels; left half red, right half blue,
    # with a transparent strip along the top.
    rows, cols = 100, 200
    img = np.zeros((4, rows, cols), dtype="uint8")
    img[0, :, :100] = 255
    img[2, :, 100:] = 255
    img[3] = 255
    img[3, :10, :] = 0
    path = tmp_path / "orthophoto.tif"
    with rasterio.open(path, "w", driver="GTiff", height=rows, width=cols, count=4, dtype="uint8",
                       crs=UTM_17N, transform=from_origin(500000.0, 4000000.0, 0.01, 0.01)) as dst:
        dst.write(img)

    rgba, bounds, crs = io.load_image(path, max_size=50)
    assert rgba.shape == (25, 50, 4)  # shrunk so the longest side is 50 pixels
    assert bounds == pytest.approx((500000.0, 3999999.0, 500002.0, 4000000.0))
    assert rgba[-1, 0, :3] == pytest.approx([1, 0, 0])  # bottom-left is red
    assert rgba[-1, -1, :3] == pytest.approx([0, 0, 1])  # bottom-right is blue
    assert rgba[0, 0, 3] == 0 and rgba[-1, 0, 3] == 1  # transparent top strip

    # Zoom to the right half only: everything should be blue.
    zoom = io.load_image(path, bounds=(500001.0, 3999999.0, 500002.0, 4000000.0))
    assert zoom[1] == pytest.approx((500001.0, 3999999.0, 500002.0, 4000000.0))
    assert np.allclose(zoom[0][..., 2], 1.0) and np.allclose(zoom[0][..., 0], 0.0)

    # An area outside the photo gives None.
    assert io.load_image(path, bounds=(600000.0, 3000000.0, 600001.0, 3000001.0)) is None


def test_repair_pits_fixes_a_deep_hole_and_leaves_the_rest_alone():
    # A tilted plane (5 cm rise per meter) at 25 cm cells, 100 m x 100 m, with a 3 m x 3 m pit 40 m deep.
    cell = 0.25
    x, y = local_grid((400, 400), cell)
    plane = 0.05 * x + 0.02 * y
    z = plane.copy()
    pit = (np.abs(x - 50) < 1.5) & (np.abs(y - 50) < 1.5)
    z[pit] -= 40.0
    fixed, changed = io.repair_pits(make_raster(z, cell), max_drop=10.0)
    assert changed[pit].all()
    assert np.abs(fixed.data[pit] - plane[pit]).max() < 0.5  # back to about the plane's height
    far = np.hypot(x - 50, y - 50) > 5
    assert not changed[far].any() and np.array_equal(fixed.data[far], z[far])


def test_repair_pits_keeps_a_wide_real_valley():
    # A V-shaped valley 15 m deep and 120 m wide is real terrain, not a pit.
    cell = 0.5
    x, y = local_grid((300, 300), cell)
    z = 15.0 * np.abs(x - 75) / 60
    fixed, changed = io.repair_pits(make_raster(z, cell), max_drop=10.0)
    assert not changed.any()
    assert np.array_equal(fixed.data, z)
