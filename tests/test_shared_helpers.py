"""Tests for helpers shared by the Phase 5 apps: vegetation indices, curvature, TWI, geometry."""

import numpy as np
import pytest

from conftest import local_grid, make_raster, to_map
from terrain import hydrology, slope, vegetation
from terrain.apps import base


def test_vari_and_exg_known_colors():
    rgb = np.array([[[0.2, 0.6, 0.2],     # green grass
                     [0.6, 0.45, 0.3],    # brown soil
                     [0.5, 0.5, 0.5],     # gray pavement
                     [0.0, 0.0, 0.0]]])   # black: VARI would divide by 0
    v, e = vegetation.vari(rgb), vegetation.exg(rgb)
    assert v[0, 0] == pytest.approx((0.6 - 0.2) / (0.6 + 0.2 - 0.2))
    assert v[0, 1] < 0  # browner than green
    assert v[0, 2] == pytest.approx(0.0)
    assert np.isnan(v[0, 3])  # safe division: no crash, just "unknown"
    assert e[0, 0] == pytest.approx(0.8) and e[0, 1] < 0.05 and e[0, 2] == pytest.approx(0.0)


def test_plan_curvature_sign_valley_negative_ridge_positive():
    x, y = local_grid((60, 60))
    valley = 0.3 * (x - 3.0) ** 2 + 0.2 * y  # U-shaped valley running north-south
    ridge = -0.3 * (x - 3.0) ** 2 + 0.2 * y
    inner = (slice(10, 50), slice(10, 50))
    assert np.all(slope.plan_curvature(valley, 0.1, 0.1)[inner] < 0)  # concave: water collects
    assert np.all(slope.plan_curvature(ridge, 0.1, 0.1)[inner] > 0)  # convex: water spreads
    assert np.all(slope.plan_curvature(np.zeros((20, 20)), 0.1, 0.1) == 0)  # flat: no crash


def test_twi_is_higher_in_valley_bottom():
    x, y = local_grid((40, 41))
    z = 0.5 * np.abs(x - 2.05) + 0.1 * y
    flow = hydrology.analyze_flow(z, 0.1, 0.1)
    twi = hydrology.topographic_wetness_index(flow["acc_area_m2"], slope.slope_degrees(z, 0.1, 0.1), 0.1)
    assert twi[30, 20] > twi[30, 5]  # valley line wetter than the valley side


def test_distance_to_shapes_point_line_polygon():
    square = [(0, 0), (2, 0), (2, 2), (0, 2)]
    assert base.distance_to_shape(1, 1, square) == 0  # inside
    assert base.distance_to_shape(5, 1, square) == pytest.approx(3)
    assert base.distance_to_shape(3, 4, [(0, 0)]) == pytest.approx(5)  # a single point
    assert base.distance_to_shape(1, 1, [(0, 0), (2, 0)], closed=False) == pytest.approx(1)  # a line


def test_polygon_mask_and_rank_normalize():
    r = make_raster(np.zeros((10, 10)))
    poly = [to_map(*v, 10) for v in [(0.2, 0.2), (0.6, 0.2), (0.6, 0.6), (0.2, 0.6)]]
    assert base.polygon_mask(r, poly).sum() == 16  # 4 x 4 cells of 10 cm
    ranks = base.rank_normalize(np.array([3.0, 1.0, np.nan, 2.0]))
    np.testing.assert_allclose(ranks, [1.0, 0.0, np.nan, 0.5])


def test_layers_remember_saves_and_reuses_results(tmp_path):
    calls = []

    def slow():
        calls.append(1)
        return {"answer": 42}

    layers = base.Layers(dtm=make_raster(np.zeros((5, 5))), cache_dir=tmp_path, cache_key="survey-1")
    assert layers.remember("demo", {"a": 1}, slow) == {"answer": 42}
    assert layers.remember("demo", {"a": 1}, slow) == {"answer": 42}  # loaded from disk
    assert len(calls) == 1
    layers.remember("demo", {"a": 2}, slow)  # new settings: worked out again
    other = base.Layers(dtm=make_raster(np.zeros((5, 5))), cache_dir=tmp_path, cache_key="survey-2")
    other.remember("demo", {"a": 1}, slow)  # another survey: worked out again
    assert len(calls) == 3
    no_cache = base.Layers(dtm=make_raster(np.zeros((5, 5))))
    no_cache.remember("demo", {"a": 1}, slow)  # no cache folder: always worked out
    assert len(calls) == 4
