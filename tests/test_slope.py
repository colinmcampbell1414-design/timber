"""Slope tests using synthetic DEMs whose correct answers we know exactly."""

import numpy as np
import pytest

from terrain import slope


def make_tilted_plane(angle_deg, rows=40, cols=50, cell_x=0.05, cell_y=0.05):
    """A flat ramp rising toward the east at `angle_deg` degrees.

    Height = (distance east) * tan(angle), so the slope everywhere is `angle_deg`.
    """
    x = np.arange(cols) * cell_x
    z_row = x * np.tan(np.radians(angle_deg))
    return np.tile(z_row, (rows, 1))


@pytest.mark.parametrize("angle", [1.0, 10.0, 30.0, 45.0, 60.0])
def test_tilted_plane_has_constant_known_slope(angle):
    z = make_tilted_plane(angle)
    result = slope.slope_degrees(z, 0.05, 0.05)
    np.testing.assert_allclose(result, angle, atol=1e-9)


def test_tilted_plane_north_south_with_non_square_cells():
    # Ramp rising toward the NORTH (row 0 is the north edge), with cells that
    # are 5 cm wide but 10 cm tall, to check we use the right spacing per axis.
    angle, rows, cell_y = 20.0, 30, 0.10
    north_distance = (rows - 1 - np.arange(rows)) * cell_y
    z = np.tile((north_distance * np.tan(np.radians(angle)))[:, None], (1, 25))
    np.testing.assert_allclose(slope.slope_degrees(z, 0.05, cell_y), angle, atol=1e-9)
    # Ground rising to the north faces (slopes down toward) the south: 180 degrees.
    np.testing.assert_allclose(slope.aspect_degrees(z, 0.05, cell_y), 180.0, atol=1e-9)


def test_aspect_of_east_rising_plane_faces_west():
    z = make_tilted_plane(15.0)
    np.testing.assert_allclose(slope.aspect_degrees(z, 0.05, 0.05), 270.0, atol=1e-9)


def test_flat_plane_has_zero_slope():
    z = np.full((30, 30), 123.4)  # absolute height must not matter
    result = slope.slope_degrees(z, 0.05, 0.05)
    assert np.all(result == 0.0)
    # Flat ground faces no direction.
    assert np.all(np.isnan(slope.aspect_degrees(z, 0.05, 0.05)))
    # Flat ground under a 45-degree sun is lit at cos(45 deg) everywhere.
    shade = slope.hillshade(z, 0.05, 0.05, altitude=45.0)
    np.testing.assert_allclose(shade, np.cos(np.radians(45.0)))


def test_slope_does_not_depend_on_cell_size_for_a_plane():
    # The same 25-degree ramp sampled with 5 cm and 50 cm cells.
    for cell in (0.05, 0.5):
        z = make_tilted_plane(25.0, cell_x=cell, cell_y=cell)
        np.testing.assert_allclose(slope.slope_degrees(z, cell, cell), 25.0, atol=1e-9)


def test_nodata_stays_nodata():
    z = make_tilted_plane(10.0)
    z[5, 5] = np.nan
    result = slope.slope_degrees(z, 0.05, 0.05)
    assert np.isnan(result[5, 5])
    # Cells far from the hole are still correct.
    np.testing.assert_allclose(result[20, 20:30], 10.0, atol=1e-9)


def test_hillshade_brightest_when_slope_faces_sun():
    z = make_tilted_plane(30.0)  # faces west (270 degrees)
    toward = slope.hillshade(z, 0.05, 0.05, azimuth=270.0)
    away = slope.hillshade(z, 0.05, 0.05, azimuth=90.0)
    assert toward.mean() > away.mean()


def test_classify_slope_and_table():
    s = np.array([[0.0, 4.9, 5.0, 14.0], [20.0, 30.0, 40.0, np.nan]])
    classes = slope.classify_slope(s, (5, 15, 25, 35))
    np.testing.assert_array_equal(classes, [[0, 0, 1, 1], [2, 3, 4, -1]])

    table = slope.slope_class_table(s, (5, 15, 25, 35), cell_area_m2=0.5)
    assert table["Area (m²)"].tolist() == [1.0, 1.0, 0.5, 0.5, 0.5]
    assert table["Percent"].sum() == pytest.approx(100.0)


def test_breaks_must_increase():
    with pytest.raises(ValueError):
        slope.classify_slope(np.zeros((3, 3)), (10, 5))


def test_cast_shadow_behind_a_wall():
    z = np.zeros((50, 10))
    z[30, :] = 2.0  # a 2 m wall across the map; the sun comes from the south (bigger row numbers)
    shadow = slope.cast_shadow_from_south(z, 0.1, altitude=45.0)
    # At 45 degrees the shadow is as long as the wall is tall: 2 m = 20 cells north of it.
    assert shadow[11:30, :].all()
    assert not shadow[:9, :].any()
    assert not shadow[31:, :].any()  # nothing shaded on the sunny side
