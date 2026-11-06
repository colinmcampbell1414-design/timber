"""Tests for the Hazards & Applications modules, using synthetic data with known answers."""

import numpy as np
import pytest

from conftest import CELL, FakeStreamlit, local_grid, make_raster, to_map
from terrain import apps
from terrain.apps import base


def run_app(app, layers, **overrides):
    """Run an app with its default slider values (plus any overrides by label)."""
    p = app.params(FakeStreamlit(overrides))
    return app.run(layers, p)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def test_every_app_has_the_shared_interface():
    assert len(apps.APPS) >= 1
    for app in apps.APPS:
        for attr in apps.REQUIRED_ATTRIBUTES:
            assert hasattr(app, attr), f"{app.__name__} is missing {attr}"
        assert set(app.REQUIRED_INPUTS) <= {"dtm", "dsm", "ortho"}
        assert 2 <= app.DESCRIPTION.count(".") <= 4  # 2-3 plain sentences
        p = app.params(FakeStreamlit())
        assert isinstance(p, dict)


def test_missing_inputs_give_a_message_not_a_crash():
    layers = base.Layers(dtm=make_raster(np.zeros((20, 20))))
    for app in apps.APPS:
        message = base.missing_inputs(app, layers)
        needs_more = any(i != "dtm" for i in app.REQUIRED_INPUTS)
        if needs_more:
            assert message and "data folder" in message
        else:
            assert message is None


# ---------------------------------------------------------------------------
# 1. Mudslide
# ---------------------------------------------------------------------------
def test_mudslide_steep_slope_scores_higher_than_flat():
    from terrain.apps import mudslide

    rows, cols = 40, 60
    x, _ = local_grid((rows, cols))
    # Left half flat, right half a 35-degree slope rising to the east.
    z = np.where(x < 3.0, 0.0, (x - 3.0) * np.tan(np.radians(35)))
    layers = base.Layers(dtm=make_raster(z))
    result = run_app(mudslide, layers)
    score = result.overlays[1].data
    flat = np.nanmean(score[:, 2:25])
    steep = np.nanmean(score[:, 35:58])
    assert steep > flat + 0.2
    classes = result.overlays[0].data
    # A dry, straight 35-degree slope alone is moderate (high needs wet or water-collecting ground too);
    # flat ground is low.
    assert np.mean(classes[:, 35:58] >= 1) > 0.9
    assert np.all(classes[:, 2:25] == 0)


# ---------------------------------------------------------------------------
# 2. Trees
# ---------------------------------------------------------------------------
def cone(x, y, cx, cy, height, radius):
    """A cone-shaped 'tree': `height` at the center, 0 at `radius` and beyond."""
    return np.clip(height * (1 - np.hypot(x - cx, y - cy) / radius), 0, None)


def test_trees_heights_and_fall_zone_targets():
    from terrain.apps import trees

    rows, cols = 200, 300  # 30 m x 20 m at 10 cm
    x, y = local_grid((rows, cols))
    ground = 0.02 * x  # gently sloping ground; must not affect tree heights
    # Two trees 12 m apart: 8 m tall on the left, 5 m tall on the right.
    # Their tops sit exactly on cell centers.
    t8, t5 = (8.05, 10.05), (20.05, 10.05)
    canopy = cone(x, y, *t8, 8.0, 3.0) + cone(x, y, *t5, 5.0, 2.5)
    dtm, dsm = make_raster(ground), make_raster(ground + canopy)

    # A target exactly halfway between them: 6 m from each tree.
    tx, ty = to_map(14.05, 10.05, rows)
    layers = base.Layers(dtm=dtm, dsm=dsm, shapes={"targets": [{"label": "shed", "xy": [(tx, ty)]}]})
    result = run_app(trees, layers)
    table = result.table.sort_values("height (m)").reset_index(drop=True)

    assert len(table) == 2
    assert table.loc[0, "height (m)"] == pytest.approx(5.0, rel=0.05)
    assert table.loc[1, "height (m)"] == pytest.approx(8.0, rel=0.05)
    # Fall reach = height: the 8 m tree reaches the shed 6 m away, the 5 m tree doesn't.
    assert not table.loc[0, "can reach marked target"]
    assert table.loc[1, "can reach marked target"]
    # The tree that can reach the target is ranked first.
    assert result.table.loc[0, "height (m)"] == pytest.approx(8.0, rel=0.05)

    # A target drawn as an outline (a shed footprint) instead of a point also works.
    shed = [to_map(*v, rows) for v in [(13.5, 9.5), (14.5, 9.5), (14.5, 10.5), (13.5, 10.5)]]
    outline_layers = base.Layers(dtm=dtm, dsm=dsm, shapes={"targets": [shed]})
    flagged = run_app(trees, outline_layers).table["can reach marked target"]
    assert flagged.sum() == 1
