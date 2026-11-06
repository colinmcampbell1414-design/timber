"""Water flow: fill depressions, flow direction, flow accumulation and ponding.

This is a simple "D8" model: water in each cell flows to ONE of its 8
neighbors, the one that is steepest downhill. Written with numpy and plain
Python (instead of the pysheds library) so every step can be explained.

Grid conventions: row 0 is the north edge, column 0 is the west edge.
Cells with no data are NaN. Water reaching the edge of the map, or the edge
of a no-data area, is assumed to flow away (we can't see where it goes).
"""

import heapq

import numpy as np
import pandas as pd
from scipy import ndimage

# The 8 neighbors of a cell as (row step, column step).
NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _outlet_cells(valid: np.ndarray) -> np.ndarray:
    """Valid cells where water can leave the map: on the edge or next to no-data."""
    padded = np.pad(valid, 1, constant_values=False)
    # A cell is "inside" only if all 8 neighbors are valid.
    inside = np.ones_like(valid)
    for dr, dc in NEIGHBORS:
        inside &= padded[1 + dr : 1 + dr + valid.shape[0], 1 + dc : 1 + dc + valid.shape[1]]
    return valid & ~inside


def fill_depressions(z: np.ndarray):
    """Fill every pit/depression up to the level where water would spill out.

    Method ("priority-flood"): imagine water rising from the edges of the map.
    1. Put all outlet cells (map edge / next to no-data) in a queue.
    2. Always take the LOWEST cell from the queue next.
    3. Look at its neighbors not seen yet. If a neighbor is lower than the
       cell we came from, it is in a depression: raise ("fill") it to that
       level. Then add the neighbor to the queue.
    The result is a surface where water can always run off the map.

    Returns (filled, order, parent):
      filled: the filled heights (NaN where z is NaN)
      order:  cell numbers in the order they were taken from the queue
              (always from low to high)
      parent: for each cell, the neighbor it was reached from (-1 for outlets).
              Used to route water across flat filled areas toward the spill point.
    """
    rows, cols = z.shape
    valid = np.isfinite(z)
    filled = np.where(valid, z, np.nan).astype("float64")
    flat_z = filled.ravel()  # same memory as `filled`: changes show up in both
    seen = ~valid.ravel()  # no-data cells count as already seen
    parent = np.full(z.size, -1, dtype=np.int64)
    order = []

    queue = []
    for idx in np.flatnonzero(_outlet_cells(valid)):
        heapq.heappush(queue, (flat_z[idx], int(idx)))
        seen[idx] = True

    while queue:
        level, idx = heapq.heappop(queue)
        order.append(idx)
        r, c = divmod(idx, cols)
        for dr, dc in NEIGHBORS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                n = nr * cols + nc
                if not seen[n]:
                    seen[n] = True
                    if flat_z[n] < level:
                        flat_z[n] = level  # fill the depression up to the spill level
                    parent[n] = idx
                    heapq.heappush(queue, (flat_z[n], n))
    return filled, np.array(order, dtype=np.int64), parent


def flow_directions(filled: np.ndarray, parent: np.ndarray, cell_x: float, cell_y: float) -> np.ndarray:
    """For each cell, the cell number its water flows to (-1 = flows off the map).

    Normal ground: water goes to the neighbor with the steepest downhill
    slope, where slope = drop in height / distance (diagonals are farther).
    Flat ground (like the top of a filled pond): there is no downhill
    neighbor, so water follows the path the flood came in by, which leads
    to the spill point.
    """
    rows, cols = filled.shape
    padded = np.pad(filled, 1, constant_values=np.nan)
    best_slope = np.zeros(filled.shape)
    receiver = np.full(filled.shape, -1, dtype=np.int64)
    r_idx, c_idx = np.indices(filled.shape)

    for dr, dc in NEIGHBORS:
        neighbor_z = padded[1 + dr : 1 + dr + rows, 1 + dc : 1 + dc + cols]
        distance = np.hypot(dr * cell_y, dc * cell_x)
        with np.errstate(invalid="ignore"):
            drop_per_m = (filled - neighbor_z) / distance
        steeper = np.nan_to_num(drop_per_m, nan=0.0) > best_slope
        best_slope[steeper] = drop_per_m[steeper]
        receiver[steeper] = ((r_idx + dr) * cols + (c_idx + dc))[steeper]

    receiver = receiver.ravel()
    no_downhill = receiver < 0
    receiver[no_downhill] = parent[no_downhill]  # flat areas: follow the flood path
    receiver[~np.isfinite(filled.ravel())] = -1
    return receiver


def flow_accumulation(receiver: np.ndarray, order: np.ndarray, shape) -> np.ndarray:
    """How many cells drain through each cell (counting the cell itself).

    Every cell starts with 1 unit of "rain". Going from the highest cell to
    the lowest (the reverse of the fill order), each cell passes everything it
    has collected to the cell it flows to. Because water only flows to cells
    that come earlier in the fill order, every cell is finished before its
    water is passed on.
    """
    acc = np.zeros(int(np.prod(shape)), dtype="float64")
    acc[order] = 1.0
    for idx in order[::-1]:
        target = receiver[idx]
        if target >= 0:
            acc[target] += acc[idx]
    acc = acc.reshape(shape)
    acc[acc == 0] = np.nan  # no-data cells
    return acc


def remove_tilt(z: np.ndarray, cell_x: float, cell_y: float):
    """Subtract the best-fit flat plane (the overall tilt) from the surface.

    Without ground control points, a drone model can be slightly tilted as a
    whole. Fitting a plane  z = a*x + b*y + c  (least squares: the plane that
    is closest to all cells) and subtracting the tilt part (a*x + b*y) lets
    you see how much the flow results depend on that overall tilt.

    Returns (detrended_z, info) where info has the removed tilt in degrees
    and the compass direction the plane slopes down toward.
    """
    x, y = _local_xy(z.shape, cell_x, cell_y)
    valid = np.isfinite(z)
    A = np.column_stack([x[valid], y[valid], np.ones(valid.sum())])
    (a, b, _), *_ = np.linalg.lstsq(A, z[valid], rcond=None)
    info = {
        "east_slope": float(a),  # meters of rise per meter going east
        "north_slope": float(b),  # meters of rise per meter going north
        "tilt_deg": float(np.degrees(np.arctan(np.hypot(a, b)))),
        "downhill_toward_deg": float(np.degrees(np.arctan2(-a, -b)) % 360),
    }
    detrended = z - tilt_plane(z.shape, cell_x, cell_y, info)
    return detrended, info


def _local_xy(shape, cell_x: float, cell_y: float):
    """Meters east and north of the bottom-left cell, for every cell."""
    rows, _ = shape
    r, c = np.indices(shape)
    return c * cell_x, (rows - 1 - r) * cell_y


def tilt_plane(shape, cell_x: float, cell_y: float, info: dict) -> np.ndarray:
    """The tilt found by remove_tilt as a grid, so the same tilt can be removed
    from another grid of the same shape (e.g. the DSM, so DSM - DTM stays right)."""
    x, y = _local_xy(shape, cell_x, cell_y)
    return info["east_slope"] * x + info["north_slope"] * y


def specific_catchment_area(acc_area_m2, cell_size: float):
    """Uphill area draining through each cell per meter of width (m²/m).

    Flow accumulation gives the AREA draining through a cell. Dividing by the
    cell width (how wide the "door" is that the water passes through) makes
    the number independent of cell size. This is the standard input for TWI
    and SPI.
    """
    return acc_area_m2 / cell_size


def topographic_wetness_index(acc_area_m2, slope_deg, cell_size: float):
    """TWI = ln(a / tan(slope)): how wet the ground tends to stay.

    a = specific catchment area. Lots of water arriving (big a) on flat
    ground (small tan(slope)) means water lingers there: high TWI = wetter.
    tan(slope) is kept above 0.001 so flat cells don't divide by zero.
    """
    a = specific_catchment_area(acc_area_m2, cell_size)
    tan_slope = np.maximum(np.tan(np.radians(slope_deg)), 0.001)
    return np.log(a / tan_slope)


def stream_power_index(acc_area_m2, slope_deg, cell_size: float):
    """SPI = ln(a x tan(slope)): how much power flowing water has to cut soil.

    Lots of water (big a) moving down steep ground (big tan(slope)) = high SPI.
    tan(slope) is kept above 0.001 so the log never gets zero.
    """
    a = specific_catchment_area(acc_area_m2, cell_size)
    tan_slope = np.maximum(np.tan(np.radians(slope_deg)), 0.001)
    return np.log(a * tan_slope)


def find_ponds(depth: np.ndarray, min_depth: float, cell_area: float, min_area: float = 0.0):
    """Group neighboring ponding cells into numbered ponds.

    A cell is "ponding" if the filled surface is more than `min_depth` meters
    above the ground there. Touching ponding cells (including diagonally)
    form one pond. Ponds smaller than `min_area` m² are ignored.

    Returns (labels, table): labels is a grid with the pond number in each
    cell (0 = no pond); table has one row per pond, biggest volume first.
    """
    wet = np.nan_to_num(depth, nan=0.0) > min_depth
    labels, count = ndimage.label(wet, structure=np.ones((3, 3)))
    if count == 0:
        return labels, pd.DataFrame(columns=["pond", "area_m2", "max_depth_m", "mean_depth_m", "volume_m3", "row", "col"])

    ids = np.arange(1, count + 1)
    cells = ndimage.sum(wet, labels, ids)
    max_depth = ndimage.maximum(depth, labels, ids)
    mean_depth = ndimage.mean(depth, labels, ids)
    centers = ndimage.center_of_mass(wet, labels, ids)
    table = pd.DataFrame({
        "label": ids,
        "area_m2": cells * cell_area,
        "max_depth_m": max_depth,
        "mean_depth_m": mean_depth,
        "volume_m3": cells * cell_area * mean_depth,  # area x average depth
        "row": [rc[0] for rc in centers],
        "col": [rc[1] for rc in centers],
    })
    table = table[table["area_m2"] >= min_area].sort_values("volume_m3", ascending=False)

    # Renumber ponds 1, 2, 3 ... from biggest volume to smallest.
    # A lookup table does it in one step: new_id_of[old label] = new label.
    # (Looping over patches with `labels == id` would scan the whole map once per
    # patch, which takes minutes on a big map with thousands of patches.)
    new_id_of = np.zeros(int(labels.max()) + 1, dtype=labels.dtype)
    new_id_of[table["label"].to_numpy()] = np.arange(1, len(table) + 1)
    new_labels = new_id_of[labels]
    table.insert(0, "pond", np.arange(1, len(table) + 1))
    return new_labels, table.drop(columns="label").reset_index(drop=True)


def suggest_path_threshold(acc_area: np.ndarray, options, max_fraction: float = 0.02) -> float:
    """Starting value for the "flow path" threshold: the smallest option that
    marks at most `max_fraction` of the map as flow paths.

    On a small yard at 5 cm, 1 m² draining through a cell is a real little
    rivulet. On a 25 cm forest map the same 1 m² is only 16 cells, so a third
    of the map would be "flow paths" and the map would be one big tangle.
    """
    a = np.nan_to_num(acc_area, nan=0.0)
    valid = max(int(np.isfinite(acc_area).sum()), 1)
    for t in sorted(options):
        if (a >= t).sum() / valid <= max_fraction:
            return t
    return max(options)


def flow_segments(receiver: np.ndarray, mask: np.ndarray, cell_x: float, cell_y: float) -> np.ndarray:
    """Line pieces from each masked cell's center to the center of the cell it flows to.

    Coordinates are meters east of the left edge and north of the bottom edge,
    matching the app's maps. Returns an array of shape (n, 2, 2).
    """
    rows, cols = mask.shape
    src = np.flatnonzero(mask.ravel())
    dst = receiver[src]
    keep = dst >= 0
    src, dst = src[keep], dst[keep]
    sr, sc = np.divmod(src, cols)
    dr, dc = np.divmod(dst, cols)
    start = np.column_stack([(sc + 0.5) * cell_x, (rows - sr - 0.5) * cell_y])
    end = np.column_stack([(dc + 0.5) * cell_x, (rows - dr - 0.5) * cell_y])
    return np.stack([start, end], axis=1)


# ---------------------------------------------------------------------------
# Fast versions for big maps (millions of cells)
# ---------------------------------------------------------------------------
# The functions above handle one cell at a time in plain Python, which is easy
# to follow but takes minutes on a forest map with ~5 million cells. These do
# the same three steps with whole-array operations.

def fill_depressions_fast(z: np.ndarray) -> np.ndarray:
    """Same result as fill_depressions, using "morphological reconstruction".

    Start with every cell as high as the highest point in the map, except
    the outlet cells (map edge / next to no-data), which keep their real
    height. Then repeatedly lower each cell to the higher of (its own ground)
    and (its lowest neighbor's current level). Water levels "drain" in from
    the outlets until nothing changes; every dip is left filled to its spill
    level. scikit-image does this in fast compiled code.
    """
    from skimage.morphology import reconstruction

    valid = np.isfinite(z)
    if not valid.any():
        return np.full(z.shape, np.nan)
    low = np.nanmin(z) - 1.0
    ground = np.where(valid, z, low)  # no-data acts like a low drain, the same as the map edge
    seed = np.full(z.shape, np.nanmax(z))
    drains = _outlet_cells(valid) | ~valid
    seed[drains] = ground[drains]
    filled = reconstruction(seed, ground, method="erosion")  # 8 neighbors, like D8
    filled[~valid] = np.nan
    return filled


def _route_flats(filled: np.ndarray, receiver: np.ndarray, outlets: np.ndarray) -> np.ndarray:
    """Give flat cells (no downhill neighbor) a direction toward where the flat drains.

    Neighboring flat cells always have exactly the same height. Starting from
    the cells that already drain (or are outlets), we step into the flat one
    ring at a time ("breadth-first search"); each newly reached flat cell
    sends its water to the cell it was reached from. So water on the flat
    top of a filled pond heads to the spill point by the shortest route.
    """
    rows, cols = filled.shape
    wide = cols + 2  # work on a grid with a 1-cell border of no-data, so neighbors never fall off
    padded = np.pad(filled, 1, constant_values=np.nan).ravel()
    recv_p = np.full(padded.size, -1, dtype=np.int64)
    inner = (np.arange(rows)[:, None] + 1) * wide + (np.arange(cols)[None, :] + 1)  # padded index of each cell
    has_recv = receiver.reshape(rows, cols) >= 0
    flat = np.isfinite(filled) & ~has_recv & ~outlets
    unresolved = np.zeros(padded.size, dtype=bool)
    unresolved[inner[flat]] = True
    if not unresolved.any():
        return receiver
    # Start from draining cells that touch a flat cell.
    drains = np.isfinite(filled) & (has_recv | outlets)
    touching = ndimage.binary_dilation(flat, structure=np.ones((3, 3)))
    frontier = inner[drains & touching]
    offsets = [dr * wide + dc for dr, dc in NEIGHBORS]
    while frontier.size:
        reached = []
        for off in offsets:
            nb = frontier + off
            ok = unresolved[nb] & (padded[nb] == padded[frontier])  # same height = same flat
            if ok.any():
                cells, first = np.unique(nb[ok], return_index=True)
                recv_p[cells] = frontier[ok][first]
                unresolved[cells] = False
                reached.append(cells)
        frontier = np.concatenate(reached) if reached else np.array([], dtype=np.int64)

    # Convert padded cell numbers back to normal ones.
    out = receiver.copy()
    flat_idx = np.flatnonzero(flat.ravel())
    targets = recv_p[inner.ravel()[flat_idx]]
    got = targets >= 0
    tr, tc = np.divmod(targets[got], wide)
    out[flat_idx[got]] = (tr - 1) * cols + (tc - 1)
    return out


def flow_accumulation_fast(receiver: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Same result as flow_accumulation, passing water downhill in batches.

    Count how many cells drain INTO each cell. Cells with none (hilltops,
    ridges) are finished first: they pass their water on, and their
    receivers' counts go down. Any cell whose count reaches 0 is finished
    next, and so on down to the outlets. Each round handles a whole batch of
    cells at once instead of one at a time.
    """
    n = receiver.size
    flat_valid = valid.ravel()
    acc = flat_valid.astype("float64")
    has_target = receiver >= 0
    waiting = np.bincount(receiver[has_target], minlength=n)  # number of uphill neighbors draining in
    frontier = np.flatnonzero(flat_valid & (waiting == 0))
    while frontier.size:
        src = frontier[has_target[frontier]]
        dst = receiver[src]
        np.add.at(acc, dst, acc[src])
        np.subtract.at(waiting, dst, 1)
        candidates = np.unique(dst)
        frontier = candidates[waiting[candidates] == 0]
    acc = acc.reshape(valid.shape)
    acc[~valid] = np.nan
    return acc


def analyze_flow(z: np.ndarray, cell_x: float, cell_y: float, detrend: bool = False, method: str = "auto") -> dict:
    """Run the whole water flow analysis on one elevation grid.

    method: "simple" = the step-by-step version above (easy to follow),
            "fast"   = the whole-array version for big maps,
            "auto"   = simple up to 1 million cells, fast above that.
    Both give the same filled surface and the same flow on sloping ground.
    """
    tilt = None
    if detrend:
        z, tilt = remove_tilt(z, cell_x, cell_y)
    if method == "fast" or (method == "auto" and z.size > 1_000_000):
        valid = np.isfinite(z)
        filled = fill_depressions_fast(z)
        receiver = flow_directions(filled, np.full(z.size, -1, dtype=np.int64), cell_x, cell_y)
        receiver = _route_flats(filled, receiver, _outlet_cells(valid))
        acc_cells = flow_accumulation_fast(receiver, valid)
    else:
        filled, order, parent = fill_depressions(z)
        receiver = flow_directions(filled, parent, cell_x, cell_y)
        acc_cells = flow_accumulation(receiver, order, z.shape)
    return {
        "z": z,
        "filled": filled,
        "depth": filled - z,  # how deep water would pond in each cell
        "receiver": receiver,
        "acc_cells": acc_cells,
        "acc_area_m2": acc_cells * cell_x * cell_y,  # area of ground draining through each cell
        "tilt": tilt,
    }
