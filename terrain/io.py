"""Loading and preparing elevation rasters (GeoTIFFs exported from WebODM).

A raster is a grid of numbers. For a DTM, each number is the ground height of
one small square ("cell" or "pixel") of land. To place the grid on a map we
also need its *transform* (where the grid sits and how big each cell is) and
its *CRS* (which map coordinate system the x/y values are in).
"""

import math
import warnings
from dataclasses import dataclass

import numpy as np
import rasterio
from scipy import ndimage
from rasterio.io import MemoryFile
from rasterio.transform import Affine
from rasterio.errors import WindowError
from rasterio.warp import Resampling, reproject
from rasterio.warp import transform as transform_coords
from rasterio.windows import Window, from_bounds
from rasterio.windows import bounds as window_bounds

# WebODM fills empty areas with -9999. If a file forgets to say what its
# "nodata" value is, we still treat this value as "no data".
WEBODM_NODATA = -9999.0


@dataclass
class Raster:
    """An elevation grid plus the information needed to place it on a map."""

    data: np.ndarray  # 2-D float array of heights in meters; NaN = no data
    transform: Affine  # converts (column, row) to map (x, y) in meters
    crs: object  # coordinate reference system, e.g. a UTM zone

    @property
    def pixel_size(self):
        """Cell width and height in map units (meters), read from the transform."""
        # transform.a is the cell width; transform.e is the cell height, which is
        # negative because row numbers increase as you go south.
        return abs(self.transform.a), abs(self.transform.e)

    @property
    def shape(self):
        """(rows, columns) of the grid."""
        return self.data.shape

    @property
    def bounds(self):
        """Map coordinates of the edges: (left, bottom, right, top)."""
        rows, cols = self.shape
        t = self.transform
        left, top = t.c, t.f  # map coordinates of the top-left corner
        right = left + cols * t.a
        bottom = top + rows * t.e  # t.e is negative, so bottom < top
        return left, bottom, right, top


def _read_dataset(src) -> Raster:
    """Read band 1 of an open rasterio dataset and mask missing values as NaN."""
    # masked=True hides cells marked as nodata (by value or by an internal mask).
    band = src.read(1, masked=True).astype("float64")
    data = band.filled(np.nan)

    # Extra safety: never let missing data turn into real-looking heights.
    data[~np.isfinite(data)] = np.nan
    if src.nodata is None:
        data[data <= WEBODM_NODATA] = np.nan

    t = src.transform
    if t.b != 0 or t.d != 0:
        raise ValueError("This raster is rotated. Only north-up rasters are supported.")
    return Raster(data=data, transform=t, crs=src.crs)


def load_raster(path) -> Raster:
    """Load a GeoTIFF from a file path."""
    with rasterio.open(path) as src:
        return _read_dataset(src)


def load_raster_bytes(file_bytes: bytes) -> Raster:
    """Load a GeoTIFF from bytes in memory (used for files uploaded in the app)."""
    with MemoryFile(file_bytes) as memfile, memfile.open() as src:
        return _read_dataset(src)


def unit_warnings(raster: Raster) -> list[str]:
    """Return warnings if the raster is not in a projected CRS measured in meters.

    Slope math needs x, y and z all in meters. Latitude/longitude degrees would
    give nonsense slopes.
    """
    crs = raster.crs
    if crs is None:
        return ["This raster has no coordinate system (CRS). Cell sizes may be wrong."]
    if crs.is_geographic:
        return [
            "This raster uses latitude/longitude (degrees), not meters. "
            "Re-export it from WebODM in its UTM projection."
        ]
    units = (crs.linear_units or "").lower()
    if units not in ("metre", "meter", "m", "metres", "meters"):
        return [f"This raster's map units are '{crs.linear_units}', not meters."]
    return []


def resample(raster: Raster, cell_size: float) -> Raster:
    """Make the grid coarser so each cell is `cell_size` meters wide.

    Each new cell gets the *average* height of the original cells inside it.
    Averaging smooths out tiny bumps (grass, noise) and makes later steps
    faster. If the requested cell is not bigger than the original cell, the
    raster is returned unchanged, because we can't invent extra detail.
    """
    if cell_size <= 0:
        raise ValueError("Cell size must be positive.")
    if cell_size <= max(raster.pixel_size) * (1 + 1e-9):
        return raster
    if raster.crs is None:
        raise ValueError("Cannot resample a raster that has no coordinate system.")

    left, _, _, top = raster.bounds
    px, py = raster.pixel_size
    old_rows, old_cols = raster.shape
    # Round up so the new grid covers everything. The small 1e-6 stops tiny
    # rounding errors (e.g. 0.04 / 0.02 = 2.0000000001) adding an empty row.
    cols = max(1, math.ceil(old_cols * px / cell_size - 1e-6))
    rows = max(1, math.ceil(old_rows * py / cell_size - 1e-6))
    new_transform = Affine(cell_size, 0, left, 0, -cell_size, top)
    new_data = np.full((rows, cols), np.nan, dtype="float64")

    reproject(
        source=raster.data,
        destination=new_data,
        src_transform=raster.transform,
        src_crs=raster.crs,
        dst_transform=new_transform,
        dst_crs=raster.crs,
        src_nodata=np.nan,  # NaN cells are ignored when averaging
        dst_nodata=np.nan,
        resampling=Resampling.average,
    )
    return Raster(data=new_data, transform=new_transform, crs=raster.crs)


def local_ground(data: np.ndarray, cell: float, window_m: float = 30.0, coarse_m: float = 2.0) -> np.ndarray:
    """A smooth "typical ground height" around every cell: the median height
    within `window_m` meters.

    For speed it works on coarse blocks (`coarse_m` wide), then smoothly
    stretches the result back onto the original grid. The median ignores a few
    extreme cells, so one deep hole doesn't pull it down.
    """
    f = max(1, int(round(coarse_m / cell)))
    rows, cols = data.shape
    padded = np.pad(data, ((0, -rows % f), (0, -cols % f)), constant_values=np.nan)
    blocks = padded.reshape(padded.shape[0] // f, f, padded.shape[1] // f, f)
    with warnings.catch_warnings():  # blocks with no data at all give NaN (handled below)
        warnings.simplefilter("ignore", RuntimeWarning)
        coarse = np.nanmedian(blocks, axis=(1, 3))
    valid = np.isfinite(coarse)
    if not valid.any():
        return np.full(data.shape, np.nan)
    # Empty blocks borrow the nearest block's value, so the edge of the data isn't dragged around.
    nearest = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    coarse = coarse[tuple(nearest)]
    size = max(3, int(round(window_m / (f * cell))) | 1)  # odd number of blocks
    med = ndimage.median_filter(coarse, size=size, mode="nearest")
    # Back to the fine grid: each fine cell takes a straight-line blend of the
    # nearest block centers (bilinear), so there are no steps at block edges.
    r = (np.arange(rows) + 0.5) / f - 0.5
    c = (np.arange(cols) + 0.5) / f - 0.5
    rr, cc = np.meshgrid(r, c, indexing="ij")
    return ndimage.map_coordinates(med, [rr, cc], order=1, mode="nearest")


def repair_pits(raster: Raster, max_drop: float = 10.0, window_m: float = 30.0, grow_m: float = 1.0):
    """Fix spots where the ground model suddenly drops far below its surroundings.

    Under thick canopy WebODM sometimes can't see the ground and its ground
    model (DTM) falls into deep "pits", tens of meters below the real ground.
    That makes trees look 60 m tall and the pit walls look like cliffs.
    A cell counts as a pit if it is more than `max_drop` meters below the
    typical ground within `window_m` meters (see local_ground). Pit cells, and
    a `grow_m` rim around them, are replaced by that typical ground height.
    Real valleys are wider than the window, so the typical ground follows them
    and they are left alone.

    Returns (repaired raster, mask of the cells that were changed).
    """
    data = raster.data
    cell = max(raster.pixel_size)
    ground = local_ground(data, cell, window_m)
    with np.errstate(invalid="ignore"):
        pit = np.isfinite(data) & (data < ground - max_drop)
    if not pit.any():
        return raster, pit
    if grow_m > 0:  # the steep rim of a pit is part of the error too
        pit = ndimage.binary_dilation(pit, iterations=max(1, int(round(grow_m / cell)))) & np.isfinite(data)
    fixed = np.where(pit, ground, data)
    return Raster(data=fixed, transform=raster.transform, crs=raster.crs), pit


def relative_heights(data: np.ndarray) -> np.ndarray:
    """Heights relative to the lowest point (lowest point = 0 m).

    Our absolute elevations are off by about a meter (no ground control
    points), but height *differences* are reliable, so we show those.
    """
    if np.all(np.isnan(data)):
        return data.copy()
    return data - np.nanmin(data)


def summarize(raster: Raster) -> dict:
    """Basic facts about a raster for display."""
    px, py = raster.pixel_size
    rows, cols = raster.shape
    valid = np.isfinite(raster.data)
    n_valid = int(valid.sum())
    rel = relative_heights(raster.data)
    return {
        "rows": rows,
        "cols": cols,
        "pixel_x_m": px,
        "pixel_y_m": py,
        "width_m": cols * px,
        "height_m": rows * py,
        "valid_fraction": n_valid / valid.size if valid.size else 0.0,
        "area_m2": n_valid * px * py,  # area that actually has data
        "relief_m": float(np.nanmax(rel)) if n_valid else float("nan"),
        "crs": raster.crs.to_string() if raster.crs is not None else "none",
    }


def latlon_to_map(lon, lat, crs):
    """Convert longitude/latitude (degrees, GPS style) to map x, y in `crs`.

    WebODM sometimes shows coordinates as latitude/longitude. Our math needs
    map coordinates in meters, so we convert them here.
    """
    xs, ys = transform_coords("EPSG:4326", crs, np.atleast_1d(lon), np.atleast_1d(lat))
    return np.asarray(xs), np.asarray(ys)


def map_to_latlon(x, y, crs):
    """The reverse of latlon_to_map: map x, y in `crs` -> (longitude, latitude)."""
    lons, lats = transform_coords(crs, "EPSG:4326", np.atleast_1d(x), np.atleast_1d(y))
    return np.asarray(lons), np.asarray(lats)


def load_image(path, bounds=None, max_size=1600):
    """Read a map photo (orthophoto) for use as a background picture.

    bounds: optional (left, bottom, right, top) in map coordinates to read
        only part of the photo (for zooming in).
    max_size: the picture is shrunk so its longest side is at most this many
        pixels, because orthophotos can be huge.

    Returns (rgba, image_bounds, crs) where rgba is a rows x cols x 4 array of
    values from 0 to 1 (4th channel = transparency), or None if `bounds` does
    not overlap the photo.
    """
    with rasterio.open(path) as src:
        full = Window(0, 0, src.width, src.height)
        if bounds is None:
            window = full
        else:
            try:
                window = from_bounds(*bounds, transform=src.transform).intersection(full)
            except WindowError:
                return None
            window = window.round_offsets().round_lengths()
            if window.width < 1 or window.height < 1:
                return None

        shrink = max(1.0, max(window.width, window.height) / max_size)
        out_h = max(1, int(window.height / shrink))
        out_w = max(1, int(window.width / shrink))

        bands = [1, 2, 3] if src.count >= 3 else [1, 1, 1]  # grayscale -> repeat band 1
        rgb = src.read(bands, window=window, out_shape=(3, out_h, out_w),
                       resampling=Resampling.average).astype("float64")
        # Transparency: WebODM orthophotos have a 4th "alpha" band; otherwise
        # use the file's own mask of valid pixels.
        if src.count >= 4:
            alpha = src.read(4, window=window, out_shape=(out_h, out_w)) > 0
        else:
            alpha = src.dataset_mask(window=window, out_shape=(out_h, out_w)) > 0

        # Scale colors to 0-1. 8-bit photos go 0-255; anything else is
        # stretched between its 2nd and 98th percentile so it isn't too dark.
        if src.dtypes[0] == "uint8":
            rgb /= 255.0
        else:
            lo, hi = np.nanpercentile(rgb[:, alpha], [2, 98]) if alpha.any() else (0.0, 1.0)
            rgb = (rgb - lo) / (hi - lo if hi > lo else 1.0)

        rgba = np.dstack([np.clip(rgb.transpose(1, 2, 0), 0, 1), alpha.astype("float64")])
        return rgba, window_bounds(window, src.transform), src.crs


def resample_to_grid(raster: Raster, target: Raster) -> Raster:
    """Put `raster` onto exactly the same grid (cells) as `target`.

    Needed so two rasters (e.g. DSM and DTM, or two DTMs from different days)
    can be subtracted cell by cell. Each target cell gets the average of the
    source cells that fall inside it.
    """
    out = np.full(target.shape, np.nan, dtype="float64")
    reproject(
        source=raster.data, destination=out,
        src_transform=raster.transform, src_crs=raster.crs,
        dst_transform=target.transform, dst_crs=target.crs,
        src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.average,
    )
    return Raster(data=out, transform=target.transform, crs=target.crs)


def read_rgb_on_grid(path, target: Raster) -> np.ndarray:
    """Read the orthophoto's colors onto the same grid as `target`.

    Returns a rows x cols x 3 array of red, green, blue from 0 to 1, with NaN
    where the photo has no data.

    Fast path (photo in the same coordinate system as the grid, the usual case
    for WebODM): WebODM photos contain smaller "overview" copies (1/2, 1/4, 1/8
    size...). Reading the smallest copy that is still at least twice as sharp as
    the analysis grid, then averaging it onto the grid, gives nearly the same
    colors 5-15x faster than reading the full 5 cm photo.
    """
    with rasterio.open(path) as src:
        same_crs = src.crs == target.crs and src.count >= 3
        factors, photo_cell = (src.overviews(1) if same_crs else []), max(src.res)
    if same_crs:
        return _read_rgb_window(path, target, factors, photo_cell)
    with rasterio.open(path) as src:
        bands = [1, 2, 3] if src.count >= 3 else [1, 1, 1]
        rgb = np.full((3, *target.shape), np.nan)
        for i, band in enumerate(bands):
            reproject(
                source=rasterio.band(src, band), destination=rgb[i],
                dst_transform=target.transform, dst_crs=target.crs,
                dst_nodata=np.nan, resampling=Resampling.average,
            )
        coverage = np.zeros(target.shape)
        if src.count >= 4:  # 4th "alpha" band: 0 = no photo here
            reproject(source=rasterio.band(src, 4), destination=coverage,
                      dst_transform=target.transform, dst_crs=target.crs, resampling=Resampling.average)
            has_photo = coverage > 127
        else:
            has_photo = np.all(np.isfinite(rgb), axis=0)
        scale = 255.0 if src.dtypes[0] == "uint8" else float(np.nanmax(rgb) or 1.0)
    rgb = rgb / scale
    rgb[:, ~has_photo] = np.nan
    return np.clip(rgb.transpose(1, 2, 0), 0, 1)


def _read_rgb_window(path, target: Raster, factors, photo_cell) -> np.ndarray:
    """read_rgb_on_grid's fast path: one windowed, averaged read of all bands."""
    cell = max(target.pixel_size)
    # The biggest overview whose cells are still at most half an analysis cell (None = full photo).
    level = max([i for i, k in enumerate(factors) if photo_cell * k <= cell / 2], default=None)
    with rasterio.open(path, overview_level=level) as src:
        window = from_bounds(*target.bounds, transform=src.transform)
        bands = [1, 2, 3, 4] if src.count >= 4 else [1, 2, 3]
        # out_shape = the analysis grid; "average" = each grid cell gets the mean of the photo pixels inside it.
        # boundless + fill 0 = parts of the grid outside the photo count as "no photo".
        arr = src.read(bands, window=window, out_shape=(len(bands), *target.shape),
                       resampling=Resampling.average, boundless=True, fill_value=0).astype("float64")
        scale = 255.0 if src.dtypes[0] == "uint8" else float(arr[:3].max() or 1.0)
    has_photo = arr[3] > 127 if len(bands) == 4 else np.any(arr[:3] > 0, axis=0)  # 4th "alpha" band: 0 = no photo
    rgb = arr[:3] / scale
    rgb[:, ~has_photo] = np.nan
    return np.clip(rgb.transpose(1, 2, 0), 0, 1)


def geotiff_bytes(data: np.ndarray, raster: Raster) -> bytes:
    """Save a grid (same cells as `raster`) as a GeoTIFF file in memory, for downloading."""
    with MemoryFile() as memfile:
        with memfile.open(driver="GTiff", height=data.shape[0], width=data.shape[1], count=1,
                          dtype="float32", crs=raster.crs, transform=raster.transform, nodata=np.nan) as dst:
            dst.write(np.asarray(data, dtype="float32"), 1)
        return memfile.read()
