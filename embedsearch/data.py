"""Sentinel-2 L2A composite, ESA WorldCover labels and chip tiling.

Imagery and labels come from the Microsoft Planetary Computer STAC API
(signed asset URLs). Both rasters are loaded onto one shared UTM grid so
that every image chip has an exactly aligned label chip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Sentinel-2 L2A bands available on Planetary Computer, in model order
# minus B10 (cirrus), which L2A products do not include.
L2A_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
             "B08", "B8A", "B09", "B11", "B12"]

# Scene Classification Layer values treated as clear: vegetation (4),
# bare soil (5), water (6), unclassified (7). Cloud, shadow, cirrus,
# saturated and no-data pixels are masked before compositing.
SCL_CLEAR = (4, 5, 6, 7)

WORLDCOVER_CLASSES = {
    10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland",
    50: "Built-up", 60: "Bare / sparse", 70: "Snow and ice",
    80: "Permanent water", 90: "Herbaceous wetland", 95: "Mangroves",
    100: "Moss and lichen",
}

# Official WorldCover legend colours.
WORLDCOVER_COLORS = {
    10: "#006400", 20: "#ffbb22", 30: "#ffff4c", 40: "#f096ff",
    50: "#fa0000", 60: "#b4b4b4", 70: "#f0f0f0", 80: "#0064c8",
    90: "#0096a0", 95: "#00cf75", 100: "#fae6a0",
}


# --------------------------------------------------------------------------
# Tiling (pure numpy, unit tested)
# --------------------------------------------------------------------------

def chip_grid(height: int, width: int, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Row and column indices (in chip units) of every full, non-overlapping chip."""
    n_rows, n_cols = height // size, width // size
    rows, cols = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
    return rows.ravel(), cols.ravel()


def tile(arr: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut a (C, H, W) or (H, W) array into non-overlapping size x size chips.

    Edge pixels that do not fill a whole chip are dropped. Returns
    (chips, rows, cols) where chips has shape (N, C, size, size) (or
    (N, size, size) for 2-D input) in row-major order.
    """
    squeeze = arr.ndim == 2
    if squeeze:
        arr = arr[None]
    c, h, w = arr.shape
    n_r, n_c = h // size, w // size
    a = arr[:, : n_r * size, : n_c * size]
    a = a.reshape(c, n_r, size, n_c, size).transpose(1, 3, 0, 2, 4)
    chips = a.reshape(n_r * n_c, c, size, size)
    rows, cols = chip_grid(h, w, size)
    if squeeze:
        chips = chips[:, 0]
    return chips, rows, cols


def dominant_class(label_chips: np.ndarray, nodata: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Most frequent class per chip and the share of the chip it covers."""
    n = label_chips.shape[0]
    flat = label_chips.reshape(n, -1)
    classes = np.unique(flat)
    classes = classes[classes != nodata]
    counts = np.stack([(flat == c).sum(1) for c in classes], axis=1)
    best = counts.argmax(1)
    return classes[best], counts[np.arange(n), best] / flat.shape[1]


# --------------------------------------------------------------------------
# Download (network; cached as GeoTIFFs in data/)
# --------------------------------------------------------------------------

def make_geobox(center_lon: float, center_lat: float, size_km: float,
                epsg: int, resolution: float = 10.0):
    """Square UTM grid of side size_km centred on a lon/lat point."""
    from odc.geo.geobox import GeoBox
    from pyproj import Transformer

    x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(center_lon, center_lat)
    half = size_km * 500.0
    x0 = np.floor((x - half) / resolution) * resolution
    y0 = np.floor((y - half) / resolution) * resolution
    n = int(round(size_km * 1000 / resolution))
    return GeoBox.from_bbox((x0, y0, x0 + n * resolution, y0 + n * resolution),
                            crs=f"EPSG:{epsg}", resolution=resolution)


def _catalog():
    import planetary_computer
    import pystac_client

    return pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)


def select_s2_scenes(geobox, start: str, end: str, max_cloud: float, n_days: int):
    """Pick the n_days clearest acquisition days that best cover the area.

    Days are ranked first by how many MGRS tiles were acquired over the
    box that day (full coverage first), then by mean scene cloud cover.
    Returns (days, items, offset) where offset is the +1000 DN added to
    L2A products from processing baseline 04.00 onward.
    """
    bbox = geobox.geographic_extent.boundingbox
    items = _catalog().search(
        collections=["sentinel-2-l2a"],
        bbox=(bbox.left, bbox.bottom, bbox.right, bbox.top),
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
    ).item_collection()
    if len(items) == 0:
        raise RuntimeError("No Sentinel-2 scenes matched the search.")
    by_day: dict[str, list] = {}
    for it in items:
        by_day.setdefault(it.datetime.strftime("%Y-%m-%d"), []).append(it)
    ranked = sorted(by_day.items(),
                    key=lambda kv: (-len(kv[1]),
                                    np.mean([i.properties["eo:cloud_cover"] for i in kv[1]])))
    days = sorted(ranked[:n_days])
    chosen = [it for _, its in days for it in its]
    baselines = {float(it.properties.get("s2:processing_baseline", "0")) for it in chosen}
    if min(baselines) < 4.0 <= max(baselines):
        raise RuntimeError("Mixed processing baselines; offset handling would be ambiguous.")
    offset = 1000 if min(baselines) >= 4.0 else 0
    return [d for d, _ in days], chosen, offset


def s2_metadata(days, items, offset) -> dict:
    return {
        "collection": "sentinel-2-l2a",
        "days": days,
        "items": [it.id for it in items],
        "scene_cloud_cover": {it.id: it.properties["eo:cloud_cover"] for it in items},
        "processing_baselines": sorted({it.properties["s2:processing_baseline"] for it in items}),
        "offset_removed": offset,
    }


def build_s2_composite(geobox, start: str, end: str, max_cloud: float,
                       n_days: int, out_path: Path) -> dict:
    """Per-pixel median of the clearest acquisition days, cloud masked by SCL.

    Written as uint16 surface reflectance x 10000 with the baseline
    04.00+ offset removed; pixels never observed clear are 0.
    """
    import odc.stac
    import rioxarray  # noqa: F401  (registers .rio)

    days, chosen, offset = select_s2_scenes(geobox, start, end, max_cloud, n_days)
    ds = odc.stac.load(chosen, bands=L2A_BANDS + ["SCL"], geobox=geobox,
                       groupby="solar_day", chunks={"x": 2048, "y": 2048},
                       resampling={"SCL": "nearest", "*": "bilinear"})
    clear = ds["SCL"].isin(list(SCL_CLEAR))
    refl = ds[L2A_BANDS].to_array("band").astype("float32")
    refl = refl.where(clear & (refl > 0))
    comp = (refl - offset).clip(min=0).median("time", skipna=True)
    comp = comp.fillna(0).round().astype("uint16").compute()

    comp = comp.rio.write_crs(geobox.crs.to_wkt())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    comp.rio.to_raster(out_path, compress="deflate", tiled=True)
    return s2_metadata(days, chosen, offset)


def build_worldcover(geobox, out_path: Path) -> dict:
    """ESA WorldCover 2021 (v200) class map on the same grid (nearest)."""
    import odc.stac
    import rioxarray  # noqa: F401

    bbox = geobox.geographic_extent.boundingbox
    items = _catalog().search(
        collections=["esa-worldcover"],
        bbox=(bbox.left, bbox.bottom, bbox.right, bbox.top),
        datetime="2021-01-01/2021-12-31",
    ).item_collection()
    ds = odc.stac.load(items, bands=["map"], geobox=geobox, resampling="nearest")
    lab = ds["map"].max("time").astype("uint8").compute()
    lab = lab.rio.write_crs(geobox.crs.to_wkt())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lab.rio.to_raster(out_path, compress="deflate", tiled=True)
    return {"collection": "esa-worldcover", "items": [it.id for it in items]}


def read_raster(path: Path) -> np.ndarray:
    import rioxarray

    da = rioxarray.open_rasterio(path)
    return np.asarray(da.values)
