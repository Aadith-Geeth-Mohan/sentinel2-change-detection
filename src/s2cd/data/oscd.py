"""OSCD (ONERA Satellite Change Detection) reader with layout discovery.

The exact directory layout of the OSCD distribution has minor variations
between mirrors/vintages, so instead of hardcoding paths we discover:
  * label cities + masks  — png files under '... Labels/<City>/cm/...'
  * image files            — band tifs, grouped by (band, date) parsed from
                             filenames / parent folder names

Every function raises with a clear message when discovery fails, so layout
surprises surface in notebook 01 (minutes) rather than mid-training.
"""

import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F

# Canonical OSCD band order
ALL_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08",
            "B8A", "B09", "B10", "B11", "B12"]


# ---------------------------------------------------------------------------
# discovery helpers
# ---------------------------------------------------------------------------
def _band_of(name: str) -> str | None:
    u = name.upper()
    if "B8A" in u:
        return "B8A"
    m = re.search(r"B(\d{2})", u)
    return f"B{m.group(1)}" if m else None


def _date_of(path: str) -> str:
    """Two temporal epochs appear as ISO dates in filename or parent folder."""
    base = os.path.basename(path)
    m = re.search(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})", base)
    if m:
        return "-".join(m.groups())
    parent = os.path.basename(os.path.dirname(path))
    return parent  # e.g. 'Images/<City>/imgs_1_rect/B04.tif' -> 'imgs_1_rect'


def _looks_like_root_component(c: str) -> bool:
    cl = c.lower()
    return ("onera" in cl) or ("label" in cl) or ("dataset" in cl and "images" in cl)


def _is_label_png(oscd_root: Path, png: Path) -> bool:
    """Ground-truth change maps live as '... Labels/<City>/cm/cm.png'.

    The Images tree also ships quicklook RGB previews at
    'Images/<City>/pair/img{1,2}.png'. Without this filter those preview pngs
    spawn a phantom 'pair' city and can even be returned as a city's mask."""
    parts = png.relative_to(oscd_root).parts
    return "cm" in parts and any("label" in p.lower() for p in parts)


def list_cities(oscd_root: str | Path) -> dict[str, list[str]]:
    """Discover labeled cities. Returns {'train': [...], 'test': [...]}."""
    oscd_root = Path(oscd_root)
    result = defaultdict(set)
    for png in oscd_root.rglob("*.png"):
        if not _is_label_png(oscd_root, png):
            continue
        comps = [p for p in png.relative_to(oscd_root).parts[:-1]]
        city = None
        for c in reversed(comps):
            if c.lower() == "cm" or _looks_like_root_component(c):
                continue
            city = c
            break
        if city is None:
            continue
        split = "test" if any("test" in c.lower() for c in comps) else "train"
        result[split].add(city)
    out = {"train": sorted(result["train"]), "test": sorted(result["test"])}
    if not out.get("train"):
        raise RuntimeError(
            f"No labeled cities found under {oscd_root}. Expected "
            "'... Labels/<City>/cm/*.png' structure. Check notebook 01 extraction."
        )
    # OSCD is a fixed 24-city dataset: 14 train / 10 test. Any other count means
    # discovery picked up or dropped a city — a phantom name like 'pair' almost
    # always means the kernel is executing stale cached modules after a pull.
    if len(out["train"]) != 14 or len(out["test"]) != 10:
        print(f"WARNING: expected 14 train / 10 test OSCD cities, got "
              f"{len(out['train'])} / {len(out['test'])}: {out}\n"
              "  If you see a phantom name (e.g. 'pair'), restart the runtime "
              "(Runtime > Restart runtime) so the pulled code is actually loaded.")
    return out


def find_mask(oscd_root: str | Path, city: str) -> Path:
    oscd_root = Path(oscd_root)
    for png in sorted(oscd_root.rglob("*.png")):
        if not _is_label_png(oscd_root, png):
            continue
        if city in png.relative_to(oscd_root).parts:
            return png
    raise FileNotFoundError(f"No label png found for city '{city}'")


# ---------------------------------------------------------------------------
# city reading
# ---------------------------------------------------------------------------
def _city_image_dir(oscd_root: Path, city: str) -> Path:
    """The city's folder in the Images tree. Layout (TorchGeo-verified):
    '... Images/<City>/imgs_1_rect/*.tif' + 'imgs_2_rect/*.tif' + 'dates.txt',
    where imgs_1_rect / imgs_2_rect are the two temporal epochs. The Labels
    trees also contain tifs (rasterized polygons) — those are excluded."""
    for tif in sorted(oscd_root.rglob("*.tif")):
        parts = tif.relative_to(oscd_root).parts
        if city not in parts:
            continue
        if any("label" in p.lower() for p in parts):
            continue
        return oscd_root.joinpath(*parts[:parts.index(city) + 1])
    raise FileNotFoundError(f"No image tifs found for city '{city}'")


def _read_tif(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


def _resample(arr: np.ndarray, shape) -> np.ndarray:
    if arr.shape == shape:
        return arr
    t = torch.from_numpy(arr.astype(np.float32))[None, None]
    t = F.interpolate(t, size=shape, mode="bilinear", align_corners=False)
    return t[0, 0].numpy()


def read_city(oscd_root: str | Path, city: str, bands: list[str],
              reflectance_divisor: float = 10000.0) -> dict:
    """Read one city as (t0, t1, mask).

    Returns dict with:
        t0, t1 : float32 (C, H, W) reflectance, band order == `bands`
        mask   : uint8 (H, W) in {0, 1}
        shape  : (H, W)
    """
    oscd_root = Path(oscd_root)
    img_dir = _city_image_dir(oscd_root, city)

    # group band files by temporal epoch (imgs_1_rect / imgs_2_rect folders,
    # or ISO dates in filenames on mirrors that use that convention)
    by_date: dict[str, dict[str, Path]] = defaultdict(dict)
    for f in img_dir.rglob("*.tif"):
        band = _band_of(f.name)
        if band:
            by_date[_date_of(str(f))][band] = f
    if len(by_date) < 2:
        raise RuntimeError(f"City '{city}': found <2 dates in {img_dir} ({sorted(by_date)})")
    dates = sorted(by_date)
    t0_date, t1_date = dates[0], dates[-1]

    # human-readable acquisition dates live in the city's dates.txt
    pretty_dates = [t0_date, t1_date]
    dt_file = img_dir / "dates.txt"
    if dt_file.exists():
        toks = [ln.split()[-1] for ln in dt_file.read_text().splitlines() if ln.split()]
        if len(toks) >= 2:
            pretty_dates = [toks[0], toks[-1]]

    missing = [b for b in bands if b not in by_date[t0_date] or b not in by_date[t1_date]]
    if missing:
        raise RuntimeError(f"City '{city}': missing bands {missing}. Available: "
                           f"{sorted(by_date[t0_date])}")

    # reference shape from the first requested band at t0
    ref = _read_tif(by_date[t0_date][bands[0]]).shape

    def stack(date) -> np.ndarray:
        chans = [_resample(_read_tif(by_date[date][b]), ref) for b in bands]
        arr = np.stack(chans, axis=0).astype(np.float32)
        return arr / reflectance_divisor

    t0, t1 = stack(t0_date), stack(t1_date)

    mask_path = find_mask(oscd_root, city)
    mask = _resample(_read_tif(mask_path), ref)
    mask = (mask > 0).astype(np.uint8)

    return {"city": city, "t0": t0, "t1": t1, "mask": mask, "shape": ref,
            "dates": pretty_dates}


def city_chip_count(shape, chip: int, stride: int) -> int:
    h, w = shape
    if h < chip or w < chip:
        return 0
    return ((h - chip) // stride + 1) * ((w - chip) // stride + 1)