"""Google Earth Engine acquisition of Sentinel-2 L2A scenes for any AOI.

Chennai case study: pick the least-cloudy scene in each of two target
windows (t0 ~2018, t1 ~2025, post-monsoon Jan-Mar), export both with the
SAME 10 bands the OSCD pipeline uses — native 10 m bands plus the 20 m
bands bilinearly resampled at export time, mirroring oscd.py's
preprocessing — as single uint16 GeoTIFFs to Drive, then read them with
s2_l2a.load_scene.

`ee` is deliberately NOT a package dependency (like torch): Colab
pip-installs earthengine-api, local dev doesn't need it. A loud ImportError
explains the fix if it's missing.
"""

import time
from datetime import datetime, timezone

# GEE band names in the canonical pipeline order (== s2_l2a.S2_L2A_BANDS
# order, == configs/data.yaml). S2_SR_HARMONIZED uses names WITHOUT the
# leading zero ('B2', not 'B02').
S2_L2A_BANDS_GEE = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
NATIVE_10M = ["B2", "B3", "B4", "B8"]
NATIVE_20M = ["B5", "B6", "B7", "B8A", "B11", "B12"]

COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"

# Full metro + fringe, no cherry-picking. [west, south, east, north] in
# degrees: ~76 x 78 km -> ~1.2 GB per date as uint16, safely under Earth
# Engine's single-file Drive-export threshold (~2 GB) so no chunk stitching
# is needed. Enlarge the rect and EE simply splits into chunks —
# s2_l2a.load_scene stitches those transparently.
CHENNAI_RECT = [79.85, 12.65, 80.55, 13.35]


def _ee():
    try:
        import ee
    except ImportError as e:
        raise ImportError(
            "earthengine-api is not installed. In Colab: "
            "!pip install -q earthengine-api"
        ) from e
    return ee


def _collection(rect, date_from: str, date_to: str, cloud_max: float):
    """Filtered collection over the AOI, least-cloudy scene first."""
    ee = _ee()
    aoi = ee.Geometry.Rectangle(rect, geodesic=False)
    return (ee.ImageCollection(COLLECTION)
            .filterBounds(aoi)
            .filterDate(date_from, date_to)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_max))
            .sort("CLOUDY_PIXEL_PERCENTAGE"))


def _scene_info(feature: dict) -> dict:
    props = feature.get("properties", {})
    ts = props.get("system:time_start")
    date = (datetime.fromtimestamp(ts / 1000, tz=timezone.utc).date().isoformat()
            if ts is not None else "?")
    index = props.get("system:index", "?")
    return {"id": f"{COLLECTION}/{index}", "date": date,
            "cloud_pct": props.get("CLOUDY_PIXEL_PERCENTAGE")}


def list_scenes(rect, date_from: str, date_to: str, cloud_max: float = 10.0,
                limit: int = 5) -> list[dict]:
    """Top-`limit` least-cloudy candidate scenes (id, date, cloud_pct)."""
    ee = _ee()
    info = _collection(rect, date_from, date_to, cloud_max).limit(limit).getInfo()
    return [_scene_info(f) for f in info.get("features", [])]


def find_scene(rect, date_from: str, date_to: str,
               cloud_max: float = 10.0) -> dict:
    """Least-cloudy scene in the window. Returns dict with the ee.Image
    (ready for export_scene) plus id/date/cloud_pct for the record."""
    ee = _ee()
    info = _collection(rect, date_from, date_to, cloud_max).limit(1).getInfo()
    feats = info.get("features", [])
    if not feats:
        raise ValueError(
            f"No S2 scene under {cloud_max:.0f}% cloud over the AOI between "
            f"{date_from} and {date_to}. Widen the dates or raise cloud_max "
            "(post-monsoon Jan-Mar is the reliable window for Chennai).")
    scene = _scene_info(feats[0])
    scene["image"] = ee.Image(scene["id"])
    return scene


def export_scene(image, rect, description: str, folder: str = "s2cd/chennai",
                 scale: int = 10):
    """Start a Drive export of the 10 pipeline bands as ONE uint16 GeoTIFF.

    The 20 m bands are bilinear-resampled at export time to match the
    pipeline's own 20 m -> 10 m preprocessing (oscd.py _resample), and
    .toUint16() keeps the export at DN 0-10000 reflectance — a float64
    export would be 4x larger for zero benefit.
    """
    ee = _ee()
    ten = image.select(NATIVE_10M)
    twenty = image.select(NATIVE_20M)
    try:
        combined = ten.addBands(twenty.resample("bilinear"))
    except Exception as e:
        print(f"WARNING: bilinear resample failed ({e}); exporting 20 m "
              "bands at default (nearest) sampling instead.")
        combined = image.select(S2_L2A_BANDS_GEE)
    img = combined.select(S2_L2A_BANDS_GEE).toUint16()

    task = ee.batch.Export.image.toDrive(
        image=img,
        description=description,
        folder=folder,
        fileNamePrefix=description,
        region=ee.Geometry.Rectangle(rect, geodesic=False),
        scale=scale,
        maxPixels=1e13,
    )
    task.start()
    print(f"export started: '{description}' -> MyDrive/{folder}/{description}.tif "
          f"(rect={rect}, {len(S2_L2A_BANDS_GEE)} bands @ {scale} m)")
    return task


def wait_for_tasks(tasks, poll_s: float = 30.0, timeout_s: float = 5400.0):
    """Block until all EE export tasks finish. Loud on failure — a FAILED
    task's error_message is raised in full rather than left in the log."""
    deadline = time.monotonic() + timeout_s
    while True:
        statuses = []
        for t in tasks:
            try:
                statuses.append(t.status())
            except Exception as e:
                raise RuntimeError(f"cannot query EE task status: {e}") from e
        print(" | ".join(f"{s.get('description', '?')}: {s['state']}" for s in statuses),
              flush=True)
        failed = [s for s in statuses
                  if s["state"] in ("FAILED", "CANCELLED")]
        if failed:
            raise RuntimeError(
                "EE export failed:\n" + "\n".join(
                    f"  {s.get('description')}: {s.get('error_message', s['state'])}"
                    for s in failed))
        if all(s["state"] == "COMPLETED" for s in statuses):
            return
        if time.monotonic() > deadline:
            raise TimeoutError(f"EE tasks still unfinished after {timeout_s}s — "
                               "check https://code.earthengine.google.com/tasks")
        time.sleep(poll_s)