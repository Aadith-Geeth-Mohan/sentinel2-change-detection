"""Google Earth Engine acquisition of Sentinel-2 L2A scenes for any AOI.

Chennai case study: for each target window (post-monsoon Jan-Mar of 2018,
2021, 2025) pick the least-cloudy acquisition DATE that fully covers the
AOI — all same-day granules mosaicked into one image, since each EE image
is a single ~100 km MGRS-tile granule and a multi-tile AOI is never covered
by one — and export it with the SAME 10 bands the OSCD pipeline uses:
native 10 m bands plus the 20 m bands bilinearly resampled at export time,
mirroring oscd.py's preprocessing, as single uint16 GeoTIFFs to Drive,
then read them with s2_l2a.load_scene.

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


def _aoi_sample_points(rect, n: int = 5) -> list[tuple[float, float]]:
    """n x n lon/lat grid over the rect for footprint-coverage tests."""
    w, s, e, north = rect
    lons = [w + (e - w) * i / (n - 1) for i in range(n)]
    lats = [s + (north - s) * i / (n - 1) for i in range(n)]
    return [(lo, la) for la in lats for lo in lons]


def _point_in_ring(x: float, y: float, ring) -> bool:
    """Ray-casting point-in-polygon on a GeoJSON exterior ring."""
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _point_in_geom(pt, geom) -> bool:
    """Point (x, y) inside a GeoJSON Polygon/MultiPolygon footprint."""
    gtype = geom.get("type") if isinstance(geom, dict) else None
    if gtype == "GeometryCollection":
        return any(_point_in_geom(pt, g) for g in geom.get("geometries", []))
    if gtype == "Polygon":
        polys = [geom["coordinates"]]
    elif gtype == "MultiPolygon":
        polys = geom["coordinates"]
    else:
        return False
    x, y = pt
    return any(_point_in_ring(x, y, poly[0]) for poly in polys)


def _scene_days(rect, date_from: str, date_to: str) -> dict:
    """Granules over the AOI grouped by acquisition DATE, coverage-tested.

    Every EE image is a single ~100 km MGRS-tile granule with a tilted
    footprint, and a multi-tile AOI is never covered by one granule — so a
    scene must be a same-day MOSAIC. Each day records the worst granule
    cloud % and whether the union of footprints covers an AOI grid.

    Note the collection is NOT cloud-filtered first: dropping a cloudy
    granule would silently punch a no-data hole in that day's mosaic; the
    cloud_max threshold is applied to the whole-day worst-cloud at
    selection time instead.
    """
    ee = _ee()
    aoi = ee.Geometry.Rectangle(rect, geodesic=False)
    fc = (ee.ImageCollection(COLLECTION)
          .filterBounds(aoi)
          .filterDate(date_from, date_to)
          .map(lambda img: ee.Feature(img.geometry(), {
              "id": img.get("system:index"),
              "ts": img.get("system:time_start"),
              "cloud": img.get("CLOUDY_PIXEL_PERCENTAGE")})))
    feats = fc.getInfo().get("features", [])
    days: dict[str, dict] = {}
    for f in feats:
        p = f.get("properties", {})
        date = (datetime.fromtimestamp(p["ts"] / 1000, tz=timezone.utc).date().isoformat()
                if p.get("ts") is not None else "?")
        day = days.setdefault(date, {"ids": [], "cloud_pct": 0.0, "footprints": []})
        day["ids"].append(p.get("id", "?"))
        day["cloud_pct"] = max(day["cloud_pct"], p.get("cloud") or 0.0)
        day["footprints"].append(f.get("geometry", {}))
    pts = _aoi_sample_points(rect)
    for day in days.values():
        day["covered"] = all(
            any(_point_in_geom(pt, g) for g in day["footprints"]) for pt in pts)
    return days


def list_scenes(rect, date_from: str, date_to: str, cloud_max: float = 10.0,
                limit: int = 5) -> list[dict]:
    """Top-`limit` candidate ACQUISITION DATES for the window, least
    worst-granule cloud first. Only dates whose same-day granule union
    fully covers the AOI are listed (id, date, cloud_pct)."""
    days = _scene_days(rect, date_from, date_to)
    ranked = sorted((d for d, v in days.items() if v["covered"]),
                    key=lambda d: days[d]["cloud_pct"])
    return [{"id": days[d]["ids"][0] + (f" (+{len(days[d]['ids']) - 1} granules)"
                                        if len(days[d]["ids"]) > 1 else ""),
             "date": d, "cloud_pct": days[d]["cloud_pct"]}
            for d in ranked[:limit]]


def find_scene(rect, date_from: str, date_to: str,
               cloud_max: float = 10.0) -> dict:
    """Best full-coverage acquisition date in the window.

    All same-day granules over the AOI are mosaicked into ONE image, dates
    are ranked by worst-granule cloud %, and only dates whose granule union
    covers the whole AOI qualify — picking a single least-cloudy granule
    exports a mosaic with a tile-sized no-data hole (see gotcha #8).
    Returns dict with the ee.Image mosaic plus id/date/cloud_pct.
    """
    ee = _ee()
    days = _scene_days(rect, date_from, date_to)
    covered = sorted((d for d, v in days.items() if v["covered"]),
                     key=lambda d: days[d]["cloud_pct"])
    ok = [d for d in covered if days[d]["cloud_pct"] <= cloud_max]
    if not ok:
        raise ValueError(
            f"No acquisition date with full-AOI coverage under {cloud_max:.0f}% "
            f"cloud between {date_from} and {date_to}. Raise cloud_max or widen "
            "the window (post-monsoon Jan-Mar is the reliable window for Chennai).")
    date = ok[0]
    day = days[date]
    mosaic = ee.ImageCollection.fromImages(
        [ee.Image(f"{COLLECTION}/{i}") for i in day["ids"]]).mosaic()
    return {"id": day["ids"][0] + (f" (+{len(day['ids']) - 1} granules)"
                                   if len(day["ids"]) > 1 else ""),
            "date": date, "cloud_pct": day["cloud_pct"], "image": mosaic,
            "granules": day["ids"]}


def export_scene(image, rect, description: str, folder: str = "s2cd",
                 scale: int = 10):
    """Start a Drive export of the 10 pipeline bands as ONE uint16 GeoTIFF.

    The 20 m bands are bilinear-resampled at export time to match the
    pipeline's own 20 m -> 10 m preprocessing (oscd.py _resample), and
    .toUint16() keeps the export at DN 0-10000 reflectance — a float64
    export would be 4x larger for zero benefit.

    `folder` must be a SINGLE-LEVEL Drive folder name: EE's toDrive folder
    parameter cannot nest paths ('s2cd/chennai' produced a root folder
    literally named 's2cd chennai'). The file lands flat in MyDrive/s2cd/
    as <description>.tif.
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