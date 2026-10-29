"""Sentinel-2 L2A GeoTIFF reader for Earth-Engine-exported scenes.

gee_acquire.py exports each acquisition as ONE multiband uint16 GeoTIFF
(DN 0-10000, 10 bands in the canonical pipeline order) to Drive. L2AScene
reads such a file LAZILY: only the requested window is read from disk, so
predict_city can run over a full-city scene without ever materializing the
(C, H, W) float32 array in RAM (2 dates x 10 bands x 4 bytes adds up fast).

L2AScene deliberately duck-types the ndarray that oscd.read_city returns —
predict_city touches only `.shape` and `[:, y0:y1, x0:x1]` slices, so the
lazy reader is a drop-in replacement with zero changes to evaluate.py.

Earth Engine splits Drive exports above ~2 GB into chunk files
'<desc>-<row>-<col>.tif'. load_scene() accepts a single .tif OR a directory
of one export's chunks (it stitches them in RAM — only needed when the AOI
outgrew the single-file threshold).
"""

import re
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

# Canonical band order == configs/data.yaml bands (OSCD naming). The GEE
# collection spells these 'B2'..'B12' (no leading zero) but in the same order;
# gee_acquire reorders/renames to match at export time.
S2_L2A_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08",
                "B8A", "B11", "B12"]

# Descriptions EE writes into the exported GeoTIFF (GEE band names).
_EE_DESCRIPTIONS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

# EE chunked-export naming: '<description>-<10 digits>-<10 digits>.tif'
_CHUNK_RE = re.compile(r"-\d{10}-\d{10}\.tif$")


def _chunk_prefix(name: str) -> str:
    """Export description shared by all chunks of one export."""
    return _CHUNK_RE.sub(".tif", name)


class L2AScene:
    """Lazy array-like reader over one multiband L2A GeoTIFF.

    Supports the subset of ndarray semantics predict_city (and the
    notebooks) need:
        scene.shape                      -> (C, H, W)
        scene[:, y0:y1, x0:x1]           -> (C, h, w) float32 reflectance
        scene[2, y0:y1, x0:x1]           -> (h, w) single-band window
        scene[[2, 1, 0]]                 -> (3, H, W) band subset (e.g. RGB)
        scene[:, ::2, ::2]               -> strided window
        scene.read(out_shape=(h, w))     -> decimated full read (quicklooks)

    Values are divided by reflectance_divisor (DN 10000 -> 0-1), matching
    oscd.read_city output. Use as a context manager, or call .close().
    """

    def __init__(self, path, bands: list[str] | None = None,
                 reflectance_divisor: float = 10000.0):
        self.path = Path(path)
        self.bands = list(bands) if bands else list(S2_L2A_BANDS)
        self.divisor = reflectance_divisor
        self._src = rasterio.open(self.path)
        if self._src.count != len(self.bands):
            raise ValueError(
                f"{self.path.name}: expected {len(self.bands)} bands "
                f"({self.bands}), found {self._src.count}. If this is an EE "
                "export, check export_scene selected all 10 pipeline bands.")
        descs = self._src.descriptions
        if all(descs) and list(descs) != _EE_DESCRIPTIONS:
            # order mismatch is exactly the bug-class that silently poisons
            # normalization (stats are per-band), so make it LOUD
            print(f"WARNING: {self.path.name}: band descriptions {list(descs)} "
                  f"don't match the canonical export order {_EE_DESCRIPTIONS}. "
                  "Check the export band selection before trusting the output.")

    # ------------------------------------------------------------- properties
    @property
    def shape(self) -> tuple[int, int, int]:
        """(C, H, W) — same layout as the ndarray returned by read_city."""
        return (self._src.count, self._src.height, self._src.width)

    @property
    def crs(self):
        return self._src.crs

    @property
    def transform(self):
        return self._src.transform

    @property
    def profile(self) -> dict:
        p = self._src.profile.copy()
        p.pop("nodata", None)
        return p

    # ----------------------------------------------------------------- reading
    def read(self, out_shape: tuple[int, int] | None = None) -> np.ndarray:
        """Full read (or rasterio-decimated read for quicklooks) as float32."""
        if out_shape is None:
            arr = self._src.read()
        else:
            h, w = out_shape
            arr = self._src.read(out_shape=(self._src.count, h, w))
        return arr.astype(np.float32) / self.divisor

    @staticmethod
    def _band_indexes(sel, count: int) -> tuple[list[int], bool]:
        """Band-axis selection -> (rasterio 1-based indexes, squeeze?)."""
        if isinstance(sel, slice):
            start, stop, step = sel.indices(count)
            return list(range(start, stop, step)), False
        if isinstance(sel, (int, np.integer)):
            if sel < 0:
                sel += count
            if not 0 <= sel < count:
                raise IndexError(f"band index {sel} out of range for {count} bands")
            return [int(sel)], True
        if isinstance(sel, (list, tuple)):
            # validate each element like the int case, 0-based -> 1-based
            out = []
            for i in sel:
                if not isinstance(i, (int, np.integer)):
                    raise TypeError(f"unsupported band selector element: {type(i)}")
                if i < 0:
                    i += count
                if not 0 <= i < count:
                    raise IndexError(f"band index {i} out of range for {count} bands")
                out.append(int(i))
            return out, False
        raise TypeError(f"unsupported band selector: {type(sel)}")

    @staticmethod
    def _win(sel, size: int) -> tuple[int, int, int]:
        """Row/col-axis selection -> (start, stop, step), clamped like numpy."""
        if isinstance(sel, slice):
            return sel.indices(size)
        if isinstance(sel, (int, np.integer)):
            if sel < 0:
                sel += size
            if not 0 <= sel < size:
                raise IndexError(f"index {sel} out of range for size {size}")
            return int(sel), int(sel) + 1, 1
        raise TypeError(f"unsupported axis selector: {type(sel)}")

    def __getitem__(self, item) -> np.ndarray:
        if not isinstance(item, tuple):
            item = (item, slice(None), slice(None))  # numpy: 1st axis = bands
        if len(item) == 3:
            bsel, rsel, csel = item
        elif len(item) == 2:
            bsel, (rsel, csel) = slice(None), item   # scene[rows, cols]
        else:
            raise IndexError(f"expected 2 or 3 indices, got {len(item)}")

        indexes, squeeze = self._band_indexes(bsel, self._src.count)
        r0, r1, rstep = self._win(rsel, self._src.height)
        c0, c1, cstep = self._win(csel, self._src.width)

        window = Window(c0, r0, c1 - c0, r1 - r0)
        arr = self._src.read(indexes=[i + 1 for i in indexes], window=window)
        if rstep != 1 or cstep != 1:
            arr = arr[:, ::rstep, ::cstep]
        out = arr.astype(np.float32) / self.divisor
        return out[0] if squeeze else out

    # ----------------------------------------------------------------- cleanup
    def close(self):
        if not self._src.closed:
            self._src.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _stack_chunks(tif_paths: list[Path], reflectance_divisor: float) -> np.ndarray:
    """Stitch one EE export's chunk GeoTIFFs into a single (C, H, W) array.

    This is the eager fallback for AOIs that outgrew EE's single-file limit
    (~2 GB): the full uint16 stack lives in RAM, which is acceptable at the
    sizes that trigger chunking. Chunk placement is computed from each
    chunk's geotransform, not from filename sort order.
    """
    srcs = [rasterio.open(p) for p in tif_paths]
    try:
        first = srcs[0]
        for s in srcs[1:]:
            if s.count != first.count or s.crs != first.crs:
                raise ValueError("chunk CRS/band-count mismatch — are all "
                                 f"files from one export? ({tif_paths[0].name} vs {s.name})")
        x0 = min(s.transform.c for s in srcs)              # west edge
        y0 = max(s.transform.f for s in srcs)              # north edge
        res_x, res_y = first.transform.a, -first.transform.e
        rows, cols, hs, ws = [], [], [], []
        for s in srcs:
            cols.append(round((s.transform.c - x0) / res_x))
            rows.append(round((y0 - s.transform.f) / res_y))
            hs.append(s.height)
            ws.append(s.width)
        H = max(r + h for r, h in zip(rows, hs))
        W = max(c + w for c, w in zip(cols, ws))
        out = np.zeros((first.count, H, W), dtype=np.uint16)
        for s, r, c in zip(srcs, rows, cols):
            out[:, r:r + s.height, c:c + s.width] = s.read()
        return out.astype(np.float32) / reflectance_divisor
    finally:
        for s in srcs:
            s.close()


def load_scene(path, bands: list[str] | None = None,
               reflectance_divisor: float = 10000.0):
    """Open an EE-exported scene: .tif file -> lazy L2AScene; directory ->
    stitched (C, H, W) float32 ndarray (EE chunked export). Both satisfy the
    predict_city contract, so callers never branch on the return type."""
    p = Path(path)
    if p.is_dir():
        tifs = sorted(p.glob("*.tif"))
        if not tifs:
            raise FileNotFoundError(f"No .tif files under {p}")
        prefixes = {_chunk_prefix(t.name) for t in tifs}
        if len(prefixes) > 1:
            # t0/t1 chunk sets share a grid, so stitching them together would
            # silently overwrite one date with the other — refuse instead
            raise ValueError(
                f"{p} mixes chunks from multiple exports {sorted(prefixes)}. "
                "Point load_scene at a single .tif or a single export's chunks.")
        if len(tifs) == 1:
            return L2AScene(tifs[0], bands, reflectance_divisor)
        print(f"stitching {len(tifs)} EE export chunks from {p} ...")
        return _stack_chunks(tifs, reflectance_divisor)
    if not p.is_file():
        raise FileNotFoundError(f"No such file or directory: {p}")
    return L2AScene(p, bands, reflectance_divisor)