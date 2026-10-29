"""Tests for the Earth-Engine L2A scene reader (s2_l2a.py).

The ENTIRE contract predict_city relies on is that L2AScene behaves like the
(C, H, W) reflectance ndarray oscd.read_city returns: .shape and window
slices. These tests pin that equivalence against synthetic GeoTIFFs written
with the same dtype (uint16 DN) and CRS conventions as real EE exports,
plus the chunked-export stitching path EE triggers above ~2 GB.
"""

import numpy as np
import pytest
import rasterio

from s2cd.data.s2_l2a import L2AScene, load_scene
from s2cd.models import SiameseUNet


def _write_tif(path, arr, west=400000.0, north=1450000.0, res=10.0):
    """Write (C, H, W) uint16 with a UTM-like geotransform (EE export style)."""
    transform = rasterio.transform.from_origin(west, north, res, res)
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[1],
                       width=arr.shape[2], count=arr.shape[0], dtype="uint16",
                       crs="EPSG:32644", transform=transform) as dst:
        dst.write(arr)


def _ref(arr):
    """The reader's exact output arithmetic: uint16 -> float32 -> /10000."""
    return arr.astype(np.float32) / 10000.0


@pytest.fixture
def scene_tif(tmp_path):
    path = tmp_path / "chennai_t0.tif"
    arr = np.random.default_rng(0).integers(0, 10001, size=(10, 300, 260))
    arr = arr.astype(np.uint16)
    _write_tif(path, arr)
    return path, arr


def test_shape_matches_full_array(scene_tif):
    path, arr = scene_tif
    with L2AScene(path) as s:
        assert s.shape == arr.shape            # (C, H, W) ndarray contract
        assert s.shape[1:] == (300, 260)


def test_window_slice_equals_ndarray(scene_tif):
    """Regression: predict_city slices [:, y0:y1, x0:x1] per window — those
    lazy reads must be bit-identical to slicing the in-RAM ndarray, in
    float32 reflectance units (DN / 10000)."""
    path, arr = scene_tif
    with L2AScene(path) as s:
        win = s[:, 40:296, 10:250]
        assert win.dtype == np.float32
        np.testing.assert_array_equal(win, _ref(arr[:, 40:296, 10:250]))
        # edge window (clamped start, includes last row/col)
        np.testing.assert_array_equal(s[:, 256:300, 200:260],
                                      _ref(arr[:, 256:300, 200:260]))
        # full read == full slice
        np.testing.assert_array_equal(s[:, :, :], _ref(arr))


def test_reflectance_scaling(scene_tif):
    """Pin the DN -> reflectance divisor (dataset.py's /10000 convention):
    1330 DN must read back as exactly 0.133."""
    path, arr = scene_tif
    arr[0, 0, 0] = 1330
    _write_tif(path, arr)
    with L2AScene(path) as s:
        assert s[0, 0, 0] == pytest.approx(0.133, abs=1e-7)


def test_int_band_and_strided_slicing(scene_tif):
    path, arr = scene_tif
    with L2AScene(path) as s:
        np.testing.assert_array_equal(s[3, :, :], _ref(arr[3]))
        np.testing.assert_array_equal(s[-1, :, :], _ref(arr[9]))
        np.testing.assert_array_equal(s[(0, 1, 2), ::2, ::4],
                                      _ref(arr[(0, 1, 2), ::2, ::4]))
        with pytest.raises(IndexError):
            _ = s[10, :, :]                    # band axis has only 0..9


def test_band_list_indexing_like_numpy(scene_tif):
    """The 06 stretch helper does a[rgb] with a bare list — L2AScene must
    follow the same first-axis-band semantics, not choke on a non-tuple."""
    path, arr = scene_tif
    rgb = [2, 1, 0]                           # B04, B03, B02 like configs
    with L2AScene(path) as s:
        np.testing.assert_array_equal(s[rgb], _ref(arr[rgb]))
        assert s[rgb].shape == (3, 300, 260)


def test_read_decimated_for_quicklooks(scene_tif):
    path, arr = scene_tif
    with L2AScene(path) as s:
        small = s.read(out_shape=(30, 26))
        assert small.shape == (10, 30, 26)
        assert small.dtype == np.float32


def test_band_count_mismatch_is_loud(tmp_path):
    """A 3-band file is not a pipeline scene — fail at open, not mid-inference
    with a cryptic normalization error."""
    path = tmp_path / "tiny.tif"
    _write_tif(path, np.zeros((3, 32, 32), dtype=np.uint16))
    with pytest.raises(ValueError, match="3"):
        L2AScene(path)


def test_predict_city_duck_typing(scene_tif):
    """The integration contract: predict_city over the LAZY scene must
    produce the same probability map as over the eager ndarray — same tiny
    model, same stats, same windows."""
    from s2cd.evaluate import predict_city

    path, arr = scene_tif
    stats = {"mean": [0.4] * 10, "std": [0.2] * 10}
    model = SiameseUNet(in_channels=10, base=32).eval()
    eager = predict_city(model, arr.astype(np.float32) / 10000.0,
                         arr.astype(np.float32) / 10000.0, stats,
                         chip=64, stride=64, device="cpu")
    with L2AScene(path) as s:
        lazy = predict_city(model, s, s, stats, chip=64, stride=64, device="cpu")
    assert lazy.shape == (300, 260)
    np.testing.assert_allclose(lazy, eager, rtol=1e-5, atol=1e-7)


def test_load_scene_single_file_returns_lazy_reader(scene_tif):
    path, _ = scene_tif
    assert isinstance(load_scene(path), L2AScene)
    assert isinstance(load_scene(path.parent, reflectance_divisor=10000.0),
                      L2AScene)          # dir with exactly one .tif -> lazy


def test_chunked_export_stitching(tmp_path):
    """EE splits >~2 GB exports into '<desc>-<row>-<col>.tif' chunks with the
    SAME grid. load_scene must stitch by geotransform into the full scene."""
    rng = np.random.default_rng(1)
    full = rng.integers(0, 10001, size=(10, 120, 100)).astype(np.uint16)
    names = ["chennai_t0-0000000000-0000000000.tif",
             "chennai_t0-0000000000-0000050000.tif"]
    _write_tif(tmp_path / names[0], full[:, :, :50])           # west half
    _write_tif(tmp_path / names[1], full[:, :, 50:], west=400500.0)  # east
    out = load_scene(tmp_path)
    assert not isinstance(out, L2AScene)          # chunked -> eager ndarray
    np.testing.assert_array_equal(out, _ref(full))


def test_mixed_exports_refused(tmp_path):
    """t0 and t1 chunk sets share a grid — stitching a directory holding
    BOTH would silently overwrite one date with the other. Must raise."""
    rng = np.random.default_rng(2)
    half = rng.integers(0, 10001, size=(10, 120, 50)).astype(np.uint16)
    for date in ["t0", "t1"]:
        _write_tif(tmp_path / f"chennai_{date}-0000000000-0000000000.tif", half)
    with pytest.raises(ValueError, match="multiple exports"):
        load_scene(tmp_path)


def test_load_scene_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_scene(tmp_path / "nope.tif")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        load_scene(empty)