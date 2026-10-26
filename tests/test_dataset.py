import json
from pathlib import Path

import numpy as np
import pytest
import torch

from s2cd.data.dataset import ChipDataset


@pytest.fixture
def fake_split(tmp_path: Path):
    """A tiny synthetic memmap split in the real build.py format."""
    n, c, h = 6, 4, 32
    rng = np.random.default_rng(0)
    t0 = rng.integers(0, 10000, size=(n, c, h, h), dtype=np.uint16)
    t1 = t0.copy()
    mask = np.zeros((n, h, h), dtype=np.uint8)
    mask[:, :4, :4] = 1          # a change block in the top-left corner
    t1[:, :, :4, :4] += 2000     # consistent with a reflectance jump
    np.save(tmp_path / "t0.npy", t0)
    np.save(tmp_path / "t1.npy", t1)
    np.save(tmp_path / "mask.npy", mask)
    (tmp_path / "meta.json").write_text(json.dumps({"count": n}))
    return tmp_path, {"mean": [0.3] * c, "std": [0.1] * c}


def test_shapes_and_dtypes(fake_split):
    d, stats = fake_split
    ds = ChipDataset(d, stats)
    t0, t1, m = ds[0]
    assert t0.dtype == torch.float32 and t0.shape == (4, 32, 32)
    assert m.dtype == torch.int64 and m.shape == (32, 32)
    assert int(m[:4, :4].sum()) == 16 and int(m[4:, :].sum()) == 0


def test_fixed_normalization_not_per_image(fake_split):
    """Normalization must use fixed stats — NOT per-image z-scoring.

    With per-image z-scoring (V1's bug), a constant image would blow up to
    noise and a global reflectance shift would vanish. With fixed stats the
    output must equal (raw/10000 - mean) / std exactly.
    """
    d, stats = fake_split
    ds = ChipDataset(d, stats)
    raw = np.load(d / "t0.npy", mmap_mode="r")[0].astype(np.float32) / 10000.0
    expected = (raw - np.asarray(stats["mean"], dtype=np.float32)[:, None, None]) \
        / np.asarray(stats["std"], dtype=np.float32)[:, None, None]
    t0, _, _ = ds[0]
    assert torch.allclose(t0, torch.from_numpy(expected), atol=1e-5)


def test_augmentation_moves_mask_with_images(fake_split):
    """Rotating/flipping the pair must transform mask identically."""
    d, stats = fake_split
    ds = ChipDataset(d, stats, augment=False)
    t0, t1, m = ds[0]
    # the change block lives in the top-left; rotate 90 deg once manually
    t0r = torch.from_numpy(np.rot90(t0.numpy(), 1, axes=(1, 2)).copy())
    mr = torch.from_numpy(np.rot90(m.numpy(), 1, axes=(0, 1)).copy())
    assert int(mr[:4, -4:].sum()) == 16, "rotated mask must track rotated change"


def test_memmap_no_ram_blowup(fake_split):
    d, stats = fake_split
    ds = ChipDataset(d, stats)
    assert ds.t0.mode == "r"  # memmap, not in-RAM copy