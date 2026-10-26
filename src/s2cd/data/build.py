"""Build memory-mapped chip datasets from raw OSCD.

Per split we write three flat arrays (random-access safe, RAM-light):
    /content/chips/<split>/t0.npy    uint16 (N, C, H, W)
    /content/chips/<split>/t1.npy    uint16 (N, C, H, W)
    /content/chips/<split>/mask.npy  uint8 (N, H, W)
    /content/chips/<split>/meta.json

then zip each split folder -> chips_{split}.zip -> Drive, with md5s recorded
in manifest.json. Raw uint16 keeps storage ~half of float32; normalization
uses fixed per-band stats (computed on the TRAIN split only) at load time.
"""

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import yaml

from . import oscd
from .store import DriveStore, _md5


def _splits(cfg: dict, oscd_root) -> dict[str, list[str]]:
    cities = oscd.list_cities(oscd_root)
    train_all = cities["train"]
    n_val = int(cfg.get("val_cities", 3))
    val = train_all[:n_val]  # deterministic holdout
    train = train_all[n_val:]
    print(f"train cities ({len(train)}): {train}")
    print(f"val cities   ({len(val)}): {val}")
    print(f"test cities ({len(cities['test'])}): {cities['test']}")
    return {"train": train, "val": val, "test": cities["test"]}


def _fill_split(oscd_root, cities, split_dir: Path, cfg: dict, writer) -> int:
    chip, stride = cfg["chip_size"], cfg["stride"]
    bands = cfg["bands"]
    div = cfg.get("reflectance_divisor", 10000)

    # pass 1: count chips from metadata shapes only (no pixel reads)
    total, per_city = 0, {}
    for city in cities:
        img_dir = oscd._city_image_dir(Path(oscd_root), city)
        # ref shape from the SAME band read_city uses as reference (bands[0],
        # first epoch), so pass-1 counts and pass-2 writes agree even when
        # bands have different native resolutions
        cand = sorted(img_dir.rglob(f"*/{bands[0]}.tif"))
        if not cand:
            raise RuntimeError(f"{city}: no {bands[0]}.tif under {img_dir} — layout surprise")
        ref = oscd._read_tif(cand[0]).shape
        n = oscd.city_chip_count(ref, chip, stride)
        per_city[city] = n
        total += n
    print(f"  {split_dir.name}: {total} chips ({per_city})")
    if total == 0:
        raise RuntimeError(f"Split {split_dir.name} produced 0 chips — chip size too large?")

    # pass 2: read pixels and fill memmaps
    meta = writer(split_dir, total, len(bands), chip)
    i = 0
    for city in cities:
        data = oscd.read_city(oscd_root, city, bands, div)
        if data["shape"][0] < chip or data["shape"][1] < chip:
            continue
        t0u = np.clip(data["t0"] * div, 0, 65535).astype(np.uint16)
        t1u = np.clip(data["t1"] * div, 0, 65535).astype(np.uint16)
        m = data["mask"]
        for y in range(0, m.shape[0] - chip + 1, stride):
            for x in range(0, m.shape[1] - chip + 1, stride):
                meta["t0"][i] = t0u[:, y:y + chip, x:x + chip]
                meta["t1"][i] = t1u[:, y:y + chip, x:x + chip]
                meta["mask"][i] = m[y:y + chip, x:x + chip]
                i += 1
    assert i == total, f"counted {total} chips but wrote {i}"
    return total


def build(oscd_root, data_cfg_path: str, drive: DriveStore,
          scratch: Path = Path("/content")) -> dict:
    cfg = yaml.safe_load(open(data_cfg_path))
    splits = _splits(cfg, oscd_root)
    chip, bands = cfg["chip_size"], cfg["bands"]

    def make_mmaps(split_dir: Path, n, c, chip):
        split_dir.mkdir(parents=True, exist_ok=True)
        return {
            "t0": np.lib.format.open_memmap(split_dir / "t0.npy", mode="w+", dtype=np.uint16,
                                            shape=(n, c, chip, chip)),
            "t1": np.lib.format.open_memmap(split_dir / "t1.npy", mode="w+", dtype=np.uint16,
                                            shape=(n, c, chip, chip)),
            "mask": np.lib.format.open_memmap(split_dir / "mask.npy", mode="w+", dtype=np.uint8,
                                              shape=(n, chip, chip)),
        }

    manifest = {"version": 1, "bands": bands, "chip_size": chip,
                "stride": cfg["stride"], "splits": {}}

    counts = {}
    for split, cities in splits.items():
        split_dir = scratch / "chips" / split
        counts[split] = _fill_split(oscd_root, cities, split_dir, cfg, make_mmaps)
        (split_dir / "meta.json").write_text(json.dumps({"count": counts[split]}))

    # fixed per-band stats from TRAIN split only
    stats_dir = scratch / "chips" / "train"
    n = counts["train"]
    mean = np.zeros(len(bands), dtype=np.float64)
    sq = np.zeros(len(bands), dtype=np.float64)
    t0 = np.load(stats_dir / "t0.npy", mmap_mode="r")
    for start in range(0, n, 256):
        block = t0[start:start + 256].astype(np.float32) / cfg.get("reflectance_divisor", 10000)
        mean += block.mean(axis=(0, 2, 3)) * len(block)
        sq += (block ** 2).mean(axis=(0, 2, 3)) * len(block)
    mean /= n
    std = np.sqrt(np.maximum(sq / n - mean ** 2, 1e-12))
    manifest["stats"] = {"mean": mean.tolist(), "std": std.tolist()}
    print(f"train band means: {np.round(mean, 4).tolist()}")
    print(f"train band stds : {np.round(std, 4).tolist()}")

    # zip each split -> Drive, record md5
    for split in counts:
        split_dir = scratch / "chips" / split
        zpath = scratch / f"chips_{split}.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for f in split_dir.iterdir():
                z.write(f, f.name)
        dest = drive.drive / "processed" / zpath.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(zpath, dest)  # streaming copy — never load the zip in RAM
        manifest["splits"][split] = {
            "chips": counts[split], "md5": _md5(dest), "zip_bytes": dest.stat().st_size,
        }
        print(f"pushed {zpath.name} -> Drive ({dest.stat().st_size / 1e9:.2f} GB)")

    mpath = drive.drive / "processed" / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {mpath}")
    return manifest