"""Torch Dataset over the memory-mapped chip splits."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ChipDataset(Dataset):
    """Reads (t0, t1, mask) chips from a memmap split folder.

    Normalization uses fixed per-band stats from the manifest (computed on
    train only) — never per-image z-scoring, which would destroy the
    absolute reflectance signal change detection depends on.

    Augmentation applies the SAME geometric transform to t0, t1 and mask,
    which is what makes it valid for a bi-temporal pair.
    """

    def __init__(self, split_dir: str | Path, stats: dict, augment: bool = False):
        self.dir = Path(split_dir)
        self.augment = augment
        self.t0 = np.load(self.dir / "t0.npy", mmap_mode="r")
        self.t1 = np.load(self.dir / "t1.npy", mmap_mode="r")
        self.mask = np.load(self.dir / "mask.npy", mmap_mode="r")
        self.mean = np.asarray(stats["mean"], dtype=np.float32)  # (C,)
        self.std = np.asarray(stats["std"], dtype=np.float32)
        self.n = int(self.t0.shape[0])

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        t0 = self.t0[idx].astype(np.float32) / 10000.0
        t1 = self.t1[idx].astype(np.float32) / 10000.0
        m = self.mask[idx]

        if self.augment:
            if random.random() < 0.5:
                t0, t1, m = t0[:, :, ::-1], t1[:, :, ::-1], m[:, ::-1]
            if random.random() < 0.5:
                t0, t1, m = t0[:, ::-1, :], t1[:, ::-1, :], m[::-1, :]
            k = random.randrange(4)
            if k:
                t0 = np.rot90(t0, k, axes=(1, 2))
                t1 = np.rot90(t1, k, axes=(1, 2))
                m = np.rot90(m, k, axes=(0, 1))

        t0 = (t0 - self.mean[:, None, None]) / self.std[:, None, None]
        t1 = (t1 - self.mean[:, None, None]) / self.std[:, None, None]

        return (torch.from_numpy(np.ascontiguousarray(t0)),
                torch.from_numpy(np.ascontiguousarray(t1)),
                torch.from_numpy(np.ascontiguousarray(m.astype(np.int64))))


def load_stats_from_manifest(manifest: dict) -> dict:
    return {"mean": manifest["stats"]["mean"], "std": manifest["stats"]["std"]}