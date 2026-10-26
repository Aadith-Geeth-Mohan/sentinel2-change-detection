"""Evaluation: test-set metrics + sliding-window predictions on full cities."""

import json
from pathlib import Path

import numpy as np
import torch

from .data import DriveStore, ChipDataset
from .metrics import ChangeMetrics
from .models import SiameseUNet
from .train import build_model, make_loader
from .utils import load_config, get_device, unwrap_model


def load_best_model(cfg_path: str, store: DriveStore, device=None):
    cfg = load_config(cfg_path)
    device = device or get_device(cfg.get("device", "cuda"))
    run_dir = store.fetch_run(cfg["run_name"])
    best = run_dir / "best.ckpt"
    if not best.exists():
        raise FileNotFoundError(f"{best} missing — train first (notebook 03).")
    ck = torch.load(best, map_location=device, weights_only=False)
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, device


@torch.no_grad()
def test_metrics(model, cfg, store: DriveStore, manifest: dict) -> dict:
    device = next(model.parameters()).device
    stats = {"mean": manifest["stats"]["mean"], "std": manifest["stats"]["std"]}
    ds = ChipDataset(store.fetch_chips("test"), stats, augment=False)
    loader = make_loader(ds, cfg, shuffle=False)
    metrics = ChangeMetrics()
    total_loss, n = 0.0, 0
    for t0, t1, mask in loader:
        t0, t1, mask = t0.to(device), t1.to(device), mask.to(device)
        logits = model(t0, t1)
        metrics.update(logits.float(), mask)
        n += 1
    r = metrics.results()
    r["n_test_chips"] = len(ds)
    return r


@torch.no_grad()
def predict_city(model, t0: np.ndarray, t1: np.ndarray, stats: dict,
                 chip: int = 256, stride: int = 256, device=None) -> np.ndarray:
    """Sliding-window change probability map for full-city reflectance arrays.

    t0, t1: (C, H, W) float arrays in reflectance units (0-1) as produced by
    oscd.read_city — already divided, unlike the raw DN stored in the memmaps.
    """
    device = device or next(model.parameters()).device
    mean = np.asarray(stats["mean"], dtype=np.float32)[:, None, None]
    std = np.asarray(stats["std"], dtype=np.float32)[:, None, None]
    h, w = t0.shape[1], t0.shape[2]
    prob = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)

    def starts(size: int) -> list[int]:
        """Window origins along one axis; a clamped final window covers the
        tail so no right/bottom strip is left uncounted when size isn't a
        multiple of stride (e.g. abudhabi 799px would otherwise lose its last
        size % stride rows). Cities smaller than one chip get a single window."""
        if size <= chip:
            return [0]
        ys = list(range(0, size - chip + 1, stride))
        if ys[-1] != size - chip:
            ys.append(size - chip)
        return ys

    for y in starts(h):
        for x in starts(w):
            yc, xc = y, x  # starts() already yields in-bounds origins
            a = (t0[:, yc:yc + chip, xc:xc + chip].astype(np.float32) - mean) / std
            b = (t1[:, yc:yc + chip, xc:xc + chip].astype(np.float32) - mean) / std
            a = torch.from_numpy(a)[None].to(device)
            b = torch.from_numpy(b)[None].to(device)
            p = torch.softmax(model(a, b), dim=1)[0, 1].cpu().numpy()
            prob[yc:yc + chip, xc:xc + chip] += p
            count[yc:yc + chip, xc:xc + chip] += 1

    return prob / np.maximum(count, 1e-6)