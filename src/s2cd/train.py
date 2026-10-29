"""Training loop: AMP, optional torch.compile, per-epoch Drive checkpoint,
auto-resume (worst case loses one epoch), early stopping on val mIoU.
A re-run over a FINISHED run detects completion (early stop or full
schedule) and skips — the recorded checkpoint is never silently retrained."""

import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .data import DriveStore, ChipDataset
from .losses import CEDiceLoss
from .metrics import ChangeMetrics
from .models import SiameseUNet
from .utils import set_seed, load_config, get_device, unwrap_model, rng_state, set_rng_state


def build_datasets(store: DriveStore, manifest: dict, cfg: dict):
    stats = {"mean": manifest["stats"]["mean"], "std": manifest["stats"]["std"]}
    ds = {}
    for split, aug in [("train", cfg.get("augment", True)), ("val", False)]:
        d = store.fetch_chips(split)
        ds[split] = ChipDataset(d, stats, augment=aug)
    return ds


def make_loader(ds, cfg, shuffle: bool) -> DataLoader:
    return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=shuffle,
                       num_workers=cfg.get("num_workers", 2),
                       pin_memory=cfg.get("pin_memory", True),
                       persistent_workers=cfg.get("num_workers", 2) > 0)


def build_model(cfg: dict) -> SiameseUNet:
    return SiameseUNet(in_channels=cfg["in_channels"], base=cfg.get("base_channels", 64),
                       num_classes=cfg.get("num_classes", 2))


def _run_epoch(model, loader, criterion, device, cfg, optimizer=None, scaler=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    metrics = ChangeMetrics()
    total_loss, n = 0.0, 0
    amp = cfg.get("amp", True) and device.type == "cuda"
    for t0, t1, mask in loader:
        t0, t1, mask = t0.to(device, non_blocking=True), t1.to(device, non_blocking=True), \
            mask.to(device, non_blocking=True)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            logits = model(t0, t1)
            loss = criterion(logits, mask)
        if train_mode:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss.item()
        n += 1
        metrics.update(logits.float(), mask)
    return total_loss / max(n, 1), metrics


def _push(store: DriveStore, cfg: dict, run_dir: Path, history: list):
    (run_dir / "history.json").write_text(json.dumps(history, indent=1))
    for name in ["last.ckpt", "best.ckpt", "history.json"]:
        store.push_run_file(cfg["run_name"], name)


def train(cfg_path: str, repo_root: str | Path = ".", store: DriveStore | None = None):
    cfg = load_config(cfg_path)
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "cuda"))
    store = store or DriveStore()
    run_dir = Path("/content/runs") / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = store.read_manifest()
    if len(manifest["bands"]) != cfg["in_channels"]:
        raise ValueError(
            f"in_channels={cfg['in_channels']} != len(bands)={len(manifest['bands'])}. "
            "Align configs/train.yaml with configs/data.yaml.")

    ds = build_datasets(store, manifest, cfg)
    print(f"dataset: train={len(ds['train'])} val={len(ds['val'])} "
          f"(chips {manifest['chip_size']}px, bands {manifest['bands']})")
    train_loader = make_loader(ds["train"], cfg, shuffle=True)
    val_loader = make_loader(ds["val"], cfg, shuffle=False)

    model = build_model(cfg).to(device)
    print(f"parameters: {model.count_params():,}")
    if cfg.get("compile", False):
        try:
            model = torch.compile(model)
            print("torch.compile: enabled")
        except Exception as e:  # queue closed, old torch, triton missing...
            print(f"torch.compile unavailable ({e}); running eager")

    criterion = CEDiceLoss(cfg.get("ce_weight", 0.5), cfg.get("dice_weight", 0.5)).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-2))
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["num_epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.get("amp", True) and device.type == "cuda")

    # ---- resume ------------------------------------------------------------
    start_epoch, best_miou, patience_left = 0, -1.0, cfg.get("patience", 7)
    history = []
    ckpt_local = run_dir / "last.ckpt"
    store.fetch_run(cfg["run_name"])
    if ckpt_local.exists():
        # map_location MUST be cpu: a cuda-mapped RNG state tensor is a
        # cuda.ByteTensor and torch.set_rng_state rejects it. Optimizer state
        # is cast back onto the params' device by load_state_dict itself.
        ck = torch.load(ckpt_local, map_location="cpu", weights_only=False)
        unwrap_model(model).load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        set_rng_state(ck["rng"])
        start_epoch, best_miou = ck["epoch"] + 1, ck["best_miou"]
        history = json.loads((run_dir / "history.json").read_text())

        # a FINISHED run must never re-train: resuming past an early stop
        # would keep optimizing and invalidate the reported checkpoint.
        # history tells us why the loop ended — an early stop is exactly
        # 'patience' non-improving epochs after the best one.
        best_i = max(range(len(history)), key=lambda i: history[i]["val_miou"])
        ran_past_best = len(history) - 1 - best_i
        if start_epoch >= cfg["num_epochs"] or ran_past_best >= cfg.get("patience", 7):
            print(f"run {cfg['run_name']!r} already complete "
                  f"({len(history)} epochs, best val mIoU {best_miou:.4f}) — "
                  "keeping the finished checkpoint; skipping training")
            return history
        # a mid-run resume keeps the REMAINING patience, not a fresh budget
        patience_left = max(cfg.get("patience", 7) - ran_past_best, 1)
        print(f"resumed from epoch {start_epoch} (best mIoU {best_miou:.4f}, "
              f"{patience_left} epochs of patience left)")

    # ---- loop --------------------------------------------------------------
    for epoch in range(start_epoch, cfg["num_epochs"]):
        t0 = time.time()
        train_loss, train_metrics = _run_epoch(model, train_loader, criterion, device, cfg,
                                               optimizer, scaler)
        with torch.no_grad():
            val_loss, val_metrics = _run_epoch(model, val_loader, criterion, device, cfg)
        scheduler.step()
        dt = time.time() - t0

        vr = val_metrics.results()
        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                  "val_miou": vr["miou"], "val_change_iou": vr["change_iou"],
                  "seconds": round(dt, 1)}
        history.append(record)
        miou = record["val_miou"]
        if miou > best_miou:
            best_miou, star = miou, " *best*"
            patience_left = cfg.get("patience", 7)
            torch.save({"model": unwrap_model(model).state_dict(),
                        "cfg": cfg, "epoch": epoch, "val_miou": miou},
                       run_dir / "best.ckpt")
        else:
            star = ""
            patience_left -= 1
            if patience_left <= 0:
                print(f"early stopping at epoch {epoch + 1} (no val mIoU gain "
                      f"for {cfg.get('patience', 7)} epochs)")
                _push(store, cfg, run_dir, history)
                break
        print(f"[{epoch + 1:3d}/{cfg['num_epochs']}] {dt:5.1f}s | "
              f"train {train_loss:.4f} | val {val_loss:.4f} | "
              f"mIoU {miou:.4f} | {val_metrics.summary()}{star}")

        torch.save({"model": unwrap_model(model).state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best_miou": best_miou, "rng": rng_state()},
                   ckpt_local)
        _push(store, cfg, run_dir, history)

    print(f"done. best val mIoU: {best_miou:.4f}")
    return history