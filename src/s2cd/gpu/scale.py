"""G1/G3: throughput scaling curves + AMP x torch.compile grid.

Measurement rules:
  * model.eval() + no_grad — pure inference throughput
  * warmup iterations cover lazy init AND torch.compile recompiles,
    then timed iterations use cuda events (GPU) or perf_counter (CPU)
"""

import time

import numpy as np
import torch


@torch.no_grad()
def _timed_iters(model, t0, t1, iters: int, warmup: int, device: torch.device,
                 amp: bool) -> float:
    """Returns seconds per iteration (after warmup)."""
    for _ in range(warmup):
        with torch.autocast("cuda", torch.float16, enabled=amp and device.type == "cuda"):
            model(t0, t1)
    if device.type == "cuda":
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            with torch.autocast("cuda", torch.float16, enabled=amp and device.type == "cuda"):
                model(t0, t1)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / (iters * 1000.0)
    start = time.perf_counter()
    for _ in range(iters):
        model(t0, t1)
    return (time.perf_counter() - start) / iters


def throughput_at_batch(model, ds_sample, batch: int, device, iters=20,
                        warmup=8, amp=True) -> float:
    """samples/sec for one batch size on a fresh random batch."""
    n, c, h, w = ds_sample
    t0 = torch.randn(batch, c, h, w, device=device)
    t1 = torch.randn(batch, c, h, w, device=device)
    sec = _timed_iters(model, t0, t1, iters, warmup, device, amp)
    return batch / sec


def batch_scaling(model, sample_shape, device, batch_sizes=(8, 16, 32, 64, 128),
                  chip_hw=256, iters=20) -> list[dict]:
    """G1a: throughput vs batch size (eager + AMP)."""
    model.eval()
    rows = []
    for bs in batch_sizes:
        sps = throughput_at_batch(model, sample_shape, bs, device, iters=iters)
        rows.append({"batch": bs, "samples_per_sec": sps})
        print(f"  batch {bs:4d}: {sps:8.1f} samples/s")
    return rows


def amp_compile_grid(model_factory, sample_shape, device, batch=32,
                    chip_hw=256, iters=20) -> list[dict]:
    """G1b/G3: 2x2 grid of AMP on/off x torch.compile on/off.

    model_factory is called fresh for each cell — a compiled model must not
    be reused across configurations, and compile time is excluded via warmup.
    """
    rows = []
    for use_amp in [False, True]:
        for use_compile in [False, True]:
            model = model_factory().to(device)
            model.eval()
            if use_compile:
                try:
                    model = torch.compile(model)
                except Exception as e:
                    print(f"  compile unavailable: {e}")
                    rows.append({"amp": use_amp, "compile": False,
                                 "samples_per_sec": None})
                    continue
            sps = throughput_at_batch(model, sample_shape, batch, device,
                                      iters=iters, warmup=15, amp=use_amp)
            rows.append({"amp": use_amp, "compile": use_compile,
                         "samples_per_sec": sps})
            print(f"  AMP={int(use_amp)} compile={int(use_compile)}: {sps:8.1f} samples/s")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return rows