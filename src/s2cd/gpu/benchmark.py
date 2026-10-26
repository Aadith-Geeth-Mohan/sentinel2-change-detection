"""G5: GPU vs CPU inference benchmark, formal methodology.

  * same host for both sides (one Colab session)
  * CPU threads pinned (torch default underuses cores otherwise)
  * cuda events for GPU, perf_counter for CPU
  * warmup excluded
  * reports compute-only AND end-to-end (incl. host->device copy)
"""

import os
import time

import numpy as np
import torch


def _measure(model, t0_cpu, t1_cpu, device, iters, warmup, amp=True):
    model.eval()
    with torch.no_grad():
        # warmup (also covers lazy CUDA init)
        for _ in range(warmup):
            data = (t0_cpu.to(device), t1_cpu.to(device))
            with torch.autocast("cuda", torch.float16, enabled=amp and device.type == "cuda"):
                model(*data)

        # end-to-end: includes host->device copy
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            data = (t0_cpu.to(device), t1_cpu.to(device))
            with torch.autocast("cuda", torch.float16, enabled=amp and device.type == "cuda"):
                model(*data)
        if device.type == "cuda":
            torch.cuda.synchronize()
        e2e = (time.perf_counter() - t0) / iters

        # compute-only: data already resident
        t0d, t1d = t0_cpu.to(device), t1_cpu.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                with torch.autocast("cuda", torch.float16, enabled=amp):
                    model(t0d, t1d)
            end.record()
            torch.cuda.synchronize()
            compute = start.elapsed_time(end) / (iters * 1000.0)
        else:
            t = time.perf_counter()
            for _ in range(iters):
                model(t0d, t1d)
            compute = (time.perf_counter() - t) / iters
    return compute, e2e


def run(model, batch, channels, chip, devices=("cuda", "cpu"), iters=20,
        warmup=10, amp=True) -> dict:
    torch.set_num_threads(os.cpu_count() or 2)  # pinned for fair CPU timing

    results = {}
    t0_cpu = torch.randn(batch, channels, chip, chip)
    t1_cpu = torch.randn(batch, channels, chip, chip)

    for name in devices:
        if name == "cuda" and not torch.cuda.is_available():
            print("cuda unavailable — skipping")
            continue
        dev = torch.device(name)
        m = model.to(dev)
        compute, e2e = _measure(m, t0_cpu, t1_cpu, dev, iters, warmup, amp)
        sps = batch / e2e
        results[name] = {"compute_sec": compute, "end_to_end_sec": e2e,
                         "samples_per_sec": sps, "threads": torch.get_num_threads()}
        print(f"{name.upper():4s}: compute {compute * 1000:8.2f} ms/batch | "
              f"end-to-end {e2e * 1000:8.2f} ms/batch | {sps:8.1f} samples/s "
              f"({torch.get_num_threads()} threads)")

    if "cuda" in results and "cpu" in results:
        sc = results["cpu"]["end_to_end_sec"] / results["cuda"]["end_to_end_sec"]
        scc = results["cpu"]["compute_sec"] / results["cuda"]["compute_sec"]
        results["speedup_end_to_end"] = sc
        results["speedup_compute_only"] = scc
        print(f"\n*** GPU speedup: {scc:.2f}x (compute-only) | "
              f"{sc:.2f}x (end-to-end incl. H2D) ***")
    return results