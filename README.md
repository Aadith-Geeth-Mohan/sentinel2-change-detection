# GPU-Accelerated Bi-temporal Change Detection (Sentinel-2)

Real per-pixel change detection on bi-temporal Copernicus Sentinel-2 imagery
(Siamese U-Net on the OSCD benchmark), a rigorous GPU measurement work
package (batch scaling, AMP × `torch.compile`, GPU-vs-CPU benchmark), and a
zero-shot transfer study applying the trained model unchanged to a
Google-Earth-Engine-acquired pair over Chennai, India.

## Results

**Change detection (OSCD test split — 10 unseen cities, 82 chips, best-val
checkpoint):**

| mIoU | change IoU | change F1 | precision | recall | pixel acc |
|---|---|---|---|---|---|
| 0.6617 | 0.3777 | 0.5483 | 0.5155 | 0.5855 | 0.9475 |

**GPU work package (Colab A100-SXM4-40GB, batch 16, 10×256×256):**

| Measurement | Result |
|---|---|
| GPU vs CPU compute speedup (12-thread CPU, same host) | **157.4×** |
| GPU vs CPU end-to-end speedup | **101.3×** |
| AMP + `torch.compile` combined (vs eager fp32) | **3.26×** (super-additive: naive product ≈ 1.97×) |
| Batch-size scaling 8 → 64 | 477 → 508 samples/s (+6.5% — memory-bandwidth-bound) |
| Host-to-device share of GPU e2e time | **36%** — the input pipeline, not the A100, is the next bottleneck |
| Full training (early stop, epoch 21) | **~2 minutes** on the A100 |

Practical meaning: two minutes of A100 training ≈ two CPU-hours of
forward-pass traffic alone.

## Method (summary)

- **Model** — Siamese U-Net (~7.8 M parameters): shared encoder over t0/t1,
  absolute-difference fusion at every encoder level, U-Net decoder with
  bilinear-upsampling + skip concatenation, 1×1 head → per-pixel 2-class
  logits (no global-pool collapse).
- **Input** — 10 Sentinel-2 bands (B02–B08, B8A, B11, B12); 20 m bands
  bilinearly resampled to the 10 m grid; reflectance = DN / 10000.
- **Normalization** — fixed per-band mean/std computed on the **train split
  only** (per-image z-scoring deliberately avoided: it would erase the
  absolute reflectance magnitude the change signal depends on).
- **Loss** — 0.5 · cross-entropy + 0.5 · soft Dice (change pixels are ~5% of
  the image; Dice keeps the rare class from collapsing).
- **Training** — 169 train chips (256×256, stride 128), AdamW 3e-4, cosine
  annealing, fp16 AMP, early stopping on val mIoU (patience 7); paired
  augmentation (same transform on t0, t1 and mask — pinned by test).
- **Inference** — `predict_city`: sliding-window over arbitrary-sized scenes
  (clamped final window, probability-averaged overlaps).
- **Evaluation** — all metrics derived from one accumulated 2×2 confusion
  matrix (never stores per-pixel arrays).

## Data

**OSCD (ONERA Satellite Change Detection)** — 24 city AOIs worldwide, each
with two co-registered Sentinel-2 acquisitions (t0, t1) and a per-pixel
binary change mask; official 14-city train / 10-city test split kept as-is
(prevents spatial leakage).

All three zips (~488 MB) live on the Hugging Face mirror
[`hkristen/oscd`](https://hf.co/datasets/hkristen/oscd). **Notebook 01
downloads them automatically** into `MyDrive/s2cd/raw/oscd/` (idempotent,
byte-size verified) — no IEEE DataPort account needed. The parser discovers
the exact in-zip layout automatically — don't rename anything.

## Colab workflow (in order)

| Notebook | Purpose |
|---|---|
| `00_setup` | clone repo, install package, mount Drive, GPU check |
| `01_download_oscd` | verify zips on Drive, extract, show discovered cities |
| `02_build_dataset` | chip cities → memory-mapped splits + manifest → Drive |
| `03_train` | train Siamese U-Net (AMP, resume-capable, ckpt → Drive) |
| `04_gpu_analysis` | batch-size scaling + AMP × compile grid |
| `05_benchmark` | GPU vs CPU benchmark, formal methodology |
| `06_report_figures` | test metrics, per-city change maps, figures → Drive |
| `07_chennai` | zero-shot transfer: GEE-acquired Chennai pair → change map |

Suggested runtime: **A100** for 02–03, any GPU for 04–07. Open notebooks via
their GitHub URLs; after a pull, restart the runtime so the kernel isn't
executing cached pre-pull modules (the clone cell purges them, the restart
guarantees it).

## Chennai — zero-shot domain transfer (branch `chennai`)

No labeled change-detection dataset exists for Chennai (OSCD covers 24 fixed
cities, Mumbai being the only Indian one), so phase 2 reframes the project
as a **zero-shot transfer study**: the OSCD-trained model is applied
*unchanged* (no fine-tuning, no Chennai labels) to a bi-temporal Sentinel-2
L2A pair over the full Chennai metro + fringe, acquired independently via
Google Earth Engine (`COPERNICUS/S2_SR_HARMONIZED`, least-cloudy scene per
date, same 10 pipeline bands, 20 m bands bilinear-resampled to 10 m at
export time — mirroring the OSCD preprocessing, same post-monsoon window
for both dates to suppress phenological "change").

Deliverables (notebook `07_chennai`, all pushed to `MyDrive/s2cd/reports/`):
`map_chennai.png` (t0 / t1 / P(change) city map),
`map_chennai_hotspots.png` (top-3 512-px cells by mean probability — the
qualitative-evidence figure), `change_prob_chennai.tif` (georeferenced
probability map, QGIS-ready) and `chennai_results.json` (scene IDs, dates,
cloud %, AOI, device, timing — the full reproducibility record). Without
labels the Chennai output is **qualitative by design**: signal = spatially
coherent high-probability clusters matching visible construction in the RGB
pair; the scored claims remain the OSCD test benchmark above. OSM-derived
labels for a scored Chennai evaluation are future work.

New modules on this branch: `src/s2cd/data/gee_acquire.py` (scene search +
Drive export; `earthengine-api` is deliberately not a package dependency,
like torch) and `src/s2cd/data/s2_l2a.py` (`L2AScene` — a lazy GeoTIFF
reader that duck-types the `read_city` ndarray contract, so `predict_city`
streams full-city scenes from disk window-by-window; also stitches EE
chunked exports automatically if the AOI outgrows EE's ~2 GB single-file
limit).

## Storage contract

- **Drive (`MyDrive/s2cd/`)** is the permanent source of truth:
  `raw/oscd/*.zip`, `processed/chips_{split}.zip + manifest.json`,
  `runs/<exp>/{last.ckpt, best.ckpt, history.json}`, `reports/`.
- **Scratch (`/content`)** is disposable and repopulated per session; all
  training reads go through local scratch, never Drive paths.
- Every fetch is **MD5-verified against the manifest** — a stale or partial
  Drive copy fails at session start, not mid-training.
- Checkpoints are pushed to Drive every epoch; on session death you lose at
  most one epoch (`03_train` auto-resumes with all RNG state restored).

## Repository structure

```
src/s2cd/
  data/    store.py (Drive/scratch contract) · oscd.py (dataset reader)
           build.py (chipping) · dataset.py (ChipDataset)
           gee_acquire.py (EE acquisition) · s2_l2a.py (lazy scene reader)
  models/  siamese_unet.py
  losses.py  metrics.py  train.py  evaluate.py  utils.py
  gpu/     batch scaling + AMP × compile grid + GPU/CPU benchmark
configs/   data.yaml · train.yaml   (single source of truth)
notebooks/colab/  00–07 (thin drivers — all logic lives in src/)
tests/     22 focused tests
```

## Testing

`pytest` — 22 focused tests, several pinning past bug-classes: the lazy
Chennai reader must be bit-identical to the eager ndarray through
`predict_city`; normalization stays fixed train-split stats (not per-image
z-scoring); paired augmentation must move the mask with the images (checked
over 30 random draws); the model head must retain spatial variance; EE
chunked-export stitching must refuse directories mixing two dates' chunks.

## Local development

```bash
python -m venv .venv && .venv\Scripts\activate   # or conda
pip install -e ".[local]"
pytest
```

All logic lives in `src/s2cd/`; notebooks are thin drivers. Configs in
`configs/*.yaml` are the single source of truth for hyperparameters.
`torch` is deliberately **not** a hard pip dependency — Colab ships a CUDA
build and pip would otherwise replace it with the CPU wheel.