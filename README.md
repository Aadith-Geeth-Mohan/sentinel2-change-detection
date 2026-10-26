# GPU-Accelerated Bi-temporal Change Detection (Sentinel-2 / OSCD)

Real bi-temporal change detection on Copernicus Sentinel-2 imagery, with a
dedicated GPU analysis work package (scaling curves, AMP / `torch.compile`,
profiling, GPU-vs-CPU benchmark). V2 of the EuroSAT course project — see
`PLAN_V2.md` in the parent folder for the full rationale.

## Data

**OSCD (ONERA Satellite Change Detection)** — real Sentinel-2 bi-temporal
pairs (~24 city AOIs, 13 bands, per-pixel change labels, predefined
train/test city split). Download (free, IEEE DataPort account required):

- Images:  "Onera Satellite Change Detection dataset - Images.zip"
- Labels:  "Onera Satellite Change Detection dataset - Train Labels.zip"
           "Onera Satellite Change Detection dataset - Test Labels.zip"

Upload all three zips to `MyDrive/s2cd/raw/oscd/` in Google Drive
(drag & drop via drive.google.com). The parser discovers the exact layout
inside the zips automatically — don't rename anything.

## Colab workflow (in order)

| Notebook | Purpose |
|---|---|
| `00_setup` | clone repo, `pip install -e .`, mount Drive, GPU check |
| `01_download_oscd` | verify zips on Drive, extract, show discovered cities |
| `02_build_dataset` | chip cities → memory-mapped splits + manifest → Drive |
| `03_train` | train Siamese U-Net (AMP, resume-capable, ckpt → Drive) |
| `04_gpu_analysis` | batch-size scaling + AMP × compile grid (G1/G3) |
| `05_benchmark` | GPU vs CPU benchmark, formal methodology (G5) |
| `06_report_figures` | test metrics, per-city change maps, figures → Drive |

Suggested runtime: **A100** for 02–03, any GPU for 04–06.

## Storage contract

- **Drive (`MyDrive/s2cd/`)** is the permanent source of truth:
  `raw/oscd/*.zip`, `processed/chips_{split}.zip + manifest.json`,
  `runs/<exp>/{last.ckpt, best.ckpt, history.json}`, `reports/`.
- **Scratch (`/content`)** is disposable and repopulated per session; all
  training reads go through local scratch, never Drive paths.
- Checkpoints are pushed to Drive every epoch; on session death you lose at
  most one epoch (`03_train` auto-resumes).

## Local development

```bash
python -m venv .venv && .venv\Scripts\activate   # or conda
pip install -e ".[local"
pytest
```

All logic lives in `src/s2cd/`; notebooks are thin drivers. Configs in
`configs/*.yaml` are the single source of truth for hyperparameters.