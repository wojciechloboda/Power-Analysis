# Companion code for the thesis results on ASCAD

## Data

Obtain the ANSSI ASCAD v1 **fixed-key** campaign, file `ATMega8515_raw_traces.h5` (100k samples × 60k traces, with plaintext / key / mask metadata). Place it at `data/ATMega8515_raw_traces.h5`, or set `ASCAD_DATA_DIR` to the directory that contains that file.

The first run of any attack script crops the 16 per-byte 700-sample windows into `results/per_byte_windows.npz` (override with `CASCA_WINDOW_CACHE`). Byte 2 is the official ASCAD window `[45400, 46100]`.

## Setup

Python 3.12. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Apple Silicon runs used MPS. Several training scripts refuse a silent CPU/CUDA fallback (`device` must be `mps`). CPA and the POI scan are NumPy-only.

## Reproduce figures and tables

| Artifact | Command |
|---|---|
| `ascad_poi_correlation_scan` | `python scripts/10_make_figures.py --figure ascad_poi_correlation_scan` |
| `plot1_individual_seeds`, `plot2_mean_ci` | `python scripts/10_make_figures.py --figure plot1_individual_seeds` |
| `plot_ge_and_val_loss` | `python scripts/10_make_figures.py --figure plot_ge_and_val_loss` |
| `aa-casca-plot2_mean_ci` | `python scripts/10_make_figures.py --figure aa-casca-plot2_mean_ci` |
| `plot_ge_and_val_loss_aa_attxatt_100ep` | `python scripts/10_make_figures.py --figure plot_ge_and_val_loss_aa_attxatt_100ep` |
| `plot_e25_e200_casca_{ge_bytes,time}` | `python scripts/10_make_figures.py --figure plot_e25_e200_casca_ge_bytes` |
| `plot_e25_casca_moc_{ge_bytes,time}` | `python scripts/10_make_figures.py --figure plot_e25_casca_moc_ge_bytes` |
| CPA thesis table | `python scripts/11_make_tables.py --table cpa` |
| 4-token attention table | `python scripts/11_make_tables.py --table attention` |
| Square-token table | `python scripts/11_make_tables.py --table squaring` |

`10_make_figures.py --figure all` runs each plot module once. Figure PDFs/PNGs were not stored with the source results; the commands above regenerate them into `figures/` (and some plot scripts still write under `results/`). Stored CSVs/JSON used by those figures are in `results/`.

## Re-run attacks

| Script | What it is |
|---|---|
| `01_locate_poi.py` | N=4000 Pearson POI scan in each 700-sample window |
| `02_run_cpa.py` | HW-CPA, 10 random subsets (`RNG_BASE=20260823`), N=5k–40k |
| `03_run_casca.py` | CA-SCA Algorithm 3, concat 14 masked bytes, 200 epochs, batch 20000 |
| `04_run_aacasca.py` | Config E att×att (`use_gate=False`, `self_multiply=True`), 200 epochs |
| `05_run_moc.py` | One MOC net per byte (not concat) |
| `06_run_budget.py` | `--mode m10` (gated E vs CA-SCA) or `--mode attxatt100` |
| `07_run_variants.py` | `--screen 4tok` or `--screen square` |

CPA uses random subsets of the 50k profiling traces. CA-SCA / AA-CASCA / MOC use the **first N** traces. Seeds are `0..9` unless a script’s argparse default is smaller (MOC default 5 seeds; 4-token screen seeds 0–2).

`configs/*.yaml` document those constants. Scripts do not read YAML.
