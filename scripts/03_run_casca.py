#!/usr/bin/env python3
"""Step 2: full 16-byte CA-SCA with the paper's concatenated dataset.

Protocol (paper §4.2 / §4.4, Algorithm 3):
  - Crop each byte's 700-sample window (diagnostic anchors; byte 2 = canonical).
  - Train ONE CNN on the concatenated windows of the 14 masked bytes (2..15)
    with plaintext-byte labels. Bytes 0 and 1 are unmasked in ASCAD and are
    excluded from training, matching §4.4.2.
  - Extract 64-d features per byte from the same N traces.
  - Score all 256 hypotheses with multi-bit CH (Eq. 11).

Each (seed, N) is an independent non-profiled full run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import sys as _casca_sys
_CASCA_ROOT = Path(__file__).resolve().parents[1]
if not (_CASCA_ROOT / 'src' / 'casca').is_dir():
    _CASCA_ROOT = Path(__file__).resolve().parents[3]
_casca_sys.path.insert(0, str(_CASCA_ROOT / "src"))
from casca.paths import raw_traces_path, repo_root, window_cache_path  # noqa: E402

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from casca.attacks.casca import Cascacnn, assert_architecture_matches_table2
from casca.attacks.clustering import assert_multibit_matches_sklearn, multibit_ch_vector, rank_of_key
from casca.data.ascad import cache_per_byte_windows, load_per_byte_windows, zscore_train_apply
from casca.metrics import guessing_entropy, success_rate
from casca.plotting import plot_ge_sr
from casca.attacks.casca import PAPER_BATCH_SIZE, PAPER_EPOCHS, extract_features, train_plaintext_cnn_fast
from casca.data.windows import ALL_BYTES, MASKED_BYTES, TRAIN_BYTES, UNMASKED_BYTES

RAW = raw_traces_path()
CACHE = window_cache_path()
OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "clean_reimpl_full16"
DEFAULT_N = (12000, 20000, 30000, 40000, 50000, 60000)
DEFAULT_SEEDS = tuple(range(10))


def concatenate_train_set(windows: np.ndarray, plaintext: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Stack the 14 masked-byte windows of the first n traces."""
    chunks_x = []
    chunks_y = []
    for b in TRAIN_BYTES:
        chunks_x.append(windows[:n, b])
        chunks_y.append(plaintext[:n, b])
    return np.concatenate(chunks_x, axis=0), np.concatenate(chunks_y, axis=0)


def run_one(
    windows: np.ndarray,
    plaintext: np.ndarray,
    true_key: np.ndarray,
    seed: int,
    n: int,
    n_epochs: int,
    out_dir: Path,
) -> dict:
    t0 = time.time()
    x_cat, y_cat = concatenate_train_set(windows, plaintext, n)
    # Z-score statistics from the concatenated training set, then apply to
    # every byte's attack windows (including unmasked 0/1, which were not trained on).
    extras = [windows[:n, b] for b in ALL_BYTES]
    scaled = zscore_train_apply(x_cat, *extras)
    x_cat_z = scaled[0]
    per_byte_z = {b: scaled[1 + b] for b in ALL_BYTES}

    result = train_plaintext_cnn_fast(x_cat_z, y_cat, seed=seed, n_epochs=n_epochs)

    ranks = []
    ch_stack = np.empty((16, 256), dtype=np.float64)
    for b in ALL_BYTES:
        feats = extract_features(result.model, per_byte_z[b])
        ch = multibit_ch_vector(feats, plaintext[:n, b])
        ch_stack[b] = ch
        ranks.append(rank_of_key(ch, int(true_key[b])))
        print(
            f"  byte {b:2d}  rank={ranks[-1]:3d}  true_ch={ch[int(true_key[b])]:.3f}  "
            f"argmax=0x{int(ch.argmax()):02x}{'  (unmasked)' if b in UNMASKED_BYTES else ''}",
            flush=True,
        )

    rec = {
        "protocol": "alg3_concat_masked14",
        "seed": seed,
        "N": n,
        "true_key": [int(k) for k in true_key],
        "ranks": ranks,
        "n_params": result.model.n_params(),
        "final_loss": result.final_loss,
        "final_acc": result.final_acc,
        "n_steps": result.n_steps,
        "train_seconds": result.seconds,
        "elapsed_seconds": time.time() - t0,
        "n_train_windows": int(len(x_cat)),
        "train_bytes": list(TRAIN_BYTES),
        "masked_recovered": int(sum(r == 0 for i, r in enumerate(ranks) if i in MASKED_BYTES)),
        "all_recovered": int(sum(r == 0 for r in ranks)),
        "full_key_rank0": bool(all(r == 0 for r in ranks)),
    }
    stem = out_dir / "runs" / f"seed{seed}_N{n}"
    stem.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(stem.with_suffix(".npz"), ch_vectors=ch_stack, ranks=np.asarray(ranks, dtype=np.int16))
    rec["ch_vector_file"] = str(stem.with_suffix(".npz"))
    stem.with_suffix(".json").write_text(json.dumps(rec, indent=2))
    print(
        f"[full16] seed={seed} N={n} recovered {rec['all_recovered']}/16 "
        f"(masked {rec['masked_recovered']}/14)  loss={result.final_loss:.4f}",
        flush=True,
    )
    del result
    return rec


def summarize(runs: list[dict], out_dir: Path) -> None:
    ns = sorted({r["N"] for r in runs})
    per_byte_rows = []
    combined_rows = []
    for n in ns:
        subset = [r for r in runs if r["N"] == n]
        ranks_mat = np.array([r["ranks"] for r in subset], dtype=np.int64)  # (trials, 16)
        n_trials = len(subset)
        for b in ALL_BYTES:
            col = ranks_mat[:, b]
            per_byte_rows.append(
                {
                    "N": n,
                    "byte": b,
                    "masked": b in MASKED_BYTES,
                    "n_trials": n_trials,
                    "GE": guessing_entropy(col),
                    "SR_1": success_rate(col, 1),
                    "SR_5": success_rate(col, 5),
                    "SR_10": success_rate(col, 10),
                    "ranks": col.tolist(),
                }
            )
        ge_all = float(ranks_mat.mean())
        ge_masked = float(ranks_mat[:, list(MASKED_BYTES)].mean())
        full_sr1 = float(np.mean(np.all(ranks_mat == 0, axis=1)))
        masked_sr1 = float(np.mean(np.all(ranks_mat[:, list(MASKED_BYTES)] == 0, axis=1)))
        combined_rows.append(
            {
                "N": n,
                "n_trials": n_trials,
                "GE_mean_16": ge_all,
                "GE_mean_masked14": ge_masked,
                "SR1_fullkey": full_sr1,
                "SR1_masked14": masked_sr1,
                "mean_bytes_recovered": float(np.mean(np.sum(ranks_mat == 0, axis=1))),
                "mean_masked_recovered": float(np.mean(np.sum(ranks_mat[:, list(MASKED_BYTES)] == 0, axis=1))),
            }
        )
        print(
            f"[summary] N={n}  GE_16={ge_all:.1f}  GE_masked={ge_masked:.1f}  "
            f"fullkey_SR1={full_sr1:.2f}  masked14_SR1={masked_sr1:.2f}  "
            f"mean_recovered={combined_rows[-1]['mean_bytes_recovered']:.1f}/16",
            flush=True,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    byte_csv = out_dir / "ge_sr_per_byte.csv"
    comb_csv = out_dir / "ge_sr_combined.csv"
    pd.DataFrame(per_byte_rows).to_csv(byte_csv, index=False)
    pd.DataFrame(combined_rows).to_csv(comb_csv, index=False)
    (out_dir / "ge_sr_per_byte.json").write_text(json.dumps(per_byte_rows, indent=2))
    (out_dir / "ge_sr_combined.json").write_text(json.dumps(combined_rows, indent=2))

    # Combined GE plot uses mean rank across 16 bytes as a single GE_N curve.
    plot_df = pd.DataFrame(
        {
            "N": [r["N"] for r in combined_rows],
            "GE": [r["GE_mean_16"] for r in combined_rows],
            "SR_1": [r["SR1_fullkey"] for r in combined_rows],
            "SR_5": [r["SR1_masked14"] for r in combined_rows],
            "SR_10": [r["mean_bytes_recovered"] / 16.0 for r in combined_rows],
        }
    )
    tmp = out_dir / "_plot_combined.csv"
    plot_df.to_csv(tmp, index=False)
    plot_ge_sr(tmp, out_dir / "combined", title_suffix="  [mean 16-byte / full-key SR_1]")
    tmp.unlink(missing_ok=True)

    # Per-byte GE heatmap-style lines
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    byte_df = pd.DataFrame(per_byte_rows)
    for b in ALL_BYTES:
        sub = byte_df[byte_df["byte"] == b]
        ls = "--" if b in UNMASKED_BYTES else "-"
        ax.plot(sub["N"], sub["GE"], ls=ls, marker="o", label=f"b{b}")
    ax.set_xlabel("N")
    ax.set_ylabel("GE_N")
    ax.set_title("Per-byte guessing entropy (dashed = unmasked bytes 0,1)")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "ge_per_byte.png", dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--N", type=int, nargs="+", default=list(DEFAULT_N))
    p.add_argument("--epochs", type=int, default=PAPER_EPOCHS)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--out", type=Path, default=OUT_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.seeds = [0]
        args.N = [2000]
        args.epochs = 2
        args.out = args.out / "smoke"

    print("[full16] architecture + CH self-checks", flush=True)
    assert_architecture_matches_table2(Cascacnn())
    assert_multibit_matches_sklearn()
    print("[full16] OK", flush=True)

    if not CACHE.exists():
        cache_per_byte_windows(RAW, CACHE)
    data = load_per_byte_windows(CACHE)
    windows, plaintext, true_key = data["windows"], data["plaintext"], data["true_key"]
    print(
        f"[full16] windows={windows.shape}  key={true_key.tolist()}\n"
        f"         train_bytes={list(TRAIN_BYTES)}  epochs={args.epochs}  batch={PAPER_BATCH_SIZE}",
        flush=True,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    runs = []
    total = len(args.seeds) * len(args.N)
    i = 0
    t_all = time.time()
    # N-major order so the paper's 12k checkpoint finishes across seeds first.
    for n in args.N:
        for seed in args.seeds:
            i += 1
            json_path = args.out / "runs" / f"seed{seed}_N{n}.json"
            if json_path.exists() and not args.smoke:
                rec = json.loads(json_path.read_text())
                print(f"[full16] resume skip {i}/{total} seed={seed} N={n} ranks={rec['ranks']}", flush=True)
                runs.append(rec)
                continue
            print(f"\n======== full16  {i}/{total}  seed={seed} N={n} ========", flush=True)
            rec = run_one(windows, plaintext, true_key, seed, n, args.epochs, args.out)
            runs.append(rec)
            summarize(runs, args.out)

    # Re-read every run on disk so a later --N {2k,5k,10k} pass cannot wipe 12k–60k.
    all_on_disk = []
    for p in sorted((args.out / "runs").glob("seed*_N*.json")):
        all_on_disk.append(json.loads(p.read_text()))
    if all_on_disk:
        summarize(all_on_disk, args.out)
    meta = {
        "seeds": list(args.seeds),
        "N": sorted({int(r["N"]) for r in all_on_disk} | set(args.N)),
        "epochs": args.epochs,
        "paper_batch_size": PAPER_BATCH_SIZE,
        "train_bytes": list(TRAIN_BYTES),
        "elapsed_seconds": time.time() - t_all,
        "independent_trial": "fresh seed -> train concatenated CNN -> CH per byte",
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[full16] finished in {meta['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
