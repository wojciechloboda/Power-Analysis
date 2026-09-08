"""Concatenated 14-byte train + 16-byte multi-bit CH attack (Algorithm 3)."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch.nn as nn

from casca.attacks.clustering import multibit_ch_vector, rank_of_key
from casca.data.ascad import zscore_train_apply
from casca.metrics import guessing_entropy, success_rate
from casca.attacks.casca import extract_features, train_plaintext_cnn_fast
from casca.data.windows import ALL_BYTES, MASKED_BYTES, TRAIN_BYTES, UNMASKED_BYTES, concatenate_train_set


def run_one(
    windows: np.ndarray,
    plaintext: np.ndarray,
    true_key: np.ndarray,
    seed: int,
    n: int,
    n_epochs: int,
    out_dir: Path,
    model_fn: Callable[[], nn.Module],
    tag: str,
    max_steps: int | None = None,
) -> dict:
    t0 = time.time()
    t_ch0 = None
    x_cat, y_cat = concatenate_train_set(windows, plaintext, n)
    extras = [windows[:n, b] for b in ALL_BYTES]
    scaled = zscore_train_apply(x_cat, *extras)
    x_cat_z = scaled[0]
    per_byte_z = {b: scaled[1 + b] for b in ALL_BYTES}

    result = train_plaintext_cnn_fast(
        x_cat_z,
        y_cat,
        seed=seed,
        n_epochs=n_epochs,
        model_fn=model_fn,
        max_steps=max_steps,
    )
    train_seconds = result.seconds

    t_ch0 = time.time()
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
    key_recovery_seconds = time.time() - t_ch0

    rec = {
        "tag": tag,
        "seed": seed,
        "N": n,
        "true_key": [int(k) for k in true_key],
        "ranks": ranks,
        "n_params": result.model.n_params(),
        "final_loss": result.final_loss,
        "final_acc": result.final_acc,
        "n_steps": result.n_steps,
        "train_seconds": train_seconds,
        "key_recovery_seconds": key_recovery_seconds,
        "elapsed_seconds": time.time() - t0,
        "n_train_windows": int(len(x_cat)),
        "train_bytes": list(TRAIN_BYTES),
        "masked_recovered": int(sum(r == 0 for i, r in enumerate(ranks) if i in MASKED_BYTES)),
        "all_recovered": int(sum(r == 0 for r in ranks)),
        "full_key_rank0": bool(all(r == 0 for r in ranks)),
        "n_tokens": int(getattr(result.model, "n_tokens", 4)),
    }
    stem = out_dir / "runs" / f"{tag}_seed{seed}_N{n}"
    stem.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        stem.with_suffix(".npz"),
        ch_vectors=ch_stack,
        ranks=np.asarray(ranks, dtype=np.int16),
    )
    rec["ch_vector_file"] = str(stem.with_suffix(".npz"))
    stem.with_suffix(".json").write_text(json.dumps(rec, indent=2))
    print(
        f"[{tag}] seed={seed} N={n} recovered {rec['all_recovered']}/16 "
        f"(masked {rec['masked_recovered']}/14)  train={train_seconds:.1f}s  "
        f"ch={key_recovery_seconds:.1f}s  loss={result.final_loss:.4f}",
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
        ranks_mat = np.array([r["ranks"] for r in subset], dtype=np.int64)
        n_trials = len(subset)
        rec_per_seed = np.sum(ranks_mat == 0, axis=1).astype(float)
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
        combined_rows.append(
            {
                "N": n,
                "n_trials": n_trials,
                "GE_mean_16": float(ranks_mat.mean()),
                "GE_mean_masked14": float(ranks_mat[:, list(MASKED_BYTES)].mean()),
                "SR1_fullkey": float(np.mean(np.all(ranks_mat == 0, axis=1))),
                "SR1_masked14": float(np.mean(np.all(ranks_mat[:, list(MASKED_BYTES)] == 0, axis=1))),
                "mean_bytes_recovered": float(rec_per_seed.mean()),
                "std_bytes_recovered": float(rec_per_seed.std(ddof=1) if n_trials > 1 else 0.0),
                "mean_masked_recovered": float(np.mean(np.sum(ranks_mat[:, list(MASKED_BYTES)] == 0, axis=1))),
                "mean_train_seconds": float(np.mean([r["train_seconds"] for r in subset])),
                "mean_key_recovery_seconds": float(np.mean([r["key_recovery_seconds"] for r in subset])),
            }
        )
        c = combined_rows[-1]
        print(
            f"[summary] N={n} trials={n_trials} GE_16={c['GE_mean_16']:.1f}  "
            f"GE_masked={c['GE_mean_masked14']:.1f}  fullkey_SR1={c['SR1_fullkey']:.2f}  "
            f"mean_recovered={c['mean_bytes_recovered']:.1f}/16  "
            f"std_recovered={c['std_bytes_recovered']:.2f}",
            flush=True,
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_byte_rows).to_csv(out_dir / "ge_sr_per_byte.csv", index=False)
    pd.DataFrame(combined_rows).to_csv(out_dir / "ge_sr_combined.csv", index=False)
    (out_dir / "ge_sr_per_byte.json").write_text(json.dumps(per_byte_rows, indent=2))
    (out_dir / "ge_sr_combined.json").write_text(json.dumps(combined_rows, indent=2))
