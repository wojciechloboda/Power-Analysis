#!/usr/bin/env python3
"""3-seed attention-combination screen on the CA-SCA 4-token conv stack.

Reuses concat_attack.run_one and existing Cascacnn / make_h2("A") flags.
Does not modify reported CA-SCA or AA-CASCA experiment scripts.
Baseline cells at N∈{12k,20k} seeds 0–2 are copied from clean_reimpl_full16.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import sys as _casca_sys
_CASCA_ROOT = Path(__file__).resolve().parents[3]
if not (_CASCA_ROOT / 'src' / 'casca').is_dir():
    _CASCA_ROOT = Path(__file__).resolve().parents[3]
_casca_sys.path.insert(0, str(_CASCA_ROOT / "src"))
from casca.paths import raw_traces_path, repo_root, window_cache_path  # noqa: E402

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from casca.attacks.aacasca import VARIANT_ORDER, make_variant  # noqa: E402
from casca.runner import run_one  # noqa: E402
from casca.data.ascad import cache_per_byte_windows, load_per_byte_windows  # noqa: E402
from casca.attacks.casca import PAPER_BATCH_SIZE, PAPER_EPOCHS, PAPER_LR, _device  # noqa: E402
from casca.data.windows import MASKED_BYTES  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
RAW = raw_traces_path()
CACHE = window_cache_path()
CASCA_RUNS = REPO / "results" / "clean_reimpl_full16" / "runs"
OUT = REPO / "results" / "attn_combine_screen_4tok"

NS = (12000, 20000)
SEEDS = (0, 1, 2)
EPOCHS = PAPER_EPOCHS
HEAD_DH = 8


def dest_path(variant: str, seed: int, n: int) -> Path:
    return OUT / "runs" / f"{variant}_seed{seed}_N{n}.json"


def attack_seconds(rec: dict) -> float:
    train = float(rec["train_seconds"])
    ch = rec.get("key_recovery_seconds")
    if ch is None and rec.get("elapsed_seconds") is not None:
        ch = max(float(rec["elapsed_seconds"]) - train, 0.0)
    if ch is None:
        ch = 0.0
    return train + float(ch)


def ge_masked14(ranks) -> float:
    ranks = np.asarray(ranks, dtype=float)
    return float(ranks[list(MASKED_BYTES)].mean())


def reuse_baseline(seed: int, n: int) -> dict:
    src = CASCA_RUNS / f"seed{seed}_N{n}.json"
    rec = json.loads(src.read_text())
    if int(rec.get("n_params", -1)) != 75288:
        raise SystemExit(f"baseline reuse {src.name} n_params={rec.get('n_params')}")
    if int(rec.get("n_steps", -1)) < 1000:
        raise SystemExit(f"baseline reuse {src.name} n_steps={rec.get('n_steps')} (not 200 ep)")
    dest = dest_path("baseline", seed, n)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out = {
        **rec,
        "tag": "baseline",
        "variant": "baseline",
        "source": str(src.relative_to(REPO)),
        "reused": True,
    }
    dest.write_text(json.dumps(out, indent=2) + "\n")
    npz = src.with_suffix(".npz")
    if npz.exists():
        shutil.copy2(npz, dest.with_suffix(".npz"))
    return out


def write_summary(rows: list[dict]) -> pd.DataFrame:
    summary = []
    for variant in VARIANT_ORDER:
        for n in NS:
            sub = [r for r in rows if r["variant"] == variant and int(r["N"]) == n]
            if len(sub) != len(SEEDS):
                raise SystemExit(f"{variant} N={n}: got {len(sub)} seeds, want {len(SEEDS)}")
            recs = [int(np.sum(np.asarray(r["ranks"]) == 0)) for r in sub]
            ges = [ge_masked14(r["ranks"]) for r in sub]
            times = [attack_seconds(r) for r in sub]
            summary.append(
                {
                    "variant": variant,
                    "N": n,
                    "n_seeds": len(sub),
                    "n_params": int(sub[0]["n_params"]),
                    "mean_bytes_recovered": float(np.mean(recs)),
                    "mean_GE_masked14": float(np.mean(ges)),
                    "mean_attack_seconds": float(np.mean(times)),
                    "per_seed_recovered": recs,
                    "per_seed_GE_masked14": [round(g, 4) for g in ges],
                    "per_seed_attack_seconds": [round(t, 3) for t in times],
                }
            )
    df = pd.DataFrame(summary)
    df.to_csv(OUT / "summary.csv", index=False)
    (OUT / "summary.json").write_text(df.to_json(orient="records", indent=2))
    return df


def print_table(df: pd.DataFrame) -> None:
    print("\n=== attention-combination screen (4 tokens, 200 ep, seeds 0–2) ===", flush=True)
    print(
        f"{'variant':<16} {'N':>6} {'params':>8} {'rec/16':>8} {'GE_14':>8} {'time_s':>8}",
        flush=True,
    )
    for _, r in df.iterrows():
        print(
            f"{r['variant']:<16} {int(r['N']):>6} {int(r['n_params']):>8} "
            f"{r['mean_bytes_recovered']:8.2f} {r['mean_GE_masked14']:8.2f} "
            f"{r['mean_attack_seconds']:8.1f}",
            flush=True,
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()
    if args.workers != 1:
        raise SystemExit(
            "This machine has one MPS GPU. Two workers share it and inflate "
            "wall-clock (seen on earlier fills). Run sequential: --workers 1."
        )

    dev = _device()
    print(f"[screen] device={dev}  workers=1 (MPS; parallel would contend)", flush=True)
    if str(dev) != "mps":
        raise SystemExit(f"refusing silent CPU/CUDA fallback; device={dev}")

    models = {v: make_variant(v) for v in VARIANT_ORDER}
    for v, m in models.items():
        x = torch.zeros(2, 1, 700)
        with torch.no_grad():
            feats = m.extract_features(x)
            logits = m(x)
        ntok = int(getattr(m, "n_tokens", 4))
        print(f"  {v:<16} tokens={ntok} params={m.n_params()} feats={tuple(feats.shape)} logits={tuple(logits.shape)}")
        if ntok != 4 or tuple(feats.shape) != (2, 64) or tuple(logits.shape) != (2, 256):
            raise SystemExit(f"{v} shape/token mismatch")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    (OUT / "protocol.json").write_text(
        json.dumps(
            {
                "conv": "CA-SCA Table 2: (4,k=7,s=7), (8,k=5,s=5), (16,k=5,s=5)",
                "n_tokens": 4,
                "epochs": EPOCHS,
                "batch_size": PAPER_BATCH_SIZE,
                "lr": PAPER_LR,
                "adam_eps": 1e-7,
                "loss": "CrossEntropy",
                "N": list(NS),
                "seeds": list(SEEDS),
                "twohead_dh_per_head": HEAD_DH,
                "twohead_note": "d_h=8 per head so total width stays 16",
                "baseline_reused_from": "results/clean_reimpl_full16",
                "variants": {
                    "baseline": "Cascacnn flatten",
                    "square_only": "tok*tok, flatten",
                    "attn_only": "make_h2('A', use_gate=False)",
                    "attn_square": "make_h2('A', use_gate=False, self_multiply=True)",
                    "twohead_prod": "(A1V1)*(A2V2), d_h=8",
                    "twohead_concat": "concat heads, d_h=8",
                },
            },
            indent=2,
        )
        + "\n"
    )

    if not CACHE.exists():
        cache_per_byte_windows(RAW, CACHE)
    data = load_per_byte_windows(CACHE)

    jobs = [(v, s, n) for v in VARIANT_ORDER for n in NS for s in SEEDS]
    t_all = time.time()
    done = 0
    for variant, seed, n in jobs:
        dest = dest_path(variant, seed, n)
        if dest.exists() and variant != "baseline":
            print(f"[screen] skip {variant} seed={seed} N={n}", flush=True)
            done += 1
            continue
        if variant == "baseline":
            print(f"\n======== reuse baseline seed={seed} N={n} ========", flush=True)
            reuse_baseline(seed, n)
            done += 1
            continue
        print(
            f"\n======== {done + 1}/{len(jobs)}  {variant}  seed={seed} N={n} ========",
            flush=True,
        )
        rec = run_one(
            data["windows"],
            data["plaintext"],
            data["true_key"],
            seed,
            n,
            EPOCHS,
            OUT,
            model_fn=lambda v=variant: make_variant(v),
            tag=variant,
        )
        rec["variant"] = variant
        rec["protocol"] = {
            "epochs": EPOCHS,
            "batch_size": PAPER_BATCH_SIZE,
            "lr": PAPER_LR,
            "adam_eps": 1e-7,
            "loss": "CrossEntropy",
            "twohead_dh_per_head": HEAD_DH,
        }
        dest.write_text(json.dumps(rec, indent=2) + "\n")
        done += 1

    rows = []
    for variant, seed, n in jobs:
        rec = json.loads(dest_path(variant, seed, n).read_text())
        rec["variant"] = variant
        rows.append(rec)
    df = write_summary(rows)
    print_table(df)
    wall = time.time() - t_all
    print(f"\n[screen] wall-clock {wall:.1f}s ({wall / 60.0:.1f} min)", flush=True)
    (OUT / "WALL_CLOCK.txt").write_text(f"{wall:.3f}\n")


if __name__ == "__main__":
    main()
