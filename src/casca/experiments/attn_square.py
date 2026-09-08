#!/usr/bin/env python3
"""Single-head attn_only vs attn_square across token counts.

Reuses make_h2 flags and concat_attack.run_one. Does not modify reported
CA-SCA / AA-CASCA experiment scripts or attn_combine_screen.py.

4-token (config A) cells are copied from results/attn_combine_screen_4tok.
All other token-resolution stores are gated H2 and are not reused.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
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

from casca.runner import run_one  # noqa: E402
from casca.data.ascad import cache_per_byte_windows, load_per_byte_windows  # noqa: E402
from casca.attacks.aacasca import CONV_CONFIGS, make_h2, measure_tokens  # noqa: E402
from casca.attacks.casca import PAPER_BATCH_SIZE, PAPER_EPOCHS, PAPER_LR, _device  # noqa: E402
from casca.data.windows import MASKED_BYTES  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
RAW = raw_traces_path()
CACHE = window_cache_path()
REUSE_4TOK = REPO / "results" / "attn_combine_screen_4tok" / "runs"
OUT = REPO / "results" / "attn_square_token_screen"

# Reduced screen: 4 / 16 / 23 / 35 tokens (A, D, E, C). B (8) and G (28)
# dropped to cut wall-clock. C keeps historical kernels (5,2,2) so it
# produces 35 tokens; (7,5,5)/(5,2,2) would be 32.
CONFIGS = ("A", "D", "E", "C")
LISTED_TOKENS = {"A": 4, "D": 16, "E": 23, "C": 35}
VARIANTS = ("attn_only", "attn_square")
NS = (20000,)
SEEDS = tuple(range(10))
EPOCHS = PAPER_EPOCHS
STEPS = {12000: 1800, 20000: 2800}


def make_model(config: str, variant: str):
    if variant == "attn_only":
        return make_h2(config, use_gate=False)
    if variant == "attn_square":
        return make_h2(config, use_gate=False, self_multiply=True)
    raise ValueError(variant)


def dest_path(config: str, variant: str, seed: int, n: int) -> Path:
    return OUT / "runs" / f"{variant}_{config}_seed{seed}_N{n}.json"


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


def confirm_tokens() -> list[dict]:
    rows = []
    print("=== measured token counts (700-sample valid padding) ===", flush=True)
    print(
        f"{'cfg':<4} {'kernels':<12} {'strides':<12} {'got':>4} {'listed':>7} {'ao_p':>8} {'as_p':>8}",
        flush=True,
    )
    for name in CONFIGS:
        cfg = CONV_CONFIGS[name]
        got = measure_tokens(cfg.kernels, cfg.strides)
        listed = LISTED_TOKENS[name]
        ao = make_h2(name, use_gate=False)
        asq = make_h2(name, use_gate=False, self_multiply=True)
        mark = "OK" if got == listed else "DIFFERS"
        print(
            f"{name:<4} {str(cfg.kernels):<12} {str(cfg.strides):<12} "
            f"{got:4d} {listed:7d} {ao.n_params():8d} {asq.n_params():8d}  {mark}",
            flush=True,
        )
        if got != listed:
            print(f"  WARNING: {name} produced {got} tokens, listed {listed}", flush=True)
        if ao.n_tokens != got or asq.n_tokens != got:
            raise SystemExit(f"{name} model token mismatch")
        if ao.n_params() != asq.n_params():
            raise SystemExit(f"{name} attn_only/attn_square param mismatch")
        rows.append(
            {
                "config": name,
                "kernels": list(cfg.kernels),
                "strides": list(cfg.strides),
                "n_tokens": got,
                "listed_tokens": listed,
                "attn_only_params": ao.n_params(),
                "attn_square_params": asq.n_params(),
            }
        )
    k755_c = measure_tokens((7, 5, 5), (5, 2, 2))
    print(
        f"note: kernels (7,5,5) + strides (5,2,2) -> {k755_c} tokens "
        "(Config C uses kernels (5,2,2) to reach 35)",
        flush=True,
    )
    return rows


def reuse_4tok(variant: str, seed: int, n: int) -> dict:
    src = REUSE_4TOK / f"{variant}_seed{seed}_N{n}.json"
    rec = json.loads(src.read_text())
    if int(rec.get("n_params", -1)) != 80264:
        raise SystemExit(f"reuse {src.name} n_params={rec.get('n_params')}")
    if int(rec.get("n_tokens", -1)) != 4:
        raise SystemExit(f"reuse {src.name} n_tokens={rec.get('n_tokens')}")
    if int(rec.get("n_steps", -1)) != STEPS[n]:
        raise SystemExit(f"reuse {src.name} n_steps={rec.get('n_steps')} want {STEPS[n]}")
    proto = rec.get("protocol") or {}
    if int(proto.get("epochs", -1)) != EPOCHS:
        raise SystemExit(f"reuse {src.name} epochs={proto.get('epochs')}")
    dest = dest_path("A", variant, seed, n)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out = {
        **rec,
        "tag": f"{variant}_A",
        "variant": variant,
        "config": "A",
        "kernels": list(CONV_CONFIGS["A"].kernels),
        "strides": list(CONV_CONFIGS["A"].strides),
        "use_gate": False,
        "self_multiply": variant == "attn_square",
        "source": str(src.relative_to(REPO)),
        "reused": True,
        "failed": False,
    }
    dest.write_text(json.dumps(out, indent=2) + "\n")
    npz = src.with_suffix(".npz")
    if npz.exists():
        shutil.copy2(npz, dest.with_suffix(".npz"))
    return out


def write_failure(config: str, variant: str, seed: int, n: int, err: str) -> dict:
    rec = {
        "tag": f"{variant}_{config}",
        "variant": variant,
        "config": config,
        "seed": seed,
        "N": n,
        "failed": True,
        "error": err,
        "n_tokens": measure_tokens(CONV_CONFIGS[config].kernels, CONV_CONFIGS[config].strides),
        "n_params": make_model(config, variant).n_params(),
        "kernels": list(CONV_CONFIGS[config].kernels),
        "strides": list(CONV_CONFIGS[config].strides),
        "use_gate": False,
        "self_multiply": variant == "attn_square",
        "reused": False,
    }
    dest = dest_path(config, variant, seed, n)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rec, indent=2) + "\n")
    return rec


def write_summary(rows: list[dict]) -> pd.DataFrame:
    summary = []
    for config in CONFIGS:
        for variant in VARIANTS:
            for n in NS:
                sub = [
                    r
                    for r in rows
                    if r["config"] == config and r["variant"] == variant and int(r["N"]) == n
                ]
                if len(sub) != len(SEEDS):
                    raise SystemExit(f"{variant} {config} N={n}: got {len(sub)} records")
                failed = [r for r in sub if r.get("failed")]
                ok = [r for r in sub if not r.get("failed")]
                recs = [int(np.sum(np.asarray(r["ranks"]) == 0)) for r in ok]
                ges = [ge_masked14(r["ranks"]) for r in ok]
                times = [attack_seconds(r) for r in ok]
                summary.append(
                    {
                        "config": config,
                        "variant": variant,
                        "strides": str(tuple(CONV_CONFIGS[config].strides)),
                        "kernels": str(tuple(CONV_CONFIGS[config].kernels)),
                        "n_tokens": int(sub[0]["n_tokens"]),
                        "N": n,
                        "n_seeds": len(ok),
                        "n_failed": len(failed),
                        "n_params": int(sub[0]["n_params"]),
                        "mean_bytes_recovered": float(np.mean(recs)) if recs else float("nan"),
                        "mean_GE_masked14": float(np.mean(ges)) if ges else float("nan"),
                        "mean_attack_seconds": float(np.mean(times)) if times else float("nan"),
                        "per_seed_recovered": recs,
                        "per_seed_GE_masked14": [round(g, 4) for g in ges],
                        "per_seed_attack_seconds": [round(t, 3) for t in times],
                        "failed_seeds": [int(r["seed"]) for r in failed],
                    }
                )
    df = pd.DataFrame(summary)
    df.to_csv(OUT / "summary.csv", index=False)
    (OUT / "summary.json").write_text(df.to_json(orient="records", indent=2))
    return df


def _cell(df: pd.DataFrame, config: str, variant: str, n: int) -> pd.Series:
    hit = df[(df["config"] == config) & (df["variant"] == variant) & (df["N"] == n)]
    if len(hit) != 1:
        raise SystemExit(f"summary lookup failed {config} {variant} N={n}")
    return hit.iloc[0]


def print_tables(df: pd.DataFrame) -> None:
    for n in NS:
        print(
            f"\n=== attn_only vs attn_square   N={n}   200 ep   seeds 0–9 ===",
            flush=True,
        )
        hdr = (
            f"{'tok':>4} {'cfg':<3} {'strides':<10} {'params':>7} "
            f"{'ao_rec':>7} {'ao_GE14':>8} {'ao_s':>7} "
            f"{'as_rec':>7} {'as_GE14':>8} {'as_s':>7} "
            f"{'d_rec':>6} {'d_GE14':>7} {'fail':>5}"
        )
        print(hdr, flush=True)
        for config in CONFIGS:
            ao = _cell(df, config, "attn_only", n)
            asq = _cell(df, config, "attn_square", n)
            d_rec = asq["mean_bytes_recovered"] - ao["mean_bytes_recovered"]
            d_ge = asq["mean_GE_masked14"] - ao["mean_GE_masked14"]
            fail = int(ao["n_failed"] + asq["n_failed"])
            print(
                f"{int(ao['n_tokens']):4d} {config:<3} {ao['strides']:<10} {int(ao['n_params']):7d} "
                f"{ao['mean_bytes_recovered']:7.2f} {ao['mean_GE_masked14']:8.2f} {ao['mean_attack_seconds']:7.1f} "
                f"{asq['mean_bytes_recovered']:7.2f} {asq['mean_GE_masked14']:8.2f} {asq['mean_attack_seconds']:7.1f} "
                f"{d_rec:6.2f} {d_ge:7.2f} {fail:5d}",
                flush=True,
            )
        print(
            "d_rec = attn_square − attn_only (bytes; + means squaring recovered more). "
            "d_GE14 = attn_square − attn_only (lower GE is better).",
            flush=True,
        )


def print_gaps(df: pd.DataFrame) -> None:
    print("\n=== squaring gap by token count (attn_square − attn_only) ===", flush=True)
    print(
        f"{'tok':>4} {'N':>6} {'d_rec':>7} {'d_GE14':>8} {'ao_params':>10}",
        flush=True,
    )
    for n in NS:
        rec_gaps = []
        ge_gaps = []
        for config in CONFIGS:
            ao = _cell(df, config, "attn_only", n)
            asq = _cell(df, config, "attn_square", n)
            d_rec = float(asq["mean_bytes_recovered"] - ao["mean_bytes_recovered"])
            d_ge = float(asq["mean_GE_masked14"] - ao["mean_GE_masked14"])
            rec_gaps.append(d_rec)
            ge_gaps.append(d_ge)
            print(
                f"{int(ao['n_tokens']):4d} {n:6d} {d_rec:7.2f} {d_ge:8.2f} {int(ao['n_params']):10d}",
                flush=True,
            )
        print(
            f"  N={n} Δrec across tokens: {['{:+.2f}'.format(x) for x in rec_gaps]}",
            flush=True,
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()
    if args.workers != 1:
        raise SystemExit(
            "This machine has one MPS GPU. Two workers share it and inflate "
            "wall-clock. Run sequential: --workers 1."
        )

    dev = _device()
    print(f"[screen] device={dev}  workers=1", flush=True)
    if str(dev) != "mps":
        raise SystemExit(f"refusing silent CPU/CUDA fallback; device={dev}")

    arch = confirm_tokens()
    for name in CONFIGS:
        for variant in VARIANTS:
            m = make_model(name, variant)
            x = torch.zeros(2, 1, 700)
            with torch.no_grad():
                feats = m.extract_features(x)
                logits = m(x)
            if tuple(feats.shape) != (2, 64) or tuple(logits.shape) != (2, 256):
                raise SystemExit(f"{variant} {name} shape mismatch")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    (OUT / "protocol.json").write_text(
        json.dumps(
            {
                "question": (
                    "Does (AV)⊙(AV) help more as token count rises, "
                    "and does that reverse above 23 tokens?"
                ),
                "variants": {
                    "attn_only": "make_h2(cfg, use_gate=False)",
                    "attn_square": "make_h2(cfg, use_gate=False, self_multiply=True)",
                },
                "configs": arch,
                "note_35_tokens": (
                    "Config C uses kernels (5,2,2) to reach 35 tokens. "
                    "Same kernels (7,5,5) with strides (5,2,2) yield 32 tokens."
                ),
                "epochs": EPOCHS,
                "batch_size": PAPER_BATCH_SIZE,
                "lr": PAPER_LR,
                "adam_eps": 1e-7,
                "loss": "CrossEntropy",
                "N": list(NS),
                "seeds": list(SEEDS),
                "reused_from": (
                    "results/attn_combine_screen_4tok "
                    "(config A seeds 0–2 only; A seeds 3–9 trained here)"
                ),
                "not_reused": [
                    "token_resolution_ablation / token_resolution_finegrid (gated H2)",
                    "attention_no_gate_ablation (25 ep, Config E)",
                    "attention_self_multiply_ablation (25 ep, Config E)",
                    "results_aacasca_200epoch (only seed0 N=40k exists)",
                ],
            },
            indent=2,
        )
        + "\n"
    )

    if not CACHE.exists():
        cache_per_byte_windows(RAW, CACHE)
    data = load_per_byte_windows(CACHE)

    jobs = [(c, v, n, s) for c in CONFIGS for v in VARIANTS for n in NS for s in SEEDS]
    t_all = time.time()
    n_reused = 0
    n_trained = 0
    n_failed = 0
    for i, (config, variant, n, seed) in enumerate(jobs, start=1):
        dest = dest_path(config, variant, seed, n)
        if dest.exists():
            rec = json.loads(dest.read_text())
            kind = "fail" if rec.get("failed") else ("reuse" if rec.get("reused") else "done")
            print(
                f"[screen] skip {i}/{len(jobs)} {kind} {variant} {config} seed={seed} N={n}",
                flush=True,
            )
            continue
        if config == "A":
            src = REUSE_4TOK / f"{variant}_seed{seed}_N{n}.json"
            if src.exists():
                print(
                    f"\n======== {i}/{len(jobs)} reuse 4tok {variant} seed={seed} N={n} ========",
                    flush=True,
                )
                reuse_4tok(variant, seed, n)
                n_reused += 1
                continue
            print(
                f"\n======== {i}/{len(jobs)}  train 4tok {variant} seed={seed} N={n} "
                f"(no reuse source) ========",
                flush=True,
            )
        else:
            print(
                f"\n======== {i}/{len(jobs)}  {variant} {config}  seed={seed} N={n} ========",
                flush=True,
            )
        try:
            rec = run_one(
                data["windows"],
                data["plaintext"],
                data["true_key"],
                seed,
                n,
                EPOCHS,
                OUT,
                model_fn=lambda c=config, v=variant: make_model(c, v),
                tag=f"{variant}_{config}",
            )
            rec["variant"] = variant
            rec["config"] = config
            rec["kernels"] = list(CONV_CONFIGS[config].kernels)
            rec["strides"] = list(CONV_CONFIGS[config].strides)
            rec["use_gate"] = False
            rec["self_multiply"] = variant == "attn_square"
            rec["reused"] = False
            rec["failed"] = False
            rec["protocol"] = {
                "epochs": EPOCHS,
                "batch_size": PAPER_BATCH_SIZE,
                "lr": PAPER_LR,
                "adam_eps": 1e-7,
                "loss": "CrossEntropy",
            }
            dest.write_text(json.dumps(rec, indent=2) + "\n")
            n_trained += 1
        except Exception as exc:
            n_failed += 1
            err = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            print(f"[screen] FAILED {variant} {config} seed={seed} N={n}\n{err}", flush=True)
            write_failure(config, variant, seed, n, err)

    rows = []
    for config, variant, n, seed in jobs:
        rec = json.loads(dest_path(config, variant, seed, n).read_text())
        rec["variant"] = variant
        rec["config"] = config
        rows.append(rec)
    df = write_summary(rows)
    print_tables(df)
    print_gaps(df)
    wall = time.time() - t_all
    print(
        f"\n[screen] wall-clock {wall:.1f}s ({wall / 60.0:.1f} min)  "
        f"reused={n_reused} trained={n_trained} failed={n_failed}",
        flush=True,
    )
    (OUT / "WALL_CLOCK.txt").write_text(f"{wall:.3f}\n")


if __name__ == "__main__":
    main()
