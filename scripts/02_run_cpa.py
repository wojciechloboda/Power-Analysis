#!/usr/bin/env python3
"""Step 3: CPA GE/SR via repeated random subsampling (10 trials, matching three_way_grid)."""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from casca.data.windows import ALL_BYTES, MASKED_BYTES, UNMASKED_BYTES  # noqa: E402
from casca.data.ascad import N_PROFILING, cache_per_byte_windows, load_per_byte_windows  # noqa: E402
from casca.metrics import guessing_entropy, success_rate  # noqa: E402
from casca.attacks.cpa import attack_byte  # noqa: E402

RAW = raw_traces_path()
CACHE = window_cache_path()
OUT = Path(__file__).resolve().parents[1] / "results" / "cpa_ge_sr_evaluation"
GATE = Path(__file__).resolve().parents[1] / "results" / "cpa_unmasked_validation/validation.json"

N_GRID = (5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000)
N_TRIALS = 10
# Reproducible subset draws; not reused from any other experiment's RNG.
RNG_BASE = 20260823


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def rng_seed(n: int, trial: int) -> int:
    return int(RNG_BASE + 1000 * n + trial)


def subset_indices(n: int, trial: int, pool: int) -> np.ndarray:
    if n > pool:
        raise ValueError(f"N={n} exceeds pool {pool}")
    rng = np.random.default_rng(rng_seed(n, trial))
    return rng.choice(pool, size=n, replace=False)


def run_trial(windows, plaintext, true_key, n: int, trial: int) -> dict:
    t0 = time.time()
    idx = subset_indices(n, trial, N_PROFILING)
    idx.sort()
    ranks = []
    scores = []
    true_scores = []
    margins = []
    peak_samples = []
    byte_seconds = []
    for b in ALL_BYTES:
        tb = time.time()
        result = attack_byte(windows[idx, b], plaintext[idx, b], int(true_key[b]))
        byte_seconds.append(time.time() - tb)
        ranks.append(result["rank"])
        scores.append(result["scores"])
        true_scores.append(result["true_score"])
        margins.append(result["margin"])
        peak_samples.append(result["peak_sample"])
        print(
            f"  [cpa] N={n} trial={trial} byte={b:2d}  "
            f"{'unmasked' if b in UNMASKED_BYTES else 'masked  '}  "
            f"rank={result['rank']:3d}  |r|={result['true_score']:.4f}  "
            f"margin={result['margin']:+.4f}  {byte_seconds[-1]:.2f}s",
            flush=True,
        )
    ranks_a = np.asarray(ranks, dtype=int)
    masked = ranks_a[list(MASKED_BYTES)]
    elapsed = time.time() - t0
    rec = {
        "tag": "cpa_ge_sr",
        "method": "CPA",
        "N": n,
        "seed": trial,
        "trial": trial,
        "rng_seed": rng_seed(n, trial),
        "pool": N_PROFILING,
        "n_traces_used": n,
        "indices_hash": int(idx.astype(np.int64).sum()),
        "true_key": true_key.tolist(),
        "ranks": ranks_a.tolist(),
        "true_scores": [float(x) for x in true_scores],
        "margins": [float(x) for x in margins],
        "peak_samples": [int(x) for x in peak_samples],
        "byte_seconds": [float(x) for x in byte_seconds],
        "bytes_recovered": int(np.sum(ranks_a == 0)),
        "masked_recovered": int(np.sum(masked == 0)),
        "GE_masked14": guessing_entropy(masked),
        "SR1": float(success_rate(ranks_a, 1)),
        "SR5": float(success_rate(ranks_a, 5)),
        "SR1_fullkey": float(np.all(ranks_a == 0)),
        "SR5_fullkey": float(np.all(ranks_a < 5)),
        "SR1_masked14": float(success_rate(masked, 1)),
        "SR5_masked14": float(success_rate(masked, 5)),
        "train_seconds": elapsed,
        "elapsed_seconds": elapsed,
        "score_file": str(OUT / "runs" / f"cpa_N{n}_trial{trial}.npz"),
    }
    return rec, np.stack(scores, axis=0)


def out_path(n: int, trial: int) -> Path:
    return OUT / "runs" / f"cpa_N{n}_trial{trial}.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timing-only", action="store_true")
    parser.add_argument("--skip-gate", action="store_true")
    args = parser.parse_args()

    if not args.skip_gate:
        if not GATE.exists():
            raise SystemExit(
                f"missing unmasked validation {GATE}. "
                "Run scripts/01_validate_unmasked.py first."
            )
        gate = json.loads(GATE.read_text())
        if not gate.get("gate_passed"):
            raise SystemExit("unmasked validation gate failed — refusing masked GE/SR grid")

    if not CACHE.exists():
        cache_per_byte_windows(RAW, CACHE)
    data = load_per_byte_windows(CACHE)
    windows = data["windows"]
    plaintext = data["plaintext"]
    true_key = data["true_key"]
    if windows.shape[0] < N_PROFILING:
        raise SystemExit(f"window cache has {windows.shape[0]} traces, need {N_PROFILING}")

    jobs = [(40000, 0)] if args.timing_only else [(n, t) for n in N_GRID for t in range(N_TRIALS)]
    pending = [(n, t) for n, t in jobs if not out_path(n, t).exists()]
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    print("=== CPA GE/SR (random subsets, 10 trials matching three_way_grid) ===")
    print(f"  N={list(N_GRID)}  trials={N_TRIALS}  pool=profiling 0:{N_PROFILING}")
    print(f"  {len(pending)} pending of {len(jobs)}", flush=True)

    t_all = time.time()
    for i, (n, trial) in enumerate(pending, start=1):
        print(f"\n======== cpa  {i}/{len(pending)}  N={n} trial={trial} ========", flush=True)
        rec, score_stack = run_trial(windows, plaintext, true_key, n, trial)
        path = out_path(n, trial)
        np.savez_compressed(path.with_suffix(".npz"), scores=score_stack, ranks=np.asarray(rec["ranks"]))
        path.write_text(json.dumps(_jsonable(rec), indent=2) + "\n")
        print(
            f"[cpa] N={n} trial={trial}  GE14={rec['GE_masked14']:.2f}  "
            f"rec={rec['bytes_recovered']}/16  masked={rec['masked_recovered']}/14  "
            f"SR1_full={rec['SR1_fullkey']:.0f}  {rec['elapsed_seconds']:.2f}s",
            flush=True,
        )
        if args.timing_only:
            proj = rec["elapsed_seconds"] * len(N_GRID) * N_TRIALS
            print(
                f"\n[timing] N=40000 trial 0 = {rec['elapsed_seconds']:.2f}s. "
                f"Upper-bound full grid ({len(N_GRID)} N × {N_TRIALS} trials) "
                f"≈ {proj:.0f}s ({proj / 60.0:.1f} min) if every cell were this expensive.",
                flush=True,
            )
            (OUT / "timing_projection.md").write_text(
                f"# CPA timing projection\n\n"
                f"Single trial at N=40000: **{rec['elapsed_seconds']:.2f}s** "
                f"(16 bytes, vectorized Pearson).\n\n"
                f"Grid: {len(N_GRID)} N × {N_TRIALS} trials = {len(N_GRID) * N_TRIALS} runs.\n"
                f"Upper bound (all cells as expensive as N=40000): "
                f"**{proj:.0f}s ({proj / 60.0:.1f} min)**.\n"
                f"Smaller N will be cheaper. CPA is deterministic given the subset; "
                f"trials differ only by which traces are drawn.\n"
            )
    print(f"\n[cpa] finished {len(pending)} combinations in {time.time() - t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
