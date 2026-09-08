"""Guessing Entropy and Success Rate for a NON-PROFILED attack.

Standaert, Malkin & Yung (CHES 2006 / "A Unified Framework for the Analysis
of Side-Channel Key Recovery Attacks"):

    GE_N  = mean rank of the true key, as a function of the number of traces N
    SR_o(N) = Pr[ true key is among the o best-ranked candidates ]

Profiled-setting convention (NOT used here)
------------------------------------------
Train one model, then draw many independent *attack-set* subsets of size N
and average the rank. The model is a fixed object; randomness is in the
attack traces.

Non-profiled-setting convention (USED here)
-------------------------------------------
CA-SCA has no separate profiling device. Algorithm 3 trains the CNN on the
same traces that are later clustered. One independent trial is therefore a
full attack:

    fresh RNG seed  ->  train CNN  ->  extract features  ->  256-way CH ranking

GE_N is the mean of those ranks across independent full runs, not across
multiple attack-set draws from a single trained model.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def guessing_entropy(ranks: Sequence[int]) -> float:
    return float(np.mean(np.asarray(ranks, dtype=np.float64)))


def success_rate(ranks: Sequence[int], order: int) -> float:
    """SR_o: fraction of trials whose 0-indexed rank is < order.

    SR_1  -> rank == 0 (true key is the unique-or-tied top candidate)
    SR_5  -> rank in {0,1,2,3,4}
    SR_10 -> rank < 10
    """
    r = np.asarray(ranks, dtype=np.int64)
    return float(np.mean(r < int(order)))


def summarize_ranks(ranks_by_n: dict[int, list[int]], orders: tuple[int, ...] = (1, 5, 10)) -> list[dict]:
    rows = []
    for n in sorted(ranks_by_n):
        ranks = ranks_by_n[n]
        row = {
            "N": n,
            "n_trials": len(ranks),
            "GE": guessing_entropy(ranks),
            "ranks": list(ranks),
        }
        for o in orders:
            row[f"SR_{o}"] = success_rate(ranks, o)
        rows.append(row)
    return rows


N_BOOT = 1000
BOOT_SEED = 0


def bootstrap_mean_ci(
    xs,
    n_resamples: int = N_BOOT,
    seed: int = BOOT_SEED,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Mean and percentile bootstrap CI. Copied from scripts/45_plot_casca_reproducibility.py."""
    xs = np.asarray(xs, dtype=np.float64)
    m = float(xs.mean()) if xs.size else float("nan")
    if xs.size < 2:
        return m, m, m
    rng = np.random.default_rng(seed)
    means = rng.choice(xs, size=(n_resamples, xs.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return m, float(lo), float(hi)
