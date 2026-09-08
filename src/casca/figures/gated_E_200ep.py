#!/usr/bin/env python3
"""Temporary gated Config E (att×G) 200-epoch mean+CI preview.

Uses only N with ≥5 seeds. Bootstrap over whatever seeds exist (5 or 10).
No interpolation of missing N.
"""
from __future__ import annotations

import json
from pathlib import Path

import sys as _casca_sys
_CASCA_ROOT = Path(__file__).resolve().parents[3]
if not (_CASCA_ROOT / 'src' / 'casca').is_dir():
    _CASCA_ROOT = Path(__file__).resolve().parents[3]
_casca_sys.path.insert(0, str(_CASCA_ROOT / "src"))
from casca.paths import raw_traces_path, repo_root, window_cache_path  # noqa: E402

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "results_aacasca_200epoch" / "gated_E_preview"
SOURCES = [
    REPO / "results" / "configE_aggregate_CH_extended" / "runs",
    REPO / "results" / "epoch_trajectory_diagnostic" / "runs",
    OUT / "runs",
]
CASCA_METRICS = REPO / "results" / "results_casca_reproducibility" / "metrics.csv"
EXCLUDE_N = {2000, 12000}
N_BOOT = 1000
BOOT_SEED = 0
MIN_SEEDS = 5
GE_FLOOR = 0.05
EXPECTED_PARAMS = 99992
EXPECTED_TOKENS = 23


def bootstrap_mean_ci(
    xs: np.ndarray,
    n_resamples: int = N_BOOT,
    seed: int = BOOT_SEED,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    xs = np.asarray(xs, dtype=np.float64)
    m = float(xs.mean()) if xs.size else float("nan")
    if xs.size < 2:
        return m, m, m
    rng = np.random.default_rng(seed)
    means = rng.choice(xs, size=(n_resamples, xs.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return m, float(lo), float(hi)


def load_raw() -> pd.DataFrame:
    rows = []
    seen: set[tuple[int, int]] = set()
    for d in SOURCES:
        for p in sorted(d.glob("h2E*.json")):
            rec = json.loads(p.read_text())
            proto = rec.get("protocol") or {}
            if int(proto.get("epochs", rec.get("epochs", -1))) != 200:
                continue
            if int(proto.get("batch_size", 20000)) != 20000:
                continue
            if int(rec.get("n_tokens", -1)) != EXPECTED_TOKENS:
                continue
            if int(rec.get("n_params", -1)) != EXPECTED_PARAMS:
                continue
            ck = [c for c in rec.get("checkpoints") or [] if int(c.get("epoch", -1)) == 200]
            if ck:
                ranks = np.asarray(ck[0]["ranks"], dtype=float)
            else:
                ranks = np.asarray(rec.get("ranks") or [], dtype=float)
            if ranks.size != 16:
                continue
            key = (int(rec["N"]), int(rec["seed"]))
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "N": key[0],
                    "seed": key[1],
                    "bytes_recovered": int(np.sum(ranks == 0)),
                    "GE": float(ranks.mean()),
                    "source": str(p.relative_to(REPO)),
                }
            )
    return pd.DataFrame(rows)


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n, sub in raw.groupby("N", sort=True):
        if int(n) in EXCLUDE_N:
            continue
        if len(sub) < MIN_SEEDS:
            continue
        sub = sub.sort_values("seed")
        rec, rec_lo, rec_hi = bootstrap_mean_ci(sub["bytes_recovered"].to_numpy())
        ge, ge_lo, ge_hi = bootstrap_mean_ci(sub["GE"].to_numpy())
        rows.append(
            {
                "N": int(n),
                "n_seeds": int(len(sub)),
                "seeds": ",".join(str(s) for s in sub["seed"]),
                "mean_bytes_recovered": rec,
                "bytes_ci_lo": rec_lo,
                "bytes_ci_hi": rec_hi,
                "mean_GE": ge,
                "GE_ci_lo": ge_lo,
                "GE_ci_hi": ge_hi,
                "per_seed_recovered": sub["bytes_recovered"].astype(int).tolist(),
                "per_seed_GE": [round(float(x), 4) for x in sub["GE"]],
            }
        )
    return pd.DataFrame(rows)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_casca() -> pd.DataFrame:
    df = pd.read_csv(CASCA_METRICS)
    return df[~df["N"].isin(EXCLUDE_N)].sort_values("N").reset_index(drop=True)


def _n_axis(ax, ns: list[int]) -> None:
    ticks = sorted(set(int(n) for n in ns))
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: f"{int(v / 1000)}k"))
    ax.set_xlim(min(ticks) - 1000, max(ticks) + 2000)


def _series(ax, df: pd.DataFrame, y, lo, hi, color: str, label: str, *, log_clip: bool = False) -> None:
    yy = df[y].to_numpy(dtype=float)
    ylo = df[lo].to_numpy(dtype=float)
    yhi = df[hi].to_numpy(dtype=float)
    if log_clip:
        yy = np.clip(yy, GE_FLOOR, None)
        ylo = np.clip(ylo, GE_FLOOR, None)
        yhi = np.clip(yhi, GE_FLOOR, None)
    ax.fill_between(df["N"], ylo, yhi, color=color, alpha=0.22, lw=0)
    ax.plot(df["N"], yy, color=color, marker="o", ms=5, lw=1.6, label=label)


def plot_combined(agg: pd.DataFrame, casca: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.8), sharex=True)
    ns = sorted(set(agg["N"].tolist()) | set(casca["N"].tolist()))

    ax = axes[0]
    _series(ax, casca, "mean_bytes_recovered", "bytes_ci_lo", "bytes_ci_hi", "C0", "CA-SCA")
    _series(ax, agg, "mean_bytes_recovered", "bytes_ci_lo", "bytes_ci_hi", "C1", "AA-CASCA")
    ax.axhline(16, color="0.35", ls="--", lw=0.9, zorder=0)
    ax.set_ylim(-0.4, 16.8)
    ax.set_ylabel("Mean bytes recovered / 16")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92)

    ax = axes[1]
    _series(ax, casca, "mean_GE", "GE_ci_lo", "GE_ci_hi", "C0", "_nolegend_", log_clip=True)
    _series(ax, agg, "mean_GE", "GE_ci_lo", "GE_ci_hi", "C1", "_nolegend_", log_clip=True)
    ax.set_yscale("log")
    ax.set_ylim(GE_FLOOR, 120)
    ax.set_ylabel("Mean GE (16-byte, log)")
    ax.set_xlabel("N (traces)")
    _n_axis(ax, ns)
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    path = OUT / "plot2_mean_ci.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    named = OUT / "aa-casca-plot2_mean_ci.png"
    fig.savefig(named, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}", flush=True)
    print(f"wrote {named}", flush=True)


def main() -> None:
    _style()
    OUT.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    dropped = raw.groupby("N").size()
    dropped = dropped[dropped < MIN_SEEDS]
    if len(dropped):
        print(f"dropped N with <{MIN_SEEDS} seeds: {dropped.to_dict()}", flush=True)
    agg = aggregate(raw)
    casca = load_casca()
    raw.to_csv(OUT / "per_seed.csv", index=False)
    cols = [
        "N",
        "n_seeds",
        "seeds",
        "mean_bytes_recovered",
        "bytes_ci_lo",
        "bytes_ci_hi",
        "mean_GE",
        "GE_ci_lo",
        "GE_ci_hi",
    ]
    agg[cols].to_csv(OUT / "summary_table.csv", index=False)
    plot_combined(agg, casca)
    hit = agg[agg["mean_GE"] < 1.0]
    dN = int(hit.iloc[0]["N"]) if len(hit) else None
    print(agg[cols].to_string(index=False), flush=True)
    print(f"gated E 200-ep disclosure N (on measured grid) = {dN}", flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
