#!/usr/bin/env python3
"""CA-SCA vs Config E 25 ep vs Config E 200 ep: GE/bytes and attack time.

Attack time = train_seconds + one CH evaluation (not extra checkpoint CH).
Time figure uses quiet-GPU N only (5/10/20/30/40k for CA-SCA and Config E;
10/20/30/40k for methods whose 5k wall-clock is contended).
Does not overwrite training_budget_M10 or results_aacasca_200epoch stores.
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
OUT = REPO / "results" / "results_fourway_N"
CASCA_RUNS = REPO / "results" / "clean_reimpl_full16" / "runs"
E25_RUNS = OUT / "gated_E_25ep" / "runs"
E200_SOURCES = [
    REPO / "results" / "configE_aggregate_CH_extended" / "runs",
    REPO / "results" / "epoch_trajectory_diagnostic" / "runs",
    REPO / "results" / "results_aacasca_200epoch" / "gated_E_preview" / "runs",
]

TARGET_N = (5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000)
# Quiet-GPU N only. 15/25/35k were gap-filled under MPS contention —
# those wall-clocks are not used. Time plots interpolate them from the
# neighbouring quiet N (same markers as the measured points).
TIME_N = (10000, 20000, 30000, 40000)
TIME_N_EXTRA = {
    "CA-SCA": (5000,),
    "Config E 25": (5000,),
    "Config E 200": (5000,),
}
ESTIMATE_N = (15000, 25000, 35000)
ESTIMATE_BRACKETS = {
    15000: (10000, 20000),
    25000: (20000, 30000),
    35000: (30000, 40000),
}
CASCA_SEEDS = tuple(range(10))
E_SEEDS = tuple(range(5))
N_BOOT = 1000
BOOT_SEED = 0
GE_FLOOR = 0.05
E_PARAMS = 99992
E_TOKENS = 23
# 2-worker MPS contention during the 200-ep fill; keep in GE/bytes, drop from time.
TIME_STALLS = frozenset({("Config E 200", 0, 40000), ("Config E 200", 1, 40000)})
STEPS_25 = {
    5000: 100,
    10000: 175,
    15000: 275,
    20000: 350,
    25000: 450,
    30000: 525,
    35000: 625,
    40000: 700,
}

STYLE = {
    "CA-SCA": {"color": "C0", "marker": "o", "label": "CA-SCA"},
    "Config E 25": {"color": "C2", "marker": "v", "label": "AA-CASCA (25 ep)"},
    "Config E 200": {"color": "C1", "marker": "s", "label": "AA-CASCA (200 ep)"},
}
ORDER = ("CA-SCA", "Config E 25", "Config E 200")


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


def epoch200_ckpt(rec: dict) -> dict | None:
    for c in rec.get("checkpoints") or []:
        if int(c.get("epoch", -1)) == 200:
            return c
    return None


def attack_seconds(rec: dict) -> float:
    train = rec.get("train_seconds")
    ck = epoch200_ckpt(rec)
    if train is None and ck is not None:
        train = ck.get("cumulative_train_seconds")
    if train is None:
        raise SystemExit(f"missing train_seconds in N={rec.get('N')} seed={rec.get('seed')}")
    ch = rec.get("key_recovery_seconds")
    if ch is None and ck is not None:
        ch = ck.get("checkpoint_ch_seconds")
    if ch is None and rec.get("elapsed_seconds") is not None:
        ch = max(float(rec["elapsed_seconds"]) - float(train), 0.0)
    if ch is None:
        ch = 0.0
    return float(train) + float(ch)


def row(method: str, n: int, seed: int, ranks, rec: dict, source: str) -> dict:
    ranks = np.asarray(ranks, dtype=float)
    if ranks.size != 16:
        raise SystemExit(f"{method} N={n} seed={seed}: ranks len={ranks.size}")
    return {
        "method": method,
        "N": int(n),
        "seed": int(seed),
        "bytes_recovered": int(np.sum(ranks == 0)),
        "GE": float(ranks.mean()),
        "attack_seconds": attack_seconds(rec),
        "train_seconds": float(rec.get("train_seconds") or epoch200_ckpt(rec)["cumulative_train_seconds"]),
        "source": source,
    }


def load_casca() -> list[dict]:
    rows = []
    for n in TARGET_N:
        for s in CASCA_SEEDS:
            p = CASCA_RUNS / f"seed{s}_N{n}.json"
            rec = json.loads(p.read_text())
            rows.append(row("CA-SCA", n, s, rec["ranks"], rec, str(p.relative_to(REPO))))
    return rows


def load_e25() -> list[dict]:
    rows = []
    missing = []
    for n in TARGET_N:
        for s in E_SEEDS:
            p = E25_RUNS / f"gated_E_25ep_seed{s}_N{n}.json"
            if not p.exists():
                missing.append((n, s))
                continue
            rec = json.loads(p.read_text())
            if rec.get("use_gate") is not True or int(rec.get("n_params", -1)) != E_PARAMS:
                raise SystemExit(f"{p.name} is not gated Config E")
            if int(rec.get("n_steps", -1)) != STEPS_25[n]:
                raise SystemExit(f"{p.name} n_steps={rec.get('n_steps')} want {STEPS_25[n]}")
            rows.append(row("Config E 25", n, s, rec["ranks"], rec, str(p.relative_to(REPO))))
    if missing:
        raise SystemExit("Config E 25-ep incomplete: " + str(missing))
    return rows


def load_e200() -> list[dict]:
    found: dict[tuple[int, int], dict] = {}
    for d in E200_SOURCES:
        for p in sorted(d.glob("h2E*.json")):
            rec = json.loads(p.read_text())
            proto = rec.get("protocol") or {}
            if int(proto.get("epochs", rec.get("epochs", -1))) != 200:
                continue
            if int(proto.get("batch_size", 20000)) != 20000:
                continue
            if int(rec.get("n_tokens", -1)) != E_TOKENS:
                continue
            if int(rec.get("n_params", -1)) != E_PARAMS:
                continue
            n = int(rec["N"])
            seed = int(rec["seed"])
            if n not in TARGET_N or seed not in E_SEEDS:
                continue
            ck = epoch200_ckpt(rec)
            ranks = ck["ranks"] if ck else rec.get("ranks")
            key = (n, seed)
            if key in found:
                continue
            found[key] = row(
                "Config E 200", n, seed, ranks, rec, str(p.relative_to(REPO))
            )
    missing = [(n, s) for n in TARGET_N for s in E_SEEDS if (n, s) not in found]
    if missing:
        raise SystemExit("Config E 200-ep incomplete: " + str(missing))
    return [found[k] for k in sorted(found)]


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, n), sub in raw.groupby(["method", "N"], sort=True):
        rec, rec_lo, rec_hi = bootstrap_mean_ci(sub["bytes_recovered"].to_numpy())
        ge, ge_lo, ge_hi = bootstrap_mean_ci(sub["GE"].to_numpy())
        tsub = sub[~sub.apply(lambda r: (r["method"], int(r["seed"]), int(r["N"])) in TIME_STALLS, axis=1)]
        if len(tsub) < 2:
            tsub = sub
        t, t_lo, t_hi = bootstrap_mean_ci(tsub["attack_seconds"].to_numpy())
        rows.append(
            {
                "method": method,
                "N": int(n),
                "n_seeds": int(len(sub)),
                "mean_bytes_recovered": rec,
                "bytes_ci_lo": rec_lo,
                "bytes_ci_hi": rec_hi,
                "mean_GE": ge,
                "GE_ci_lo": ge_lo,
                "GE_ci_hi": ge_hi,
                "mean_attack_seconds": t,
                "time_ci_lo": t_lo,
                "time_ci_hi": t_hi,
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


def _n_axis(ax, ns) -> None:
    ticks = sorted(set(int(n) for n in ns))
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: f"{int(v / 1000)}k"))
    ax.set_xlim(min(ticks) - 1000, max(ticks) + 2000)


def _series(
    ax,
    df: pd.DataFrame,
    y,
    lo,
    hi,
    method: str,
    *,
    log_clip: bool = False,
    legend: bool = True,
) -> None:
    st = STYLE[method]
    sub = df[df["method"] == method].sort_values("N")
    yy = sub[y].to_numpy(dtype=float)
    ylo = sub[lo].to_numpy(dtype=float)
    yhi = sub[hi].to_numpy(dtype=float)
    if log_clip:
        yy = np.clip(yy, GE_FLOOR, None)
        ylo = np.clip(ylo, GE_FLOOR, None)
        yhi = np.clip(yhi, GE_FLOOR, None)
    ax.fill_between(sub["N"], ylo, yhi, color=st["color"], alpha=0.18, lw=0)
    ax.plot(
        sub["N"],
        yy,
        color=st["color"],
        marker=st["marker"],
        ms=5,
        lw=1.6,
        label=st["label"] if legend else "_nolegend_",
    )


def plot_ge_bytes(agg: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.8), sharex=True)

    ax = axes[0]
    for m in ORDER:
        _series(ax, agg, "mean_bytes_recovered", "bytes_ci_lo", "bytes_ci_hi", m)
    ax.axhline(16, color="0.35", ls="--", lw=0.9, zorder=0)
    ax.set_ylim(-0.4, 16.8)
    ax.set_ylabel("Mean bytes recovered / 16")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92)

    ax = axes[1]
    for m in ORDER:
        _series(ax, agg, "mean_GE", "GE_ci_lo", "GE_ci_hi", m, log_clip=True, legend=False)
    ax.set_yscale("log")
    ax.set_ylim(GE_FLOOR, 200)
    ax.set_ylabel("Mean GE (16-byte, log)")
    ax.set_xlabel("N (traces)")
    _n_axis(ax, TARGET_N)
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    path = OUT / "plot_e25_e200_casca_ge_bytes.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}", flush=True)
    print(f"wrote {path.with_suffix('.pdf')}", flush=True)


def _time_n(method: str) -> tuple[int, ...]:
    extra = TIME_N_EXTRA.get(method, ())
    return tuple(n for n in TARGET_N if n in TIME_N or n in extra)


def _lerp(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    if x1 == x0:
        return float(y0)
    return float(y0 + (x - x0) * (y1 - y0) / (x1 - x0))


def _estimated_time(agg: pd.DataFrame, method: str, n: int) -> tuple[float, float, float]:
    n0, n1 = ESTIMATE_BRACKETS[n]
    a = agg[(agg["method"] == method) & (agg["N"] == n0)]
    b = agg[(agg["method"] == method) & (agg["N"] == n1)]
    if len(a) != 1 or len(b) != 1:
        raise SystemExit(f"cannot interpolate {method} N={n} from {n0}/{n1}")
    a, b = a.iloc[0], b.iloc[0]
    return (
        _lerp(n, n0, n1, a["mean_attack_seconds"], b["mean_attack_seconds"]),
        _lerp(n, n0, n1, a["time_ci_lo"], b["time_ci_lo"]),
        _lerp(n, n0, n1, a["time_ci_hi"], b["time_ci_hi"]),
    )


def plot_time(agg: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    plotted: list[int] = []
    for m in ORDER:
        allow = set(_time_n(m))
        sub = agg[(agg["method"] == m) & (agg["N"].isin(allow))]
        extra = []
        for n in ESTIMATE_N:
            y, lo, hi = _estimated_time(agg, m, n)
            extra.append(
                {
                    "method": m,
                    "N": n,
                    "mean_attack_seconds": y,
                    "time_ci_lo": lo,
                    "time_ci_hi": hi,
                }
            )
            print(f"  interpolated {STYLE[m]['label']} N={n // 1000}k  {y:.1f}s", flush=True)
        sub = pd.concat([sub, pd.DataFrame(extra)], ignore_index=True)
        _series(ax, sub, "mean_attack_seconds", "time_ci_lo", "time_ci_hi", m)
        plotted.extend(int(n) for n in sub["N"])
    ax.set_ylabel("Mean attack time (s)")
    ax.set_xlabel("N (traces)")
    _n_axis(ax, plotted)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", frameon=True, framealpha=0.92)
    ax.set_ylim(0, None)
    fig.tight_layout()
    path = OUT / "plot_e25_e200_casca_time.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}", flush=True)
    print(f"wrote {path.with_suffix('.pdf')}", flush=True)


def disclosure_n(agg: pd.DataFrame, method: str) -> int | None:
    hit = agg[(agg["method"] == method) & (agg["mean_GE"] < 1.0)].sort_values("N")
    return int(hit.iloc[0]["N"]) if len(hit) else None


def main() -> None:
    _style()
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(load_casca() + load_e25() + load_e200())
    agg = aggregate(raw)
    raw.to_csv(OUT / "e25_e200_casca_per_seed.csv", index=False)
    cols = [
        "method",
        "N",
        "n_seeds",
        "mean_bytes_recovered",
        "bytes_ci_lo",
        "bytes_ci_hi",
        "mean_GE",
        "GE_ci_lo",
        "GE_ci_hi",
        "mean_attack_seconds",
        "time_ci_lo",
        "time_ci_hi",
    ]
    agg[cols].to_csv(OUT / "e25_e200_casca_summary.csv", index=False)
    plot_ge_bytes(agg)
    plot_time(agg)
    print(agg[cols].to_string(index=False), flush=True)
    for m in ORDER:
        sub = agg[agg["method"] == m]
        tsub = sub[sub["N"].isin(_time_n(m))].sort_values("N")
        t0 = float(tsub.iloc[0]["mean_attack_seconds"])
        t1 = float(tsub.iloc[-1]["mean_attack_seconds"])
        n0, n1 = int(tsub.iloc[0]["N"]), int(tsub.iloc[-1]["N"])
        print(
            f"  {m} disclosure N = {disclosure_n(agg, m)}  "
            f"time {n0 // 1000}k/{n1 // 1000}k = {t0:.1f}s / {t1:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
