#!/usr/bin/env python3
"""Thesis figures: reproducibility of CA-SCA on the 10-seed 5k–40k sweep.

Pure plotting. Does not train. Halts if any required (N, seed) cell is missing.
"""
from __future__ import annotations

import json
import sys
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
SRC = REPO / "results" / "clean_reimpl_full16" / "runs"
OUT = REPO / "results" / "results_casca_reproducibility"

GRID_N = (5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000)
PAPER_N = 12000
# Grid plus the paper's claimed recovery point (needed for the summary table
# and as a measured knot on both figures).
PLOT_N = (5000, 10000, 12000, 15000, 20000, 25000, 30000, 35000, 40000)
SEEDS = tuple(range(10))
N_BOOT = 1000
BOOT_SEED = 0
PROTOCOL = "alg3_concat_masked14"


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


def load_cell(n: int, seed: int) -> dict | None:
    p = SRC / f"seed{seed}_N{n}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def completeness() -> tuple[pd.DataFrame, bool, int, int]:
    rows = []
    complete = True
    n_ch = 0
    n_cells = 0
    for n in PLOT_N:
        present: list[int] = []
        missing: list[int] = []
        bad: list[str] = []
        ch_ok = 0
        for seed in SEEDS:
            rec = load_cell(n, seed)
            n_cells += 1
            if rec is None:
                missing.append(seed)
                complete = False
                continue
            ranks = rec.get("ranks")
            proto = rec.get("protocol")
            if proto != PROTOCOL:
                bad.append(f"seed{seed}: protocol={proto!r}")
                missing.append(seed)
                complete = False
                continue
            if not isinstance(ranks, list) or len(ranks) != 16:
                bad.append(f"seed{seed}: ranks len={0 if ranks is None else len(ranks)}")
                missing.append(seed)
                complete = False
                continue
            if int(rec.get("seed", -1)) != seed or int(rec.get("N", -1)) != n:
                bad.append(f"seed{seed}: field mismatch")
                missing.append(seed)
                complete = False
                continue
            present.append(seed)
            npz = SRC / f"seed{seed}_N{n}.npz"
            ch_path = rec.get("ch_vector_file")
            if npz.exists() or (ch_path and Path(ch_path).exists()):
                ch_ok += 1
                n_ch += 1
        rows.append(
            {
                "N": n,
                "n_present": len(present),
                "seeds_present": ",".join(str(s) for s in present) if present else "—",
                "seeds_missing": ",".join(str(s) for s in missing) if missing else "—",
                "schema_issues": "; ".join(bad) if bad else "—",
                "ch_vectors": f"{ch_ok}/{len(present)}" if present else "0/0",
                "required_grid": n in GRID_N or n == PAPER_N,
            }
        )
    df = pd.DataFrame(rows)
    return df, complete, n_ch, n_cells


def load_raw() -> pd.DataFrame:
    rows = []
    for n in PLOT_N:
        for seed in SEEDS:
            rec = load_cell(n, seed)
            if rec is None:
                raise SystemExit(f"missing {SRC / f'seed{seed}_N{n}.json'}")
            ranks = np.asarray(rec["ranks"], dtype=float)
            rows.append(
                {
                    "N": n,
                    "seed": seed,
                    "bytes_recovered": int(np.sum(ranks == 0)),
                    "GE": float(ranks.mean()),
                    "full_key": bool(np.all(ranks == 0)),
                    "source": f"results/clean_reimpl_full16/runs/seed{seed}_N{n}.json",
                }
            )
    return pd.DataFrame(rows)


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n in PLOT_N:
        sub = raw[raw.N == n]
        rec, rec_lo, rec_hi = bootstrap_mean_ci(sub["bytes_recovered"].to_numpy())
        ge, ge_lo, ge_hi = bootstrap_mean_ci(sub["GE"].to_numpy())
        rows.append(
            {
                "N": n,
                "mean_bytes_recovered": rec,
                "bytes_ci_lo": rec_lo,
                "bytes_ci_hi": rec_hi,
                "mean_GE": ge,
                "GE_ci_lo": ge_lo,
                "GE_ci_hi": ge_hi,
                "per_seed_recovered": sub.sort_values("seed")["bytes_recovered"].astype(int).tolist(),
                "per_seed_GE": [round(float(x), 4) for x in sub.sort_values("seed")["GE"]],
            }
        )
    return pd.DataFrame(rows)


def disclosure_n(agg: pd.DataFrame, ns: tuple[int, ...]) -> int | None:
    sub = agg[agg.N.isin(ns)].sort_values("N")
    hit = sub[sub["mean_GE"] < 1.0]
    if hit.empty:
        return None
    return int(hit.iloc[0]["N"])


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


def _n_formatter(ax) -> None:
    ax.set_xticks(list(PLOT_N))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: f"{int(v/1000)}k"))
    ax.set_xlim(4000, 42000)


def _paper_vline(ax) -> None:
    ax.axvline(PAPER_N, color="0.25", ls="--", lw=1.0, zorder=5)
    ax.annotate(
        "paper's reported\nfull-key recovery",
        xy=(PAPER_N, 0.86),
        xycoords=("data", "axes fraction"),
        xytext=(-5, 0),
        textcoords="offset points",
        fontsize=7,
        color="0.2",
        va="top",
        ha="right",
    )


def plot1(raw: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    colors = plt.cm.tab10(np.linspace(0, 1, 10, endpoint=False))
    for seed in SEEDS:
        sub = raw[raw.seed == seed].sort_values("N")
        ax.plot(
            sub["N"],
            sub["bytes_recovered"],
            color=colors[seed],
            lw=1.15,
            alpha=0.7,
            marker="o",
            ms=3.4,
            label=f"seed {seed}",
        )
    ax.axhline(16, color="0.35", ls="--", lw=0.9, zorder=0)
    ax.set_ylim(-0.4, 16.8)
    _paper_vline(ax)
    ax.set_xlabel("N (traces)")
    ax.set_ylabel("Bytes recovered / 16")
    _n_formatter(ax)
    ax.grid(True, alpha=0.3)
    ax.legend(
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        frameon=False,
        handlelength=1.6,
        columnspacing=1.0,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    path = OUT / "plot1_individual_seeds.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}", flush=True)


def plot2(agg: pd.DataFrame) -> bool:
    # Mean GE spans ~82 → 0.16; on a linear axis the y=1 disclosure line
    # sits on the baseline. Log y-scale keeps that threshold readable.
    # CI lower bounds of 0 are clipped to the axis floor for the band only.
    ge_floor = 0.05
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.8), sharex=True)

    ax = axes[0]
    ax.fill_between(agg["N"], agg["bytes_ci_lo"], agg["bytes_ci_hi"], color="C0", alpha=0.22, lw=0)
    ax.plot(agg["N"], agg["mean_bytes_recovered"], color="C0", marker="o", ms=5, lw=1.6)
    ax.axhline(16, color="0.35", ls="--", lw=0.9, zorder=0)
    _paper_vline(ax)
    ax.set_ylim(-0.4, 16.8)
    ax.set_ylabel("Mean bytes recovered / 16")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ge_lo = np.clip(agg["GE_ci_lo"].to_numpy(dtype=float), ge_floor, None)
    ax.fill_between(agg["N"], ge_lo, agg["GE_ci_hi"], color="C0", alpha=0.22, lw=0)
    ax.plot(agg["N"], agg["mean_GE"], color="C0", marker="o", ms=5, lw=1.6)
    ax.axvline(PAPER_N, color="0.25", ls="--", lw=1.0, zorder=5)
    ax.set_yscale("log")
    ax.set_ylim(ge_floor, 120)
    ax.set_ylabel("Mean GE (16-byte, log)")
    ax.set_xlabel("N (traces)")
    _n_formatter(ax)
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    path = OUT / "plot2_mean_ci.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}", flush=True)
    return True


def write_table(agg: pd.DataFrame, dN: int | None) -> pd.DataFrame:
    row12 = agg[agg.N == PAPER_N].iloc[0]
    rec_lo, rec_hi = float(row12.bytes_ci_lo), float(row12.bytes_ci_hi)
    ge_lo, ge_hi = float(row12.GE_ci_lo), float(row12.GE_ci_hi)
    includes_full = (rec_lo <= 16.0 <= rec_hi) and (rec_lo > 15.0)
    # "include full recovery (16 bytes / GE near 0)" — CI must reach 16 and GE near 0.
    ge_near0 = ge_hi < 1.0
    if rec_hi < 16.0 and not ge_near0:
        verdict = (
            f"The 95% CI at N=12000 clearly excludes full-key recovery: "
            f"recovered {row12.mean_bytes_recovered:.2f} [{rec_lo:.2f}, {rec_hi:.2f}] / 16, "
            f"GE {row12.mean_GE:.2f} [{ge_lo:.2f}, {ge_hi:.2f}] (CI does not reach 16 bytes or GE < 1)."
        )
    elif includes_full and ge_near0:
        verdict = (
            "The 95% CI at N=12000 includes full-key recovery "
            "(recovered CI reaches 16 and GE CI sits below 1)."
        )
    else:
        verdict = (
            f"The 95% CI at N=12000 does not support the paper's full-key claim: "
            f"recovered {row12.mean_bytes_recovered:.2f} [{rec_lo:.2f}, {rec_hi:.2f}] / 16, "
            f"GE {row12.mean_GE:.2f} [{ge_lo:.2f}, {ge_hi:.2f}]."
        )
    table = pd.DataFrame(
        [
            {
                "paper_claimed_N_full_key": PAPER_N,
                "thesis_trace_count_to_disclosure_GE_lt_1": dN if dN is not None else "",
                "mean_bytes_recovered_at_12000": float(row12.mean_bytes_recovered),
                "bytes_ci_lo_12000": rec_lo,
                "bytes_ci_hi_12000": rec_hi,
                "mean_GE_at_12000": float(row12.mean_GE),
                "GE_ci_lo_12000": ge_lo,
                "GE_ci_hi_12000": ge_hi,
                "verdict": verdict,
            }
        ]
    )
    table.to_csv(OUT / "summary_table.csv", index=False)
    return table


def write_note(
    comp: pd.DataFrame,
    agg: pd.DataFrame,
    dN: int | None,
    n_ch: int,
    n_cells: int,
    table: pd.DataFrame,
    used_log: bool,
) -> None:
    row12 = agg[agg.N == PAPER_N].iloc[0]
    lines = [
        "# CA-SCA reproducibility figures",
        "",
        "No retraining. Source: `results/clean_reimpl_full16/runs/seed{s}_N{n}.json` "
        "(protocol `alg3_concat_masked14`, Table-2 CNN, 200 epochs, concat-14, multi-bit CH). "
        "Seeds **0–9**.",
        "",
        "## Step 0 completeness",
        "",
        "| N | seeds present | seeds missing | CH vectors |",
        "|---:|---|---|---|",
    ]
    for _, r in comp.iterrows():
        lines.append(
            f"| {int(r.N)} | {r.seeds_present} | {r.seeds_missing} | {r.ch_vectors} |"
        )
    lines += [
        "",
        f"Required 8-point grid (5k–40k step 5k): **complete, 80/80**. "
        f"Paper point N=12000: **complete, 10/10**. "
        f"Full 256-hypothesis CH vectors (sibling `.npz`): **{n_ch}/{n_cells}** cells. "
        "GE uses per-byte rank only (mean of 16 ranks in 0–255); CH vectors are not required.",
        "",
        "## Metrics",
        "",
        "Bytes recovered = count of rank-0 bytes / 16. "
        "GE(N) = mean rank across all 16 bytes. "
        "95% CIs are percentile bootstrap, 1000 resamples over the 10 seeds.",
        "",
        f"- Trace-count-to-key-disclosure (smallest on-grid N with mean GE < 1): "
        f"**{dN if dN is not None else 'not reached ≤ 40k'}**.",
        f"- At the paper's claimed N=12000: mean recovered "
        f"**{row12.mean_bytes_recovered:.2f}** "
        f"[{row12.bytes_ci_lo:.2f}, {row12.bytes_ci_hi:.2f}] / 16; "
        f"mean GE **{row12.mean_GE:.2f}** "
        f"[{row12.GE_ci_lo:.2f}, {row12.GE_ci_hi:.2f}].",
        "",
        f"**N=12000 verdict:** {table.iloc[0]['verdict']}",
        "",
        "## Figures",
        "",
        "- `plot1_individual_seeds.pdf` — 10 seed trajectories; N=12000 and y=16 marked.",
        "- `plot2_mean_ci.pdf` — mean + 95% CI for recovered bytes and GE. "
        + (
            "GE panel uses a log y-scale."
            if used_log
            else "GE panel is linear (log scale was considered; zeros at large N and the y=1 threshold are clearer on a linear axis)."
        ),
        "- N=12000 is a measured 10-seed point (same store), not interpolated, so the "
        "paper marker sits on real data.",
        "",
        "## Files",
        "",
        "- `summary_table.csv`",
        "- `metrics.csv` — per-N means and CIs",
        "- `per_seed.csv` — per-(N, seed) recovered bytes and GE",
        "",
    ]
    (OUT / "NOTE.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    _style()
    OUT.mkdir(parents=True, exist_ok=True)
    comp, ok, n_ch, n_cells = completeness()
    print("=== Step 0 completeness ===", flush=True)
    print(comp.to_string(index=False), flush=True)
    print(f"CH vectors available: {n_ch}/{n_cells}", flush=True)
    comp.to_csv(OUT / "completeness.csv", index=False)
    if not ok:
        sys.exit("INCOMPLETE SWEEP — not plotting partial data.")

    raw = load_raw()
    agg = aggregate(raw)
    raw.to_csv(OUT / "per_seed.csv", index=False)
    agg.to_csv(OUT / "metrics.csv", index=False)
    dN = disclosure_n(agg, GRID_N)
    plot1(raw)
    used_log = plot2(agg)
    table = write_table(agg, dN)
    write_note(comp, agg, dN, n_ch, n_cells, table, used_log)
    print(table.to_string(index=False), flush=True)
    print(f"disclosure N = {dN}", flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
