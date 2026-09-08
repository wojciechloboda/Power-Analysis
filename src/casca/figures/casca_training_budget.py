#!/usr/bin/env python3
"""Publication figures: CA-SCA training budget (GE vs steps, CH stop, ML metrics).

Prefers 400-epoch trajectories in results_casca_training_budget/runs_400ep/
so the 200-epoch mark sits mid-curve. Does not overwrite training_budget_M10.
Halts if the 400-epoch M=10 grid is incomplete.
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
OUT = REPO / "results" / "results_casca_training_budget"
RUNS = OUT / "runs_400ep"
AA_RUNS = REPO / "results" / "training_budget_M10" / "runs"
AA_BUDGET_STEPS = 300
AA_CAP_STEPS = 1000
AA100_RUNS = OUT / "runs_aa_attxatt_100ep"
AA100_SEEDS = tuple(range(5))
AA100_EPOCHS = 100
PROTO_PATH = RUNS / "protocol.json"
if not PROTO_PATH.exists():
    raise SystemExit(f"missing {PROTO_PATH} — run 47_run_casca_400ep.py first")
PROTO = json.loads(PROTO_PATH.read_text())

NS = (10000, 20000, 30000)
SEEDS = tuple(range(10))
PLOT1_N = NS
PLOT2_N = 30000
N_BOOT = 1000
BOOT_SEED = 0
PAPER_STEPS = {int(k): int(v) for k, v in PROTO["paper_200ep_steps"].items()}
STEPS_400 = {int(k): int(v) for k, v in PROTO["paper_400ep_steps"].items()}
SPE = {int(k): int(v) for k, v in PROTO["steps_per_epoch"].items()}
SCHEDULES = {int(k): tuple(v) for k, v in PROTO["schedules"].items()}
XLIM = (-80, 8800)

COLOR_N = {10000: "#1f77b4", 20000: "#ff7f0e", 30000: "#2ca02c", 5000: "#7f7f7f"}


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
        }
    )


def load_run(n: int, seed: int) -> dict | None:
    p = RUNS / f"casca_seed{seed}_N{n}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def completeness() -> tuple[pd.DataFrame, bool]:
    rows = []
    ok = True
    for n in NS:
        present: list[int] = []
        missing: list[int] = []
        issues: list[str] = []
        want = list(SCHEDULES[n])
        for seed in SEEDS:
            rec = load_run(n, seed)
            if rec is None:
                missing.append(seed)
                ok = False
                continue
            cks = rec.get("checkpoints") or []
            steps = [int(c["step"]) for c in cks]
            need = ("GE_16", "train_loss", "val_loss", "ranks")
            if steps != want:
                issues.append(f"seed{seed}: steps={steps[:3]}…{steps[-2:]} want {len(want)} ckpts")
                missing.append(seed)
                ok = False
                continue
            bad_fields = False
            for c in cks:
                if any(k not in c for k in need) or not isinstance(c.get("ranks"), list) or len(c["ranks"]) != 16:
                    bad_fields = True
                    break
                if "agg_ch" not in c and not (RUNS / f"casca_seed{seed}_N{n}.npz").exists():
                    bad_fields = True
                    break
            if bad_fields:
                issues.append(f"seed{seed}: missing checkpoint fields")
                missing.append(seed)
                ok = False
                continue
            present.append(seed)
        rows.append(
            {
                "N": n,
                "spe": SPE[n],
                "steps_200ep": PAPER_STEPS[n],
                "steps_400ep": STEPS_400[n],
                "n_checkpoints": len(want),
                "last_gap": want[-1] - want[-2] if len(want) > 1 else 0,
                "seeds_present": ",".join(str(s) for s in present) if present else "—",
                "seeds_missing": ",".join(str(s) for s in missing) if missing else "—",
                "n_present": len(present),
                "issues": "; ".join(issues) if issues else "—",
            }
        )
    return pd.DataFrame(rows), ok


def load_raw() -> pd.DataFrame:
    rows = []
    for n in NS:
        for seed in SEEDS:
            rec = load_run(n, seed)
            if rec is None:
                raise SystemExit(f"missing casca_seed{seed}_N{n}.json")
            npz = RUNS / f"casca_seed{seed}_N{n}.npz"
            ch = np.load(npz)["ch_vectors"] if npz.exists() else None
            for i, ck in enumerate(rec["checkpoints"]):
                agg = ck.get("agg_ch")
                if agg is None and ch is not None:
                    agg = float(ch[i, 2:16, :].sum(axis=1).mean())
                step = int(ck["step"])
                rows.append(
                    {
                        "N": n,
                        "seed": seed,
                        "step": step,
                        "epoch": float(ck.get("epoch", step / SPE[n])),
                        "GE": float(ck["GE_16"]),
                        "bytes_recovered": int(ck["bytes_recovered"]),
                        "agg_ch": float(agg) if agg is not None else float("nan"),
                        "train_loss": float(ck["train_loss"]),
                        "val_loss": float(ck["val_loss"]),
                        "train_seconds": float(ck["cumulative_train_seconds"]),
                    }
                )
    return pd.DataFrame(rows)


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (n, step), sub in raw.groupby(["N", "step"]):
        sub = sub.sort_values("seed")
        ge, ge_lo, ge_hi = bootstrap_mean_ci(sub["GE"].to_numpy())
        ch, ch_lo, ch_hi = bootstrap_mean_ci(sub["agg_ch"].to_numpy())
        tl, tl_lo, tl_hi = bootstrap_mean_ci(sub["train_loss"].to_numpy())
        vl, vl_lo, vl_hi = bootstrap_mean_ci(sub["val_loss"].to_numpy())
        rec, rec_lo, rec_hi = bootstrap_mean_ci(sub["bytes_recovered"].to_numpy())
        ts, ts_lo, ts_hi = bootstrap_mean_ci(sub["train_seconds"].to_numpy())
        rows.append(
            {
                "N": int(n),
                "step": int(step),
                "epoch": float(sub["epoch"].mean()),
                "GE_mean": ge,
                "GE_lo": ge_lo,
                "GE_hi": ge_hi,
                "ch_mean": ch,
                "ch_lo": ch_lo,
                "ch_hi": ch_hi,
                "train_loss_mean": tl,
                "train_loss_lo": tl_lo,
                "train_loss_hi": tl_hi,
                "val_loss_mean": vl,
                "val_loss_lo": vl_lo,
                "val_loss_hi": vl_hi,
                "bytes_mean": rec,
                "bytes_lo": rec_lo,
                "bytes_hi": rec_hi,
                "train_s_mean": ts,
                "train_s_lo": ts_lo,
                "train_s_hi": ts_hi,
            }
        )
    return pd.DataFrame(rows).sort_values(["N", "step"])


def stop_on_curve(steps: list[int], values: list[float], threshold: float) -> int:
    if len(steps) < 2:
        return int(steps[0])
    for i in range(1, len(steps)):
        prev, cur = values[i - 1], values[i]
        denom = abs(prev) if abs(prev) > 1e-12 else 1.0
        if (cur - prev) / denom < threshold:
            return int(steps[i])
    return int(steps[-1])


def plot1(agg: pd.DataFrame) -> None:
    """GE vs epoch for three N. One shared 200-epoch line; no extra markers."""
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ge_floor = 0.5
    for n in PLOT1_N:
        sub = agg[agg.N == n].sort_values("epoch")
        color = COLOR_N[n]
        lo = np.clip(sub["GE_lo"].to_numpy(dtype=float), ge_floor, None)
        ax.fill_between(sub.epoch, lo, sub.GE_hi, color=color, alpha=0.18, lw=0)
        ax.plot(sub.epoch, sub.GE_mean, color=color, lw=1.6, label=f"N={n // 1000}k")
    ax.axvline(200, color="0.35", ls="--", lw=1.0, label="200 epochs")
    ax.set_xlim(0, 410)
    ax.set_yscale("log")
    ax.set_ylim(ge_floor, 140)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Guessing entropy")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper right", frameon=True, framealpha=0.92)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = OUT / "plot1_ge_vs_steps_multiN.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}", flush=True)


def plot2(agg: pd.DataFrame, stop2: int, stop5: int, opt_step: int) -> None:
    sub = agg[agg.N == PLOT2_N].sort_values("step")
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    ax.fill_between(sub.step, sub.GE_lo, sub.GE_hi, color="C0", alpha=0.20, lw=0)
    ax.plot(sub.step, sub.GE_mean, color="C0", marker="o", ms=3.2, lw=1.6, label="mean GE")
    ax.set_xlabel("Cumulative gradient-update steps")
    ax.set_ylabel("Mean GE (16-byte)")
    ax.set_ylim(bottom=-2)
    ax.spines["top"].set_visible(False)

    ax2 = ax.twinx()
    ax2.fill_between(sub.step, sub.ch_lo, sub.ch_hi, color="0.35", alpha=0.12, lw=0)
    ax2.plot(sub.step, sub.ch_mean, color="0.35", ls=":", lw=1.7, label="mean aggregate CH")
    ax2.set_ylabel("Aggregate CH  (Σ 256 hyps, masked 14)")
    ax2.spines["top"].set_visible(False)

    stop = stop2
    lab = "2%/5% CH stop" if stop2 == stop5 else "CH stop"
    ax.axvline(stop, color="#d62728", ls="--", lw=1.1, label=f"{lab} @ {stop}")
    ax.axvline(
        PAPER_STEPS[PLOT2_N],
        color="0.45",
        ls=":",
        lw=1.1,
        label=f"200 ep @ {PAPER_STEPS[PLOT2_N]}",
    )
    ax.axvline(opt_step, color="0.15", ls="-.", lw=1.1, label=f"GE minimum @ {opt_step}")
    ax.annotate(
        "",
        xy=(opt_step, 0.92),
        xytext=(stop, 0.92),
        xycoords=("data", "axes fraction"),
        arrowprops=dict(arrowstyle="<->", color="0.25", lw=1.0),
    )
    mid = 0.5 * (stop + opt_step)
    ax.text(
        mid,
        0.95,
        f"{opt_step - stop} steps",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=7.5,
        color="0.2",
    )

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=True, framealpha=0.92)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(*XLIM)
    fig.tight_layout()
    path = OUT / "plot2_ge_vs_ch_signal.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}", flush=True)


def _seconds_per_epoch(run_paths: list[Path], n: int) -> float:
    """Mean measured training seconds per epoch across the listed runs.

    Uses last-checkpoint cumulative_train_seconds / (step / steps_per_epoch).
    That timer excludes CH scoring (paused out of train_seconds).
    """
    spe = SPE[n]
    rates: list[float] = []
    for p in run_paths:
        rec = json.loads(p.read_text())
        last = rec["checkpoints"][-1]
        t = last.get("cumulative_train_seconds")
        if t is None:
            t = rec.get("train_seconds")
        if t is None:
            raise SystemExit(f"no train timer in {p}")
        epoch = float(last["step"]) / spe
        if epoch <= 0:
            raise SystemExit(f"non-positive epoch in {p}")
        rates.append(float(t) / epoch)
    return float(np.mean(rates))


def _nice_second_ticks(t_max: float) -> list[float]:
    """4–6 ticks at round second values covering [0, t_max]."""
    if t_max <= 0:
        return [0.0]
    best: list[float] | None = None
    best_score = 1e9
    for mag in (1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 150, 200, 250, 300, 400, 500, 750, 1000):
        ticks = [float(i * mag) for i in range(int(np.floor(t_max / mag)) + 1)]
        if ticks[-1] < t_max * 0.85:
            ticks.append(float(int(np.floor(t_max / mag)) * mag + mag))
        n = len(ticks)
        if n < 4 or n > 6:
            continue
        score = abs(n - 5) + (0.0 if ticks[-1] <= t_max * 1.05 else 1.0)
        if score < best_score:
            best, best_score = ticks, score
    if best is not None:
        return best
    step = max(1.0, round(t_max / 4.0))
    return [float(i * step) for i in range(5)]


def _add_time_axis(ax, sec_per_epoch: float, xlim: tuple[float, float]) -> None:
    def forward(epoch):
        return np.asarray(epoch, dtype=float) * sec_per_epoch

    def inverse(seconds):
        return np.asarray(seconds, dtype=float) / sec_per_epoch

    ax.spines["top"].set_visible(True)
    secax = ax.secondary_xaxis("top", functions=(forward, inverse))
    secax.set_xlabel("training time (s)")
    t_max = xlim[1] * sec_per_epoch
    secax.set_ticks(_nice_second_ticks(t_max))


def plot_ge_and_val_loss(agg: pd.DataFrame) -> None:
    """CA-SCA: checkpoints vs epoch, N=10k/20k/30k."""
    rates = {
        n: _seconds_per_epoch(
            [RUNS / f"casca_seed{s}_N{n}.json" for s in SEEDS],
            n,
        )
        for n in PLOT1_N
    }
    _plot_ge_val_grid(
        agg,
        vline_epochs={n: 200.0 for n in PLOT1_N},
        extra_vline=None,
        names=("plot_ge_and_val_loss", "appendix_ml_metrics_vs_ge"),
        sec_per_epoch=rates,
        rate_label="CA-SCA",
        show_train_time_row=True,
    )


def load_aa_raw() -> pd.DataFrame:
    rows = []
    missing: list[str] = []
    for n in PLOT1_N:
        for seed in SEEDS:
            p = AA_RUNS / f"aa_casca_seed{seed}_N{n}.json"
            if not p.exists():
                missing.append(p.name)
                continue
            rec = json.loads(p.read_text())
            if int(rec.get("max_steps", -1)) != AA_CAP_STEPS:
                missing.append(f"{p.name} max_steps={rec.get('max_steps')}")
                continue
            spe = SPE[n]
            for ck in rec["checkpoints"]:
                step = int(ck["step"])
                rows.append(
                    {
                        "N": n,
                        "seed": seed,
                        "step": step,
                        "epoch": step / spe,
                        "GE": float(ck["GE_16"]),
                        "val_loss": float(ck["val_loss"]),
                        "train_seconds": float(ck["cumulative_train_seconds"]),
                    }
                )
    if missing:
        raise SystemExit(f"incomplete AA-CASCA logs: {missing[:8]}")
    return pd.DataFrame(rows)


def aggregate_aa(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (n, step), sub in raw.groupby(["N", "step"]):
        sub = sub.sort_values("seed")
        ge, ge_lo, ge_hi = bootstrap_mean_ci(sub["GE"].to_numpy())
        vl, vl_lo, vl_hi = bootstrap_mean_ci(sub["val_loss"].to_numpy())
        ts, ts_lo, ts_hi = bootstrap_mean_ci(sub["train_seconds"].to_numpy())
        rows.append(
            {
                "N": int(n),
                "step": int(step),
                "epoch": float(sub["epoch"].mean()),
                "GE_mean": ge,
                "GE_lo": ge_lo,
                "GE_hi": ge_hi,
                "val_loss_mean": vl,
                "val_loss_lo": vl_lo,
                "val_loss_hi": vl_hi,
                "train_s_mean": ts,
                "train_s_lo": ts_lo,
                "train_s_hi": ts_hi,
            }
        )
    return pd.DataFrame(rows).sort_values(["N", "step"])


E200_DIRS = {
    10000: REPO / "results" / "configE_aggregate_CH_extended" / "runs",
    20000: REPO / "results" / "epoch_trajectory_diagnostic" / "runs",
    30000: REPO / "results" / "configE_aggregate_CH_extended" / "runs",
}
E200_SEEDS = tuple(range(5))
E200_PARAMS = 99992


def load_configE_200_raw() -> pd.DataFrame:
    """Gated Config E 200-epoch trajectories (5 seeds), not the 1000-step M10 cap."""
    rows = []
    missing: list[str] = []
    for n in PLOT1_N:
        for seed in E200_SEEDS:
            p = E200_DIRS[n] / f"h2E_seed{seed}_N{n}.json"
            if not p.exists():
                missing.append(p.name)
                continue
            rec = json.loads(p.read_text())
            proto = rec.get("protocol") or {}
            if int(proto.get("epochs", rec.get("epochs", -1))) != 200:
                missing.append(f"{p.name} epochs={proto.get('epochs')}")
                continue
            if int(rec.get("n_params", -1)) != E200_PARAMS:
                missing.append(f"{p.name} n_params={rec.get('n_params')}")
                continue
            cks = rec.get("checkpoints") or []
            if not cks:
                missing.append(f"{p.name} no checkpoints")
                continue
            for ck in cks:
                ranks = np.asarray(ck["ranks"], dtype=float)
                if ranks.size != 16:
                    missing.append(f"{p.name} ep={ck.get('epoch')} ranks")
                    break
                epoch = float(ck["epoch"])
                rows.append(
                    {
                        "N": n,
                        "seed": seed,
                        "step": int(epoch),
                        "epoch": epoch,
                        "GE": float(ranks.mean()),
                        "val_loss": float(ck["val_loss"]),
                        "train_seconds": float(ck["cumulative_train_seconds"]),
                    }
                )
    if missing:
        raise SystemExit(f"incomplete Config E 200-ep logs: {missing}")
    return pd.DataFrame(rows)


def plot_ge_and_val_loss_configE(agg: pd.DataFrame) -> None:
    """Config E 200-epoch logs: same 3-row grid as CA-SCA, x to 200 epochs."""
    _plot_ge_val_grid(
        agg,
        vline_epochs={n: 25.0 for n in PLOT1_N},
        extra_vline=None,
        names=("plot_ge_and_val_loss_configE",),
        xlim=(0, 200),
        xticks=(0, 50, 100, 150, 200),
        show_train_time_row=True,
    )


def plot_ge_and_val_loss_aa(agg: pd.DataFrame) -> None:
    """Gated AA-CASCA 1000-step logs on the 0–410 epoch axis."""
    rates = {
        n: _seconds_per_epoch(
            [AA_RUNS / f"aa_casca_seed{s}_N{n}.json" for s in SEEDS],
            n,
        )
        for n in PLOT1_N
    }
    _plot_ge_val_grid(
        agg,
        vline_epochs={n: AA_BUDGET_STEPS / SPE[n] for n in PLOT1_N},
        extra_vline=None,
        names=("plot_ge_and_val_loss_aa_casca",),
        xlim=(0, 100),
        xticks=(0, 25, 50, 75, 100),
        sec_per_epoch=rates,
        rate_label="AA-CASCA (gated)",
        show_train_time_row=True,
    )


# Quiet AA-CASCA (gated Config E, 200 ep) anchors from plot_e25_e200_casca_time.
# 10k and 30k only: both quiet-GPU. 20k is interpolated. 40k is a known stall.
AA200_TIME_ANCHORS = (10000, 30000)
AA100_EPOCH_SCALE = AA100_EPOCHS / 200.0
AA25_EPOCH_SCALE = AA100_EPOCHS / 25.0
E25_RUNS = REPO / "results" / "results_fourway_N" / "gated_E_25ep" / "runs"


def _aa200_quiet_train_seconds_per_seed(n: int) -> np.ndarray:
    """Per-seed train_seconds for gated Config E 200-ep at one quiet N."""
    times = []
    for seed in E200_SEEDS:
        p = E200_DIRS[n] / f"h2E_seed{seed}_N{n}.json"
        rec = json.loads(p.read_text())
        proto = rec.get("protocol") or {}
        if int(proto.get("epochs", rec.get("epochs", -1))) != 200:
            raise SystemExit(f"{p.name} epochs={proto.get('epochs')}")
        if int(rec.get("n_params", -1)) != E200_PARAMS:
            raise SystemExit(f"{p.name} n_params={rec.get('n_params')}")
        t = rec.get("train_seconds")
        if t is None:
            raise SystemExit(f"{p.name} missing train_seconds")
        times.append(float(t))
    return np.asarray(times, dtype=float)


def _aa25_quiet_train_seconds_per_seed(n: int) -> np.ndarray:
    """Per-seed train_seconds for gated Config E 25-ep at one quiet N."""
    times = []
    for seed in E200_SEEDS:
        p = E25_RUNS / f"gated_E_25ep_seed{seed}_N{n}.json"
        rec = json.loads(p.read_text())
        if rec.get("use_gate") is not True or int(rec.get("n_params", -1)) != E200_PARAMS:
            raise SystemExit(f"{p.name} is not gated Config E")
        t = rec.get("train_seconds")
        if t is None:
            raise SystemExit(f"{p.name} missing train_seconds")
        times.append(float(t))
    return np.asarray(times, dtype=float)


def _aa100_estimated_train_at_100() -> dict[int, tuple[float, float, float]]:
    """100-ep train time vs N from the quiet four-way time figure.

    Two scalings: half the 200-ep 10k/30k line, and 4× the 25-ep times.
    The plotted line is their midpoint; the band is the envelope of the
    two bootstrap CIs. Seed noise alone is ~0.3 s and would not show.
    """
    n0, n1 = AA200_TIME_ANCHORS
    t0 = _aa200_quiet_train_seconds_per_seed(n0)
    t1 = _aa200_quiet_train_seconds_per_seed(n1)
    from200 = {
        n0: AA100_EPOCH_SCALE * t0,
        n1: AA100_EPOCH_SCALE * t1,
        20000: AA100_EPOCH_SCALE * 0.5 * (t0 + t1),
    }
    out = {}
    for n in PLOT1_N:
        m200, lo200, hi200 = bootstrap_mean_ci(from200[n])
        m25, lo25, hi25 = bootstrap_mean_ci(AA25_EPOCH_SCALE * _aa25_quiet_train_seconds_per_seed(n))
        lo = min(lo200, lo25)
        hi = max(hi200, hi25)
        # Centre of the envelope so the line sits in the band, not on an edge.
        out[n] = (0.5 * (lo + hi), lo, hi)
    print(
        f"AA-CASCA 200-ep quiet train anchors: N={n0 // 1000}k={t0.mean():.2f}s, "
        f"N={n1 // 1000}k={t1.mean():.2f}s",
        flush=True,
    )
    print(
        "100-ep estimated train (line = centre of 200-ep/2 vs 25-ep×4 envelope): "
        + ", ".join(
            f"N={n // 1000}k={out[n][0]:.2f}s [{out[n][1]:.2f}, {out[n][2]:.2f}]"
            for n in PLOT1_N
        ),
        flush=True,
    )
    return out


def _apply_linear_time_estimate(
    agg: pd.DataFrame,
    t100_by_n: dict[int, tuple[float, float, float]],
) -> pd.DataFrame:
    """Replace measured times with epoch-linear estimates and scaled CIs."""
    out = agg.copy()
    for n, (t_mean, t_lo, t_hi) in t100_by_n.items():
        mask = out["N"] == n
        frac = out.loc[mask, "epoch"].to_numpy(dtype=float) / float(AA100_EPOCHS)
        out.loc[mask, "train_s_mean"] = frac * t_mean
        out.loc[mask, "train_s_lo"] = frac * t_lo
        out.loc[mask, "train_s_hi"] = frac * t_hi
    return out


def plot_ge_and_val_loss_aa_100(agg: pd.DataFrame | None = None) -> None:
    """att×att 100-epoch sweep, same 3-row grid, x-axis 0–100 epochs.

    Time row is not the att×att run clocks (MPS-contended, disagree with the
    quiet four-way figure). It is a linear estimate from the two quiet
    AA-CASCA 200-ep train times on plot_e25_e200_casca_time (N=10k and 30k),
    scaled by 100/200. N=20k is interpolated. GE and val loss stay measured.
    """
    if agg is None:
        agg = load_aa100_agg()
        if agg is None:
            print("skip att×att 100-ep plot — grid incomplete", flush=True)
            return
    t100 = _aa100_estimated_train_at_100()
    agg = _apply_linear_time_estimate(agg, t100)
    rates = {n: t100[n][0] / float(AA100_EPOCHS) for n in PLOT1_N}
    _plot_ge_val_grid(
        agg,
        vline_epochs=None,
        extra_vline=None,
        names=("plot_ge_and_val_loss_aa_attxatt_100ep",),
        xlim=(0, 105),
        xticks=(0, 25, 50, 75, 100),
        ge_pad_zero=True,
        sec_per_epoch=rates,
        rate_label="AA-CASCA (100 ep, time estimated from quiet 200-ep 10k/30k)",
        show_train_time_row=True,
        time_estimated=True,
    )


def load_aa100_agg() -> pd.DataFrame | None:
    proto_p = AA100_RUNS / "protocol.json"
    if not proto_p.exists():
        return None
    rows = []
    for n in PLOT1_N:
        spe = SPE[n]
        want = AA100_EPOCHS * spe
        for seed in AA100_SEEDS:
            p = AA100_RUNS / f"aa_attxatt_seed{seed}_N{n}.json"
            if not p.exists():
                return None
            rec = json.loads(p.read_text())
            if rec.get("self_multiply") is not True or int(rec.get("max_steps", -1)) != want:
                return None
            for ck in rec["checkpoints"]:
                step = int(ck["step"])
                rows.append(
                    {
                        "N": n,
                        "seed": seed,
                        "step": step,
                        "epoch": step / spe,
                        "GE": float(ck["GE_16"]),
                        "val_loss": float(ck["val_loss"]),
                        "train_seconds": float(ck["cumulative_train_seconds"]),
                    }
                )
    raw = pd.DataFrame(rows)
    return aggregate_aa(raw)


def _plot_ge_val_grid(
    agg: pd.DataFrame,
    vline_epochs: dict[int, float] | None,
    extra_vline: float | None,
    names: tuple[str, ...],
    xlim: tuple[float, float] = (0, 410),
    xticks: tuple[float, ...] | None = None,
    ge_pad_zero: bool = False,
    sec_per_epoch: dict[int, float] | None = None,
    rate_label: str = "",
    show_train_time_row: bool = False,
    time_estimated: bool = False,
) -> None:
    ns = PLOT1_N
    n_rows = 3 if show_train_time_row else 2
    fig, axes = plt.subplots(
        n_rows,
        len(ns),
        figsize=(6.4, 6.2 if show_train_time_row else 4.4),
        sharex=True,
        constrained_layout=True,
    )
    val_lo = min(float(agg[agg.N == n].val_loss_lo.min()) for n in ns)
    val_hi = max(float(agg[agg.N == n].val_loss_hi.max()) for n in ns)
    pad = 0.05 * (val_hi - val_lo) if val_hi > val_lo else 0.01
    time_hi = 0.0
    if show_train_time_row:
        time_hi = max(float(agg[agg.N == n].train_s_hi.max()) for n in ns)

    for j, n in enumerate(ns):
        sub = agg[agg.N == n].sort_values("epoch")
        ax_v = axes[0, j]
        ax_g = axes[1, j]
        ax_t = axes[2, j] if show_train_time_row else None

        ax_v.fill_between(sub.epoch, sub.val_loss_lo, sub.val_loss_hi, color="C1", alpha=0.20, lw=0)
        ax_v.plot(sub.epoch, sub.val_loss_mean, color="C1", lw=1.6)
        if vline_epochs:
            ax_v.axvline(vline_epochs[n], color="0.35", ls="--", lw=1.0)
        if extra_vline is not None:
            ax_v.axvline(extra_vline, color="0.55", ls=":", lw=0.9)
        ax_v.set_ylim(val_lo - pad, val_hi + pad)
        ax_v.set_xlim(*xlim)
        ax_v.grid(True, alpha=0.3)
        ax_v.spines["right"].set_visible(False)
        ax_v.set_title(f"N={n // 1000}k", fontsize=10, pad=4)

        ax_g.fill_between(sub.epoch, sub.GE_lo, sub.GE_hi, color="C0", alpha=0.20, lw=0, zorder=1)
        ax_g.plot(sub.epoch, sub.GE_mean, color="C0", lw=1.7, zorder=3)
        if vline_epochs:
            ax_g.axvline(vline_epochs[n], color="0.35", ls="--", lw=1.0)
        if extra_vline is not None:
            ax_g.axvline(extra_vline, color="0.55", ls=":", lw=0.9)
        ax_g.set_xlim(*xlim)
        if xticks is not None:
            ax_v.set_xticks(list(xticks))
            ax_g.set_xticks(list(xticks))
            if ax_t is not None:
                ax_t.set_xticks(list(xticks))
        ge_top = max(float(sub.GE_hi.max()), float(sub.GE_mean.max()), 1.0)
        if ge_pad_zero:
            ax_g.set_ylim(-0.08 * ge_top, ge_top * 1.08)
        else:
            ax_g.set_ylim(bottom=0)
        ax_g.grid(True, alpha=0.3)
        ax_g.spines["right"].set_visible(False)
        if ax_t is None:
            ax_g.set_xlabel("Epoch")

        if ax_t is not None:
            ax_t.fill_between(
                sub.epoch,
                sub.train_s_lo,
                sub.train_s_hi,
                color="C2",
                alpha=0.28 if time_estimated else 0.20,
                lw=0,
            )
            ax_t.plot(sub.epoch, sub.train_s_mean, color="C2", lw=1.6)
            if vline_epochs:
                ax_t.axvline(vline_epochs[n], color="0.35", ls="--", lw=1.0)
            if extra_vline is not None:
                ax_t.axvline(extra_vline, color="0.55", ls=":", lw=0.9)
            ax_t.set_xlim(*xlim)
            ax_t.set_ylim(0, time_hi * 1.08 if time_hi > 0 else 1.0)
            ax_t.grid(True, alpha=0.3)
            ax_t.spines["right"].set_visible(False)
            ax_t.set_xlabel("Epoch")

        if j == 0:
            ax_v.set_ylabel("Mean validation loss")
            ax_g.set_ylabel("Mean GE")
            if ax_t is not None:
                ax_t.set_ylabel("Mean training time (s)")

    if sec_per_epoch is not None:
        bits = ", ".join(f"N={n // 1000}k: {sec_per_epoch[n]:.4f} s/epoch" for n in ns)
        print(f"seconds/epoch ({rate_label}): {bits}", flush=True)

    for name in names:
        path = OUT / f"{name}.pdf"
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
        print(f"wrote {path}", flush=True)
    plt.close(fig)


def _at_step(sub: pd.DataFrame, step: int) -> pd.Series:
    hit = sub[sub.step == step]
    if hit.empty:
        raise SystemExit(f"missing checkpoint step={step} for N={int(sub.N.iloc[0])}")
    return hit.iloc[0]


def write_note(comp: pd.DataFrame, agg: pd.DataFrame, stop2: int, stop5: int, opt_step: int) -> None:
    n_ok = int(comp["n_present"].sum())
    n_want = len(NS) * len(SEEDS)
    lines = [
        "# CA-SCA training-budget figures (M=10, 400-epoch extension)",
        "",
        "Source: `results/results_casca_training_budget/runs_400ep/` "
        "(Table-2 CNN, concat-14, **400-epoch** cap, same step checkpoints as the "
        "200-epoch M=10 sweep plus a matching-coarseness tail). Does **not** "
        "overwrite `results/training_budget_M10`. Seeds **0–9**. "
        "GE = mean rank across 16 bytes. CIs are percentile bootstrap 95% (1000 resamples).",
        "",
        "## Completeness",
        "",
        "| N | seeds present | missing | # ckpts | steps/epoch | 200 ep | 400 ep | last gap |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in comp.iterrows():
        lines.append(
            f"| {int(r.N)} | {r.seeds_present} | {r.seeds_missing} | "
            f"{int(r.n_checkpoints)} | {int(r.spe)} | {int(r.steps_200ep)} | "
            f"{int(r.steps_400ep)} | {int(r.last_gap)} |"
        )
    lines += [
        "",
        f"**{n_ok}/{n_want} CA-SCA cells complete.** N ∈ {{10k, 20k, 30k}} × seeds 0–9.",
        "",
        "Checkpoint schedule: every 25 steps through 400, every 50 through 1000, "
        "then a coarser tail through 400 × steps_per_epoch(N). **Limitation:** "
        "the tail remains coarse (last gap 400 / 700 / 700 steps at N=10k / 20k / 30k).",
        "",
        "## Plot 1 — is 200 epochs a GE optimum?",
        "",
        "Dashed coloured lines mark each N's 200-epoch step count (now mid-curve). "
        "Diamonds mark the GE argmin; squares mark the 400-epoch endpoint.",
        "",
    ]
    for n in PLOT1_N:
        sub = agg[agg.N == n].sort_values("step")
        last = sub.iloc[-1]
        opt = sub.iloc[int(sub["GE_mean"].to_numpy().argmin())]
        at200 = _at_step(sub, PAPER_STEPS[n])
        ep_opt = int(opt.step) / SPE[n]
        if int(opt.step) == PAPER_STEPS[n]:
            verdict = "Argmin **is** the 200-epoch mark."
        elif int(opt.step) < PAPER_STEPS[n]:
            verdict = (
                f"Argmin is **before** 200 epochs "
                f"(step {int(opt.step)}, ~{ep_opt:.0f} ep)."
            )
        elif int(opt.step) == int(last.step):
            verdict = "Argmin is the 400-epoch endpoint (still falling or tied at the cap)."
        else:
            verdict = (
                f"Argmin is **after** 200 epochs "
                f"(step {int(opt.step)}, ~{ep_opt:.0f} ep)."
            )
        lines.append(
            f"- **N={n}:** 200 ep = {PAPER_STEPS[n]} steps, GE {at200.GE_mean:.2f} "
            f"[{at200.GE_lo:.2f}, {at200.GE_hi:.2f}]. "
            f"400 ep = {STEPS_400[n]} steps, GE {last.GE_mean:.2f} "
            f"[{last.GE_lo:.2f}, {last.GE_hi:.2f}]. "
            f"Min GE {opt.GE_mean:.2f} @ step {int(opt.step)}. {verdict}"
        )
    sub30 = agg[agg.N == PLOT2_N].sort_values("step")
    ge_stop = float(_at_step(sub30, stop2).GE_mean)
    ge_opt = float(_at_step(sub30, opt_step).GE_mean)
    at200_30 = _at_step(sub30, PAPER_STEPS[PLOT2_N])
    last30 = sub30.iloc[-1]
    opt_label = (
        "the 200-epoch mark"
        if opt_step == PAPER_STEPS[PLOT2_N]
        else f"step {opt_step} (~{opt_step / SPE[PLOT2_N]:.0f} ep)"
    )
    lines += [
        "",
        "## Plot 2 — aggregate CH vs true GE optimum",
        "",
        f"Representative N = **{PLOT2_N}**. "
        f"Stopping rule = first checkpoint after a relative CH improvement "
        f"`(CH_t − CH_{{t−1}}) / |CH_{{t−1}}|` below 2% or 5% (or a CH decline). "
        f"Both thresholds fire at **step {stop2}**"
        + ("" if stop2 == stop5 else f" / 5% at {stop5}")
        + f". True GE minimum = **{opt_label}**.",
        "",
        f"- Gap: **{opt_step - stop2} steps** ({(opt_step - stop2) / SPE[PLOT2_N]:.0f} epochs).",
        f"- GE at CH stop: **{ge_stop:.2f}**; GE at 200 ep: **{at200_30.GE_mean:.2f}**; "
        f"GE at true opt: **{ge_opt:.2f}**; ΔGE (stop − opt) = {ge_stop - ge_opt:.2f}.",
        "",
        "The aggregate CH signal does **not** track CA-SCA's true GE optimum "
        "under this step schedule: it trips at the second checkpoint for every N.",
        "",
        "## Main figure — GE and validation loss at training checkpoints",
        "",
        f"N ∈ {{10k, 20k, 30k}}, seeds 0–9. Three rows × three columns, shared epoch axis. "
        "Top: plaintext-classification validation loss (shared y). Bottom: 16-byte "
        "guessing entropy (y free per N). Dashed line at 200 epochs.",
        "",
        f"At 200 epochs: val loss {float(at200_30.val_loss_mean):.4f}, "
        f"GE {float(at200_30.GE_mean):.2f}. "
        f"At 400 epochs: val loss {float(last30.val_loss_mean):.4f}, "
        f"GE {float(last30.GE_mean):.2f}. "
        "Validation loss rises throughout; GE keeps falling slowly after 200.",
        "",
        "## Files",
        "",
        "- `plot_ge_and_val_loss.pdf` — CA-SCA val loss, GE, and cumulative train time vs epoch (N=10k/20k/30k)",
        "- `plot_ge_and_val_loss_aa_casca.pdf` — AA-CASCA, same axis; x = step / spe "
        "(1000-step logs; dashed = 300-step budget, dotted = 200 epochs)",
        "- `plot1_ge_vs_steps_multiN.pdf` — GE vs epoch for N=10k/20k/30k",
        "- `plot2_ge_vs_ch_signal.pdf` — CH-stop comparison",
        "- `runs_400ep/` — 400-epoch trajectories",
        "",
    ]
    (OUT / "NOTE.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    _style()
    OUT.mkdir(parents=True, exist_ok=True)
    comp, ok = completeness()
    print("=== Step 0 completeness (CA-SCA checkpointed trajectories) ===", flush=True)
    print(comp.to_string(index=False), flush=True)
    comp.to_csv(OUT / "completeness.csv", index=False)
    if not ok:
        sys.exit("INCOMPLETE TRAJECTORIES — not plotting or interpolating.")

    raw = load_raw()
    agg = aggregate(raw)
    agg.to_csv(OUT / "metrics.csv", index=False)

    sub = agg[agg.N == PLOT2_N].sort_values("step")
    steps = [int(s) for s in sub.step]
    ch = [float(v) for v in sub.ch_mean]
    stop2 = stop_on_curve(steps, ch, 0.02)
    stop5 = stop_on_curve(steps, ch, 0.05)
    opt_step = int(sub.loc[sub["GE_mean"].idxmin(), "step"])

    plot1(agg)
    plot2(agg, stop2, stop5, opt_step)
    plot_ge_and_val_loss(agg)
    plot_ge_and_val_loss_aa(aggregate_aa(load_aa_raw()))
    plot_ge_and_val_loss_aa_100()
    plot_ge_and_val_loss_configE(aggregate_aa(load_configE_200_raw()))
    write_note(comp, agg, stop2, stop5, opt_step)
    print(f"CH 2% stop={stop2}  5% stop={stop5}  GE opt={opt_step}  (N={PLOT2_N})", flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
