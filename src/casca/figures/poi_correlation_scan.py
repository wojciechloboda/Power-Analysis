#!/usr/bin/env python3
"""Thesis figure: full-trace POI correlation scans that produced Table 3.2.

The stored Table 3.2 results are peak locations and ρ only. This script
recomputes the N=4000 signed Pearson scans (same key, masks, first traces)
and draws the two-panel figure. Scans are cached so a re-run does not
touch the HDF5 traces.
"""
from __future__ import annotations

import sys
from pathlib import Path

import sys as _casca_sys
_CASCA_ROOT = Path(__file__).resolve().parents[3]
if not (_CASCA_ROOT / 'src' / 'casca').is_dir():
    _CASCA_ROOT = Path(__file__).resolve().parents[3]
_casca_sys.path.insert(0, str(_CASCA_ROOT / "src"))
from casca.paths import raw_traces_path, repo_root, window_cache_path  # noqa: E402

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from casca.data.ascad import sbox  # noqa: E402
from casca.data.ascad import ASCAD_KEY  # noqa: E402
from casca.data.windows import BYTE_WINDOWS, MASKED_BYTES  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
RAW = raw_traces_path()
TABLE = REPO / "results" / "poi_full16_scan" / "poi_scan_table.csv"
CACHE = REPO / "results" / "poi_full16_scan" / "full_trace_scans.npz"
OUT = Path(__file__).resolve().parents[3] / "figures" / "ascad_poi_correlation_scan.pdf"

N_TRACES = 4000
TRACE_LEN = 100_000
HW_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

# Table 3.2 (in-window argmax |r|).
TABLE_MASK = {
    2: 45556,
    3: 33063,
    4: 47638,
    5: 41392,
    6: 37227,
    7: 35145,
    8: 26816,
    9: 39309,
    10: 28898,
    11: 43474,
    12: 20569,
    13: 22651,
    14: 49721,
    15: 18486,
}
TABLE_VAL = {
    0: 31338,
    1: 25091,
    2: 45917,
    3: 33420,
    4: 47996,
    5: 41753,
    6: 37617,
    7: 35506,
    8: 27177,
    9: 39667,
    10: 29259,
    11: 43835,
    12: 20958,
    13: 23012,
    14: 50082,
    15: 18844,
}

BYTE2_WINDOW = (45_400, 46_100)


def pearson_1d(traces: np.ndarray, hyp: np.ndarray) -> np.ndarray:
    """Signed Pearson r vs every sample. Same formula as script 39."""
    x = np.asarray(hyp, dtype=np.float64).reshape(-1)
    y = np.asarray(traces, dtype=np.float64)
    xc = x - x.mean()
    yc = y - y.mean(axis=0, keepdims=True)
    num = xc @ yc
    xnorm = float(np.sqrt(np.dot(xc, xc)))
    ynorm = np.sqrt(np.einsum("ij,ij->j", yc, yc))
    den = xnorm * ynorm
    r = np.zeros_like(num)
    np.divide(num, den, out=r, where=den > 1e-15)
    return r


def peak_abs(r: np.ndarray, lo: int, hi: int) -> tuple[int, float]:
    sl = r[lo:hi]
    i = int(np.abs(sl).argmax())
    return lo + i, float(sl[i])


def load_or_compute_scans() -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    if CACHE.exists():
        z = np.load(CACHE)
        r_mask = {int(k.split("_")[1]): z[k] for k in z.files if k.startswith("mask_")}
        r_val = {int(k.split("_")[1]): z[k] for k in z.files if k.startswith("val_")}
        if set(r_mask) == set(MASKED_BYTES) and set(r_val) == set(range(16)):
            print(f"reusing cached scans {CACHE}", flush=True)
            return r_mask, r_val
        print("cache incomplete; recomputing", flush=True)

    print(f"loading N={N_TRACES} traces from {RAW}", flush=True)
    with h5py.File(RAW, "r") as f:
        traces = f["traces"][:N_TRACES].astype(np.float64)
        meta = f["metadata"]
        pt = meta["plaintext"][:N_TRACES].astype(np.int64)
        key = meta["key"][0].astype(np.int64)
        masks = meta["masks"][:N_TRACES].astype(np.int64)

    if traces.shape[1] != TRACE_LEN:
        raise SystemExit(f"expected {TRACE_LEN} samples, got {traces.shape}")
    if not np.array_equal(key, ASCAD_KEY):
        raise SystemExit(f"HDF5 key != published ASCAD_KEY: {key.tolist()}")

    r_mask: dict[int, np.ndarray] = {}
    r_val: dict[int, np.ndarray] = {}
    for b in range(16):
        sb = sbox((pt[:, b] ^ int(key[b])) & 0xFF)
        if b in MASKED_BYTES:
            m = masks[:, b - 2].astype(np.int64) & 0xFF
            print(f"  byte {b:2d}: HW(mask) …", flush=True)
            r_mask[b] = pearson_1d(traces, HW_LUT[m].astype(np.float64))
            hw_v = HW_LUT[(sb.astype(np.int64) ^ m) & 0xFF].astype(np.float64)
        else:
            hw_v = HW_LUT[sb].astype(np.float64)
        print(f"  byte {b:2d}: HW(masked value) …", flush=True)
        r_val[b] = pearson_1d(traces, hw_v)

    payload = {f"mask_{b}": r_mask[b] for b in MASKED_BYTES}
    payload.update({f"val_{b}": r_val[b] for b in range(16)})
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **payload)
    print(f"wrote {CACHE}", flush=True)
    return r_mask, r_val


def check_against_table(
    r_mask: dict[int, np.ndarray], r_val: dict[int, np.ndarray]
) -> tuple[list[str], list[str]]:
    """Return (position mismatches, negative-peak notes)."""
    stored = pd.read_csv(TABLE)
    mismatches: list[str] = []
    negatives: list[str] = []
    print("=== peak check vs Table 3.2 (in-window argmax |r|) ===", flush=True)
    print(
        f"{'byte':>4}  {'kind':<6}  {'table t':>8}  {'scan t':>8}  "
        f"{'table ρ':>8}  {'signed r':>9}  {'ok':>4}",
        flush=True,
    )
    for b in range(16):
        lo, hi = BYTE_WINDOWS[b]
        row = stored.loc[stored.byte == b].iloc[0]
        checks = []
        if b in MASKED_BYTES:
            t, rho = peak_abs(r_mask[b], lo, hi)
            checks.append(("mask", int(TABLE_MASK[b]), t, float(row.rho_mask), rho))
        t, rho = peak_abs(r_val[b], lo, hi)
        checks.append(("value", int(TABLE_VAL[b]), t, float(row.rho_maskedvalue), rho))
        for kind, t_tab, t_scan, rho_tab, r_signed in checks:
            pos_ok = t_scan == t_tab
            if not pos_ok:
                mismatches.append(
                    f"byte {b} {kind}: scan {t_scan} vs table {t_tab}"
                )
            if r_signed < 0:
                negatives.append(
                    f"byte {b} {kind}: t={t_scan} signed r={r_signed:+.4f} "
                    f"(table reports +{rho_tab:.3f})"
                )
            print(
                f"{b:4d}  {kind:<6}  {t_tab:8d}  {t_scan:8d}  "
                f"{rho_tab:8.3f}  {r_signed:+9.4f}  "
                f"{'yes' if pos_ok else 'NO':>4}",
                flush=True,
            )
    return mismatches, negatives


def plot_figure(
    r_mask: dict[int, np.ndarray], r_val: dict[int, np.ndarray]
) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
        }
    )

    mask_c = "C0"
    val_c = "C2"
    trace_c = "0.75"
    start, stop = BYTE2_WINDOW
    t_mask = TABLE_MASK[2]
    t_val = TABLE_VAL[2]

    with h5py.File(RAW, "r") as f:
        trace = np.asarray(f["traces"][0, start:stop], dtype=np.int16)

    fig, ax = plt.subplots(figsize=(12.0 / 2.54, 9.0 / 2.54))
    x = np.arange(start, stop)

    ax_t = ax.twinx()
    ax_t.plot(x, trace, color=trace_c, lw=0.55, zorder=0, label="trace")
    ax_t.set_ylabel("measured value")
    ax_t.set_zorder(0)
    ax_t.spines["top"].set_visible(False)
    ax.set_zorder(1)
    ax.patch.set_visible(False)

    ax.plot(x, r_mask[2][start:stop], color=mask_c, ls="-", lw=1.1, label="HW(mask)", zorder=3)
    ax.plot(
        x,
        r_val[2][start:stop],
        color=val_c,
        ls="-",
        lw=1.1,
        label="HW(masked value)",
        zorder=3,
    )
    ax.axvline(
        t_mask, color=mask_c, ls=":", lw=1.1, zorder=2, label="mask peak"
    )
    ax.axvline(
        t_val, color=val_c, ls=":", lw=1.1, zorder=2, label="masked-value peak"
    )
    ax.axhline(0.0, color="0.55", lw=0.4, zorder=2)
    y0 = min(float(r_mask[2][start:stop].min()), float(r_val[2][start:stop].min()))
    y1 = max(float(r_mask[2][start:stop].max()), float(r_val[2][start:stop].max()))
    ax.set_xlim(start, stop)
    ax.set_ylim(y0 - 0.04 * (y1 - y0), y1 + 0.08 * (y1 - y0))
    ax.set_xlabel("sample index")
    ax.set_ylabel("correlation coefficient")
    ax.set_xticks((45400, t_mask, 45800, t_val, 46100))
    ax.grid(True, alpha=0.25)
    h_c, l_c = ax.get_legend_handles_labels()
    h_t, l_t = ax_t.get_legend_handles_labels()
    fig.tight_layout(rect=(0.0, 0.20, 1.0, 1.0))
    fig.legend(
        [h_c[0], h_c[1], h_c[2], h_c[3], h_t[0]],
        [l_c[0], l_c[1], l_c[2], l_c[3], l_t[0]],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        bbox_transform=fig.transFigure,
        ncol=2,
        frameon=True,
        framealpha=0.95,
        fontsize=8,
        handlelength=1.5,
        handletextpad=0.5,
        columnspacing=1.2,
        labelspacing=0.3,
        borderpad=0.35,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}", flush=True)
    print(f"wrote {OUT.with_suffix('.png')}", flush=True)


def main() -> None:
    r_mask, r_val = load_or_compute_scans()
    mismatches, negatives = check_against_table(r_mask, r_val)
    if negatives:
        print("\nNEGATIVE PEAKS — not plotting.", flush=True)
        for line in negatives:
            print(f"  {line}", flush=True)
        raise SystemExit(1)
    print("scans are signed Pearson r; every Table 3.2 peak is positive.", flush=True)
    if mismatches:
        print("POSITION MISMATCHES:", flush=True)
        for line in mismatches:
            print(f"  {line}", flush=True)
    else:
        print("every in-window peak matches Table 3.2.", flush=True)
    plot_figure(r_mask, r_val)
    print(f"x-range = {BYTE2_WINDOW[0]}–{BYTE2_WINDOW[1]}  trace_index=0", flush=True)


if __name__ == "__main__":
    main()
