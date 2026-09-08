#!/usr/bin/env python3
"""Full 16-byte POI correlation scan, matching the scan that produced the
thesis byte-2 / byte-3 numbers.

Those numbers are the in-window argmax |r| from
  magisterka/window_diagnostic.py  (N=4000, first traces)
  magisterka/results/window_diagnostic/leakage_vs_window.csv
quoted again in magisterka/joint_leakage_check.py.

They are NOT the SubBytes-band ([16000:54000]) global argmax from the same
script — that band's peaks (byte 2: 47007 / 46594) do not match the thesis.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import sys as _casca_sys
_CASCA_ROOT = Path(__file__).resolve().parents[1]
if not (_CASCA_ROOT / 'src' / 'casca').is_dir():
    _CASCA_ROOT = Path(__file__).resolve().parents[3]
_casca_sys.path.insert(0, str(_CASCA_ROOT / "src"))
from casca.paths import raw_traces_path, repo_root, window_cache_path  # noqa: E402

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from casca.data.ascad import sbox  # noqa: E402
from casca.data.ascad import ASCAD_KEY  # noqa: E402
from casca.data.windows import BYTE_WINDOWS, MASKED_BYTES, UNMASKED_BYTES  # noqa: E402

HW_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

RAW = raw_traces_path()
PRIOR_CSV = Path(__file__).resolve().parents[1] / "results" / "window_diagnostic" / "leakage_vs_window.csv"
OUT = Path(__file__).resolve().parents[1] / "results" / "poi_full16_scan"

N_TRACES = 4000
# Per-byte search is that byte's 700-sample diagnostic window. The unused
# SubBytes-band from the same script is recorded only to flag the mismatch.
SUB_BAND = (16000, 54000)
PROCESSING_ORDER = (15, 12, 13, 1, 8, 10, 0, 3, 7, 6, 9, 5, 11, 2, 4, 14)

QUOTED = {
    2: dict(mask_t=45556, mask_rho=0.63, val_t=45917, val_rho=0.84),
    3: dict(mask_t=33063, mask_rho=0.62, val_t=33420, val_rho=0.86),
}


def peak_abs(r: np.ndarray, lo: int, hi: int) -> tuple[int, float]:
    sl = r[lo:hi]
    i = int(np.abs(sl).argmax())
    return lo + i, float(sl[i])


def pearson_1d(traces: np.ndarray, hyp: np.ndarray) -> np.ndarray:
    """Pearson r vs every sample. Same formula as cpa_clean_reimpl.pearson_matrix."""
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


def load_prior() -> pd.DataFrame | None:
    if not PRIOR_CSV.exists():
        print(f"[poi] prior CSV missing ({PRIOR_CSV}); scanning all 16 bytes", flush=True)
        return None
    return pd.read_csv(PRIOR_CSV)


def confirm_quoted(prior: pd.DataFrame) -> list[str]:
    flags = []
    print("=== STEP 0  existing methodology ===")
    print(f"  source script   = magisterka/window_diagnostic.py")
    print(f"  source results  = {PRIOR_CSV}")
    print(f"  N               = {N_TRACES}  (first traces in file order)")
    print(
        "  search window   = each byte's own 700-sample diagnostic crop "
        f"(BYTE_WINDOWS); NOT the SubBytes band {SUB_BAND}"
    )
    print("  key / masks     = raw HDF5 metadata ground truth")
    print("  hypotheses      = HW(mask r[b]), HW(Sbox(p⊕k) ⊕ r[b])")
    print("  distinguisher   = Pearson r (CPA formula), peak = argmax |r|")
    print()
    print("  byte 2 / 3 from the existing CSV (in-window columns):")
    for b, q in QUOTED.items():
        row = prior.loc[prior.byte_index == b].iloc[0]
        t_m = int(row.mask_peak_in_window_t)
        t_v = int(row.masked_value_peak_in_window_t)
        r_m = float(row.mask_peak_in_window_rho)
        r_v = float(row.masked_value_peak_in_window_rho)
        ok_t = t_m == q["mask_t"] and t_v == q["val_t"]
        ok_r = round(r_m, 2) == q["mask_rho"] and round(r_v, 2) == q["val_rho"]
        status = "MATCH" if ok_t and ok_r else "MISMATCH"
        print(
            f"    byte {b}: mask {t_m} ρ={r_m:.3f}  val {t_v} ρ={r_v:.3f}  "
            f"thesis {q['mask_t']}/{q['mask_rho']}, {q['val_t']}/{q['val_rho']}  [{status}]"
        )
        if not ok_t or not ok_r:
            flags.append(f"byte {b} in-window peak does not match thesis quote")
        band_m = int(row.mask_peak_location)
        band_v = int(row.masked_value_peak_location)
        print(
            f"             (SubBytes-band argmax was {band_m} / {band_v} — "
            "NOT the thesis numbers)"
        )
    if flags:
        print("  ** DISCREPANCY ** " + "; ".join(flags))
    else:
        print("  byte 2 and byte 3 reproduce the thesis quotes exactly (positions;")
        print("  ρ matches at the two-decimal rounding used in the text).")
    return flags


def scan_byte(
    traces_win: np.ndarray,
    plaintext: np.ndarray,
    key: np.ndarray,
    masks: np.ndarray,
    byte: int,
) -> dict:
    lo, hi = BYTE_WINDOWS[byte]
    pt = plaintext[:, byte].astype(np.int64)
    kb = int(key[byte])
    sb = sbox((pt ^ kb) & 0xFF)
    if byte in MASKED_BYTES:
        m = masks[:, byte - 2].astype(np.int64) & 0xFF
        hw_m = HW_LUT[m].astype(np.float64)
        hw_v = HW_LUT[(sb.astype(np.int64) ^ m) & 0xFF].astype(np.float64)
        r_m = pearson_1d(traces_win, hw_m)
        r_v = pearson_1d(traces_win, hw_v)
        # traces_win is already the crop; peaks are offsets into the crop
        i_m = int(np.abs(r_m).argmax())
        i_v = int(np.abs(r_v).argmax())
        return {
            "byte": byte,
            "masked": True,
            "window_start": lo,
            "window_end": hi,
            "mask_point_position": lo + i_m,
            "rho_mask": float(r_m[i_m]),
            "maskedvalue_point_position": lo + i_v,
            "rho_maskedvalue": float(r_v[i_v]),
            "source": "this_run",
        }
    # Unmasked: mask is the constant 0. HW(0) has zero variance → no peak.
    # Masked-value hypothesis collapses to HW(Sbox) (r = 0).
    hw_v = HW_LUT[sb].astype(np.float64)
    r_v = pearson_1d(traces_win, hw_v)
    i_v = int(np.abs(r_v).argmax())
    return {
        "byte": byte,
        "masked": False,
        "window_start": lo,
        "window_end": hi,
        "mask_point_position": np.nan,
        "rho_mask": np.nan,
        "maskedvalue_point_position": lo + i_v,
        "rho_maskedvalue": float(r_v[i_v]),
        "source": "this_run_unmasked",
    }


def row_from_prior(prior: pd.DataFrame, byte: int) -> dict:
    row = prior.loc[prior.byte_index == byte].iloc[0]
    lo, hi = BYTE_WINDOWS[byte]
    return {
        "byte": byte,
        "masked": True,
        "window_start": lo,
        "window_end": hi,
        "mask_point_position": int(row.mask_peak_in_window_t),
        "rho_mask": float(row.mask_peak_in_window_rho),
        "maskedvalue_point_position": int(row.masked_value_peak_in_window_t),
        "rho_maskedvalue": float(row.masked_value_peak_in_window_rho),
        "source": "reused_window_diagnostic_n4000",
    }


def spacing_stat(table: pd.DataFrame) -> dict:
    pos = {
        int(r.byte): int(r.maskedvalue_point_position)
        for _, r in table.iterrows()
    }
    ordered = [pos[b] for b in PROCESSING_ORDER]
    diffs = np.diff(np.asarray(ordered, dtype=np.int64)).astype(np.float64)
    stat = {
        "processing_order": list(PROCESSING_ORDER),
        "ordered_maskedvalue_positions": ordered,
        "consecutive_differences": [int(d) for d in diffs],
        "n_diffs": int(len(diffs)),
        "mean": float(diffs.mean()),
        "std_sample": float(diffs.std(ddof=1)),
        "std_population": float(diffs.std(ddof=0)),
        "min": int(diffs.min()),
        "max": int(diffs.max()),
    }
    return stat


def thesis_sentence(stat: dict) -> str:
    return (
        f"{stat['mean']:.1f} samples between consecutively-processed bytes "
        f"(std {stat['std_sample']:.1f}, range {stat['min']}-{stat['max']})"
    )


def write_outputs(table: pd.DataFrame, stat: dict, flags: list[str], key_ok: bool) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "poi_scan_table.csv"
    export = table[
        [
            "byte",
            "mask_point_position",
            "rho_mask",
            "maskedvalue_point_position",
            "rho_maskedvalue",
        ]
    ].copy()
    export["byte"] = export["byte"].astype(int)
    export["maskedvalue_point_position"] = export["maskedvalue_point_position"].astype(int)
    export["mask_point_position"] = export["mask_point_position"].astype("Int64")
    export.to_csv(csv_path, index=False, na_rep="")
    (OUT / "protocol.json").write_text(
        json.dumps(
            {
                "N": N_TRACES,
                "search_window": "per-byte 700-sample diagnostic crop (BYTE_WINDOWS)",
                "subbytes_band_not_used_for_peaks": list(SUB_BAND),
                "why_not_the_band": (
                    "argmax |r| in [16000:54000] is 47007/46594 for byte 2; "
                    "thesis quotes the in-window peaks 45556/45917"
                ),
                "key": "HDF5 metadata ground truth (ASCAD fixed key)",
                "masks": "HDF5 metadata; masks[:, b-2] for byte b in 2..15",
                "pearson": "cpa_clean_reimpl.src.cpa.pearson_matrix",
                "byte_2_3": "reused from leakage_vs_window.csv",
                "quoted_match": not flags,
                "key_matches_published_ascad": key_ok,
            },
            indent=2,
        )
        + "\n"
    )
    (OUT / "spacing.json").write_text(json.dumps(stat, indent=2) + "\n")

    latex_rows = []
    for _, r in export.iterrows():
        b = int(r.byte)
        if pd.isna(r.mask_point_position):
            latex_rows.append(
                f"{b} & --- & --- & {int(r.maskedvalue_point_position)} & "
                f"{r.rho_maskedvalue:.2f} \\\\"
            )
        else:
            latex_rows.append(
                f"{b} & {int(r.mask_point_position)} & {r.rho_mask:.2f} & "
                f"{int(r.maskedvalue_point_position)} & {r.rho_maskedvalue:.2f} \\\\"
            )

    match_note = (
        "Byte 2 and byte 3 in-window peaks from the existing N=4000 scan "
        "reproduce the thesis quotes exactly (45556 / 0.63 and 45917 / 0.84; "
        "33063 / 0.62 and 33420 / 0.86)."
        if not flags
        else "DISCREPANCY: " + "; ".join(flags)
    )
    md = "\n".join(
        [
            "# Full 16-byte POI correlation scan",
            "",
            match_note,
            "",
            "## Methodology (confirmed against the existing byte 2/3 numbers)",
            "",
            f"- **N = {N_TRACES}** (first traces, file order).",
            "- **Search window:** each byte's own 700-sample diagnostic crop "
            "(`casca_clean_reimpl/src/windows.py`). Peak = argmax |Pearson r| "
            "inside that crop.",
            f"- The same script also searched the SubBytes band `{SUB_BAND}`; "
            "those global argmax locations do **not** match the thesis and "
            "were not used.",
            "- Hypotheses: `HW(mask)` and `HW(Sbox(p ⊕ k) ⊕ mask)`, true key "
            "and true mask from the raw HDF5 metadata.",
            "- Pearson formula: `cpa_clean_reimpl` vectorized Pearson "
            "(same as this project's CPA).",
            "- Bytes 2 and 3: **reused** from "
            "`magisterka/results/window_diagnostic/leakage_vs_window.csv`.",
            "- Bytes 0 and 1 are unmasked (mask fixed at 0). Mask-point is "
            "undefined (zero-variance HW). Masked-value point is the "
            "in-window `HW(Sbox)` peak, which is what the product "
            "hypothesis collapses to.",
            "",
            "## Table (numerical byte order)",
            "",
            "| byte | mask_point_position | rho_mask | "
            "maskedvalue_point_position | rho_maskedvalue |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for _, r in export.iterrows():
        if pd.isna(r.mask_point_position):
            md += (
                f"\n| {int(r.byte)} | --- | --- | "
                f"{int(r.maskedvalue_point_position)} | {r.rho_maskedvalue:.3f} |"
            )
        else:
            md += (
                f"\n| {int(r.byte)} | {int(r.mask_point_position)} | "
                f"{r.rho_mask:.3f} | {int(r.maskedvalue_point_position)} | "
                f"{r.rho_maskedvalue:.3f} |"
            )
    md += "\n\n### LaTeX rows for Table~\\ref{tab:poi-scan}\n\n```\n"
    md += "\n".join(latex_rows) + "\n```\n"
    md += "\n## Spacing (masked-value points, processing order)\n\n"
    md += f"Order: `{PROCESSING_ORDER}`\n\n"
    md += (
        "Ordered masked-value positions: "
        + ", ".join(str(p) for p in stat["ordered_maskedvalue_positions"])
        + "\n\n"
    )
    md += (
        "Consecutive differences: "
        + ", ".join(str(d) for d in stat["consecutive_differences"])
        + "\n\n"
    )
    md += (
        f"- mean = **{stat['mean']:.3f}**\n"
        f"- std (sample, ddof=1) = **{stat['std_sample']:.3f}**\n"
        f"- min = **{stat['min']}**, max = **{stat['max']}**\n\n"
        "Thesis placeholder:\n\n"
        f"> {thesis_sentence(stat)}\n"
    )
    (OUT / "REPORT.md").write_text(md + "\n")
    print(f"\n  wrote {csv_path}")
    print(f"  wrote {OUT / 'REPORT.md'}")
    print(f"  thesis N placeholder     : {N_TRACES}")
    print(f"  thesis spacing sentence  : {thesis_sentence(stat)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full 16-byte POI correlation scan.")
    p.add_argument("--prior-csv", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    global PRIOR_CSV
    if args.prior_csv is not None:
        PRIOR_CSV = args.prior_csv
    prior = load_prior()
    flags = confirm_quoted(prior) if prior is not None else []

    print("\n=== STEP 1  full 16-byte table ===")
    rows = {}
    for b in MASKED_BYTES:
        if prior is not None and b in QUOTED:
            rows[b] = row_from_prior(prior, b)
            print(
                f"  byte {b:2d}: REUSED  mask {rows[b]['mask_point_position']} "
                f"ρ={rows[b]['rho_mask']:+.3f}  val "
                f"{rows[b]['maskedvalue_point_position']} "
                f"ρ={rows[b]['rho_maskedvalue']:+.3f}",
                flush=True,
            )

    need = [b for b in range(16) if b not in rows]
    print(f"  loading N={N_TRACES} traces for bytes {need} ...", flush=True)
    with h5py.File(RAW, "r") as f:
        meta = f["metadata"]
        pt = meta["plaintext"][:N_TRACES].astype(np.int64)
        key = meta["key"][0].astype(np.int64)
        masks = meta["masks"][:N_TRACES].astype(np.int64)
        key_ok = bool(np.array_equal(key, ASCAD_KEY))
        print(f"  key == published ASCAD_KEY : {key_ok}  {key.tolist()}", flush=True)
        if not key_ok:
            flags.append("HDF5 key does not match published ASCAD_KEY")
        for b in need:
            lo, hi = BYTE_WINDOWS[b]
            win = f["traces"][:N_TRACES, lo:hi].astype(np.float64)
            rec = scan_byte(win, pt, key, masks, b)
            if prior is not None and b in MASKED_BYTES:
                old = row_from_prior(prior, b)
                if (
                    int(rec["mask_point_position"]) != int(old["mask_point_position"])
                    or int(rec["maskedvalue_point_position"])
                    != int(old["maskedvalue_point_position"])
                ):
                    flags.append(f"byte {b} recomputed peak != prior CSV")
                    print(
                        f"  ** byte {b} position mismatch vs prior CSV "
                        f"(mask {old['mask_point_position']} val "
                        f"{old['maskedvalue_point_position']})",
                        flush=True,
                    )
            rows[b] = rec
            print(
                f"  byte {b:2d}: NEW     mask {rec['mask_point_position']} "
                f"ρ={rec['rho_mask'] if rec['rho_mask'] == rec['rho_mask'] else float('nan'):+.3f}  "
                f"val {rec['maskedvalue_point_position']} "
                f"ρ={rec['rho_maskedvalue']:+.3f}  window [{lo}:{hi}]",
                flush=True,
            )

    table = pd.DataFrame([rows[b] for b in range(16)])
    print("\n=== STEP 2  spacing (masked-value points, processing order) ===")
    stat = spacing_stat(table)
    print(f"  order     = {PROCESSING_ORDER}")
    print(f"  positions = {stat['ordered_maskedvalue_positions']}")
    print(f"  diffs     = {stat['consecutive_differences']}")
    print(
        f"  mean={stat['mean']:.3f}  std={stat['std_sample']:.3f}  "
        f"range={stat['min']}-{stat['max']}"
    )
    print(f"  sentence  = {thesis_sentence(stat)}")
    write_outputs(table, stat, flags, key_ok)


if __name__ == "__main__":
    main()
