#!/usr/bin/env python3
"""16-byte MOC attack, one independent network per byte window.

CNN/Config E train ONE concatenated model on the 14 masked bytes. MOC cannot
do that: each byte has its own 256 key-byte hypotheses and LSB labels. The
fair 16-byte protocol is therefore:

    for each byte b in 0..15:
        train MOC on that byte's 700-sample window (first N traces)
        rank = argmax of the 256 branch training accuracies

Same N/seed trial definition as CNN Algorithm 3 (non-profiled, first-N
profiling traces). Architecture and training protocol are unchanged from
the byte-2 validation.

Default grid: 5 seeds × {2000, 5000, 8000, 10000, 12000, 20000}.
That matches the CNN comparison table in the region where CNN is still
incomplete, without a 10-seed × 60k run. Resume-safe.
"""

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
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from casca.data.windows import ALL_BYTES, MASKED_BYTES, UNMASKED_BYTES  # noqa: E402
from casca.data.ascad import cache_per_byte_windows, load_per_byte_windows  # noqa: E402
from casca.plotting import plot_ge_per_byte, plot_mean_recovered_vs_n  # noqa: E402
from casca.attacks.moc import lsb_onehot, mean_remove_scale_pm1  # noqa: E402
from casca.attacks.moc import MOC_BATCH_SIZE, MOC_EPOCHS, train_moc  # noqa: E402

from casca.attacks.clustering import rank_of_key  # noqa: E402
from casca.runner import summarize  # noqa: E402

RAW = raw_traces_path()
CACHE = window_cache_path()
OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "moc_full16"
CNN_DIR = Path(__file__).resolve().parents[1] / "results" / "clean_reimpl_full16"
# 5-seed pilot covering the CNN table's incomplete-recovery region.
DEFAULT_N = (2000, 5000, 8000, 10000, 12000, 20000)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)


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


def run_one_byte(
    traces_raw: np.ndarray,
    plaintext_byte: np.ndarray,
    true_key: int,
    seed: int,
    n_epochs: int,
    batch_size: int,
) -> tuple[int, float, float, float, np.ndarray, float]:
    traces, = mean_remove_scale_pm1(traces_raw)
    onehot = lsb_onehot(plaintext_byte)
    result = train_moc(
        traces,
        onehot,
        true_key=true_key,
        seed=seed,
        n_epochs=n_epochs,
        batch_size=batch_size,
        log_every=n_epochs,
        verbose=False,
    )
    acc = result.acc_per_hyp_final
    assert acc is not None
    rank = int(rank_of_key(acc, true_key))
    wrong = np.delete(acc, true_key)
    seconds = float(result.seconds)
    del result
    return rank, float(acc[true_key]), float(wrong.mean()), float(wrong.std(ddof=1)), acc, seconds


def run_one(
    windows: np.ndarray,
    plaintext: np.ndarray,
    true_key: np.ndarray,
    seed: int,
    n: int,
    n_epochs: int,
    batch_size: int,
    out_dir: Path,
) -> dict:
    t0 = time.time()
    ranks = []
    true_acc = []
    wrong_mean = []
    wrong_std = []
    byte_seconds = []
    acc_stack = np.empty((16, 256), dtype=np.float64)
    for b in ALL_BYTES:
        rank, acc_t, acc_w, acc_s, acc, dt = run_one_byte(
            windows[:n, b],
            plaintext[:n, b],
            int(true_key[b]),
            seed,
            n_epochs,
            batch_size,
        )
        ranks.append(rank)
        true_acc.append(acc_t)
        wrong_mean.append(acc_w)
        wrong_std.append(acc_s)
        byte_seconds.append(dt)
        acc_stack[b] = acc
        tag = "  (unmasked)" if b in UNMASKED_BYTES else ""
        print(
            f"  byte {b:2d}  rank={rank:3d}  acc_true={acc_t:.4f}  "
            f"acc_wrong={acc_w:.4f}±{acc_s:.4f}  argmax=0x{int(acc.argmax()):02x}  "
            f"{dt:.1f}s{tag}",
            flush=True,
        )
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    rec = {
        "method": "MOC",
        "protocol": "independent_per_byte",
        "seed": int(seed),
        "N": int(n),
        "true_key": [int(k) for k in true_key],
        "ranks": ranks,
        "true_key_acc": true_acc,
        "wrong_acc_mean": wrong_mean,
        "wrong_acc_std": wrong_std,
        "byte_train_seconds": byte_seconds,
        "n_params": 180932,
        "train_seconds": float(sum(byte_seconds)),
        "key_recovery_seconds": 0.0,
        "elapsed_seconds": time.time() - t0,
        "masked_recovered": int(sum(r == 0 for i, r in enumerate(ranks) if i in MASKED_BYTES)),
        "all_recovered": int(sum(r == 0 for r in ranks)),
        "full_key_rank0": bool(all(r == 0 for r in ranks)),
        "shared_width_unconfirmed": True,
        "epochs": int(n_epochs),
        "batch": int(batch_size),
    }
    stem = out_dir / "runs" / f"moc_seed{seed}_N{n}"
    stem.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        stem.with_suffix(".npz"),
        acc_per_hyp=acc_stack,
        ranks=np.asarray(ranks, dtype=np.int16),
    )
    rec["acc_vector_file"] = str(stem.with_suffix(".npz"))
    stem.with_suffix(".json").write_text(json.dumps(_jsonable(rec), indent=2))
    print(
        f"[full16] seed={seed} N={n} recovered {rec['all_recovered']}/16 "
        f"(masked {rec['masked_recovered']}/14)  train={rec['train_seconds']:.1f}s",
        flush=True,
    )
    return rec


def cnn_agg(seeds: tuple[int, ...]) -> list[dict]:
    rows = []
    if not (CNN_DIR / "runs").is_dir():
        return rows
    by_n: dict[int, list[dict]] = {}
    for p in sorted((CNN_DIR / "runs").glob("seed*_N*.json")):
        rec = json.loads(p.read_text())
        if int(rec["seed"]) not in seeds:
            continue
        by_n.setdefault(int(rec["N"]), []).append(rec)
    masked = list(MASKED_BYTES)
    for n in sorted(by_n):
        subset = by_n[n]
        ranks = np.array([r["ranks"] for r in subset], dtype=int)
        rec = np.sum(ranks == 0, axis=1).astype(float)
        rows.append(
            {
                "method": "CNN",
                "N": n,
                "n_trials": len(subset),
                "mean_bytes_recovered": float(rec.mean()),
                "std_bytes_recovered": float(rec.std(ddof=1) if len(subset) > 1 else 0.0),
                "SR1_fullkey": float(np.mean(np.all(ranks == 0, axis=1))),
                "GE_masked14": float(ranks[:, masked].mean()),
                "per_seed_recovered": rec.astype(int).tolist(),
            }
        )
    return rows


def write_comparison(moc_runs: list[dict], seeds: tuple[int, ...], out_dir: Path) -> list[dict]:
    cnn_rows = cnn_agg(seeds)
    cnn_by_n = {r["N"]: r for r in cnn_rows}
    moc_by_n: dict[int, list[dict]] = {}
    for r in moc_runs:
        moc_by_n.setdefault(int(r["N"]), []).append(r)
    masked = list(MASKED_BYTES)
    rows = []
    for n in sorted(moc_by_n):
        subset = moc_by_n[n]
        ranks = np.array([r["ranks"] for r in subset], dtype=int)
        rec = np.sum(ranks == 0, axis=1).astype(float)
        moc = {
            "method": "MOC",
            "N": n,
            "n_trials": len(subset),
            "mean_bytes_recovered": float(rec.mean()),
            "std_bytes_recovered": float(rec.std(ddof=1) if len(subset) > 1 else 0.0),
            "SR1_fullkey": float(np.mean(np.all(ranks == 0, axis=1))),
            "GE_masked14": float(ranks[:, masked].mean()),
            "per_seed_recovered": rec.astype(int).tolist(),
            "mean_train_seconds": float(np.mean([r["train_seconds"] for r in subset])),
        }
        c = cnn_by_n.get(n)
        rows.append(
            {
                **{f"MOC_{k}": v for k, v in moc.items() if k != "method"},
                "N": n,
                "GE_masked14_MOC": moc["GE_masked14"],
                "GE_masked14_CNN": None if c is None else c["GE_masked14"],
                "recovered_MOC": moc["mean_bytes_recovered"],
                "recovered_CNN": None if c is None else c["mean_bytes_recovered"],
                "SR1_MOC": moc["SR1_fullkey"],
                "SR1_CNN": None if c is None else c["SR1_fullkey"],
                "std_recovered_MOC": moc["std_bytes_recovered"],
                "std_recovered_CNN": None if c is None else c["std_bytes_recovered"],
                "per_seed_MOC": moc["per_seed_recovered"],
                "per_seed_CNN": None if c is None else c["per_seed_recovered"],
                "n_trials_MOC": moc["n_trials"],
                "n_trials_CNN": None if c is None else c["n_trials"],
                "mean_train_seconds_MOC": moc["mean_train_seconds"],
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "comparison_vs_cnn.csv", index=False)
    (out_dir / "comparison_vs_cnn.json").write_text(json.dumps(_jsonable(rows), indent=2))

    moc_series_n = [r["N"] for r in rows]
    moc_mean = [r["recovered_MOC"] for r in rows]
    moc_std = [r["std_recovered_MOC"] for r in rows]
    cnn_n = [r["N"] for r in cnn_rows]
    cnn_mean = [r["mean_bytes_recovered"] for r in cnn_rows]
    cnn_std = [r["std_bytes_recovered"] for r in cnn_rows]
    plot_mean_recovered_vs_n(
        [
            (f"MOC ({len(seeds)}-seed, per-byte)", moc_series_n, moc_mean, moc_std),
            (f"CNN ({len(seeds)}-seed, concat-14)", cnn_n, cnn_mean, cnn_std),
        ],
        out_dir / "mean_recovered_vs_n.png",
        title="16-byte mean recovered vs N  [MOC vs CNN, same seeds]",
    )
    return rows


def write_verdict(moc_runs: list[dict], cmp_rows: list[dict], wall: float, out_dir: Path) -> None:
    by_n = {int(r["N"]): r for r in cmp_rows}
    ns = sorted(by_n)
    lines = [
        "# MOC 16-byte validation",
        "",
        "Independent non-profiled trials. **One MOC network per byte** "
        "(not concatenated). Same 5 seeds as the byte-2 gate. CNN numbers "
        "below are the same seeds from `clean_reimpl_full16` (concatenated "
        "14-byte CNN + CH), for a same-seed comparison — not a claim that "
        "the two protocols are identical.",
        "",
        "## Mean bytes recovered",
        "",
        "| N | MOC recovered | CNN recovered | SR1 MOC | SR1 CNN | GE masked-14 MOC | GE masked-14 CNN | MOC train s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n in ns:
        r = by_n[n]
        cnn_rec = "—" if r["recovered_CNN"] is None else f"{r['recovered_CNN']:.2f} ± {r['std_recovered_CNN']:.2f}"
        cnn_sr = "—" if r["SR1_CNN"] is None else f"{r['SR1_CNN']:.2f}"
        cnn_ge = "—" if r["GE_masked14_CNN"] is None else f"{r['GE_masked14_CNN']:.1f}"
        lines.append(
            f"| {n} | {r['recovered_MOC']:.2f} ± {r['std_recovered_MOC']:.2f} "
            f"({r['per_seed_MOC']}) | {cnn_rec} | {r['SR1_MOC']:.2f} | {cnn_sr} | "
            f"{r['GE_masked14_MOC']:.1f} | {cnn_ge} | {r['mean_train_seconds_MOC']:.0f} |"
        )

    last = by_n[ns[-1]]
    first = by_n[ns[0]]
    improving = last["recovered_MOC"] >= first["recovered_MOC"]
    not_stuck = last["recovered_MOC"] > 2.1  # more than the two unmasked bytes
    if improving and not_stuck:
        verdict = "PASS"
        follow = (
            "16-byte MOC behaves sensibly (recovers more than the unmasked "
            "pair, and more traces help). Ready to extend the N grid or seed "
            "count if a fuller comparison with CNN's 10-seed × 60k table is needed."
        )
    else:
        verdict = "FAIL"
        follow = (
            "16-byte MOC did not show a clear improvement over recovering "
            "only the unmasked bytes. Do not treat the byte-2 win as a "
            "16-byte result."
        )
    lines[2:2] = [f"**Verdict: {verdict}.**", ""]
    lines += [
        "",
        "## Notes",
        "",
        "- MOC trains 16 separate networks; CNN trains one concatenated model.",
        "- Unmasked ASCAD bytes 0 and 1 should be easy for both.",
        f"- Wall-clock for this run: **{wall:.1f}s** ({wall / 60.0:.2f} min).",
        "- Shared-layer width=20 remains an unconfirmed default.",
        "",
        follow,
        "",
    ]
    (out_dir / "VERDICT.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--N", type=int, nargs="+", default=list(DEFAULT_N))
    p.add_argument("--epochs", type=int, default=MOC_EPOCHS)
    p.add_argument("--batch", type=int, default=MOC_BATCH_SIZE)
    p.add_argument("--out", type=Path, default=OUT_DIR)
    p.add_argument("--smoke", action="store_true", help="1 seed, N=2000, 2 epochs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.seeds = [0]
        args.N = [2000]
        args.epochs = 2
        args.out = args.out / "smoke"

    if not CACHE.exists():
        if not RAW.exists():
            raise FileNotFoundError(f"Raw traces missing: {RAW}")
        cache_per_byte_windows(RAW, CACHE)
    data = load_per_byte_windows(CACHE)
    windows, plaintext, true_key = data["windows"], data["plaintext"], data["true_key"]
    print(
        f"[full16] windows={windows.shape}  key={true_key.tolist()}\n"
        f"         method=MOC independent-per-byte  epochs={args.epochs}  "
        f"batch={args.batch}  seeds={args.seeds}  N={args.N}",
        flush=True,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    seeds = tuple(args.seeds)
    ns = tuple(args.N)
    total = len(seeds) * len(ns)
    t_all = time.time()
    i = 0
    for n in ns:
        for seed in seeds:
            i += 1
            json_path = args.out / "runs" / f"moc_seed{seed}_N{n}.json"
            if json_path.exists() and not args.smoke:
                rec = json.loads(json_path.read_text())
                print(
                    f"[full16] resume skip {i}/{total} seed={seed} N={n} "
                    f"recovered={rec['all_recovered']}/16",
                    flush=True,
                )
                continue
            print(f"\n======== MOC full16  {i}/{total}  seed={seed} N={n} ========", flush=True)
            rec = run_one(
                windows, plaintext, true_key, seed, n, args.epochs, args.batch, args.out
            )
            if i == 1:
                est = rec["elapsed_seconds"] * total
                print(
                    f"[timing] first cell {rec['elapsed_seconds']:.1f}s  "
                    f"projected {total} cells ≈ {est / 60.0:.1f} min",
                    flush=True,
                )

    all_on_disk = []
    for p in sorted((args.out / "runs").glob("moc_seed*_N*.json")):
        all_on_disk.append(json.loads(p.read_text()))
    if all_on_disk:
        summarize(all_on_disk, args.out)
        per_byte = json.loads((args.out / "ge_sr_per_byte.json").read_text())
        plot_ge_per_byte(per_byte, UNMASKED_BYTES, args.out / "ge_per_byte.png")
        cmp_rows = write_comparison(all_on_disk, seeds, args.out)
        wall = time.time() - t_all
        write_verdict(all_on_disk, cmp_rows, wall, args.out)
    wall = time.time() - t_all
    meta = {
        "method": "MOC",
        "protocol": "independent_per_byte",
        "seeds": list(seeds),
        "N": list(ns),
        "epochs": args.epochs,
        "batch": args.batch,
        "elapsed_seconds": wall,
        "shared_width_unconfirmed": True,
        "note": (
            "Each (seed, N) trains 16 independent MOC networks, one per byte "
            "window. Not the CNN concatenated-14 protocol. GE/SR via "
            "casca_clean_reimpl concat_attack.summarize."
        ),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[full16] finished in {wall:.1f}s  outputs in {args.out}", flush=True)


if __name__ == "__main__":
    main()
