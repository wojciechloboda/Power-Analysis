#!/usr/bin/env python3
"""M=10 step-checkpointed training-budget sweep: CA-SCA vs AA-CASCA.

Step 0 convention (confirmed from train_plaintext_cnn_fast):
  bs = min(20000, n_windows); for i in range(0, n_windows, bs); no drop_last.
  steps_per_epoch = ceil(14N / min(20000, 14N)).

Both architectures share the same cumulative-step checkpoint schedule
through 1000 updates. CA-SCA continues to its published 200-epoch
equivalent (200 × steps_per_epoch) so the full-budget claim can be
tested under M=10. Architectures are unchanged; only the trainer's
optional step-checkpoint callback is used.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import sys as _casca_sys
_CASCA_ROOT = Path(__file__).resolve().parents[3]
if not (_CASCA_ROOT / 'src' / 'casca').is_dir():
    _CASCA_ROOT = Path(__file__).resolve().parents[3]
_casca_sys.path.insert(0, str(_CASCA_ROOT / "src"))
from casca.paths import raw_traces_path, repo_root, window_cache_path  # noqa: E402

import numpy as np
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from casca.attacks.casca import Cascacnn
from casca.attacks.clustering import multibit_ch_vector, rank_of_key
from casca.runner import concatenate_train_set
from casca.data.ascad import cache_per_byte_windows, load_per_byte_windows, zscore_train_apply
from casca.attacks.aacasca import CONV_CONFIGS, make_h2
from casca.metrics import guessing_entropy, success_rate
from casca.attacks.casca import (
    PAPER_BATCH_SIZE,
    PAPER_EPOCHS,
    PAPER_LR,
    _device,
    extract_features,
    train_plaintext_cnn_fast,
)
from casca.data.windows import ALL_BYTES, MASKED_BYTES, TRAIN_BYTES

RAW = raw_traces_path()
CACHE = window_cache_path()
OUT = Path(__file__).resolve().parents[3] / "results" / "training_budget_M10"

ARCHS = ("casca", "aa_casca")
NS = (5000, 10000, 20000, 30000)
SEEDS = tuple(range(10))
BATCH = PAPER_BATCH_SIZE
VAL_FRAC = 0.10
AA_CASCA_MAX_STEPS = 1000

# Common schedule: every 25 steps through 400 (AA-CASCA's previously
# reported 250–700 optimum sits here), then every 50 through 1000.
COMMON_STEPS = tuple(list(range(25, 401, 25)) + list(range(450, 1001, 50)))
# CA-SCA tail out to the 200-epoch equivalent at N=30000 (21×200=4200).
CASCA_TAIL = (1200, 1400, 1600, 1800, 2000, 2400, 2800, 3500, 4200)


def steps_per_epoch(n: int, batch: int = BATCH) -> int:
    """Exact trainer convention: ceil(14N / min(batch, 14N)), no drop_last."""
    n_windows = int(n) * len(TRAIN_BYTES)
    bs = min(int(batch), n_windows)
    return (n_windows + bs - 1) // bs


def paper_steps(n: int) -> int:
    return PAPER_EPOCHS * steps_per_epoch(n)


def max_steps_for(arch: str, n: int) -> int:
    if arch == "casca":
        return paper_steps(n)
    if arch == "aa_casca":
        return AA_CASCA_MAX_STEPS
    raise ValueError(arch)


def schedule_for(arch: str, n: int) -> tuple[int, ...]:
    cap = max_steps_for(arch, n)
    steps = [s for s in COMMON_STEPS if s <= cap]
    if arch == "casca":
        steps += [s for s in CASCA_TAIL if s <= cap]
    if cap not in steps:
        steps.append(cap)
    return tuple(sorted(set(steps)))


def json_path(arch: str, seed: int, n: int) -> Path:
    return OUT / "runs" / f"{arch}_seed{seed}_N{n}.json"


def make_val_concat(windows: np.ndarray, plaintext: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, int]:
    n_val = max(1, int(round(VAL_FRAC * n)))
    stop = n + n_val
    if stop > windows.shape[0]:
        raise SystemExit(f"val slice [{n}:{stop}] exceeds {windows.shape[0]} traces")
    x = np.concatenate([windows[n:stop, b] for b in TRAIN_BYTES], axis=0)
    y = np.concatenate([plaintext[n:stop, b] for b in TRAIN_BYTES], axis=0)
    return x, y, n_val


def score_ch(
    model: nn.Module,
    per_byte_z: dict[int, np.ndarray],
    plaintext: np.ndarray,
    true_key: np.ndarray,
    n: int,
) -> tuple[list[int], np.ndarray]:
    ranks = []
    ch_stack = np.empty((16, 256), dtype=np.float64)
    for b in ALL_BYTES:
        feats = extract_features(model, per_byte_z[b])
        ch = multibit_ch_vector(feats, plaintext[:n, b])
        ch_stack[b] = ch
        ranks.append(rank_of_key(ch, int(true_key[b])))
    return ranks, ch_stack


def aggregate_ch(ch_stack: np.ndarray) -> float:
    """Sum over 256 hypotheses, mean over the 14 masked bytes."""
    return float(ch_stack[list(MASKED_BYTES), :].sum(axis=1).mean())


def model_fn_for(arch: str):
    if arch == "casca":
        return Cascacnn
    if arch == "aa_casca":
        return lambda: make_h2("E")
    raise ValueError(arch)


def run_one(
    windows: np.ndarray,
    plaintext: np.ndarray,
    true_key: np.ndarray,
    arch: str,
    seed: int,
    n: int,
) -> dict:
    t0 = time.time()
    spe = steps_per_epoch(n)
    cap = max_steps_for(arch, n)
    ckpt_steps = schedule_for(arch, n)
    n_epochs_ceiling = max(PAPER_EPOCHS, math.ceil(cap / spe) + 1)

    x_cat, y_cat = concatenate_train_set(windows, plaintext, n)
    x_val, y_val, n_val = make_val_concat(windows, plaintext, n)
    extras = [windows[:n, b] for b in ALL_BYTES] + [x_val]
    scaled = zscore_train_apply(x_cat, *extras)
    x_cat_z = scaled[0]
    per_byte_z = {b: scaled[1 + b] for b in ALL_BYTES}
    x_val_z = scaled[1 + 16]

    tag = "cnn" if arch == "casca" else "h2E"
    checkpoints: list[dict] = []
    ch_by_step: dict[int, np.ndarray] = {}

    def on_step_checkpoint(step: int, snap: dict, model: nn.Module) -> None:
        t_ch = time.time()
        ranks, ch_stack = score_ch(model, per_byte_z, plaintext, true_key, n)
        ch_seconds = time.time() - t_ch
        ch_by_step[step] = ch_stack
        masked = [ranks[b] for b in MASKED_BYTES]
        rec = {
            "architecture": arch,
            "tag": tag,
            "N": n,
            "seed": seed,
            "step": step,
            "epoch": int(snap["epoch"]),
            "ranks": ranks,
            "GE_16": guessing_entropy(ranks),
            "GE_masked14": guessing_entropy(masked),
            "SR1": float(success_rate(ranks, 1)),
            "SR5": float(success_rate(ranks, 5)),
            "SR1_fullkey": float(all(r == 0 for r in ranks)),
            "SR5_fullkey": float(all(r < 5 for r in ranks)),
            "bytes_recovered": int(sum(r == 0 for r in ranks)),
            "masked_recovered": int(sum(r == 0 for r in masked)),
            "agg_ch": aggregate_ch(ch_stack),
            "train_loss": snap["train_loss"],
            "train_acc": snap["train_acc"],
            "train_loss_last": snap.get("train_loss_last", snap["train_loss"]),
            "val_loss": snap["val_loss"],
            "val_acc": snap["val_acc"],
            "cumulative_train_seconds": snap["cumulative_train_seconds"],
            "checkpoint_ch_seconds": ch_seconds,
        }
        checkpoints.append(rec)
        print(
            f"  [{arch}] seed={seed} N={n} step={step:4d} (ep {rec['epoch']})  "
            f"GE16={rec['GE_16']:.2f}  GE14={rec['GE_masked14']:.2f}  "
            f"rec={rec['bytes_recovered']}/16  aggCH={rec['agg_ch']:.1f}  "
            f"tr={rec['train_loss']:.4f} val={rec['val_loss']:.4f}  "
            f"train={rec['cumulative_train_seconds']:.1f}s  ch={ch_seconds:.1f}s",
            flush=True,
        )

    result = train_plaintext_cnn_fast(
        x_cat_z,
        y_cat,
        seed=seed,
        n_epochs=n_epochs_ceiling,
        batch_size=BATCH,
        lr=PAPER_LR,
        model_fn=model_fn_for(arch),
        val_traces=x_val_z,
        val_labels=y_val,
        max_steps=cap,
        checkpoint_steps=set(ckpt_steps),
        on_step_checkpoint=on_step_checkpoint,
    )
    got = [c["step"] for c in checkpoints]
    if got != list(ckpt_steps):
        raise RuntimeError(f"missing checkpoints for {arch} seed={seed} N={n}: got {got} want {list(ckpt_steps)}")
    if result.n_steps != cap:
        raise RuntimeError(f"expected {cap} steps, got {result.n_steps} ({arch} seed={seed} N={n})")

    out = {
        "architecture": arch,
        "tag": tag,
        "seed": seed,
        "N": n,
        "n_params": result.model.n_params(),
        "n_tokens": int(getattr(result.model, "n_tokens", 4 if arch == "casca" else 23)),
        "n_train_windows": int(len(x_cat)),
        "n_val_traces": n_val,
        "n_val_windows": int(len(x_val)),
        "val_trace_slice": [n, n + n_val],
        "steps_per_epoch": spe,
        "max_steps": cap,
        "paper_200ep_steps": paper_steps(n),
        "protocol": {
            "batch_size": BATCH,
            "lr": PAPER_LR,
            "adam_eps": 1e-7,
            "loss": "CrossEntropy",
            "checkpoint_steps": list(ckpt_steps),
            "steps_per_epoch_convention": "ceil(14N / min(20000, 14N)), no drop_last",
        },
        "train_seconds": result.seconds,
        "elapsed_seconds": time.time() - t0,
        "final_loss": result.final_loss,
        "final_acc": result.final_acc,
        "n_steps": result.n_steps,
        "checkpoints": checkpoints,
    }
    stem = json_path(arch, seed, n)
    stem.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        stem.with_suffix(".npz"),
        steps=np.asarray(ckpt_steps, dtype=np.int32),
        ch_vectors=np.stack([ch_by_step[s] for s in ckpt_steps], axis=0),
        ranks=np.asarray([c["ranks"] for c in checkpoints], dtype=np.int16),
    )
    stem.write_text(json.dumps(out, indent=2))
    print(
        f"[{arch}] seed={seed} N={n} wrote {stem.name}  "
        f"train={result.seconds:.1f}s  wall={out['elapsed_seconds']:.1f}s  "
        f"final GE16={checkpoints[-1]['GE_16']:.2f} rec={checkpoints[-1]['bytes_recovered']}/16",
        flush=True,
    )
    del result
    return out


def print_protocol() -> None:
    cnn = Cascacnn()
    e = make_h2("E")
    print("=== training-budget M=10: protocol ===", flush=True)
    print(
        f"  CA-SCA          = Table-2 CNN  params={cnn.n_params()}  "
        f"max_steps = 200 × steps_per_epoch(N)",
        flush=True,
    )
    print(
        f"  AA-CASCA        = Config E  tokens={e.n_tokens} params={e.n_params()} "
        f"strides={CONV_CONFIGS['E'].strides}  max_steps = {AA_CASCA_MAX_STEPS}",
        flush=True,
    )
    print(f"  batch_size      = {BATCH}  (applied to concat-14 windows)", flush=True)
    print(f"  optimizer       = Adam(lr={PAPER_LR}, eps=1e-7)", flush=True)
    print("  loss            = CrossEntropy", flush=True)
    print(f"  labels          = plaintext-byte, concat TRAIN_BYTES={list(TRAIN_BYTES)}", flush=True)
    print(f"  seeds           = {list(SEEDS)}   (fixed M=10 list)", flush=True)
    print(f"  N               = {list(NS)}", flush=True)
    print(
        f"  val split       = traces [N : N+round({VAL_FRAC}*N)], concat TRAIN_BYTES, "
        "z-scored with train stats; not used for gradient steps",
        flush=True,
    )
    print(
        f"  common steps    = every 25 through 400, every 50 through 1000 "
        f"({len(COMMON_STEPS)} points)",
        flush=True,
    )
    print(f"  CA-SCA tail     = {list(CASCA_TAIL)}  (clipped to 200×spe)", flush=True)
    print("  steps/epoch (concat-14, batch min(20000, 14N), no drop_last):", flush=True)
    for n in NS:
        spe = steps_per_epoch(n)
        print(
            f"    N={n:5d}  windows={n * 14:6d}  spe={spe:2d}  "
            f"CA-SCA max={paper_steps(n):4d} ({PAPER_EPOCHS} ep)  "
            f"AA-CASCA max={AA_CASCA_MAX_STEPS:4d}  "
            f"ckpts casca={len(schedule_for('casca', n))}  "
            f"aa={len(schedule_for('aa_casca', n))}",
            flush=True,
        )
    print(
        "  note            = step checkpoints do not change either architecture "
        "or the optimizer updates. Mid-epoch stops are allowed.",
        flush=True,
    )


def protocol_dict(seeds: tuple[int, ...], ns: tuple[int, ...]) -> dict:
    return {
        "architectures": {
            "casca": "Table-2 CNN (CA-SCA), unchanged",
            "aa_casca": "Config E H2Net (AA-CASCA), unchanged",
        },
        "batch_size": BATCH,
        "batch_applied_to": "concat-14 windows (14N)",
        "steps_per_epoch_convention": "ceil(14N / min(20000, 14N)), no drop_last",
        "steps_per_epoch": {str(n): steps_per_epoch(n) for n in ns},
        "paper_200ep_steps": {str(n): paper_steps(n) for n in ns},
        "aa_casca_max_steps": AA_CASCA_MAX_STEPS,
        "lr": PAPER_LR,
        "adam_eps": 1e-7,
        "loss": "CrossEntropy",
        "common_checkpoint_steps": list(COMMON_STEPS),
        "casca_tail_steps": list(CASCA_TAIL),
        "schedules": {
            arch: {str(n): list(schedule_for(arch, n)) for n in ns} for arch in ARCHS
        },
        "val_frac": VAL_FRAC,
        "N": list(ns),
        "seeds": list(seeds),
        "M": len(seeds),
        "GE": "per-byte rank; GE_16 = mean of 16 ranks; GE_masked14 = mean of bytes 2-15",
        "agg_ch": "sum of 256-hypothesis CH, mean over 14 masked bytes",
        "note": (
            "Full M=10 protocol. Step-based checkpoints only. "
            "This supersedes earlier 3–5 seed exploratory training-budget screens."
        ),
    }


def planned_jobs(archs: tuple[str, ...], seeds: tuple[int, ...], ns: tuple[int, ...]) -> list[tuple[str, int, int]]:
    # Longest jobs first so the timing cell finishes early and workers stay busy.
    return [(arch, seed, n) for n in sorted(ns, reverse=True) for arch in archs for seed in seeds]


def write_timing_projection(elapsed_s: float, arch: str, n: int, seed: int) -> None:
    """Scale the timed cell across the full grid using relative step × N weights."""
    timed_steps = max_steps_for(arch, n)
    timed_ckpts = len(schedule_for(arch, n))
    # Attribute ~60% of wall to gradient steps, 40% to CH (matches prior
    # CNN/Config E checkpointed runs). Scale steps by step-count; CH by N × n_ckpts.
    jobs = planned_jobs(ARCHS, SEEDS, NS)
    weights = []
    for a, _s, nn in jobs:
        steps = max_steps_for(a, nn)
        ckpts = len(schedule_for(a, nn))
        # Config E is ~2.6× slower per step than the CNN (epoch-trajectory).
        step_cost = steps * (2.6 if a == "aa_casca" else 1.0)
        ch_cost = ckpts * (nn / n)
        weights.append(0.55 * step_cost / timed_steps + 0.45 * ch_cost / max(timed_ckpts, 1))
    timed_w = 0.55 * 1.0 + 0.45 * 1.0
    # The timed cell is CA-SCA, so its weight uses the CNN multiplier.
    if arch == "casca":
        scale = elapsed_s / timed_w
    else:
        scale = elapsed_s / (0.55 * 2.6 + 0.45)
    total = float(scale * sum(weights))
    per_cell = {f"{a}/N{nn}": None for a in ARCHS for nn in NS}
    # rewrite as means
    cell_sums: dict[str, list[float]] = {}
    for (a, _s, nn), w in zip(jobs, weights):
        cell_sums.setdefault(f"{a}/N{nn}", []).append(scale * w)
    per_cell = {k: float(np.mean(v)) for k, v in cell_sums.items()}
    lines = [
        f"# Timing projection (from {arch} N={n} seed={seed})",
        "",
        f"Timed cell wall-clock: **{elapsed_s:.1f}s** ({elapsed_s / 60:.2f} min).",
        f"Checkpoints in timed cell: {timed_ckpts}. Steps: {timed_steps}.",
        "",
        f"Projected full grid (2 arch × {len(NS)} N × {len(SEEDS)} seeds = {len(jobs)} cells):",
        f"- sequential: **{total / 3600:.2f} h** ({total / 60:.0f} min)",
        f"- 2 workers:  **{total / 2 / 3600:.2f} h** ({total / 2 / 60:.0f} min)",
        "",
        "Per-(architecture, N) mean projected wall (one seed):",
    ]
    for k, v in per_cell.items():
        lines.append(f"- {k}: {v:.0f}s")
    lines.append("")
    lines.append(
        "Scaling: 55% of wall ~ gradient-step count (AA-CASCA ×2.6 per-step vs CA-SCA), "
        "45% ~ checkpoint count × N (CH + val)."
    )
    text = "\n".join(lines) + "\n"
    (OUT / "timing_projection.md").write_text(text)
    print(text, flush=True)


def run_job_list(
    jobs: list[tuple[str, int, int]],
    worker_id: int,
    windows,
    plaintext,
    true_key,
) -> None:
    for i, (arch, seed, n) in enumerate(jobs, start=1):
        dest = json_path(arch, seed, n)
        if dest.exists():
            print(
                f"[budget w{worker_id}] skip {i}/{len(jobs)}  {arch} seed={seed} N={n}",
                flush=True,
            )
            continue
        print(
            f"\n======== budget w{worker_id} {i}/{len(jobs)}  {arch}  "
            f"seed={seed} N={n}  max_steps={max_steps_for(arch, n)}  "
            f"spe={steps_per_epoch(n)}  ckpts={len(schedule_for(arch, n))} ========",
            flush=True,
        )
        run_one(windows, plaintext, true_key, arch, seed, n)


def _worker_entry(jobs: list[tuple[str, int, int]], worker_id: int) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    if str(_device()) != "mps":
        raise SystemExit(f"w{worker_id}: not MPS ({_device()})")
    data = load_per_byte_windows(CACHE)
    print(f"[budget w{worker_id}] {len(jobs)} jobs device=mps", flush=True)
    run_job_list(jobs, worker_id, data["windows"], data["plaintext"], data["true_key"])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--timing-only",
        action="store_true",
        help="Most expensive cell only: CA-SCA N=30000 seed 0, then write projection",
    )
    p.add_argument("--arch", choices=ARCHS, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--N", type=int, nargs="+", default=None)
    p.add_argument("--workers", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print_protocol()
    dev = _device()
    print(f"  device          = {dev}", flush=True)
    print(f"  workers         = {args.workers}", flush=True)
    if str(dev) != "mps":
        raise SystemExit(f"refusing silent CPU/CUDA fallback; device={dev}")

    if not CACHE.exists():
        cache_per_byte_windows(RAW, CACHE)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "runs").mkdir(parents=True, exist_ok=True)

    seeds = tuple(args.seeds) if args.seeds is not None else SEEDS
    ns = tuple(args.N) if args.N is not None else NS
    archs = (args.arch,) if args.arch is not None else ARCHS
    if args.seed is not None:
        seeds = (args.seed,)

    if args.timing_only:
        archs, seeds, ns = ("casca",), (0,), (30000,)

    (OUT / "protocol.json").write_text(json.dumps(protocol_dict(seeds, ns), indent=2))

    jobs = planned_jobs(archs, seeds, ns)
    pending = [j for j in jobs if not json_path(*j).exists()]
    print(f"[budget] {len(pending)} pending of {len(jobs)} requested  workers={args.workers}", flush=True)
    if not pending:
        print("[budget] nothing to train", flush=True)
        return

    t_all = time.time()
    if args.workers <= 1 or len(pending) == 1:
        data = load_per_byte_windows(CACHE)
        run_job_list(pending, 0, data["windows"], data["plaintext"], data["true_key"])
    else:
        chunks = [pending[i :: args.workers] for i in range(args.workers)]
        ctx = mp.get_context("spawn")
        procs = []
        for wid, chunk in enumerate(chunks):
            if not chunk:
                continue
            print(
                f"[budget] worker {wid}: {len(chunk)} jobs "
                f"{chunk[:3]}{'...' if len(chunk) > 3 else ''}",
                flush=True,
            )
            p = ctx.Process(target=_worker_entry, args=(chunk, wid))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
            if p.exitcode not in (0, None):
                raise SystemExit(f"worker failed: {[q.exitcode for q in procs]}")

    elapsed = time.time() - t_all
    print(f"[budget] finished {len(pending)} runs in {elapsed:.1f}s", flush=True)
    if args.timing_only:
        write_timing_projection(elapsed, "casca", 30000, 0)


if __name__ == "__main__":
    main()
