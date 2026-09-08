#!/usr/bin/env python3
"""AA-CASCA att×att (Config E, no gate) training trajectories to 100 epochs.

N ∈ {10k, 20k, 30k}, seeds 0–4. Step-checkpointed GE + val loss, same
trainer convention as the CA-SCA budget plots. Does not overwrite
training_budget_M10 (gated, 1000-step) or results_aacasca_200epoch.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
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

REPO = Path(__file__).resolve().parents[3]
ROOT = Path(__file__).resolve().parents[3]
TB_PATH = Path(__file__).resolve().parent / "budget_m10.py"
OUT = REPO / "results" / "results_casca_training_budget"
RUNS = OUT / "runs_aa_attxatt_100ep"

NS = (10000, 20000, 30000)
SEEDS = tuple(range(5))
TARGET_EPOCHS = 100
EXPECTED_PARAMS = 99720
EXPECTED_TOKENS = 23
ARCH = "aa_casca"
# Tail through 100 ep at N=30k (21×100=2100).
AA_TAIL = (1200, 1400, 1600, 1800, 2000, 2100)


def load_tb():
    spec = importlib.util.spec_from_file_location("tb40_aa100", TB_PATH)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {TB_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tb40_aa100"] = mod
    spec.loader.exec_module(mod)
    return mod


def stamp(path: Path) -> None:
    rec = json.loads(path.read_text())
    rec["variant"] = "attxatt"
    rec["use_gate"] = False
    rec["self_multiply"] = True
    rec["target_epochs"] = TARGET_EPOCHS
    path.write_text(json.dumps(rec, indent=2))


def is_done(path: Path, tb, n: int) -> bool:
    if not path.exists():
        return False
    rec = json.loads(path.read_text())
    return (
        rec.get("self_multiply") is True
        and rec.get("use_gate") is False
        and int(rec.get("max_steps", -1)) == tb.max_steps_for(ARCH, n)
        and int(rec.get("n_params", -1)) == EXPECTED_PARAMS
    )


def patch(tb) -> None:
    from casca.attacks.aacasca import make_h2  # noqa: E402

    tb.OUT = OUT
    tb.CASCA_TAIL = AA_TAIL

    def max_steps_for(arch: str, n: int) -> int:
        return TARGET_EPOCHS * tb.steps_per_epoch(n)

    def json_path(arch: str, seed: int, n: int) -> Path:
        return RUNS / f"aa_attxatt_seed{seed}_N{n}.json"

    def model_fn_for(arch: str):
        return lambda: make_h2("E", use_gate=False, self_multiply=True)

    def schedule_for(arch: str, n: int) -> tuple[int, ...]:
        cap = max_steps_for(arch, n)
        steps = [s for s in tb.COMMON_STEPS if s <= cap]
        steps += [s for s in AA_TAIL if s <= cap]
        if cap not in steps:
            steps.append(cap)
        return tuple(sorted(set(steps)))

    tb.max_steps_for = max_steps_for
    tb.json_path = json_path
    tb.model_fn_for = model_fn_for
    tb.schedule_for = schedule_for


def print_plan(tb) -> None:
    from casca.attacks.aacasca import make_h2  # noqa: E402

    model = make_h2("E", use_gate=False, self_multiply=True)
    print("=== AA-CASCA att×att 100-epoch trajectories ===", flush=True)
    print(f"  out             = {RUNS}", flush=True)
    print(
        f"  architecture    = Config E att×att  tokens={model.n_tokens} "
        f"params={model.n_params()}  (no G; comb = att ⊙ att)",
        flush=True,
    )
    print(f"  N               = {list(NS)}", flush=True)
    print(f"  seeds           = {list(SEEDS)}", flush=True)
    print(f"  cap             = {TARGET_EPOCHS} × steps_per_epoch(N)", flush=True)
    if model.n_params() != EXPECTED_PARAMS or model.n_tokens != EXPECTED_TOKENS:
        raise SystemExit(
            f"expected {EXPECTED_TOKENS} tok / {EXPECTED_PARAMS} params, "
            f"got {model.n_tokens} / {model.n_params()}"
        )
    for n in NS:
        spe = tb.steps_per_epoch(n)
        cap = TARGET_EPOCHS * spe
        sched = tb.schedule_for(ARCH, n)
        print(
            f"    N={n}  spe={spe}  100ep={cap}  300steps={300 / spe:.1f} ep  "
            f"n_ckpts={len(sched)}  last={sched[-1]}",
            flush=True,
        )


def _finish_cell(tb, dest: Path, n: int) -> None:
    stamp(dest)
    rec = json.loads(dest.read_text())
    if int(rec.get("n_params", -1)) != EXPECTED_PARAMS:
        dest.unlink(missing_ok=True)
        raise SystemExit(f"param mismatch in {dest.name}: {rec.get('n_params')}")
    if int(rec.get("max_steps", -1)) != tb.max_steps_for(ARCH, n):
        raise SystemExit(f"max_steps mismatch in {dest.name}")


def _worker_entry(jobs: list[tuple[str, int, int]], worker_id: int) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    tb = load_tb()
    patch(tb)
    from casca.attacks.casca import _device  # noqa: E402

    if str(_device()) != "mps":
        raise SystemExit(f"w{worker_id}: not MPS ({_device()})")
    data = tb.load_per_byte_windows(tb.CACHE)
    print(f"[aa100 w{worker_id}] {len(jobs)} jobs device=mps", flush=True)
    for i, (arch, seed, n) in enumerate(jobs, start=1):
        dest = tb.json_path(arch, seed, n)
        if is_done(dest, tb, n):
            print(f"[aa100 w{worker_id}] skip {i}/{len(jobs)}  seed={seed} N={n}", flush=True)
            continue
        print(
            f"\n======== aa100 w{worker_id} {i}/{len(jobs)}  att×att  "
            f"seed={seed} N={n}  max_steps={tb.max_steps_for(arch, n)} ========",
            flush=True,
        )
        tb.run_one(data["windows"], data["plaintext"], data["true_key"], arch, seed, n)
        _finish_cell(tb, dest, n)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--timing-only", action="store_true")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--N", type=int, nargs="+", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    args = p.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    tb = load_tb()
    patch(tb)
    from casca.attacks.casca import _device  # noqa: E402

    print_plan(tb)
    dev = _device()
    print(f"  device          = {dev}", flush=True)
    if str(dev) != "mps":
        raise SystemExit(f"refusing silent CPU/CUDA fallback; device={dev}")

    RUNS.mkdir(parents=True, exist_ok=True)
    ns = tuple(args.N) if args.N is not None else NS
    seeds = tuple(args.seeds) if args.seeds is not None else SEEDS
    if args.timing_only:
        ns, seeds = (10000,), (0,)

    proto = {
        "architecture": "AA-CASCA Config E att×att",
        "use_gate": False,
        "self_multiply": True,
        "operator": "comb = (A V) ⊙ (A V); no G",
        "n_params": EXPECTED_PARAMS,
        "n_tokens": EXPECTED_TOKENS,
        "target_epochs": TARGET_EPOCHS,
        "N": list(NS),
        "seeds": list(SEEDS),
        "cap": "100 × steps_per_epoch(N)",
        "does_not_overwrite": [
            str(REPO / "results" / "training_budget_M10"),
            str(REPO / "results" / "results_aacasca_200epoch"),
        ],
        "steps_per_epoch": {str(n): tb.steps_per_epoch(n) for n in NS},
        "steps_100ep": {str(n): TARGET_EPOCHS * tb.steps_per_epoch(n) for n in NS},
        "schedules": {str(n): list(tb.schedule_for(ARCH, n)) for n in NS},
    }
    (RUNS / "protocol.json").write_text(json.dumps(proto, indent=2))

    jobs = [(ARCH, s, n) for n in sorted(ns, reverse=True) for s in seeds]
    pending = [j for j in jobs if not is_done(tb.json_path(*j), tb, j[2])]
    print(f"[aa100] {len(pending)} pending of {len(jobs)}  workers={args.workers}", flush=True)
    if not pending:
        print("[aa100] nothing to train", flush=True)
        return

    t0 = time.time()
    if args.workers <= 1 or len(pending) == 1:
        data = tb.load_per_byte_windows(tb.CACHE)
        for i, (arch, seed, n) in enumerate(pending, start=1):
            print(f"\n======== aa100 {i}/{len(pending)}  seed={seed} N={n} ========", flush=True)
            rec = tb.run_one(data["windows"], data["plaintext"], data["true_key"], arch, seed, n)
            dest = tb.json_path(arch, seed, n)
            _finish_cell(tb, dest, n)
            if args.timing_only:
                proj = float(rec["elapsed_seconds"])
                rest = [(s, nn) for nn in NS for s in SEEDS if not (nn == n and s == seed)]
                scale = sum(nn / n for _s, nn in rest)
                print(
                    f"[timing] N={n} seed={seed} {proj:.1f}s  "
                    f"full 15-cell sequential ≈ {(scale * proj + proj) / 60:.1f} min  "
                    f"2 workers ≈ {(scale * proj / 2 + proj) / 60:.1f} min",
                    flush=True,
                )
                (OUT / "timing_aa_attxatt_100ep.md").write_text(
                    f"# att×att 100-epoch timing\n\nTimed N={n} seed={seed}: **{proj:.1f}s**.\n"
                    f"Remaining {len(rest)} cells ≈ **{scale * proj / 60:.1f} min** sequential "
                    f"(~{scale * proj / 2 / 60:.1f} min with 2 workers).\n"
                )
    else:
        chunks = [pending[i :: args.workers] for i in range(args.workers)]
        ctx = mp.get_context("spawn")
        procs = []
        for wid, chunk in enumerate(chunks):
            if not chunk:
                continue
            proc = ctx.Process(target=_worker_entry, args=(chunk, wid))
            proc.start()
            procs.append(proc)
        for proc in procs:
            proc.join()
            if proc.exitcode not in (0, None):
                raise SystemExit(f"worker failed: {[q.exitcode for q in procs]}")

    print(f"[aa100] finished in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
