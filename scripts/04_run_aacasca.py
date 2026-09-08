#!/usr/bin/env python3
"""Train AA-CASCA (23-token att×att, no gate) at CA-SCA's 200-epoch protocol.

Does not reuse gated att×G cells or the 25-epoch / 300-step shorter budgets.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import sys as _casca_sys
_CASCA_ROOT = Path(__file__).resolve().parents[1]
if not (_CASCA_ROOT / 'src' / 'casca').is_dir():
    _CASCA_ROOT = Path(__file__).resolve().parents[3]
_casca_sys.path.insert(0, str(_CASCA_ROOT / "src"))
from casca.paths import raw_traces_path, repo_root, window_cache_path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from casca.runner import run_one
from casca.data.ascad import cache_per_byte_windows, load_per_byte_windows
from casca.attacks.aacasca import CONV_CONFIGS, make_h2
from casca.attacks.casca import PAPER_BATCH_SIZE, PAPER_EPOCHS, PAPER_LR, _device
from casca.data.windows import TRAIN_BYTES

RAW = raw_traces_path()
CACHE = window_cache_path()
OUT = Path(__file__).resolve().parents[1] / "results" / "results_aacasca_200epoch"
AUDIT = Path(__file__).resolve().parents[1] / "results" / "results_aacasca_200epoch/AUDIT.md"

GRID_N = (5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000)
SEEDS = tuple(range(10))
EPOCHS = PAPER_EPOCHS  # 200 — CA-SCA paper budget, not AA-CASCA's 300-step rec
BATCH = PAPER_BATCH_SIZE
EXPECTED_TOKENS = 23
EXPECTED_PARAMS = 99720
TAG = "h2E_selfmul_200ep"


def steps_per_epoch(n: int, batch: int = BATCH) -> int:
    nw = 14 * int(n)
    bs = min(int(batch), nw)
    return (nw + bs - 1) // bs


def json_path(seed: int, n: int) -> Path:
    return OUT / "runs" / f"{TAG}_seed{seed}_N{n}.json"


def is_valid(path: Path) -> bool:
    if not path.exists():
        return False
    rec = json.loads(path.read_text())
    proto = rec.get("protocol") or {}
    epochs = proto.get("epochs", rec.get("epochs"))
    return (
        rec.get("self_multiply") is True
        and rec.get("use_gate") is False
        and int(rec.get("n_tokens", -1)) == EXPECTED_TOKENS
        and int(rec.get("n_params", -1)) == EXPECTED_PARAMS
        and int(epochs) == EPOCHS
        and int(rec.get("n_steps", -1)) == EPOCHS * steps_per_epoch(int(rec["N"]))
        and isinstance(rec.get("ranks"), list)
        and len(rec["ranks"]) == 16
    )


def print_audit() -> list[tuple[int, int]]:
    print("=== Step 0 — AA-CASCA 200-epoch att×att audit ===", flush=True)
    print(
        "Required: Config E 23 tok, att×att (no gate), 200 epochs, "
        f"batch {BATCH}, Adam lr={PAPER_LR} eps=1e-7, CE, concat-14.",
        flush=True,
    )
    print(f"  (audit file: {AUDIT})", flush=True)
    pending: list[tuple[int, int]] = []
    print(
        f"{'N':>6}  reusable seeds                    still needed",
        flush=True,
    )
    for n in GRID_N:
        ok = [s for s in SEEDS if is_valid(json_path(s, n))]
        need = [s for s in SEEDS if s not in ok]
        for s in need:
            pending.append((s, n))
        ok_s = ",".join(str(s) for s in ok) if ok else "—"
        need_s = ",".join(str(s) for s in need) if need else "—"
        print(f"{n:6d}  {ok_s:<32} {need_s}", flush=True)
    print(
        f"reusable={80 - len(pending)}/80  pending={len(pending)}/80  "
        "excluded: gated att×G, 25-ep att×att screen, 300-step / step-budget cells "
        "(see AUDIT.md).",
        flush=True,
    )
    return pending


def print_protocol() -> dict:
    model = make_h2("E", use_gate=False, self_multiply=True)
    gated = make_h2("E", use_gate=True)
    info = {
        "architecture": "AA-CASCA Config E att×att",
        "n_tokens": model.n_tokens,
        "n_params": model.n_params(),
        "gated_params_not_used": gated.n_params(),
        "kernels": list(CONV_CONFIGS["E"].kernels),
        "strides": list(CONV_CONFIGS["E"].strides),
        "use_gate": False,
        "self_multiply": True,
        "operator": "Q,K,V = Linear(16→16); A=softmax(QK^T/√16); comb=(A V)⊙(A V); no G",
        "feature_fc": "64→256→128→64 SeLU (same stack as CA-SCA)",
        "epochs": EPOCHS,
        "batch_size": BATCH,
        "batch_applied_to": "concat-14 windows (TRAIN_BYTES 2..15)",
        "lr": PAPER_LR,
        "adam_eps": 1e-7,
        "loss": "CrossEntropy",
        "windows": "shared per-byte 700-sample windows with CA-SCA",
        "N": list(GRID_N),
        "seeds": list(SEEDS),
        "note": (
            "200-epoch CA-SCA paper protocol applied to AA-CASCA att×att. "
            "NOT the 300-step recommended budget from training_budget_M10."
        ),
    }
    print("=== Step 1 — protocol ===", flush=True)
    print(
        f"  architecture    = Config E att×att  tokens={info['n_tokens']} "
        f"params={info['n_params']} strides={info['strides']}",
        flush=True,
    )
    print(f"  operator        = {info['operator']}", flush=True)
    print(f"  gate            = OFF (no G layer; not att×G)", flush=True)
    print(f"  epochs          = {EPOCHS}  (CA-SCA paper budget, not 300-step rec)", flush=True)
    print(f"  batch_size      = {BATCH}  on concat TRAIN_BYTES={list(TRAIN_BYTES)}", flush=True)
    print(f"  optimizer       = Adam(lr={PAPER_LR}, eps=1e-7)  loss=CrossEntropy", flush=True)
    print("  windows         = shared with CA-SCA (unchanged)", flush=True)
    print("  steps/epoch     = ceil(14N / 20000):", flush=True)
    for n in GRID_N:
        spe = steps_per_epoch(n)
        print(
            f"    N={n:5d}  spe={spe:2d}  200-ep steps={200 * spe}",
            flush=True,
        )
    if info["n_tokens"] != EXPECTED_TOKENS:
        raise SystemExit(f"expected {EXPECTED_TOKENS} tokens, got {info['n_tokens']}")
    if info["n_params"] != EXPECTED_PARAMS:
        raise SystemExit(f"expected {EXPECTED_PARAMS} params, got {info['n_params']}")
    if info["n_params"] >= info["gated_params_not_used"]:
        raise SystemExit("att×att should have fewer params than gated")
    return info


def run_job_list(jobs: list[tuple[int, int]], worker_id: int, windows, plaintext, true_key) -> None:
    for i, (seed, n) in enumerate(jobs, start=1):
        dest = json_path(seed, n)
        if is_valid(dest):
            print(f"[200ep w{worker_id}] skip {i}/{len(jobs)} seed={seed} N={n}", flush=True)
            continue
        print(
            f"\n======== 200ep w{worker_id} {i}/{len(jobs)}  att×att  "
            f"seed={seed} N={n}  epochs={EPOCHS}  spe={steps_per_epoch(n)} ========",
            flush=True,
        )
        rec = run_one(
            windows,
            plaintext,
            true_key,
            seed,
            n,
            EPOCHS,
            OUT,
            model_fn=lambda: make_h2("E", use_gate=False, self_multiply=True),
            tag=TAG,
        )
        rec["variant"] = "self_multiply"
        rec["use_gate"] = False
        rec["self_multiply"] = True
        rec["protocol"] = {
            "epochs": EPOCHS,
            "batch_size": BATCH,
            "lr": PAPER_LR,
            "adam_eps": 1e-7,
            "loss": "CrossEntropy",
            "actual_steps": int(rec["n_steps"]),
        }
        expect = EPOCHS * steps_per_epoch(n)
        if rec["n_steps"] != expect:
            raise SystemExit(f"expected {expect} steps, got {rec['n_steps']} seed={seed} N={n}")
        if rec["n_tokens"] != EXPECTED_TOKENS or rec["n_params"] != EXPECTED_PARAMS:
            raise SystemExit(f"architecture mismatch seed={seed} N={n}")
        dest.write_text(json.dumps(rec, indent=2) + "\n")


def _worker_entry(jobs: list[tuple[int, int]], worker_id: int) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    if str(_device()) != "mps":
        raise SystemExit(f"w{worker_id}: not MPS ({_device()})")
    data = load_per_byte_windows(CACHE)
    print(f"[200ep w{worker_id}] {len(jobs)} jobs device=mps", flush=True)
    run_job_list(jobs, worker_id, data["windows"], data["plaintext"], data["true_key"])


def write_timing(elapsed_s: float, n: int, seed: int) -> None:
    # Scale by step count (spe × 200). Dual-worker ~1.45× from prior MPS sharing.
    timed_steps = EPOCHS * steps_per_epoch(n)
    jobs = [(s, nn) for nn in GRID_N for s in SEEDS]
    weights = []
    for _s, nn in jobs:
        weights.append(EPOCHS * steps_per_epoch(nn) / timed_steps)
    total_seq = elapsed_s * sum(weights)
    lines = [
        f"# Timing projection (from att×att N={n} seed={seed}, 200 epochs)",
        "",
        f"Timed cell wall-clock: **{elapsed_s:.1f}s** ({elapsed_s / 60:.2f} min), "
        f"{timed_steps} steps.",
        "",
        f"Projected 80-cell grid:",
        f"- sequential: **{total_seq / 3600:.2f} h** ({total_seq / 60:.0f} min)",
        f"- 2 workers (×1.45 GPU-share): **{total_seq * 1.45 / 2 / 3600:.2f} h** "
        f"({total_seq * 1.45 / 2 / 60:.0f} min)",
        "",
        "200 epochs is much longer than AA-CASCA's 300-step recommended budget "
        f"(at N={n}, 300 steps vs {timed_steps}).",
        "",
    ]
    text = "\n".join(lines)
    (OUT / "timing_projection.md").write_text(text)
    print(text, flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--timing-only", action="store_true", help="N=40000 seed 0")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--N", type=int, nargs="+", default=None)
    p.add_argument("--workers", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    info = print_protocol()
    pending = print_audit()
    if args.audit_only:
        return

    dev = _device()
    print(f"  device          = {dev}", flush=True)
    if str(dev) != "mps":
        raise SystemExit(f"refusing silent CPU/CUDA fallback; device={dev}")

    if not CACHE.exists():
        cache_per_byte_windows(RAW, CACHE)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    (OUT / "protocol.json").write_text(json.dumps(info, indent=2))

    if args.timing_only:
        pending = [(0, 40000)]
    else:
        seeds = tuple(args.seeds) if args.seeds is not None else SEEDS
        ns = tuple(args.N) if args.N is not None else GRID_N
        if args.seed is not None:
            seeds = (args.seed,)
        wanted = {(s, n) for n in ns for s in seeds}
        pending = [j for j in pending if j in wanted]

    print(f"[200ep] {len(pending)} pending  workers={args.workers}", flush=True)
    if not pending:
        print("[200ep] nothing to train", flush=True)
        return

    t_all = time.time()
    if args.workers <= 1 or len(pending) == 1:
        data = load_per_byte_windows(CACHE)
        run_job_list(pending, 0, data["windows"], data["plaintext"], data["true_key"])
    else:
        # Longest N first so workers stay balanced.
        pending = sorted(pending, key=lambda t: (-t[1], t[0]))
        chunks = [pending[i :: args.workers] for i in range(args.workers)]
        ctx = mp.get_context("spawn")
        procs = []
        for wid, chunk in enumerate(chunks):
            if not chunk:
                continue
            print(f"[200ep] worker {wid}: {len(chunk)} jobs", flush=True)
            p = ctx.Process(target=_worker_entry, args=(chunk, wid))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
            if p.exitcode not in (0, None):
                raise SystemExit(f"worker failed: {[q.exitcode for q in procs]}")
    elapsed = time.time() - t_all
    print(f"[200ep] finished {len(pending)} runs in {elapsed:.1f}s", flush=True)
    if args.timing_only:
        write_timing(elapsed, 40000, 0)


if __name__ == "__main__":
    main()
