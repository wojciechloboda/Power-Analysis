#!/usr/bin/env python3
"""Training-budget sweeps: M=10 gated Config E, or att×att 100-epoch."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Training-budget sweeps (M=10 gated E, or att×att 100-epoch)."
    )
    p.add_argument("--mode", choices=("m10", "attxatt100"), default="m10")
    args, rest = p.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    if args.mode == "attxatt100":
        from casca.experiments.budget_attxatt100 import main as run
    else:
        from casca.experiments.budget_m10 import main as run
    run()


if __name__ == "__main__":
    main()
