#!/usr/bin/env python3
"""Appendix attention-combination (4-token) and square-token screens."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Appendix attention-combination / square-token screens."
    )
    p.add_argument("--screen", choices=("4tok", "square"), default="4tok")
    args, rest = p.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    if args.screen == "square":
        from casca.experiments.attn_square import main as run
    else:
        from casca.experiments.attn_4tok import main as run
    run()


if __name__ == "__main__":
    main()
