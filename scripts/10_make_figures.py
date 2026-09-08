#!/usr/bin/env python3
"""Rebuild the named thesis figures from stored run JSON / CSVs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FIGURES = {
    "ascad_poi_correlation_scan": "casca.figures.poi_correlation_scan",
    "plot1_individual_seeds": "casca.figures.casca_reproducibility",
    "plot2_mean_ci": "casca.figures.casca_reproducibility",
    "plot_ge_and_val_loss": "casca.figures.casca_training_budget",
    "aa-casca-plot2_mean_ci": "casca.figures.gated_E_200ep",
    "plot_ge_and_val_loss_aa_attxatt_100ep": "casca.figures.casca_training_budget",
    "plot_e25_e200_casca_ge_bytes": "casca.figures.casca_vs_configE",
    "plot_e25_e200_casca_time": "casca.figures.casca_vs_configE",
    "plot_e25_casca_moc_ge_bytes": "casca.figures.e25_casca_moc",
    "plot_e25_casca_moc_time": "casca.figures.e25_casca_moc",
}


def main() -> None:
    p = argparse.ArgumentParser(description="Write thesis figures into figures/.")
    p.add_argument(
        "--figure",
        choices=list(FIGURES) + ["all"],
        default="all",
        help="Named figure, or all (default).",
    )
    args = p.parse_args()
    names = list(FIGURES) if args.figure == "all" else [args.figure]
    seen: set[str] = set()
    for name in names:
        modname = FIGURES[name]
        if modname in seen:
            continue
        seen.add(modname)
        print(f"[figures] {name} -> {modname}", flush=True)
        mod = __import__(modname, fromlist=["main"])
        mod.main()


if __name__ == "__main__":
    main()
