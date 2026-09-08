"""GE / SR plots for the byte-2 validation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_ge_sr(summary_csv: Path, out_dir: Path, title_suffix: str = "") -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(summary_csv)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.plot(df["N"], df["GE"], marker="o", color="C0")
    ax.set_xlabel("N (attack traces clustered)")
    ax.set_ylabel("Guessing entropy GE_N")
    ax.set_title(f"GE_N{title_suffix}")
    ax.set_ylim(0, 255)
    ax.axhline(127.5, color="gray", ls="--", lw=0.8, label="random (127.5)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for col, label in (("SR_1", "SR_1"), ("SR_5", "SR_5"), ("SR_10", "SR_10")):
        if col in df.columns:
            ax.plot(df["N"], df[col], marker="o", label=label)
    ax.set_xlabel("N (attack traces clustered)")
    ax.set_ylabel("Success rate")
    ax.set_title(f"SR_o(N){title_suffix}")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = out_dir / "ge_sr.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {path}", flush=True)

def plot_mean_recovered_vs_n(
    series: list[tuple[str, list[int], list[float], list[float] | None]],
    out_path: Path,
    title: str = "Mean bytes recovered vs N",
    n_bytes: int = 16,
) -> None:
    """``series``: (label, Ns, means, optional stds)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for label, ns, means, stds in series:
        ax.plot(ns, means, marker="o", label=label)
        if stds is not None:
            m = np.asarray(means, dtype=np.float64)
            s = np.asarray(stds, dtype=np.float64)
            ax.fill_between(ns, m - s, m + s, alpha=0.15)
    ax.set_xlabel("N (traces)")
    ax.set_ylabel(f"mean bytes recovered / {n_bytes}")
    ax.set_title(title)
    ax.set_ylim(-0.2, n_bytes + 0.4)
    ax.axhline(16, color="gray", ls="--", lw=0.8)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out_path}", flush=True)


def plot_ge_per_byte(per_byte_rows: list[dict], unmasked: tuple[int, ...], out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    byte_df = pd.DataFrame(per_byte_rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    for b in sorted(byte_df["byte"].unique()):
        sub = byte_df[byte_df["byte"] == b]
        ls = "--" if int(b) in unmasked else "-"
        ax.plot(sub["N"], sub["GE"], ls=ls, marker="o", label=f"b{int(b)}")
    ax.set_xlabel("N")
    ax.set_ylabel("GE_N")
    ax.set_title("MOC per-byte guessing entropy (dashed = unmasked 0,1)")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out_path}", flush=True)
