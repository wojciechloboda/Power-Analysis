"""Repo-relative paths. Absolute machine paths from the original tree are not used."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    # src/casca/paths.py -> clean-repo/
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    env = os.environ.get("ASCAD_DATA_DIR")
    if env:
        return Path(env)
    return repo_root() / "data"


def raw_traces_path() -> Path:
    return data_dir() / "ATMega8515_raw_traces.h5"


def window_cache_path() -> Path:
    env = os.environ.get("CASCA_WINDOW_CACHE")
    if env:
        return Path(env)
    return repo_root() / "results" / "per_byte_windows.npz"


def results_dir() -> Path:
    return repo_root() / "results"


def figures_dir() -> Path:
    return repo_root() / "figures"


def tables_dir() -> Path:
    return repo_root() / "tables"
