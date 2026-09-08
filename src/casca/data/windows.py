"""Per-byte 700-sample windows from the already-validated diagnostic.

Source of the numbers (NOT training/model code):
  magisterka/results/window_diagnostic/leakage_vs_window.csv

Byte 2 is the canonical ASCAD window [45400:46100]. The remaining bytes use
the same 700-sample length with the diagnostic's per-byte anchors. Bytes 0
and 1 are unmasked in ASCAD (mask fixed at 0); the paper trains the S-2 CNN
on the other 14 (masked) bytes only.
"""

from __future__ import annotations

# (start, end) with end exclusive, length 700.
BYTE_WINDOWS: dict[int, tuple[int, int]] = {
    0: (30824, 31524),
    1: (24577, 25277),
    2: (45400, 46100),
    3: (32906, 33606),
    4: (47482, 48182),
    5: (41235, 41935),
    6: (37071, 37771),
    7: (34989, 35689),
    8: (26659, 27359),
    9: (39153, 39853),
    10: (28742, 29442),
    11: (43318, 44018),
    12: (20413, 21113),
    13: (22495, 23195),
    14: (49565, 50265),
    15: (18330, 19030),
}

UNMASKED_BYTES = (0, 1)
MASKED_BYTES = tuple(range(2, 16))
ALL_BYTES = tuple(range(16))
WINDOW_LEN = 700
# Paper §4.4.2: train on SubBytes windows other than the two unmasked bytes.
TRAIN_BYTES = MASKED_BYTES


import numpy as np

VAL_FRAC = 0.10


def concatenate_train_set(windows: np.ndarray, plaintext: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Stack the 14 masked-byte windows of the first n traces."""
    chunks_x = [windows[:n, b] for b in TRAIN_BYTES]
    chunks_y = [plaintext[:n, b] for b in TRAIN_BYTES]
    return np.concatenate(chunks_x, axis=0), np.concatenate(chunks_y, axis=0)


def make_val_concat(windows: np.ndarray, plaintext: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, int]:
    n_val = max(1, int(round(VAL_FRAC * n)))
    stop = n + n_val
    if stop > windows.shape[0]:
        raise SystemExit(f"val slice [{n}:{stop}] exceeds {windows.shape[0]} traces")
    x = np.concatenate([windows[n:stop, b] for b in TRAIN_BYTES], axis=0)
    y = np.concatenate([plaintext[n:stop, b] for b in TRAIN_BYTES], axis=0)
    return x, y, n_val
