"""Pearson POI helpers used by the 16-byte correlation scan."""

from __future__ import annotations

import numpy as np

HW_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def peak_abs(r: np.ndarray, lo: int, hi: int) -> tuple[int, float]:
    sl = r[lo:hi]
    i = int(np.abs(sl).argmax())
    return lo + i, float(sl[i])


def pearson_1d(traces: np.ndarray, hyp: np.ndarray) -> np.ndarray:
    """Pearson r vs every sample. Same formula as casca.attacks.cpa.pearson_matrix."""
    x = np.asarray(hyp, dtype=np.float64).reshape(-1)
    y = np.asarray(traces, dtype=np.float64)
    xc = x - x.mean()
    yc = y - y.mean(axis=0, keepdims=True)
    num = xc @ yc
    xnorm = float(np.sqrt(np.dot(xc, xc)))
    ynorm = np.sqrt(np.einsum("ij,ij->j", yc, yc))
    den = xnorm * ynorm
    r = np.zeros_like(num)
    np.divide(num, den, out=r, where=den > 1e-15)
    return r
