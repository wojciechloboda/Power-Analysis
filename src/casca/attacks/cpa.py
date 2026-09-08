"""Vectorized Hamming-weight CPA (Brier, Clavier & Olivier, CHES 2004).

For each key-byte hypothesis k:

    v = Sbox(p XOR k)
    predicted leakage = HW(v)

Score(k) = max over time samples t of |Pearson(HW(v), traces[:, t])|.
Rank 0 = the correct key uniquely (or jointly) maximises that score.
"""

from __future__ import annotations

import numpy as np

from casca.attacks.clustering import rank_of_key
from casca.data.ascad import sbox

KEYS = np.arange(256, dtype=np.uint8)
HW_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hw_hypotheses(plaintext_byte: np.ndarray) -> np.ndarray:
    """Predicted HW leakage for all 256 key hypotheses. Shape (N, 256)."""
    pt = np.asarray(plaintext_byte, dtype=np.uint8).reshape(-1)
    values = sbox(np.bitwise_xor(pt[:, None], KEYS[None, :]))
    return HW_LUT[values]


def pearson_matrix(hypotheses: np.ndarray, traces: np.ndarray) -> np.ndarray:
    """Pearson r between every hypothesis column and every time sample.

    hypotheses: (N, K), traces: (N, T)  ->  corr: (K, T)
    Zero-variance columns yield r = 0 (no NaNs).
    """
    x = np.asarray(hypotheses, dtype=np.float64)
    y = np.asarray(traces, dtype=np.float64)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"trace/hypothesis length mismatch: {x.shape} vs {y.shape}")
    if x.shape[0] < 2:
        raise ValueError("need at least 2 traces for Pearson correlation")
    xc = x - x.mean(axis=0, keepdims=True)
    yc = y - y.mean(axis=0, keepdims=True)
    num = xc.T @ yc
    xnorm = np.sqrt(np.einsum("ij,ij->j", xc, xc))
    ynorm = np.sqrt(np.einsum("ij,ij->j", yc, yc))
    den = xnorm[:, None] * ynorm[None, :]
    corr = np.zeros_like(num)
    np.divide(num, den, out=corr, where=den > 1e-15)
    return corr


def cpa_scores(traces: np.ndarray, plaintext_byte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (scores, corr) with scores[k] = max_t |corr[k, t]|."""
    hw = hw_hypotheses(plaintext_byte)
    corr = pearson_matrix(hw, traces)
    scores = np.max(np.abs(corr), axis=1)
    return scores, corr


def attack_byte(
    traces: np.ndarray,
    plaintext_byte: np.ndarray,
    true_key: int,
) -> dict:
    """CPA on one key byte. Rank uses the project's rank_of_key (strictly greater)."""
    scores, corr = cpa_scores(traces, plaintext_byte)
    true_key = int(true_key)
    peak_sample = int(np.argmax(np.abs(corr[true_key])))
    true_corr = float(corr[true_key, peak_sample])
    abs_scores = scores.copy()
    runner_up = float(np.max(np.delete(abs_scores, true_key))) if scores.size > 1 else 0.0
    return {
        "scores": scores,
        "corr": corr,
        "rank": rank_of_key(scores, true_key),
        "true_key": true_key,
        "true_score": float(scores[true_key]),
        "true_corr_at_peak": true_corr,
        "peak_sample": peak_sample,
        "runner_up_score": runner_up,
        "margin": float(scores[true_key] - runner_up),
    }


def assert_pearson_matches_definition(n: int = 80, t: int = 5, seed: int = 0) -> None:
    """Guard: matrix Pearson matches the textbook formula on a small draw."""
    rng = np.random.default_rng(seed)
    hyp = rng.integers(0, 9, size=(n, 4)).astype(np.float64)
    tr = rng.normal(size=(n, t))
    fast = pearson_matrix(hyp, tr)
    xc = hyp - hyp.mean(axis=0)
    yc = tr - tr.mean(axis=0)
    slow = np.empty((4, t), dtype=np.float64)
    for k in range(4):
        for j in range(t):
            num = float(np.dot(xc[:, k], yc[:, j]))
            den = float(np.linalg.norm(xc[:, k]) * np.linalg.norm(yc[:, j]))
            slow[k, j] = num / den if den > 1e-15 else 0.0
    if not np.allclose(fast, slow, rtol=1e-12, atol=1e-12):
        raise AssertionError("vectorized Pearson diverges from the textbook formula")
