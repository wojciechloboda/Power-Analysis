"""Calinski-Harabasz scoring over 256 key-byte hypotheses.

Paper Eq. (6):  v = S-Box(k XOR P_t)
Paper Eq. (11) (ASCAD / S-2 reporting choice, multi-bit):
    s(k) = sum_{b=0..7} CH(X, f_mono(v, b))
    f_mono(v, b) = v AND (2^b)

CH is sklearn's calinski_harabasz_score, as the paper states.
The full 256-vector is returned so later analyses can ensemble scores
without retraining.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import calinski_harabasz_score

from casca.data.ascad import sbox


def _ch(features: np.ndarray, labels: np.ndarray) -> float:
    # sklearn raises if there is only one unique label.
    if np.unique(labels).size < 2:
        return 0.0
    return float(calinski_harabasz_score(features, labels))


def _two_class_ch_all_keys(features: np.ndarray, labels01: np.ndarray) -> np.ndarray:
    """Calinski-Harabasz for 256 independent 2-cluster labelings.

    Mathematically the same as sklearn.metrics.calinski_harabasz_score
    with k=2:  CH = (B_k / W_k) * (N - 2).
    `labels01` has shape (N, 256) with values in {0, 1}.
    """
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels01, dtype=np.float64)
    n, d = x.shape
    n1 = y.sum(axis=0)
    n0 = n - n1
    valid = (n0 >= 1) & (n1 >= 1)
    sum1 = x.T @ y
    sum_all = x.sum(axis=0)
    sum0 = sum_all[:, None] - sum1
    m = sum_all / n
    m1 = np.zeros_like(sum1)
    m0 = np.zeros_like(sum0)
    np.divide(sum1, n1, out=m1, where=n1 > 0)
    np.divide(sum0, n0, out=m0, where=n0 > 0)
    bk = n0 * np.sum((m0 - m[:, None]) ** 2, axis=0) + n1 * np.sum((m1 - m[:, None]) ** 2, axis=0)
    xfrob = np.square(x).sum()
    wk = xfrob - n0 * np.sum(m0**2, axis=0) - n1 * np.sum(m1**2, axis=0)
    scores = np.zeros(256, dtype=np.float64)
    np.divide(bk, wk, out=scores, where=(wk > 1e-12) & valid)
    scores *= n - 2
    scores[~valid] = 0.0
    return scores


def multibit_ch_vector(
    features: np.ndarray,
    plaintext_byte: np.ndarray,
) -> np.ndarray:
    """Return shape (256,) CH scores, one per AES key-byte hypothesis.

    This is the S-2 selected function (multi-bit, Eq. 11).
    """
    features = np.asarray(features, dtype=np.float64)
    pt = np.asarray(plaintext_byte, dtype=np.uint8)
    keys = np.arange(256, dtype=np.uint8)
    v = sbox(np.bitwise_xor(pt[:, None], keys[None, :]))
    scores = np.zeros(256, dtype=np.float64)
    for b in range(8):
        # f_mono(v, b) = v AND 2^b  — two clusters {0, 2^b}. Using the bit
        # itself as a 0/1 label is equivalent for CH (only uniqueness matters).
        labels01 = ((v >> b) & 1).astype(np.float64)
        scores += _two_class_ch_all_keys(features, labels01)
    return scores


def hw_ch_vector(features: np.ndarray, plaintext_byte: np.ndarray) -> np.ndarray:
    """Eq. (8), HW labelling. Implemented for completeness, not used in S-2."""
    features = np.asarray(features, dtype=np.float64)
    pt = np.asarray(plaintext_byte, dtype=np.uint8)
    scores = np.empty(256, dtype=np.float64)
    for k in range(256):
        v = sbox(np.bitwise_xor(pt, np.uint8(k)))
        hw = np.bitwise_count(v) if hasattr(np, "bitwise_count") else _hw_fallback(v)
        scores[k] = _ch(features, hw)
    return scores


def _hw_fallback(v: np.ndarray) -> np.ndarray:
    x = v.astype(np.uint8)
    table = np.array([bin(i).count("1") for i in range(256)], dtype=np.int16)
    return table[x]


def assert_multibit_matches_sklearn(n: int = 400, seed: int = 0) -> None:
    """Guard: vectorized Eq. 11 must match sklearn on a small random set."""
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, 64))
    pt = rng.integers(0, 256, size=n, dtype=np.uint8)
    fast = multibit_ch_vector(features, pt)
    slow = np.empty(256, dtype=np.float64)
    for k in range(256):
        v = sbox(np.bitwise_xor(pt, np.uint8(k)))
        total = 0.0
        for b in range(8):
            total += _ch(features, np.bitwise_and(v, np.uint8(1 << b)))
        slow[k] = total
    if not np.allclose(fast, slow, rtol=1e-5, atol=1e-4):
        raise AssertionError("vectorized multi-bit CH diverges from sklearn")


def rank_of_key(scores: np.ndarray, true_key: int) -> int:
    """0-indexed rank: number of hypotheses with a strictly higher score.

    Rank 0 means the true key uniquely (or jointly) maximises the CH index.
    Ties: any strictly-better hypothesis counts; equal scores do not.
    """
    true = scores[true_key]
    return int(np.sum(scores > true))
