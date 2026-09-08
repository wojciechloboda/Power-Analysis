"""ASCAD loaders, z-score, window cache, and AES SubBytes lookup."""

import numpy as np

SBOX = np.array(
    [
        0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
        0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
        0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
        0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
        0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
        0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
        0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
        0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
        0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
        0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
        0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
        0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
        0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
        0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
        0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
        0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
    ],
    dtype=np.uint8,
)


def sbox(values: np.ndarray) -> np.ndarray:
    return SBOX[np.asarray(values, dtype=np.uint8)]

"""Load the canonical ASCAD 700-sample byte-2 traces.

Official construction (ANSSI ASCAD documentation / ePrint 2018/053):
  raw traces[:, 45400:46100]  ->  700 samples
  traces 0:50000              ->  Profiling_traces
  traces 50000:60000          ->  Attack_traces
  label                       ->  Sbox(plaintext[2] XOR key[2])

CA-SCA does not train on that S-box label; it trains on plaintext[2].
The official labels are stored only so the file matches the public format.
"""

from pathlib import Path

import h5py
import numpy as np

ASCAD_WINDOW = (45400, 46100)  # 700 samples
TARGET_BYTE = 2
N_PROFILING = 50000
N_ATTACK = 10000
TRACE_LEN = 700

# Official ASCAD fixed key (also stored per-trace in the raw metadata).
ASCAD_KEY = np.array(
    [0x4D, 0xFB, 0xE0, 0xF2, 0x72, 0x21, 0xFE, 0x10, 0xA7, 0x8D, 0x4A, 0xDC, 0x8E, 0x49, 0x04, 0x69],
    dtype=np.uint8,
)


def build_canonical_ascad_h5(raw_path: Path, out_path: Path) -> None:
    """Crop the official window from ATMega8515_raw_traces.h5 into ASCAD.h5."""
    raw_path = Path(raw_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    start, stop = ASCAD_WINDOW
    print(f"[data] Reading {raw_path}")
    print(f"[data] Cropping samples [{start}:{stop}] (len={stop - start})")
    with h5py.File(raw_path, "r") as src:
        traces = src["traces"][:, start:stop]
        metadata = src["metadata"][:]
    if traces.shape != (N_PROFILING + N_ATTACK, TRACE_LEN):
        raise ValueError(f"Unexpected cropped shape {traces.shape}")

    plaintext = np.stack(metadata["plaintext"]).astype(np.uint8)
    key = np.stack(metadata["key"]).astype(np.uint8)
    ciphertext = np.stack(metadata["ciphertext"]).astype(np.uint8)
    labels = sbox(plaintext[:, TARGET_BYTE] ^ key[:, TARGET_BYTE])

    # Sanity: ASCAD is a fixed-key campaign.
    if not np.all(key == ASCAD_KEY):
        uniq = np.unique(key, axis=0)
        print(f"[data] WARNING: key is not constant ({len(uniq)} unique keys)")

    print(f"[data] Writing {out_path}")
    with h5py.File(out_path, "w") as dst:
        for split, sl in (("Profiling_traces", slice(0, N_PROFILING)), ("Attack_traces", slice(N_PROFILING, None))):
            grp = dst.create_group(split)
            grp.create_dataset("traces", data=traces[sl], compression="gzip")
            grp.create_dataset("labels", data=labels[sl])
            dt = np.dtype(
                [
                    ("plaintext", "u1", (16,)),
                    ("ciphertext", "u1", (16,)),
                    ("key", "u1", (16,)),
                ]
            )
            meta = np.empty(traces[sl].shape[0], dtype=dt)
            meta["plaintext"] = plaintext[sl]
            meta["ciphertext"] = ciphertext[sl]
            meta["key"] = key[sl]
            grp.create_dataset("metadata", data=meta)
    print(f"[data] Done. profiling={N_PROFILING} attack={N_ATTACK} len={TRACE_LEN}")


def load_canonical_ascad(ascad_h5: Path) -> dict:
    """Return profiling/attack traces, plaintext, key as numpy arrays."""
    with h5py.File(ascad_h5, "r") as f:
        out = {
            "profiling_traces": f["Profiling_traces/traces"][:],
            "profiling_plaintext": f["Profiling_traces/metadata"]["plaintext"][:],
            "profiling_key": f["Profiling_traces/metadata"]["key"][:],
            "profiling_labels": f["Profiling_traces/labels"][:],
            "attack_traces": f["Attack_traces/traces"][:],
            "attack_plaintext": f["Attack_traces/metadata"]["plaintext"][:],
            "attack_key": f["Attack_traces/metadata"]["key"][:],
            "attack_labels": f["Attack_traces/labels"][:],
        }
    out["true_key_byte"] = int(out["profiling_key"][0, TARGET_BYTE])
    if out["true_key_byte"] != int(ASCAD_KEY[TARGET_BYTE]):
        raise ValueError("Profiling key byte 2 does not match the published ASCAD key")
    return out


def zscore_train_apply(
    train_traces: np.ndarray, *others: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Per-timepoint z-score. Not specified in the paper; required for SeLU scale.

    Statistics are computed on the training traces only and applied to every set.
    """
    x = train_traces.astype(np.float32)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    scaled = [(x - mean) / std]
    for arr in others:
        scaled.append((arr.astype(np.float32) - mean) / std)
    return tuple(scaled)


def cache_per_byte_windows(raw_path: Path, out_path: Path, chunk: int = 4000) -> None:
    """Crop all 16 diagnostic windows from the raw 100k-sample traces once."""
    from casca.data.windows import ALL_BYTES, BYTE_WINDOWS, WINDOW_LEN

    raw_path = Path(raw_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[data] Caching 16 x {WINDOW_LEN}-sample windows from {raw_path}", flush=True)
    with h5py.File(raw_path, "r") as src:
        n_traces = src["traces"].shape[0]
        windows = np.empty((n_traces, 16, WINDOW_LEN), dtype=np.int8)
        for start in range(0, n_traces, chunk):
            stop = min(start + chunk, n_traces)
            block = src["traces"][start:stop]
            for b in ALL_BYTES:
                s, e = BYTE_WINDOWS[b]
                windows[start:stop, b] = block[:, s:e]
            print(f"[data]   traces {start}:{stop}", flush=True)
        metadata = src["metadata"][:]
    plaintext = np.stack(metadata["plaintext"]).astype(np.uint8)
    key = np.stack(metadata["key"]).astype(np.uint8)
    np.savez_compressed(out_path, windows=windows, plaintext=plaintext, key=key)
    print(f"[data] wrote {out_path}  windows={windows.shape}", flush=True)


def load_per_byte_windows(npz_path: Path) -> dict:
    z = np.load(npz_path)
    return {
        "windows": z["windows"],
        "plaintext": z["plaintext"],
        "key": z["key"],
        "true_key": z["key"][0].astype(np.uint8),
    }
