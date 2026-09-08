"""CA-SCA CNN for ASCAD (paper Table 2, case study S-2).

Feature vector X is the 64-d SeLU activation immediately before the
256-way softmax classifier (Algorithm 3, line 4: X = M_θf(T)).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Table 2 reports 294.09 KB. With float32 that is exactly 75,288 parameters
# when every Conv1D / Linear layer has a bias (Keras default).
EXPECTED_N_PARAMS = 75288
EXPECTED_SIZE_KB = EXPECTED_N_PARAMS * 4 / 1024.0  # 294.09375


class Cascacnn(nn.Module):
    """Plaintext-labelled CNN from Table 2.

    Channels-first internally (PyTorch). Equivalent to the paper's
    channels-last Keras shapes (None, L, C).
    """

    def __init__(self) -> None:
        super().__init__()
        # valid padding: output lengths 700 -> 100 -> 20 -> 4
        self.conv1 = nn.Conv1d(1, 4, kernel_size=7, stride=7, padding=0)
        self.conv2 = nn.Conv1d(4, 8, kernel_size=5, stride=5, padding=0)
        self.conv3 = nn.Conv1d(8, 16, kernel_size=5, stride=5, padding=0)
        self.fc1 = nn.Linear(64, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.classifier = nn.Linear(64, 256)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Keras default for Conv1D/Dense is Glorot uniform + zero bias, even
        # when activation='selu'. The paper does not specify an initializer.
        for module in (self.conv1, self.conv2, self.conv3, self.fc1, self.fc2, self.fc3, self.classifier):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def extract_features(self, traces: torch.Tensor) -> torch.Tensor:
        """Return X, the Table-2 feature-extractor output, shape (N, 64)."""
        x = F.selu(self.conv1(traces))
        x = F.selu(self.conv2(x))
        x = F.selu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.selu(self.fc1(x))
        x = F.selu(self.fc2(x))
        x = F.selu(self.fc3(x))
        return x

    def forward(self, traces: torch.Tensor) -> torch.Tensor:
        """Logits of the 256-way plaintext classifier (no softmax).

        Training uses PyTorch CrossEntropyLoss, which applies log-softmax
        internally and is equivalent to Keras softmax + categorical cross-entropy.
        """
        return self.classifier(self.extract_features(traces))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def assert_architecture_matches_table2(model: Cascacnn | None = None) -> None:
    model = model or Cascacnn()
    n = model.n_params()
    if n != EXPECTED_N_PARAMS:
        raise AssertionError(
            f"Parameter count {n} != Table 2 ({EXPECTED_N_PARAMS}, {EXPECTED_SIZE_KB:.2f} KB)"
        )
    dummy = torch.zeros(2, 1, 700)
    with torch.no_grad():
        feats = model.extract_features(dummy)
        logits = model(dummy)
    if tuple(feats.shape) != (2, 64):
        raise AssertionError(f"Feature shape {tuple(feats.shape)} != (N, 64)")
    if tuple(logits.shape) != (2, 256):
        raise AssertionError(f"Logit shape {tuple(logits.shape)} != (N, 256)")
    # Intermediate conv lengths: 100, 20, 4
    with torch.no_grad():
        x = F.selu(model.conv1(dummy))
        if tuple(x.shape) != (2, 4, 100):
            raise AssertionError(f"conv1 out {tuple(x.shape)} != (N, 4, 100)")
        x = F.selu(model.conv2(x))
        if tuple(x.shape) != (2, 8, 20):
            raise AssertionError(f"conv2 out {tuple(x.shape)} != (N, 8, 20)")
        x = F.selu(model.conv3(x))
        if tuple(x.shape) != (2, 16, 4):
            raise AssertionError(f"conv3 out {tuple(x.shape)} != (N, 16, 4)")

"""Train the Table-2 CNN as a 256-way plaintext classifier (Algorithm 3)."""

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PAPER_EPOCHS = 200
PAPER_BATCH_SIZE = 20000
PAPER_LR = 1e-3


@dataclass
class TrainResult:
    model: nn.Module
    losses: list[float] = field(default_factory=list)
    seconds: float = 0.0
    n_steps: int = 0
    final_loss: float = float("nan")
    final_acc: float = float("nan")


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_plaintext_cnn(
    traces: np.ndarray,
    plaintext_byte: np.ndarray,
    seed: int,
    n_epochs: int = PAPER_EPOCHS,
    batch_size: int = PAPER_BATCH_SIZE,
    lr: float = PAPER_LR,
    log_every: int = 25,
) -> TrainResult:
    """Supervised training of M_θf, M_θc on (traces, plaintext).

    Paper Table 2: CrossEntropy, Adam(lr=0.001), batch 20000, 200 epochs.
    When N < 20000 the batch is min(20000, N) so there is still one step/epoch.
    """
    set_seed(seed)
    device = _device()
    n = len(traces)
    bs = min(int(batch_size), n)
    model = Cascacnn().to(device)
    # TF/Keras Adam uses epsilon=1e-7; PyTorch default is 1e-8.
    opt = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-7)
    loss_fn = torch.nn.CrossEntropyLoss()

    x = torch.from_numpy(np.ascontiguousarray(traces[:, None, :], dtype=np.float32))
    y = torch.from_numpy(np.ascontiguousarray(plaintext_byte, dtype=np.int64))
    loader = DataLoader(TensorDataset(x, y), batch_size=bs, shuffle=True, drop_last=False)

    print(
        f"[train] seed={seed} N={n} epochs={n_epochs} batch={bs} "
        f"device={device} params={model.n_params()}",
        flush=True,
    )
    t0 = time.time()
    losses: list[float] = []
    n_steps = 0
    model.train()
    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_n = 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            n_steps += 1
            epoch_loss += float(loss.item()) * len(yb)
            epoch_correct += int((logits.argmax(dim=1) == yb).sum().item())
            epoch_n += len(yb)
        mean_loss = epoch_loss / max(epoch_n, 1)
        acc = epoch_correct / max(epoch_n, 1)
        losses.append(mean_loss)
        if epoch == 1 or epoch == n_epochs or epoch % log_every == 0:
            print(
                f"[train]   epoch {epoch:3d}/{n_epochs}  loss={mean_loss:.4f}  acc={acc:.4f}",
                flush=True,
            )

    dt = time.time() - t0
    print(f"[train] done in {dt:.1f}s  steps={n_steps}  final_loss={losses[-1]:.4f}", flush=True)
    return TrainResult(
        model=model,
        losses=losses,
        seconds=dt,
        n_steps=n_steps,
        final_loss=losses[-1],
        final_acc=acc,
    )


@torch.no_grad()
def eval_loss_acc(
    model: nn.Module,
    traces: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
) -> tuple[float, float]:
    """Plaintext-classification loss/acc on an already-device-resident set."""
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    n = int(labels.shape[0])
    total_loss = 0.0
    correct = 0
    bs = min(int(batch_size), n)
    for i in range(0, n, bs):
        xb = traces[i : i + bs]
        yb = labels[i : i + bs]
        logits = model(xb)
        total_loss += float(loss_fn(logits, yb).item()) * len(yb)
        correct += int((logits.argmax(dim=1) == yb).sum().item())
    return total_loss / max(n, 1), correct / max(n, 1)


def train_plaintext_cnn_fast(
    traces: np.ndarray,
    plaintext_byte: np.ndarray,
    seed: int,
    n_epochs: int = PAPER_EPOCHS,
    batch_size: int = PAPER_BATCH_SIZE,
    lr: float = PAPER_LR,
    log_every: int = 25,
    model_fn: Callable[[], nn.Module] | None = None,
    val_traces: np.ndarray | None = None,
    val_labels: np.ndarray | None = None,
    checkpoint_epochs: set[int] | frozenset[int] | None = None,
    on_checkpoint: Callable[[int, dict, nn.Module], None] | None = None,
    max_steps: int | None = None,
    checkpoint_steps: set[int] | frozenset[int] | None = None,
    on_step_checkpoint: Callable[[int, dict, nn.Module], None] | None = None,
) -> TrainResult:
    """Same protocol as train_plaintext_cnn, with GPU-resident tensors.

    Used for concatenated 14-byte training sets that are too large for a
    naive DataLoader shuffle to be the bottleneck.

    Optional `val_*` / `checkpoint_epochs` / `on_checkpoint` /
    `checkpoint_steps` / `on_step_checkpoint` do not change the optimizer
    updates. Checkpoint-callback time is excluded from `TrainResult.seconds`
    (only gradient steps count). Step checkpoints fire after that many
    optimizer updates, including mid-epoch; they do not alter the architecture
    or the update rule.

    If `max_steps` is set, training stops after that many optimizer updates,
    even mid-epoch. `n_epochs` is then only a ceiling.

    Steps per epoch is `ceil(n_windows / min(batch_size, n_windows))` with
    no `drop_last` (a smaller final batch is included when n_windows is not
    a multiple of batch_size). For the concatenated 14-byte set that is
    `ceil(14N / min(20000, 14N))`.
    """
    set_seed(seed)
    device = _device()
    n = len(traces)
    bs = min(int(batch_size), n)
    model = (model_fn or Cascacnn)().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-7)
    loss_fn = torch.nn.CrossEntropyLoss()

    x = torch.from_numpy(np.ascontiguousarray(traces[:, None, :], dtype=np.float32)).to(device)
    y = torch.from_numpy(np.ascontiguousarray(plaintext_byte, dtype=np.int64)).to(device)
    val_x = val_y = None
    if val_traces is not None:
        if val_labels is None:
            raise ValueError("val_labels required when val_traces is set")
        val_x = torch.from_numpy(np.ascontiguousarray(val_traces[:, None, :], dtype=np.float32)).to(device)
        val_y = torch.from_numpy(np.ascontiguousarray(val_labels, dtype=np.int64)).to(device)

    print(
        f"[train] seed={seed} N={n} epochs={n_epochs} batch={bs} "
        f"device={device} params={model.n_params()} (gpu-resident)"
        + (f"  max_steps={max_steps}" if max_steps is not None else "")
        + (f"  val_N={int(val_y.shape[0])}" if val_y is not None else ""),
        flush=True,
    )
    t_resume = time.time()
    train_seconds = 0.0
    losses: list[float] = []
    n_steps = 0
    acc = 0.0
    last_loss = float("nan")
    ckpt = set(checkpoint_epochs) if checkpoint_epochs is not None else set()
    step_ckpt = set(checkpoint_steps) if checkpoint_steps is not None else set()
    model.train()
    hit_step_limit = False

    def _emit(callback: Callable[[int, dict, nn.Module], None], key: int, epoch: int) -> None:
        nonlocal train_seconds, t_resume
        train_seconds += time.time() - t_resume
        denom = epoch_seen if epoch_seen else 1
        snap = {
            "epoch": epoch,
            "step": n_steps,
            "train_loss": epoch_loss / denom,
            "train_acc": epoch_correct / denom,
            "train_loss_last": last_loss,
            "val_loss": float("nan"),
            "val_acc": float("nan"),
            "cumulative_train_seconds": train_seconds,
        }
        if val_x is not None and val_y is not None:
            vloss, vacc = eval_loss_acc(model, val_x, val_y, bs)
            snap["val_loss"] = vloss
            snap["val_acc"] = vacc
        was_training = model.training
        model.eval()
        callback(key, snap, model)
        if was_training:
            model.train()
        t_resume = time.time()

    for epoch in range(1, n_epochs + 1):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_seen = 0
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            opt.zero_grad(set_to_none=True)
            logits = model(x.index_select(0, idx))
            yb = y.index_select(0, idx)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            n_steps += 1
            last_loss = float(loss.item())
            epoch_loss += last_loss * len(idx)
            epoch_correct += int((logits.argmax(dim=1) == yb).sum().item())
            epoch_seen += len(idx)
            if on_step_checkpoint is not None and n_steps in step_ckpt:
                _emit(on_step_checkpoint, n_steps, epoch)
            if max_steps is not None and n_steps >= max_steps:
                hit_step_limit = True
                break
        denom = epoch_seen if epoch_seen else n
        mean_loss = epoch_loss / denom
        acc = epoch_correct / denom
        losses.append(mean_loss)
        if epoch == 1 or epoch == n_epochs or epoch % log_every == 0 or hit_step_limit:
            print(
                f"[train]   epoch {epoch:3d}/{n_epochs}  loss={mean_loss:.4f}  acc={acc:.4f}",
                flush=True,
            )
        if on_checkpoint is not None and epoch in ckpt:
            _emit(on_checkpoint, epoch, epoch)
        if hit_step_limit:
            break

    train_seconds += time.time() - t_resume
    print(f"[train] done in {train_seconds:.1f}s  steps={n_steps}  final_loss={losses[-1]:.4f}", flush=True)
    return TrainResult(
        model=model,
        losses=losses,
        seconds=train_seconds,
        n_steps=n_steps,
        final_loss=losses[-1],
        final_acc=acc,
    )


@torch.no_grad()
def extract_features(model: nn.Module, traces: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """X = M_θf(T), shape (N, 64)."""
    model.eval()
    device = next(model.parameters()).device
    chunks = []
    for start in range(0, len(traces), batch_size):
        xb = torch.from_numpy(np.ascontiguousarray(traces[start : start + batch_size, None, :], dtype=np.float32))
        xb = xb.to(device)
        chunks.append(model.extract_features(xb).cpu().numpy())
    return np.concatenate(chunks, axis=0)
