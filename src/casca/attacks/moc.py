"""MOC network: one shared layer, then 256 DDLA-identical hypothesis branches.

Per-branch MLP is Timon TCHES 2019 Appendix A (MLPexp), which MOC states it
reuses:

    Dense(20) + ReLU  ->  Dense(10) + ReLU  ->  Dense(2) + softmax

The 256 branches are stored as stacked tensors and applied with einsum.
That is algebraically identical to 256 independent Sequential modules and
is the only practical way to train them jointly.

UNCONFIRMED (flagged — not an established architectural fact):
    Secondary MOC sources describe "a shared layer followed by k branches"
    (singular) but do not give the shared layer's width or depth from a
    primary source available to this project. This implementation uses a
    SINGLE shared Dense with width=20 (matching the first MLPexp hidden
    layer) and ReLU, sitting between the raw (normalized) trace and the
    256 branches. Reasonable default, not confirmed from a primary source.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Confirmed (Timon Appendix A / MOC reuse of MLPexp).
BRANCH_H1 = 20
BRANCH_H2 = 10
N_CLASSES = 2
N_HYPOTHESES = 256

# UNCONFIRMED default: shared-layer width. See module docstring.
# Reasonable default, not confirmed from a primary source.
SHARED_WIDTH_UNCONFIRMED_DEFAULT = 20


def ddla_mlp_n_params(input_len: int) -> int:
    """Parameter count of one independent MLPexp network on ``input_len`` samples."""
    d20 = input_len * BRANCH_H1 + BRANCH_H1
    d10 = BRANCH_H1 * BRANCH_H2 + BRANCH_H2
    d2 = BRANCH_H2 * N_CLASSES + N_CLASSES
    return d20 + d10 + d2


def independent_ddla_n_params(input_len: int, n_hyp: int = N_HYPOTHESES) -> int:
    """What 256 fully separate DDLA networks would cost (MOC's efficiency claim)."""
    return n_hyp * ddla_mlp_n_params(input_len)


class MOC(nn.Module):
    """Shared trunk + 256 LSB-classification branches.

    Forward returns softmax probabilities of shape ``(batch, 256, 2)``.
    """

    def __init__(
        self,
        input_len: int,
        shared_width: int = SHARED_WIDTH_UNCONFIRMED_DEFAULT,
        n_hyp: int = N_HYPOTHESES,
    ) -> None:
        super().__init__()
        self.input_len = int(input_len)
        self.shared_width = int(shared_width)
        self.n_hyp = int(n_hyp)
        # UNCONFIRMED: single Dense(shared_width)+ReLU as the entire shared
        # component. Width default 20 is a reasonable default, not confirmed
        # from a primary source. See module docstring.
        self.shared = nn.Linear(self.input_len, self.shared_width)
        # Branch weights: (K, out, in), independent Linear per hypothesis.
        self.b1_weight = nn.Parameter(torch.empty(self.n_hyp, BRANCH_H1, self.shared_width))
        self.b1_bias = nn.Parameter(torch.empty(self.n_hyp, BRANCH_H1))
        self.b2_weight = nn.Parameter(torch.empty(self.n_hyp, BRANCH_H2, BRANCH_H1))
        self.b2_bias = nn.Parameter(torch.empty(self.n_hyp, BRANCH_H2))
        self.b3_weight = nn.Parameter(torch.empty(self.n_hyp, N_CLASSES, BRANCH_H2))
        self.b3_bias = nn.Parameter(torch.empty(self.n_hyp, N_CLASSES))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Keras Dense default (Timon's stack): Glorot uniform, zero bias.
        nn.init.xavier_uniform_(self.shared.weight)
        nn.init.zeros_(self.shared.bias)
        for w, b in (
            (self.b1_weight, self.b1_bias),
            (self.b2_weight, self.b2_bias),
            (self.b3_weight, self.b3_bias),
        ):
            for k in range(self.n_hyp):
                nn.init.xavier_uniform_(w.data[k])
                nn.init.zeros_(b.data[k])

    def forward(self, traces: torch.Tensor) -> torch.Tensor:
        """Softmax probabilities, shape ``(B, 256, 2)``."""
        # traces: (B, L)
        h = F.relu(self.shared(traces))
        # Each einsum is Linear: y[k] = h[k] @ W[k].T + b[k]
        h = F.relu(torch.einsum("bi,koi->bko", h, self.b1_weight) + self.b1_bias)
        h = F.relu(torch.einsum("bki,koi->bko", h, self.b2_weight) + self.b2_bias)
        logits = torch.einsum("bki,koi->bko", h, self.b3_weight) + self.b3_bias
        return F.softmax(logits, dim=-1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def efficiency_report(self) -> dict:
        moc = self.n_params()
        ddla = independent_ddla_n_params(self.input_len, self.n_hyp)
        shared = self.shared.weight.numel() + self.shared.bias.numel()
        branches = moc - shared
        return {
            "moc_n_params": moc,
            "independent_ddla_n_params": ddla,
            "ratio_moc_over_ddla": moc / ddla,
            "shared_n_params": shared,
            "branches_n_params": branches,
            "input_len": self.input_len,
            "shared_width": self.shared_width,
            "shared_width_confirmed": False,
            "shared_width_note": (
                "reasonable default, not confirmed from primary source"
            ),
        }


def assert_moc_shapes(model: MOC | None = None, input_len: int = 700) -> None:
    model = model or MOC(input_len)
    dummy = torch.zeros(3, input_len)
    with torch.no_grad():
        out = model(dummy)
    if tuple(out.shape) != (3, N_HYPOTHESES, N_CLASSES):
        raise AssertionError(f"MOC output {tuple(out.shape)} != (3, 256, 2)")
    if out.min() < 0 or out.max() > 1:
        raise AssertionError("softmax outputs out of [0, 1]")
    row_sums = out.sum(dim=-1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5):
        raise AssertionError("softmax rows do not sum to 1")
    expected = (
        input_len * model.shared_width
        + model.shared_width
        + N_HYPOTHESES
        * (
            model.shared_width * BRANCH_H1
            + BRANCH_H1
            + BRANCH_H1 * BRANCH_H2
            + BRANCH_H2
            + BRANCH_H2 * N_CLASSES
            + N_CLASSES
        )
    )
    if model.n_params() != expected:
        raise AssertionError(f"param count {model.n_params()} != {expected}")

"""Train MOC with Timon's DDLA protocol (carried over as MOC's default).

Confirmed from Timon TCHES 2019, reused because MOC reuses MLPexp and does
not state different training hyperparameters in the secondary sources:

    loss          MSE on the 2-class softmax (not cross-entropy)
    batch size    1000
    optimizer     Adam(lr=0.001, betas=(0.9, 0.999), eps=1e-8), no LR decay
    epochs        50 (Timon's ASCAD reference; even 5 succeeded for him)
    distinguisher per-branch training accuracy (highest accuracy = key byte)

Loss reduction: SUM of the 256 per-branch MSEs (each branch MSE = mean over
batch and the 2 class dimensions). Sum vs mean changes the effective
learning rate per branch and may need tuning later. Sum is the right
starting point: each branch's own parameters then see the same gradient
scale as a standalone DDLA network trained with MSE; the shared layer
receives the sum of all 256 branch gradients (joint backprop).
"""

import time
from dataclasses import dataclass, field

import numpy as np
import torch

MOC_EPOCHS = 50
MOC_BATCH_SIZE = 1000
MOC_LR = 1e-3
MOC_BETAS = (0.9, 0.999)
MOC_EPS = 1e-8


@dataclass
class MOCTrainResult:
    model: MOC
    losses: list[float] = field(default_factory=list)
    acc_correct: list[float] = field(default_factory=list)
    acc_wrong_mean: list[float] = field(default_factory=list)
    acc_wrong_std: list[float] = field(default_factory=list)
    acc_per_hyp_final: np.ndarray | None = None
    seconds: float = 0.0
    n_steps: int = 0
    n_params: int = 0
    efficiency: dict = field(default_factory=dict)


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


def _sum_of_branch_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Sum over hypotheses of per-branch mean-squared error.

    ``pred``/``target``: (B, 256, 2). Each branch MSE averages over batch and
    the 2 softmax bins (Keras ``mse`` on a single DDLA head). The 256 values
    are then summed, not averaged — see module docstring.
    """
    # (sum over B, K, C of sq) / (B * 2)  ==  sum_k mean_{b,c} sq[b,k,c]
    sq = (pred - target) ** 2
    return sq.sum() / (sq.shape[0] * sq.shape[2])


def train_moc(
    traces: np.ndarray,
    onehot: np.ndarray,
    true_key: int,
    seed: int,
    n_epochs: int = MOC_EPOCHS,
    batch_size: int = MOC_BATCH_SIZE,
    lr: float = MOC_LR,
    log_every: int = 10,
    verbose: bool = True,
) -> MOCTrainResult:
    """Jointly train the shared layer and 256 LSB branches.

    ``traces``: (N, L) already mean-removed and scaled to [-1, 1].
    ``onehot``: (N, 256, 2) LSB-of-Sbox(p XOR k) one-hots.
    """
    set_seed(seed)
    device = _device()
    n, input_len = traces.shape
    bs = min(int(batch_size), n)
    model = MOC(input_len).to(device)
    efficiency = model.efficiency_report()
    opt = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=MOC_BETAS,
        eps=MOC_EPS,
    )

    x = torch.from_numpy(np.ascontiguousarray(traces, dtype=np.float32)).to(device)
    y = torch.from_numpy(np.ascontiguousarray(onehot, dtype=np.float32)).to(device)
    y_cls = y.argmax(dim=-1)  # (N, 256)

    if verbose:
        print(
            f"[moc] seed={seed} N={n} L={input_len} epochs={n_epochs} batch={bs} "
            f"device={device}",
            flush=True,
        )
        print(
            f"[moc] params: MOC={efficiency['moc_n_params']:,}  "
            f"256 independent DDLA={efficiency['independent_ddla_n_params']:,}  "
            f"ratio={efficiency['ratio_moc_over_ddla']:.4f}  "
            f"(shared={efficiency['shared_n_params']:,}, "
            f"branches={efficiency['branches_n_params']:,})",
            flush=True,
        )
        print(
            f"[moc] UNCONFIRMED shared-layer width={efficiency['shared_width']} "
            f"({efficiency['shared_width_note']})",
            flush=True,
        )
        print(
            f"[moc] loss=sum of 256 per-branch MSEs  Adam(lr={lr}, betas={MOC_BETAS}, "
            f"eps={MOC_EPS})  true_key=0x{true_key:02x}",
            flush=True,
        )

    t0 = time.time()
    losses: list[float] = []
    acc_correct_hist: list[float] = []
    acc_wrong_mean_hist: list[float] = []
    acc_wrong_std_hist: list[float] = []
    n_steps = 0
    wrong_idx = np.ones(256, dtype=bool)
    wrong_idx[true_key] = False

    model.train()
    for epoch in range(1, n_epochs + 1):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        correct_counts = torch.zeros(256, device=device)
        for start in range(0, n, bs):
            idx = perm[start : start + bs]
            opt.zero_grad(set_to_none=True)
            pred = model(x.index_select(0, idx))
            yb = y.index_select(0, idx)
            loss = _sum_of_branch_mse(pred, yb)
            loss.backward()
            opt.step()
            n_steps += 1
            epoch_loss += float(loss.item()) * len(idx)
            pred_cls = pred.argmax(dim=-1)
            yb_cls = y_cls.index_select(0, idx)
            correct_counts += (pred_cls == yb_cls).sum(dim=0).float()

        mean_loss = epoch_loss / n
        acc_per_hyp = (correct_counts / n).detach().cpu().numpy()
        acc_c = float(acc_per_hyp[true_key])
        acc_w = acc_per_hyp[wrong_idx]
        acc_w_mean = float(acc_w.mean())
        acc_w_std = float(acc_w.std(ddof=1)) if len(acc_w) > 1 else 0.0
        losses.append(mean_loss)
        acc_correct_hist.append(acc_c)
        acc_wrong_mean_hist.append(acc_w_mean)
        acc_wrong_std_hist.append(acc_w_std)

        if verbose and (epoch == 1 or epoch == n_epochs or epoch % log_every == 0):
            gap = acc_c - acc_w_mean
            print(
                f"[moc]   epoch {epoch:3d}/{n_epochs}  loss={mean_loss:.4f}  "
                f"acc_correct={acc_c:.4f}  acc_wrong={acc_w_mean:.4f}±{acc_w_std:.4f}  "
                f"gap={gap:+.4f}",
                flush=True,
            )

    dt = time.time() - t0
    recovered = int(np.argmax(acc_per_hyp))
    rank = int(np.sum(acc_per_hyp > acc_per_hyp[true_key]))
    if verbose:
        print(
            f"[moc] done in {dt:.1f}s  steps={n_steps}  "
            f"final acc_correct={acc_correct_hist[-1]:.4f}  "
            f"argmax_k=0x{recovered:02x}  rank_of_true={rank}",
            flush=True,
        )
    return MOCTrainResult(
        model=model,
        losses=losses,
        acc_correct=acc_correct_hist,
        acc_wrong_mean=acc_wrong_mean_hist,
        acc_wrong_std=acc_wrong_std_hist,
        acc_per_hyp_final=acc_per_hyp,
        seconds=dt,
        n_steps=n_steps,
        n_params=model.n_params(),
        efficiency=efficiency,
    )


"""MOC-specific labels and input normalization.

The CA-SCA CNN pipeline z-scores traces for SeLU scale. MOC / DDLA do not:
Timon (TCHES 2019) mean-removes each sample and scales to [-1, 1]. That
transform is implemented here and must not be replaced by z-score.
"""

import numpy as np

from casca.data.ascad import SBOX


def mean_remove_scale_pm1(
    train_traces: np.ndarray, *others: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Mean-removed, scaled to [-1, 1], statistics from ``train_traces`` only.

    Timon: "the mean of each sample is removed, then the traces are scaled
    to have values in [-1, 1]." Sample = time sample (column). Mean is
    per-timepoint over the training traces; scale is the max-abs of the
    mean-removed training set so training values lie in [-1, 1]. The same
    affine transform is applied to any extra arrays.
    """
    x = np.ascontiguousarray(train_traces, dtype=np.float32)
    mean = x.mean(axis=0, keepdims=True)
    centered = x - mean
    scale = float(np.max(np.abs(centered)))
    if scale < 1e-12:
        scale = 1.0
    out = [centered / scale]
    for arr in others:
        a = np.ascontiguousarray(arr, dtype=np.float32)
        out.append((a - mean) / scale)
    return tuple(out)


def lsb_sbox_labels(plaintext_byte: np.ndarray) -> np.ndarray:
    """LSB of Sbox(p XOR k_hyp) for every key-byte hypothesis.

    Shape ``(N, 256)``, values in {0, 1}.

    Timon used LSB specifically for the ASCAD attack (MSB was used on other
    datasets in the same paper). MOC reuses DDLA's labelling.
    """
    p = np.asarray(plaintext_byte, dtype=np.uint8).reshape(-1, 1)
    k = np.arange(256, dtype=np.uint8).reshape(1, -1)
    return (SBOX[p ^ k] & np.uint8(1)).astype(np.int64)


def lsb_onehot(plaintext_byte: np.ndarray) -> np.ndarray:
    """One-hot of :func:`lsb_sbox_labels`, shape ``(N, 256, 2)``, float32."""
    lsb = lsb_sbox_labels(plaintext_byte)
    return np.eye(2, dtype=np.float32)[lsb]
