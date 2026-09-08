"""H2: Table-2 conv stack plus multiplicative attention (documented H2Net).

Source of the operator (not copied as a training loop):
  magisterka/casca_h2.py module docstring and H2Net.

    conv  (B, 16, T)
    tokens (B, T, 16)
    Q,K,V,G = Linear_16→16(tokens)     # four independent projections, bias on
    A = softmax(Q K^T / sqrt(16), dim=-1)
    comb = (A V) ⊙ G                   # no residual, no sigmoid on G
    aout = Linear(T·16 → 64)(flatten(comb))
    feat = FC 64→256→128→64 + SeLU     # clustered 64-d vector
    head = FC 64→256                   # training only

`aout` is not in the high-level sketch (for T=4, T·16 is already 64) but is
part of the documented H2Net: a learned mix of the gated tokens before the
feature MLP. For the token-resolution ablation it becomes Linear(T·16, 64).

Init is LeCun-normal (SELU-correct), matching documented H2Net.selu_init —
not the Glorot-uniform used for the paper CNN (Keras default). Dropout is 0.

No positional encoding, no extra heads in the documented H2Net. Optional
`use_posenc` / `use_residual` / `use_gate` / `self_multiply` flags (defaults
preserve documented H2Net) are screening additions. They must not change
the default `make_h2("B")` / `make_h2("E")` path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

FEATURE_DIM = 64
ATTN_DH = 16
CONV_CHANNELS = (4, 8, 16)


@dataclass(frozen=True)
class ConvConfig:
    name: str
    kernels: tuple[int, int, int]
    strides: tuple[int, int, int]
    note: str


# Token counts below are measured with valid padding on 700-sample input.
CONV_CONFIGS = {
    "A": ConvConfig(
        "A",
        kernels=(7, 5, 5),
        strides=(7, 5, 5),
        note="baseline H2: 4 tokens, ~175 samples/token",
    ),
    "B": ConvConfig(
        "B",
        kernels=(7, 5, 5),
        strides=(7, 5, 2),
        note="medium: last stride 5→2. Valid padding gives 8 tokens (not 10).",
    ),
    "C": ConvConfig(
        "C",
        kernels=(5, 2, 2),
        strides=(5, 2, 2),
        note="high-res: kernels match strides so 700 divides cleanly to 35 tokens.",
    ),
    # Fine-grid between B (8) and C (35). Layer-1 stays A/B (k=7,s=7); kernels
    # stay (7,5,5); only later strides change. 24 is not achievable; 23 is closest.
    "D": ConvConfig(
        "D",
        kernels=(7, 5, 5),
        strides=(7, 5, 1),
        note="fine-grid: last stride 2→1. Valid padding gives exactly 16 tokens.",
    ),
    "E": ConvConfig(
        "E",
        kernels=(7, 5, 5),
        strides=(7, 1, 4),
        note="fine-grid: strides (7,1,4). Valid padding gives 23 tokens (target 24).",
    ),
    "F": ConvConfig(
        "F",
        kernels=(7, 5, 5),
        strides=(7, 1, 5),
        note="bracket peak: strides (7,1,5). Valid padding gives exactly 19 tokens.",
    ),
    "G": ConvConfig(
        "G",
        kernels=(7, 5, 5),
        strides=(7, 3, 1),
        note="bracket peak: strides (7,3,1). Valid padding gives exactly 28 tokens.",
    ),
}


class _ContiguousGrad(torch.autograd.Function):
    """Identity forward; contiguous incoming gradient. MPS layout fix only."""

    @staticmethod
    def forward(ctx, t):  # noqa: ANN001
        return t

    @staticmethod
    def backward(ctx, g):  # noqa: ANN001
        return g.contiguous()


def _lecun_init(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def measure_tokens(kernels: tuple[int, int, int], strides: tuple[int, int, int], length: int = 700) -> int:
    conv = nn.Sequential(
        nn.Conv1d(1, 4, kernels[0], strides[0]),
        nn.Conv1d(4, 8, kernels[1], strides[1]),
        nn.Conv1d(8, 16, kernels[2], strides[2]),
    )
    with torch.no_grad():
        return int(conv(torch.zeros(1, 1, length)).shape[-1])


class H2Net(nn.Module):
    """Documented multiplicative-attention CA-SCA model. Token count is not hardcoded."""

    def __init__(
        self,
        input_length: int = 700,
        kernels: tuple[int, int, int] = (7, 5, 5),
        strides: tuple[int, int, int] = (7, 5, 5),
        dh: int = ATTN_DH,
        use_posenc: bool = False,
        use_residual: bool = False,
        use_gate: bool = True,
        self_multiply: bool = False,
    ) -> None:
        super().__init__()
        if use_posenc and use_residual:
            raise ValueError("Screening variants are one addition at a time.")
        if self_multiply and use_gate:
            raise ValueError("self_multiply replaces the gate; set use_gate=False.")
        self.C = CONV_CHANNELS[-1]
        self.dh = dh
        self.kernels = kernels
        self.strides = strides
        self.use_posenc = use_posenc
        self.use_residual = use_residual
        self.use_gate = use_gate
        self.self_multiply = self_multiply
        self.conv = nn.Sequential(
            nn.Conv1d(1, CONV_CHANNELS[0], kernels[0], strides[0]),
            nn.SELU(),
            nn.Conv1d(CONV_CHANNELS[0], CONV_CHANNELS[1], kernels[1], strides[1]),
            nn.SELU(),
            nn.Conv1d(CONV_CHANNELS[1], CONV_CHANNELS[2], kernels[2], strides[2]),
            nn.SELU(),
        )
        with torch.no_grad():
            self.n_tokens = int(self.conv(torch.zeros(1, 1, input_length)).shape[-1])
        self.q = nn.Linear(self.C, dh)
        self.k = nn.Linear(self.C, dh)
        self.v = nn.Linear(self.C, dh)
        # Gate: raw Linear(16→16), no sigmoid. Ablation can drop it entirely.
        self.g = nn.Linear(self.C, dh) if use_gate else None
        self.aout = nn.Linear(self.n_tokens * dh, FEATURE_DIM)
        self.fc1 = nn.Linear(FEATURE_DIM, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, FEATURE_DIM)
        self.classifier = nn.Linear(FEATURE_DIM, 256)
        _lecun_init(self)
        # Learned PE, zeros so the network starts as Config B until PE moves.
        if use_posenc:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.n_tokens, self.C))
        else:
            self.pos_embed = None
        if use_residual and self.C != dh:
            raise ValueError(
                f"Residual needs C==dh for a parameter-free add; got C={self.C} dh={dh}"
            )

    def _interact(self, tok: torch.Tensor) -> torch.Tensor:
        # tok: (B, T, C)  — pre-attention conv feature map
        conv_tok = tok
        if self.use_posenc:
            tok = tok + self.pos_embed
        b, t, _ = tok.shape
        q, k, v = self.q(tok), self.k(tok), self.v(tok)
        attn = torch.softmax(q @ k.transpose(1, 2) / math.sqrt(self.dh), dim=-1)
        att = attn @ v
        if self.use_gate:
            att = att * self.g(tok)
        elif self.self_multiply:
            att = att * att
        if self.use_residual:
            att = att + conv_tok
        return self.aout(att.reshape(b, t * self.dh))

    def extract_features(self, traces: torch.Tensor) -> torch.Tensor:
        fmap = _ContiguousGrad.apply(self.conv(traces))
        h = self._interact(fmap.transpose(1, 2))
        h = F.selu(self.fc1(h))
        h = F.selu(self.fc2(h))
        h = F.selu(self.fc3(h))
        return h

    def forward(self, traces: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(traces))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def make_h2(
    config_name: str = "A",
    *,
    use_posenc: bool = False,
    use_residual: bool = False,
    use_gate: bool = True,
    self_multiply: bool = False,
) -> H2Net:
    cfg = CONV_CONFIGS[config_name]
    return H2Net(
        kernels=cfg.kernels,
        strides=cfg.strides,
        use_posenc=use_posenc,
        use_residual=use_residual,
        use_gate=use_gate,
        self_multiply=self_multiply,
    )


def describe_configs(length: int = 700) -> list[dict]:
    rows = []
    for name, cfg in CONV_CONFIGS.items():
        n_tok = measure_tokens(cfg.kernels, cfg.strides, length)
        model = H2Net(length, cfg.kernels, cfg.strides)
        rows.append(
            {
                "config": name,
                "kernels": cfg.kernels,
                "strides": cfg.strides,
                "n_tokens": n_tok,
                "samples_per_token": length / n_tok,
                "n_params": model.n_params(),
                "attn_matrix": f"{n_tok}x{n_tok}",
                "note": cfg.note,
            }
        )
    return rows

"""Attention-combination screen models on the CA-SCA 4-token conv stack.

Does not change Cascacnn or H2Net defaults. Two-head variants use d_h=8 per
head so total attention width stays 16 (comparable to single-head d_h=16).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from casca.attacks.casca import Cascacnn

KERNELS = (7, 5, 5)
STRIDES = (7, 5, 5)
HEAD_DH = 8
N_HEADS = 2
VARIANT_ORDER = (
    "baseline",
    "square_only",
    "attn_only",
    "attn_square",
    "twohead_prod",
    "twohead_concat",
)


def _casca_conv() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(1, CONV_CHANNELS[0], KERNELS[0], STRIDES[0]),
        nn.SELU(),
        nn.Conv1d(CONV_CHANNELS[0], CONV_CHANNELS[1], KERNELS[1], STRIDES[1]),
        nn.SELU(),
        nn.Conv1d(CONV_CHANNELS[1], CONV_CHANNELS[2], KERNELS[2], STRIDES[2]),
        nn.SELU(),
    )


class SquareOnlyNet(nn.Module):
    """Conv tokens squared element-wise, flatten to 64, CA-SCA classifier."""

    def __init__(self, input_length: int = 700) -> None:
        super().__init__()
        self.conv = _casca_conv()
        with torch.no_grad():
            self.n_tokens = int(self.conv(torch.zeros(1, 1, input_length)).shape[-1])
        if self.n_tokens != 4:
            raise ValueError(f"square_only expected 4 tokens, got {self.n_tokens}")
        flat = self.n_tokens * CONV_CHANNELS[-1]
        self.fc1 = nn.Linear(flat, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, FEATURE_DIM)
        self.classifier = nn.Linear(FEATURE_DIM, 256)
        _lecun_init(self)

    def extract_features(self, traces: torch.Tensor) -> torch.Tensor:
        fmap = _ContiguousGrad.apply(self.conv(traces))
        tok = fmap.transpose(1, 2)
        tok = tok * tok
        h = tok.reshape(tok.size(0), -1)
        h = F.selu(self.fc1(h))
        h = F.selu(self.fc2(h))
        h = F.selu(self.fc3(h))
        return h

    def forward(self, traces: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(traces))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class TwoHeadNet(nn.Module):
    """Two heads, d_h=8 each. mode='prod' or 'concat'."""

    def __init__(self, mode: str, input_length: int = 700) -> None:
        super().__init__()
        if mode not in ("prod", "concat"):
            raise ValueError(mode)
        self.mode = mode
        self.dh = HEAD_DH
        self.conv = _casca_conv()
        with torch.no_grad():
            self.n_tokens = int(self.conv(torch.zeros(1, 1, input_length)).shape[-1])
        if self.n_tokens != 4:
            raise ValueError(f"two-head expected 4 tokens, got {self.n_tokens}")
        c = CONV_CHANNELS[-1]
        self.q = nn.ModuleList([nn.Linear(c, HEAD_DH) for _ in range(N_HEADS)])
        self.k = nn.ModuleList([nn.Linear(c, HEAD_DH) for _ in range(N_HEADS)])
        self.v = nn.ModuleList([nn.Linear(c, HEAD_DH) for _ in range(N_HEADS)])
        aout_in = self.n_tokens * (HEAD_DH if mode == "prod" else HEAD_DH * N_HEADS)
        self.aout = nn.Linear(aout_in, FEATURE_DIM)
        self.fc1 = nn.Linear(FEATURE_DIM, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, FEATURE_DIM)
        self.classifier = nn.Linear(FEATURE_DIM, 256)
        _lecun_init(self)

    def _one_head(self, tok: torch.Tensor, i: int) -> torch.Tensor:
        q, k, v = self.q[i](tok), self.k[i](tok), self.v[i](tok)
        attn = torch.softmax(q @ k.transpose(1, 2) / math.sqrt(self.dh), dim=-1)
        return attn @ v

    def extract_features(self, traces: torch.Tensor) -> torch.Tensor:
        fmap = _ContiguousGrad.apply(self.conv(traces))
        tok = fmap.transpose(1, 2)
        h1, h2 = self._one_head(tok, 0), self._one_head(tok, 1)
        if self.mode == "prod":
            comb = h1 * h2
        else:
            comb = torch.cat((h1, h2), dim=-1)
        h = self.aout(comb.reshape(comb.size(0), -1))
        h = F.selu(self.fc1(h))
        h = F.selu(self.fc2(h))
        h = F.selu(self.fc3(h))
        return h

    def forward(self, traces: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(traces))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def make_variant(name: str) -> nn.Module:
    if name == "baseline":
        return Cascacnn()
    if name == "square_only":
        return SquareOnlyNet()
    if name == "attn_only":
        return make_h2("A", use_gate=False)
    if name == "attn_square":
        return make_h2("A", use_gate=False, self_multiply=True)
    if name == "twohead_prod":
        return TwoHeadNet("prod")
    if name == "twohead_concat":
        return TwoHeadNet("concat")
    raise ValueError(f"unknown variant {name}")
