import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """Precomputes cos/sin tables; returns slices of shape (T, head_dim)."""

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10_000.0):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"

        exponents = torch.arange(0, head_dim, 2).float() / head_dim
        inv_freq = 1.0 / (theta**exponents)  # (D/2,)
        positions = torch.arange(max_seq_len).float()  # (L,)

        freqs = torch.outer(positions, inv_freq)  # (L, D/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (L, D)

        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        assert seq_len <= self.cos.shape[0], "sequence longer than max_seq_len"
        cos = self.cos[:seq_len]
        sin = self.sin[:seq_len]
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, D); cos/sin: (T, D). Computed in fp32, cast back to x's dtype."""
    x_float = x.float()
    rotated = rotate_half(x_float)
    out = x_float * cos + rotated * sin
    return out.type_as(x)
