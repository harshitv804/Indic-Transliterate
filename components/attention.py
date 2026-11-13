import torch
import torch.nn.functional as F
from torch import nn

from config import Config

from .liger_ops import RMSNorm
from .rope import apply_rope


def _merge_causal_mask(
    attn_mask: torch.Tensor,
    q_len: int,
    k_len: int,
) -> torch.Tensor:
    # SDPA rejects attn_mask + is_causal=True, so merge them. Diagonal stays
    # visible so a fully masked query row cannot produce NaNs.
    q_idx = torch.arange(q_len, device=attn_mask.device)[:, None]
    k_idx = torch.arange(k_len, device=attn_mask.device)[None, :]
    causal = q_idx >= k_idx

    if attn_mask.dtype == torch.bool:
        merged = attn_mask & causal
        merged.diagonal(dim1=-2, dim2=-1).fill_(True)
    else:
        merged = attn_mask.masked_fill(~causal, float("-inf"))
        merged.diagonal(dim1=-2, dim2=-1).fill_(0.0)

    return merged


class GatedMultiHeadAttention(nn.Module):
    """Standard MHA + SDPA + elementwise sigmoid output gate."""

    def __init__(self, cfg: Config):
        super().__init__()

        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.dropout = cfg.dropout

        q_dim = self.n_heads * self.head_dim

        self.q_proj = nn.Linear(cfg.dim, q_dim, bias=False)
        self.k_proj = nn.Linear(cfg.dim, q_dim, bias=False)
        self.v_proj = nn.Linear(cfg.dim, q_dim, bias=False)

        # One gate per query head/channel.
        self.gate_proj = nn.Linear(cfg.dim, q_dim, bias=False)

        self.o_proj = nn.Linear(q_dim, cfg.dim, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=cfg.norm_eps) if cfg.qk_norm else None

        self.k_norm = RMSNorm(self.head_dim, eps=cfg.norm_eps) if cfg.qk_norm else None

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        rope: tuple[torch.Tensor, torch.Tensor] | None = None,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:

        B, T, _ = x.shape

        kv_input = x if context is None else context
        S = kv_input.shape[1]

        # -------------------------------------------------- #
        # Q, K, V projections
        # -------------------------------------------------- #

        q = self.q_proj(x)
        k = self.k_proj(kv_input)
        v = self.v_proj(kv_input)

        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, S, self.n_heads, self.head_dim)
        v = v.view(B, S, self.n_heads, self.head_dim)

        # -------------------------------------------------- #
        # QK normalization
        # -------------------------------------------------- #

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # -------------------------------------------------- #
        # SDPA layout: (B, H, T, D)
        # -------------------------------------------------- #

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # -------------------------------------------------- #
        # RoPE
        # -------------------------------------------------- #

        if rope is not None:
            cos, sin = rope
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        # -------------------------------------------------- #
        # Standard Multi-Head Attention
        # -------------------------------------------------- #

        if is_causal and attn_mask is not None:
            attn_mask = _merge_causal_mask(attn_mask, T, S)
            is_causal = False

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        # -------------------------------------------------- #
        # Elementwise sigmoid gate
        # -------------------------------------------------- #

        attn_out = attn_out.transpose(1, 2)
        attn_out = attn_out.reshape(B, T, -1)

        gate = torch.sigmoid(self.gate_proj(x))

        gated_out = attn_out * gate

        # -------------------------------------------------- #
        # Output projection
        # -------------------------------------------------- #

        return self.o_proj(gated_out)
