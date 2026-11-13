from torch import nn

from config import Config

from .attention import GatedMultiHeadAttention
from .ffn import SwiGLU
from .liger_ops import RMSNorm


class EncoderLayer(nn.Module):
    """Pre-norm bidirectional self-attention block + feed-forward."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.attn = GatedMultiHeadAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.mlp = SwiGLU(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, rope, attn_mask=None):
        # 1) bidirectional self-attention (is_causal=False; only padding is masked)
        residual = x
        h = self.attn_norm(x)
        h = self.attn(h, rope=rope, attn_mask=attn_mask)
        h = self.drop(h)
        x = residual + h

        # 2) feed-forward
        residual = x
        h = self.mlp_norm(x)
        h = self.mlp(h)
        h = self.drop(h)
        x = residual + h

        return x


class DecoderLayer(nn.Module):
    """Pre-norm causal self-attention + cross-attention + feed-forward."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.self_norm = RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.self_attn = GatedMultiHeadAttention(cfg)
        self.cross_norm = RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.cross_attn = GatedMultiHeadAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.mlp = SwiGLU(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, memory, rope, cross_mask=None, tgt_mask=None):
        # 1) causal self-attention (RoPE on q/k)
        residual = x
        h = self.self_norm(x)
        h = self.self_attn(h, rope=rope, attn_mask=tgt_mask, is_causal=True)
        h = self.drop(h)
        x = residual + h

        # 2) cross-attention to encoder output (no RoPE: q/k come from different sequences)
        residual = x
        h = self.cross_norm(x)
        h = self.cross_attn(h, context=memory, attn_mask=cross_mask)
        h = self.drop(h)
        x = residual + h

        # 3) feed-forward
        residual = x
        h = self.mlp_norm(x)
        h = self.mlp(h)
        h = self.drop(h)
        x = residual + h

        return x
