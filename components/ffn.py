import torch
import torch.nn.functional as F
from torch import nn

from config import Config

from .liger_ops import liger_swiglu, use_liger


class SwiGLU(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)
        self.up_proj = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)
        self.down_proj = nn.Linear(cfg.ffn_dim, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        if use_liger(gate):
            return self.down_proj(liger_swiglu(gate, up))

        return self.down_proj(F.silu(gate) * up)
