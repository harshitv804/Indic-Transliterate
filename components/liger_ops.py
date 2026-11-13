from __future__ import annotations

import torch
from torch import nn

try:
    from liger_kernel.functional import swiglu as liger_swiglu
    from liger_kernel.transformers import LigerCrossEntropyLoss, LigerRMSNorm

    HAS_LIGER = True
except ImportError:  # pragma: no cover - liger is optional
    liger_swiglu = None
    LigerCrossEntropyLoss = None
    LigerRMSNorm = None
    HAS_LIGER = False


def use_liger(target: torch.Tensor | torch.device | str | None = None) -> bool:
    """Liger's kernels are Triton and CUDA-only."""
    if not HAS_LIGER:
        return False

    if target is None:
        return torch.cuda.is_available()

    if isinstance(target, torch.Tensor):
        return target.is_cuda

    return torch.device(target).type == "cuda"


RMSNorm = LigerRMSNorm if use_liger() else nn.RMSNorm

__all__ = [
    "HAS_LIGER",
    "LigerCrossEntropyLoss",
    "LigerRMSNorm",
    "RMSNorm",
    "liger_swiglu",
    "use_liger",
]
