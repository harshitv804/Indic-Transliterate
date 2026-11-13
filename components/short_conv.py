import torch.nn.functional as F
from torch import nn


class ShortConv(nn.Module):
    def __init__(self, dim: int, kernel: int = 5):
        super().__init__()
        assert kernel % 2 == 1, "Kernel size must be odd for centered padding."

        self.depthwise_conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel,
            padding=kernel // 2,
            groups=dim,
            bias=False,
        )

        self.pointwise_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, attn_mask=None):
        if attn_mask is not None:
            token_mask = attn_mask.unsqueeze(-1)
            x = x * token_mask

        # Conv1D expects [B, D, T].
        conv_features = self.depthwise_conv(x.transpose(1, 2)).transpose(1, 2)

        # Nonlinear activation followed by channel mixing.
        output = self.pointwise_proj(F.silu(conv_features))

        if attn_mask is not None:
            output = output * token_mask

        return output
