import math

from torch import nn

from config import Config

from .attention import GatedMultiHeadAttention
from .ffn import SwiGLU
from .layers import DecoderLayer, EncoderLayer
from .liger_ops import RMSNorm
from .rope import RotaryEmbedding
from .short_conv import ShortConv


def _is_right_padded(mask):
    if mask.shape[1] <= 1:
        return True

    # No False -> True transition in any row.
    return bool((mask[:, :-1] | ~mask[:, 1:]).all())


class EncoderDecoderTransformer(nn.Module):
    def __init__(self, cfg: Config = Config()):
        super().__init__()
        self.cfg = cfg

        # Separate embedding modules; weights are tied below.
        self.enc_tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.dec_tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)

        # Encoder stem: short depthwise convolution (residual branch).
        self.enc_conv = ShortConv(cfg.dim, kernel=cfg.conv_kernel)

        # Rotary positional embeddings
        self.rope = RotaryEmbedding(
            cfg.head_dim,
            cfg.max_seq_len,
            cfg.rope_theta,
        )

        # Encoder
        self.enc_layers = nn.ModuleList(
            EncoderLayer(cfg) for _ in range(cfg.n_enc_layers)
        )
        self.enc_norm = RMSNorm(
            cfg.dim,
            eps=cfg.norm_eps,
        )

        # Decoder
        self.dec_layers = nn.ModuleList(
            DecoderLayer(cfg) for _ in range(cfg.n_dec_layers)
        )
        self.dec_norm = RMSNorm(
            cfg.dim,
            eps=cfg.norm_eps,
        )

        # Output projection
        self.lm_head = nn.Linear(
            cfg.dim,
            cfg.vocab_size,
            bias=False,
        )

        # Three-way weight tying (structural, independent of initialization).
        self.dec_tok_emb.weight = self.enc_tok_emb.weight
        self.lm_head.weight = self.enc_tok_emb.weight

        # Weight initialization is explicit: call ``init_weights()`` for a fresh
        # run, or load a checkpoint to skip it.

    def init_weights(self):
        """Initialize all model weights. Call once for a fresh training run."""

        def _init_module(module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

            elif isinstance(module, nn.Conv1d):
                # Depthwise conv in the encoder stem.
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, RMSNorm):
                # Normalization scales start at 1 (Liger's default, made
                # explicit here so all initialization lives in one place).
                nn.init.ones_(module.weight)

        self.apply(_init_module)

        # Scale each residual branch by 1/sqrt(branches * depth) so the residual
        # stream stays roughly constant regardless of stack depth. Structural
        # (not name-based), so it still works under torch.compile/DDP wrappers.
        for layers, depth, branches in (
            (self.enc_layers, self.cfg.n_enc_layers, 2),  # attn + mlp
            (self.dec_layers, self.cfg.n_dec_layers, 3),  # self + cross + mlp
        ):
            std = 0.02 / math.sqrt(branches * depth)
            for layer in layers:
                for module in layer.modules():
                    if isinstance(module, GatedMultiHeadAttention):
                        nn.init.normal_(module.o_proj.weight, mean=0.0, std=std)
                    elif isinstance(module, SwiGLU):
                        nn.init.normal_(module.down_proj.weight, mean=0.0, std=std)

    def encode(self, src_ids, src_mask=None):
        x = self.enc_tok_emb(src_ids)
        x = x + self.enc_conv(x, src_mask)
        T = src_ids.shape[1]

        rope = self.rope(T)  # fixed

        attn_mask = src_mask[:, None, None, :] if src_mask is not None else None

        for layer in self.enc_layers:
            x = layer(x, rope=rope, attn_mask=attn_mask)

        return self.enc_norm(x)

    def decode(self, tgt_ids, memory, cross_mask=None, tgt_mask=None):
        # cross_mask is [B, S] over source (memory) positions; tgt_mask is
        # [B, T] over target positions.
        x = self.dec_tok_emb(tgt_ids)
        T = tgt_ids.shape[1]

        rope = self.rope(T)  # fixed

        cross_attn_mask = (
            cross_mask[:, None, None, :] if cross_mask is not None else None
        )

        # Redundant under causal masking; drop it to keep SDPA's fused kernel.
        self_attn_mask = None
        if tgt_mask is not None and not _is_right_padded(tgt_mask):
            self_attn_mask = tgt_mask[:, None, None, :]

        for layer in self.dec_layers:
            x = layer(
                x,
                memory=memory,
                rope=rope,
                cross_mask=cross_attn_mask,
                tgt_mask=self_attn_mask,
            )

        return self.dec_norm(x)

    def forward(self, src_ids, tgt_ids, src_mask=None, tgt_mask=None):
        memory = self.encode(src_ids, src_mask)

        hidden = self.decode(
            tgt_ids,
            memory,
            cross_mask=src_mask,
            tgt_mask=tgt_mask,
        )

        logits = self.lm_head(hidden)

        return logits
