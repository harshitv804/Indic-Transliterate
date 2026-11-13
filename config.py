from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from tokenizers import Tokenizer

TOKENIZER_PATH = Path(__file__).with_name("byte_tokenizer") / "tokenizer.json"


@lru_cache(maxsize=1)
def tokenizer_vocab_size() -> int:
    """Vocabulary size read from the tokenizer, special tokens included."""
    return Tokenizer.from_file(str(TOKENIZER_PATH)).get_vocab_size()


@dataclass
class Config:
    """Single source of truth for the encoder-decoder model."""

    vocab_size: int = field(default_factory=tokenizer_vocab_size)
    dim: int = 128
    n_heads: int = 4
    n_enc_layers: int = 12
    n_dec_layers: int = 12
    ffn_dim: int = 256
    conv_kernel: int = 7
    max_seq_len: int = 2048
    rope_theta: float = 50_000.0
    norm_eps: float = 1e-6
    dropout: float = 0.2
    qk_norm: bool = True

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads


@dataclass
class TrainingConfig:
    """Single source of truth for the transliteration training loop."""

    data_dir: str = "F:\\all_langs"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    run_id: str | None = None

    # data / batching
    batch_size: int = 256
    eval_batch_size: int = 64
    num_workers: int = 4
    max_src_len: int = 2048
    max_tgt_len: int = 2048
    shuffle: bool = False

    # optimizer
    lr_adamw: float = 1e-5
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    max_grad_norm: float = 1.0

    # schedule / loop
    num_epochs: int = 2
    num_warmup_steps: int = 10_000
    num_training_steps: int | None = None
    use_lr_scheduler: bool = True
    use_bf16: bool = True

    # logging / checkpoints
    log_every: int = 100
    checkpoint_every: int = 5000
    resume_from: str | None = None
    load_optimizer: bool = True
    seed: int = 42
