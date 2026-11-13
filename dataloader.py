from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Self

import torch
from tokenizers import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from config import TOKENIZER_PATH

if TYPE_CHECKING:
    from datasets import Dataset
    from torch.utils.data import Dataset as TorchDataset

__all__ = [
    "BOS_TOKEN",
    "DEFAULT_TOKENIZER_PATH",
    "DIRECTION_IDS",
    "EOS_TOKEN",
    "IGNORE_INDEX",
    "PAD_TOKEN",
    "SpecialTokens",
    "TransliterationCollator",
    "create_dataloader",
    "load_tokenizer",
    "make_collate_fn",
    "make_dataloaders",
    "tokenize_dataset",
]

BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|eos|>"
PAD_TOKEN = "<|pad|>"  # dedicated pad token, id 12
IGNORE_INDEX = -100

DEFAULT_MAX_SRC_LEN = 512
DEFAULT_MAX_TGT_LEN = 512

# Direction tag ids, exactly as in byte_tokenizer/tokenizer.json. Prepended to
# every source sequence so one model can serve both directions of every language.
DIRECTION_IDS = {
    "ta-en": 2,
    "en-ta": 3,
    "hi-en": 4,
    "en-hi": 5,
    "te-en": 6,
    "en-te": 7,
    "ml-en": 8,
    "en-ml": 9,
    "kn-en": 10,
    "en-kn": 11,
}

# Public alias used by the export scripts.
DEFAULT_TOKENIZER_PATH = TOKENIZER_PATH


# --------------------------------------------------------------------------- #
# Validation and encoding helpers
# --------------------------------------------------------------------------- #
def _validate_max_lengths(max_src_len: int, max_tgt_len: int) -> None:
    if max_src_len < 2:
        raise ValueError("max_src_len must be >= 2 to fit a direction tag and EOS")
    if max_tgt_len < 2:
        raise ValueError("max_tgt_len must be >= 2 to fit BOS and EOS")


def _validate_direction_ids(tokenizer: Tokenizer) -> None:
    mismatched = []
    for direction, expected_id in DIRECTION_IDS.items():
        actual_id = tokenizer.token_to_id(f"<|{direction}|>")
        if actual_id != expected_id:
            mismatched.append(
                f"<|{direction}|> (expected {expected_id}, got {actual_id})"
            )
    if mismatched:
        raise ValueError(
            "tokenizer direction ids do not match DIRECTION_IDS: "
            + ", ".join(mismatched)
        )


def _require_token_id(tokenizer: Tokenizer, token: str) -> int:
    token_id = tokenizer.token_to_id(token)
    if token_id is None:
        raise ValueError(f"tokenizer is missing required token {token!r}")
    return token_id


def _direction_id(direction: str) -> int:
    try:
        return DIRECTION_IDS[direction]
    except KeyError:
        raise ValueError(
            f"unknown direction {direction!r}; expected one of {sorted(DIRECTION_IDS)}"
        ) from None


def _build_source_ids(
    token_ids: Sequence[int],
    direction: str,
    max_len: int,
    specials: SpecialTokens,
) -> list[int]:
    body = list(token_ids[: max_len - 2])
    return [_direction_id(direction)] + body + [specials.eos_id]


def _build_target_ids(
    token_ids: Sequence[int],
    max_len: int,
    specials: SpecialTokens,
) -> tuple[list[int], list[int]]:
    body = list(token_ids[: max_len - 2])
    return [specials.bos_id] + body, body + [specials.eos_id]


def _pad_sequences(
    sequences: Sequence[Sequence[int]],
    padding_value: int,
) -> torch.Tensor:
    return pad_sequence(
        [torch.as_tensor(ids, dtype=torch.long) for ids in sequences],
        batch_first=True,
        padding_value=padding_value,
    )


def _length_mask(
    sequences: Sequence[Sequence[int]],
    width: int,
) -> torch.Tensor:
    lengths = torch.tensor([len(ids) for ids in sequences], dtype=torch.long)
    return torch.arange(width).unsqueeze(0) < lengths.unsqueeze(1)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SpecialTokens:
    bos_id: int
    eos_id: int
    pad_id: int

    @classmethod
    def from_tokenizer(cls, tokenizer: Tokenizer) -> Self:
        return cls(
            bos_id=_require_token_id(tokenizer, BOS_TOKEN),
            eos_id=_require_token_id(tokenizer, EOS_TOKEN),
            pad_id=_require_token_id(tokenizer, PAD_TOKEN),
        )


def load_tokenizer(path: str | Path = DEFAULT_TOKENIZER_PATH) -> Tokenizer:
    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    return tokenizer


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: Tokenizer,
    *,
    src_column: str = "input",
    tgt_column: str = "target",
    direction_column: str = "direction",
    max_src_len: int = DEFAULT_MAX_SRC_LEN,
    max_tgt_len: int = DEFAULT_MAX_TGT_LEN,
    num_proc: int | None = None,
    load_from_cache_file: bool = True,
) -> Dataset:
    _validate_max_lengths(max_src_len, max_tgt_len)
    specials = SpecialTokens.from_tokenizer(tokenizer)
    _validate_direction_ids(tokenizer)

    def _encode(batch: dict[str, list]) -> dict[str, list[list[int]]]:
        src_batch = tokenizer.encode_batch(batch[src_column])
        tgt_batch = tokenizer.encode_batch(batch[tgt_column])

        src_ids = [
            _build_source_ids(enc.ids, direction, max_src_len, specials)
            for enc, direction in zip(src_batch, batch[direction_column], strict=True)
        ]
        target_pairs = [
            _build_target_ids(enc.ids, max_tgt_len, specials) for enc in tgt_batch
        ]

        return {
            "src_ids": src_ids,
            "tgt_ids": [decoder_ids for decoder_ids, _ in target_pairs],
            "labels": [labels for _, labels in target_pairs],
        }

    return dataset.map(
        _encode,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        load_from_cache_file=load_from_cache_file,
        desc="Tokenizing transliteration pairs",
    )


class TransliterationCollator:
    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        src_column: str = "input",
        tgt_column: str = "target",
        direction_column: str = "direction",
        max_src_len: int = DEFAULT_MAX_SRC_LEN,
        max_tgt_len: int = DEFAULT_MAX_TGT_LEN,
    ) -> None:
        _validate_max_lengths(max_src_len, max_tgt_len)
        self.tokenizer = tokenizer
        self.src_column = src_column
        self.tgt_column = tgt_column
        self.direction_column = direction_column
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.specials = SpecialTokens.from_tokenizer(tokenizer)
        _validate_direction_ids(tokenizer)

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        if "src_ids" in batch[0]:
            return self._collate_tokenized(batch)
        return self._tokenize_and_collate(batch)

    def _tokenize_and_collate(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        src_encodings = self.tokenizer.encode_batch(
            [example[self.src_column] for example in batch]
        )
        tgt_encodings = self.tokenizer.encode_batch(
            [example[self.tgt_column] for example in batch]
        )

        src, tgt, labels = [], [], []
        for src_encoding, tgt_encoding, example in zip(
            src_encodings, tgt_encodings, batch, strict=True
        ):
            direction = example[self.direction_column]
            src.append(
                _build_source_ids(
                    src_encoding.ids,
                    direction,
                    self.max_src_len,
                    self.specials,
                )
            )
            decoder_ids, label_ids = _build_target_ids(
                tgt_encoding.ids,
                self.max_tgt_len,
                self.specials,
            )
            tgt.append(decoder_ids)
            labels.append(label_ids)

        return self._pad(src, tgt, labels)

    def _collate_tokenized(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        src = [list(example["src_ids"]) for example in batch]
        tgt = [list(example["tgt_ids"]) for example in batch]
        labels = [list(example["labels"]) for example in batch]
        return self._pad(src, tgt, labels)

    def _pad(
        self,
        src: Sequence[Sequence[int]],
        tgt: Sequence[Sequence[int]],
        labels: Sequence[Sequence[int]],
    ) -> dict[str, torch.Tensor]:
        src_ids = _pad_sequences(src, self.specials.pad_id)
        tgt_ids = _pad_sequences(tgt, self.specials.pad_id)
        label_ids = _pad_sequences(labels, IGNORE_INDEX)

        # Build masks from lengths rather than ``ids != pad_id`` so a real token
        # that happens to equal pad_id can never be masked by accident.
        src_mask = _length_mask(src, src_ids.size(1))
        tgt_mask = _length_mask(tgt, tgt_ids.size(1))

        return {
            "src_ids": src_ids,
            "tgt_ids": tgt_ids,
            "labels": label_ids,
            "src_mask": src_mask,
            "tgt_mask": tgt_mask,
        }


def make_collate_fn(
    tokenizer: Tokenizer,
    **kwargs,
) -> TransliterationCollator:
    return TransliterationCollator(tokenizer, **kwargs)


def create_dataloader(
    dataset: TorchDataset,
    tokenizer: Tokenizer,
    *,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
    generator: torch.Generator | None = None,
    max_src_len: int = DEFAULT_MAX_SRC_LEN,
    max_tgt_len: int = DEFAULT_MAX_TGT_LEN,
    src_column: str = "input",
    tgt_column: str = "target",
    direction_column: str = "direction",
    **loader_kwargs,
) -> DataLoader:
    collate_fn = TransliterationCollator(
        tokenizer,
        src_column=src_column,
        tgt_column=tgt_column,
        direction_column=direction_column,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        **loader_kwargs,
    )


def make_dataloaders(
    train_dataset: TorchDataset,
    val_dataset: TorchDataset,
    tokenizer: Tokenizer,
    *,
    batch_size: int = 32,
    eval_batch_size: int | None = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    generator: torch.Generator | None = None,
    max_src_len: int = DEFAULT_MAX_SRC_LEN,
    max_tgt_len: int = DEFAULT_MAX_TGT_LEN,
    src_column: str = "input",
    tgt_column: str = "target",
    direction_column: str = "direction",
    shuffle: bool = True,
) -> tuple[DataLoader, DataLoader]:
    if eval_batch_size is None:
        eval_batch_size = batch_size

    loader_kwargs = {
        "tokenizer": tokenizer,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "max_src_len": max_src_len,
        "max_tgt_len": max_tgt_len,
        "src_column": src_column,
        "tgt_column": tgt_column,
        "direction_column": direction_column,
    }

    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        **loader_kwargs,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader
