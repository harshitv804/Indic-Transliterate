"""CPU-only ONNX inference for the transliteration model.

Uses beam search to decode with :class:`OnnxTransliterator`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

DEFAULT_BEAM_SIZE = 4
DEFAULT_LENGTH_PENALTY = 1.0

__all__ = ["OnnxTransliterator", "normalize_direction"]


def normalize_direction(direction: str) -> str:
    """Normalize ``en-ta`` / ``<|en-ta|>`` to ``en-ta``."""
    direction = direction.strip()
    if direction.startswith("<|") and direction.endswith("|>"):
        direction = direction[2:-2]
    return direction


def _log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))


def _validate_beam_options(beam_size: int, length_penalty: float) -> None:
    if beam_size < 1:
        raise ValueError("beam_size must be >= 1")
    if length_penalty < 0.0:
        raise ValueError("length_penalty must be >= 0")


class OnnxTransliterator:
    """Decoder around the exported ONNX encoder/decoder."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        max_src_len: int | None = None,
        max_new_tokens: int | None = None,
    ):
        self.model_dir = Path(model_dir)
        config_path = self.model_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"{config_path} not found; run export_onnx.py first"
            )

        self.meta = json.loads(config_path.read_text(encoding="utf-8"))
        self.tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()

        options = ort.SessionOptions()
        options.log_severity_level = 3
        providers = ["CPUExecutionProvider"]
        self.encoder = ort.InferenceSession(
            str(self.model_dir / "encoder.onnx"), options, providers=providers
        )
        self.decoder = ort.InferenceSession(
            str(self.model_dir / "decoder.onnx"), options, providers=providers
        )

        self.direction_ids = {k: int(v) for k, v in self.meta["direction_ids"].items()}
        specials = self.meta["special_tokens"]
        self.bos_id = int(specials["bos"])
        self.eos_id = int(specials["eos"])
        self.pad_id = int(specials["pad"])

        model_max_len = int(self.meta["model"]["max_seq_len"])
        self.max_src_len = min(
            max_src_len or int(self.meta["max_src_len"]), model_max_len
        )
        self.max_new_tokens = min(
            max_new_tokens or int(self.meta["max_new_tokens"]), model_max_len - 1
        )
        self._suppress_ids = np.array(
            [
                token_id
                for token, token_id in self.tokenizer.get_vocab().items()
                if token.startswith("<|") and token_id != self.eos_id
            ],
            dtype=np.int64,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def translate(
        self,
        text: str,
        direction: str,
        *,
        beam_size: int = DEFAULT_BEAM_SIZE,
        length_penalty: float = DEFAULT_LENGTH_PENALTY,
    ) -> str:
        return self.translate_batch(
            [text],
            direction,
            beam_size=beam_size,
            length_penalty=length_penalty,
        )[0]

    def translate_batch(
        self,
        texts: Sequence[str],
        direction: str | Sequence[str],
        *,
        beam_size: int = DEFAULT_BEAM_SIZE,
        length_penalty: float = DEFAULT_LENGTH_PENALTY,
    ) -> list[str]:
        _validate_beam_options(beam_size, length_penalty)
        texts = list(texts)
        if not texts:
            return []

        directions = self._expand_directions(len(texts), direction)
        source_ids = [
            self._encode_source(text, tag) for text, tag in zip(texts, directions)
        ]
        output_ids = self._generate_ids(
            source_ids,
            beam_size=beam_size,
            length_penalty=length_penalty,
        )
        return [self._decode(ids) for ids in output_ids]

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _expand_directions(
        self, n: int, direction: str | Sequence[str]
    ) -> list[str]:
        if isinstance(direction, str):
            directions = [normalize_direction(direction)] * n
        else:
            directions = [normalize_direction(d) for d in direction]
            if len(directions) != n:
                raise ValueError(
                    f"got {n} texts but {len(directions)} directions"
                )

        unknown = sorted({d for d in directions if d not in self.direction_ids})
        if unknown:
            raise ValueError(
                f"unknown direction(s) {unknown}; expected "
                f"{sorted(self.direction_ids)}"
            )
        return directions

    def _encode_source(self, text: str, direction: str) -> list[int]:
        body = self.tokenizer.encode(text).ids[: self.max_src_len - 2]
        return [self.direction_ids[direction]] + body + [self.eos_id]

    def _prepare_source(
        self, source_ids: list[list[int]]
    ) -> tuple[np.ndarray, np.ndarray]:
        batch = len(source_ids)
        width = max(len(ids) for ids in source_ids)

        src_ids = np.full((batch, width), self.pad_id, dtype=np.int64)
        src_mask = np.zeros((batch, width), dtype=np.bool_)
        for row, ids in enumerate(source_ids):
            src_ids[row, : len(ids)] = ids
            src_mask[row, : len(ids)] = True
        return src_ids, src_mask

    def _suppress_specials(self, logits: np.ndarray) -> np.ndarray:
        if self._suppress_ids.size == 0:
            return logits
        logits = logits.copy()
        logits[..., self._suppress_ids] = -np.inf
        return logits

    def _generate_ids(
        self,
        source_ids: list[list[int]],
        *,
        beam_size: int = DEFAULT_BEAM_SIZE,
        length_penalty: float = DEFAULT_LENGTH_PENALTY,
    ) -> list[list[int]]:
        src_ids, src_mask = self._prepare_source(source_ids)
        memory = self.encoder.run(
            None,
            {"src_ids": src_ids, "src_mask": src_mask},
        )[0]
        return self._generate_beam(
            memory,
            src_mask,
            beam_size=beam_size,
            length_penalty=length_penalty,
        )

    def _generate_beam(
        self,
        memory: np.ndarray,
        src_mask: np.ndarray,
        *,
        beam_size: int,
        length_penalty: float,
    ) -> list[list[int]]:
        batch = memory.shape[0]
        beams = beam_size

        memory = np.repeat(memory, beams, axis=0)
        cross_mask = np.repeat(src_mask, beams, axis=0)
        generated = np.full((batch * beams, 1), self.bos_id, dtype=np.int64)
        scores = np.full((batch, beams), -np.inf, dtype=np.float32)
        scores[:, 0] = 0.0
        finished = np.zeros((batch, beams), dtype=np.bool_)

        for _ in range(self.max_new_tokens):
            logits = self.decoder.run(
                None,
                {
                    "tgt_ids": generated,
                    "memory": memory,
                    "src_mask": cross_mask,
                },
            )[0]
            logits = logits[:, -1, :]
            vocab_size = logits.shape[-1]
            logits = logits.reshape(batch, beams, vocab_size)
            logits = self._suppress_specials(logits)
            log_probs = _log_softmax(logits, axis=-1)

            candidate_scores = scores[:, :, None] + log_probs
            candidate_scores = np.where(
                finished[:, :, None], -np.inf, candidate_scores
            )

            flat_scores = candidate_scores.reshape(batch, beams * vocab_size)
            frozen_scores = np.where(finished, scores, -np.inf)
            combined = np.concatenate((flat_scores, frozen_scores), axis=1)

            top_indices = np.argpartition(-combined, beams - 1, axis=1)[:, :beams]
            top_scores = np.take_along_axis(combined, top_indices, axis=1)
            order = np.argsort(-top_scores, axis=1)
            top_indices = np.take_along_axis(top_indices, order, axis=1)
            top_scores = np.take_along_axis(top_scores, order, axis=1)

            frozen = top_indices >= beams * vocab_size
            frozen_beam = np.clip(top_indices - beams * vocab_size, 0, beams - 1)
            active_beam = np.clip(top_indices // vocab_size, 0, beams - 1)
            token_ids = top_indices % vocab_size
            selected_beam = np.where(frozen, frozen_beam, active_beam)

            flat_generated = generated.reshape(batch, beams, -1)
            gather_index = np.broadcast_to(
                selected_beam[:, :, None],
                (batch, beams, flat_generated.shape[-1]),
            )
            next_generated = np.take_along_axis(
                flat_generated, gather_index, axis=1
            )
            next_token = np.where(frozen, self.pad_id, token_ids)
            next_generated = np.concatenate(
                (next_generated, next_token[:, :, None]), axis=-1
            )

            old_finished = np.take_along_axis(
                finished, selected_beam, axis=1
            )
            next_finished = old_finished | (
                (~frozen) & (token_ids == self.eos_id)
            )

            generated = next_generated.reshape(batch * beams, -1)
            scores = top_scores.astype(np.float32)
            finished = next_finished

            if bool(finished.all()):
                break

        sequences = generated.reshape(batch, beams, -1)[:, :, 1:]
        eos_mask = sequences == self.eos_id
        has_eos = eos_mask.any(axis=-1)
        eos_pos = np.argmax(eos_mask, axis=-1)
        lengths = np.where(
            has_eos,
            eos_pos + 1,
            sequences.shape[-1],
        ).astype(np.float64)
        lengths = np.maximum(lengths, 1.0)

        adjusted = scores.astype(np.float64) / (lengths**length_penalty)
        best = np.argmax(adjusted, axis=1)
        best_sequences = sequences[np.arange(batch), best]
        return self._collect_results(best_sequences)

    def _collect_results(self, sequences: np.ndarray) -> list[list[int]]:
        results: list[list[int]] = []
        for row in sequences.tolist():
            if self.eos_id in row:
                row = row[: row.index(self.eos_id)]
            results.append(row)
        return results

    def _decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True).strip()
