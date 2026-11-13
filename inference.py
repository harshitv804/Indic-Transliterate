"""Inference for the multilingual transliteration model.

Import and reuse a loaded model:

    from inference import Transliterator

    translator = Transliterator.from_checkpoint("checkpoints/last.pt")
    print(translator.translate("How are you?", "en-ta"))
    print(translator.translate("मैं ठीक हूँ", "hi-en"))

Load only the model (for custom generation code):

    from components import Config
    from inference import load_model

    model = load_model("checkpoints/last.pt", mcfg=Config())

Command line:

    python inference.py --checkpoint checkpoints/last.pt --direction en-ta \
        --text "How are you?"

    # no --text -> interactive prompt; empty line exits
    python inference.py --checkpoint checkpoints/last.pt --direction en-ta

Notes:
  * Decoding uses beam search.
  * The decoder has no KV cache, so every new token re-runs the full target
    prefix. This is fine for testing, but slow for large-scale generation.
  * If you trained with a non-default ``Config``, pass the same instance as
    ``mcfg``; the checkpoint does not store the model config.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import torch

from components import Config, EncoderDecoderTransformer
from dataloader import (
    DIRECTION_IDS,
    SpecialTokens,
    load_tokenizer,
)

DEFAULT_MAX_SRC_LEN = 128
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_BEAM_SIZE = 4
DEFAULT_LENGTH_PENALTY = 1.0

__all__ = ["Transliterator", "load_model", "transliterate"]


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _check_direction_ids(tokenizer) -> None:
    mismatched = [
        f"<|{direction}|> (expected {expected_id}, "
        f"got {tokenizer.token_to_id(f'<|{direction}|>')})"
        for direction, expected_id in DIRECTION_IDS.items()
        if tokenizer.token_to_id(f"<|{direction}|>") != expected_id
    ]
    if mismatched:
        raise ValueError(
            "tokenizer direction ids do not match DIRECTION_IDS: "
            + ", ".join(mismatched)
        )


def _normalize_direction(direction: str) -> str:
    direction = direction.strip()
    if direction.startswith("<|") and direction.endswith("|>"):
        direction = direction[2:-2]
    if direction not in DIRECTION_IDS:
        raise ValueError(
            f"unknown direction {direction!r}; expected one of "
            f"{sorted(DIRECTION_IDS)}"
        )
    return direction


def _validate_beam_options(beam_size: int, length_penalty: float) -> None:
    if beam_size < 1:
        raise ValueError("beam_size must be >= 1")
    if length_penalty < 0.0:
        raise ValueError("length_penalty must be >= 0")


def load_model(
    checkpoint: str | Path | None = None,
    *,
    mcfg: Config | None = None,
    device: str | torch.device | None = None,
) -> EncoderDecoderTransformer:
    """Build the model and load a training checkpoint.

    ``checkpoint`` may be a file saved by ``train.py`` (state stored under
    ``"model"``) or a file containing a raw state dict. ``None`` returns a
    randomly initialized model, useful for shape/forward tests.
    """
    device = _resolve_device(device)
    mcfg = mcfg or Config()

    model = EncoderDecoderTransformer(mcfg)

    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(state_dict)

        step = ckpt.get("step") if isinstance(ckpt, dict) else None
        epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
        if epoch is not None and not ckpt.get("epoch_1_based", False):
            epoch += 1  # old checkpoints stored 0-based epochs
        if step is not None:
            print(f"[ckpt] loaded step={step} epoch={epoch} from {checkpoint}")
        else:
            print(f"[ckpt] loaded weights from {checkpoint}")
    else:
        model.init_weights()
        print("[ckpt] no checkpoint given; using freshly initialized weights")

    return model.to(device).eval()


class Transliterator:
    """Small decoding wrapper around the encoder-decoder model."""

    def __init__(
        self,
        model: EncoderDecoderTransformer,
        tokenizer=None,
        *,
        device: str | torch.device | None = None,
        max_src_len: int = DEFAULT_MAX_SRC_LEN,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ):
        if max_src_len < 2:
            raise ValueError("max_src_len must be >= 2 (direction tag + EOS)")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")

        self.model = model
        self.tokenizer = tokenizer or load_tokenizer()
        _check_direction_ids(self.tokenizer)

        self.device = (
            _resolve_device(device)
            if device is not None
            else next(model.parameters()).device
        )
        self.model.to(self.device).eval()

        self.specials = SpecialTokens.from_tokenizer(self.tokenizer)
        self.max_src_len = min(max_src_len, model.cfg.max_seq_len)
        self.max_new_tokens = max_new_tokens
        self._suppress_ids = [
            token_id
            for token, token_id in self.tokenizer.get_vocab().items()
            if token.startswith("<|") and token_id != self.specials.eos_id
        ]

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path | None = None,
        *,
        mcfg: Config | None = None,
        tokenizer=None,
        device: str | torch.device | None = None,
        max_src_len: int = DEFAULT_MAX_SRC_LEN,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> "Transliterator":
        model = load_model(checkpoint, mcfg=mcfg, device=device)
        return cls(
            model,
            tokenizer=tokenizer,
            device=device,
            max_src_len=max_src_len,
            max_new_tokens=max_new_tokens,
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
        """Transliterate one string. ``direction`` is e.g. ``"en-ta"``."""
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
        """Decode a batch with beam search; ``direction`` may be one tag or
        one per row."""
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
    @staticmethod
    def _expand_directions(
        n: int, direction: str | Sequence[str]
    ) -> list[str]:
        if isinstance(direction, str):
            return [_normalize_direction(direction)] * n

        directions = [_normalize_direction(d) for d in direction]
        if len(directions) != n:
            raise ValueError(
                f"got {n} texts but {len(directions)} directions"
            )
        return directions

    def _encode_source(self, text: str, direction: str) -> list[int]:
        direction = _normalize_direction(direction)
        body = self.tokenizer.encode(text).ids[: self.max_src_len - 2]
        return [DIRECTION_IDS[direction]] + body + [self.specials.eos_id]

    @torch.inference_mode()
    def _generate_ids(
        self,
        source_ids: list[list[int]],
        *,
        beam_size: int = DEFAULT_BEAM_SIZE,
        length_penalty: float = DEFAULT_LENGTH_PENALTY,
    ) -> list[list[int]]:
        src_ids, src_mask = self._prepare_source(source_ids)
        memory = self.model.encode(src_ids, src_mask)
        return self._generate_beam(
            memory,
            src_mask,
            beam_size=beam_size,
            length_penalty=length_penalty,
        )

    def _prepare_source(
        self, source_ids: list[list[int]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = len(source_ids)
        width = max(len(ids) for ids in source_ids)

        src_ids = torch.full(
            (batch, width),
            self.specials.pad_id,
            dtype=torch.long,
            device=self.device,
        )
        src_mask = torch.zeros(
            (batch, width), dtype=torch.bool, device=self.device
        )
        for row, ids in enumerate(source_ids):
            src_ids[row, : len(ids)] = torch.tensor(
                ids, dtype=torch.long, device=self.device
            )
            src_mask[row, : len(ids)] = True
        return src_ids, src_mask

    def _suppress_specials(self, logits: torch.Tensor) -> torch.Tensor:
        if not self._suppress_ids:
            return logits
        ids = torch.tensor(self._suppress_ids, dtype=torch.long, device=logits.device)
        return logits.index_fill(-1, ids, torch.finfo(logits.dtype).min)

    def _generate_beam(
        self,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        *,
        beam_size: int,
        length_penalty: float,
    ) -> list[list[int]]:
        batch = memory.size(0)
        beams = beam_size
        vocab_size = self.model.cfg.vocab_size
        device = self.device

        memory = memory.repeat_interleave(beams, dim=0)
        cross_mask = src_mask.repeat_interleave(beams, dim=0)
        generated = torch.full(
            (batch * beams, 1),
            self.specials.bos_id,
            dtype=torch.long,
            device=device,
        )
        scores = torch.full(
            (batch, beams),
            float("-inf"),
            dtype=torch.float32,
            device=device,
        )
        scores[:, 0] = 0.0
        finished = torch.zeros((batch, beams), dtype=torch.bool, device=device)
        max_new = min(self.max_new_tokens, self.model.cfg.max_seq_len - 1)

        for _ in range(max_new):
            hidden = self.model.decode(
                generated,
                memory,
                cross_mask=cross_mask,
            )
            logits = self.model.lm_head(hidden[:, -1, :]).view(
                batch, beams, vocab_size
            )
            logits = self._suppress_specials(logits)
            log_probs = torch.log_softmax(logits, dim=-1)

            candidate_scores = scores.unsqueeze(-1) + log_probs
            min_score = torch.finfo(candidate_scores.dtype).min
            candidate_scores = candidate_scores.masked_fill(
                finished.unsqueeze(-1), min_score
            )

            flat_scores = candidate_scores.reshape(batch, beams * vocab_size)
            frozen_scores = scores.masked_fill(~finished, min_score)
            combined = torch.cat((flat_scores, frozen_scores), dim=1)

            top_scores, top_indices = combined.topk(beams, dim=1)

            frozen = top_indices >= beams * vocab_size
            frozen_beam = (top_indices - beams * vocab_size).clamp(min=0)
            active_beam = (top_indices // vocab_size).clamp(max=beams - 1)
            token_ids = top_indices % vocab_size
            selected_beam = torch.where(frozen, frozen_beam, active_beam)

            flat_generated = generated.view(batch, beams, -1)
            gather_index = selected_beam.unsqueeze(-1).expand(
                -1, -1, flat_generated.size(-1)
            )
            next_generated = flat_generated.gather(1, gather_index)

            next_token = torch.where(
                frozen,
                torch.full_like(token_ids, self.specials.pad_id),
                token_ids,
            )
            next_generated = torch.cat(
                (next_generated, next_token.unsqueeze(-1)), dim=-1
            )

            old_finished = finished.gather(1, selected_beam)
            next_finished = old_finished | (
                ~frozen & token_ids.eq(self.specials.eos_id)
            )

            generated = next_generated.reshape(batch * beams, -1)
            scores = top_scores
            finished = next_finished

            if bool(finished.all()):
                break

        sequences = generated.view(batch, beams, -1)[:, :, 1:]
        eos_mask = sequences.eq(self.specials.eos_id)
        has_eos = eos_mask.any(dim=-1)
        eos_pos = eos_mask.to(torch.long).argmax(dim=-1)
        lengths = torch.where(
            has_eos,
            eos_pos + 1,
            torch.full_like(eos_pos, sequences.size(-1)),
        ).clamp(min=1)

        adjusted = scores / lengths.to(scores.dtype).pow(length_penalty)
        best = adjusted.argmax(dim=-1)
        best_sequences = sequences[
            torch.arange(batch, device=device),
            best,
        ]
        return self._collect_results(best_sequences)

    def _collect_results(self, sequences: torch.Tensor) -> list[list[int]]:
        results: list[list[int]] = []
        for row in sequences.tolist():
            if self.specials.eos_id in row:
                row = row[: row.index(self.specials.eos_id)]
            results.append(row)
        return results

    def _decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True).strip()


def transliterate(
    text: str,
    direction: str,
    *,
    checkpoint: str | Path | None = None,
    mcfg: Config | None = None,
    tokenizer=None,
    device: str | torch.device | None = None,
    max_src_len: int = DEFAULT_MAX_SRC_LEN,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    beam_size: int = DEFAULT_BEAM_SIZE,
    length_penalty: float = DEFAULT_LENGTH_PENALTY,
) -> str:
    """One-shot convenience wrapper. Reuse ``Transliterator`` for many calls."""
    translator = Transliterator.from_checkpoint(
        checkpoint,
        mcfg=mcfg,
        tokenizer=tokenizer,
        device=device,
        max_src_len=max_src_len,
        max_new_tokens=max_new_tokens,
    )
    return translator.translate(
        text,
        direction,
        beam_size=beam_size,
        length_penalty=length_penalty,
    )


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Beam-search transliteration with the trained encoder-decoder."
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="path to a train.py checkpoint (omit for random weights)",
    )
    parser.add_argument(
        "--direction",
        default="en-ta",
        help="direction tag, e.g. en-ta, ta-en, hi-en (default: en-ta)",
    )
    parser.add_argument(
        "--text",
        default=None,
        help="text to transliterate; omit for an interactive prompt",
    )
    parser.add_argument("--device", default=None, help="cuda, cpu, or cuda:1")
    parser.add_argument("--max-src-len", type=int, default=DEFAULT_MAX_SRC_LEN)
    parser.add_argument(
        "--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    # Windows consoles often default to a legacy code page; force UTF-8 so
    # Indic output prints correctly.
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except (AttributeError, ValueError):
                pass

    args = _parse_args(argv)
    translator = Transliterator.from_checkpoint(
        args.checkpoint,
        device=args.device,
        max_src_len=args.max_src_len,
        max_new_tokens=args.max_new_tokens,
    )

    if args.text is not None:
        print(translator.translate(args.text, args.direction))
        return

    print(
        f"Interactive mode ({args.direction}). "
        "Type a sentence; empty line exits."
    )
    while True:
        try:
            text = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not text.strip():
            break

        print(translator.translate(text, args.direction))


if __name__ == "__main__":
    main()
