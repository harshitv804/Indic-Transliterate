"""Word-by-word sentence transliteration preserving non-text runs."""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence

from onnx_inference import (
    DEFAULT_BEAM_SIZE,
    DEFAULT_LENGTH_PENALTY,
    OnnxTransliterator,
    normalize_direction,
)

__all__ = ["IndicTransliterator", "split_text_runs"]


def _is_word_char(char: str) -> bool:
    """True for Unicode letters and combining marks."""
    return unicodedata.category(char)[0] in ("L", "M")


def _has_letter(run: str) -> bool:
    return any(unicodedata.category(char).startswith("L") for char in run)


def split_text_runs(text: str) -> list[tuple[bool, str]]:
    """Split into ``(is_word, run)`` pieces; joining restores the input."""
    if not text:
        return []

    runs: list[tuple[bool, str]] = []
    start = 0
    current = _is_word_char(text[0])
    for index in range(1, len(text)):
        is_word = _is_word_char(text[index])
        if is_word != current:
            runs.append((current, text[start:index]))
            start = index
            current = is_word
    runs.append((current, text[start:]))

    # Demote mark-only runs, then merge adjacent non-text runs.
    cleaned: list[tuple[bool, str]] = []
    for is_word, run in runs:
        is_word = is_word and _has_letter(run)
        if cleaned and cleaned[-1][0] == is_word:
            cleaned[-1] = (is_word, cleaned[-1][1] + run)
        else:
            cleaned.append((is_word, run))
    return cleaned


class IndicTransliterator:
    """Transliterate sentences while preserving formatting.

    English source text is lowercased before transliteration. Decoding options
    are forwarded to the underlying :class:`OnnxTransliterator`.
    """

    def __init__(
        self,
        model_dir: str,
        *,
        max_src_len: int | None = None,
        max_new_tokens: int | None = None,
    ):
        self.engine = OnnxTransliterator(
            model_dir,
            max_src_len=max_src_len,
            max_new_tokens=max_new_tokens,
        )

    @property
    def direction_ids(self) -> dict[str, int]:
        return self.engine.direction_ids

    def translate(
        self,
        text: str,
        direction: str,
        *,
        beam_size: int = DEFAULT_BEAM_SIZE,
        length_penalty: float = DEFAULT_LENGTH_PENALTY,
    ) -> str:
        """Transliterate one sentence."""
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
        """Transliterate sentences, batching the words in each."""
        texts = list(texts)
        if not texts:
            return []

        directions = self._resolve_directions(len(texts), direction)
        output: list[str] = []

        for text, direction_tag in zip(texts, directions):
            if direction_tag.startswith("en-"):
                text = text.lower()
            runs = split_text_runs(text)
            words = [run for is_word, run in runs if is_word]
            translated_words = (
                self.engine.translate_batch(
                    words,
                    direction_tag,
                    beam_size=beam_size,
                    length_penalty=length_penalty,
                )
                if words
                else []
            )

            pieces: list[str] = []
            cursor = 0
            for is_word, run in runs:
                if is_word:
                    pieces.append(translated_words[cursor])
                    cursor += 1
                else:
                    pieces.append(run)
            output.append("".join(pieces))
        return output

    def __call__(self, text: str, direction: str, **kwargs) -> str:
        return self.translate(text, direction, **kwargs)

    def _resolve_directions(
        self, count: int, direction: str | Sequence[str]
    ) -> list[str]:
        if isinstance(direction, str):
            directions = [normalize_direction(direction)] * count
        else:
            directions = [normalize_direction(item) for item in direction]
            if len(directions) != count:
                raise ValueError(f"got {count} texts but {len(directions)} directions")

        unknown = sorted(
            {item for item in directions if item not in self.engine.direction_ids}
        )
        if unknown:
            raise ValueError(
                f"unknown direction(s) {unknown}; "
                f"expected {sorted(self.engine.direction_ids)}"
            )
        return directions
