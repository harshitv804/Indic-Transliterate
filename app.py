"""Gradio Space for live, word-by-word Indic transliteration.

The model is loaded from the Hugging Face model repo
``harshitv804/Indic-Transliterate`` unless a local ``onnx_model`` directory is
present next to this file.
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from pathlib import Path

import gradio as gr

try:  # preinstalled only on Hugging Face ZeroGPU runtimes
    import spaces
except ImportError:  # local CPU runs do not have it
    spaces = None

from onnx_inference import DEFAULT_BEAM_SIZE
from transliterate import IndicTransliterator, split_text_runs

MODEL_REPO = os.getenv("MODEL_REPO", "harshitv804/Indic-Transliterate")
MODEL_SUBDIR = "onnx_model"
DEFAULT_DIRECTION = "en-ta"
BEAM_SIZE = int(os.getenv("BEAM_SIZE", str(DEFAULT_BEAM_SIZE)))
CACHE_SIZE = int(os.getenv("WORD_CACHE_SIZE", "8192"))
DIAG = os.getenv("DIAG_TIMING") == "1"

LANGUAGE_NAMES = {
    "en": "English",
    "ta": "Tamil",
    "hi": "Hindi",
    "te": "Telugu",
    "ml": "Malayalam",
    "kn": "Kannada",
}


def _resolve_model_dir() -> str:
    """Use a local export when present, otherwise download it from the Hub."""
    local_dir = Path(__file__).resolve().parent / MODEL_SUBDIR
    if (local_dir / "config.json").exists():
        return str(local_dir)

    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(
        repo_id=MODEL_REPO,
        allow_patterns=[f"{MODEL_SUBDIR}/*"],
    )
    return str(Path(snapshot) / MODEL_SUBDIR)


TRANSLITERATOR = IndicTransliterator(_resolve_model_dir())
if DIAG:
    engine = TRANSLITERATOR.engine
    print(
        f"[diag] cpus={os.cpu_count()} "
        f"intra_op={engine.num_threads} "
        f"providers={engine.encoder.get_providers()}",
        flush=True,
    )
DIRECTIONS = list(TRANSLITERATOR.direction_ids)
if DEFAULT_DIRECTION not in DIRECTIONS:
    DEFAULT_DIRECTION = DIRECTIONS[0]


def _direction_label(tag: str) -> str:
    source, _, target = tag.partition("-")
    source_name = LANGUAGE_NAMES.get(source, source)
    target_name = LANGUAGE_NAMES.get(target, target)
    return f"{source_name} → {target_name} ({tag})"


DIRECTION_CHOICES = [
    (_direction_label(tag), tag) for tag in DIRECTIONS
]

# Word translations are cached so typing a sentence does not re-run the model
# for every already-completed word.
_WORD_CACHE: "OrderedDict[tuple[str, str], str]" = OrderedDict()
_CACHE_LOCK = threading.Lock()


def _translate_words(words: list[str], direction: str) -> dict[str, str]:
    """Translate unique words, using and updating the cache."""
    unique_words = list(dict.fromkeys(words))
    if not unique_words:
        return {}

    keys = [(word, direction) for word in unique_words]
    with _CACHE_LOCK:
        missing = [
            word
            for word, key in zip(unique_words, keys)
            if key not in _WORD_CACHE
        ]

    if missing:
        start = time.perf_counter()
        translated = TRANSLITERATOR.engine.translate_batch(
            missing,
            direction,
            beam_size=BEAM_SIZE,
        )
        if DIAG:
            elapsed = time.perf_counter() - start
            print(
                f"[diag] translate {len(missing)} word(s) "
                f"beam={BEAM_SIZE} threads={TRANSLITERATOR.engine.num_threads} "
                f"in {elapsed * 1000:.0f} ms: {missing[:5]}",
                flush=True,
            )
        with _CACHE_LOCK:
            for word, value in zip(missing, translated):
                key = (word, direction)
                _WORD_CACHE[key] = value
                _WORD_CACHE.move_to_end(key)
            while len(_WORD_CACHE) > CACHE_SIZE:
                _WORD_CACHE.popitem(last=False)

    with _CACHE_LOCK:
        result: dict[str, str] = {}
        for word in unique_words:
            key = (word, direction)
            result[word] = _WORD_CACHE[key]
            _WORD_CACHE.move_to_end(key)
        return result


def live_transliterate(text: str, direction: str | None) -> str:
    """Translate completed words only; keep the word being typed intact.

    A word is considered complete once it is followed by whitespace or
    punctuation. Because the whole output is rebuilt from the current input,
    backspacing also removes the corresponding translated text.
    """
    if not text:
        return ""

    direction = direction or DEFAULT_DIRECTION
    runs = split_text_runs(text)
    if not runs:
        return text

    lower_english = direction.startswith("en-")

    # The final run is left untouched when it is still a word being typed.
    word_runs: list[str] = []
    for index, (is_word, run) in enumerate(runs):
        if is_word and index < len(runs) - 1:
            word_runs.append(run.lower() if lower_english else run)

    translations = _translate_words(word_runs, direction) if word_runs else {}

    pieces: list[str] = []
    for index, (is_word, run) in enumerate(runs):
        if is_word and index < len(runs) - 1:
            key = run.lower() if lower_english else run
            pieces.append(translations[key])
        else:
            pieces.append(run)
    return "".join(pieces)


def _translate_from_ui(text: str, direction: str | None) -> str:
    try:
        return live_transliterate(text, direction)
    except Exception as exc:  # keep the UI alive if one word fails
        print(f"[live_transliterate] {type(exc).__name__}: {exc}")
        return f"⚠️ {exc}"


if spaces is not None:

    @spaces.GPU
    def zerogpu_startup_probe() -> str:
        """No-op placeholder for the ZeroGPU runtime.

        ZeroGPU Spaces refuse to boot unless the app defines at least one
        ``@spaces.GPU`` function. This app does all inference on CPU with
        ONNX Runtime, so the probe is never called and never reserves GPU
        time.
        """
        return "cpu-only"


with gr.Blocks(
    title="Indic Transliterate",
    fill_width=True,
) as demo:
    gr.Markdown(
        "# Indic Transliterate\n"
        "Type in the left box. A word is transliterated as soon as you "
        "press **space** (or add punctuation). Backspacing removes the "
        "corresponding output too.\n\n"
        f"Decoding: **beam search**, beam size **{BEAM_SIZE}**."
    )

    direction = gr.Dropdown(
        choices=DIRECTION_CHOICES,
        value=DEFAULT_DIRECTION,
        label="Direction",
        filterable=True,
        scale=1,
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            source = gr.Textbox(
                label="Input",
                lines=16,
                max_lines=40,
                placeholder="Type here…",
                autofocus=True,
            )
        with gr.Column(scale=1):
            target = gr.Textbox(
                label="Output",
                lines=16,
                max_lines=40,
                interactive=False,
                buttons=["copy"],
                placeholder="Transliteration appears here…",
            )

    gr.Markdown(
        "Words are translated incrementally. The word currently being typed "
        "stays in the input script until you press space."
    )

    source.input(
        fn=_translate_from_ui,
        inputs=[source, direction],
        outputs=target,
        show_progress="hidden",
        trigger_mode="always_last",
        concurrency_limit=1,
    )
    direction.change(
        fn=_translate_from_ui,
        inputs=[source, direction],
        outputs=target,
        show_progress="hidden",
        trigger_mode="always_last",
        concurrency_limit=1,
    )


demo.queue(default_concurrency_limit=1)


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
