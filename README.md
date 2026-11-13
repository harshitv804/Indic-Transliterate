---
title: Indic Transliterate
emoji: 🔤
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.32.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Live word-by-word Indic transliteration
---

# Indic Transliterate

A live, word-by-word transliteration app for English, Tamil, Hindi, Telugu,
Malayalam, and Kannada.

The left panel is the input and the right panel is the output. Choose a
direction from the dropdown, then start typing. A word is transliterated as
soon as it is followed by a space or punctuation. Deleting text from the input
also removes the corresponding output.

Decoding uses beam search with a default beam size of 4.

## Model

The ONNX model is loaded from
[`harshitv804/Indic-Transliterate`](https://huggingface.co/harshitv804/Indic-Transliterate).

The app uses a local `onnx_model/` directory when present; otherwise it
downloads the model from the Hub.

## Local run

```bash
pip install -r requirements.txt
python app.py
```

## Deploying as a Space

Create a **Gradio** Space and upload these files to its repository root:

- `app.py`
- `requirements.txt`
- `README.md`
- `transliterate.py`
- `onnx_inference.py`

You do not need to upload the local `onnx_model/` folder; the app downloads
it from the model repo at startup. If an `onnx_model/` folder is present, that
local copy is used instead.

Optional environment variables:

- `MODEL_REPO` — override the Hugging Face model repo.
- `BEAM_SIZE` — beam width, default `4`.
- `WORD_CACHE_SIZE` — number of cached word translations, default `8192`.
- `ONNX_NUM_THREADS` — ONNX Runtime intra-op threads, default
  `min(4, cpu_count)`. Oversubscribing small containers is much slower.
- `DIAG_TIMING` — set to `1` to log per-request timings.
