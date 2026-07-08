# CLAUDE.md — StemForge

StemForge is a **working local audio workstation**, verified end-to-end on an RTX 4090:
Demucs separation, beat_this analysis, Basic Pitch → MIDI via ONNX (no TensorFlow),
Rubber Band stretch, a Typer CLI, and a Gradio UI.

## Architecture (read these first)

- `src/stemforge/orchestrator.py` — `RunConfig` (nested dataclasses, mirrors
  `configs/default.yaml` 1:1) + `load_config(path, overrides)` with dotted-key
  overrides (`{"separation.model": "htdemucs_6s"}`) + a **fail-soft** `Pipeline`
  DAG that writes `out/<song>/manifest.json`. Stages record
  `{"skipped": ...}` / `{"error": ...}` instead of crashing the run.
- `src/stemforge/io_utils.py` — `AudioTensor` (channels-first float32 numpy +
  sample rate; lazy torch conversion), save/load, `slugify`, receipts.
- `src/stemforge/separate.py` — Demucs backend. Returns
  `(dict[str, AudioTensor], model)` so the Pipeline can cache weights across a batch.
- `src/stemforge/separate_uvr.py` — SOTA backend via **audio-separator**
  (BS-Roformer & the UVR model zoo). Same `(stems, engine)` contract.
- `src/stemforge/cli.py` — Typer CLI (`stemforge doctor|separate|analyze|run|ui`).
- `src/stemforge/app.py` — Gradio UI (preset dropdown, progress, open-output-folder).
- `tests/` — **GPU-free**; every heavy dep is mocked (see `tests/test_separation_backends.py`
  for the fake `audio_separator` module injection). Synthetic audio from `conftest.py`.

## Separation backends & quality presets

`separation.backend` = `demucs` | `uvr` | `hybrid`, routed in
`Pipeline._run_separation`. `separation.preset` (CLI `--preset`, UI dropdown)
resolves in `load_config` with precedence **yaml < preset < dotted overrides**
(`--set separation.shifts=1` beats `--preset max`; `--preset` beats `--model`):

| Preset | Backend | What runs |
|--------|---------|-----------|
| `fast` | demucs  | `htdemucs`, shifts=1 |
| `best` | demucs  | `htdemucs_ft`, shifts=2 |
| `sota` | uvr     | BS-Roformer `model_bs_roformer_ep_317_sdr_12.9755.ckpt` (2-stem: vocals + instrumental) |
| `max`  | hybrid  | BS-Roformer vocals + `htdemucs_ft` on the instrumental residual → merged 4-stem (vocals/drums/bass/other) |

audio-separator usage (see `separate_uvr.py`):

```python
from audio_separator.separator import Separator  # LAZY import only
sep = Separator(output_dir=..., output_format="WAV", model_file_dir="models/uvr",
                use_autocast=True,
                demucs_params={"shifts": 2, "overlap": 0.25, "segment_size": "Default"})
sep.load_model(model_filename="model_bs_roformer_ep_317_sdr_12.9755.ckpt")
files = sep.separate("audio.wav")   # list of output paths; models auto-download on first load
```

Output filenames carry a parenthesized stem token —
`{track}_(Vocals)_{model}.wav`, `(Instrumental)`, and for 4-stem models
`(Drums)/(Bass)/(Other)` — mapped to canonical stem names by
`separate_uvr.canonical_stem_name()`.

Backend semantics worth knowing:
- Missing optional deps fail soft with an install hint (`_SEPARATION_DEP_HINTS`);
  any other `ModuleNotFoundError` is a real error and keeps its stage trace.
- `hybrid` degrades to the finished 2-stem UVR result (with a manifest `notes`
  entry) if demucs is unavailable — completed GPU work is never discarded.
- The UVR backend can't honor `device`/`segment` — audio-separator auto-picks
  CUDA and manages its own chunking. Known upstream limitation.
- Model caches are keyed `"<backend>:<model>"`; the Gradio app shares one cache
  across clicks (`app._MODEL_CACHE`) so preset switches don't reload weights.
- Relative `uvr_model_dir` is anchored at the project root (not cwd) so the
  checkpoint cache doesn't re-download per working directory; `UvrEngine`
  removes its scratch dir via `weakref.finalize`.

## Verified install recipe (order matters)

```bash
# 1. python 3.11 env, then CUDA-matched torch FIRST (cu124 on the 4090 box)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.cuda.is_available())"   # must print True

# 2. StemForge + conflict-free extras
pip install -e .[all]        # demucs, beat_this, onnxruntime-gpu, pyrubberband, gradio

# 3. Melodic MIDI, TensorFlow-free (basic-pitch's deps would pull TF):
pip install --no-deps basic-pitch
pip install "resampy<0.4.3" scikit-learn mir_eval

# 4. SOTA separation (BS-Roformer / UVR zoo; reuses torch + onnxruntime-gpu, NO TF):
pip install "audio-separator[gpu]"      # or: pip install -e .[separation-sota]

# 5. System binaries: ffmpeg + rubberband (apt/brew/choco)
```

## Gotchas (hard-won — do not regress)

- **(a)** NO `from __future__ import annotations` in `cli.py` — it breaks Typer
  bool flags and dataclass-from-dict building (annotation strings don't resolve).
  Other modules may use it; `cli.py` may not.
- **(b)** Every Typer **bool** option needs an explicit flag name, e.g.
  `typer.Option(False, "--midi")` — otherwise Typer/Click flag inference breaks.
- **(c)** Require `typer>=0.15` — older typer + Click 8.4 raises
  `make_metavar() TypeError`.
- **(d)** Keep `torch` / `demucs` / `onnxruntime` / `gradio` / `audio_separator`
  **lazily imported** (inside functions), so `import stemforge` and pytest stay
  GPU-free. `python -c "import sys, stemforge.app, stemforge.cli; ..."` must not
  pull any of them.
- **(e)** Pipeline stages stay **fail-soft**: a missing model/dependency records
  `{"skipped": ...}` in the manifest (e.g. uvr backend without audio-separator
  installed) — never crash the run.
- **(f)** Output dirs are **slugified** (`io_utils.slugify`): `out/<slug(song)>/…`.
- Container/CI note: `pretty_midi` 0.2.10 can fail to build on Debian-patched
  setuptools (`AttributeError: install_layout`). Workaround:
  `pip install "setuptools==59.8.0" && pip install --no-build-isolation pretty_midi`,
  then restore `setuptools>=68`.
- Gradio 6: pass `theme=` to `launch()`, **not** `Blocks()` — `app.launch`
  branches on `gradio.__version__` (gradio 4/5 still take it on `Blocks()`).

## Tests

```bash
python -m pytest          # must stay green, GPU-free, no network/model downloads
```

Mock `audio_separator` via `sys.modules` injection (never import the real thing
in tests), mock Demucs by monkeypatching `stemforge.separate.separate`.
