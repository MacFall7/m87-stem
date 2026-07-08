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
  (BS-Roformer & the UVR model zoo), run as a **subprocess from an isolated
  venv** (`.venv-uvr`) — never imported in-process. Same `(stems, engine)`
  contract. Also hosts `separate_drums()` + drum-model discovery
  (`--list_filter=drums`), reused by the drum splitter.
- `src/stemforge/drum_split.py` — drum teardown. `backend: uvr` runs a DrumSep
  model through the **same** `.venv-uvr` subprocess (`separate_uvr.separate_drums`);
  accepts a raw drum LOOP (`from_input`) or the separated drums stem; parts feed
  the existing parts-based `drum_midi` (onset + RMS-velocity + GM mapping).
- `src/stemforge/cli.py` — Typer CLI (`doctor|setup-sota|separate|analyze|run|ui|desktop-shortcut`).
- `src/stemforge/app.py` — the **M87 Space-Tech workstation** (Gradio): four
  workflow panels (Extract Stems / Drum Teardown / Melodic → MIDI / Full
  Teardown), a dark CSS theme whose colors/fonts are CSS variables at the top of
  `M87_CSS` (`--m87-bg` … `--m87-mono`), progress, per-stem audition, and an
  open-output-folder button. Theme + CSS pass via `launch()` on Gradio 6.
- `tests/` — **GPU-free**; every heavy dep is mocked. The audio-separator CLI
  subprocess (stem + drum) is mocked in `tests/test_separation_backends.py` /
  `tests/test_drum_split.py`; synthetic audio from `conftest.py`.

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

## audio-separator isolation (CRITICAL — do not regress)

**audio-separator is NEVER imported in-process.** It lives in its own venv
(default `<project-root>/.venv-uvr`, config `separation.uvr_venv`, git-ignored)
and is invoked as a subprocess (see `separate_uvr.UvrEngine.separate_file`):

```
<venv>/bin/audio-separator input.wav --model_filename <ckpt> \
    --output_dir <scratch> --output_format WAV --model_file_dir models/uvr [--use_autocast]
```

**Why (production incident, Windows/py3.11, 2026-07):** installing
`audio-separator[gpu]` into the main env pulled a **CPU-only torch 2.13.0+cpu**
that replaced torch 2.6.0+cu124 and bumped numpy to 2.4.6 — silently breaking
the working demucs GPU stack. Isolation makes the clobber structurally
impossible; the main interpreter's packages are never touched.

- Provision with **`stemforge setup-sota`** (idempotent): creates the venv,
  installs CUDA torch from the cu124 index FIRST, then `audio-separator[gpu]`,
  then **force-reinstalls the cu124 torch LAST** (`--force-reinstall --no-deps`)
  because audio-separator's deps otherwise leave a **CPU torch** behind (that's
  why `--preset max` once ran on CPU). It then probes `torch.cuda.is_available()`
  *inside the venv* and prints the GPU name — into that venv ONLY. The
  force-reinstall is fail-soft (warns, never crashes) and skipped when torch is
  already a `+cu` build (idempotent). `stemforge doctor` shows the venv status
  **plus the `.venv-uvr` torch version + CUDA state** (`separate_uvr.venv_torch_status`).
- `separate_uvr.preflight()` verifies ffmpeg + the venv CLI before every run
  and raises `SotaEnvError` → the orchestrator records `{"skipped": ...}` with
  a one-line fix hint. Separation never creates the venv or downloads anything.
- Output filenames carry a parenthesized stem token —
  `{track}_(Vocals)_{model}.wav`, `(Instrumental)`, and for 4-stem models
  `(Drums)/(Bass)/(Other)` — mapped to canonical stem names by
  `separate_uvr.canonical_stem_name()`; produced files are detected by
  scratch-dir diff, loaded as `AudioTensor`, then deleted.

Backend semantics worth knowing:
- Missing demucs/torch fail soft with an install hint (`_SEPARATION_DEP_HINTS`);
  any other `ModuleNotFoundError` is a real error and keeps its stage trace. A
  nonzero audio-separator exit is also a real error (stderr tail in the manifest).
- `hybrid` degrades to the finished 2-stem UVR result (with a manifest `notes`
  entry) if demucs is unavailable — completed GPU work is never discarded.
- The UVR backend can't honor `device`/`segment` — audio-separator auto-picks
  CUDA inside its venv and manages its own chunking. Known upstream limitation.
- Model caches are keyed `"<backend>:<model>"`; the Gradio app shares one cache
  across clicks (`app._MODEL_CACHE`). The cached `UvrEngine` holds no weights
  in-process (they live in the subprocess) but skips repeated preflights.
- Relative `uvr_venv` / `uvr_model_dir` are anchored at the project root (not
  cwd) so neither the venv nor the checkpoint cache is duplicated per working
  directory; `UvrEngine` removes its scratch dir via `weakref.finalize`.

## Drum teardown (same isolated venv — do not regress)

`drums.split.backend = uvr` (default) tears a **drum loop** or the separated
drums stem into hit parts through the **same** `.venv-uvr` subprocess runner —
`separate_uvr.separate_drums()` builds a `UvrEngine.from_cfg(drum_cfg, model=…)`
and runs it exactly like stem separation. audio-separator is still never
imported in-process.

- Model: `drums.split.uvr_model` when set, else **auto-discovered** via
  `audio-separator --list_filter=drums` (`discover_drum_model`, prefers a
  6-piece kit). Output tokens `(Kick)/(Snare)/(Toms)/(HiHat)/(Ride)/(Crash)` map
  to canonical parts through `separate_uvr.canonical_drum_part()`.
- Input: `drums.split.from_input=true` tears down the **raw input loop** (no
  separation needed); otherwise it uses `ctx.stems["drums"]`. The orchestrator
  passes `ctx.audio` as an `AudioTensor` for the from-input case.
- The produced parts feed the existing parts-based `drum_midi` (onset detection +
  RMS→velocity + GM note mapping, 5→7 hi-hat expansion), so a drum loop yields
  **both** individual hit stems **and** a GM drum `.mid`.
- Fail-soft: missing venv/ffmpeg/**drum model** → manifest `skipped` with a
  `stemforge setup-sota` hint; a nonzero CLI exit → `error` (stderr tail). Never
  crashes, never touches the main env.

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

# 4. SOTA separation (BS-Roformer / UVR zoo) — ISOLATED venv, never the main env:
stemforge setup-sota      # .venv-uvr: CUDA torch + audio-separator[gpu], cu124 torch forced LAST
# (do NOT `pip install audio-separator` into the main env — it clobbers CUDA torch/numpy)

# 5. System binaries: ffmpeg + rubberband (apt/brew/choco)
```

## Launch the app

- The UI is the **M87 Space-Tech workstation** (`app.py`): four workflow panels
  (Extract Stems · Drum Teardown · Melodic → MIDI · Full Teardown), not a
  checkbox grid. Colors/fonts are CSS variables at the top of `M87_CSS`
  (`--m87-bg`, `--m87-surface`, `--m87-accent`, `--m87-accent-2`, `--m87-text`,
  `--m87-mono`) — swap them for real M87 tokens. Theme + CSS pass via `launch()`
  on Gradio 6 (`build_ui(theme=, css=)` on 4/5).
- **CLI:** `stemforge ui` starts Gradio and (by default) opens the browser via
  gradio's `inbrowser=True`; `--no-open` suppresses it (`app.launch(open_browser=…)`).
- **Double-click:** `stemforge desktop-shortcut` drops a Desktop launcher —
  a `.lnk` on Windows (PowerShell `WScript.Shell.CreateShortcut`), `.command`
  on macOS, `.desktop` on Linux (`src/stemforge/desktop.py`). It points at the
  committed `scripts/launch_ui.{bat,sh}`, which **prepend** the winget Links dir
  (`%LOCALAPPDATA%\Microsoft\WinGet\Links`, for ffmpeg) and the repo dir to PATH,
  then run `python -m stemforge.cli ui`. Keeping the PATH-prepend in the scripts
  (not the shortcut) means both the double-click and the raw scripts behave the same.

## Gotchas (hard-won — do not regress)

- **(a)** NO `from __future__ import annotations` in `cli.py` — it breaks Typer
  bool flags and dataclass-from-dict building (annotation strings don't resolve).
  Other modules may use it; `cli.py` may not.
- **(b)** Every Typer **bool** option needs an explicit flag name, e.g.
  `typer.Option(False, "--midi")` — otherwise Typer/Click flag inference breaks.
- **(c)** Require `typer>=0.15` — older typer + Click 8.4 raises
  `make_metavar() TypeError`.
- **(d)** Keep `torch` / `demucs` / `onnxruntime` / `gradio` **lazily imported**
  (inside functions), so `import stemforge` and pytest stay GPU-free.
  `audio_separator` is stricter still: **never imported in this process at
  all** — subprocess only (see the isolation section above for the
  CPU-torch/numpy clobber that rule prevents).
- **(e)** Pipeline stages stay **fail-soft**: a missing model/dependency records
  `{"skipped": ...}` in the manifest (e.g. uvr backend without the `.venv-uvr`
  env set up, or ffmpeg off PATH) — never crash the run, never mutate the main env.
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

Mock the audio-separator CLI by monkeypatching `separate_uvr.subprocess.run`
(plus `find_cli` / `have_ffmpeg` for preflight) — no venv creation and no model
downloads in CI; mock Demucs by monkeypatching `stemforge.separate.separate`.
See `tests/test_separation_backends.py::fake_uvr_cli`.
