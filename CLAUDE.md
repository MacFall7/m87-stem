# CLAUDE.md — StemForge

StemForge is a **working local audio workstation**, verified end-to-end on an RTX 4090:
Demucs separation, beat_this analysis, Basic Pitch → MIDI via ONNX (no TensorFlow),
Rubber Band stretch, real drum teardown (inagoy/drumsep), a Typer CLI, and a
bespoke **FastAPI web app** (the M87 workstation — Gradio has been retired).

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
- `src/stemforge/drum_split.py` — drum teardown. Default `backend:
  demucs_inagoy` runs the **inagoy/drumsep Demucs checkpoint in the MAIN env**
  (via `demucs`, no venv, GPU-fast), auto-downloaded to `models/drumsep/`,
  splitting a loop/stem into kick/snare/toms/other. `backend: uvr` (isolated
  `.venv-uvr`) is kept but is NOT the default — audio-separator's registry has
  no per-hit drum model. Accepts a raw drum LOOP (`from_input`); parts feed the
  existing parts-based `drum_midi` (onset + RMS-velocity + GM mapping).
- `src/stemforge/stretch.py` — pitch-preserving time-stretch (rubberband →
  signalsmith → librosa chain). `stretch_stems` retimes the separated stems;
  `detect_bpm()` + `match_bpm_file()` power the whole-file **Match BPM** workflow
  (no separation): decode → detect (or `source_bpm` override for half/double
  errors) → stretch the whole file → one `<stem>_<bpm>bpm.wav`. Fail-soft.
- `src/stemforge/cli.py` — Typer CLI (`doctor|setup-sota|separate|analyze|run|match-bpm|ui|desktop-shortcut`).
- `src/stemforge/webapp.py` + `src/stemforge/web/` — the **M87 workstation**: a
  FastAPI backend (job-based; `/api/{extract,drum-teardown,melodic-midi,full-teardown,match-bpm}`
  + `/api/detect-bpm`, poll `/api/job/{id}` or SSE `/api/progress/{id}`,
  `/api/file`, `/api/download-all`, `/api/open-folder`, `/api/health`) serving a
  static single-page front-end (`web/index.html` + `web/assets/{styles.css,app.js}`,
  wavesurfer.js from cdnjs). All colors/fonts are CSS variables at the top of
  `styles.css` (`--bg`, `--cyan`, `--violet`, `--grad-primary`, …). Launched by
  `stemforge ui` (uvicorn).
- `tests/` — **GPU-free**; every heavy dep is mocked. The FastAPI routes run with
  the Pipeline mocked (`tests/test_webapp.py`); the drum backends mock demucs /
  the audio-separator CLI (`tests/test_drum_split.py`); synthetic audio from
  `conftest.py`.

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

## Drum teardown (default: inagoy/drumsep in the MAIN env)

**Why not audio-separator:** its model registry has NO per-hit drum model —
`--list_filter=drums` only lists kit-isolation models. So the drum teardown uses
the **inagoy/drumsep Demucs checkpoint** run in the MAIN env via the already-
installed `demucs` package (it's a Demucs checkpoint — no venv, GPU-fast).

`drums.split.backend = demucs_inagoy` (default) — `drum_split._demucs_inagoy_split`:
- **Checkpoint**: auto-downloaded on first use to `drums.split.inagoy_model_dir`
  (`models/drumsep/`, git-ignored) from `drums.split.inagoy_url` (overridable) —
  the public, non-gated `Eddycrack864/Drumsep/modelo_final.th` mirror (the old
  `mnstrmnl/drumsep` URL is now gated → HTTP 401). Loaded with
  `demucs.states.load_model` and applied via the shared
  `separate.apply_model_to_audio`. **torch≥2.6:** `torch.load` defaults
  `weights_only=True`, which rejects the pickled `HDemucs`/`HTDemucs` globals;
  `_load_demucs_checkpoint` retries with `weights_only=False` (trusted MIT file)
  — works on older torch too, fail-soft on genuine corruption.
- **Mapping**: the model's 4 sources → canonical kick/snare/toms/other by
  keyword (English or Spanish labels), with a positional kit-order fallback so
  nothing is dropped (`_map_inagoy_sources`).
- **Input**: `drums.split.from_input=true` tears down the raw input loop;
  otherwise it uses `ctx.stems["drums"]`. The orchestrator passes `ctx.audio` as
  an `AudioTensor` for the from-input case.
- The produced parts feed the parts-based `drum_midi` (onset + RMS→velocity + GM,
  5→7 hi-hat expansion), so a loop yields **both** hit stems **and** a GM `.mid`.
- **Fail-soft**: a checkpoint that can't download/load, or missing demucs, →
  manifest `skipped` with a fix hint; a runtime apply failure → `error`. Never
  crashes, never mutates anything.

`backend: uvr` still exists (isolated `.venv-uvr`, `separate_uvr.separate_drums`,
`--list_filter=drums` discovery) but is not the default and produces kit
isolation, not per-hit stems.

## Verified install recipe (order matters)

```bash
# 1. python 3.11 env, then CUDA-matched torch FIRST (cu124 on the 4090 box)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.cuda.is_available())"   # must print True

# 2. StemForge + conflict-free extras
pip install -e .[all]        # demucs, beat_this, onnxruntime-gpu, pyrubberband, fastapi+uvicorn

# 3. Melodic MIDI, TensorFlow-free (basic-pitch's deps would pull TF):
pip install --no-deps basic-pitch
pip install "resampy<0.4.3" scikit-learn mir_eval

# 4. SOTA separation (BS-Roformer / UVR zoo) — ISOLATED venv, never the main env:
stemforge setup-sota      # .venv-uvr: CUDA torch + audio-separator[gpu], cu124 torch forced LAST
# (do NOT `pip install audio-separator` into the main env — it clobbers CUDA torch/numpy)

# 5. System binaries: ffmpeg + rubberband (apt/brew/choco)
```

## Launch the app

- The UI is the **M87 workstation** — a bespoke FastAPI backend (`webapp.py`)
  serving a static SPA (`web/`), **not Gradio** (retired). One page, five
  workflows via a left rail (Extract Stems · Drum Teardown · Melodic → MIDI ·
  Full Teardown · Match BPM), no page reloads. Match BPM has its own controls
  (Detect BPM button → surfaces detected/half/double, a Target + Source-override
  box, engine select). Deep-space theme; all colors/fonts are CSS variables at
  the top of `web/assets/styles.css` (`--bg`, `--surface`, `--cyan`, `--violet`,
  `--indigo`, `--grad-primary`, `--text`, `--font-mono`, …) — swap them for exact
  M87 tokens. Real waveforms via wavesurfer.js (cdnjs).
- **CLI:** `stemforge ui` runs uvicorn and (by default) opens the browser once
  the server is up; `--no-open` suppresses it (`webapp.launch(open_browser=…)`).
- Progress: each POST starts a background job; the SPA polls `/api/job/{id}`
  (SSE stream also at `/api/progress/{id}`). The Pipeline's `on_stage` hook feeds
  coarse per-stage progress. `web/` ships as setuptools `package-data`.
- **Double-click:** `stemforge desktop-shortcut` drops a Desktop launcher —
  a `.lnk` on Windows (PowerShell `WScript.Shell.CreateShortcut`), `.command`
  on macOS, `.desktop` on Linux (`src/stemforge/desktop.py`). It points at the
  committed `scripts/launch_ui.{bat,sh}`, which **prepend** the winget Links dir
  (`%LOCALAPPDATA%\Microsoft\WinGet\Links`, for ffmpeg) and the repo dir to PATH,
  then run `python -m stemforge.cli ui`. Keeping the PATH-prepend in the scripts
  (not the shortcut) means both the double-click and the raw scripts behave the same.

## Gotchas (hard-won — do not regress)

- **(a)** NO `from __future__ import annotations` in `cli.py` **or `webapp.py`** —
  it breaks Typer bool flags / dataclass-from-dict in the CLI, and FastAPI's
  request-time resolution of route param annotations (`UploadFile`/`Form`) in
  the web app (stringized annotations raise `PydanticUserError`). Other modules
  may use it; these two may not.
- **(b)** Every Typer **bool** option needs an explicit flag name, e.g.
  `typer.Option(False, "--midi")` — otherwise Typer/Click flag inference breaks.
- **(c)** Require `typer>=0.15` — older typer + Click 8.4 raises
  `make_metavar() TypeError`.
- **(d)** Keep `torch` / `demucs` / `onnxruntime` / `fastapi` / `uvicorn`
  **lazily imported** (inside functions), so `import stemforge` and pytest stay
  GPU-free and dependency-light. `audio_separator` is stricter still: **never
  imported in this process at all** — subprocess only (see the isolation section
  above for the CPU-torch/numpy clobber that rule prevents).
- **(e)** Pipeline stages stay **fail-soft**: a missing model/dependency records
  `{"skipped": ...}` in the manifest (e.g. the inagoy checkpoint can't download,
  or ffmpeg off PATH) — never crash the run, never mutate the main env. The web
  job runner mirrors this: a stage skip → `done` with a note; a real exception →
  job `error`.
- **(f)** Output dirs are **slugified** (`io_utils.slugify`): `out/<slug(song)>/…`.
  The web `/api/file` + `/api/download-all` routes only serve paths under the
  output root / upload dir (path-guarded).
- Container/CI note: `pretty_midi` 0.2.10 can fail to build on Debian-patched
  setuptools (`AttributeError: install_layout`). Workaround:
  `pip install "setuptools==59.8.0" && pip install --no-build-isolation pretty_midi`,
  then restore `setuptools>=68`.

## Tests

```bash
python -m pytest          # must stay green, GPU-free, no network/model downloads
```

FastAPI routes are tested with the Pipeline mocked (`tests/test_webapp.py`, needs
`fastapi`+`httpx`, skipped if absent); the drum backends mock demucs / the
audio-separator CLI. No model downloads, no real separation in CI.

Mock the audio-separator CLI by monkeypatching `separate_uvr.subprocess.run`
(plus `find_cli` / `have_ffmpeg` for preflight) — no venv creation and no model
downloads in CI; mock Demucs by monkeypatching `stemforge.separate.separate`.
See `tests/test_separation_backends.py::fake_uvr_cli`.
