# StemForge

Local, GPU-accelerated audio workstation. One machine, four capabilities over a single shared analysis pass:

1. **Stem separation** — full mix → `vocals / drums / bass / other` (+ `guitar / piano`), three backends: Demucs, **BS-Roformer (SOTA)** via audio-separator, and a hybrid of both
2. **Melodic MIDI** — each melodic stem → `.mid` (notes + pitch bends)
3. **BPM time-stretch** — any stem → retimed to a target BPM, pitch preserved
4. **Drum decomposition** — drum stem → `kick / snare / toms / hi-hat / ride / crash`, plus drum `.mid` (GM-mapped, velocity-aware)

One analysis pass produces the tempo map + beat grid **once**; every downstream module consumes it, so all outputs stay phase- and grid-aligned.

---

## Install (order matters)

The #1 thing that breaks these builds is the **TensorFlow ↔ PyTorch CUDA collision** and the **madmom numpy pin**. StemForge sidesteps both: Basic Pitch runs via **ONNX Runtime** (no TF), and **beat_this** replaces madmom. Result: one clean PyTorch env.

### 1. Create the env

```bash
conda create -n env_torch python=3.11 -y
conda activate env_torch
# or: python3.11 -m venv env_torch && source env_torch/bin/activate
```

### 2. Install CUDA-matched PyTorch FIRST

```bash
nvidia-smi                      # note your CUDA driver version
# pick the matching index (cu121 / cu124 ...)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print('CUDA:', torch.cuda.is_available())"   # must print True
```

### 3. Install StemForge + conflict-free extras

```bash
pip install -e ".[all]"        # demucs, beat_this, onnxruntime-gpu, pyrubberband, python-stretch, gradio
```

> Do **not** install the `midi-tf` extra unless you accept that it pulls TensorFlow and can break the torch CUDA stack. The default `midi` path is ONNX-only.

Optional — SOTA separation (BS-Roformer & the UVR model zoo) runs in an **isolated venv**:

```bash
stemforge setup-sota      # one-time: .venv-uvr with CUDA torch + audio-separator[gpu]
stemforge doctor          # shows the .venv-uvr torch version + CUDA state
```

`setup-sota` installs audio-separator, then **force-reinstalls the cu124 torch
last** (audio-separator's deps otherwise pull a CPU torch — the reason SOTA once
ran on CPU) and verifies `torch.cuda.is_available()` inside the venv.

> Do **not** `pip install audio-separator` into the main env. On Windows/py3.11 it
> replaced CUDA torch with a CPU-only build and bumped numpy, breaking the demucs GPU
> stack — StemForge therefore calls it via subprocess from `.venv-uvr` only.

### 4. System binaries

```bash
# ffmpeg (decode) + rubberband (highest-quality stretch)
# Windows: choco install ffmpeg rubberband     |  macOS: brew install ffmpeg rubberband
# Ubuntu:  sudo apt install ffmpeg rubberband-cli
```

### 5. Model weights & external drum repos

Auto-downloaded on first use:
- **Demucs** (`htdemucs_ft`, `htdemucs_6s`) — via the `demucs` package.
- **BS-Roformer / UVR models** — via `audio-separator` into `models/uvr/` (presets `sota` / `max`).

Manual, dropped into `models/` (see `models/README.md`):
- **Basic Pitch ONNX** (`basic_pitch.onnx`, ~230 KB) — melodic transcription.
- **DrumSep** (Jarredou/aufr33 MelBand Roformer 6-stem) — drum decomposition.
- **ADTOF** (CRNN checkpoint) — drum transcription.

`beat_this` weights download automatically on first inference.

---

## Quickstart

```bash
# Day-one deliverable: separate a track into stems
stemforge separate song.wav --model htdemucs_ft -o out/

# Quality presets (also in the UI dropdown)
stemforge separate song.wav --preset fast   # demucs htdemucs (draft)
stemforge separate song.wav --preset best   # demucs htdemucs_ft, shifts=2
stemforge separate song.wav --preset sota   # BS-Roformer (vocals + instrumental)
stemforge separate song.wav --preset max    # hybrid 4-stem: roformer vocals + demucs residual

# Full pipeline
stemforge run song.wav \
  --preset max --target-bpm 120 \
  --midi --drum-split --drum-midi --stretch -o out/

# Launch the local web UI (opens your browser automatically)
stemforge ui
```

### Launch the app

```bash
stemforge ui                 # start the web UI and open the browser
stemforge ui --no-open       # start it without opening a browser
stemforge desktop-shortcut   # create a double-clickable Desktop launcher
```

`stemforge desktop-shortcut` drops a native launcher on your Desktop — a `.lnk`
on Windows, a `.command` on macOS, a `.desktop` entry on Linux. Double-click it
to start StemForge; it prepends ffmpeg (the winget Links dir on Windows) and the
repo to `PATH` first, so the app and the isolated `.venv-uvr` resolve. The same
logic lives in `scripts/launch_ui.bat` (Windows) and `scripts/launch_ui.sh`
(macOS/Linux) if you'd rather run a script directly.

Every run writes a per-song bundle:

```
out/<song>/
├─ stems/       vocals.wav drums.wav bass.wav other.wav [guitar.wav piano.wav]
├─ midi/        bass.mid other.mid ...
├─ drums/       kick.wav snare.wav toms.wav hihat.wav ride.wav crash.wav  +  drums.mid
├─ stretched/   <stem>_<bpm>bpm.wav
└─ manifest.json   source/target BPM, beat grid, model versions, per-stem ratios
```

---

## Build status (roadmap phases)

| Phase | Module | Status |
|-------|--------|--------|
| 0 | env / GPU sanity | ✅ `stemforge doctor` |
| 1 | Demucs separation | ✅ working |
| 2 | analysis (beat_this/librosa) + Rubber Band stretch | ✅ working |
| 3 | melodic MIDI (Basic Pitch ONNX) | 🟡 interface complete; drop in `basic_pitch.onnx` |
| 4 | drum decomposition (DrumSep) | 🟡 adapter; wire external repo/weights |
| 5 | drum MIDI (ADTOF + 7-class + velocity) | 🟡 velocity/mapping done; wire ADT model |
| 6 | orchestrator + Gradio UI | ✅ working |
| 7 | hardening / golden-file tests | 🟡 in progress |

✅ = runs today · 🟡 = interface + logic in place, needs the external model weight/repo wired

`import stemforge` never requires a GPU — heavy libs load lazily, so tests and CI run clean.

---

## Layout

```
stemforge/
├─ pyproject.toml
├─ configs/default.yaml       # every knob, maps to RunConfig
├─ models/                    # downloaded weights (git-ignored)
├─ src/stemforge/
│  ├─ ingest.py  analysis.py  separate.py  midi_melodic.py
│  ├─ drum_split.py  drum_midi.py  stretch.py
│  ├─ orchestrator.py  io_utils.py  cli.py  app.py
└─ tests/
```

## License

MIT.
