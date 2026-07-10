# Model weights

Weights are **not** committed (git-ignored). Drop them here.

| File / dir | Used by | Source |
|------------|---------|--------|
| `basic_pitch.onnx` | `midi_melodic.py` | Basic Pitch ONNX export (~230 KB) — HuggingFace `AEmotionStudio/basic-pitch-onnx-models` |
| `larsnet/` | `drum_split.py` (fallback) | LarsNet 5-stem U-Nets — `polimi-ispl/larsnet` |
| `adtof/` | `drum_midi.py` (optional SOTA drum MIDI) | ADTOF CRNN checkpoint — from the ADTOF repo |

Auto-downloaded (no action needed):
- **Demucs** `htdemucs_ft` / `htdemucs_6s` — cached by the `demucs` package on first run.
- **inagoy/drumsep** (`drums.split` default) — a Demucs checkpoint fetched to
  `models/drumsep/modelo_final.th` on first use. It is **sha256-pinned**
  (`drums.split.inagoy_sha256`) and refused unless the hash matches — no verified
  hash, no unsafe `torch.load`. See the security note in the top-level README.
- **MDX23C DrumSep** (`backend: uvr_drumsep`, optional) — per-hit model run in the
  isolated `.venv-uvr` (`stemforge setup-sota`), cached under `models/uvr/`.
- **beat_this** — weights fetched on first inference.

Point non-default locations at these via `configs/default.yaml`
(`midi.onnx_model_path`, `drums.split.inagoy_url`/`inagoy_sha256`,
`drums.split.external_cmd`, `drums.midi.external_cmd`).
