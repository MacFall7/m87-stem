# Model weights

Weights are **not** committed (git-ignored). Drop them here.

| File / dir | Used by | Source |
|------------|---------|--------|
| `basic_pitch.onnx` | `midi_melodic.py` | Basic Pitch ONNX export (~230 KB) — HuggingFace `AEmotionStudio/basic-pitch-onnx-models` |
| `drumsep_roformer/` | `drum_split.py` | Jarredou/aufr33 MelBand Roformer 6-stem (config + checkpoint) |
| `larsnet/` | `drum_split.py` (fallback) | LarsNet 5-stem U-Nets — `polimi-ispl/larsnet` |
| `adtof/` | `drum_midi.py` | ADTOF CRNN checkpoint — from the ADTOF repo |

Auto-downloaded (no action needed):
- **Demucs** `htdemucs_ft` / `htdemucs_6s` — cached by the `demucs` package on first run.
- **beat_this** — weights fetched on first inference.

Point non-default locations at these via `configs/default.yaml`
(`midi.onnx_model_path`, `drums.split.external_cmd`, `drums.midi.external_cmd`).
