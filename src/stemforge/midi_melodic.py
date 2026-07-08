"""§5.4 midi_melodic — Basic Pitch (polyphonic AMT) via ONNX Runtime.

Runs Basic Pitch through its ONNX backend (pass a ``.onnx`` model path) so no
TensorFlow is pulled into the torch CUDA env. Exposes the standard knobs, an
optional monophonic constraint (far higher accuracy on bass/vocal), and optional
quantization of note starts to the shared beat grid.

If Basic Pitch / the ONNX model is not installed, ``transcribe_stems`` returns a
``skipped`` record rather than raising — the rest of the pipeline still runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import PathLike

try:  # pragma: no cover
    from .orchestrator import MidiCfg
except Exception:  # pragma: no cover
    MidiCfg = Any  # type: ignore


class BasicPitchNotWired(RuntimeError):
    pass


def transcribe_stems(
    stems: dict[str, Path],
    cfg: "MidiCfg",
    out_dir: PathLike,
    beat_grid: Any = None,
) -> dict[str, Any]:
    if not stems:
        return {"skipped": "no stems (run separation first)"}

    out_dir = Path(out_dir)
    result: dict[str, Any] = {"engine": cfg.engine, "files": {}, "notes": {}}
    targeted = [s for s in cfg.stems if s in stems]
    if not targeted:
        return {"skipped": f"none of the requested stems {cfg.stems} were produced"}

    for name in targeted:
        try:
            pm = transcribe_file(stems[name], cfg,
                                 monophonic=name in cfg.monophonic_stems,
                                 beat_grid=beat_grid)
        except BasicPitchNotWired as e:
            result.setdefault("skipped_stems", {})[name] = str(e)
            continue
        outp = out_dir / f"{name}.mid"
        pm.write(str(outp))
        result["files"][name] = str(outp)
        result["notes"][name] = sum(len(i.notes) for i in pm.instruments)
    if not result["files"] and "skipped_stems" in result:
        return {"skipped": "Basic Pitch not available", "detail": result["skipped_stems"]}
    return result


def transcribe_file(
    audio_path: PathLike,
    cfg: "MidiCfg",
    monophonic: bool = False,
    beat_grid: Any = None,
):
    """Return a pretty_midi.PrettyMIDI for one melodic stem."""
    model = _load_model(cfg)
    from basic_pitch.inference import predict  # type: ignore

    _model_output, midi_data, _note_events = predict(
        str(audio_path),
        model,
        onset_threshold=cfg.onset_threshold,
        frame_threshold=cfg.frame_threshold,
        minimum_note_length=cfg.min_note_length_ms,
        minimum_frequency=cfg.min_frequency,
        maximum_frequency=cfg.max_frequency,
        multiple_pitch_bends=cfg.include_pitch_bends,
        melodia_trick=cfg.melodia_trick,
    )

    if monophonic:
        _make_monophonic(midi_data)
    if cfg.quantize_to_grid and beat_grid is not None and getattr(beat_grid, "beats", None):
        _quantize_to_grid(midi_data, beat_grid)
    return midi_data


# --------------------------------------------------------------------------- #
# Model loading (ONNX preferred, no TensorFlow)
# --------------------------------------------------------------------------- #
def _load_model(cfg: "MidiCfg"):
    try:
        from basic_pitch.inference import Model  # type: ignore
    except ImportError as e:
        raise BasicPitchNotWired(
            "basic-pitch not installed. `pip install .[midi]` and place the ONNX "
            "model at configs' midi.onnx_model_path (see models/README.md)."
        ) from e

    if cfg.engine == "onnx":
        path = Path(cfg.onnx_model_path)
        if not path.is_file():
            raise BasicPitchNotWired(
                f"ONNX model not found at {path}. Download basic_pitch.onnx "
                "(AEmotionStudio/basic-pitch-onnx-models) into models/."
            )
        return Model(str(path))  # onnxruntime backend, no TF

    # engine == 'basic_pitch' -> default packaged model (may pull TF/coreml/tflite)
    from basic_pitch import ICASSP_2022_MODEL_PATH  # type: ignore

    return Model(ICASSP_2022_MODEL_PATH)


# --------------------------------------------------------------------------- #
# Post-processing
# --------------------------------------------------------------------------- #
def _make_monophonic(pm) -> None:
    """Remove temporal overlaps, keeping the louder note. One voice at a time."""
    for inst in pm.instruments:
        notes = sorted(inst.notes, key=lambda n: (n.start, -n.velocity))
        kept: list = []
        for n in notes:
            if kept and n.start < kept[-1].end:
                prev = kept[-1]
                if n.velocity > prev.velocity:
                    prev.end = min(prev.end, n.start)  # truncate previous
                    if prev.end <= prev.start:
                        kept.pop()
                    kept.append(n)
                else:
                    continue  # drop the overlapping quieter note
            else:
                kept.append(n)
        inst.notes = kept


def _quantize_to_grid(pm, beat_grid, subdivisions: int = 4) -> None:
    """Snap note starts to the nearest grid line (default: 16th notes)."""
    grid = _grid_lines(beat_grid.beats, subdivisions)
    if grid.size == 0:
        return
    for inst in pm.instruments:
        for n in inst.notes:
            j = int(np.argmin(np.abs(grid - n.start)))
            delta = grid[j] - n.start
            n.start = max(0.0, n.start + delta)
            n.end = max(n.start + 1e-3, n.end + delta)  # preserve duration


def _grid_lines(beats: list[float], subdivisions: int) -> np.ndarray:
    if len(beats) < 2:
        return np.asarray(beats, dtype=np.float64)
    b = np.asarray(beats, dtype=np.float64)
    lines: list[float] = []
    for i in range(len(b) - 1):
        step = (b[i + 1] - b[i]) / subdivisions
        for k in range(subdivisions):
            lines.append(b[i] + k * step)
    lines.append(float(b[-1]))
    return np.asarray(lines, dtype=np.float64)
