"""§5.4 midi_melodic — Basic Pitch (polyphonic AMT) via ONNX Runtime, with an
auditable transformation ledger.

Runs Basic Pitch through its ONNX backend (pass a ``.onnx`` model path) so no
TensorFlow is pulled into the torch CUDA env. Exposes the standard knobs, an
optional monophonic constraint (far higher accuracy on bass/vocal), and optional
quantization of note starts to the shared beat grid.

**Evidence (PR-A3).** Every transcription keeps three artifacts per stem —
``raw.mid`` (the untouched Basic Pitch output, never overwritten), ``cleaned.mid``
(after the monophonic constraint), and ``quantized.mid`` (after grid-snap) — plus
``events.json``: a per-note ledger carrying a stable id, confidence, status
(retained/dropped, with the drop reason), and the quantize delta. The ledger is
complete: :func:`reconstruct` rebuilds ``cleaned``/``quantized`` from ``raw`` +
``events.json`` exactly, so every transformation is inspectable and reversible.

If Basic Pitch / the ONNX model is not installed, ``transcribe_stems`` returns a
``skipped`` record rather than raising — the rest of the pipeline still runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import PathLike, ensure_dir, write_json

try:  # pragma: no cover
    from .orchestrator import MidiCfg
except Exception:  # pragma: no cover
    MidiCfg = Any  # type: ignore


class BasicPitchNotWired(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Per-note evidence model
# --------------------------------------------------------------------------- #
@dataclass
class NoteEvent:
    """One Basic Pitch note, tracked from raw prediction through every transform.

    ``id`` is stable and derived from the raw note's position, so the ledger can
    be matched back to ``raw.mid`` without storing IDs in the MIDI file itself.
    """

    id: str
    pitch: int
    start: float
    end: float
    confidence: float                 # Basic Pitch amplitude, retained (was discarded)
    status: str = "retained"          # "retained" | "dropped" (monophonic cleanup)
    drop_reason: str = ""
    quantize_delta: float = 0.0       # start shift applied by grid-snap (seconds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "pitch": self.pitch,
            "start": round(self.start, 4),
            "end": round(self.end, 4),
            "confidence": round(self.confidence, 4),
            "status": self.status,
            "drop_reason": self.drop_reason,
            "quantize_delta": round(self.quantize_delta, 6),
        }


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def transcribe_stems(
    stems: dict[str, Path],
    cfg: "MidiCfg",
    out_dir: PathLike,
    beat_grid: Any = None,
) -> dict[str, Any]:
    if not stems:
        return {"skipped": "no stems (run separation first)"}

    out_dir = Path(out_dir)
    result: dict[str, Any] = {"engine": cfg.engine, "files": {}, "notes": {}, "evidence": {}}
    targeted = [s for s in cfg.stems if s in stems]
    if not targeted:
        return {"skipped": f"none of the requested stems {cfg.stems} were produced"}

    for name in targeted:
        try:
            raw_pm, note_events = _predict_raw(stems[name], cfg)
        except BasicPitchNotWired as e:
            result.setdefault("skipped_stems", {})[name] = str(e)
            continue

        ledger, cleaned_pm, quantized_pm = _transcribe_one(
            raw_pm, note_events, cfg,
            monophonic=name in cfg.monophonic_stems, beat_grid=beat_grid)

        stem_dir = ensure_dir(out_dir / name)
        raw_pm.write(str(stem_dir / "raw.mid"))              # never overwritten
        cleaned_pm.write(str(stem_dir / "cleaned.mid"))
        quantized = ledger["quantize"]["applied"]
        if quantized:
            quantized_pm.write(str(stem_dir / "quantized.mid"))
        write_json(stem_dir / "events.json", ledger)

        final_pm = quantized_pm if quantized else cleaned_pm
        final_name = "quantized.mid" if quantized else "cleaned.mid"
        result["files"][name] = str(stem_dir / final_name)
        result["notes"][name] = sum(len(i.notes) for i in final_pm.instruments)
        result["evidence"][name] = {
            "raw": str(stem_dir / "raw.mid"),
            "cleaned": str(stem_dir / "cleaned.mid"),
            "quantized": str(stem_dir / "quantized.mid") if quantized else None,
            "events": str(stem_dir / "events.json"),
            "retained": ledger["counts"]["retained"],
            "dropped": ledger["counts"]["dropped"],
        }

    if not result["files"] and "skipped_stems" in result:
        return {"skipped": "Basic Pitch not available", "detail": result["skipped_stems"]}
    return result


def transcribe_file(
    audio_path: PathLike,
    cfg: "MidiCfg",
    monophonic: bool = False,
    beat_grid: Any = None,
):
    """Return a pretty_midi.PrettyMIDI for one melodic stem (the final transform).

    Backwards-compatible: yields the quantized MIDI when quantization applies,
    else the monophonic-cleaned MIDI, else the raw prediction. The full evidence
    (raw/cleaned/quantized + ledger) is available via :func:`transcribe_stems`.
    """
    raw_pm, note_events = _predict_raw(audio_path, cfg)
    ledger, cleaned_pm, quantized_pm = _transcribe_one(
        raw_pm, note_events, cfg, monophonic=monophonic, beat_grid=beat_grid)
    if ledger["quantize"]["applied"]:
        return quantized_pm
    if monophonic:
        return cleaned_pm
    return raw_pm


def _transcribe_one(
    raw_pm: Any, note_events: Any, cfg: "MidiCfg",
    monophonic: bool, beat_grid: Any,
) -> tuple[dict[str, Any], Any, Any]:
    """Build the ledger and the cleaned/quantized MIDI from a raw prediction."""
    events = build_events(raw_pm)
    if monophonic:
        mark_monophonic(events)
    quantize_meta: dict[str, Any] = {"applied": False, "subdivisions": 4, "max_displacement_s": 0.0}
    if cfg.quantize_to_grid and beat_grid is not None and getattr(beat_grid, "beats", None):
        mark_quantize(events, beat_grid, quantize_meta)
    ledger = build_ledger(events, monophonic, quantize_meta, raw_pm, note_events)
    cleaned_pm, quantized_pm = reconstruct(raw_pm, ledger)
    return ledger, cleaned_pm, quantized_pm


# --------------------------------------------------------------------------- #
# Basic Pitch prediction (raw)
# --------------------------------------------------------------------------- #
def _predict_raw(audio_path: PathLike, cfg: "MidiCfg"):
    """Run Basic Pitch and return ``(raw_pretty_midi, note_events)`` untouched."""
    model = _load_model(cfg)
    from basic_pitch.inference import predict  # type: ignore

    _model_output, midi_data, note_events = predict(
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
    return midi_data, note_events


# --------------------------------------------------------------------------- #
# Ledger construction + transforms (operate on NoteEvent, never mutate raw)
# --------------------------------------------------------------------------- #
def _sorted_notes(pm) -> list:
    """All notes across instruments in a stable order (start, pitch, end)."""
    notes = [n for inst in pm.instruments for n in inst.notes]
    return sorted(notes, key=lambda n: (round(n.start, 6), n.pitch, round(n.end, 6)))


def _event_id(i: int) -> str:
    return f"e{i:04d}"


def build_events(raw_pm) -> list[NoteEvent]:
    """Assign a stable id + confidence to every raw note, before any transform.

    Confidence is the Basic Pitch amplitude, carried on each note's velocity
    (0–127) — previously discarded with ``note_events``; here it is retained.
    """
    events: list[NoteEvent] = []
    for i, n in enumerate(_sorted_notes(raw_pm)):
        events.append(NoteEvent(
            id=_event_id(i), pitch=int(n.pitch), start=float(n.start), end=float(n.end),
            confidence=float(n.velocity) / 127.0))
    return events


def mark_monophonic(events: list[NoteEvent]) -> None:
    """One voice at a time: among temporally-overlapping notes keep the loudest;
    mark the rest ``dropped`` with the reason (logged, never silently removed)."""
    kept: list[NoteEvent] = []
    for e in sorted(events, key=lambda x: (x.start, -x.confidence, x.pitch)):
        overlapping = [k for k in kept if _overlaps(k, e)]
        louder = [k for k in overlapping if k.confidence >= e.confidence]
        if louder:
            e.status = "dropped"
            e.drop_reason = f"overlaps louder retained {louder[0].id}"
        else:
            for k in overlapping:                       # e is louder — supersede quieter kept
                k.status = "dropped"
                k.drop_reason = f"superseded by louder {e.id}"
                kept.remove(k)
            kept.append(e)


def _overlaps(a: NoteEvent, b: NoteEvent) -> bool:
    return a.start < b.end and b.start < a.end


def mark_quantize(events: list[NoteEvent], beat_grid: Any, meta: dict[str, Any],
                  subdivisions: int = 4) -> None:
    """Snap each RETAINED note's start to the nearest grid line, recording the
    per-note delta and the max displacement (so the snap is auditable)."""
    grid = _grid_lines(getattr(beat_grid, "beats", []), subdivisions)
    if grid.size == 0:
        return
    max_disp = 0.0
    for e in events:
        if e.status != "retained":
            continue
        j = int(np.argmin(np.abs(grid - e.start)))
        e.quantize_delta = float(grid[j] - e.start)
        max_disp = max(max_disp, abs(e.quantize_delta))
    meta["applied"] = True
    meta["subdivisions"] = subdivisions
    meta["max_displacement_s"] = round(float(max_disp), 6)


def build_ledger(events: list[NoteEvent], monophonic: bool, quantize_meta: dict[str, Any],
                 raw_pm: Any, note_events: Any = None) -> dict[str, Any]:
    pitch_bends = [{"time": round(float(pb.time), 4), "pitch": int(pb.pitch)}
                   for inst in raw_pm.instruments for pb in getattr(inst, "pitch_bends", [])]
    return {
        "schema": "stemforge.midi.events/1",
        "monophonic": monophonic,
        "quantize": {
            "applied": bool(quantize_meta.get("applied", False)),
            "subdivisions": int(quantize_meta.get("subdivisions", 4)),
            "max_displacement_s": float(quantize_meta.get("max_displacement_s", 0.0)),
        },
        "pitch_bends": pitch_bends,
        "raw_note_events": (len(note_events) if note_events is not None else None),
        "counts": {
            "raw": len(events),
            "retained": sum(1 for e in events if e.status == "retained"),
            "dropped": sum(1 for e in events if e.status == "dropped"),
        },
        "events": [e.to_dict() for e in events],
    }


def reconstruct(raw_pm: Any, ledger: dict[str, Any]) -> tuple[Any, Any]:
    """Rebuild ``(cleaned_pm, quantized_pm)`` from ``raw_pm`` + the ledger alone.

    The ledger is complete: matching each raw note to its ledger entry by the
    stable id, ``cleaned`` keeps the retained notes and ``quantized`` shifts each
    by its recorded delta. Instrument-level pitch bends are copied verbatim.
    """
    import pretty_midi

    by_id = {e["id"]: e for e in ledger["events"]}
    cleaned = pretty_midi.PrettyMIDI()
    quantized = pretty_midi.PrettyMIDI()
    ci = pretty_midi.Instrument(program=0, is_drum=False, name="cleaned")
    qi = pretty_midi.Instrument(program=0, is_drum=False, name="quantized")

    for i, n in enumerate(_sorted_notes(raw_pm)):
        rec = by_id.get(_event_id(i))
        if rec is None or rec["status"] != "retained":
            continue
        ci.notes.append(pretty_midi.Note(velocity=n.velocity, pitch=n.pitch,
                                          start=n.start, end=n.end))
        d = float(rec.get("quantize_delta", 0.0))
        qs = max(0.0, n.start + d)
        qi.notes.append(pretty_midi.Note(velocity=n.velocity, pitch=n.pitch,
                                         start=qs, end=max(qs + 1e-3, n.end + d)))
    cleaned.instruments.append(ci)
    quantized.instruments.append(qi)
    for inst in raw_pm.instruments:                       # preserve expression evidence
        for pb in getattr(inst, "pitch_bends", []):
            ci.pitch_bends.append(pretty_midi.PitchBend(pitch=pb.pitch, time=pb.time))
            qi.pitch_bends.append(pretty_midi.PitchBend(pitch=pb.pitch, time=pb.time))
    return cleaned, quantized


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
# Grid helpers
# --------------------------------------------------------------------------- #
def _grid_lines(beats: list[float], subdivisions: int = 4) -> np.ndarray:
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
