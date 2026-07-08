"""§5.6 drum_midi — drum transcription to a General-MIDI, velocity-aware .mid.

Two onset sources:
  1. **ADTOF** (external, CRNN 5-class) via ``external_cmd`` that emits a .mid we
     adopt directly — the SOTA path.
  2. **Parts-based** (no ADT model needed): when drum_split has produced per-part
     WAVs, we detect onsets per part and assign the correct GM note. This makes
     drum MIDI work off Phase-4 output alone.

Regardless of source, **velocity** comes from each part's RMS loudness envelope
(equal-ish weighting, configurable window/hop): the max RMS in a short window
after each onset, scaled to MIDI 1–127. The 5→7 expansion (open/closed hi-hat via
a decay test) is applied on the hi-hat part.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .io_utils import PathLike, load_audio

try:  # pragma: no cover
    from .orchestrator import DrumMidiCfg
except Exception:  # pragma: no cover
    DrumMidiCfg = Any  # type: ignore

# General MIDI percussion note numbers.
GM = {
    "kick": 36,
    "snare": 38,
    "hihat": 42,        # closed by default
    "hihat_closed": 42,
    "hihat_open": 46,
    "toms": 47,         # mid tom
    "ride": 51,
    "crash": 49,
}

NOTE_LEN_S = 0.06


def transcribe(
    drums_path: PathLike | None,
    drum_parts: dict[str, str],
    cfg: "DrumMidiCfg",
    out_dir: PathLike,
    beat_grid: Any = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    bpm = float(getattr(beat_grid, "source_bpm", 0) or 120.0)

    # 1) SOTA path: let ADTOF (or another ADT) write the MIDI, we adopt it.
    if cfg.external_cmd:
        return _adt_external(cfg.external_cmd, drums_path, out_dir)

    # 2) Parts-based path: build the MIDI from separated parts.
    if drum_parts:
        return _from_parts(drum_parts, cfg, out_dir, bpm)

    return {
        "skipped": (
            "no drum parts and no ADT external_cmd. Enable drums.split (Phase 4) for "
            "the parts-based path, or set drums.midi.external_cmd to an ADTOF command."
        )
    }


# --------------------------------------------------------------------------- #
# ADT external adapter
# --------------------------------------------------------------------------- #
def _adt_external(cmd_template: str, drums_path: PathLike | None, out_dir: Path) -> dict[str, Any]:
    if drums_path is None or not Path(drums_path).is_file():
        return {"skipped": "no drums stem for ADT"}
    cmd = cmd_template.format(input=str(drums_path), output_dir=str(out_dir))
    try:
        subprocess.run(shlex.split(cmd), check=True, capture_output=True, text=True)
    except FileNotFoundError as e:
        return {"error": f"ADT external_cmd not found: {e}"}
    except subprocess.CalledProcessError as e:
        return {"error": f"ADT external_cmd failed: {e.stderr[-500:]}"}

    mids = list(out_dir.glob("*.mid"))
    if not mids:
        return {"error": "ADT external_cmd produced no .mid"}
    target = out_dir / "drums.mid"
    if mids[0] != target:
        shutil.copy(mids[0], target)
    return {"source": "adt_external", "file": str(target)}


# --------------------------------------------------------------------------- #
# Parts-based transcription (fully implemented)
# --------------------------------------------------------------------------- #
def _from_parts(drum_parts: dict[str, str], cfg: "DrumMidiCfg", out_dir: Path, bpm: float) -> dict[str, Any]:
    import pretty_midi

    events: list[tuple[float, int, float]] = []  # (start, note, raw_peak)
    global_peak = 1e-9
    counts: dict[str, int] = {}

    for part, file in drum_parts.items():
        if part not in GM and part != "hihat":
            continue
        p = Path(file)
        if not p.is_file():
            continue
        audio = load_audio(p).to_mono()
        y = audio.samples[0]
        sr = audio.sample_rate
        onsets = _onsets(y, sr)
        env, env_t = _rms_env(y, sr, cfg.velocity_window_ms, cfg.velocity_hop_ms)
        note_fn = _note_selector(part, cfg, y, sr)
        for t in onsets:
            peak = _peak_in_window(env, env_t, t, cfg.velocity_window_ms / 1000.0) \
                if cfg.velocity_from_stems else 1.0
            events.append((float(t), note_fn(t), float(peak)))
            global_peak = max(global_peak, peak)
        counts[part] = len(onsets)

    if not events:
        return {"skipped": "no onsets detected in drum parts"}

    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    inst = pretty_midi.Instrument(program=0, is_drum=True, name="drums")
    for start, note, peak in events:
        vel = _scale_velocity(peak, global_peak) if cfg.velocity_from_stems else 100
        inst.notes.append(pretty_midi.Note(velocity=vel, pitch=note,
                                            start=start, end=start + NOTE_LEN_S))
    inst.notes.sort(key=lambda n: n.start)
    pm.instruments.append(inst)

    target = out_dir / "drums.mid"
    pm.write(str(target))
    return {
        "source": "parts",
        "file": str(target),
        "note_count": len(events),
        "onsets_per_part": counts,
        "expanded_7class": bool(cfg.expand_to_7class),
    }


def _note_selector(part: str, cfg: "DrumMidiCfg", y: np.ndarray, sr: int) -> Callable[[float], int]:
    """Return a function onset_time -> GM note, applying 5->7 hi-hat expansion."""
    if part == "hihat" and cfg.expand_to_7class:
        return lambda t: GM["hihat_open"] if _hat_is_open(y, sr, t) else GM["hihat_closed"]
    return lambda t, n=GM.get(part, GM["snare"]): n


# --------------------------------------------------------------------------- #
# DSP helpers
# --------------------------------------------------------------------------- #
def _onsets(y: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    return librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True)


def _rms_env(y: np.ndarray, sr: int, window_ms: float, hop_ms: float) -> tuple[np.ndarray, np.ndarray]:
    import librosa

    frame = max(256, int(sr * window_ms / 1000.0))
    hop = max(64, int(sr * hop_ms / 1000.0))
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    return rms, times


def _peak_in_window(env: np.ndarray, env_t: np.ndarray, t: float, window_s: float) -> float:
    mask = (env_t >= t) & (env_t <= t + window_s)
    if not np.any(mask):
        idx = int(np.argmin(np.abs(env_t - t)))
        return float(env[idx])
    return float(np.max(env[mask]))


def _scale_velocity(peak: float, global_peak: float) -> int:
    """Perceptual-ish scaling: sqrt compression, floor at 10 so hits stay audible."""
    ratio = (peak / global_peak) ** 0.5 if global_peak > 0 else 0.0
    return int(np.clip(round(10 + 117 * ratio), 1, 127))


def _hat_is_open(y: np.ndarray, sr: int, t: float, short_ms: float = 25.0, long_ms: float = 130.0) -> bool:
    """Open vs closed hi-hat via decay: open hats sustain, closed die fast."""
    i0 = int(t * sr)
    a = _rms_at(y, i0, int(short_ms / 1000.0 * sr))
    b = _rms_at(y, i0 + int(long_ms / 1000.0 * sr), int(short_ms / 1000.0 * sr))
    if a <= 1e-9:
        return False
    return (b / a) > 0.35  # still ringing well after the hit => open


def _rms_at(y: np.ndarray, start: int, length: int) -> float:
    seg = y[max(0, start):max(0, start) + max(1, length)]
    if seg.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))
