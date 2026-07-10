"""Cymbal MIDI on the default (inagoy) drum path — A-series PR-A1.

Receipt (verified at main): ``drum_split`` maps every cymbal (platillos/cymbals/
hihat/ride/crash) into the single ``other`` stem, and ``drum_midi._from_parts``
skipped any part not in ``GM`` and not ``"hihat"`` — so the default ``demucs_inagoy``
teardown emitted kick/snare/toms only, with zero cymbal MIDI and the open/closed
hat classifier unreachable. This regression pins the loss: the default-path drum
MIDI must carry at least one cymbal event.

GPU-free and deterministic — synthetic decaying-noise cymbal hits, seeded RNG.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from stemforge import drum_midi
from stemforge.io_utils import ensure_dir
from stemforge.orchestrator import load_config

CYMBAL_PITCHES = {
    drum_midi.GM["hihat_closed"],  # 42
    drum_midi.GM["hihat_open"],    # 46
    drum_midi.GM["crash"],         # 49
    drum_midi.GM["ride"],          # 51
}
SR = 44100


# --------------------------------------------------------------------------- #
# Synthetic drum-part audio (deterministic)
# --------------------------------------------------------------------------- #
def _decay_noise(dur: float, tau: float, seed: int, amp: float = 1.0, sr: int = SR) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * sr)) / sr
    return (amp * rng.standard_normal(t.size) * np.exp(-t / tau)).astype(np.float32)


def _tone(dur: float, freq: float, tau: float, amp: float = 0.9, sr: int = SR) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t) * np.exp(-t / tau)).astype(np.float32)


def _place(hits: list[tuple[float, np.ndarray]], total_s: float = 2.0, sr: int = SR) -> np.ndarray:
    y = np.zeros(int(total_s * sr), dtype=np.float32)
    for start, sig in hits:
        i = int(start * sr)
        end = min(y.size, i + sig.size)
        y[i:end] += sig[: end - i]
    return y


def _write(path: Path, y: np.ndarray, sr: int = SR) -> str:
    sf.write(str(path), y, sr, subtype="FLOAT")
    return str(path)


def _cymbal_bucket(seed_base: int = 100) -> np.ndarray:
    """An 'other' stem holding three cymbal hits: closed / open / crash decays."""
    return _place([
        (0.1, _decay_noise(0.5, 0.020, seed_base + 0)),   # closed hi-hat
        (0.7, _decay_noise(0.5, 0.200, seed_base + 1)),   # open hi-hat
        (1.3, _decay_noise(0.6, 0.600, seed_base + 2)),   # crash
    ])


def _pitches(mid_path: str) -> set[int]:
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(mid_path)
    return {n.pitch for inst in pm.instruments for n in inst.notes}


def _midi_cfg():
    return load_config().drums.midi


# --------------------------------------------------------------------------- #
# RED regression: the default 'other' bucket must yield cymbal MIDI
# --------------------------------------------------------------------------- #
def test_default_path_other_bucket_yields_cymbal_events(tmp_path):
    """Fails on the pre-fix tree (the 'other' cymbal bucket was dropped): the
    default-path drum MIDI must contain at least one cymbal event."""
    parts = {
        "kick": _write(tmp_path / "kick.wav", _place([
            (0.1, _tone(0.4, 80.0, 0.08)),
            (0.7, _tone(0.4, 80.0, 0.08)),
            (1.3, _tone(0.4, 80.0, 0.08)),
        ])),
        "other": _write(tmp_path / "other.wav", _cymbal_bucket()),
    }
    res = drum_midi._from_parts(parts, _midi_cfg(), ensure_dir(tmp_path / "out"), bpm=120.0)
    assert "file" in res, res
    got = _pitches(res["file"])
    assert CYMBAL_PITCHES & got, (
        f"expected at least one cymbal event from the default 'other' bucket, got {sorted(got)}")
