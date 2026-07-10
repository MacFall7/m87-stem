"""Drum calibration — A-series PR-A4.

Receipt (verified at main): drum_midi had NO cross-part de-dup, so separation
bleed (a weak phantom in one part coincident with a loud hit in another) survived
as a duplicate coincident note. This regression pins that: coincident cross-part
bleed must not produce duplicate notes.

GPU-free and deterministic — synthetic hits, seeded RNG.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

pretty_midi = pytest.importorskip("pretty_midi")

from stemforge import drum_midi                     # noqa: E402
from stemforge.io_utils import ensure_dir           # noqa: E402
from stemforge.orchestrator import load_config      # noqa: E402

SR = 44100


def _noise(dur: float, tau: float, seed: int, amp: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * SR)) / SR
    return (amp * rng.standard_normal(t.size) * np.exp(-t / tau)).astype(np.float32)


def _sine(dur: float, hz: float, tau: float, amp: float = 0.9) -> np.ndarray:
    t = np.arange(int(dur * SR)) / SR
    return (amp * np.sin(2 * np.pi * hz * t) * np.exp(-t / tau)).astype(np.float32)


def _place(hits: list[tuple[float, np.ndarray]], total_s: float = 2.0) -> np.ndarray:
    y = np.zeros(int(total_s * SR), dtype=np.float32)
    for start, sig in hits:
        i = int(start * SR)
        end = min(y.size, i + sig.size)
        y[i:end] += sig[: end - i]
    return y


def _write(path: Path, y: np.ndarray) -> str:
    sf.write(str(path), y, SR, subtype="FLOAT")
    return str(path)


def _run(parts, tmp_path, **cfg_over):
    cfg = load_config().drums.midi
    for k, v in cfg_over.items():
        setattr(cfg, k, v)
    return drum_midi._from_parts(parts, cfg, ensure_dir(tmp_path / "out"), bpm=120.0)


# --------------------------------------------------------------------------- #
# RED — separation bleed must not produce coincident cross-part duplicates
# --------------------------------------------------------------------------- #
def test_cross_part_bleed_is_deduped(tmp_path):
    """A weak phantom in the toms part, coincident with each loud kick (bleed),
    must not survive as a duplicate note. Fails pre-fix (no cross-part de-dup)."""
    kick = _place([(0.3, _noise(0.3, 0.05, 1, amp=1.0)),
                   (0.9, _noise(0.3, 0.05, 2, amp=1.0)),
                   (1.5, _noise(0.3, 0.05, 3, amp=1.0))])
    toms = _place([(0.304, _noise(0.3, 0.05, 4, amp=0.12)),   # phantom bleed x3
                   (0.904, _noise(0.3, 0.05, 5, amp=0.12)),
                   (1.504, _noise(0.3, 0.05, 6, amp=0.12)),
                   (0.6, _sine(0.4, 90.0, 0.12, amp=0.9))])    # one real (low) tom
    res = _run({"kick": _write(tmp_path / "kick.wav", kick),
                "toms": _write(tmp_path / "toms.wav", toms)}, tmp_path)
    mid = pretty_midi.PrettyMIDI(res["file"])
    times = sorted((round(n.start, 4), n.pitch) for inst in mid.instruments for n in inst.notes)
    coincident = sum(1 for a, b in zip(times, times[1:])
                     if b[0] - a[0] <= 0.018 and a[1] != b[1])
    assert coincident == 0, f"cross-part bleed duplicates survived: {coincident}"
