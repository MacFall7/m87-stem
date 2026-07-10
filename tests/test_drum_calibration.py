"""Drum calibration — A-series PR-A4.

Receipts (verified at main): drum_midi scaled every hit against ONE global peak
(no per-part normalization), had NO cross-part de-dup (separation bleed produced
phantom coincident notes), voiced all toms as one GM note, hard-coded the hi-hat
decay test, and recorded no per-event strength. This pins the fixes: per-part
velocity normalization, cross-part bleed de-dup (keep-most-probable), toms split
by pitch, config-driven thresholds (validated), and a per-event manifest.

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
from stemforge.orchestrator import DrumMidiCfg, load_config   # noqa: E402

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


def _cfg(**over):
    cfg = load_config().drums.midi
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def _run(parts, tmp_path, **cfg_over):
    return drum_midi._from_parts(parts, _cfg(**cfg_over), ensure_dir(tmp_path / "out"), bpm=120.0)


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


# --------------------------------------------------------------------------- #
# Per-part velocity normalization + ghost survival
# --------------------------------------------------------------------------- #
def test_velocity_normalized_per_part_and_ghosts_survive(tmp_path):
    # a quiet part (hi-hat-ish) and a loud part (kick): both reach a strong velocity
    # for their own loudest hit, and a quiet ghost stays clearly audible.
    kick = _place([(0.3, _noise(0.3, 0.05, 1, amp=1.0)), (0.9, _noise(0.3, 0.05, 2, amp=1.0))])
    snare = _place([(0.6, _noise(0.3, 0.05, 3, amp=0.15)),   # whole part is quiet
                    (1.2, _noise(0.3, 0.05, 4, amp=0.15)),
                    (1.5, _noise(0.3, 0.05, 5, amp=0.04))])  # a ghost within the quiet part
    res = _run({"kick": _write(tmp_path / "k.wav", kick),
                "snare": _write(tmp_path / "s.wav", snare)}, tmp_path)
    by_part: dict[str, list[int]] = {}
    for e in res["events"]:
        by_part.setdefault(e["part"], []).append(e["velocity"])
    # per-part normalization: the quiet snare part still reaches a strong velocity
    assert max(by_part["snare"]) >= 90
    assert max(by_part["kick"]) >= 90
    # every hit (incl. the ghost) stays audible (velocity floor)
    assert all(v >= 10 for vs in by_part.values() for v in vs)
    assert min(by_part["snare"]) >= 10


# --------------------------------------------------------------------------- #
# Toms split by pitch
# --------------------------------------------------------------------------- #
def test_toms_split_low_mid_high_by_pitch(tmp_path):
    toms = _place([(0.2, _sine(0.4, 80.0, 0.12)),     # low
                   (0.8, _sine(0.4, 150.0, 0.12)),    # mid
                   (1.4, _sine(0.4, 300.0, 0.12))])   # high
    res = _run({"toms": _write(tmp_path / "toms.wav", toms)}, tmp_path)
    pitches = {n.pitch for inst in pretty_midi.PrettyMIDI(res["file"]).instruments for n in inst.notes}
    assert drum_midi.GM["toms_low"] in pitches
    assert drum_midi.GM["toms_high"] in pitches
    # tom_split off -> everything collapses to the single mid-tom note
    res2 = _run({"toms": _write(tmp_path / "t2.wav", toms)}, tmp_path, tom_split=False)
    p2 = {n.pitch for inst in pretty_midi.PrettyMIDI(res2["file"]).instruments for n in inst.notes}
    assert p2 == {drum_midi.GM["toms"]}


def test_comparable_simultaneous_hits_survive(tmp_path):
    # a real, comparable kick+snare on the same beat is NOT bleed — both survive
    kick = _place([(0.5, _noise(0.3, 0.05, 1, amp=1.0))])
    snare = _place([(0.502, _noise(0.3, 0.05, 2, amp=0.9))])
    res = _run({"kick": _write(tmp_path / "k.wav", kick),
                "snare": _write(tmp_path / "s.wav", snare)}, tmp_path)
    pitches = [n.pitch for inst in pretty_midi.PrettyMIDI(res["file"]).instruments for n in inst.notes]
    assert drum_midi.GM["kick"] in pitches and drum_midi.GM["snare"] in pitches
    assert res["duplicates_removed"] == 0


# --------------------------------------------------------------------------- #
# Per-event manifest + config validation
# --------------------------------------------------------------------------- #
def test_manifest_records_per_event_strengths(tmp_path):
    kick = _place([(0.3, _noise(0.3, 0.05, 1)), (0.9, _noise(0.3, 0.05, 2))])
    res = _run({"kick": _write(tmp_path / "k.wav", kick)}, tmp_path)
    assert "events" in res and res["events"]
    ev = res["events"][0]
    for key in ("time", "part", "note", "raw_strength", "normalized_strength", "velocity"):
        assert key in ev
    assert 1 <= ev["velocity"] <= 127
    assert 0.0 <= ev["normalized_strength"] <= 1.0
    assert "duplicates_removed" in res


def test_config_validation_rejects_bad_values():
    assert DrumMidiCfg().validate() is not None                   # defaults are valid
    with pytest.raises(ValueError):
        DrumMidiCfg(velocity_percentile=0).validate()
    with pytest.raises(ValueError):
        DrumMidiCfg(dedupe_weak_ratio=2.0).validate()
    with pytest.raises(ValueError):
        DrumMidiCfg(cymbal_ambiguous="nonsense").validate()
    with pytest.raises(ValueError):
        DrumMidiCfg(tom_low_hz=500, tom_mid_hz=100).validate()
    # load_config runs validation -> a bad override is rejected at load time
    with pytest.raises(ValueError):
        load_config(overrides={"drums.midi.cymbal_ambiguous": "bogus"})
