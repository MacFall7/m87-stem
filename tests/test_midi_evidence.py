"""Basic Pitch transformation ledger — A-series PR-A3.

Receipts (verified at main): midi_melodic discarded Basic Pitch confidence
(`:73`), overwrote the raw output (only the final .mid was kept), and applied the
monophonic + quantize transforms with no record of what changed. This pins the
fix: raw is preserved untouched, and a per-note ledger (id, confidence, status,
quantize delta) reconstructs cleaned/quantized from raw EXACTLY.

GPU-free and deterministic — hand-built pretty_midi objects, Basic Pitch mocked.
"""

from __future__ import annotations

import json
import types

import pytest

pretty_midi = pytest.importorskip("pretty_midi")

from stemforge import midi_melodic as mm            # noqa: E402
from stemforge.orchestrator import load_config      # noqa: E402


def _raw_pm(notes: list[tuple[int, float, float, int]]):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for pitch, start, end, vel in notes:
        inst.notes.append(pretty_midi.Note(velocity=vel, pitch=pitch, start=start, end=end))
    pm.instruments.append(inst)
    return pm


def _notes(pm, prec: int = 2) -> list[tuple[int, float, float]]:
    return sorted((n.pitch, round(n.start, prec), round(n.end, prec))
                  for inst in pm.instruments for n in inst.notes)


def _pitches(pm) -> set[int]:
    return {n.pitch for inst in pm.instruments for n in inst.notes}


def _grid(beats):
    return types.SimpleNamespace(beats=beats)


def _midi_cfg(**over):
    cfg = load_config().midi
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


# --------------------------------------------------------------------------- #
# Event model + transforms
# --------------------------------------------------------------------------- #
def test_build_events_assigns_stable_ids_and_retains_confidence():
    pm = _raw_pm([(60, 0.0, 0.5, 127), (64, 0.5, 1.0, 64)])
    events = mm.build_events(pm)
    assert [e.id for e in events] == ["e0000", "e0001"]
    assert events[0].confidence == 1.0                       # velocity 127 -> full confidence
    assert abs(events[1].confidence - 64 / 127) < 1e-6       # retained, not discarded
    assert all(e.status == "retained" and e.quantize_delta == 0.0 for e in events)


def test_monophonic_marks_dropped_with_reason():
    # two overlapping notes: the quieter is dropped and the drop is logged
    events = mm.build_events(_raw_pm([(60, 0.0, 1.0, 120), (67, 0.2, 0.8, 50)]))
    mm.mark_monophonic(events)
    by_pitch = {e.pitch: e for e in events}
    assert by_pitch[60].status == "retained"
    assert by_pitch[67].status == "dropped" and by_pitch[67].drop_reason
    # non-overlapping notes are all kept
    ev2 = mm.build_events(_raw_pm([(60, 0.0, 0.4, 100), (62, 0.5, 0.9, 100)]))
    mm.mark_monophonic(ev2)
    assert all(e.status == "retained" for e in ev2)


def test_quantize_records_deltas_and_max_displacement():
    events = mm.build_events(_raw_pm([(60, 0.03, 0.5, 100), (62, 1.03, 1.4, 100)]))
    meta = {"applied": False, "subdivisions": 4, "max_displacement_s": 0.0}
    mm.mark_quantize(events, _grid([0.0, 0.5, 1.0, 1.5, 2.0]), meta)
    assert meta["applied"] is True
    assert meta["max_displacement_s"] >= 0.0
    assert any(abs(e.quantize_delta) > 0 for e in events)     # near-grid notes get a delta


# --------------------------------------------------------------------------- #
# Acceptance — the ledger reconstructs cleaned/quantized from raw
# --------------------------------------------------------------------------- #
def test_transform_is_concrete_and_correct():
    raw = _raw_pm([(60, 0.0, 1.0, 120), (67, 0.2, 0.8, 50), (62, 1.03, 1.4, 100)])
    cfg = _midi_cfg(quantize_to_grid=True)
    ledger, cleaned, quantized = mm._transcribe_one(
        raw, None, cfg, monophonic=True, beat_grid=_grid([0.0, 0.5, 1.0, 1.5, 2.0]))
    # the overlapping quieter note (67) is dropped; the non-overlapping 62 survives
    assert _pitches(cleaned) == {60, 62}
    assert ledger["counts"] == {"raw": 3, "retained": 2, "dropped": 1}
    # 62 started at 1.03 -> snapped toward 1.0
    q62 = next(n for inst in quantized.instruments for n in inst.notes if n.pitch == 62)
    assert abs(q62.start - 1.0) < 0.13


def test_reconstruct_matches_generated_output():
    raw = _raw_pm([(60, 0.0, 1.0, 120), (67, 0.2, 0.8, 50), (62, 1.03, 1.4, 100)])
    cfg = _midi_cfg(quantize_to_grid=True)
    ledger, cleaned, quantized = mm._transcribe_one(
        raw, None, cfg, monophonic=True, beat_grid=_grid([0.0, 0.5, 1.0, 1.5, 2.0]))
    rc, rq = mm.reconstruct(raw, ledger)      # from raw + ledger alone
    assert _notes(rc) == _notes(cleaned)
    assert _notes(rq) == _notes(quantized)


def test_acceptance_disk_roundtrip_reconstructs_exactly(tmp_path, monkeypatch):
    """The persisted ledger + raw.mid rebuild the persisted cleaned/quantized."""
    raw = _raw_pm([(60, 0.0, 1.0, 120), (67, 0.2, 0.8, 50), (62, 1.03, 1.4, 100)])
    monkeypatch.setattr(mm, "_predict_raw", lambda path, cfg: (raw, None))
    cfg = _midi_cfg(stems=["bass"], monophonic_stems=["bass"], quantize_to_grid=True)
    grid = _grid([0.0, 0.5, 1.0, 1.5, 2.0])
    mm.transcribe_stems({"bass": tmp_path / "bass.wav"}, cfg, tmp_path / "midi", beat_grid=grid)

    d = tmp_path / "midi" / "bass"
    raw_loaded = pretty_midi.PrettyMIDI(str(d / "raw.mid"))
    ledger = json.loads((d / "events.json").read_text())
    rc, rq = mm.reconstruct(raw_loaded, ledger)
    assert _notes(rc) == _notes(pretty_midi.PrettyMIDI(str(d / "cleaned.mid")))
    assert _notes(rq) == _notes(pretty_midi.PrettyMIDI(str(d / "quantized.mid")))


# --------------------------------------------------------------------------- #
# transcribe_stems writes evidence; raw is never overwritten
# --------------------------------------------------------------------------- #
def test_transcribe_stems_writes_evidence_artifacts(tmp_path, monkeypatch):
    raw = _raw_pm([(60, 0.0, 1.0, 120), (67, 0.2, 0.8, 50)])
    monkeypatch.setattr(mm, "_predict_raw", lambda path, cfg: (raw, None))
    cfg = _midi_cfg(stems=["bass"], monophonic_stems=["bass"], quantize_to_grid=False)
    res = mm.transcribe_stems({"bass": tmp_path / "bass.wav"}, cfg, tmp_path / "midi",
                              beat_grid=None)
    d = tmp_path / "midi" / "bass"
    assert (d / "raw.mid").is_file() and (d / "cleaned.mid").is_file()
    assert (d / "events.json").is_file()
    assert not (d / "quantized.mid").exists()                # quantize off -> no artifact
    assert res["files"]["bass"].endswith("cleaned.mid")
    ev = res["evidence"]["bass"]
    assert ev["retained"] == 1 and ev["dropped"] == 1
    assert ev["quantized"] is None


def test_raw_is_never_overwritten(tmp_path, monkeypatch):
    raw = _raw_pm([(60, 0.0, 1.0, 120), (67, 0.2, 0.8, 50)])
    monkeypatch.setattr(mm, "_predict_raw", lambda path, cfg: (raw, None))
    cfg = _midi_cfg(stems=["bass"], monophonic_stems=["bass"], quantize_to_grid=False)
    mm.transcribe_stems({"bass": tmp_path / "bass.wav"}, cfg, tmp_path / "midi")
    d = tmp_path / "midi" / "bass"
    raw_loaded = pretty_midi.PrettyMIDI(str(d / "raw.mid"))
    cleaned_loaded = pretty_midi.PrettyMIDI(str(d / "cleaned.mid"))
    assert sum(len(i.notes) for i in raw_loaded.instruments) == 2       # both raw notes intact
    assert sum(len(i.notes) for i in cleaned_loaded.instruments) == 1   # one dropped in cleaning
