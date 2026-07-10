"""Tempo/beat evidence — A-series PR-A2.

Receipts (verified at main): analysis.py silently returned 120 BPM on several
failure paths (`:96,:113,:119,:123`), `:98` assumed 4/4 (`downbeats=beats[::4]`),
and BeatGrid carried no confidence. drum_midi baked 120 into the .mid when the
grid was absent. stretch swallowed fallback reasons at both hops. This pins the
fixes: unknown tempo is 0.0 (never a silent 120) with confidence/candidates,
the librosa meter is flagged assumed, and both analysis and stretch record a
shared, sanitized EngineAttempt chain.

GPU-free and deterministic — analyze is driven to librosa either explicitly
(engine="librosa") or by forcing the beat_this hop to fail, so the tests do not
depend on whether beat_this happens to be installed.
"""

from __future__ import annotations

import inspect
import types
from pathlib import Path

import numpy as np
import soundfile as sf

from stemforge import analysis, drum_midi
from stemforge.analysis import (
    BeatGrid,
    _bpm_candidates,
    _bpm_confidence,
    _bpm_from_beats,
    _meter_confidence,
    analyze,
)
from stemforge.io_utils import AudioTensor, EngineAttempt, ensure_dir, sanitize_reason
from stemforge.orchestrator import load_config
from stemforge.stretch import match_bpm_file

SR = 44100


def _burst_track(times: list[float], dur: float = 2.5, tau: float = 0.05, seed: int = 0) -> np.ndarray:
    """Broadband decaying-noise hits (librosa onset_detect finds these)."""
    rng = np.random.default_rng(seed)
    y = np.zeros(int(dur * SR), dtype=np.float32)
    for t0 in times:
        i = int(t0 * SR)
        n = int(0.3 * SR)
        tt = np.arange(n) / SR
        seg = (rng.standard_normal(n) * np.exp(-tt / tau)).astype(np.float32)
        end = min(y.size, i + n)
        y[i:end] += seg[: end - i]
    return y


def _write(path: Path, y: np.ndarray) -> str:
    sf.write(str(path), y, SR, subtype="FLOAT")
    return str(path)


def _two_clicks() -> AudioTensor:
    y = np.zeros(SR, dtype=np.float32)
    y[100:130] = 0.8
    y[SR // 2:SR // 2 + 30] = 0.8
    return AudioTensor(y, SR)


# --------------------------------------------------------------------------- #
# commit 1 — RED: an un-estimable tempo is UNKNOWN (0.0), never a silent 120
# --------------------------------------------------------------------------- #
def test_bpm_unknown_is_zero_not_silent_120():
    assert _bpm_from_beats([1.0, 2.0], regression=True) == 0.0          # <3 beats
    assert _bpm_from_beats([3.0, 2.0, 1.0], regression=True) == 0.0     # non-positive slope
    assert _bpm_from_beats([1.0, 1.0, 1.0], regression=False) == 0.0    # non-positive median


# --------------------------------------------------------------------------- #
# commit 3 — evidence, candidates, meter confidence, chains, drum/stretch wiring
# --------------------------------------------------------------------------- #
def test_bpm_known_sequence_and_confidence():
    beats = [i * 0.5 for i in range(9)]  # exactly 120 BPM
    assert abs(_bpm_from_beats(beats, regression=True) - 120.0) < 0.5   # contract preserved
    assert _bpm_confidence(beats, regression=True) > 0.99               # near-perfect line
    assert _bpm_confidence([1.0, 2.0], regression=True) == 0.0          # unknown -> 0 conf


def test_bpm_candidates_include_half_and_double():
    assert _bpm_candidates(120.0) == [60.0, 120.0, 240.0]
    assert _bpm_candidates(0.0) == []                                    # unknown -> no candidates


def test_meter_confidence_consistent_vs_irregular():
    beats = [i * 0.5 for i in range(17)]
    downbeats = beats[::4]                     # a clean 4/4 grid
    assert _meter_confidence(beats, downbeats) > 0.9
    irregular = [beats[0], beats[3], beats[9], beats[11]]  # ragged bar spacing
    assert _meter_confidence(beats, irregular) < _meter_confidence(beats, downbeats)


def test_beatgrid_to_dict_carries_evidence():
    g = BeatGrid(source_bpm=120.0, beats=[0.0, 0.5, 1.0, 1.5], downbeats=[0.0, 1.0],
                 bpm_confidence=0.9, bpm_candidates=[60.0, 120.0, 240.0],
                 meter_confidence=0.8, tempo_assumed=False,
                 fallback_chain=[{"engine": "beat_this", "status": "used", "reason": ""}])
    d = g.to_dict()
    assert d["source_bpm"] == 120.0
    for k in ("bpm_confidence", "bpm_candidates", "meter_confidence", "tempo_assumed",
              "fallback_chain"):
        assert k in d
    assert d["tempo_assumed"] is False
    assert d["bpm_candidates"] == [60.0, 120.0, 240.0]


def test_analyze_records_fallback_chain(monkeypatch):
    """A beat_this failure -> analyze records a fell_through hop with a sanitized
    reason, then librosa used. Force the failure so the test is isolated from the
    environment (beat_this may or may not be installed — it is on Windows)."""
    def _boom(*a, **k):
        raise ImportError("No module named 'beat_this'")

    monkeypatch.setattr(analysis, "_analyze_beat_this", _boom)
    g = analyze(_two_clicks())  # default engine="beat_this" -> forced fallback to librosa
    assert g.engine == "librosa"
    chain = g.fallback_chain
    assert [c["engine"] for c in chain] == ["beat_this", "librosa"]
    assert chain[0]["status"] == "fell_through" and chain[0]["reason"]
    assert chain[1]["status"] == "used"
    assert "/" not in chain[0]["reason"] or "<path>" in chain[0]["reason"]


def test_analyze_unknown_source_bpm_is_flagged():
    g = analyze(_two_clicks(), engine="librosa")
    assert g.source_bpm == 0.0            # unknown, not 120
    assert g.tempo_assumed is True
    assert g.bpm_confidence == 0.0
    assert g.bpm_candidates == []


def test_librosa_path_marks_meter_assumed():
    g = analyze(_two_clicks(), engine="librosa")
    assert g.tempo_assumed is True         # librosa downbeats are a 4/4 assumption
    assert g.meter_confidence <= 0.2


def test_engine_attempt_and_sanitize_reason():
    assert EngineAttempt("rubberband", "fell_through", "boom").to_dict() == {
        "engine": "rubberband", "status": "fell_through", "reason": "boom"}
    r = sanitize_reason("failed at /usr/local/bin/rubberband\nstack line 2\nline 3")
    assert "/usr/local" not in r and "<path>" in r
    assert "\n" not in r                                    # subprocess dump collapsed
    assert sanitize_reason(RuntimeError("x")) == "RuntimeError: x"   # deterministic


def test_match_bpm_file_manifest_carries_attempts(tmp_path, sine):
    samples, sr = sine
    wav = _write(tmp_path / "loop.wav", samples.T if samples.ndim == 2 else samples)
    res = match_bpm_file(Path(wav), target_bpm=140, out_dir=tmp_path / "m",
                         source_bpm=120, engine="librosa")
    assert "attempts" in res and isinstance(res["attempts"], list)
    assert res["attempts"], "stretch manifest must carry a non-empty attempts[]"
    last = res["attempts"][-1]
    assert last["engine"] == "librosa" and last["status"] == "used"


def test_stretch_records_fallthrough_reasons(monkeypatch, sine):
    from stemforge import stretch as st

    def raise_rb(*a, **k):
        raise RuntimeError("rubberband missing at /usr/bin/rubberband")

    def raise_ss(*a, **k):
        raise RuntimeError("signalsmith boom")

    monkeypatch.setattr(st, "_rubberband", raise_rb)
    monkeypatch.setattr(st, "_signalsmith", raise_ss)
    attempts: list[dict] = []
    _out, used = st.stretch_with_engine(AudioTensor(*sine), 1.5, engine="rubberband",
                                        attempts=attempts)
    assert used == "librosa"
    kinds = {(a["engine"], a["status"]) for a in attempts}
    assert ("rubberband", "fell_through") in kinds
    assert ("signalsmith", "fell_through") in kinds
    assert ("librosa", "used") in kinds
    rb = next(a for a in attempts if a["engine"] == "rubberband")
    assert "<path>" in rb["reason"] and "/usr/bin" not in rb["reason"]


def _drum_parts(tmp_path) -> dict[str, str]:
    return {"kick": _write(tmp_path / "kick.wav", _burst_track([0.1, 0.6, 1.1, 1.6]))}


def test_drum_midi_flags_serialization_default_tempo(tmp_path):
    res = drum_midi.transcribe(None, _drum_parts(tmp_path), load_config().drums.midi,
                               ensure_dir(tmp_path / "out"), beat_grid=None)
    assert "file" in res
    assert res["tempo_source"] == "midi_serialization_default"
    assert res["tempo_assumed"] is True
    assert res["tempo_bpm"] == 120.0


def test_drum_midi_uses_measured_tempo(tmp_path):
    grid = types.SimpleNamespace(source_bpm=128.0)
    res = drum_midi.transcribe(None, _drum_parts(tmp_path), load_config().drums.midi,
                               ensure_dir(tmp_path / "out"), beat_grid=grid)
    assert res["tempo_source"] == "beat_grid"
    assert res["tempo_assumed"] is False
    assert res["tempo_bpm"] == 128.0


def test_analysis_source_has_no_silent_120_fallback():
    """Guard against reintroducing a silent 120: the BPM helpers must not
    manufacture a tempo on failure."""
    src = inspect.getsource(analysis)
    assert "return 120" not in src
    assert "else 120" not in src
