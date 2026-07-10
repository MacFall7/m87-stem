"""§5.2 analysis — tempo / beat / downbeat, run ONCE and shared by every stage.

Primary engine: ``beat_this`` (PyTorch, no numpy pin). Fallback: ``librosa`` so
analysis always works even without the GPU model installed. Source BPM is derived
by linear regression over beat times (robust to jitter).
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import AudioTensor, EngineAttempt, sanitize_reason, save_audio


@dataclass
class BeatGrid:
    source_bpm: float                                      # 0.0 == UNKNOWN (never a silent 120)
    beats: list[float] = field(default_factory=list)       # beat times, seconds
    downbeats: list[float] = field(default_factory=list)   # downbeat times, seconds
    engine: str = "librosa"
    beats_per_bar: int = 4
    # Tempo/meter evidence — so downstream stages (and the manifest) can tell a
    # measured tempo from an assumed one instead of trusting a bare number.
    bpm_confidence: float = 0.0                            # 0..1 (0.0 when unknown)
    bpm_candidates: list[float] = field(default_factory=list)   # always incl. half/double
    meter_confidence: float = 0.0                          # 0..1 confidence in beats_per_bar
    tempo_assumed: bool = False                            # True => bpm/meter is a fallback assumption
    fallback_chain: list[dict] = field(default_factory=list)    # EngineAttempt dicts

    @property
    def seconds_per_beat(self) -> float:
        return 60.0 / self.source_bpm if self.source_bpm > 0 else 0.0

    def nearest_beat(self, t: float) -> float:
        if not self.beats:
            return t
        arr = np.asarray(self.beats)
        return float(arr[int(np.argmin(np.abs(arr - t)))])

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "source_bpm": round(self.source_bpm, 3),
            "bpm_confidence": round(self.bpm_confidence, 3),
            "bpm_candidates": [round(c, 3) for c in self.bpm_candidates],
            "tempo_assumed": self.tempo_assumed,
            "beats_per_bar": self.beats_per_bar,
            "meter_confidence": round(self.meter_confidence, 3),
            "fallback_chain": self.fallback_chain,
            "num_beats": len(self.beats),
            "num_downbeats": len(self.downbeats),
            "beats": [round(b, 4) for b in self.beats],
            "downbeats": [round(b, 4) for b in self.downbeats],
        }


def analyze(
    audio: AudioTensor,
    engine: str = "beat_this",
    device: str = "cuda",
    bpm_from_regression: bool = True,
) -> BeatGrid:
    """Analyze tempo/beats, recording the engine fallback chain as evidence.

    Never silently substitutes 120 BPM: when the tempo can't be measured the grid
    reports ``source_bpm == 0.0`` (unknown) with ``tempo_assumed`` set.
    """
    chain: list[dict] = []
    if engine == "beat_this":
        try:
            grid = _analyze_beat_this(audio, device, bpm_from_regression)
            chain.append(EngineAttempt("beat_this", "used").to_dict())
            grid.fallback_chain = chain
            return grid
        except Exception as e:  # noqa: BLE001 - fall through to librosa, but record why
            chain.append(EngineAttempt("beat_this", "fell_through", sanitize_reason(e)).to_dict())
            engine = "librosa"
    grid = _analyze_librosa(audio, bpm_from_regression)
    chain.append(EngineAttempt("librosa", "used").to_dict())
    grid.fallback_chain = chain
    return grid


# --------------------------------------------------------------------------- #
# beat_this (primary)
# --------------------------------------------------------------------------- #
def _analyze_beat_this(audio: AudioTensor, device: str, regression: bool) -> BeatGrid:
    from beat_this.inference import File2Beats  # type: ignore

    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "analysis.wav"
        save_audio(wav, audio)
        f2b = File2Beats(device=device)
        beats, downbeats = f2b(str(wav))

    beats = [float(b) for b in np.asarray(beats).ravel()]
    downbeats = [float(b) for b in np.asarray(downbeats).ravel()]
    bpm = _bpm_from_beats(beats, regression)
    bpb = _beats_per_bar(beats, downbeats)
    # beat_this returns real downbeats: the meter is measured (not assumed), and
    # tempo is assumed only when we couldn't derive a BPM at all.
    return BeatGrid(
        source_bpm=bpm, beats=beats, downbeats=downbeats,
        engine="beat_this", beats_per_bar=bpb,
        bpm_confidence=_bpm_confidence(beats, regression),
        bpm_candidates=_bpm_candidates(bpm),
        meter_confidence=_meter_confidence(beats, downbeats),
        tempo_assumed=(bpm <= 0.0),
    )


# --------------------------------------------------------------------------- #
# librosa (fallback — always available)
# --------------------------------------------------------------------------- #
def _analyze_librosa(audio: AudioTensor, regression: bool) -> BeatGrid:
    import librosa

    y = audio.to_mono().samples[0]
    sr = audio.sample_rate
    _tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    beats = [float(t) for t in beat_times]
    # UNKNOWN (0.0) — never a silent 120 — when the beats can't support a BPM.
    bpm = _bpm_from_beats(beats, regression)
    downbeats = beats[::4]  # heuristic: assume 4/4, downbeat every 4 beats
    # The librosa downbeats are a 4/4 ASSUMPTION, so the meter is always assumed
    # here (low meter confidence), independent of the BPM estimate quality.
    return BeatGrid(
        source_bpm=bpm, beats=beats, downbeats=downbeats,
        engine="librosa", beats_per_bar=4,
        bpm_confidence=_bpm_confidence(beats, regression),
        bpm_candidates=_bpm_candidates(bpm),
        meter_confidence=min(_meter_confidence(beats, downbeats), 0.2),
        tempo_assumed=True,
    )


# --------------------------------------------------------------------------- #
# BPM estimation + evidence
# --------------------------------------------------------------------------- #
def _bpm_from_beats(beats: list[float], regression: bool) -> float:
    """Linear regression of beat index -> time gives seconds/beat robustly.

    (Averaging raw inter-beat intervals is jitter-sensitive; the slope of a line
    fit through (index, time) is not — see madmom issue #416.)

    Returns ``0.0`` (UNKNOWN) — never a silent 120 — when there is too little
    evidence (<3 beats), a non-positive regression slope, or a non-positive median
    inter-beat interval.
    """
    if len(beats) < 3:
        return 0.0
    arr = np.asarray(beats, dtype=np.float64)
    if regression:
        idx = np.arange(len(arr))
        slope, _ = np.polyfit(idx, arr, 1)  # seconds per beat
        if slope <= 0:
            return 0.0
        return 60.0 / slope
    intervals = np.diff(arr)
    med = float(np.median(intervals))
    return 60.0 / med if med > 0 else 0.0


def _bpm_confidence(beats: list[float], regression: bool) -> float:
    """0..1 confidence in the BPM: R² of the index→time fit (regression) or
    ``1 - CV`` of the inter-beat intervals. 0.0 whenever the BPM is unknown."""
    if len(beats) < 3:
        return 0.0
    arr = np.asarray(beats, dtype=np.float64)
    if regression:
        idx = np.arange(len(arr))
        slope, intercept = np.polyfit(idx, arr, 1)
        if slope <= 0:
            return 0.0
        pred = slope * idx + intercept
        ss_res = float(np.sum((arr - pred) ** 2))
        ss_tot = float(np.sum((arr - arr.mean()) ** 2))
        return float(np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0)) if ss_tot > 0 else 0.0
    intervals = np.diff(arr)
    med = float(np.median(intervals))
    if med <= 0:
        return 0.0
    cv = float(np.std(intervals) / med)
    return float(np.clip(1.0 - cv, 0.0, 1.0))


def _bpm_candidates(bpm: float) -> list[float]:
    """Always surface the half/double alternates alongside the estimate so a
    half/double detection error is visible, not silently locked in. Empty when
    the BPM is unknown."""
    if bpm <= 0:
        return []
    return sorted({round(bpm, 3), round(bpm / 2, 3), round(bpm * 2, 3)})


def _beats_per_bar(beats: list[float], downbeats: list[float]) -> int:
    """Estimate meter from spacing of downbeats within the beat sequence."""
    if len(downbeats) < 2 or len(beats) < 2:
        return 4
    b = np.asarray(beats)
    d = np.asarray(downbeats)
    counts = [int(np.sum((b >= d[i]) & (b < d[i + 1]))) for i in range(len(d) - 1)]
    counts = [c for c in counts if c > 0]
    return int(np.median(counts)) if counts else 4


def _meter_confidence(beats: list[float], downbeats: list[float]) -> float:
    """0..1 confidence in the meter: how consistent the beats-per-bar counts are
    across the piece (all bars equal → 1.0). 0.0 when there is too little to tell."""
    if len(downbeats) < 3 or len(beats) < 2:
        return 0.0
    b = np.asarray(beats)
    d = np.asarray(downbeats)
    counts = [int(np.sum((b >= d[i]) & (b < d[i + 1]))) for i in range(len(d) - 1)]
    counts = [c for c in counts if c > 0]
    if len(counts) < 2:
        return 0.0
    arr = np.asarray(counts, dtype=np.float64)
    med = float(np.median(arr))
    if med <= 0:
        return 0.0
    frac_agree = float(np.mean(np.abs(arr - med) < 0.5))  # share of bars at the modal count
    return float(np.clip(frac_agree, 0.0, 1.0))
