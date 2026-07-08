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

from .io_utils import AudioTensor, save_audio


@dataclass
class BeatGrid:
    source_bpm: float
    beats: list[float] = field(default_factory=list)       # beat times, seconds
    downbeats: list[float] = field(default_factory=list)   # downbeat times, seconds
    engine: str = "librosa"
    beats_per_bar: int = 4

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
            "beats_per_bar": self.beats_per_bar,
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
    if engine == "beat_this":
        try:
            return _analyze_beat_this(audio, device, bpm_from_regression)
        except Exception:
            # fall through to librosa rather than fail the whole run
            engine = "librosa"
    return _analyze_librosa(audio, bpm_from_regression)


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
    return BeatGrid(source_bpm=bpm, beats=beats, downbeats=downbeats,
                    engine="beat_this", beats_per_bar=bpb)


# --------------------------------------------------------------------------- #
# librosa (fallback — always available)
# --------------------------------------------------------------------------- #
def _analyze_librosa(audio: AudioTensor, regression: bool) -> BeatGrid:
    import librosa

    y = audio.to_mono().samples[0]
    sr = audio.sample_rate
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    beats = [float(t) for t in beat_times]
    tempo_val = float(np.asarray(tempo).ravel()[0]) if np.size(tempo) else 120.0
    bpm = _bpm_from_beats(beats, regression) if len(beats) >= 3 else tempo_val
    downbeats = beats[::4]  # heuristic: assume 4/4, downbeat every 4 beats
    return BeatGrid(source_bpm=bpm, beats=beats, downbeats=downbeats,
                    engine="librosa", beats_per_bar=4)


# --------------------------------------------------------------------------- #
# BPM estimation
# --------------------------------------------------------------------------- #
def _bpm_from_beats(beats: list[float], regression: bool) -> float:
    """Linear regression of beat index -> time gives seconds/beat robustly.

    (Averaging raw inter-beat intervals is jitter-sensitive; the slope of a line
    fit through (index, time) is not — see madmom issue #416.)
    """
    if len(beats) < 3:
        return 120.0
    arr = np.asarray(beats, dtype=np.float64)
    if regression:
        idx = np.arange(len(arr))
        slope, _ = np.polyfit(idx, arr, 1)  # seconds per beat
        if slope <= 0:
            return 120.0
        return 60.0 / slope
    intervals = np.diff(arr)
    med = float(np.median(intervals))
    return 60.0 / med if med > 0 else 120.0


def _beats_per_bar(beats: list[float], downbeats: list[float]) -> int:
    """Estimate meter from spacing of downbeats within the beat sequence."""
    if len(downbeats) < 2 or len(beats) < 2:
        return 4
    spb = 60.0  # placeholder, use beat spacing
    b = np.asarray(beats)
    d = np.asarray(downbeats)
    # median beats between consecutive downbeats
    counts = []
    for i in range(len(d) - 1):
        counts.append(int(np.sum((b >= d[i]) & (b < d[i + 1]))))
    counts = [c for c in counts if c > 0]
    _ = spb
    return int(np.median(counts)) if counts else 4
