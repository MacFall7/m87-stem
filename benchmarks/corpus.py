"""Deterministic synthetic corpus with a known ground truth.

Every fixture is rendered from parameters the metrics can check against — beat
grids at a known BPM, drum grooves with a known hit list per part, and a bleed
variant with deliberate cross-part phantoms. No external audio, no GPU, no
network. Determinism (seeded ``default_rng``) makes the fixture SHA-256 stable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from stemforge.io_utils import sha256_bytes

SR = 22050


# --------------------------------------------------------------------------- #
# Renderers
# --------------------------------------------------------------------------- #
def _noise_hit(n: int, seed: int, amp: float, tau: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    return (amp * rng.standard_normal(n) * np.exp(-t / tau)).astype(np.float32)


def _tone_hit(n: int, hz: float, amp: float, tau: float) -> np.ndarray:
    t = np.arange(n) / SR
    return (amp * np.sin(2 * np.pi * hz * t) * np.exp(-t / tau)).astype(np.float32)


def _render(hits: list[tuple[float, np.ndarray]], dur: float) -> np.ndarray:
    y = np.zeros(int(dur * SR), dtype=np.float32)
    for start, sig in hits:
        i = int(start * SR)
        end = min(y.size, i + sig.size)
        y[i:end] += sig[: end - i]
    return y


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@dataclass
class TempoFixture:
    name: str
    true_bpm: float
    audio: np.ndarray
    sr: int = SR


@dataclass
class DrumFixture:
    name: str
    parts: dict[str, np.ndarray]                 # part -> mono audio
    truth: dict[str, list[float]]                # part -> hit times (seconds)
    amps: dict[str, list[float]]                 # part -> render amplitude per hit (for velocity corr)
    bleed_pairs: list[tuple[str, float]] = field(default_factory=list)  # (part, time) phantom hits
    sr: int = SR


def tempo_fixtures() -> list[TempoFixture]:
    """Constant-tempo noise-pulse trains at 90 / 120 / 140 BPM."""
    out: list[TempoFixture] = []
    for k, bpm in enumerate((90.0, 120.0, 140.0)):
        dur = 6.0
        period = 60.0 / bpm
        n = int(0.04 * SR)
        hits = [(t, _noise_hit(n, seed=100 + k, amp=0.9, tau=0.02))
                for t in np.arange(0.2, dur, period)]
        out.append(TempoFixture(name=f"tempo_{int(bpm)}", true_bpm=bpm, audio=_render(hits, dur)))
    return out


def drum_fixtures() -> list[DrumFixture]:
    """A backbeat groove with per-part ground truth, plus a bleed variant."""
    dur = 4.0
    period = 0.5  # 120 BPM quarter notes
    beats = list(np.arange(0.2, dur, period))
    n = int(0.28 * SR)

    kick_times = [b for i, b in enumerate(beats) if i % 2 == 0]
    snare_times = [b for i, b in enumerate(beats) if i % 2 == 1]
    tom_times = [3.45, 3.6]                             # a short low-tom fill, off the kick/snare grid
    ghost_times = [t + 0.25 for t in snare_times[:-1]]  # quiet ghost snares off the beat

    kick = _render([(t, _noise_hit(n, 10 + i, amp=1.0, tau=0.05)) for i, t in enumerate(kick_times)], dur)
    snare_hits = [(t, _noise_hit(n, 20 + i, amp=0.9, tau=0.05)) for i, t in enumerate(snare_times)]
    ghost_hits = [(t, _noise_hit(n, 40 + i, amp=0.12, tau=0.04)) for i, t in enumerate(ghost_times)]
    snare = _render(snare_hits + ghost_hits, dur)
    toms = _render([(t, _tone_hit(n, 90.0, amp=0.9, tau=0.12)) for t in tom_times], dur)

    clean = DrumFixture(
        name="groove_clean",
        parts={"kick": kick, "snare": snare, "toms": toms},
        truth={"kick": kick_times, "snare": snare_times + ghost_times, "toms": tom_times},
        amps={"kick": [1.0] * len(kick_times),
              "snare": [0.9] * len(snare_times) + [0.12] * len(ghost_times),
              "toms": [0.9] * len(tom_times)},
    )

    # bleed variant: weak phantom kick coincident in the toms part (separation spill)
    phantom_times = kick_times
    toms_bleed = _render(
        [(t, _tone_hit(n, 90.0, amp=0.9, tau=0.12)) for t in tom_times]
        + [(t + 0.004, _noise_hit(n, 60 + i, amp=0.12, tau=0.05)) for i, t in enumerate(phantom_times)],
        dur)
    bleed = DrumFixture(
        name="groove_bleed",
        parts={"kick": kick, "snare": snare, "toms": toms_bleed},
        truth={"kick": kick_times, "snare": snare_times + ghost_times, "toms": tom_times},
        amps={"kick": [1.0] * len(kick_times),
              "snare": [0.9] * len(snare_times) + [0.12] * len(ghost_times),
              "toms": [0.9] * len(tom_times)},
        bleed_pairs=[("toms", t + 0.004) for t in phantom_times],
    )
    return [clean, bleed]


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def fixture_sha256() -> str:
    """A stable SHA-256 over the whole rendered corpus (provenance for reports)."""
    h_parts: list[bytes] = []
    for tf in tempo_fixtures():
        h_parts.append(tf.audio.tobytes())
    for df in drum_fixtures():
        for name in sorted(df.parts):
            h_parts.append(df.parts[name].tobytes())
    return sha256_bytes(b"".join(h_parts))


def write_corpus(dest: Path) -> dict[str, Any]:
    """Materialize the corpus as .wav files under ``dest`` (for inspection/CI)."""
    dest.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"tempo": {}, "drums": {}, "sha256": fixture_sha256()}
    for tf in tempo_fixtures():
        p = dest / f"{tf.name}.wav"
        sf.write(str(p), tf.audio, tf.sr, subtype="FLOAT")
        manifest["tempo"][tf.name] = {"path": str(p), "true_bpm": tf.true_bpm}
    for df in drum_fixtures():
        for part, y in df.parts.items():
            sf.write(str(dest / f"{df.name}_{part}.wav"), y, df.sr, subtype="FLOAT")
        manifest["drums"][df.name] = {"truth": df.truth}
    return manifest
