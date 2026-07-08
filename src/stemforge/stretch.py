"""§5.7 stretch — pitch-preserving time-stretch to a target BPM.

Engine chain: Rubber Band (highest quality, needs the system ``rubberband`` CLI)
-> signalsmith (pip ``python-stretch``) -> librosa phase-vocoder (always present).
The chain guarantees stretch works in any environment; quality degrades gracefully.

``ratio = target_bpm / source_bpm`` — ratio > 1 speeds up (shorter), < 1 slows down.
Batch stretches every stem with the same ratio so the set stays grid-aligned;
per-stem target overrides are supported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import AudioTensor, PathLike, load_audio, save_audio

try:  # pragma: no cover
    from .orchestrator import StretchCfg
except Exception:  # pragma: no cover
    StretchCfg = Any  # type: ignore


def time_stretch(
    audio: AudioTensor,
    ratio: float,
    engine: str = "rubberband",
    crisp: int = 5,
    preserve_formant: bool = False,
) -> AudioTensor:
    if abs(ratio - 1.0) < 1e-6:
        return audio
    if engine == "rubberband":
        try:
            return _rubberband(audio, ratio, crisp, preserve_formant)
        except Exception:
            engine = "signalsmith"
    if engine == "signalsmith":
        try:
            return _signalsmith(audio, ratio)
        except Exception:
            pass
    return _librosa(audio, ratio)


# --------------------------------------------------------------------------- #
# Engines
# --------------------------------------------------------------------------- #
def _rubberband(audio: AudioTensor, ratio: float, crisp: int, preserve_formant: bool) -> AudioTensor:
    import pyrubberband as pyrb

    y = audio.samples.T  # pyrb wants (samples, channels)
    rbargs = {"--crisp": str(int(crisp))}
    if preserve_formant:
        # Formant preservation is only meaningful alongside pitch shift; for a
        # pure tempo change pitch is already preserved. Kept for the future
        # pitch-shift path; '-F' takes no value so we pass it with a dummy the
        # CLI ignores rather than an empty argv element.
        rbargs["--formant"] = "1"
    out = pyrb.time_stretch(y, audio.sample_rate, ratio, rbargs=rbargs)
    return AudioTensor(np.ascontiguousarray(out.T, dtype=np.float32), audio.sample_rate)


def _signalsmith(audio: AudioTensor, ratio: float) -> AudioTensor:
    import stretch  # python-stretch

    processor = stretch.Signalsmith(audio.num_channels, audio.sample_rate)
    processor.setTimeFactor(1.0 / ratio)  # signalsmith: factor > 1 => longer
    out = processor.process(audio.samples.astype(np.float32))
    return AudioTensor(np.ascontiguousarray(out, dtype=np.float32), audio.sample_rate)


def _librosa(audio: AudioTensor, ratio: float) -> AudioTensor:
    import librosa

    chans = [librosa.effects.time_stretch(audio.samples[c], rate=ratio)
             for c in range(audio.num_channels)]
    n = min(len(c) for c in chans)
    out = np.stack([c[:n] for c in chans], axis=0)
    return AudioTensor(out.astype(np.float32), audio.sample_rate)


# --------------------------------------------------------------------------- #
# Batch over stems
# --------------------------------------------------------------------------- #
def stretch_stems(
    stems: dict[str, Path],
    cfg: "StretchCfg",
    out_dir: PathLike,
    beat_grid: Any = None,
) -> dict[str, Any]:
    if not stems:
        return {"skipped": "no stems to stretch (run separation first)"}
    source_bpm = getattr(beat_grid, "source_bpm", None)
    if not source_bpm or source_bpm <= 0:
        return {"skipped": "no source BPM (enable analysis)"}

    out_dir = Path(out_dir)
    result: dict[str, Any] = {
        "engine": cfg.engine,
        "source_bpm": round(float(source_bpm), 3),
        "target_bpm": cfg.target_bpm,
        "ratios": {},
        "files": {},
    }
    for name, path in stems.items():
        target = cfg.per_stem_target_bpm.get(name, cfg.target_bpm)
        if not target:
            continue
        ratio = float(target) / float(source_bpm)
        audio = load_audio(path)
        formant = cfg.preserve_formant and name == "vocals"
        out = time_stretch(audio, ratio, engine=cfg.engine, crisp=cfg.crisp,
                           preserve_formant=formant)
        outp = save_audio(out_dir / f"{name}_{int(round(float(target)))}bpm.wav", out)
        result["ratios"][name] = round(ratio, 4)
        result["files"][name] = str(outp)
    return result
