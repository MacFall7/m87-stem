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

from .io_utils import AudioTensor, PathLike, load_audio, save_audio, slugify

try:  # pragma: no cover
    from .orchestrator import StretchCfg
except Exception:  # pragma: no cover
    StretchCfg = Any  # type: ignore


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def stretch_with_engine(
    audio: AudioTensor,
    ratio: float,
    engine: str = "rubberband",
    crisp: int = 5,
    preserve_formant: bool = False,
) -> tuple[AudioTensor, str]:
    """Stretch and report the engine ACTUALLY used (after any fallback).

    The chain degrades rubberband → signalsmith → librosa; the returned label is
    the engine that actually produced the audio — so callers record the truth,
    not the request (the old code fell through to librosa without ever updating
    the label). Identity ratio does no work and reports ``"none"``.
    """
    if abs(ratio - 1.0) < 1e-6:
        return audio, "none"
    if engine == "rubberband":
        try:
            return _rubberband(audio, ratio, crisp, preserve_formant), "rubberband"
        except Exception:
            engine = "signalsmith"
    if engine == "signalsmith":
        try:
            return _signalsmith(audio, ratio), "signalsmith"
        except Exception:
            pass
    return _librosa(audio, ratio), "librosa"


def time_stretch(
    audio: AudioTensor,
    ratio: float,
    engine: str = "rubberband",
    crisp: int = 5,
    preserve_formant: bool = False,
) -> AudioTensor:
    """Backwards-compatible wrapper returning only the audio (engine discarded)."""
    return stretch_with_engine(audio, ratio, engine, crisp, preserve_formant)[0]


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
# Whole-file "Match BPM" (no separation)
# --------------------------------------------------------------------------- #
def detect_bpm(audio: AudioTensor, engine: str = "beat_this", device: str = "auto") -> float:
    """Return the source BPM of ``audio`` (0.0 if none detectable).

    Runs the shared analysis engine (``beat_this`` → librosa fallback). Never
    raises — a detection failure returns 0.0 so callers can fail soft.
    """
    from . import analysis

    try:
        grid = analysis.analyze(audio, engine=engine, device=_resolve_device(device))
        bpm = float(getattr(grid, "source_bpm", 0.0) or 0.0)
        return bpm if bpm > 0 else 0.0
    except Exception:  # noqa: BLE001 - detection is best-effort
        return 0.0


def match_bpm_file(
    input_path: PathLike,
    target_bpm: float,
    out_dir: PathLike,
    source_bpm: float | None = None,
    engine: str = "rubberband",
    crisp: int = 5,
    detect_engine: str = "beat_this",
    device: str = "auto",
    out_name: str | None = None,
) -> dict[str, Any]:
    """Stretch a WHOLE file to ``target_bpm`` (pitch preserved, no separation).

    ``source_bpm`` (>0) overrides detection — the escape hatch for half/double
    tempo detection errors. Fail-soft: returns ``{"skipped": ...}`` for a
    non-positive target or an undetectable source, ``{"error": ...}`` for a
    decode/stretch failure; never raises.
    """
    if not target_bpm or float(target_bpm) <= 0:
        return {"skipped": "target_bpm must be > 0"}

    try:
        from . import ingest

        audio = ingest.decode(input_path)  # mp3/m4a via ffmpeg, wav/flac direct

        overridden = source_bpm is not None and float(source_bpm) > 0
        detected = 0.0
        if overridden:
            src = float(source_bpm)
        else:
            detected = detect_bpm(audio, engine=detect_engine, device=device)
            src = detected
        if src <= 0:
            return {"skipped": "no detectable BPM; pass source_bpm to override"}

        ratio = float(target_bpm) / src
        out, used_engine = stretch_with_engine(audio, ratio, engine=engine, crisp=crisp)
        stem = out_name or f"{slugify(Path(input_path).stem)}_{int(round(float(target_bpm)))}bpm.wav"
        outp = save_audio(Path(out_dir) / stem, out)
    except Exception as e:  # noqa: BLE001 - never raise; surface as a soft error
        return {"error": f"match-bpm failed: {e}"}

    return {
        "engine": used_engine,           # actual engine after any fallback
        "engine_requested": engine,
        "detect_engine": detect_engine,
        "source_bpm": round(src, 3),
        "source_bpm_detected": round(detected, 3),
        "source_bpm_overridden": overridden,
        "target_bpm": float(target_bpm),
        "ratio": round(ratio, 4),
        "input": str(input_path),
        "output": str(outp),
    }


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
        "engine": cfg.engine,            # actual engine used (overwritten below)
        "engine_requested": cfg.engine,
        "source_bpm": round(float(source_bpm), 3),
        "target_bpm": cfg.target_bpm,
        "ratios": {},
        "files": {},
    }
    engines_used: list[str] = []
    for name, path in stems.items():
        target = cfg.per_stem_target_bpm.get(name, cfg.target_bpm)
        if not target:
            continue
        ratio = float(target) / float(source_bpm)
        audio = load_audio(path)
        formant = cfg.preserve_formant and name == "vocals"
        out, used = stretch_with_engine(audio, ratio, engine=cfg.engine, crisp=cfg.crisp,
                                        preserve_formant=formant)
        engines_used.append(used)
        outp = save_audio(out_dir / f"{name}_{int(round(float(target)))}bpm.wav", out)
        result["ratios"][name] = round(ratio, 4)
        result["files"][name] = str(outp)
    if engines_used:
        uniq = sorted(set(engines_used))
        result["engine"] = uniq[0] if len(uniq) == 1 else uniq
    return result
