"""§5.1 ingest — decode any input to canonical float32 / 44.1 kHz / stereo and
loudness-normalize for consistent downstream behavior (ADT velocity depends on it).
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .io_utils import AudioTensor, PathLike, load_audio

_SNDFILE_OK = {".wav", ".flac", ".ogg", ".aiff", ".aif"}


def decode(path: PathLike, sample_rate: int = 44100, channels: int = 2) -> AudioTensor:
    """Decode to AudioTensor at the target rate/channel count.

    libsndfile handles wav/flac/ogg directly; everything else (mp3/m4a/opus/...)
    is routed through ffmpeg to a temporary float WAV.
    """
    p = Path(path)
    if p.suffix.lower() in _SNDFILE_OK:
        audio = load_audio(p)
    else:
        audio = _ffmpeg_decode(p, sample_rate, channels)

    audio = _match_channels(audio, channels)
    audio = _resample(audio, sample_rate)
    return audio


def _ffmpeg_decode(path: Path, sample_rate: int, channels: int) -> AudioTensor:
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "decoded.wav"
        cmd = [
            "ffmpeg", "-nostdin", "-y", "-i", str(path),
            "-ac", str(channels), "-ar", str(sample_rate),
            "-c:a", "pcm_f32le", str(wav),
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True)
        except FileNotFoundError as e:
            raise RuntimeError("ffmpeg not found — install it to decode this format.") from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg failed to decode {path.name}:\n{e.stderr.decode()}") from e
        return load_audio(wav)


def _match_channels(audio: AudioTensor, channels: int) -> AudioTensor:
    if channels == 1:
        return audio.to_mono()
    if channels == 2:
        return audio.to_stereo()
    return audio


def _resample(audio: AudioTensor, target_sr: int) -> AudioTensor:
    if audio.sample_rate == target_sr:
        return audio
    import librosa  # lazy

    out = librosa.resample(audio.samples, orig_sr=audio.sample_rate, target_sr=target_sr, axis=1)
    return AudioTensor(out.astype(np.float32), target_sr)


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
def normalize(audio: AudioTensor, method: str = "replaygain", target_lufs: float = -18.0) -> AudioTensor:
    """method: 'replaygain' (LUFS via pyloudnorm) | 'peak' | 'none'."""
    if method == "none":
        return audio
    if method == "peak":
        return _peak_normalize(audio, target_dbfs=-1.0)
    if method == "replaygain":
        return _lufs_normalize(audio, target_lufs)
    raise ValueError(f"unknown normalize method: {method}")


def _peak_normalize(audio: AudioTensor, target_dbfs: float = -1.0) -> AudioTensor:
    peak = audio.peak()
    if peak <= 0:
        return audio
    target = 10 ** (target_dbfs / 20.0)
    return AudioTensor((audio.samples * (target / peak)).astype(np.float32), audio.sample_rate)


def _lufs_normalize(audio: AudioTensor, target_lufs: float) -> AudioTensor:
    try:
        import pyloudnorm as pyln  # lazy
    except ImportError:
        # Graceful fallback: peak-normalize instead of failing the run.
        return _peak_normalize(audio, target_dbfs=-1.0)

    meter = pyln.Meter(audio.sample_rate)
    interleaved = audio.samples.T  # (frames, channels) as pyloudnorm expects
    loudness = meter.integrated_loudness(interleaved)
    if not np.isfinite(loudness):
        return _peak_normalize(audio, target_dbfs=-1.0)
    normalized = pyln.normalize.loudness(interleaved, loudness, target_lufs)
    # guard against clipping introduced by the gain
    out = AudioTensor(normalized.T.astype(np.float32), audio.sample_rate)
    if out.peak() > 1.0:
        out = _peak_normalize(out, target_dbfs=-0.5)
    return out


def ingest(path: PathLike, sample_rate: int = 44100, channels: int = 2,
           normalize_method: str = "replaygain", target_lufs: float = -18.0) -> AudioTensor:
    """Full ingest: decode -> resample -> channel match -> normalize."""
    audio = decode(path, sample_rate=sample_rate, channels=channels)
    return normalize(audio, method=normalize_method, target_lufs=target_lufs)
