"""I/O + small utilities shared by every module.

The canonical audio container is :class:`AudioTensor` — a channels-first float32
numpy array plus a sample rate. Torch conversion is lazy so this module (and the
whole package) imports without torch installed.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf

PathLike = str | Path


# --------------------------------------------------------------------------- #
# Engine fallback evidence (shared by analysis + stretch)
# --------------------------------------------------------------------------- #
@dataclass
class EngineAttempt:
    """One hop in an engine fallback chain — the analysis beat_this→librosa hop or
    the stretch rubberband→signalsmith→librosa chain.

    ``status`` is ``"used"`` (this engine produced the output), ``"fell_through"``
    (it failed and the chain continued), or ``"failed"``. ``reason`` is a
    sanitized, single-line message (no absolute paths or subprocess dumps), empty
    when the engine was used.
    """

    engine: str
    status: str
    reason: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"engine": self.engine, "status": self.status, "reason": self.reason}


_PATH_RE = re.compile(r"(?:[A-Za-z]:\\|/)[\w./\\-]+")


def sanitize_reason(err: Any, limit: int = 200) -> str:
    """A deterministic, log-safe reason string for a fallback record: drop
    absolute paths, collapse newlines/whitespace (kills multi-line subprocess
    dumps), and truncate — so manifests stay reproducible and leak no host paths.
    """
    s = err if isinstance(err, str) else f"{type(err).__name__}: {err}"
    s = _PATH_RE.sub("<path>", s)
    s = " ".join(s.split())
    return s[:limit]


# --------------------------------------------------------------------------- #
# Audio container
# --------------------------------------------------------------------------- #
@dataclass
class AudioTensor:
    """Channels-first float32 audio. Shape: (channels, num_samples)."""

    samples: np.ndarray
    sample_rate: int

    def __post_init__(self) -> None:
        if self.samples.ndim == 1:
            self.samples = self.samples[np.newaxis, :]
        if self.samples.ndim != 2:
            raise ValueError(f"AudioTensor expects 1D or 2D array, got {self.samples.ndim}D")
        if self.samples.dtype != np.float32:
            self.samples = self.samples.astype(np.float32)

    # -- properties -------------------------------------------------------- #
    @property
    def num_channels(self) -> int:
        return self.samples.shape[0]

    @property
    def num_samples(self) -> int:
        return self.samples.shape[1]

    @property
    def duration_seconds(self) -> float:
        return self.num_samples / self.sample_rate

    # -- transforms -------------------------------------------------------- #
    def to_mono(self) -> "AudioTensor":
        if self.num_channels == 1:
            return self
        return AudioTensor(self.samples.mean(axis=0, keepdims=True), self.sample_rate)

    def to_stereo(self) -> "AudioTensor":
        if self.num_channels == 2:
            return self
        if self.num_channels == 1:
            return AudioTensor(np.repeat(self.samples, 2, axis=0), self.sample_rate)
        return AudioTensor(self.samples[:2], self.sample_rate)

    def peak(self) -> float:
        return float(np.max(np.abs(self.samples))) if self.num_samples else 0.0

    def torch(self):  # -> "torch.Tensor"
        """Lazy torch view (channels, samples). Imports torch only when called."""
        import torch

        return torch.from_numpy(self.samples)

    @classmethod
    def from_torch(cls, tensor, sample_rate: int) -> "AudioTensor":
        arr = tensor.detach().cpu().numpy()
        return cls(np.ascontiguousarray(arr), sample_rate)


# --------------------------------------------------------------------------- #
# Load / save
# --------------------------------------------------------------------------- #
def load_audio(path: PathLike) -> AudioTensor:
    """Read a WAV/FLAC/OGG via libsndfile. For mp3/m4a/exotic, use ingest.decode()."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (frames, channels)
    return AudioTensor(data.T.copy(), sr)


def save_audio(path: PathLike, audio: AudioTensor, subtype: str | None = None) -> Path:
    """Write channels-first audio. subtype e.g. 'FLOAT', 'PCM_24', 'PCM_16'."""
    p = Path(path)
    ensure_dir(p.parent)
    fmt = p.suffix.lstrip(".").upper() or "WAV"
    if subtype is None:
        subtype = "FLOAT" if fmt == "WAV" else None
    sf.write(str(p), audio.samples.T, audio.sample_rate, subtype=subtype, format=fmt)
    return p


def ffprobe_duration(path: PathLike) -> float | None:
    """Best-effort media duration via ffprobe; None if unavailable."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, check=True,
        )
        return float(out.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Filesystem + hashing (receipts / reproducibility)
# --------------------------------------------------------------------------- #
def ensure_dir(path: PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256_file(path: PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: PathLike, obj: Any) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")
    return p


def read_json(path: PathLike) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    # np.generic covers EVERY numpy scalar — bool_, floating, integer — across
    # numpy 1.x and 2.x (2.0 renamed np.bool_'s repr to numpy.bool). The A1 cymbal
    # evidence was the first path to put an np.bool_ into a manifest, which the old
    # (np.floating, np.integer) branch missed → write_json crashed on drum teardown.
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON-serializable: {type(o)}")


def slugify(name: str) -> str:
    """Filesystem-safe stem name from a song filename."""
    keep = "-_.() "
    cleaned = "".join(c if c.isalnum() or c in keep else "_" for c in name).strip()
    return "_".join(cleaned.split()) or "song"


def receipt(files: Iterable[PathLike], root: PathLike) -> dict[str, str]:
    """Map of run-root-relative POSIX path -> sha256 (the completion receipt).

    Keys are relative to ``root`` (the run's output dir), NOT basenames, so two
    files sharing a name in different subdirs (``stems/other.wav`` and
    ``drums/other.wav``) get distinct, collision-free keys. A file outside
    ``root`` falls back to its basename.
    """
    root_res = Path(root).resolve()
    out: dict[str, str] = {}
    for f in files:
        p = Path(f)
        if not p.is_file():
            continue
        try:
            key = p.resolve().relative_to(root_res).as_posix()
        except ValueError:
            key = p.name
        out[key] = sha256_file(p)
    return out
