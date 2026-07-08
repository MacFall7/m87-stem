"""§5.3b separate_uvr — SOTA separation via `audio-separator` (UVR model zoo).

Wraps ``audio_separator.separator.Separator`` to run BS-Roformer / MDX / VR /
Demucs checkpoints from the UVR ecosystem. audio-separator is imported lazily so
``import stemforge`` (and pytest) never needs it; install with the
``separation-sota`` extra (``pip install "audio-separator[gpu]"`` — reuses the
existing torch + onnxruntime-gpu stack, NO TensorFlow).

audio-separator is file-based: it reads an input path and writes stem files
whose names carry a parenthesized stem token, e.g.
``{track}_(Vocals)_{model}.wav`` / ``(Instrumental)`` / ``(Drums)`` /
``(Bass)`` / ``(Other)``. We round-trip through a scratch directory and map
those tokens back to canonical stem names, returning ``dict[str, AudioTensor]``
like the Demucs backend. Models auto-download into ``cfg.uvr_model_dir`` on
first load.
"""

from __future__ import annotations

import re
import shutil
import tempfile
import uuid
import weakref
from pathlib import Path
from typing import Any

from .io_utils import AudioTensor, PathLike, ensure_dir, load_audio, save_audio, slugify

# forward ref only; avoids importing orchestrator at module load
try:  # pragma: no cover - typing convenience
    from .orchestrator import SeparationCfg
except Exception:  # pragma: no cover
    SeparationCfg = Any  # type: ignore

# Parenthesized stem token in audio-separator output filenames -> canonical name.
_STEM_TOKEN_MAP = {
    "vocals": "vocals",
    "instrumental": "instrumental",
    "no vocals": "instrumental",
    "drums": "drums",
    "bass": "bass",
    "other": "other",
    "guitar": "guitar",
    "piano": "piano",
}

_TOKEN_RE = re.compile(r"\(([^()]+)\)")

# Anchor for relative model dirs (same anchor as configs/ in orchestrator), so
# the multi-hundred-MB checkpoint cache doesn't re-download per working dir.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_model_dir(model_file_dir: str) -> str:
    p = Path(model_file_dir).expanduser()
    return str(p if p.is_absolute() else _PROJECT_ROOT / p)


def canonical_stem_name(filename: PathLike) -> str:
    """Map an audio-separator output filename to a canonical stem name.

    Scans parenthesized tokens (``mix_(Vocals)_model_bs_roformer....wav`` ->
    ``vocals``). Unrecognized outputs keep a slugified version of their stem so
    no file is silently dropped.
    """
    stem = Path(filename).stem
    for token in _TOKEN_RE.findall(stem):
        key = token.strip().lower()
        if key in _STEM_TOKEN_MAP:
            return _STEM_TOKEN_MAP[key]
    return slugify(stem).lower()


class UvrEngine:
    """A loaded audio-separator model plus its scratch output directory.

    audio-separator binds ``output_dir`` at model-load time, so the scratch dir
    lives as long as the engine; per-call outputs are read back and deleted.
    Cache an instance across a batch exactly like the Demucs model object.
    """

    def __init__(self, cfg: "SeparationCfg", work_dir: PathLike | None = None):
        from audio_separator.separator import Separator  # lazy: optional dep

        self.model_filename: str = cfg.uvr_model
        owns_work_dir = work_dir is None
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="stemforge-uvr-"))
        ensure_dir(self.work_dir)
        if owns_work_dir:  # remove the scratch dir when the engine is collected
            self._cleanup = weakref.finalize(self, shutil.rmtree, str(self.work_dir), True)
        self.separator = Separator(
            output_dir=str(self.work_dir),
            output_format="WAV",
            model_file_dir=_resolve_model_dir(cfg.uvr_model_dir),
            use_autocast=cfg.uvr_use_autocast,
            demucs_params={
                "shifts": cfg.shifts,
                "overlap": cfg.overlap,
                "segment_size": "Default",
            },
        )
        self.separator.load_model(model_filename=cfg.uvr_model)

    def separate_file(self, path: PathLike) -> dict[str, Path]:
        """Run the loaded model on an audio file; return canonical name -> path."""
        produced = self.separator.separate(str(path))
        out: dict[str, Path] = {}
        for f in produced:
            p = Path(f)
            if not p.is_absolute():  # some versions return names relative to output_dir
                p = self.work_dir / p
            out[canonical_stem_name(p.name)] = p
        return out


def separate(
    audio: AudioTensor,
    cfg: "SeparationCfg",
    device: str = "cuda",
    engine: UvrEngine | None = None,
) -> tuple[dict[str, AudioTensor], UvrEngine]:
    """Separate ``audio`` with a UVR-family model.

    Returns ``(stems, engine)`` — the engine is returned so the caller can cache
    the loaded model across a batch, mirroring :func:`stemforge.separate.separate`.
    ``device`` is accepted for backend-signature parity; audio-separator picks
    CUDA automatically when available.
    """
    if engine is None or engine.model_filename != cfg.uvr_model:
        engine = UvrEngine(cfg)

    in_path = engine.work_dir / f"{uuid.uuid4().hex[:12]}.wav"
    save_audio(in_path, audio)
    by_name: dict[str, Path] = {}
    try:
        by_name = engine.separate_file(in_path)
        stems = {name: load_audio(p) for name, p in by_name.items()}
    finally:
        in_path.unlink(missing_ok=True)
        for p in by_name.values():
            p.unlink(missing_ok=True)

    from .separate import _select  # honor cfg.stems the same way the demucs backend does

    return _select(stems, cfg), engine
