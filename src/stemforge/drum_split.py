"""§5.5 drum_split — decompose a drum loop / drums stem into kick/snare/toms/hh/…

Primary path (``backend: uvr``): run a DrumSep-family model through the SAME
isolated ``.venv-uvr`` audio-separator subprocess used for stem separation
(``separate_uvr.separate_drums``) — never the main env. The drum model is
``drums.split.uvr_model`` when set, else auto-discovered via
``audio-separator --list_filter=drums``. Accepts a raw drum LOOP as input, not
only the demucs drums stem.

Fallback (``external_cmd``): any repo-based separator via a shell template::

    external_cmd: "python /opt/drumsep/infer.py --in {input} --out {output_dir}"

Placeholders ``{input}`` / ``{output_dir}`` are substituted. All paths fail
soft: a missing venv / ffmpeg / drum model records ``{"skipped": ...}`` with a
fix hint so the pipeline continues.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any

from .io_utils import AudioTensor, PathLike, ensure_dir, load_audio, save_audio

try:  # pragma: no cover
    from .orchestrator import DrumSplitCfg
except Exception:  # pragma: no cover
    DrumSplitCfg = Any  # type: ignore

# Canonical parts and the filename keywords we use to map arbitrary outputs to them.
_PART_KEYWORDS = {
    "kick": ("kick", "bd", "bassdrum"),
    "snare": ("snare", "sd"),
    "toms": ("tom", "toms"),
    "hihat": ("hihat", "hi-hat", "hh", "hat"),
    "ride": ("ride",),
    "crash": ("crash", "cymbal", "cym"),
}


def split(
    drums_path: PathLike | None,
    cfg: "DrumSplitCfg",
    out_dir: PathLike,
    device: str = "cuda",
    audio: AudioTensor | None = None,
) -> dict[str, Any]:
    """Split a drum source into hit parts.

    ``audio`` is the source to tear down (a raw drum loop or the drums stem) as
    an ``AudioTensor``; when omitted it is loaded from ``drums_path``. The
    ``external_cmd`` path always works from a file (``drums_path``).
    """
    out_dir = ensure_dir(Path(out_dir))
    backend = (getattr(cfg, "backend", "uvr") or "uvr").strip().lower()

    # Explicit external command wins (repo-based DrumSep/LarsNet), any backend.
    if cfg.external_cmd:
        if drums_path is None or not Path(drums_path).is_file():
            return {"skipped": "external_cmd needs a drums file (run separation first)"}
        return _run_external(cfg.external_cmd, drums_path, out_dir, cfg)

    if backend == "uvr":
        return _uvr_split(audio, drums_path, cfg, out_dir)

    if cfg.model == "larsnet":
        if drums_path is None or not Path(drums_path).is_file():
            return {"skipped": "no drums stem available (run separation first)"}
        try:
            return _larsnet(drums_path, out_dir, cfg, device)
        except _NotWired as e:
            return {"skipped": str(e), "model": cfg.model}

    return {
        "skipped": (
            f"model '{cfg.model}' needs external weights; set drums.split.external_cmd "
            "to your DrumSep/LarsNet inference command (see models/README.md)."
        ),
        "model": cfg.model,
    }


class _NotWired(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# UVR backend (isolated .venv-uvr subprocess) — the real, wired path
# --------------------------------------------------------------------------- #
def _uvr_split(
    audio: AudioTensor | None, drums_path: PathLike | None,
    cfg: "DrumSplitCfg", out_dir: Path,
) -> dict[str, Any]:
    from . import separate_uvr as uvr

    if audio is None:
        if drums_path is None or not Path(drums_path).is_file():
            return {"skipped": "no drum audio (drop a drum loop or run separation first)"}
        audio = load_audio(drums_path)

    try:
        parts, model, _engine = uvr.separate_drums(audio, cfg)
    except uvr.SotaEnvError as e:  # missing venv / ffmpeg / drum model -> fail soft
        return {"skipped": f"{e} — run `stemforge setup-sota`"}
    except RuntimeError as e:  # CLI crashed inside a ready env -> real error
        return {"error": f"drum separation failed: {e}", "backend": "uvr"}

    if not parts:
        return {"skipped": "drum model produced no recognizable parts", "backend": "uvr"}

    wanted = set(cfg.parts) if getattr(cfg, "parts", None) else None
    files: dict[str, str] = {}
    for part, tensor in parts.items():
        if wanted is not None and part not in wanted:
            continue  # honor drums.split.parts (drop residual/unrequested parts)
        files[part] = str(save_audio(out_dir / f"{part}.wav", tensor))

    if not files:  # model output didn't intersect requested parts -> keep everything
        for part, tensor in parts.items():
            files[part] = str(save_audio(out_dir / f"{part}.wav", tensor))

    return {"backend": "uvr", "model": model, "parts": list(files), "files": files}


# --------------------------------------------------------------------------- #
# External-command adapter (works for any repo-based separator)
# --------------------------------------------------------------------------- #
def _run_external(cmd_template: str, drums_path: PathLike, out_dir: Path, cfg: "DrumSplitCfg") -> dict[str, Any]:
    cmd = cmd_template.format(input=str(drums_path), output_dir=str(out_dir))
    try:
        subprocess.run(shlex.split(cmd), check=True, capture_output=True, text=True)
    except FileNotFoundError as e:
        return {"error": f"external_cmd program not found: {e}"}
    except subprocess.CalledProcessError as e:
        return {"error": f"external_cmd failed: {e.stderr[-500:]}"}

    files = _map_parts(out_dir, cfg.parts)
    if not files:
        return {"error": "external_cmd produced no recognizable part files", "model": cfg.model}
    return {"model": cfg.model, "parts": list(files), "files": files}


def _map_parts(out_dir: Path, parts: list[str]) -> dict[str, str]:
    """Match produced .wav files to canonical part names by filename keyword."""
    produced = list(out_dir.glob("*.wav")) + list(out_dir.glob("*.flac"))
    mapping: dict[str, str] = {}
    for part in parts:
        keywords = _PART_KEYWORDS.get(part, (part,))
        for f in produced:
            low = f.stem.lower()
            if any(k in low for k in keywords) and part not in mapping:
                mapping[part] = str(f)
                break
    return mapping


# --------------------------------------------------------------------------- #
# LarsNet in-process adapter (if the package is importable)
# --------------------------------------------------------------------------- #
def _larsnet(drums_path: PathLike, out_dir: Path, cfg: "DrumSplitCfg", device: str) -> dict[str, Any]:
    try:
        import larsnet  # type: ignore  # noqa: F401
    except ImportError as e:
        raise _NotWired(
            "LarsNet not importable. Clone polimi-ispl/larsnet and expose it, or use "
            "external_cmd. alpha-Wiener crosstalk control: larsnet_wiener="
            f"{cfg.larsnet_wiener}."
        ) from e
    # Repo API is not a stable package; drive it via external_cmd in practice.
    raise _NotWired("LarsNet found but has no stable in-process API; use external_cmd.")


def available_models() -> list[str]:
    return ["uvr", "drumsep_roformer", "larsnet", "inagoy"]
