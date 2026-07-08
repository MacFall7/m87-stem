"""§5.5 drum_split — decompose the drum stem into kick/snare/toms/hh/ride/crash.

Primary: DrumSep (Jarredou/aufr33 MelBand Roformer 6-stem). Fallbacks: LarsNet
(5-stem, alpha-Wiener option) and inagoy drumsep (4-stem HDemucs). These ship as
repos + weights rather than clean PyPI packages, so the robust integration point
is a configurable ``external_cmd`` template that runs the repo's inference and
writes part WAVs, which we then map to canonical names.

Set ``drums.split.external_cmd`` in the config, e.g.::

    external_cmd: "python /opt/drumsep/infer.py --in {input} --out {output_dir}"

Placeholders ``{input}`` and ``{output_dir}`` are substituted. If no command is
configured and no in-process adapter is available, a ``skipped`` record is
returned so the pipeline continues.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any

from .io_utils import PathLike, ensure_dir

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
) -> dict[str, Any]:
    if drums_path is None or not Path(drums_path).is_file():
        return {"skipped": "no drums stem available (run separation first)"}

    out_dir = ensure_dir(Path(out_dir))

    if cfg.external_cmd:
        return _run_external(cfg.external_cmd, drums_path, out_dir, cfg)

    if cfg.model == "larsnet":
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
    return ["drumsep_roformer", "larsnet", "inagoy"]
