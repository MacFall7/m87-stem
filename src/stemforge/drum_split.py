"""§5.5 drum_split — decompose a drum loop / drums stem into kick/snare/toms/other.

Primary path (``backend: demucs_inagoy``, the default): run the **inagoy/drumsep
Demucs checkpoint in the MAIN env** via the already-installed ``demucs`` package
(no venv, GPU-fast). The checkpoint auto-downloads on first use into
``models/drumsep/`` (``drums.split.inagoy_url`` / ``inagoy_model_dir``). It
splits a drum loop/stem into kick/snare/toms/other. audio-separator's registry
has NO per-hit drum model — its ``--list_filter=drums`` models only isolate the
kit — so the previous ``uvr`` drum path never produced hits; it remains as an
option but is no longer the default.

Option (``backend: uvr_drumsep``): the MDX23C per-hit DrumSep model
(``MDX23C-DrumSep-aufr33-jarredou.ckpt``, in audio-separator's registry) run
through the isolated ``.venv-uvr`` subprocess — no torch-2.6 weights_only hack
and no gated demucs checkpoint. Not the default only because its ``.ckpt``
couldn't be download-verified in this build env (the CI proxy blocks GitHub
release assets); it resolves on an open network.

Fallback (``external_cmd``): any repo-based separator via a shell template::

    external_cmd: "python /opt/drumsep/infer.py --in {input} --out {output_dir}"

All paths fail soft: a model that can't download / load records
``{"skipped": ...}`` with a fix hint so the pipeline continues. The produced
parts feed the existing parts-based ``drum_midi`` (onset + RMS-velocity + GM).
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

# Anchor for a relative model dir, so the checkpoint cache is shared, not per-cwd.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Canonical parts and the filename keywords we use to map arbitrary outputs to them.
_PART_KEYWORDS = {
    "kick": ("kick", "bd", "bassdrum"),
    "snare": ("snare", "sd"),
    "toms": ("tom", "toms"),
    "hihat": ("hihat", "hi-hat", "hh", "hat"),
    "ride": ("ride",),
    "crash": ("crash", "cymbal", "cym"),
}

# inagoy/drumsep emits 4 sources (labels vary / are Spanish in some builds);
# map by keyword, with a positional fallback in kit order.
_INAGOY_ALIASES = {
    "kick": ("kick", "bombo", "bd", "bass drum", "bassdrum"),
    "snare": ("snare", "redoblante", "caja", "sd"),
    "toms": ("toms", "tom"),
    "other": ("other", "platillos", "cymbals", "cymbal", "hihat", "hi-hat",
              "hh", "crash", "ride", "resto", "rest"),
}
_INAGOY_POSITIONAL = ["kick", "snare", "toms", "other"]


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
    backend = (getattr(cfg, "backend", "demucs_inagoy") or "demucs_inagoy").strip().lower()

    # Explicit external command wins (repo-based DrumSep/LarsNet), any backend.
    if cfg.external_cmd:
        if drums_path is None or not Path(drums_path).is_file():
            return {"skipped": "external_cmd needs a drums file (run separation first)"}
        return _run_external(cfg.external_cmd, drums_path, out_dir, cfg)

    if backend == "demucs_inagoy":
        return _demucs_inagoy_split(_source_audio(audio, drums_path), cfg, out_dir, device)

    if backend == "uvr_drumsep":
        # MDX23C per-hit DrumSep via the isolated .venv-uvr subprocess. Keep ALL
        # recognized hits (kick/snare/toms/hihat/ride/crash) — the model may emit
        # more than the inagoy 4. (The MelBand-Roformer drum model is NOT in
        # audio-separator's registry — it needs the separate MSST framework — so
        # it's intentionally not wired here; a future option.)
        return _uvr_split(audio, drums_path, cfg, out_dir,
                          model=getattr(cfg, "uvr_drumsep_model", None), keep_all=True)

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


class _DrumModelUnavailable(RuntimeError):
    """The inagoy checkpoint can't be fetched/loaded — fail soft (skip)."""


def _source_audio(audio: AudioTensor | None, drums_path: PathLike | None) -> AudioTensor | None:
    if audio is not None:
        return audio
    if drums_path is not None and Path(drums_path).is_file():
        return load_audio(drums_path)
    return None


# --------------------------------------------------------------------------- #
# inagoy/drumsep backend — Demucs checkpoint in the MAIN env (default)
# --------------------------------------------------------------------------- #
def _demucs_inagoy_split(
    audio: AudioTensor | None, cfg: "DrumSplitCfg", out_dir: Path, device: str,
) -> dict[str, Any]:
    if audio is None:
        return {"skipped": "no drum audio (drop a drum loop or run separation first)"}

    try:
        model, model_name = _load_inagoy_model(cfg, device)
    except _DrumModelUnavailable as e:
        return {"skipped": f"{e}", "backend": "demucs_inagoy"}
    except ModuleNotFoundError as e:  # demucs / torch not installed
        return {"skipped": f"demucs not installed ({e.name}); pip install demucs",
                "backend": "demucs_inagoy"}

    try:
        from . import separate as sep

        sources = sep.apply_model_to_audio(model, audio, device=device)
    except RuntimeError as e:  # OOM / runtime failure inside a loaded model
        return {"error": f"inagoy drumsep failed: {e}", "backend": "demucs_inagoy"}

    parts = _map_inagoy_sources(sources)
    wanted = set(cfg.parts) if getattr(cfg, "parts", None) else None
    files: dict[str, str] = {}
    for part, tensor in parts.items():
        if wanted is not None and part not in wanted and part != "other":
            continue
        files[part] = str(save_audio(out_dir / f"{part}.wav", tensor))
    if not files:
        return {"skipped": "inagoy drumsep produced no recognizable parts",
                "backend": "demucs_inagoy"}
    return {"backend": "demucs_inagoy", "model": model_name,
            "parts": list(files), "files": files}


def _map_inagoy_sources(sources: dict[str, AudioTensor]) -> dict[str, "AudioTensor"]:
    """Map the model's 4 sources to canonical kick/snare/toms/other.

    Prefer name/keyword matches (English or Spanish labels); fall back to kit
    order for anything unmatched so no source is dropped."""
    out: dict[str, AudioTensor] = {}
    remaining = dict(sources)
    for canon, aliases in _INAGOY_ALIASES.items():
        for name, tensor in list(remaining.items()):
            if any(a in name.lower() for a in aliases):
                out[canon] = tensor
                remaining.pop(name)
                break
    leftovers = list(remaining.values())
    for canon in _INAGOY_POSITIONAL:
        if canon not in out and leftovers:
            out[canon] = leftovers.pop(0)
    return out


def _inagoy_filename(url: str) -> str:
    name = url.rsplit("/", 1)[-1].split("?")[0]
    return name or "inagoy_drumsep.th"


def _ensure_inagoy_checkpoint(cfg: "DrumSplitCfg") -> Path:
    model_dir = Path(cfg.inagoy_model_dir).expanduser()
    if not model_dir.is_absolute():
        model_dir = _PROJECT_ROOT / model_dir
    ensure_dir(model_dir)
    dest = model_dir / _inagoy_filename(cfg.inagoy_url or "")
    if dest.is_file():
        return dest
    if not cfg.inagoy_url:
        raise _DrumModelUnavailable(
            "no inagoy drumsep URL — set drums.split.inagoy_url to the checkpoint"
        )
    import urllib.request

    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        urllib.request.urlretrieve(cfg.inagoy_url, tmp)  # noqa: S310 - user-configured URL
        tmp.replace(dest)
    except Exception as e:  # noqa: BLE001 - any network/FS failure -> fail soft
        tmp.unlink(missing_ok=True)
        raise _DrumModelUnavailable(
            f"could not download inagoy drumsep model ({e}); set drums.split.inagoy_url "
            f"or drop the checkpoint into {model_dir}"
        ) from e
    return dest


def _load_inagoy_model(cfg: "DrumSplitCfg", device: str):
    """Download (if needed) + load the inagoy drumsep Demucs checkpoint."""
    dest = _ensure_inagoy_checkpoint(cfg)
    from demucs.states import load_model  # lazy: main-env demucs (MNFE -> caller skips)

    try:
        model = _load_demucs_checkpoint(load_model, dest)
    except Exception as e:  # noqa: BLE001 - corrupt/incompatible checkpoint -> fail soft
        raise _DrumModelUnavailable(
            f"could not load inagoy drumsep checkpoint {dest.name} ({e}); "
            f"re-download or set drums.split.inagoy_url"
        ) from e
    model.to(device)
    model.eval()
    return model, dest.name


def _load_demucs_checkpoint(load_model, dest: Path):
    """Load a trusted demucs checkpoint across torch versions.

    torch>=2.6 flipped ``torch.load(weights_only=True)`` on by default, which
    rejects the pickled demucs model classes (``UnpicklingError: Unsupported
    global: GLOBAL demucs.hdemucs.HDemucs was not an allowed global by
    default``). The drumsep checkpoint is a trusted MIT file, so on that failure
    we retry with ``weights_only=False`` (``demucs.states.load_model`` doesn't
    expose the flag, so we patch ``torch.load`` for the retry). On older torch
    the first attempt already succeeds and the retry never runs.
    """
    try:
        return load_model(str(dest))
    except Exception:  # noqa: BLE001 - retry allowing the trusted full unpickle
        import torch

        orig_load = torch.load

        def _trusted_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return orig_load(*args, **kwargs)

        torch.load = _trusted_load
        try:
            return load_model(str(dest))
        finally:
            torch.load = orig_load


# --------------------------------------------------------------------------- #
# UVR backend (isolated .venv-uvr subprocess) — the real, wired path
# --------------------------------------------------------------------------- #
def _uvr_split(
    audio: AudioTensor | None, drums_path: PathLike | None,
    cfg: "DrumSplitCfg", out_dir: Path,
    model: str | None = None, keep_all: bool = False,
) -> dict[str, Any]:
    from . import separate_uvr as uvr

    backend = "uvr_drumsep" if model else "uvr"
    if audio is None:
        if drums_path is None or not Path(drums_path).is_file():
            return {"skipped": "no drum audio (drop a drum loop or run separation first)"}
        audio = load_audio(drums_path)

    try:
        parts, used_model, _engine = uvr.separate_drums(audio, cfg, model=model)
    except uvr.SotaEnvError as e:  # missing venv / ffmpeg / drum model -> fail soft
        return {"skipped": f"{e} — run `stemforge setup-sota`", "backend": backend}
    except RuntimeError as e:  # CLI crashed inside a ready env -> real error
        return {"error": f"drum separation failed: {e}", "backend": backend}

    if not parts:
        return {"skipped": "drum model produced no recognizable parts", "backend": backend}

    # keep_all (uvr_drumsep): keep every recognized hit; otherwise honor drums.split.parts.
    wanted = None if keep_all else (set(cfg.parts) if getattr(cfg, "parts", None) else None)
    files: dict[str, str] = {}
    for part, tensor in parts.items():
        if wanted is not None and part not in wanted:
            continue  # drop residual/unrequested parts
        files[part] = str(save_audio(out_dir / f"{part}.wav", tensor))

    if not files:  # model output didn't intersect requested parts -> keep everything
        for part, tensor in parts.items():
            files[part] = str(save_audio(out_dir / f"{part}.wav", tensor))

    return {"backend": backend, "model": used_model, "parts": list(files), "files": files}


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
    return ["demucs_inagoy", "uvr_drumsep", "uvr", "larsnet", "external"]
