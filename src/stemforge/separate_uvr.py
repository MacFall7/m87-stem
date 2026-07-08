"""§5.3b separate_uvr — SOTA separation via `audio-separator`, ISOLATED in its own venv.

audio-separator (BS-Roformer / MDX / VR / the UVR model zoo) is run as a
**subprocess** from a dedicated virtual environment (default
``<project-root>/.venv-uvr``, configurable via ``separation.uvr_venv``) and is
NEVER imported in-process. Hard-won reason (Windows / py3.11, 2026-07):
``pip install "audio-separator[gpu]"`` into the main env pulled a **CPU-only
torch 2.13.0+cpu** that replaced torch 2.6.0+cu124 and bumped numpy to 2.4.6,
silently breaking the working demucs GPU stack. Isolation makes that
structurally impossible — the main interpreter's packages are never touched.

Provision the venv with ``stemforge setup-sota`` (idempotent). Separation
preflights the venv + ffmpeg and fails soft (manifest ``skipped`` + fix hint)
when either is missing; it never creates the venv or downloads anything
mid-pipeline.

The CLI is file-based: it reads an input path and writes stem files whose
names carry a parenthesized stem token — ``{track}_(Vocals)_{model}.wav`` /
``(Instrumental)`` / ``(Drums)`` / ``(Bass)`` / ``(Other)`` — which we map back
to canonical stem names and load as :class:`AudioTensor`. Models auto-download
(inside the subprocess) into ``cfg.uvr_model_dir`` on first use.
"""

from __future__ import annotations

import os
import re
import json
import shutil
import subprocess
import sys
import tempfile
import uuid
import weakref
from pathlib import Path
from typing import Any, Callable

from .io_utils import AudioTensor, PathLike, ensure_dir, load_audio, save_audio, slugify

# forward ref only; avoids importing orchestrator at module load
try:  # pragma: no cover - typing convenience
    from .orchestrator import SeparationCfg
except Exception:  # pragma: no cover
    SeparationCfg = Any  # type: ignore

#: What `stemforge setup-sota` installs into the ISOLATED venv (never the main env).
AUDIO_SEPARATOR_SPEC = "audio-separator[gpu]>=0.44"
TORCH_CU124_INDEX = "https://download.pytorch.org/whl/cu124"

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

# Parenthesized stem token in drum-model output filenames -> canonical part name.
_DRUM_TOKEN_MAP = {
    "kick": "kick",
    "bass drum": "kick",
    "bassdrum": "kick",
    "bd": "kick",
    "snare": "snare",
    "sd": "snare",
    "toms": "toms",
    "tom": "toms",
    "hihat": "hihat",
    "hi-hat": "hihat",
    "hh": "hihat",
    "hat": "hihat",
    "ride": "ride",
    "crash": "crash",
    "cymbals": "crash",
    "cymbal": "crash",
    "other": "other",
}

# Model-file extensions audio-separator lists / caches, used to parse `--list_*`.
_MODEL_EXT_RE = re.compile(r"[\w.\-]+\.(?:ckpt|onnx|yaml|th|pt|pth|safetensors)", re.IGNORECASE)

# Anchor for relative venv/model dirs (same anchor as configs/ in orchestrator),
# so neither the venv nor the checkpoint cache is duplicated per working dir.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class SotaEnvError(RuntimeError):
    """Preflight failure of the isolated audio-separator environment.

    The orchestrator records this as a manifest ``skipped`` entry — a missing
    venv or ffmpeg must never crash the run (and never mutate the main env).
    """


# --------------------------------------------------------------------------- #
# Filename token mapping
# --------------------------------------------------------------------------- #
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


def canonical_drum_part(filename: PathLike) -> str:
    """Map a drum-model output filename to a canonical part name.

    ``loop_(Kick)_drumsep.wav`` -> ``kick``; ``(Hi-Hat)`` -> ``hihat``. Unknown
    tokens keep a slugified stem so no produced file is silently dropped.
    """
    stem = Path(filename).stem
    for token in _TOKEN_RE.findall(stem):
        key = token.strip().lower()
        if key in _DRUM_TOKEN_MAP:
            return _DRUM_TOKEN_MAP[key]
    return slugify(stem).lower()


# --------------------------------------------------------------------------- #
# Isolated venv discovery / provisioning
# --------------------------------------------------------------------------- #
def resolve_venv_dir(venv: PathLike) -> Path:
    p = Path(venv).expanduser()
    return p if p.is_absolute() else _PROJECT_ROOT / p


def _resolve_model_dir(model_file_dir: str) -> str:
    p = Path(model_file_dir).expanduser()
    return str(p if p.is_absolute() else _PROJECT_ROOT / p)


def venv_python(venv_dir: Path) -> Path:
    sub = "Scripts" if os.name == "nt" else "bin"
    exe = "python.exe" if os.name == "nt" else "python"
    return venv_dir / sub / exe


def find_cli(venv_dir: Path) -> Path | None:
    """Path of the venv's `audio-separator` console script, or None."""
    sub = "Scripts" if os.name == "nt" else "bin"
    for name in ("audio-separator.exe", "audio-separator"):
        cli = venv_dir / sub / name
        if cli.exists():
            return cli
    return None


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def list_drum_models(cli: Path, model_file_dir: str) -> list[str]:
    """Discover installed/available drum-separation models in the isolated venv.

    Runs ``audio-separator --list_filter=drums`` (the CLI's own filtered listing)
    and extracts model filenames from its output. Returns an empty list on any
    failure — the caller fails soft.
    """
    cmd = [str(cli), "--list_filter=drums", "--model_file_dir", str(model_file_dir)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except OSError:
        return []
    if proc.returncode != 0:
        return []
    out: list[str] = []
    for match in _MODEL_EXT_RE.findall(proc.stdout or ""):
        if match not in out:
            out.append(match)
    return out


def discover_drum_model(cli: Path, model_file_dir: str) -> str | None:
    """Pick the best drum model available, preferring a full 6-piece kit.

    Preference order: names hinting at a 6-stem / kick-snare-toms-hh-ride-crash
    kit, then a dedicated ``drumsep`` model, then whatever is listed first.
    Returns None when no drum model is available (fail-soft skip upstream).
    """
    models = list_drum_models(cli, model_file_dir)
    if not models:
        return None
    for hint in ("6", "kick", "drumsep", "drum_sep"):
        for m in models:
            if hint in m.lower():
                return m
    return models[0]


def venv_torch_status(venv_dir: Path) -> dict | None:
    """Report the venv's torch build by running a probe inside it.

    Returns ``{"version", "cuda", "name"}`` or ``None`` if torch is absent /
    not importable. Runs in the isolated interpreter — never imports torch here.
    """
    py = venv_python(venv_dir)
    if not py.exists():
        return None
    code = (
        "import json, torch;"
        "print(json.dumps({"
        "'version': torch.__version__,"
        "'cuda': torch.cuda.is_available(),"
        "'name': (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
        "}))"
    )
    try:
        proc = subprocess.run([str(py), "-c", code], capture_output=True, text=True)
    except OSError:
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None


def _ensure_cuda_torch(venv_dir: Path, log: Callable[[str], None]) -> None:
    """Force a CUDA torch build to win in the venv.

    audio-separator's dependency resolution downgrades torch to a CPU wheel
    (``+cpu``), which is why ``--preset max`` ran on CPU. We reinstall the cu124
    build LAST with ``--force-reinstall --no-deps`` so nothing re-drags a CPU
    torch back in. Idempotent (skips when torch is already a ``+cu`` build) and
    fail-soft (a failed reinstall warns, never raises).
    """
    status = venv_torch_status(venv_dir)
    if status and "+cu" in str(status.get("version", "")):
        _log_torch(status, log)  # already a CUDA build -> nothing to do
        return

    log("forcing CUDA torch to win in the venv (audio-separator downgraded it to CPU)")
    py = venv_python(venv_dir)
    try:
        subprocess.run(
            [str(py), "-m", "pip", "install", "--force-reinstall", "--no-deps",
             "torch", "torchaudio", "--index-url", TORCH_CU124_INDEX],
            check=True,
        )
    except (subprocess.CalledProcessError, OSError) as e:
        log(f"WARNING: could not reinstall CUDA torch ({e}); SOTA separation may run on CPU")
        return

    status = venv_torch_status(venv_dir)
    if status is None:
        log("WARNING: torch not importable in the venv after reinstall")
    else:
        _log_torch(status, log)


def _log_torch(status: dict, log: Callable[[str], None]) -> None:
    version = status.get("version", "?")
    if status.get("cuda"):
        log(f"venv torch: {version} · CUDA available · {status.get('name')}")
    else:
        log(f"venv torch: {version} · CUDA NOT available at runtime "
            "(check the GPU driver / `nvidia-smi`)")


def setup_sota_env(cfg: "SeparationCfg", venv: PathLike | None = None,
                   log: Callable[[str], None] = print) -> bool:
    """Create/repair the isolated audio-separator venv. Idempotent.

    Installs CUDA-matched torch/torchaudio FIRST (cu124 index), then
    audio-separator, then force-reinstalls the cu124 torch LAST so
    audio-separator's deps can't leave a CPU torch behind — into the venv ONLY.
    The main interpreter's packages are never touched. Returns True when the
    venv is ready (CLI present + ffmpeg on PATH); a CPU-only GPU state warns but
    does not fail.
    """
    venv_dir = resolve_venv_dir(venv or cfg.uvr_venv)
    py = venv_python(venv_dir)

    if py.exists():
        log(f"venv present: {venv_dir}")
    else:
        log(f"creating venv: {venv_dir}")
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
        if not venv_python(venv_dir).exists():
            log("venv creation did not produce a python executable")
            return False

    cli = find_cli(venv_dir)
    if cli is None:
        py = venv_python(venv_dir)
        log(f"installing CUDA torch ({TORCH_CU124_INDEX}) into the venv — this can take a while")
        subprocess.run(
            [str(py), "-m", "pip", "install", "torch", "torchaudio",
             "--index-url", TORCH_CU124_INDEX],
            check=True,
        )
        log(f"installing {AUDIO_SEPARATOR_SPEC} into the venv")
        subprocess.run([str(py), "-m", "pip", "install", AUDIO_SEPARATOR_SPEC], check=True)
        cli = find_cli(venv_dir)
        if cli is None:
            log("install finished but the audio-separator CLI was not found in the venv")
            return False
    log(f"audio-separator CLI: {cli}")

    # LAST: make sure the CUDA torch build wins over audio-separator's CPU downgrade.
    _ensure_cuda_torch(venv_dir, log)

    if have_ffmpeg():
        log("ffmpeg: ok")
    else:
        log("ffmpeg: NOT on PATH — install it (apt/brew/choco install ffmpeg)")
        return False
    return True


def preflight(cfg: "SeparationCfg") -> Path:
    """Verify the isolated env is usable; return the CLI path.

    Raises :class:`SotaEnvError` (fail-soft at the orchestrator) — never
    creates the venv and never installs anything.
    """
    if not have_ffmpeg():
        raise SotaEnvError("ffmpeg not on PATH (install ffmpeg, e.g. `apt install ffmpeg`)")
    venv_dir = resolve_venv_dir(cfg.uvr_venv)
    cli = find_cli(venv_dir)
    if cli is None:
        raise SotaEnvError(
            f"audio-separator venv not ready at {venv_dir} (run `stemforge setup-sota`)"
        )
    return cli


# --------------------------------------------------------------------------- #
# Subprocess-backed engine
# --------------------------------------------------------------------------- #
class UvrEngine:
    """A preflighted isolated-venv CLI runner plus its scratch output directory.

    Holds no model in-process — the checkpoint lives inside the subprocess —
    but caching an instance across a batch still skips repeated preflights and
    scratch-dir churn. Mirrors the ``(stems, engine)`` contract of the demucs
    backend.
    """

    def __init__(self, cli: Path, model_filename: str, model_file_dir: str,
                 use_autocast: bool = True, work_dir: PathLike | None = None):
        self.cli = cli
        self.model_filename = model_filename
        self.model_file_dir = model_file_dir
        self.use_autocast = use_autocast
        owns_work_dir = work_dir is None
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="stemforge-uvr-"))
        ensure_dir(self.work_dir)
        if owns_work_dir:  # remove the scratch dir when the engine is collected
            self._cleanup = weakref.finalize(self, shutil.rmtree, str(self.work_dir), True)

    @classmethod
    def from_cfg(cls, cfg: Any, model_filename: str | None = None,
                 work_dir: PathLike | None = None) -> "UvrEngine":
        """Build a preflighted engine from any cfg exposing the ``uvr_*`` fields
        (``SeparationCfg`` or ``DrumSplitCfg``). ``model_filename`` overrides
        ``cfg.uvr_model`` (used when a drum model is auto-discovered)."""
        return cls(
            cli=preflight(cfg),
            model_filename=model_filename or cfg.uvr_model,
            model_file_dir=_resolve_model_dir(cfg.uvr_model_dir),
            use_autocast=getattr(cfg, "uvr_use_autocast", True),
            work_dir=work_dir,
        )

    def run(self, path: PathLike) -> list[Path]:
        """Run the venv CLI on an audio file; return the produced stem files."""
        before = {p for p in self.work_dir.iterdir()}
        cmd = [
            str(self.cli), str(path),
            "--model_filename", self.model_filename,
            "--output_dir", str(self.work_dir),
            "--output_format", "WAV",
            "--model_file_dir", self.model_file_dir,
        ]
        if self.use_autocast:
            cmd.append("--use_autocast")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]
            raise RuntimeError(
                f"audio-separator exited with {proc.returncode}: " + " | ".join(tail)
            )
        return sorted(p for p in self.work_dir.iterdir() if p not in before and p.is_file())

    def separate_file(self, path: PathLike) -> dict[str, Path]:
        """Run the venv CLI on an audio file; return canonical stem name -> path."""
        return {canonical_stem_name(p.name): p for p in self.run(path)}


def separate(
    audio: AudioTensor,
    cfg: "SeparationCfg",
    device: str = "cuda",
    engine: UvrEngine | None = None,
) -> tuple[dict[str, AudioTensor], UvrEngine]:
    """Separate ``audio`` with a UVR-family model in the isolated venv.

    Returns ``(stems, engine)`` — the engine is returned so the caller can cache
    it across a batch, mirroring :func:`stemforge.separate.separate`.
    ``device`` is accepted for backend-signature parity; audio-separator picks
    CUDA automatically inside its own venv.
    """
    if engine is None or engine.model_filename != cfg.uvr_model:
        engine = UvrEngine.from_cfg(cfg)

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


def separate_drums(
    audio: AudioTensor,
    cfg: Any,
    engine: UvrEngine | None = None,
) -> tuple[dict[str, AudioTensor], str, UvrEngine]:
    """Tear a drum loop / drums stem into hit parts, in the isolated venv.

    ``cfg`` is a ``DrumSplitCfg`` (exposes the ``uvr_*`` fields). The drum model
    is ``cfg.uvr_model`` when set, otherwise auto-discovered via
    ``audio-separator --list_filter=drums``. Returns ``(parts, model, engine)``
    with parts keyed by canonical name (kick/snare/toms/hihat/ride/crash/…).

    Raises :class:`SotaEnvError` (fail-soft upstream) when ffmpeg / the venv /
    a drum model is missing; the main env is never touched.
    """
    cli = preflight(cfg)  # ffmpeg + venv, else SotaEnvError
    model_dir = _resolve_model_dir(cfg.uvr_model_dir)
    model = cfg.uvr_model or discover_drum_model(cli, model_dir)
    if not model:
        raise SotaEnvError(
            "no drum-separation model available in the venv — set "
            "`drums.split.uvr_model` to a DrumSep checkpoint, or install one "
            "(the model auto-downloads on first use once named)"
        )

    if engine is None or engine.model_filename != model:
        engine = UvrEngine.from_cfg(cfg, model_filename=model)

    in_path = engine.work_dir / f"{uuid.uuid4().hex[:12]}.wav"
    save_audio(in_path, audio)
    produced: list[Path] = []
    try:
        produced = engine.run(in_path)
        parts = {canonical_drum_part(p.name): load_audio(p) for p in produced}
    finally:
        in_path.unlink(missing_ok=True)
        for p in produced:
            p.unlink(missing_ok=True)
    return parts, model, engine
