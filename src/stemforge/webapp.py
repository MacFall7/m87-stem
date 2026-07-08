"""§5.9 webapp — the M87 workstation as a bespoke FastAPI app (no Gradio).

A tiny FastAPI backend runs the existing :class:`~stemforge.orchestrator.Pipeline`
and serves a static single-page front-end (``web/``). Four workflow endpoints
kick off a background job; the SPA polls ``/api/job/{id}`` for stage progress,
then renders result cards with per-stem/per-hit waveforms.

Routes
------
- ``GET  /``                     → the SPA (``web/index.html``)
- ``GET  /static/*``             → SPA assets
- ``GET  /api/health``           → device/GPU + venv/model status (status chip)
- ``POST /api/extract``          → mix → stems
- ``POST /api/drum-teardown``    → drum loop/stem → hit stems + drum MIDI
- ``POST /api/melodic-midi``     → stem/mix → .mid  (monophonic, quantize flags)
- ``POST /api/full-teardown``    → everything, grid-aligned
- ``GET  /api/job/{job_id}``     → poll job progress/result
- ``GET  /api/progress/{job_id}``→ same, as an SSE stream
- ``GET  /api/file``             → download one output (path-guarded)
- ``GET  /api/download-all``     → zip an output folder
- ``POST /api/open-folder``      → reveal an output folder locally

fastapi/uvicorn are imported lazily (inside :func:`create_app` / :func:`launch`),
so ``import stemforge`` stays dependency-light; the Pipeline's heavy work stays
lazy and fail-soft exactly as in the CLI.

NOTE: no ``from __future__ import annotations`` here — FastAPI resolves route
parameter annotations (``UploadFile``/``Form``) at request time, and stringized
annotations break that (same rule as ``cli.py``).
"""

import io
import json
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .orchestrator import Pipeline, load_config, preset_names

WEB_DIR = Path(__file__).resolve().parent / "web"

# Stage → (progress fraction, human label) for the poll/SSE payload.
_STAGE_PROGRESS = {
    "analysis": (0.15, "Analyzing tempo & beat grid"),
    "separation": (0.40, "Separating stems"),
    "midi": (0.60, "Transcribing MIDI"),
    "drum_split": (0.75, "Tearing down the kit"),
    "drum_midi": (0.88, "Building drum MIDI"),
    "stretch": (0.95, "Time-stretching"),
}

# In-memory job store: job_id -> {status, progress, stage, message, result, error}
_JOBS: dict[str, dict[str, Any]] = {}
_MODEL_CACHE: dict[str, Any] = {}
_UPLOAD_DIR = WEB_DIR.parent / ".uploads"  # under the package dir, transient

_MELODIC_STEMS = ["bass", "other", "guitar", "piano", "vocals"]


# --------------------------------------------------------------------------- #
# Workflow → config overrides (mirror the CLI/pipeline semantics)
# --------------------------------------------------------------------------- #
def _overrides(workflow: str, params: dict[str, Any]) -> dict[str, Any]:
    preset = str(params.get("preset", "best")).lower()
    device = params.get("device", "auto")
    if workflow == "extract":
        return {"separation.preset": preset, "device": device, "analysis.enabled": False}
    if workflow == "drum-teardown":
        return {
            "device": device, "analysis.enabled": True,
            "separation.enabled": False,
            "drums.split.enabled": True, "drums.split.from_input": True,
            "drums.midi.enabled": True,
        }
    if workflow == "melodic-midi":
        quantize = bool(params.get("quantize"))
        return {
            "separation.preset": "best", "device": device,
            "analysis.enabled": quantize,
            "midi.enabled": True, "midi.quantize_to_grid": quantize,
            "midi.monophonic_stems": _MELODIC_STEMS if params.get("monophonic") else [],
        }
    if workflow == "full-teardown":
        return {
            "separation.preset": preset, "device": device,
            "analysis.enabled": True, "midi.enabled": True,
            "drums.split.enabled": True, "drums.midi.enabled": True,
        }
    raise ValueError(f"unknown workflow {workflow!r}")


def _out_dir_for(cfg, manifest: dict) -> Path:
    from .io_utils import slugify

    return Path(cfg.output.root) / slugify(Path(manifest["input"]["filename"]).stem)


def _cards(out_dir: Path) -> list[dict[str, Any]]:
    """Build result cards for every produced audio/MIDI file (with download URLs)."""
    cards: list[dict[str, Any]] = []
    order = {"stems": 0, "drums": 1, "midi": 2, "stretched": 3}
    files = [p for p in out_dir.rglob("*") if p.is_file() and p.name != "manifest.json"]
    for p in sorted(files, key=lambda p: (order.get(p.parent.name, 9), p.name)):
        kind = "audio" if p.suffix.lower() in {".wav", ".flac", ".mp3"} else (
            "midi" if p.suffix.lower() in {".mid", ".midi"} else "file")
        cards.append({
            "name": p.stem,
            "group": p.parent.name,
            "kind": kind,
            "filename": p.name,
            "url": "/api/file?path=" + quote(str(p.resolve())),
        })
    return cards


def run_job(job_id: str, workflow: str, input_path: Path, params: dict[str, Any]) -> None:
    """Execute one workflow, updating the job store. Never raises."""
    job = _JOBS[job_id]
    try:
        cfg = load_config(overrides=_overrides(workflow, params))

        def on_stage(key: str, enabled: bool) -> None:
            if enabled and key in _STAGE_PROGRESS:
                frac, label = _STAGE_PROGRESS[key]
                job.update(progress=frac, stage=key, message=label)

        job.update(status="running", progress=0.05, message="Ingesting audio")
        pipe = Pipeline(cfg, model_cache=_MODEL_CACHE, on_stage=on_stage)
        manifest = pipe.run(input_path)

        out_dir = _out_dir_for(cfg, manifest)
        analysis = manifest.get("analysis", {})
        job.update(
            status="done", progress=1.0, message="Done",
            result={
                "workflow": workflow,
                "out_dir": str(out_dir.resolve()),
                "bpm": analysis.get("source_bpm"),
                "key": analysis.get("key"),
                "cards": _cards(out_dir),
                "manifest": manifest,
            },
        )
    except Exception as e:  # noqa: BLE001 - surface as a job error, don't crash the server
        job.update(status="error", progress=1.0, message=str(e), error=str(e))
    finally:
        try:
            input_path.unlink(missing_ok=True)
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# Path guarding for file downloads
# --------------------------------------------------------------------------- #
def _allowed_roots() -> list[Path]:
    cfg = load_config()
    return [Path(cfg.output.root).resolve(), _UPLOAD_DIR.resolve()]


def _safe(path_str: str) -> Path | None:
    try:
        p = Path(path_str).resolve()
    except (OSError, RuntimeError):
        return None
    if not p.is_file():
        return None
    return p if any(r in p.parents for r in _allowed_roots()) else None


# --------------------------------------------------------------------------- #
# App factory
# --------------------------------------------------------------------------- #
def create_app():
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="M87 · StemForge", docs_url=None, redoc_url=None)
    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    if (WEB_DIR / "assets").is_dir():
        app.mount("/static", StaticFiles(directory=str(WEB_DIR / "assets")), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (WEB_DIR / "index.html").read_text(encoding="utf-8")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return health_status()

    async def _accept(workflow: str, file: UploadFile, params: dict[str, Any]) -> dict[str, str]:
        job_id = uuid.uuid4().hex[:12]
        dest = _UPLOAD_DIR / f"{job_id}_{Path(file.filename or 'input').name}"
        dest.write_bytes(await file.read())
        _JOBS[job_id] = {"status": "queued", "progress": 0.0, "stage": None,
                         "message": "Queued", "result": None, "error": None}
        threading.Thread(target=run_job, args=(job_id, workflow, dest, params),
                         daemon=True).start()
        return {"job_id": job_id}

    @app.post("/api/extract")
    async def extract(file: UploadFile = File(...), preset: str = Form("best"),
                      device: str = Form("auto")):
        return await _accept("extract", file, {"preset": preset, "device": device})

    @app.post("/api/drum-teardown")
    async def drum_teardown(file: UploadFile = File(...), device: str = Form("auto")):
        return await _accept("drum-teardown", file, {"device": device})

    @app.post("/api/melodic-midi")
    async def melodic_midi(file: UploadFile = File(...), monophonic: bool = Form(False),
                           quantize: bool = Form(False), device: str = Form("auto")):
        return await _accept("melodic-midi", file,
                             {"monophonic": monophonic, "quantize": quantize, "device": device})

    @app.post("/api/full-teardown")
    async def full_teardown(file: UploadFile = File(...), preset: str = Form("best"),
                            device: str = Form("auto")):
        return await _accept("full-teardown", file, {"preset": preset, "device": device})

    @app.get("/api/job/{job_id}")
    def job_status(job_id: str) -> dict[str, Any]:
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job")
        return job

    @app.get("/api/progress/{job_id}")
    def progress(job_id: str):
        if job_id not in _JOBS:
            raise HTTPException(status_code=404, detail="unknown job")

        def stream():
            last = None
            while True:
                job = _JOBS.get(job_id, {})
                snap = json.dumps(job, default=str)
                if snap != last:
                    yield f"data: {snap}\n\n"
                    last = snap
                if job.get("status") in {"done", "error"}:
                    break
                time.sleep(0.3)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/api/file")
    def get_file(path: str):
        p = _safe(path)
        if p is None:
            raise HTTPException(status_code=404, detail="file not found or not allowed")
        return FileResponse(str(p), filename=p.name)

    @app.get("/api/download-all")
    def download_all(dir: str):
        d = Path(dir).resolve()
        if not d.is_dir() or not any(r == d or r in d.parents for r in _allowed_roots()):
            raise HTTPException(status_code=404, detail="folder not found or not allowed")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(d.rglob("*")):
                if f.is_file():
                    zf.write(f, f.relative_to(d))
        buf.seek(0)
        headers = {"Content-Disposition": f'attachment; filename="{d.name}.zip"'}
        return StreamingResponse(buf, media_type="application/zip", headers=headers)

    @app.post("/api/open-folder")
    def open_folder(payload: dict[str, str]):
        return {"message": reveal_folder(payload.get("path", ""))}

    return app


# --------------------------------------------------------------------------- #
# Health + folder reveal (module-level so they're unit-testable without FastAPI)
# --------------------------------------------------------------------------- #
def health_status() -> dict[str, Any]:
    import shutil

    cfg = load_config()
    device = Pipeline(cfg).resolve_device()
    gpu = None
    try:
        import torch

        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass

    from .separate_uvr import find_cli, resolve_venv_dir

    venv_ready = find_cli(resolve_venv_dir(cfg.separation.uvr_venv)) is not None
    drum_dir = Path(cfg.drums.split.inagoy_model_dir)
    if not drum_dir.is_absolute():
        drum_dir = Path(__file__).resolve().parents[2] / drum_dir
    drum_ready = drum_dir.is_dir() and any(drum_dir.glob("*.th"))

    return {
        "device": device,
        "gpu": gpu,
        "cuda": gpu is not None,
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "uvr_venv": venv_ready,
        "drumsep_model": bool(drum_ready),
        "presets": preset_names(),
        "version": _version(),
    }


def reveal_folder(folder: str) -> str:
    """Open `folder` in the local file manager; degrade to a message when headless."""
    if not folder or not Path(folder).is_dir():
        return "No output folder yet — run a workflow first."
    import subprocess
    import sys

    try:
        if sys.platform == "win32":
            import os

            os.startfile(folder)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", folder])
        else:
            subprocess.Popen(["xdg-open", folder])
        return f"Asked the OS to open {folder}"
    except Exception as e:  # noqa: BLE001
        return f"Couldn't open a file manager ({e}). Folder: {folder}"


def _version() -> str:
    from . import __version__

    return __version__


# --------------------------------------------------------------------------- #
# Launch (uvicorn) — used by `stemforge ui`
# --------------------------------------------------------------------------- #
def launch(host: str = "127.0.0.1", port: int = 7860, open_browser: bool = True) -> None:
    import uvicorn

    if open_browser:
        _open_when_up(host, port)
    uvicorn.run(create_app(), host=host, port=port, log_level="info")


def _open_when_up(host: str, port: int) -> None:
    """Open the browser shortly after the server starts (best-effort, non-blocking)."""
    import webbrowser

    url = f"http://{host}:{port}/"

    def _open() -> None:
        time.sleep(1.2)
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001
            pass

    threading.Thread(target=_open, daemon=True).start()
