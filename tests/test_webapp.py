"""FastAPI web backend — routes exercised with the Pipeline fully mocked.

GPU-free and hermetic: no real separation, no model downloads. fastapi/httpx are
required (skip if absent); the background job runs the FakePipeline synchronously
fast, so a bounded poll resolves immediately in CI.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("multipart")

from fastapi.testclient import TestClient  # noqa: E402

from stemforge import webapp  # noqa: E402
from stemforge.io_utils import slugify  # noqa: E402


class FakePipeline:
    """Stand-in for the real Pipeline: writes fake outputs per enabled stage."""

    def __init__(self, cfg, model_cache=None, on_stage=None):
        self.cfg = cfg
        self.on_stage = on_stage

    def resolve_device(self):
        return "cpu"

    def _fire(self, key):
        if self.on_stage:
            self.on_stage(key, True)

    def run(self, path):
        cfg = self.cfg
        out = Path(cfg.output.root) / slugify(Path(path).stem)
        (out).mkdir(parents=True, exist_ok=True)
        manifest = {"input": {"filename": Path(path).name}, "analysis": {}}

        if cfg.analysis.enabled:
            self._fire("analysis")
            manifest["analysis"] = {"source_bpm": 120.0}
        if cfg.separation.enabled:
            self._fire("separation")
            (out / "stems").mkdir(exist_ok=True)
            for s in ("vocals", "drums"):
                (out / "stems" / f"{s}.wav").write_bytes(b"RIFFfake")
        if cfg.drums.split.enabled:
            self._fire("drum_split")
            (out / "drums").mkdir(exist_ok=True)
            for s in ("kick", "snare"):
                (out / "drums" / f"{s}.wav").write_bytes(b"RIFFfake")
        if cfg.drums.midi.enabled:
            self._fire("drum_midi")
            (out / "drums").mkdir(exist_ok=True)
            (out / "drums" / "drums.mid").write_bytes(b"MThd")
        if cfg.midi.enabled:
            self._fire("midi")
            (out / "midi").mkdir(exist_ok=True)
            (out / "midi" / "bass.mid").write_bytes(b"MThd")

        (out / "manifest.json").write_text("{}")
        return manifest


@pytest.fixture
def client(monkeypatch, tmp_path):
    from stemforge import orchestrator

    real_load = orchestrator.load_config

    def fake_load(overrides=None):
        cfg = real_load(overrides=overrides)
        cfg.output.root = str(tmp_path / "out")   # keep all output under tmp
        return cfg

    monkeypatch.setattr(webapp, "load_config", fake_load)
    monkeypatch.setattr(webapp, "Pipeline", FakePipeline)
    monkeypatch.setattr(webapp, "_UPLOAD_DIR", tmp_path / "uploads")
    webapp._JOBS.clear()
    return TestClient(webapp.create_app())


def _poll(client, job_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/job/{job_id}").json()
        if job["status"] in {"done", "error"}:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish: {job}")


def _wav_bytes() -> bytes:
    # a minimal file; the mocked pipeline never decodes it
    return b"RIFF" + b"\x00" * 40


# --------------------------------------------------------------------------- #
def test_index_and_static_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "STEMFORGE" in r.text.upper()
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/styles.css").status_code == 200


def test_health_reports_device_and_status(client):
    h = client.get("/api/health").json()
    assert h["device"] in {"cpu", "cuda"}
    assert set(h) >= {"device", "cuda", "ffmpeg", "uvr_venv", "drumsep_model", "presets"}
    assert h["presets"] == ["fast", "best", "sota", "max"]


def test_extract_produces_stem_cards(client):
    r = client.post("/api/extract", files={"file": ("mix.wav", _wav_bytes(), "audio/wav")},
                    data={"preset": "sota"})
    job = _poll(client, r.json()["job_id"])
    assert job["status"] == "done"
    result = job["result"]
    names = {c["name"] for c in result["cards"]}
    assert {"vocals", "drums"} <= names
    assert all(c["url"].startswith("/api/file?path=") for c in result["cards"])
    assert result["out_dir"].endswith(slugify("mix"))

    # each card is downloadable through the guarded file route (url is pre-encoded)
    assert client.get(result["cards"][0]["url"]).status_code == 200


def test_drum_teardown_cards_include_midi(client):
    r = client.post("/api/drum-teardown", files={"file": ("loop.wav", _wav_bytes(), "audio/wav")})
    job = _poll(client, r.json()["job_id"])
    result = job["result"]
    kinds = {c["kind"] for c in result["cards"]}
    names = {c["name"] for c in result["cards"]}
    assert "midi" in kinds and "audio" in kinds
    assert {"kick", "snare", "drums"} <= names  # hit stems + the drum midi ("drums")
    assert result["bpm"] == 120.0


def test_melodic_midi_flags_and_output(client):
    r = client.post("/api/melodic-midi", files={"file": ("bass.wav", _wav_bytes(), "audio/wav")},
                    data={"monophonic": "true", "quantize": "true"})
    job = _poll(client, r.json()["job_id"])
    assert job["status"] == "done"
    assert any(c["kind"] == "midi" for c in job["result"]["cards"])


def test_full_teardown_bundles_everything(client):
    r = client.post("/api/full-teardown", files={"file": ("song.wav", _wav_bytes(), "audio/wav")},
                    data={"preset": "max"})
    result = _poll(client, r.json()["job_id"])["result"]
    groups = {c["group"] for c in result["cards"]}
    assert {"stems", "drums", "midi"} <= groups
    assert result["bpm"] == 120.0


def test_download_all_zips_output(client):
    r = client.post("/api/extract", files={"file": ("mix.wav", _wav_bytes(), "audio/wav")})
    out_dir = _poll(client, r.json()["job_id"])["result"]["out_dir"]
    z = client.get("/api/download-all", params={"dir": out_dir})
    assert z.status_code == 200
    assert z.headers["content-type"] == "application/zip"
    assert z.content[:2] == b"PK"  # zip magic


def test_job_progress_advances_via_on_stage(client):
    r = client.post("/api/full-teardown", files={"file": ("song.wav", _wav_bytes(), "audio/wav")})
    job = _poll(client, r.json()["job_id"])
    assert job["progress"] == 1.0 and job["message"] == "Done"


def test_error_in_pipeline_becomes_job_error(client, monkeypatch):
    class Boom(FakePipeline):
        def run(self, path):
            raise RuntimeError("gpu melted")

    monkeypatch.setattr(webapp, "Pipeline", Boom)
    r = client.post("/api/extract", files={"file": ("mix.wav", _wav_bytes(), "audio/wav")})
    job = _poll(client, r.json()["job_id"])
    assert job["status"] == "error" and "gpu melted" in job["error"]


def test_file_route_rejects_paths_outside_output_root(client, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("nope")
    assert client.get("/api/file", params={"path": str(secret)}).status_code == 404


def test_unknown_job_is_404(client):
    # the SPA poller relies on this 404 to stop polling a dead job after a restart
    assert client.get("/api/job/deadbeef").status_code == 404


def test_favicon_returns_204_not_404(client):
    # a favicon route silences the per-load 404 in the local server console
    assert client.get("/favicon.ico").status_code == 204


def test_job_status_serializes_numpy_scalars(client):
    """A producer may leave a numpy scalar in a job; FastAPI's pydantic serializer
    has no default hook and 500s on np.bool_. GET /api/job/{id} must round-trip it
    to plain, valid JSON (the frontend JSON.parse depends on this)."""
    import numpy as np

    jid = "npjob00beef1"
    webapp._JOBS[jid] = {
        "status": "done", "progress": 1.0, "stage": None, "message": "Done",
        "error": None, "outcome": "success", "finished_at": None,
        "result": {"flag": np.bool_(True), "score": np.float64(0.5),
                   "n": np.int64(3), "arr": np.arange(2)},
    }
    try:
        r = client.get(f"/api/job/{jid}")
        assert r.status_code == 200            # was 500 (PydanticSerializationError)
        body = r.json()                        # valid JSON
        assert body["result"]["flag"] is True
        assert body["result"]["score"] == 0.5
        assert body["result"]["n"] == 3
        assert body["result"]["arr"] == [0, 1]
    finally:
        webapp._JOBS.pop(jid, None)


# --------------------------------------------------------------------------- #
# Match BPM (whole-file) — stretch.match_bpm_file mocked
# --------------------------------------------------------------------------- #
def test_match_bpm_route(client, monkeypatch, tmp_path):
    seen: dict = {}

    def fake_match(input_path, target_bpm, out_dir, source_bpm=None,
                   engine="rubberband", detect_engine="beat_this", device="auto", out_name=None):
        seen.update(target_bpm=target_bpm, source_bpm=source_bpm, engine=engine)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "song_140bpm.wav").write_bytes(b"RIFFfake")
        return {"engine": engine, "detect_engine": detect_engine, "source_bpm": 120.0,
                "source_bpm_detected": 0.0, "source_bpm_overridden": source_bpm is not None,
                "target_bpm": target_bpm, "ratio": round(target_bpm / 120.0, 4),
                "input": str(input_path), "output": str(out / "song_140bpm.wav")}

    import stemforge.stretch as stretch_mod
    monkeypatch.setattr(stretch_mod, "match_bpm_file", fake_match)

    r = client.post("/api/match-bpm", files={"file": ("song.wav", _wav_bytes(), "audio/wav")},
                    data={"target_bpm": "140", "source_bpm": "120", "engine": "librosa"})
    job = _poll(client, r.json()["job_id"])
    assert job["status"] == "done"
    assert seen["target_bpm"] == 140.0 and seen["source_bpm"] == 120.0 and seen["engine"] == "librosa"

    result = job["result"]
    assert result["bpm"] == 140.0 and result["source_bpm"] == 120.0
    assert [c["name"] for c in result["cards"]] == ["song_140bpm"]
    assert result["cards"][0]["group"] == "matched"
    assert client.get(result["cards"][0]["url"]).status_code == 200


def test_match_bpm_skip_is_done_with_note(client, monkeypatch):
    import stemforge.stretch as stretch_mod
    monkeypatch.setattr(stretch_mod, "match_bpm_file",
                        lambda *a, **k: {"skipped": "no detectable BPM; pass source_bpm to override"})
    r = client.post("/api/match-bpm", files={"file": ("song.wav", _wav_bytes(), "audio/wav")},
                    data={"target_bpm": "140"})
    job = _poll(client, r.json()["job_id"])
    assert job["status"] == "done"
    assert job["result"]["cards"] == []
    assert "source_bpm" in job["message"]


def test_detect_bpm_route(client, monkeypatch):
    import stemforge.stretch as stretch_mod
    monkeypatch.setattr(stretch_mod, "detect_bpm", lambda audio, **k: 87.0)
    # decode is invoked before detect_bpm; stub it so no real audio is needed
    import stemforge.ingest as ingest_mod
    monkeypatch.setattr(ingest_mod, "decode", lambda *a, **k: object())

    r = client.post("/api/detect-bpm", files={"file": ("song.wav", _wav_bytes(), "audio/wav")},
                    data={"detect_engine": "librosa"})
    assert r.status_code == 200
    body = r.json()
    assert body == {"bpm": 87.0, "half": 43.5, "double": 174.0}


def test_detect_bpm_route_failsoft_zero(client, monkeypatch):
    import stemforge.ingest as ingest_mod

    def boom(*a, **k):
        raise RuntimeError("bad audio")

    monkeypatch.setattr(ingest_mod, "decode", boom)
    r = client.post("/api/detect-bpm", files={"file": ("x.wav", _wav_bytes(), "audio/wav")})
    assert r.status_code == 200 and r.json() == {"bpm": 0.0, "half": 0.0, "double": 0.0}


# --------------------------------------------------------------------------- #
# C7 (H1) — server hardening
# --------------------------------------------------------------------------- #
def test_open_folder_outside_allowed_roots_rejected(client):
    # R2/AC2 — a path outside the output/upload roots is refused (was 200 before).
    r = client.post("/api/open-folder", json={"path": "/etc"})
    assert r.status_code == 404
    r2 = client.post("/api/open-folder", json={"path": "/definitely/not/a/dir"})
    assert r2.status_code == 404


def test_open_folder_allowed_dir_ok(client, monkeypatch, tmp_path):
    monkeypatch.setattr(webapp, "reveal_folder", lambda p: f"opened {p}")
    allowed = tmp_path / "out" / "song"
    allowed.mkdir(parents=True)
    r = client.post("/api/open-folder", json={"path": str(allowed)})
    assert r.status_code == 200 and "opened" in r.json()["message"]


def test_oversize_upload_rejected_413(client, monkeypatch):
    # R3/AC3 — oversize upload rejected (streamed; capped before a full read).
    monkeypatch.setattr(webapp, "_max_upload_bytes", lambda: 10)
    r = client.post("/api/extract", files={"file": ("big.wav", b"x" * 4096, "audio/wav")})
    assert r.status_code == 413


def test_download_all_still_streams(client):
    r = client.post("/api/extract", files={"file": ("mix.wav", _wav_bytes(), "audio/wav")})
    out_dir = _poll(client, r.json()["job_id"])["result"]["out_dir"]
    z = client.get("/api/download-all", params={"dir": out_dir})
    assert z.status_code == 200 and z.content[:2] == b"PK"  # streamed from a temp file


# --------------------------------------------------------------------------- #
# C5 (H2v2) AC1 — jobs enqueue on a single-worker executor and are correctly
# attributed with no cross-contamination (serialized, non-interleaved).
# --------------------------------------------------------------------------- #
def test_concurrent_jobs_attributed_and_serialized(client):
    r1 = client.post("/api/extract", files={"file": ("alpha.wav", _wav_bytes(), "audio/wav")},
                     data={"preset": "best"})
    r2 = client.post("/api/extract", files={"file": ("beta.wav", _wav_bytes(), "audio/wav")},
                     data={"preset": "best"})
    j1 = _poll(client, r1.json()["job_id"])
    j2 = _poll(client, r2.json()["job_id"])

    assert j1["status"] == "done" and j2["status"] == "done"
    assert j1["result"]["out_dir"].endswith(slugify("alpha"))
    assert j2["result"]["out_dir"].endswith(slugify("beta"))
    # each job's cards live under its OWN output dir — no cross-attribution
    assert j1["result"]["cards"] and all("alpha" in c["url"] for c in j1["result"]["cards"])
    assert j2["result"]["cards"] and all("beta" in c["url"] for c in j2["result"]["cards"])


def test_uploads_do_not_spawn_free_threads(client, monkeypatch):
    """R1: uploads submit to the bounded executor; no raw Thread() is spawned."""
    import stemforge.webapp as webapp_mod

    def boom(*a, **k):
        raise AssertionError("uploads must enqueue on the executor, not spawn a Thread")

    monkeypatch.setattr(webapp_mod.threading, "Thread", boom)
    r = client.post("/api/extract", files={"file": ("mix.wav", _wav_bytes(), "audio/wav")})
    job = _poll(client, r.json()["job_id"])
    assert job["status"] == "done"
