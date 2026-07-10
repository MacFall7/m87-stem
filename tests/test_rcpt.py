"""C4 (EC-SF-RCPT) — artifact integrity + honest outcome semantics.

GPU-free, no ML deps: the pipeline runs with separation/analysis disabled or with
an unknown backend (fail-soft error), and stretch-engine truth is checked directly.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf
from typer.testing import CliRunner

from stemforge import cli
from stemforge import stretch as stretch_mod
from stemforge.io_utils import AudioTensor, receipt, save_audio
from stemforge.orchestrator import Pipeline, StretchCfg, _compute_outcome, load_config

runner = CliRunner()


def _wav(tmp_path, name="in.wav", dur=0.5):
    sr = 44100
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    y = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    p = tmp_path / name
    sf.write(str(p), np.stack([y, y]).T, sr, subtype="FLOAT")
    return p


# --------------------------------------------------------------------------- #
# AC1 — receipt keys are run-root-relative POSIX paths; same basename in
# different subdirs -> distinct keys + distinct hashes (collision impossible).
# --------------------------------------------------------------------------- #
def test_receipt_distinct_keys_for_same_basename(tmp_path):
    root = tmp_path / "run"
    (root / "stems").mkdir(parents=True)
    (root / "drums").mkdir(parents=True)
    p1 = save_audio(root / "stems" / "other.wav", AudioTensor(np.zeros((1, 16), np.float32), 44100))
    p2 = save_audio(root / "drums" / "other.wav", AudioTensor(np.ones((1, 16), np.float32), 44100))
    r = receipt([p1, p2], root)
    assert set(r) == {"stems/other.wav", "drums/other.wav"}   # distinct keys
    assert r["stems/other.wav"] != r["drums/other.wav"]       # distinct hashes


# --------------------------------------------------------------------------- #
# AC2 — a stale file in a reused output dir does NOT enter the new manifest,
# and is gone from the published bundle (run staged fresh, then finalized).
# --------------------------------------------------------------------------- #
def test_stale_file_excluded_from_manifest(tmp_path):
    wav = _wav(tmp_path)
    out_root = tmp_path / "out"
    song_dir = out_root / "in"
    song_dir.mkdir(parents=True)
    (song_dir / "STALE.wav").write_bytes(b"RIFFstale")  # prior-run leftover

    cfg = load_config(overrides={
        "ingest.normalize": "none", "analysis.enabled": False,
        "separation.enabled": False, "output.root": str(out_root),
    })
    m = Pipeline(cfg).run(wav)

    assert all("STALE" not in k for k in m["receipt"])       # never certified
    assert not (out_root / "in" / "STALE.wav").exists()      # finalize replaced the dir
    assert (out_root / "in" / "manifest.json").is_file()     # fresh bundle published


# --------------------------------------------------------------------------- #
# AC3 — required-stage failure -> outcome "failed" (+ CLI exit != 0);
# a non-required-stage skip -> outcome "partial".
# --------------------------------------------------------------------------- #
def test_outcome_failed_on_required_stage_error(tmp_path):
    wav = _wav(tmp_path)
    cfg = load_config(overrides={
        "ingest.normalize": "none", "analysis.enabled": False,
        "separation.enabled": True, "separation.backend": "spleeter",  # unknown -> error
        "output.root": str(tmp_path / "out"),
    })
    m = Pipeline(cfg, required_stages={"separation"}).run(wav)
    assert "error" in m["separation"]
    assert m["stages"]["separation"] == "error"
    assert m["outcome"] == "failed"


def test_outcome_partial_on_nonrequired_skip(tmp_path):
    wav = _wav(tmp_path)
    cfg = load_config(overrides={
        "ingest.normalize": "none", "analysis.enabled": False,
        "separation.enabled": False,
        "stretch.enabled": True, "stretch.target_bpm": 120,  # skips: no source bpm
        "output.root": str(tmp_path / "out"),
    })
    m = Pipeline(cfg, required_stages=set()).run(wav)   # nothing required
    assert "skipped" in m["stretch"]
    assert m["stages"]["stretch"] == "skipped"
    assert m["outcome"] == "partial"


def test_compute_outcome_matrix():
    assert _compute_outcome({"separation": "success"}, {"separation"}) == "success"
    assert _compute_outcome({"separation": "error"}, {"separation"}) == "failed"
    assert _compute_outcome({"separation": "skipped"}, {"separation"}) == "failed"
    assert _compute_outcome({"midi": "skipped"}, set()) == "partial"
    # 'disabled' never counts against the outcome
    assert _compute_outcome({"separation": "success", "midi": "disabled"}, {"separation"}) == "success"


def test_cli_exit_nonzero_on_required_failure(tmp_path, sine_wav):
    result = runner.invoke(cli.app, [
        "run", str(sine_wav),
        "--set", "separation.backend=spleeter",
        "--set", "analysis.enabled=false",
        "--set", "ingest.normalize=none",
        "-o", str(tmp_path / "out"),
    ])
    assert result.exit_code != 0    # required separation errored -> failed -> exit 1


# --------------------------------------------------------------------------- #
# AC4 — manifest records the ACTUAL stretch engine after a fallback (the probe
# showed the old code recorded the REQUESTED engine and never labeled librosa).
# stretch.py in scope per amendment C4A1.
# --------------------------------------------------------------------------- #
def test_stretch_records_actual_engine_after_fallback(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("engine unavailable")

    monkeypatch.setattr(stretch_mod, "_rubberband", boom)
    monkeypatch.setattr(stretch_mod, "_signalsmith", boom)

    p = save_audio(tmp_path / "bass.wav", AudioTensor(np.zeros((1, 44100), np.float32), 44100))
    cfg = StretchCfg(enabled=True, target_bpm=100.0, engine="rubberband")

    class _Grid:
        source_bpm = 120.0

    res = stretch_mod.stretch_stems({"bass": p}, cfg, tmp_path / "out", beat_grid=_Grid())
    assert res["engine"] == "librosa"            # actual engine used (after fallback)
    assert res["engine_requested"] == "rubberband"


def test_match_bpm_reports_actual_engine(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no engine")

    monkeypatch.setattr(stretch_mod, "_rubberband", boom)
    monkeypatch.setattr(stretch_mod, "_signalsmith", boom)
    wav = _wav(tmp_path)
    res = stretch_mod.match_bpm_file(wav, target_bpm=140, out_dir=tmp_path / "m",
                                     source_bpm=120, engine="rubberband")
    assert res["engine"] == "librosa"            # actual
    assert res["engine_requested"] == "rubberband"


# --------------------------------------------------------------------------- #
# R5/R6 — actual values recorded: seed applied + per-stage status present.
# --------------------------------------------------------------------------- #
def test_seed_and_stage_status_recorded(tmp_path):
    wav = _wav(tmp_path)
    cfg = load_config(overrides={
        "ingest.normalize": "none", "analysis.enabled": False,
        "separation.enabled": False, "seed": 1234, "output.root": str(tmp_path / "out"),
    })
    m = Pipeline(cfg).run(wav)
    assert m["seed"] == 1234
    assert m["stages"]["separation"] == "disabled"
    assert m["outcome"] in {"success", "partial", "failed"}
