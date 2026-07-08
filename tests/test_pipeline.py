"""End-to-end smoke test that needs NO ML deps: ingest + orchestrator + manifest."""

from __future__ import annotations

import soundfile as sf

from stemforge.orchestrator import Pipeline, load_config


def test_resolve_device_without_torch():
    p = Pipeline(load_config())
    assert p.resolve_device() in {"cpu", "cuda"}


def test_pipeline_runs_with_stages_disabled(tmp_path, sine):
    samples, sr = sine
    wav = tmp_path / "in.wav"
    sf.write(str(wav), samples.T, sr, subtype="FLOAT")

    cfg = load_config(overrides={
        "ingest.normalize": "none",
        "analysis.enabled": False,
        "separation.enabled": False,
        "output.root": str(tmp_path / "out"),
    })
    m = Pipeline(cfg).run(wav)

    assert m["separation"]["skipped"] == "disabled"
    assert m["analysis"]["skipped"] == "disabled"
    assert m["input"]["sample_rate"] == sr
    assert m["input"]["sha256"] and len(m["input"]["sha256"]) == 64
    assert "receipt" in m
    assert (tmp_path / "out").exists()
    assert (tmp_path / "out" / "in" / "manifest.json").is_file()


def test_stretch_needs_source_bpm(tmp_path, sine):
    """Stretch enabled but analysis off -> recorded as skipped, run still succeeds."""
    samples, sr = sine
    wav = tmp_path / "in.wav"
    sf.write(str(wav), samples.T, sr, subtype="FLOAT")

    cfg = load_config(overrides={
        "ingest.normalize": "none",
        "analysis.enabled": False,
        "separation.enabled": False,
        "stretch.enabled": True,
        "stretch.target_bpm": 120,
        "output.root": str(tmp_path / "out"),
    })
    m = Pipeline(cfg).run(wav)
    assert "skipped" in m["stretch"]
