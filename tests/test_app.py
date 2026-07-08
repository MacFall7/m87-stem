"""M87 workstation UI — builds GPU-free, CSS variables present, handlers wired.

The gradio import is skipped if gradio isn't installed; the pipeline is mocked so
no models load and nothing hits the network.
"""

from __future__ import annotations

import pytest

from stemforge import app


# --------------------------------------------------------------------------- #
# Theme / CSS tokens
# --------------------------------------------------------------------------- #
def test_css_exposes_m87_variables():
    for var in ("--m87-bg", "--m87-surface", "--m87-accent", "--m87-accent-2",
                "--m87-text", "--m87-mono"):
        assert var in app.M87_CSS, f"missing CSS variable {var}"


def test_preset_choices_from_source_of_truth():
    assert app.PRESET_CHOICES == ["Fast", "Best", "SOTA", "Max"]


# --------------------------------------------------------------------------- #
# Empty-input guards (no gradio update objects, no pipeline)
# --------------------------------------------------------------------------- #
def test_handlers_guard_empty_input(monkeypatch):
    # if any handler tried to run, this would explode -> proves the guard returns first
    monkeypatch.setattr(app, "_run", lambda *a, **k: pytest.fail("should not run"))
    assert "mix" in app.extract_stems(None, "Best", "auto")[0].lower()
    assert "drum" in app.drum_teardown(None, "auto")[0].lower()
    assert "stem" in app.melodic_midi(None, False, False, "auto")[0].lower()
    assert "track" in app.full_teardown(None, "Best", "auto")[0].lower()


def test_open_folder_no_output():
    assert "run a workflow" in app._open_folder("").lower()


# --------------------------------------------------------------------------- #
# Handlers drive the pipeline with the right overrides (pipeline mocked)
# --------------------------------------------------------------------------- #
def test_drum_teardown_overrides(monkeypatch, tmp_path):
    seen: dict = {}

    def fake_run(overrides, audio_file):
        seen.update(overrides)
        out = tmp_path / "loop"
        (out / "drums").mkdir(parents=True, exist_ok=True)
        (out / "drums" / "kick.wav").write_bytes(b"")
        (out / "drums" / "drums.mid").write_bytes(b"")
        manifest = {"drum_split": {"backend": "uvr", "files": {"kick": "x"}},
                    "drum_midi": {"source": "parts", "note_count": 12},
                    "analysis": {"source_bpm": 120}}
        return manifest, out

    monkeypatch.setattr(app, "_run", fake_run)
    status, _pick, _audio, midi, files, out_dir = app.drum_teardown("loop.wav", "auto")

    assert seen["drums.split.enabled"] is True
    assert seen["drums.split.from_input"] is True
    assert seen["drums.midi.enabled"] is True
    assert seen["separation.enabled"] is False
    assert "12 MIDI notes" in status
    assert midi and midi.endswith("drums.mid")
    assert out_dir.endswith("loop")


def test_extract_stems_overrides(monkeypatch, tmp_path):
    seen: dict = {}

    def fake_run(overrides, audio_file):
        seen.update(overrides)
        out = tmp_path / "song"
        (out / "stems").mkdir(parents=True, exist_ok=True)
        (out / "stems" / "vocals.wav").write_bytes(b"")
        return {"separation": {"model": "bs_roformer", "files": {"vocals.wav": "x"}}}, out

    monkeypatch.setattr(app, "_run", fake_run)
    status, _pick, preview, files, out_dir = app.extract_stems("song.wav", "SOTA", "auto")
    assert seen["separation.preset"] == "sota"
    assert "✓ Extract" in status
    assert preview and preview.endswith("vocals.wav")


def test_melodic_monophonic_and_quantize_flags(monkeypatch, tmp_path):
    seen: dict = {}

    def fake_run(overrides, audio_file):
        seen.update(overrides)
        return {"midi": {"files": {}, "skipped": "Basic Pitch not available"}}, tmp_path / "s"

    monkeypatch.setattr(app, "_run", fake_run)
    app.melodic_midi("bass.wav", True, True, "auto")
    assert seen["midi.enabled"] is True
    assert seen["midi.quantize_to_grid"] is True
    assert seen["analysis.enabled"] is True          # grid needed for quantize
    assert seen["midi.monophonic_stems"] == app._MELODIC_STEMS

    seen.clear()
    app.melodic_midi("bass.wav", False, False, "auto")
    assert seen["midi.monophonic_stems"] == []
    assert seen["analysis.enabled"] is False


# --------------------------------------------------------------------------- #
# Build (requires gradio) — GPU-free, no launch
# --------------------------------------------------------------------------- #
def test_build_ui_constructs():
    pytest.importorskip("gradio")
    demo = app.build_ui()
    assert demo.__class__.__name__ == "Blocks"
