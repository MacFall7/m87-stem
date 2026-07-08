"""Presets + UVR/hybrid backend routing — GPU-free, audio_separator fully mocked."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from stemforge.io_utils import AudioTensor
from stemforge.orchestrator import BS_ROFORMER_MODEL, Pipeline, load_config, preset_names
from stemforge.separate_uvr import canonical_stem_name


# --------------------------------------------------------------------------- #
# Quality presets
# --------------------------------------------------------------------------- #
def test_preset_names_order():
    assert preset_names() == ["fast", "best", "sota", "max"]


def test_preset_fast():
    sep = load_config(overrides={"separation.preset": "fast"}).separation
    assert (sep.backend, sep.model, sep.shifts) == ("demucs", "htdemucs", 1)


def test_preset_best():
    sep = load_config(overrides={"separation.preset": "best"}).separation
    assert (sep.backend, sep.model, sep.shifts) == ("demucs", "htdemucs_ft", 2)


def test_preset_sota():
    sep = load_config(overrides={"separation.preset": "sota"}).separation
    assert sep.backend == "uvr"
    assert sep.uvr_model == BS_ROFORMER_MODEL
    assert sep.stems == ["vocals", "instrumental"]


def test_preset_max():
    sep = load_config(overrides={"separation.preset": "max"}).separation
    assert sep.backend == "hybrid"
    assert sep.model == "htdemucs_ft"
    assert sep.uvr_model == BS_ROFORMER_MODEL
    assert sep.shifts == 2
    assert sep.stems == ["vocals", "drums", "bass", "other"]


def test_preset_case_insensitive_and_normalized():
    sep = load_config(overrides={"separation.preset": "SOTA"}).separation
    assert sep.backend == "uvr"
    assert sep.preset == "sota"  # normalized for the manifest


def test_explicit_override_wins_over_preset():
    """Precedence: yaml < preset < dotted overrides (--set escape hatch works)."""
    sep = load_config(overrides={
        "separation.preset": "max", "separation.shifts": 1,
    }).separation
    assert sep.backend == "hybrid"  # from the preset
    assert sep.shifts == 1          # explicit override beats the preset's 2


def test_unknown_preset_rejected():
    with pytest.raises(ValueError, match="preset"):
        load_config(overrides={"separation.preset": "ultra"})


def test_no_preset_leaves_defaults():
    sep = load_config().separation
    assert sep.preset is None
    assert (sep.backend, sep.model) == ("demucs", "htdemucs_ft")


# --------------------------------------------------------------------------- #
# Output-filename stem-token mapping
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("filename,expected", [
    ("mix_(Vocals)_model_bs_roformer_ep_317_sdr_12.9755.wav", "vocals"),
    ("mix_(Instrumental)_model_bs_roformer_ep_317_sdr_12.9755.wav", "instrumental"),
    ("track_(Drums)_htdemucs.wav", "drums"),
    ("track_(Bass)_htdemucs.wav", "bass"),
    ("track_(Other)_htdemucs.wav", "other"),
    ("track_(VOCALS)_x.wav", "vocals"),          # case-insensitive
    ("track_(No Vocals)_x.wav", "instrumental"),  # alias
    ("weird_output.wav", "weird_output"),         # no token -> slugified fallback
])
def test_canonical_stem_name(filename, expected):
    assert canonical_stem_name(filename) == expected


# --------------------------------------------------------------------------- #
# Mocked audio_separator
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_audio_separator(monkeypatch):
    """Inject a fake `audio_separator` package; records calls, writes real WAVs."""
    calls: dict = {"init_count": 0}

    class FakeSeparator:
        def __init__(self, output_dir=None, output_format="WAV", model_file_dir=None,
                     use_autocast=False, demucs_params=None, **kwargs):
            calls["init_count"] += 1
            calls["init"] = dict(
                output_dir=output_dir, output_format=output_format,
                model_file_dir=model_file_dir, use_autocast=use_autocast,
                demucs_params=demucs_params,
            )
            self.output_dir = output_dir
            self.model = None

        def load_model(self, model_filename=None):
            calls["model"] = model_filename
            self.model = model_filename

        def separate(self, path):
            calls.setdefault("inputs", []).append(path)
            src = Path(path)
            data, sr = sf.read(str(src), always_2d=True)
            out = []
            for token in ("Vocals", "Instrumental"):
                name = f"{src.stem}_({token})_{Path(self.model).stem}.wav"
                sf.write(str(Path(self.output_dir) / name), data * 0.5, sr)
                out.append(name)  # relative to output_dir, like some versions
            return out

    pkg = types.ModuleType("audio_separator")
    mod = types.ModuleType("audio_separator.separator")
    mod.Separator = FakeSeparator
    pkg.separator = mod
    monkeypatch.setitem(sys.modules, "audio_separator", pkg)
    monkeypatch.setitem(sys.modules, "audio_separator.separator", mod)
    return calls


def _base_overrides(tmp_path) -> dict:
    return {
        "ingest.normalize": "none",
        "analysis.enabled": False,
        "output.root": str(tmp_path / "out"),
    }


def test_separate_uvr_maps_and_caches(fake_audio_separator, sine):
    from stemforge import separate_uvr

    samples, sr = sine
    audio = AudioTensor(samples, sr)
    cfg = load_config(overrides={"separation.preset": "sota"}).separation

    stems, engine = separate_uvr.separate(audio, cfg, device="cpu")
    assert set(stems) == {"vocals", "instrumental"}
    assert all(isinstance(v, AudioTensor) for v in stems.values())
    assert stems["vocals"].sample_rate == sr
    assert fake_audio_separator["model"] == BS_ROFORMER_MODEL
    assert fake_audio_separator["init"]["output_format"] == "WAV"
    assert fake_audio_separator["init"]["model_file_dir"].endswith("models/uvr")
    assert fake_audio_separator["init"]["use_autocast"] is True
    assert fake_audio_separator["init"]["demucs_params"]["segment_size"] == "Default"

    # scratch files are cleaned up after each call
    assert list(engine.work_dir.iterdir()) == []

    # same model -> engine (and loaded weights) are reused, not rebuilt
    _, engine2 = separate_uvr.separate(audio, cfg, device="cpu", engine=engine)
    assert engine2 is engine
    assert fake_audio_separator["init_count"] == 1

    # different model -> new engine
    cfg.uvr_model = "some_other_model.ckpt"
    _, engine3 = separate_uvr.separate(audio, cfg, device="cpu", engine=engine)
    assert engine3 is not engine
    assert fake_audio_separator["init_count"] == 2


def test_pipeline_uvr_backend(fake_audio_separator, tmp_path, sine_wav):
    wav = sine_wav
    cfg = load_config(overrides={**_base_overrides(tmp_path), "separation.preset": "sota"})
    m = Pipeline(cfg).run(wav)

    sep = m["separation"]
    assert sep["backend"] == "uvr"
    assert sep["preset"] == "sota"
    assert set(sep["files"]) == {"vocals.wav", "instrumental.wav"}
    assert m["models"]["uvr"] == BS_ROFORMER_MODEL
    out = tmp_path / "out" / "in" / "stems"
    assert (out / "vocals.wav").is_file() and (out / "instrumental.wav").is_file()


def test_pipeline_hybrid_backend(fake_audio_separator, monkeypatch, tmp_path, sine_wav):
    import stemforge.separate as demucs_mod

    seen: dict = {}

    def fake_demucs(audio, cfg, device="cpu", model=None):
        seen["stems_requested"] = list(cfg.stems)
        seen["input_samples"] = audio.num_samples
        mk = lambda: AudioTensor(np.zeros((2, audio.num_samples), np.float32), audio.sample_rate)
        # demucs always emits its own (near-silent) vocals; hybrid must drop it
        return {"vocals": mk(), "drums": mk(), "bass": mk(), "other": mk()}, "cached-model"

    monkeypatch.setattr(demucs_mod, "separate", fake_demucs)

    wav = sine_wav
    cfg = load_config(overrides={**_base_overrides(tmp_path), "separation.preset": "max"})
    m = Pipeline(cfg).run(wav)

    sep = m["separation"]
    assert sep["backend"] == "hybrid"
    # merged 4-stem: roformer vocals + demucs residual; instrumental dropped
    assert set(sep["files"]) == {"vocals.wav", "drums.wav", "bass.wav", "other.wav"}
    assert seen["stems_requested"] == ["drums", "bass", "other"]
    assert set(m["models"]) == {"uvr", "demucs"}
    assert m["models"]["demucs"] == "htdemucs_ft"
    assert "model_bs_roformer" in sep["model"] and "htdemucs_ft" in sep["model"]


def test_uvr_missing_dependency_fails_soft(monkeypatch, tmp_path, sine_wav):
    """No audio-separator installed -> stage records a skip, run still completes."""
    monkeypatch.setitem(sys.modules, "audio_separator", None)
    monkeypatch.setitem(sys.modules, "audio_separator.separator", None)

    wav = sine_wav
    cfg = load_config(overrides={**_base_overrides(tmp_path), "separation.preset": "sota"})
    m = Pipeline(cfg).run(wav)

    assert "skipped" in m["separation"]
    assert "audio-separator" in m["separation"]["skipped"]
    assert (tmp_path / "out" / "in" / "manifest.json").is_file()


def test_unknown_backend_recorded_as_error(tmp_path, sine_wav):
    wav = sine_wav
    cfg = load_config(overrides={**_base_overrides(tmp_path), "separation.backend": "spleeter"})
    m = Pipeline(cfg).run(wav)
    assert "error" in m["separation"]
    assert "spleeter" in m["separation"]["error"]


def test_hybrid_keeps_uvr_result_when_demucs_missing(fake_audio_separator, monkeypatch, tmp_path, sine_wav):
    """Hybrid must not throw away the finished roformer pass if demucs is absent."""
    import stemforge.separate as demucs_mod

    def no_demucs(*a, **kw):
        raise ModuleNotFoundError("No module named 'demucs'", name="demucs")

    monkeypatch.setattr(demucs_mod, "separate", no_demucs)
    cfg = load_config(overrides={**_base_overrides(tmp_path), "separation.preset": "max"})
    m = Pipeline(cfg).run(sine_wav)

    sep = m["separation"]
    assert sep["backend"] == "hybrid"
    assert set(sep["files"]) == {"vocals.wav", "instrumental.wav"}  # degraded 2-stem
    assert any("demucs" in n for n in sep["notes"])


def test_non_dependency_import_error_is_an_error_not_a_skip(monkeypatch, tmp_path, sine_wav):
    """Only known optional backend deps fail soft; other MNFEs surface as errors."""
    import stemforge.separate as demucs_mod

    def broken(*a, **kw):
        raise ModuleNotFoundError("No module named 'einops'", name="einops")

    monkeypatch.setattr(demucs_mod, "separate", broken)
    cfg = load_config(overrides=_base_overrides(tmp_path))  # default demucs backend
    m = Pipeline(cfg).run(sine_wav)

    assert "error" in m["separation"] and "skipped" not in m["separation"]
    assert "einops" in m["separation"]["error"]
