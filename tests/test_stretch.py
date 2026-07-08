from pathlib import Path

import pytest
import soundfile as sf

from stemforge.io_utils import AudioTensor
from stemforge.stretch import match_bpm_file, time_stretch


def test_identity_ratio_is_noop(sine):
    samples, sr = sine
    a = AudioTensor(samples, sr)
    out = time_stretch(a, 1.0)
    assert out.num_samples == a.num_samples


def test_librosa_fallback_length(sine):
    pytest.importorskip("librosa")
    from stemforge.stretch import _librosa

    samples, sr = sine
    a = AudioTensor(samples, sr)
    faster = _librosa(a, 2.0)      # 2x tempo -> ~half length
    slower = _librosa(a, 0.5)      # half tempo -> ~2x length
    assert faster.num_samples < a.num_samples < slower.num_samples
    assert abs(faster.num_samples - a.num_samples / 2) / (a.num_samples / 2) < 0.15


# --------------------------------------------------------------------------- #
# Match BPM (whole-file) — deterministic via source override + librosa engine
# --------------------------------------------------------------------------- #
def _write_wav(tmp_path, sine) -> Path:
    samples, sr = sine
    wav = tmp_path / "loop.wav"
    sf.write(str(wav), samples.T, sr, subtype="FLOAT")
    return wav


def test_match_bpm_file_ratio_math_with_override(tmp_path, sine):
    pytest.importorskip("librosa")
    wav = _write_wav(tmp_path, sine)
    n_in = AudioTensor(*sine).num_samples

    res = match_bpm_file(wav, target_bpm=140, out_dir=tmp_path / "matched",
                         source_bpm=120, engine="librosa")

    assert res["source_bpm"] == 120.0
    assert res["source_bpm_overridden"] is True
    assert res["source_bpm_detected"] == 0.0          # detection skipped when overridden
    assert res["target_bpm"] == 140.0
    assert res["ratio"] == pytest.approx(140 / 120, rel=1e-3)
    assert res["engine"] == "librosa"

    out = Path(res["output"])
    assert out.is_file() and out.name == "loop_140bpm.wav"
    # ratio > 1 speeds up -> shorter file (length ~ input / ratio)
    data, _ = sf.read(str(out), always_2d=True)
    expected = n_in / (140 / 120)
    assert abs(data.shape[0] - expected) / expected < 0.15


def test_match_bpm_file_zero_target_skips(tmp_path, sine):
    wav = _write_wav(tmp_path, sine)
    res = match_bpm_file(wav, target_bpm=0, out_dir=tmp_path / "m")
    assert "skipped" in res and "target_bpm" in res["skipped"]


def test_match_bpm_file_undetectable_source_skips(tmp_path, sine, monkeypatch):
    import stemforge.stretch as st

    monkeypatch.setattr(st, "detect_bpm", lambda *a, **k: 0.0)  # nothing detectable
    wav = _write_wav(tmp_path, sine)
    res = match_bpm_file(wav, target_bpm=120, out_dir=tmp_path / "m")  # no source override
    assert "skipped" in res and "source_bpm" in res["skipped"]


def test_detect_bpm_returns_float(sine):
    pytest.importorskip("librosa")
    from stemforge.stretch import detect_bpm

    bpm = detect_bpm(AudioTensor(*sine), engine="librosa", device="cpu")
    assert isinstance(bpm, float) and bpm >= 0.0
