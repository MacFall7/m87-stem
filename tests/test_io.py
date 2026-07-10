from pathlib import Path

import numpy as np

from stemforge.io_utils import AudioTensor, load_audio, read_json, receipt, save_audio, slugify, write_json


def test_audiotensor_shapes(sine):
    samples, sr = sine
    a = AudioTensor(samples, sr)
    assert a.num_channels == 2
    assert a.num_samples == samples.shape[1]
    assert abs(a.duration_seconds - 0.5) < 1e-3
    assert a.to_mono().num_channels == 1
    assert a.to_mono().to_stereo().num_channels == 2


def test_1d_promoted_to_2d():
    a = AudioTensor(np.zeros(1000, dtype=np.float32), 44100)
    assert a.samples.ndim == 2
    assert a.num_channels == 1


def test_save_load_roundtrip(tmp_path, sine):
    samples, sr = sine
    p = save_audio(tmp_path / "x.wav", AudioTensor(samples, sr))
    back = load_audio(p)
    assert back.sample_rate == sr
    assert back.num_channels == 2
    assert np.allclose(back.samples, samples, atol=1e-4)


def test_receipt_hashes(tmp_path, sine):
    samples, sr = sine
    p = save_audio(tmp_path / "a.wav", AudioTensor(samples, sr))
    r = receipt([p], tmp_path)
    assert "a.wav" in r
    assert len(r["a.wav"]) == 64  # sha256 hex


def test_slugify():
    assert slugify("My Song") == "My_Song"
    assert slugify("a/b\\c") == "a_b_c"
    assert slugify("") == "song"


def test_write_json_serializes_numpy_scalars(tmp_path):
    """A manifest can carry any numpy scalar. np.bool_ (the A1 cymbal evidence hit
    it first) previously crashed write_json — _json_default now covers np.generic."""
    obj = {
        "flag": np.bool_(True),          # the case that broke drum teardown on Windows
        "f": np.float64(1.5),
        "i": np.int64(7),
        "arr": np.arange(3),             # ndarray -> list
        "path": tmp_path / "out",        # Path -> str
    }
    p = write_json(tmp_path / "m.json", obj)   # must not raise
    back = read_json(p)
    assert back["flag"] is True
    assert back["f"] == 1.5 and isinstance(back["f"], float)
    assert back["i"] == 7 and isinstance(back["i"], int)
    assert back["arr"] == [0, 1, 2]
    assert back["path"] == str(Path(tmp_path / "out"))
