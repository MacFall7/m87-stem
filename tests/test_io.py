import numpy as np

from stemforge.io_utils import AudioTensor, load_audio, receipt, save_audio, slugify


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
