import pytest

from stemforge.io_utils import AudioTensor
from stemforge.stretch import time_stretch


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
