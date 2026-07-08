import random

from stemforge.analysis import BeatGrid, _bpm_from_beats


def test_bpm_from_clean_beats():
    beats = [i * 0.5 for i in range(9)]  # exactly 120 BPM
    assert abs(_bpm_from_beats(beats, regression=True) - 120.0) < 0.5
    assert abs(_bpm_from_beats(beats, regression=False) - 120.0) < 0.5


def test_bpm_regression_robust_to_jitter():
    random.seed(1)
    beats = [i * 0.5 + random.uniform(-0.012, 0.012) for i in range(17)]  # ~120 BPM + jitter
    assert abs(_bpm_from_beats(beats, regression=True) - 120.0) < 3.0


def test_beatgrid_dict_and_helpers():
    g = BeatGrid(source_bpm=120.0, beats=[0.0, 0.5, 1.0, 1.5], downbeats=[0.0, 1.0])
    d = g.to_dict()
    assert d["source_bpm"] == 120.0
    assert d["num_beats"] == 4 and d["num_downbeats"] == 2
    assert abs(g.seconds_per_beat - 0.5) < 1e-9
    assert g.nearest_beat(0.6) == 0.5
