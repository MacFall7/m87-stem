import numpy as np

from stemforge.drum_midi import GM, _hat_is_open, _peak_in_window, _rms_at, _velocity_from_norm


def test_gm_mapping():
    assert (GM["kick"], GM["snare"], GM["hihat"], GM["ride"], GM["crash"]) == (36, 38, 42, 51, 49)
    assert GM["hihat_open"] == 46 and GM["hihat_closed"] == 42


def test_velocity_from_norm_monotonic_and_bounded():
    # PR-A4: velocity now maps normalized (0..1) per-part strength (was global-peak)
    vals = [_velocity_from_norm(p) for p in (0.0, 0.25, 0.5, 1.0)]
    assert vals == sorted(vals)
    assert all(1 <= v <= 127 for v in vals)
    assert vals[-1] == 127          # full normalized strength => full velocity
    assert _velocity_from_norm(0.0) >= 1  # floor keeps quiet/ghost hits audible


def test_peak_in_window():
    env = np.array([0.1, 0.9, 0.2, 0.05])
    t = np.array([0.0, 0.05, 0.10, 0.15])
    assert _peak_in_window(env, t, 0.0, 0.06) == 0.9
    assert _peak_in_window(env, t, 0.14, 0.06) == 0.05


def test_rms_at_energy():
    y = np.ones(1000, dtype=np.float32)
    assert abs(_rms_at(y, 0, 500) - 1.0) < 1e-6
    assert _rms_at(y, 5000, 100) == 0.0  # out of range -> empty -> 0


def test_hat_open_vs_closed():
    sr = 44100
    closed = np.zeros(sr, dtype=np.float32)
    closed[: int(0.01 * sr)] = np.hanning(int(0.01 * sr)).astype(np.float32)
    rng = np.random.default_rng(0)
    openh = np.zeros(sr, dtype=np.float32)
    openh[: int(0.3 * sr)] = (0.5 * rng.standard_normal(int(0.3 * sr))).astype(np.float32)
    assert _hat_is_open(openh, sr, 0.0) is True
    assert _hat_is_open(closed, sr, 0.0) is False
