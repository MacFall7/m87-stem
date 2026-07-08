"""Shared fixtures: synthetic audio so tests need no external files or GPU."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def sine():
    """0.5 s stereo 220 Hz sine at 44.1 kHz, shape (2, N) float32."""
    sr = 44100
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    mono = 0.3 * np.sin(2 * np.pi * 220 * t)
    return np.stack([mono, mono]).astype(np.float32), sr


@pytest.fixture
def click_track():
    """4 s mono click track at 120 BPM (beat every 0.5 s), 44.1 kHz."""
    sr = 44100
    dur = 4.0
    y = np.zeros(int(sr * dur), dtype=np.float32)
    for beat in np.arange(0, dur, 0.5):  # 120 BPM
        i = int(beat * sr)
        y[i:i + 200] = np.hanning(200).astype(np.float32)  # short percussive blip
    return y, sr
