"""Tempo/beat evidence — A-series PR-A2.

Receipts (verified at main): analysis.py silently returned 120 BPM on several
failure paths (`:96,:113,:119,:123`), `:98` assumed 4/4 (`downbeats=beats[::4]`),
and BeatGrid carried no confidence. This regression pins the first fix: an
un-estimable tempo must be UNKNOWN (0.0), never a silent 120.
"""

from __future__ import annotations

from stemforge.analysis import _bpm_from_beats


# --------------------------------------------------------------------------- #
# RED: an un-estimable tempo is UNKNOWN (0.0), never a silent 120
# --------------------------------------------------------------------------- #
def test_bpm_unknown_is_zero_not_silent_120():
    assert _bpm_from_beats([1.0, 2.0], regression=True) == 0.0          # <3 beats
    assert _bpm_from_beats([3.0, 2.0, 1.0], regression=True) == 0.0     # non-positive slope
    assert _bpm_from_beats([1.0, 1.0, 1.0], regression=False) == 0.0    # non-positive median
