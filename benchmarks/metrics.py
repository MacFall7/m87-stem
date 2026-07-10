"""Accuracy metrics for the benchmark corpus.

Pure functions over ground truth + StemForge output — no I/O, deterministic.
"""

from __future__ import annotations

import numpy as np


def tempo_abs_error(true_bpm: float, est_bpm: float) -> float:
    """Absolute BPM error. An unknown estimate (0.0) is a full miss (= true_bpm)."""
    if est_bpm <= 0:
        return float(true_bpm)
    return float(abs(true_bpm - est_bpm))


def half_double_ok(true_bpm: float, candidates: list[float], rel_tol: float = 0.05) -> bool:
    """True if the true BPM matches the estimate OR one of its half/double
    candidates within an octave-relative tolerance (a half/double detection error
    is recoverable — the correct octave is surfaced in the candidate list)."""
    return any(abs(true_bpm - c) <= rel_tol * true_bpm for c in (candidates or []))


def match_events(truth: list[float], pred: list[float], tol_s: float = 0.05) -> tuple[int, int, int]:
    """Greedy one-to-one match of predicted onset times to truth within tol.
    Returns (true_positives, false_positives, false_negatives)."""
    truth_sorted = sorted(truth)
    used = [False] * len(truth_sorted)
    tp = 0
    for p in sorted(pred):
        best, best_d = -1, tol_s
        for i, t in enumerate(truth_sorted):
            if used[i]:
                continue
            d = abs(p - t)
            if d <= best_d:
                best, best_d = i, d
        if best >= 0:
            used[best] = True
            tp += 1
    fp = len(pred) - tp
    fn = len(truth_sorted) - tp
    return tp, fp, fn


def f1(tp: int, fp: int, fn: int) -> float:
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return float(2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0


def duplicate_rate(note_times_pitches: list[tuple[float, int]], window_s: float = 0.018) -> float:
    """Fraction of adjacent note pairs that are coincident (<= window) but a
    DIFFERENT pitch — i.e. cross-part duplicates. 0.0 is clean."""
    seq = sorted(note_times_pitches)
    if len(seq) < 2:
        return 0.0
    dup = sum(1 for a, b in zip(seq, seq[1:])
              if b[0] - a[0] <= window_s and a[1] != b[1])
    return float(dup) / (len(seq) - 1)


def rank_corr(a: list[float], b: list[float]) -> float:
    """Spearman rank correlation (velocity-vs-render-amplitude ordering).
    Returns 0.0 when undefined (constant input or < 2 points)."""
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    ra = _rankdata(a)
    rb = _rankdata(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    if denom <= 1e-12:
        return 0.0
    return float((ra * rb).sum() / denom)


def _rankdata(x: list[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    order = arr.argsort()
    ranks = np.empty(len(arr), dtype=np.float64)
    ranks[order] = np.arange(len(arr), dtype=np.float64)
    return ranks
