"""Benchmark harness — A-series PR-A5.

Verifies the accuracy harness runs offline, computes the gated metrics, enforces
thresholds (a strict bound must trip a violation), carries provenance, and reports
— never silently drops — the metrics that need an unavailable model.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pretty_midi")
pytest.importorskip("librosa")

from benchmarks import corpus, metrics                     # noqa: E402
from benchmarks import run_benchmarks as rb                # noqa: E402


# --------------------------------------------------------------------------- #
# Pure metric functions
# --------------------------------------------------------------------------- #
def test_match_events_and_f1():
    tp, fp, fn = metrics.match_events([1.0, 2.0, 3.0], [1.01, 2.0, 5.0], tol_s=0.05)
    assert (tp, fp, fn) == (2, 1, 1)
    assert metrics.f1(2, 1, 1) == pytest.approx(2 / 3, rel=1e-6)
    assert metrics.f1(0, 3, 3) == 0.0


def test_duplicate_rate_counts_only_cross_pitch_coincidence():
    # two different-pitch notes within 18 ms -> a duplicate; same pitch -> not
    assert metrics.duplicate_rate([(0.0, 36), (0.005, 45)]) == 1.0
    assert metrics.duplicate_rate([(0.0, 36), (0.005, 36)]) == 0.0
    assert metrics.duplicate_rate([(0.0, 36), (0.5, 45)]) == 0.0


def test_half_double_and_rank_corr():
    assert metrics.half_double_ok(120.0, [60.0, 120.0, 240.0]) is True
    assert metrics.half_double_ok(140.0, [142.3, 71.1, 284.6]) is True   # right octave, small drift
    assert metrics.half_double_ok(100.0, [150.0, 75.0]) is False
    assert metrics.rank_corr([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert metrics.rank_corr([1, 2, 3], [30, 20, 10]) == pytest.approx(-1.0)
    assert metrics.tempo_abs_error(120.0, 0.0) == 120.0                  # unknown -> full miss


# --------------------------------------------------------------------------- #
# Corpus determinism
# --------------------------------------------------------------------------- #
def test_fixture_sha256_is_stable():
    assert corpus.fixture_sha256() == corpus.fixture_sha256()
    assert len(corpus.tempo_fixtures()) == 3
    assert {df.name for df in corpus.drum_fixtures()} == {"groove_clean", "groove_bleed"}


# --------------------------------------------------------------------------- #
# End-to-end harness + threshold enforcement + provenance
# --------------------------------------------------------------------------- #
def test_harness_runs_and_passes_thresholds():
    report = rb.run()
    m = report["metrics"]
    for key in ("tempo_abs_error_bpm_max", "half_double_correct", "drum_f1_min",
                "duplicate_rate_max", "velocity_rank_corr_min"):
        assert key in m
    # the shipped thresholds pass on the current tree
    assert rb.evaluate(report, rb.load_thresholds()) == []
    # provenance present
    prov = report["provenance"]
    assert len(prov["fixture_sha256"]) == 64
    assert len(prov["config_hash"]) == 64
    assert prov["versions"]["stemforge"]
    # skipped metrics are reported with a reason, not dropped
    assert "melodic_onset_pitch_f1" in report["skipped"]


def test_thresholds_actually_gate():
    report = rb.run()
    # an impossibly strict duplicate-rate bound must trip a violation
    strict = dict(rb.load_thresholds())
    strict["drum_f1_min"] = 1.01           # unreachable (F1 <= 1)
    violations = rb.evaluate(report, strict)
    assert any("drum_f1_min" in v for v in violations)


def test_main_exit_code_and_report(tmp_path):
    rc = rb.main(["--report", str(tmp_path / "r.json")])
    assert rc == 0                                    # baseline passes
    assert (tmp_path / "r.json").is_file()
    import json
    saved = json.loads((tmp_path / "r.json").read_text())
    assert saved["passed"] is True and saved["violations"] == []
