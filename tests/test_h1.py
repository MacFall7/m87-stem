"""C7 (EC-SF-H1) — local server hardening: bind guard (R1), TTL eviction (R4).

Pure helpers, no HTTP server needed. Route-level checks live in test_webapp.py.
"""

from __future__ import annotations

import pytest

from stemforge import webapp


# --------------------------------------------------------------------------- #
# R1/AC1 — non-loopback bind refused unless allow_remote + token
# --------------------------------------------------------------------------- #
def test_require_safe_bind_loopback_ok():
    webapp.require_safe_bind("127.0.0.1", False, None)
    webapp.require_safe_bind("localhost", False, None)
    webapp.require_safe_bind("::1", False, None)


def test_require_safe_bind_refuses_remote_without_optin():
    with pytest.raises(RuntimeError):
        webapp.require_safe_bind("0.0.0.0", False, None)


def test_require_safe_bind_refuses_remote_without_token():
    with pytest.raises(RuntimeError):
        webapp.require_safe_bind("192.168.1.10", True, None)


def test_require_safe_bind_allows_remote_with_flag_and_token():
    webapp.require_safe_bind("0.0.0.0", True, "s3cret")  # no raise


# --------------------------------------------------------------------------- #
# R4/AC3 — finished jobs are TTL-evicted; running/fresh jobs are kept
# --------------------------------------------------------------------------- #
def test_prune_jobs_evicts_finished_after_ttl(monkeypatch):
    monkeypatch.setattr(webapp, "_JOB_TTL_S", 100.0)
    webapp._JOBS.clear()
    webapp._JOBS["old"] = {"status": "done", "finished_at": 0.0}
    webapp._JOBS["errold"] = {"status": "error", "finished_at": 0.0}
    webapp._JOBS["fresh"] = {"status": "done", "finished_at": 1_000.0}
    webapp._JOBS["running"] = {"status": "running", "finished_at": None}

    webapp._prune_jobs(now=1_000.0)   # old finished 1000s ago > 100s TTL

    assert "old" not in webapp._JOBS and "errold" not in webapp._JOBS   # evicted
    assert "fresh" in webapp._JOBS and "running" in webapp._JOBS        # kept
    webapp._JOBS.clear()
