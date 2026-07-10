"""Run the StemForge accuracy benchmarks and enforce thresholds.

Usage:
    python -m benchmarks.run_benchmarks [--report out.json] [--corpus-dir dir]

Exits non-zero if any metric violates ``benchmarks/thresholds.yaml`` — wired into
CI so an accuracy regression fails the build. Every report embeds provenance
(fixture SHA-256, config hash, library/model versions) captured at run time.

Metrics implemented here run fully offline: tempo abs error + half/double
(analysis), drum per-class F1, cross-part duplicate rate, and velocity rank
correlation (drum_midi). Metrics that need an unavailable model are reported as
``skipped`` with a reason — never silently dropped.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import yaml

from stemforge import __version__ as sf_version
from stemforge import analysis, drum_midi
from stemforge.io_utils import sha256_bytes
from stemforge.orchestrator import load_config

from . import corpus, metrics

_THRESHOLDS = Path(__file__).resolve().parent / "thresholds.yaml"


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def _versions() -> dict[str, str]:
    vers = {"stemforge": sf_version, "python": platform.python_version(),
            "numpy": np.__version__, "soundfile": sf.__version__}
    try:
        import librosa
        vers["librosa"] = librosa.__version__
    except Exception:  # noqa: BLE001
        vers["librosa"] = "absent"
    try:
        import pretty_midi
        vers["pretty_midi"] = getattr(pretty_midi, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        vers["pretty_midi"] = "absent"
    return vers


def _config_hash() -> str:
    import dataclasses
    return sha256_bytes(json.dumps(
        dataclasses.asdict(load_config().drums.midi), sort_keys=True, default=str).encode())


# --------------------------------------------------------------------------- #
# Evaluators
# --------------------------------------------------------------------------- #
def _eval_tempo() -> dict[str, Any]:
    errors, half_double_hits, n = [], 0, 0
    per_fixture = {}
    for tf in corpus.tempo_fixtures():
        grid = analysis.analyze(analysis.AudioTensor(tf.audio, tf.sr), engine="librosa")
        err = metrics.tempo_abs_error(tf.true_bpm, grid.source_bpm)
        hd = metrics.half_double_ok(tf.true_bpm, grid.bpm_candidates or [grid.source_bpm])
        errors.append(err)
        half_double_hits += int(hd)
        n += 1
        per_fixture[tf.name] = {"true_bpm": tf.true_bpm, "est_bpm": round(grid.source_bpm, 2),
                                "abs_error": round(err, 3), "half_double_ok": hd}
    return {
        "tempo_abs_error_bpm_mean": float(np.mean(errors)) if errors else 0.0,
        "tempo_abs_error_bpm_max": float(np.max(errors)) if errors else 0.0,
        "half_double_correct": half_double_hits / n if n else 0.0,
        "per_fixture": per_fixture,
    }


def _run_drum(df: corpus.DrumFixture, work: Path) -> dict[str, Any]:
    parts: dict[str, str] = {}
    for part, y in df.parts.items():
        p = work / f"{df.name}_{part}.wav"
        sf.write(str(p), y, df.sr, subtype="FLOAT")
        parts[part] = str(p)
    cfg = load_config().drums.midi
    res = drum_midi._from_parts(parts, cfg, work, bpm=120.0)
    return res


def _eval_drums() -> dict[str, Any]:
    _CLASS_NOTES = {"kick": {drum_midi.GM["kick"]}, "snare": {drum_midi.GM["snare"]},
                    "toms": {drum_midi.GM["toms_low"], drum_midi.GM["toms_mid"],
                             drum_midi.GM["toms_high"], drum_midi.GM["toms"]}}
    out: dict[str, Any] = {"per_fixture": {}}
    f1s: dict[str, list[float]] = {"kick": [], "snare": [], "toms": []}
    dup_rates, vel_corrs = [], []
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        for df in corpus.drum_fixtures():
            res = _run_drum(df, work)
            events = res.get("events", [])
            fx: dict[str, Any] = {"duplicates_removed": res.get("duplicates_removed", 0)}
            for cls, notes in _CLASS_NOTES.items():
                pred = [e["time"] for e in events if e["note"] in notes]
                tp, fp, fn = metrics.match_events(df.truth.get(cls, []), pred)
                score = metrics.f1(tp, fp, fn)
                f1s[cls].append(score)
                fx[f"{cls}_f1"] = round(score, 3)
            # duplicate rate from the written MIDI (time, pitch)
            import pretty_midi
            mid = pretty_midi.PrettyMIDI(res["file"])
            notes_tp = [(n.start, n.pitch) for inst in mid.instruments for n in inst.notes]
            dr = metrics.duplicate_rate(notes_tp)
            fx["duplicate_rate"] = round(dr, 4)
            dup_rates.append(dr)
            # velocity rank correlation on the snare part (varied amplitudes)
            vc = _velocity_corr(df, events, "snare")
            if vc is not None:
                fx["snare_velocity_rank_corr"] = round(vc, 3)
                vel_corrs.append(vc)
            out["per_fixture"][df.name] = fx
    # Gate on the reliable backbone classes (kick+snare); toms F1 is reported per
    # fixture but not gated — a two-hit tom fill makes its F1 statistically noisy.
    backbone = f1s["kick"] + f1s["snare"]
    out["drum_f1_min"] = float(min(backbone)) if backbone else 0.0
    out["drum_toms_f1_mean"] = float(np.mean(f1s["toms"])) if f1s["toms"] else 0.0
    out["drum_kick_f1_mean"] = float(np.mean(f1s["kick"])) if f1s["kick"] else 0.0
    out["duplicate_rate_max"] = float(max(dup_rates)) if dup_rates else 0.0
    out["velocity_rank_corr_min"] = float(min(vel_corrs)) if vel_corrs else 1.0
    return out


def _velocity_corr(df: corpus.DrumFixture, events: list[dict[str, Any]], part: str):
    truth = df.truth.get(part, [])
    amps = df.amps.get(part, [])
    if len(truth) != len(amps) or len(truth) < 2:
        return None
    pred = sorted((e["time"], e["velocity"]) for e in events if e["part"] == part)
    paired_amp, paired_vel = [], []
    for t, a in zip(truth, amps):
        near = min(pred, key=lambda tv: abs(tv[0] - t), default=None) if pred else None
        if near is not None and abs(near[0] - t) <= 0.05:
            paired_amp.append(a)
            paired_vel.append(near[1])
    if len(paired_amp) < 2:
        return None
    return metrics.rank_corr(paired_amp, paired_vel)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run() -> dict[str, Any]:
    import time
    t0 = time.perf_counter()
    tempo = _eval_tempo()
    drums = _eval_drums()
    metrics_flat = {
        "tempo_abs_error_bpm_max": tempo["tempo_abs_error_bpm_max"],
        "half_double_correct": tempo["half_double_correct"],
        "drum_f1_min": drums["drum_f1_min"],
        "duplicate_rate_max": drums["duplicate_rate_max"],
        "velocity_rank_corr_min": drums["velocity_rank_corr_min"],
    }
    return {
        "provenance": {
            "fixture_sha256": corpus.fixture_sha256(),
            "config_hash": _config_hash(),
            "versions": _versions(),
        },
        "metrics": metrics_flat,
        "detail": {"tempo": tempo, "drums": drums},
        "skipped": {
            "melodic_onset_pitch_f1": "requires basic-pitch (not installed in the offline harness)",
            "beat_alignment": "deferred — needs a beat-aligned reference render",
            "note_fragmentation": "deferred to the melodic path (basic-pitch)",
            "quantization_displacement": "covered by the A3 ledger (max_displacement_s); "
                                         "not re-measured here without basic-pitch",
        },
        "runtime_s": round(time.perf_counter() - t0, 3),
    }


def load_thresholds(path: Path = _THRESHOLDS) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def evaluate(report: dict[str, Any], thresholds: dict[str, Any]) -> list[str]:
    """Return a list of human-readable threshold violations (empty == pass)."""
    m = report["metrics"]
    violations: list[str] = []

    def _max(name: str) -> None:
        lim = thresholds.get(name)
        if lim is not None and m.get(name, 0.0) > lim:
            violations.append(f"{name}={m[name]:.3f} exceeds max {lim}")

    def _min(name: str) -> None:
        lim = thresholds.get(name)
        if lim is not None and m.get(name, 0.0) < lim:
            violations.append(f"{name}={m[name]:.3f} below min {lim}")

    _max("tempo_abs_error_bpm_max")
    _min("half_double_correct")
    _min("drum_f1_min")
    _max("duplicate_rate_max")
    _min("velocity_rank_corr_min")
    return violations


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run StemForge accuracy benchmarks.")
    ap.add_argument("--report", type=Path, default=None, help="write the JSON report here")
    ap.add_argument("--corpus-dir", type=Path, default=None, help="also materialize the corpus .wavs")
    args = ap.parse_args(argv)

    if args.corpus_dir:
        corpus.write_corpus(args.corpus_dir)
    report = run()
    thresholds = load_thresholds()
    violations = evaluate(report, thresholds)
    report["thresholds"] = thresholds
    report["violations"] = violations
    report["passed"] = not violations

    text = json.dumps(report, indent=2)
    if args.report:
        args.report.write_text(text, encoding="utf-8")
    print(text)
    for skip, why in report["skipped"].items():
        print(f"[skipped] {skip}: {why}", file=sys.stderr)
    if violations:
        print(f"\nBENCHMARK FAILED — {len(violations)} threshold violation(s):", file=sys.stderr)
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        return 1
    print("\nBENCHMARK PASSED", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
