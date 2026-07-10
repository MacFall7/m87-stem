"""Cymbal MIDI on the default (inagoy) drum path — A-series PR-A1.

Receipt (verified at main): ``drum_split`` maps every cymbal (platillos/cymbals/
hihat/ride/crash) into the single ``other`` stem, and ``drum_midi._from_parts``
skipped any part not in ``GM`` and not ``"hihat"`` — so the default ``demucs_inagoy``
teardown emitted kick/snare/toms only, with zero cymbal MIDI and the open/closed
hat classifier unreachable. These tests pin the fix: the ``other`` bucket is
routed to a ``cymbals`` super-class and classified per-onset.

GPU-free and deterministic — synthetic decaying-noise cymbal hits, seeded RNG.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from stemforge import drum_midi
from stemforge.io_utils import ensure_dir
from stemforge.orchestrator import load_config

CYMBAL_PITCHES = {
    drum_midi.GM["hihat_closed"],  # 42
    drum_midi.GM["hihat_open"],    # 46
    drum_midi.GM["crash"],         # 49
    drum_midi.GM["ride"],          # 51
}
SR = 44100


# --------------------------------------------------------------------------- #
# Synthetic drum-part audio (deterministic)
# --------------------------------------------------------------------------- #
def _decay_noise(dur: float, tau: float, seed: int, amp: float = 1.0, sr: int = SR) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * sr)) / sr
    return (amp * rng.standard_normal(t.size) * np.exp(-t / tau)).astype(np.float32)


def _tone(dur: float, freq: float, tau: float, amp: float = 0.9, sr: int = SR) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t) * np.exp(-t / tau)).astype(np.float32)


def _place(hits: list[tuple[float, np.ndarray]], total_s: float = 2.0, sr: int = SR) -> np.ndarray:
    y = np.zeros(int(total_s * sr), dtype=np.float32)
    for start, sig in hits:
        i = int(start * sr)
        end = min(y.size, i + sig.size)
        y[i:end] += sig[: end - i]
    return y


def _write(path: Path, y: np.ndarray, sr: int = SR) -> str:
    sf.write(str(path), y, sr, subtype="FLOAT")
    return str(path)


def _cymbal_bucket(seed_base: int = 100) -> np.ndarray:
    """An 'other' stem holding three cymbal hits: closed / open / crash decays."""
    return _place([
        (0.1, _decay_noise(0.5, 0.020, seed_base + 0)),   # closed hi-hat
        (0.7, _decay_noise(0.5, 0.200, seed_base + 1)),   # open hi-hat
        (1.3, _decay_noise(0.6, 0.600, seed_base + 2)),   # crash
    ])


def _pitches(mid_path: str) -> set[int]:
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(mid_path)
    return {n.pitch for inst in pm.instruments for n in inst.notes}


def _midi_cfg():
    return load_config().drums.midi


# --------------------------------------------------------------------------- #
# commit 1 — RED regression: the default 'other' bucket must yield cymbal MIDI
# --------------------------------------------------------------------------- #
def test_default_path_other_bucket_yields_cymbal_events(tmp_path):
    """Fails on the pre-fix tree (the 'other' cymbal bucket was dropped): the
    default-path drum MIDI must contain at least one cymbal event."""
    parts = {
        "kick": _write(tmp_path / "kick.wav", _place([
            (0.1, _tone(0.4, 80.0, 0.08)),
            (0.7, _tone(0.4, 80.0, 0.08)),
            (1.3, _tone(0.4, 80.0, 0.08)),
        ])),
        "other": _write(tmp_path / "other.wav", _cymbal_bucket()),
    }
    res = drum_midi._from_parts(parts, _midi_cfg(), ensure_dir(tmp_path / "out"), bpm=120.0)
    assert "file" in res, res
    got = _pitches(res["file"])
    assert CYMBAL_PITCHES & got, (
        f"expected at least one cymbal event from the default 'other' bucket, got {sorted(got)}")


# --------------------------------------------------------------------------- #
# commit 3 — canonicalization, classifier profiles, ambiguity, no-regression
# --------------------------------------------------------------------------- #
def test_canonical_drum_part_routes_other_and_survives_explicit():
    cdp = drum_midi.canonical_drum_part
    # the inagoy cymbal bucket -> generic cymbals super-class
    assert cdp("other") == "cymbals"
    assert cdp("platillos") == "cymbals"
    assert cdp("Cymbals") == "cymbals"
    # explicit UVR full-kit labels survive unchanged (not re-bucketed)
    assert cdp("hihat") == "hihat"
    assert cdp("Hi-Hat") == "hihat"
    assert cdp("ride") == "ride"
    assert cdp("crash") == "crash"
    # core hits + aliases
    assert cdp("kick") == "kick" and cdp("BD") == "kick" and cdp("bombo") == "kick"
    assert cdp("snare") == "snare" and cdp("caja") == "snare"
    assert cdp("Toms") == "toms"
    # unknown -> None (skipped, as before)
    assert cdp("mystery") is None
    assert cdp("") is None


def test_classify_cymbal_profiles():
    """Feature-dict unit tests: each clean profile lands on its GM class; weak
    evidence is rejected or flagged ambiguous — never fabricated."""
    cfg = _midi_cfg()
    clf = drum_midi._classify_cymbal
    strong = {"peak": 1.0, "hf_ratio": 0.6}
    assert clf({**strong, "decay": 0.05, "centroid": 9000}, cfg) == "hihat_closed"
    assert clf({**strong, "decay": 0.42, "centroid": 9000}, cfg) == "hihat_open"
    assert clf({**strong, "decay": 0.70, "centroid": 9000}, cfg) == "crash"
    assert clf({**strong, "decay": 0.70, "centroid": 4000}, cfg) == "ride"
    # ambiguous cymbal (mid decay, dark) -> None
    assert clf({**strong, "decay": 0.42, "centroid": 4000}, cfg) is None
    # gates: too quiet, or not enough HF content -> reject (not a cymbal)
    assert clf({"peak": 1e-6, "hf_ratio": 0.9, "decay": 0.4, "centroid": 9000}, cfg) == "reject"
    assert clf({"peak": 1.0, "hf_ratio": 0.05, "decay": 0.4, "centroid": 9000}, cfg) == "reject"


def test_cymbal_note_ambiguity_policy():
    cfg = _midi_cfg()
    assert drum_midi._cymbal_note("reject", cfg) is None            # gated -> no event
    assert drum_midi._cymbal_note("crash", cfg) == drum_midi.GM["crash"]
    # ambiguous (cls is None): generic note by default, dropped when configured
    assert drum_midi._cymbal_note(None, cfg) == drum_midi.CYMBAL_GENERIC_NOTE
    cfg.cymbal_ambiguous = "drop"
    assert drum_midi._cymbal_note(None, cfg) is None
    cfg.cymbal_ambiguous = "generic"
    cfg.cymbal_generic_note = 51
    assert drum_midi._cymbal_note(None, cfg) == 51


def test_hat_classifier_exercised_on_default_path(tmp_path):
    """A closed + open hi-hat in the 'other' bucket produce distinct GM classes,
    proving the open/closed classifier now runs on the default path."""
    other = _place([
        (0.1, _decay_noise(0.5, 0.020, 200)),   # closed
        (0.7, _decay_noise(0.5, 0.200, 201)),   # open
    ])
    parts = {"other": _write(tmp_path / "other.wav", other)}
    res = drum_midi._from_parts(parts, _midi_cfg(), ensure_dir(tmp_path / "out"), bpm=120.0)
    classes = res["cymbal_classes"]
    assert classes.get("hihat_closed", 0) >= 1
    assert classes.get("hihat_open", 0) >= 1
    got = _pitches(res["file"])
    assert drum_midi.GM["hihat_closed"] in got and drum_midi.GM["hihat_open"] in got


def test_kick_snare_toms_unregressed(tmp_path):
    """The core parts still map 1:1 to their GM notes and emit no cymbal pitches."""
    parts = {
        "kick": _write(tmp_path / "kick.wav", _place([(0.1, _tone(0.4, 70, 0.08)),
                                                      (0.9, _tone(0.4, 70, 0.08))])),
        "snare": _write(tmp_path / "snare.wav", _place([(0.5, _tone(0.4, 200, 0.08)),
                                                        (1.3, _tone(0.4, 200, 0.08))])),
        "toms": _write(tmp_path / "toms.wav", _place([(0.3, _tone(0.4, 120, 0.10))])),
    }
    res = drum_midi._from_parts(parts, _midi_cfg(), ensure_dir(tmp_path / "out"), bpm=120.0)
    got = _pitches(res["file"])
    assert drum_midi.GM["kick"] in got
    assert drum_midi.GM["snare"] in got
    assert drum_midi.GM["toms"] in got
    assert not (CYMBAL_PITCHES & got)              # no cymbals invented from tonal drums
    assert res["cymbal_classes"] == {}             # no cymbal bucket present
    assert res["cymbal_rejected"] == 0


def test_residual_noise_does_not_flood(tmp_path):
    """A quiet, cymbal-less 'other' bed must not flood the MIDI: the energy/HF
    gate rejects it, so every emitted note comes from the loud kick part."""
    quiet = _decay_noise(2.0, 5.0, 300, amp=2e-4)  # near-steady, well below the energy floor
    parts = {
        "kick": _write(tmp_path / "kick.wav", _place([(0.1, _tone(0.4, 80, 0.08)),
                                                      (0.7, _tone(0.4, 80, 0.08)),
                                                      (1.3, _tone(0.4, 80, 0.08))])),
        "other": _write(tmp_path / "other.wav", quiet),
    }
    res = drum_midi._from_parts(parts, _midi_cfg(), ensure_dir(tmp_path / "out"), bpm=120.0)
    assert res["cymbal_classes"] == {}                      # nothing voiced from the noise bed
    assert res["note_count"] == res["onsets_per_part"]["kick"]  # only kick events survive
    assert not (CYMBAL_PITCHES & _pitches(res["file"]))


def test_manifest_reports_cymbal_counts(tmp_path):
    """The manifest carries per-class and rejected cymbal counts; the per-class
    counts sum to the number of cymbal events voiced."""
    parts = {"other": _write(tmp_path / "other.wav", _cymbal_bucket())}
    res = drum_midi._from_parts(parts, _midi_cfg(), ensure_dir(tmp_path / "out"), bpm=120.0)
    assert isinstance(res["cymbal_classes"], dict)
    assert isinstance(res["cymbal_rejected"], int)
    assert sum(res["cymbal_classes"].values()) >= 1
    voiced = sum(res["cymbal_classes"].values())
    assert voiced + res["cymbal_rejected"] == res["onsets_per_part"]["other"]
    assert all(k in {"hihat_closed", "hihat_open", "ride", "crash", "generic"}
               for k in res["cymbal_classes"])


def test_stock_config_needs_no_new_fields(tmp_path):
    """Prior configs parse unchanged: the stock DrumMidiCfg has no cymbal_* fields,
    yet the cymbal path runs with module defaults (getattr fallbacks)."""
    cfg = _midi_cfg()
    assert not hasattr(cfg, "cymbal_ambiguous")     # not a dataclass field
    assert not hasattr(cfg, "cymbal_generic_note")
    parts = {"other": _write(tmp_path / "other.wav", _cymbal_bucket())}
    res = drum_midi._from_parts(parts, cfg, ensure_dir(tmp_path / "out"), bpm=120.0)
    assert sum(res["cymbal_classes"].values()) >= 1  # defaults -> generic policy, still voices
