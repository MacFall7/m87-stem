"""§5.6 drum_midi — drum transcription to a General-MIDI, velocity-aware .mid.

Two onset sources:
  1. **ADTOF** (external, CRNN 5-class) via ``external_cmd`` that emits a .mid we
     adopt directly — the SOTA path.
  2. **Parts-based** (no ADT model needed): when drum_split has produced per-part
     WAVs, we detect onsets per part and assign the correct GM note. This makes
     drum MIDI work off Phase-4 output alone.

Regardless of source, **velocity** comes from each part's RMS loudness envelope
(equal-ish weighting, configurable window/hop): the max RMS in a short window
after each onset, scaled to MIDI 1–127. The 5→7 expansion (open/closed hi-hat via
a decay test) is applied on the hi-hat part.

The default ``demucs_inagoy`` backend bundles every cymbal into one ``other``
stem. :func:`canonical_drum_part` routes that stem to a ``cymbals`` super-class
whose onsets are classified per-hit (spectral centroid, HF-energy ratio, decay,
transient/sustain) into closed/open hi-hat, ride, or crash — so the default path
emits cymbal MIDI instead of silently dropping it. Onsets that fail the energy or
high-frequency gate are rejected; genuinely ambiguous cymbals go to a configurable
generic note (or are dropped). Per-class and rejected counts are reported in the
manifest. Explicit ``hihat``/``ride``/``crash`` parts (UVR full-kit path) survive
unchanged.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import PathLike, load_audio

try:  # pragma: no cover
    from .orchestrator import DrumMidiCfg
except Exception:  # pragma: no cover
    DrumMidiCfg = Any  # type: ignore

# General MIDI percussion note numbers.
GM = {
    "kick": 36,
    "snare": 38,
    "hihat": 42,        # closed by default
    "hihat_closed": 42,
    "hihat_open": 46,
    "toms": 47,         # mid tom (default when tom-split is off)
    "toms_low": 45,     # low/floor tom
    "toms_mid": 47,
    "toms_high": 50,    # high tom
    "ride": 51,
    "crash": 49,
}

NOTE_LEN_S = 0.06


def transcribe(
    drums_path: PathLike | None,
    drum_parts: dict[str, str],
    cfg: "DrumMidiCfg",
    out_dir: PathLike,
    beat_grid: Any = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    # A real, measured tempo drives the .mid; when it's UNKNOWN (0.0) we still must
    # write *some* tempo to serialize the file, so we use 120 but flag it as a
    # serialization default (tempo_assumed) rather than passing it off as detected.
    src_bpm = float(getattr(beat_grid, "source_bpm", 0) or 0.0)
    tempo_assumed = src_bpm <= 0.0
    bpm = src_bpm if src_bpm > 0 else 120.0
    tempo_source = "beat_grid" if src_bpm > 0 else "midi_serialization_default"

    # 1) SOTA path: let ADTOF (or another ADT) write the MIDI, we adopt it.
    if cfg.external_cmd:
        return _adt_external(cfg.external_cmd, drums_path, out_dir)

    # 2) Parts-based path: build the MIDI from separated parts.
    if drum_parts:
        res = _from_parts(drum_parts, cfg, out_dir, bpm)
        if "file" in res:  # annotate a produced MIDI with its tempo provenance
            res["tempo_bpm"] = round(bpm, 3)
            res["tempo_source"] = tempo_source
            res["tempo_assumed"] = tempo_assumed
        return res

    return {
        "skipped": (
            "no drum parts and no ADT external_cmd. Enable drums.split (Phase 4) for "
            "the parts-based path, or set drums.midi.external_cmd to an ADTOF command."
        )
    }


# --------------------------------------------------------------------------- #
# ADT external adapter
# --------------------------------------------------------------------------- #
def _adt_external(cmd_template: str, drums_path: PathLike | None, out_dir: Path) -> dict[str, Any]:
    if drums_path is None or not Path(drums_path).is_file():
        return {"skipped": "no drums stem for ADT"}
    cmd = cmd_template.format(input=str(drums_path), output_dir=str(out_dir))
    try:
        subprocess.run(shlex.split(cmd), check=True, capture_output=True, text=True)
    except FileNotFoundError as e:
        return {"error": f"ADT external_cmd not found: {e}"}
    except subprocess.CalledProcessError as e:
        return {"error": f"ADT external_cmd failed: {e.stderr[-500:]}"}

    mids = list(out_dir.glob("*.mid"))
    if not mids:
        return {"error": "ADT external_cmd produced no .mid"}
    target = out_dir / "drums.mid"
    if mids[0] != target:
        shutil.copy(mids[0], target)
    return {"source": "adt_external", "file": str(target)}


# --------------------------------------------------------------------------- #
# Boundary adapter: normalize an incoming drum-part LABEL to a canonical part
# --------------------------------------------------------------------------- #
# The default ``demucs_inagoy`` backend bundles ALL cymbals into a single
# ``other`` stem (drum_split maps platillos/cymbals/hihat/ride/crash there), so a
# part labeled ``other``/``platillos`` maps to the ``cymbals`` super-class, which
# is classified per-onset below. Explicit ``hihat``/``ride``/``crash`` labels
# (e.g. from the UVR 6-stem path) survive as themselves. This adapter lives in
# drum_midi only — it renames nothing on disk and does not touch drum_split's own
# part mapping (no repo-wide rename).
_DRUM_PART_ALIASES: dict[str, tuple[str, ...]] = {
    "kick": ("kick", "bd", "bassdrum", "bass drum", "bombo"),
    "snare": ("snare", "sd", "caja", "redoblante"),
    "toms": ("toms", "tom"),
    "hihat": ("hihat", "hi-hat", "hh", "hat"),
    "ride": ("ride",),
    "crash": ("crash",),
    "cymbals": ("cymbals", "cymbal", "cym", "platillos", "other"),
}


def canonical_drum_part(part: str) -> str | None:
    """Map an incoming part label to a canonical drum part, or ``None`` if unknown.

    ``other``/``platillos`` (the inagoy cymbal bucket) → ``cymbals`` (per-onset
    classified); explicit ``hihat``/``ride``/``crash`` survive unchanged so the
    UVR full-kit path is never re-bucketed.
    """
    low = (part or "").strip().lower()
    if not low:
        return None
    for canon, aliases in _DRUM_PART_ALIASES.items():
        if low == canon or any(a == low for a in aliases):
            return canon
    for canon, aliases in _DRUM_PART_ALIASES.items():  # substring fallback (decorated labels)
        if any(a in low for a in aliases):
            return canon
    return None


# --------------------------------------------------------------------------- #
# Parts-based transcription (calibrated — PR-A4)
# --------------------------------------------------------------------------- #
def _from_parts(drum_parts: dict[str, str], cfg: "DrumMidiCfg", out_dir: Path, bpm: float) -> dict[str, Any]:
    import pretty_midi

    raw_events: list[dict[str, Any]] = []   # {time, part, note, raw_strength}
    counts: dict[str, int] = {}
    cymbal_classes: dict[str, int] = {}     # GM-class -> count (default-path evidence)
    cymbal_rejected = 0                      # onsets in a cymbal bucket that were gated/dropped

    for part, file in drum_parts.items():
        canon = canonical_drum_part(part)
        if canon is None:
            continue
        p = Path(file)
        if not p.is_file():
            continue
        audio = load_audio(p).to_mono()
        y = audio.samples[0]
        sr = audio.sample_rate
        onsets = _onsets(y, sr)
        env, env_t = _rms_env(y, sr, cfg.velocity_window_ms, cfg.velocity_hop_ms)
        counts[part] = len(onsets)

        def _peak(t: float) -> float:
            return _peak_in_window(env, env_t, t, cfg.velocity_window_ms / 1000.0) \
                if cfg.velocity_from_stems else 1.0

        for t in onsets:
            if canon == "cymbals":
                cls = _classify_cymbal(_cymbal_features(y, sr, float(t), cfg), cfg)
                note = _cymbal_note(cls, cfg)
                if note is None:            # gated (not a cymbal) or ambiguous-dropped
                    cymbal_rejected += 1
                    continue
                label = cls if cls in GM else "generic"
                cymbal_classes[label] = cymbal_classes.get(label, 0) + 1
            elif canon == "toms":
                note = _tom_note(y, sr, float(t), cfg)
            elif canon == "hihat":
                note = _hihat_note(y, sr, float(t), cfg)
            else:
                note = GM.get(canon, GM["snare"])
            raw_events.append({"time": float(t), "part": canon, "note": int(note),
                               "raw_strength": float(_peak(t))})

    if not raw_events:
        return {"skipped": "no onsets detected in drum parts"}

    _normalize_velocities(raw_events, cfg)                       # per-part, then kit balance
    kept, duplicates_removed = _dedupe_cross_part(raw_events, cfg)

    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    inst = pretty_midi.Instrument(program=0, is_drum=True, name="drums")
    records: list[dict[str, Any]] = []
    for e in sorted(kept, key=lambda x: x["time"]):
        vel = _velocity_from_norm(e["normalized_strength"]) if cfg.velocity_from_stems else 100
        inst.notes.append(pretty_midi.Note(velocity=vel, pitch=e["note"],
                                            start=e["time"], end=e["time"] + NOTE_LEN_S))
        records.append({
            "time": round(e["time"], 4), "part": e["part"], "note": e["note"],
            "raw_strength": round(e["raw_strength"], 6),
            "normalized_strength": round(e["normalized_strength"], 4),
            "velocity": vel,
        })
    pm.instruments.append(inst)

    target = out_dir / "drums.mid"
    pm.write(str(target))
    return {
        "source": "parts",
        "file": str(target),
        "note_count": len(kept),
        "onsets_per_part": counts,
        "expanded_7class": bool(cfg.expand_to_7class),
        "cymbal_classes": cymbal_classes,
        "cymbal_rejected": cymbal_rejected,
        "duplicates_removed": duplicates_removed,
        "events": records,
    }


# --------------------------------------------------------------------------- #
# Calibration: per-part velocity normalization, cross-part de-dup, part voicing
# --------------------------------------------------------------------------- #
def _normalize_velocities(events: list[dict[str, Any]], cfg: "DrumMidiCfg") -> None:
    """Normalize each part against its own ~98th-percentile strength (so a quiet
    hi-hat and a loud kick both use the full velocity range and ghost notes stay
    audible), then apply the kit-level balance gain. Adds ``normalized_strength``."""
    pct = float(getattr(cfg, "velocity_percentile", 98.0))
    balance = getattr(cfg, "kit_balance", {}) or {}
    by_part: dict[str, list[dict[str, Any]]] = {}
    for e in events:
        by_part.setdefault(e["part"], []).append(e)
    for part, evs in by_part.items():
        strengths = np.asarray([e["raw_strength"] for e in evs], dtype=np.float64)
        ref = float(np.percentile(strengths, pct)) if strengths.size else 1.0
        if ref <= 1e-9:
            ref = float(strengths.max()) if strengths.size else 1.0
        ref = ref if ref > 1e-9 else 1.0
        gain = float(balance.get(part, 1.0))
        for e in evs:
            e["normalized_strength"] = float(np.clip(e["raw_strength"] / ref * gain, 0.0, 1.0))


def _dedupe_cross_part(events: list[dict[str, Any]], cfg: "DrumMidiCfg",
                       ) -> tuple[list[dict[str, Any]], int]:
    """Cluster onsets within ``dedupe_window_ms`` and drop a CROSS-part hit that is
    clearly weaker than the loudest in the cluster (separation bleed) — keeping the
    most probable. Comparable simultaneous hits (a real kick+snare) and same-part
    ghost notes survive, since only clearly-weaker cross-part coincidences go."""
    if not events:
        return [], 0
    win = float(getattr(cfg, "dedupe_window_ms", 18.0)) / 1000.0
    ratio = float(getattr(cfg, "dedupe_weak_ratio", 0.5))
    order = sorted(events, key=lambda e: e["time"])
    kept: list[dict[str, Any]] = []
    removed = 0
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and order[j + 1]["time"] - order[i]["time"] <= win:
            j += 1
        cluster = order[i:j + 1]
        if len(cluster) == 1:
            kept.append(cluster[0])
        else:
            strongest = max(cluster, key=lambda e: e["raw_strength"])
            for e in cluster:
                if e is strongest:
                    kept.append(e)
                elif e["part"] != strongest["part"] and \
                        e["raw_strength"] < ratio * strongest["raw_strength"]:
                    removed += 1                      # separation bleed → drop
                else:
                    kept.append(e)                    # same-part or comparable → keep
        i = j + 1
    return kept, removed


def _velocity_from_norm(norm: float) -> int:
    """Normalized strength (0..1) -> MIDI velocity with a floor so ghosts stay audible."""
    ratio = float(np.clip(norm, 0.0, 1.0)) ** 0.5   # perceptual-ish compression
    return int(np.clip(round(10 + 117 * ratio), 1, 127))


def _hihat_note(y: np.ndarray, sr: int, t: float, cfg: "DrumMidiCfg") -> int:
    if cfg.expand_to_7class and _hat_is_open(y, sr, t, cfg):
        return GM["hihat_open"]
    return GM["hihat_closed"]


def _tom_note(y: np.ndarray, sr: int, t: float, cfg: "DrumMidiCfg") -> int:
    """Split toms into low/mid/high by the onset's dominant pitch."""
    if not getattr(cfg, "tom_split", True):
        return GM["toms"]
    hz = _dominant_hz(y, sr, t)
    if hz < float(getattr(cfg, "tom_low_hz", 100.0)):
        return GM["toms_low"]
    if hz < float(getattr(cfg, "tom_mid_hz", 220.0)):
        return GM["toms_mid"]
    return GM["toms_high"]


def _dominant_hz(y: np.ndarray, sr: int, t: float, win_ms: float = 60.0) -> float:
    """Dominant (peak-magnitude) frequency in a short window after the onset."""
    i0 = max(0, int(t * sr))
    seg = y[i0:i0 + max(1, int(win_ms / 1000.0 * sr))].astype(np.float64)
    if seg.size < 4:
        return 0.0
    spec = np.abs(np.fft.rfft(seg * np.hanning(seg.size)))
    freqs = np.fft.rfftfreq(seg.size, d=1.0 / sr)
    return float(freqs[int(np.argmax(spec))])


# --------------------------------------------------------------------------- #
# Cymbal classification — default inagoy path only (the "other" stem is all
# cymbals). Thresholds are module defaults, read via getattr(cfg, ...) so the
# drum-calibration contract (PR-A4) can move them into config without a schema
# change here. A hit that fails the energy or high-frequency gate is REJECTED
# (never voiced); a cymbal matching no clear profile is AMBIGUOUS and routed to a
# configurable generic note or dropped — confidence is never fabricated.
# --------------------------------------------------------------------------- #
CYMBAL_HF_CUT_HZ = 6000.0          # boundary for the high-frequency energy ratio
CYMBAL_MIN_HF_RATIO = 0.30         # below this an onset is bleed/tom spill, not a cymbal
CYMBAL_MIN_PEAK = 1e-3             # energy floor — quieter onsets are rejected (no flooding)
CYMBAL_ATTACK_MS = 30.0            # attack window for peak / centroid / HF ratio
CYMBAL_SUSTAIN_MS = 180.0          # offset at which the sustain RMS is measured (decay probe)
CYMBAL_CLOSED_DECAY_MAX = 0.30     # sustain/attack RMS <= this => fast decay => closed hi-hat
CYMBAL_CRASH_DECAY_MIN = 0.55      # >= this => long wash => ride (dark) / crash (bright)
CYMBAL_BRIGHT_CENTROID_HZ = 7000.0 # bright (hi-hat / crash) vs darker/tonal (ride) split
CYMBAL_GENERIC_NOTE = GM["hihat_closed"]  # 42 — safe generic when a cymbal is ambiguous


def _cfg_num(cfg: "DrumMidiCfg", name: str, default: float) -> float:
    v = getattr(cfg, name, default)
    return default if v is None else float(v)


def _cymbal_features(y: np.ndarray, sr: int, t: float, cfg: "DrumMidiCfg") -> dict[str, float]:
    """Spectral + temporal features of one onset in a cymbal stem:
    ``peak`` (attack amplitude), ``centroid`` (Hz), ``hf_ratio`` (energy above the
    HF cut / total), ``decay`` (sustain-RMS / attack-RMS — higher = longer wash)."""
    attack_ms = _cfg_num(cfg, "cymbal_attack_ms", CYMBAL_ATTACK_MS)
    sustain_ms = _cfg_num(cfg, "cymbal_sustain_ms", CYMBAL_SUSTAIN_MS)
    hf_cut = _cfg_num(cfg, "cymbal_hf_cut_hz", CYMBAL_HF_CUT_HZ)

    i0 = max(0, int(t * sr))
    n = max(1, int(attack_ms / 1000.0 * sr))
    seg = y[i0:i0 + n].astype(np.float64)
    if seg.size < 2:
        return {"peak": 0.0, "centroid": 0.0, "hf_ratio": 0.0, "decay": 0.0}

    peak = float(np.max(np.abs(seg)))
    spec = np.abs(np.fft.rfft(seg * np.hanning(seg.size)))
    freqs = np.fft.rfftfreq(seg.size, d=1.0 / sr)
    total = float(spec.sum()) + 1e-12
    centroid = float((freqs * spec).sum() / total)
    hf_ratio = float(spec[freqs >= hf_cut].sum() / total)

    a = _rms_at(y, i0, n)
    b = _rms_at(y, i0 + int(sustain_ms / 1000.0 * sr), n)
    decay = float(b / a) if a > 1e-9 else 0.0
    return {"peak": peak, "centroid": centroid, "hf_ratio": hf_ratio, "decay": decay}


def _classify_cymbal(feat: dict[str, float], cfg: "DrumMidiCfg") -> str | None:
    """Map cymbal-onset features to a GM cymbal class.

    Returns one of ``hihat_closed``/``hihat_open``/``ride``/``crash``; the sentinel
    ``"reject"`` when the onset fails the energy/HF gate (not a cymbal); or ``None``
    when it is a cymbal but matches no clear profile (ambiguous — the caller applies
    the generic-or-drop policy). No class is ever fabricated on weak evidence.
    """
    min_peak = _cfg_num(cfg, "cymbal_min_peak", CYMBAL_MIN_PEAK)
    min_hf = _cfg_num(cfg, "cymbal_min_hf_ratio", CYMBAL_MIN_HF_RATIO)
    closed_max = _cfg_num(cfg, "cymbal_closed_decay_max", CYMBAL_CLOSED_DECAY_MAX)
    crash_min = _cfg_num(cfg, "cymbal_crash_decay_min", CYMBAL_CRASH_DECAY_MIN)
    bright = _cfg_num(cfg, "cymbal_bright_centroid_hz", CYMBAL_BRIGHT_CENTROID_HZ)

    if feat["peak"] < min_peak or feat["hf_ratio"] < min_hf:
        return "reject"
    decay, centroid = feat["decay"], feat["centroid"]
    if decay <= closed_max:
        return "hihat_closed"
    if decay >= crash_min:
        return "ride" if centroid < bright else "crash"
    if centroid >= bright:
        return "hihat_open"
    return None  # ambiguous cymbal -> generic-or-drop


def _cymbal_note(cls: str | None, cfg: "DrumMidiCfg") -> int | None:
    """Resolve a cymbal class to a GM note, honoring the ambiguity policy.

    ``reject`` → ``None`` (gated, no event). A concrete class → its GM note. An
    ambiguous cymbal (``cls is None``) → a configurable generic note, or ``None``
    when ``cfg.cymbal_ambiguous == "drop"``.
    """
    if cls == "reject":
        return None
    if cls in GM:
        return GM[cls]
    policy = str(getattr(cfg, "cymbal_ambiguous", "generic") or "generic").strip().lower()
    if policy == "drop":
        return None
    return int(_cfg_num(cfg, "cymbal_generic_note", CYMBAL_GENERIC_NOTE))


# --------------------------------------------------------------------------- #
# DSP helpers
# --------------------------------------------------------------------------- #
def _onsets(y: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    return librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True)


def _rms_env(y: np.ndarray, sr: int, window_ms: float, hop_ms: float) -> tuple[np.ndarray, np.ndarray]:
    import librosa

    frame = max(256, int(sr * window_ms / 1000.0))
    hop = max(64, int(sr * hop_ms / 1000.0))
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    return rms, times


def _peak_in_window(env: np.ndarray, env_t: np.ndarray, t: float, window_s: float) -> float:
    mask = (env_t >= t) & (env_t <= t + window_s)
    if not np.any(mask):
        idx = int(np.argmin(np.abs(env_t - t)))
        return float(env[idx])
    return float(np.max(env[mask]))


def _hat_is_open(y: np.ndarray, sr: int, t: float, cfg: "DrumMidiCfg" = None) -> bool:
    """Open vs closed hi-hat via decay: open hats sustain, closed die fast.

    Windows and the decay ratio are configurable (PR-A4); the defaults reproduce
    the previous hard-coded behavior."""
    short_ms = _cfg_num(cfg, "hat_open_short_ms", 25.0) if cfg is not None else 25.0
    long_ms = _cfg_num(cfg, "hat_open_long_ms", 130.0) if cfg is not None else 130.0
    ratio = _cfg_num(cfg, "hat_open_ratio", 0.35) if cfg is not None else 0.35
    i0 = int(t * sr)
    a = _rms_at(y, i0, int(short_ms / 1000.0 * sr))
    b = _rms_at(y, i0 + int(long_ms / 1000.0 * sr), int(short_ms / 1000.0 * sr))
    if a <= 1e-9:
        return False
    return (b / a) > ratio  # still ringing well after the hit => open


def _rms_at(y: np.ndarray, start: int, length: int) -> float:
    seg = y[max(0, start):max(0, start) + max(1, length)]
    if seg.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))
