"""§5.8 orchestrator — config model, pipeline DAG, and the per-song manifest.

One analysis pass fills a shared :class:`RunContext`; every stage reads the same
beat grid so stems, MIDI, and stretched audio stay grid-aligned. Stages fail
*soft*: a missing external model records ``{"skipped": reason}`` in the manifest
instead of crashing the whole run, so Phases 1-2 work before 3-5 are wired.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import traceback
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from . import __version__
from .io_utils import (
    AudioTensor,
    PathLike,
    ensure_dir,
    receipt,
    save_audio,
    sha256_file,
    slugify,
    write_json,
)

_CONFIG_DEFAULT = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


# --------------------------------------------------------------------------- #
# Config model (mirrors configs/default.yaml)
# --------------------------------------------------------------------------- #
@dataclass
class IngestCfg:
    normalize: str = "replaygain"
    target_lufs: float = -18.0


@dataclass
class AnalysisCfg:
    enabled: bool = True
    engine: str = "beat_this"
    bpm_from_regression: bool = True


BS_ROFORMER_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"


# Verified multi-model ensemble sets from audio-separator's own
# ensemble_presets.json (all four filenames are keys in its models.json registry).
# Run via the CLI's native ensemble (`-m primary --extra_models ... --ensemble_algorithm`).
ENSEMBLE_VOCALS = [  # preset 'vocal_balanced'
    "bs_roformer_vocals_resurrection_unwa.ckpt",
    "melband_roformer_big_beta6x.ckpt",
]
ENSEMBLE_INSTRUMENTAL = [  # preset 'instrumental_full' — inst SDR ~17.55
    "melband_roformer_inst_v1e_plus.ckpt",
    "mel_band_roformer_instrumental_becruily.ckpt",
]


@dataclass
class SeparationCfg:
    enabled: bool = True
    backend: str = "demucs"  # demucs | uvr (audio-separator) | hybrid (uvr vocals + demucs residual)
    preset: str | None = None  # fast | best | sota | max — see SEPARATION_PRESETS
    model: str = "htdemucs_ft"
    uvr_model: str = BS_ROFORMER_MODEL
    uvr_model_dir: str = "models/uvr"
    uvr_venv: str = ".venv-uvr"  # isolated audio-separator venv (see `stemforge setup-sota`)
    uvr_use_autocast: bool = True
    # Multi-model ensemble (audio-separator's native ensemble, in the venv subprocess).
    ensemble_enabled: bool = False
    ensemble_models: list[str] = field(default_factory=list)
    uvr_ensemble_algorithm: str = "uvr_max_spec"  # max-spec merge
    stems: list[str] = field(default_factory=lambda: ["vocals", "drums", "bass", "other"])
    segment: float = 10.0
    overlap: float = 0.25
    shifts: int = 1
    export_formats: list[str] = field(default_factory=lambda: ["wav"])
    no_cuda_memory_caching: bool = False


# Quality presets: named bundles of separation settings. load_config layers a
# preset over the YAML values but under explicit dotted overrides, so
# `--set separation.shifts=1` still beats `--preset max`.
SEPARATION_PRESETS: dict[str, dict[str, Any]] = {
    # quick draft: plain htdemucs, single pass
    "fast": {"backend": "demucs", "model": "htdemucs", "shifts": 1},
    # best Demucs quality: fine-tuned bag + test-time augmentation
    "best": {"backend": "demucs", "model": "htdemucs_ft", "shifts": 2},
    # SOTA 2-stem: BS-Roformer vocals/instrumental via audio-separator
    "sota": {
        "backend": "uvr",
        "uvr_model": BS_ROFORMER_MODEL,
        "stems": ["vocals", "instrumental"],
    },
    # Max 4-stem: a real multi-model vocal ensemble (uvr_max_spec) for the
    # vocals/instrumental split + htdemucs_ft on the instrumental residual.
    # Falls back to demucs 4-stem if the venv/ensemble isn't available.
    "max": {
        "backend": "hybrid",
        "model": "htdemucs_ft",
        "uvr_model": BS_ROFORMER_MODEL,          # single-model fallback primary
        "ensemble_enabled": True,
        "ensemble_models": list(ENSEMBLE_VOCALS),
        "uvr_ensemble_algorithm": "uvr_max_spec",
        "shifts": 2,
        "stems": ["vocals", "drums", "bass", "other"],
    },
}


def preset_names() -> list[str]:
    """Preset names in quality order (for CLI/UI choices)."""
    return list(SEPARATION_PRESETS)


# Optional backend deps: missing one is a fail-soft skip (with an install hint);
# any other ModuleNotFoundError is a real error and surfaces via the stage trace.
# audio-separator is NOT here — it is never imported in-process; its isolated
# venv is preflighted by separate_uvr and reported via SotaEnvError.
_SEPARATION_DEP_HINTS = {
    "demucs": "pip install demucs",
    "torch": "install CUDA-matched torch first (see README)",
}


@dataclass
class StretchCfg:
    enabled: bool = False
    target_bpm: float | None = None
    engine: str = "rubberband"
    crisp: int = 5
    preserve_formant: bool = True
    per_stem_target_bpm: dict[str, float] = field(default_factory=dict)


@dataclass
class MidiCfg:
    enabled: bool = False
    engine: str = "onnx"
    onnx_model_path: str = "models/basic_pitch.onnx"
    stems: list[str] = field(default_factory=lambda: ["bass", "other", "guitar", "piano"])
    monophonic_stems: list[str] = field(default_factory=lambda: ["bass"])
    onset_threshold: float = 0.5
    frame_threshold: float = 0.3
    min_note_length_ms: float = 58.0
    min_frequency: float = 32.7
    max_frequency: float = 3000.0
    melodia_trick: bool = True
    include_pitch_bends: bool = True
    quantize_to_grid: bool = False


@dataclass
class DrumSplitCfg:
    enabled: bool = False
    # demucs_inagoy (main-env Demucs, default) | uvr_drumsep (MDX23C via .venv-uvr)
    # | uvr (audio-separator kit-isolation, discovery) | larsnet | external
    backend: str = "demucs_inagoy"
    model: str = "drumsep_roformer"  # legacy label for the larsnet/external paths
    parts: list[str] = field(default_factory=lambda: ["kick", "snare", "toms", "other"])
    from_input: bool = False  # tear down the raw input loop instead of the separated drums stem
    # inagoy/drumsep (default): a Demucs checkpoint run in the MAIN env, auto-downloaded.
    # Public, non-gated mirror of the exact inagoy checkpoint (MIT); the old
    # mnstrmnl/drumsep URL is now gated and returns HTTP 401.
    inagoy_url: str = "https://huggingface.co/Eddycrack864/Drumsep/resolve/main/modelo_final.th"
    inagoy_model_dir: str = "models/drumsep"
    # sha256 integrity pin for the inagoy checkpoint (audit finding C1). Empty =>
    # the built-in default pin (drum_split.INAGOY_DEFAULT_SHA256) is honored *only*
    # for the default URL; a custom inagoy_url without a matching inagoy_sha256 is
    # refused at load time (fail-closed). See drum_split._expected_inagoy_sha256.
    inagoy_sha256: str = ""
    # uvr_drumsep: MDX23C per-hit DrumSep via the isolated .venv-uvr (registry-listed;
    # not the default because its .ckpt couldn't be download-verified in-env).
    uvr_drumsep_model: str = "MDX23C-DrumSep-aufr33-jarredou.ckpt"
    # `uvr` path (isolated venv) — auto-discovers a kit-isolation model, not per-hit.
    uvr_model: str | None = None
    uvr_model_dir: str = "models/uvr"
    uvr_venv: str = ".venv-uvr"
    uvr_use_autocast: bool = True
    larsnet_wiener: float = 1.0
    external_cmd: str | None = None


@dataclass
class DrumMidiCfg:
    enabled: bool = False
    adt_model: str = "adtof"
    expand_to_7class: bool = True
    velocity_from_stems: bool = True
    velocity_window_ms: float = 50.0
    velocity_hop_ms: float = 10.0
    external_cmd: str | None = None


@dataclass
class DrumsCfg:
    split: DrumSplitCfg = field(default_factory=DrumSplitCfg)
    midi: DrumMidiCfg = field(default_factory=DrumMidiCfg)


@dataclass
class OutputCfg:
    root: str = "out"
    write_manifest: bool = True


@dataclass
class RunConfig:
    device: str = "auto"
    sample_rate: int = 44100
    channels: int = 2
    seed: int = 0
    ingest: IngestCfg = field(default_factory=IngestCfg)
    analysis: AnalysisCfg = field(default_factory=AnalysisCfg)
    separation: SeparationCfg = field(default_factory=SeparationCfg)
    stretch: StretchCfg = field(default_factory=StretchCfg)
    midi: MidiCfg = field(default_factory=MidiCfg)
    drums: DrumsCfg = field(default_factory=DrumsCfg)
    output: OutputCfg = field(default_factory=OutputCfg)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------- #
# Config loading (YAML -> nested dataclasses, with dotted overrides)
# --------------------------------------------------------------------------- #
def load_config(path: PathLike | None = None, overrides: dict[str, Any] | None = None) -> RunConfig:
    """Load default.yaml (or `path`), deep-merge `overrides`, build a RunConfig.

    `overrides` accepts dotted keys, e.g. {"separation.model": "htdemucs_6s"}.
    Precedence (low to high): yaml < separation preset < dotted overrides, so
    `--set separation.shifts=1` still wins over `--preset max`.
    """
    src = Path(path) if path else _CONFIG_DEFAULT
    data: dict[str, Any] = {}
    if src.is_file():
        data = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
    overrides = dict(overrides or {})
    preset = overrides.get("separation.preset", (data.get("separation") or {}).get("preset"))
    if preset:
        key = str(preset).strip().lower()
        if key not in SEPARATION_PRESETS:
            raise ValueError(
                f"unknown separation preset {preset!r}; choose from {sorted(SEPARATION_PRESETS)}"
            )
        sep = data.get("separation") or {}
        data["separation"] = sep
        for name, value in SEPARATION_PRESETS[key].items():
            sep[name] = list(value) if isinstance(value, list) else value
        sep["preset"] = key
        if "separation.preset" in overrides:
            overrides["separation.preset"] = key  # normalized for the manifest
    for dotted, value in overrides.items():
        _set_dotted(data, dotted, value)
    return _build(RunConfig, data)


def _set_dotted(d: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = d
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def _build(dc_type: type, data: Any) -> Any:
    """Recursively construct a (possibly nested) dataclass from a dict.

    Nested dataclass fields are detected via their ``default_factory`` (every
    nested config field defines one). This avoids resolving annotation strings,
    which are fragile under ``from __future__ import annotations``.
    """
    if not is_dataclass(dc_type):
        return data
    if data is None:
        return dc_type()
    if not isinstance(data, dict):
        raise TypeError(f"expected mapping for {dc_type.__name__}, got {type(data)}")

    field_names: set[str] = set()
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(dc_type):
        field_names.add(f.name)
        if f.name not in data:
            continue
        nested = _nested_dataclass_type(f)
        if nested is not None:
            kwargs[f.name] = _build(nested, data[f.name])
        else:
            kwargs[f.name] = data[f.name]
    unknown = set(data) - field_names
    if unknown:
        raise ValueError(f"{dc_type.__name__}: unknown config keys {sorted(unknown)}")
    return dc_type(**kwargs)


def _nested_dataclass_type(f) -> type | None:
    """Dataclass type of a field (via its default_factory), else None."""
    df = getattr(f, "default_factory", dataclasses.MISSING)
    if df is dataclasses.MISSING:
        return None
    try:
        sample = df()
    except Exception:
        return None
    if is_dataclass(sample) and not isinstance(sample, type):
        return type(sample)
    return None


# --------------------------------------------------------------------------- #
# Shared run context
# --------------------------------------------------------------------------- #
@dataclass
class RunContext:
    input_path: Path
    song: str
    out_dir: Path
    audio: AudioTensor
    device: str
    manifest: dict[str, Any] = field(default_factory=dict)
    beat_grid: Any = None
    stems: dict[str, Path] = field(default_factory=dict)

    def subdir(self, name: str) -> Path:
        return ensure_dir(self.out_dir / name)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
class Pipeline:
    """Runs the four capabilities over one shared analysis pass.

    Models are cached on the instance so a batch of files does not reload weights.
    """

    def __init__(self, cfg: RunConfig, model_cache: dict[str, Any] | None = None,
                 on_stage=None):
        self.cfg = cfg
        # Keyed "<backend>:<model name>" so a shared cache (e.g. across UI runs
        # with different presets) can never hand back the wrong weights.
        self._model_cache: dict[str, Any] = model_cache if model_cache is not None else {}
        # Optional callback(stage_key, enabled) fired as each stage starts — the
        # web UI maps it to progress. Never affects results; failures are ignored.
        self._on_stage = on_stage

    def resolve_device(self) -> str:
        if self.cfg.device != "auto":
            return self.cfg.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def run(self, input_path: PathLike) -> dict[str, Any]:
        from . import ingest as ingest_mod

        cfg = self.cfg
        ipath = Path(input_path)
        song = slugify(ipath.stem)
        out_dir = ensure_dir(Path(cfg.output.root) / song)
        device = self.resolve_device()

        audio = ingest_mod.ingest(
            ipath,
            sample_rate=cfg.sample_rate,
            channels=cfg.channels,
            normalize_method=cfg.ingest.normalize,
            target_lufs=cfg.ingest.target_lufs,
        )

        ctx = RunContext(input_path=ipath, song=song, out_dir=out_dir, audio=audio, device=device)
        ctx.manifest = {
            "stemforge_version": __version__,
            "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "device": device,
            "input": {
                "path": str(ipath),
                "filename": ipath.name,
                "sha256": sha256_file(ipath) if ipath.is_file() else None,
                "duration_s": round(audio.duration_seconds, 3),
                "sample_rate": audio.sample_rate,
            },
            "config": cfg.to_dict(),
            "models": {},
        }

        self._stage(ctx, "analysis", self._run_analysis, cfg.analysis.enabled)
        self._stage(ctx, "separation", self._run_separation, cfg.separation.enabled)
        self._stage(ctx, "midi", self._run_midi, cfg.midi.enabled)
        self._stage(ctx, "drum_split", self._run_drum_split, cfg.drums.split.enabled)
        self._stage(ctx, "drum_midi", self._run_drum_midi, cfg.drums.midi.enabled)
        self._stage(ctx, "stretch", self._run_stretch, cfg.stretch.enabled)

        produced = [p for p in out_dir.rglob("*") if p.is_file() and p.name != "manifest.json"]
        ctx.manifest["receipt"] = receipt(produced)

        if cfg.output.write_manifest:
            write_json(out_dir / "manifest.json", ctx.manifest)
        return ctx.manifest

    def run_batch(self, paths: list[PathLike]) -> list[dict[str, Any]]:
        return [self.run(p) for p in paths]

    def _stage(self, ctx: RunContext, key: str, fn, enabled: bool) -> None:
        if self._on_stage is not None:
            try:
                self._on_stage(key, enabled)
            except Exception:  # noqa: BLE001 - progress must never break the run
                pass
        if not enabled:
            ctx.manifest.setdefault(key, {"skipped": "disabled"})
            return
        try:
            fn(ctx)
        except Exception as e:  # noqa: BLE001 - record, don't crash the run
            ctx.manifest[key] = {
                "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc(limit=3),
            }

    def _run_analysis(self, ctx: RunContext) -> None:
        from . import analysis

        grid = analysis.analyze(
            ctx.audio, engine=self.cfg.analysis.engine, device=ctx.device,
            bpm_from_regression=self.cfg.analysis.bpm_from_regression,
        )
        ctx.beat_grid = grid
        ctx.manifest["analysis"] = grid.to_dict()

    def _run_separation(self, ctx: RunContext) -> None:
        from .separate_uvr import SotaEnvError

        cfg = self.cfg.separation
        backend = (cfg.backend or "demucs").strip().lower()
        models_used: dict[str, str] = {}
        notes: list[str] = []
        try:
            if backend == "demucs":
                stems = self._separate_demucs(ctx, cfg, models_used)
            elif backend == "uvr":
                stems = self._separate_uvr(ctx, cfg, models_used)
            elif backend == "hybrid":
                stems = self._separate_hybrid(ctx, cfg, models_used, notes)
            else:
                raise ValueError(f"unknown separation backend {cfg.backend!r} (demucs | uvr | hybrid)")
        except SotaEnvError as e:
            # fail-soft: isolated audio-separator env not ready -> skip, never crash
            ctx.manifest["separation"] = {"skipped": str(e)}
            return
        except ModuleNotFoundError as e:
            root = (e.name or "").split(".")[0]
            hint = _SEPARATION_DEP_HINTS.get(root)
            if hint is None:  # not an optional backend dep -> real bug, let _stage record it
                raise
            ctx.manifest["separation"] = {"skipped": f"missing dependency {root!r} ({hint})"}
            return

        stems_dir = ctx.subdir("stems")
        written: dict[str, str] = {}
        for name, tensor in stems.items():
            for fmt in cfg.export_formats:
                path = save_audio(stems_dir / f"{name}.{fmt}", tensor)
                if fmt == "wav":
                    ctx.stems[name] = path
                written[f"{name}.{fmt}"] = str(path)
        ctx.manifest["separation"] = {
            "backend": backend,
            "preset": cfg.preset,
            "model": " + ".join(models_used.values()),
            "models": models_used,
            "files": written,
        }
        if notes:
            ctx.manifest["separation"]["notes"] = notes
        ctx.manifest["models"].update(models_used)

    def _separate_demucs(
        self, ctx: RunContext, cfg: SeparationCfg, models_used: dict[str, str],
        audio: AudioTensor | None = None, stems_subset: list[str] | None = None,
    ) -> dict[str, AudioTensor]:
        from . import separate

        dcfg = cfg if stems_subset is None else dataclasses.replace(cfg, stems=stems_subset)
        cache_key = f"demucs:{cfg.model}"
        stems, model = separate.separate(
            ctx.audio if audio is None else audio, dcfg,
            device=ctx.device, model=self._model_cache.get(cache_key),
        )
        self._model_cache[cache_key] = model
        models_used["demucs"] = cfg.model
        return stems

    def _separate_uvr(
        self, ctx: RunContext, cfg: SeparationCfg, models_used: dict[str, str],
    ) -> dict[str, AudioTensor]:
        from . import separate_uvr

        primary, extra, _algo = separate_uvr.ensemble_spec(cfg)
        label = " + ".join([primary, *extra])
        cache_key = f"uvr:{label}"
        stems, engine = separate_uvr.separate(
            ctx.audio, cfg, device=ctx.device, engine=self._model_cache.get(cache_key),
        )
        self._model_cache[cache_key] = engine
        models_used["uvr"] = label
        return stems

    def _separate_hybrid(
        self, ctx: RunContext, cfg: SeparationCfg, models_used: dict[str, str], notes: list[str],
    ) -> dict[str, AudioTensor]:
        """UVR (single or ensemble) vocals + demucs on the instrumental residual.

        If the UVR stage is unavailable (venv not set up, or a gated/failed model
        download), Max degrades to a full demucs 4-stem so it never regresses to
        an error — the demucs default is the safety net.
        """
        from .separate_uvr import SotaEnvError

        try:
            uvr_stems = self._separate_uvr(ctx, cfg, models_used)
        except (SotaEnvError, RuntimeError) as e:
            notes.append(f"uvr ensemble unavailable ({e}); fell back to demucs {cfg.model}")
            models_used.pop("uvr", None)
            return self._separate_demucs(ctx, cfg, models_used)
        instrumental = uvr_stems.get("instrumental")
        if instrumental is None:  # 4-stem UVR model: nothing left to refine
            notes.append("uvr model produced no instrumental stem; demucs refinement skipped")
            return uvr_stems
        try:
            residual = self._separate_demucs(
                ctx, cfg, models_used, audio=instrumental, stems_subset=["drums", "bass", "other"],
            )
        except ModuleNotFoundError as e:
            # don't discard the finished roformer pass just because demucs is absent
            notes.append(f"demucs unavailable ({e.name}); kept the 2-stem uvr result")
            return uvr_stems
        merged = {k: v for k, v in uvr_stems.items() if k != "instrumental"}
        merged.update({k: v for k, v in residual.items() if k != "vocals"})
        return merged

    def _run_midi(self, ctx: RunContext) -> None:
        from . import midi_melodic

        ctx.manifest["midi"] = midi_melodic.transcribe_stems(
            ctx.stems, self.cfg.midi, ctx.subdir("midi"), beat_grid=ctx.beat_grid,
        )

    def _run_drum_split(self, ctx: RunContext) -> None:
        from . import drum_split

        scfg = self.cfg.drums.split
        drums_path = ctx.stems.get("drums")
        # from_input tears down the raw input loop itself; otherwise use the
        # separated drums stem (drum_split fails soft if neither is available).
        source_audio = ctx.audio if scfg.from_input else None
        ctx.manifest["drum_split"] = drum_split.split(
            drums_path, scfg, ctx.subdir("drums"), device=ctx.device, audio=source_audio,
        )

    def _run_drum_midi(self, ctx: RunContext) -> None:
        from . import drum_midi

        ctx.manifest["drum_midi"] = drum_midi.transcribe(
            ctx.stems.get("drums"),
            drum_parts=ctx.manifest.get("drum_split", {}).get("files", {}),
            cfg=self.cfg.drums.midi,
            out_dir=ctx.subdir("drums"),
            beat_grid=ctx.beat_grid,
        )

    def _run_stretch(self, ctx: RunContext) -> None:
        from . import stretch as stretch_mod

        ctx.manifest["stretch"] = stretch_mod.stretch_stems(
            ctx.stems, self.cfg.stretch, ctx.subdir("stretched"), beat_grid=ctx.beat_grid,
        )
