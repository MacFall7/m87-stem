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


@dataclass
class SeparationCfg:
    enabled: bool = True
    model: str = "htdemucs_ft"
    stems: list[str] = field(default_factory=lambda: ["vocals", "drums", "bass", "other"])
    segment: float = 10.0
    overlap: float = 0.25
    shifts: int = 1
    export_formats: list[str] = field(default_factory=lambda: ["wav"])
    no_cuda_memory_caching: bool = False


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
    model: str = "drumsep_roformer"
    parts: list[str] = field(default_factory=lambda: ["kick", "snare", "toms", "hihat", "ride", "crash"])
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
    """
    src = Path(path) if path else _CONFIG_DEFAULT
    data: dict[str, Any] = {}
    if src.is_file():
        data = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
    if overrides:
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

    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self._model_cache: dict[str, Any] = {}

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
        from . import separate

        model = self._model_cache.get("demucs")
        stems, model = separate.separate(ctx.audio, self.cfg.separation, device=ctx.device, model=model)
        self._model_cache["demucs"] = model
        stems_dir = ctx.subdir("stems")
        written: dict[str, str] = {}
        for name, tensor in stems.items():
            for fmt in self.cfg.separation.export_formats:
                path = save_audio(stems_dir / f"{name}.{fmt}", tensor)
                if fmt == "wav":
                    ctx.stems[name] = path
                written[f"{name}.{fmt}"] = str(path)
        ctx.manifest["separation"] = {"model": self.cfg.separation.model, "files": written}
        ctx.manifest["models"]["demucs"] = self.cfg.separation.model

    def _run_midi(self, ctx: RunContext) -> None:
        from . import midi_melodic

        ctx.manifest["midi"] = midi_melodic.transcribe_stems(
            ctx.stems, self.cfg.midi, ctx.subdir("midi"), beat_grid=ctx.beat_grid,
        )

    def _run_drum_split(self, ctx: RunContext) -> None:
        from . import drum_split

        ctx.manifest["drum_split"] = drum_split.split(
            ctx.stems.get("drums"), self.cfg.drums.split, ctx.subdir("drums"), device=ctx.device,
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
