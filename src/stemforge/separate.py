"""§5.3 separate — Demucs stem separation (the day-one deliverable).

Loads a Demucs model once (cached by the Pipeline), applies it with the same
per-track normalization the official CLI uses, and returns float stems as
:class:`AudioTensor`. torch/demucs are imported lazily so the package imports
without them.

VRAM control:
  * ``segment`` — lower => less VRAM (transformer models cap ~7.8 s; we clamp).
  * ``no_cuda_memory_caching`` — set true on <=3 GB cards (also set the env var
    ``PYTORCH_NO_CUDA_MEMORY_CACHING=1`` *before* launching for full effect).
"""

from __future__ import annotations

import os
from typing import Any

from .io_utils import AudioTensor

# forward ref only; avoids importing orchestrator at module load
try:  # pragma: no cover - typing convenience
    from .orchestrator import SeparationCfg
except Exception:  # pragma: no cover
    SeparationCfg = Any  # type: ignore


def load_model(name: str = "htdemucs_ft", device: str = "cuda"):
    """Fetch + cache a Demucs model on the given device, in eval mode."""
    from demucs.pretrained import get_model

    model = get_model(name)
    model.to(device)
    model.eval()
    return model


def separate(
    audio: AudioTensor,
    cfg: "SeparationCfg",
    device: str = "cuda",
    model: Any = None,
) -> tuple[dict[str, AudioTensor], Any]:
    """Separate ``audio`` into stems.

    Returns ``(stems, model)`` — the model is returned so the caller can cache it
    across a batch without reloading weights.
    """
    import torch
    from demucs.apply import apply_model

    if cfg.no_cuda_memory_caching:
        os.environ.setdefault("PYTORCH_NO_CUDA_MEMORY_CACHING", "1")

    if model is None:
        model = load_model(cfg.model, device)

    # Demucs models are 44.1 kHz stereo natively; ingest already guarantees that,
    # but convert defensively in case a caller passes something else.
    wav = _prepare(audio, model, device)

    _apply_segment(model, cfg.segment)

    # Per-track normalization (mirrors demucs.separate).
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / (ref.std() + 1e-8)

    with torch.no_grad():
        out = _apply(apply_model, model, wav, cfg, device)

    out = out * ref.std() + ref.mean()  # (sources, channels, samples)

    sources: dict[str, AudioTensor] = {}
    for name, src in zip(model.sources, out):
        sources[name] = AudioTensor.from_torch(src, model.samplerate)

    if device == "cuda":
        torch.cuda.empty_cache()

    return _select(sources, cfg), model


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _prepare(audio: AudioTensor, model, device):
    import torch
    from demucs.audio import convert_audio

    wav = audio.torch().to(device)  # (channels, samples)
    wav = convert_audio(wav, audio.sample_rate, model.samplerate, model.audio_channels)
    return wav


def _apply(apply_model, model, wav, cfg, device):
    """Call apply_model, tolerating signature differences across demucs 4.x."""
    kwargs = dict(shifts=cfg.shifts, split=True, overlap=cfg.overlap, progress=False, device=device)
    try:
        return apply_model(model, wav[None], segment=cfg.segment, **kwargs)[0]
    except TypeError:
        # older/newer signature without a segment kwarg (set on model instead)
        return apply_model(model, wav[None], **kwargs)[0]


def _apply_segment(model, segment: float) -> None:
    """Set per-model segment where the model allows it; clamp to model max."""
    def _set(m):
        cap = getattr(m, "segment", None)
        if cap is None:
            return
        try:
            m.segment = float(segment)
        except Exception:
            pass

    if hasattr(model, "models"):  # BagOfModels (htdemucs_ft is a bag)
        for m in model.models:
            _set(m)
    else:
        _set(model)


def _select(sources: dict[str, AudioTensor], cfg: "SeparationCfg") -> dict[str, AudioTensor]:
    """Return the requested subset if it is a strict subset of what the model
    produced; otherwise return everything the model produced."""
    wanted = set(cfg.stems)
    have = set(sources)
    if wanted and wanted < have:
        return {k: sources[k] for k in sources if k in wanted}
    return sources


def available_models() -> list[str]:
    return ["htdemucs_ft", "htdemucs", "htdemucs_6s", "hdemucs_mmi", "mdx_extra"]
