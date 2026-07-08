"""StemForge — local, GPU-accelerated audio workstation.

Four capabilities over one shared analysis pass:
  1. Stem separation      (Demucs htdemucs_ft / _6s)
  2. Melodic MIDI          (Basic Pitch via ONNX Runtime)
  3. BPM time-stretch      (Rubber Band, pitch-preserved)
  4. Drum decomposition    (DrumSep) + drum MIDI (ADTOF, 7-class + velocity)

Heavy frameworks (torch, demucs, onnxruntime, gradio) are imported lazily inside
the modules that need them, so `import stemforge` works on any machine — including
CI without a GPU. See README §Install for the CUDA-first install order.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .orchestrator import RunConfig, Pipeline, load_config  # noqa: E402
from .io_utils import AudioTensor  # noqa: E402

__all__ = ["__version__", "RunConfig", "Pipeline", "load_config", "AudioTensor"]
