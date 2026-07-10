"""C5 (EC-SF-H2v2) — concurrency safety.

R2: each UVR invocation writes into its own fresh dir (no shared-scratch snapshot
    attribution). R3: the torch.load weights_only relaxation is a LOCKED,
    self-restoring scope, not a leaky process-global monkeypatch.
GPU-free: the audio-separator CLI and torch are mocked.
"""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import soundfile as sf

from stemforge import drum_split, separate_uvr
from stemforge.io_utils import AudioTensor
from stemforge.orchestrator import load_config


def _sine() -> AudioTensor:
    sr = 44100
    y = 0.2 * np.sin(2 * np.pi * 220 * np.linspace(0, 0.3, int(sr * 0.3), endpoint=False))
    return AudioTensor(np.stack([y, y]).astype(np.float32), sr)


# --------------------------------------------------------------------------- #
# R2 — per-invocation UVR work dir (no snapshot-diff attribution)
# --------------------------------------------------------------------------- #
def test_uvr_run_uses_unique_per_invocation_dir(monkeypatch):
    monkeypatch.setattr(separate_uvr, "have_ffmpeg", lambda: True)
    monkeypatch.setattr(separate_uvr, "find_cli",
                        lambda venv_dir: Path(venv_dir) / "bin" / "audio-separator")
    seen_dirs: list[Path] = []

    def fake_run(cmd, capture_output=False, text=False, check=False, **kw):
        args = {cmd[i]: cmd[i + 1] for i in range(len(cmd) - 1)}
        out_dir = Path(args["--output_dir"])
        seen_dirs.append(out_dir)
        src = Path(cmd[1])
        data, sr = sf.read(str(src), always_2d=True)
        for token in ("Vocals", "Instrumental"):
            sf.write(str(out_dir / f"{src.stem}_({token})_m.wav"), data * 0.5, sr)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(separate_uvr.subprocess, "run", fake_run)

    cfg = load_config(overrides={"separation.preset": "sota"}).separation
    _, engine = separate_uvr.separate(_sine(), cfg, device="cpu")
    separate_uvr.separate(_sine(), cfg, device="cpu", engine=engine)

    # each invocation used a DISTINCT call- subdir under the engine's work dir
    assert len(seen_dirs) == 2 and seen_dirs[0] != seen_dirs[1]
    assert all(d.name.startswith("call-") and d.parent == engine.work_dir for d in seen_dirs)
    # scratch fully cleaned up afterward — no leftover call dirs or files
    assert list(engine.work_dir.iterdir()) == []


# --------------------------------------------------------------------------- #
# R3 — locked, self-restoring torch.load scope
# --------------------------------------------------------------------------- #
def test_relaxed_torch_load_locked_and_self_restoring(monkeypatch):
    fake_torch = types.ModuleType("torch")
    sentinel = object()
    fake_torch.load = sentinel
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert hasattr(drum_split._TORCH_LOAD_LOCK, "acquire")  # a real lock guards it

    with drum_split._relaxed_torch_load():
        assert fake_torch.load is not sentinel   # relaxed only inside the scope
    assert fake_torch.load is sentinel           # restored on normal exit

    try:
        with drum_split._relaxed_torch_load():
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert fake_torch.load is sentinel           # restored even on exception


def test_load_demucs_checkpoint_relaxes_only_within_scope(monkeypatch):
    fake_torch = types.ModuleType("torch")
    calls: list[bool] = []

    def fake_load(path, *a, weights_only=True, **k):
        calls.append(weights_only)
        if weights_only:  # mimic torch>=2.6 rejecting the demucs pickle
            raise Exception("Unsupported global: GLOBAL demucs.hdemucs.HDemucs")
        return "MODEL"

    fake_torch.load = fake_load
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    def load_model(p):
        import torch
        return torch.load(p)

    out = drum_split._load_demucs_checkpoint(load_model, Path("x.th"))
    assert out == "MODEL"
    assert calls == [True, False]          # strict first (rejected), retry relaxed
    assert fake_torch.load is fake_load    # restored — nothing left globally patched
