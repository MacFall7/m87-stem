"""Drum teardown via the isolated .venv-uvr subprocess — GPU-free, CLI mocked.

No venv is built and nothing is downloaded: the audio-separator CLI (both the
`--list_filter=drums` discovery call and the separation call) is mocked, and
preflight (ffmpeg + venv) is stubbed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from stemforge import drum_split, separate_uvr
from stemforge.io_utils import AudioTensor
from stemforge.orchestrator import Pipeline, load_config
from stemforge.separate_uvr import canonical_drum_part, discover_drum_model

_DRUM_MODEL = "drumsep_mel_band_roformer_6stem.ckpt"
_DRUM_LIST_OUTPUT = f"""\
Model Filename                                   Arch        Stems
-----------------------------------------------  ----------  ----------------------------
{_DRUM_MODEL}   MelRoformer  Kick, Snare, Toms, HiHat, Ride, Crash
some_4stem_drum_model.ckpt                        Demucs      Kick, Snare, Toms, Cymbals
"""

_DRUM_TOKENS = ("Kick", "Snare", "Toms", "HiHat", "Ride", "Crash")


# --------------------------------------------------------------------------- #
# Token mapping + model discovery (pure)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("filename,expected", [
    ("loop_(Kick)_drumsep.wav", "kick"),
    ("loop_(Snare)_drumsep.wav", "snare"),
    ("loop_(Toms)_drumsep.wav", "toms"),
    ("loop_(HiHat)_drumsep.wav", "hihat"),
    ("loop_(Hi-Hat)_x.wav", "hihat"),
    ("loop_(Ride)_x.wav", "ride"),
    ("loop_(Crash)_x.wav", "crash"),
    ("loop_(Cymbals)_x.wav", "crash"),
    ("loop_(BD)_x.wav", "kick"),
    ("mystery_output.wav", "mystery_output"),  # unknown -> slug fallback
])
def test_canonical_drum_part(filename, expected):
    assert canonical_drum_part(filename) == expected


def test_discover_drum_model_prefers_six_stem(monkeypatch, tmp_path):
    def fake_run(cmd, capture_output=False, text=False, **kw):
        assert "--list_filter=drums" in cmd
        return subprocess.CompletedProcess(cmd, 0, stdout=_DRUM_LIST_OUTPUT, stderr="")

    monkeypatch.setattr(separate_uvr.subprocess, "run", fake_run)
    model = discover_drum_model(Path("cli"), str(tmp_path))
    assert model == _DRUM_MODEL  # the 6-stem kit wins over the 4-stem


def test_discover_drum_model_none_when_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(
        separate_uvr.subprocess, "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout="no models\n", stderr=""),
    )
    assert discover_drum_model(Path("cli"), str(tmp_path)) is None


# --------------------------------------------------------------------------- #
# Mocked drum CLI (discovery + separation)
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_drum_cli(monkeypatch):
    """Preflight passes; the CLI lists a drum model and writes per-hit WAVs."""
    monkeypatch.setattr(separate_uvr, "have_ffmpeg", lambda: True)
    monkeypatch.setattr(
        separate_uvr, "find_cli", lambda venv_dir: Path(venv_dir) / "bin" / "audio-separator",
    )
    calls: dict = {"list": 0, "separate": 0}

    def fake_run(cmd, capture_output=False, text=False, check=False, **kw):
        cmd = [str(c) for c in cmd]
        if any("--list_filter=drums" in c for c in cmd):
            calls["list"] += 1
            return subprocess.CompletedProcess(cmd, 0, stdout=_DRUM_LIST_OUTPUT, stderr="")
        # separation call: {cli} {in} --model_filename ... --output_dir <dir> ...
        calls["separate"] += 1
        args = {cmd[i]: cmd[i + 1] for i in range(len(cmd) - 1)}
        src = Path(cmd[1])
        out_dir = Path(args["--output_dir"])
        model = Path(args["--model_filename"]).stem
        data, sr = sf.read(str(src), always_2d=True)
        for token in _DRUM_TOKENS:
            sf.write(str(out_dir / f"{src.stem}_({token})_{model}.wav"), data, sr)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(separate_uvr.subprocess, "run", fake_run)
    return calls


def _drum_cfg(**over):
    """A drums.split cfg forced to the (optional) uvr backend for these tests."""
    cfg = load_config().drums.split
    cfg.enabled = True
    cfg.backend = "uvr"
    cfg.parts = ["kick", "snare", "toms", "hihat", "ride", "crash"]  # full-kit UVR request
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def _click_loop() -> AudioTensor:
    sr = 44100
    y = np.zeros(int(sr * 2.0), dtype=np.float32)
    for beat in np.arange(0, 2.0, 0.5):  # 120 BPM
        i = int(beat * sr)
        y[i:i + 200] = np.hanning(200).astype(np.float32)
    return AudioTensor(np.stack([y, y]), sr)


def test_separate_drums_maps_parts_and_uses_list_filter(fake_drum_cli):
    parts, model, engine = separate_uvr.separate_drums(_click_loop(), _drum_cfg())
    assert model == _DRUM_MODEL
    assert set(parts) == {"kick", "snare", "toms", "hihat", "ride", "crash"}
    assert all(isinstance(v, AudioTensor) for v in parts.values())
    assert fake_drum_cli["list"] == 1 and fake_drum_cli["separate"] == 1
    assert list(engine.work_dir.iterdir()) == []  # scratch cleaned up


def test_split_writes_part_files(fake_drum_cli, tmp_path):
    cfg = _drum_cfg()
    res = drum_split.split(None, cfg, tmp_path, audio=_click_loop())
    assert res["backend"] == "uvr"
    assert res["model"] == _DRUM_MODEL
    assert set(res["files"]) == {"kick", "snare", "toms", "hihat", "ride", "crash"}
    for p in res["files"].values():
        assert Path(p).is_file()


def test_split_honors_parts_subset(fake_drum_cli, tmp_path):
    cfg = _drum_cfg(parts=["kick", "snare"])
    res = drum_split.split(None, cfg, tmp_path, audio=_click_loop())
    assert set(res["files"]) == {"kick", "snare"}


def test_explicit_uvr_model_skips_discovery(fake_drum_cli, tmp_path):
    cfg = _drum_cfg(uvr_model="my_pinned_drum_model.ckpt")
    res = drum_split.split(None, cfg, tmp_path, audio=_click_loop())
    assert res["model"] == "my_pinned_drum_model.ckpt"
    assert fake_drum_cli["list"] == 0  # no discovery call when pinned


# --------------------------------------------------------------------------- #
# Fail-soft (isolation preserved)
# --------------------------------------------------------------------------- #
def test_missing_venv_fails_soft(monkeypatch, tmp_path):
    monkeypatch.setattr(separate_uvr, "have_ffmpeg", lambda: True)
    monkeypatch.setattr(separate_uvr, "find_cli", lambda venv_dir: None)
    res = drum_split.split(None, _drum_cfg(), tmp_path, audio=_click_loop())
    assert "skipped" in res and "setup-sota" in res["skipped"]


def test_missing_ffmpeg_fails_soft(monkeypatch, tmp_path):
    monkeypatch.setattr(separate_uvr, "have_ffmpeg", lambda: False)
    res = drum_split.split(None, _drum_cfg(), tmp_path, audio=_click_loop())
    assert "skipped" in res and "ffmpeg" in res["skipped"]


def test_no_drum_model_fails_soft(monkeypatch, tmp_path):
    monkeypatch.setattr(separate_uvr, "have_ffmpeg", lambda: True)
    monkeypatch.setattr(
        separate_uvr, "find_cli", lambda venv_dir: Path(venv_dir) / "bin" / "audio-separator",
    )
    monkeypatch.setattr(
        separate_uvr.subprocess, "run",
        lambda cmd, **kw: subprocess.CompletedProcess([str(c) for c in cmd], 0, stdout="", stderr=""),
    )
    res = drum_split.split(None, _drum_cfg(), tmp_path, audio=_click_loop())
    assert "skipped" in res and "drum" in res["skipped"].lower()


def test_cli_crash_is_error_not_skip(monkeypatch, tmp_path):
    monkeypatch.setattr(separate_uvr, "have_ffmpeg", lambda: True)
    monkeypatch.setattr(
        separate_uvr, "find_cli", lambda venv_dir: Path(venv_dir) / "bin" / "audio-separator",
    )

    def fake_run(cmd, capture_output=False, text=False, **kw):
        cmd = [str(c) for c in cmd]
        if any("--list_filter=drums" in c for c in cmd):
            return subprocess.CompletedProcess(cmd, 0, stdout=_DRUM_LIST_OUTPUT, stderr="")
        return subprocess.CompletedProcess(cmd, 2, stdout="", stderr="CUDA OOM")

    monkeypatch.setattr(separate_uvr.subprocess, "run", fake_run)
    res = drum_split.split(None, _drum_cfg(), tmp_path, audio=_click_loop())
    assert "error" in res and "CUDA OOM" in res["error"]


def test_pipeline_drum_loop_to_stems_and_midi_uvr(fake_drum_cli, tmp_path):
    loop = _click_loop()
    wav = tmp_path / "break.wav"
    sf.write(str(wav), loop.samples.T, loop.sample_rate, subtype="FLOAT")

    cfg = load_config(overrides={
        "ingest.normalize": "none",
        "analysis.enabled": False,
        "separation.enabled": False,     # a raw drum loop: no upstream separation
        "drums.split.enabled": True,
        "drums.split.backend": "uvr",
        "drums.split.parts": ["kick", "snare", "toms", "hihat", "ride", "crash"],
        "drums.split.from_input": True,  # tear down the input loop itself
        "drums.midi.enabled": True,
        "output.root": str(tmp_path / "out"),
    })
    m = Pipeline(cfg).run(wav)

    ds = m["drum_split"]
    assert ds["backend"] == "uvr"
    assert set(ds["files"]) == {"kick", "snare", "toms", "hihat", "ride", "crash"}
    drums_dir = tmp_path / "out" / "break" / "drums"
    assert (drums_dir / "kick.wav").is_file()

    dm = m["drum_midi"]
    assert dm["source"] == "parts"
    assert dm["note_count"] > 0
    assert (drums_dir / "drums.mid").is_file()


# --------------------------------------------------------------------------- #
# inagoy/drumsep backend (default) — main-env Demucs, mocked (no download/apply)
# --------------------------------------------------------------------------- #
class _FakeDemucs:
    sources = ["bombo", "redoblante", "toms", "platillos"]  # kick/snare/toms/cymbals (ES labels)
    samplerate = 44100

    def to(self, _d):
        return self

    def eval(self):
        return self


@pytest.fixture
def fake_inagoy(monkeypatch):
    """inagoy checkpoint 'loads' without download; apply returns 4 named sources."""
    monkeypatch.setattr(
        drum_split, "_load_inagoy_model", lambda cfg, device: (_FakeDemucs(), "drumsep.th"),
    )

    def fake_apply(model, audio, device="cuda", **kw):
        return {name: audio for name in model.sources}

    import stemforge.separate as sep
    monkeypatch.setattr(sep, "apply_model_to_audio", fake_apply)


def _inagoy_cfg(**over):
    cfg = load_config().drums.split       # backend defaults to demucs_inagoy
    cfg.enabled = True
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def test_inagoy_maps_spanish_sources_to_canonical(fake_inagoy, tmp_path):
    res = drum_split.split(None, _inagoy_cfg(), tmp_path, audio=_click_loop())
    assert res["backend"] == "demucs_inagoy"
    assert res["model"] == "drumsep.th"
    assert set(res["files"]) == {"kick", "snare", "toms", "other"}
    for p in res["files"].values():
        assert Path(p).is_file()


def test_inagoy_download_failure_fails_soft(monkeypatch, tmp_path):
    def boom(cfg):
        raise drum_split._DrumModelUnavailable("could not download inagoy drumsep model (offline)")

    monkeypatch.setattr(drum_split, "_ensure_inagoy_checkpoint", boom)
    res = drum_split.split(None, _inagoy_cfg(), tmp_path, audio=_click_loop())
    assert "skipped" in res and "inagoy" in res["skipped"].lower()
    assert res["backend"] == "demucs_inagoy"


def test_inagoy_bad_checkpoint_fails_soft(monkeypatch, tmp_path):
    """A checkpoint that downloads but won't load -> skip (not a crash)."""
    dest = tmp_path / "drumsep.th"
    dest.write_bytes(b"not a real checkpoint")
    monkeypatch.setattr(drum_split, "_ensure_inagoy_checkpoint", lambda cfg: dest)

    import types
    fake_demucs = types.ModuleType("demucs")
    fake_states = types.ModuleType("demucs.states")

    def boom(_p):
        raise RuntimeError("bad magic")

    fake_states.load_model = boom
    monkeypatch.setitem(__import__("sys").modules, "demucs", fake_demucs)
    monkeypatch.setitem(__import__("sys").modules, "demucs.states", fake_states)

    res = drum_split.split(None, _inagoy_cfg(), tmp_path, audio=_click_loop())
    assert "skipped" in res and "load" in res["skipped"].lower()


def test_inagoy_missing_demucs_fails_soft(monkeypatch, tmp_path):
    monkeypatch.setattr(drum_split, "_ensure_inagoy_checkpoint", lambda cfg: tmp_path / "drumsep.th")

    def no_demucs(cfg, device):
        raise ModuleNotFoundError("No module named 'demucs'", name="demucs")

    monkeypatch.setattr(drum_split, "_load_inagoy_model", no_demucs)
    res = drum_split.split(None, _inagoy_cfg(), tmp_path, audio=_click_loop())
    assert "skipped" in res and "demucs" in res["skipped"]


def test_inagoy_checkpoint_download_and_cache(monkeypatch, tmp_path):
    """Downloads once into inagoy_model_dir; reuses the cached file next time."""
    calls = {"n": 0}

    def fake_urlretrieve(url, dest):
        calls["n"] += 1
        Path(dest).write_bytes(b"fake-checkpoint")

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlretrieve", fake_urlretrieve)
    cfg = _inagoy_cfg(inagoy_model_dir=str(tmp_path / "dr"),
                      inagoy_url="http://example.test/drumsep.th")

    p1 = drum_split._ensure_inagoy_checkpoint(cfg)
    p2 = drum_split._ensure_inagoy_checkpoint(cfg)
    assert p1 == p2 and p1.is_file() and p1.name == "drumsep.th"
    assert calls["n"] == 1  # second call hits the cache


def test_default_inagoy_url_is_public_mirror_and_caches_modelo_final(tmp_path):
    """The default points at the public Eddycrack864 mirror; cache basename is
    modelo_final.th under models/drumsep/."""
    cfg = load_config().drums.split
    assert cfg.inagoy_url == "https://huggingface.co/Eddycrack864/Drumsep/resolve/main/modelo_final.th"
    assert cfg.inagoy_model_dir == "models/drumsep"
    assert drum_split._inagoy_filename(cfg.inagoy_url) == "modelo_final.th"

    cfg.inagoy_model_dir = str(tmp_path / "dr")
    (tmp_path / "dr").mkdir()
    (tmp_path / "dr" / "modelo_final.th").write_bytes(b"x")  # pretend already cached
    assert drum_split._ensure_inagoy_checkpoint(cfg) == tmp_path / "dr" / "modelo_final.th"


def test_inagoy_loader_torch26_weights_only_fallback(monkeypatch, tmp_path):
    """torch>=2.6 rejects the pickled HDemucs global under weights_only=True;
    the loader retries with weights_only=False and succeeds."""
    import sys
    import types

    dest = tmp_path / "modelo_final.th"
    dest.write_bytes(b"ckpt")
    monkeypatch.setattr(drum_split, "_ensure_inagoy_checkpoint", lambda cfg: dest)

    weights_only_seen: list = []

    class FakeModel:
        def to(self, _d):
            return self

        def eval(self):
            return self

    # A fake `torch` whose load mimics torch 2.6: weights_only defaults True and
    # rejects the demucs global; passing weights_only=False succeeds.
    fake_torch = types.ModuleType("torch")

    def fake_load(path, *args, weights_only=True, **kwargs):
        weights_only_seen.append(weights_only)
        if weights_only:
            raise Exception(  # noqa: TRY002 - mirrors torch's UnpicklingError text
                "Unsupported global: GLOBAL demucs.hdemucs.HDemucs was not an "
                "allowed global by default"
            )
        return FakeModel()

    fake_torch.load = fake_load
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    # A fake demucs.states.load_model that routes through torch.load, like the real one.
    fake_demucs = types.ModuleType("demucs")
    fake_states = types.ModuleType("demucs.states")

    def load_model(p):
        import torch

        return torch.load(p)  # no weights_only kwarg -> default path / our patch

    fake_states.load_model = load_model
    monkeypatch.setitem(sys.modules, "demucs", fake_demucs)
    monkeypatch.setitem(sys.modules, "demucs.states", fake_states)

    model, name = drum_split._load_inagoy_model(_inagoy_cfg(), "cpu")
    assert isinstance(model, FakeModel)
    assert name == "modelo_final.th"
    assert weights_only_seen == [True, False]  # rejected first, retried without
    assert fake_torch.load is fake_load        # torch.load restored after the retry


def test_pipeline_drum_loop_inagoy_to_stems_and_midi(fake_inagoy, tmp_path):
    loop = _click_loop()
    wav = tmp_path / "break.wav"
    sf.write(str(wav), loop.samples.T, loop.sample_rate, subtype="FLOAT")

    cfg = load_config(overrides={
        "ingest.normalize": "none",
        "analysis.enabled": False,
        "separation.enabled": False,
        "drums.split.enabled": True,      # backend defaults to demucs_inagoy
        "drums.split.from_input": True,
        "drums.midi.enabled": True,
        "output.root": str(tmp_path / "out"),
    })
    m = Pipeline(cfg).run(wav)

    ds = m["drum_split"]
    assert ds["backend"] == "demucs_inagoy"
    assert set(ds["files"]) == {"kick", "snare", "toms", "other"}
    dm = m["drum_midi"]
    assert dm["source"] == "parts"
    assert dm["note_count"] > 0
    assert (tmp_path / "out" / "break" / "drums" / "drums.mid").is_file()
