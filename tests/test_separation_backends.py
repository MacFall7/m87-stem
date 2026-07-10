"""Presets + UVR/hybrid backend routing — GPU-free; the audio-separator CLI
subprocess is fully mocked (no venv creation, no model downloads in CI)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from stemforge import separate_uvr
from stemforge.io_utils import AudioTensor
from stemforge.orchestrator import BS_ROFORMER_MODEL, Pipeline, load_config, preset_names
from stemforge.separate_uvr import SotaEnvError, canonical_stem_name


# --------------------------------------------------------------------------- #
# Quality presets
# --------------------------------------------------------------------------- #
def test_preset_names_order():
    assert preset_names() == ["fast", "best", "sota", "max"]


def test_preset_fast():
    sep = load_config(overrides={"separation.preset": "fast"}).separation
    assert (sep.backend, sep.model, sep.shifts) == ("demucs", "htdemucs", 1)


def test_preset_best():
    sep = load_config(overrides={"separation.preset": "best"}).separation
    assert (sep.backend, sep.model, sep.shifts) == ("demucs", "htdemucs_ft", 2)


def test_preset_sota():
    sep = load_config(overrides={"separation.preset": "sota"}).separation
    assert sep.backend == "uvr"
    assert sep.uvr_model == BS_ROFORMER_MODEL
    assert sep.stems == ["vocals", "instrumental"]


def test_preset_max():
    sep = load_config(overrides={"separation.preset": "max"}).separation
    assert sep.backend == "hybrid"
    assert sep.model == "htdemucs_ft"
    assert sep.uvr_model == BS_ROFORMER_MODEL
    assert sep.shifts == 2
    assert sep.stems == ["vocals", "drums", "bass", "other"]


def test_preset_case_insensitive_and_normalized():
    sep = load_config(overrides={"separation.preset": "SOTA"}).separation
    assert sep.backend == "uvr"
    assert sep.preset == "sota"  # normalized for the manifest


def test_explicit_override_wins_over_preset():
    """Precedence: yaml < preset < dotted overrides (--set escape hatch works)."""
    sep = load_config(overrides={
        "separation.preset": "max", "separation.shifts": 1,
    }).separation
    assert sep.backend == "hybrid"  # from the preset
    assert sep.shifts == 1          # explicit override beats the preset's 2


def test_unknown_preset_rejected():
    with pytest.raises(ValueError, match="preset"):
        load_config(overrides={"separation.preset": "ultra"})


def test_no_preset_leaves_defaults():
    sep = load_config().separation
    assert sep.preset is None
    assert (sep.backend, sep.model) == ("demucs", "htdemucs_ft")


# --------------------------------------------------------------------------- #
# Output-filename stem-token mapping
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("filename,expected", [
    ("mix_(Vocals)_model_bs_roformer_ep_317_sdr_12.9755.wav", "vocals"),
    ("mix_(Instrumental)_model_bs_roformer_ep_317_sdr_12.9755.wav", "instrumental"),
    ("track_(Drums)_htdemucs.wav", "drums"),
    ("track_(Bass)_htdemucs.wav", "bass"),
    ("track_(Other)_htdemucs.wav", "other"),
    ("track_(VOCALS)_x.wav", "vocals"),          # case-insensitive
    ("track_(No Vocals)_x.wav", "instrumental"),  # alias
    ("weird_output.wav", "weird_output"),         # no token -> slugified fallback
])
def test_canonical_stem_name(filename, expected):
    assert canonical_stem_name(filename) == expected


# --------------------------------------------------------------------------- #
# Mocked audio-separator CLI subprocess (no venv, no downloads, no import)
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_uvr_cli(monkeypatch):
    """Pretend the isolated venv exists and its CLI works: preflight passes and
    `subprocess.run` writes real stem WAVs into --output_dir."""
    calls: dict = {"cmds": []}

    monkeypatch.setattr(separate_uvr, "have_ffmpeg", lambda: True)
    monkeypatch.setattr(
        separate_uvr, "find_cli", lambda venv_dir: Path(venv_dir) / "bin" / "audio-separator",
    )

    def fake_run(cmd, capture_output=False, text=False, check=False, **kwargs):
        calls["cmds"].append([str(c) for c in cmd])
        args = {cmd[i]: cmd[i + 1] for i in range(len(cmd) - 1)}
        src = Path(str(cmd[1]))
        out_dir = Path(str(args["--output_dir"]))
        model = Path(str(args["--model_filename"])).stem
        data, sr = sf.read(str(src), always_2d=True)
        for token in ("Vocals", "Instrumental"):
            sf.write(str(out_dir / f"{src.stem}_({token})_{model}.wav"), data * 0.5, sr)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(separate_uvr.subprocess, "run", fake_run)
    return calls


def _base_overrides(tmp_path) -> dict:
    return {
        "ingest.normalize": "none",
        "analysis.enabled": False,
        "output.root": str(tmp_path / "out"),
    }


def test_max_preset_wires_verified_vocal_ensemble():
    from stemforge.orchestrator import ENSEMBLE_VOCALS

    sep = load_config(overrides={"separation.preset": "max"}).separation
    assert sep.ensemble_enabled is True
    assert sep.ensemble_models == ENSEMBLE_VOCALS
    assert sep.uvr_ensemble_algorithm == "uvr_max_spec"


def test_ensemble_cli_invocation(fake_uvr_cli, sine):
    """Max's ensemble runs `-m primary --extra_models m2 --ensemble_algorithm uvr_max_spec`."""
    from stemforge.orchestrator import ENSEMBLE_VOCALS

    audio = AudioTensor(*sine)
    cfg = load_config(overrides={"separation.preset": "max"}).separation
    # route through the uvr stage directly (hybrid would then run demucs on the residual)
    stems, engine = separate_uvr.separate(audio, cfg, device="cpu")
    assert set(stems) == {"vocals", "instrumental"}

    cmd = fake_uvr_cli["cmds"][0]
    assert cmd[cmd.index("--model_filename") + 1] == ENSEMBLE_VOCALS[0]  # primary
    assert "--extra_models" in cmd
    extra_i = cmd.index("--extra_models")
    assert cmd[extra_i + 1] == ENSEMBLE_VOCALS[1]                        # the extra model
    assert cmd[cmd.index("--ensemble_algorithm") + 1] == "uvr_max_spec"
    assert engine.extra_models == ENSEMBLE_VOCALS[1:]


def test_non_ensemble_uvr_has_no_extra_models(fake_uvr_cli, sine):
    cfg = load_config(overrides={"separation.preset": "sota"}).separation
    separate_uvr.separate(AudioTensor(*sine), cfg, device="cpu")
    cmd = fake_uvr_cli["cmds"][0]
    assert "--extra_models" not in cmd and "--ensemble_algorithm" not in cmd


def test_separate_uvr_maps_and_caches(fake_uvr_cli, sine):
    samples, sr = sine
    audio = AudioTensor(samples, sr)
    cfg = load_config(overrides={"separation.preset": "sota"}).separation

    stems, engine = separate_uvr.separate(audio, cfg, device="cpu")
    assert set(stems) == {"vocals", "instrumental"}
    assert all(isinstance(v, AudioTensor) for v in stems.values())
    assert stems["vocals"].sample_rate == sr

    cmd = fake_uvr_cli["cmds"][0]
    assert cmd[0].endswith("audio-separator")
    assert cmd[cmd.index("--model_filename") + 1] == BS_ROFORMER_MODEL
    assert cmd[cmd.index("--output_format") + 1] == "WAV"
    # compare path components, not a "/"-joined suffix — portable across Windows backslashes
    assert Path(cmd[cmd.index("--model_file_dir") + 1]).parts[-2:] == ("models", "uvr")
    assert "--use_autocast" in cmd

    # scratch files are cleaned up after each call
    assert list(engine.work_dir.iterdir()) == []

    # same model -> engine (and its preflight/scratch dir) is reused
    _, engine2 = separate_uvr.separate(audio, cfg, device="cpu", engine=engine)
    assert engine2 is engine

    # different model -> new engine
    cfg.uvr_model = "some_other_model.ckpt"
    _, engine3 = separate_uvr.separate(audio, cfg, device="cpu", engine=engine)
    assert engine3 is not engine


def test_no_in_process_audio_separator_import(fake_uvr_cli, sine):
    """The whole point of the isolation: audio_separator never enters this process."""
    samples, sr = sine
    cfg = load_config(overrides={"separation.preset": "sota"}).separation
    separate_uvr.separate(AudioTensor(samples, sr), cfg, device="cpu")
    assert "audio_separator" not in sys.modules


def test_pipeline_uvr_backend(fake_uvr_cli, tmp_path, sine_wav):
    cfg = load_config(overrides={**_base_overrides(tmp_path), "separation.preset": "sota"})
    m = Pipeline(cfg).run(sine_wav)

    sep = m["separation"]
    assert sep["backend"] == "uvr"
    assert sep["preset"] == "sota"
    assert set(sep["files"]) == {"vocals.wav", "instrumental.wav"}
    assert m["models"]["uvr"] == BS_ROFORMER_MODEL
    out = tmp_path / "out" / "in" / "stems"
    assert (out / "vocals.wav").is_file() and (out / "instrumental.wav").is_file()


def test_pipeline_hybrid_backend(fake_uvr_cli, monkeypatch, tmp_path, sine_wav):
    import stemforge.separate as demucs_mod

    seen: dict = {}

    def fake_demucs(audio, cfg, device="cpu", model=None):
        seen["stems_requested"] = list(cfg.stems)
        seen["input_samples"] = audio.num_samples
        mk = lambda: AudioTensor(np.zeros((2, audio.num_samples), np.float32), audio.sample_rate)
        # demucs always emits its own (near-silent) vocals; hybrid must drop it
        return {"vocals": mk(), "drums": mk(), "bass": mk(), "other": mk()}, "cached-model"

    monkeypatch.setattr(demucs_mod, "separate", fake_demucs)

    cfg = load_config(overrides={**_base_overrides(tmp_path), "separation.preset": "max"})
    m = Pipeline(cfg).run(sine_wav)

    sep = m["separation"]
    assert sep["backend"] == "hybrid"
    # merged 4-stem: roformer vocals + demucs residual; instrumental dropped
    assert set(sep["files"]) == {"vocals.wav", "drums.wav", "bass.wav", "other.wav"}
    assert seen["stems_requested"] == ["drums", "bass", "other"]
    assert set(m["models"]) == {"uvr", "demucs"}
    assert m["models"]["demucs"] == "htdemucs_ft"
    # Max now runs the verified vocal ensemble (uvr_max_spec) + demucs residual.
    assert "bs_roformer_vocals_resurrection_unwa" in sep["model"]
    assert "melband_roformer_big_beta6x" in sep["model"]
    assert "htdemucs_ft" in sep["model"]


def test_hybrid_keeps_uvr_result_when_demucs_missing(fake_uvr_cli, monkeypatch, tmp_path, sine_wav):
    """Hybrid must not throw away the finished roformer pass if demucs is absent."""
    import stemforge.separate as demucs_mod

    def no_demucs(*a, **kw):
        raise ModuleNotFoundError("No module named 'demucs'", name="demucs")

    monkeypatch.setattr(demucs_mod, "separate", no_demucs)
    cfg = load_config(overrides={**_base_overrides(tmp_path), "separation.preset": "max"})
    m = Pipeline(cfg).run(sine_wav)

    sep = m["separation"]
    assert sep["backend"] == "hybrid"
    assert set(sep["files"]) == {"vocals.wav", "instrumental.wav"}  # degraded 2-stem
    assert any("demucs" in n for n in sep["notes"])


def test_max_degrades_to_demucs_when_uvr_unavailable(monkeypatch, tmp_path, sine_wav):
    """Max never regresses to an error: no venv -> full demucs 4-stem + a note."""
    import stemforge.separate as demucs_mod

    monkeypatch.setattr(separate_uvr, "have_ffmpeg", lambda: True)
    monkeypatch.setattr(separate_uvr, "find_cli", lambda venv_dir: None)  # venv not set up

    def fake_demucs(audio, cfg, device="cpu", model=None):
        mk = lambda: AudioTensor(np.zeros((2, audio.num_samples), np.float32), audio.sample_rate)
        return {"vocals": mk(), "drums": mk(), "bass": mk(), "other": mk()}, "demucs-bag"

    monkeypatch.setattr(demucs_mod, "separate", fake_demucs)
    cfg = load_config(overrides={**_base_overrides(tmp_path), "separation.preset": "max"})
    m = Pipeline(cfg).run(sine_wav)

    sep = m["separation"]
    assert "error" not in sep and "skipped" not in sep
    assert set(sep["files"]) == {"vocals.wav", "drums.wav", "bass.wav", "other.wav"}
    assert set(m["models"]) == {"demucs"}                      # UVR label dropped
    assert any("fell back to demucs" in n for n in sep.get("notes", []))


# --------------------------------------------------------------------------- #
# Preflight fail-soft (no venv / no ffmpeg) + runtime CLI failure
# --------------------------------------------------------------------------- #
def test_missing_venv_fails_soft_with_setup_hint(monkeypatch, tmp_path, sine_wav):
    monkeypatch.setattr(separate_uvr, "have_ffmpeg", lambda: True)
    monkeypatch.setattr(separate_uvr, "find_cli", lambda venv_dir: None)

    cfg = load_config(overrides={**_base_overrides(tmp_path), "separation.preset": "sota"})
    m = Pipeline(cfg).run(sine_wav)

    assert "setup-sota" in m["separation"]["skipped"]
    assert (tmp_path / "out" / "in" / "manifest.json").is_file()


def test_missing_ffmpeg_fails_soft(monkeypatch, tmp_path, sine_wav):
    monkeypatch.setattr(separate_uvr, "have_ffmpeg", lambda: False)

    cfg = load_config(overrides={**_base_overrides(tmp_path), "separation.preset": "sota"})
    m = Pipeline(cfg).run(sine_wav)

    assert "ffmpeg" in m["separation"]["skipped"]


def test_preflight_raises_sota_env_error(monkeypatch):
    monkeypatch.setattr(separate_uvr, "have_ffmpeg", lambda: True)
    monkeypatch.setattr(separate_uvr, "find_cli", lambda venv_dir: None)
    with pytest.raises(SotaEnvError, match="setup-sota"):
        separate_uvr.preflight(load_config().separation)


def test_cli_crash_is_an_error_not_a_skip(monkeypatch, tmp_path, sine_wav):
    """A broken run inside a ready env is a real error (with stderr), not a skip."""
    monkeypatch.setattr(separate_uvr, "have_ffmpeg", lambda: True)
    monkeypatch.setattr(
        separate_uvr, "find_cli", lambda venv_dir: Path(venv_dir) / "bin" / "audio-separator",
    )

    def failing_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="CUDA out of memory")

    monkeypatch.setattr(separate_uvr.subprocess, "run", failing_run)
    cfg = load_config(overrides={**_base_overrides(tmp_path), "separation.preset": "sota"})
    m = Pipeline(cfg).run(sine_wav)

    assert "error" in m["separation"] and "skipped" not in m["separation"]
    assert "CUDA out of memory" in m["separation"]["error"]


def test_non_dependency_import_error_is_an_error_not_a_skip(monkeypatch, tmp_path, sine_wav):
    """Only known optional backend deps fail soft; other MNFEs surface as errors."""
    import stemforge.separate as demucs_mod

    def broken(*a, **kw):
        raise ModuleNotFoundError("No module named 'einops'", name="einops")

    monkeypatch.setattr(demucs_mod, "separate", broken)
    cfg = load_config(overrides=_base_overrides(tmp_path))  # default demucs backend
    m = Pipeline(cfg).run(sine_wav)

    assert "error" in m["separation"] and "skipped" not in m["separation"]
    assert "einops" in m["separation"]["error"]


def test_unknown_backend_recorded_as_error(tmp_path, sine_wav):
    cfg = load_config(overrides={**_base_overrides(tmp_path), "separation.backend": "spleeter"})
    m = Pipeline(cfg).run(sine_wav)
    assert "error" in m["separation"]
    assert "spleeter" in m["separation"]["error"]


# --------------------------------------------------------------------------- #
# setup_sota_env — venv provisioning, mocked subprocess, idempotent
# --------------------------------------------------------------------------- #
def _bindir(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts" if separate_uvr.os.name == "nt" else "bin")


def _touch_python(venv_dir: Path) -> None:
    b = _bindir(venv_dir)
    b.mkdir(parents=True, exist_ok=True)
    (b / ("python.exe" if separate_uvr.os.name == "nt" else "python")).touch()


def _stateful_pip_mock(venv_dir: Path, cmds: list, state: dict):
    """Mock subprocess.run modeling: audio-separator downgrades torch to +cpu;
    the force-reinstall restores +cu124; `python -c` probes report `state`."""
    def fake_run(cmd, check=False, capture_output=False, text=False, **kwargs):
        cmd = [str(c) for c in cmd]
        cmds.append(cmd)
        if cmd[1:3] == ["-m", "venv"]:
            _touch_python(Path(cmd[3]))
        elif "--force-reinstall" in cmd:
            state["torch"] = "2.6.0+cu124"
        elif "pip" in cmd and any("audio-separator" in c for c in cmd):
            (_bindir(venv_dir) / "audio-separator").touch()
            state["torch"] = "2.13.0+cpu"  # audio-separator's deps clobber torch
        elif "pip" in cmd and "torch" in cmd:
            state["torch"] = "2.6.0+cu124"
        elif cmd[1:2] == ["-c"]:  # torch status probe
            if state["torch"] is None:
                return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="no torch")
            cuda = "+cu" in state["torch"]
            payload = json.dumps({
                "version": state["torch"], "cuda": cuda,
                "name": "Mock RTX 4090" if cuda else None,
            })
            return subprocess.CompletedProcess(cmd, 0, stdout=payload + "\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0)

    return fake_run


def test_setup_sota_env_creates_venv_forces_cuda_torch_and_is_idempotent(monkeypatch, tmp_path):
    venv_dir = tmp_path / ".venv-uvr"
    cmds: list[list[str]] = []
    state = {"torch": None}

    monkeypatch.setattr(separate_uvr.subprocess, "run", _stateful_pip_mock(venv_dir, cmds, state))
    monkeypatch.setattr(separate_uvr, "have_ffmpeg", lambda: True)
    logs: list[str] = []

    cfg = load_config().separation
    assert separate_uvr.setup_sota_env(cfg, venv=venv_dir, log=logs.append) is True

    installs = [c for c in cmds if "pip" in c and "install" in c]
    # order: CUDA torch first, then audio-separator, then the cu124 force-reinstall LAST
    assert "torch" in installs[0] and separate_uvr.TORCH_CU124_INDEX in installs[0]
    assert any("audio-separator" in c for c in installs[1])
    assert "--force-reinstall" in installs[-1] and "--no-deps" in installs[-1]
    assert separate_uvr.TORCH_CU124_INDEX in installs[-1]
    # the force-reinstall runs AFTER the audio-separator install
    assert "torch" in installs[-1]
    assert cmds.index(installs[-1]) > cmds.index(installs[1])
    # every pip install targets the VENV python, never the main interpreter
    assert all(c[0] == str(separate_uvr.venv_python(venv_dir)) for c in installs)
    # verified CUDA in the venv and surfaced the GPU name
    assert any("Mock RTX 4090" in line for line in logs)

    # second run: venv + CLI present, torch already +cu124 -> NO installs at all
    cmds.clear()
    assert separate_uvr.setup_sota_env(cfg, venv=venv_dir, log=logs.append) is True
    assert [c for c in cmds if "pip" in c and "install" in c] == []


def test_setup_sota_env_cuda_reinstall_failsoft(monkeypatch, tmp_path):
    """A failed cu124 reinstall warns but never crashes; setup still succeeds."""
    venv_dir = tmp_path / ".venv-uvr"
    _touch_python(venv_dir)
    (_bindir(venv_dir) / "audio-separator").touch()  # CLI already present

    def fake_run(cmd, check=False, capture_output=False, text=False, **kwargs):
        cmd = [str(c) for c in cmd]
        if "--force-reinstall" in cmd:
            raise subprocess.CalledProcessError(1, cmd)
        if cmd[1:2] == ["-c"]:  # torch present but CPU-only -> triggers a reinstall attempt
            payload = json.dumps({"version": "2.13.0+cpu", "cuda": False, "name": None})
            return subprocess.CompletedProcess(cmd, 0, stdout=payload + "\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(separate_uvr.subprocess, "run", fake_run)
    monkeypatch.setattr(separate_uvr, "have_ffmpeg", lambda: True)
    logs: list[str] = []

    assert separate_uvr.setup_sota_env(load_config().separation, venv=venv_dir, log=logs.append) is True
    assert any("WARNING" in line and "CPU" in line for line in logs)


def test_venv_torch_status_none_when_absent(tmp_path):
    """No venv python -> None, no crash (doctor tolerates it)."""
    assert separate_uvr.venv_torch_status(tmp_path / "nope") is None


def test_setup_sota_env_reports_missing_ffmpeg(monkeypatch, tmp_path):
    venv_dir = tmp_path / ".venv-uvr"
    _touch_python(venv_dir)
    (_bindir(venv_dir) / "audio-separator").touch()

    monkeypatch.setattr(separate_uvr, "have_ffmpeg", lambda: False)

    def fake_run(cmd, check=False, capture_output=False, text=False, **kwargs):
        cmd = [str(c) for c in cmd]
        if "install" in cmd:
            pytest.fail("no pip install expected when the venv is complete")
        if cmd[1:2] == ["-c"]:  # already a CUDA build -> no reinstall
            payload = json.dumps({"version": "2.6.0+cu124", "cuda": True, "name": "Mock GPU"})
            return subprocess.CompletedProcess(cmd, 0, stdout=payload + "\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(separate_uvr.subprocess, "run", fake_run)
    logs: list[str] = []
    assert separate_uvr.setup_sota_env(load_config().separation, venv=venv_dir, log=logs.append) is False
    assert any("ffmpeg" in line for line in logs)
