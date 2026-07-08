"""CLI flag plumbing (Typer) — pipeline mocked, no models, no GPU."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from stemforge import cli

runner = CliRunner()


class _FakePipeline:
    """Captures the config the CLI built; returns a minimal manifest."""

    captured: dict = {}

    def __init__(self, cfg):
        _FakePipeline.captured["cfg"] = cfg

    def run(self, path):
        return {"input": {"filename": Path(path).name}}


def test_run_preset_and_bool_flags(monkeypatch, sine_wav):
    monkeypatch.setattr(cli, "Pipeline", _FakePipeline)

    result = runner.invoke(cli.app, ["run", str(sine_wav), "--preset", "SOTA", "--midi"])
    assert result.exit_code == 0, result.output

    cfg = _FakePipeline.captured["cfg"]
    assert cfg.separation.backend == "uvr"
    assert cfg.separation.preset == "sota"
    assert cfg.midi.enabled is True
    assert cfg.stretch.enabled is False


def test_run_without_preset_keeps_demucs_default(monkeypatch, sine_wav):
    monkeypatch.setattr(cli, "Pipeline", _FakePipeline)

    result = runner.invoke(cli.app, ["run", str(sine_wav)])
    assert result.exit_code == 0, result.output

    cfg = _FakePipeline.captured["cfg"]
    assert cfg.separation.backend == "demucs"
    assert cfg.separation.preset is None
    assert cfg.separation.model == "htdemucs_ft"


def test_separate_preset_flag(monkeypatch, sine_wav):
    monkeypatch.setattr(cli, "Pipeline", _FakePipeline)

    result = runner.invoke(cli.app, ["separate", str(sine_wav), "--preset", "max"])
    assert result.exit_code == 0, result.output

    cfg = _FakePipeline.captured["cfg"]
    assert cfg.separation.backend == "hybrid"
    assert cfg.separation.model == "htdemucs_ft"
    assert cfg.separation.shifts == 2


def test_preset_picks_model_over_model_default(monkeypatch, sine_wav):
    """--preset drops the --model default so the preset's model applies."""
    monkeypatch.setattr(cli, "Pipeline", _FakePipeline)

    result = runner.invoke(cli.app, ["separate", str(sine_wav), "--preset", "fast"])
    assert result.exit_code == 0, result.output
    assert _FakePipeline.captured["cfg"].separation.model == "htdemucs"


def test_set_override_beats_preset(monkeypatch, sine_wav):
    monkeypatch.setattr(cli, "Pipeline", _FakePipeline)

    result = runner.invoke(
        cli.app, ["run", str(sine_wav), "--preset", "max", "--set", "separation.shifts=1"],
    )
    assert result.exit_code == 0, result.output

    cfg = _FakePipeline.captured["cfg"]
    assert cfg.separation.backend == "hybrid"
    assert cfg.separation.shifts == 1


def test_bad_preset_fails_loudly(monkeypatch, sine_wav):
    monkeypatch.setattr(cli, "Pipeline", _FakePipeline)

    result = runner.invoke(cli.app, ["separate", str(sine_wav), "--preset", "ultra"])
    assert result.exit_code != 0


def test_help_lists_preset():
    for cmd in ("run", "separate"):
        result = runner.invoke(cli.app, [cmd, "--help"])
        assert result.exit_code == 0
        assert "--preset" in result.output


def test_setup_sota_invokes_provisioner(monkeypatch, tmp_path):
    import stemforge.separate_uvr as uvr_mod

    seen = {}

    def fake_setup(cfg, venv=None, log=print):
        seen["venv"] = venv
        seen["default_venv"] = cfg.uvr_venv
        return True

    monkeypatch.setattr(uvr_mod, "setup_sota_env", fake_setup)
    result = runner.invoke(cli.app, ["setup-sota", "--venv", str(tmp_path / "v")])
    assert result.exit_code == 0, result.output
    assert seen["venv"] == tmp_path / "v"
    assert seen["default_venv"] == ".venv-uvr"


def test_setup_sota_reports_failure(monkeypatch):
    import stemforge.separate_uvr as uvr_mod

    monkeypatch.setattr(uvr_mod, "setup_sota_env", lambda cfg, venv=None, log=print: False)
    result = runner.invoke(cli.app, ["setup-sota"])
    assert result.exit_code == 1


def test_ui_open_flag_defaults_true_and_toggles(monkeypatch):
    import stemforge.app as app_mod

    seen: dict = {}
    monkeypatch.setattr(app_mod, "launch", lambda **kw: seen.update(kw))

    assert runner.invoke(cli.app, ["ui"]).exit_code == 0
    assert seen["open_browser"] is True

    assert runner.invoke(cli.app, ["ui", "--no-open"]).exit_code == 0
    assert seen["open_browser"] is False


def test_desktop_shortcut_command(monkeypatch, tmp_path):
    import stemforge.desktop as desktop_mod

    made = tmp_path / "StemForge.lnk"
    monkeypatch.setattr(desktop_mod, "create_shortcut", lambda log=print: made)
    result = runner.invoke(cli.app, ["desktop-shortcut"])
    assert result.exit_code == 0, result.output
    assert "StemForge.lnk" in result.output


def test_desktop_shortcut_command_failure(monkeypatch):
    import stemforge.desktop as desktop_mod

    monkeypatch.setattr(desktop_mod, "create_shortcut", lambda log=print: None)
    result = runner.invoke(cli.app, ["desktop-shortcut"])
    assert result.exit_code == 1
