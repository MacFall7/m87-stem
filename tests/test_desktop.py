"""Desktop launcher creation — GPU-free; PowerShell/file writes mocked or tmp'd.

No real .lnk is written in CI: the Windows path mocks the PowerShell subprocess,
and the posix paths write into a tmp Desktop ($HOME pointed at tmp_path).
"""

from __future__ import annotations

import subprocess

import pytest

from stemforge import desktop


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Point the user's home (and Desktop) at a tmp dir for both posix + windows."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    return tmp_path


def test_windows_shortcut_mocks_powershell(home, monkeypatch):
    monkeypatch.setattr(desktop.platform, "system", lambda: "Windows")
    seen: dict = {}

    def fake_run(cmd, check=False, **kwargs):
        seen["cmd"] = [str(c) for c in cmd]
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(desktop.subprocess, "run", fake_run)
    logs: list[str] = []

    path = desktop.create_shortcut(log=logs.append)
    assert path is not None and path.name == "StemForge.lnk"
    ps = " ".join(seen["cmd"])
    assert "powershell" in seen["cmd"][0]
    assert "CreateShortcut" in ps
    assert "launch_ui.bat" in ps
    assert "$s.Save()" in ps


def test_windows_shortcut_failsoft(home, monkeypatch):
    monkeypatch.setattr(desktop.platform, "system", lambda: "Windows")

    def boom(cmd, check=False, **kwargs):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(desktop.subprocess, "run", boom)
    logs: list[str] = []
    assert desktop.create_shortcut(log=logs.append) is None
    assert any("could not create" in line for line in logs)


def test_linux_shortcut_writes_desktop_entry(home, monkeypatch):
    monkeypatch.setattr(desktop.platform, "system", lambda: "Linux")
    # subprocess must NOT be used on the posix path
    monkeypatch.setattr(
        desktop.subprocess, "run",
        lambda *a, **k: pytest.fail("posix launcher must not shell out"),
    )
    path = desktop.create_shortcut(log=lambda *_: None)
    assert path is not None and path.name == "StemForge.desktop"
    assert path.is_file()
    body = path.read_text()
    assert "launch_ui.sh" in body and "Name=StemForge" in body


def test_macos_shortcut_writes_command(home, monkeypatch):
    monkeypatch.setattr(desktop.platform, "system", lambda: "Darwin")
    path = desktop.create_shortcut(log=lambda *_: None)
    assert path is not None and path.name == "StemForge.command"
    body = path.read_text()
    assert body.startswith("#!/bin/bash")
    assert "launch_ui.sh" in body


def test_launcher_scripts_exist_and_prepend_path():
    """The committed launchers exist and do the PATH-prepend the shortcut relies on."""
    bat = desktop._LAUNCH_BAT
    sh = desktop._LAUNCH_SH
    assert bat.is_file() and sh.is_file()
    bat_txt = bat.read_text()
    assert "WinGet\\Links" in bat_txt and "stemforge.cli ui" in bat_txt
    sh_txt = sh.read_text()
    assert "PATH=" in sh_txt and "stemforge.cli ui" in sh_txt
