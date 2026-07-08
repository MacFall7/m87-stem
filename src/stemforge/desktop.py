"""§5.10 desktop — a double-clickable launcher for the Gradio UI.

`stemforge desktop-shortcut` drops a platform-native shortcut on the Desktop:
a ``.lnk`` on Windows (via PowerShell ``WScript.Shell``), a ``.command`` on
macOS, a ``.desktop`` entry on Linux. Each points at the bundled launcher
(``scripts/launch_ui.{bat,sh}``), which prepends the tool dirs (ffmpeg / the
winget Links dir on Windows) and the repo dir to PATH before running
``python -m stemforge.cli ui`` — so ffmpeg and the isolated ``.venv-uvr``
resolve when the app is started by a double-click, not just from a dev shell.

Everything here shells out or writes files, so it stays out of import paths and
is fully mockable (tests patch ``desktop.subprocess.run`` / ``platform.system``
and point ``$HOME`` at a tmp dir — no real ``.lnk`` writes in CI).
"""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path
from typing import Callable

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _PROJECT_ROOT / "scripts"
_LAUNCH_BAT = _SCRIPTS / "launch_ui.bat"
_LAUNCH_SH = _SCRIPTS / "launch_ui.sh"


def desktop_dir() -> Path:
    """The user's Desktop (honors $HOME / %USERPROFILE%)."""
    return Path.home() / "Desktop"


def create_shortcut(log: Callable[[str], None] = print) -> Path | None:
    """Create a Desktop launcher for the current OS. Returns its path, or None."""
    system = platform.system()
    if system == "Windows":
        return _windows_shortcut(log)
    if system == "Darwin":
        return _macos_shortcut(log)
    return _linux_shortcut(log)


def _windows_shortcut(log: Callable[[str], None]) -> Path | None:
    lnk = desktop_dir() / "StemForge.lnk"
    ps = (
        "$W = New-Object -ComObject WScript.Shell; "
        f"$s = $W.CreateShortcut('{lnk}'); "
        f"$s.TargetPath = '{_LAUNCH_BAT}'; "
        f"$s.WorkingDirectory = '{_PROJECT_ROOT}'; "
        "$s.Description = 'Launch the StemForge UI'; "
        "$s.Save()"
    )
    try:
        _ensure_desktop()
        subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], check=True
        )
    except (subprocess.CalledProcessError, OSError) as e:
        log(f"could not create the Windows shortcut ({e})")
        return None
    log(f"created {lnk}")
    return lnk


def _macos_shortcut(log: Callable[[str], None]) -> Path | None:
    cmd = desktop_dir() / "StemForge.command"
    body = f'#!/bin/bash\ncd "{_PROJECT_ROOT}"\nexec bash "{_LAUNCH_SH}"\n'
    return _write_launcher(cmd, body, log)


def _linux_shortcut(log: Callable[[str], None]) -> Path | None:
    entry = desktop_dir() / "StemForge.desktop"
    body = (
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=StemForge\n"
        "Comment=Local stem separation UI\n"
        f'Exec=bash "{_LAUNCH_SH}"\n'
        f"Path={_PROJECT_ROOT}\n"
        "Terminal=true\n"
        "Categories=AudioVideo;Audio;\n"
    )
    return _write_launcher(entry, body, log)


def _write_launcher(path: Path, body: str, log: Callable[[str], None]) -> Path | None:
    try:
        _ensure_desktop()
        path.write_text(body, encoding="utf-8")
        path.chmod(0o755)
    except OSError as e:
        log(f"could not create the desktop launcher ({e})")
        return None
    log(f"created {path}")
    return path


def _ensure_desktop() -> None:
    desktop_dir().mkdir(parents=True, exist_ok=True)
