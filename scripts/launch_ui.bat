@echo off
REM StemForge UI launcher (Windows).
REM Prepend the winget Links dir (ffmpeg) and the repo dir to PATH so ffmpeg and
REM the isolated .venv-uvr resolve, then start the UI (opens the browser).
setlocal
set "REPO=%~dp0.."
set "PATH=%LOCALAPPDATA%\Microsoft\WinGet\Links;%REPO%;%PATH%"
cd /d "%REPO%"
python -m stemforge.cli ui
