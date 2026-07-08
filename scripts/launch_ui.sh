#!/usr/bin/env bash
# StemForge UI launcher (macOS/Linux).
# Prepend common tool dirs (ffmpeg via Homebrew/usr-local) and the repo dir to
# PATH so ffmpeg and the isolated .venv-uvr resolve, then start the UI
# (opens the browser).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$REPO:/usr/local/bin:/opt/homebrew/bin:$PATH"
cd "$REPO"
exec python -m stemforge.cli ui
