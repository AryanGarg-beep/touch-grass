#!/usr/bin/env bash
# Adds "Field Notes" to your app launcher. Run it once, from the venv that has faster-whisper.
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
PY="${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}"
PY="${PY:-$(command -v python3)}"
mkdir -p "$HOME/.local/share/applications"
cat > "$HOME/.local/share/applications/field-notes.desktop" <<DESK
[Desktop Entry]
Type=Application
Name=Field Notes
Comment=Voice notes from outside, turned into a journal
Exec="$PY" "$DIR/app.py"
Icon=audio-input-microphone
Terminal=false
Categories=AudioVideo;Utility;
DESK
echo "Added Field Notes to your app launcher (it runs with $PY)."
