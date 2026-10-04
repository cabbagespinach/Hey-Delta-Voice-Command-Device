#!/bin/bash
# One line on the Raspberry Pi 5 (Raspberry Pi OS 64-bit), after copying assistant_kit.zip to the Pi's home folder:
#
#   unzip -o assistant_kit.zip && bash assistant_kit/install.sh
#
# Installs what the assistant needs (PortAudio, ffmpeg, gdbus; a Python environment in ~/assistant with Piper),
# creates the private settings file ~/.heydelta/config.json if there is none (empty = everything simulated), runs
# the offline tests, then starts the assistant (the dashboard address is printed in a box).
# Running it again is quick (nothing is installed twice) and is also the way to start the assistant later.
#   bash assistant_kit/install.sh --no-run       install + tests only
#   bash assistant_kit/install.sh --autostart    install + tests, then start at every boot (install_autostart.sh)
#   bash assistant_kit/install.sh --model hf_only   further options go to assistant.py
# Asks for your password once if a system package is missing (sudo apt-get). Calls and texts need the one-time
# phone setup in README_ASSISTANT.md ("Phone"); devices are set in ~/.heydelta/config.json ("Your devices").
set -e
KIT="$(cd "$(dirname "$0")" && pwd)"
VENV="${HEYDELTA_VENV:-$HOME/assistant}"
MODE=run
case "$1" in --no-run) MODE=none; shift;; --autostart) MODE=autostart; shift;; esac
step() { echo; echo "== $*"; }

step "1/5 system packages"
NEED=()
ldconfig -p | grep -q "libportaudio.so.2" || NEED+=(libportaudio2)
command -v ffmpeg >/dev/null || NEED+=(ffmpeg)
command -v gdbus >/dev/null || NEED+=(libglib2.0-bin)
python3 -c "import ensurepip, venv" 2>/dev/null || NEED+=(python3-venv)
if [ ${#NEED[@]} -gt 0 ]; then
    echo "installing ${NEED[*]} (sudo)"; sudo apt-get update -qq && sudo apt-get install -y "${NEED[@]}"
else echo "ok"; fi

step "2/5 Python environment $VENV"
[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q numpy scipy soundfile sounddevice onnxruntime piper-tts==1.8.0 python-kasa
echo "ok"

cd "$KIT"
step "3/5 private settings"
"$VENV/bin/python" assistant.py --init-config

step "4/5 tests (fake bulb and phone, Piper voice, no microphone)"
"$VENV/bin/python" test_assistant.py --tts
"$VENV/bin/python" assistant.py --selftest
"$VENV/bin/python" -c "import sounddevice as sd; print('microphone:', sd.query_devices(kind='input')['name'])" 2>/dev/null ||
    { echo "no microphone found: plug in a USB microphone and run this again"; exit 1; }

case "$MODE" in
none)
    step "done. Start it with:  bash $KIT/install.sh   (or at every boot: bash $KIT/install.sh --autostart)";;
autostart)
    step "5/5 autostart at every boot"
    source "$VENV/bin/activate"
    exec bash install_autostart.sh "$@";;
run)
    step "5/5 starting: say \"Hey Delta\", wait for the chime, say a command (Ctrl+C to quit)"
    exec "$VENV/bin/python" assistant.py "$@";;
esac
