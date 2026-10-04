#!/bin/bash
# One line on the Raspberry Pi 5 (Raspberry Pi OS 64-bit), after copying model_kit.zip to the Pi's home folder:
#
#   unzip -o model_kit.zip && bash model_kit/install.sh
#
# Installs what the kit needs (the PortAudio library, a Python environment in ~/heydelta), checks that the models
# load and that there is a microphone, then starts listening: say "Hey Delta", wait for the chime, say a command.
# Running it again is quick (nothing is installed twice) and is also the way to start the kit later.
#   bash model_kit/install.sh --no-run                 install + check only
#   bash model_kit/install.sh hf_only                  the model trained on the class data only
#   bash model_kit/install.sh hf_plus --rule balanced  further options go to command_pi.py (argmax|cautious|balanced)
# Asks for your password once if a system package is missing (sudo apt-get).
set -e
KIT="$(cd "$(dirname "$0")" && pwd)"
VENV="${HEYDELTA_VENV:-$HOME/heydelta}"
RUN=1; MODEL=hf_plus
[ "$1" = "--no-run" ] && { RUN=0; shift; }
case "$1" in hf_plus|hf_only) MODEL=$1; shift;; esac
step() { echo; echo "== $*"; }

step "1/4 system packages"
NEED=()
ldconfig -p | grep -q "libportaudio.so.2" || NEED+=(libportaudio2)
python3 -c "import ensurepip, venv" 2>/dev/null || NEED+=(python3-venv)
if [ ${#NEED[@]} -gt 0 ]; then
    echo "installing ${NEED[*]} (sudo)"; sudo apt-get update -qq && sudo apt-get install -y "${NEED[@]}"
else echo "ok"; fi

step "2/4 Python environment $VENV"
[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q numpy onnxruntime sounddevice scipy
echo "ok"

step "3/4 checks"
cd "$KIT"
"$VENV/bin/python" - <<'EOF'
import glob, onnxruntime as ort, sounddevice as sd
for f in sorted(glob.glob("listener/*.onnx") + glob.glob("models/*/*.onnx")):
    ort.InferenceSession(f, providers=["CPUExecutionProvider"])
    print("model loads:", f)
try:
    mic = sd.query_devices(kind="input")
    print("microphone:", mic["name"])
except Exception as e:
    raise SystemExit(f"no microphone found ({e}): plug in a USB microphone and run this again")
EOF
if [ -f "models/bcresnet6_$MODEL/reference_clips.npz" ]; then
    "$VENV/bin/python" "models/bcresnet6_$MODEL/command_pi.py" check
else echo "(no reference clips in this kit: probability check skipped)"; fi

if [ "$RUN" = 0 ]; then
    step "done. Start it with:  bash $KIT/install.sh"
    exit 0
fi
step "4/4 listening with bcresnet6_$MODEL: say \"Hey Delta\", wait for the chime, say a command (Ctrl+C to quit)"
exec "$VENV/bin/python" "models/bcresnet6_$MODEL/command_pi.py" live --deploy listener "$@"
