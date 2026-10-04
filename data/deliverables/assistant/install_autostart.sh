#!/bin/bash
# One-time setup: the assistant starts by itself at every boot (no login, no monitor) and restarts if it stops.
#
#   cd ~/AI231ME2RedondoAssistantDeploy
#   conda activate <your env>            # or: source ~/AI231ME2RedondoAssistant/bin/activate  (the Python that runs the assistant)
#   bash install_autostart.sh            # options are passed to assistant.py, e.g.: bash install_autostart.sh --model hf_only
#   bash install_autostart.sh --remove   # undo: no more autostart
#
# What it sets up:
#   ~/.config/systemd/user/heydelta-assistant.service   runs start_assistant.sh with this Python
#   loginctl enable-linger                              your services start at boot without logging in
#   /etc/sudoers.d/heydelta-ofono                       lets the launcher run ONE command without a password:
#                                                       "systemctl restart ofono" (fixes calls if oFono starts too early)
set -e
UNIT="$HOME/.config/systemd/user/heydelta-assistant.service"

if [ "$1" = "--remove" ]; then
    systemctl --user disable --now heydelta-assistant 2>/dev/null || true
    rm -f "$UNIT"
    systemctl --user daemon-reload
    sudo rm -f /etc/sudoers.d/heydelta-ofono
    echo "Autostart removed (linger left on: sudo loginctl disable-linger $USER to undo that too)."
    exit 0
fi

KIT="$(cd "$(dirname "$0")" && pwd)"
PY="$(command -v python3)"
"$PY" -c "import numpy, onnxruntime, sounddevice, piper" 2>/dev/null || {
    echo "This python3 ($PY) does not have the assistant's packages. Activate the environment you run the"
    echo "assistant with (conda activate ... / source .../bin/activate) and run this script again."
    exit 1
}
[ -f "$HOME/.heydelta/config.json" ] || echo "Note: no ~/.heydelta/config.json yet (python3 assistant.py --init-config); devices stay simulated."
chmod +x "$KIT/start_assistant.sh"

mkdir -p "$(dirname "$UNIT")"
cat > "$UNIT" <<EOF
[Unit]
Description=Hey Delta voice assistant
After=pipewire.service wireplumber.service
Wants=pipewire.service wireplumber.service

[Service]
Environment=HEYDELTA_PYTHON=$PY
ExecStart="$KIT/start_assistant.sh" $*
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
EOF

sudo loginctl enable-linger "$USER"
echo "$USER ALL=(root) NOPASSWD: /usr/bin/systemctl restart ofono" | sudo tee /etc/sudoers.d/heydelta-ofono >/dev/null
sudo chmod 440 /etc/sudoers.d/heydelta-ofono
sudo visudo -cf /etc/sudoers.d/heydelta-ofono >/dev/null

systemctl --user daemon-reload
systemctl --user enable heydelta-assistant
systemctl --user restart heydelta-assistant
sleep 3
systemctl --user --no-pager status heydelta-assistant | head -5
echo
echo "Done: the assistant now starts at every boot."
echo "  live output:   journalctl --user -u heydelta-assistant -f"
echo "  stop / start:  systemctl --user stop heydelta-assistant   /   systemctl --user start heydelta-assistant"
echo "  (stop it before running assistant.py by hand: only one program can use the microphone)"
