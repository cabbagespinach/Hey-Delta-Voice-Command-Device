#!/bin/bash
# Starts the Hey Delta assistant. Run by the heydelta-assistant service at every boot (install_autostart.sh);
# also fine to run by hand. Options are passed on to assistant.py (e.g. --model hf_only).
#
# Before the assistant starts it:
#   - waits (up to 60 s) for the audio system and a microphone,
#   - fixes calls if oFono started before WirePlumber this boot (restarts oFono once; allowed by install_autostart.sh),
#   - reconnects the phone from ~/.heydelta/config.json in the background (calls/texts; fine if it is away).
cd "$(dirname "$(readlink -f "$0")")" || exit 1
PY="${HEYDELTA_PYTHON:-python3}"
export OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
BUS="/run/user/$(id -u)/bus"
[ -z "$DBUS_SESSION_BUS_ADDRESS" ] && [ -S "$BUS" ] && export DBUS_SESSION_BUS_ADDRESS="unix:path=$BUS"

for _ in $(seq 30); do
    systemctl --user is-active --quiet wireplumber && arecord -l 2>/dev/null | grep -q "^card" && break
    sleep 2
done

if systemctl is-active --quiet ofono; then
    since=$(systemctl show ofono -p ActiveEnterTimestamp --value)
    if journalctl -u ofono --since "$since" --no-pager 2>/dev/null | grep -q "UUID already registered"; then
        echo "(oFono started before WirePlumber: restarting it so calls work)"
        sudo -n /usr/bin/systemctl restart ofono && sleep 3
    fi
fi

ADDR=$("$PY" -c "import json, os; c = json.load(open(os.path.expanduser('~/.heydelta/config.json'))); print((c.get('phone') or {}).get('bt_address', ''))" 2>/dev/null)
if [ -n "$ADDR" ]; then
    (for _ in 1 2 3 4 5 6; do
        bluetoothctl info "$ADDR" 2>/dev/null | grep -q "Connected: yes" && break
        bluetoothctl connect "$ADDR" >/dev/null 2>&1
        sleep 10
    done) &
fi

exec "$PY" assistant.py "$@"
