# "Hey Delta" assistant: the commands, actually carried out

Say **"Hey Delta"**, wait for the rising chime, say a command. The Pi recognises it (BC-ResNet-6, HF + our data,
balanced rule by default), carries it out with your real devices where you have them (Tapo bulb, phone over
Bluetooth, Open-Meteo weather, music in `~/Music/Party Music`) and a simulated home for the rest (thermostat), and
answers out loud with an offline voice (Piper, "Amy"). The home's state is shown live in a browser on your laptop.

## What each command does

| Command | What happens | Spoken reply (example) |
|---|---|---|
| PLAY_MUSIC | plays the songs in `~/Music/Party Music` (shuffled; `--music` for another folder); after PAUSE it resumes. If no song can be opened it says so (the reason is printed) | "Playing music." |
| PAUSE / STOP | pauses / stops the music; STOP with no music playing cancels running timers | "Paused." / "Stopped." |
| NEXT | next song | "Next song." |
| VOLUME_UP / VOLUME_DOWN | ±10% for music, replies and rings (0–100%) | "Volume 70 percent." |
| TIMER_10S / 30S / 1MIN | a real countdown; rings and speaks when done (several can run) | "Timer set for 30 seconds." |
| ALARM_6AM / 8AM / 9PM | rings at the next 6 AM / 8 AM / 9 PM; kept across restarts | "Alarm set for 6 AM tomorrow." |
| CREATE_REMINDER_* | adds "drink water" / "study" / "exercise" to your reminders | "Okay, I'll remind you to study." |
| LIST_REMINDERS | reads your reminders and pending alarms | "You have reminders to study and drink water." |
| TIME | the Pi's clock | "It's 3:42 PM." |
| WEATHER | live forecast from Open-Meteo for the location in the config (internet needed); without it, `weather.json` or "offline" | "In Quezon City it's 31 degrees and partly cloudy…" |
| LIGHT_ON / LIGHT_OFF, BRIGHTNESS_*, COLOR_* | the Tapo bulb (simulated if not set up: the reply then ends in "(simulated)"); a white-only bulb answers that it can't change colour | "Changing the lights to blue." |
| TEMPERATURE_* | simulated thermostat (shown on the dashboard) | "Setting the temperature to 22 degrees." |
| CALL | calls the contact in the config through the phone (Bluetooth HFP, oFono) | "Calling Mom." |
| MESSAGE | texts the fixed message to that contact (Bluetooth MAP, obexd) | "Sending I'm on my way. to Mom." |
| not a command / not sure | nothing happens | "Sorry, I didn't catch that." |

After every command the screen shows the home's state, e.g.
`[home] lights ON (60%, blue) | thermostat 22°C | volume 60% | music playing | timers 1 | alarms 1 | reminders 2`.
State is saved in `~/.heydelta/state.json`.

While the assistant is talking, a wake-up is ignored (it would be its own voice), and music is turned down while it
listens and answers. For the class benchmark, use the separate `vcm_bench_kit` (no actions there).

## Build the kit from the repository (any computer with Python 3)

```bash
git clone https://github.com/cabbagespinach/Hey-Delta-Voice-Command-Device.git && cd Hey-Delta-Voice-Command-Device
python3 data/deliverables/assistant/make_assistant_kit.py     # -> data/deliverables/assistant/AI231ME2RedondoAssistantDeploy.zip
scp data/deliverables/assistant/AI231ME2RedondoAssistantDeploy.zip <user>@<pi-ip>:~     # <pi-ip>: run  hostname -I  on the Pi (first number)
```

The kit uses the trained models committed in the repository (wake word `data/deliverables/model/deploy/`, commands
`data/deliverables/command_classifier/model/export/bcresnet6_hf_plus` and `_hf_only`). The Piper voice "Amy" is
downloaded once from [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) (v1.0.0, about 63 MB; voice
licence: see its model card), because it is not ours to redistribute. The 16 reference clips used by
`assistant.py --selftest` are audio and therefore not in the repository: a kit built from a fresh clone skips the
selftest (everything else works). They are made again by the ONNX export step of a full reproduction
(`docs/REPRODUCE.md`).

### What you have to change for your own setup

Nothing in the code. Everything personal goes in one private file on the Pi, never in the repository:

| What | Where | Needed for |
|---|---|---|
| weather location | `~/.heydelta/config.json` → `location` | WEATHER (else "offline") |
| Tapo bulb IP + TP-Link login | `~/.heydelta/config.json` → `tapo` | lights (else simulated) |
| contact name + number, message text | `~/.heydelta/config.json` → `contact`, `message` | CALL, MESSAGE |
| phone's Bluetooth address | `~/.heydelta/config.json` → `phone.bt_address` | CALL, MESSAGE (+ one-time phone setup below) |
| songs | `~/Music/Party Music` (or `--music <folder>`) | PLAY_MUSIC |
| microphone / speaker | `--mic N`, `--speaker N` (only if the defaults are wrong) | listening, replies |
| autostart options | arguments to `bash install_autostart.sh`, e.g. `--model hf_only` | starting at boot |

`python3 assistant.py --init-config` writes the file with empty fields (readable only by you); details in
"Your devices" below. Anything left empty stays simulated, so the assistant also runs with no devices at all.

## Install and run: one line (Pi 5, Raspberry Pi OS 64-bit)

```bash
unzip -o AI231ME2RedondoAssistantDeploy.zip && bash AI231ME2RedondoAssistantDeploy/install.sh
```

`install.sh` installs PortAudio, ffmpeg and gdbus (asks for your password once if missing) and a Python environment
in `~/AI231ME2RedondoAssistant`, creates `~/.heydelta/config.json` if there is none (empty = everything simulated), runs the
offline tests, then starts the assistant. Run `bash AI231ME2RedondoAssistantDeploy/install.sh` again to start it later (nothing is
installed twice); `--no-run` only installs and tests, `--autostart` makes it start at every boot instead (see
"Start at every boot"), further options go to `assistant.py` (e.g. `--model hf_only`). Your devices and the phone
are set up separately (below).

By hand instead:

```bash
unzip AI231ME2RedondoAssistantDeploy.zip && cd AI231ME2RedondoAssistantDeploy
python3 -m venv ~/AI231ME2RedondoAssistant && source ~/AI231ME2RedondoAssistant/bin/activate   # or activate your own environment (e.g. conda); use the same one later
pip install numpy scipy soundfile sounddevice onnxruntime piper-tts==1.8.0 python-kasa
which gdbus || sudo apt install libglib2.0-bin       # used for the phone (oFono / obexd over D-Bus); calls/texts: see Phone below
which ffmpeg || sudo apt install ffmpeg             # songs in .m4a / .aac (e.g. from Apple Music) and odd .mp3 files
python3 test_assistant.py --tts --net   # 22 tests with fake bulb/phone (+ Piper voice, + live Open-Meteo)
python3 assistant.py --selftest         # reference clips -> model -> actions (no audio; skipped in a kit built from a fresh clone)
```

## Your devices: the private settings file

```bash
python3 assistant.py --init-config      # creates ~/.heydelta/config.json, readable only by you
nano ~/.heydelta/config.json
```

| Field | What to put |
|---|---|
| `location` | `name`, `latitude`, `longitude` of where the weather is for (e.g. Quezon City 14.65, 121.05) |
| `tapo` | the bulb's IP address (give it a fixed address on the router) + your TP-Link (Tapo app) email and password. If it fails to log in, turn on third-party compatibility in the Tapo app |
| `contact` | `name` (spoken) and `number` of the one person CALL and MESSAGE go to |
| `message` | the text MESSAGE sends, e.g. "I'm on my way." |
| `phone` | `bt_address` of your paired phone (`bluetoothctl devices`) |

Anything left empty stays simulated, and the reply says so. Calls and texts need a one-time setup: see Phone below.

## Phone: calls and texts (one-time setup on the Pi)

Pairing alone is not enough: the Pi needs two Bluetooth phone services that Raspberry Pi OS does not install.

```bash
sudo apt install ofono bluez-obexd python3-dbus     # oFono = calls (HFP), obexd + python3-dbus = texts (MAP)
mkdir -p ~/.config/wireplumber/wireplumber.conf.d   # let oFono, not PipeWire, handle the phone's call link
printf 'monitor.bluez.properties = {\n  bluez5.hfphsp-backend = "ofono"\n}\n' \
  > ~/.config/wireplumber/wireplumber.conf.d/51-hfp-ofono.conf
sudo systemctl enable --now ofono
systemctl --user restart wireplumber
bluetoothctl disconnect XX:XX:XX:XX:XX:XX; bluetoothctl connect XX:XX:XX:XX:XX:XX   # reconnect the phone
```

Check (each should show your phone's address as `dev_XX_XX_...`):

```bash
gdbus call --system --dest org.ofono --object-path / --method org.ofono.Manager.GetModems
python3 assistant.py --do CALL                      # rings your contact; the reason is printed if it fails
python3 assistant.py --do MESSAGE
```

- **iPhone:** calls work, **texts do not**: iOS lets a Bluetooth device read message notifications but not send texts
  (Apple does not support sending over MAP). MESSAGE then says it could not send. On Android, allow "Message access"
  (and "Phone calls") for the Pi in the phone's Bluetooth settings.
- Texts are sent by `map_send.py` in one process (obexd drops the connection when the process that opened it
  exits). It uses the Pi's own Python (`python3-dbus`) when your environment (e.g. conda) lacks `dbus`, and finds
  the desktop session's D-Bus by itself, also over SSH.
- If calls stop working after a reboot: `systemctl --user restart wireplumber; sudo systemctl restart ofono`, then
  reconnect the phone (oFono must start after WirePlumber has the "ofono" setting).
- While oFono handles the call link, the Pi is not a Bluetooth headset for the phone's audio. The call's sound
  stays on the phone unless you route it, which is fine for the demo ("Calling Mom" + the phone dials).

## Run

```bash
python3 assistant.py                    # Ctrl+C to quit; prints the dashboard address
python3 assistant.py --do LIGHT_ON      # one command, no microphone and no model: test a device
python3 assistant.py --do PLAY_MUSIC    # test the music: keeps playing until Ctrl+C
python3 assistant.py --type             # type command names one by one (TIMER_10S, CALL, ...)
python3 assistant.py --list             # all command names
```

Options: `--model hf_only`, `--rule balanced|argmax`, `--music <folder>` (default `~/Music/Party Music`), `--mic N`,
`--speaker N` (list devices: `python3 -c "import sounddevice; print(sounddevice.query_devices())"`), `--no-voice`
(print replies only), `--dashboard-port 0` (no dashboard).

## Start automatically at every boot

Once everything works by hand, make the assistant start by itself whenever the Pi is switched on (no login, no
monitor needed; it also restarts if it ever stops):

```bash
cd ~/AI231ME2RedondoAssistantDeploy                      # wherever the kit is
source ~/AI231ME2RedondoAssistant/bin/activate         # the same Python environment you installed into (conda: conda activate <env>)
bash install_autostart.sh               # asks for your password once; options go to assistant.py, e.g. --model hf_only
```

After a reboot, wait about 30-60 s, then say "Hey Delta". Useful commands:

```bash
journalctl --user -u heydelta-assistant -f      # what it hears and does, live (Ctrl+C leaves it running)
systemctl --user stop heydelta-assistant        # stop it, e.g. before running assistant.py by hand
systemctl --user start heydelta-assistant       # start it again
bash install_autostart.sh --remove              # no more autostart
```

At each start, `start_assistant.sh` waits for the audio system and microphone, reconnects your phone, and restarts
oFono once if it started too early (the only command it may run as root). If you move or replace the kit folder,
run `bash install_autostart.sh` again.

## The dashboard (no monitor needed)

When it starts, the assistant prints a DASHBOARD box with the Pi's number-style address, e.g. `http://192.168.1.23:8080`
(the `.local` name often does not work; use the number), what to type, and what to do if the page does not open
(browser forcing https, macOS/iPhone "Local Network" permission, same network). The box is printed again whenever the
address changes, e.g. when the hotspot connects after an autostart; with autostart, see it with
`journalctl --user -u heydelta-assistant | grep -E "http://[0-9]+\." | tail -2`. Open it in your laptop's browser: lights (with the bulb's colour), thermostat, music and volume, timers counting down, alarms,
reminders, phone, weather, and the last commands with their replies, updated twice a second. It is read-only and
needs no internet, only the same network.

**Demo day:** put the Pi, the laptop and the bulb on the same network. A phone hotspot works, but the bulb must be
connected to that hotspot beforehand (Tapo app). Weather needs internet (the hotspot's).

Use the echo-cancelling output as the speaker if you set one up for the listener (`data/deliverables/model/deploy/README_DEPLOY.md` in the repository), so the Pi does
not hear its own replies.

## Files

`assistant.py` (main loop), `actions.py` (what each command does), `devices.py` (weather, Tapo bulb, phone),
`dashboard.py` (the browser page), `audio_out.py` (one speaker output: music, replies, rings), `tts.py` (Piper voice, 2 CPU threads), `command_pi.py` + `models/` (command models, official cutoffs),
`listener/` (wake word), `voices/` (Piper "Amy" medium, from rhasspy/piper-voices; see its model card for the
voice's licence), `test_assistant.py`.
