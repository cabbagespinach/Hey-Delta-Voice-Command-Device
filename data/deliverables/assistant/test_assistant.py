#!/usr/bin/env python3
"""
Tests for the assistant's actions and audio mixer, without microphone or speaker.

    python3 test_assistant.py            # or: python3 -m pytest test_assistant.py
    python3 test_assistant.py --tts      # also synthesise one reply with the Piper voice (slower)
"""
from pathlib import Path
import csv, datetime as dt, json, sys, tempfile, time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from actions import Home, next_occurrence, spoken_time, State                # noqa: E402
from audio_out import AudioOut, OUT_SR                                       # noqa: E402

LABELS = HERE / "label_map_schema_b.csv"
if not LABELS.exists():
    LABELS = HERE.parents[0] / "model/deploy/label_map_schema_b.csv"
CLASSES = [r["class"] for r in csv.DictReader(open(LABELS))] + ["unknown"]


class FakeAudio:
    def __init__(self, music=True):
        self.music, self.playing, self.paused, self.volume, self.rings, self.calls = music, False, False, None, 0, []

    def has_music(self): return self.music
    def play_music(self): self.playing, self.paused = True, False; self.calls.append("play")
    def pause_music(self): self.paused = True; self.calls.append("pause")
    def resume_music(self): self.paused = False
    def stop_music(self): self.playing = self.paused = False; self.calls.append("stop")
    def next_track(self): self.playing, self.paused = True, False; self.calls.append("next")
    def is_playing(self): return self.playing and not self.paused
    def is_paused(self): return self.playing and self.paused
    def set_volume(self, v): self.volume = v
    def ring(self): self.rings += 1


class Clock:
    def __init__(self, t): self.t = t
    def __call__(self): return self.t


def home(**kw):
    said = []
    h = Home(kw.pop("audio", FakeAudio()), said.append, **kw)
    return h, said


def test_every_class_has_a_reply():
    h, _ = home(timer_scale=1000)                          # timers far in the future
    for c in CLASSES:
        r = h.handle(c)
        assert isinstance(r, str) and r.strip(), c
    assert h.handle("unknown") == "Sorry, I didn't catch that."
    h.close()


def test_timer_rings():
    a = FakeAudio()
    h, said = home(audio=a, timer_scale=0.01)              # 10 s timer -> 0.1 s
    assert h.handle("TIMER_10S") == "Timer set for 10 seconds."
    assert h.handle("TIMER_1MIN") == "Timer set for one minute."
    time.sleep(1.0)
    assert a.rings == 2 and any("10 second timer is done" in s for s in said)
    h.close()


def test_stop_cancels_timer_when_nothing_plays():
    a = FakeAudio()
    h, said = home(audio=a, timer_scale=0.05)
    h.handle("TIMER_30S")
    assert h.handle("STOP") == "Timer cancelled."
    time.sleep(2.0)
    assert a.rings == 0
    h.close()


def test_alarms_today_tomorrow_and_ring():
    clock = Clock(dt.datetime(2026, 10, 2, 7, 0))
    a = FakeAudio()
    h, said = home(audio=a, clock=clock)
    assert h.handle("ALARM_6AM") == "Alarm set for 6 AM tomorrow."
    assert h.handle("ALARM_8AM") == "Alarm set for 8 AM today."
    assert h.handle("ALARM_9PM") == "Alarm set for 9 PM today."
    assert "Alarms at 8 AM, 9 PM, 6 AM." in h.handle("LIST_REMINDERS")
    clock.t = dt.datetime(2026, 10, 2, 8, 0, 1)
    time.sleep(1.5)
    assert a.rings == 1 and any("8 AM" in s for s in said) and len(h.state.alarms) == 2
    h.close()


def test_volume_clamps():
    a = FakeAudio()
    h, _ = home(audio=a)
    for _ in range(10):
        h.handle("VOLUME_UP")
    assert h.state.volume == 100 and a.volume == 1.0 and "maximum" in h.handle("VOLUME_UP")
    for _ in range(12):
        h.handle("VOLUME_DOWN")
    assert h.state.volume == 0 and a.volume == 0.0
    h.close()


def test_music_flow():
    a = FakeAudio()
    h, _ = home(audio=a)
    assert h.handle("PAUSE") == "Nothing is playing."
    assert h.handle("PLAY_MUSIC") == "Playing music." and a.is_playing()
    assert h.handle("PAUSE") == "Paused." and a.is_paused()
    assert h.handle("NEXT") == "Next song." and a.is_playing()
    assert h.handle("STOP") == "Stopped." and not a.is_playing()
    h2, _ = home(audio=FakeAudio(music=False))
    assert "music folder" in h2.handle("PLAY_MUSIC")
    h.close(); h2.close()


def test_real_player_files():
    """The real music player (no sound card): hidden macOS copies skipped, a broken song is reported, not faked."""
    import tempfile
    import soundfile as sf
    from audio_out import AudioOut
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "._song.wav").write_bytes(b"junk")                      # macOS AppleDouble copy
        (d / "broken.mp3").write_bytes(b"not audio")
        bad = AudioOut(d, enabled=False)
        h, _ = home(audio=bad)
        assert bad.has_music() and [s.name for s in bad._songs()] == ["broken.mp3"]
        assert h.handle("PLAY_MUSIC").startswith("Sorry, I couldn't play") and "broken.mp3" in bad.error
        sf.write(d / "song.wav", 0.3 * np.sin(np.arange(22050) / 10).astype(np.float32), 22050)
        h.close()
        good = AudioOut(d, enabled=False, seed=0)
        h, _ = home(audio=good)
        assert h.handle("PLAY_MUSIC") == "Playing music." and good.is_playing()
        out = np.zeros((1024, 1), np.float32)
        while good.track != "song" and good.is_playing():             # skip past broken.mp3 if it came first
            good._advance()
        good._callback(out, 1024)
        assert good.track == "song" and np.abs(out).max() > 0.01     # music actually reaches the output
        h.close()


def test_lights_temperature_reminders_and_state_file():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "state.json"
        h, _ = home(state_path=p)
        assert h.handle("COLOR_BLUE") == "Changing the lights to blue (simulated)."      # no bulb set up
        h.handle("BRIGHTNESS_20"); h.handle("TEMPERATURE_18"); assert h.handle("LIGHT_OFF") == "Lights off (simulated)."
        h.handle("CREATE_REMINDER_STUDY"); h.handle("CREATE_REMINDER_DRINK_WATER"); h.handle("CREATE_REMINDER_STUDY")
        h.close()
        s = json.loads(p.read_text())
        assert s["color"] == "blue" and s["brightness"] == 20 and s["temperature"] == 18 and not s["lights_on"]
        assert s["reminders"] == ["study", "drink water"]
        h2, _ = home(state_path=p)                           # survives a restart
        assert h2.handle("LIST_REMINDERS") == "You have reminders to study and drink water."
        h2.close()


def test_weather_offline_and_file():
    with tempfile.TemporaryDirectory() as d:
        h, _ = home(weather_path=Path(d) / "weather.json")
        assert "offline" in h.handle("WEATHER")
        (Path(d) / "weather.json").write_text(json.dumps({"summary": "light rain", "temperature": 27}))
        assert h.handle("WEATHER") == "Today: light rain, around 27 degrees."
        h.close()


def test_time_and_helpers():
    assert spoken_time(dt.datetime(2026, 1, 1, 21, 5)) == "9:05 PM"
    assert spoken_time(dt.datetime(2026, 1, 1, 0, 0)) == "12 AM"
    now = dt.datetime(2026, 1, 1, 6, 0)
    assert next_occurrence(now, 6, 0) == dt.datetime(2026, 1, 2, 6, 0)   # exactly now -> tomorrow
    h, _ = home(clock=Clock(dt.datetime(2026, 1, 1, 15, 42)))
    assert h.handle("TIME") == "It's 3:42 PM."
    h.close()


def test_mixer_music_and_oneshots():
    import soundfile as sf
    with tempfile.TemporaryDirectory() as d:
        sf.write(str(Path(d) / "song.wav"), 0.5 * np.ones(22050, np.float32), 22050)   # 1 s at 22.05 kHz
        a = AudioOut(Path(d), enabled=False, seed=0)
        assert a.has_music()
        a.play_music()
        assert a.is_playing() and len(a._song) == OUT_SR
        a.enabled = True                                    # feed the mixer by hand (no sound device)
        a._pos = 10000                                      # mid-song: away from the resampling filter's edge
        a.play_clip(0.2 * np.ones(4800, np.float32))
        out = np.zeros((1024, 1), np.float32)
        a.set_volume(1.0)
        a._callback(out, 1024)
        assert np.allclose(out[:, 0], 0.5 * 0.5 + 0.2, atol=1e-3)   # music (x0.5 below speech) + one-shot
        a.set_duck(True)
        a._callback(out, 1024)
        assert out[0, 0] < 0.3
        a.pause_music()
        assert a.is_paused()
        a.play_music()                                      # resumes, does not restart
        assert a.is_playing() and a._pos > 0


def test_tts_synthesises():
    from tts import Speaker
    voice = HERE / "voices/en_US-amy-medium.onnx"
    if not voice.exists():
        voice = HERE.parents[2] / "voices/piper_raw/en/en_US/amy/medium/en_US-amy-medium.onnx"
    s = Speaker(AudioOut(enabled=False), voice, threads=2)
    y, sr = s.synth("Timer set for 30 seconds.")
    assert sr > 0 and 0.8 < len(y) / sr < 5 and np.abs(y).max() > 0.05


# ---------------------------------------------------------------- devices (fakes; no Bluetooth / bulb needed)
import devices                                                              # noqa: E402


class FakeBulb:
    def __init__(self, ok=True, color=True): self.ok, self.color_ok, self.calls = ok, color, []
    def _do(self, x):
        if not self.ok: raise OSError("no route to host")
        self.calls.append(x)
    def on(self): self._do("on")
    def off(self): self._do("off")
    def brightness(self, p): self._do(("brightness", p))
    def can_color(self): return self.color_ok
    def color(self, c): self._do(("color", c))


class FakePhone:
    def __init__(self, ok=True): self.ok, self.calls = ok, []
    def call(self):
        if not self.ok: raise RuntimeError("no modem")
        self.calls.append("call")
    def text(self):
        if not self.ok: raise RuntimeError("refused")
        self.calls.append("text")


class FakeDevices:
    def __init__(self, bulb=None, phone=None, weather=None):
        self.bulb, self.phone, self.weather = bulb, phone, weather
        self.contact_name, self.message = "Mom", "I'm on my way."


def test_bulb_actions_and_failures():
    b = FakeBulb()
    h, _ = home(devices=FakeDevices(bulb=b))
    assert h.handle("LIGHT_ON") == "Lights on." and h.state.lights_on
    assert h.handle("BRIGHTNESS_60") == "Brightness set to 60 percent." and ("brightness", 60) in b.calls
    assert h.handle("COLOR_GREEN") == "Changing the lights to green." and ("color", "green") in b.calls
    assert h.handle("LIGHT_OFF") == "Lights off." and not h.state.lights_on
    h2, _ = home(devices=FakeDevices(bulb=FakeBulb(ok=False)))
    assert h2.handle("LIGHT_ON") == "I couldn't reach the light." and not h2.state.lights_on   # state unchanged
    h3, _ = home(devices=FakeDevices(bulb=FakeBulb(color=False)))
    assert h3.handle("COLOR_RED") == "This light can't change color."
    for x in (h, h2, h3): x.close()


def test_call_and_message():
    p = FakePhone()
    h, said = home(devices=FakeDevices(phone=p))
    assert h.handle("CALL") == "Calling Mom." and p.calls == ["call"]
    assert h.handle("MESSAGE") == "Sending I'm on my way. to Mom."
    time.sleep(0.3)
    assert "text" in p.calls and h.state.last_message
    h2, said2 = home(devices=FakeDevices(phone=FakePhone(ok=False)))
    assert "couldn't start the call" in h2.handle("CALL")
    h2.handle("MESSAGE"); time.sleep(0.3)
    assert any("didn't go through" in s for s in said2)
    h3, _ = home()
    assert h3.handle("CALL") == "Calling isn't set up yet." and h3.handle("MESSAGE") == "Texting isn't set up yet."
    for x in (h, h2, h3): x.close()


def test_weather_device_and_error():
    class W:
        def __init__(self, ok): self.ok = ok
        def report(self):
            if not self.ok: raise OSError("offline")
            return "In Quezon City it's 30 degrees."
    h, _ = home(devices=FakeDevices(weather=W(True)))
    assert h.handle("WEATHER") == "In Quezon City it's 30 degrees."
    h2, _ = home(devices=FakeDevices(weather=W(False)))
    assert "couldn't reach the weather" in h2.handle("WEATHER")
    h.close(); h2.close()


def test_bmessage_length():
    m = devices.bmessage("+639170000000", "I'm on my way.")
    body = "BEGIN:MSG\r\nI'm on my way.\r\nEND:MSG\r\n"
    assert f"LENGTH:{len(body.encode())}\r\n" + body in m and "TEL:+639170000000" in m and m.endswith("END:BMSG\r\n")


def test_phone_dbus_calls(monkeypatch=None):
    seen = []
    modems = ("([(objectpath '/phonesim', {'Online': <false>, 'Interfaces': <['org.ofono.VoiceCallManager']>}), "
              "(objectpath '/hfp/org/bluez/hci0/dev_AA_BB_CC_DD_EE_FF', {'Online': <true>, "
              "'Interfaces': <['org.ofono.VoiceCallManager', 'org.ofono.Handsfree']>})],)")

    def fake(bus, dest, path, method, *args, timeout=10.0):
        seen.append((bus, path, method, args))
        if method.endswith("GetModems"): return modems
        if method.endswith("CreateSession"): return "(objectpath '/org/bluez/obex/client/session0',)"
        if method.endswith("PushMessage"):
            assert "TEL:+639170000000" in Path(args[0]).read_text()
            return "(objectpath '/org/bluez/obex/client/session0/transfer0', {'Status': <'queued'>})"
        if method.endswith("Properties.Get"): return "(<'complete'>,)"
        return "()"
    old = devices.gdbus
    devices.gdbus = fake
    try:
        ph = devices.Phone({"contact": {"name": "Mom", "number": "+639170000000"}, "message": "I'm on my way.",
                            "phone": {"bt_address": "aa:bb:cc:dd:ee:ff"}})
        ph.call()
        dial = [s for s in seen if s[2].endswith("Dial")][0]
        assert dial[1] == "/hfp/org/bluez/hci0/dev_AA_BB_CC_DD_EE_FF" and dial[3] == ("+639170000000", "")
        # texts: ONE helper process (map_send.py) with the address and a bMessage file
        import subprocess, types
        ran = []

        def fake_run(cmd, **kw):
            ran.append((cmd, Path(cmd[3]).read_text(), kw["env"]))
            return types.SimpleNamespace(returncode=0, stderr="", stdout="")
        old_run, devices.subprocess.run = devices.subprocess.run, fake_run
        try:
            ph.text()
            cmd, bmsg, env = ran[0]
            assert cmd[1].endswith("map_send.py") and cmd[2] == "AA:BB:CC:DD:EE:FF" and "TEL:+639170000000" in bmsg
            assert not Path(cmd[3]).exists()                          # temp file removed
            devices.subprocess.run = lambda cmd, **kw: types.SimpleNamespace(
                returncode=1, stdout="", stderr="ModuleNotFoundError: No module named 'dbus'")
            try:
                ph.text()
                raise AssertionError("a failed send must raise")
            except RuntimeError as e:
                assert "python3-dbus" in str(e)
        finally:
            devices.subprocess.run = old_run

        def slow_dial(bus, dest, path, method, *args, timeout=10.0):  # Dial times out but the phone is calling
            if method.endswith("Dial"): raise RuntimeError("Timeout was reached")
            if method.endswith("GetCalls"): return "([(objectpath '/x/voicecall01', {'State': <'alerting'>})],)"
            return fake(bus, dest, path, method, *args)
        devices.gdbus = slow_dial
        ph.call()                                                     # no exception: counted as a call
        devices.gdbus = lambda *x, **k: ("([],)" if x[3].endswith("GetCalls") else slow_dial(*x, **k))
        try:
            ph.call()
            raise AssertionError("a failed dial with no call must raise")
        except RuntimeError as e:
            assert "dial failed" in str(e)
    finally:
        devices.gdbus = old


def test_tapo_bulb_async_plumbing():
    import types
    calls = []

    class Light:
        brightness = 40
        def has_feature(self, f): return f == "hsv"
        async def set_brightness(self, b): calls.append(("brightness", b))
        async def set_hsv(self, h, s, v): calls.append(("hsv", h, s, v))

    class Dev:
        modules = {"Light": Light()}
        async def update(self): pass
        async def turn_on(self): calls.append("on")
        async def turn_off(self): calls.append("off")

    class Discover:
        @staticmethod
        async def discover_single(host, username=None, password=None):
            calls.append(("connect", host, username)); return Dev()
    fake = types.SimpleNamespace(Discover=Discover, Module=types.SimpleNamespace(Light="Light"))
    old = sys.modules.get("kasa")
    sys.modules["kasa"] = fake
    try:
        b = devices.TapoBulb({"host": "192.168.1.50", "username": "me@example.com", "password": "x"})
        b.on(); b.brightness(60); assert b.can_color(); b.color("blue"); b.off()
        assert calls[0] == ("connect", "192.168.1.50", "me@example.com")
        assert ("brightness", 60) in calls and ("hsv", 240, 100, 40) in calls and calls[-1] == "off"
        assert sum(isinstance(c, tuple) and c[0] == "connect" for c in calls) == 1   # one connection, reused
    finally:
        if old is None: sys.modules.pop("kasa")
        else: sys.modules["kasa"] = old


def test_config_template_private():
    import os, stat
    with tempfile.TemporaryDirectory() as d:
        p = devices.init_config(Path(d) / "hd/config.json")
        assert stat.S_IMODE(os.stat(p).st_mode) == 0o600 and stat.S_IMODE(os.stat(p.parent).st_mode) == 0o700
        cfg = devices.load_config(p)
        assert devices.Devices(cfg).summary().count("not set up") == 3


def test_open_meteo_live():
    """Real request (needs internet): the reply is built from Open-Meteo's live answer."""
    r = devices.Weather({"latitude": 14.65, "longitude": 121.05, "name": "Quezon City"}).report()
    assert r.startswith("In Quezon City it's ") and "high is" in r
    print("    ", r)


def test_dashboard_snapshot_and_http():
    import urllib.request
    import dashboard
    a = FakeAudio()
    h, _ = home(audio=a, devices=FakeDevices(bulb=FakeBulb(), phone=FakePhone()), timer_scale=1000)
    h.handle("COLOR_BLUE", 0.97); h.handle("TEMPERATURE_22"); h.handle("TIMER_30S"); h.handle("CREATE_REMINDER_STUDY")
    s = h.snapshot()
    assert s["lights"]["color"] == "blue" and s["lights"]["real"] and s["thermostat"] == 22
    assert s["timers"][0]["name"] == "30 second" and s["timers"][0]["seconds_left"] > 20000
    assert s["reminders"] == ["study"] and s["events"][-1]["command"] == "COLOR_BLUE" and s["events"][-1]["confidence"] == 0.97
    assert s["phone"]["contact"] == "Mom" and "number" not in json.dumps(s["phone"])
    srv = dashboard.serve(h, port=0, host="127.0.0.1")
    port = srv.server_address[1]
    page = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5).read().decode()
    state = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/state", timeout=5).read())
    assert "Hey Delta Home" in page and state["thermostat"] == 22 and json.loads(json.dumps(state)) == state
    srv.shutdown(); h.close()


def test_dashboard_address_instructions():
    import dashboard
    real = dashboard.lan_ips
    try:
        dashboard.lan_ips = lambda: ["192.168.1.23"]
        assert dashboard.addresses(8080)[0] == "http://192.168.1.23:8080"          # number-style address first
        txt = dashboard.instructions(8080)
        assert "http://192.168.1.23:8080" in txt and "Local Network" in txt and "Always use secure connections" in txt
        dashboard.lan_ips = lambda: []
        assert "not on a network yet" in dashboard.instructions(8080)
    finally:
        dashboard.lan_ips = real
    assert all(ip.count(".") == 3 and not ip.startswith("127.") for ip in real())


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and (k not in ("test_tts_synthesises", "test_open_meteo_live") or "--tts" in sys.argv and k == "test_tts_synthesises" or "--net" in sys.argv and k == "test_open_meteo_live")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
