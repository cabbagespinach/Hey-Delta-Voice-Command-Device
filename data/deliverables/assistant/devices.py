"""
Real devices behind the commands (owner, 2026-10-02):

  weather   Open-Meteo forecast API (no key), for the location in the config
  lights    TP-Link Tapo bulb over Wi-Fi (python-kasa 0.10, TP-Link account login in the config)
  calls     the paired phone over Bluetooth HFP, through oFono (D-Bus: VoiceCallManager.Dial)
  texts     the paired phone over Bluetooth MAP, through BlueZ obexd (D-Bus: MessageAccess1.PushMessage)

Personal settings live in a private config file, ~/.heydelta/config.json (owner-only permissions), written as a
template by `assistant.py --init-config`. Anything not filled in stays simulated, and the reply says it is not set up.
D-Bus is reached with the `gdbus` command (libglib2.0-bin), so no extra Python packages are needed for the phone.
"""
from __future__ import annotations

from pathlib import Path
import asyncio, json, os, re, subprocess, sys, tempfile, threading, time, urllib.parse, urllib.request

CONFIG = Path.home() / ".heydelta/config.json"
TEMPLATE = {
    "_help": "Private settings for the Hey Delta assistant. Keep this file owner-only (chmod 600).",
    "location": {"name": "", "latitude": None, "longitude": None},
    "tapo": {"host": "", "username": "", "password": ""},
    "contact": {"name": "", "number": ""},
    "message": "I'm on my way.",
    "phone": {"bt_address": ""},
}


def init_config(path: Path = CONFIG) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    if not path.exists():
        path.write_text(json.dumps(TEMPLATE, indent=2) + "\n")
    os.chmod(path, 0o600)
    return path


def load_config(path: Path = CONFIG) -> dict:
    if not path.exists():
        return {}
    if path.stat().st_mode & 0o077:
        print(f"(warning: {path} is readable by other users; run: chmod 600 {path})")
    return json.loads(path.read_text())


# ---------------------------------------------------------------- weather
WMO = {0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 45: "foggy", 48: "foggy",
       51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
       61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
       71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains", 80: "light showers", 81: "showers",
       82: "heavy showers", 85: "snow showers", 86: "snow showers", 95: "a thunderstorm",
       96: "a thunderstorm with hail", 99: "a thunderstorm with hail"}


class Weather:
    URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, loc: dict):
        self.lat, self.lon, self.name = loc["latitude"], loc["longitude"], loc.get("name") or "your area"

    def report(self, timeout: float = 6.0) -> str:
        q = urllib.parse.urlencode(dict(latitude=self.lat, longitude=self.lon, timezone="auto", forecast_days=1,
                                        current="temperature_2m,weather_code",
                                        daily="temperature_2m_max,temperature_2m_min,precipitation_probability_max"))
        with urllib.request.urlopen(f"{self.URL}?{q}", timeout=timeout) as r:
            d = json.load(r)
        c, day = d["current"], d["daily"]
        now = WMO.get(int(c["weather_code"]), "unsettled")
        rain = day["precipitation_probability_max"][0]
        s = (f"In {self.name} it's {round(c['temperature_2m'])} degrees and {now}. Today's high is "
             f"{round(day['temperature_2m_max'][0])}, the low {round(day['temperature_2m_min'][0])}")
        return s + (f", with a {rain} percent chance of rain." if rain is not None else ".")


# ---------------------------------------------------------------- Tapo bulb
COLORS_HSV = {"red": (0, 100), "green": (120, 100), "blue": (240, 100)}


class TapoBulb:
    """python-kasa runs on asyncio: one event loop in a background thread; each call waits at most `timeout` s."""

    def __init__(self, cfg: dict, timeout: float = 6.0):
        self.cfg, self.timeout, self.dev = cfg, timeout, None
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(self.timeout)

    async def _device(self):
        if self.dev is None:
            from kasa import Discover
            self.dev = await Discover.discover_single(self.cfg["host"], username=self.cfg["username"],
                                                      password=self.cfg["password"])
        await self.dev.update()
        return self.dev

    async def _light(self):
        from kasa import Module
        return (await self._device()).modules[Module.Light]

    def on(self):
        self._run(self._on())

    async def _on(self):
        await (await self._device()).turn_on()

    def off(self):
        self._run(self._off())

    async def _off(self):
        await (await self._device()).turn_off()

    def brightness(self, percent: int):
        self._run(self._brightness(percent))

    async def _brightness(self, percent):
        light = await self._light()
        await self.dev.turn_on()
        await light.set_brightness(percent)

    def can_color(self) -> bool:
        return self._run(self._can_color())

    async def _can_color(self):
        return (await self._light()).has_feature("hsv")

    def color(self, name: str):
        self._run(self._color(name))

    async def _color(self, name):
        light = await self._light()
        h, s = COLORS_HSV[name]
        await self.dev.turn_on()
        await light.set_hsv(h, s, light.brightness or 100)


# ---------------------------------------------------------------- phone (oFono HFP calls, obexd MAP texts)
# the system's gdbus: a conda environment's copy (glib from conda-forge) can look for the system bus in the wrong place
GDBUS = "/usr/bin/gdbus" if os.path.exists("/usr/bin/gdbus") else "gdbus"


def gdbus(bus: str, dest: str, path: str, method: str, *args: str, timeout: float = 10.0) -> str:
    cmd = [GDBUS, "call", f"--{bus}", "--dest", dest, "--object-path", path, "--method", method, *args]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or f"gdbus failed: {method}")
    return r.stdout


def bmessage(number: str, text: str) -> str:
    """bMessage 1.0 (Bluetooth MAP) for one outgoing SMS, in the format of the owner's working send_sms.py.
    LENGTH counts BEGIN:MSG ... END:MSG with CRLFs, in bytes."""
    body = f"BEGIN:MSG\r\n{text}\r\nEND:MSG\r\n"
    return ("BEGIN:BMSG\r\nVERSION:1.0\r\nSTATUS:UNREAD\r\nTYPE:SMS_GSM\r\nFOLDER:telecom/msg/outbox\r\n"
            f"BEGIN:BENV\r\nBEGIN:VCARD\r\nVERSION:2.1\r\nN:\r\nTEL:{number}\r\nEND:VCARD\r\n"
            f"BEGIN:BBODY\r\nCHARSET:UTF-8\r\nLENGTH:{len(body.encode())}\r\n{body}END:BBODY\r\nEND:BENV\r\nEND:BMSG\r\n")


def map_python() -> str:
    """A Python with dbus-python for map_send.py: this one if it has it, else the Pi's own (python3-dbus)."""
    try:
        import dbus  # noqa: F401
        return sys.executable
    except ImportError:
        return "/usr/bin/python3"


def session_bus_env() -> dict:
    """Environment with the desktop session's D-Bus (obexd lives there), also when started over SSH."""
    env = dict(os.environ)
    bus = f"/run/user/{os.getuid()}/bus"
    if not env.get("DBUS_SESSION_BUS_ADDRESS") and os.path.exists(bus):
        env["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path={bus}"
    return env


class Phone:
    def __init__(self, cfg: dict):
        self.addr = (cfg.get("phone", {}) or {}).get("bt_address", "").upper()
        self.contact, self.message = cfg["contact"], cfg.get("message") or "I'm on my way."

    def _modems(self) -> list[tuple[str, str]]:
        try:
            out = gdbus("system", "org.ofono", "/", "org.ofono.Manager.GetModems")
        except RuntimeError as e:
            if "ServiceUnknown" in str(e) or "was not provided" in str(e):
                raise RuntimeError("oFono is not running (sudo apt install ofono; see README_ASSISTANT.md, Phone)") from None
            raise
        want = "dev_" + self.addr.replace(":", "_") if self.addr else ""
        ms = [(re.match(r"'([^']+)'", m).group(1), m) for m in re.split(r"\(objectpath ", out)[1:]]
        return [(path, m) for path, m in ms if not want or want in path]

    def _modem(self) -> str:
        ms = self._modems()
        for path, m in ms:
            if "org.ofono.VoiceCallManager" in m and "'Online': <true>" in m:
                return path
        if not ms:
            raise RuntimeError("oFono sees no phone: reconnect it (bluetoothctl connect ADDRESS) after oFono started, "
                               "and check that PipeWire hands calls to oFono (README_ASSISTANT.md, Phone)")
        for path, _ in ms:                    # the phone's link exists but is switched off: switch it on
            for prop in ("Powered", "Online"):
                try:
                    gdbus("system", "org.ofono", path, "org.ofono.Modem.SetProperty", prop, "<true>")
                except RuntimeError:
                    pass
        time.sleep(2)
        for path, m in self._modems():
            if "org.ofono.VoiceCallManager" in m and "'Online': <true>" in m:
                return path
        raise RuntimeError("the phone is known to oFono but would not come online (is it connected?)")

    def _in_call(self, modem: str) -> bool:
        try:
            out = gdbus("system", "org.ofono", modem, "org.ofono.VoiceCallManager.GetCalls")
        except Exception:
            return False
        return any(f"'State': <'{s}'>" in out for s in ("dialing", "alerting", "active"))

    def call(self):
        modem = self._modem()
        try:                                  # the phone can take a while to confirm the dial
            gdbus("system", "org.ofono", modem, "org.ofono.VoiceCallManager.Dial", self.contact["number"], "",
                  timeout=30.0)
        except Exception as e:                # error or no answer in time, but the phone may be calling anyway
            time.sleep(1)
            if not self._in_call(modem):
                raise RuntimeError(f"dial failed: {e}") from None
            print(f"(call: dial reported '{e}', but the phone is calling)")

    def text(self, timeout: float = 45.0):
        """Send the message over MAP in one helper process (map_send.py): obexd drops a MAP session as soon as
        the process that opened it exits, so step-by-step `gdbus call`s cannot work."""
        if not self.addr:
            raise RuntimeError("phone.bt_address not set in the config")
        with tempfile.NamedTemporaryFile("w", suffix=".bmsg", delete=False, newline="") as f:
            f.write(bmessage(self.contact["number"], self.message))
        os.chmod(f.name, 0o644)
        try:
            r = subprocess.run([map_python(), str(Path(__file__).with_name("map_send.py")), self.addr, f.name],
                               capture_output=True, text=True, timeout=timeout, env=session_bus_env())
        except subprocess.TimeoutExpired:
            raise RuntimeError("sending timed out") from None
        finally:
            os.unlink(f.name)
        if r.returncode:
            err = (r.stderr.strip().splitlines() or ["unknown error"])[-1]
            if "No module named 'dbus'" in err:
                err = "dbus-python missing: sudo apt install python3-dbus"
            raise RuntimeError(err)


class Devices:
    """Whichever devices the config sets up; the others are None (their commands stay simulated)."""

    def __init__(self, cfg: dict):
        loc, tapo, contact = cfg.get("location") or {}, cfg.get("tapo") or {}, cfg.get("contact") or {}
        self.weather = Weather(loc) if loc.get("latitude") is not None and loc.get("longitude") is not None else None
        self.bulb = TapoBulb(tapo) if all(tapo.get(k) for k in ("host", "username", "password")) else None
        self.phone = Phone(cfg) if contact.get("number") else None
        self.contact_name = contact.get("name") or "your contact"
        self.message = cfg.get("message") or "I'm on my way."

    def summary(self) -> str:
        on = lambda x: "set up" if x else "not set up"
        return f"[devices] weather {on(self.weather)} | Tapo bulb {on(self.bulb)} | phone {on(self.phone)}"
