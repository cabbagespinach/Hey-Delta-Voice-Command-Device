"""
What each recognised command does: a simulated smart home (owner decision 2026-10-02).

Real behaviour on the Pi itself: timers and alarms count down and ring, reminders are kept and read back, music plays
from ~/Music/Party Music, volume changes what you hear, the clock is read. With devices set up (devices.py, owner
2026-10-02): the weather comes from Open-Meteo, lights / brightness / colour drive the Tapo bulb, calls and texts go
through the paired phone (oFono HFP, obexd MAP) to the one contact in the private config. A device that is not set up
or does not answer is said so in the reply. The thermostat stays simulated (no device).

Home.handle(command) -> the sentence to say back. All side effects go through the injected `audio` (music / rings)
and `say` callbacks, so the logic is tested without a speaker (test_assistant.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import datetime as dt
import collections, json, threading, time

TIMERS = {"TIMER_10S": (10, "10 second"), "TIMER_30S": (30, "30 second"), "TIMER_1MIN": (60, "1 minute")}
ALARMS = {"ALARM_6AM": (6, 0), "ALARM_8AM": (8, 0), "ALARM_9PM": (21, 0)}
TEMPERATURES = {"TEMPERATURE_18": 18, "TEMPERATURE_22": 22, "TEMPERATURE_26": 26}
BRIGHTNESS = {"BRIGHTNESS_20": 20, "BRIGHTNESS_60": 60, "BRIGHTNESS_100": 100}
COLORS = {"COLOR_RED": "red", "COLOR_BLUE": "blue", "COLOR_GREEN": "green"}
REMINDERS = {"CREATE_REMINDER_DRINK_WATER": "drink water", "CREATE_REMINDER_STUDY": "study",
             "CREATE_REMINDER_EXERCISE": "exercise"}
VOLUME_STEP = 10


@dataclass
class State:
    """Everything the simulated home remembers between runs (saved as JSON)."""
    lights_on: bool = False
    brightness: int = 100
    color: str = "white"
    temperature: int = 24
    volume: int = 60                                   # percent, applies to music, replies and rings
    reminders: list = field(default_factory=list)
    alarms: list = field(default_factory=list)         # ISO datetimes of pending alarms
    last_call: str = ""
    last_message: str = ""


def spoken_time(t: dt.datetime) -> str:
    h = t.hour % 12 or 12
    return f"{h}:{t.minute:02d} {'AM' if t.hour < 12 else 'PM'}" if t.minute else f"{h} {'AM' if t.hour < 12 else 'PM'}"


def next_occurrence(now: dt.datetime, hour: int, minute: int) -> dt.datetime:
    t = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return t if t > now else t + dt.timedelta(days=1)


class Home:
    def __init__(self, audio, say, state_path: Path | None = None, weather_path: Path | None = None,
                 clock=dt.datetime.now, timer_scale: float = 1.0, devices=None):
        """audio: object with play_music(), pause_music(), resume_music(), stop_music(), next_track(), is_playing(),
        has_music(), set_volume(0..1), ring(). say(text): speak (non-blocking). timer_scale shortens timers in tests."""
        self.audio, self.say, self.clock, self.timer_scale = audio, say, clock, timer_scale
        self.state_path, self.weather_path, self.devices = state_path, weather_path, devices
        self.state = State()
        if state_path and state_path.exists():
            self.state = State(**{**asdict(State()), **json.loads(state_path.read_text())})
        self.audio.set_volume(self.state.volume / 100)
        self._timers: list[threading.Timer] = []
        self._background: list[threading.Thread] = []    # texts being sent
        self._timer_info: list[dict] = []                # name + due time (epoch s), for the dashboard
        self.events = collections.deque(maxlen=12)        # last commands: time, command, confidence, reply
        self.listening = False                            # set by the assistant between wake-up and decision
        self.last_weather = ""
        self._lock = threading.Lock()
        self._closing = threading.Event()
        self._alarm_thread = threading.Thread(target=self._alarm_loop, daemon=True)
        self._alarm_thread.start()

    # ---------------------------------------------------------------- dispatch
    def handle(self, command: str, confidence: float | None = None) -> str:
        reply = self._handle(command)
        self.events.appendleft(dict(t=time.time(), command=command, confidence=confidence, reply=reply))
        return reply

    def _handle(self, command: str) -> str:
        c = command
        if c in TIMERS:
            return self._timer(*TIMERS[c])
        if c in ALARMS:
            return self._alarm(*ALARMS[c])
        if c in TEMPERATURES:
            self.state.temperature = TEMPERATURES[c]
            return self._done(f"Setting the temperature to {self.state.temperature} degrees.")
        if c in BRIGHTNESS:
            b = BRIGHTNESS[c]
            return self._bulb(lambda bulb: bulb.brightness(b), dict(brightness=b, lights_on=True),
                              f"Brightness set to {b} percent.")
        if c in COLORS:
            col = COLORS[c]
            bulb = self.devices.bulb if self.devices else None
            try:
                if bulb is not None and not bulb.can_color():
                    return "This light can't change color."
            except Exception:
                return "I couldn't reach the light."
            return self._bulb(lambda bulb: bulb.color(col), dict(color=col, lights_on=True), f"Changing the lights to {col}.")
        if c in REMINDERS:
            task = REMINDERS[c]
            if task not in self.state.reminders:
                self.state.reminders.append(task)
            return self._done(f"Okay, I'll remind you to {task}.")
        fn = getattr(self, "_" + c.lower(), None)
        if fn is None or c == "unknown":
            return "Sorry, I didn't catch that."
        return fn()

    # ---------------------------------------------------------------- fixed commands
    def _play_music(self):
        if not self.audio.has_music():
            return "I don't have any music yet. Put some songs in the music folder."
        if self.audio.play_music() is False:           # every song failed to load (reason printed by audio_out)
            return "Sorry, I couldn't play the songs in the music folder."
        return "Playing music."

    def _pause(self):
        if not self.audio.is_playing():
            return "Nothing is playing."
        self.audio.pause_music()
        return "Paused."

    def _stop(self):
        timers = [t for t in self._timers if t.is_alive()]
        if self.audio.is_playing() or self.audio.is_paused():
            self.audio.stop_music()
            return "Stopped."
        if timers:                                      # "stop" while nothing plays: cancel running timers
            for t in timers:
                t.cancel()
            return "Timer cancelled."
        return "Nothing is playing."

    def _next(self):
        if not self.audio.has_music():
            return "I don't have any music yet."
        if self.audio.next_track() is False:
            return "Sorry, I couldn't play the songs in the music folder."
        return "Next song."

    def _volume_up(self):
        return self._volume(+VOLUME_STEP)

    def _volume_down(self):
        return self._volume(-VOLUME_STEP)

    def _volume(self, step):
        self.state.volume = max(0, min(100, self.state.volume + step))
        self.audio.set_volume(self.state.volume / 100)
        edge = " That's the maximum." if self.state.volume == 100 and step > 0 else \
            " That's the minimum." if self.state.volume == 0 and step < 0 else ""
        return self._done(f"Volume {self.state.volume} percent.{edge}")

    def _weather(self):
        if self.devices and self.devices.weather:
            try:
                self.last_weather = self.devices.weather.report()
                return self.last_weather
            except Exception as e:
                print(f"(weather: {e})")
                return "I couldn't reach the weather service right now."
        if self.weather_path and self.weather_path.exists():
            w = json.loads(self.weather_path.read_text())
            return w.get("say") or f"Today: {w.get('summary', 'no forecast')}, around {w.get('temperature', '?')} degrees."
        return "I'm offline, so I can't check the weather right now."

    def _time(self):
        return f"It's {spoken_time(self.clock())}."

    def _light_on(self):
        return self._bulb(lambda bulb: bulb.on(), dict(lights_on=True), "Lights on.")

    def _light_off(self):
        return self._bulb(lambda bulb: bulb.off(), dict(lights_on=False), "Lights off.")

    def _bulb(self, action, changes: dict, reply: str) -> str:
        """Drive the Tapo bulb if one is set up (simulated otherwise); the state changes only if the bulb obeyed."""
        bulb = self.devices.bulb if self.devices else None
        if bulb is not None:
            try:
                action(bulb)
            except Exception as e:
                print(f"(light: {e!r})")
                return "I couldn't reach the light."
        else:                                          # no bulb in the config: say so, it did not really happen
            reply = reply.rstrip(".") + " (simulated)."
        for k, v in changes.items():
            setattr(self.state, k, v)
        return self._done(reply)

    def _call(self):
        phone = self.devices.phone if self.devices else None
        if phone is None:
            return "Calling isn't set up yet."
        try:
            phone.call()
        except Exception as e:
            print(f"(call: {e})")
            return "I couldn't start the call. Is the phone connected?"
        self.state.last_call = self.clock().isoformat(timespec="seconds")
        return self._done(f"Calling {self.devices.contact_name}.")

    def _message(self):
        phone = self.devices.phone if self.devices else None
        if phone is None:
            return "Texting isn't set up yet."
        name, msg = self.devices.contact_name, self.devices.message

        def send():                                     # MAP can take seconds: send in the background
            try:
                phone.text()
                self.state.last_message = self.clock().isoformat(timespec="seconds")
                self.save()
            except Exception as e:
                print(f"(message: {e})")
                self.say(f"The message to {name} didn't go through.")
        th = threading.Thread(target=send, daemon=True)
        th.start()
        self._background.append(th)
        return f"Sending {msg} to {name}."

    def wait_background(self, timeout: float = 60.0):
        """Wait for background jobs (texts being sent); used by one-shot runs before they exit."""
        for th in self._background:
            th.join(timeout)

    def _list_reminders(self):
        r = self.state.reminders
        alarms = sorted(dt.datetime.fromisoformat(a) for a in self.state.alarms)
        parts = []
        if r:
            parts.append("You have reminders to " + (r[0] if len(r) == 1 else ", ".join(r[:-1]) + " and " + r[-1]) + ".")
        if alarms:
            parts.append("Alarms at " + ", ".join(spoken_time(a) for a in alarms) + ".")
        return " ".join(parts) if parts else "You have no reminders."

    # ---------------------------------------------------------------- timers and alarms
    def _timer(self, seconds, name):
        t = threading.Timer(seconds * self.timer_scale, self._ring, args=(f"Your {name} timer is done.",))
        t.daemon = True
        t.start()
        self._timers.append(t)
        self._timer_info.append(dict(name=name, due=time.time() + seconds * self.timer_scale, timer=t))
        return f"Timer set for {name.replace(' second', ' seconds').replace('1 minute', 'one minute')}."

    def _alarm(self, hour, minute):
        when = next_occurrence(self.clock(), hour, minute)
        iso = when.isoformat(timespec="minutes")
        with self._lock:
            if iso not in self.state.alarms:
                self.state.alarms.append(iso)
        day = "today" if when.date() == self.clock().date() else "tomorrow"
        return self._done(f"Alarm set for {spoken_time(when)} {day}.")

    def _alarm_loop(self):
        while not self._closing.wait(1.0):
            now = self.clock()
            with self._lock:
                due = [a for a in self.state.alarms if dt.datetime.fromisoformat(a) <= now]
                if not due:
                    continue
                self.state.alarms = [a for a in self.state.alarms if a not in due]
            self.save()
            for a in due:
                self._ring(f"It's {spoken_time(dt.datetime.fromisoformat(a))}. This is your alarm.")

    def _ring(self, text):
        self.audio.ring()
        self.say(text)

    # ---------------------------------------------------------------- state
    def _done(self, reply):
        self.save()
        return reply

    def save(self):
        if self.state_path:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(asdict(self.state), indent=2) + "\n")
            tmp.replace(self.state_path)

    def snapshot(self) -> dict:
        """Everything the laptop dashboard shows (dashboard.py), JSON-ready."""
        s, now = self.state, time.time()
        timers = [dict(name=i["name"], seconds_left=max(0.0, round(i["due"] - now, 1)))
                  for i in self._timer_info if i["timer"].is_alive()]
        d = self.devices
        return dict(
            lights=dict(on=s.lights_on, brightness=s.brightness, color=s.color, real=bool(d and d.bulb)),
            thermostat=s.temperature, volume=s.volume,
            music=dict(state="playing" if self.audio.is_playing() else "paused" if self.audio.is_paused() else "stopped",
                       track=getattr(self.audio, "track", "") or ""),
            timers=timers, alarms=sorted(s.alarms), reminders=list(s.reminders),
            phone=dict(contact=d.contact_name if d and d.phone else "", last_call=s.last_call, last_message=s.last_message),
            weather=self.last_weather, listening=self.listening, clock=spoken_time(self.clock()),
            devices=dict(weather=bool(d and d.weather), bulb=bool(d and d.bulb), phone=bool(d and d.phone)),
            events=[dict(e) for e in self.events])

    def status(self) -> str:
        s = self.state
        light = f"lights {'ON' if s.lights_on else 'off'} ({s.brightness}%, {s.color})"
        music = "playing" if self.audio.is_playing() else "paused" if self.audio.is_paused() else "stopped"
        timers = sum(t.is_alive() for t in self._timers)
        return (f"[home] {light} | thermostat {s.temperature}°C | volume {s.volume}% | music {music} | "
                f"timers {timers} | alarms {len(s.alarms)} | reminders {len(s.reminders)}")

    def close(self):
        self._closing.set()
        for t in self._timers:
            t.cancel()
        self.save()
