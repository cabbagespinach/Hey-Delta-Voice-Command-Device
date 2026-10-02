#!/usr/bin/env python3
"""
HeyDeltaListener: always-on "Hey Delta" detection, then capture of the spoken command from the SAME microphone
stream, handed to your code as audio. Needs: numpy, onnxruntime, sounddevice (+ scipy if the mic can't do 16 kHz).

    from heydelta_listener import HeyDeltaListener

    def on_command(cap):                       # cap: Capture (see below)
        if cap.audio is None:
            print("woke up, but nothing was said")
        else:
            label = my_command_classifier(cap.audio)       # 16 kHz mono float32, -1..1

    HeyDeltaListener(on_command=on_command).run()          # blocks; Ctrl+C or .stop() ends it

How it works
  LISTENING   every 100 ms the last 1.5 s of audio goes through the wakeword model (BC-ResNet-6, front end included).
              It fires when at least k of the last n scores reach the threshold (deploy_config.json: 2 of 3, 0.7335).
  CAPTURING   from the moment it fires, audio keeps flowing into a command buffer, starting `pre_roll_sec` earlier so a
              command said without a pause isn't clipped. Without a VAD, "speech" = a 20 ms frame at least
              `speech_margin_db` above the room's background level (re-estimated all the time while listening); once the
              command has started, `continue_margin_db` above it is enough, so quieter parts don't count as silence.
              Levels are averaged over `smooth_sec` (100 ms), so single room clicks don't keep a capture open.
              Sound in the first `ignore_after_wake_sec` is ignored (the tail of "Delta"), and a command only counts as
              started after `min_speech_sec` of speech frames, so a click or breath can't start it.
              Capture ends when, after speech has started, there has been `end_silence_sec` of non-speech
              (reason "end_of_speech"), or `max_command_sec` after speech started ("max_length"). If no speech starts within
              `max_wait_sec`, the capture ends there: with `always_capture` (default) the audio so far is still handed
              over (reason "no_speech_detected", ~pre_roll + max_wait seconds); otherwise audio=None ("no_speech").
  COOLDOWN    the wakeword ignores the next `cooldown_sec`, so the tail of a command can't re-trigger it; then LISTENING.
  CHIMES      (`chimes=True`, default) a short rising tone when it wakes up and a falling tone when the command has been
              captured, played through the default output without blocking. Route that output to your PulseAudio
              echo-cancel sink (default sink, or PULSE_SINK=...) so AEC removes the chime from the microphone signal.
              While the start chime plays, the capture ignores the microphone (see ignore_after_wake_sec).
Replace the loudness-based end-of-speech rule with a real VAD later: only `_is_speech` changes.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import json, queue, time

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
SR, WIN, HOP, FRAME = 16000, 24000, 1600, 320          # 16 kHz; 1.5 s window; 100 ms hop; 20 ms frame


@dataclass
class Capture:
    audio: np.ndarray | None           # command audio, 16 kHz mono float32 (None if nothing was said)
    reason: str                        # "end_of_speech" | "max_length" | "no_speech_detected" | "no_speech"
    wake_prob: float                   # wakeword probability at the moment it fired
    wake_time: float                   # seconds since the listener started
    wake_window: np.ndarray = field(repr=False, default=None)   # the 1.5 s window that fired (e.g. to keep as data)

    @property
    def duration_sec(self):
        return 0.0 if self.audio is None else len(self.audio) / SR


@dataclass
class Settings:
    pre_roll_sec: float = 0.3                # captured audio starts this long before the wake-up fired
    max_wait_sec: float = 3.0
    end_silence_sec: float = 1.0             # this much non-speech after speech ends the capture (mid-sentence pauses)
    max_command_sec: float = 6.0             # counted from when speech starts (the wait before it is max_wait_sec)
    speech_margin_db: float = 6.0            # dB above background to START a command (calibrated on owner Pi recordings)
    continue_margin_db: float = 4.0          # dB above background that still counts as speech once it has started
    smooth_sec: float = 0.1                  # speech is judged on loudness averaged over this window, so brief room
                                             # clicks don't count as speech and brief dips inside words don't count as silence
    ignore_after_wake_sec: float = 0.25      # the tail of "Delta" is still sounding when the model fires
    min_speech_sec: float = 0.15             # this much speech (not necessarily continuous) before a command counts
    cooldown_sec: float = 1.0
    chimes: bool = True                      # rising tone on wake-up, falling tone when the capture ends
    chime_volume: float = 0.25               # 0..1 (full scale); keep it soft
    chime_output_device: int | str | None = None   # None = default output (should be the echo-cancel sink)
    always_capture: bool = True              # keep the audio even when no speech was detected (reason
                                             # "no_speech_detected"); False = hand over audio=None ("no_speech")


CHIME_SR = 48000


def make_chime(rising: bool, volume: float) -> np.ndarray:
    """Two soft 90 ms notes (E5 -> A5 rising, A5 -> E5 falling) with 10 ms fades; ~0.19 s at 48 kHz."""
    notes = (659.25, 880.0) if rising else (880.0, 659.25)
    n = int(0.09 * CHIME_SR)
    t = np.arange(n) / CHIME_SR
    fade = np.minimum(1, np.minimum(t, t[::-1]) / 0.01)
    gap = np.zeros(int(0.01 * CHIME_SR))
    x = np.concatenate([np.sin(2 * np.pi * notes[0] * t) * fade, gap, np.sin(2 * np.pi * notes[1] * t) * fade])
    return (volume * x).astype(np.float32)


class HeyDeltaListener:
    def __init__(self, on_command=None, on_wake=None, device=None, settings: Settings | None = None,
                 config_path=HERE / "deploy_config.json", verbose=True):
        cfg = json.loads(Path(config_path).read_text())
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1                     # one core; the rest stays free for the other models
        so.inter_op_num_threads = 1
        self.model = ort.InferenceSession(str(Path(config_path).parent / cfg["model_file"]), so,
                                          providers=["CPUExecutionProvider"])
        self.thr, (self.k, self.n) = cfg["threshold"], cfg["k_of_n"]
        self.on_command, self.on_wake = on_command, on_wake
        self.device, self.s, self.verbose = device, settings or Settings(), verbose
        self._q, self._running = queue.Queue(), False
        self._chime_up, self._chime_down = make_chime(True, self.s.chime_volume), make_chime(False, self.s.chime_volume)
        # the capture ignores the microphone while the start chime plays (+0.1 s for output latency)
        self._ignore_sec = max(self.s.ignore_after_wake_sec,
                               len(self._chime_up) / CHIME_SR + 0.1 if self.s.chimes else 0.0)
        self._reset()

    def _play(self, x):
        if not self.s.chimes:
            return
        try:
            import sounddevice as sd
            sd.play(x, CHIME_SR, device=self.s.chime_output_device, blocking=False)
        except Exception as e:                           # no speaker / no audio library: carry on silently
            if self.verbose and not getattr(self, "_chime_warned", False):
                print(f"(chime not played: {e})")
                self._chime_warned = True

    # ---- state ---------------------------------------------------------------
    def _reset(self):
        self.t = 0.0                                          # seconds of audio processed
        self.window = np.zeros(WIN, np.float32)               # last 1.5 s
        self.filled = 0
        self.recent = deque(maxlen=self.n)
        self.state, self.cooldown_until = "LISTENING", 0.0
        self.floor_db = deque(maxlen=int(10 * SR / FRAME))    # last 10 s of frame levels while listening
        self.pending = np.zeros(0, np.float32)

    def _frame_db(self, x):
        n = len(x) // FRAME
        if n == 0:
            return np.array([])
        f = x[:n * FRAME].reshape(n, FRAME)
        return 20 * np.log10(np.sqrt((f ** 2).mean(1)) + 1e-9)

    def _background_db(self):
        return float(np.percentile(self.floor_db, 20)) if len(self.floor_db) >= 25 else -60.0

    def _is_speech(self, frame_db, started=False):
        """Two thresholds (hysteresis): louder to start a command, quieter to keep it going, so a voice that drops,
        soft sounds ("s", "f") and trailing words don't count as silence."""
        margin = self.s.continue_margin_db if started else self.s.speech_margin_db
        return frame_db >= self._background_db() + margin

    # ---- per 100 ms ----------------------------------------------------------
    def _hop(self, hop):
        self.t += HOP / SR
        self.window = np.concatenate([self.window[HOP:], hop])
        self.filled = min(WIN, self.filled + HOP)
        levels = self._frame_db(hop)
        if self.state == "LISTENING":
            self.floor_db.extend(levels)
            if self.filled < WIN or self.t < self.cooldown_until:
                return
            p = float(self.model.run(["prob"], {"audio": self.window[None, :]})[0][0])
            self.recent.append(p >= self.thr)
            if p >= self.thr and sum(self.recent) >= self.k:
                self._start_capture(p)
        else:
            self._capture_hop(hop, levels)

    def _start_capture(self, p):
        pre = int(self.s.pre_roll_sec * SR)
        self.cap = dict(wake_prob=p, wake_time=round(self.t, 2), wake_window=self.window.copy(),
                        buf=[self.window[-pre:].copy()] if pre else [], started=None, last_speech=None, t0=self.t,
                        speech_frames=0, recent_power=deque(maxlen=max(1, int(round(self.s.smooth_sec * SR / FRAME)))))
        self.state = "CAPTURING"
        self._play(self._chime_up)
        if self.verbose:
            print(f"[{self.t:7.1f}s] WAKE (p={p:.3f}) - listening for a command")
        if self.on_wake:
            self.on_wake(p, self.t)

    def _capture_hop(self, hop, levels):
        c = self.cap
        c["buf"].append(hop)
        for i, db_raw in enumerate(levels):
            ft = self.t - HOP / SR + (i + 1) * FRAME / SR
            c["recent_power"].append(10 ** (db_raw / 10))
            db = 10 * np.log10(np.mean(c["recent_power"]) + 1e-18)        # smoothed level (smooth_sec window)
            if ft - c["t0"] < self._ignore_sec or not self._is_speech(db, started=c["started"] is not None):
                continue
            c["speech_frames"] += 1
            c["last_speech"] = ft
            if c["started"] is None and c["speech_frames"] * FRAME / SR >= self.s.min_speech_sec:
                c["started"] = ft
        el = self.t - c["t0"]
        if c["started"] is None:
            if el >= self.s.max_wait_sec:
                if self.s.always_capture:
                    self._finish(np.concatenate(c["buf"]), "no_speech_detected")
                else:
                    self._finish(None, "no_speech")
        elif self.t - c["last_speech"] >= self.s.end_silence_sec:
            self._finish(np.concatenate(c["buf"]), "end_of_speech")
        elif self.t - c["started"] >= self.s.max_command_sec:
            self._finish(np.concatenate(c["buf"]), "max_length")

    def _finish(self, audio, reason):
        c = self.cap
        cap = Capture(audio=audio, reason=reason, wake_prob=c["wake_prob"], wake_time=c["wake_time"],
                      wake_window=c["wake_window"])
        self.state, self.cooldown_until = "LISTENING", self.t + self.s.cooldown_sec
        self.recent.clear()
        self._play(self._chime_down)
        if self.verbose:
            print(f"[{self.t:7.1f}s] command captured: {cap.duration_sec:.2f} s ({reason})")
        if self.on_command:
            self.on_command(cap)

    # ---- audio input ---------------------------------------------------------
    def feed(self, audio16k: np.ndarray):
        """Push 16 kHz mono float32 audio of any length (use this instead of run() to supply audio yourself)."""
        self.pending = np.concatenate([self.pending, np.asarray(audio16k, np.float32)])
        while len(self.pending) >= HOP:
            hop, self.pending = self.pending[:HOP], self.pending[HOP:]
            self._hop(hop)

    def run(self, minutes: float | None = None):
        """Open the microphone and process audio until stop(), Ctrl+C, or `minutes` of audio."""
        import sounddevice as sd
        rate = SR
        try:
            sd.check_input_settings(device=self.device, samplerate=SR, channels=1, dtype="float32")
        except Exception:
            rate = int(sd.query_devices(self.device, "input")["default_samplerate"])
        resample = None
        if rate != SR:
            from scipy.signal import resample_poly
            g = np.gcd(rate, SR)
            resample = lambda x: resample_poly(x, SR // g, rate // g).astype(np.float32)
        self._running = True
        with sd.InputStream(device=self.device, samplerate=rate, channels=1, dtype="float32",
                            blocksize=int(rate * 0.1), callback=lambda d, *_: self._q.put(d[:, 0].copy())):
            if self.verbose:
                print("listening for 'Hey Delta' ... (Ctrl+C to stop)")
            try:
                while self._running and (minutes is None or self.t < minutes * 60):
                    try:
                        x = self._q.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    self.feed(resample(x) if resample else x)
            except KeyboardInterrupt:
                pass

    def stop(self):
        self._running = False
