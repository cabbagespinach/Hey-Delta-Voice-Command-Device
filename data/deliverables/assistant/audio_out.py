"""
One audio output for the assistant: music, spoken replies and alarm rings are mixed into a single stream, so they
never fight over the speaker. Music is ducked (turned down) while the assistant listens or talks.

Music: every song file in the music folder (shuffled), loaded one song at a time. .wav / .flac / .ogg / .mp3 are
read directly; .m4a / .aac / .opus / .wma need ffmpeg (`sudo apt install ffmpeg`), which is also tried when a file
cannot be read directly. Hidden files (e.g. macOS "._song.mp3" copies) are skipped.
"""
from __future__ import annotations

from pathlib import Path
import random, shutil, subprocess, threading

import numpy as np

OUT_SR = 48000
MUSIC_EXT = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac", ".opus", ".wma"}


def read_song(p: Path) -> np.ndarray:
    """Decode a song to mono float32 at OUT_SR: soundfile first, ffmpeg as the fallback."""
    try:
        import soundfile as sf
        y, sr = sf.read(str(p), dtype="float32")
        return to_rate(y, sr)
    except Exception as e:
        if not shutil.which("ffmpeg"):
            raise RuntimeError(f"{e} (installing ffmpeg may help: sudo apt install ffmpeg)") from None
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(p), "-f", "f32le", "-ac", "1", "-ar", str(OUT_SR), "-"],
                       capture_output=True, timeout=120)
    if r.returncode or not r.stdout:
        raise RuntimeError(r.stderr.decode(errors="replace").strip()[-200:] or "ffmpeg could not decode it")
    return np.frombuffer(r.stdout, np.float32).copy()


def to_rate(y: np.ndarray, sr: int, out_sr: int = OUT_SR) -> np.ndarray:
    y = y.mean(axis=1) if y.ndim > 1 else y
    if sr != out_sr:
        from scipy.signal import resample_poly
        g = np.gcd(int(sr), out_sr)
        y = resample_poly(y, out_sr // g, int(sr) // g)
    return y.astype(np.float32)


def ring_tone(seconds: float = 2.4) -> np.ndarray:
    """Three-beep alarm pattern, repeated (soft attack/release, 880/1320 Hz)."""
    beep = lambda f, d: np.sin(2 * np.pi * f * np.arange(int(d * OUT_SR)) / OUT_SR) * \
        np.minimum(1, np.minimum(np.arange(int(d * OUT_SR)), np.arange(int(d * OUT_SR))[::-1]) / (0.01 * OUT_SR))
    gap = np.zeros(int(0.08 * OUT_SR))
    unit = np.concatenate([beep(880, 0.12), gap, beep(1320, 0.12), gap, beep(880, 0.12), np.zeros(int(0.4 * OUT_SR))])
    reps = max(1, int(seconds / (len(unit) / OUT_SR)))
    return (0.5 * np.tile(unit, reps)).astype(np.float32)


class AudioOut:
    def __init__(self, music_dir: Path | None = None, device=None, enabled: bool = True, seed: int | None = None):
        self.music_dir, self.device, self.enabled = music_dir, device, enabled
        self.volume, self.duck = 0.6, 1.0
        self._lock = threading.Lock()
        self._oneshots: list[list] = []                  # [array, position]
        self._song, self._pos, self._playing, self._paused = None, 0, False, False
        self._playlist, self._rng = [], random.Random(seed)
        self.track = ""                                   # name of the song playing (dashboard)
        self.error = ""                                   # why the last song could not be loaded
        self._stream = None
        if enabled:
            import sounddevice as sd
            self._stream = sd.OutputStream(samplerate=OUT_SR, channels=1, dtype="float32", device=device,
                                           blocksize=1024, callback=self._callback)
            self._stream.start()

    # ---------------------------------------------------------------- mixer
    def _callback(self, out, frames, *_):
        mix = np.zeros(frames, np.float32)
        with self._lock:
            if self._playing and not self._paused and self._song is not None:
                n = min(frames, len(self._song) - self._pos)
                mix[:n] += self._song[self._pos:self._pos + n] * self.duck
                self._pos += n
                if self._pos >= len(self._song):
                    threading.Thread(target=self._advance, daemon=True).start()
                    self._song = None
            for s in self._oneshots:
                n = min(frames, len(s[0]) - s[1])
                mix[:n] += s[0][s[1]:s[1] + n]
                s[1] += n
            self._oneshots = [s for s in self._oneshots if s[1] < len(s[0])]
        out[:, 0] = np.clip(mix * self.volume, -1, 1)

    def play_clip(self, y: np.ndarray, sr: int = OUT_SR, block: bool = False):
        """Mix a one-shot sound (a spoken reply, a ring) over whatever plays."""
        y = to_rate(np.asarray(y, np.float32), sr)
        if not self.enabled:
            return
        item = [y, 0]
        with self._lock:
            self._oneshots.append(item)
        if block:
            import time
            while item[1] < len(item[0]):
                time.sleep(0.05)

    def ring(self):
        self.play_clip(ring_tone())

    # ---------------------------------------------------------------- music
    def _songs(self):
        if not self.music_dir or not Path(self.music_dir).is_dir():
            return []
        root = Path(self.music_dir)
        return sorted(p for p in root.rglob("*") if p.suffix.lower() in MUSIC_EXT and p.is_file()
                      and not any(part.startswith(".") for part in p.relative_to(root).parts))

    def has_music(self) -> bool:
        return bool(self._songs())

    def _load_next(self):
        if not self._playlist:
            self._playlist = self._songs()
            self._rng.shuffle(self._playlist)
        while self._playlist:
            p = self._playlist.pop()
            try:
                y = read_song(p) * 0.5                    # music sits below speech
                self.track, self.error = p.stem, ""
                return y
            except Exception as e:                        # unreadable file: skip it
                self.error = f"{p.name}: {e}"
                print(f"(skipping {self.error})", flush=True)
        return None

    def _advance(self):
        song = self._load_next()
        with self._lock:
            self._song, self._pos = song, 0
            self._playing = song is not None and self._playing

    def play_music(self) -> bool:
        """Start (or resume) the music; False if no song in the folder could be loaded."""
        with self._lock:
            if self._paused and self._song is not None:   # "play music" after "pause" resumes
                self._paused = False
                return True
            self._playing, self._paused = True, False
        self._advance()
        return self._song is not None

    def pause_music(self):
        with self._lock:
            self._paused = True

    def resume_music(self):
        with self._lock:
            self._paused = False

    def stop_music(self):
        with self._lock:
            self._playing, self._paused, self._song, self._pos = False, False, None, 0
            self.track = ""

    def next_track(self) -> bool:
        with self._lock:
            self._playing, self._paused = True, False
        self._advance()
        return self._song is not None

    def is_playing(self) -> bool:
        return self._playing and not self._paused

    def is_paused(self) -> bool:
        return self._playing and self._paused

    def set_volume(self, v: float):
        self.volume = max(0.0, min(1.0, v))

    def set_duck(self, ducked: bool):
        self.duck = 0.15 if ducked else 1.0

    def close(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
