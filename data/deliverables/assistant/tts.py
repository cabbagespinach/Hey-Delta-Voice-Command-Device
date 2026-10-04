"""
Spoken replies with an offline Piper voice (piper-tts 1.8 API: PiperVoice.load, synthesize_wav).

say(text) queues the sentence and returns at once; a background thread synthesises it and plays it through AudioOut.
`speaking` is True from queueing until the sound has finished, so the assistant can ignore its own voice.
Piper's onnxruntime session would use every core; it is capped (threads=2 by default) so the wakeword listener keeps
running smoothly. If Piper or the voice is missing, replies are printed only.
"""
from __future__ import annotations

from pathlib import Path
import io, queue, threading, wave

import numpy as np


def _cap_onnx_threads(n: int):
    """Piper builds its InferenceSession without SessionOptions (= one thread per core): force n threads."""
    import onnxruntime as ort
    if getattr(ort.InferenceSession, "_capped", False):
        return
    base = ort.InferenceSession

    class Capped(base):
        _capped = True

        def __init__(self, path, sess_options=None, providers=None, **kw):
            so = sess_options or ort.SessionOptions()
            so.intra_op_num_threads, so.inter_op_num_threads = n, 1
            super().__init__(path, sess_options=so, providers=providers, **kw)
    ort.InferenceSession = Capped


class Speaker:
    def __init__(self, audio, voice_path: Path | None, threads: int = 2, length_scale: float = 1.0, enabled=True):
        self.audio, self.length_scale = audio, length_scale
        self.voice, self.speaking = None, False
        self._q: queue.Queue = queue.Queue()
        if enabled and voice_path and Path(voice_path).exists():
            try:
                _cap_onnx_threads(threads)
                from piper import PiperVoice
                self.voice = PiperVoice.load(str(voice_path))
            except Exception as e:
                print(f"(Piper voice not loaded, replies are printed only: {e})")
        elif enabled:
            print(f"(no Piper voice at {voice_path}; replies are printed only)")
        threading.Thread(target=self._worker, daemon=True).start()

    def synth(self, text: str) -> tuple[np.ndarray, int]:
        from piper import SynthesisConfig
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            self.voice.synthesize_wav(text, w, syn_config=SynthesisConfig(length_scale=self.length_scale))
        buf.seek(0)
        with wave.open(buf) as w:
            sr = w.getframerate()
            y = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        return y, sr

    def say(self, text: str):
        print(f"  assistant: {text}", flush=True)
        if self.voice is None:
            return
        self.speaking = True
        self._q.put(text)

    def _worker(self):
        while True:
            text = self._q.get()
            try:
                y, sr = self.synth(text)
                self.audio.play_clip(y, sr, block=True)
            except Exception as e:
                print(f"(could not speak: {e})")
            finally:
                if self._q.empty():
                    self.speaking = False

    def wait(self, timeout: float = 30.0):
        import time
        t = time.time()
        while self.speaking and time.time() - t < timeout:
            time.sleep(0.05)
