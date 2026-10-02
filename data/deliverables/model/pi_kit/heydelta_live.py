#!/usr/bin/env python3
"""
"Hey Delta" streaming test on a laptop (Windows, macOS or Linux), all model sizes side by side.

    pip install numpy onnxruntime sounddevice          (scipy too, if your microphone can't do 16 kHz)

    python heydelta_laptop.py devices                  # list microphones
    python heydelta_laptop.py live --minutes 10        # listen; press Enter each time you say "Hey Delta"
    python heydelta_laptop.py wav recording.wav        # same test on a recording (16 kHz mono 16-bit WAV)

Every model in models.json listens to the SAME audio. Each gets a new 1.5 s window every 100 ms and wakes when
at least 2 of its last 3 window scores reach its own validated threshold, then stays quiet for 1 s: the same
rule as the evaluation on the server. Results go to results/ (a JSON summary and a CSV of every wake-up).

Efficiency (every run): results/<time>_efficiency.txt (readable) and .json: per model the time to score one
window, its share of one CPU core, model load time and file size; for the whole process peak memory and total CPU;
in live mode whether processing kept up with the microphone (delay, dropped audio); on a Raspberry Pi also CPU
temperature and throttling.

Marking what you said (live mode): press Enter right AFTER each "Hey Delta" you say. At the end, for every model:
  caught      = a wake-up within 2.5 s before your Enter press
  missed      = an Enter press with no wake-up in that range
  false wakes = wake-ups not near any Enter press
"""
from pathlib import Path
import argparse, csv, datetime, json, os, platform, queue, subprocess, sys, threading, time, wave

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
SR, WIN, HOP = 16000, 24000, 1600
RESULTS = HERE / "results"
MATCH_BEFORE_SEC = 2.5          # a wake-up up to 2.5 s before the Enter press counts as catching it


class Detector:
    def __init__(self, spec):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        self.name, self.thr = spec["name"], spec["threshold"]
        self.k, self.n = spec["k_of_n"]
        self.refractory = spec["refractory_sec"]
        t0 = time.perf_counter()
        self.s = ort.InferenceSession(str(HERE / spec["file"]), so, providers=["CPUExecutionProvider"])
        self.load_sec = time.perf_counter() - t0
        self.file_mb = (HERE / spec["file"]).stat().st_size / 1e6
        self.recent, self.last, self.wakes, self.score = [], -1e9, [], 0.0
        self.wall_ms, self.cpu_ms = [], []             # per window: elapsed time, CPU time of this thread

    def step(self, window, t):
        w0, c0 = time.perf_counter(), time.thread_time()
        p = float(self.s.run(["prob"], {"audio": window[None, :]})[0][0])
        self.cpu_ms.append((time.thread_time() - c0) * 1000)
        self.wall_ms.append((time.perf_counter() - w0) * 1000)
        self.score = p
        self.recent = (self.recent + [p >= self.thr])[-self.n:]
        if p >= self.thr and sum(self.recent) >= self.k and t - self.last >= self.refractory:
            self.last = t
            self.wakes.append(round(t, 2))
            return True
        return False


def load_models(only):
    cfg = json.loads((HERE / "models.json").read_text())
    specs = [m for m in cfg["models"] if not only or m["name"] in only]
    if not specs:
        sys.exit(f"no models match {only}; available: {[m['name'] for m in cfg['models']]}")
    return [Detector(m) for m in specs]


def to16k(x, sr):
    if sr == SR:
        return x
    from scipy.signal import resample_poly           # only needed when the source is not 16 kHz
    g = np.gcd(int(sr), SR)
    return resample_poly(x, SR // g, int(sr) // g).astype(np.float32)


class Perf:
    """Whole-run efficiency counters (live: how far processing lags behind the microphone)."""

    def __init__(self, live):
        self.live, self.wall0, self.cpu0 = live, time.perf_counter(), time.process_time()
        self.lag_sec, self.max_queue, self.overflows = [], 0, 0
        self.last_arrival = None            # perf_counter() when the newest microphone block arrived
        self.temp_start, self.throttle_start = cpu_temp_c(), pi_throttled()


def cpu_temp_c():
    """CPU temperature (Linux / Raspberry Pi); None elsewhere."""
    try:
        return round(int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000, 1)
    except (OSError, ValueError):
        return None


def pi_throttled():
    """Raspberry Pi firmware throttle flags (`vcgencmd get_throttled`); None if not a Pi."""
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=5).stdout
        return int(out.strip().split("=")[1], 16)
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def throttle_text(v):
    if v is None:
        return "n/a (not a Raspberry Pi)"
    if v == 0:
        return "none"
    bits = {0: "under-voltage now", 1: "frequency capped now", 2: "throttled now", 3: "soft temperature limit now",
            16: "under-voltage occurred", 17: "frequency capping occurred", 18: "throttling occurred",
            19: "soft temperature limit occurred"}
    return ", ".join(t for b, t in bits.items() if v >> b & 1) + f" (0x{v:x})"


def peak_rss_mb():
    try:
        import resource                                  # Linux / macOS
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(r / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)
    except ImportError:
        try:
            import psutil                                # Windows, if installed
            return round(psutil.Process().memory_info().peak_wset / 2 ** 20, 1)
        except Exception:
            return None


def system_info():
    info = dict(platform=platform.platform(), machine=platform.machine(), cpu_cores=os.cpu_count(),
                python=platform.python_version(), onnxruntime=ort.__version__)
    try:
        info["device"] = Path("/proc/device-tree/model").read_text().strip("\x00")
    except OSError:
        pass
    try:
        info["mem_total_mb"] = int(open("/proc/meminfo").readline().split()[1]) // 1024
    except (OSError, ValueError):
        pass
    return info


def efficiency_report(dets, perf, seconds, stamp):
    wall = time.perf_counter() - perf.wall0
    cpu = time.process_time() - perf.cpu0
    per_hop = HOP / SR * 1000                            # 100 ms of audio per window step
    models = {}
    for d in dets:
        w, c = np.array(d.wall_ms or [np.nan]), np.array(d.cpu_ms or [np.nan])
        models[d.name] = dict(
            windows=len(d.wall_ms), ms_per_window_median=round(float(np.nanmedian(w)), 2),
            ms_per_window_p95=round(float(np.nanpercentile(w, 95)), 2), ms_per_window_max=round(float(np.nanmax(w)), 2),
            cpu_share_of_one_core_pct=round(float(np.nanmean(c)) / per_hop * 100, 2),
            load_sec=round(d.load_sec, 3), file_mb=round(d.file_mb, 2))
    total_ms = sum(v["ms_per_window_median"] for v in models.values())
    lag = np.array(perf.lag_sec) if perf.lag_sec else None
    out = dict(system=system_info(), audio_minutes=round(seconds / 60, 2), wall_minutes=round(wall / 60, 2),
               models=models,
               all_models_ms_per_100ms_audio=round(total_ms, 2),
               all_models_cpu_share_of_one_core_pct=round(sum(v["cpu_share_of_one_core_pct"] for v in models.values()), 2),
               process_cpu_share_of_one_core_pct=round(cpu / wall * 100, 1) if perf.live and wall > 0 else None,
               peak_memory_mb=peak_rss_mb(),
               cpu_temp_c=dict(start=perf.temp_start, end=cpu_temp_c()),
               throttling=dict(start=throttle_text(perf.throttle_start), end=throttle_text(pi_throttled())))
    if perf.live:
        out["keeping_up"] = dict(
            delay_behind_microphone_sec_median=round(float(np.median(lag)), 3) if lag is not None else None,
            delay_behind_microphone_sec_max=round(float(lag.max()), 3) if lag is not None else None,
            max_audio_blocks_waiting=perf.max_queue, input_overflows=perf.overflows,
            verdict=("kept up" if lag is not None and lag.max() < 1.0 and perf.overflows == 0 else
                     "FELL BEHIND (see delay / overflows)"))
    (RESULTS / f"{stamp}_efficiency.json").write_text(json.dumps(out, indent=2) + "\n")
    L = [f"Efficiency report {stamp}", f"Device: {out['system'].get('device', out['system']['platform'])}, "
         f"{out['system']['cpu_cores']} cores" + (f", {out['system']['mem_total_mb']} MB RAM" if 'mem_total_mb' in out['system'] else ""),
         f"Audio processed: {out['audio_minutes']} min (wall clock {out['wall_minutes']} min)", "",
         f"{'model':<14}{'ms/window (median / p95 / max)':>32}{'% of one core':>15}{'load s':>8}{'file MB':>9}"]
    for k, v in models.items():
        L.append(f"{k:<14}{v['ms_per_window_median']:>14} / {v['ms_per_window_p95']} / {v['ms_per_window_max']:<8}"
                 f"{v['cpu_share_of_one_core_pct']:>13}{v['load_sec']:>8}{v['file_mb']:>9}")
    L += ["", f"All models together: {out['all_models_ms_per_100ms_audio']} ms per 100 ms of audio "
          f"({out['all_models_cpu_share_of_one_core_pct']}% of one core for the models)",
          (f"Whole process (models + audio + display): {out['process_cpu_share_of_one_core_pct']}% of one core, "
           if perf.live else "Whole process: (CPU share only measured in live mode; a file is processed as fast as possible), ")
          + f"peak memory {out['peak_memory_mb']} MB",
          "CPU temperature: " + (f"{out['cpu_temp_c']['start']} -> {out['cpu_temp_c']['end']} C"
                                 if out['cpu_temp_c']['end'] is not None else "n/a") + "; "
          f"throttling: {out['throttling']['end']}"]
    if perf.live:
        k = out["keeping_up"]
        L.append(f"Keeping up with the microphone: {k['verdict']} (delay median {k['delay_behind_microphone_sec_median']} s, "
                 f"max {k['delay_behind_microphone_sec_max']} s; dropped-audio events {k['input_overflows']})")
    txt = "\n".join(L) + "\n"
    (RESULTS / f"{stamp}_efficiency.txt").write_text(txt)
    print("\n" + txt + f"saved results/{stamp}_efficiency.txt and .json")


def run_stream(chunks, dets, markers_fn=None, show=True, perf=None):
    """chunks: iterator of float32 16 kHz arrays. Returns seconds of audio processed."""
    buf, t, pending = np.zeros(0, np.float32), 0.0, np.zeros(0, np.float32)
    for c in chunks:
        pending = np.concatenate([pending, c])
        while len(pending) >= HOP:
            hop, pending = pending[:HOP], pending[HOP:]
            buf = np.concatenate([buf, hop])[-WIN:]
            t += HOP / SR
            if len(buf) < WIN:
                continue
            woke = [d.name for d in dets if d.step(buf, t)]
            if perf is not None and perf.live and perf.last_arrival is not None:
                # delay from the arrival of the newest microphone block to the end of scoring this window
                perf.lag_sec.append(time.perf_counter() - perf.last_arrival)
            if show:
                level = 20 * np.log10(np.abs(hop).max() + 1e-9)
                bar = "#" * int(max(0, min(30, (level + 60) / 2)))
                scores = "  ".join(f"{d.name} {d.score:.2f}" for d in dets)
                sys.stdout.write(f"\r{t:7.1f}s  mic {level:6.1f} dBFS {bar:<30s} | {scores}   ")
                if woke:
                    sys.stdout.write(f"\n{t:7.1f}s  WAKE: {', '.join(woke)}\n")
                sys.stdout.flush()
    return t


def summarise(dets, seconds, markers, source, note):
    out = dict(created=datetime.datetime.now().isoformat(timespec="seconds"), source=source, note=note,
               minutes=round(seconds / 60, 2), enter_presses=[round(m, 2) for m in markers], models={})
    for d in dets:
        used, caught = set(), 0
        for m in markers:
            hit = [w for w in d.wakes if m - MATCH_BEFORE_SEC <= w <= m and w not in used]
            if hit:
                used.add(hit[0])
                caught += 1
        false = [w for w in d.wakes if w not in used]
        out["models"][d.name] = dict(threshold=d.thr, wakeups=len(d.wakes), caught=caught if markers else None,
                                     missed=(len(markers) - caught) if markers else None, false_wakes=len(false),
                                     false_wakes_per_hour=round(len(false) / (seconds / 3600), 1) if seconds else None,
                                     wake_times_sec=d.wakes)
    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    (RESULTS / f"{stamp}_summary.json").write_text(json.dumps(out, indent=2) + "\n")
    with open(RESULTS / f"{stamp}_wakeups.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "wake_time_sec"])
        for d in dets:
            w.writerows([d.name, x] for x in d.wakes)
    print("\n\n" + "model".ljust(16) + "wake-ups  caught  missed  false wakes  (per hour)")
    for k, v in out["models"].items():
        print(f"{k:<16}{v['wakeups']:>8}  {str(v['caught'] if v['caught'] is not None else '-'):>6}  "
              f"{str(v['missed'] if v['missed'] is not None else '-'):>6}  {v['false_wakes']:>11}  ({v['false_wakes_per_hour']})")
    print(f"\nsaved results/{stamp}_summary.json and results/{stamp}_wakeups.csv")
    return stamp


def cmd_devices(_):
    import sounddevice as sd
    print(sd.query_devices())
    print("\nUse the number with --device, e.g. --device 2")


def cmd_live(a):
    import sounddevice as sd
    dets = load_models(a.models)
    q = queue.Queue()
    rate = SR
    try:
        sd.check_input_settings(device=a.device, samplerate=SR, channels=1, dtype="float32")
    except Exception:
        rate = int(sd.query_devices(a.device, "input")["default_samplerate"])
        print(f"microphone can't record at 16 kHz; recording at {rate} Hz and converting (needs scipy)")
    perf = None

    def callback(data, frames, ti, status):
        if status and status.input_overflow and perf is not None:
            perf.overflows += 1                                # the OS dropped microphone audio
        q.put((time.perf_counter(), data[:, 0].copy()))
        if perf is not None:
            perf.max_queue = max(perf.max_queue, q.qsize())

    stream = sd.InputStream(device=a.device, samplerate=rate, channels=1, dtype="float32", blocksize=int(rate * 0.1),
                            callback=callback)
    markers, start = [], None
    recorded = [] if a.save_audio else None

    def enter_watcher():
        for _ in sys.stdin:
            if start is not None:
                markers.append(time_audio[0])
                sys.stdout.write(f"\n  [marked 'Hey Delta' at {time_audio[0]:.1f}s]\n")

    time_audio = [0.0]
    threading.Thread(target=enter_watcher, daemon=True).start()
    print(f"models: {', '.join(f'{d.name} (threshold {d.thr:.3f})' for d in dets)}")
    print(f"listening for {a.minutes} min. Say 'Hey Delta' and press Enter right after each one. Ctrl+C stops.\n")

    def chunks():
        nonlocal start
        start = time.time()
        while time.time() - start < a.minutes * 60:
            try:
                arrived, c = q.get(timeout=1.0)
            except queue.Empty:
                continue
            perf.last_arrival = arrived
            c = to16k(c, rate)
            if recorded is not None:
                recorded.append(c)
            time_audio[0] += len(c) / SR
            yield c

    seconds = 0.0
    perf = Perf(live=True)                                   # the clock starts with the microphone stream
    with stream:
        try:
            seconds = run_stream(chunks(), dets, perf=perf)
        except KeyboardInterrupt:
            seconds = time_audio[0]
    stamp = summarise(dets, seconds, markers, f"microphone {a.device if a.device is not None else '(default)'}", a.note)
    efficiency_report(dets, perf, seconds, stamp)
    if recorded:
        y = np.clip(np.concatenate(recorded), -1, 1)
        with wave.open(str(RESULTS / f"{stamp}_audio.wav"), "wb") as w:
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(SR)
            w.writeframes((y * 32767).astype("<i2").tobytes())
        print(f"saved results/{stamp}_audio.wav (16 kHz) - send it with the summary if you want it analysed")


def cmd_wav(a):
    dets = load_models(a.models)
    with wave.open(a.file, "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        if sw != 2:
            sys.exit("need a 16-bit PCM WAV")
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    x = to16k(x, sr)
    markers = [float(v) for v in a.marks.split(",")] if a.marks else []
    perf = Perf(live=False)
    seconds = run_stream((x[i:i + HOP] for i in range(0, len(x), HOP)), dets, show=not a.quiet, perf=perf)
    stamp = summarise(dets, seconds, markers, a.file, a.note)
    efficiency_report(dets, perf, seconds, stamp)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("devices")
    l = sub.add_parser("live")
    l.add_argument("--minutes", type=float, default=10)
    l.add_argument("--device", type=int, default=None, help="microphone number from `devices` (default: system default)")
    l.add_argument("--models", nargs="*", help="subset of model names (default: all in models.json)")
    l.add_argument("--note", default="", help="saved with the result, e.g. 'TV on, laptop mic, 1 m away'")
    l.add_argument("--save-audio", action="store_true", help="also save the recorded audio (off by default)")
    w = sub.add_parser("wav")
    w.add_argument("file")
    w.add_argument("--models", nargs="*")
    w.add_argument("--marks", default="", help="comma-separated times (s) right after each 'Hey Delta' in the file")
    w.add_argument("--note", default="")
    w.add_argument("--quiet", action="store_true", help="don't print the live meter")
    a = ap.parse_args()
    {"devices": cmd_devices, "live": cmd_live, "wav": cmd_wav}[a.cmd](a)


if __name__ == "__main__":
    main()
