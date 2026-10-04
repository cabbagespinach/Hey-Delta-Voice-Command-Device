# Class benchmark kit: "Hey Delta" on the Raspberry Pi 5

For the class's live benchmark, **github.com/airimonda/vcm-benchmark** (checked against commit `ab39857`, 2026-10-02 18:57). The laptop
plays "Hey Delta" + a holdout command out loud; the Pi runs our assistant and writes one line per command; the
laptop scores it. Models: `hf_plus` = class HF data + our data, `hf_only` = HF data only, `hf_ourlabels` (only in
the owner's copy) = hf_plus retrained 2026-10-02 with our labels on the owner's HF clips. The benchmark scores **one model per run**; to compare
several models on the **same** run, see section 4.

This kit is for the standard test.

## 0. Build the kit (from a clone of the repository)

```bash
python3 data/deliverables/command_classifier/model/make_bench_kit.py   # -> .../model/vcm_bench_kit.zip
scp data/deliverables/command_classifier/model/vcm_bench_kit.zip <user>@<pi-ip>:~
```

It downloads the pinned holdout (`holdout_da92a79.parquet`, 15 MB) from Hugging Face once. A kit built from the
public repository has no reference clips (audio), so `--selftest` uses three noise captures instead.

## 1. On the Pi

```bash
unzip vcm_bench_kit.zip && cd vcm_bench_kit
python3 -m venv ~/bench && source ~/bench/bin/activate      # or reuse the venv of the live kit
pip install numpy onnxruntime sounddevice scipy
python3 vcm_bench_assistant.py --selftest --model hf_plus   # 5 JSON lines, no microphone needed
```

Then start the assistant and leave it running (in tmux or a second SSH window):

```bash
python3 vcm_bench_assistant.py --model hf_plus --log ~/vcm_bench_kit/bench.log
```

Each decision is written to `bench.log` as one JSON line, e.g.
`{"intent": "TIMER_30S", "slot": "", "infer_ms": 61.2, "audio_ms": 5000, ...}` and `{"event": "wake", ...}`.
Default rule: `cautious` (the model's default). Other rules: `--rule balanced` or `--rule argmax`.
Microphone: `--device <number>` (list them with `python3 -c "import sounddevice; print(sounddevice.query_devices())"`).

## 2. On the laptop

```bash
git clone https://github.com/airimonda/vcm-benchmark.git && cd vcm-benchmark
git checkout ab39857                                   # the version this kit was checked against
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python benchmark.py --mode ssh --host <user>@<pi-address> \
    --log ~/vcm_bench_kit/bench.log --proc vcm_bench_assistant.py \
    --wake-word "Hey Delta" --holdout <path>/vcm_bench_kit/holdout_da92a79.parquet \
    --model pi:~/vcm_bench_kit/models/bcresnet6_hf_plus/command_bcresnet6_hf_plus.onnx \
    --student "Arvir Jane R. Redondo (hf_plus, cautious)"
```

* `--holdout ...parquet`: the class holdout **pinned** to dataset revision `da92a79` (202 clips: 186 commands + 16
  out of scope), shipped in this kit, so every run tests the same clips. Without it, the benchmark downloads the
  dataset's *latest* revision, which changed on 2026-10-02.
* `--model pi:...`: lets the benchmark count FLOPs / parameters of the command model (use the `hf_only` path
  when benchmarking that model).
* Wake word: the script records you saying "Hey Delta" 3 times on the laptop. The default pause between wake word
  and command (0.8 s) suits our listener (it ignores ~0.3 s after waking while the chime plays); if the first word of
  commands gets cut, use `--wake-gap 1.0`.
* For the second model, stop the assistant (Ctrl+C), restart it with `--model hf_only`, and run `benchmark.py`
  again with the `hf_only` model path and a different `--student` note. Use the **same `--seed`** both times so both
  models hear the same order.

## 3. Fair test (from the benchmark README)

Laptop speaker about 1 m from the Pi microphone, quiet room, volume set in the sound check and then left alone, no
other audio on the laptop. Results: `runs/<date-time>/report.md` on the laptop.

## Notes

* `audio_ms` = 5000: the model always processes a 5 s window (shorter captures are padded), so the real-time factor
  is `infer_ms / 5000`. The actual capture length is in `capture_ms`.
* Out of scope: our `unknown` is written as `OUT_OF_SCOPE`; if nothing is said after the wake word, nothing is
  written (scored as no response).
* Licence of the model weights: CC BY-NC 4.0.

## 4. Several models in one run (same captures)

Every capture goes to all the models; the benchmark scores the first (`--model`), the others are scored afterwards on
the laptop with the benchmark's own scoring (`--rescore`), so all of them heard exactly the same 202 clips.

On the Pi (instead of the command in section 1):

```bash
python3 vcm_bench_assistant.py --selftest --model hf_plus --also hf_only    # check: 5 lines + 5 others
rm -f ~/vcm_bench_kit/bench.log ~/vcm_bench_kit/bench_others.jsonl                       # start clean
python3 vcm_bench_assistant.py --model hf_plus --also hf_only --log ~/vcm_bench_kit/bench.log
```

* `bench.log`: the `hf_plus` (cautious) answers, written first, so the benchmark's timing is that model alone; each
  line has a `seq` number.
* `bench_others.jsonl`: for the same `seq`, every model's answer under all three rules (argmax / cautious /
  balanced) + its own inference time. The benchmark does not read this file.

On the laptop: run `benchmark.py` exactly as in section 2 (with the `hf_plus` model path). When it has finished:

```bash
scp <user>@<pi-address>:~/vcm_bench_kit/bench_others.jsonl runs/<date-time>/
python <path>/vcm_bench_kit/score_multi.py runs/<date-time>
```

Default variants: `hf_plus:cautious` (official), `hf_plus:balanced`, `hf_only:cautious`,
`hf_only:balanced`; others with e.g. `--variants hf_plus:balanced,hf_only:argmax`. Output:

* `runs/<date-time>/multi_model_report.md`: one table for all variants + the trials where they disagree.
* `runs/<date-time>__<model>_<rule>/report.md`: the benchmark's full report for each variant.

Per model: all classification numbers, slots, inference time, FLOPs / parameters. **Shared** by all variants (one
capture): the wake word (a missed wake is a miss for every model), response latency, and the Pi's CPU / RAM /
temperature (the process runs all the models). Only the `--model` result is a normal, official benchmark run.
