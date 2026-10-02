# Phase 4: validation and streaming evaluation (command classifier)

| File | Role |
|---|---|
| `evaluate_commands.py` | Confidence cutoffs chosen on VALIDATION (cautious: ≤ 2% of unknown clips trigger a command; balanced: ≤ 5%; mean over unknown sources), then TEST results per model and cutoff, per source group (owner test session, speaker2, public, synthetic): correct / wrong command / asks again; unknown clips that trigger a command, per kind |
| `streaming_commands.py` | The whole Pi path on continuous audio: 40 × 60 s streams from TEST material over the owner's real room background, with owner events (real "Hey Delta" + real command), cloned-voice events (command or off-list sentence) and distractor speech. Runs the deployed `HeyDeltaListener` (wakeword ONNX, capture, end-of-speech rule), then the classifier. Reports wake rate, command outcome, off-list actions, end-to-end success, false wake-ups and the actions they cause. |
| `phase4_report.py` | One summary page (`results/phase4_summary.md`) and the recommended model (by validation score only) |
| `run_phase4.sh` | Runs all of the above |

Results are written to `results/`.
