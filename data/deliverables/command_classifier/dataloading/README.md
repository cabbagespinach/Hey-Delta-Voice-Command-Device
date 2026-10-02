# Phase 3: dataset and dataloader (command classifier)

| File | Role |
|---|---|
| `../build_commands_all.py` | One list of every clip (`data/commands_all.csv`): path, label, class, source dataset, speaker, split, real/synthetic, licence |
| `dataloader_config.json` | Sampling shares, format alignment, workers (reasons in `_*` keys) |
| `command_data.py` | `TrainDraws` (weighted, augmented training draws per epoch), `EvalClips` (every clip once, unaugmented), `make_loader` |
| `test_dataloader.py` | Split integrity (no voice in two splits, owner session/speaker rules, converted clips from training recordings only, fragments follow their source), draw mix, reproducibility, train-only prefix bank |

**Splits (owner decisions of 2026-09-30).**
- **Owner recordings:** test = all of `speaker2` plus the owner's 16:18–16:33 session; validation = the owner's
  13:36–13:55 sessions; train = the rest. Clips Whisper found empty or cut off are left out, except for the unknown prompts.
- **Public datasets:** their own speaker-disjoint splits.
- **Synthetic voices:** each voice in one split (80/10/10 by name hash). The 10 voice-conversion target voices are in train.
- **Converted clips:** only from training recordings.
- **Reused unknowns and fragments:** follow the split of their source recording.

**Training draws.** An epoch is 12,000 weighted draws, not a pass over the clips (the sources differ in size by 100x):
- 20% `unknown`, spread over its sources: owner prompts, owner room, MUSAN, FLEURS, MSWC, public requests, synthetic,
  fragments.
- 80% commands: the class is chosen uniformly, then the source group (owner 25%, public 25%, synthetic 50%,
  renormalised over what the class has), then a clip.

Each draw has its own seed from (seed, epoch, index), so an epoch is identical for any worker count.

**Format alignment.** Every capture starts with the "Delta" tail and a pause, but public clips and reused unknowns
don't. Each of those gets a real prefix from a bank of TRAINING prefixes, in every split (random in training, fixed
per clip in validation/test). Otherwise "no tail" would mean "public dataset".

**Augmentation.** Training only, through `../augment_commands.py` (see `../coverage/README.md`).
