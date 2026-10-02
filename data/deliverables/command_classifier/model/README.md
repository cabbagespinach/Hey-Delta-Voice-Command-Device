# Command classifier model and training

- **`command_model.py`:** BC-ResNet (the wakeword project's implementation) with a 30-way head, plus `CommandNet`
  (front end + network, raw 5 s capture in, logits out). Widths: tau 1 = 9.8k, tau 3 = 56k, tau 6 = 191k parameters.
- **`train_commands.py` / `train_config.json`:** AdamW, lr 0.002, 2 warm-up epochs then cosine, label smoothing 0.05,
  SpecAugment (time masks ≤ 150 ms), 60 epochs × 48,000 draws (375 steps each). Each epoch scores the whole validation split.
  Selection score = mean of the command macro accuracy on owner / public / synthetic clips and 1 − the rate at which
  validation unknown clips trigger a command. Test data is never read.
- **Small-scale test (before the full training, owner request):**
  - `probe_shortcuts.py`: can the class be guessed from things that are not the words (the first 0.25 s, the overall
    colour/level, the length)?
  - Two short trainings (8 epochs × 24,000 draws = 1,500 steps, tau 3): `runs/probe_all` (all data) and `runs/probe_no_owner`
    (without the owner's recordings or anything derived from them).
  - `probe_report.py` compares them per command on the owner's validation session, showing which commands depend on
    the owner's own recordings, i.e. where more real recordings would help.
- **Full training:** `runs/bcresnet3`, `runs/bcresnet6` (`best.pt`, `history.json`, `val_predictions.csv`).
