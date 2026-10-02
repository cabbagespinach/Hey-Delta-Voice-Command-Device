#!/usr/bin/env python3
"""
Relabel every command clip into the class benchmark schema "Dataset Schema - Option B" (owner, 2026-10-01).

    python schema_b.py          # -> data/commands_schema_b.csv, data/commands_schema_b_dropped.csv,
                                #    model/deploy/label_map_schema_b.csv, evaluation/results/schema_b_counts.md

Classes: label + value (owner decision): the 13 Fixed labels, and each Slotted label once per value (TIMER_10S,
ALARM_8AM, COLOR_RED, ...) = 31 classes + `unknown`. Rules (owner decisions 2026-10-01):
- an old class that means the same as a schema class is relabelled (skip -> NEXT, louder -> VOLUME_UP,
  call_jane -> CALL, text_jane -> MESSAGE, dim_lights_20 -> BRIGHTNESS_20, remind_study -> CREATE_REMINDER_STUDY, ...);
- old commands with no schema class (play, party, timer 5/10 min, alarm 7 AM, dim 50/80, remind trash) become
  `unknown` near-misses (the value matters: "timer for 5 minutes" is not TIMER_1MIN);
- classmates' clips (recorded against this schema) take their class from the prompt id <LABEL>_V<variation>[_<value>];
- `unknown` clips whose words are a schema command are relabelled, in the schema's words (public "set an alarm for
  8AM", cut-off "Call") or in other words when the intent is clear ("Call Mom", "Pause for a moment"); ambiguous ones
  ("Stop talking", "Lights out at ten", SLURP "call a taxi") are dropped.
Splits are copied from commands_all (provisional: the class has not fixed its benchmark split yet).
"""
from pathlib import Path
import re, sys

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
D = ROOT / "data"
sys.path.insert(0, str(HERE))
import build_commands_all as bca                                  # noqa: E402
from generate_synthetic_commands import norm                      # noqa: E402

SCHEMA = ROOT / "Dataset Schema - Option B.csv"

OLD_TO_NEW = {
    "play_music": "PLAY_MUSIC", "weather": "WEATHER", "time": "TIME", "lights_on": "LIGHT_ON", "lights_off": "LIGHT_OFF",
    "pause": "PAUSE", "stop": "STOP", "next": "NEXT", "skip": "NEXT", "volume_up": "VOLUME_UP", "louder": "VOLUME_UP",
    "volume_down": "VOLUME_DOWN", "call_jane": "CALL", "text_jane": "MESSAGE", "list_reminders": "LIST_REMINDERS",
    "set_timer_1": "TIMER_1MIN", "set_alarm_6am": "ALARM_6AM", "set_alarm_9pm": "ALARM_9PM",
    "set_temperature_18": "TEMPERATURE_18", "set_temperature_22": "TEMPERATURE_22",
    "set_temperature_26": "TEMPERATURE_26", "dim_lights_20": "BRIGHTNESS_20", "remind_study": "CREATE_REMINDER_STUDY",
    # no schema class -> unknown near-misses
    "play": "unknown", "party": "unknown", "set_timer_5": "unknown", "set_timer_10": "unknown",
    "set_alarm_7am": "unknown", "dim_lights_50": "unknown", "dim_lights_80": "unknown", "remind_trash": "unknown",
}
# unknown sentences that now ask for a schema command in other words. Owner 2026-10-01: "we can still use data that
# don't have the same wording as long as it can be placed into an existing intent" -> relabelled when the intent is
# clear; None = ambiguous, dropped ("Stop talking" is not stop-playback, "Lights out at ten" is a schedule)
CONFLICT_TEXTS = {"call mom": "CALL", "call the doctor": "CALL", "text john": "MESSAGE", "message john": "MESSAGE",
                  "pause for a moment": "PAUSE", "pause for a second": "PAUSE", "keep it down": "VOLUME_DOWN",
                  "stop talking": None, "stop talking please": None, "lights out at ten": None}
# public SLURP requests: "call a taxi" / "post a message to facebook" are other services -> ambiguous, dropped
CONFLICT_WEB = re.compile(r"^(can you |please |i need to get to location )?call\b|\bcall (a|an) \w+|\bmessage\b")


def value_slug(v):
    v = v.strip()
    for pat, rep in [(r"^(\d+):00 ([AP]M)$", r"\1\2"), (r"^(\d+) seconds?$", r"\1S"), (r"^(\d+) minutes?$", r"\1MIN"),
                     (r"^(\d+) (degrees|percent)$", r"\1")]:
        if re.match(pat, v, re.I):
            return re.sub(pat, rep, v, flags=re.I).upper()
    return re.sub(r"\W+", "_", v).upper()


def schema_classes():
    s = pd.read_csv(SCHEMA)
    rows = []
    for r in s.itertuples(index=False):
        wordings = [r[2], r[3], r[4]]
        if r.Type.strip().lower() == "fixed":
            rows.append(dict(**{"class": r.Label}, label=r.Label, type="fixed", value="", value_index=0,
                             wordings=" | ".join(wordings)))
        else:
            for k, v in enumerate([r[5], r[6], r[7]], 1):
                rows.append(dict(**{"class": f"{r.Label}_{value_slug(v)}"}, label=r.Label, type="slotted", value=v,
                                 value_index=k, wordings=" | ".join(re.sub(r"\{\w+\}", v, w) for w in wordings)))
    return pd.DataFrame(rows)


def main():
    sc = schema_classes()
    assert len(sc) == 31 and sc["class"].is_unique, sc
    classes = set(sc["class"])
    assert set(OLD_TO_NEW.values()) - {"unknown"} <= classes, set(OLD_TO_NEW.values()) - classes
    by_prompt = dict(zip(zip(sc.label, sc.value_index), sc["class"]))
    exact = {norm(w): c for c, ws in zip(sc["class"], sc.wordings) for w in ws.split(" | ")}

    a = bca.collect(phrasing_test_exclusions=False)
    a = a.rename(columns={"class": "old_class"})
    # texts known per source
    txt = {}
    w = pd.read_csv(D / "commands_real_web/manifest.csv")
    txt.update(zip("data/commands_real_web/" + w.file, w.transcript))
    s = pd.read_csv(D / "commands_synthetic/manifest.csv")
    txt.update(zip("data/commands_synthetic/" + s.file, s.text))
    q = pd.read_csv(D / "commands_variants_train/manifest.csv")
    txt.update(zip(q.path, q.text))
    f = pd.read_csv(D / "commands_fragments/manifest.csv")
    txt.update(zip("data/commands_fragments/" + f.file, f.kept_text))
    frag_src = dict(zip("data/commands_fragments/" + f.file, f.source_label))
    o = pd.read_csv(D / "commands_other_speakers/manifest.csv")
    txt.update(zip(o.file, o.heard))
    prompt = dict(zip(o.file, o.prompt_id.fillna("")))
    a["text"] = a.path.map(txt)

    cls, rule = [], []
    for r in a.itertuples(index=False):
        t = norm(str(r.text)) if isinstance(r.text, str) else ""
        pid = prompt.get(r.path, "")
        m = re.fullmatch(r"([A-Z_]+?)_V(\d)(?:_(\d))?", pid) if pid else None
        if m:                                                    # classmates: the schema prompt they read
            c = by_prompt[(m.group(1), int(m.group(3) or 0))]
            cls.append(c); rule.append(f"prompt {pid}")
        elif r.old_class != "unknown":
            c = OLD_TO_NEW[r.label] if r.label in OLD_TO_NEW else OLD_TO_NEW[r.old_class]
            cls.append(c); rule.append(f"old {r.label}" + (" (no schema class)" if c == "unknown" else ""))
        elif r.dataset.startswith("fragments") and (t in exact or (frag_src.get(r.path) == "call_jane" and len(t.split()) == 1)):
            c = exact.get(t, "CALL")                             # the cut-off "Call Jane" clips now just say "Call"
            cls.append(c); rule.append(f"cut-off clip now says a whole command ('{r.text}')")
        elif t in exact:
            cls.append(exact[t]); rule.append(f"unknown clip says schema wording '{r.text}'")
        elif t in CONFLICT_TEXTS and CONFLICT_TEXTS[t]:
            cls.append(CONFLICT_TEXTS[t]); rule.append(f"unknown sentence fits the intent ('{r.text}')")
        elif t in CONFLICT_TEXTS or (r.dataset == "web_slurp" and CONFLICT_WEB.search(t)):
            cls.append("drop"); rule.append(f"unknown sentence, ambiguous intent ('{r.text}')")
        else:
            mm = re.fullmatch(r"(set|start) (an |the |my |a )?(alarm|timer) for (.+)", t)
            hit = None
            if mm and r.dataset.startswith("web_"):
                for c, v in zip(sc["class"], sc.value):                # the whole value, exactly ("eight a m")
                    if c.startswith(("ALARM_", "TIMER_")) and mm.group(4) == norm(v):
                        hit = c
            if hit:
                cls.append(hit); rule.append(f"unknown clip says '{r.text}' (schema value)")
            else:
                cls.append("unknown"); rule.append("unknown")
    a["class"], a["rule"] = cls, rule

    dropped = a[a["class"] == "drop"]
    a = a[a["class"] != "drop"]
    assert set(a["class"]) <= classes | {"unknown"}
    cols = ["path", "label", "class", "dataset", "speaker", "split", "real", "license", "old_class", "text", "rule"]
    a[cols].to_csv(D / "commands_schema_b.csv", index=False)
    dropped[cols].to_csv(D / "commands_schema_b_dropped.csv", index=False)
    sc.to_csv(HERE.parent / "model/deploy/label_map_schema_b.csv", index=False)

    # report
    grp = a.dataset.str.replace(r"^fragments_.*", "fragments", regex=True).str.replace(r"^reuse_.*", "reused unknowns", regex=True) \
        .str.replace(r"^web_.*", "public datasets", regex=True).str.replace(r"^synthetic_.*", "synthetic voices", regex=True) \
        .str.replace(r"^variants_.*", "other wordings (synthetic)", regex=True)
    grp = grp.replace({"owner_recordings": "your recordings", "converted_owner": "voice-converted (yours)",
                       "other_speakers": "other speakers"})
    order = list(sc["class"]) + ["unknown"]
    t = pd.crosstab(a["class"], grp).reindex(order).fillna(0).astype(int)
    t["total"] = t.sum(axis=1)
    real = a[a.real.astype(bool)]
    t["real speakers"] = real.groupby("class").speaker.nunique().reindex(order).fillna(0).astype(int)
    t.loc["TOTAL"] = t.sum()
    ch = a.groupby(["old_class", "class"]).size()
    out = HERE / "evaluation/results/schema_b_counts.md"
    lines = ["# Clips per schema-B class (existing data, relabelled; no new data yet)", "",
             f"{len(a)} clips kept, {len(dropped)} dropped (`data/commands_schema_b_dropped.csv`). Splits provisional.", "",
             t.to_markdown(), "", "## dropped, by reason", "",
             dropped.rule.str.replace(r" \(.*", "", regex=True).value_counts().to_markdown(), "",
             "## relabelled unknown -> class", "",
             a[(a.old_class == "unknown") & (a["class"] != "unknown")].assign(rule=lambda d: d.rule.str.replace(r" \('.*", "", regex=True))
             .groupby(["class", "rule"]).size().rename("clips").to_markdown()]
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
