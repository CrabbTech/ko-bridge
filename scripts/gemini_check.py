#!/usr/bin/env python3
"""The two Gemini 3.1 Pro calls of the producer layer, for real (paid, fractions of a cent each): compile a beat plan from a
prompt, and critique a recorded take. Results land as JSON under ~/Music/ko-bridge/notes/ so another session can pick them up.

    uv run python scripts/gemini_check.py                       # the plan for the chop & screw prompt (90 BPM, 2 bars)
    uv run python scripts/gemini_check.py --take <take_id>      # also the critique of a rendered take (the WAV leaves the machine)

GEMINI_API_KEY comes from ~/Music/ko-bridge/secrets.env (or op-bridge's); it is never printed. A failed request is an
error here, never a silent heuristic plan: `model` in the output names the model that answered.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from ko_bridge import gemini_producer as GP

NOTES = os.path.expanduser("~/Music/ko-bridge/notes")
DEFAULT_PROMPT = ("Houston chop and screw beat built on a two-bar 1970s soul Rhodes-and-horns loop chopped across group C "
                  "(pads 1..8 grid chops in order, 9/0/. accents, enter the reversed pickup); heavy double-up stutters on "
                  "bar 2 and the reversed pickup on beat 4.4")


def main() -> int:
    ap = argparse.ArgumentParser(description="Gemini 3.1 Pro beat plan and take critique (paid)")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--bpm", type=float, default=90.0)
    ap.add_argument("--bars", type=int, default=2)
    ap.add_argument("--take", default=None, help="a take id (list_takes) to critique with Gemini listening to the WAV")
    ap.add_argument("--no-plan", action="store_true", help="skip the plan, only critique --take")
    a = ap.parse_args()
    os.makedirs(NOTES, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out: dict = {}
    if not a.no_plan:
        plan, model = GP.compile_beat_with_gemini(a.prompt, bpm=a.bpm, bars=a.bars)
        path = os.path.join(NOTES, f"gemini-plan-{stamp}.json")
        with open(path, "w") as f:
            f.write(plan.model_dump_json(indent=1))
        out["plan"] = {"model": model, "saved": path, "title": plan.title, "bpm": plan.bpm, "bars": plan.bars, "swing": plan.swing,
                       "snare_delay_ticks": plan.snare_delay_ticks, "screw_semitones": plan.screw_semitones,
                       "drums": len(plan.drums), "bass": len(plan.bass), "chops": len(plan.chops), "fx_dives": len(plan.fx_dives),
                       "chop_pads_used": sorted({c.pad for c in plan.chops}), "stutters": sorted({c.stutter for c in plan.chops}),
                       "producer_notes": plan.producer_notes}
    if a.take:
        from ko_bridge import server as srv
        crit = srv.critique_take(a.take)
        path = os.path.join(NOTES, f"gemini-critique-{stamp}.json")
        with open(path, "w") as f:
            json.dump(crit, f, indent=1, default=str)
        out["critique"] = {"model": crit["model"], "saved": path, **{k: v for k, v in crit["critique"].items() if k != "pad_adjustments"},
                           "pad_adjustments": len(crit["critique"].get("pad_adjustments", []))}
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
