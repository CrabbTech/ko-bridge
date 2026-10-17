#!/usr/bin/env python3
"""Hear a Lyria stem on the EP-133 with nobody at the device: chop it with the server's chop_sample into the scratch
project from ONE shared slot (trim start + length per pad, choke, the pickup reversed), write a two-bar chop pattern (the
eight grid chops in order on the beats, so the loop replays in its screwed tempo, an accent, and the reversed pickup as the
turnaround), render it with render_song, measure the take, then write the project back and delete the slot.

    uv run python scripts/lyria_chop_render.py --stem ~/Music/ko-bridge/library/wav/lyria-stem-<id>.wav [--project 9] [--screw -3.5] [--bpm 90] [--keep]
    uv run python scripts/lyria_chop_render.py --stem <wav> --plan ~/Music/ko-bridge/notes/gemini-plan-<ts>.json

With --plan (a GeminiBeatPlan saved by scripts/gemini_check.py) the chop pattern, fader bases and FX come from the plan
through plan_to_project_spec, with the drum and bass groups silenced so only the chops are heard. Every write is
checkpointed by the server; --keep leaves the chopped loop in its slot and the project as written.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from ko_bridge import gemini_producer as GP
from ko_bridge import server as srv
from ko_bridge.ep133 import DELETE_ACK, EP133
from ko_bridge.project import NoteEvent, PadSpec, PatternSpec, ProjectSpec, SceneSpec


def main() -> int:
    ap = argparse.ArgumentParser(description="Chop a stem into the scratch project and render it")
    ap.add_argument("--stem", required=True, help="the loop WAV (scripts/lyria_stem.py prints wav_path)")
    ap.add_argument("--project", type=int, default=9, help="the scratch slot (default 9)")
    ap.add_argument("--bpm", type=float, default=90.0, help="the stem's tempo")
    ap.add_argument("--screw", type=float, default=-3.5, help="turntable pitch+speed shift in semitones (0 for none)")
    ap.add_argument("--group", default="C")
    ap.add_argument("--keep", action="store_true", help="leave the chopped loop in its slot and the project as written")
    ap.add_argument("--plan", default=None, help="a GeminiBeatPlan JSON (scripts/gemini_check.py) to play instead of the fixed chop pattern")
    a = ap.parse_args()
    stem = os.path.expanduser(a.stem)
    if not os.path.isfile(stem):
        print(f"no such WAV: {stem}", file=sys.stderr)
        return 1

    ep = EP133(lock_owner="lyria_chop_render")
    active_before = ep.active_project().get("project")
    ckpt, before, _s = ep.checkpoint(a.project, "before-lyria-chop")
    occupied = {int(n["id"]) for n in ep.list_sounds()}
    upload_slot = ep.free_slots(occupied)[0]
    out: dict = {"stem": stem, "project": a.project, "checkpoint": str(ckpt)}
    try:
        chop = srv.chop_sample(stem, group=a.group, mode="hybrid", num_slices=12, screw_semitones=a.screw, source_bpm=a.bpm,
                               samplerate=46875, mono=False, reverse_last_pad=True, shared_slot=True, project=a.project, wait=True)
        out["chop"] = {"slot": chop["slot"], "audio_info": chop["audio_info"], "project_write_identical": chop["project_write"]["identical"],
                       "slices": [(s["pad"], s["kind"], s["start_frame"], s["end_frame"] - s["start_frame"], s["rms_db"]) for s in chop["slices"]]}
        tempo = round(a.bpm * float(chop["audio_info"].get("speed_ratio", 1.0)), 2)   # the screwed loop's tempo
        silent = [PatternSpec(group=g, number=1, bars=1) for g in "ABCD" if g != a.group]
        if a.plan:
            plan = GP.GeminiBeatPlan.model_validate_json(open(os.path.expanduser(a.plan)).read())
            chop_pads = [PadSpec.model_validate(p) for p in chop["pads"]]
            full, _clock = GP.plan_to_project_spec(plan, chop_pads=chop_pads, chop_group=a.group, build_two_scenes=False)
            chop_pattern = next(p for p in full.patterns if p.group == a.group)
            spec = ProjectSpec(bpm=tempo, pads=chop_pads, patterns=[chop_pattern] + silent, scene=SceneSpec(patterns={"A": 1, "B": 1, "C": 1, "D": 1}),
                               fader_bases=[fb for fb in full.fader_bases if fb.group == a.group], fx=full.fx)
            out["plan"] = {"file": a.plan, "title": plan.title, "chops": len(plan.chops), "bars": plan.bars, "swing": plan.swing, "fx": plan.fx_type}
        else:
            events = [NoteEvent(pad=str(i + 1), step=4 * i, velocity=110) for i in range(8)]      # one grid chop per beat, in order
            events += [NoteEvent(pad="9", step=14, velocity=100), NoteEvent(pad="enter", step=30, velocity=108)]   # accent, reversed turnaround
            spec = ProjectSpec(bpm=tempo, patterns=[PatternSpec(group=a.group, number=1, bars=2, events=events)] + silent,
                               scene=SceneSpec(patterns={"A": 1, "B": 1, "C": 1, "D": 1}))
        wr = srv.write_project(a.project, spec, reload=True, wait=True)
        out["pattern_write"] = {"identical": wr["identical"], "bpm": tempo, "bars": spec.patterns[0].bars, "events": len(spec.patterns[0].events)}
        r = srv.render_song(a.project, positions=[1], name="lyria-chop", wait=True)
        pos = r["positions"][0]
        grade = pos.get("grade") or {}
        high = [e["offset_ms"] for e in grade.get("events", []) if e.get("band") == "high"]
        out["render"] = {"take_id": r["song"]["take_id"], "wav": srv.get_take(r["song"]["take_id"])["wav"], "seconds": r["song"]["seconds"],
                         "midi_note_ons": pos["midi_note_ons"], "expected": grade.get("expected"), "matched": grade.get("matched"),
                         "high_band_offsets_ms": high, "peak_dbfs": grade.get("peak_dbfs"), "spectral": (grade.get("spectral_mix") or {}).get("bands_dbfs")}
        out["dsp_critique"] = srv.critique_take(r["song"]["take_id"], use_gemini=False)["critique"]["spectral_balance_assessment"]
    finally:
        if not a.keep:
            ep.write_project(a.project, before, "after-lyria-chop", reload=True)
            now = {int(n["id"]) for n in ep.list_sounds()}
            if upload_slot in now and upload_slot not in occupied:
                srv.delete_sound(upload_slot, acknowledge=DELETE_ACK.format(slot=upload_slot))
            if active_before and active_before != a.project:
                ep.select_project(active_before)
            out["restored"] = True
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
