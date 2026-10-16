#!/usr/bin/env python3
"""A fast jungle breakbeat in E minor at 170 BPM, written into a project slot through the ko-bridge server and played back.

    uv run python scripts/jungle.py --project 9 [--bars 8] [--no-play]

Factory sounds only: kick hard, snare smack, rim mid (ghosts), closed hat mid, open hat drive, shaker dmx, crash, clap real;
reese + thirty seven sub for bass (key mode, transposed into E minor); liquid chord em7, soothe string and octave stab for
harmony; the s6i2 vocal chops and the birds tape for colour. Transpositions come from the samples' root notes
(reese G, sub B, strings C, stab A#, vox F# and F).
"""
from __future__ import annotations

import argparse
import json
import sys

from ko_bridge import server as srv
from ko_bridge.project import Lane, LanePoint, NoteEvent, PadSpec, PatternSpec, ProjectSpec, FaderBase, FxSpec, pad_note

BPM = 170.0
S = 16  # steps per bar

PADS = [
    # group A: drums (one-shots play through)
    PadSpec(group="A", pad="1", slot=16, amplitude=112),               # kick hard
    PadSpec(group="A", pad="2", slot=111, amplitude=110),              # snare smack
    PadSpec(group="A", pad="3", slot=134, amplitude=104),              # rim mid (ghost notes)
    PadSpec(group="A", pad="4", slot=204, amplitude=118),              # closed hat mid
    PadSpec(group="A", pad="5", slot=222, amplitude=112),              # open hat drive
    PadSpec(group="A", pad="6", slot=336, amplitude=100),              # shaker dmx
    PadSpec(group="A", pad="7", slot=248, amplitude=80),               # crash
    PadSpec(group="A", pad="8", slot=303, amplitude=86),               # clap real
    # group B: bass, key mode so the note length and release shape each note
    PadSpec(group="B", pad="1", slot=429, play_mode="key", release=90, amplitude=96),    # reese (G)
    PadSpec(group="B", pad="2", slot=426, play_mode="key", release=120, amplitude=82),   # thirty seven sub (B)
    # group C: harmony
    PadSpec(group="C", pad="1", slot=542, amplitude=92),               # liquid chord em7 (E), plays through
    PadSpec(group="C", pad="2", slot=522, amplitude=72),               # soothe string (C), plays through
    PadSpec(group="C", pad="3", slot=511, amplitude=96),               # octave stab (A#)
    # group D: colour
    PadSpec(group="D", pad="1", slot=557, amplitude=94),               # vox s6i2 gated (F#)
    PadSpec(group="D", pad="2", slot=553, amplitude=90),               # vox s6i2 eh (F)
    PadSpec(group="D", pad="3", slot=556, amplitude=66),               # fx birds tape
]
ROOT = {(p.group, p.pad): p.root for p in PADS}


def ev(group: str, bar: int, step: float, pad: str, vel: int = 100, semis: int = 0, steps: float = 1.0) -> NoteEvent:
    """A note at bar (0-based), step (16ths), transposed by `semis`, lasting `steps` 16ths (matters in key mode)."""
    return NoteEvent(pad=pad, step=bar * S + step, note=ROOT[(group, pad)] + semis, velocity=vel, duration=max(1, min(384, int(steps * 24))))


def drums() -> PatternSpec:
    e = []
    hat_vel = [110, 72, 92, 72] * 4
    opens = {6, 14}
    for bar in range(4):
        bar_opens = opens if bar != 3 else {6}
        for st in range(S):
            if st in bar_opens:
                e.append(ev("A", bar, st, "5", 100 if st == 6 else 94))
            else:
                e.append(ev("A", bar, st, "4", hat_vel[st]))
        for st in (2, 6, 10, 14):
            e.append(ev("A", bar, st, "6", 70))
    kicks = {0: [(0, 120), (10, 112)], 1: [(0, 120), (6, 105), (11, 96)], 2: [(0, 120), (10, 112), (13, 92)], 3: [(0, 120), (8, 104)]}
    snares = {0: [(4, 118), (12, 118)], 1: [(4, 118), (11, 108)], 2: [(4, 118), (12, 118)], 3: [(4, 118), (12, 112), (13, 88), (14, 100), (15, 116)]}
    rims = {0: [(2, 45), (7, 62), (15, 55)], 1: [(7, 60), (13, 66), (15, 52)], 2: [(2, 48), (7, 62), (15, 55)], 3: [(6, 58), (7, 66), (10, 60)]}
    claps = {2: [(12, 85)], 3: [(4, 88), (12, 85)]}
    for bar in range(4):
        e += [ev("A", bar, st, "1", v) for st, v in kicks[bar]]
        e += [ev("A", bar, st, "2", v) for st, v in snares[bar]]
        e += [ev("A", bar, st, "3", v) for st, v in rims[bar]]
        e += [ev("A", bar, st, "8", v) for st, v in claps.get(bar, [])]
    e.append(ev("A", 0, 0, "7", 100))   # crash on the one
    return PatternSpec(group="A", number=1, bars=4, events=e)


def bass() -> PatternSpec:
    E, D, G, B = -3, -5, 0, 4          # reese is G: offsets into E minor
    e = [ev("B", 0, 0, "1", 110, E, 12), ev("B", 1, 6, "1", 100, D, 5), ev("B", 1, 12, "1", 104, E, 4),
         ev("B", 2, 0, "1", 110, E, 8), ev("B", 2, 10, "1", 100, G, 6), ev("B", 3, 0, "1", 104, D, 6), ev("B", 3, 8, "1", 106, B, 8),
         ev("B", 0, 0, "2", 100, 5, 16), ev("B", 2, 0, "2", 100, 5, 16)]   # sub (B) +5 = E, a bar each
    return PatternSpec(group="B", number=1, bars=4, events=e)


def harmony() -> PatternSpec:
    e = [ev("C", 0, 0, "1", 100, 0), ev("C", 1, 10, "1", 92, -5), ev("C", 2, 0, "1", 100, 0), ev("C", 3, 8, "1", 92, 5),   # Em7, Bm7, Em7, Am7
         ev("C", 0, 0, "2", 72, 4), ev("C", 2, 0, "2", 70, -1),                                                           # strings on E, then B
         ev("C", 1, 14, "3", 90, -6), ev("C", 3, 6, "3", 86, -6), ev("C", 3, 14, "3", 92, 1)]                             # stabs on E, E, B
    lanes = [Lane(parameter="lowpass", points=[LanePoint(step=0, value=0.45), LanePoint(step=S, value=0.7), LanePoint(step=2 * S, value=1.0)])]
    return PatternSpec(group="C", number=1, bars=4, events=e, lanes=lanes)


def colour() -> PatternSpec:
    e = [ev("D", 0, 0, "3", 60), ev("D", 0, 14, "1", 92, 0), ev("D", 1, 8, "2", 90, -1), ev("D", 2, 6, "1", 90, -2), ev("D", 2, 7, "1", 80, 0),
         ev("D", 3, 2, "2", 86, -1)]
    return PatternSpec(group="D", number=1, bars=4, events=e)


def spec() -> ProjectSpec:
    return ProjectSpec(bpm=BPM, pads=PADS, patterns=[drums(), bass(), harmony(), colour()],
                       fader_bases=[FaderBase(group="C", parameter="fx", value=0.3), FaderBase(group="D", parameter="fx", value=0.5)],
                       fx=FxSpec(type="reverb", x=0.45, y=0.5, output_compressor_drive=0.35, output_compressor_speed=0.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=int, default=9)
    ap.add_argument("--bars", type=int, default=8)
    ap.add_argument("--no-play", action="store_true")
    args = ap.parse_args()
    s = spec()
    records = {p.id: len(p.events) + sum(len(l.points) for l in p.lanes) for p in s.patterns}
    print("pattern records:", records)
    w = srv.write_project(args.project, s, wait=True)
    print(json.dumps({k: w[k] for k in ("project", "bytes", "write_seconds", "identical", "readback_seconds", "reload_seconds", "checkpoint")}, indent=1))
    after = w["project_after"]
    print("pads:", {k: v["slot"] for k, v in after["pads"].items()}, "| scene", after["scenes"][0]["patterns"], "| bpm", after["bpm"], "| fx", after.get("fx"))
    if args.no_play:
        return 0
    r = srv.play_pattern(bpm=BPM, bars=args.bars, record=True, name="jungle", wait=True)
    g = r.get("grade", {})
    print(json.dumps({"midi_note_ons": r["midi_note_ons"], "expected": g.get("expected"), "matched": g.get("matched"),
                      "offset_ms": g.get("offset_stats_ms"), "jitter_ms": g.get("jitter_relative_to_first_ms"), "settled": r.get("settled_seconds"),
                      "warning": r.get("warning") or r.get("take", {}).get("warning"), "take": r.get("take", {}).get("wav"),
                      "peak_dbfs": r.get("take", {}).get("peak_dbfs"), "seconds": r.get("take", {}).get("seconds")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
