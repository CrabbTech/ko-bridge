#!/usr/bin/env python3
"""The jungle track as a four-scene song: intro, drop, breakdown with a melody, second drop; song order 1-2-3-4-2.

    uv run python scripts/jungle_song.py --project 9 [--play-scene 3 --bars 4]

Builds on scripts/jungle.py (scene 2 = its drums, bass, harmony and colour patterns). New patterns: A02 intro hats,
A03 breakdown drums, A04 drop variation; B02 sub drone; C02 filtered strings, C03/C04 chords plus an "exotic pluck"
melody in E minor written in concert pitch; D02/D03 vocal colour. current_scene is the drop so a Mac START plays it;
the device's song mode plays the whole order (hold MAIN + ENTER, SHIFT+PLAY).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jungle  # noqa: E402

from ko_bridge import server as srv  # noqa: E402
from ko_bridge.project import Lane, LanePoint, NoteEvent, PadSpec, PatternSpec, ProjectSpec, SceneSpec  # noqa: E402

S = 16
BPM = jungle.BPM
MELODY_PAD = PadSpec(group="C", pad="4", slot=539, play_mode="key", release=110, amplitude=98)   # exotic pluck
PADS = jungle.PADS + [MELODY_PAD]
jungle.ROOT[("C", "4")] = MELODY_PAD.root


def ev(group, bar, step, pad, vel=100, semis=0, steps=1.0):
    return jungle.ev(group, bar, step, pad, vel, semis, steps)


def mel(bar: int, step: float, name: str, steps: float, vel: int = 104) -> NoteEvent:
    return NoteEvent(pad="4", step=bar * S + step, concert=name, velocity=vel, duration=max(1, min(384, int(steps * 24))))


MELODY_A = [  # breakdown, bars 0-3: (bar, step, note, length in steps)
    (0, 0, "E4", 3), (0, 3, "G4", 3), (0, 6, "B4", 2), (0, 8, "A4", 4), (0, 12, "G4", 4),
    (1, 0, "F#4", 6), (1, 6, "E4", 2), (1, 8, "D4", 8),
    (2, 0, "E4", 3), (2, 3, "G4", 3), (2, 6, "B4", 2), (2, 8, "D5", 4), (2, 12, "B4", 4),
    (3, 0, "A4", 6), (3, 6, "G4", 2), (3, 8, "E4", 8),
]
MELODY_B = [(0, 0, "E4", 4), (0, 8, "B4", 4), (1, 0, "G4", 4), (1, 8, "F#4", 4), (2, 0, "E4", 4), (2, 8, "D5", 4), (3, 0, "B4", 8)]


def intro_drums() -> PatternSpec:
    e = []
    for bar in range(4):
        for st in range(S):
            if st in (6, 14):
                e.append(ev("A", bar, st, "5", 90))
            else:
                e.append(ev("A", bar, st, "4", [92, 60, 78, 60][st % 4]))
        e += [ev("A", bar, st, "6", 68) for st in (2, 6, 10, 14)]
        e += [ev("A", bar, st, "3", v) for st, v in ((7, 58), (15, 52))]
    e.append(ev("A", 0, 0, "7", 96))
    e += [ev("A", 3, st, "2", v) for st, v in ((12, 96), (13, 88), (14, 104), (15, 118))]   # roll into the drop
    return PatternSpec(group="A", number=2, bars=4, events=e)


def breakdown_drums() -> PatternSpec:
    e = []
    for bar in range(4):
        e += [ev("A", bar, st, "4", v) for st, v in ((2, 70), (6, 80), (10, 70), (14, 80))]
        e += [ev("A", bar, st, "2", 104) for st in (4, 12)]
        e += [ev("A", bar, st, "3", v) for st, v in ((7, 60), (11, 56), (15, 62))]
        e.append(ev("A", bar, 14, "6", 70))
    e += [ev("A", 3, st, "1", 112) for st in (8, 10)]   # kicks announce the second drop
    e += [ev("A", 3, st, "2", v) for st, v in ((13, 92), (14, 104), (15, 118))]
    return PatternSpec(group="A", number=3, bars=4, events=e)


def drop_two_drums() -> PatternSpec:
    base = jungle.drums()
    e = list(base.events)
    e += [ev("A", 1, st, "1", 100) for st in (13,)]
    e += [ev("A", 2, st, "8", 90) for st in (4, 12)]
    e += [ev("A", 3, st, "3", v) for st, v in ((1, 64), (3, 70), (5, 76))]
    return PatternSpec(group="A", number=4, bars=4, events=e)


def sub_drone() -> PatternSpec:
    return PatternSpec(group="B", number=2, bars=4, events=[ev("B", 0, 0, "2", 96, 5, 16), ev("B", 1, 0, "2", 90, 5, 16),
                                                            ev("B", 2, 0, "2", 96, 5, 16), ev("B", 3, 0, "2", 90, 5, 16)])


def intro_harmony() -> PatternSpec:
    e = [ev("C", 0, 0, "2", 74, 4), ev("C", 2, 0, "2", 72, -1), ev("C", 1, 8, "1", 84, 0), ev("C", 3, 8, "1", 84, -5)]
    lanes = [Lane(parameter="lowpass", points=[LanePoint(step=0, value=0.3), LanePoint(step=2 * S, value=0.5), LanePoint(step=4 * S - 1, value=0.75)])]
    return PatternSpec(group="C", number=2, bars=4, events=e, lanes=lanes)


def breakdown_harmony() -> PatternSpec:
    e = [ev("C", 0, 0, "1", 92, 0), ev("C", 1, 8, "1", 88, -5), ev("C", 2, 0, "1", 92, 0), ev("C", 3, 0, "1", 88, 5), ev("C", 0, 0, "2", 70, 4)]
    e += [mel(*m) for m in MELODY_A]
    return PatternSpec(group="C", number=3, bars=4, events=e)


def drop_two_harmony() -> PatternSpec:
    base = jungle.harmony()
    e = list(base.events) + [mel(*m, vel=100) for m in MELODY_B]
    return PatternSpec(group="C", number=4, bars=4, events=e, lanes=[])


def intro_colour() -> PatternSpec:
    return PatternSpec(group="D", number=2, bars=4, events=[ev("D", 0, 0, "3", 70), ev("D", 2, 0, "3", 64), ev("D", 3, 10, "2", 88, -1), ev("D", 3, 14, "1", 92, 0)])


def breakdown_colour() -> PatternSpec:
    return PatternSpec(group="D", number=3, bars=4, events=[ev("D", 1, 14, "1", 86, -2), ev("D", 3, 6, "2", 84, -1), ev("D", 3, 7, "1", 80, 0)])


def spec() -> ProjectSpec:
    base = jungle.spec()
    patterns = base.patterns + [intro_drums(), breakdown_drums(), drop_two_drums(), sub_drone(), intro_harmony(), breakdown_harmony(),
                                drop_two_harmony(), intro_colour(), breakdown_colour()]
    scenes = [SceneSpec(patterns={"A": 2, "B": 2, "C": 2, "D": 2}),   # 1 intro
              SceneSpec(patterns={"A": 1, "B": 1, "C": 1, "D": 1}),   # 2 drop
              SceneSpec(patterns={"A": 3, "B": 1, "C": 3, "D": 3}),   # 3 breakdown with melody
              SceneSpec(patterns={"A": 4, "B": 1, "C": 4, "D": 1})]   # 4 second drop with melody
    return ProjectSpec(bpm=BPM, pads=PADS, patterns=patterns, scenes=scenes, song=[1, 2, 3, 4, 2], current_scene=2,
                       fader_bases=base.fader_bases, fx=base.fx)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=int, default=9)
    ap.add_argument("--play-scene", type=int, default=0, help="after writing, point the cursor at this scene and play it for --bars")
    ap.add_argument("--bars", type=int, default=4)
    args = ap.parse_args()
    s = spec()
    print("records per pattern:", {p.id: len(p.events) + sum(len(l.points) for l in p.lanes) for p in s.patterns})
    w = srv.write_project(args.project, s, wait=True)
    print(json.dumps({k: w[k] for k in ("project", "bytes", "write_seconds", "identical", "reload_seconds")}, indent=1))
    after = w["project_after"]
    print("scenes:", after["scenes"], "| song", after["song"], "| cursor", after["scene_cursor"], "| bpm", after["bpm"])
    c3 = after["patterns"]["C03"]
    print("C03 melody notes (pad 4):", [(e["step"], e["note"]) for e in c3["events"] if e["pad"] == "4"][:8])
    if args.play_scene:
        s2 = ProjectSpec(scenes=s.scenes, song=s.song, current_scene=args.play_scene)
        w2 = srv.write_project(args.project, s2, wait=True)
        print("cursor ->", args.play_scene, "identical", w2["identical"])
        r = srv.play_pattern(bpm=BPM, bars=args.bars, record=True, name=f"song-scene{args.play_scene}", wait=True)
        print(json.dumps({"midi_note_ons": r["midi_note_ons"], "expected": r["expected_events"], "take": r.get("take", {}).get("wav"),
                          "warning": r.get("warning") or r.get("take", {}).get("warning")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
