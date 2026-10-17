#!/usr/bin/env python3
"""Drum and bass at 174 BPM in E minor from the synthesized dnb-174 set: six 8-bar scenes with a real arc.

    uv run python scripts/dnb.py --project 9            # write (slots from library/sets/dnb-174/slots.json)

Scenes: 1 intro (pad, bells, crackle, choir; riser into 2) · 2 groove (drums in, sub, epiano, pluck; fill + riser)
· 3 drop (full break, baked-wobble reese loops, growl answers, impact) · 4 breakdown (choir, epiano melody, pluck
arpeggio, sub; drums thin; riser, snare roll, one silent beat) · 5 drop 2 (heavier: 1/16 and triplet wobbles, stutter,
growl, double kicks, open hats) · 6 outro (drums fall away over a level lane, pad and bells remain). Song 1-2-3-4-5-6.
Bass loops are one bar long and retriggered on every bar at native pitch, so their baked LFOs stay on the grid.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from ko_bridge import server as srv
from ko_bridge.project import FaderBase, FxSpec, Lane, LanePoint, NoteEvent, PadSpec, PatternSpec, ProjectSpec, SceneSpec, SidechainSource

BPM = 174.0
S = 16
BARS = 8
SET = os.path.expanduser("~/Music/ko-bridge/library/sets/dnb-174")

def _slots() -> dict[str, int]:
    path = os.path.join(SET, "slots.json")
    if os.path.exists(path):
        return json.load(open(path))["slots"]
    manifest = json.load(open(os.path.join(SET, "manifest.json")))["items"]
    return {item["name"]: 602 + i for i, item in enumerate(manifest)}     # dry-run numbering before the upload


SLOTS = _slots()

# pad layout: group A drums, B bass, C keys/pads (pad C3 on three pads for chords), D colour/fx
LAYOUT = {
    ("A", "1"): ("kick punch", dict(amplitude=100)), ("A", "2"): ("kick deep", dict(amplitude=88)), ("A", "3"): ("snare", dict(amplitude=98)),
    ("A", "4"): ("clap", dict(amplitude=80)), ("A", "5"): ("hat closed", dict(amplitude=94, choke=True)), ("A", "6"): ("hat open", dict(amplitude=88, choke=True)),
    ("A", "7"): ("ride", dict(amplitude=80)), ("A", "8"): ("shaker", dict(amplitude=84)), ("A", "9"): ("rim", dict(amplitude=94)),
    ("A", "."): ("tom", dict(amplitude=94)), ("A", "0"): ("zap", dict(amplitude=86)),
    ("B", "1"): ("reese E1 wub8", dict(amplitude=86)), ("B", "2"): ("reese E1 wub16", dict(amplitude=86)), ("B", "3"): ("reese E1 wub4", dict(amplitude=86)),
    ("B", "4"): ("reese E1 trip", dict(amplitude=86)), ("B", "5"): ("reese E1 stutter", dict(amplitude=86)), ("B", "6"): ("reese G1 wub8", dict(amplitude=86)),
    ("B", "7"): ("reese A1 wub8", dict(amplitude=86)), ("B", "8"): ("reese B1 wub8", dict(amplitude=86)), ("B", "9"): ("reese D2 wub8", dict(amplitude=86)),
    ("B", "."): ("growl E1 wub8", dict(amplitude=86)), ("B", "0"): ("growl G1 wub8", dict(amplitude=86)), ("B", "enter"): ("growl E1 wub4", dict(amplitude=86)),
    ("C", "1"): ("sub E1", dict(amplitude=94, play_mode="key", release=120)), ("C", "2"): ("bass pluck E2", dict(amplitude=90)),
    ("C", "3"): ("epiano C4", dict(amplitude=92, play_mode="key", release=120)), ("C", "4"): ("string pluck A3", dict(amplitude=90)),
    ("C", "5"): ("bell C5", dict(amplitude=82)), ("C", "6"): ("pad C3", dict(amplitude=76, play_mode="key", release=90)),
    ("C", "7"): ("pad C3", dict(amplitude=76, play_mode="key", release=90)), ("C", "8"): ("pad C3", dict(amplitude=76, play_mode="key", release=90)),
    ("C", "9"): ("choir A2", dict(amplitude=78, play_mode="key", release=110)), ("C", "."): ("choir A2", dict(amplitude=78, play_mode="key", release=110)),
    ("D", "1"): ("vox E4", dict(amplitude=86)), ("D", "2"): ("riser 2 bars", dict(amplitude=90)), ("D", "3"): ("impact", dict(amplitude=96)),
    ("D", "4"): ("crackle 4 bars", dict(amplitude=84)),
}
PADS = [PadSpec(group=g, pad=p, slot=SLOTS[name], **kw) for (g, p), (name, kw) in LAYOUT.items()]
ROOT = {(p.group, p.pad): p.root for p in PADS}


from ko_bridge.groove import snap_step_to_clock


def hit(group, bar, step, pad, vel=100, semis=0, steps=1.0) -> NoteEvent:
    return NoteEvent(pad=pad, step=snap_step_to_clock(bar * S + step), note=ROOT[(group, pad)] + semis, velocity=vel, duration=max(1, min(384, int(steps * 24))))


def tone(bar, step, pad, name, steps, vel=100) -> NoteEvent:
    return NoteEvent(pad=pad, step=snap_step_to_clock(bar * S + step), concert=name, velocity=vel, duration=max(1, min(384, int(steps * 24))))


def loop(bar, pad, vel=112) -> NoteEvent:      # a one-bar bass loop on the one, native pitch
    return hit("B", bar, 0, pad, vel, 0, 16)


def chord(bar, step, names, steps, vel=90, pads=("6", "7", "8")) -> list[NoteEvent]:
    """Place chord tones on exact 24-PPQN MIDI clock boundaries (0 or +4 ticks) to avoid mid-clock interpolation jitter."""
    return [tone(bar, step + (k // 2) * (4 / 24), pad, name, steps, vel) for k, (pad, name) in enumerate(zip(pads, names))]


EM9 = ("E3", "G3", "B3"); CMAJ7 = ("C3", "E3", "B3"); AM7 = ("A2", "C3", "G3"); BM7 = ("B2", "D3", "A3")   # voiced near the pad's C3


# ----------------------------------------------------------------------------- drums

def break_bar(e, b, variant=0, hats=True, vel=1.0):
    v = lambda x: max(1, min(127, int(x * vel)))
    e += [hit("A", b, 0, "1", v(124)), hit("A", b, 0, "2", v(112))]
    kicks = {0: [10], 1: [10, 14], 2: [9], 3: [10, 13]}[variant % 4]
    e += [hit("A", b, st, "1", v(108)) for st in kicks]
    e += [hit("A", b, 4, "3", v(122)), hit("A", b, 12, "3", v(124)), hit("A", b, 4, "4", v(96)), hit("A", b, 12, "4", v(100))]
    ghosts = {0: [(7, 56), (11, 62), (15, 52)], 1: [(2, 48), (7, 60), (15, 56)], 2: [(7, 58), (13, 64), (15, 50)], 3: [(1, 46), (7, 60), (11, 64), (15, 70)]}[variant % 4]
    e += [hit("A", b, st, "9", v(gv)) for st, gv in ghosts[:2]]
    if hats:
        for st in range(S):
            if st in (6, 14) and variant % 2 == 1:
                e.append(hit("A", b, st, "6", v(96)))
            else:
                e.append(hit("A", b, st, "5", v([100, 62, 84, 62][st % 4])))
        e += [hit("A", b, st, "8", v(74)) for st in (2, 10)]


def drop_drums(number: int, second: bool = False) -> PatternSpec:
    e = []
    for b in range(BARS):
        break_bar(e, b, variant=b, vel=1.0)
        if second and b in (1, 5):
            e.append(hit("A", b, 2, "1", 104))
    e += [hit("A", 3, st, "3", vv) for st, vv in ((13, 86), (14, 100), (15, 116))]
    e += [hit("A", 7, st, ".", vv) for st, vv in ((12, 100), (13, 106), (14, 112), (15, 118))]
    e += [hit("A", 7, st, "3", vv) for st, vv in ((14, 104), (15, 120))]
    e += [hit("A", 0, 0, "7", 90)]
    if second:
        e += [hit("A", b, 15, "0", 90) for b in (1, 5)]
    return PatternSpec(group="A", number=number, bars=BARS, events=e, lanes=[Lane(parameter="level", points=[LanePoint(step=0, value=0.8)])])


def intro_drums() -> PatternSpec:
    e = []
    for b in range(BARS):
        if b >= 4:
            e += [hit("A", b, st, "5", [70, 44, 58, 44][st % 4]) for st in range(0, S, 2)]
            e += [hit("A", b, st, "8", 60) for st in (2, 6, 10, 14)]
        e += [hit("A", b, 7, "9", 44), hit("A", b, 15, "9", 48)]
    e += [hit("A", 7, 12, "3", 80), hit("A", 7, 14, "3", 96), hit("A", 7, 15, "3", 110)]
    return PatternSpec(group="A", number=2, bars=BARS, events=e, lanes=[Lane(parameter="level", points=[LanePoint(step=0, value=0.75), LanePoint(step=4 * S, value=0.9)])])


def groove_drums() -> PatternSpec:
    e = []
    for b in range(BARS):
        e += [hit("A", b, 0, "1", 112), hit("A", b, 4, "3", 112), hit("A", b, 12, "3", 114), hit("A", b, 10, "1", 100)]
        e += [hit("A", b, st, "5", [92, 56, 76, 56][st % 4]) for st in range(S)]
        e += [hit("A", b, 7, "9", 54), hit("A", b, 15, "9", 50)]
        if b % 2 == 1:
            e.append(hit("A", b, 14, "6", 90))
    e += [hit("A", 7, st, "3", vv) for st, vv in ((12, 84), (13, 92), (14, 104), (15, 118))]
    e += [hit("A", 7, st, ".", vv) for st, vv in ((8, 96), (10, 104))]
    return PatternSpec(group="A", number=3, bars=BARS, events=e, lanes=[Lane(parameter="level", points=[LanePoint(step=0, value=0.8)])])


def breakdown_drums() -> PatternSpec:
    e = []
    for b in range(BARS):
        if b < 6:
            e += [hit("A", b, 4, "4", 70), hit("A", b, 12, "4", 74), hit("A", b, 7, "9", 46), hit("A", b, 15, "9", 50)]
            e += [hit("A", b, st, "7", 64) for st in (0, 8)]
        if b in (6,):                                         # roll into the silence
            e += [hit("A", b, st, "3", 70 + 4 * st) for st in range(0, 12)]
            e += [hit("A", b, st, "5", 90) for st in range(0, 12)]
        # bar 7: one beat of nothing at the start (steps 0-3), then a pickup
        if b == 7:
            e += [hit("A", b, 12, "1", 110), hit("A", b, 14, "3", 112), hit("A", b, 15, "3", 124)]
    return PatternSpec(group="A", number=4, bars=BARS, events=e, lanes=[Lane(parameter="level", points=[LanePoint(step=0, value=0.8)])])


def outro_drums() -> PatternSpec:
    e = []
    for b in range(BARS):
        if b < 4:
            break_bar(e, b, variant=b, hats=(b < 2), vel=1.0 - 0.08 * b)
        elif b < 6:
            e += [hit("A", b, 0, "1", 96), hit("A", b, 12, "3", 96)] + [hit("A", b, st, "5", 60) for st in range(0, S, 4)]
        e += [hit("A", b, 7, "9", 44)]
    return PatternSpec(group="A", number=6, bars=BARS, events=e,
                       lanes=[Lane(parameter="level", points=[LanePoint(step=0, value=1.0), LanePoint(step=4 * S, value=0.7), LanePoint(step=8 * S - 1, value=0.3)])])


# ----------------------------------------------------------------------------- bass

def drop_bass() -> PatternSpec:
    e = []
    plan = {0: "1", 1: "1", 2: "6", 3: "3", 4: "1", 5: "2", 6: "7", 7: "5"}        # E wub8, E, G, E 1/4, E, E 1/16, A, E stutter
    for b, pad in plan.items():
        e.append(loop(b, pad))
    e += [hit("B", 1, 12, ".", 108, 0, 4), hit("B", 3, 12, "0", 104, 0, 4), hit("B", 5, 12, ".", 108, 0, 4), hit("B", 7, 8, "enter", 110, 0, 8)]
    return PatternSpec(group="B", number=1, bars=BARS, events=e, lanes=[Lane(parameter="level", points=[LanePoint(step=0, value=0.8)]),
                                                                        Lane(parameter="lowpass", points=[LanePoint(step=0, value=1.0)])])


def drop2_bass() -> PatternSpec:
    e = []
    plan = {0: "2", 1: "5", 2: "6", 3: "4", 4: "2", 5: "5", 6: "8", 7: "9"}        # 1/16, stutter, G, triplets, 1/16, stutter, B, D
    for b, pad in plan.items():
        e.append(loop(b, pad))
    e += [hit("B", b, 12, ".", 108, 0, 4) for b in (1, 5)] + [hit("B", 3, 8, "enter", 110, 0, 8), hit("B", 7, 12, "0", 106, 0, 4)]
    return PatternSpec(group="B", number=2, bars=BARS, events=e, lanes=[Lane(parameter="level", points=[LanePoint(step=0, value=0.8)]),
                                                                        Lane(parameter="lowpass", points=[LanePoint(step=0, value=1.0)])])


def quiet_bass(number: int) -> PatternSpec:    # intro / breakdown / outro: no reese; the sub lives in group C
    return PatternSpec(group="B", number=number, bars=BARS, events=[], lanes=[Lane(parameter="level", points=[LanePoint(step=0, value=1.0)])])


# ----------------------------------------------------------------------------- keys, pads, sub

def intro_keys() -> PatternSpec:
    e = []
    e += chord(0, 0, EM9, 16, 70) + chord(2, 0, CMAJ7, 16, 70) + chord(4, 0, AM7, 16, 76) + chord(6, 0, BM7, 16, 78)
    e += [tone(1, 4 / 24, "9", "E3", 16, 60), tone(5, 4 / 24, "9", "A2", 16, 64), tone(5, 5 / 24, ".", "E3", 16, 60)]
    bells = [(0, 8, -8), (1, 4, -5), (1, 12, -1), (3, 0, -8), (3, 10, -3), (5, 8, -8), (6, 4, -5), (6, 12, 2), (7, 6, -1)]   # semitones from C5: E4 G4 B4 ...
    e += [hit("C", b, st, "5", 78, semis, 4) for b, st, semis in bells]
    lanes = [Lane(parameter="lowpass", points=[LanePoint(step=0, value=0.55), LanePoint(step=4 * S, value=0.75), LanePoint(step=8 * S - 1, value=1.0)]),
             Lane(parameter="level", points=[LanePoint(step=0, value=0.9)]), Lane(parameter="fx", points=[LanePoint(step=0, value=0.6)])]
    return PatternSpec(group="C", number=1, bars=BARS, events=e, lanes=lanes)


def groove_keys() -> PatternSpec:
    e = [tone(b, 0, "1", n, 16, 104) for b, n in zip(range(BARS), ("E1", "E1", "C1", "C1", "A1", "A1", "B1", "B1"))]   # sub follows the chords
    e += chord(0, 0, EM9, 16, 66) + chord(2, 0, CMAJ7, 16, 66) + chord(4, 0, AM7, 16, 70) + chord(6, 0, BM7, 16, 72)
    ep = [(0, 8, "B4", 2), (0, 10, "E5", 2), (0, 12, "G4", 4), (1, 4, "F#4", 4), (1, 8, "E4", 8),
          (2, 8, "G4", 2), (2, 10, "C5", 2), (2, 12, "B4", 4), (3, 4, "A4", 4), (3, 8, "G4", 8),
          (4, 8, "E4", 2), (4, 10, "A4", 2), (4, 12, "C5", 4), (5, 4, "B4", 4), (5, 8, "A4", 8),
          (6, 8, "F#4", 2), (6, 10, "B4", 2), (6, 12, "D5", 4), (7, 0, "B4", 8)]
    e += [tone(b, st, "3", n, ln, 96) for b, st, n, ln in ep]
    e += [hit("C", b, st, "4", 84, semis, 1) for b in (1, 3, 5, 7) for st, semis in ((14, 7), (15, 10))]     # string pluck pickups (A3 -> E4, G4)
    lanes = [Lane(parameter="lowpass", points=[LanePoint(step=0, value=1.0)]), Lane(parameter="level", points=[LanePoint(step=0, value=0.95)]),
             Lane(parameter="fx", points=[LanePoint(step=0, value=0.45)])]
    return PatternSpec(group="C", number=2, bars=BARS, events=e, lanes=lanes)


def drop_keys(number: int = 3) -> PatternSpec:
    e = chord(0, 0, EM9, 8, 72) + chord(2, 0, CMAJ7, 8, 72) + chord(4, 0, AM7, 8, 74) + chord(6, 0, BM7, 8, 74)
    e += [hit("C", b, 14, "4", 90, 7 if b % 2 else 10, 1) for b in range(BARS)]
    e += [tone(3, 12, "5", "E5", 4, 80), tone(7, 12, "5", "B4", 4, 80)]
    lanes = [Lane(parameter="lowpass", points=[LanePoint(step=0, value=1.0)]), Lane(parameter="level", points=[LanePoint(step=0, value=0.7)]),
             Lane(parameter="fx", points=[LanePoint(step=0, value=0.35)])]
    return PatternSpec(group="C", number=number, bars=BARS, events=e, lanes=lanes)


MELODY = [(0, 0, "E4", 3), (0, 3, "G4", 3), (0, 6, "B4", 2), (0, 8, "A4", 4), (0, 12, "G4", 4), (1, 0, "F#4", 6), (1, 6, "E4", 2), (1, 8, "D4", 8),
          (2, 0, "C4", 3), (2, 3, "E4", 3), (2, 6, "G4", 2), (2, 8, "B4", 4), (2, 12, "C5", 4), (3, 0, "B4", 6), (3, 6, "A4", 2), (3, 8, "G4", 8),
          (4, 0, "A4", 3), (4, 3, "C5", 3), (4, 6, "E5", 2), (4, 8, "D5", 4), (4, 12, "C5", 4), (5, 0, "B4", 6), (5, 6, "A4", 2), (5, 8, "G4", 8),
          (6, 0, "F#4", 4), (6, 4, "A4", 4), (6, 8, "B4", 4), (6, 12, "D5", 4)]
ARP = ["E4", "G4", "B4", "E5", "B4", "G4"]


def breakdown_keys() -> PatternSpec:
    e = [tone(b, 0, "1", n, 16, 100) for b, n in zip(range(6), ("E1", "E1", "C1", "C1", "A1", "B1"))]
    e += chord(0, 0, EM9[:2], 16, 74, pads=("6", "7")) + chord(2, 0, CMAJ7[:2], 16, 74, pads=("6", "7")) + chord(4, 0, AM7[:2], 16, 76, pads=("6", "7")) + chord(5, 0, BM7[:2], 16, 78, pads=("6", "7"))
    e += [tone(b, 4 / 24, "9", n, 16, 68) for b, n in ((0, "E3"), (2, "E3"), (4, "A2"))]
    e += [tone(b, st, "3", n, ln, 100) for b, st, n, ln in MELODY]
    for b in range(6):
        e += [tone(b, st + 2 / 24, "4", ARP[(st // 2 + b) % len(ARP)], 1, 72 + (st % 4 == 0) * 14) for st in range(0, S, 2)]
    e += [tone(6, 0, "3", "E4", 16, 96), tone(6, 1 / 24, "6", "E3", 16, 70), tone(6, 2 / 24, "7", "B3", 16, 70)]
    lanes = [Lane(parameter="lowpass", points=[LanePoint(step=0, value=1.0)]), Lane(parameter="level", points=[LanePoint(step=0, value=1.0), LanePoint(step=6 * S, value=1.0), LanePoint(step=7 * S, value=0.0)]),
             Lane(parameter="fx", points=[LanePoint(step=0, value=0.55), LanePoint(step=6 * S, value=0.8)])]
    return PatternSpec(group="C", number=4, bars=BARS, events=e, lanes=lanes)


def outro_keys() -> PatternSpec:
    e = chord(0, 0, EM9, 16, 72) + chord(2, 0, CMAJ7, 16, 70) + chord(4, 0, AM7, 16, 68) + chord(6, 0, EM9, 16, 66)
    e += [tone(b, 0, "1", "E1", 16, 90) for b in (0, 2)]
    e += [hit("C", b, st, "5", 74, semis, 4) for b, st, semis in ((1, 8, -8), (3, 4, -5), (5, 8, -1), (7, 0, -8))]
    e += [tone(4, 4 / 24, "9", "E3", 16, 60), tone(6, 4 / 24, "9", "E3", 16, 56)]
    lanes = [Lane(parameter="lowpass", points=[LanePoint(step=0, value=1.0), LanePoint(step=4 * S, value=0.8), LanePoint(step=8 * S - 1, value=0.4)]),
             Lane(parameter="level", points=[LanePoint(step=0, value=0.9), LanePoint(step=6 * S, value=0.8), LanePoint(step=8 * S - 1, value=0.5)]),
             Lane(parameter="fx", points=[LanePoint(step=0, value=0.6)])]
    return PatternSpec(group="C", number=5, bars=BARS, events=e, lanes=lanes)


# ----------------------------------------------------------------------------- colour

def colour(number: int, crackle_bars: list[int], riser_bar: int | None, impact: bool, vox: list[tuple[int, int, str]]) -> PatternSpec:
    e = [hit("D", b, 6 / 24, "4", 70, 0, 16) for b in crackle_bars]
    if riser_bar is not None:
        e.append(hit("D", riser_bar, 0, "2", 100, 0, 16))
    if impact:
        e.append(hit("D", 0, 0, "3", 110))
    e += [tone(b, st, "1", n, 2, 88) for b, st, n in vox]
    return PatternSpec(group="D", number=number, bars=BARS, events=e, lanes=[Lane(parameter="fx", points=[LanePoint(step=0, value=0.5)])])


def spec() -> ProjectSpec:
    patterns = [drop_drums(1), drop_bass(), drop_keys(3), colour(1, [], None, True, [(3, 14, "E4"), (7, 14, "B4")]),
                intro_drums(), quiet_bass(3), intro_keys(), colour(2, [0, 4], 6, False, [(3, 14, "E4")]),
                groove_drums(), quiet_bass(4), groove_keys(), colour(3, [], 6, False, [(1, 14, "E4"), (5, 14, "G4")]),
                breakdown_drums(), quiet_bass(5), breakdown_keys(), colour(4, [], 5, False, [(3, 14, "E4"), (7, 12, "B4")]),
                drop_drums(5, second=True), drop2_bass(), drop_keys(6), colour(5, [], None, True, [(1, 14, "E4"), (5, 14, "B4"), (7, 12, "E5")]),
                outro_drums(), quiet_bass(6), outro_keys(), colour(6, [0, 4], None, False, [(3, 14, "E4")])]
    scenes = [SceneSpec(patterns={"A": 2, "B": 3, "C": 1, "D": 2}),   # 1 intro
              SceneSpec(patterns={"A": 3, "B": 4, "C": 2, "D": 3}),   # 2 groove
              SceneSpec(patterns={"A": 1, "B": 1, "C": 3, "D": 1}),   # 3 drop
              SceneSpec(patterns={"A": 4, "B": 5, "C": 4, "D": 4}),   # 4 breakdown
              SceneSpec(patterns={"A": 5, "B": 2, "C": 6, "D": 5}),   # 5 drop 2
              SceneSpec(patterns={"A": 6, "B": 6, "C": 5, "D": 6})]   # 6 outro
    return ProjectSpec(bpm=BPM, pads=PADS, patterns=patterns, scenes=scenes, song=[1, 2, 3, 4, 5, 6], current_scene=3,
                       fader_bases=[FaderBase(group="A", parameter="fx", value=0.1), FaderBase(group="B", parameter="fx", value=0.0),
                                    FaderBase(group="C", parameter="fx", value=0.45), FaderBase(group="D", parameter="fx", value=0.5)],
                       fx=FxSpec(type="reverb", x=0.55, y=0.5, output_compressor_drive=0.22, output_compressor_speed=0.5,
                                 sidechain_sources=[SidechainSource(group="A", pad="1"), SidechainSource(group="A", pad="2")],
                                 sidechain_destinations=["B", "C"], sidechain_length=0.3, sidechain_shape=0.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=int, default=9)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    s = spec()
    records = {p.id: len(p.events) + sum(len(l.points) for l in p.lanes) for p in s.patterns}
    print("records per pattern:", records, "max", max(records.values()))
    if args.no_write:
        return 0
    w = srv.write_project(args.project, s, wait=True)
    print(json.dumps({k: w[k] for k in ("project", "bytes", "write_seconds", "identical", "reload_seconds")}, indent=1))
    after = w["project_after"]
    print("scenes:", [(sc["scene"], sc["patterns"]) for sc in after["scenes"]], "| song", after["song"], "| cursor", after["scene_cursor"], "| bpm", after["bpm"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
