#!/usr/bin/env python3
"""Riddim at 140 BPM in F minor: sampled factory drums and percussion, Tyler's megaman hits, vocal shots, and synthesized
riddim basses baked at tempo (scripts/synth.py riddim set). Six 8-bar scenes as a song.

    uv run python scripts/riddim.py --project 9

Lessons applied: every event sits on a MIDI-clock multiple (4 ticks) so the device never has to guess between clock
pulses (swing is +8 ticks, triplets 16/32 ticks); velocities are humanized; drops stay under ten voices; dynamics come
from level lanes, density and silence; two-bar bass loops retrigger every two bars at native pitch.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

from ko_bridge import server as srv
from ko_bridge.project import FaderBase, FxSpec, Lane, LanePoint, NoteEvent, PadSpec, PatternSpec, ProjectSpec, SceneSpec, SidechainSource

BPM = 140.0
S = 16
BARS = 8
SET = os.path.expanduser("~/Music/ko-bridge/library/sets/riddim-140")
RNG = random.Random(140)


def _slots() -> dict[str, int]:
    path = os.path.join(SET, "slots.json")
    if os.path.exists(path):
        return json.load(open(path))["slots"]
    manifest = json.load(open(os.path.join(SET, "manifest.json")))["items"]
    return {item["name"]: 640 + i for i, item in enumerate(manifest)}


SLOTS = _slots()
SW = 1 / 3          # swing: off-16ths land 8 ticks (two MIDI clocks) late
T16 = 2 / 3         # a 16th-note triplet in steps (16 ticks)

LAYOUT = {
    # A: drums (factory, layered)
    ("A", "1"): dict(slot=21, amplitude=104),                 # nt alt kick: the weight
    ("A", "2"): dict(slot=10, amplitude=100),                 # kick bump: the punch
    ("A", "3"): dict(slot=106, amplitude=104),                # snare 6x6 1: the big snare on 3
    ("A", "4"): dict(slot=309, amplitude=92),                 # clap sp disco b (layer)
    ("A", "5"): dict(slot=114, amplitude=88),                 # nt snare alt (stereo layer)
    ("A", "6"): dict(slot=210, amplitude=92, choke=True),     # hat closed 6x6
    ("A", "7"): dict(slot=217, amplitude=90, choke=True),     # crunch hat 2
    ("A", "8"): dict(slot=139, amplitude=96),                 # rimshot dmx
    ("A", "9"): dict(slot=339, amplitude=84),                 # cowbell
    ("A", "."): dict(slot=251, amplitude=80),                 # splash
    ("A", "0"): dict(slot=752, amplitude=92),                 # megaman_beep_hit (Tyler's)
    ("A", "enter"): dict(slot=748, amplitude=86),             # megaman_cybal (Tyler's)
    # B: basses
    ("B", "1"): dict(slot=SLOTS["riddim A"], amplitude=100), ("B", "2"): dict(slot=SLOTS["riddim B"], amplitude=100),
    ("B", "3"): dict(slot=SLOTS["riddim fill"], amplitude=100), ("B", "4"): dict(slot=SLOTS["riddim C"], amplitude=100),
    ("B", "5"): dict(slot=SLOTS["wonk F1"], amplitude=98), ("B", "6"): dict(slot=SLOTS["laser"], amplitude=90), ("B", "7"): dict(slot=SLOTS["yoi F1"], amplitude=98),
    ("B", "8"): dict(slot=SLOTS["sub drop 1 bar"], amplitude=100),
    ("B", "9"): dict(slot=426, amplitude=96, play_mode="key", release=110),    # thirty seven sub (B1)
    ("B", "."): dict(slot=749, amplitude=94),                                   # boop_bass_megman (Tyler's)
    # C: atmosphere and melody (factory)
    ("C", "1"): dict(slot=513, amplitude=88, play_mode="key", release=150),    # scary vibes (E4)
    ("C", "2"): dict(slot=549, amplitude=90),                                   # sitar one shot (C5)
    ("C", "3"): dict(slot=530, amplitude=84, play_mode="key", release=200),    # cello 360 (F2): the drone
    ("C", "4"): dict(slot=448, amplitude=86),                                   # kaa wa bell air (E2)
    ("C", "5"): dict(slot=526, amplitude=88),                                   # dr organ chord (B1)
    ("C", "6"): dict(slot=541, amplitude=74, play_mode="key", release=200),    # strings dr (C4)
    ("C", "7"): dict(slot=246, amplitude=80),                                   # toaster bell
    # D: colour
    ("D", "1"): dict(slot=553, amplitude=94),                                   # vox s6i2 eh (F3, native in key)
    ("D", "2"): dict(slot=548, amplitude=90),                                   # vox s6i2 dark (D#3)
    ("D", "3"): dict(slot=509, amplitude=84),                                   # bg vocal (C5)
    ("D", "4"): dict(slot=751, amplitude=88),                                   # megaman_beep (Tyler's)
    ("D", "5"): dict(slot=SLOTS["reverse crash 1 bar"], amplitude=92), ("D", "6"): dict(slot=SLOTS["riser 1 bar"], amplitude=92),
}
PADS = [PadSpec(group=g, pad=p, **kw) for (g, p), kw in LAYOUT.items()]
ROOT = {(p.group, p.pad): p.root for p in PADS}


def hv(v: int, spread: int = 8) -> int:
    return max(1, min(127, v + RNG.randint(-spread, spread)))


def hit(group, bar, step, pad, vel=100, semis=0, steps=1.0, humanize=True) -> NoteEvent:
    return NoteEvent(pad=pad, step=bar * S + step, note=ROOT[(group, pad)] + semis, velocity=hv(vel) if humanize else vel, duration=max(1, min(384, int(steps * 24))))


def tone(bar, step, pad, name, steps, vel=100) -> NoteEvent:
    return NoteEvent(pad=pad, step=bar * S + step, concert=name, velocity=hv(vel, 4), duration=max(1, min(384, int(steps * 24))))


def loop2(bar, pad, vel=112) -> NoteEvent:     # a two-bar bass loop on the one, native pitch, held two bars (two events of a bar each)
    return hit("B", bar, 0, pad, vel, 0, 16, humanize=False)


# ----------------------------------------------------------------------------- drums

def half_time(e, b, full=True, fill=False, vel=1.0):
    v = lambda x: int(x * vel)
    e += [hit("A", b, 0, "1", v(124)), hit("A", b, 0, "2", v(116))]
    if full and b % 2 == 1:
        e += [hit("A", b, 10 + SW, "2", v(104))]                     # a swung pickup kick
    e += [hit("A", b, 8, "3", v(126)), hit("A", b, 8, "4", v(104)), hit("A", b, 8, "5", v(96))]
    if full:
        e += [hit("A", b, st + (SW if st % 2 else 0), "6", v(96 if st % 4 == 0 else 70)) for st in (2, 6, 10, 14)]   # swung off-beat hats
        e += [hit("A", b, st, "7", v(60)) for st in (4, 12)]
        e += [hit("A", b, 5 + SW, "8", v(62)), hit("A", b, 13 + SW, "8", v(58))]
    if fill:
        e += [hit("A", b, 12 + k * T16, "3", v(84 + 7 * k)) for k in range(6)]     # 16th-triplet snare roll
    return e


def drop_drums(number: int, variant: int) -> PatternSpec:
    e = []
    for b in range(BARS):
        half_time(e, b, full=True, fill=(b in (3, 7)))
        if variant >= 1 and b % 2 == 1:
            e += [hit("A", b, 14, "9", 86)]                           # cowbell answer
        if variant >= 2 and b % 4 == 2:
            e += [hit("A", b, 6 + T16 * k, "0", 90) for k in range(3)]     # megaman beep hit triplet
    e.append(hit("A", 0, 0, ".", 96))
    if variant >= 1:
        e.append(hit("A", 4, 0, "enter", 90))
    return PatternSpec(group="A", number=number, bars=BARS, events=e, lanes=[Lane(parameter="level", points=[LanePoint(step=0, value=0.85)])])


def intro_drums() -> PatternSpec:
    e = []
    for b in range(BARS):
        e += [hit("A", b, 8, "8", 58), hit("A", b, 14 + SW, "6", 54)]
        if b >= 4:
            e += [hit("A", b, 4 + T16 * k, "enter", 64) for k in range(2)]
    e += [hit("A", 7, 8 + k * T16, "3", 70 + 8 * k) for k in range(6)]
    return PatternSpec(group="A", number=2, bars=BARS, events=e, lanes=[Lane(parameter="level", points=[LanePoint(step=0, value=0.7)])])


def build_drums() -> PatternSpec:
    e = []
    for b in range(BARS):
        half_time(e, b, full=(b >= 4), fill=(b == 7))
        if b < 4:
            e += [hit("A", b, 6 + SW, "6", 72), hit("A", b, 14 + SW, "6", 72)]
    e += [hit("A", 7, 0, "0", 92), hit("A", 7, 4, "0", 92)]
    return PatternSpec(group="A", number=3, bars=BARS, events=e,
                       lanes=[Lane(parameter="level", points=[LanePoint(step=0, value=0.75), LanePoint(step=4 * S, value=0.85), LanePoint(step=8 * S - 1, value=0.95)])])


def break_drums() -> PatternSpec:
    e = []
    for b in range(BARS):
        if b < 6:
            e += [hit("A", b, 8, "4", 80), hit("A", b, 8, "5", 76), hit("A", b, 13 + SW, "8", 54)]
        if b == 6:
            e += [hit("A", b, k * T16, "3", 72 + 3 * k) for k in range(12)]       # roll over beats 1-2
        # bar 7: a beat of silence (steps 0-3), then the pickup
        if b == 7:
            e += [hit("A", b, 8, "1", 110), hit("A", b, 8, "2", 104)] + [hit("A", b, 12 + k * T16, "3", 96 + 8 * k) for k in range(6)]
    return PatternSpec(group="A", number=4, bars=BARS, events=e, lanes=[Lane(parameter="level", points=[LanePoint(step=0, value=0.75)])])


# ----------------------------------------------------------------------------- bass

def drop_bass(number: int, plan: list[str], stabs: list[tuple], subs: bool = True, drop_bar: int | None = None) -> PatternSpec:
    e = []
    for b in range(0, BARS, 2):
        e.append(loop2(b, plan[(b // 2) % len(plan)]))      # a two-bar loop started on odd bars runs through the next bar
    for b, st, pad, name, ln in stabs:
        e.append(tone(b, st, pad, name, ln, 108) if name else hit("B", b, st, pad, 108, 0, ln))
    if subs:
        e += [tone(b, 0, "9", "F1", 16, 100) for b in range(BARS) if b % 2 == 0]
    if drop_bar is not None:
        e.append(hit("B", drop_bar, 0, "8", 110, 0, 16))
    lanes = [Lane(parameter="level", points=[LanePoint(step=0, value=0.85)]), Lane(parameter="lowpass", points=[LanePoint(step=0, value=1.0)])]
    return PatternSpec(group="B", number=number, bars=BARS, events=e, lanes=lanes)


def build_bass() -> PatternSpec:
    e = [loop2(b, "1", 104) for b in (0, 2, 4, 6)] + [tone(b, 0, "9", "F1", 16, 96) for b in (4, 6)]
    lanes = [Lane(parameter="level", points=[LanePoint(step=0, value=0.8)]),
             Lane(parameter="lowpass", points=[LanePoint(step=0, value=0.3), LanePoint(step=4 * S, value=0.55), LanePoint(step=8 * S - 1, value=1.0)])]
    return PatternSpec(group="B", number=5, bars=BARS, events=e, lanes=lanes)


def quiet_bass(number: int, subs: list[int]) -> PatternSpec:
    e = [tone(b, 0, "9", "F1", 16, 96) for b in subs]
    return PatternSpec(group="B", number=number, bars=BARS, events=e, lanes=[Lane(parameter="level", points=[LanePoint(step=0, value=0.85)]), Lane(parameter="lowpass", points=[LanePoint(step=0, value=1.0)])])


# ----------------------------------------------------------------------------- atmosphere and melody

SITAR = [(0, 0, "F4", 2), (0, 2, "G#4", 2), (0, 4, "C5", 4), (1, 0, "A#4", 2), (1, 2, "G#4", 2), (1, 4, "F4", 4),
         (4, 0, "F4", 2), (4, 2, "G#4", 2), (4, 4, "C5", 2), (4, 6, "C#5", 2), (5, 0, "C5", 4), (5, 4, "G#4", 4)]
VIBES = [(2, 0, "F4", 6), (2, 6, "G#4", 2), (2, 8, "G4", 8), (6, 0, "F4", 6), (6, 6, "C5", 2), (6, 8, "A#4", 8)]


def intro_keys() -> PatternSpec:
    e = [tone(b, 0, "3", "F2", 16, 72 + 2 * b) for b in range(BARS)]                      # cello drone swells
    e += [tone(b, st, "2", n, ln, 86) for b, st, n, ln in SITAR]
    e += [tone(b, st, "1", n, ln, 80) for b, st, n, ln in VIBES]
    e += [hit("C", b, 12 + SW, "4", 70, 1) for b in (1, 3, 5)]                               # bell air on F (+1 from E)
    lanes = [Lane(parameter="lowpass", points=[LanePoint(step=0, value=0.6), LanePoint(step=8 * S - 1, value=1.0)]),
             Lane(parameter="level", points=[LanePoint(step=0, value=0.85)]), Lane(parameter="fx", points=[LanePoint(step=0, value=0.6)])]
    return PatternSpec(group="C", number=1, bars=BARS, events=e, lanes=lanes)


def build_keys() -> PatternSpec:
    e = [tone(b, 0, "3", "F2", 16, 78) for b in range(0, BARS, 2)]
    e += [tone(b, st, "2", n, ln, 84) for b, st, n, ln in SITAR if b < 2] + [tone(b + 4, st, "2", n, ln, 88) for b, st, n, ln in SITAR if b < 2]
    e += [hit("C", b, 8, "5", 88, 6, 2) for b in (1, 3, 5)] + [hit("C", 7, 8, "5", 92, 6, 2), hit("C", 7, 12, "5", 96, 9, 2)]   # organ stabs on F, then G#
    e += [tone(b, 0, "6", "F3", 16, 56 + 3 * b) for b in range(4, 8)]
    lanes = [Lane(parameter="lowpass", points=[LanePoint(step=0, value=0.8), LanePoint(step=8 * S - 1, value=1.0)]), Lane(parameter="level", points=[LanePoint(step=0, value=0.8)]),
             Lane(parameter="fx", points=[LanePoint(step=0, value=0.5)])]
    return PatternSpec(group="C", number=2, bars=BARS, events=e, lanes=lanes)


def drop_keys(number: int, stabs_on: tuple = (3, 7)) -> PatternSpec:
    e = [hit("C", b, 14 + SW, "5", 90, 6, 1) for b in stabs_on] + [hit("C", b, 12, "7", 78, 0, 2) for b in (1, 5)]
    e += [tone(7, 8 + k * T16, "2", n, 1, 86) for k, n in enumerate(("F4", "G#4", "C5"))]
    lanes = [Lane(parameter="lowpass", points=[LanePoint(step=0, value=1.0)]), Lane(parameter="level", points=[LanePoint(step=0, value=0.75)]),
             Lane(parameter="fx", points=[LanePoint(step=0, value=0.4)])]
    return PatternSpec(group="C", number=number, bars=BARS, events=e, lanes=lanes)


def break_keys() -> PatternSpec:
    e = [tone(b, 0, "3", "F2", 16, 76) for b in range(0, 6, 2)] + [tone(b, 0, "6", "F3", 16, 66) for b in (0, 2, 4)]
    e += [tone(b, st, "2", n, ln, 88) for b, st, n, ln in SITAR]
    e += [tone(b, st, "1", n, ln, 84) for b, st, n, ln in VIBES if b == 2] + [tone(4, 0, "1", "G#4", 16, 80), tone(5, 0, "1", "F4", 16, 80)]
    lanes = [Lane(parameter="lowpass", points=[LanePoint(step=0, value=1.0)]), Lane(parameter="level", points=[LanePoint(step=0, value=0.9), LanePoint(step=6 * S + 12, value=0.9), LanePoint(step=7 * S, value=0.0), LanePoint(step=7 * S + 8, value=0.9)]),
             Lane(parameter="fx", points=[LanePoint(step=0, value=0.6)])]
    return PatternSpec(group="C", number=4, bars=BARS, events=e, lanes=lanes)


# ----------------------------------------------------------------------------- colour

def colour(number: int, eh: list[tuple], dark: list[tuple], bg: list[tuple], beeps: list[tuple], reverse_bars: list[int], riser_bars: list[int]) -> PatternSpec:
    e = [tone(b, st, "1", "F3", 2, 92) for b, st in eh] + [tone(b, st, "2", "F3", 2, 88) for b, st in dark] + [tone(b, st, "3", "F4", 8, 82) for b, st in bg]
    e += [hit("D", b, st, "4", 84, 0, 1) for b, st in beeps]
    e += [hit("D", b, 0, "5", 90, 0, 16) for b in reverse_bars] + [hit("D", b, 0, "6", 92, 0, 16) for b in riser_bars]
    return PatternSpec(group="D", number=number, bars=BARS, events=e, lanes=[Lane(parameter="fx", points=[LanePoint(step=0, value=0.5)])])


def spec() -> ProjectSpec:
    patterns = [
        # scene 3: drop A — riff A, A, A, fill
        drop_drums(1, 0), drop_bass(1, ["1", "1", "1", "3"], [(1, 14 + SW, "6", None, 1), (5, 14 + SW, "6", None, 1), (3, 12, "7", "F1", 2)], subs=True),
        drop_keys(3), colour(1, [(3, 14), (7, 14)], [], [], [(1, 15), (5, 15)], [], []),
        # scene 1: intro
        intro_drums(), quiet_bass(6, [4, 6]), intro_keys(), colour(2, [(3, 8)], [(5, 12)], [(6, 0)], [(1, 6), (1, 6 + T16), (5, 6)], [], [6, 7]),
        # scene 2: build
        build_drums(), build_bass(), build_keys(), colour(3, [(1, 14), (5, 14)], [], [(3, 8)], [(2, 10), (6, 10), (6, 10 + T16)], [7], []),
        # scene 4: drop B — riff B with wonk stabs in the gaps, sub drop at the end
        drop_drums(5, 1), drop_bass(2, ["2", "2", "2", "3"], [(0, 12, "5", "F2", 1), (1, 4 + SW, "5", "G#1", 1), (2, 12, "5", "A#1", 1), (4, 12, "5", "F2", 1), (5, 4 + SW, "5", "C2", 1), (6, 12, "5", "G#1", 1), (3, 12, "7", "F1", 2)], subs=True, drop_bar=7),
        drop_keys(5, (1, 5)), colour(4, [(1, 14), (5, 14)], [(3, 12)], [], [(2, 15), (6, 15)], [], []),
        # scene 5: break
        break_drums(), quiet_bass(7, [0, 2, 4]), break_keys(), colour(5, [(1, 8)], [], [(2, 0), (5, 0)], [(3, 6), (3, 6 + T16), (3, 6 + 2 * T16)], [6], [7]),
        # scene 6: drop C — riff C and A alternating, yoi, beeps, cowbell
        drop_drums(6, 2), drop_bass(3, ["4", "1", "4", "3"], [(1, 12, "7", "F1", 2), (3, 14 + SW, "6", None, 1), (5, 12, "7", "G#1", 2), (6, 14 + SW, "6", None, 1), (7, 8, ".", None, 1), (7, 8 + T16, ".", None, 1), (7, 8 + 2 * T16, ".", None, 1)], subs=True),
        drop_keys(6, (1, 3, 5, 7)), colour(6, [(1, 14), (3, 14), (5, 14), (7, 14)], [(2, 12), (6, 12)], [], [(0, 15), (4, 15)], [], []),
    ]
    scenes = [SceneSpec(patterns={"A": 2, "B": 6, "C": 1, "D": 2}), SceneSpec(patterns={"A": 3, "B": 5, "C": 2, "D": 3}), SceneSpec(patterns={"A": 1, "B": 1, "C": 3, "D": 1}),
              SceneSpec(patterns={"A": 5, "B": 2, "C": 5, "D": 4}), SceneSpec(patterns={"A": 4, "B": 7, "C": 4, "D": 5}), SceneSpec(patterns={"A": 6, "B": 3, "C": 6, "D": 6})]
    return ProjectSpec(bpm=BPM, pads=PADS, patterns=patterns, scenes=scenes, song=[1, 2, 3, 4, 5, 6], current_scene=3,
                       fader_bases=[FaderBase(group="A", parameter="fx", value=0.12), FaderBase(group="B", parameter="fx", value=0.0),
                                    FaderBase(group="C", parameter="fx", value=0.5), FaderBase(group="D", parameter="fx", value=0.55)],
                       fx=FxSpec(type="reverb", x=0.5, y=0.5, output_compressor_drive=0.22, output_compressor_speed=0.5,
                                 sidechain_sources=[SidechainSource(group="A", pad="1"), SidechainSource(group="A", pad="2")],
                                 sidechain_destinations=["B", "C"], sidechain_length=0.35, sidechain_shape=0.5))


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
