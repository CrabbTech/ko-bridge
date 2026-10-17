#!/usr/bin/env python3
"""A dubstep track in E minor at 140 BPM, written through the ko-bridge server: four 8-bar scenes as a song.

    uv run python scripts/dubstep.py --project 9 [--no-write]

What it leans on: the sample library for sounds and their pitches (bass, melody and vocal lines are written in concert
pitch), a low-pass fader lane on the bass group as the wobble LFO (1/8, 1/16, triplet and 1/4 rates), a high-pass lane
and a pitch lane for the riser, the kick pair ducking bass and chords through the device's note-triggered sidechain, group
FX sends into a master reverb, and the output compressor. Scenes: 1 intro, 2 drop, 3 breakdown, 4 second drop; song 1-2-3-4.
"""
from __future__ import annotations

import argparse
import json
import sys

from ko_bridge import server as srv
from ko_bridge.project import FaderBase, FxSpec, Lane, LanePoint, NoteEvent, PadSpec, PatternSpec, ProjectSpec, SceneSpec, SidechainSource

BPM = 140.0
S = 16                      # 16th steps per bar
BARS = 8

PADS = [
    # A: drums
    PadSpec(group="A", pad="1", slot=15, amplitude=114),                    # kick 9x9 dirt e: the punch
    PadSpec(group="A", pad="2", slot=25, amplitude=98),                     # kick sub: the weight (C2, ~2 s)
    PadSpec(group="A", pad="3", slot=110, amplitude=110),                   # snare fat
    PadSpec(group="A", pad="4", slot=304, amplitude=94),                    # clap hard (layer)
    PadSpec(group="A", pad="5", slot=129, amplitude=90),                    # london snare 2 (layer, stereo)
    PadSpec(group="A", pad="6", slot=211, amplitude=118, choke=True),       # hat closed 8x8
    PadSpec(group="A", pad="7", slot=228, amplitude=110, choke=True),       # open hat lo
    PadSpec(group="A", pad="8", slot=142, amplitude=112),                   # rim snap (ghost notes)
    PadSpec(group="A", pad="9", slot=352, amplitude=106),                   # tom lo flam
    PadSpec(group="A", pad=".", slot=252, amplitude=84),                    # crash cym
    PadSpec(group="A", pad="0", slot=250, amplitude=80),                    # china 626
    PadSpec(group="A", pad="enter", slot=237, amplitude=90),                # ride dark
    # B: bass (key mode: note length and release shape the notes; the low-pass lane is the wobble)
    PadSpec(group="B", pad="1", slot=422, play_mode="key", release=100, amplitude=102),   # buzz bass (C2) wobble voice
    PadSpec(group="B", pad="2", slot=429, play_mode="key", release=100, amplitude=76),    # reese (G2) body
    PadSpec(group="B", pad="3", slot=426, play_mode="key", release=130, amplitude=90),    # thirty seven sub (B1)
    PadSpec(group="B", pad="4", slot=443, amplitude=102),                                 # growly (C3) stab
    PadSpec(group="B", pad="5", slot=453, play_mode="key", release=100, amplitude=88),    # phonec grr (C1) second-drop wobble
    # C: harmony
    PadSpec(group="C", pad="1", slot=513, play_mode="key", release=150, amplitude=92),    # scary vibes (E4) motif
    PadSpec(group="C", pad="2", slot=542, amplitude=86),                                  # liquid chord em7, plays through
    PadSpec(group="C", pad="3", slot=533, play_mode="key", release=160, amplitude=82),    # waterchord am7
    PadSpec(group="C", pad="4", slot=541, play_mode="key", release=200, amplitude=80),    # strings dr (C4) riser pad
    PadSpec(group="C", pad="5", slot=523, play_mode="key", release=120, amplitude=98),    # cello pluckz (C4) melody
    PadSpec(group="C", pad="6", slot=554, amplitude=88),                                  # guzheng (C3) accent
    # D: colour
    PadSpec(group="D", pad="1", slot=548, amplitude=94),                                  # vox s6i2 dark (D#3)
    PadSpec(group="D", pad="2", slot=509, amplitude=86),                                  # bg vocal (C5)
    PadSpec(group="D", pad="3", slot=556, amplitude=58),                                  # fx birds tape
    PadSpec(group="D", pad="4", slot=549, amplitude=82),                                  # sitar one shot
]
ROOT = {(p.group, p.pad): p.root for p in PADS}


from ko_bridge.groove import snap_step_to_clock


def hit(group, bar, step, pad, vel=100, semis=0, steps=1.0) -> NoteEvent:
    return NoteEvent(pad=pad, step=snap_step_to_clock(bar * S + step), note=ROOT[(group, pad)] + semis, velocity=vel, duration=max(1, min(384, int(steps * 24))))


def tone(bar, step, pad, name, steps, vel=100) -> NoteEvent:
    """A note in concert pitch, resolved through the library's measured pitch of the pad's sample."""
    return NoteEvent(pad=pad, step=snap_step_to_clock(bar * S + step), concert=name, velocity=vel, duration=max(1, min(384, int(steps * 24))))


def wobble(bar: int, rate: str, lo: float = 0.18, hi: float = 0.95) -> list[LanePoint]:
    """Low-pass lane points across one bar, clamped to >=2.0 steps (1/8 note) and pre-led by 0.5 steps
    so the EP-133's ~100 ms group-fader slew reaches peak cutoff on the beat instead of flattening out."""
    raw_period = {"1/4": 4.0, "1/8": 2.0, "1/16": 2.0, "8T": 8.0 / 3.0, "16T": 2.0}[rate]
    period = max(2.0, raw_period)
    lead = 0.5 if bar > 0 else 0.0
    pts, t = [], 0.0
    while t < S - 1e-6:
        open_step = max(0.0, bar * S + t - (lead if t > 0 or bar > 0 else 0.0))
        close_step = min((bar + 1) * S - 0.25, bar * S + t + period * 0.5)
        pts.append(LanePoint(step=snap_step_to_clock(open_step), value=hi))
        pts.append(LanePoint(step=snap_step_to_clock(close_step), value=lo))
        t += period
    return pts


# ----------------------------------------------------------------------------- scene 2: the drop

def drop_drums(number: int = 1, second: bool = False) -> PatternSpec:
    e = []
    for b in range(BARS):
        e += [hit("A", b, 0, "1", 124), hit("A", b, 0, "2", 118)]
        if b % 2 == 1:
            e += [hit("A", b, 10, "1", 110), hit("A", b, 10, "2", 100)]
        if b in (3, 7):
            e += [hit("A", b, 6, "1", 104)]
        if second and b in (1, 5):
            e += [hit("A", b, 3, "1", 100)]
        e += [hit("A", b, 8, "3", 124), hit("A", b, 8, "4", 104), hit("A", b, 8, "5", 96)]
        if b in (3, 7):
            e.append(hit("A", b, 15, "3", 92))
        if second and b in (3, 7):
            e.append(hit("A", b, 15, "4", 90))
        for st in range(0, S, 2):
            e.append(hit("A", b, st, "6", 94 if st % 4 == 0 else 72))
        if b in (3, 7):
            e += [hit("A", b, st, "6", 62 + 6 * (st - 12)) for st in (13, 15)]
        if b in (1, 5) or (second and b % 2 == 1):
            e.append(hit("A", b, 14, "7", 96))
        e += [hit("A", b, 5, "8", 54), hit("A", b, 11, "8", 62), hit("A", b, 13, "8", 58)]
    e += [hit("A", 7, st, "9", v) for st, v in ((12, 96), (13, 104), (14, 112), (15, 118))]
    e += [hit("A", 3, st, "9", v) for st, v in ((14, 100), (15, 112))]
    if second:
        e += [hit("A", 7, st, "9", v) for st, v in ((8, 90), (10, 96))]
    e.append(hit("A", 0, 0, ".", 100))
    e.append(hit("A", 4, 0, "0", 88))
    if second:
        e.append(hit("A", 0, 0, "0", 84))
    return PatternSpec(group="A", number=number, bars=BARS, events=e)


def drop_bass() -> PatternSpec:
    e = []
    riff = {0: [("E1", 0, 16)], 1: [("E1", 0, 10), ("G1", 10, 6)], 2: [("E1", 0, 16)], 3: [("B1", 0, 8), ("A1", 8, 4), ("G1", 12, 4)],
            4: [("E1", 0, 16)], 5: [("E1", 0, 10), ("G1", 10, 6)], 6: [("E1", 0, 12), ("D2", 12, 4)], 7: [("E1", 0, 8), ("E2", 8, 4)]}
    for b, notes in riff.items():
        for name, st, ln in notes:
            e.append(tone(b, st, "1", name, ln, 112))
    for b in (0, 2, 4, 6):
        e.append(tone(b, 0, "2", "E2", 16, 96))                      # reese body under the wobble
        e.append(tone(b, 0, "3", "E1", 16, 104))                     # sub on the ones
    e.append(tone(3, 0, "3", "B1", 8, 100))
    e += [tone(1, 14, "4", "E2", 2, 104), tone(3, 14, "4", "E2", 2, 104), tone(5, 6, "4", "G2", 2, 100), tone(7, 12, "4", "E2", 2, 108), tone(7, 14, "4", "E3", 2, 108)]
    lp = []
    for b, rate in enumerate(("1/8", "1/8", "1/4", "8T", "1/8", "1/16", "1/8", "16T")):
        lp += wobble(b, rate)
    lanes = [Lane(parameter="lowpass", points=lp),
             Lane(parameter="pitch", points=[LanePoint(step=0, value=0.5), LanePoint(step=7 * S + 12, value=0.5), LanePoint(step=7 * S + 15.9, value=0.32)])]
    return PatternSpec(group="B", number=1, bars=BARS, events=e, lanes=lanes)


def drop_harmony() -> PatternSpec:
    e = [hit("C", 0, 0, "2", 96), hit("C", 4, 0, "2", 96), hit("C", 2, 0, "3", 90, 0, 16), hit("C", 6, 0, "3", 90, 0, 16)]
    for b in (3, 7):
        e += [tone(b, 8, "1", "E4", 2, 90), tone(b, 10, "1", "G4", 2, 90), tone(b, 12, "1", "B4", 4, 94)]
    e.append(tone(7, 14, "6", "E4", 2, 92))
    lanes = [Lane(parameter="lowpass", points=[LanePoint(step=0, value=1.0)]), Lane(parameter="highpass", points=[LanePoint(step=0, value=0.0)]),
             Lane(parameter="pitch", points=[LanePoint(step=0, value=0.5)])]
    return PatternSpec(group="C", number=1, bars=BARS, events=e, lanes=lanes)


def drop_colour(number: int = 1) -> PatternSpec:
    e = [tone(1, 14, "1", "E3", 2, 92), tone(5, 14, "1", "E3", 2, 92), hit("D", 3, 12, "2", 86), hit("D", 7, 8, "2", 84), hit("D", 0, 0, "3", 56)]
    return PatternSpec(group="D", number=number, bars=BARS, events=e)


# ----------------------------------------------------------------------------- scene 1: intro

def intro_drums() -> PatternSpec:
    e = [hit("A", 0, 0, ".", 92)]
    for b in range(BARS):
        for st in range(0, S, 2):
            e.append(hit("A", b, st, "6", 76 if st % 4 == 0 else 58))
        e += [hit("A", b, 5, "8", 46), hit("A", b, 13, "8", 52)]
        if b >= 4:
            e += [hit("A", b, 0, "1", 100), hit("A", b, 8, "3", 100), hit("A", b, 8, "4", 86)]
        if b in (5, 6):
            e.append(hit("A", b, 14, "7", 90))
    e += [hit("A", 7, st, "9", 80 + 5 * (st - 8)) for st in range(8, 16)]
    e += [hit("A", 7, st, "3", v) for st, v in ((12, 90), (13, 96), (14, 108), (15, 120))]
    return PatternSpec(group="A", number=2, bars=BARS, events=e)


def intro_bass() -> PatternSpec:
    e = [tone(b, 0, "3", "E1", 16, 96) for b in (4, 5, 6, 7)] + [tone(6, 0, "2", "E2", 16, 88), tone(7, 0, "2", "E2", 16, 92)]
    lanes = [Lane(parameter="lowpass", points=[LanePoint(step=0, value=0.95), LanePoint(step=6 * S, value=0.2), LanePoint(step=7 * S, value=0.55), LanePoint(step=8 * S - 0.1, value=0.95)]),
             Lane(parameter="pitch", points=[LanePoint(step=0, value=0.5)])]
    return PatternSpec(group="B", number=2, bars=BARS, events=e, lanes=lanes)


MOTIF = [(0, 0, "E4", 4), (0, 4, "G4", 4), (0, 8, "B4", 4), (0, 12, "A4", 4), (1, 0, "G4", 6), (1, 6, "F#4", 2), (1, 8, "E4", 8),
         (2, 0, "E4", 4), (2, 4, "G4", 4), (2, 8, "D5", 4), (2, 12, "B4", 4), (3, 0, "A4", 8), (3, 8, "G4", 8)]


def intro_harmony() -> PatternSpec:
    e = []
    for rep in (0, 4):
        e += [tone(b + rep, st, "1", name, ln, 88 if rep == 0 else 96) for b, st, name, ln in MOTIF]
    e += [tone(b, 0, "4", "E3", 16, 56 + 3 * b) for b in range(BARS)]                      # strings swell
    e += [hit("C", 2, 0, "3", 84, 0, 16), hit("C", 6, 0, "3", 88, 0, 16)]                   # Am7
    e += [tone(5, 14, "6", "B3", 2, 86), tone(7, 14, "6", "E4", 2, 90)]
    lanes = [Lane(parameter="highpass", points=[LanePoint(step=0, value=0.0), LanePoint(step=4 * S, value=0.0), LanePoint(step=8 * S - 0.1, value=0.62)]),   # riser thins out
             Lane(parameter="pitch", points=[LanePoint(step=0, value=0.5), LanePoint(step=6 * S, value=0.5), LanePoint(step=8 * S - 0.1, value=0.58)]),     # and lifts
             Lane(parameter="lowpass", points=[LanePoint(step=0, value=0.7), LanePoint(step=4 * S, value=1.0)])]
    return PatternSpec(group="C", number=2, bars=BARS, events=e, lanes=lanes)


def intro_colour() -> PatternSpec:
    return PatternSpec(group="D", number=2, bars=BARS, events=[hit("D", 0, 0, "3", 62), hit("D", 4, 0, "3", 58), tone(3, 14, "1", "E3", 2, 88), hit("D", 7, 0, "2", 90)])


# ----------------------------------------------------------------------------- scene 3: breakdown

def breakdown_drums() -> PatternSpec:
    e = []
    for b in range(BARS):
        if b % 2 == 0:
            e.append(hit("A", b, 0, "1", 100))
        e += [hit("A", b, 8, "3", 102), hit("A", b, 8, "4", 84)]
        e += [hit("A", b, st, "enter", 70) for st in (0, 4, 8, 12)]
        e += [hit("A", b, 5, "8", 48), hit("A", b, 13, "8", 54)]
    e += [hit("A", 7, st, "9", 86 + 6 * (st - 12)) for st in range(12, 16)]
    e += [hit("A", 7, st, "3", v) for st, v in ((14, 100), (15, 116))]
    return PatternSpec(group="A", number=3, bars=BARS, events=e)


def breakdown_bass() -> PatternSpec:
    e = [tone(0, 0, "3", "E1", 16, 98), tone(4, 0, "3", "E1", 16, 98)]
    for b, name in ((1, "E2"), (2, "E2"), (3, "B1"), (5, "A1"), (6, "A1"), (7, "G1")):
        e.append(tone(b, 0, "2", name, 16, 90))
    lp = []
    for b in range(BARS):
        lp += wobble(b, "1/4", lo=0.3, hi=0.8)
    return PatternSpec(group="B", number=3, bars=BARS, events=e, lanes=[Lane(parameter="lowpass", points=lp), Lane(parameter="pitch", points=[LanePoint(step=0, value=0.5)])])


MELODY = [(0, 0, "E4", 3), (0, 3, "G4", 3), (0, 6, "F#4", 2), (0, 8, "E4", 4), (0, 12, "D4", 4), (1, 0, "B3", 8), (1, 8, "E4", 8),
          (2, 0, "G4", 3), (2, 3, "A4", 3), (2, 6, "B4", 2), (2, 8, "A4", 4), (2, 12, "G4", 4), (3, 0, "F#4", 6), (3, 6, "E4", 2), (3, 8, "D4", 8),
          (4, 0, "E4", 3), (4, 3, "G4", 3), (4, 6, "B4", 2), (4, 8, "D5", 4), (4, 12, "B4", 4), (5, 0, "A4", 8), (5, 8, "G4", 8),
          (6, 0, "F#4", 4), (6, 4, "E4", 4), (6, 8, "D4", 4), (6, 12, "B3", 4), (7, 0, "E4", 16)]


def breakdown_harmony() -> PatternSpec:
    e = [tone(b, st, "5", name, ln, 100) for b, st, name, ln in MELODY]
    e += [hit("C", 0, 0, "2", 90), hit("C", 4, 0, "2", 90), hit("C", 2, 0, "3", 86, 0, 16), hit("C", 6, 0, "3", 86, 0, 16), tone(6, 0, "1", "E4", 16, 80)]
    lanes = [Lane(parameter="lowpass", points=[LanePoint(step=0, value=1.0)]), Lane(parameter="highpass", points=[LanePoint(step=0, value=0.0)]),
             Lane(parameter="pitch", points=[LanePoint(step=0, value=0.5)])]
    return PatternSpec(group="C", number=3, bars=BARS, events=e, lanes=lanes)


def breakdown_colour() -> PatternSpec:
    return PatternSpec(group="D", number=3, bars=BARS, events=[hit("D", 1, 8, "4", 84), hit("D", 5, 8, "4", 80), tone(3, 14, "1", "E3", 2, 86), hit("D", 7, 12, "2", 88)])


# ----------------------------------------------------------------------------- scene 4: second drop

def drop2_bass() -> PatternSpec:
    e = []
    for b in range(BARS):
        e.append(tone(b, 0, "5", "E1", 16 if b % 4 != 3 else 12, 108))      # phonec grr carries the wobble
        if b % 4 == 3:
            e.append(tone(b, 12, "5", "G1", 4, 104))
    for b in (0, 2, 4, 6):
        e.append(tone(b, 0, "3", "E1", 16, 100))
    e += [tone(1, 14, "1", "E2", 2, 104), tone(3, 6, "1", "E2", 2, 100), tone(5, 14, "1", "B1", 2, 104), tone(7, 12, "4", "E2", 2, 108), tone(7, 14, "4", "E3", 2, 108)]
    lp = []
    for b, rate in enumerate(("1/8", "1/16", "8T", "1/4", "1/8", "16T", "1/8", "1/16")):
        lp += wobble(b, rate, lo=0.15, hi=0.95)
    lanes = [Lane(parameter="lowpass", points=lp),
             Lane(parameter="pitch", points=[LanePoint(step=0, value=0.5), LanePoint(step=3 * S + 14, value=0.5), LanePoint(step=4 * S - 0.1, value=0.38),
                                             LanePoint(step=4 * S, value=0.5), LanePoint(step=7 * S + 12, value=0.5), LanePoint(step=8 * S - 0.1, value=0.3)])]
    return PatternSpec(group="B", number=4, bars=BARS, events=e, lanes=lanes)


def drop2_harmony() -> PatternSpec:
    e = [hit("C", 0, 0, "2", 96), hit("C", 4, 0, "2", 96), hit("C", 2, 0, "3", 90, 0, 8), hit("C", 6, 0, "3", 90, 0, 8)]
    e += [tone(b, 14, "6", "E4" if b in (1, 5) else "B3", 2, 90) for b in (1, 3, 5, 7)]
    e += [tone(7, 8, "1", "E4", 2, 92), tone(7, 10, "1", "G4", 2, 92), tone(7, 12, "1", "B4", 4, 96)]
    lanes = [Lane(parameter="lowpass", points=[LanePoint(step=0, value=1.0)]), Lane(parameter="highpass", points=[LanePoint(step=0, value=0.0)]),
             Lane(parameter="pitch", points=[LanePoint(step=0, value=0.5)])]
    return PatternSpec(group="C", number=4, bars=BARS, events=e, lanes=lanes)


def spec() -> ProjectSpec:
    patterns = [drop_drums(1), drop_bass(), drop_harmony(), drop_colour(1),
                intro_drums(), intro_bass(), intro_harmony(), intro_colour(),
                breakdown_drums(), breakdown_bass(), breakdown_harmony(), breakdown_colour(),
                drop_drums(4, second=True), drop2_bass(), drop2_harmony(), drop_colour(4)]
    scenes = [SceneSpec(patterns={"A": 2, "B": 2, "C": 2, "D": 2}), SceneSpec(patterns={"A": 1, "B": 1, "C": 1, "D": 1}),
              SceneSpec(patterns={"A": 3, "B": 3, "C": 3, "D": 3}), SceneSpec(patterns={"A": 4, "B": 4, "C": 4, "D": 4})]
    return ProjectSpec(bpm=BPM, pads=PADS, patterns=patterns, scenes=scenes, song=[1, 2, 3, 4], current_scene=2,
                       fader_bases=[FaderBase(group="A", parameter="fx", value=0.12), FaderBase(group="B", parameter="fx", value=0.0),
                                    FaderBase(group="C", parameter="fx", value=0.4), FaderBase(group="D", parameter="fx", value=0.55)],
                       fx=FxSpec(type="reverb", x=0.5, y=0.55, output_compressor_drive=0.5, output_compressor_speed=0.45,
                                 sidechain_sources=[SidechainSource(group="A", pad="1"), SidechainSource(group="A", pad="2")],
                                 sidechain_destinations=["B", "C"], sidechain_length=0.4, sidechain_shape=0.5))


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
    print("scenes:", [(sc["scene"], sc["patterns"]) for sc in after["scenes"]], "| song", after["song"], "| cursor", after["scene_cursor"], "| bpm", after["bpm"], "| fx", after.get("fx"))
    print("B01 first notes:", [(e["step"], e["note"]) for e in after["patterns"]["B01"]["events"]][:6], "| lanes:", {k: len(v) for k, v in after["patterns"]["B01"]["lanes"].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
