#!/usr/bin/env python3
"""Riddim at 140 BPM in F minor, library sounds only: factory drums and basses, Tyler's Pocket Operator imports and
megaman hits, vocal shots. Six 8-bar scenes as a song.

    uv run python scripts/riddim.py --project 9

How the riddim character is made from one-shots: every bass pad sits in the device's mute group, so each new bass hit
cuts the last (the gated bounce); riffs are programmed on 16ths and 16th-triplets with humanized velocities; pitch drops
and sub drops are the group pitch lane; risers are pitch and high-pass lanes on a held string; every event sits on a MIDI
clock multiple (4 ticks) so playback under the Mac's clock stays tight; drops stay under ten voices.
"""
from __future__ import annotations

import argparse
import json
import random
import sys

from ko_bridge import server as srv
from ko_bridge.project import FaderBase, FxSpec, Lane, LanePoint, NoteEvent, PadSpec, PatternSpec, ProjectSpec, SceneSpec, SidechainSource

BPM = 140.0
S = 16
BARS = 8
RNG = random.Random(140)
SW = 1 / 3          # swing: off-16ths land 8 ticks (two MIDI clocks) late
T16 = 2 / 3         # 16th-note triplet in steps (16 ticks)

LAYOUT = {
    # A: drums, factory with PO layers
    ("A", "1"): dict(slot=21, amplitude=108),                 # nt alt kick: the weight
    ("A", "2"): dict(slot=741, amplitude=104),                # PO kick (sub-heavy click): the punch layer
    ("A", "3"): dict(slot=106, amplitude=112),                # snare 6x6 1 on the three
    ("A", "4"): dict(slot=309, amplitude=96),                 # clap sp disco b (layer)
    ("A", "5"): dict(slot=822, amplitude=92),                 # PO snare, bright (layer)
    ("A", "6"): dict(slot=802, amplitude=100),                 # PO hat, 28 ms, clicky
    ("A", "7"): dict(slot=757, amplitude=96),                 # PO hat, 89 ms
    ("A", "8"): dict(slot=803, amplitude=96),                 # PO open hat
    ("A", "9"): dict(slot=139, amplitude=100),                 # rimshot dmx
    ("A", "."): dict(slot=339, amplitude=90),                 # cowbell
    ("A", "0"): dict(slot=752, amplitude=96),                 # megaman_beep_hit
    ("A", "enter"): dict(slot=739, amplitude=96),             # PO snare tick (ghost notes)
    # B: basses in the mute group (gated bounce); the sub sustains outside it
    ("B", "1"): dict(slot=457, amplitude=104, choke=True),    # wlobb wobb (C2): main wonk
    ("B", "2"): dict(slot=443, amplitude=106, choke=True),     # growly (C3): the answer, an octave up
    ("B", "3"): dict(slot=906, amplitude=104, choke=True),    # PO bass tone, native F1
    ("B", "4"): dict(slot=411, amplitude=102, choke=True),    # tb3x3 punch (E1)
    ("B", "5"): dict(slot=428, amplitude=104, choke=True),    # mud (C1)
    ("B", "6"): dict(slot=453, amplitude=102, choke=True),     # phonec grr (C1), long growl
    ("B", "7"): dict(slot=456, amplitude=98, choke=True),     # tock (C2), percussive
    ("B", "8"): dict(slot=426, amplitude=96, play_mode="key", release=110),   # thirty seven sub (B1)
    ("B", "9"): dict(slot=412, amplitude=102, choke=True),     # uggbass (D2)
    ("B", "."): dict(slot=749, amplitude=98, choke=True),     # boop_bass_megman (D#2)
    ("B", "0"): dict(slot=459, amplitude=102, choke=True),     # rude twang
    ("B", "enter"): dict(slot=765, amplitude=102, choke=True), # PO bass tone F2
    # C: atmosphere and melody
    ("C", "1"): dict(slot=513, amplitude=88, play_mode="key", release=150),   # scary vibes (E4)
    ("C", "2"): dict(slot=549, amplitude=90),                                  # sitar one shot (C5)
    ("C", "3"): dict(slot=530, amplitude=86, play_mode="key", release=200),   # cello 360 (F2): drone
    ("C", "4"): dict(slot=448, amplitude=84),                                  # kaa wa bell air (E2)
    ("C", "5"): dict(slot=526, amplitude=88),                                  # dr organ chord (B1)
    ("C", "6"): dict(slot=541, amplitude=74, play_mode="key", release=200),   # strings dr (C4): riser pad
    ("C", "7"): dict(slot=246, amplitude=80),                                  # toaster bell
    ("C", "8"): dict(slot=851, amplitude=92),                                  # PO melodic hit F3
    ("C", "9"): dict(slot=904, amplitude=84),                                  # PO bright blip C5
    # D: colour
    ("D", "1"): dict(slot=553, amplitude=94),                                  # vox s6i2 eh (F3)
    ("D", "2"): dict(slot=548, amplitude=90),                                  # vox s6i2 dark (D#3)
    ("D", "3"): dict(slot=509, amplitude=84),                                  # bg vocal (C5)
    ("D", "4"): dict(slot=751, amplitude=88),                                  # megaman_beep (G3)
    ("D", "5"): dict(slot=801, amplitude=88),                                  # PO perc, bright
    ("D", "6"): dict(slot=556, amplitude=60),                                  # fx birds tape
    ("D", "7"): dict(slot=819, amplitude=84),                                  # PO noise swell, 530 ms
    ("D", "8"): dict(slot=865, amplitude=82),                                  # PO long bright decay (cymbal-like)
}
PADS = [PadSpec(group=g, pad=p, **kw) for (g, p), kw in LAYOUT.items()]
ROOT = {(p.group, p.pad): p.root for p in PADS}


def hv(v: int, spread: int = 8) -> int:
    return max(1, min(127, v + RNG.randint(-spread, spread)))


def hit(group, bar, step, pad, vel=100, semis=0, steps=1.0, humanize=True) -> NoteEvent:
    return NoteEvent(pad=pad, step=bar * S + step, note=ROOT[(group, pad)] + semis, velocity=hv(vel) if humanize else vel, duration=max(1, min(384, int(steps * 24))))


def tone(bar, step, pad, name, steps=1.0, vel=100) -> NoteEvent:
    return NoteEvent(pad=pad, step=bar * S + step, concert=name, velocity=hv(vel, 5), duration=max(1, min(384, int(steps * 24))))


# ----------------------------------------------------------------------------- drums

def half_time(e, b, full=True, fill=False, vel=1.0):
    v = lambda x: int(x * vel)
    e += [hit("A", b, 0, "1", v(124)), hit("A", b, 0, "2", v(112))]
    if full and b % 2 == 1:
        e += [hit("A", b, 10 + SW, "2", v(100))]                                   # swung pickup
    e += [hit("A", b, 8, "3", v(126)), hit("A", b, 8, "4", v(102)), hit("A", b, 8, "5", v(94))]
    if full:
        e += [hit("A", b, st + (SW if st % 4 == 2 else 0), "6", v(96 if st % 4 == 0 else 68)) for st in (2, 4, 6, 10, 12, 14)]   # hats, swung on the off-beats
        e += [hit("A", b, st, "7", v(58)) for st in (1 + SW, 9 + SW)]
        e += [hit("A", b, 5 + SW, "enter", v(60)), hit("A", b, 13 + SW, "9", v(64))]
    if fill:
        e += [hit("A", b, 12 + k * T16, "3", v(84 + 7 * k)) for k in range(6)]     # 16th-triplet snare roll
    return e


def drop_drums(number: int, variant: int) -> PatternSpec:
    e = []
    for b in range(BARS):
        half_time(e, b, full=True, fill=(b in (3, 7)))
        if variant >= 1 and b % 2 == 1:
            e.append(hit("A", b, 14, ".", 84))                                     # cowbell answer
        if b in (1, 5):
            e.append(hit("A", b, 14 + SW, "8", 88))                                # open hat lift
        if variant >= 2 and b % 4 == 2:
            e += [hit("A", b, 6 + T16 * k, "0", 90) for k in range(3)]             # beep-hit triplet
    e.append(hit("A", 0, 0, "8", 100))
    return PatternSpec(group="A", number=number, bars=BARS, events=e, lanes=[Lane(parameter="level", points=[LanePoint(step=0, value=1.0)])])


def intro_drums() -> PatternSpec:
    e = []
    for b in range(BARS):
        e += [hit("A", b, 8, "9", 56), hit("A", b, 14 + SW, "6", 52)]
        if b >= 4:
            e += [hit("A", b, 4 + T16 * k, "enter", 62) for k in range(2)]
    e += [hit("A", 7, 8 + k * T16, "3", 70 + 8 * k) for k in range(6)]
    return PatternSpec(group="A", number=2, bars=BARS, events=e, lanes=[Lane(parameter="level", points=[LanePoint(step=0, value=0.7)])])


def build_drums() -> PatternSpec:
    e = []
    for b in range(BARS):
        half_time(e, b, full=(b >= 4), fill=(b == 7))
        if b < 4:
            e += [hit("A", b, 6 + SW, "6", 70), hit("A", b, 14 + SW, "6", 70)]
    e += [hit("A", 7, 0, "0", 92), hit("A", 7, 4, "0", 92)]
    return PatternSpec(group="A", number=3, bars=BARS, events=e,
                       lanes=[Lane(parameter="level", points=[LanePoint(step=0, value=0.75), LanePoint(step=4 * S, value=0.85), LanePoint(step=8 * S - 1, value=0.95)])])


def break_drums() -> PatternSpec:
    e = []
    for b in range(BARS):
        if b < 6:
            e += [hit("A", b, 8, "4", 78), hit("A", b, 8, "5", 74), hit("A", b, 13 + SW, "9", 52)]
        if b == 6:
            e += [hit("A", b, k * T16, "3", 72 + 3 * k) for k in range(12)]       # roll over beats 1-2
        if b == 7:                                                                  # beat 1 silent, then the pickup
            e += [hit("A", b, 8, "1", 110), hit("A", b, 8, "2", 100)] + [hit("A", b, 12 + k * T16, "3", 96 + 8 * k) for k in range(6)]
    return PatternSpec(group="A", number=4, bars=BARS, events=e, lanes=[Lane(parameter="level", points=[LanePoint(step=0, value=0.75)])])


# ----------------------------------------------------------------------------- bass riffs (two bars each, in 16ths)

RIFF_A = [(0, "1", "F1", 120), (3, "4", "F1", 108), (6, "1", "G#1", 112), (8, "2", "F2", 108), (10, "7", "F1", 92), (12, "3", "F1", 118), (14, "0", None, 100),
          (16, "1", "F1", 120), (19, "4", "F1", 108), (22, "1", "A#1", 112), (24, "2", "G#2", 108), (26, "7", "F1", 92), (28, "3", "F1", 118), (30, "5", "F1", 116)]
RIFF_B = [(0, "6", "F1", 118), (4, "1", "C2", 104), (6, "1", "A#1", 104), (8, "2", "F2", 112), (12, "1", "F1", 118), (12 + 2 * T16, "1", "F1", 100), (12 + 4 * T16, "1", "F1", 100),
          (16, "6", "F1", 118), (20, "9", "F2", 104), (22, "1", "G#1", 104), (24, "2", "G#2", 110), (27, "7", "F1", 92), (28, "3", "F1", 118), (30, ".", None, 100)]
RIFF_C = [(0, "3", "F1", 120), (6, "5", "F1", 112), (12, "1", "G#1", 104), (14, "1", "A#1", 104), (16, "3", "F1", 120), (22, "6", "F1", 114), (28, "4", "F1", 108), (30, "4", "F1", 108)]
FILL = RIFF_A[:7] + [(16 + k * T16, "2", n, 104 + 2 * k) for k, n in enumerate(("F2", "G#2", "A#2", "C3", "C#3", "D#3", "F3", "G#3", "A#3", "C4", "C#4", "D#4"))] + [(28, "3", "F1", 124)]


def riff(e, bar, notes):
    for st, pad, name, vel in notes:
        e.append(tone(bar, st, pad, name, 1, vel) if name else hit("B", bar, st, pad, vel))


def drop_bass(number: int, plan: list, subs: bool = True, sub_drop_bar: int | None = None) -> PatternSpec:
    e = []
    pitch = [LanePoint(step=0, value=0.5)]
    for i, name in enumerate(plan):
        bar = 2 * i
        riff(e, bar, {"A": RIFF_A, "B": RIFF_B, "C": RIFF_C, "fill": FILL}[name])
        # a pitch drop on the last 16th of each two-bar riff (group pitch lane), reset on the next one
        pitch += [LanePoint(step=(bar + 2) * S - 2, value=0.5), LanePoint(step=(bar + 2) * S - 0.1, value=0.3 if name != "fill" else 0.05), LanePoint(step=(bar + 2) * S, value=0.5)]
    if subs:
        e += [tone(b, 0, "8", "F1", 16, 100) for b in range(BARS) if b % 2 == 0]
    if sub_drop_bar is not None:
        pitch += [LanePoint(step=sub_drop_bar * S + 8, value=0.5), LanePoint(step=(sub_drop_bar + 1) * S - 0.1, value=0.0)]
    pitch = sorted(pitch, key=lambda p: p.step)
    lanes = [Lane(parameter="level", points=[LanePoint(step=0, value=0.95)]), Lane(parameter="lowpass", points=[LanePoint(step=0, value=1.0)]), Lane(parameter="pitch", points=pitch)]
    return PatternSpec(group="B", number=number, bars=BARS, events=e, lanes=lanes)


def build_bass() -> PatternSpec:
    e = []
    for bar in (0, 2, 4, 6):
        riff(e, bar, RIFF_A)
    e += [tone(b, 0, "8", "F1", 16, 96) for b in (4, 6)]
    lanes = [Lane(parameter="level", points=[LanePoint(step=0, value=0.8)]), Lane(parameter="pitch", points=[LanePoint(step=0, value=0.5)]),
             Lane(parameter="lowpass", points=[LanePoint(step=0, value=0.3), LanePoint(step=4 * S, value=0.55), LanePoint(step=8 * S - 1, value=1.0)])]
    return PatternSpec(group="B", number=5, bars=BARS, events=e, lanes=lanes)


def quiet_bass(number: int, subs: list[int]) -> PatternSpec:
    e = [tone(b, 0, "8", "F1", 16, 96) for b in subs]
    return PatternSpec(group="B", number=number, bars=BARS, events=e, lanes=[Lane(parameter="level", points=[LanePoint(step=0, value=0.85)]),
                                                                            Lane(parameter="lowpass", points=[LanePoint(step=0, value=1.0)]), Lane(parameter="pitch", points=[LanePoint(step=0, value=0.5)])])


# ----------------------------------------------------------------------------- atmosphere and melody

SITAR = [(0, 0, "F4", 2), (0, 2, "G#4", 2), (0, 4, "C5", 4), (1, 0, "A#4", 2), (1, 2, "G#4", 2), (1, 4, "F4", 4),
         (4, 0, "F4", 2), (4, 2, "G#4", 2), (4, 4, "C5", 2), (4, 6, "C#5", 2), (5, 0, "C5", 4), (5, 4, "G#4", 4)]
VIBES = [(2, 0, "F4", 6), (2, 6, "G#4", 2), (2, 8, "G4", 8), (6, 0, "F4", 6), (6, 6, "C5", 2), (6, 8, "A#4", 8)]
PO_MOTIF = [(0, 0, "F3", 1), (0, 2, "G#3", 1), (0, 3, "F3", 1), (0, 6, "C4", 1), (1, 0, "A#3", 1), (1, 2, "G#3", 1), (1, 4, "F3", 1), (1, 6 + SW, "F3", 1)]


def intro_keys() -> PatternSpec:
    e = [tone(b, 0, "3", "F2", 16, 72 + 2 * b) for b in range(BARS)]
    e += [tone(b, st, "2", n, ln, 86) for b, st, n, ln in SITAR] + [tone(b, st, "1", n, ln, 80) for b, st, n, ln in VIBES]
    e += [hit("C", b, 12 + SW, "4", 70, 1) for b in (1, 3, 5)]
    e += [tone(b, 0, "6", "F3", 16, 50 + 6 * (b - 4)) for b in (6, 7)]
    lanes = [Lane(parameter="lowpass", points=[LanePoint(step=0, value=0.6), LanePoint(step=8 * S - 1, value=1.0)]), Lane(parameter="level", points=[LanePoint(step=0, value=0.85)]),
             Lane(parameter="fx", points=[LanePoint(step=0, value=0.6)]),
             Lane(parameter="pitch", points=[LanePoint(step=0, value=0.5), LanePoint(step=6 * S, value=0.5), LanePoint(step=8 * S - 0.1, value=0.62)]),       # riser lift
             Lane(parameter="highpass", points=[LanePoint(step=0, value=0.0), LanePoint(step=6 * S, value=0.0), LanePoint(step=8 * S - 0.1, value=0.6)])]
    return PatternSpec(group="C", number=1, bars=BARS, events=e, lanes=lanes)


def build_keys() -> PatternSpec:
    e = [tone(b, 0, "3", "F2", 16, 78) for b in range(0, BARS, 2)]
    e += [tone(b, st, "2", n, ln, 84) for b, st, n, ln in SITAR if b < 2] + [tone(b + 4, st, "2", n, ln, 88) for b, st, n, ln in SITAR if b < 2]
    e += [hit("C", b, 8, "5", 88, 6) for b in (1, 3, 5)] + [hit("C", 7, 8, "5", 92, 6), hit("C", 7, 12, "5", 96, 9)]
    e += [tone(b, 0, "6", "F3", 16, 56 + 4 * (b - 4)) for b in range(4, 8)]
    lanes = [Lane(parameter="lowpass", points=[LanePoint(step=0, value=0.8), LanePoint(step=8 * S - 1, value=1.0)]), Lane(parameter="level", points=[LanePoint(step=0, value=0.8)]),
             Lane(parameter="fx", points=[LanePoint(step=0, value=0.5)]),
             Lane(parameter="pitch", points=[LanePoint(step=0, value=0.5), LanePoint(step=6 * S, value=0.5), LanePoint(step=8 * S - 0.1, value=0.66)]),
             Lane(parameter="highpass", points=[LanePoint(step=0, value=0.0), LanePoint(step=6 * S, value=0.0), LanePoint(step=8 * S - 0.1, value=0.7)])]
    return PatternSpec(group="C", number=2, bars=BARS, events=e, lanes=lanes)


def drop_keys(number: int, stabs_on: tuple, motif: bool = False) -> PatternSpec:
    e = [hit("C", b, 14 + SW, "5", 90, 6) for b in stabs_on] + [hit("C", b, 12, "7", 78) for b in (1, 5)]
    e += [tone(7, 8 + k * T16, "2", n, 1, 86) for k, n in enumerate(("F4", "G#4", "C5"))]
    if motif:
        e += [tone(b + 4, st, "8", n, ln, 94) for b, st, n, ln in PO_MOTIF] + [tone(3, 12, "9", "F5", 1, 80), tone(3, 14, "9", "G#5", 1, 80)]
    lanes = [Lane(parameter="lowpass", points=[LanePoint(step=0, value=1.0)]), Lane(parameter="level", points=[LanePoint(step=0, value=0.82)]),
             Lane(parameter="fx", points=[LanePoint(step=0, value=0.4)]), Lane(parameter="pitch", points=[LanePoint(step=0, value=0.5)]), Lane(parameter="highpass", points=[LanePoint(step=0, value=0.0)])]
    return PatternSpec(group="C", number=number, bars=BARS, events=e, lanes=lanes)


def break_keys() -> PatternSpec:
    e = [tone(b, 0, "3", "F2", 16, 76) for b in range(0, 6, 2)] + [tone(b, 0, "6", "F3", 16, 66) for b in (0, 2, 4)]
    e += [tone(b, st, "8", n, ln, 92) for b, st, n, ln in PO_MOTIF] + [tone(b + 2, st, "8", n, ln, 92) for b, st, n, ln in PO_MOTIF]
    e += [tone(b, st, "2", n, ln, 86) for b, st, n, ln in SITAR if b >= 4]
    e += [tone(4, 0, "1", "G#4", 16, 80), tone(5, 0, "1", "F4", 16, 80)]
    e += [tone(b, 0, "6", "F3", 16, 60 + 8 * (b - 6)) for b in (6, 7)]
    lanes = [Lane(parameter="lowpass", points=[LanePoint(step=0, value=1.0)]), Lane(parameter="fx", points=[LanePoint(step=0, value=0.6)]),
             Lane(parameter="level", points=[LanePoint(step=0, value=0.9), LanePoint(step=6 * S + 12, value=0.9), LanePoint(step=7 * S, value=0.0), LanePoint(step=7 * S + 7.9, value=0.0), LanePoint(step=7 * S + 8, value=0.9)]),
             Lane(parameter="pitch", points=[LanePoint(step=0, value=0.5), LanePoint(step=6 * S, value=0.5), LanePoint(step=7 * S + 8, value=0.66)]),
             Lane(parameter="highpass", points=[LanePoint(step=0, value=0.0), LanePoint(step=6 * S, value=0.0), LanePoint(step=7 * S + 8, value=0.7)])]
    return PatternSpec(group="C", number=4, bars=BARS, events=e, lanes=lanes)


# ----------------------------------------------------------------------------- colour

def colour(number: int, eh=(), dark=(), bg=(), beeps=(), perc=(), swell_bars=(), birds=(), cym=()) -> PatternSpec:
    e = [tone(b, st, "1", "F3", 2, 92) for b, st in eh] + [tone(b, st, "2", "F3", 2, 88) for b, st in dark] + [tone(b, st, "3", "F4", 8, 82) for b, st in bg]
    e += [hit("D", b, st, "4", 84) for b, st in beeps] + [hit("D", b, st, "5", 86) for b, st in perc]
    e += [hit("D", b, 12, "7", 92) for b in swell_bars] + [hit("D", b, 0, "6", 62, 0, 16) for b in birds] + [hit("D", b, 0, "8", 90) for b in cym]
    return PatternSpec(group="D", number=number, bars=BARS, events=e, lanes=[Lane(parameter="fx", points=[LanePoint(step=0, value=0.5)])])


def spec() -> ProjectSpec:
    patterns = [
        drop_drums(1, 0), drop_bass(1, ["A", "A", "A", "fill"]), drop_keys(3, (3, 7)), colour(1, eh=[(3, 14), (7, 14)], beeps=[(1, 15), (5, 15)], cym=[0]),            # 3 drop A
        intro_drums(), quiet_bass(6, [4, 6]), intro_keys(), colour(2, eh=[(3, 8)], dark=[(5, 12)], bg=[(6, 0)], beeps=[(1, 6), (1, 6 + T16), (5, 6)], birds=[0, 4], swell_bars=[7]),   # 1 intro
        build_drums(), build_bass(), build_keys(), colour(3, eh=[(1, 14), (5, 14)], bg=[(3, 8)], beeps=[(2, 10), (6, 10), (6, 10 + T16)], swell_bars=[7], perc=[(3, 15), (7, 15)]),   # 2 build
        drop_drums(5, 1), drop_bass(2, ["B", "B", "B", "fill"], sub_drop_bar=7), drop_keys(5, (1, 5), motif=True), colour(4, eh=[(1, 14), (5, 14)], dark=[(3, 12)], beeps=[(2, 15), (6, 15)], cym=[0, 4]),   # 4 drop B
        break_drums(), quiet_bass(7, [0, 2, 4]), break_keys(), colour(5, eh=[(1, 8)], bg=[(2, 0), (5, 0)], beeps=[(3, 6), (3, 6 + T16), (3, 6 + 2 * T16)], swell_bars=[6, 7], birds=[0]),   # 5 break
        drop_drums(6, 2), drop_bass(3, ["C", "A", "C", "fill"]), drop_keys(6, (1, 3, 5, 7), motif=True), colour(6, eh=[(1, 14), (3, 14), (5, 14), (7, 14)], dark=[(2, 12), (6, 12)], beeps=[(0, 15), (4, 15)], cym=[0]),   # 6 drop C
    ]
    scenes = [SceneSpec(patterns={"A": 2, "B": 6, "C": 1, "D": 2}), SceneSpec(patterns={"A": 3, "B": 5, "C": 2, "D": 3}), SceneSpec(patterns={"A": 1, "B": 1, "C": 3, "D": 1}),
              SceneSpec(patterns={"A": 5, "B": 2, "C": 5, "D": 4}), SceneSpec(patterns={"A": 4, "B": 7, "C": 4, "D": 5}), SceneSpec(patterns={"A": 6, "B": 3, "C": 6, "D": 6})]
    return ProjectSpec(bpm=BPM, pads=PADS, patterns=patterns, scenes=scenes, song=[1, 2, 3, 4, 5, 6], current_scene=3,
                       fader_bases=[FaderBase(group="A", parameter="fx", value=0.12), FaderBase(group="B", parameter="fx", value=0.0),
                                    FaderBase(group="C", parameter="fx", value=0.5), FaderBase(group="D", parameter="fx", value=0.55)],
                       fx=FxSpec(type="reverb", x=0.5, y=0.5, output_compressor_drive=0.3, output_compressor_speed=0.5,
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
