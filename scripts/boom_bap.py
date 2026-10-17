#!/usr/bin/env python3
"""Boom bap from a chopped loop, library sounds only: the EP-133 plays a two-bar soul phrase (electric piano, wurlitzer,
muted guitar, electric bass, shaker; C minor at 107 BPM), the Mac records it, screws it down three semitones (A minor at
90 BPM) and chops it across the twelve pads of group C in ONE sample slot (trim start + length per pad, choke, the
turnaround reversed). Fresh factory drums, an upright sub and vocal shots go around the chops in four 4-bar scenes.

    uv run python scripts/boom_bap.py source            # write the source phrase to P9, record it, cut the 2-bar loop WAV
    uv run python scripts/boom_bap.py chop [--dry-run]  # screw + slice the newest loop, upload it to the first free slot from 600
    uv run python scripts/boom_bap.py song              # write the four-scene track to P9 (pads from the newest chop JSON)
    uv run python scripts/boom_bap.py render            # render_song with nobody at the device, print the grades
    uv run python scripts/boom_bap.py all

Every stage is checkpointed by the server (P9 is the scratch slot); nothing here asks anyone to touch the device.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time

from ko_bridge import config as C
from ko_bridge import groove as GR
from ko_bridge import server as srv
from ko_bridge.project import FaderBase, FxSpec, Lane, LanePoint, NoteEvent, PadSpec, PatternSpec, ProjectSpec, SceneSpec

# ----------------------------------------------------------------------------- the source phrase

SRC_BPM = 107.0           # the band plays here ...
SCREW = -3.0              # ... and the record is slowed three semitones: 107 * 2^(-3/12) = 89.98 BPM
BPM = 90.0                # the song tempo (0.02 % off the screwed loop, 1.5 ms over two bars)
S = 16                    # steps per bar
CK = 1 / 6                # one MIDI clock pulse (4 ticks) in steps: the only sub-step offset the Mac's 24 PPQN clock can place exactly
LOOPS = C.folder("library", "loops")
CHOPS = C.folder("library", "chops")

SOURCE_PADS = [
    PadSpec(group="A", pad="1", slot=340, amplitude=54),                                        # shaker (texture baked into the chops)
    PadSpec(group="B", pad="1", slot=403, amplitude=86, play_mode="key", release=100),           # e bass round, A1
    PadSpec(group="C", pad="1", slot=515, amplitude=90, play_mode="key", release=110),           # epiano 360, C3 (three pads = three voices)
    PadSpec(group="C", pad="2", slot=515, amplitude=90, play_mode="key", release=110),
    PadSpec(group="C", pad="3", slot=515, amplitude=90, play_mode="key", release=110),
    PadSpec(group="C", pad="4", slot=502, amplitude=86),                                        # wurli clean, G3
    PadSpec(group="C", pad="5", slot=503, amplitude=78),                                        # mute strato, G2
]


def n(pad: str, step: float, semis: int = 0, steps: float = 1.0, vel: int = 100) -> NoteEvent:
    return NoteEvent(pad=pad, step=step, semitones=semis, velocity=vel, duration=max(1, min(384, int(round(steps * 24)))))


def source_keys() -> PatternSpec:
    """C minor, two bars, one distinct thing per beat so each grid chop has its own face."""
    e = []
    chords = [(0, (3, 7, 10), 3.5, 104),      # beat 1  Cm7   E♭3 G3 B♭3
              (8, (2, 7, 10), 3.5, 96),       # beat 3  Cm9 colour  D3 G3 B♭3
              (16, (-4, 0, 3), 3.5, 102),     # beat 5  Fm7   A♭2 C3 E♭3
              (24, (-5, -1, 5), 3.5, 100)]    # beat 7  G7    G2 B2 F3
    for step, offsets, length, vel in chords:
        e += [n(pad, step, off, length, vel - 4 * i) for i, (pad, off) in enumerate(zip(("1", "2", "3"), offsets))]
    e += [n("4", 4, 8, 1, 92), n("4", 6, 7, 1, 86),            # beat 2  wurli E♭4, D4
          n("4", 12, 5, 1, 90), n("4", 14, 3, 1, 84),          # beat 4  wurli C4, B♭3
          n("4", 20, 8, 1, 90), n("4", 22, 10, 1, 86),         # beat 6  wurli E♭4, F4
          n("4", 28, 3, 1, 90), n("4", 28 + 4 * CK, 1, 1, 86), n("4", 29 + 2 * CK, 0, 1, 84), n("4", 30, -2, 1, 88)]   # beat 8  run B♭3 A♭3 G3 F3
    e += [n("5", 2, 5, 1, 84), n("5", 10, 5, 1, 80), n("5", 18, -2, 1, 84), n("5", 26, 0, 1, 82)]   # guitar chucks on the 'and' of 1, 3, 5, 7
    return PatternSpec(group="C", number=1, bars=2, events=e)


def source_bass() -> PatternSpec:
    e = [n("1", 0, 3, 3.5, 104), n("1", 6, 3, 1.5, 78), n("1", 8, 3, 3.5, 100), n("1", 12, 3, 2, 84), n("1", 14, 6, 2, 88),   # C2 ... E♭2
         n("1", 16, 8, 3.5, 104), n("1", 20, 8, 1.5, 92), n("1", 22, 11, 1.5, 80),                                        # F2 ... A♭2
         n("1", 24, -2, 3.5, 102), n("1", 28, -2, 2, 90), n("1", 30, 1, 2, 86)]                                             # G1 ... B♭1
    return PatternSpec(group="B", number=1, bars=2, events=e)


def source_shaker() -> PatternSpec:
    e = [n("1", st + (CK if st % 4 else 0), 0, 1, 70 if st % 4 == 0 else 50) for st in range(0, 2 * S, 2)]
    return PatternSpec(group="A", number=1, bars=2, events=e)


def source_spec() -> ProjectSpec:
    return ProjectSpec(bpm=SRC_BPM, pads=SOURCE_PADS,
                       patterns=[source_shaker(), source_bass(), source_keys(), PatternSpec(group="D", number=1, bars=2, events=[])],
                       scenes=[SceneSpec(patterns={"A": 1, "B": 1, "C": 1, "D": 1})], song=[1], current_scene=1,
                       fader_bases=[FaderBase(group="A", parameter="fx", value=0.3), FaderBase(group="B", parameter="fx", value=0.0),
                                    FaderBase(group="C", parameter="fx", value=0.2), FaderBase(group="D", parameter="fx", value=0.0)],
                       # clear_sidechain: the riddim left A "1"/"2" ducking B and C in P9, which gated the first recording on every shaker hit
                       fx=FxSpec(type="reverb", x=0.3, y=0.45, output_compressor_drive=0.1, output_compressor_speed=0.5, clear_sidechain=True))


def stage_source(project: int) -> str:
    import numpy as np
    import soundfile as sf

    s = source_spec()
    print("source records:", {p.id: len(p.events) for p in s.patterns})
    w = srv.write_project(project, s, wait=True)
    print("write:", {k: w[k] for k in ("bytes", "write_seconds", "identical", "checkpoint")})
    r = srv.play_pattern(bpm=SRC_BPM, bars=2, record=True, name="boombap-src", wait=True)
    take = r["take"]
    g = r.get("grade", {})
    high = [ev for ev in g.get("events", []) if ev.get("band") == "high"]
    print("source take:", take["take_id"], "| midi note-ons", r["midi_note_ons"], "of", r["expected_events"], "| matched", g.get("matched"), "high-band", len(high),
          "| peak", take.get("peak_dbfs"), "dBFS", "| warning", r.get("warning") or take.get("warning"))
    audio, sr = sf.read(take["wav"], dtype="float32", always_2d=True)
    start = int(srv.get_take(take["take_id"])["start_sample"])
    first = [ev["offset_ms"] for ev in g.get("events", []) if ev.get("expected_s") == 0.0 and ev.get("offset_ms") is not None]
    lead_ms = min(first) if first and 0.0 <= min(first) <= 15.0 else srv._cfg().latency_ms
    a = start + int(round(lead_ms / 1000.0 * sr))
    length = int(round(2 * 240.0 / SRC_BPM * sr))
    loop = audio[a: a + length]
    if len(loop) < length:
        raise SystemExit(f"the take is too short for a two-bar loop ({len(loop)} of {length} frames)")
    path = os.path.join(LOOPS, f"boombap-src-{time.strftime('%Y%m%d-%H%M%S')}.wav")
    sf.write(path, loop, sr, subtype="PCM_24")
    print(f"loop: {path} ({length / sr:.3f} s from {lead_ms:.1f} ms after START, peak {20 * np.log10(max(1e-9, float(np.max(np.abs(loop))))):.1f} dBFS)")
    return path


# ----------------------------------------------------------------------------- the chop

def newest(pattern: str) -> str:
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not files:
        raise SystemExit(f"nothing matches {pattern}; run the earlier stage first")
    return files[-1]


def stage_chop(loop: str | None, dry_run: bool) -> str | None:
    loop = loop or newest(os.path.join(LOOPS, "boombap-src-*.wav"))
    kw = dict(wav_path=loop, group="C", mode="hybrid", num_slices=12, screw_semitones=SCREW, source_bpm=SRC_BPM, target_bpm=BPM,
              samplerate=46875, mono=False, reverse_last_pad=True, shared_slot=True)
    r = srv.chop_sample(upload=not dry_run, wait=True, **kw)
    info = r["audio_info"]
    print("audio:", {k: info[k] for k in ("frames", "seconds", "channels", "effective_semitones", "speed_ratio")}, "| slot", r["slot"])
    beat = info["seconds"] / 8.0
    for sl in r["slices"]:
        print(f"  pad {sl['pad']:>5} {sl['kind']:<9} {sl['start_s']:6.3f}-{sl['end_s']:6.3f} s  beat {sl['start_s'] / beat + 1:5.2f}  "
              f"len {sl['duration_s']:.3f} s  rms {sl['rms_db']:6.1f} dB  peak {sl['peak_db']:6.1f}{'  reverse' if sl['reverse'] else ''}")
    if dry_run:
        return None
    up = r["upload"]
    print("upload:", {k: up.get(k) for k in ("slot", "bytes", "upload_seconds", "readback_seconds", "identical", "name")})
    path = os.path.join(CHOPS, f"boombap-{time.strftime('%Y%m%d-%H%M%S')}.json")
    C.write_json(path, {"loop": loop, "slot": r["slot"], "audio_info": info, "slices": r["slices"], "pads": r["pads"], "screw_semitones": SCREW,
                        "source_bpm": SRC_BPM, "bpm": BPM})
    print("chop json:", path)
    return path


# ----------------------------------------------------------------------------- the song

RNG = random.Random(90)
LATE = CK                 # the lazy pocket: snares one clock pulse (4 ticks, 11 ms at 90 BPM) behind the grid

SONG_PADS = [
    # v2 levels: v1 read 12 dB between the mid and high bands (the jungle Tyler likes has 5.6) and 10.6 dB between low and mid
    # (jungle 8.8): hats up and brighter, chops up with their low end thinned, sub down a notch
    PadSpec(group="A", pad="1", slot=26, amplitude=112),                 # kick sp: body at 39.7 Hz (E1), in key; "kick thud" sat at 68 Hz against the A1 sub
    PadSpec(group="A", pad="2", slot=10, amplitude=98),                  # kick bump (the double)
    PadSpec(group="A", pad="3", slot=110, amplitude=110),                # snare fat
    PadSpec(group="A", pad="4", slot=121, amplitude=98),                 # snare vinyl (crack layer)
    PadSpec(group="A", pad="5", slot=204, amplitude=114),                # closed hat mid (very bright)
    PadSpec(group="A", pad="6", slot=203, amplitude=104),                # closed hat lo (the other hand)
    PadSpec(group="A", pad="7", slot=228, amplitude=96),                 # open hat lo
    PadSpec(group="A", pad="8", slot=137, amplitude=90),                 # rimshot vinyl (ghosts)
    PadSpec(group="A", pad="9", slot=248, amplitude=70),                 # crash
    PadSpec(group="A", pad=".", slot=306, amplitude=88),                 # clap vinyl
    PadSpec(group="A", pad="0", slot=325, amplitude=66),                 # tamb on the backbeat (at 78 it peaked level with the snare)
    PadSpec(group="B", pad="1", slot=407, amplitude=90, play_mode="key", release=110, choke=False),   # upright sub; never in the mute group
    PadSpec(group="D", pad="1", slot=553, amplitude=90),                 # vox s6i2 eh (F3)
    PadSpec(group="D", pad="2", slot=548, amplitude=84),                 # vox s6i2 dark (D#3)
    PadSpec(group="D", pad="3", slot=512, amplitude=88, play_mode="key", release=120),   # trumpet breezy (F3)
]
CHOP_AMPLITUDE = 92
CHOP_HIGHPASS = 0.18      # thins the bass baked into the chops; the sub owns the low end


def hv(v: int, spread: int = 6) -> int:
    return max(1, min(127, v + RNG.randint(-spread, spread)))


def hit(bar: int, step: float, pad: str, vel: int = 100, humanize: bool = True) -> NoteEvent:
    return NoteEvent(pad=pad, step=bar * S + step, velocity=hv(vel) if humanize else vel, duration=24)


# The library hears "upright sub" at F#1 (45.9 Hz); an FFT of its PCM peaks at 48.8 Hz, a G1, and written in concert pitch the
# sub came out a semitone sharp against the bass baked into the chops (58 Hz against 55 Hz in the v1 render). Offsets from G1:
SUB_SEMITONES = {"G1": 0, "A1": 2, "D2": 7, "E1": -3}


def sub(bar: int, step: float, name: str, steps: float, vel: int = 100) -> NoteEvent:
    return NoteEvent(pad="1", step=bar * S + step, semitones=SUB_SEMITONES[name], velocity=hv(vel, 4), duration=max(1, min(384, int(round(steps * 24)))))


def chop(bar: int, step: float, pad: str, vel: int = 104) -> NoteEvent:
    return NoteEvent(pad=pad, step=bar * S + step, velocity=hv(vel, 5), duration=24)


def lane(parameter: str, points: list[tuple[float, float]], bars: int) -> Lane:
    """A fader lane with the 120 ms slew lead applied (points as (step, value))."""
    raw = [LanePoint(step=st, value=v) for st, v in points]
    return Lane(parameter=parameter, points=GR.compensate_fader_slew(raw, bars=bars, bpm=BPM))


# drums: a two-bar boom-bap phrase, bars 0-1 and 2-3 differ in their second bar

def drum_bar(e: list, bar: int, variant: int, hats: bool = True, kick: bool = True, snare: bool = True, fill: bool = False, ghosts: bool = True, crash: bool = False):
    if crash:
        e.append(hit(bar, 0, "9", 96))
    if kick:
        kicks = {0: [(0, 124), (6, 102), (10, 114)], 1: [(0, 124), (7, 98), (10, 112), (14, 88)],
                 2: [(0, 124), (3, 92), (10, 114)], 3: [(0, 124), (6, 104), (11, 96), (13, 110)]}[variant % 4]
        e += [hit(bar, st, "1", v) for st, v in kicks]
        e.append(hit(bar, 0, "2", 104))                                             # the bump under the one
    if snare:
        e += [hit(bar, 4 + LATE, "3", 118), hit(bar, 12 + LATE, "3", 116), hit(bar, 4 + LATE, "4", 96), hit(bar, 12 + LATE, "4", 92)]
        e += [hit(bar, 4 + LATE, "0", 82), hit(bar, 12 + LATE, "0", 78)]                           # tambourine with the snare
        if variant % 2 == 1:
            e.append(hit(bar, 12 + LATE, ".", 84))                                  # clap on the four of odd bars
    if ghosts:
        g = {0: [(7, 56), (11, 50), (15, 62)], 1: [(2, 46), (7, 58), (15, 60)], 2: [(7, 54), (9, 44), (15, 64)], 3: [(3, 48), (7, 58), (11, 52)]}[variant % 4]
        e += [hit(bar, st + CK, "8", v) for st, v in g]
    if hats:
        for st in range(0, S, 2):
            pad = "5" if st % 4 == 0 else "6"
            e.append(hit(bar, st + (CK if st % 4 else 0), pad, 100 if st % 4 == 0 else 70))
        extra = {0: [(3, 56), (11, 58)], 1: [(7, 54), (11, 58), (15, 60)], 2: [(3, 56), (13, 54)], 3: [(7, 54), (11, 58)]}[variant % 4]
        e += [hit(bar, st + CK, "6", v) for st, v in extra]
        if variant % 2 == 1:
            e.append(hit(bar, 14 + CK, "7", 88))                                    # open hat on the 'and' of four
    if fill:
        e += [hit(bar, 13, "3", 84), hit(bar, 14, "3", 96), hit(bar, 14 + 3 * CK, "4", 90), hit(bar, 15, "3", 112), hit(bar, 15, "1", 110)]


def drums_main() -> PatternSpec:
    e = []
    for bar in range(4):
        drum_bar(e, bar, bar, fill=(bar == 3), crash=(bar == 0))
    return PatternSpec(group="A", number=1, bars=4, events=e, lanes=[Lane(parameter="level", points=[LanePoint(step=0, value=0.92)])])


def drums_intro() -> PatternSpec:
    e = []
    for bar in range(4):
        if bar < 2:
            e += [hit(bar, st, "6", 62 if st % 4 else 80) for st in range(0, S, 2)]                     # bare hats, no swing yet
        else:
            drum_bar(e, bar, bar, kick=(bar == 3), snare=False, ghosts=(bar == 3), fill=(bar == 3))
    e += [hit(3, 4 + LATE, "8", 90), hit(3, 12 + LATE, "3", 108)]                                    # rim on the two, the snare arrives on the four
    return PatternSpec(group="A", number=2, bars=4, events=e,
                       lanes=[Lane(parameter="level", points=[LanePoint(step=0, value=0.8), LanePoint(step=2 * S - 2, value=0.88), LanePoint(step=3 * S - 2, value=0.92)])])


def drums_flip() -> PatternSpec:
    e = []
    for bar in range(4):
        drum_bar(e, bar, bar + 2, fill=(bar == 3))
    e += [hit(1, 10 + CK, "7", 86), hit(3, 6 + CK, "2", 96)]                                        # one more open hat, one more bump
    e = GR.apply_lazy_pocket(e, snare_pads=(), hat_pads=(), bars=4)                                  # (snares are already placed late above)
    return PatternSpec(group="A", number=3, bars=4, events=e, lanes=[Lane(parameter="level", points=[LanePoint(step=0, value=0.94)])])


def drums_outro() -> PatternSpec:
    e = []
    for bar in range(3):
        drum_bar(e, bar, bar, hats=True, kick=True, snare=(bar < 2), ghosts=(bar < 2))
    e += [hit(2, 12 + LATE, "8", 84), hit(3, 0, "1", 120), hit(3, 0, "9", 90)]                       # a rim where the last snare was, then the final one
    return PatternSpec(group="A", number=4, bars=4, events=e,
                       lanes=[Lane(parameter="level", points=[LanePoint(step=0, value=0.9), LanePoint(step=2 * S - 2, value=0.8), LanePoint(step=3 * S - 2, value=0.7)])])


# chops: pads 1-8 are beats 1-8 of the screwed loop (A minor: Am7 | wurli | Am9 | wurli | Dm7 | wurli | E7 | run), 9 0 . are accents, enter is the run reversed
# the bass baked into each chop: beats 1-4 A, 5-6 D, 7-8 E; the sub below follows the chop that owns each half bar

def chops_main() -> PatternSpec:
    e = []
    for pair in (0, 2):
        b1, b2 = pair, pair + 1
        e += [chop(b1, 0, "1", 110), chop(b1, 3, "2", 98), chop(b1, 6, "3", 104), chop(b1, 8, "1", 106), chop(b1, 11, "4", 100), chop(b1, 14, "9", 96)]
        e += [chop(b2, 0, "5", 110), chop(b2, 4, "6", 98), chop(b2, 6, "5", 102), chop(b2, 8, "7", 108), chop(b2, 12, "8", 100)]
        e.append(chop(b2, 14, "enter" if pair == 2 else ".", 98))                                  # the turnaround: accent, then the reversed run
    return PatternSpec(group="C", number=1, bars=4, events=e, lanes=chop_lanes())


def chop_lanes(lowpass: list[tuple[float, float]] | None = None) -> list[Lane]:
    lanes = [lane("lowpass", lowpass, 4) if lowpass else Lane(parameter="lowpass", points=[LanePoint(step=0, value=1.0)]),
             Lane(parameter="highpass", points=[LanePoint(step=0, value=CHOP_HIGHPASS)]), Lane(parameter="level", points=[LanePoint(step=0, value=1.0)])]
    return lanes


def chops_intro() -> PatternSpec:
    e = [chop(0, 0, "1", 104), chop(0, 8, "3", 98), chop(1, 0, "5", 104), chop(1, 8, "7", 100), chop(1, 12, "8", 96),
         chop(2, 0, "1", 108), chop(2, 3, "2", 96), chop(2, 6, "3", 102), chop(2, 8, "1", 104), chop(2, 11, "4", 98), chop(2, 14, "9", 94),
         chop(3, 0, "5", 108), chop(3, 4, "6", 96), chop(3, 8, "7", 104), chop(3, 12, "8", 100), chop(3, 14, "enter", 100)]
    return PatternSpec(group="C", number=2, bars=4, events=e, lanes=chop_lanes([(0, 0.32), (S, 0.45), (2 * S, 0.68), (3 * S, 1.0)]))


def chops_flip() -> PatternSpec:
    e = []
    # bar 0: the Am7 stab double-up (a 1/32 crossfader pre-hit), the wurli answer twice, the accent
    e += GR.make_screw_double_up("1", 0, velocity=110)
    e += [chop(0, 4, "2", 96), chop(0, 6, "2", 92), chop(0, 8, "3", 104), chop(0, 10, "0", 96), chop(0, 12, "4", 100), chop(0, 14, "9", 96)]
    # bar 1: Dm7 then the E7 pushed a clock early, run, reversed run
    e += [chop(1, 0, "5", 110), chop(1, 3, "6", 96), chop(1, 6, "5", 100), chop(1, 8 - CK, "7", 108), chop(1, 11, "7", 96), chop(1, 12, "8", 100), chop(1, 14, "enter", 100)]
    # bar 2: triplet roll into the Am9 colour, accents traded
    e += GR.make_screw_double_up("3", 4, velocity=106, style="triplet_roll")
    e += [chop(2, 0, "1", 110), chop(2, 8, "1", 104), chop(2, 10, ".", 96), chop(2, 12, "4", 100), chop(2, 13, "0", 90), chop(2, 14, "9", 94)]
    # bar 3: the E7 held, the run answered by its mirror, silence on the last 8th before the next scene
    e += [chop(3, 0, "7", 110), chop(3, 4, "5", 100), chop(3, 6, "6", 96), chop(3, 8, "7", 106), chop(3, 10, "8", 98), chop(3, 12, "enter", 104)]
    return PatternSpec(group="C", number=3, bars=4, events=e, lanes=chop_lanes())


def chops_outro() -> PatternSpec:
    e = [chop(0, 0, "1", 108), chop(0, 6, "3", 100), chop(0, 8, "1", 102), chop(0, 12, "4", 96),
         chop(1, 0, "5", 106), chop(1, 8, "7", 100), chop(1, 12, "8", 96), chop(1, 14, "enter", 96),
         chop(2, 0, "1", 104), chop(2, 8, "3", 96), chop(3, 0, "enter", 100)]
    return PatternSpec(group="C", number=4, bars=4, events=e, lanes=chop_lanes([(0, 1.0), (S, 0.75), (2 * S, 0.5), (3 * S, 0.3)]))


# sub: follows the chops' baked bass (A under beats 1-4, D under 5-6, E under 7-8 of the loop)

def sub_main(number: int, notes: list[tuple[int, float, str, float, int]]) -> PatternSpec:
    e = [sub(b, st, name, ln, v) for b, st, name, ln, v in notes]
    return PatternSpec(group="B", number=number, bars=4, events=e, lanes=[Lane(parameter="lowpass", points=[LanePoint(step=0, value=0.6)]), Lane(parameter="level", points=[LanePoint(step=0, value=0.9)])])


SUB_MAIN = [(0, 0, "A1", 6, 104), (0, 8, "A1", 5, 98), (0, 14, "G1", 2, 90), (1, 0, "D2", 7, 104), (1, 8, "E1", 7, 100),
            (2, 0, "A1", 6, 104), (2, 8, "A1", 5, 98), (2, 14, "G1", 2, 90), (3, 0, "D2", 7, 104), (3, 8, "E1", 6, 100), (3, 14, "E1", 2, 92)]
SUB_INTRO = [(2, 0, "A1", 8, 96), (2, 8, "A1", 6, 92), (3, 0, "D2", 7, 100), (3, 8, "E1", 7, 96)]
SUB_FLIP = [(0, 0, "A1", 3, 104), (0, 3 + CK, "A1", 4, 92), (0, 8, "A1", 6, 100), (1, 0, "D2", 7, 104), (1, 8 - CK, "E1", 7, 100),
            (2, 0, "A1", 7, 104), (2, 8, "A1", 6, 98), (3, 0, "E1", 4, 104), (3, 4, "D2", 4, 98), (3, 8, "E1", 6, 100)]
SUB_OUTRO = [(0, 0, "A1", 6, 100), (0, 8, "A1", 6, 94), (1, 0, "D2", 7, 100), (1, 8, "E1", 7, 96), (2, 0, "A1", 14, 96), (3, 0, "A1", 8, 90)]


def colour(number: int, eh=(), dark=(), horn=()) -> PatternSpec:
    e = [NoteEvent(pad="1", step=b * S + st, velocity=hv(92, 4), duration=24) for b, st in eh]
    e += [NoteEvent(pad="2", step=b * S + st, velocity=hv(86, 4), duration=24) for b, st in dark]
    e += [NoteEvent(pad="3", step=b * S + st, concert=name, velocity=hv(90, 4), duration=max(1, int(ln * 24))) for b, st, name, ln in horn]
    return PatternSpec(group="D", number=number, bars=4, events=e, lanes=[Lane(parameter="fx", points=[LanePoint(step=0, value=0.45)])])


def song_spec(chop_pads: list[PadSpec]) -> ProjectSpec:
    pads = SONG_PADS + [p.model_copy(update={"amplitude": CHOP_AMPLITUDE}) for p in chop_pads]
    patterns = [drums_main(), drums_intro(), drums_flip(), drums_outro(),
                sub_main(1, SUB_MAIN), sub_main(2, SUB_INTRO), sub_main(3, SUB_FLIP), sub_main(4, SUB_OUTRO),
                chops_main(), chops_intro(), chops_flip(), chops_outro(),
                colour(1, eh=[(3, 14)], dark=[(1, 12)]), colour(2, dark=[(3, 10)]), colour(3, eh=[(1, 14), (3, 6)], horn=[(2, 0, "E4", 3), (2, 4, "C4", 3), (2, 8, "A3", 6)]),
                colour(4, dark=[(2, 8)])]
    scenes = [SceneSpec(patterns={"A": 2, "B": 2, "C": 2, "D": 2}),   # 1 intro: filtered chops, hats, the kit arrives in bar 4
              SceneSpec(patterns={"A": 1, "B": 1, "C": 1, "D": 1}),   # 2 the beat
              SceneSpec(patterns={"A": 3, "B": 3, "C": 3, "D": 3}),   # 3 the flip: double-ups, reversed runs, the horn
              SceneSpec(patterns={"A": 4, "B": 4, "C": 4, "D": 4})]   # 4 outro: filter closing, last hit
    return ProjectSpec(bpm=BPM, pads=pads, patterns=patterns, scenes=scenes, song=[1, 2, 3, 2, 4], current_scene=2,
                       fader_bases=[FaderBase(group="A", parameter="fx", value=0.1), FaderBase(group="B", parameter="fx", value=0.0),
                                    FaderBase(group="C", parameter="fx", value=0.28), FaderBase(group="D", parameter="fx", value=0.45)],
                       fx=FxSpec(type="reverb", x=0.4, y=0.5, output_compressor_drive=0.2, output_compressor_speed=0.5, clear_sidechain=True))


def check_grid(spec: ProjectSpec) -> None:
    """Every event and lane point on the 4-tick MIDI clock grid; chord-free group C (one voice, the mute group)."""
    bad = [(p.id, ev.tick_value()) for p in spec.patterns for ev in p.events if ev.tick_value() % GR.CLOCK_TICKS]
    bad += [(p.id, pt.tick_value()) for p in spec.patterns for l in p.lanes for pt in l.points if pt.tick_value() % GR.CLOCK_TICKS]
    if bad:
        raise SystemExit(f"off-grid events: {bad[:10]}")


def load_chop(path: str | None) -> tuple[dict, list[PadSpec]]:
    path = path or newest(os.path.join(CHOPS, "boombap-*.json"))
    data = C.read_json(path)
    return data, [PadSpec.model_validate(p) for p in data["pads"]]


def stage_song(project: int, chop_json: str | None) -> None:
    data, chop_pads = load_chop(chop_json)
    s = song_spec(chop_pads)
    check_grid(s)
    print("chop slot", data["slot"], "| records per pattern:", {p.id: len(p.events) + sum(len(l.points) for l in p.lanes) for p in s.patterns})
    w = srv.write_project(project, s, wait=True)
    print("write:", {k: w[k] for k in ("bytes", "write_seconds", "identical", "checkpoint")})
    after = w["project_after"]
    print("scenes:", [(sc["scene"], sc["patterns"]) for sc in after["scenes"]], "| song", after["song"], "| bpm", after["bpm"])
    cpads = {k: (v["slot"], v["trim_start"], v["trim_end"], v["time_mode"], v["choke"]) for k, v in after["pads"].items() if k.startswith("C:")}
    print("chop pads:", cpads)


def stage_render(project: int, name: str, positions: list[int] | None) -> None:
    r = srv.render_song(project, positions=positions, name=name, wait=True)
    print("song:", r["song"]["take_id"], r["song"]["wav"], f"{r['seconds']} s", "peak", r["song"].get("peak_dbfs"), "dBFS")
    for p in r["positions"]:
        g = srv.get_take(p["take_id"]).get("grade", {})
        high = [ev["offset_ms"] for ev in g.get("events", []) if ev.get("band") == "high" and ev.get("offset_ms") is not None]
        import numpy as np
        hs = f"{len(high)} high-band, mean {np.mean(high):.1f} ms, std {np.std(high):.1f} ms" if high else "no high-band matches"
        mix = g.get("spectral_mix", {})
        print(f"  pos {p['position']} scene {p['scene']} {p['bars']} bars: midi {p['midi_note_ons']}/{p['expected_events']} | {hs} | "
              f"peak {g.get('peak_dbfs')} | bands {mix.get('bands_dbfs')} | crest {mix.get('crest_factor_db')} | clipped {mix.get('clipped_samples')} | {p.get('warning') or ''}")
        for wmsg in mix.get("warnings", []):
            print("     ", wmsg)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["source", "chop", "song", "render", "all"])
    ap.add_argument("--project", type=int, default=9)
    ap.add_argument("--loop", help="loop WAV for the chop stage (default: the newest boombap-src-*.wav)")
    ap.add_argument("--chop-json", help="chop description for the song stage (default: the newest)")
    ap.add_argument("--dry-run", action="store_true", help="chop: slice offline without uploading")
    ap.add_argument("--name", default="boombap")
    ap.add_argument("--positions", type=int, nargs="*", help="render: song positions (default: the whole song)")
    args = ap.parse_args()
    if args.stage in ("source", "all"):
        args.loop = stage_source(args.project)
    if args.stage in ("chop", "all"):
        args.chop_json = stage_chop(args.loop, args.dry_run) or args.chop_json
        if args.dry_run:
            return 0
    if args.stage in ("song", "all"):
        stage_song(args.project, args.chop_json)
    if args.stage in ("render", "all"):
        stage_render(args.project, args.name, args.positions)
    return 0


if __name__ == "__main__":
    sys.exit(main())
