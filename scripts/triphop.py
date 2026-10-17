#!/usr/bin/env python3
"""Trip hop in F minor at 75 BPM from four Lyria pieces, chopped and screwed over the EP-133's own drums and sub bass.

    uv run python scripts/triphop.py --dry-run          # pick, cut, screw and slice the pieces; print the plan; no device
    uv run python scripts/triphop.py                    # upload the four chop slots, write P9, render the song hands-free
    uv run python scripts/triphop.py --render-only      # P9 already written: render again

The pieces (scripts/lyria_pieces.py, set "triphop") came back in different keys and tempi, as Lyria does: strings and
trumpet in F minor (about 74.5 and 87 BPM), guitar in E minor (88), the voice around E-flat minor. The song keeps the
strings' tempo and key, screws the guitar up one semitone and the voice two so all four meet in F minor, and plays
everything but the string bed as short retriggered chops, so tempo differences never accumulate. Group A is factory drums,
B the upright sub plus guitar stabs, C the strings bed (one shared slot, twelve chops, the mute group), D voice and trumpet
phrases in key mode so note length, not the mute group, ends them.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import soundfile as sf

from ko_bridge import analysis as an
from ko_bridge import chopper as CH
from ko_bridge import groove as GR
from ko_bridge import loops
from ko_bridge.project import (FaderBase, FxSpec, Lane, LanePoint, NoteEvent, PadSpec, PatternSpec, ProjectSpec, SceneSpec,
                               SidechainSource)

BPM = 75.0
BARS = 4
S = 16                       # steps per bar
PROJECT = int(os.environ.get("KO_BRIDGE_SCRATCH_PROJECT", "9"))
WAV_DIR = os.path.expanduser("~/Music/ko-bridge/library/wav")
NOTES = os.path.expanduser("~/Music/ko-bridge/notes")
SR = 46875

# piece -> how it is cut, screwed and sliced
PIECES = {
    "strings": {"bars": 4, "tempo_hint": 86.0, "window_bpm": None, "screw": 0.0, "mono": False, "slices": 12, "mode": "hybrid",
                "group": "C", "pads": CH.CHOP_PAD_ORDER, "play_mode": "oneshot", "choke": True, "release": 255, "amplitude": 92},
    "vocal":   {"bars": 4, "tempo_hint": 86.0, "window_bpm": BPM, "screw": 2.0, "mono": True, "slices": 6, "mode": "transients",
                "group": "D", "pads": ("1", "2", "3", "4", "5", "6"), "play_mode": "key", "choke": False, "release": 90, "amplitude": 94},
    "trumpet": {"bars": 4, "tempo_hint": 86.0, "window_bpm": None, "screw": 0.0, "mono": True, "slices": 6, "mode": "transients",
                "group": "D", "pads": ("7", "8", "9", "0", ".", "enter"), "play_mode": "key", "choke": False, "release": 90, "amplitude": 90},
    "guitar":  {"bars": 2, "tempo_hint": 86.0, "window_bpm": None, "screw": 1.0, "mono": True, "slices": 6, "mode": "transients",
                "group": "B", "pads": ("1", "2", "3", "4", "5", "6"), "play_mode": "key", "choke": False, "release": 70, "amplitude": 86},
}

DRUMS = [  # factory slots, picked from the library index for a dusty trip hop kit
    PadSpec(group="A", pad=".", slot=7, amplitude=104),        # kick grit
    PadSpec(group="A", pad="enter", slot=8, amplitude=98),     # kick verb
    PadSpec(group="A", pad="0", slot=121, amplitude=98),       # snare vinyl
    PadSpec(group="A", pad="6", slot=103, amplitude=84),       # snare lo (ghosts)
    PadSpec(group="A", pad="1", slot=203, amplitude=74),       # closed hat lo
    PadSpec(group="A", pad="2", slot=206, amplitude=70),       # closed hat pedal
    PadSpec(group="A", pad="3", slot=133, amplitude=82),       # rim lo
    PadSpec(group="A", pad="4", slot=340, amplitude=60),       # shaker
    PadSpec(group="A", pad="5", slot=325, amplitude=66),       # tamb
    PadSpec(group="A", pad="7", slot=306, amplitude=80),       # clap vinyl
]
SUB = PadSpec(group="B", pad=".", slot=407, amplitude=100, play_mode="key", release=140)   # upright sub; the library hears it a semitone low (true G1)
SUB_ROOT_OFFSET = -2   # semitones from the sub's native pitch to F1


def ev(pad, bar, step, vel=100, beats=1.0, semis=None):
    return NoteEvent(pad=pad, step=bar * S + step, velocity=vel, duration=max(4, min(384, int(round(beats * 24 * 4)))), semitones=semis)


def drums_main() -> list[NoteEvent]:
    out = []
    for bar in range(BARS):
        out += [ev(".", bar, 0, 120), ev(".", bar, 10, 102), ev("0", bar, 4, 110), ev("0", bar, 12, 116)]
        if bar % 2 == 1:
            out += [ev("6", bar, 7, 48), ev("6", bar, 15, 44), ev("7", bar, 12, 72)]
        if bar == 3:
            out += [ev("enter", bar, 14, 96)]
        for h in range(0, 16, 2):
            out.append(ev("1", bar, h, 82 if h % 4 == 0 else 66))
        out += [ev("2", bar, 6, 60), ev("2", bar, 14, 58), ev("3", bar, 11, 70)]
        for s in range(16):
            out.append(ev("4", bar, s, 56 if s % 4 == 0 else 44))
        out += [ev("5", bar, 4, 64), ev("5", bar, 12, 68)]
    return GR.apply_mpc_swing(out, swing=0.58, bars=BARS, snap_clock=True)


def drums_intro() -> list[NoteEvent]:
    out = []
    for bar in range(BARS):
        if bar in (0, 2):
            out.append(ev(".", bar, 0, 108))
        out += [ev("3", bar, 4, 62), ev("3", bar, 12, 66)]
        for s in range(0, 16, 2):
            out.append(ev("4", bar, s, 50 if s % 4 == 0 else 40))
    return GR.apply_mpc_swing(out, swing=0.58, bars=BARS, snap_clock=True)


def drums_break() -> list[NoteEvent]:
    out = []
    for bar in range(BARS):
        out += [ev(".", bar, 0, 112), ev("0", bar, 12, 104), ev("3", bar, 4, 66)]
        for s in range(0, 16, 4):
            out.append(ev("4", bar, s, 48))
        if bar == 3:
            out += [ev("6", bar, 13, 56), ev("6", bar, 14, 62), ev("6", bar, 15, 70)]
    return out


def bass_line(with_guitar: bool, guitar_pads: list[str]) -> list[NoteEvent]:
    out = []
    r = SUB_ROOT_OFFSET
    for bar in range(BARS):
        out += [ev(".", bar, 0, 112, beats=2.5, semis=r), ev(".", bar, 10, 100, beats=1.0, semis=r)]
        if bar % 2 == 1:
            out.append(ev(".", bar, 14, 92, beats=0.5, semis=r - 2))      # E-flat below, leading back to F
        if bar == 3:
            out.append(ev(".", bar, 12, 96, beats=0.5, semis=r + 7))      # C, the fifth, as a pickup
    if with_guitar and guitar_pads:
        stabs = [(2, 0), (6, 1), (11, 2), (14, 3)]
        for bar in range(BARS):
            for step, k in stabs:
                if bar % 2 == 0 or k % 2 == 1:
                    out.append(ev(guitar_pads[k % len(guitar_pads)], bar, step, 88 if k % 2 == 0 else 78, beats=0.75))
    return GR.align_events_to_clock(out, bars=BARS)


def strings_bed(variant: int) -> list[NoteEvent]:
    """Grid chops 1..8 in order, one per half bar, so the bed is rebuilt from its own slices; variant 2 adds accents,
    a double-up stutter and the reversed pickup into the downbeat."""
    out = [ev(str(k + 1), k // 2, (k % 2) * 8, 100, beats=2.0) for k in range(8)]
    if variant == 2:
        out = [e for e in out if not (e.step == 2 * S + 0)]        # bar 3 starts with the stutter instead
        out += GR.make_screw_double_up(pad="5", step=2 * S + 0, velocity=104, style="double_up", duration_steps=8)
        out += [ev("9", 1, 14, 96, beats=0.5), ev("0", 3, 6, 92, beats=0.5), ev("enter", 3, 15, 100, beats=0.25)]
    return GR.align_events_to_clock(out, bars=BARS)


def voice_phrases(vocal_pads: list[str], trumpet_pads: list[str], which: str) -> list[NoteEvent]:
    out = []
    if which in ("vocal", "both"):
        out += [ev(vocal_pads[0], 0, 2, 100, beats=3.0), ev(vocal_pads[2], 1, 4, 96, beats=2.5),
                ev(vocal_pads[4], 2, 2, 100, beats=3.5), ev(vocal_pads[1], 3, 8, 92, beats=2.0)]
    if which in ("trumpet", "both"):
        out += [ev(trumpet_pads[0], 0, 8, 98, beats=2.5), ev(trumpet_pads[1], 1, 10, 94, beats=2.0),
                ev(trumpet_pads[2], 2, 8, 100, beats=3.0), ev(trumpet_pads[5], 3, 12, 96, beats=1.5)]
    if which == "both":   # answer phrases do not overlap the voice
        out = [e for e in out if not (e.pad in trumpet_pads and e.step in (2 * S + 8,))]
    return GR.align_events_to_clock(out, bars=BARS)


def intro_sweep() -> Lane:
    pts = [LanePoint(step=0, value=0.3), LanePoint(step=16, value=0.5), LanePoint(step=32, value=0.72), LanePoint(step=48, value=0.9),
           LanePoint(step=62, value=1.0)]
    return Lane(parameter="lowpass", points=GR.compensate_fader_slew(pts, bars=BARS, bpm=BPM, lead_ms=120))


def prepare_pieces(manifest: dict, stamp: str) -> dict[str, dict]:
    """For every piece: real tempo, strongest window, cut, screw, resample, slice; writes the loop and chop WAVs."""
    by_name = {p["name"]: p for p in manifest["pieces"] if "wav" in p}
    out = {}
    for name, cfg in PIECES.items():
        p = by_name[name]
        x, sr = sf.read(p["wav"], dtype="float32", always_2d=True)
        tempo = loops.estimate_bpm(x, sr, hint=cfg["tempo_hint"])
        wbpm = cfg["window_bpm"] or tempo["bpm"]
        picks = loops.pick_loops(x, sr, bpm=wbpm, bars=cfg["bars"], top=3)
        pick = picks[0]
        loop = loops.cut_loop(x, sr, pick["start_s"], pick["seconds"])
        loop_wav = os.path.join(WAV_DIR, f"triphop-{name}-loop-{stamp}.wav")
        sf.write(loop_wav, loop, sr, subtype="PCM_16")
        audio, pcm, info = CH.prepare_chopped_audio(loop_wav, samplerate=SR, mono=cfg["mono"], normalize_dbfs=-1.0, screw_semitones=cfg["screw"])
        slices = CH.detect_chop_slices(audio, SR, num_slices=cfg["slices"], mode=cfg["mode"],
                                       reverse_last_pad=(cfg["mode"] == "hybrid"), pad_order=tuple(cfg["pads"]))
        chop_wav = os.path.join(WAV_DIR, f"triphop-{name}-chop-{stamp}.wav")
        sf.write(chop_wav, audio, SR, subtype="PCM_16")
        out[name] = {"source": p["wav"], "tempo": tempo["bpm"], "tempo_confidence": tempo["confidence"], "window_bpm": wbpm, "pick": pick,
                     "alternatives": picks[1:], "screw": cfg["screw"], "loop_wav": loop_wav, "chop_wav": chop_wav, "info": info,
                     "slices": slices, "cfg": cfg}
        print(f"{name:8s} tempo {tempo['bpm']:6.2f} (conf {tempo['confidence']:.2f})  window {pick['seconds']:5.2f} s from {pick['start_s']:6.2f} s "
              f"rms {pick['rms_dbfs']} dBFS, {pick['high_onsets']} high onsets  -> screw {cfg['screw']:+.1f} -> {info['seconds']} s, {info['frames']} frames, "
              f"{info['channels']} ch, {len(slices)} slices", file=sys.stderr)
    return out


def build_spec(prep: dict[str, dict], slots: dict[str, int]) -> ProjectSpec:
    pads: list[PadSpec] = list(DRUMS) + [SUB]
    for name, pr in prep.items():
        cfg = pr["cfg"]
        chop_pads = CH.build_shared_slot_chop_pads(pr["slices"], slot=slots[name], group=cfg["group"], sample_bpm=BPM,
                                                   amplitude=cfg["amplitude"], choke=cfg["choke"], play_mode=cfg["play_mode"])
        pads += [cp.model_copy(update={"release": cfg["release"]}) for cp in chop_pads]
    vocal_pads, trumpet_pads, guitar_pads = list(PIECES["vocal"]["pads"]), list(PIECES["trumpet"]["pads"]), list(PIECES["guitar"]["pads"])
    patterns = [
        PatternSpec(group="A", number=1, bars=BARS, events=drums_intro()),
        PatternSpec(group="A", number=2, bars=BARS, events=drums_main()),
        PatternSpec(group="A", number=3, bars=BARS, events=drums_break()),
        PatternSpec(group="B", number=1, bars=BARS, events=bass_line(False, guitar_pads)),
        PatternSpec(group="B", number=2, bars=BARS, events=bass_line(True, guitar_pads)),
        PatternSpec(group="B", number=4, bars=1, events=[]),
        PatternSpec(group="C", number=1, bars=BARS, events=strings_bed(1), lanes=[intro_sweep()]),
        PatternSpec(group="C", number=2, bars=BARS, events=strings_bed(2)),
        PatternSpec(group="C", number=3, bars=BARS, events=strings_bed(1)),
        PatternSpec(group="D", number=1, bars=BARS, events=voice_phrases(vocal_pads, trumpet_pads, "vocal")),
        PatternSpec(group="D", number=2, bars=BARS, events=voice_phrases(vocal_pads, trumpet_pads, "trumpet")),
        PatternSpec(group="D", number=3, bars=BARS, events=voice_phrases(vocal_pads, trumpet_pads, "both")),
        PatternSpec(group="D", number=4, bars=1, events=[]),
    ]
    scenes = [
        SceneSpec(patterns={"A": 1, "B": 4, "C": 1, "D": 4}),   # 1 intro: strings opening up, sparse drums
        SceneSpec(patterns={"A": 2, "B": 1, "C": 3, "D": 1}),   # 2 verse: beat, sub, bed, voice
        SceneSpec(patterns={"A": 2, "B": 2, "C": 2, "D": 3}),   # 3 chorus: guitar stabs, stutters, voice and trumpet
        SceneSpec(patterns={"A": 3, "B": 1, "C": 3, "D": 2}),   # 4 break: half drums, trumpet alone
        SceneSpec(patterns={"A": 1, "B": 4, "C": 2, "D": 2}),   # 5 outro: strings with the reversed pickup, trumpet
    ]
    return ProjectSpec(
        bpm=BPM, pads=pads, patterns=patterns, scenes=scenes, song=[1, 2, 3, 2, 4, 3, 5], current_scene=1,
        fader_bases=[FaderBase(group="A", parameter="level", value=0.95), FaderBase(group="B", parameter="level", value=0.9),
                     FaderBase(group="C", parameter="lowpass", value=1.0), FaderBase(group="D", parameter="fx", value=0.45)],
        fx=FxSpec(type="reverb", x=0.45, y=0.5, output_compressor_drive=0.25, output_compressor_speed=0.5,
                  sidechain_sources=[SidechainSource(group="A", pad=".")], sidechain_destinations=["C"], sidechain_length=0.3, sidechain_shape=0.5),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Trip hop from four Lyria pieces on the EP-133")
    ap.add_argument("--manifest", default=None, help="lyria_pieces manifest (default: the newest triphop one)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--render-only", action="store_true")
    ap.add_argument("--name", default="triphop-v1")
    a = ap.parse_args()
    manifest_path = a.manifest or sorted(glob.glob(os.path.join(NOTES, "lyria-triphop-*.json")))[-1]
    manifest = json.load(open(manifest_path))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    note: dict = {"manifest": manifest_path, "bpm": BPM, "key": "F minor", "time": stamp}
    from ko_bridge import server as srv
    if not a.render_only:
        prep = prepare_pieces(manifest, stamp)
        note["pieces"] = {n: {k: v for k, v in pr.items() if k not in ("slices", "cfg")} | {"slices": [s.model_dump() for s in pr["slices"]]} for n, pr in prep.items()}
        if a.dry_run:
            spec = build_spec(prep, {n: 900 + i for i, n in enumerate(prep)})
            print(json.dumps({"patterns": [(p.id, p.bars, len(p.events), [l.parameter for l in p.lanes]) for p in spec.patterns],
                              "pads": len(spec.pads), "scenes": [s.patterns for s in spec.scenes], "song": spec.song,
                              "chops": {n: [(s.pad, s.start_frame, s.end_frame - s.start_frame, s.kind) for s in pr["slices"]] for n, pr in prep.items()}}, indent=1))
            return 0
        slots: dict[str, int] = {}
        for name, pr in prep.items():
            up = srv.upload_sound(pr["chop_wav"], name=f"th {name}"[:16], samplerate=SR, mono=pr["cfg"]["mono"], wait=True)
            slots[name] = int(up["slot"])
            print(f"uploaded {name} -> slot {slots[name]} ({up.get('bytes')} B, verified {up.get('verified', up.get('identical'))})", file=sys.stderr)
        note["slots"] = slots
        spec = build_spec(prep, slots)
        wr = srv.write_project(PROJECT, spec, reload=True, wait=True)
        note["write"] = {k: wr.get(k) for k in ("identical", "payloads_identical", "checkpoint", "bytes", "write_seconds")}
        print("P9 written:", note["write"], file=sys.stderr)
        if not wr.get("identical"):
            print("readback differs from what was written; stopping before the render (restore_checkpoint undoes it)", file=sys.stderr)
            json.dump(note, open(os.path.join(NOTES, f"triphop-build-{stamp}.json"), "w"), indent=1, default=str)
            return 1
    res = srv.render_song(PROJECT, name=a.name, wait=True)
    song = res.get("song", {})
    note["render"] = {"wav": song.get("wav"), "seconds": res.get("seconds"), "peak_dbfs": song.get("peak_dbfs"), "rms_dbfs": song.get("rms_dbfs"),
                      "positions": [{k: p.get(k) for k in ("position", "scene", "midi_note_ons", "expected_events", "warning")} for p in res.get("positions", [])]}
    if song.get("wav"):
        y, sr = sf.read(song["wav"], dtype="float32", always_2d=True)
        note["render"]["spectral"] = an.spectral_mix_report(y, sr)
    json.dump(note, open(os.path.join(NOTES, f"triphop-build-{stamp}.json"), "w"), indent=1, default=str)
    print(json.dumps(note["render"], indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
