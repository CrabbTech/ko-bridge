#!/usr/bin/env python3
"""Step 7: write a 4-bar pattern (two pads + one fader lane) into the scratch project, play it under Mac clock, grade timing.

    uv run python scripts/pattern_play.py --scratch 9 --bpm 120 [--skip-write] [--bars-to-clock 5]

The device must have MIDI clock set to receive (system setting 101) and be in MAIN mode; the Mac sends START then 24-PPQN
clock. The project is compiled over the scratch slot's current archive (settings/fx_settings/pads preserved), written with
checkpoint + readback + reload, then after a 6 s settle the clock runs for `--bars-to-clock` bars while USB audio and the
device's MIDI note output are captured. Onsets are matched to the tick grid; MIDI note-ons are matched too.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from ko_bridge import analysis as an
from ko_bridge.audio import Recorder, find_audio_input
from ko_bridge.ep133 import EP133, describe_archive, save_json
from ko_bridge.midi import ClockDriver, Midi

OUT = Path(__file__).resolve().parents[1] / "docs" / "spike"
WAV_DIR = Path(os.path.expanduser("~/Music/ko-bridge/spike"))
PPQN = 96

EVENTS = [  # tick, pad (archive number), note, velocity
    (0, 10, 60, 120), (192, 11, 61, 100), (384, 10, 60, 120), (576, 11, 61, 100),
    (768, 10, 60, 120), (864, 10, 60, 90), (960, 11, 61, 100),
    (1152, 10, 60, 120), (1344, 11, 61, 100), (1440, 11, 61, 80),
]
AUTOMATION = [  # low-pass fader (parameter 3) lane: open, half closed for bar 2, open again
    {"tick": 0, "parameterID": 3, "value": 32767}, {"tick": 384, "parameterID": 3, "value": 14000},
    {"tick": 768, "parameterID": 3, "value": 32767},
]


def pattern_doc(bpm: float) -> dict:
    return {
        "schema": "ep.project.v1", "product": "ep133",
        "patterns": [
            {"id": "C01", "bars": 4,
             "events": [{"tick": t, "pad": p, "note": n, "velocity": v, "duration": 24} for t, p, n, v in EVENTS],
             "automation": AUTOMATION},
            {"id": "B01", "bars": 1, "events": []},
            {"id": "D01", "bars": 1, "events": []},
        ],
        "scenes": [{"groupPatterns": [1, 1, 1, 1], "timeSignature": [4, 4]}],
        "song": [1], "currentScene": 1,
        "settings": {"bpm": bpm},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", type=int, required=True)
    ap.add_argument("--bpm", type=float, default=120.0)
    ap.add_argument("--bars-to-clock", type=int, default=5)
    ap.add_argument("--skip-write", action="store_true", help="the pattern is already on the device; just play and grade")
    ap.add_argument("--duplex", action="store_true")
    ap.add_argument("--label", default="pattern")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    ep = EP133(lock_owner="ko-bridge pattern_play")
    report = {"scratch": args.scratch, "bpm": args.bpm, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "events": EVENTS,
              "automation": AUTOMATION}

    doc = pattern_doc(args.bpm)
    (OUT / "pattern-project.json").write_text(json.dumps(doc, indent=2))
    if not args.skip_write:
        _ckpt, base, _s = ep.checkpoint(args.scratch, "before-pattern-base")
        tar = ep.compile(doc, base)
        w = ep.write_project(args.scratch, tar, "pattern", reload=True)
        report["write"] = w
        print("pattern write:", json.dumps(w, default=str))
        if not w.get("identical"):
            print("readback mismatch; stopping before playback")
            save_json(OUT / f"{args.label}.json", report)
            return 1
        after, _m, _s = ep.read_project(args.scratch)
        report["archive_after"] = describe_archive(after)
        print("settling 6 s after reload")
        time.sleep(6.0)

    dev = find_audio_input("EP-133")
    sr = 48000
    clocks = args.bars_to_clock * 4 * 24
    rec = Recorder(dev["index"], channels=2, samplerate=sr, blocksize=128, duplex=args.duplex)
    midi_marks = []

    def on_midi(_t, msg):  # mark the audio timeline when the device's own note-on arrives
        if msg.type == "note_on" and msg.velocity > 0 and rec._stream is not None:
            midi_marks.append((rec.mark(f"midi{len(midi_marks)}"), msg.note))

    with Midi("EP-133", on_message=on_midi) as midi:
        midi.capture_start()
        rec.start()
        try:
            time.sleep(0.7)
            driver = ClockDriver(midi, args.bpm, clocks=clocks, send_stop=True, on_start=lambda: rec.mark("start"))
            driver.start()
            driver.join()
            time.sleep(1.0)
        finally:
            r = rec.stop()
        midi_events = midi.capture_stop()
    r.save(str(WAV_DIR / f"{args.label}.wav"))
    report["wav"] = str(WAV_DIR / f"{args.label}.wav")

    start_sample = r.mark_sample("start")
    t0 = driver.started_at
    tick_s = 60.0 / (args.bpm * PPQN)
    bar_ticks = 4 * PPQN
    expected = []
    # the 4-bar pattern loops: list every occurrence inside the clocked window
    pattern_ticks = 4 * bar_ticks
    total_ticks = args.bars_to_clock * bar_ticks
    for rep in range((total_ticks + pattern_ticks - 1) // pattern_ticks):
        for t, p, n, v in EVENTS:
            tick = t + rep * pattern_ticks
            if tick < total_ticks:
                expected.append({"tick": tick, "pad": p, "note": n, "velocity": v, "time_s": tick * tick_s})
    x = an.mono(r.audio)
    hp = an.transients(x, sr)
    exp_samples = [start_sample + int(round(e["time_s"] * sr)) for e in expected]
    found = an.onsets_near(hp, sr, exp_samples, before_ms=40.0, after_ms=160.0)
    offsets = []
    for e, es, f in zip(expected, exp_samples, found):
        e["onset_offset_ms"] = None if f is None else round((f - es) / sr * 1000.0, 2)
        if f is not None:
            offsets.append(e["onset_offset_ms"])
    # sub-millisecond onsets: first high-passed sample above -30 dBFS within 6 ms of the frame-quantized onset
    # (the high-pass keeps a still-ringing 880/440 Hz tail from the previous hit below the threshold)
    absx = np.abs(hp)
    for e, f in zip(expected, found):
        e["onset_fine_ms"] = None
        if f is not None:
            seg = absx[max(0, f - 288): f + 288]
            hits = np.nonzero(seg > 0.0316)[0]
            if len(hits):
                fine = max(0, f - 288) + int(hits[0])
                e["onset_fine_ms"] = round((fine - (start_sample + int(round(e["time_s"] * sr)))) / sr * 1000.0, 3)
    fine_vals = [e["onset_fine_ms"] for e in expected if e["onset_fine_ms"] is not None]
    report["audio_offset_fine_stats_ms"] = an.stats_ms(fine_vals)
    if fine_vals:
        report["audio_offset_fine_relative_to_first_ms"] = an.stats_ms([v - fine_vals[0] for v in fine_vals])
    report["start_sample"] = start_sample
    report["midi_marks"] = midi_marks
    report["expected"] = expected
    report["audio_onsets_matched"] = len(offsets)
    report["audio_offset_stats_ms"] = an.stats_ms(offsets)
    if offsets:
        rel = [o - offsets[0] for o in offsets]
        report["audio_offset_relative_to_first_ms"] = an.stats_ms(rel)
    report["all_onsets_s"] = [round((s - start_sample) / sr, 3) for s in an.all_onsets(hp, sr) if s >= start_sample - sr // 10]
    report["silent_spans_s"] = [(round(a - start_sample / sr, 2), round(b - start_sample / sr, 2)) for a, b in an.silent_spans(x, sr)]
    report["peak_dbfs"] = round(an.peak_db(x), 1)
    report["clock_jitter_ms"] = an.stats_ms([(b - a) * 1000.0 - 60000.0 / (args.bpm * 24) for a, b in zip(driver.clock_times, driver.clock_times[1:])])
    notes = [(t - t0, m) for t, m in midi_events if m.type == "note_on" and m.velocity > 0]
    report["midi_note_ons"] = [{"time_s": round(t, 4), "note": m.note, "velocity": m.velocity, "channel": m.channel + 1} for t, m in notes]
    midi_offsets = []
    for e in expected:
        cands = [t for t, m in notes if m.note == e["note"] and abs(t - e["time_s"]) < 0.15]
        e["midi_offset_ms"] = round((min(cands, key=lambda t: abs(t - e["time_s"])) - e["time_s"]) * 1000.0, 2) if cands else None
        if cands:
            midi_offsets.append(e["midi_offset_ms"])
    report["midi_offset_stats_ms"] = an.stats_ms(midi_offsets)
    # audio onset minus the arrival of the device's own note-on, both on the audio clock: a cross-check of the clock base
    audio_vs_midi = []
    for ms, note in midi_marks:
        s_on = an.onset_sample(hp, sr, start=max(0, ms - int(0.03 * sr)), floor_window_ms=30.0)
        if s_on is not None and s_on < ms + int(0.1 * sr):
            seg = absx[max(0, s_on - 288): s_on + 288]
            hits = np.nonzero(seg > 0.0316)[0]
            fine = max(0, s_on - 288) + int(hits[0]) if len(hits) else s_on
            audio_vs_midi.append(round((fine - ms) / sr * 1000.0, 3))
    report["audio_onset_minus_midi_arrival_ms"] = an.stats_ms(audio_vs_midi)
    report["audio_onset_minus_midi_arrival_values"] = audio_vs_midi
    report["midi_other"] = [{"time_s": round(t - t0, 4), "msg": str(m)} for t, m in midi_events if not (m.type in ("note_on", "note_off"))][:20]
    save_json(OUT / f"{args.label}.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("expected", "midi_note_ons", "events", "automation")}, indent=1, default=str))
    print("per-event offsets (ms):", [(e["tick"], e["onset_offset_ms"], e["midi_offset_ms"]) for e in expected])
    return 0 if offsets else 2


if __name__ == "__main__":
    sys.exit(main())
