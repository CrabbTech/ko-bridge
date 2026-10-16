#!/usr/bin/env python3
"""Armed recording of the device's own song-mode playback: waits for the first note the device sends, records USB audio
and MIDI from there, stops when the notes stop (or after --max seconds), and names each 4-bar song position by the scene
it matches (kick count, melody notes).

    uv run python scripts/record_song.py --wait 25 --max 45 --bpm 170 --bars-per-position 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import soundfile as sf

from ko_bridge import analysis as an
from ko_bridge.audio import Recorder, find_audio_input
from ko_bridge.midi import Midi

TAKES = os.path.expanduser("~/Music/ko-bridge/takes")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait", type=float, default=25.0, help="seconds to wait for the first note")
    ap.add_argument("--max", type=float, default=45.0, help="seconds to record after the first note at most")
    ap.add_argument("--silence", type=float, default=4.0, help="stop when no note arrived for this long")
    ap.add_argument("--bpm", type=float, default=170.0)
    ap.add_argument("--bars-per-position", type=int, default=4)
    ap.add_argument("--name", default="song-mode")
    args = ap.parse_args()
    dev = find_audio_input("EP-133")
    rec = Recorder(dev["index"], channels=2, samplerate=48000, blocksize=256)
    first = {"t": None, "sample": None}
    last = {"t": None}
    marks = []

    def on_midi(t, msg):
        if msg.type == "note_on" and msg.velocity > 0:
            if first["t"] is None:
                first["t"] = t
                first["sample"] = rec.mark("first_note")
            last["t"] = t

    with Midi("EP-133", on_message=on_midi) as midi:
        midi.capture_start()
        rec.start()
        print("armed: press SHIFT+PLAY on the device", flush=True)
        t0 = time.perf_counter()
        while first["t"] is None and time.perf_counter() - t0 < args.wait:
            time.sleep(0.05)
        if first["t"] is None:
            rec.stop(); midi.capture_stop()
            print("no note arrived; nothing recorded")
            return 2
        print(f"first note after {first['t'] - t0:.1f} s; recording", flush=True)
        while time.perf_counter() - first["t"] < args.max:
            if last["t"] is not None and time.perf_counter() - last["t"] > args.silence:
                break
            time.sleep(0.05)
        time.sleep(0.5)
        recording = rec.stop()
        events = midi.capture_stop()
    sr = recording.samplerate
    start = recording.mark_sample("first_note")
    ons = [(t - first["t"], m.note, m.velocity) for t, m in events if m.type == "note_on" and m.velocity > 0]
    length_s = ons[-1][0] if ons else 0.0
    # the tempo the device actually ran: closed-hat (note 42) 16th spacing when present, else the nominal BPM
    hats = np.sort([t for t, n, v in ons if n == 42])
    gaps = np.diff(hats)
    gaps = gaps[(gaps > 0.04) & (gaps < 0.25)]
    step_s = float(np.median(gaps)) if len(gaps) > 8 else 60.0 / (args.bpm * 4)
    bpm_measured = 60.0 / (step_s * 4)
    bar_s = step_s * 16
    burst = sum(1 for t, n, v in ons if t < 0.1)
    pos_s = args.bars_per_position * bar_s
    positions = []
    for p in range(int(np.ceil((length_s + 0.01) / pos_s))):
        a, b = p * pos_s, (p + 1) * pos_s
        notes = [n for t, n, v in ons if a - 0.02 <= t < b - 0.02]
        kicks = sum(1 for n in notes if n == 39)
        hats_n = sum(1 for n in notes if n == 42)
        per_bar = args.bars_per_position
        if hats_n / per_bar <= 8:
            scene = "breakdown-like (sparse hats)"
        elif kicks / per_bar < 1:
            scene = "intro-like (hats, no kick)"
        else:
            scene = "drop-like (kicks + hats)"
        positions.append({"position": p + 1, "seconds": [round(a, 2), round(b, 2)], "notes": len(notes), "kicks": kicks, "hats": hats_n, "scene": scene})
    stopped = (time.perf_counter() - (first["t"] + length_s)) > args.silence - 0.5
    x = recording.audio
    lo = max(0, start - int(0.3 * sr)); hi = min(len(x), start + int((length_s + 2.0) * sr))
    os.makedirs(TAKES, exist_ok=True)
    stem = f"{args.name}-{time.strftime('%Y%m%d-%H%M%S')}"
    wav = os.path.join(TAKES, stem + ".wav")
    sf.write(wav, x[lo:hi], sr, subtype="PCM_24")
    report = {"wav": wav, "seconds": round((hi - lo) / sr, 2), "first_note_at_s": round(first["t"] - t0, 2), "note_ons": len(ons), "song_length_s": round(length_s, 2),
              "bpm_measured": round(bpm_measured, 1), "song_bars": round(length_s / bar_s, 2), "notes_in_first_100ms": burst,
              "start_burst": burst > 40, "stopped_by_itself": bool(stopped), "positions": positions,
              "peak_dbfs": round(an.peak_db(x[lo:hi]), 1), "statuses": recording.statuses[:3]}
    with open(os.path.join(TAKES, stem + ".json"), "w") as f:
        json.dump({**report, "midi_note_ons": [{"time_s": round(t, 4), "note": n, "velocity": v} for t, n, v in ons]}, f, indent=1)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
