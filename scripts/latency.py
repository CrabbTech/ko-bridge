#!/usr/bin/env python3
"""Step 6: MIDI note -> USB audio latency and jitter.

    uv run python scripts/latency.py --note 60 --trials 20 [--duplex] [--blocksize 128]

Records the EP-133 USB audio input, marks the stream clock right before each note-on, and finds the first onset after the
mark on a high-passed copy (the test samples open with an 8 ms noise burst). Reports mean/std/min/max, the raw-signal
detector for comparison, PortAudio's reported input latency, and whether any audio arrived at all (if not, rerun with
--duplex to answer the full-duplex question).
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
from ko_bridge.ep133 import save_json
from ko_bridge.midi import Midi

OUT = Path(__file__).resolve().parents[1] / "docs" / "spike"
WAV_DIR = Path(os.path.expanduser("~/Music/ko-bridge/spike"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--note", type=int, default=60)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--spacing", type=float, default=1.5)
    ap.add_argument("--duplex", action="store_true")
    ap.add_argument("--blocksize", type=int, default=128)
    ap.add_argument("--velocity", type=int, default=110)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    dev = find_audio_input("EP-133")
    if dev is None:
        print("no EP-133 audio input")
        return 1
    with Midi("EP-133") as midi:
        rec = Recorder(dev["index"], channels=2, samplerate=48000, blocksize=args.blocksize, duplex=args.duplex).start()
        try:
            latency_report = rec._stream.latency
            time.sleep(0.6)
            send_times = []
            for i in range(args.trials):
                m = rec.mark(f"n{i}")
                t_send = midi.note_on(args.note, args.velocity)
                send_times.append((m, t_send))
                time.sleep(0.3)
                midi.note_off(args.note)
                time.sleep(max(0.1, args.spacing - 0.3))
            time.sleep(0.6)
        finally:
            r = rec.stop()
    sr = r.samplerate
    x = an.mono(r.audio)
    hp = an.transients(x, sr)
    rows = []
    for i in range(args.trials):
        m = r.mark_sample(f"n{i}")
        start = max(0, m - int(0.010 * sr))
        s_hp = an.onset_sample(hp, sr, start=start, floor_window_ms=40.0)
        s_raw = an.onset_sample(x, sr, start=start, floor_window_ms=40.0)
        seg = r.audio[m: m + int(0.5 * sr)]
        rows.append({"trial": i, "mark": m,
                     "latency_ms": None if s_hp is None else round((s_hp - m) / sr * 1000.0, 2),
                     "latency_raw_ms": None if s_raw is None else round((s_raw - m) / sr * 1000.0, 2),
                     "peak_dbfs": round(an.peak_db(seg), 1),
                     "levels_dbfs": [round(v, 1) for v in an.channel_levels(seg)] if len(seg) else None})
    lat = [row["latency_ms"] for row in rows if row["latency_ms"] is not None and row["latency_ms"] < 400]
    report = {"note": args.note, "trials": args.trials, "duplex": args.duplex, "blocksize": args.blocksize, "samplerate": sr,
              "portaudio_input_latency_s": latency_report, "capture_seconds": round(r.seconds, 2),
              "stream_statuses": r.statuses[:10], "detected": len(lat), "stats": an.stats_ms(lat),
              "stats_raw": an.stats_ms([row["latency_raw_ms"] for row in rows if row["latency_raw_ms"] is not None and row["latency_raw_ms"] < 400]),
              "floor_dbfs": round(an.rms_db(x[: int(0.5 * sr)]), 1), "peak_dbfs": round(an.peak_db(x), 1), "rows": rows}
    tag = f"{args.label or 'note' + str(args.note)}{'-duplex' if args.duplex else ''}"
    wav = WAV_DIR / f"latency-{tag}.wav"
    r.save(str(wav))
    report["wav"] = str(wav)
    save_json(OUT / f"latency-{tag}.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1))
    if not lat:
        print("NO ONSETS DETECTED: no audio reached the Mac (peak %.1f dBFS); try --duplex, or check the active project/pad" % report["peak_dbfs"])
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
