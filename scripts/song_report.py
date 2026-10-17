#!/usr/bin/env python3
"""Report on a rendered song take: per-position note-on matches and per-section band levels.

    uv run python scripts/song_report.py ~/Music/ko-bridge/takes/<song take>.json [--names intro,groove,...]
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

from ko_bridge import analysis as an


def band(x, sr, lo, hi):
    if lo == 0:
        sos = butter(4, hi / (sr / 2), btype="lowpass", output="sos")
    elif hi is None:
        sos = butter(4, lo / (sr / 2), btype="highpass", output="sos")
    else:
        sos = butter(4, [lo / (sr / 2), hi / (sr / 2)], btype="bandpass", output="sos")
    return sosfilt(sos, x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("take_json")
    ap.add_argument("--names", default="")
    args = ap.parse_args()
    info = json.load(open(args.take_json))
    x, sr = sf.read(info["wav"], dtype="float32", always_2d=True)
    mono = x.mean(axis=1)
    bpm = float(info["bpm"])
    bar = 240.0 / bpm
    names = [n for n in args.names.split(",") if n]
    bands = {"sub": band(mono, sr, 0, 120), "low": band(mono, sr, 120, 400), "mid": band(mono, sr, 400, 3000), "high": band(mono, sr, 3000, None)}
    print(f"{info['take_id']}: {len(x) / sr:.2f} s, peak {an.peak_db(x):.1f} dBFS, clipped samples {int((np.abs(x) >= 0.999).sum())}")
    start = 0.0
    for i, pos in enumerate(info.get("positions", [])):
        length = pos["bars"] * bar
        a, b = int(start * sr), int((start + length) * sr)
        levels = {k: round(an.rms_db(v[a:b]), 1) for k, v in bands.items()}
        label = names[i] if i < len(names) else f"scene {pos['scene']}"
        ok = f"{pos['midi_note_ons']}/{pos['expected_events']}"
        print(f"  pos {pos['position']} {label:10s} {pos['bars']} bars  notes {ok:9s} {levels}  peak {an.peak_db(x[a:b]):.1f}" + (f"  WARNING {pos['warning']}" if pos.get("warning") else ""))
        start += length
    spans = an.silent_spans(mono, sr, threshold_db=-50, min_ms=300)
    print("  silent spans >300 ms:", [(round(a, 2), round(b, 2)) for a, b in spans][:10])
    return 0


if __name__ == "__main__":
    sys.exit(main())
