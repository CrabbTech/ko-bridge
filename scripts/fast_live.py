#!/usr/bin/env python3
"""Step 8: one write while the device is playing, on the epsysex fast-live path (write + reload in one session).

    uv run python scripts/fast_live.py --scratch 9 --bpm 120

Takes a checkpoint first, starts Mac clock playback of the scratch project (setting 101 on the device), lets two bars
play, then calls write_project_archive_and_reload with a denser C01 pattern while the clock keeps running, records USB
audio throughout, and measures the audible interruption and whether the new pattern took over. A normal readback verifies
the bytes afterwards. The docs call this path opt-in (it skips checkpoint and immediate readback), not safe; the default
6 ms page pacing is kept.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from ko_bridge import analysis as an
from ko_bridge.audio import Recorder, find_audio_input
from ko_bridge.ep133 import EP133, describe_archive, save_json
from ko_bridge.midi import ClockDriver, Midi

OUT = Path(__file__).resolve().parents[1] / "docs" / "spike"
WAV_DIR = Path(os.path.expanduser("~/Music/ko-bridge/spike"))

DENSE_EVENTS = [(t, 10 if (t // 48) % 2 == 0 else 11, 60 if (t // 48) % 2 == 0 else 61, 110) for t in range(0, 1536, 48)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", type=int, required=True)
    ap.add_argument("--bpm", type=float, default=120.0)
    ap.add_argument("--bars-before", type=int, default=2)
    ap.add_argument("--bars-after", type=int, default=4)
    ap.add_argument("--duplex", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    ep = EP133(lock_owner="ko-bridge fast_live")
    report = {"scratch": args.scratch, "bpm": args.bpm, "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    ckpt, base, s = ep.checkpoint(args.scratch, "before-fastlive")
    report["checkpoint"] = {"path": str(ckpt), "bytes": len(base), "seconds": round(s, 2), "archive": describe_archive(base)}
    doc = {"schema": "ep.project.v1", "product": "ep133",
           "patterns": [{"id": "C01", "bars": 4,
                         "events": [{"tick": t, "pad": p, "note": n, "velocity": v, "duration": 24} for t, p, n, v in DENSE_EVENTS]}]}
    new_tar = ep.compile(doc, base)
    report["new_bytes"] = len(new_tar)

    dev = find_audio_input("EP-133")
    sr = 48000
    bar_s = 240.0 / args.bpm
    with Midi("EP-133") as midi:
        midi.capture_start()
        rec = Recorder(dev["index"], channels=2, samplerate=sr, blocksize=128, duplex=args.duplex).start()
        driver = ClockDriver(midi, args.bpm, clocks=None, send_stop=True, on_start=lambda: rec.mark("start"))
        try:
            time.sleep(0.7)
            driver.start()
            driver.wait_started()
            time.sleep(args.bars_before * bar_s)
            rec.mark("write_begin")
            t0 = time.monotonic()
            try:
                result = ep.client.write_project_archive_and_reload(args.scratch, new_tar)
                report["fast_live_result"] = result
            except Exception as exc:  # reported, playback is still stopped cleanly below
                report["fast_live_error"] = f"{type(exc).__name__}: {exc}"
                print("FAST-LIVE ERROR:", report["fast_live_error"])
            report["fast_live_seconds"] = round(time.monotonic() - t0, 2)
            rec.mark("write_end")
            time.sleep(args.bars_after * bar_s)
            driver.halt()
            driver.join()
            time.sleep(0.8)
        finally:
            r = rec.stop()
        midi_events = midi.capture_stop()
    r.save(str(WAV_DIR / "fast-live.wav"))
    report["wav"] = str(WAV_DIR / "fast-live.wav")
    start = r.mark_sample("start")
    wb, we = r.mark_sample("write_begin"), r.mark_sample("write_end")
    x = an.mono(r.audio)
    hp = an.transients(x, sr)
    onsets = an.all_onsets(hp, sr)
    rel = lambda s: round((s - start) / sr, 3)
    report["write_window_s"] = [rel(wb), rel(we)]
    report["onsets_s"] = [rel(s) for s in onsets]
    report["onsets_before_write"] = len([s for s in onsets if s < wb])
    report["onsets_during_write"] = len([s for s in onsets if wb <= s < we])
    report["onsets_after_write"] = len([s for s in onsets if s >= we])
    report["onsets_per_bar_before"] = round(report["onsets_before_write"] / args.bars_before, 2)
    report["onsets_per_bar_after"] = round(report["onsets_after_write"] / args.bars_after, 2)
    spans = an.silent_spans(x, sr, threshold_db=-55.0, min_ms=150.0)
    report["silent_spans_s"] = [(round(a - start / sr, 2), round(b - start / sr, 2)) for a, b in spans]
    gaps = [(a, b) for a, b in spans if b > wb / sr - 0.05 and a < we / sr + 3.0]
    report["interruption_s"] = round(sum(b - a for a, b in gaps), 2) if gaps else 0.0
    report["midi_note_ons_after_write"] = len([1 for t, m in midi_events if m.type == "note_on" and m.velocity > 0 and (t - driver.started_at) > we / sr - start / sr])
    report["midi_other"] = [str(m) for t, m in midi_events if m.type not in ("note_on", "note_off")][:10]

    time.sleep(1.0)
    after, _m, rs = ep.read_project(args.scratch)
    report["deferred_readback"] = {"identical": after == new_tar, "bytes": len(after), "seconds": round(rs, 2)}
    report["active_after"] = ep.active_project()
    save_json(OUT / "fast-live.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "onsets_s"}, indent=1, default=str))
    return 0 if report["deferred_readback"]["identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
