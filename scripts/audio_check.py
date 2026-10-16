#!/usr/bin/env python3
"""Open the EP-133 USB audio input for two seconds and report what arrives (no MIDI, no FILE traffic)."""
from __future__ import annotations

import json
import sys
import time

from ko_bridge import analysis as an
from ko_bridge.audio import Recorder, find_audio_input


def main() -> int:
    duplex = "--duplex" in sys.argv
    dev = find_audio_input("EP-133")
    if dev is None:
        print("no EP-133 audio input")
        return 1
    with Recorder(dev["index"], channels=2, samplerate=48000, blocksize=128, duplex=duplex) as rec:
        time.sleep(2.0)
        rec.mark("end")
    r = rec.stop()
    print(json.dumps({"device": dev, "duplex": duplex, "seconds": round(r.seconds, 3), "blocks_statuses": r.statuses[:5],
                      "levels_dbfs": [round(v, 1) for v in an.channel_levels(r.audio)],
                      "peak_dbfs": round(an.peak_db(r.audio), 1), "mark_end": r.mark_sample("end"),
                      "clock": "perf_counter" if rec._use_perf else "adc"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
