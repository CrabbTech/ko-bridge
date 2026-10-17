#!/usr/bin/env python3
"""Upload a rendered sound set (manifest.json from synth.py) into free slots, tag each slot in the library, save slots.json.

    uv run python scripts/upload_set.py ~/Music/ko-bridge/library/sets/dnb-174
"""
from __future__ import annotations

import json
import os
import sys
import time

from ko_bridge import server as srv


def main() -> int:
    set_dir = os.path.expanduser(sys.argv[1])
    manifest = json.load(open(os.path.join(set_dir, "manifest.json")))
    items = manifest["items"]
    mono = [i for i in items if i["channels"] == 1]
    stereo = [i for i in items if i["channels"] == 2]
    t0 = time.time()
    slots: dict[str, int] = {}
    for group, is_mono in ((mono, True), (stereo, False)):
        if not group:
            continue
        print(f"uploading {len(group)} {'mono' if is_mono else 'stereo'} sounds...", flush=True)
        r = srv.upload_sounds([i["path"] for i in group], samplerate=46875, mono=is_mono, wait=True)
        for item, up in zip(group, r["uploaded"]):
            slots[item["name"]] = up["slot"]
            print(f"  {item['name']:18s} -> slot {up['slot']} {up['bytes']} B {up['upload_kbps']} KB/s identical={up.get('identical')}", flush=True)
    for item in items:
        if item["name"] in slots:
            srv.tag_sound(str(slots[item["name"]]), add=item["tags"] + ["dnb-174", item["kind"]])
    json.dump({"bpm": manifest["bpm"], "slots": slots, "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S")}, open(os.path.join(set_dir, "slots.json"), "w"), indent=1)
    print(f"done: {len(slots)} slots in {time.time() - t0:.0f} s -> slots.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
