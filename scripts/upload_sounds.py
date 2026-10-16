#!/usr/bin/env python3
"""Step 5: upload the two test WAVs into free slots, assign them to group C pads in the scratch project, verify.

    uv run python scripts/upload_sounds.py --scratch 9 --slot46 600 --slot26 601 --base ~/Music/ko-bridge/backups/<ts>/projects/P02.tar

Refuses an occupied slot. Times each upload (KB/s) and projects a 12-chop crate of 2 s mono chops. The pad assignment
is compiled over `--base` (a device-made archive) so settings/fx_settings/scenes stay native, then written to the scratch
slot with checkpoint, readback and reload. Pads: C "." (p10, MIDI 60) <- 46,875 Hz sample, C "0" (p11, MIDI 61) <- 26,250 Hz.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import soundfile as sf
from epsysex.dependencies import wav_to_pcm16
from epsysex.tar import member_payloads

from ko_bridge.ep133 import EP133, decode_pad, describe_archive, save_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "spike"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", type=int, required=True)
    ap.add_argument("--slot46", type=int, default=600)
    ap.add_argument("--slot26", type=int, default=601)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--allow-overwrite", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ep = EP133(lock_owner="ko-bridge upload_sounds")
    report = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "uploads": []}

    sounds = ep.list_sounds()
    occupied = sorted(int(n["id"]) for n in sounds)
    report["occupied_before"] = {"count": len(occupied), "600_plus": [i for i in occupied if i >= 600]}
    for slot in (args.slot46, args.slot26):
        print(f"slot {slot}: {'OCCUPIED' if slot in occupied else 'free'}")
        if slot in occupied and not args.allow_overwrite:
            print("refusing to overwrite; pick another slot")
            return 2

    specs = [(args.slot46, ROOT / "assets" / "kob-test-46875.wav", "kob test 46k"),
             (args.slot26, ROOT / "assets" / "kob-test-26250.wav", "kob test 26k")]
    frames = {}
    for slot, path, name in specs:
        info = sf.info(path)
        pcm, stream = wav_to_pcm16(path, target_rate=info.samplerate, target_channels=1)
        frames[slot] = stream["frames"]
        r = ep.upload_sound(slot, pcm, name=name, samplerate=info.samplerate, channels=1,
                            allow_overwrite=args.allow_overwrite, verify=True)
        r.update({"wav": str(path), "samplerate": info.samplerate, "frames": stream["frames"], "seconds": round(info.duration, 3)})
        report["uploads"].append(r)
        print(f"upload slot {slot}: {len(pcm)} B in {r['upload_seconds']} s = {r['upload_kbps']} KB/s; "
              f"readback identical={r['identical']} in {r['readback_seconds']} s; metadata={json.dumps(r['metadata'])}")

    kbps = [u["upload_kbps"] for u in report["uploads"]]
    mean_kbps = sum(kbps) / len(kbps)
    crate_bytes = 12 * 2 * 46875 * 2
    report["projection"] = {"mean_upload_kbps": round(mean_kbps, 1), "crate_12x2s_mono_46875_bytes": crate_bytes,
                            "crate_seconds_at_mean_kbps": round(crate_bytes / 1024 / mean_kbps, 1),
                            "per_call_overhead_note": "each upload is one FILE session: init, put open, pages, EOF, re-init, metadata"}
    print("projection:", json.dumps(report["projection"]))

    base = args.base.read_bytes()
    doc = {
        "schema": "ep.project.v1", "product": "ep133",
        "pads": [
            {"group": "C", "pad": 10, "slot": args.slot46, "trimStart": 0, "trimEnd": frames[args.slot46], "sampleBpm": 120,
             "amplitude": 100, "release": 255, "timeMode": 0, "playMode": 0, "rootNote": 60},
            {"group": "C", "pad": 11, "slot": args.slot26, "trimStart": 0, "trimEnd": frames[args.slot26], "sampleBpm": 120,
             "amplitude": 100, "release": 255, "timeMode": 0, "playMode": 0, "rootNote": 61},
        ],
    }
    tar = ep.compile(doc, base)
    (OUT / "assign-pads.json").write_text(json.dumps(doc, indent=2))
    w = ep.write_project(args.scratch, tar, "assign-pads", reload=True)
    report["assign_write"] = w
    print("assign write:", json.dumps(w, default=str))
    after, _m, _s = ep.read_project(args.scratch)
    pads = member_payloads(after)
    report["pads_readback"] = {name: decode_pad(pads[name]) for name in ("pads/c/p10", "pads/c/p11")}
    report["archive_after"] = describe_archive(after)
    print("pads read back:", json.dumps(report["pads_readback"]))
    save_json(OUT / "upload_sounds.json", report)
    ok = all(u["identical"] for u in report["uploads"]) and w.get("identical") \
        and report["pads_readback"]["pads/c/p10"]["slot"] == args.slot46 and report["pads_readback"]["pads/c/p11"]["slot"] == args.slot26
    print("UPLOAD+ASSIGN", "VERIFIED" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
