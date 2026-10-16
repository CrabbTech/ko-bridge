#!/usr/bin/env python3
"""Step 4: project round trip into the scratch slot.

    uv run python scripts/roundtrip.py --scratch 9 --source-tar ~/Music/ko-bridge/backups/<ts>/projects/P01.tar

A: read the scratch slot, write the same bytes back, read again, diff (native-shaped archive, so byte-exact expected).
B: write `--source-tar` (a device-made archive from another slot) into the scratch slot, read back, diff, then run the
activation dance so the device loads it. Records bytes and seconds for every transfer and the active project before.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from ko_bridge.ep133 import EP133, describe_archive, save_json

OUT = Path(__file__).resolve().parents[1] / "docs" / "spike"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", type=int, required=True)
    ap.add_argument("--source-tar", type=Path, required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ep = EP133(lock_owner="ko-bridge roundtrip")
    report = {"scratch": args.scratch, "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    report["active_before"] = ep.active_project()
    print("active project before:", report["active_before"])

    ckpt, original, s = ep.checkpoint(args.scratch, "initial")
    report["initial"] = {"checkpoint": str(ckpt), "bytes": len(original), "read_seconds": round(s, 2),
                         "archive": describe_archive(original)}
    print(f"A: scratch P{args.scratch} read {len(original)} B in {s:.2f} s -> {ckpt}")

    a = ep.write_project(args.scratch, original, "roundtrip-self", reload=False)
    report["A_self_roundtrip"] = a
    print("A result:", json.dumps({k: v for k, v in a.items() if k != "reload"}))

    source = args.source_tar.read_bytes()
    b = ep.write_project(args.scratch, source, "roundtrip-copy", reload=True)
    b["source"] = str(args.source_tar)
    b["source_archive"] = describe_archive(source)
    report["B_copy_roundtrip"] = b
    print("B result:", json.dumps(b, default=str))
    report["active_after"] = ep.active_project()
    print("active project after reload:", report["active_after"])
    save_json(OUT / "roundtrip.json", report)
    ok = a.get("identical") and b.get("identical")
    print("ROUNDTRIP", "VERIFIED" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
