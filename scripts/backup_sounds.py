#!/usr/bin/env python3
"""Phase 2 of a device backup: every sample slot's raw PCM and metadata, resumable.

    uv run python scripts/backup_sounds.py ~/Music/ko-bridge/backups/<timestamp>

Needs the sounds-list.json that backup_projects.py wrote. Skips slots already saved with the
right size. Also STATs a few tree nodes (root, /sounds, /projects, P1) into stat.json.
Stops at once on a debug frame or a timeout (never retries a half-open stream); a clean
status rejection is logged and the next slot is tried. Read-only against the device.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from epsysex.fileclient import FileClient
from epsysex.sysex import EP133_PRODUCT, DebugFrameError, StatusRejectionError


def main() -> int:
    root = Path(sys.argv[1]).expanduser()
    sounds = json.loads((root / "sounds-list.json").read_text())
    out = root / "sounds"
    out.mkdir(exist_ok=True)
    client = FileClient(product_byte=EP133_PRODUCT, port_hint="EP-133", lock_owner="ko-bridge backup_sounds")

    stats = {}
    for fid in (0, 1000, 2000, 3000):
        try:
            stats[str(fid)] = client.stat(fid)
        except DebugFrameError:
            raise
        except Exception as exc:  # unknown API shape is not a device problem
            stats[str(fid)] = f"{type(exc).__name__}: {exc}"
    (root / "stat.json").write_text(json.dumps(stats, indent=2, default=str))
    print("stat:", json.dumps(stats, default=str), flush=True)

    progress = (root / "sounds-progress.jsonl").open("a")
    total_bytes = 0
    total_seconds = 0.0
    done = 0
    skipped = 0
    errors = []
    wall0 = time.monotonic()
    ordered = sorted(sounds, key=lambda n: int(n["id"]))
    for node in ordered:
        slot, size = int(node["id"]), int(node["size"])
        pcm_path, meta_path = out / f"{slot:03d}.pcm", out / f"{slot:03d}.json"
        if pcm_path.exists() and pcm_path.stat().st_size == size and meta_path.exists():
            skipped += 1
            continue
        t0 = time.monotonic()
        try:
            data, open_meta = client.read_sound(slot)
            metadata = client.get_sample_metadata(slot)
        except StatusRejectionError as exc:
            errors.append({"slot": slot, "error": str(exc)})
            print(f"slot {slot}: REJECTED {exc}", flush=True)
            continue
        dt = time.monotonic() - t0
        pcm_path.write_bytes(data)
        meta_path.write_text(json.dumps({"node": node, "open": open_meta, "metadata": metadata}, indent=2, default=str))
        total_bytes += len(data)
        total_seconds += dt
        done += 1
        record = {"slot": slot, "bytes": len(data), "listed": size, "seconds": round(dt, 3),
                  "kbps": round(len(data) / 1024 / dt, 1) if dt else None, "name": metadata.get("name")}
        progress.write(json.dumps(record) + "\n")
        progress.flush()
        print(f"slot {slot:3d} {len(data):>8} B {dt:6.2f} s {record['kbps']:6.1f} KB/s name={metadata.get('name')!r} "
              f"[{done + skipped}/{len(ordered)}; cumulative {total_bytes / 1024 / total_seconds:.1f} KB/s, "
              f"{total_bytes / 1e6:.1f} MB, {(time.monotonic() - wall0) / 60:.1f} min]", flush=True)

    summary = {"slots": len(ordered), "downloaded": done, "skipped_existing": skipped, "errors": errors,
               "bytes": total_bytes, "seconds": round(total_seconds, 1),
               "kbps": round(total_bytes / 1024 / total_seconds, 1) if total_seconds else None,
               "wall_minutes": round((time.monotonic() - wall0) / 60, 1)}
    (root / "sounds-summary.json").write_text(json.dumps(summary, indent=2))
    print("DONE", json.dumps(summary), flush=True)
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
