#!/usr/bin/env python3
"""Phase 1 of a device backup: identity, every project TAR (1-9), the sound list.

Writes ~/Music/ko-bridge/backups/<timestamp>/{identity.json,projects/PNN.tar,sounds-list.json,manifest.json}
and prints a one-line summary per project so an expendable scratch slot can be chosen.
Read-only against the device; serialized through the shared epsysex lock.
"""
from __future__ import annotations

import json
import os
import struct
import sys
import time
from pathlib import Path

from epsysex.fileclient import FileClient
from epsysex.identity import identity_from_device
from epsysex.sysex import EP133_PRODUCT
from epsysex.tar import iter_members

BACKUP_ROOT = Path(os.path.expanduser(os.environ.get("KO_BRIDGE_BACKUPS", "~/Music/ko-bridge/backups")))


def summarize_project(data: bytes) -> dict:
    members = list(iter_members(data))
    files = {name: data[start:start + size] for name, start, size, t in members if t != "5"}
    pads = {}
    for name, payload in files.items():
        if name.startswith("pads/") and len(payload) >= 3:
            slot = struct.unpack_from("<H", payload, 1)[0]
            if slot:
                pads[name] = slot
    patterns = {n: p for n, p in files.items() if n.startswith("patterns/") and len(p) >= 4}
    nonempty = {n: p[2] for n, p in patterns.items() if p[2] > 0}
    bpm = None
    if "settings" in files and len(files["settings"]) >= 8:
        bpm = round(struct.unpack_from("<f", files["settings"], 4)[0], 2)
    scenes = files.get("scenes")
    return {
        "bytes": len(data),
        "members": len(members),
        "assigned_pads": len(pads),
        "assigned_slots": sorted(set(pads.values())),
        "pattern_members": len(patterns),
        "nonempty_patterns": {n: c for n, c in sorted(nonempty.items())},
        "has_scenes": scenes is not None,
        "scene_cursor": scenes[-111 + 3] if scenes and len(scenes) == 712 else None,
        "has_settings": "settings" in files,
        "settings_bytes": len(files["settings"]) if "settings" in files else None,
        "has_fx_settings": "fx_settings" in files,
        "fx_settings_bytes": len(files["fx_settings"]) if "fx_settings" in files else None,
        "bpm": bpm,
    }


def main() -> int:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    root = BACKUP_ROOT / stamp
    (root / "projects").mkdir(parents=True, exist_ok=True)
    print(f"backup dir: {root}")

    ident = identity_from_device("EP-133")
    (root / "identity.json").write_text(json.dumps(ident, indent=2))
    print("identity:", json.dumps(ident))

    client = FileClient(product_byte=EP133_PRODUCT, port_hint="EP-133", lock_owner="ko-bridge backup_projects")
    manifest = {"timestamp": stamp, "identity": ident, "projects": {}, "sounds": None}
    for p in range(1, 10):
        t0 = time.monotonic()
        data, meta = client.read_project_archive(p)
        dt = time.monotonic() - t0
        (root / "projects" / f"P{p:02d}.tar").write_bytes(data)
        summary = summarize_project(data)
        summary.update({"name": meta.get("name"), "fid": meta.get("fid"), "seconds": round(dt, 2),
                        "kbps": round(len(data) / 1024 / dt, 1) if dt else None})
        manifest["projects"][str(p)] = summary
        print(f"P{p}: {len(data)} B in {dt:.2f} s ({summary['kbps']} KB/s) name={meta.get('name')!r} "
              f"pads={summary['assigned_pads']} patterns={summary['pattern_members']} "
              f"nonempty={len(summary['nonempty_patterns'])} scenes={summary['has_scenes']} "
              f"bpm={summary['bpm']} settings={summary['settings_bytes']} fx={summary['fx_settings_bytes']}")
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    t0 = time.monotonic()
    sounds = client.list_sounds()
    dt = time.monotonic() - t0
    (root / "sounds-list.json").write_text(json.dumps(sounds, indent=2))
    total = sum(int(n["size"]) for n in sounds)
    ids = sorted(int(n["id"]) for n in sounds)
    manifest["sounds"] = {"count": len(sounds), "total_bytes": total, "list_seconds": round(dt, 2),
                          "lowest": ids[0] if ids else None, "highest": ids[-1] if ids else None,
                          "occupied_600_plus": [i for i in ids if i >= 600]}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"sounds: {len(sounds)} slots, {total / 1e6:.2f} MB, list took {dt:.2f} s; "
          f"occupied >= 600: {manifest['sounds']['occupied_600_plus']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
