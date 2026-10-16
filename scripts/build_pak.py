#!/usr/bin/env python3
"""Assemble a .pak (TE Sample Tool style ZIP) from a ko-bridge backup directory.

    uv run python scripts/build_pak.py ~/Music/ko-bridge/backups/<timestamp> --device-version 2.5.1

/meta.json, /projects/PNN.tar from projects/, /sounds/NNN name.wav built from sounds/NNN.pcm + NNN.json metadata
(samplerate, channels). Leading slashes are significant to the official tool. Acceptance by the official tool is not
verified; the per-slot PCM + metadata files remain the primary restore path (epsysex upload_sound).
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
import wave
import zipfile
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("backup", type=Path)
    ap.add_argument("--device-version", default="2.5.1")
    ap.add_argument("--sku", default="TE032AS001")
    args = ap.parse_args()
    root = args.backup.expanduser()
    out = root / "ep133-backup.pak"
    meta = {"info": "teenage engineering - pak file", "pak_version": 1, "pak_type": "user", "pak_release": "1.2.0",
            "device_name": "EP", "device_sku": args.sku, "device_version": args.device_version,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()), "author": "ko-bridge", "base_sku": args.sku}
    count_sounds = 0
    total = 0
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("/meta.json", json.dumps(meta, indent=1))
        for tar in sorted((root / "projects").glob("P*.tar")):
            zf.write(tar, f"/projects/{tar.name}")
        for pcm_path in sorted((root / "sounds").glob("*.pcm")):
            slot = int(pcm_path.stem)
            meta_path = pcm_path.with_suffix(".json")
            if not meta_path.exists():
                print(f"slot {slot}: no metadata, skipped")
                continue
            md = json.loads(meta_path.read_text())["metadata"]
            name = re.sub(r"[^A-Za-z0-9 _.-]", "_", str(md.get("name") or f"slot{slot:03d}")).strip() or f"slot{slot:03d}"
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(int(md.get("channels", 1)))
                w.setsampwidth(2)
                w.setframerate(int(md.get("samplerate", 46875)))
                w.writeframes(pcm_path.read_bytes())
            zf.writestr(f"/sounds/{slot:03d} {name}.wav", buf.getvalue())
            count_sounds += 1
            total += pcm_path.stat().st_size
    print(f"{out}: {out.stat().st_size / 1e6:.1f} MB, {count_sounds} sounds ({total / 1e6:.1f} MB PCM), projects {len(list((root / 'projects').glob('P*.tar')))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
