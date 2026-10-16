"""The EP-133 device layer: epsysex FileClient plus the rules this project adds.

Every project write goes through `write_project`, which checkpoints the slot to ~/Music/ko-bridge/backups first, writes,
reads back, compares, and (optionally) runs the activation dance. Sound uploads refuse occupied slots unless told
otherwise. All FILE I/O is serialized by epsysex's /tmp/ep-device.lock; nothing here parallelizes it.
"""
from __future__ import annotations

import json
import os
import struct
import time
from pathlib import Path

from epsysex.compile import compile_project
from epsysex.fileclient import PROJECTS_NODE_ID, FileClient, project_fid
from epsysex.sysex import EP133_PRODUCT
from epsysex.tar import iter_members, member_payloads

BACKUP_ROOT = Path(os.path.expanduser(os.environ.get("KO_BRIDGE_BACKUPS", "~/Music/ko-bridge/backups")))

# MIDI note offset inside a group (0..11) -> archive pad member and keypad label (epsysex midi-reference.md, live-calibrated)
PAD_BY_OFFSET = (("p10", "."), ("p11", "0"), ("p12", "enter"), ("p07", "1"), ("p08", "2"), ("p09", "3"),
                 ("p04", "4"), ("p05", "5"), ("p06", "6"), ("p01", "7"), ("p02", "8"), ("p03", "9"))
GROUP_BASE_NOTE = {"A": 36, "B": 48, "C": 60, "D": 72}


def pad_for_note(note: int) -> dict:
    if not 36 <= note <= 83:
        raise ValueError(f"note {note} is outside the pad range 36..83")
    group = "ABCD"[(note - 36) // 12]
    member, label = PAD_BY_OFFSET[(note - 36) % 12]
    return {"note": note, "group": group, "pad": int(member[1:]), "member": f"pads/{group.lower()}/{member}", "label": label}


def note_for_pad(group: str, pad_number: int) -> int:
    for offset, (member, _label) in enumerate(PAD_BY_OFFSET):
        if int(member[1:]) == pad_number:
            return GROUP_BASE_NOTE[group.upper()] + offset
    raise ValueError(f"pad {pad_number} is outside 1..12")


def decode_pad(record: bytes) -> dict:
    return {"valid0": record[0], "slot": struct.unpack_from("<H", record, 1)[0], "midi_channel": record[3],
            "trim_start": struct.unpack_from("<I", record, 4)[0], "trim_length": struct.unpack_from("<I", record, 8)[0],
            "sample_bpm": round(struct.unpack_from("<f", record, 12)[0], 2), "amplitude": record[16],
            "pitch": struct.unpack_from("<b", record, 17)[0], "pan": struct.unpack_from("<b", record, 18)[0],
            "attack": record[19], "release": record[20], "time_mode": record[21], "mute_group": record[22],
            "play_mode": record[23], "root_note": record[24], "size": len(record)}


def describe_archive(tar: bytes) -> dict:
    members = list(iter_members(tar))
    files = member_payloads(tar)
    pads = {n: decode_pad(p)["slot"] for n, p in files.items() if n.startswith("pads/") and len(p) >= 26}
    patterns = {n: {"bars": p[1], "records": p[2]} for n, p in files.items() if n.startswith("patterns/") and len(p) >= 4}
    out = {"bytes": len(tar), "members": len(members), "assigned_pads": {n: s for n, s in pads.items() if s},
           "patterns": patterns, "has_scenes": "scenes" in files, "has_settings": "settings" in files,
           "has_fx_settings": "fx_settings" in files}
    if "settings" in files and len(files["settings"]) >= 8:
        out["bpm"] = round(struct.unpack_from("<f", files["settings"], 4)[0], 2)
    if "scenes" in files and len(files["scenes"]) == 712:
        sc = files["scenes"]
        out["scene1"] = list(sc[7:13])
        out["scene_cursor"] = sc[7 + 99 * 6 + 3]
    return out


class EP133:
    def __init__(self, port_hint: str = "EP-133", backup_root: Path = BACKUP_ROOT, put_page_delay: float = 0.006,
                 lock_owner: str = "ko-bridge"):
        self.client = FileClient(product_byte=EP133_PRODUCT, port_hint=port_hint, lock_owner=lock_owner,
                                 put_page_delay=put_page_delay)
        self.backup_root = Path(backup_root)
        self.checkpoint_dir = self.backup_root / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # -- state ------------------------------------------------------------------
    def active_project(self) -> dict:
        fid = self.client.get_metadata(PROJECTS_NODE_ID, key="active").get("active")
        number = (fid - 3000) // 1000 + 1 if isinstance(fid, int) and fid >= 3000 else None
        return {"fid": fid, "project": number}

    # -- projects ---------------------------------------------------------------
    def read_project(self, project: int) -> tuple[bytes, dict, float]:
        t0 = time.monotonic()
        data, meta = self.client.read_project_archive(project)
        return data, meta, time.monotonic() - t0

    def checkpoint(self, project: int, label: str) -> tuple[Path, bytes, float]:
        data, _meta, seconds = self.read_project(project)
        path = self.checkpoint_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-P{project:02d}-{label}.tar"
        path.write_bytes(data)
        return path, data, seconds

    def write_project(self, project: int, tar: bytes, label: str, reload: bool = True, verify: bool = True,
                      settle: float = 0.25) -> dict:
        """Checkpoint, FILE_PUT, read back, compare, activation dance. Returns timings and the verification result."""
        ckpt, before, ckpt_s = self.checkpoint(project, f"before-{label}")
        t0 = time.monotonic()
        self.client.write_project_archive(project, tar)
        write_s = time.monotonic() - t0
        result = {"project": project, "label": label, "checkpoint": str(ckpt), "checkpoint_bytes": len(before),
                  "checkpoint_seconds": round(ckpt_s, 2), "bytes": len(tar), "write_seconds": round(write_s, 2),
                  "write_kbps": round(len(tar) / 1024 / write_s, 1)}
        if verify:
            time.sleep(settle)
            after, _m, read_s = self.read_project(project)
            result.update({"readback_seconds": round(read_s, 2), "readback_bytes": len(after), "identical": after == tar})
            if after != tar:
                result["payloads_identical"] = member_payloads(after) == member_payloads(tar)
                result["readback_path"] = str(self.checkpoint_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-P{project:02d}-readback-{label}.tar")
                Path(result["readback_path"]).write_bytes(after)
        if reload:
            t0 = time.monotonic()
            result["reload"] = self.client.reload_project(project)
            result["reload_seconds"] = round(time.monotonic() - t0, 2)
        return result

    def compile(self, doc: dict, base: bytes) -> bytes:
        return compile_project(doc, base_archive=base)

    # -- sounds -----------------------------------------------------------------
    def list_sounds(self) -> list[dict]:
        return self.client.list_sounds()

    def upload_sound(self, slot: int, pcm: bytes, name: str, samplerate: int, channels: int = 1,
                     allow_overwrite: bool = False, verify: bool = True) -> dict:
        occupied = {int(n["id"]) for n in self.list_sounds()}
        if slot in occupied and not allow_overwrite:
            raise RuntimeError(f"sound slot {slot} is occupied; refusing to overwrite")
        t0 = time.monotonic()
        summary = self.client.upload_sound(slot, pcm, name=name, samplerate=samplerate, channels=channels)
        up_s = time.monotonic() - t0
        result = {"slot": slot, "bytes": len(pcm), "upload_seconds": round(up_s, 2), "upload_kbps": round(len(pcm) / 1024 / up_s, 1),
                  "summary": summary, "was_occupied": slot in occupied}
        if verify:
            t0 = time.monotonic()
            back, _meta = self.client.read_sound(slot)
            result["readback_seconds"] = round(time.monotonic() - t0, 2)
            result["identical"] = back == pcm
            result["metadata"] = self.client.get_sample_metadata(slot)
        return result


def save_json(path: Path | str, data) -> None:
    Path(path).write_text(json.dumps(data, indent=2, default=str))
