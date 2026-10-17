"""The EP-133 device layer: epsysex FileClient plus the rules ko-bridge adds.

Every project write checkpoints the slot first, writes, reads back, compares, and runs the activation dance. Sound
uploads refuse occupied slots unless told otherwise; deletes need the acknowledgement string and spare the factory banks.
All FILE I/O is serialized by epsysex's /tmp/ep-device.lock; nothing here parallelizes it. Pad numbering helpers live
in ko_bridge.project.
"""
from __future__ import annotations

import json
import os
import struct
import time
from pathlib import Path
from typing import Any, Callable

from epsysex.compile import compile_project
from epsysex.fileclient import PROJECTS_NODE_ID, FileClient
from epsysex.sysex import EP133_PRODUCT
from epsysex.tar import member_payloads

from . import config as C
from .jobs import check
from .project import describe

FACTORY_SLOTS = range(1, 600)

# monotonic time of the last FILE session on this process: playback waits a settle window after it (see server.SETTLE_SECONDS)
LAST_FILE_IO = 0.0


def _touch() -> None:
    global LAST_FILE_IO
    LAST_FILE_IO = time.monotonic()


class _TouchingClient:
    """Every FileClient call stamps LAST_FILE_IO, so the player can keep its distance from FILE traffic."""

    def __init__(self, client: FileClient):
        self._client = client

    def __getattr__(self, name):
        attr = getattr(self._client, name)
        if not callable(attr):
            return attr

        def call(*a, **k):
            _touch()
            try:
                return attr(*a, **k)
            finally:
                _touch()
        return call
DELETE_ACK = "DELETE SOUND {slot}"
OVERWRITE_ACK = "OVERWRITE SOUND {slot}"


def decode_pad(record: bytes) -> dict:   # kept for the spike scripts
    from .project import decode_pad as _d
    return _d(record)


def describe_archive(tar: bytes) -> dict:  # kept for the spike scripts
    return describe(tar)


def pad_for_note(note: int) -> dict:     # kept for the spike scripts
    from .project import pad_for_note as _p
    d = _p(note)
    return {"note": note, "group": d["group"], "pad": int(d["member"][-2:]), "member": d["member"], "label": d["pad"]}


def note_for_pad(group: str, pad_number: int) -> int:
    from .project import label_for_member, pad_note
    return pad_note(group, label_for_member(pad_number))


def save_json(path, data) -> None:
    Path(path).write_text(json.dumps(data, indent=2, default=str))


class EP133:
    def __init__(self, port_hint: str = "EP-133", backup_root: str | Path | None = None, put_page_delay: float = 0.006,
                 lock_owner: str = "ko-bridge"):
        self.client = _TouchingClient(FileClient(product_byte=EP133_PRODUCT, port_hint=port_hint, lock_owner=lock_owner, put_page_delay=put_page_delay))
        self.backup_root = Path(backup_root) if backup_root else Path(C.HOME) / "backups"
        self.checkpoint_dir = self.backup_root / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # -- state ------------------------------------------------------------------
    def active_project(self) -> dict[str, Any]:
        fid = self.client.get_metadata(PROJECTS_NODE_ID, key="active").get("active")
        number = (fid - 3000) // 1000 + 1 if isinstance(fid, int) and fid >= 3000 else None
        return {"fid": fid, "project": number}

    def select_project(self, project: int) -> dict[str, Any]:
        """Make `project` the active project (the activation dance; cycles through another slot when it is already active)."""
        active = self.active_project().get("project")
        t0 = time.monotonic()
        result = self.client.reload_project(project, cycle=(active == project))
        return {"project": project, "was_active": active, "seconds": round(time.monotonic() - t0, 2), "reload": result}

    # -- projects ---------------------------------------------------------------
    def read_project(self, project: int) -> tuple[bytes, dict, float]:
        t0 = time.monotonic()
        data, meta = self.client.read_project_archive(project)
        return data, meta, time.monotonic() - t0

    def describe_project(self, project: int) -> dict[str, Any]:
        data, meta, seconds = self.read_project(project)
        out = describe(data)
        out.update({"project": project, "read_seconds": round(seconds, 2)})
        return out

    def checkpoint(self, project: int, label: str) -> tuple[Path, bytes, float]:
        data, _meta, seconds = self.read_project(project)
        path = self.checkpoint_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-P{project:02d}-{label}.tar"
        path.write_bytes(data)
        return path, data, seconds

    def checkpoints(self, project: int | None = None) -> list[dict[str, Any]]:
        out = []
        for p in sorted(self.checkpoint_dir.glob("*.tar")):
            parts = p.stem.split("-", 3)
            if len(parts) < 4 or not parts[2].startswith("P"):
                continue
            number = int(parts[2][1:])
            if project is not None and number != project:
                continue
            out.append({"path": str(p), "project": number, "label": parts[3], "bytes": p.stat().st_size, "time": f"{parts[0]}-{parts[1]}"})
        return out

    def write_project(self, project: int, tar: bytes, label: str, reload: bool = True, verify: bool = True, settle: float = 0.25,
                      base: bytes | None = None) -> dict[str, Any]:
        """Checkpoint, FILE_PUT, read back, compare, activation dance. Returns timings and the verification result.
        `base` is the slot's archive when the caller has just read it (it becomes the checkpoint without a second read)."""
        if base is None:
            ckpt, before, ckpt_s = self.checkpoint(project, f"before-{label}")
        else:
            before, ckpt_s = base, 0.0
            ckpt = self.checkpoint_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-P{project:02d}-before-{label}.tar"
            ckpt.write_bytes(base)
        t0 = time.monotonic()
        self.client.write_project_archive(project, tar)
        write_s = time.monotonic() - t0
        result: dict[str, Any] = {"project": project, "label": label, "checkpoint": str(ckpt), "checkpoint_bytes": len(before),
                                  "checkpoint_seconds": round(ckpt_s, 2), "bytes": len(tar), "write_seconds": round(write_s, 2),
                                  "write_kbps": round(len(tar) / 1024 / write_s, 1) if write_s else None}
        if verify:
            time.sleep(settle)
            after, _m, read_s = self.read_project(project)
            result.update({"readback_seconds": round(read_s, 2), "readback_bytes": len(after), "identical": after == tar})
            if after != tar:
                result["payloads_identical"] = member_payloads(after) == member_payloads(tar)
                path = self.checkpoint_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-P{project:02d}-readback-{label}.tar"
                path.write_bytes(after)
                result["readback_path"] = str(path)
        if reload:
            t0 = time.monotonic()
            result["reload"] = self.client.reload_project(project)
            result["reload_seconds"] = round(time.monotonic() - t0, 2)
        return result

    def restore_checkpoint(self, project: int, path: str | Path | None = None) -> dict[str, Any]:
        """Write a checkpoint TAR back into `project` (the newest one for that slot when `path` is None)."""
        if path is None:
            cps = self.checkpoints(project)
            if not cps:
                raise FileNotFoundError(f"no checkpoint for project {project} under {self.checkpoint_dir}")
            path = cps[-1]["path"]
        tar = Path(path).read_bytes()
        result = self.write_project(project, tar, "restore", reload=True)
        result["restored_from"] = str(path)
        return result

    def compile(self, doc: dict, base: bytes) -> bytes:
        return compile_project(doc, base_archive=base)

    # -- sounds -----------------------------------------------------------------
    def list_sounds(self) -> list[dict]:
        nodes = self.client.list_sounds()
        C.write_json(C.file_path("library.json"), {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "sounds": nodes})
        return nodes

    def cached_library(self) -> dict[str, Any] | None:
        return C.read_json(os.path.join(C.HOME, "library.json"))

    def sound_info(self, slot: int) -> dict[str, Any]:
        stat = self.client.stat(slot)
        meta = self.client.get_sample_metadata(slot)
        channels = int(meta.get("channels", 1) or 1)
        rate = int(meta.get("samplerate", 46875) or 46875)
        frames = stat["byteSize"] // (2 * channels)
        return {"slot": slot, "name": meta.get("name"), "bytes": stat["byteSize"], "channels": channels, "samplerate": rate,
                "frames": frames, "seconds": round(frames / rate, 3), "metadata": meta}

    def frames_for_slots(self, slots: set[int] | list[int]) -> dict[int, int]:
        return {s: self.sound_info(s)["frames"] for s in sorted(set(slots))}

    def upload_sound(self, slot: int, pcm: bytes, name: str, samplerate: int, channels: int = 1, acknowledge_overwrite: str | None = None,
                     verify: bool = True, occupied: set[int] | None = None) -> dict[str, Any]:
        if occupied is None:
            occupied = {int(n["id"]) for n in self.list_sounds()}
        if slot in occupied and acknowledge_overwrite != OVERWRITE_ACK.format(slot=slot):
            raise RuntimeError(f"sound slot {slot} is occupied; to replace it pass acknowledge_overwrite={OVERWRITE_ACK.format(slot=slot)!r}")
        name = (name or f"slot {slot}")[:16]      # 19 characters were rejected with "invalid length" on this device
        t0 = time.monotonic()
        summary = self.client.upload_sound(slot, pcm, name=name, samplerate=samplerate, channels=channels)
        up_s = time.monotonic() - t0
        result: dict[str, Any] = {"slot": slot, "bytes": len(pcm), "frames": len(pcm) // (2 * channels), "samplerate": samplerate, "channels": channels,
                                  "seconds": round(len(pcm) / (2 * channels) / samplerate, 3), "upload_seconds": round(up_s, 2),
                                  "upload_kbps": round(len(pcm) / 1024 / up_s, 1) if up_s else None, "name": summary["metadata"]["name"] if summary.get("metadata") else name,
                                  "replaced": slot in occupied}
        if verify:
            t0 = time.monotonic()
            back, _meta = self.client.read_sound(slot)
            result["readback_seconds"] = round(time.monotonic() - t0, 2)
            result["identical"] = back == pcm
            if back != pcm:
                raise RuntimeError(f"slot {slot} read back differently ({len(back)} B vs {len(pcm)} B written)")
        return result

    def delete_sound(self, slot: int, acknowledge: str, allow_factory: bool = False) -> dict[str, Any]:
        if acknowledge != DELETE_ACK.format(slot=slot):
            raise RuntimeError(f"deleting is destructive; pass acknowledge={DELETE_ACK.format(slot=slot)!r}")
        if slot in FACTORY_SLOTS and not allow_factory:
            raise RuntimeError(f"slot {slot} is in the factory banks 1-599; pass allow_factory=True to delete it anyway")
        self.client.delete_sound(slot)
        remaining = {int(n["id"]) for n in self.list_sounds()}
        return {"slot": slot, "deleted": slot not in remaining}

    def free_slots(self, occupied: set[int], minimum: int = 600) -> list[int]:
        return [s for s in range(minimum, 1000) if s not in occupied]

    def audition(self, slot: int) -> None:
        self.client.audition_sound(slot)

    # -- backup -----------------------------------------------------------------
    def backup(self, sounds: bool = True, stage: Callable[[str], None] | None = None, cancel=None) -> dict[str, Any]:
        """Projects 1-9 (always) and every sound slot (optionally) into backups/<timestamp>/, plus a .pak."""
        stamp = time.strftime("%Y%m%d-%H%M%S")
        root = self.backup_root / stamp
        (root / "projects").mkdir(parents=True, exist_ok=True)
        manifest: dict[str, Any] = {"timestamp": stamp, "projects": {}, "sounds": None}
        t_all = time.monotonic()
        for p in range(1, 10):
            check(cancel)
            if stage:
                stage(f"project {p}")
            data, meta, s = self.read_project(p)
            (root / "projects" / f"P{p:02d}.tar").write_bytes(data)
            manifest["projects"][str(p)] = {"bytes": len(data), "seconds": round(s, 2), "name": meta.get("name")}
        nodes = self.list_sounds()
        save_json(root / "sounds-list.json", nodes)
        manifest["sounds"] = {"count": len(nodes), "total_bytes": sum(int(n["size"]) for n in nodes), "downloaded": 0}
        if sounds:
            out = root / "sounds"
            out.mkdir(exist_ok=True)
            total = 0
            for i, node in enumerate(sorted(nodes, key=lambda n: int(n["id"]))):
                check(cancel)
                slot = int(node["id"])
                if stage:
                    stage(f"sound {slot} ({i + 1}/{len(nodes)}, {total / 1e6:.1f} MB)")
                data, _open = self.client.read_sound(slot)
                meta = self.client.get_sample_metadata(slot)
                (out / f"{slot:03d}.pcm").write_bytes(data)
                save_json(out / f"{slot:03d}.json", {"node": node, "metadata": meta})
                total += len(data)
            manifest["sounds"]["downloaded"] = len(nodes)
            manifest["sounds"]["bytes"] = total
            build_pak(root)
        manifest["seconds"] = round(time.monotonic() - t_all, 1)
        save_json(root / "manifest.json", manifest)
        manifest["path"] = str(root)
        return manifest


def build_pak(root: Path, device_version: str = "2.5.1", sku: str = "TE032AS001") -> Path:
    """A Sample-Tool-layout .pak from a backup directory (leading-slash entries; acceptance by TE's tool untested)."""
    import io
    import re
    import wave
    import zipfile

    out = root / "ep133-backup.pak"
    meta = {"info": "teenage engineering - pak file", "pak_version": 1, "pak_type": "user", "pak_release": "1.2.0", "device_name": "EP",
            "device_sku": sku, "device_version": device_version, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            "author": "ko-bridge", "base_sku": sku}
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("/meta.json", json.dumps(meta, indent=1))
        for tar in sorted((root / "projects").glob("P*.tar")):
            zf.write(tar, f"/projects/{tar.name}")
        for pcm_path in sorted((root / "sounds").glob("*.pcm")):
            meta_path = pcm_path.with_suffix(".json")
            if not meta_path.exists():
                continue
            md = json.loads(meta_path.read_text())["metadata"]
            name = re.sub(r"[^A-Za-z0-9 _.-]", "_", str(md.get("name") or f"slot{pcm_path.stem}")).strip() or f"slot{pcm_path.stem}"
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(int(md.get("channels", 1)))
                w.setsampwidth(2)
                w.setframerate(int(md.get("samplerate", 46875)))
                w.writeframes(pcm_path.read_bytes())
            zf.writestr(f"/sounds/{int(pcm_path.stem):03d} {name}.wav", buf.getvalue())
    return out
