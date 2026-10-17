"""The ko-bridge MCP server: lets a model drive a connected EP-133 K.O. II.

Runs over stdio (Claude Desktop, Claude Code) or Streamable HTTP. Every tool talks to the real device; one device
operation at a time (a process lock here, epsysex's file lock across processes); anything longer than a chat tool call
runs as a job.
"""
from __future__ import annotations

import contextlib
import functools
import glob
import importlib.metadata
import math
import os
import threading
import time
from typing import Annotated, Any, Literal
from uuid import uuid4

import mido
from mcp import MCPError
from mcp.server import CacheHint, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field as Param

from . import config as C
from . import knowledge, player
from . import library as L
from . import project as PJ
from .audio import find_audio_input
from . import ep133 as E
from .ep133 import DELETE_ACK, EP133, OVERWRITE_ACK
from .jobs import Job, Jobs
from .midi import Midi, find_port
from .project import Group, PadLabel, PadSpec, PatternSpec, ProjectSpec

INSTRUCTIONS = """ko-bridge controls a Teenage Engineering EP-133 K.O. II plugged into this computer.
Start with get_guide("quickstart") (five moves, a minute to read), then get_status. The device's sounds live in a
shared library (list_sounds, upload_sound); a project slot (1-9) holds pads, patterns, scenes, BPM and FX and is
written as a whole from a spec (write_project, with assign_pads and write_pattern as short forms); every write is
checkpointed and can be undone (restore_checkpoint). Playback from the Mac needs the device set to receive MIDI clock
(human_steps("midi_clock_in")); play_pattern then records the result and grades it against the grid, play_notes plays
pads live. Anything longer than about 25 s comes back at once as a job: poll job_status(job_id) and send no other
device calls meanwhile. System settings cannot be read or written over USB: human_steps gives the key presses."""


def _version() -> str:
    try:
        return importlib.metadata.version("ko-bridge")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0"


_STATIC = CacheHint(ttl_ms=3_600_000, scope="public")
CACHE_HINTS = {"tools/list": _STATIC, "prompts/list": _STATIC, "resources/list": _STATIC, "resources/templates/list": _STATIC,
               "server/discover": _STATIC}

_server = MCPServer("ko-bridge", title="EP-133 K.O. II bridge", instructions=INSTRUCTIONS, version=_version(), cache_hints=CACHE_HINTS)


class _Registrar:
    """Wraps every tool so refusals and device errors reach the model as readable tool errors."""

    def __init__(self, server):
        self._s = server

    def __getattr__(self, name):
        return getattr(self._s, name)

    def tool(self, **kw):
        def deco(fn):
            kw.setdefault("title", fn.__name__.replace("_", " ").capitalize())

            @functools.wraps(fn)
            def wrapped(*a, **k):
                try:
                    return fn(*a, **k)
                except (ToolError, MCPError):
                    raise
                except Exception as e:
                    raise ToolError(f"{type(e).__name__}: {e}") from e
            self._s.tool(**kw)(wrapped)
            return wrapped
        return deco


mcp = _Registrar(_server)
_device_lock = threading.Lock()
JOBS = Jobs()
SYNC_LIMIT = float(os.environ.get("KO_BRIDGE_SYNC_LIMIT", "25"))
# After a project switch or reload the device needs a moment: the epsysex docs saw MIDI floods when transport started at
# once, and this bridge saw the USB audio output stuck on one repeating packet when a note hit 3 s after a switch.
SETTLE_SECONDS = 6.0

RO = ToolAnnotations(read_only_hint=True, open_world_hint=False)
RW = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False)
SETS = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=True, open_world_hint=False)
OVERWRITES = ToolAnnotations(read_only_hint=False, destructive_hint=True, open_world_hint=False)

Project = Annotated[int, Param(ge=1, le=9, description="project slot 1-9")]
Slot = Annotated[int, Param(ge=1, le=999, description="sample slot 1-999")]
Bpm = Annotated[float, Param(ge=40, le=399, description="BPM")]
Bars = Annotated[int, Param(ge=1, le=99)]
SampleRate = Literal[26250, 32000, 46875, 48000]
GuideTopic = Literal["quickstart", "device", "project", "human"]
HumanStepTopic = Literal[tuple(sorted(knowledge.HUMAN_STEPS))]
MidiClock = Literal["unknown", "off", "in", "out"]


# ------------------------------------------------------------------ plumbing

def _cfg() -> C.Config:
    return C.Config.load()


def _ports(cfg: C.Config) -> dict[str, Any]:
    return {"midi_in": find_port(mido.get_input_names(), cfg.midi_hint), "midi_out": find_port(mido.get_output_names(), cfg.midi_hint),
            "audio": find_audio_input(cfg.audio_hint)}


def _require_midi(cfg: C.Config) -> None:
    if find_port(mido.get_output_names(), cfg.midi_hint) is None:
        raise RuntimeError("EP-133 not found over USB MIDI: connect it with USB-C and leave it on its normal screen")


@contextlib.contextmanager
def _device(timeout: float = 3.0):
    """The connected device, held exclusively by this process. A call that cannot get it in `timeout` says which job has it."""
    if not _device_lock.acquire(timeout=timeout):
        running = JOBS.running()
        if running:
            j = running[0].snapshot()
            raise RuntimeError(f"the EP-133 is busy with job {j['job_id']} ({j['kind']} '{j['name']}', {j['stage']}, about {j['remaining_seconds']} s left): poll job_status, or cancel_job, then retry")
        raise RuntimeError("the EP-133 is busy with another call; retry in a moment")
    try:
        cfg = _cfg()
        _require_midi(cfg)
        yield EP133(port_hint=cfg.midi_hint), cfg
    finally:
        _device_lock.release()


def _stage(job: Job | None, text: str) -> None:
    if job is not None:
        job.stage = text


def _cancel(job: Job | None):
    return job.cancel if job is not None else None


def _run_or_job(kind: str, name: str, expected_seconds: float, wait: bool | None, work) -> dict[str, Any]:
    """Run `work(job)` inline when it fits in a tool call, else on a background job. wait=True forces inline, wait=False forces a job."""
    if wait is True or (wait is None and expected_seconds <= SYNC_LIMIT):
        return work(None)
    job = JOBS.start(f"{C.new_id(f'job-{name}')}-{uuid4().hex[:4]}", kind, name, expected_seconds, work)
    return {"job_id": job.id, "status": "running", "kind": kind, "expected_seconds": round(expected_seconds, 1),
            "note": f"about {expected_seconds:.0f} s of work, longer than a chat tool call allows, so it runs in the background: poll job_status(\"{job.id}\") every few seconds until status is done; its result is exactly what a direct call returns."}


def _settle(job: Job | None) -> float:
    """Wait until SETTLE_SECONDS have passed since the last FILE session; returns how long it waited."""
    remaining = E.LAST_FILE_IO + SETTLE_SECONDS - time.monotonic()
    if remaining > 0:
        _stage(job, f"settling {remaining:.1f} s after FILE traffic")
        time.sleep(remaining)
        return remaining
    return 0.0


def _audio_index(cfg: C.Config, record: bool) -> int | None:
    if not record:
        return None
    dev = find_audio_input(cfg.audio_hint)
    return None if dev is None else dev["index"]


def _take_id(kind: str) -> str:
    return f"{C.new_id(kind)}-{uuid4().hex[:4]}"


def _save_take(tid: str, recording, meta: dict[str, Any]) -> dict[str, Any]:
    wav = C.file_path("takes", tid + ".wav")
    recording.save(wav)
    info = {"take_id": tid, "wav": wav, "samplerate": recording.samplerate, "marks": recording.marks, **meta, **player.levels(recording)}
    C.write_json(C.file_path("takes", tid + ".json"), info)
    return {k: v for k, v in info.items() if k != "marks"}


def _safe_id(take_id: str) -> str:
    if not take_id or any(c in take_id for c in "/\\\x00") or take_id in (".", ".."):
        raise ValueError(f"{take_id!r} is not a take id (list_takes shows them)")
    return take_id


def _wav_to_pcm(path: str, samplerate: int, mono: bool, normalize_dbfs: float | None) -> tuple[bytes, dict[str, Any]]:
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no such WAV: {path}")
    x, sr = sf.read(path, dtype="float64", always_2d=True)
    if mono and x.shape[1] > 1:
        x = x.mean(axis=1, keepdims=True)
    if x.shape[1] > 2:
        raise ValueError("the EP-133 takes mono or stereo samples")
    if sr != samplerate:
        g = math.gcd(int(sr), int(samplerate))
        x = resample_poly(x, samplerate // g, int(sr) // g, axis=0)
    if normalize_dbfs is not None:
        peak = float(np.max(np.abs(x))) if len(x) else 0.0
        if peak > 0:
            x = x * (10 ** (normalize_dbfs / 20.0) / peak)
    x = np.clip(x, -1.0, 1.0)
    seconds = len(x) / samplerate
    limit = 40.0 if x.shape[1] == 1 else 20.0
    if seconds > limit:
        raise ValueError(f"{seconds:.1f} s is over the device limit of {limit:.0f} s for {'mono' if x.shape[1] == 1 else 'stereo'} samples; trim the file first")
    pcm = (x * 32767.0).astype("<i2").tobytes()
    return pcm, {"source_samplerate": int(sr), "source_channels": int(sf.info(path).channels), "channels": int(x.shape[1]),
                 "frames": int(len(x)), "seconds": round(seconds, 3)}


# ------------------------------------------------------------------ status and knowledge

@mcp.tool(annotations=RO)
def get_status() -> dict[str, Any]:
    """Connection state of the EP-133, what the human has told the bridge about its settings, the active project, the
    library summary, running jobs and recent takes. Call this first."""
    cfg = _cfg()
    ports = _ports(cfg)
    connected = bool(ports["midi_out"] and ports["midi_in"])
    out: dict[str, Any] = {
        "device": {"midi_in": ports["midi_in"], "midi_out": ports["midi_out"], "audio_device": ports["audio"] and ports["audio"]["name"],
                   "connected": connected,
                   "hint": None if connected else "no EP-133 MIDI port on this computer: connect it over USB-C and leave it on its normal screen"},
        "settings_per_human": {"midi_clock": cfg.midi_clock, "note": "system settings cannot be read over USB; set_config records what the human set"},
        "default_bpm": cfg.default_bpm,
        "scratch_project": cfg.scratch_project,
        "latency_ms": cfg.latency_ms,
        "jobs_running": [j.snapshot() for j in JOBS.running()],
        "home": C.HOME,
    }
    lib = C.read_json(os.path.join(C.HOME, "library.json"))
    if lib:
        occupied = {int(n["id"]) for n in lib["sounds"]}
        out["library"] = {"listed_at": lib["time"], "sounds": len(occupied), "mb": round(sum(int(n["size"]) for n in lib["sounds"]) / 1e6, 2),
                          "first_free_slots_600_plus": [s for s in range(600, 1000) if s not in occupied][:8]}
    takes_dir = os.path.join(C.HOME, "takes")
    if os.path.isdir(takes_dir):
        files = sorted((f for f in os.listdir(takes_dir) if f.endswith(".json")), key=lambda f: os.path.getmtime(os.path.join(takes_dir, f)))
        out["takes"] = [f[:-5] for f in files][-10:]
    if connected and not JOBS.running() and _device_lock.acquire(timeout=0.2):
        try:
            out["active_project"] = EP133(port_hint=cfg.midi_hint).active_project()
        except Exception as e:
            out["active_project"] = {"error": f"{type(e).__name__}: {e}"}
        finally:
            _device_lock.release()
    if cfg.midi_clock != "in":
        out["notes_for_human"] = ["play_pattern needs MIDI clock set to receive on the device: human_steps('midi_clock_in'), then set_config(midi_clock='in')"]
    return out


@mcp.tool(annotations=RO)
def get_guide(topic: GuideTopic = "quickstart") -> str:
    """Reference: quickstart (a beat in five moves), device (what was verified on the real EP-133), project (the write_project spec), human (the human_steps topics)."""
    return knowledge.guide(topic)


@mcp.tool(annotations=RO)
def human_steps(topic: HumanStepTopic) -> str:
    """Exact key presses for what only the human can do at the device: MIDI clock and channel settings, USB audio input mode, sample rate, selecting a project or group, sound edit, sampling, power-cycling after an error."""
    return knowledge.human_steps(topic)


@mcp.tool(annotations=SETS)
def set_config(midi_clock: MidiClock | None = None, default_bpm: Bpm | None = None, scratch_project: Project | None = None) -> dict[str, Any]:
    """Record facts the bridge cannot read from the device: the MIDI clock setting the human made (off/in/out), the default BPM, and a project slot the human has declared expendable."""
    cfg = _cfg()
    if midi_clock is not None:
        cfg.midi_clock = midi_clock
    if default_bpm is not None:
        cfg.default_bpm = float(default_bpm)
    if scratch_project is not None:
        cfg.scratch_project = int(scratch_project)
    cfg.save()
    return {"midi_clock": cfg.midi_clock, "default_bpm": cfg.default_bpm, "scratch_project": cfg.scratch_project}


# ------------------------------------------------------------------ sounds

@mcp.tool(annotations=RO)
def list_sounds(min_slot: Slot = 1, max_slot: Slot = 999, refresh: bool = True) -> dict[str, Any]:
    """Occupied sample slots with name and size (factory banks: kick 1-99, snare 100-199, hat 200-299, perc 300-399, bass 400-499, melodic 500-599), and the free slots from 600 up. refresh=False reads the cached list."""
    lib = None if refresh else C.read_json(os.path.join(C.HOME, "library.json"))
    if lib is None:
        with _device() as (ep, _cfg_):
            nodes = ep.list_sounds()
        listed_at = time.strftime("%Y-%m-%d %H:%M:%S")
    else:
        nodes, listed_at = lib["sounds"], lib["time"]
    occupied = {int(n["id"]) for n in nodes}
    lib = L.Library()
    rows = []
    for n in sorted(nodes, key=lambda n: int(n["id"])):
        slot = int(n["id"])
        if not min_slot <= slot <= max_slot:
            continue
        rec = lib.get(slot)
        rows.append({"slot": slot, "name": (rec or {}).get("name") or n["name"], "category": (rec or {}).get("category"),
                     "kb": round(int(n["size"]) / 1024, 1)})
    free = [s for s in range(600, 1000) if s not in occupied]
    return {"listed_at": listed_at, "occupied": len(occupied), "total_mb": round(sum(int(n["size"]) for n in nodes) / 1e6, 2),
            "capacity_mb": 128, "sounds": rows, "free_slots_600_plus": free[:30], "free_count_600_plus": len(free),
            "note": "sizes are raw PCM bytes (2 bytes per frame per channel); describe_sound gives rate, channels and seconds"}


@mcp.tool(annotations=RO)
def describe_sound(slot: Slot) -> dict[str, Any]:
    """One slot's name, size, sample rate, channels, length in frames and seconds, and its metadata (trim, play mode, root note, bpm)."""
    with _device() as (ep, _c):
        return ep.sound_info(slot)


@mcp.tool(annotations=RW)
def audition_sound(slot: Slot) -> str:
    """Play a slot's raw sample on the device (no envelope or trim applied), to hear what is in it."""
    with _device() as (ep, _c):
        ep.audition(slot)
    return f"auditioned slot {slot}"


@mcp.tool(annotations=RW)
def upload_sound(wav_path: str, slot: Slot | None = None, name: str | None = None, samplerate: SampleRate = 46875, mono: bool = True,
                 normalize_dbfs: Annotated[float | None, Param(le=0, description="peak-normalise to this dBFS; None keeps the level")] = None,
                 acknowledge_overwrite: str | None = None, wait: bool | None = None) -> dict[str, Any]:
    """Convert a WAV (any rate, mono or stereo) to the device's 16-bit PCM at `samplerate` and upload it into `slot` (default: the first free slot from 600), byte-verified by reading it back. Replacing an occupied slot needs acknowledge_overwrite="OVERWRITE SOUND N". Long files run as a job (~24 KB/s)."""
    pcm, info = _wav_to_pcm(wav_path, samplerate, mono, normalize_dbfs)
    label = (name or os.path.splitext(os.path.basename(wav_path))[0])[:20]
    expected = len(pcm) / 24000.0 + len(pcm) / 30000.0 + 3.0

    def work(job: Job | None) -> dict[str, Any]:
        with _device() as (ep, _c):
            _stage(job, "listing slots")
            occupied = {int(n["id"]) for n in ep.list_sounds()}
            target = slot
            if target is None:
                free = ep.free_slots(occupied)
                if not free:
                    raise RuntimeError("no free slot from 600 up; delete_sound frees one")
                target = free[0]
            _stage(job, f"uploading {len(pcm) // 1024} KB to slot {target}")
            r = ep.upload_sound(target, pcm, name=label, samplerate=samplerate, channels=info["channels"],
                                acknowledge_overwrite=acknowledge_overwrite, occupied=occupied)
            ep.list_sounds()   # refresh the cache
            rec = L.Library().note_upload(target, pcm, samplerate, info["channels"], label, wav_path)
            r.update({"wav": wav_path, "source": info, "library": L.Library.row(rec),
                      "pad_hint": f"assign with assign_pads: slot {target}, trim_end {r['frames']} (or leave trim_end null)"})
            return r
    return _run_or_job("upload_sound", label, expected, wait, work)


@mcp.tool(annotations=RW)
def upload_sounds(wav_paths: list[str], start_slot: Slot | None = None, samplerate: SampleRate = 46875, mono: bool = True,
                  normalize_dbfs: float | None = None, wait: bool | None = None) -> dict[str, Any]:
    """Upload several WAVs into consecutive free slots (from start_slot, default the first free from 600), each byte-verified; runs as a job (a 12-chop crate of 2 s chops takes about 3 minutes)."""
    prepared = [(_wav_to_pcm(p, samplerate, mono, normalize_dbfs), os.path.splitext(os.path.basename(p))[0][:20], p) for p in wav_paths]
    total = sum(len(pcm) for (pcm, _i), _n, _p in prepared)
    expected = total / 24000.0 + total / 30000.0 + 3.0 * len(prepared)

    def work(job: Job | None) -> dict[str, Any]:
        results = []
        lib = L.Library()
        with _device() as (ep, _c):
            occupied = {int(n["id"]) for n in ep.list_sounds()}
            free = [s for s in ep.free_slots(occupied) if start_slot is None or s >= start_slot]
            if len(free) < len(prepared):
                raise RuntimeError(f"only {len(free)} free slots from {start_slot or 600}; need {len(prepared)}")
            for i, (((pcm, info), label, path), target) in enumerate(zip(prepared, free)):
                if job is not None and job.cancel.is_set():
                    break
                _stage(job, f"uploading {i + 1}/{len(prepared)}: {label} -> slot {target}")
                r = ep.upload_sound(target, pcm, name=label, samplerate=samplerate, channels=info["channels"], occupied=occupied)
                occupied.add(target)
                rec = lib.note_upload(target, pcm, samplerate, info["channels"], label, path)
                r.update({"wav": path, "source": info, "category": rec["category"]})
                results.append(r)
            ep.list_sounds()
        return {"uploaded": results, "slots": [r["slot"] for r in results], "bytes": sum(r["bytes"] for r in results),
                "seconds": round(sum(r["upload_seconds"] + r.get("readback_seconds", 0) for r in results), 1)}
    return _run_or_job("upload_sounds", f"crate-{len(prepared)}", expected, wait, work)


@mcp.tool(annotations=OVERWRITES)
def delete_sound(slot: Slot, acknowledge: Annotated[str, Param(description='must be exactly "DELETE SOUND <slot>"')], allow_factory: bool = False) -> dict[str, Any]:
    """Delete a sample slot (no undo on the device; the last full backup holds its PCM). Factory slots 1-599 need allow_factory=True."""
    with _device() as (ep, _c):
        out = ep.delete_sound(slot, acknowledge, allow_factory=allow_factory)
    if out.get("deleted"):
        L.Library().forget_slot(slot)
    return out


# ------------------------------------------------------------------ projects

@mcp.tool(annotations=RO)
def describe_project(project: Project) -> dict[str, Any]:
    """Decode a project slot: assigned pads by group and keypad label (slot, trim, root, release, play mode), patterns with their events and fader lanes, scenes and the scene cursor, BPM, fader assignment, FX."""
    with _device() as (ep, _c):
        return ep.describe_project(project)


def _spec(spec) -> ProjectSpec:
    return spec if isinstance(spec, ProjectSpec) else ProjectSpec.model_validate(spec)


def _write(project: int, spec: ProjectSpec, label: str, reload: bool, wait: bool | None) -> dict[str, Any]:
    expected = 15.0 + 1.5 * len({p.slot for p in spec.pads if p.slot and p.trim_end is None})

    def work(job: Job | None) -> dict[str, Any]:
        with _device() as (ep, _c):
            _stage(job, f"reading project {project}")
            ckpt, base, _s = ep.checkpoint(project, f"before-{label}")
            slots = PJ.referenced_slots(spec)
            frames: dict[int, int] = {}
            if slots:
                _stage(job, "checking sound slots")
                occupied = {int(n["id"]) for n in ep.list_sounds()}
                missing = sorted(slots - occupied)
                if missing:
                    raise RuntimeError(f"sound slots {missing} are empty on the device; a pad assigned to them would play nothing (list_sounds, upload_sound)")
                frames = ep.frames_for_slots({p.slot for p in spec.pads if p.slot and p.trim_end is None})
            template = PJ.find_settings_template(str(ep.backup_root)) if (spec.bpm is not None or spec.fader_bases) else None
            pitches = None
            if PJ.concert_slots(spec):
                lib = L.Library()
                pitches = {}
                for rec in lib.records.values():
                    f = rec.get("features", {})
                    if rec.get("slot") is not None and f.get("midi_note") is not None and f.get("pitch_confidence", 0) > 0.5:
                        pitches[rec["slot"]] = float(f["midi_note"])
            tar, doc = PJ.compile_spec(spec, base, frames, template, pitch_by_slot=pitches)
            _stage(job, f"writing {len(tar) // 1024} KB")
            result = ep.write_project(project, tar, label, reload=reload, base=base)
            result["checkpoint"] = str(ckpt)
            if not result.get("identical"):
                raise RuntimeError(f"project {project} read back differently after the write ({result.get('readback_bytes')} B); the checkpoint is {ckpt}; restore_checkpoint undoes it")
            result["project_after"] = PJ.describe(tar)
            result["document"] = doc
            result["note"] = "allow about 6 s before starting playback so the sequencer settles on the new project"
            return result
    return _run_or_job("write_project", f"P{project}-{label}", expected, wait, work)


@mcp.tool(annotations=OVERWRITES)
def write_project(project: Project, spec: Annotated[ProjectSpec, Param(description='the format of get_guide("project")')], reload: bool = True, wait: bool | None = None) -> dict[str, Any]:
    """Write pads, patterns, fader lanes, scene, BPM and FX into a project slot: checkpoint the slot, compile the spec over its current archive (unmentioned pads, patterns and members survive), write, read back and compare byte for byte, then reload so the device plays it. About 13 s for a 60 KB project."""
    return _write(project, _spec(spec), "write", reload, wait)


@mcp.tool(annotations=OVERWRITES)
def assign_pads(project: Project, pads: list[PadSpec], wait: bool | None = None) -> dict[str, Any]:
    """Put sample slots on pads (group A-D, keypad label "." "0" "enter" "1".."9") with trim, root note, release and play mode; the rest of the project is untouched."""
    pads = [p if isinstance(p, PadSpec) else PadSpec.model_validate(p) for p in pads]
    return _write(project, ProjectSpec(pads=pads), "pads", True, wait)


@mcp.tool(annotations=OVERWRITES)
def write_pattern(project: Project, pattern: PatternSpec, bpm: Bpm | None = None, set_scene: bool = True, wait: bool | None = None) -> dict[str, Any]:
    """Write one group pattern (events by 16th step or tick, fader lanes) and, by default, point scene 1 at it so play_pattern plays it; other groups keep their scene-1 pattern. bpm patches the project tempo."""
    pattern = pattern if isinstance(pattern, PatternSpec) else PatternSpec.model_validate(pattern)
    spec = ProjectSpec(bpm=bpm, patterns=[pattern], scene=PJ.SceneSpec(patterns={pattern.group: pattern.number}) if set_scene else None)
    return _write(project, spec, f"{pattern.id}", True, wait)


@mcp.tool(annotations=SETS)
def select_project(project: Project) -> dict[str, Any]:
    """Make a project slot the active one (what pads, notes and playback use), from the Mac."""
    with _device() as (ep, _c):
        out = ep.select_project(project)
    out["note"] = f"allow about {SETTLE_SECONDS:.0f} s before playing: the sequencer and the USB audio output settle after a switch"
    return out


@mcp.tool(annotations=RO)
def list_checkpoints(project: Project | None = None) -> list[dict[str, Any]]:
    """The archives saved before every write (newest last), for restore_checkpoint."""
    return EP133(port_hint=_cfg().midi_hint).checkpoints(project)


@mcp.tool(annotations=OVERWRITES)
def restore_checkpoint(project: Project, checkpoint: str | None = None) -> dict[str, Any]:
    """Write a saved archive back into a project slot (the newest checkpoint of that slot when none is named); itself checkpointed first."""
    with _device() as (ep, _c):
        if checkpoint is not None:
            real = os.path.realpath(os.path.expanduser(checkpoint))
            root = os.path.realpath(str(ep.checkpoint_dir))
            if not real.startswith(root + os.sep):
                raise PermissionError(f"checkpoints live under {root}; list_checkpoints shows them")
            checkpoint = real
        return ep.restore_checkpoint(project, checkpoint)


@mcp.tool(annotations=RW)
def backup_device(sounds: bool = True, wait: bool | None = None) -> dict[str, Any]:
    """Save every project TAR (seconds) and, with sounds=True, every sample slot with its metadata plus a .pak (about 30 minutes for a full library) under the bridge's backups folder. Runs as a job."""
    expected = 30.0 + (1800.0 if sounds else 0.0)

    def work(job: Job | None) -> dict[str, Any]:
        with _device() as (ep, _c):
            return ep.backup(sounds=sounds, stage=lambda t: _stage(job, t), cancel=_cancel(job))
    return _run_or_job("backup_device", "backup", expected, wait, work)


# ------------------------------------------------------------------ playing and listening

class LiveNote(BaseModel):
    """One pad hit at a time, by group and keypad label or by MIDI note."""
    time_s: float = Param(ge=0, description="when, from the start")
    group: Group | None = None
    pad: PadLabel | None = None
    note: int | None = Param(default=None, ge=0, le=127, description="MIDI note; alternative to group+pad (36-83 are the pads)")
    velocity: int = Param(default=100, ge=1, le=127)
    duration_s: float = Param(default=0.25, gt=0, le=30)
    channel: int | None = Param(default=None, ge=1, le=16)

    def midi_note(self) -> int:
        if self.note is not None:
            return self.note
        if self.group is None or self.pad is None:
            raise ValueError("a live note needs note, or group and pad")
        return PJ.pad_note(self.group, self.pad)


@mcp.tool(annotations=RW)
def play_pattern(bpm: Bpm | None = None, bars: Bars = 4, record: bool = True, project: Project | None = None, name: str | None = None,
                 wait: bool | None = None) -> dict[str, Any]:
    """Play the active project's current scene from the Mac: MIDI START then clock at `bpm` (default the bridge's default BPM) for `bars` bars, then STOP. Records the USB audio to a take and the device's MIDI note output, and grades onsets against the grid. The device must be set to receive MIDI clock (human_steps("midi_clock_in")). `project` selects a slot first."""
    cfg = _cfg()
    tempo = float(bpm or cfg.default_bpm)
    expected = bars * 240.0 / tempo + 6.0 + (4.0 + SETTLE_SECONDS if project else 0.0)

    def work(job: Job | None) -> dict[str, Any]:
        with _device() as (ep, cfg2):
            if project is not None:
                _stage(job, f"selecting project {project}")
                ep.select_project(project)
                _stage(job, "settling after the project switch")
                time.sleep(SETTLE_SECONDS)
            active = ep.active_project()
            desc = ep.describe_project(active["project"]) if active.get("project") else {}
            out, res, expected_events = _clocked_play(ep, cfg2, desc, tempo, bars, record, job)
        out["project"] = active.get("project")
        if "recording" in res:
            rec = res["recording"]
            grade = player.grade(rec, res["start_sample"], [e["time_s"] for e in expected_events]) if expected_events else {}
            take = _save_take(_take_id(name or "pattern"), rec, {"kind": "pattern", "bpm": tempo, "bars": bars, "start_sample": res["start_sample"],
                                                               "expected": expected_events, "grade": grade, "midi_note_ons": res["midi_note_ons"]})
            out.update({"take": take, "grade": grade})
        return out
    return _run_or_job("play_pattern", name or "pattern", expected, wait, work)


def _clocked_play(ep: EP133, cfg: C.Config, desc: dict[str, Any], tempo: float, bars: int, record: bool, job: Job | None):
    """Settle after FILE traffic, then START + clock for `bars` bars on the active project; returns (summary, raw result, expected events)."""
    expected_events = PJ.expected_events(desc, tempo, bars) if desc else []
    settled = _settle(job)
    _stage(job, "playing")
    with Midi(cfg.midi_hint) as midi:
        res = player.play_pattern(midi, tempo, bars, audio_device=_audio_index(cfg, record), stage=lambda t: _stage(job, t), cancel=_cancel(job))
    out: dict[str, Any] = {"bpm": tempo, "bars": bars, "midi_note_ons": len(res["midi_note_ons"]), "first_midi_note_ons": res["midi_note_ons"][:12],
                           "clock_jitter_ms": res["clock_jitter_ms"], "expected_events": len(expected_events), "settled_seconds": round(settled, 1)}
    if res.get("note"):
        out["note"] = res["note"]
    ons = res["midi_note_ons"]
    if ons and all(e["time_s"] < 0.05 for e in ons) and bars * 240.0 / tempo > 1.0:
        out["warning"] = ("the device fired notes only at START and then nothing: the sequencer was not settled (this happens "
                          "when playback follows FILE traffic too closely); wait about 6 s and play again")
    if ons and cfg.midi_clock != "in":
        cfg2 = C.Config.load(); cfg2.midi_clock = "in"; cfg2.save()      # the device clearly follows the clock; remember that
    return out, res, expected_events


class ClockNotReceived(RuntimeError):
    pass


@mcp.tool(annotations=RW)
def render_song(project: Project, positions: list[int] | None = None, bpm: Bpm | None = None, name: str | None = None, wait: bool | None = None) -> dict[str, Any]:
    """Record a project's whole song from the Mac with nobody at the device: for each song position the scene cursor is
    moved to its scene (a checkpointed cursor-only write, about 13 s), the scene plays under MIDI clock for the length of
    its longest pattern and is recorded, and the pieces are stitched into one WAV (plus one take per position with its
    MIDI grade). The device must be set to receive MIDI clock (101) once; the song mode of the device itself is not used.
    Runs as a job; cancel_job stops after the current position."""
    def work(job: Job | None) -> dict[str, Any]:
        import numpy as np

        with _device() as (ep, cfg):
            desc = ep.describe_project(project)
            song = [int(p) for p in (positions or desc.get("song") or [desc.get("scene_cursor") or 1])]
            tempo = float(bpm or desc.get("bpm") or cfg.default_bpm)
            scenes = {s["scene"]: s for s in desc.get("scenes", [])}
            if not scenes:
                raise RuntimeError(f"project {project} has no scenes to render")
            original_cursor = int(desc.get("scene_cursor") or 1)
            active = ep.active_project().get("project")
            if active != project:
                _stage(job, f"selecting project {project}")
                ep.select_project(project)
            cursor = original_cursor
            pieces, takes = [], []
            sr = 48000
            bar_s = 240.0 / tempo
            for index, scene_no in enumerate(song):
                if job is not None and job.cancel.is_set():
                    break
                scene = scenes.get(scene_no)
                if scene is None:
                    raise RuntimeError(f"song position {index + 1} names scene {scene_no}, which is not defined")
                bars = max([desc["patterns"][f"{g}{n:02d}"]["bars"] for g, n in scene["patterns"].items() if n and f"{g}{n:02d}" in desc["patterns"]] or [1])
                if cursor != scene_no:
                    _stage(job, f"position {index + 1}/{len(song)}: cursor to scene {scene_no}")
                    base, _m, _s = ep.read_project(project)
                    tar, _doc = PJ.compile_spec(ProjectSpec(current_scene=scene_no), base, {})
                    w = ep.write_project(project, tar, f"cursor-{scene_no}", reload=True, base=base)
                    if not w.get("identical"):
                        raise RuntimeError(f"cursor write for scene {scene_no} read back differently; checkpoint {w['checkpoint']}")
                    cursor = scene_no
                _stage(job, f"position {index + 1}/{len(song)}: scene {scene_no}, {bars} bars")
                out, res, expected = _clocked_play(ep, cfg, {**desc, "scene_cursor": scene_no}, tempo, bars, True, job)
                if not res["midi_note_ons"] and expected:
                    raise ClockNotReceived("the device sent no notes under the Mac's clock: set MIDI clock to receive once "
                                           "(SHIFT+ERASE, 1 0 1, MAIN) and render again; nothing else is needed at the device")
                rec = res["recording"]
                sr = rec.samplerate
                start = res["start_sample"]
                length = int(round(bars * bar_s * sr))
                pieces.append(rec.audio[start: start + length])
                grade = player.grade(rec, start, [e["time_s"] for e in expected]) if expected else {}
                take = _save_take(_take_id(f"{name or 'song'}-pos{index + 1}"), rec, {"kind": "pattern", "bpm": tempo, "bars": bars, "scene": scene_no,
                                                                                      "start_sample": start, "expected": expected, "grade": grade,
                                                                                      "midi_note_ons": res["midi_note_ons"]})
                takes.append({"position": index + 1, "scene": scene_no, "bars": bars, "midi_note_ons": out["midi_note_ons"], "expected_events": out["expected_events"],
                              "take_id": take["take_id"], "warning": out.get("warning")})
            if cursor != original_cursor:
                _stage(job, f"cursor back to scene {original_cursor}")
                base, _m, _s = ep.read_project(project)
                tar, _doc = PJ.compile_spec(ProjectSpec(current_scene=original_cursor), base, {})
                ep.write_project(project, tar, f"cursor-{original_cursor}", reload=True, base=base)
        if not pieces:
            return {"positions": takes, "note": "nothing rendered"}
        audio = np.concatenate(pieces, axis=0)
        from .audio import Recording
        stitched = Recording(audio=audio, samplerate=sr)
        take = _save_take(_take_id(name or "song"), stitched, {"kind": "song", "project": project, "bpm": tempo, "song": song[: len(pieces)], "start_sample": 0,
                                                              "positions": takes})
        return {"song": take, "bpm": tempo, "seconds": round(len(audio) / sr, 2), "positions": takes}
    bars_total = 0
    try:
        with _device(timeout=1.0) as (ep, cfg):
            desc = ep.describe_project(project)
        song = positions or desc.get("song") or [desc.get("scene_cursor") or 1]
        scenes = {s["scene"]: s for s in desc.get("scenes", [])}
        for scene_no in song:
            sc = scenes.get(int(scene_no))
            if sc:
                bars_total += max([desc["patterns"][f"{g}{n:02d}"]["bars"] for g, n in sc["patterns"].items() if n and f"{g}{n:02d}" in desc["patterns"]] or [1])
        tempo = float(bpm or desc.get("bpm") or cfg.default_bpm)
        expected_seconds = len(song) * (14.0 + SETTLE_SECONDS + 2.0) + bars_total * 240.0 / tempo + 16.0
    except Exception:
        expected_seconds = 180.0
    return _run_or_job("render_song", name or "song", expected_seconds, wait, work)


@mcp.tool(annotations=RW)
def play_notes(notes: list[LiveNote], record: bool = True, name: str | None = None, wait: bool | None = None) -> dict[str, Any]:
    """Play pads live from the Mac at the given times (group + keypad label, or MIDI note), on the active project; records the result and reports the measured note-to-audio latency of each hit."""
    notes = [n if isinstance(n, LiveNote) else LiveNote.model_validate(n) for n in notes]
    events = [{"time_s": n.time_s, "note": n.midi_note(), "velocity": n.velocity, "duration_s": n.duration_s, "channel": n.channel} for n in notes]
    length = max((e["time_s"] + e["duration_s"] for e in events), default=0.0)
    expected = length + 3.0

    def work(job: Job | None) -> dict[str, Any]:
        with _device() as (ep, cfg):
            settled = _settle(job)
            with Midi(cfg.midi_hint) as midi:
                res = player.play_notes(midi, events, audio_device=_audio_index(cfg, record), cancel=_cancel(job))
        out: dict[str, Any] = {"sent": res["sent"], "settled_seconds": round(settled, 1)}
        if "recording" in res:
            out["latency_stats_ms"] = res.get("latency_stats_ms")
            out["take"] = _save_take(_take_id(name or "notes"), res["recording"], {"kind": "notes", "start_sample": res["start_sample"], "sent": res["sent"]})
        return out
    return _run_or_job("play_notes", name or "notes", expected, wait, work)


@mcp.tool(annotations=RW)
def record_audio(seconds: Annotated[float, Param(gt=0, le=600)], name: str | None = None, wait: bool | None = None) -> dict[str, Any]:
    """Record the EP-133's USB audio for a while (the human plays, or the device runs its own sequencer) into a take."""
    def work(job: Job | None) -> dict[str, Any]:
        with _device() as (ep, cfg):
            dev = find_audio_input(cfg.audio_hint)
            if dev is None:
                raise RuntimeError("no EP-133 audio device: the device must be connected over USB with OS 2.5 or later")
            _settle(job)
            _stage(job, f"recording {seconds:g} s")
            rec = player.record(dev["index"], float(seconds))
        return {"take": _save_take(_take_id(name or "record"), rec, {"kind": "record", "start_sample": rec.mark_sample("start")})}
    return _run_or_job("record_audio", name or "record", float(seconds) + 2.0, wait, work)


@mcp.tool(annotations=RO)
def list_takes() -> list[dict[str, Any]]:
    """Recorded takes (newest last): id, kind, seconds, peak."""
    d = os.path.join(C.HOME, "takes")
    if not os.path.isdir(d):
        return []
    out = []
    for f in sorted((f for f in os.listdir(d) if f.endswith(".json")), key=lambda f: os.path.getmtime(os.path.join(d, f))):
        info = C.read_json(os.path.join(d, f), {}) or {}
        out.append({k: info.get(k) for k in ("take_id", "kind", "seconds", "peak_dbfs", "bpm", "bars")})
    return out


@mcp.tool(annotations=RO)
def get_take(take_id: str) -> dict[str, Any]:
    """A take's record: WAV path, levels, the events that were played or expected, and the timing grade."""
    info = C.read_json(os.path.join(C.HOME, "takes", _safe_id(take_id) + ".json"))
    if info is None:
        raise FileNotFoundError(f"no take {take_id!r} (list_takes shows them)")
    return {k: v for k, v in info.items() if k != "marks"}


@mcp.tool(annotations=RO)
def measure_take(take_id: str, expected_times_s: list[float] | None = None, bpm: Bpm | None = None, ticks: list[int] | None = None) -> dict[str, Any]:
    """Grade a take's onsets against expected times (seconds from the take's start mark, or ticks at a BPM); without them, list every onset and silent span."""
    import soundfile as sf
    from .audio import Recording

    info = get_take(take_id)
    audio, sr = sf.read(info["wav"], dtype="float32", always_2d=True)
    rec = Recording(audio=audio, samplerate=sr)
    start = int(info.get("start_sample") or 0)
    if ticks is not None:
        expected_times_s = [t * 60.0 / (float(bpm or _cfg().default_bpm) * PJ.PPQN) for t in ticks]
    return player.grade(rec, start, expected_times_s or [])


# ------------------------------------------------------------------ the sample library

Category = Literal[tuple(L.CATEGORIES)]
Bank = Literal["kick", "snare", "hat", "perc", "bass", "melodic", "user"]


@mcp.tool(annotations=RO)
def library_status() -> dict[str, Any]:
    """The companion sample library: how many sounds are indexed (device slots from the last backup, imports, uploads), how many are loaded on the device, and when it was built."""
    return L.Library().status()


@mcp.tool(annotations=RO)
def search_sounds(query: str | None = None, labels: list[str] | None = None, exclude: list[str] | None = None,
                  min_seconds: float | None = None, max_seconds: float | None = None, pitched: bool | None = None,
                  bank: Bank | None = None, on_device: bool | None = None, pitch_class: str | None = None, limit: int = 25) -> list[dict[str, Any]]:
    """Find sounds by words in their name or labels, by labels (a category such as kick, snare, hat_closed, hat_open, bass, reese, chord, pad, vocal, fx, plus descriptors such as short, long, punchy, bright, dark, tonal, noisy, stereo, and bank:kick..bank:melodic), by length, pitch, bank, or whether they are loaded on the device. list_labels shows every label with counts."""
    return L.Library().search(query=query, labels=labels, exclude=exclude, min_seconds=min_seconds, max_seconds=max_seconds, pitched=pitched,
                              bank=bank, on_device=on_device, pitch_class=pitch_class, limit=limit)


@mcp.tool(annotations=RO)
def list_labels() -> dict[str, Any]:
    """Every label in the library with its count and three example sounds, plus the fixed category and descriptor vocabularies."""
    return L.Library().labels()


@mcp.tool(annotations=RO)
def describe_library_sound(sound: str) -> dict[str, Any]:
    """Everything the library measured about one sound (a slot number or a library id): length, attack, decay, sustain, brightness, pitch and confidence, low/high energy shares, device metadata, labels and tags."""
    rec = L.Library().get(sound)
    if rec is None:
        raise KeyError(f"no library record for {sound!r} (search_sounds finds ids; rebuild_library indexes the last backup)")
    return rec


@mcp.tool(annotations=SETS)
def tag_sound(sound: str, add: list[str] | None = None, remove: list[str] | None = None) -> dict[str, Any]:
    """Add or remove free-form tags on a library sound (a slot number or id), e.g. "boop", "amen", "dark"; tags are searchable like labels."""
    return L.Library.row(L.Library().tag(sound, add, remove))


@mcp.tool(annotations=RW)
def import_sounds(paths: list[str], tags: list[str] | None = None, origin: str | None = None, license_note: str | None = None) -> list[dict[str, Any]]:
    """Copy WAVs into the library and index them with measured features and guessed labels, without touching the device; load_sound puts one on a slot later. Record where they came from and under what licence."""
    lib = L.Library()
    out = []
    for p in paths:
        p = os.path.expanduser(p)
        files = sorted(glob.glob(os.path.join(p, "*.wav")) + glob.glob(os.path.join(p, "*.WAV"))) if os.path.isdir(p) else [p]
        for f in files:
            out.append(L.Library.row(lib.import_wav(f, tags=tags, origin=origin, license_note=license_note)))
    return out


@mcp.tool(annotations=RW)
def load_sound(sound: str, slot: Slot | None = None, samplerate: SampleRate = 46875, mono: bool = True, wait: bool | None = None) -> dict[str, Any]:
    """Upload a library sound (by id) to a device slot (default the first free from 600), converted and byte-verified; the library then knows which slot holds it."""
    rec = L.Library().get(sound)
    if rec is None:
        raise KeyError(f"no library record {sound!r}")
    path = (rec.get("source") or {}).get("path")
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"the library has no audio file for {sound!r} ({path})")
    if path.endswith(".pcm"):
        raise ValueError(f"{sound!r} is a device slot already (slot {rec.get('slot')}); upload_sound takes WAVs")
    return upload_sound(path, slot=slot, name=rec.get("name"), samplerate=samplerate, mono=mono, wait=wait)


@mcp.tool(annotations=RW)
def sync_library(wait: bool | None = None) -> dict[str, Any]:
    """Bring the library in line with the device: index slots it does not know (read from the device, about 7 s each) and forget slots that were deleted on the device."""
    def work(job: Job | None) -> dict[str, Any]:
        with _device() as (ep, _c):
            return L.Library().sync_with_device(ep, stage=lambda t: _stage(job, t))
    return _run_or_job("sync_library", "library-sync", 20.0, wait, work)


@mcp.tool(annotations=RW)
def rebuild_library(backup: str | None = None) -> dict[str, Any]:
    """Re-index every device slot from the newest backup (or the named backup folder), keeping imports and tags; a few seconds, no device traffic."""
    return L.Library().rebuild_from_backup(backup)


# ------------------------------------------------------------------ jobs

@mcp.tool(annotations=RO)
def job_status(job_id: str) -> dict[str, Any]:
    """Progress of a background job; when done, its result is exactly what a direct call would have returned."""
    job = JOBS.get(job_id)
    if job is None:
        raise KeyError(f"no job {job_id!r} (list_jobs shows them)")
    return job.snapshot()


@mcp.tool(annotations=RO)
def list_jobs() -> list[dict[str, Any]]:
    """Running and recent jobs."""
    return [j.snapshot() for j in JOBS.all()]


@mcp.tool(annotations=RW)
def cancel_job(job_id: str) -> dict[str, Any]:
    """Ask a running job to stop between steps (a backup or crate upload stops after the current file)."""
    return JOBS.cancel(job_id).snapshot()


def run(transport: str = "stdio", host: str = "127.0.0.1", port: int = 8766, path_secret: str | None = None, public: bool = False) -> None:
    if transport == "stdio":
        _server.run(transport="stdio")
        return
    from mcp.server.transport_security import TransportSecuritySettings
    path = f"/{path_secret}/mcp" if path_secret else "/mcp"
    sec = TransportSecuritySettings(enable_dns_rebinding_protection=False) if public else None
    _server.run(transport="streamable-http", host=host, port=port, streamable_http_path=path, transport_security=sec)
