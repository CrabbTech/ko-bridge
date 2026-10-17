"""Authoring an EP-133 project: typed specs in, an epsysex JSON document (compiled over the slot's current archive) out.

Pads are addressed the way the device labels them: group A-D and the keypad label "." "0" "enter" "1".."9". The archive's
pNN numbering and the MIDI note are derived (epsysex midi-reference.md, calibrated on the device):

    label:   .    0   enter  1    2    3    4    5    6    7    8    9
    note:   +0   +1   +2    +3   +4   +5   +6   +7   +8   +9  +10  +11   (group base: A 36, B 48, C 60, D 72)
    member: p10  p11  p12   p07  p08  p09  p04  p05  p06  p01  p02  p03

Timing is 96 PPQN: a 16th step is 24 ticks, a bar 384. A pattern event's note is the pitch relative to the pad's root
note and is also what the device sends out over MIDI, so the default root is the pad's own note and the default event
note is that root: native pitch unless asked otherwise. Fader lanes take parameter names and 0..1 values.
"""
from __future__ import annotations

import base64
import glob
import os
import re
import struct
from typing import Any, Literal

from epsysex.compile import compile_project
from epsysex.tar import iter_members, member_payloads
from pydantic import BaseModel, Field

PPQN = 96
STEP_TICKS = 24
BAR_TICKS = 384

Group = Literal["A", "B", "C", "D"]
PAD_LABELS = (".", "0", "enter", "1", "2", "3", "4", "5", "6", "7", "8", "9")   # by note offset 0..11
PAD_MEMBER_BY_OFFSET = (10, 11, 12, 7, 8, 9, 4, 5, 6, 1, 2, 3)                   # archive pNN for the same offsets
GROUP_BASE_NOTE = {"A": 36, "B": 48, "C": 60, "D": 72}
PadLabel = Literal[".", "0", "enter", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

FADER_PARAMETERS = ("level", "pitch", "time", "lowpass", "highpass", "fx", "attack", "release", "pan", "tune", "velocity", "modulation")
FaderParameter = Literal["level", "pitch", "time", "lowpass", "highpass", "fx", "attack", "release", "pan", "tune", "velocity", "modulation"]
TIME_MODES = ("off", "bpm", "bar", "reverse")
PLAY_MODES = ("oneshot", "key", "legato")
FX_TYPES = ("off", "delay", "reverb", "distortion", "chorus", "filter", "compressor")


# ----------------------------------------------------------------------------- pad addressing

def pad_offset(label: str) -> int:
    label = str(label).lower()
    if label not in PAD_LABELS:
        raise ValueError(f"pad {label!r} is not a keypad label; use one of {PAD_LABELS}")
    return PAD_LABELS.index(label)


def pad_member_number(label: str) -> int:
    """Archive pad number 1-12 (pads/<g>/pNN) for a keypad label."""
    return PAD_MEMBER_BY_OFFSET[pad_offset(label)]


def pad_note(group: str, label: str) -> int:
    return GROUP_BASE_NOTE[group.upper()] + pad_offset(label)


def label_for_member(number: int) -> str:
    return PAD_LABELS[PAD_MEMBER_BY_OFFSET.index(number)]


def pad_for_note(note: int) -> dict[str, Any]:
    if not 36 <= note <= 83:
        raise ValueError(f"note {note} is outside the pad range 36..83")
    group = "ABCD"[(note - 36) // 12]
    offset = (note - 36) % 12
    return {"note": note, "group": group, "pad": PAD_LABELS[offset], "member": f"pads/{group.lower()}/p{PAD_MEMBER_BY_OFFSET[offset]:02d}"}


# ----------------------------------------------------------------------------- specs

class PadSpec(BaseModel):
    """One pad's sound assignment and sound design (the per-project pad record)."""
    group: Group
    pad: PadLabel = Field(description='keypad label: "." "0" "enter" or "1".."9"')
    slot: int = Field(ge=0, le=999, description="sample slot 1-999; 0 clears the pad")
    trim_start: int = Field(default=0, ge=0, description="frames at the sample's native rate")
    trim_end: int | None = Field(default=None, ge=1, description="trim LENGTH in frames from trim_start (not an end frame; the device resets a pad whose start + length exceeds the sample to the whole sample); None = the whole sample")
    sample_bpm: float = Field(default=120.0, ge=1, le=399)
    amplitude: int = Field(default=100, ge=0, le=200, description="100 = unity, ~0.5 dB per unit below")
    pitch: int = Field(default=0, ge=-12, le=12, description="semitones")
    pan: int = Field(default=0, ge=-16, le=16)
    attack: int = Field(default=0, ge=0, le=255)
    release: int = Field(default=255, ge=0, le=255, description="255 = play through; lower = decay from note-on in oneshot")
    time_mode: Literal["off", "bpm", "bar", "reverse"] = "off"
    choke: bool = Field(default=False, description="member of the single shared mute group")
    play_mode: Literal["oneshot", "key", "legato"] = "oneshot"
    root_note: int | None = Field(default=None, ge=0, le=127, description="None = the pad's own MIDI note (native pitch when that note plays)")
    midi_channel: int = Field(default=0, ge=0, le=16, description="0 = the device's channel setting")

    @property
    def note(self) -> int:
        return pad_note(self.group, self.pad)

    @property
    def root(self) -> int:
        return self.note if self.root_note is None else self.root_note


NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
FLATS = {"DB": "C#", "EB": "D#", "GB": "F#", "AB": "G#", "BB": "A#"}


def note_number(name: str) -> int:
    """MIDI number of a note name such as "E3", "Bb2", "F#4" (C4 = 60)."""
    m = re.fullmatch(r"\s*([A-Ga-g])([#b]?)(-?\d+)\s*", str(name))
    if not m:
        raise ValueError(f"{name!r} is not a note name like E3 or Bb2")
    letter = (m.group(1).upper() + ("#" if m.group(2) == "#" else "")).upper()
    if m.group(2) == "b":
        letter = FLATS[(m.group(1).upper() + "B")]
    return NOTE_NAMES.index(letter) + 12 * (int(m.group(3)) + 1)


class NoteEvent(BaseModel):
    """One hit. Pitch, three ways: `note` (the raw byte, relative to the pad's root), `semitones` (offset from the pad's
    native pitch), or `concert` (a note name the sound should actually sound at, resolved through the library's measured
    pitch of the sample). Leaving all three out plays the sample at its native pitch."""
    pad: PadLabel
    step: float | None = Field(default=None, ge=0, description="position in 16th steps from the pattern start (24 ticks each); fractions allowed")
    tick: int | None = Field(default=None, ge=0, description="position in ticks (96 per beat); alternative to step")
    note: int | None = Field(default=None, ge=0, le=127, description="raw note byte: the pad's root = native pitch; other values repitch")
    semitones: int | None = Field(default=None, ge=-60, le=60, description="offset from the pad's native pitch; alternative to note")
    concert: str | None = Field(default=None, description='note name the sound should sound at, e.g. "E3" (needs the sample pitch from the library)')
    velocity: int = Field(default=100, ge=1, le=127)
    duration: int = Field(default=24, ge=1, le=BAR_TICKS, description="ticks; audible only in key/legato play modes")

    def tick_value(self) -> int:
        if self.tick is not None:
            return int(self.tick)
        if self.step is not None:
            return int(round(self.step * STEP_TICKS))
        raise ValueError("a note event needs step or tick")


class LanePoint(BaseModel):
    step: float | None = Field(default=None, ge=0)
    tick: int | None = Field(default=None, ge=0)
    value: float = Field(ge=0.0, le=1.0, description="0..1 of the fader's range")

    def tick_value(self) -> int:
        if self.tick is not None:
            return int(self.tick)
        if self.step is not None:
            return int(round(self.step * STEP_TICKS))
        raise ValueError("a lane point needs step or tick")


class Lane(BaseModel):
    """A group-fader automation lane; values latch until the next point and slew over tens of milliseconds."""
    parameter: FaderParameter
    points: list[LanePoint]


class PatternSpec(BaseModel):
    group: Group
    number: int = Field(default=1, ge=1, le=99)
    bars: int = Field(default=1, ge=1, le=99)
    events: list[NoteEvent] = []
    lanes: list[Lane] = []

    @property
    def member(self) -> str:
        return f"patterns/{self.group.lower()}{self.number:02d}"

    @property
    def id(self) -> str:
        return f"{self.group}{self.number:02d}"


class SceneSpec(BaseModel):
    """Which pattern each group plays; groups left out keep that scene's current reference (or scene 1's, or an empty pattern)."""
    patterns: dict[Group, int] = Field(default_factory=dict, description="group -> pattern number")
    time_signature: list[int] = Field(default=[4, 4], min_length=2, max_length=2)


class SidechainSource(BaseModel):
    group: Group
    pad: PadLabel


class FxSpec(BaseModel):
    type: Literal["off", "delay", "reverb", "distortion", "chorus", "filter", "compressor"] = "off"
    x: float | None = Field(default=None, ge=0, le=1, description="knob X: delay length, reverb length, drive, modulation, cutoff, drive")
    y: float | None = Field(default=None, ge=0, le=1, description="knob Y: feedback, color, color, feedback, resonance, speed")
    output_compressor_drive: float | None = Field(default=None, ge=0, le=1)
    output_compressor_speed: float | None = Field(default=None, ge=0, le=1)
    sidechain_sources: list[SidechainSource] = Field(default=[], description="pads whose hits duck the destination groups (note-triggered sidechain)")
    sidechain_destinations: list[Group] = Field(default=[], description="groups that get ducked")
    sidechain_length: float | None = Field(default=None, ge=0, le=1, description="duck length (SHIFT+FX knob X)")
    sidechain_shape: float | None = Field(default=None, ge=0, le=1, description="duck shape (SHIFT+FX knob Y)")


class FaderBase(BaseModel):
    group: Group
    parameter: FaderParameter
    value: float = Field(ge=0, le=1)


class ProjectSpec(BaseModel):
    """Everything write_project can change in a slot. Unmentioned members, pads and patterns are preserved."""
    bpm: float | None = Field(default=None, ge=40, le=399)
    pads: list[PadSpec] = []
    patterns: list[PatternSpec] = []
    scene: SceneSpec | None = Field(default=None, description="set scene 1; None keeps the slot's scenes untouched unless patterns are written")
    scenes: list[SceneSpec] = Field(default=[], description="scenes 1..N in order (an arrangement); overrides `scene`")
    song: list[int] = Field(default=[], description="song order as 1-based scene numbers, played by the device's own song mode")
    current_scene: int | None = Field(default=None, ge=1, le=99, description="the scene external MIDI START plays (default 1 when scenes are written; alone, it moves the cursor and keeps the scenes)")
    fader_bases: list[FaderBase] = Field(default=[], description="group fader base positions and assignment (settings member)")
    fx: FxSpec | None = None


# ----------------------------------------------------------------------------- archive decoding

def decode_pad(record: bytes) -> dict[str, Any]:
    return {"slot": struct.unpack_from("<H", record, 1)[0], "midi_channel": record[3],
            "trim_start": struct.unpack_from("<I", record, 4)[0], "trim_end": struct.unpack_from("<I", record, 8)[0],
            "sample_bpm": round(struct.unpack_from("<f", record, 12)[0], 2), "amplitude": record[16],
            "pitch": struct.unpack_from("<b", record, 17)[0], "pan": struct.unpack_from("<b", record, 18)[0],
            "attack": record[19], "release": record[20], "time_mode": TIME_MODES[record[21]] if record[21] < 4 else record[21],
            "choke": bool(record[22]), "play_mode": PLAY_MODES[record[23]] if record[23] < 3 else record[23],
            "root_note": record[24], "record_bytes": len(record)}


def decode_pattern(payload: bytes) -> dict[str, Any]:
    bars, count = payload[1], payload[2]
    events, lanes = [], {}
    for i in range(count):
        r = payload[4 + i * 8: 12 + i * 8]
        if len(r) < 8:
            break
        tick = r[0] | (r[1] << 8)
        if r[2] == 1 and r[4] == 0 and r[3] < 12:      # automation record
            lanes.setdefault(FADER_PARAMETERS[r[3]], []).append({"tick": tick, "step": round(tick / STEP_TICKS, 3),
                                                                  "value": round((r[5] | (r[6] << 8)) / 32767, 4)})
        else:
            number = r[2] // 8 + 1
            events.append({"tick": tick, "step": round(tick / STEP_TICKS, 3), "pad": label_for_member(number) if 1 <= number <= 12 else r[2],
                           "note": r[3], "velocity": r[4], "duration": r[5] | (r[6] << 8), "flag": r[7]})
    return {"bars": bars, "records": count, "events": events, "lanes": lanes}


def describe(tar: bytes) -> dict[str, Any]:
    """A readable view of a project archive: assigned pads by label, decoded patterns, scene 1, cursor, BPM, FX."""
    members = list(iter_members(tar))
    files = member_payloads(tar)
    pads = {}
    for name, p in files.items():
        if name.startswith("pads/") and len(p) >= 26:
            d = decode_pad(p)
            if d["slot"]:
                group = name.split("/")[1].upper()
                number = int(name[-2:])
                label = label_for_member(number)
                d.update({"note": pad_note(group, label), "member": name})
                pads[f"{group}:{label}"] = d
    patterns = {n[9:].upper(): decode_pattern(p) for n, p in files.items() if n.startswith("patterns/") and len(p) >= 4}
    out: dict[str, Any] = {"bytes": len(tar), "members": len(members), "pads": pads, "patterns": patterns,
                           "has_settings": "settings" in files, "has_fx_settings": "fx_settings" in files}
    if "settings" in files and len(files["settings"]) >= 220:
        st = files["settings"]
        out["bpm"] = round(struct.unpack_from("<f", st, 4)[0], 2)
        out["fader_assignment"] = {g: FADER_PARAMETERS[st[216 + i]] if st[216 + i] < 12 else st[216 + i] for i, g in enumerate("ABCD")}
    if "fx_settings" in files and len(files["fx_settings"]) >= 12:
        fx = files["fx_settings"]
        out["fx"] = {"type": FX_TYPES[fx[4]] if fx[4] < 7 else fx[4]}
        if 1 <= fx[4] <= 6 and len(fx) >= 100:
            out["fx"]["x"] = round(struct.unpack_from("<f", fx, 12 + (fx[4] - 1) * 4)[0], 3)
            out["fx"]["y"] = round(struct.unpack_from("<f", fx, 76 + (fx[4] - 1) * 4)[0], 3)
    if "scenes" in files and len(files["scenes"]) == 712:
        sc = files["scenes"]
        cursor = sc[7 + 99 * 6 + 3]
        scenes = []
        for i in range(99):
            chunk = sc[7 + i * 6: 13 + i * 6]
            if any(chunk[:4]):
                scenes.append({"scene": i + 1, "patterns": {g: chunk[j] for j, g in enumerate("ABCD")}, "time_signature": [chunk[4], chunk[5]]})
        out["scenes"] = scenes
        out["scene_cursor"] = cursor
        out["song"] = list(sc[7 + 99 * 6 + 12: 7 + 99 * 6 + 12 + sc[7 + 99 * 6 + 11]])
    return out


def scene_refs(tar: bytes, scene: int = 1) -> dict[str, int]:
    """Scene `scene` (1-based) group references from an archive's scenes member, {} when absent or undefined."""
    files = member_payloads(tar)
    if "scenes" in files and len(files["scenes"]) == 712:
        chunk = files["scenes"][7 + (scene - 1) * 6: 11 + (scene - 1) * 6]
        if any(chunk):
            return {g: chunk[i] for i, g in enumerate("ABCD")}
    return {}


def scene_one(tar: bytes) -> dict[str, int]:
    return scene_refs(tar, 1)


def scenes_from_base(tar: bytes) -> tuple[list["SceneSpec"], list[int]]:
    """Every defined scene of an archive as SceneSpecs, plus its song order (for cursor-only rewrites)."""
    files = member_payloads(tar)
    sc = files.get("scenes")
    if sc is None or len(sc) != 712:
        return [], []
    scenes = []
    for i in range(99):
        chunk = sc[7 + i * 6: 13 + i * 6]
        if not any(chunk[:4]):
            break
        scenes.append(SceneSpec(patterns={g: chunk[j] for j, g in enumerate("ABCD") if chunk[j]}, time_signature=[chunk[4] or 4, chunk[5] or 4]))
    length = sc[7 + 99 * 6 + 11]
    song = [int(b) for b in sc[7 + 99 * 6 + 12: 7 + 99 * 6 + 12 + length] if b]
    return scenes, song


def find_settings_template(backup_root: str) -> bytes | None:
    """A genuine device-written settings member from the newest backup that has one (never fabricate one)."""
    candidates = sorted(glob.glob(os.path.join(backup_root, "*", "projects", "P*.tar")) + glob.glob(os.path.join(backup_root, "checkpoints", "*.tar")),
                        key=os.path.getmtime, reverse=True)
    for path in candidates:
        try:
            files = member_payloads(open(path, "rb").read())
        except Exception:
            continue
        st = files.get("settings")
        if st is not None and len(st) in (222, 224):
            return st
    return None


# ----------------------------------------------------------------------------- compiling

def to_document(spec: ProjectSpec, base: bytes, frames_by_slot: dict[int, int], settings_template: bytes | None = None,
                write_scene_for_patterns: bool = True, pitch_by_slot: dict[int, float] | None = None) -> dict[str, Any]:
    """Lower a ProjectSpec onto the slot's current archive as an epsysex ep.project.v1 document.

    `pitch_by_slot` gives each sample's native sounding pitch as a MIDI number (from the library) for `concert` events."""
    files = member_payloads(base)
    roots: dict[tuple[str, str], int] = {}
    slots: dict[tuple[str, str], int] = {}
    for name, p in files.items():
        if name.startswith("pads/") and len(p) >= 26:
            group, number = name.split("/")[1].upper(), int(name[-2:])
            roots[(group, label_for_member(number))] = p[24]
            slots[(group, label_for_member(number))] = struct.unpack_from("<H", p, 1)[0]
    doc: dict[str, Any] = {"schema": "ep.project.v1", "product": "ep133"}

    pads = []
    for pad in spec.pads:
        roots[(pad.group, pad.pad)] = pad.root
        slots[(pad.group, pad.pad)] = pad.slot
        trim_end = pad.trim_end
        if pad.slot and pad.slot in frames_by_slot:
            # trim_end is a LENGTH from trim_start and start + length may not exceed the sample: the device silently resets
            # such a pad to the whole sample and the readback then differs (measured 2026-10-17)
            frames = frames_by_slot[pad.slot]
            if pad.trim_start >= frames:
                raise ValueError(f"pad {pad.group} {pad.pad}: trim_start {pad.trim_start} is at or beyond the end of slot {pad.slot} ({frames} frames)")
            if trim_end is None:
                trim_end = frames - pad.trim_start          # the rest of the sample, not the whole sample
            elif pad.trim_start + trim_end > frames:
                raise ValueError(f"pad {pad.group} {pad.pad}: trim_start {pad.trim_start} + trim_end {trim_end} exceeds slot {pad.slot}'s {frames} frames (trim_end is a length)")
        elif pad.slot and trim_end is None:
            raise ValueError(f"slot {pad.slot} has no known length; pass trim_end or let the device layer look the slot up")
        pads.append({"group": pad.group, "pad": pad_member_number(pad.pad), "slot": pad.slot, "trimStart": pad.trim_start,
                     "trimEnd": trim_end or 0, "sampleBpm": pad.sample_bpm, "amplitude": pad.amplitude, "pitch": pad.pitch,
                     "pan": pad.pan, "attack": pad.attack, "release": pad.release, "timeMode": TIME_MODES.index(pad.time_mode),
                     "chokeGroup": 1 if pad.choke else 0, "playMode": PLAY_MODES.index(pad.play_mode), "rootNote": pad.root,
                     "midiChannel": pad.midi_channel})
    if pads:
        doc["pads"] = pads

    patterns = []
    for pat in spec.patterns:
        events = []
        for ev in pat.events:
            tick = ev.tick_value()
            if tick >= pat.bars * BAR_TICKS:
                raise ValueError(f"event at tick {tick} is beyond the {pat.bars}-bar pattern {pat.id}")
            root = roots.get((pat.group, ev.pad), pad_note(pat.group, ev.pad))
            if ev.note is not None:
                note = ev.note
            elif ev.semitones is not None:
                note = root + ev.semitones
            elif ev.concert is not None:
                slot = slots.get((pat.group, ev.pad))
                native = (pitch_by_slot or {}).get(slot) if slot else None
                if native is None:
                    raise ValueError(f"event on {pat.group} {ev.pad!r} asks for concert pitch {ev.concert} but the library has no pitch for slot {slot}; "
                                     "use semitones or note, or index the slot (sync_library)")
                note = root + int(round(note_number(ev.concert) - native))
            else:
                note = root
            if not 0 <= note <= 127:
                raise ValueError(f"event on {pat.group} {ev.pad!r} resolves to note {note}, outside 0..127")
            events.append({"tick": tick, "pad": pad_member_number(ev.pad), "note": note, "velocity": ev.velocity, "duration": ev.duration})
        automation = []
        for lane in pat.lanes:
            for pt in lane.points:
                automation.append({"tick": pt.tick_value(), "parameterID": FADER_PARAMETERS.index(lane.parameter),
                                   "value": int(round(pt.value * 32767))})
        entry: dict[str, Any] = {"id": pat.id, "bars": pat.bars, "events": events}
        if automation:
            entry["automation"] = automation
        patterns.append(entry)

    scene_list = list(spec.scenes)
    base_song: list[int] = []
    if not scene_list:
        scene = spec.scene
        if scene is None and spec.patterns and write_scene_for_patterns:
            scene = SceneSpec(patterns={p.group: p.number for p in spec.patterns})
        if scene is not None:
            scene_list = [scene]
        elif spec.current_scene is not None:           # cursor-only: keep every scene and the song, move the cursor
            scene_list, base_song = scenes_from_base(base)
            if not scene_list:
                raise ValueError("current_scene alone needs a project that already has scenes")
    if scene_list:
        authored = {p["id"] for p in patterns}
        chunks = []
        fallback = scene_one(base)
        for index, scene in enumerate(scene_list, start=1):
            current = scene_refs(base, index) or fallback
            refs = {g: scene.patterns.get(g) or current.get(g) or 1 for g in "ABCD"}
            for g in "ABCD":
                member = f"patterns/{g.lower()}{refs[g]:02d}"
                if member not in files and f"{g}{refs[g]:02d}" not in authored:
                    patterns.append({"id": f"{g}{refs[g]:02d}", "bars": 1, "events": []})   # a silent group: an empty pattern member
                    authored.add(f"{g}{refs[g]:02d}")
            chunks.append({"groupPatterns": [refs[g] for g in "ABCD"], "timeSignature": list(scene.time_signature)})
        song = [int(s) for s in spec.song] or base_song or [1]
        if any(s < 1 or s > len(chunks) for s in song):
            raise ValueError(f"song {song} references a scene outside 1..{len(chunks)}")
        cursor = spec.current_scene if spec.current_scene is not None else 1
        if cursor > len(chunks):
            raise ValueError(f"current_scene {cursor} is outside the {len(chunks)} scenes written")
        doc["scenes"] = chunks
        doc["song"] = song
        doc["currentScene"] = cursor
    if patterns:
        doc["patterns"] = patterns

    if spec.bpm is not None or spec.fader_bases:
        settings: dict[str, Any] = {}
        if spec.bpm is not None:
            settings["bpm"] = spec.bpm
        if spec.fader_bases:
            settings["groupFaders"] = [{"group": fb.group, "parameter": FADER_PARAMETERS.index(fb.parameter), "baseValue": fb.value} for fb in spec.fader_bases]
        if "settings" not in files:
            if settings_template is None:
                raise ValueError("this slot has no settings member and no device-written template was found in the backups; "
                                 "take a backup of a project that has one first (never fabricate settings: ERR CLOCK 43)")
            settings["templateBase64"] = base64.b64encode(settings_template).decode("ascii")
        doc["settings"] = settings

    if spec.fx is not None:
        fx: dict[str, Any] = {"type": spec.fx.type}
        params = {k: v for k, v in (("x", spec.fx.x), ("y", spec.fx.y)) if v is not None}
        if params:
            if spec.fx.type == "off":
                raise ValueError("fx x/y need a non-off fx type")
            fx["parameters"] = params
        comp = {k: v for k, v in (("drive", spec.fx.output_compressor_drive), ("speed", spec.fx.output_compressor_speed)) if v is not None}
        if comp:
            fx["outputCompressor"] = comp
        if spec.fx.sidechain_sources or spec.fx.sidechain_destinations:
            routing: dict[str, dict[str, Any]] = {g: {"sourcePads": [], "destination": False} for g in "ABCD"}   # unnamed groups are cleared
            for src in spec.fx.sidechain_sources:
                routing.setdefault(src.group, {}).setdefault("sourcePads", []).append(pad_member_number(src.pad))
            for g in spec.fx.sidechain_destinations:
                routing.setdefault(g, {})["destination"] = True
            side: dict[str, Any] = {"routing": routing}
            if spec.fx.sidechain_length is not None:
                side["length"] = spec.fx.sidechain_length
            if spec.fx.sidechain_shape is not None:
                side["shape"] = spec.fx.sidechain_shape
            fx["sidechain"] = side
        doc["fxSettings"] = fx
    return doc


def compile_spec(spec: ProjectSpec, base: bytes, frames_by_slot: dict[int, int], settings_template: bytes | None = None,
                 pitch_by_slot: dict[int, float] | None = None) -> tuple[bytes, dict[str, Any]]:
    doc = to_document(spec, base, frames_by_slot, settings_template, pitch_by_slot=pitch_by_slot)
    return compile_project(doc, base_archive=base), doc


def concert_slots(spec: ProjectSpec) -> bool:
    return any(ev.concert is not None for pat in spec.patterns for ev in pat.events)


def referenced_slots(spec: ProjectSpec) -> set[int]:
    return {p.slot for p in spec.pads if p.slot}


def expected_events(spec_or_desc: dict[str, Any], bpm: float, bars_window: int) -> list[dict[str, Any]]:
    """Note events of scene 1's patterns, repeated through `bars_window` bars, with their nominal times at `bpm`
    (from a describe() view)."""
    desc = spec_or_desc
    refs = next((s["patterns"] for s in desc.get("scenes", []) if s["scene"] == desc.get("scene_cursor", 1)), None)
    if refs is None:
        return []
    out = []
    tick_s = 60.0 / (bpm * PPQN)
    for g, n in refs.items():
        pat = desc.get("patterns", {}).get(f"{g}{n:02d}")
        if not pat or not pat["events"]:
            continue
        length = pat["bars"] * BAR_TICKS
        reps = (bars_window * BAR_TICKS + length - 1) // length
        for rep in range(reps):
            for ev in pat["events"]:
                tick = ev["tick"] + rep * length
                if tick < bars_window * BAR_TICKS:
                    out.append({"group": g, "pad": ev["pad"], "note": ev["note"], "velocity": ev["velocity"], "tick": tick, "time_s": tick * tick_s})
    return sorted(out, key=lambda e: (e["tick"], e["note"]))
