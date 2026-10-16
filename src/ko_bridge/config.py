"""Where ko-bridge keeps its files, and the few facts about the device the protocol cannot read (set by the human).

KO_BRIDGE_HOME (default ~/Music/ko-bridge) holds backups/, backups/checkpoints/, takes/, library.json and config.json.
Secrets, when there are any, live in secrets.env there and never in the repo.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field

HOME = os.path.expanduser(os.environ.get("KO_BRIDGE_HOME", "~/Music/ko-bridge"))


def folder(*parts: str) -> str:
    p = os.path.join(HOME, *parts)
    os.makedirs(p, exist_ok=True)
    return p


def file_path(*parts: str) -> str:
    p = os.path.join(HOME, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def new_id(prefix: str) -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}"


def write_json(path: str, data) -> str:
    with open(path, "w") as f:
        json.dump(data, f, indent=1, default=str)
    return path


def read_json(path: str, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


MIDI_CLOCK_SETTINGS = ("unknown", "off", "in", "out")   # system setting 100 / 101 / 102


@dataclass
class Config:
    midi_clock: str = "unknown"          # what the human set with SHIFT+ERASE 100/101/102; the bridge cannot read or set it
    default_bpm: float = 120.0
    scratch_project: int | None = None   # a slot the human has declared expendable; tools never pick it on their own
    midi_hint: str = "EP-133"
    audio_hint: str = "EP-133"
    latency_ms: float = 3.5              # note-on to USB audio onset, measured 2026-10-16
    notes: dict = field(default_factory=dict)

    @classmethod
    def load(cls) -> "Config":
        data = read_json(os.path.join(HOME, "config.json"), {}) or {}
        cfg = cls()
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    def save(self) -> None:
        write_json(file_path("config.json"), asdict(self))
