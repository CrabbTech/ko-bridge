"""Where ko-bridge keeps its files, and the few facts about the device the protocol cannot read (set by the human).

KO_BRIDGE_HOME (default ~/Music/ko-bridge) holds backups/, backups/checkpoints/, takes/, library.json and config.json.
Secrets live in secrets.env there (KEY=VALUE lines) and never in the repo: load_secrets reads them into the environment
at server start, falling back to op-bridge's ~/Music/op-bridge/secrets.env, and never logs a value.
"""
from __future__ import annotations

import json
import os
import re
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


SECRETS_FILES = (os.path.join(HOME, "secrets.env"), os.path.expanduser("~/Music/op-bridge/secrets.env"))
# A variable name: an identifier of at most 64 characters. Tyler's op-bridge file carries one malformed line whose "name" is
# a 106-character key blob, which this skips.
_SECRET_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_SECRETS_REPORT: dict | None = None


def load_secrets(paths: tuple[str, ...] = SECRETS_FILES, environ=os.environ, force: bool = False) -> dict:
    """Read KEY=VALUE lines from the first secrets file that exists into `environ`, never overriding a variable that is
    already set. Blank lines, comments, a leading "export " and quotes around a value are handled; a line without "=" or
    with a name that is not an identifier is skipped. Returns names and counts only (never a value), and runs once per
    process unless `force` is set."""
    global _SECRETS_REPORT
    if _SECRETS_REPORT is not None and not force:
        return _SECRETS_REPORT
    report: dict = {"file": None, "loaded": [], "already_set": [], "skipped_lines": 0}
    for candidate in paths:
        path = os.path.expanduser(candidate)
        if not os.path.isfile(path):
            continue
        report["file"] = path
        with open(path) as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].lstrip()
                name, sep, value = line.partition("=")
                name = name.strip()
                if not sep or not _SECRET_NAME.match(name):
                    report["skipped_lines"] += 1
                    continue
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                if name in environ:
                    report["already_set"].append(name)
                    continue
                environ[name] = value
                report["loaded"].append(name)
        break
    _SECRETS_REPORT = report
    return report


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
