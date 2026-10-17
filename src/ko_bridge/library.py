"""The companion sample library: every saved sound indexed with measured features and labels, searchable.

Records come from the device (every slot of the newest backup, with TE's display names and metadata), from WAVs imported
on the Mac (not necessarily loaded on the device) and from uploads. Each record carries measured features (length,
attack, decay, brightness, pitch and its confidence, low/high energy shares), labels (a fixed category vocabulary plus
descriptors; from the name when it says something, otherwise guessed from the features and marked so) and free user
tags. The index lives at ~/Music/ko-bridge/library/index.json; imported audio is copied to ~/Music/ko-bridge/library/wav/.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import shutil
import time
from typing import Any

import numpy as np

from . import config as C

CATEGORIES = ("kick", "snare", "rim", "clap", "hat_closed", "hat_open", "ride", "crash", "cymbal", "tom", "shaker", "tambourine",
              "conga", "bongo", "cowbell", "clave", "perc", "bass", "sub", "reese", "chord", "pad", "string", "piano", "keys", "organ",
              "pluck", "stab", "lead", "brass", "wind", "guitar", "bell", "vocal", "fx", "texture", "loop", "unknown")
DESCRIPTORS = ("short", "medium", "long", "sustained", "decaying", "bright", "dark", "punchy", "soft", "noisy", "tonal", "pitched",
               "stereo", "low", "high")
BANKS = (("kick", 1, 99), ("snare", 100, 199), ("hat", 200, 299), ("perc", 300, 399), ("bass", 400, 499), ("melodic", 500, 599), ("user", 600, 999))
BANK_CATEGORY = {"kick": "kick", "snare": "snare", "hat": "hat_closed", "perc": "perc", "bass": "bass", "melodic": "keys"}

# name fragments -> categories (TE's factory names are descriptive; checked longest fragment first)
NAME_RULES = [
    ("closed hat", "hat_closed"), ("hat closed", "hat_closed"), ("hh closed", "hat_closed"), ("open hat", "hat_open"), ("hat open", "hat_open"),
    ("hh open", "hat_open"), ("hihat", "hat_closed"), ("dbl hat", "hat_closed"), ("golden hat", "hat_closed"), ("shine hat", "hat_closed"),
    ("crunch hat", "hat_closed"), ("brush hat", "hat_open"), ("hat", "hat_closed"),
    ("rimshot", "rim"), ("rim", "rim"), ("snare", "snare"), ("snr", "snare"), ("quik roll", "snare"), ("kick", "kick"), ("kik", "kick"),
    ("clap", "clap"), ("ride", "ride"), ("crash", "crash"), ("china", "cymbal"), ("splash", "cymbal"), ("cymbal", "cymbal"), ("cym ", "cymbal"),
    ("tom", "tom"), ("shaker", "shaker"), ("shake", "shaker"), ("cabasa", "shaker"), ("cabase", "shaker"), ("maraca", "shaker"),
    ("tambo", "tambourine"), ("tamb", "tambourine"), ("conga", "conga"), ("bongo", "bongo"), ("cowbell", "cowbell"), ("agogo", "cowbell"),
    ("clave", "clave"), ("guiro", "perc"), ("hand drum", "perc"), ("hundobell", "perc"), ("perc", "perc"),
    ("reese", "reese"), ("sub", "sub"), ("bass", "bass"), ("chordy", "chord"), ("chord", "chord"), ("soothe", "pad"), ("string", "string"),
    ("cello", "string"), ("epiano", "keys"), ("piano", "piano"), ("wurli", "keys"), ("clav", "keys"), ("vibes", "keys"), ("organ", "organ"),
    ("pluck", "pluck"), ("stab", "stab"), ("lead", "lead"), ("trumpet", "brass"), ("trombone", "brass"), ("horn", "brass"), ("flute", "wind"),
    ("guitar", "guitar"), ("strato", "guitar"), ("sitar", "guitar"), ("guzheng", "guitar"), ("vox", "vocal"), ("vocal", "vocal"),
    ("bell", "bell"), ("birds", "fx"), ("tape", "fx"), ("fx", "fx"), ("noise", "fx"), ("sky", "pad"), ("search chord", "chord"),
    ("square", "bass"), ("buzz", "bass"), ("growl", "bass"), ("wobb", "bass"), ("twang", "bass"), ("thmp", "bass"), ("thub", "bass"),
    ("mud", "bass"), ("boing", "fx"), ("puff", "fx"), ("tock", "perc"), ("revy", "fx"), ("orga", "organ"),
]


def _dir() -> str:
    return C.folder("library")


def index_path() -> str:
    return os.path.join(_dir(), "index.json")


# ----------------------------------------------------------------------------- features

def _env_db(x: np.ndarray, sr: int, win_ms: float = 1.0) -> np.ndarray:
    win = max(1, int(sr * win_ms / 1000))
    n = len(x) // win
    if n == 0:
        return np.full(1, -120.0)
    frames = x[: n * win].reshape(n, win).astype(np.float64)
    return 20 * np.log10(np.sqrt(np.mean(frames * frames, axis=1)) + 1e-12)


_PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_FLAT_TO_SHARP = {"DB": "C#", "EB": "D#", "GB": "F#", "AB": "G#", "BB": "A#"}
_TONAL_WORDS = {"bass", "reese", "growl", "wub", "chord", "chords", "stab", "key", "keys", "note", "pad", "lead", "pluck",
                "piano", "epiano", "rhodes", "organ", "synth", "arp", "bell", "string", "strings", "vox", "vocal", "choir", "root", "tone",
                "in", "808"}


def parse_note_from_name(name: str | None) -> dict[str, Any] | None:
    """Extract an explicit musical pitch class (and optional octave/MIDI note) from a sample name or filename.

    Examples:
    - `"Cymatics - Dubstep Bass E"` -> `{"pitch_class": "E", "octave": None, "midi_note": None}`
    - `"808_F#1"` -> `{"pitch_class": "F#", "octave": 1, "midi_note": 30}`
    - `"liquid chord em7"` -> `{"pitch_class": "E", "octave": None, "midi_note": None}`
    """
    if not name:
        return None
    stem = re.sub(r"\.(wav|pcm|aif|aiff|flac)$", "", str(name), flags=re.IGNORECASE).strip()
    tokens = [t for t in re.split(r"[\s_\-]+", stem) if t]
    for idx in range(len(tokens) - 1, -1, -1):
        tok = tokens[idx]
        # Avoid matching a standalone English article "a" in the middle of a phrase unless it's the final token
        if tok.lower() == "a" and idx < len(tokens) - 1:
            continue
        m = re.fullmatch(r"([A-Ga-g])([#b]?)(?:m|min|maj|m7|maj7|7)?([0-6])?", tok)
        if not m:
            continue
        letter = m.group(1).upper()
        acc = m.group(2)
        oct_str = m.group(3)
        # A bare letter (no octave, no accidental) is a variant marker in TE's naming ("nt kick b", "snare mini bm":
        # 46 factory names) unless a tonal word precedes it ("dubstep bass e", "chord em7").
        if oct_str is None and not acc:
            prev = tokens[idx - 1].lower() if idx > 0 else ""
            if prev not in _TONAL_WORDS:
                continue
        pc = letter + ("#" if acc == "#" else "")
        if acc == "b":
            pc = _FLAT_TO_SHARP.get(letter + "B", letter)
        if pc not in _PITCH_CLASSES:
            continue
        octave = int(oct_str) if oct_str is not None else None
        midi_note = (_PITCH_CLASSES.index(pc) + 12 * (octave + 1)) if octave is not None else None
        return {"pitch_class": pc, "octave": octave, "midi_note": midi_note}
    return None


def _f0(seg: np.ndarray, sr: int, fmin: float = 30.0, fmax: float = 2500.0) -> tuple[float | None, float]:
    seg = seg.astype(np.float64)
    seg = seg - seg.mean()
    if len(seg) < sr * 0.03 or not np.any(seg):
        return None, 0.0
    n = len(seg)
    f = np.fft.rfft(seg * np.hanning(n), 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    lo, hi = int(sr / fmax), min(int(sr / fmin), n - 2)
    if hi <= lo:
        return None, 0.0
    # skip the slow fall from lag 0 (a low sine has barely changed after a few samples): search only past the first
    # dip below zero, so the first real peak is the period
    below = np.nonzero(ac[1:hi] < 0)[0]
    start = max(lo, int(below[0]) + 1) if len(below) else lo
    if hi <= start + 1:
        return None, 0.0
    lag = start + int(np.argmax(ac[start:hi]))
    conf = float(ac[lag] / (ac[0] + 1e-12))
    # no octave correction: for a periodic tone the autocorrelation at twice the period is as high as at the period,
    # so the highest peak (the shortest strong period) is the fundamental; TE's root-note metadata is kept beside it
    a, b, c = ac[lag - 1], ac[lag], ac[lag + 1]
    denom = a - 2 * b + c
    if denom != 0:
        lag = lag + 0.5 * (a - c) / denom
    if not np.isfinite(lag) or lag <= 0 or not np.isfinite(conf):
        return None, 0.0
    f0 = float(sr / lag)
    if not (fmin * 0.5 <= f0 <= fmax * 2):
        return None, 0.0
    return f0, max(0.0, min(1.0, conf))


def features(x: np.ndarray, sr: int, channels: int = 1, name: str | None = None) -> dict[str, Any]:
    """Measured features of one sound (x = mono float32 in [-1, 1])."""
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    seconds = n / sr
    if n == 0:
        return {"seconds": 0.0}
    peak = float(np.max(np.abs(x))) if n else 0.0
    env = _env_db(x, sr, 1.0)                           # 1 ms RMS envelope in dB
    peak_i = int(np.argmax(env))
    peak_db = float(env[peak_i])
    lin = 10 ** (env / 20)
    # attack: first 10% to first 90% of the peak amplitude
    i10 = int(np.argmax(lin >= 0.1 * lin[peak_i])) if lin[peak_i] > 0 else 0
    i90 = int(np.argmax(lin >= 0.9 * lin[peak_i])) if lin[peak_i] > 0 else 0
    attack_ms = max(0.0, (i90 - i10) * 1.0)
    # decay: ms from the peak until the envelope stays 20 dB under it
    below = np.nonzero(env[peak_i:] < peak_db - 20)[0]
    decay_ms = float(below[0]) if len(below) else None
    # sustain: level in the middle third relative to the first 100 ms after the peak
    third = max(1, len(env) // 3)
    mid = float(np.mean(env[third: 2 * third])) if len(env) >= 3 else peak_db
    head = float(np.mean(env[peak_i: peak_i + 100])) if len(env) > peak_i else peak_db
    sustain_db = (mid - head) if seconds > 0.3 else -60.0
    # spectra: first 100 ms after onset and whole
    def centroid_and_shares(seg):
        if len(seg) < 64:
            return None, None, None
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
        freqs = np.fft.rfftfreq(len(seg), 1 / sr)
        total = float(spec.sum()) + 1e-18
        return float((freqs * spec).sum() / total), float(spec[freqs < 150].sum() / total), float(spec[freqs > 4000].sum() / total)
    onset = int(i10 * sr / 1000)
    cen_head, low_head, high_head = centroid_and_shares(x[onset: onset + int(0.1 * sr)])
    cen_all, low_all, high_all = centroid_and_shares(x[onset:])
    # pitch on the 200 ms after the attack (skipping the first 20 ms transient)
    seg = x[onset + int(0.02 * sr): onset + int(0.22 * sr)]
    f0, conf = _f0(seg, sr)
    # Measure pitch drift between early (20-120 ms) and late (120-220 ms) windows to detect sweeping FX / 808 drops
    seg_early = x[onset + int(0.02 * sr): onset + int(0.12 * sr)]
    seg_late = x[onset + int(0.12 * sr): onset + int(0.22 * sr)]
    f0_early, conf_early = _f0(seg_early, sr)
    f0_late, conf_late = _f0(seg_late, sr)
    pitch_drift_cents = None
    if f0_early and f0_late and conf_early > 0.45 and conf_late > 0.45:
        pitch_drift_cents = round(float(1200.0 * np.log2(f0_late / f0_early)), 1)
    pitch_stable = True if pitch_drift_cents is None else (abs(pitch_drift_cents) <= 150.0)

    midi = float(69 + 12 * np.log2(f0 / 440)) if (f0 and f0 > 0) else None
    hint = parse_note_from_name(name)
    # If the sample name explicitly states a note (e.g. "Dubstep Bass E") and autocorrelation locked onto the 3rd
    # harmonic (+7 semitones / fifth above), disambiguate back to the fundamental pitch class.
    if hint and midi is not None and conf > 0.45:
        hint_pc_idx = _PITCH_CLASSES.index(hint["pitch_class"])
        meas_pc_idx = int(round(midi)) % 12
        diff = (meas_pc_idx - hint_pc_idx) % 12
        if diff == 7:  # 3rd harmonic: 19.02 semitones above the fundamental (an octave and a fifth), so drop 19, not 7
            midi = round(midi) - 19.0
            f0 = float(440.0 * (2.0 ** ((midi - 69.0) / 12.0)))

    # spectral flatness (0 tonal .. 1 noisy), bandwidth and the share of energy in body (150-800) and presence (800-4000) bands
    body = x[onset: onset + int(0.4 * sr)] if n - onset > 64 else x
    spec = np.abs(np.fft.rfft(body * np.hanning(len(body)))) ** 2 + 1e-18
    freqs = np.fft.rfftfreq(len(body), 1 / sr)
    band = (freqs > 60) & (freqs < 16000)
    flatness = float(np.exp(np.mean(np.log(spec[band]))) / np.mean(spec[band]))
    total = float(spec.sum())
    cen_b = float((freqs * spec).sum() / total)
    bandwidth = float(np.sqrt(((freqs - cen_b) ** 2 * spec).sum() / total))
    mid_share = float(spec[(freqs >= 150) & (freqs < 800)].sum() / total)
    presence_share = float(spec[(freqs >= 800) & (freqs < 4000)].sum() / total)
    head = x[onset: onset + int(0.02 * sr)]
    crest = float(np.max(np.abs(head)) / (np.sqrt(np.mean(head.astype(np.float64) ** 2)) + 1e-9)) if len(head) else 0.0
    # decay shape: how much level is left a third and two thirds of the way through
    third = max(1, len(env) // 3)
    late_db = float(np.mean(env[2 * third:])) - peak_db if len(env) > 3 else -60.0
    return {
        "flatness": round(flatness, 3), "bandwidth_hz": round(bandwidth), "mid_share": round(mid_share, 3), "presence_share": round(presence_share, 3),
        "crest": round(crest, 1), "late_level_db": round(late_db, 1),
        "seconds": round(seconds, 3), "peak_dbfs": round(20 * np.log10(peak + 1e-12), 1), "rms_dbfs": round(float(20 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12)), 1),
        "attack_ms": round(attack_ms, 1), "decay_ms": None if decay_ms is None else round(decay_ms, 0), "sustain_db": round(sustain_db, 1),
        "centroid_hz": None if cen_head is None else round(cen_head), "centroid_hz_overall": None if cen_all is None else round(cen_all),
        "low_share": None if low_all is None else round(low_all, 3), "high_share": None if high_all is None else round(high_all, 3),
        "f0_hz": None if f0 is None else round(f0, 1), "pitch_confidence": round(conf, 2),
        "midi_note": None if midi is None else round(float(midi), 1),
        "pitch_class": None if midi is None else _PITCH_CLASSES[int(round(midi)) % 12],
        "pitch_drift_cents": pitch_drift_cents,
        "pitch_stable": pitch_stable,
        "name_pitch_hint": hint["pitch_class"] if hint else None,
        "stereo": channels > 1,
    }


# ----------------------------------------------------------------------------- labels

def category_from_name(name: str | None) -> str | None:
    if not name:
        return None
    low = f" {name.lower()} "
    for frag, cat in NAME_RULES:
        if frag in low:
            return cat
    return None


def bank_of(slot: int | None) -> str | None:
    if slot is None:
        return None
    for label, lo, hi in BANKS:
        if lo <= slot <= hi:
            return label
    return None


def guess_category(f: dict[str, Any], bank: str | None) -> str:
    """A category from features alone, for sounds whose name says nothing (user slots, imports)."""
    sec, att = f.get("seconds", 0), f.get("attack_ms", 0)
    dec = f.get("decay_ms")
    dec = sec * 1000.0 if dec is None else dec          # a file that ends before falling 20 dB decayed with its length
    cen = f.get("centroid_hz") or 0
    low, high = f.get("low_share") or 0, f.get("high_share") or 0
    conf, f0 = f.get("pitch_confidence", 0), f.get("f0_hz") or 0
    sustained = f.get("sustain_db", -60) > -12
    percussive = att < 15 and dec < 500 and not sustained
    # factory banks are a strong prior when the name says nothing
    if bank == "kick":
        return "kick"
    if bank == "snare":
        return "snare"
    if bank == "hat":
        return "hat_closed" if sec < 0.15 else ("hat_open" if sec < 1.2 else "cymbal")
    if sec > 2.5 and not (conf > 0.75 and f0 < 150):
        return "loop" if (f.get("rms_dbfs", -60) > -30 and not sustained) else "texture"
    if percussive:
        if low > 0.45 and (f0 < 120 or conf < 0.5):
            return "kick"
        if cen > 5000 and sec < 0.15:
            return "hat_closed"
        if cen > 3500 and sec >= 0.15:
            return "hat_open" if sec < 1.2 else "cymbal"
        if conf > 0.7 and 70 < f0 < 450 and low < 0.45:
            return "tom"
        if 1000 < cen <= 3500:
            return "snare"
        return "perc"
    if conf > 0.5:
        if f0 and f0 < 130:
            return "bass"
        return "keys" if sec < 1.5 else "pad"
    if bank in BANK_CATEGORY:
        return BANK_CATEGORY[bank]
    return "unknown"


def descriptors(f: dict[str, Any]) -> list[str]:
    out = []
    sec = f.get("seconds", 0)
    out.append("short" if sec < 0.15 else "long" if sec > 1.0 else "medium")
    if f.get("sustain_db", -40) > -12:
        out.append("sustained")
    elif f.get("decay_ms") is not None:
        out.append("decaying")
    cen = f.get("centroid_hz") or 0
    if cen > 3000:
        out.append("bright")
    elif cen and cen < 800:
        out.append("dark")
    if f.get("attack_ms", 0) < 5:
        out.append("punchy")
    elif f.get("attack_ms", 0) > 40:
        out.append("soft")
    conf = f.get("pitch_confidence", 0)
    if conf > 0.75:
        out.append("tonal")
    if conf > 0.5:
        out.append("pitched")
    elif conf < 0.3 and (f.get("high_share") or 0) > 0.15:
        out.append("noisy")
    if f.get("stereo"):
        out.append("stereo")
    if (f.get("low_share") or 0) > 0.5:
        out.append("low")
    if (f.get("high_share") or 0) > 0.5:
        out.append("high")
    return out


def character(record: dict[str, Any]) -> str:
    """One readable line about what a sound is like, from its measurements."""
    f = record.get("features", {})
    sec = f.get("seconds", 0)
    words = []
    words.append("very short" if sec < 0.08 else "short" if sec < 0.25 else "medium" if sec < 1.0 else "long" if sec < 3 else "very long")
    conf = f.get("pitch_confidence", 0)
    if conf > 0.75 and f.get("pitch_class"):
        words.append(f"clearly pitched at {f['pitch_class']}{'' if f.get('midi_note') is None else int(round(f['midi_note'])) // 12 - 1}")
    elif conf > 0.5 and f.get("pitch_class"):
        words.append(f"loosely pitched around {f['pitch_class']}")
    else:
        words.append("unpitched")
    flat = f.get("flatness") or 0
    words.append("noisy" if flat > 0.3 else "textured" if flat > 0.08 else "tonal")
    cen = f.get("centroid_hz") or 0
    words.append("very bright" if cen > 6000 else "bright" if cen > 2500 else "mid" if cen > 800 else "dark" if cen > 200 else "sub-heavy")
    att = f.get("attack_ms", 0)
    words.append("clicky attack" if (f.get("crest") or 0) > 6 and att < 5 else "hard attack" if att < 10 else "soft attack" if att > 60 else "medium attack")
    late = f.get("late_level_db", -60)
    words.append("sustains" if late > -12 else "rings" if late > -30 else "decays fast")
    if f.get("stereo"):
        words.append("stereo" + (" wide" if (f.get("stereo_width") or 0) > 0.5 else ""))
    return ", ".join(words)


def auto_name(record: dict[str, Any]) -> str:
    """A descriptive stand-in for placeholder names like '845 sample_01'."""
    f = record.get("features", {})
    cat = record.get("category", "sound")
    pitch = f.get("pitch_class") if f.get("pitch_confidence", 0) > 0.5 else None
    ms = int(round((f.get("seconds") or 0) * 1000))
    bits = [cat]
    if pitch:
        bits.append(pitch + ("" if f.get("midi_note") is None else str(int(round(f["midi_note"])) // 12 - 1)))
    cen = f.get("centroid_hz") or 0
    bits.append("bright" if cen > 3000 else "dark" if cen < 400 else "mid")
    bits.append(f"{ms}ms" if ms < 1000 else f"{ms / 1000:.1f}s")
    return " ".join(bits)


def is_placeholder(name: str | None) -> bool:
    return not name or bool(re.fullmatch(r"\s*\d*\s*sample_\d+\s*|\s*\d+\.pcm\s*|\s*\d+[-_]\d+_\d+\s*", name))


def label(record: dict[str, Any]) -> None:
    f = record.get("features", {})
    name_cat = category_from_name(record.get("name"))
    if name_cat:
        record["category"], record["label_source"] = name_cat, "name"
    else:
        record["category"], record["label_source"] = guess_category(f, record.get("bank")), "guessed"
    record["labels"] = [record["category"]] + descriptors(f)
    if record.get("bank") and record["bank"] != "user":
        record["labels"].append(f"bank:{record['bank']}")
    record["character"] = character(record)
    if is_placeholder(record.get("name")):
        record["auto_name"] = auto_name(record)
        record["labels"].append("placeholder-name")


# ----------------------------------------------------------------------------- the index

class Library:
    def __init__(self, path: str | None = None):
        self.path = path or index_path()
        data = C.read_json(self.path, {}) or {}
        self.records: dict[str, dict[str, Any]] = data.get("records", {})
        self.meta: dict[str, Any] = data.get("meta", {})

    def save(self) -> None:
        C.write_json(self.path, {"meta": {**self.meta, "saved": time.strftime("%Y-%m-%d %H:%M:%S"), "count": len(self.records)}, "records": self.records})

    # -- building ---------------------------------------------------------------
    def add_pcm(self, rid: str, pcm: bytes, samplerate: int, channels: int, name: str | None, slot: int | None, source: dict[str, Any],
                metadata: dict[str, Any] | None = None, user_tags: list[str] | None = None) -> dict[str, Any]:
        arr = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        mono = arr.reshape(-1, channels).mean(axis=1) if channels > 1 else arr
        hint_name = name or os.path.basename(str((source or {}).get("original") or (source or {}).get("path") or ""))
        feats = features(mono, samplerate, channels, name=hint_name)
        if channels == 2:
            lr = arr.reshape(-1, 2)
            side = float(np.sqrt(np.mean((lr[:, 0] - lr[:, 1]) ** 2)) + 1e-9)
            midv = float(np.sqrt(np.mean((lr[:, 0] + lr[:, 1]) ** 2)) + 1e-9)
            feats["stereo_width"] = round(side / midv, 3)
        rec: dict[str, Any] = {"id": rid, "name": name, "slot": slot, "bank": bank_of(slot), "samplerate": samplerate, "channels": channels,
                               "frames": len(mono), "source": source, "features": feats,
                               "user_tags": list(user_tags or self.records.get(rid, {}).get("user_tags", []))}
        if metadata:
            rec["device"] = {k: metadata.get(k) for k in ("sound.rootnote", "sound.playmode", "sound.bpm", "time.mode", "sound.amplitude", "sample.start", "sample.end") if k in metadata}
        label(rec)
        self.records[rid] = rec
        return rec

    def rebuild_from_backup(self, backup_dir: str | None = None) -> dict[str, Any]:
        """Index every slot of a backup (default: the newest one with a sounds/ folder)."""
        if backup_dir is None:
            dirs = sorted(d for d in glob.glob(os.path.join(C.HOME, "backups", "*")) if os.path.isdir(os.path.join(d, "sounds")))
            if not dirs:
                raise FileNotFoundError("no backup with a sounds/ folder under ~/Music/ko-bridge/backups; run backup_device first")
            backup_dir = dirs[-1]
        count = 0
        for meta_path in sorted(glob.glob(os.path.join(backup_dir, "sounds", "*.json"))):
            slot = int(os.path.basename(meta_path)[:3])
            d = json.load(open(meta_path))
            md = d["metadata"]
            pcm_path = meta_path[:-5] + ".pcm"
            if not os.path.exists(pcm_path):
                continue
            self.add_pcm(f"slot-{slot:03d}", open(pcm_path, "rb").read(), int(md.get("samplerate", 46875) or 46875), int(md.get("channels", 1) or 1),
                         md.get("name"), slot, {"kind": "device-backup", "path": pcm_path, "backup": os.path.basename(backup_dir)}, md)
            count += 1
        self.meta["backup"] = backup_dir
        refreshed = self.reanalyze_local(exclude_prefix="slot-", only_without_backup=True)
        self.save()
        return {"indexed": count, "backup": backup_dir, "records": len(self.records), "reanalyzed": refreshed}

    def reanalyze_local(self, exclude_prefix: str | None = None, only_without_backup: bool = False) -> int:
        """Recompute features and labels for records whose audio is on this Mac (imports, cached device reads, uploads)."""
        import soundfile as sf

        count = 0
        for rid, rec in list(self.records.items()):
            src = rec.get("source") or {}
            if only_without_backup and src.get("kind") == "device-backup":
                continue
            path = src.get("path")
            if not path or not os.path.exists(path):
                continue
            if path.endswith(".pcm"):
                pcm = open(path, "rb").read()
                sr, ch = int(rec.get("samplerate", 46875)), int(rec.get("channels", 1))
            else:
                x, sr = sf.read(path, dtype="float32", always_2d=True)
                ch = x.shape[1]
                pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
            self.add_pcm(rid, pcm, sr, ch, rec.get("name"), rec.get("slot"), src, (rec.get("device") or None) and {k.replace("sound.", "sound."): v for k, v in rec["device"].items()},
                         rec.get("user_tags"))
            count += 1
        return count

    def import_wav(self, path: str, tags: list[str] | None = None, origin: str | None = None, license_note: str | None = None,
                   name: str | None = None) -> dict[str, Any]:
        """Copy a WAV into the library and index it (not loaded on the device until load_sound)."""
        import soundfile as sf

        path = os.path.expanduser(path)
        data = open(path, "rb").read()
        rid = "wav-" + hashlib.sha1(data).hexdigest()[:10]
        dest_dir = C.folder("library", "wav")
        dest = os.path.join(dest_dir, f"{rid}-{re.sub(r'[^A-Za-z0-9_.-]', '_', os.path.basename(path))}")
        if not os.path.exists(dest):
            shutil.copy(path, dest)
        x, sr = sf.read(dest, dtype="float32", always_2d=True)
        channels = x.shape[1]
        pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
        rec = self.add_pcm(rid, pcm, int(sr), channels, name or os.path.splitext(os.path.basename(path))[0][:20], None,
                           {"kind": "import", "path": dest, "original": path, "origin": origin, "license": license_note}, None, tags)
        self.save()
        return rec

    def note_upload(self, slot: int, pcm: bytes, samplerate: int, channels: int, name: str, source_path: str | None) -> dict[str, Any]:
        rec = self.add_pcm(f"slot-{slot:03d}", pcm, samplerate, channels, name, slot,
                           {"kind": "upload", "path": self._cache_pcm(slot, pcm), "wav": source_path, "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        self.save()
        return rec

    def _cache_pcm(self, slot: int, pcm: bytes) -> str:
        path = os.path.join(C.folder("library", "pcm"), f"slot-{slot:03d}.pcm")
        with open(path, "wb") as f:
            f.write(pcm)
        return path

    def sync_with_device(self, ep, stage=None) -> dict[str, Any]:
        """Index every device slot missing from the library (read from the device) and forget slots that are gone."""
        nodes = ep.list_sounds()
        present = {int(n["id"]) for n in nodes}
        indexed = {r["slot"] for r in self.records.values() if r.get("slot") is not None}
        added = []
        for slot in sorted(present - indexed):
            if stage:
                stage(f"indexing slot {slot}")
            pcm, _open = ep.client.read_sound(slot)
            md = ep.client.get_sample_metadata(slot)
            self.add_pcm(f"slot-{slot:03d}", pcm, int(md.get("samplerate", 46875) or 46875), int(md.get("channels", 1) or 1), md.get("name"), slot,
                         {"kind": "device", "path": self._cache_pcm(slot, pcm), "time": time.strftime("%Y-%m-%d %H:%M:%S")}, md)
            added.append(slot)
        gone = sorted(indexed - present)
        for slot in gone:
            self.records.pop(f"slot-{slot:03d}", None)
        self.save()
        return {"added": added, "forgotten": gone, "on_device": len(present), "records": len(self.records)}

    def forget_slot(self, slot: int) -> None:
        self.records.pop(f"slot-{slot:03d}", None)
        self.save()

    # -- querying ---------------------------------------------------------------
    def get(self, key: str | int) -> dict[str, Any] | None:
        if isinstance(key, int) or (isinstance(key, str) and key.isdigit()):
            return self.records.get(f"slot-{int(key):03d}")
        return self.records.get(key)

    def tag(self, key: str | int, add: list[str] | None = None, remove: list[str] | None = None) -> dict[str, Any]:
        rec = self.get(key)
        if rec is None:
            raise KeyError(f"no library record {key!r}")
        tags = set(rec.get("user_tags", []))
        tags |= {t.strip().lower() for t in (add or []) if t.strip()}
        tags -= {t.strip().lower() for t in (remove or [])}
        rec["user_tags"] = sorted(tags)
        self.save()
        return rec

    def labels(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        examples: dict[str, list[str]] = {}
        for rec in self.records.values():
            for lab in rec.get("labels", []) + rec.get("user_tags", []):
                counts[lab] = counts.get(lab, 0) + 1
                if len(examples.setdefault(lab, [])) < 3 and rec.get("name"):
                    examples[lab].append(rec["name"])
        return {"labels": {k: {"count": counts[k], "examples": examples[k]} for k in sorted(counts, key=lambda k: -counts[k])},
                "categories": list(CATEGORIES), "descriptors": list(DESCRIPTORS)}

    def search(self, query: str | None = None, labels: list[str] | None = None, exclude: list[str] | None = None,
               min_seconds: float | None = None, max_seconds: float | None = None, pitched: bool | None = None,
               bank: str | None = None, on_device: bool | None = None, pitch_class: str | None = None, limit: int = 25) -> list[dict[str, Any]]:
        want = {l.lower() for l in (labels or [])}
        avoid = {l.lower() for l in (exclude or [])}
        words = [w for w in re.split(r"[\s,]+", (query or "").lower()) if w]
        rows = []
        for rec in self.records.values():
            f = rec.get("features", {})
            labs = set(rec.get("labels", [])) | set(rec.get("user_tags", []))
            if want and not want <= labs:
                continue
            if avoid & labs:
                continue
            if bank and rec.get("bank") != bank:
                continue
            if on_device is not None and (rec.get("slot") is not None) != on_device:
                continue
            if min_seconds is not None and f.get("seconds", 0) < min_seconds:
                continue
            if max_seconds is not None and f.get("seconds", 0) > max_seconds:
                continue
            if pitched is not None and (f.get("pitch_confidence", 0) > 0.5) != pitched:
                continue
            if pitch_class and f.get("pitch_class") != pitch_class:
                continue
            score = 0.0
            text = f"{rec.get('name') or ''} {rec.get('auto_name') or ''} {rec.get('character') or ''} {' '.join(labs)}".lower()
            for w in words:
                if w in text:
                    score += 2.0 if w in (rec.get("name") or "").lower() else 1.0
            if words and score == 0:
                continue
            rows.append((score, rec))
        rows.sort(key=lambda r: (-r[0], r[1].get("slot") or 9999, r[1].get("name") or ""))
        return [self.row(rec) for _s, rec in rows[:limit]]

    @staticmethod
    def row(rec: dict[str, Any]) -> dict[str, Any]:
        f = rec.get("features", {})
        return {"id": rec["id"], "slot": rec.get("slot"), "name": rec.get("name"), "auto_name": rec.get("auto_name"), "character": rec.get("character"),
                "category": rec.get("category"), "label_source": rec.get("label_source"),
                "labels": [l for l in rec.get("labels", []) if l != rec.get("category")], "tags": rec.get("user_tags", []),
                "seconds": f.get("seconds"), "stereo": f.get("stereo"), "samplerate": rec.get("samplerate"),
                "pitch": (f"{f.get('pitch_class')}{'' if f.get('midi_note') is None else int(round(f['midi_note'])) // 12 - 1} ({f.get('f0_hz')} Hz)" if f.get("pitch_confidence", 0) > 0.5 else None),
                "root_note_device": (rec.get("device") or {}).get("sound.rootnote"), "attack_ms": f.get("attack_ms"), "decay_ms": f.get("decay_ms"),
                "centroid_hz": f.get("centroid_hz")}

    def status(self) -> dict[str, Any]:
        on_dev = sum(1 for r in self.records.values() if r.get("slot") is not None)
        return {"records": len(self.records), "on_device": on_dev, "imports_not_loaded": len(self.records) - on_dev, "index": self.path,
                "backup": self.meta.get("backup"), "saved": self.meta.get("saved"),
                "guessed_labels": sum(1 for r in self.records.values() if r.get("label_source") == "guessed")}
