"""Auto-chopper and Chop & Screw DSP engine for the EP-133 K.O. II.

Solves the core musical limitation of treating the EP-133 as a static single-sample ROMpler:
instead of transposing a single one-shot across +/-12 semitones, this module slices musical stems,
breakbeats (Amen, Think, Apache), and Lyria 3 loops across a 12-pad group (C or D) using either:

1. **Shared-Slot Non-Destructive Trim Mode** (`build_shared_slot_chop_pads`):
   Uploads the loop (optionally pitched/screwed or BPM-matched) into a *single* sample slot and
   assigns all 12 pads in the group to that slot with frame-accurate `trim_start` and `trim_end`
   boundaries, `choke=True` (shared mute group), and `time_mode="reverse"` on the turnaround pad.
   Saves 11 SysEx uploads (~2.5 minutes of transfer time) while preserving instant hardware chop playability.

2. **Multi-Slot Pre-Sliced Crate Mode** (`export_chop_crate_wavs`):
   Slices the audio into 12 discrete WAV files with zero-crossing snap and 2 ms cosine anti-click
   micro-fades, ready for `upload_sounds`.
"""
from __future__ import annotations

import math
import os
from typing import Any, Literal

import numpy as np
import soundfile as sf
from pydantic import BaseModel, Field
from scipy.signal import resample_poly

from . import analysis as an
from .project import BAR_TICKS, Group, NoteEvent, PadLabel, PadSpec, PatternSpec, STEP_TICKS

# Natural producer keypad order: pads 1..9 for main slices, 0 and . for accents, enter for turnaround/reverse pickup
CHOP_PAD_ORDER: tuple[PadLabel, ...] = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "0", ".", "enter")
ChopMode = Literal["hybrid", "equal", "transients"]


class ChopSlice(BaseModel):
    """Metadata and frame boundaries for one pad slice."""
    pad: PadLabel
    index: int = Field(ge=0, le=11)
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=1)
    start_s: float = Field(ge=0.0)
    end_s: float = Field(ge=0.0)
    duration_s: float = Field(ge=0.0)
    rms_db: float
    peak_db: float
    kind: Literal["grid", "transient", "pickup", "equal"] = "grid"
    reverse: bool = False
    pitch_semitones: int = 0
    pan: int = 0


def snap_to_zero_crossing(mono: np.ndarray, frame: int, window: int = 128) -> int:
    """Snap a frame index to the nearest zero-crossing within +/-`window` frames to eliminate click artifacts."""
    n = len(mono)
    if n <= 2:
        return max(0, min(frame, n))
    frame = max(0, min(int(frame), n - 1))
    lo = max(0, frame - window)
    hi = min(n - 1, frame + window)
    seg = mono[lo:hi + 1]
    if len(seg) < 2:
        return frame
    signs = np.signbit(seg)
    crossings = np.nonzero(signs[:-1] != signs[1:])[0]
    if len(crossings) == 0:
        return lo + int(np.argmin(np.abs(seg)))
    best = crossings[int(np.argmin(np.abs((lo + crossings) - frame)))]
    return int(lo + best)


def apply_micro_fades(audio: np.ndarray, sr: int, fade_ms: float = 2.0) -> np.ndarray:
    """Apply short half-cosine fade-in and fade-out ramps to prevent slice boundary clicks."""
    fade_len = min(max(1, int(round(sr * fade_ms / 1000.0))), len(audio) // 4)
    if fade_len <= 1:
        return audio
    out = audio.copy()
    ramp_in = (0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, fade_len)))).astype(out.dtype)
    ramp_out = ramp_in[::-1]
    if out.ndim == 1:
        out[:fade_len] *= ramp_in
        out[-fade_len:] *= ramp_out
    else:
        out[:fade_len, :] *= ramp_in[:, None]
        out[-fade_len:, :] *= ramp_out[:, None]
    return out


def screw_resample(
    audio: np.ndarray,
    sr: int,
    screw_semitones: float = 0.0,
    source_bpm: float | None = None,
    target_bpm: float | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Resample audio like a turntable / SP-1200 / Houston DJ Screw deck so pitch and speed shift together.

    If `source_bpm` and `target_bpm` are provided and `screw_semitones == 0`, computes the exact semitone
    shift needed to lock the loop to `target_bpm`. Otherwise applies `screw_semitones` directly (e.g. -3.5
    semitones slows the loop by ~18.3% and drops formants/bass weight).
    """
    effective_semitones = float(screw_semitones)
    if effective_semitones == 0.0 and source_bpm and target_bpm and source_bpm > 0 and target_bpm > 0:
        effective_semitones = 12.0 * math.log2(float(target_bpm) / float(source_bpm))

    speed_ratio = 2.0 ** (effective_semitones / 12.0)
    if abs(speed_ratio - 1.0) < 1e-4:
        return audio, {"effective_semitones": 0.0, "speed_ratio": 1.0}

    # Resample by 1 / speed_ratio so playing back at `sr` runs at `speed_ratio` speed & pitch
    scale = 1.0 / speed_ratio
    up = int(round(scale * 1000))
    down = 1000
    g = math.gcd(up, down)
    up //= g
    down //= g
    resampled = resample_poly(audio.astype(np.float64), up, down, axis=0).astype(np.float32)
    return resampled, {"effective_semitones": round(effective_semitones, 2), "speed_ratio": round(speed_ratio, 4)}


def detect_chop_slices(
    audio: np.ndarray,
    sr: int,
    num_slices: int = 12,
    mode: ChopMode = "hybrid",
    reverse_last_pad: bool = True,
    pad_order: tuple[PadLabel, ...] = CHOP_PAD_ORDER,
) -> list[ChopSlice]:
    """Slice an audio buffer into up to 12 musical chops mapped to `pad_order`.

    Modes:
    - `"hybrid"`:
      - Pads 1..8 (indices 0..7): 8 structural grid chops across the loop, each snapped within +/-35 ms
        to the nearest broadband or low-mid transient so drum/chord attacks stay crisp.
      - Pads 9, 0, . (indices 8..10): 3 high-energy transient accent chops selected from the punchiest
        onsets in the loop.
      - Pad enter (index 11): turnaround/pickup slice from the final beat, flagged `reverse=True` when
        `reverse_last_pad=True` (maps to EP-133 OS 2.5 `time_mode="reverse"`).
    - `"equal"`: 12 evenly spaced grid slices (matching EP-133 `EQU` auto-chop), zero-crossing snapped.
    - `"transients"`: 12 strongest transient onsets sorted chronologically across the loop.
    """
    num_slices = max(1, min(int(num_slices), len(pad_order)))
    mono_sig = an.mono(audio).astype(np.float64)
    total_frames = len(mono_sig)
    if total_frames < num_slices * 64:
        raise ValueError(f"audio is too short ({total_frames} frames) to slice into {num_slices} chops")

    # Detect both broadband transients and full-band energy onsets
    hp = an.transients(mono_sig, sr, cutoff_hz=1200.0)
    raw_onsets = an.all_onsets(hp, sr, rise_db=8.5, lookback_ms=40.0, refractory_ms=55.0, floor_db=-55.0)
    full_onsets = an.all_onsets(mono_sig, sr, rise_db=7.5, lookback_ms=45.0, refractory_ms=65.0, floor_db=-50.0)
    candidates = sorted(set([0] + raw_onsets + full_onsets))

    snap_tol = int(0.035 * sr)  # 35 ms transient snap window

    def nearest_transient(target: int) -> int:
        if not candidates:
            return target
        best = min(candidates, key=lambda c: abs(c - target))
        return best if abs(best - target) <= snap_tol else target

    raw_bounds: list[tuple[int, int, Literal["grid", "transient", "pickup", "equal"]]] = []

    if mode == "equal" or num_slices < 6:
        step_frames = total_frames / num_slices
        pts = [snap_to_zero_crossing(mono_sig, int(round(i * step_frames))) for i in range(num_slices)] + [total_frames]
        for i in range(num_slices):
            s, e = pts[i], max(pts[i] + 64, pts[i + 1])
            raw_bounds.append((s, min(e, total_frames), "equal"))

    elif mode == "transients":
        # Score candidate onsets by RMS in the 120 ms following each onset
        win = max(64, int(0.12 * sr))
        scored = []
        min_gap = max(int(0.08 * sr), total_frames // (num_slices * 3))
        filtered: list[int] = []
        for c in candidates:
            if not filtered or c - filtered[-1] >= min_gap:
                filtered.append(c)
        for c in filtered:
            seg = mono_sig[c: min(total_frames, c + win)]
            scored.append((an.rms_db(seg), c))
        scored.sort(reverse=True)
        chosen = sorted(c for _rms, c in scored[:num_slices])
        while len(chosen) < num_slices:
            fill = int(round(len(chosen) * total_frames / num_slices))
            chosen.append(fill)
        chosen = sorted(set(chosen))[:num_slices]
        if chosen[0] > int(0.05 * sr):
            chosen[0] = 0
        pts = [snap_to_zero_crossing(mono_sig, c) for c in chosen] + [total_frames]
        for i in range(num_slices):
            s = pts[i]
            e = pts[i + 1] if i + 1 < len(pts) else total_frames
            e = max(s + 64, min(e, total_frames))
            raw_bounds.append((s, e, "transient"))

    else:
        # "hybrid": 8 structural grid slices + 3 punchy transient accents + 1 pickup turnaround chop
        n_grid = min(8, num_slices)
        grid_step = total_frames / n_grid
        grid_starts = [0]
        for i in range(1, n_grid):
            snapped = nearest_transient(int(round(i * grid_step)))
            grid_starts.append(snap_to_zero_crossing(mono_sig, snapped))
        grid_starts.append(total_frames)

        for i in range(n_grid):
            s = grid_starts[i]
            e = max(s + 64, min(grid_starts[i + 1], total_frames))
            raw_bounds.append((s, e, "grid"))

        remaining = num_slices - n_grid
        if remaining > 0:
            n_accents = max(0, remaining - 1) if reverse_last_pad else remaining
            # Pick transient accents that don't duplicate existing grid_starts
            win = max(64, int(0.15 * sr))
            accent_len = max(int(0.25 * sr), int(grid_step * 0.5))
            accent_pool: list[tuple[float, int]] = []
            for c in candidates:
                if all(abs(c - gs) > int(0.04 * sr) for gs in grid_starts[:-1]) and c + 256 < total_frames:
                    accent_pool.append((an.rms_db(mono_sig[c: min(total_frames, c + win)]), c))
            accent_pool.sort(reverse=True)
            chosen_accents = sorted(c for _r, c in accent_pool[:n_accents])
            while len(chosen_accents) < n_accents:
                fallback = int(round((len(chosen_accents) + 0.5) * grid_step))
                chosen_accents.append(min(total_frames - accent_len, max(0, fallback)))
            for c in chosen_accents[:n_accents]:
                s = snap_to_zero_crossing(mono_sig, c)
                e = snap_to_zero_crossing(mono_sig, min(total_frames, s + accent_len))
                raw_bounds.append((s, max(s + 64, e), "transient"))

            if len(raw_bounds) < num_slices:
                # Final pad ("enter"): pickup slice from the last 1/8th or half-bar
                pickup_len = min(total_frames // 2, max(int(0.25 * sr), int(round(grid_step))))
                s = snap_to_zero_crossing(mono_sig, max(0, total_frames - pickup_len))
                raw_bounds.append((s, total_frames, "pickup"))

    slices: list[ChopSlice] = []
    for idx, (s, e, kind) in enumerate(raw_bounds[:num_slices]):
        seg = mono_sig[s:e]
        is_rev = bool(reverse_last_pad and idx == num_slices - 1 and kind == "pickup")
        slices.append(
            ChopSlice(
                pad=pad_order[idx],
                index=idx,
                start_frame=int(s),
                end_frame=int(e),
                start_s=round(s / sr, 4),
                end_s=round(e / sr, 4),
                duration_s=round((e - s) / sr, 4),
                rms_db=round(an.rms_db(seg), 1),
                peak_db=round(an.peak_db(seg), 1),
                kind=kind,
                reverse=is_rev,
                pitch_semitones=0,
                pan=0,
            )
        )
    return slices


def prepare_chopped_audio(
    wav_path: str,
    samplerate: int = 46875,
    mono: bool = False,
    normalize_dbfs: float | None = -1.0,
    screw_semitones: float = 0.0,
    source_bpm: float | None = None,
    target_bpm: float | None = None,
    max_seconds: float | None = None,
) -> tuple[np.ndarray, bytes, dict[str, Any]]:
    """Load, optionally pitch/screw or BPM-match, resample to `samplerate`, normalize, and encode to 16-bit PCM."""
    path = os.path.expanduser(wav_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no such WAV: {path}")
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    orig_sr, orig_ch = int(sr), int(x.shape[1])
    if mono and x.shape[1] > 1:
        x = x.mean(axis=1, keepdims=True)
    elif x.shape[1] > 2:
        x = x[:, :2]

    if max_seconds is not None and len(x) > int(max_seconds * sr):
        x = x[: int(max_seconds * sr)]

    x, screw_info = screw_resample(x, int(sr), screw_semitones=screw_semitones, source_bpm=source_bpm, target_bpm=target_bpm)

    if int(sr) != int(samplerate):
        g = math.gcd(int(sr), int(samplerate))
        x = resample_poly(x.astype(np.float64), int(samplerate) // g, int(sr) // g, axis=0).astype(np.float32)

    if normalize_dbfs is not None and len(x):
        peak = float(np.max(np.abs(x)))
        if peak > 1e-9:
            x = x * float(10.0 ** (normalize_dbfs / 20.0) / peak)

    x = np.clip(x, -1.0, 1.0)
    limit = 40.0 if x.shape[1] == 1 else 20.0
    seconds = len(x) / samplerate
    if seconds > limit:
        max_frames = int((limit - 0.05) * samplerate)
        x = x[:max_frames]
        seconds = len(x) / samplerate

    x = apply_micro_fades(x, samplerate, fade_ms=2.0)
    pcm = (x * 32767.0).astype("<i2").tobytes()
    info = {
        "source_samplerate": orig_sr,
        "source_channels": orig_ch,
        "samplerate": int(samplerate),
        "channels": int(x.shape[1]),
        "frames": int(len(x)),
        "seconds": round(seconds, 3),
        **screw_info,
    }
    return x, pcm, info


def build_shared_slot_chop_pads(
    slices: list[ChopSlice],
    slot: int,
    group: Group = "C",
    sample_bpm: float = 120.0,
    amplitude: int = 100,
    pitch: int = 0,
    choke: bool = True,
    play_mode: Literal["oneshot", "key", "legato"] = "oneshot",
) -> list[PadSpec]:
    """Build 12 `PadSpec` records referencing a single uploaded sample slot via non-destructive `trim_start`/`trim_end`."""
    pads: list[PadSpec] = []
    for sl in slices:
        pads.append(
            PadSpec(
                group=group,
                pad=sl.pad,
                slot=int(slot),
                trim_start=int(sl.start_frame),
                trim_end=int(sl.end_frame),
                sample_bpm=float(sample_bpm),
                amplitude=int(amplitude),
                pitch=max(-12, min(12, int(pitch + sl.pitch_semitones))),
                pan=max(-16, min(16, int(sl.pan))),
                attack=0,
                release=255,
                time_mode="reverse" if sl.reverse else "off",
                choke=bool(choke),
                play_mode=play_mode,
            )
        )
    return pads


def export_chop_crate_wavs(
    audio: np.ndarray,
    sr: int,
    slices: list[ChopSlice],
    out_dir: str,
    prefix: str = "chop",
) -> list[dict[str, Any]]:
    """Write each chop slice as its own anti-clicked WAV file on disk for multi-slot crate uploads."""
    os.makedirs(out_dir, exist_ok=True)
    exported: list[dict[str, Any]] = []
    for sl in slices:
        seg = audio[sl.start_frame: sl.end_frame].copy()
        if sl.reverse:
            seg = seg[::-1].copy()
        seg = apply_micro_fades(seg, sr, fade_ms=2.0)
        fname = f"{prefix}_{sl.index + 1:02d}_pad_{sl.pad.replace('.', 'dot')}.wav"
        fpath = os.path.join(out_dir, fname)
        sf.write(fpath, seg, sr, subtype="PCM_16")
        exported.append({"pad": sl.pad, "index": sl.index, "wav": fpath, "frames": int(len(seg)), "slice": sl.model_dump()})
    return exported


def sequence_chop_pattern(
    group: Group,
    number: int,
    bars: int,
    hits: list[dict[str, Any]],
    legato_choke: bool = True,
) -> PatternSpec:
    """Compile a list of chop triggers `{pad, step, velocity?, semitones?, duration_steps?}` into a `PatternSpec`.

    When `legato_choke=True`, automatically sizes each event's tick duration so it cuts cleanly right before the
    next chop trigger (useful for `key` / `legato` play modes as well as tidy MIDI output).
    """
    sorted_hits = sorted(hits, key=lambda h: float(h["step"]))
    total_ticks = int(bars) * BAR_TICKS
    events: list[NoteEvent] = []
    for i, h in enumerate(sorted_hits):
        step = float(h["step"])
        tick = int(round(step * STEP_TICKS))
        if tick >= total_ticks:
            continue
        if "duration_steps" in h and h["duration_steps"] is not None:
            dur_ticks = max(4, int(round(float(h["duration_steps"]) * STEP_TICKS)))
        elif legato_choke and i + 1 < len(sorted_hits):
            next_tick = int(round(float(sorted_hits[i + 1]["step"]) * STEP_TICKS))
            dur_ticks = max(6, min(BAR_TICKS, next_tick - tick - 2))
        else:
            dur_ticks = min(BAR_TICKS, max(24, total_ticks - tick))
        events.append(
            NoteEvent(
                pad=h["pad"],
                tick=tick,
                velocity=int(h.get("velocity", 105)),
                semitones=h.get("semitones"),
                concert=h.get("concert"),
                duration=min(BAR_TICKS, dur_ticks),
            )
        )
    return PatternSpec(group=group, number=number, bars=bars, events=events)
