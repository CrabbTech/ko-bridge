"""Loop picking for long generations: the real tempo of a piece and its strongest bar-aligned windows.

A Lyria generation is a minute of song-shaped music, not a loop: the first bars are usually a quiet intro and the stated
tempo is approximate (a request for 90 BPM came back at 91.85). `estimate_bpm` reads the tempo off the onset train and
`pick_loops` ranks candidate windows of `bars` bars that start on an onset by level, transient density and how cleanly
the window closes on another onset, so the chopper gets material with attacks in it.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from . import analysis as an


def _onset_strength(x: np.ndarray, sr: int, win_ms: float = 10.0) -> tuple[int, np.ndarray]:
    """Half-wave rectified level rise per frame, summed over a high-passed and a full-band copy."""
    hop, lev_hp = an.frame_levels_db(an.transients(x, sr, cutoff_hz=1500.0), sr, win_ms)
    _hop, lev_full = an.frame_levels_db(x, sr, win_ms)
    n = min(len(lev_hp), len(lev_full))
    rise = np.maximum(np.diff(lev_hp[:n], prepend=lev_hp[0]), 0) + 0.5 * np.maximum(np.diff(lev_full[:n], prepend=lev_full[0]), 0)
    rise[lev_full[:n] < -55] = 0.0
    return hop, rise


def estimate_bpm(x: np.ndarray, sr: int, hint: float | None = None, lo: float = 60.0, hi: float = 180.0, tolerance: float = 0.15) -> dict[str, Any]:
    """Tempo from the autocorrelation of the onset strength: the beat period with the strongest periodicity, preferring a
    value within `tolerance` of `hint` (half and double tempos are folded onto the hint's octave)."""
    hop, s = _onset_strength(an.mono(x), sr)
    s = s - s.mean()
    if len(s) < 64 or not np.any(s):
        return {"bpm": hint, "confidence": 0.0, "candidates": []}
    ac = np.correlate(s, s, mode="full")[len(s) - 1:]
    ac = ac / (ac[0] + 1e-12)
    frame_s = hop / sr
    lag_lo, lag_hi = int(60.0 / hi / frame_s), int(60.0 / lo / frame_s)
    lags = np.arange(lag_lo, min(lag_hi, len(ac) - 1))
    # reward lags whose multiples also correlate (a true beat period repeats at 2x and 4x)
    score = np.array([ac[l] + 0.5 * ac[2 * l] + 0.25 * ac[4 * l] if 4 * l < len(ac) else ac[l] for l in lags])
    order = np.argsort(score)[::-1][:12]
    cands = []
    for i in order:
        l = lags[i]
        bpm = 60.0 / (l * frame_s)
        cands.append({"bpm": round(bpm, 2), "score": round(float(score[i]), 3)})
    best = cands[0]
    if hint:
        folded = []
        for c in cands:
            b = c["bpm"]
            while b < hint / 1.5:
                b *= 2
            while b > hint * 1.5:
                b /= 2
            folded.append({"bpm": round(b, 2), "score": c["score"], "raw": c["bpm"]})
        near = [c for c in folded if abs(c["bpm"] - hint) / hint <= tolerance]
        if near:
            best = max(near, key=lambda c: c["score"])
    # refine the best lag by parabolic interpolation around the autocorrelation peak
    l0 = int(round(60.0 / best["bpm"] / frame_s))
    if 1 <= l0 < len(ac) - 1:
        a, b, c = ac[l0 - 1], ac[l0], ac[l0 + 1]
        denom = (a - 2 * b + c)
        shift = 0.5 * (a - c) / denom if abs(denom) > 1e-9 else 0.0
        bpm = 60.0 / ((l0 + float(np.clip(shift, -0.5, 0.5))) * frame_s)
    else:
        bpm = best["bpm"]
    return {"bpm": round(float(bpm), 2), "confidence": round(float(max(0.0, min(1.0, best["score"]))), 3), "candidates": cands[:6]}


def pick_loops(x: np.ndarray, sr: int, bpm: float, bars: int = 2, top: int = 5, min_start_s: float = 0.0, beats_per_bar: int = 4) -> list[dict[str, Any]]:
    """Rank windows of `bars` bars at `bpm` that start on an onset. Score: level, high-band transient count, a strong
    attack at the start, an onset near the window's end (so the loop closes), and a penalty for a silent tail."""
    m = an.mono(x).astype(np.float64)
    loop_s = bars * beats_per_bar * 60.0 / float(bpm)
    loop_n = int(round(loop_s * sr))
    if loop_n >= len(m):
        return [{"start_s": 0.0, "seconds": round(len(m) / sr, 3), "score": 0.0, "note": "shorter than the window"}]
    hp = an.transients(m, sr, cutoff_hz=1500.0)
    full_onsets = an.all_onsets(m, sr, rise_db=8.0, lookback_ms=60.0, refractory_ms=80.0, floor_db=-50.0, win_ms=10.0)
    hp_onsets = an.all_onsets(hp, sr, rise_db=9.0, lookback_ms=50.0, refractory_ms=60.0, floor_db=-55.0, win_ms=5.0)
    starts = sorted(set(o for o in full_onsets + hp_onsets if o / sr >= min_start_s and o + loop_n <= len(m)))
    hp_arr = np.array(hp_onsets)
    out = []
    for s0 in starts:
        seg = m[s0:s0 + loop_n]
        rms = an.rms_db(seg)
        if rms < -45:
            continue
        hits = int(np.sum((hp_arr >= s0) & (hp_arr < s0 + loop_n))) if len(hp_arr) else 0
        attack = 20 * np.log10(np.max(np.abs(hp[s0:s0 + int(0.05 * sr)])) + 1e-9)
        tail = an.rms_db(seg[-int(0.25 * loop_n):])
        end_near = min((abs(o - (s0 + loop_n)) for o in full_onsets), default=loop_n) / sr
        score = rms + 0.8 * min(hits, 40) + 0.3 * max(-40.0, attack) - 8.0 * max(0.0, (end_near - 0.03) / 0.1) - 0.5 * max(0.0, (rms - tail) - 12)
        out.append({"start_s": round(s0 / sr, 3), "seconds": round(loop_s, 3), "rms_dbfs": round(rms, 1), "high_onsets": hits,
                    "attack_dbfs": round(float(attack), 1), "end_onset_gap_ms": round(end_near * 1000, 1), "tail_rms_dbfs": round(tail, 1),
                    "score": round(float(score), 2)})
    out.sort(key=lambda r: -r["score"])
    # keep picks apart by at least a bar so the top list spans the piece
    kept: list[dict[str, Any]] = []
    for r in out:
        if all(abs(r["start_s"] - k["start_s"]) >= loop_s / bars for k in kept):
            kept.append(r)
        if len(kept) >= top:
            break
    return kept


def cut_loop(x: np.ndarray, sr: int, start_s: float, seconds: float, preroll_ms: float = 8.0, fade_ms: float = 4.0) -> np.ndarray:
    """The window with a little pre-roll before the attack and short fades at both ends."""
    s0 = max(0, int(round((start_s - preroll_ms / 1000.0) * sr)))
    n = int(round(seconds * sr)) + int(round(preroll_ms / 1000.0 * sr))
    seg = np.array(x[s0:s0 + n], dtype=np.float32, copy=True)
    f = max(1, int(sr * fade_ms / 1000.0))
    ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
    if seg.ndim == 1:
        seg[:f] *= ramp
        seg[-f:] *= ramp[::-1]
    else:
        seg[:f] *= ramp[:, None]
        seg[-f:] *= ramp[::-1][:, None]
    return seg
