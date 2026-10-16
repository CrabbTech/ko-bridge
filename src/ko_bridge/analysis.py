"""Measurements on captured USB audio: levels, onsets, silent spans.

The onset detector follows op-bridge's: the first 2 ms frame whose level rises `rise_db` above the floor measured just
before the search start. For percussive test material a high-pass first (`transients`) keeps a ringing tail from the
previous hit out of the floor estimate.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import butter, sosfiltfilt

EPS = 1e-12


def db(x: float) -> float:
    return 20.0 * math.log10(max(float(x), EPS))


def rms_db(x: np.ndarray) -> float:
    if len(x) == 0:
        return -120.0
    return db(math.sqrt(float(np.mean(np.square(x, dtype=np.float64)))))


def peak_db(x: np.ndarray) -> float:
    if len(x) == 0:
        return -120.0
    return db(float(np.max(np.abs(x))))


def channel_levels(audio: np.ndarray) -> list[float]:
    """RMS in dBFS per channel of a frames x channels array."""
    return [rms_db(audio[:, c]) for c in range(audio.shape[1])]


def mono(audio: np.ndarray) -> np.ndarray:
    return audio if audio.ndim == 1 else audio.mean(axis=1)


def transients(x: np.ndarray, sr: int, cutoff_hz: float = 2500.0) -> np.ndarray:
    """High-passed copy of `x`, so broadband attacks stand out from sustained low tones."""
    sos = butter(4, cutoff_hz / (sr / 2.0), btype="highpass", output="sos")
    return sosfiltfilt(sos, x.astype(np.float64))


def frame_levels_db(x: np.ndarray, sr: int, win_ms: float = 2.0) -> tuple[int, np.ndarray]:
    win = max(1, int(sr * win_ms / 1000.0))
    n = len(x) // win
    if n == 0:
        return win, np.zeros(0)
    frames = x[: n * win].reshape(n, win).astype(np.float64)
    return win, 20.0 * np.log10(np.sqrt(np.mean(frames * frames, axis=1)) + EPS)


def onset_sample(x: np.ndarray, sr: int, start: int = 0, rise_db: float = 12.0, floor_window_ms: float = 60.0,
                 win_ms: float = 2.0, floor_min_db: float = -90.0) -> int | None:
    """First sample at or after `start` where the level rises `rise_db` above the floor measured just before `start`."""
    fw = int(sr * floor_window_ms / 1000.0)
    pre = x[max(0, start - fw): start]
    floor = max(rms_db(pre) if len(pre) > 0 else -90.0, floor_min_db)
    win, lev = frame_levels_db(x[start:], sr, win_ms)
    hits = np.nonzero(lev > floor + rise_db)[0]
    if len(hits) == 0:
        return None
    return start + int(hits[0]) * win


def onsets_near(x: np.ndarray, sr: int, expected_samples: list[int], before_ms: float = 60.0, after_ms: float = 150.0,
                rise_db: float = 12.0) -> list[int | None]:
    """For each expected onset position, the detected onset inside [-before, +after] ms, or None."""
    found = []
    for e in expected_samples:
        start = max(0, e - int(sr * before_ms / 1000.0))
        s = onset_sample(x, sr, start=start, rise_db=rise_db)
        if s is None or s > e + int(sr * after_ms / 1000.0):
            found.append(None)
        else:
            found.append(s)
    return found


def all_onsets(x: np.ndarray, sr: int, rise_db: float = 12.0, lookback_ms: float = 50.0, refractory_ms: float = 60.0,
               floor_db: float = -60.0, win_ms: float = 2.0) -> list[int]:
    """Every frame whose level exceeds both the absolute floor and the minimum of the preceding `lookback` by `rise_db`."""
    win, lev = frame_levels_db(x, sr, win_ms)
    look = max(1, int(lookback_ms / win_ms))
    refractory = max(1, int(refractory_ms / win_ms))
    out = []
    last = -refractory
    for i in range(look, len(lev)):
        if i - last < refractory:
            continue
        base = float(np.min(lev[i - look: i]))
        if lev[i] > floor_db and lev[i] > base + rise_db:
            out.append(i * win)
            last = i
    return out


def silent_spans(x: np.ndarray, sr: int, threshold_db: float = -55.0, min_ms: float = 80.0, win_ms: float = 5.0) -> list[tuple[float, float]]:
    """Spans (start_s, end_s) where the level stays below `threshold_db` for at least `min_ms`."""
    win, lev = frame_levels_db(x, sr, win_ms)
    quiet = lev < threshold_db
    spans = []
    i = 0
    n = len(quiet)
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            if (j - i) * win_ms >= min_ms:
                spans.append((i * win / sr, j * win / sr))
            i = j
        else:
            i += 1
    return spans


def stats_ms(values: list[float]) -> dict:
    a = np.asarray(values, dtype=np.float64)
    if len(a) == 0:
        return {"n": 0}
    return {"n": int(len(a)), "mean_ms": round(float(a.mean()), 2), "std_ms": round(float(a.std()), 2),
            "min_ms": round(float(a.min()), 2), "max_ms": round(float(a.max()), 2),
            "p2p_ms": round(float(a.max() - a.min()), 2), "median_ms": round(float(np.median(a)), 2)}
