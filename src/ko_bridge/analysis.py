"""Measurements on captured USB audio: levels, onsets, silent spans.

The onset detector follows op-bridge's: the first 2 ms frame whose level rises `rise_db` above the floor measured just
before the search start. For percussive test material a high-pass first (`transients`) keeps a ringing tail from the
previous hit out of the floor estimate.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt

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
    sos = butter(4, min(0.95, cutoff_hz / (sr / 2.0)), btype="highpass", output="sos")
    return sosfiltfilt(sos, x.astype(np.float64))


def low_transients(x: np.ndarray, sr: int, low_hz: float = 35.0, high_hz: float = 450.0) -> np.ndarray:
    """Causal band-passed low/low-mid copy of `x` (no backward pre-ringing) so 808s, kicks, and warm chops register accurate onsets."""
    nyq = sr / 2.0
    lo = max(0.005, min(0.9, low_hz / nyq))
    hi = max(lo + 0.01, min(0.95, high_hz / nyq))
    sos = butter(3, [lo, hi], btype="bandpass", output="sos")
    return sosfilt(sos, x.astype(np.float64))


def bandpass(x: np.ndarray, sr: int, low_hz: float, high_hz: float) -> np.ndarray:
    """Zero-phase 4th-order Butterworth bandpass filter."""
    if len(x) < 32:
        return x.astype(np.float64)
    nyq = sr / 2.0
    lo = max(0.002, min(0.92, low_hz / nyq))
    hi = max(lo + 0.005, min(0.98, high_hz / nyq))
    if hi >= 0.97:
        sos = butter(4, lo, btype="highpass", output="sos")
    elif lo <= 0.005:
        sos = butter(4, hi, btype="lowpass", output="sos")
    else:
        sos = butter(4, [lo, hi], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x.astype(np.float64))


def spectral_mix_report(audio: np.ndarray, sr: int) -> dict:
    """4-band spectral balance (sub, low, mid, high), crest factor, clipping count, stereo width, and mix warnings."""
    m = mono(audio).astype(np.float64)
    if len(m) < 64:
        return {"seconds": 0.0, "warnings": ["capture is empty"]}

    overall_rms = rms_db(m)
    overall_peak = peak_db(audio)
    crest_db = round(overall_peak - overall_rms, 1)
    clipped_samples = int(np.sum(np.abs(audio) >= 0.998))

    bands_hz = {
        "sub_20_80hz": (20.0, 80.0),
        "low_80_250hz": (80.0, 250.0),
        "mid_250_4khz": (250.0, 4000.0),
        "high_4k_20khz": (4000.0, min(20000.0, sr * 0.48)),
    }
    bands_db = {name: round(rms_db(bandpass(m, sr, lo, hi)), 1) for name, (lo, hi) in bands_hz.items()}

    stereo_width = 0.0
    if audio.ndim == 2 and audio.shape[1] >= 2:
        side = math.sqrt(float(np.mean((audio[:, 0] - audio[:, 1]) ** 2)) + EPS)
        mid_ch = math.sqrt(float(np.mean((audio[:, 0] + audio[:, 1]) ** 2)) + EPS)
        stereo_width = round(side / mid_ch, 3)

    warnings: list[str] = []
    if clipped_samples > 10:
        warnings.append(f"hard digital clipping ({clipped_samples} clipped samples at >= -0.02 dBFS); lower group fader or pad amplitude")
    if bands_db["low_80_250hz"] - bands_db["mid_250_4khz"] > 6.5 and overall_rms > -40.0:
        warnings.append(f"muddy low-mids: 80-250 Hz ({bands_db['low_80_250hz']} dBFS) exceeds 250-4k Hz ({bands_db['mid_250_4khz']} dBFS) by >6.5 dB")
    if bands_db["sub_20_80hz"] < overall_rms - 18.0 and overall_rms > -35.0:
        warnings.append(f"thin sub-bass: 20-80 Hz is {bands_db['sub_20_80hz']} dBFS ({round(overall_rms - bands_db['sub_20_80hz'], 1)} dB below overall RMS)")
    if bands_db["high_4k_20khz"] < overall_rms - 22.0 and overall_rms > -35.0:
        warnings.append(f"dull top-end: 4-20 kHz is {bands_db['high_4k_20khz']} dBFS; hats/chops may be overly low-passed")
    if crest_db < 5.5 and overall_rms > -25.0:
        warnings.append(f"over-compressed / flat dynamics (crest factor {crest_db} dB); add sidechain ducking or reduce compressor drive")

    return {
        "rms_dbfs": round(overall_rms, 1),
        "peak_dbfs": round(overall_peak, 1),
        "crest_factor_db": crest_db,
        "clipped_samples": clipped_samples,
        "stereo_width": stereo_width,
        "bands_dbfs": bands_db,
        "warnings": warnings,
    }


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
