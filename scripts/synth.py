#!/usr/bin/env python3
"""Original sound design for the EP-133, rendered on the Mac at the device's native 46,875 Hz.

Drums (kicks, snare, clap, hats, ride, shaker, tom, zap), basses with the wobble BAKED IN at the song's tempo (a
resonant 24 dB low-pass swept by an LFO over detuned band-limited saws, plus FM growls), a sub, an FM electric piano,
an FM bell, a Karplus-Strong pluck, a detuned-saw pad, a formant "vowel" pad, a two-bar riser, an impact and a crackle
texture. pedalboard adds chorus, reverb, compression and drive where it helps. Everything is synthesized here: no
sampled material.

    uv run python scripts/synth.py --bpm 174 --out ~/Music/ko-bridge/library/sets/dnb-174
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter, lfilter_zi, sosfilt

SR = 46875
RNG = np.random.default_rng(20261016)

try:
    from pedalboard import Chorus, Compressor, Distortion, HighpassFilter, Pedalboard, Reverb
    HAVE_PB = True
except Exception:  # pedalboard is optional
    HAVE_PB = False


# ----------------------------------------------------------------------------- building blocks

def tvec(n: int) -> np.ndarray:
    return np.arange(n) / SR


def midi_hz(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def exp_env(n: int, tau: float, start: float = 1.0) -> np.ndarray:
    return start * np.exp(-tvec(n) / tau)


def adsr(n: int, a: float, d: float, s: float, r: float) -> np.ndarray:
    t = tvec(n)
    env = np.ones(n) * s
    ai = min(n, int(a * SR)); di = min(n, ai + int(d * SR)); ri = max(0, n - int(r * SR))
    env[:ai] = np.linspace(0, 1, ai, endpoint=False) if ai else env[:ai]
    if di > ai:
        env[ai:di] = np.linspace(1, s, di - ai, endpoint=False)
    if ri < n:
        env[ri:] = env[ri:] * np.linspace(1, 0, n - ri)
    return env


def bl_saw(freq: float, n: int, phase: float = 0.0, max_hz: float = 12000.0) -> np.ndarray:
    """Band-limited sawtooth by additive synthesis (harmonics up to max_hz)."""
    t = tvec(n)
    k_max = max(1, int(max_hz / freq))
    out = np.zeros(n)
    for k in range(1, k_max + 1):
        out += np.sin(2 * np.pi * (k * freq * t + k * phase)) / k
    return out * (2 / np.pi)


def bl_square(freq: float, n: int, max_hz: float = 12000.0) -> np.ndarray:
    t = tvec(n)
    out = np.zeros(n)
    k = 1
    while k * freq < max_hz:
        out += np.sin(2 * np.pi * k * freq * t) / k
        k += 2
    return out * (4 / np.pi)


def noise(n: int) -> np.ndarray:
    return RNG.uniform(-1, 1, n)


def drive(x: np.ndarray, amount: float) -> np.ndarray:
    return np.tanh(x * amount) / math.tanh(amount)


def normalize(x: np.ndarray, peak_dbfs: float = -1.0) -> np.ndarray:
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    return x * (10 ** (peak_dbfs / 20) / peak) if peak > 0 else x


def bp(x: np.ndarray, lo: float, hi: float, order: int = 2) -> np.ndarray:
    sos = butter(order, [lo / (SR / 2), min(hi, SR / 2 - 100) / (SR / 2)], btype="bandpass", output="sos")
    return sosfilt(sos, x)


def hp(x: np.ndarray, hz: float, order: int = 2) -> np.ndarray:
    return sosfilt(butter(order, hz / (SR / 2), btype="highpass", output="sos"), x)


def lp(x: np.ndarray, hz: float, order: int = 2) -> np.ndarray:
    return sosfilt(butter(order, min(hz, SR / 2 - 100) / (SR / 2), btype="lowpass", output="sos"), x)


def rbj_lowpass(fc: float, q: float):
    w0 = 2 * np.pi * min(fc, SR * 0.45) / SR
    alpha = np.sin(w0) / (2 * q)
    cosw = np.cos(w0)
    b = np.array([(1 - cosw) / 2, 1 - cosw, (1 - cosw) / 2])
    a = np.array([1 + alpha, -2 * cosw, 1 - alpha])
    return b / a[0], a / a[0]


def rbj_bandpass(fc: float, q: float):
    w0 = 2 * np.pi * min(fc, SR * 0.45) / SR
    alpha = np.sin(w0) / (2 * q)
    b = np.array([alpha, 0.0, -alpha])
    a = np.array([1 + alpha, -2 * np.cos(w0), 1 - alpha])
    return b / a[0], a / a[0]


def rbj_highpass(fc: float, q: float):
    w0 = 2 * np.pi * min(fc, SR * 0.45) / SR
    alpha = np.sin(w0) / (2 * q)
    cosw = np.cos(w0)
    b = np.array([(1 + cosw) / 2, -(1 + cosw), (1 + cosw) / 2])
    a = np.array([1 + alpha, -2 * cosw, 1 - alpha])
    return b / a[0], a / a[0]


def swept_filter(x: np.ndarray, cutoff: np.ndarray, q: float, kind: str = "lowpass", stages: int = 2, block: int = 64) -> np.ndarray:
    """Time-varying biquad (RBJ), coefficients updated every `block` samples, `stages` cascaded (2 = 24 dB/oct)."""
    y = x.astype(np.float64).copy()
    design = {"lowpass": rbj_lowpass, "bandpass": rbj_bandpass, "highpass": rbj_highpass}[kind]
    for _stage in range(stages):
        out = np.empty_like(y)
        zi = None
        for i in range(0, len(y), block):
            fc = float(cutoff[min(i, len(cutoff) - 1)])
            b, a = design(fc, q)
            if zi is None:
                zi = lfilter_zi(b, a) * y[0]
            seg, zi = lfilter(b, a, y[i:i + block], zi=zi)
            out[i:i + block] = seg
        y = out
    return y


def lfo(n: int, period_s: float, shape: str = "wub", phase: float = 0.0) -> np.ndarray:
    """0..1 modulation. wub = peaky rise and fall, square = smoothed square, sine, saw_down = fast open then close."""
    ph = (tvec(n) / period_s + phase) % 1.0
    if shape == "sine":
        return 0.5 - 0.5 * np.cos(2 * np.pi * ph)
    if shape == "square":
        return 1 / (1 + np.exp(-24 * (0.5 - np.abs(ph - 0.5)) + 6))
    if shape == "saw_down":
        return (1 - ph) ** 1.5
    return (0.5 - 0.5 * np.cos(2 * np.pi * ph)) ** 0.6     # wub


def polish(x: np.ndarray, chain: list) -> np.ndarray:
    if not HAVE_PB or not chain:
        return x
    y = Pedalboard(chain)(x.astype(np.float32)[None, :], SR)
    return y[0].astype(np.float64)


def stereo_polish(x: np.ndarray, chain_l: list, chain_r: list) -> np.ndarray:
    if not HAVE_PB:
        return np.stack([x, x], axis=1)
    l = Pedalboard(chain_l)(x.astype(np.float32)[None, :], SR)[0]
    r = Pedalboard(chain_r)(x.astype(np.float32)[None, :], SR)[0]
    return np.stack([l, r], axis=1).astype(np.float64)


# ----------------------------------------------------------------------------- drums

def kick(punchy: bool = True) -> np.ndarray:
    n = int(0.55 * SR)
    t = tvec(n)
    f = (52 + 120 * np.exp(-t / 0.022)) if punchy else (44 + 70 * np.exp(-t / 0.035))
    phase = np.cumsum(f) / SR
    body = np.sin(2 * np.pi * phase) * np.exp(-t / (0.26 if punchy else 0.42))
    click = noise(n) * np.exp(-t / 0.0015) * (0.9 if punchy else 0.25)
    x = drive(body * 1.1 + hp(click, 2000), 1.8 if punchy else 1.2)
    if not punchy:
        x = lp(x, 180, 2)
    return normalize(polish(x, [Compressor(threshold_db=-10, ratio=3)]), -0.8)


def snare() -> np.ndarray:
    n = int(0.6 * SR)
    t = tvec(n)
    body = np.sin(2 * np.pi * 186 * t) * np.exp(-t / 0.075) + 0.6 * np.sin(2 * np.pi * 334 * t) * np.exp(-t / 0.045)
    rattle = bp(noise(n), 1200, 7500) * np.exp(-t / 0.13)
    crack = noise(n) * np.exp(-t / 0.004)
    x = drive(0.9 * body + 1.0 * rattle + 0.5 * crack, 2.2)
    return normalize(polish(x, [Compressor(threshold_db=-12, ratio=4), Reverb(room_size=0.22, wet_level=0.14, dry_level=0.86)]), -0.8)


def clap() -> np.ndarray:
    n = int(0.5 * SR)
    t = tvec(n)
    x = np.zeros(n)
    for k, off in enumerate((0.0, 0.011, 0.021, 0.033)):
        i = int(off * SR)
        burst = bp(noise(n - i), 900, 5200) * np.exp(-tvec(n - i) / (0.012 if k < 3 else 0.14))
        x[i:] += burst * (0.8 if k < 3 else 1.0)
    return normalize(polish(drive(x, 1.6), [Reverb(room_size=0.35, wet_level=0.2, dry_level=0.8)]), -1.0)


def metal(n: int, tau: float, hpf: float = 7000) -> np.ndarray:
    freqs = [205.3, 304.4, 369.6, 522.7, 540.0, 800.0]
    x = sum(bl_square(f * 7.9, n, max_hz=18000) for f in freqs) / 6
    return hp(x, hpf, 2) * exp_env(n, tau)


def hat_closed() -> np.ndarray:
    return normalize(metal(int(0.14 * SR), 0.028), -2.0)


def hat_open() -> np.ndarray:
    return normalize(metal(int(0.7 * SR), 0.23), -2.0)


def ride() -> np.ndarray:
    n = int(1.6 * SR)
    x = metal(n, 0.55, 3000) + 0.4 * bp(noise(n), 4000, 11000) * exp_env(n, 0.7)
    return normalize(polish(x, [Reverb(room_size=0.3, wet_level=0.12, dry_level=0.88)]), -3.0)


def shaker() -> np.ndarray:
    n = int(0.16 * SR)
    env = adsr(n, 0.012, 0.06, 0.0, 0.02)
    return normalize(bp(noise(n), 4000, 9500) * env, -4.0)


def rim() -> np.ndarray:
    n = int(0.09 * SR)
    t = tvec(n)
    return normalize(bp(noise(n), 1800, 5000) * np.exp(-t / 0.012) + 0.6 * np.sin(2 * np.pi * 820 * t) * np.exp(-t / 0.02), -3.0)


def tom() -> np.ndarray:
    n = int(0.6 * SR)
    t = tvec(n)
    f = 85 + 60 * np.exp(-t / 0.06)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.22) + 0.3 * noise(n) * np.exp(-t / 0.003)
    return normalize(drive(x, 1.5), -1.0)


def zap() -> np.ndarray:
    n = int(0.25 * SR)
    t = tvec(n)
    f = 150 + 2400 * np.exp(-t / 0.03)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR + 2.0 * np.sin(2 * np.pi * np.cumsum(f * 1.5) / SR)) * np.exp(-t / 0.07)
    return normalize(x, -2.0)


# ----------------------------------------------------------------------------- basses (loops baked at tempo)

def reese_loop(freq: float, bars: int, bpm: float, period_steps: float, shape: str = "wub", lo_hz: float = 110, hi_hz: float = 2600,
               q: float = 4.5, pattern: list[float] | None = None) -> np.ndarray:
    """Detuned saws through a resonant 24 dB low-pass swept by an LFO whose period is `period_steps` 16ths.
    `pattern`: optional per-16th multipliers (0..1) that gate the LFO depth for stutter rhythms."""
    n = int(round(bars * 240 / bpm * SR))
    step_s = 60 / bpm / 4
    src = (bl_saw(freq * 0.994, n) + bl_saw(freq, n, 0.33) + bl_saw(freq * 1.006, n, 0.66)) / 3
    src += 0.45 * np.sin(2 * np.pi * freq * tvec(n))
    mod = lfo(n, period_steps * step_s, shape)
    if pattern:
        gate = np.repeat(np.array(pattern, dtype=float), int(step_s * SR) + 1)[:n]
        if len(gate) < n:
            gate = np.pad(gate, (0, n - len(gate)), constant_values=gate[-1])
        mod = mod * gate
    cutoff = lo_hz * (hi_hz / lo_hz) ** mod
    x = swept_filter(src, cutoff, q)
    x = drive(x, 2.2)
    x = x + 0.5 * np.sin(2 * np.pi * freq * tvec(n))               # steady sub under the wub
    x = hp(x, 28, 2)
    env = np.ones(n); a = int(0.004 * SR); env[:a] = np.linspace(0, 1, a); env[-a:] *= np.linspace(1, 0, a)
    return normalize(polish(x * env, [Compressor(threshold_db=-8, ratio=3)]), -1.0)


def growl_loop(freq: float, bars: int, bpm: float, period_steps: float) -> np.ndarray:
    n = int(round(bars * 240 / bpm * SR))
    t = tvec(n)
    step_s = 60 / bpm / 4
    mod = lfo(n, period_steps * step_s, "wub")
    index = 1.5 + 9.0 * mod
    car = np.sin(2 * np.pi * freq * t + index * np.sin(2 * np.pi * freq * t))
    car2 = np.sin(2 * np.pi * 2 * freq * t + 0.6 * index * np.sin(2 * np.pi * 2.01 * freq * t))
    formant = swept_filter(car + 0.6 * car2, 380 * (1900 / 380) ** mod, 2.2, kind="bandpass", stages=1)
    x = drive(0.6 * (car + 0.6 * car2) + 1.6 * formant, 3.0) + 0.5 * np.sin(2 * np.pi * freq * t)
    env = np.ones(n); a = int(0.004 * SR); env[:a] = np.linspace(0, 1, a); env[-a:] *= np.linspace(1, 0, a)
    return normalize(polish(hp(x * env, 28), [Compressor(threshold_db=-8, ratio=3)]), -1.0)


def sub(freq: float, seconds: float = 2.5) -> np.ndarray:
    n = int(seconds * SR)
    t = tvec(n)
    x = np.sin(2 * np.pi * freq * t) + 0.1 * np.sin(2 * np.pi * 2 * freq * t)
    env = adsr(n, 0.008, 0.1, 0.9, 0.3)
    return normalize(drive(x, 1.3) * env, -1.0)


def bass_pluck(freq: float, seconds: float = 0.7) -> np.ndarray:
    n = int(seconds * SR)
    t = tvec(n)
    src = (bl_saw(freq, n) + bl_saw(freq * 1.004, n, 0.5)) / 2 + 0.4 * np.sin(2 * np.pi * freq * t)
    cutoff = 150 + 3000 * np.exp(-t / 0.12)
    x = swept_filter(src, cutoff, 3.0) * np.exp(-t / 0.35)
    return normalize(drive(x, 1.6), -1.0)


# ----------------------------------------------------------------------------- keys, pads, bells, pluck

def epiano(freq: float, seconds: float = 2.0) -> np.ndarray:
    n = int(seconds * SR)
    t = tvec(n)
    index = 0.4 + 2.8 * np.exp(-t / 0.3)
    tone = np.sin(2 * np.pi * freq * t + index * np.sin(2 * np.pi * freq * t))
    tine = 0.08 * np.sin(2 * np.pi * freq * 14 * t) * np.exp(-t / 0.05)
    trem = 1 + 0.08 * np.sin(2 * np.pi * 5.2 * t)
    x = (tone + tine) * np.exp(-t / 1.1) * trem
    env = adsr(n, 0.004, 0.2, 1.0, 0.25)
    return normalize(polish(x * env, [Chorus(rate_hz=0.6, depth=0.15, mix=0.25), Reverb(room_size=0.4, wet_level=0.18, dry_level=0.82)]), -1.5)


def bell(freq: float, seconds: float = 2.4) -> np.ndarray:
    n = int(seconds * SR)
    t = tvec(n)
    index = 2.4 * np.exp(-t / 0.4)
    x = np.sin(2 * np.pi * freq * t + index * np.sin(2 * np.pi * freq * 3.5 * t)) * np.exp(-t / 0.9)
    x += 0.35 * np.sin(2 * np.pi * freq * 2 * t) * np.exp(-t / 0.5)
    return normalize(polish(x, [Reverb(room_size=0.7, wet_level=0.3, dry_level=0.7)]), -2.0)


def ks_pluck(freq: float, seconds: float = 1.0, damping: float = 0.996) -> np.ndarray:
    """Karplus-Strong string: a noise burst circulating in a delay line with a low-pass in the loop."""
    n = int(seconds * SR)
    period = int(round(SR / freq))
    buf = RNG.uniform(-1, 1, period)
    out = np.empty(n)
    for i in range(n):
        out[i] = buf[i % period]
        nxt = buf[(i + 1) % period]
        buf[i % period] = damping * 0.5 * (buf[i % period] + nxt)
    return normalize(polish(out * adsr(n, 0.001, 0.1, 1.0, 0.1), [Reverb(room_size=0.3, wet_level=0.15, dry_level=0.85)]), -2.0)


def pad(freq: float, seconds: float = 3.0) -> np.ndarray:
    n = int(seconds * SR)
    t = tvec(n)
    voices = [bl_saw(freq * 2 ** (c / 1200), n, ph) for c, ph in ((-14, 0.0), (-6, 0.2), (0, 0.4), (7, 0.6), (13, 0.8))]
    src = sum(voices) / 5 + 0.25 * bl_saw(freq * 2, n, 0.1) + 0.3 * np.sin(2 * np.pi * freq * t)
    cutoff = 350 + 1700 * (1 - np.exp(-t / 0.9))
    x = swept_filter(src, cutoff, 0.9) * adsr(n, 0.3, 0.5, 0.8, 0.6)
    return stereo_polish(normalize(x, -4.0), [Chorus(rate_hz=0.35, depth=0.28, mix=0.45), Reverb(room_size=0.6, wet_level=0.25, dry_level=0.75)],
                         [Chorus(rate_hz=0.41, depth=0.28, mix=0.45, centre_delay_ms=9.0), Reverb(room_size=0.6, wet_level=0.25, dry_level=0.75)])


def vowel_pad(freq: float, seconds: float = 3.5) -> np.ndarray:
    """Saw + sub through three formant band-passes morphing oo -> ah: a synthetic choir."""
    n = int(seconds * SR)
    t = tvec(n)
    src = bl_saw(freq, n) + 0.5 * bl_saw(freq * 1.005, n, 0.3) + 0.4 * np.sin(2 * np.pi * freq * t)
    morph = 0.5 - 0.5 * np.cos(np.pi * np.clip(t / seconds, 0, 1))
    oo, ah = (300, 870, 2240), (700, 1220, 2600)
    x = np.zeros(n)
    for f_oo, f_ah, gain in zip(oo, ah, (1.0, 0.7, 0.35)):
        centre = f_oo * (f_ah / f_oo) ** morph
        x += gain * swept_filter(src, centre, 6.0, kind="bandpass", stages=1)
    x = x * adsr(n, 0.35, 0.4, 0.85, 0.7)
    return stereo_polish(normalize(x, -4.0), [Chorus(rate_hz=0.25, depth=0.2, mix=0.35), Reverb(room_size=0.75, wet_level=0.3, dry_level=0.7)],
                         [Chorus(rate_hz=0.29, depth=0.2, mix=0.35, centre_delay_ms=8.0), Reverb(room_size=0.75, wet_level=0.3, dry_level=0.7)])


def vox_chop(freq: float, seconds: float = 0.45) -> np.ndarray:
    n = int(seconds * SR)
    t = tvec(n)
    src = bl_saw(freq, n) + 0.4 * np.sin(2 * np.pi * freq * t)
    x = sum(g * swept_filter(src, np.full(n, f), 7.0, kind="bandpass", stages=1) for f, g in ((650, 1.0), (1080, 0.6), (2650, 0.3)))
    return normalize(polish(x * adsr(n, 0.01, 0.1, 0.8, 0.12), [Reverb(room_size=0.5, wet_level=0.25, dry_level=0.75)]), -2.0)


# ----------------------------------------------------------------------------- fx

def riser(bars: int, bpm: float) -> np.ndarray:
    n = int(round(bars * 240 / bpm * SR))
    t = tvec(n)
    u = t / t[-1]
    nz = swept_filter(noise(n), 200 * (9000 / 200) ** u, 0.8, kind="lowpass", stages=1)
    nz = swept_filter(nz, 150 + 3000 * u ** 2, 0.7, kind="highpass", stages=1)
    tone = np.sin(2 * np.pi * np.cumsum(180 * (1800 / 180) ** u) / SR) * 0.4
    x = (nz * 0.8 + tone) * (0.08 + 0.92 * u ** 2.2)
    return normalize(polish(x, [Reverb(room_size=0.5, wet_level=0.2, dry_level=0.8)]), -1.5)


def impact() -> np.ndarray:
    n = int(2.6 * SR)
    t = tvec(n)
    boom = np.sin(2 * np.pi * np.cumsum(38 + 50 * np.exp(-t / 0.08)) / SR) * np.exp(-t / 0.7)
    burst = lp(noise(n), 2500) * np.exp(-t / 0.25)
    x = drive(boom * 1.2 + burst * 0.8, 1.6)
    return normalize(polish(x, [Reverb(room_size=0.92, wet_level=0.45, dry_level=0.6)]), -0.8)


def crackle(bars: int, bpm: float) -> np.ndarray:
    n = int(round(bars * 240 / bpm * SR))
    x = lp(noise(n), 3000) * 0.02
    clicks = RNG.integers(0, n, 60)
    for c in clicks:
        k = int(0.0012 * SR)
        x[c:c + k] += RNG.uniform(0.1, 0.5) * noise(k) * np.exp(-tvec(k) / 0.0003)
    return normalize(hp(x, 300), -12.0)


# ----------------------------------------------------------------------------- the set

def render_set(bpm: float, out_dir: str) -> list[dict]:
    os.makedirs(out_dir, exist_ok=True)
    E1, G1, A1, B1, D2, E2 = (midi_hz(m) for m in (28, 31, 33, 35, 38, 40))
    items = [
        ("kick punch", kick(True), "drums", None, ["kick", "punchy"]), ("kick deep", kick(False), "drums", None, ["kick", "sub"]),
        ("snare", snare(), "drums", None, ["snare"]), ("clap", clap(), "drums", None, ["clap"]),
        ("hat closed", hat_closed(), "drums", None, ["hat_closed"]), ("hat open", hat_open(), "drums", None, ["hat_open"]),
        ("ride", ride(), "drums", None, ["ride"]), ("shaker", shaker(), "drums", None, ["shaker"]), ("rim", rim(), "drums", None, ["rim"]),
        ("tom", tom(), "drums", None, ["tom"]), ("zap", zap(), "drums", None, ["perc", "fx"]),
        ("reese E1 wub8", reese_loop(E1, 1, bpm, 2), "bass", 28, ["reese", "wobble", "loop", "E1", "1/8"]),
        ("reese E1 wub16", reese_loop(E1, 1, bpm, 1, lo_hz=140), "bass", 28, ["reese", "wobble", "loop", "E1", "1/16"]),
        ("reese E1 wub4", reese_loop(E1, 1, bpm, 4, shape="saw_down"), "bass", 28, ["reese", "wobble", "loop", "E1", "1/4"]),
        ("reese E1 trip", reese_loop(E1, 1, bpm, 4 / 3), "bass", 28, ["reese", "wobble", "loop", "E1", "8T"]),
        ("reese E1 stutter", reese_loop(E1, 1, bpm, 1, pattern=[1, 1, 0.1, 1, 1, 0.1, 1, 0.1, 1, 1, 0.1, 1, 1, 0.1, 1, 1]), "bass", 28, ["reese", "wobble", "loop", "E1", "stutter"]),
        ("reese G1 wub8", reese_loop(G1, 1, bpm, 2), "bass", 31, ["reese", "wobble", "loop", "G1"]),
        ("reese A1 wub8", reese_loop(A1, 1, bpm, 2), "bass", 33, ["reese", "wobble", "loop", "A1"]),
        ("reese B1 wub8", reese_loop(B1, 1, bpm, 2), "bass", 35, ["reese", "wobble", "loop", "B1"]),
        ("reese D2 wub8", reese_loop(D2, 1, bpm, 2), "bass", 38, ["reese", "wobble", "loop", "D2"]),
        ("growl E1 wub8", growl_loop(E1, 1, bpm, 2), "bass", 28, ["growl", "loop", "E1"]),
        ("growl E1 wub4", growl_loop(E1, 1, bpm, 4), "bass", 28, ["growl", "loop", "E1", "1/4"]),
        ("growl G1 wub8", growl_loop(G1, 1, bpm, 2), "bass", 31, ["growl", "loop", "G1"]),
        ("sub E1", sub(E1), "bass", 28, ["sub", "E1"]), ("bass pluck E2", bass_pluck(E2), "bass", 40, ["bass", "pluck", "E2"]),
        ("epiano C4", epiano(midi_hz(60)), "keys", 60, ["keys", "piano", "fm"]), ("bell C5", bell(midi_hz(72)), "keys", 72, ["bell", "fm"]),
        ("string pluck A3", ks_pluck(midi_hz(57)), "keys", 57, ["pluck", "string"]),
        ("pad C3", pad(midi_hz(48)), "pad", 48, ["pad", "saw"]), ("choir A2", vowel_pad(midi_hz(45)), "pad", 45, ["pad", "vocal", "choir"]),
        ("vox E4", vox_chop(midi_hz(64)), "vocal", 64, ["vocal", "chop"]),
        ("riser 2 bars", riser(2, bpm), "fx", None, ["riser", "fx", "2bars"]), ("impact", impact(), "fx", None, ["impact", "fx"]),
        ("crackle 4 bars", crackle(4, bpm), "fx", None, ["texture", "loop", "4bars"]),
    ]
    manifest = []
    for name, audio, kind, note, tags in items:
        path = os.path.join(out_dir, name.replace(" ", "_").replace("/", "") + ".wav")
        data = audio if audio.ndim == 2 else audio[:, None]
        sf.write(path, np.clip(data, -1, 1).astype(np.float32), SR, subtype="PCM_16")
        frames = len(data)
        manifest.append({"name": name, "path": path, "kind": kind, "midi_note": note, "tags": tags, "frames": frames, "seconds": round(frames / SR, 3),
                         "channels": data.shape[1]})
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump({"bpm": bpm, "samplerate": SR, "items": manifest}, f, indent=1)
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bpm", type=float, default=174.0)
    ap.add_argument("--out", default=os.path.expanduser("~/Music/ko-bridge/library/sets/dnb-174"))
    args = ap.parse_args()
    import time
    t0 = time.time()
    items = render_set(args.bpm, os.path.expanduser(args.out))
    total = sum(i["frames"] * 2 * i["channels"] for i in items)
    print(f"{len(items)} sounds, {total / 1e6:.2f} MB of PCM, rendered in {time.time() - t0:.1f} s -> {args.out}")
    for i in items:
        print(f"  {i['name']:18s} {i['seconds']:6.3f} s {'st' if i['channels'] == 2 else '  '} {i['kind']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
