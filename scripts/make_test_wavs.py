#!/usr/bin/env python3
"""Generate the two spike test samples: mono 16-bit WAVs with an 8 ms noise burst then a decaying sine.

    assets/kob-test-46875.wav   2.0 s at 46,875 Hz, 880 Hz, tau 0.25 s
    assets/kob-test-26250.wav   3.0 s at 26,250 Hz, 440 Hz, tau 0.40 s

The burst gives a broadband attack for onset detection; the tone makes the pad recognisable by ear.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

OUT = Path(__file__).resolve().parents[1] / "assets"


def make(sr: int, seconds: float, hz: float, tau: float, seed: int) -> np.ndarray:
    n = int(sr * seconds)
    t = np.arange(n) / sr
    tone = 0.6 * np.sin(2 * np.pi * hz * t) * np.exp(-t / tau)
    burst_n = int(sr * 0.008)
    rng = np.random.default_rng(seed)
    burst = rng.uniform(-0.9, 0.9, burst_n) * np.linspace(1.0, 0.2, burst_n)
    x = tone.copy()
    x[:burst_n] = np.clip(x[:burst_n] + burst, -0.95, 0.95)
    fade = int(sr * 0.01)
    x[-fade:] *= np.linspace(1.0, 0.0, fade)
    return x.astype(np.float32)


def main() -> int:
    OUT.mkdir(exist_ok=True)
    specs = [(46875, 2.0, 880.0, 0.25, 1, "kob-test-46875.wav"), (26250, 3.0, 440.0, 0.40, 2, "kob-test-26250.wav")]
    for sr, sec, hz, tau, seed, name in specs:
        x = make(sr, sec, hz, tau, seed)
        path = OUT / name
        sf.write(path, x, sr, subtype="PCM_16")
        info = sf.info(path)
        print(f"{path.name}: {info.samplerate} Hz {info.channels} ch {info.subtype} {info.frames} frames {info.duration:.2f} s {path.stat().st_size} B")
    return 0


if __name__ == "__main__":
    sys.exit(main())
