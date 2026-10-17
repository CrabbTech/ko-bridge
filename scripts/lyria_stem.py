#!/usr/bin/env python3
"""Generate ONE instrumental loop with Lyria 3.5 (a paid call, about $0.08) and print where it landed.

    uv run python scripts/lyria_stem.py                                   # the default soul prompt, 90 BPM, 2 bars
    uv run python scripts/lyria_stem.py --prompt "..." --bpm 68 --bars 4 [--keep-whole]

GEMINI_API_KEY comes from ~/Music/ko-bridge/secrets.env (or op-bridge's) through config.load_secrets; the key is never
printed. The whole generation is kept as <stem>-full.wav under ~/Music/ko-bridge/library/wav and the printed wav_path
holds the first `bars` bars from the first onset, ready for chop_sample or scripts/lyria_chop_render.py. When Lyria is
unavailable or fails the error is printed and nothing is synthesized in its place.
"""
from __future__ import annotations

import argparse
import json
import sys

import soundfile as sf

from ko_bridge import analysis as an
from ko_bridge import gemini_producer as GP

DEFAULT_PROMPT = ("Warm 1970s soul instrumental for sampling: Rhodes electric piano chords, soft muted horn stabs, "
                  "round electric bass, no drums, dusty vinyl warmth, C minor")


def main() -> int:
    ap = argparse.ArgumentParser(description="One Lyria 3.5 loop (paid) for chopping on the EP-133")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--bpm", type=float, default=90.0)
    ap.add_argument("--bars", type=int, default=2)
    ap.add_argument("--keep-whole", action="store_true", help="do not trim the loop to --bars from the first onset (the whole generation is kept as -full.wav anyway)")
    a = ap.parse_args()
    try:
        res = GP.generate_lyria_stem(a.prompt, bpm=a.bpm, bars=a.bars, trim_to_bars=not a.keep_whole)
    except (GP.GeminiUnavailable, GP.LyriaError) as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 1
    for label in ("full_wav_path", "wav_path"):
        x, sr = sf.read(res[label], dtype="float32", always_2d=True)
        m = an.mono(x)
        onsets = an.all_onsets(an.transients(m, sr), sr)
        res[label.replace("wav_path", "audio")] = {"seconds": round(len(x) / sr, 3), "peak_dbfs": round(an.peak_db(m), 1), "rms_dbfs": round(an.rms_db(m), 1),
                                                   "onsets": len(onsets), "bands_dbfs": an.spectral_mix_report(x, sr).get("bands_dbfs")}
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
