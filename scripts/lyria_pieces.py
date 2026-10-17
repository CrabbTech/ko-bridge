#!/usr/bin/env python3
"""Render a set of Lyria 3.5 pieces for a song (paid: about $0.08 per piece) and write a manifest for the chop step.

    uv run python scripts/lyria_pieces.py --set triphop                 # every piece of the set (4 calls)
    uv run python scripts/lyria_pieces.py --set triphop --only strings vocal
    uv run python scripts/lyria_pieces.py --list

Each piece is one Lyria call. The whole generation (about a minute of music) is kept as a 16-bit WAV under
~/Music/ko-bridge/library/wav, nothing is trimmed here, and a manifest JSON lands in ~/Music/ko-bridge/notes for the loop
picking and chopping that follow on the Mac. GEMINI_API_KEY comes from the secrets file through config.load_secrets and is
never printed; a failed piece is reported and the next one is still attempted.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import soundfile as sf

from ko_bridge import analysis as an
from ko_bridge import gemini_producer as GP

WAV_DIR = os.path.expanduser("~/Music/ko-bridge/library/wav")
NOTES = os.path.expanduser("~/Music/ko-bridge/notes")

SETS: dict[str, dict] = {
    "triphop": {
        "bpm": 86.0,
        "key": "D minor",
        "about": "Bristol trip hop around 86 BPM in D minor: a string-and-Rhodes bed, a wordless voice, a dub guitar riff and a muted trumpet, "
                 "all to be screwed and chopped on the EP-133 over its own drums and sub bass",
        "pieces": [
            {"name": "strings", "bars": 4, "instrumental": True,
             "prompt": "1990s Bristol trip hop, cinematic and melancholic: a slow string ensemble swelling through minor chords with a dusty "
                       "Rhodes electric piano underneath, vinyl crackle and tape warmth, no drums, no percussion, no bass guitar, "
                       "D minor, 86 BPM, a four-bar chord cycle that starts on the downbeat and repeats"},
            {"name": "vocal", "bars": 4, "instrumental": False,
             "prompt": "A haunting wordless female voice, a cappella: slow soulful hums, sighs and long held notes with a lot of reverb, "
                       "no lyrics, no words, no instruments, no drums, intimate and close, D minor, 86 BPM, phrases that begin on the downbeat"},
            {"name": "guitar", "bars": 2, "instrumental": True,
             "prompt": "Dark dub-influenced trip hop: a sparse tremolo electric guitar riff with spring reverb and a deep round electric "
                       "bass motif, lots of space and silence between notes, slow and heavy, no drums, no percussion, D minor, 86 BPM, "
                       "a two-bar riff that repeats"},
            {"name": "trumpet", "bars": 4, "instrumental": True,
             "prompt": "A lonely muted trumpet playing a smoky, breathy, slow melody with silence between short phrases, late-night film noir, "
                       "nothing else playing, no drums, no bass, no chords, D minor, 86 BPM, phrases that start on the downbeat"},
        ],
    },
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Render a set of Lyria 3.5 pieces (paid) for a ko-bridge song")
    ap.add_argument("--set", default="triphop", choices=sorted(SETS))
    ap.add_argument("--only", nargs="*", default=None, help="piece names to render (default: all)")
    ap.add_argument("--list", action="store_true", help="print the set and exit without calling anything")
    a = ap.parse_args()
    s = SETS[a.set]
    pieces = [p for p in s["pieces"] if not a.only or p["name"] in a.only]
    if a.list or not pieces:
        print(json.dumps({"set": a.set, "bpm": s["bpm"], "key": s["key"], "about": s["about"],
                          "pieces": [{k: p[k] for k in ("name", "bars", "instrumental", "prompt")} for p in pieces]}, indent=1))
        return 0
    os.makedirs(WAV_DIR, exist_ok=True)
    os.makedirs(NOTES, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    manifest = {"set": a.set, "bpm": s["bpm"], "key": s["key"], "about": s["about"], "time": stamp, "pieces": []}
    failures = 0
    for p in pieces:
        path = os.path.join(WAV_DIR, f"lyria-{a.set}-{p['name']}-{stamp}.wav")
        print(f"rendering {p['name']} ...", file=sys.stderr)
        try:
            res = GP.generate_lyria_stem(p["prompt"], bpm=s["bpm"], bars=p["bars"], out_wav_path=path,
                                         instrumental=p["instrumental"], trim_to_bars=False)
        except Exception as e:   # the producer layer keeps the key out of its error text
            failures += 1
            manifest["pieces"].append({"name": p["name"], "error": f"{type(e).__name__}: {e}", "prompt": p["prompt"]})
            print(f"  {p['name']}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        wav = res.get("full_wav_path") or res.get("wav_path")
        x, sr = sf.read(wav, dtype="float32", always_2d=True)
        m = an.mono(x)
        onsets = an.all_onsets(an.transients(m, sr), sr)
        entry = {"name": p["name"], "wav": wav, "model": res.get("model"), "seconds": round(len(x) / sr, 2), "samplerate": int(sr),
                 "channels": int(x.shape[1]), "peak_dbfs": round(an.peak_db(m), 1), "rms_dbfs": round(an.rms_db(m), 1),
                 "onsets": len(onsets), "bands_dbfs": an.spectral_mix_report(x, sr).get("bands_dbfs"),
                 "lyria_text": res.get("text"), "bars_requested": p["bars"], "instrumental": p["instrumental"], "prompt": p["prompt"]}
        manifest["pieces"].append(entry)
        print(f"  {p['name']}: {entry['seconds']} s, peak {entry['peak_dbfs']} dBFS, rms {entry['rms_dbfs']} dBFS, {entry['onsets']} onsets -> {wav}", file=sys.stderr)
    out = os.path.join(NOTES, f"lyria-{a.set}-{stamp}.json")
    with open(out, "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps({"manifest": out, "rendered": len(pieces) - failures, "failed": failures,
                      "pieces": [{k: e.get(k) for k in ("name", "wav", "seconds", "peak_dbfs", "rms_dbfs", "onsets", "error")} for e in manifest["pieces"]]}, indent=1))
    return 1 if failures == len(pieces) else 0


if __name__ == "__main__":
    sys.exit(main())
