"""Houston Chop & Screw + Sample-Chopped Studio workflow on the EP-133 K.O. II.

Demonstrates the upgraded musical pipeline in `ko-bridge`:
1. Generates (or loads) a soulful stereo loop via `gemini_producer.generate_lyria_stem`.
2. Resamples it down -3.5 semitones (`chopper.prepare_chopped_audio`) so tempo and formants slow down
   authentically like a turntable / SP-1200.
3. Slices it into 12 chops (`chopper.detect_chop_slices` in `"hybrid"` mode: 8 structural beat-grid chops
   snapped to transients on pads `1..8`, 3 transient accents on `9, 0, .`, and 1 reversed `REV` pickup
   on `enter`) referencing a **single shared sample slot** via `trim_start` and `trim_end`.
4. Compiles a multi-scene arrangement with `gemini_producer.compile_beat_with_gemini` + `groove.py`
   (4-tick 24-PPQN MIDI clock alignment, 58% MPC swing, +4 tick lazy snare pocket, DJ Screw 1/32nd
   crossfader double-ups, and slew-compensated filter automation).

Run offline dry-run:
    uv run python scripts/chop_and_screw.py --dry-run
Run on connected EP-133 (project 9; the loop goes to the first free sample slot from 600):
    uv run python scripts/chop_and_screw.py --project 9
"""
from __future__ import annotations

import argparse
import json

from ko_bridge import chopper as CH
from ko_bridge import gemini_producer as GP
from ko_bridge.project import PadSpec


# Stock EP-133 factory drum & bass slots for Group A & B
DRUM_PADS = [
    PadSpec(group="A", pad=".", slot=10, amplitude=118),   # punchy kick
    PadSpec(group="A", pad="0", slot=111, amplitude=114),  # crisp snare
    PadSpec(group="A", pad="1", slot=205, amplitude=92),   # closed hat
    PadSpec(group="A", pad="2", slot=225, amplitude=88),   # open hat
    PadSpec(group="A", pad="3", slot=145, amplitude=102),  # clap / rim layer
]

BASS_PADS = [
    PadSpec(group="B", pad=".", slot=403, amplitude=112, play_mode="legato", choke=False),  # deep sub / 808; choke is the device's one global mute group, so a choke bass would cut the chops
]


def build_chop_and_screw_project(
    prompt: str = "Houston DJ Screw chopped soul Rhodes & horn loop at 68 BPM with heavy double-up stutters on bar 2 and 4 and reversed pickup on beat 4.4",
    bpm: float = 68.0,
    bars: int = 4,
    screw_semitones: float = -3.5,
    chop_slot: int = 605,
) -> dict:
    stem = GP.generate_lyria_stem(prompt=prompt, bpm=bpm, bars=bars, samplerate=46875)
    audio, pcm, audio_info = CH.prepare_chopped_audio(
        stem["wav_path"],
        samplerate=46875,
        mono=False,
        normalize_dbfs=-1.0,
        screw_semitones=screw_semitones,
        source_bpm=bpm,
    )
    slices = CH.detect_chop_slices(audio, 46875, num_slices=12, mode="hybrid", reverse_last_pad=True)
    chop_pads = CH.build_shared_slot_chop_pads(slices, slot=chop_slot, group="C", sample_bpm=bpm)

    plan, model_used = GP.compile_beat_with_gemini(prompt=prompt, bpm=bpm, bars=bars)
    spec, clock_auto = GP.plan_to_project_spec(
        plan,
        chop_pads=chop_pads,
        drum_pads=DRUM_PADS,
        bass_pads=BASS_PADS,
        chop_group="C",
        build_two_scenes=True,
    )
    return {
        "model": model_used,
        "stem": stem,
        "audio": audio,
        "audio_info": audio_info,
        "pcm": pcm,
        "slice_objs": slices,
        "slices": [s.model_dump() for s in slices],
        "plan": plan,
        "chop_slot": chop_slot,
        "spec": spec,
        "clock_automation": clock_auto,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Houston Chop & Screw session for the EP-133 K.O. II")
    parser.add_argument("--prompt", default="Houston DJ Screw chopped soul Rhodes & horn beat at 68 BPM with double-up stutters on bar 2 and 4")
    parser.add_argument("--bpm", type=float, default=68.0)
    parser.add_argument("--screw", type=float, default=-3.5, help="Turntable pitch/speed shift in semitones")
    parser.add_argument("--slot", type=int, default=None, help="Sample slot for the chopped loop (default: the first free slot from 600; an occupied slot needs --overwrite)")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting the sound already in --slot (otherwise an occupied slot is refused)")
    parser.add_argument("--allow-synth", action="store_true", help="Upload the built-in synthesized fallback stem when Lyria is unavailable (the library rule is real samples only)")
    parser.add_argument("--project", type=int, default=9, help="EP-133 project slot (1-9)")
    parser.add_argument("--dry-run", action="store_true", help="Compile offline without writing to the hardware")
    args = parser.parse_args()

    built = build_chop_and_screw_project(
        prompt=args.prompt,
        bpm=args.bpm,
        screw_semitones=args.screw,
        chop_slot=args.slot or 600,   # placeholder when no slot is given; the pads are rebuilt for the slot actually used
    )

    if args.dry_run:
        summary = {
            "model": built["model"],
            "plan_title": built["plan"].title,
            "stem": built["stem"],
            "audio_info": built["audio_info"],
            "slices_count": len(built["slices"]),
            "first_three_slices": built["slices"][:3],
            "reverse_pickup_slice": built["slices"][-1],
            "patterns": [
                {"id": p.id, "bars": p.bars, "events": len(p.events), "lanes": [l.parameter for l in p.lanes]}
                for p in built["spec"].patterns
            ],
            "clock_automation_pulses": len(built["clock_automation"]),
        }
        print(json.dumps(summary, indent=2))
        return

    import soundfile as sf

    from ko_bridge import server as srv

    if str(built["stem"].get("source", "")).startswith("fallback") and not args.allow_synth:
        raise SystemExit("the stem is the built-in synthesized fallback (no Lyria access): the library holds real samples only; "
                         "pass --allow-synth to upload it anyway")
    # Upload the PROCESSED audio (screwed, resampled, normalised): the slice frames were measured on it, not on the source stem.
    processed = built["stem"]["wav_path"].replace(".wav", f"-screw{args.screw:+.1f}.wav")
    sf.write(processed, built["audio"], 46875, subtype="PCM_16")
    up = srv.upload_sound(
        processed,
        slot=args.slot,
        name="screw-stem",
        samplerate=46875,
        mono=False,
        acknowledge_overwrite=f"OVERWRITE SOUND {args.slot}" if (args.slot and args.overwrite) else None,
        wait=True,
    )
    if up["slot"] != built["chop_slot"]:
        chop_pads = CH.build_shared_slot_chop_pads(built["slice_objs"], slot=up["slot"], group="C", sample_bpm=args.bpm)
        built["spec"], built["clock_automation"] = GP.plan_to_project_spec(
            built["plan"], chop_pads=chop_pads, drum_pads=DRUM_PADS, bass_pads=BASS_PADS, chop_group="C", build_two_scenes=True)
    wr = srv.write_project(args.project, built["spec"], reload=True, wait=True)
    print(json.dumps({"upload": up["slot"], "project": wr["project"], "identical": wr["identical"]}, indent=2))


if __name__ == "__main__":
    main()
