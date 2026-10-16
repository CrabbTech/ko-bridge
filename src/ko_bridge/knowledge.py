"""What a model needs to know to wield the EP-133 through ko-bridge: guides and exact human key presses."""
from __future__ import annotations

QUICKSTART = """ko-bridge quickstart: a beat on the EP-133 in five moves.

1. get_status: the device must show midi and audio connected. Read midi_clock; play_pattern needs "in" (setting 101),
   which only the human can set: human_steps("midi_clock_in").
2. list_sounds: the shared library (slots 1-999, 128 MB). Factory banks: kicks 1-99, snares 100-199, hats 200-299,
   perc 300-399, bass 400-499, melodic 500-599. User sounds live above 600 but the range is NOT free by default:
   the list shows what is occupied and the first free slots. upload_sound puts a WAV into a free slot (byte-verified).
3. describe_project(n): what slot n holds. Pick an expendable slot (ask the human; get_status shows one if declared).
4. write_project(n, spec): pads (group A-D, keypad label "." "0" "enter" "1".."9", slot, trim, root, release, play
   mode), patterns (group, number, bars, events by 16th step or tick, fader lanes), the scene, BPM, FX. The bridge
   checkpoints the slot, compiles your spec over its current archive, writes, reads back, compares and reloads. The
   whole cycle is about 13 s for a 60 KB project. write_pattern and assign_pads are the short forms.
5. play_pattern(bpm, bars): MIDI START + clock from the Mac, recording the USB audio and the device's own note output;
   the result grades onsets against the grid (expected within ~4 ms, jitter well under 1 ms). play_notes plays pads
   live (3.5 ms latency). Listen with get_take / the WAV path.

Rules the bridge enforces: every write is checkpointed (restore_checkpoint undoes it); sound slots are never
overwritten or deleted without the acknowledgement string; factory slots 1-599 are protected; system settings are
never touched (there is no protocol for them); one FILE operation at a time (long ones run as jobs: poll job_status)."""

DEVICE = """EP-133 K.O. II as verified on Tyler's unit (128 MB edition, OS 2.5.x) on 2026-10-16; see docs/spike-results.md.

Transport: USB MIDI (ports "EP-133") carries notes, clock and the TE SysEx FILE protocol; USB audio (OS 2.5+) is the
stereo main mix at 48 kHz on the Mac side (46,875 Hz inside the device). No mass-storage mode.
Pads: notes 36-83 trigger the 48 pads of the ACTIVE project: group A 36-47, B 48-59, C 60-71, D 72-83; inside a group
the keypad order is . 0 enter 1 2 3 4 5 6 7 8 9. The device receives on all channels by default and sends on channel 1.
Latency: note-on to audio 3.5 ms (0.3 ms jitter). Pattern hits under Mac clock: within ~4 ms of the grid, jitter 0.4 ms.
Pitch: an event's note byte is pitch relative to the pad's root byte (default root 60); root = pad note gives native
pitch, and any sound becomes a melody instrument by varying the note byte (KEYS mode by hand, semitones/concert events
here). A 26,250 Hz sample plays correctly (the device resamples per slot).
Fader lanes: level pitch time lowpass highpass fx attack release pan tune velocity modulation (ids 0-11), values latch
and SLEW: a change is not complete within ~60 ms, so place it ahead of the hit it should shape. The group FX send gates
the master effect entirely.
Project archive: TAR with 48 pad records (27 B here), pattern members (4-byte header + 8-byte records, 96 PPQN),
scenes (712 B, cursor = the scene external START plays), settings (224 B: BPM, fader bases, assignment; never fabricated,
only patched), fx_settings (160 B). Writes replace the whole archive; the bridge compiles your spec over the current one.
Throughput: sound upload ~24 KB/s, project write ~11 KB/s (+2 s fixed), reads ~25-29 KB/s; a 2 s mono chop uploads in
8 s, a 12-chop crate in ~1.5 min (3 min verified); a full 52 MB library backup takes 29 min.
After select_project or a write's reload, wait about 6 s before playing: the sequencer settles, and once the USB audio
output was caught stuck on one repeating 49-sample packet (constant -9 dBFS buzz) when a pad was hit 3 s after a switch;
takes carry a warning when that happens.
Not reachable from the Mac: every system setting (MIDI clock mode, channel, USB audio input mode, record sample rate,
swing...), so human_steps gives the exact key presses. Program Change is unmapped since OS 2.0.2: sounds and patterns
are chosen by writing the project, not by MIDI.
Hazards (epsysex docs, honoured here): pattern header byte 0 and pad byte 0 must be 0; a defined scene must reference a
pattern for all four groups (use an empty pattern for silence); automation values 0..32767; never fabricate settings;
never run two FILE sessions at once; an unsolicited debug SysEx frame means stop and power-cycle."""

PROJECT_FORMAT = """write_project spec (JSON):
{
 "bpm": 120,                                   // optional; patches the settings member (a device-written template is used if the slot has none)
 "pads": [ {"group": "C", "pad": ".", "slot": 600, "trim_end": null, "release": 255, "play_mode": "oneshot",
            "root_note": null, "amplitude": 100, "pitch": 0, "pan": 0, "attack": 0, "time_mode": "off", "choke": false} ],
 "patterns": [ {"group": "C", "number": 1, "bars": 4,
                "events": [ {"pad": ".", "step": 0, "velocity": 120}, {"pad": "0", "step": 8, "velocity": 100, "note": 63} ],
                "lanes": [ {"parameter": "lowpass", "points": [ {"step": 0, "value": 1.0}, {"step": 16, "value": 0.4} ]} ]} ],
 "scene": {"patterns": {"C": 1}, "time_signature": [4, 4]},   // optional; defaults to the written patterns, other groups keep scene 1
 "fader_bases": [ {"group": "C", "parameter": "fx", "value": 0.3} ],
 "fx": {"type": "reverb", "x": 0.4, "y": 0.5, "output_compressor_drive": 0.2}
}
step = 16th notes from the pattern start (24 ticks); tick is the alternative (96 per beat, 384 per bar). trim_end is in
frames at the sample's native rate; null means the whole sample (looked up on the device).
Melody on any sound (what KEYS mode does by hand): give an event "semitones" (offset from the sample's native pitch),
"concert" (a note name such as "E3" the sound should actually sound at, resolved through the library's measured pitch of
that sample; search_sounds shows each sound's pitch), or the raw "note" byte (the pad's root = native). Chords are the
same idea on a chord sample: "liquid chord em7" at semitones -5 is Bm7. Use play_mode "key" with a release below 255 when
note lengths should matter.
Arrangements: "scenes": [ {"patterns": {"A": 1, "B": 1}}, {"patterns": {"A": 2}} ] writes scenes 1..N (groups left out keep
their current reference, missing patterns become empty members), "song": [1, 2, 1] is the order the device's own song
mode plays (hold MAIN + ENTER for the song list editor, SHIFT+PLAY plays from the start), "current_scene" is the scene the
Mac's START plays. Unmentioned pads, patterns and members survive."""

HUMAN_STEPS = {
    "system_settings": "System settings: press SHIFT+ERASE. Navigate with - and + and ENTER, or type a setting's code on the pads (e.g. 1 0 1); the display shows the new value briefly. SHIFT+ENTER goes back a page; MAIN returns to the normal screen.",
    "midi_clock_in": "MIDI clock receive (needed for play_pattern): SHIFT+ERASE, type 1 0 1 on the pads; the display flashes IN; press MAIN. (Observed 2026-10-16: no ENTER needed.) Then tell the bridge with set_config(midi_clock=\"in\").",
    "midi_clock_out": "MIDI clock send: SHIFT+ERASE, type 1 0 2, MAIN. The device then clocks the Mac; play_pattern cannot start the sequencer in this mode.",
    "midi_clock_off": "MIDI clock off (TE default): SHIFT+ERASE, type 1 0 0, MAIN.",
    "midi_channel": "MIDI channel: SHIFT+ERASE, type 1 1 0 for receive-all/send-1 (default), 1 1 1 .. 1 2 6 for channels 1-16 in and out, 1 2 7 to send only pads with a channel assigned in sound edit; MAIN.",
    "usb_audio_input": "USB audio input mode (what the sampler records from USB): SHIFT+ERASE, 5 1 0 sampler only (default) or 5 1 1 sampler + live monitoring; MAIN. USB audio OUT to the Mac needs no setting.",
    "record_sample_rate": "Recording sample rate: SHIFT+ERASE, 5 0 0 = 26,250 Hz, 5 0 1 = 32,000 Hz, 5 0 2 = 46,875 Hz (default); MAIN. Imported sounds keep their own rate.",
    "select_project": "Select a project: hold MAIN and press pad 1-9 (the bridge can also do this: select_project).",
    "select_group": "Select a group: press GROUP A, B, C or D. Hold a group button to pick its pattern.",
    "sound_edit": "Sound edit for the selected pad: SHIFT+SOUND. Hold SHIFT+SOUND two seconds to save edits.",
    "sample_mode": "Sampling on the device: press SAMPLE (LEDs blink), hold a pad to record, +/- to choose mic/line/resample/USB input, MAIN to exit. Hold SOUND, type a slot number, ENTER, then SAMPLE to sample into a chosen slot.",
    "play_stop": "Transport: PLAY starts, STOP stops (also MIDI STOP from the Mac chokes voices). RECORD arms recording of pads into the pattern.",
    "song_mode": "Song mode (plays the song order the bridge wrote): hold MAIN and press ENTER to open the song list editor; - and + move between song positions; SHIFT+PLAY plays the song from the start, PLAY from the current position; STOP stops; MAIN returns to the normal screen. The device's own transport runs it, so MIDI clock must be internal (SHIFT+ERASE, 1 0 0, MAIN) unless the Mac is sending clock.",
    "commit": "Commit (SHIFT+MAIN) duplicates the current scene into the next one so you can vary it; the bridge writes scenes directly, so commit is only needed when composing by hand.",
    "keys_mode": "Keys mode: select a pad, press KEYS; the 12 pads play that sound chromatically. Hold KEYS and press - or + for the octave; hold KEYS and press a pad to transpose the scale. Record notes with RECORD + pad (step) or RECORD then PLAY (live). The bridge writes the same thing as events with semitones or concert pitches.",
    "power_cycle": "If the display shows an error or the bridge reports a debug SysEx frame: stop everything, hold the power switch off, wait, power on, then run get_status and describe_project before any write.",
    "never": "Never: SHIFT+ERASE while powering on (formats the drive and erases all sounds). The bridge never writes settings and never formats.",
}

GUIDES = {"quickstart": QUICKSTART, "device": DEVICE, "project": PROJECT_FORMAT,
          "human": "Topics for human_steps: " + ", ".join(sorted(HUMAN_STEPS))}


def guide(topic: str) -> str:
    return GUIDES.get(topic, "unknown guide; known: " + ", ".join(sorted(GUIDES)))


def human_steps(topic: str) -> str:
    return HUMAN_STEPS.get(topic, "unknown topic; known: " + ", ".join(sorted(HUMAN_STEPS)))
