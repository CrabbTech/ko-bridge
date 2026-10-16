# EP-133 K.O. II write-path spike (2026-10-16)

Status: **complete.** Every planned write verified byte-for-byte; no device error, status rejection or unsolicited
debug frame appeared at any point. The one skipped item is the write-during-playback test, skipped at Tyler's decision.

Goal: prove, from this Mac, that an EP-133 K.O. II can be driven the way ko-bridge needs before any UI work: project
round trip over SysEx, sample upload, note-to-audio latency over USB audio, a clocked pattern with a fader lane.
Tools: [epsysex](https://github.com/kmorrill/ep-series-sysex) at commit `891a905c` (MIT) for the FILE protocol,
`ko_bridge.*` for MIDI, audio and the checkpoint-before-write rule. Raw results: `docs/spike/*.json`; captures:
`~/Music/ko-bridge/spike/*.wav`; backups and checkpoints: `~/Music/ko-bridge/backups/`.

## Device

| Item | Value | Source |
|---|---|---|
| MIDI ports | input `EP-133`, output `EP-133` | mido port list |
| USB audio | `EP-133`, 2 in / 2 out, CoreAudio 48,000 Hz | sounddevice |
| Identity reply | `f0 7e 33 06 02 00 20 76 20 00 02 00 00 00 00 00 f7` → TE, family 0x0020, member 2, **version bytes all zero** | MIDI Universal Identity Request |
| Firmware | OS 2.5.x: USB audio is present and the identity version is zeroed, both OS 2.5 behaviours; the exact point release has to be read from the device's own screen (not done) | behaviour, epsysex sysex-protocol.md |
| Sample memory | **128 MB edition** (Tyler); TE guide: "up to 999 samples, or 128 MB". The protocol exposes no total: STAT on `/`, `/sounds`, `/projects` returns flags 0x0e, size 0. 52.26 MB in use before the spike | Tyler, TE guide, STAT |
| Sound library | 610 occupied slots, 52.26 MB; 1-599 factory banks, **700-792, 800-870, 900-978 also occupied**; free before the spike: 600-699, 793-799, 871-899, 979-999. Metadata census: 489 mono 46,875 Hz, 62 stereo 46,875 Hz, 58 mono 48,000 Hz, 1 stereo 44,100 Hz, all `s16` | FILE LIST + metadata |
| Projects | P1 81,920 B (48 pads, 24 pattern members, 17 non-empty, scenes, 126 BPM); P2 57,856 B (48 pads, one empty pattern, scenes, 114 BPM; the active project when the spike began); P3-P5 40,960 B (36 pads, no patterns, no settings member); P6-P9 4,096 B (directory skeleton only) | FILE GET |
| Scratch slot | **P9**, the only slot written (Tyler, via the parent session: P9 over P6; P1-P8 untouched) | Tyler |
| Write during playback | **Skipped at Tyler's decision** ("Skip it" over "Try once on the scratch slot"). The epsysex docs call `--fast-live` opt-in, not safe: it drops the checkpoint and the immediate readback | Tyler, epsysex live-deployment.md |

## Backup (before any write)

`~/Music/ko-bridge/backups/20261016-132738/`: `identity.json`, `projects/P01-P09.tar`, `sounds-list.json`,
`sounds/NNN.pcm` + `NNN.json` (raw PCM plus the slot's 19-field metadata) for all 610 slots, verified against the
listed sizes, `manifest.json`, `stat.json`, `sounds-progress.jsonl`, and `ep133-backup.pak` (41.6 MB ZIP in the Sample
Tool layout, WAVs built from the PCM; acceptance by TE's tool not tested). Every later project write left a checkpoint
in `backups/checkpoints/` (`<time>-P09-before-<label>.tar`). The sample backup took 29.3 min.

## Throughput

| Transfer | Bytes | Seconds | KB/s | Notes |
|---|---:|---:|---:|---|
| Project GET P1 | 81,920 | 3.25 | 24.6 | 324 B GET pages |
| Project GET P2 | 57,856 | 2.19 | 25.8 | |
| Project GET P9 skeleton | 4,096 | 0.19-0.23 | ~20 | session overhead dominates |
| Sound GET, whole library | 52,255,024 | 1,758.5 | 29.0 | 610 slots, per slot 20.8-30.7 KB/s |
| Sound GET slot 600 (readback) | 187,500 | 5.93 | 30.9 | |
| Project PUT P9 ← own skeleton | 4,096 | 2.13 | 1.9 | fixed cost ≈ 2 s: 1.2 s debug-loop preflight + init/open/EOF/re-init |
| Project PUT P9 ← P1 copy | 81,920 | 6.85 | 11.7 | 433 B pages, 6 ms pacing |
| Project PUT P9 ← pads archive | 57,856 | 5.19 | 10.9 | |
| Project PUT P9 ← pattern archive | 60,928 | 5.55 | 10.7 | |
| Sound PUT slot 600 (46,875 Hz, 2.0 s mono) | 187,500 | 7.69 | 23.8 | includes metadata write |
| Sound PUT slot 601 (26,250 Hz, 3.0 s mono) | 157,500 | 6.48 | 23.7 | |
| Reload / activation dance | — | 2.5-2.9 | — | project cycle + readbacks |
| Whole 60 KB project cycle: checkpoint, write, readback, reload | 60,928 | 12.9 | — | 2.22 + 5.55 + 2.45 + 2.70 |

Projection for a 12-chop crate of 2 s mono 46,875 Hz chops (2.25 MB): **about 92 s** of upload at 23.8 KB/s, plus
about 71 s if every chop is read back for verification, plus one ~13 s project cycle to assign the pads: roughly 3 minutes
verified, 1.75 minutes unverified. Sound reads are ~2.5× faster than project writes per byte.

## Latency (MIDI note → USB audio)

Method: input-only `sounddevice` stream on `EP-133` at 48 kHz, 128-frame blocks; the stream's ADC clock is pinned to a
sample index at the first block and a mark is taken right before each note-on (op-bridge's probe method). The test
samples open with an 8 ms noise burst, so onsets are read on a 2.5 kHz high-passed copy.

| Measurement | Result |
|---|---|
| Pad C "." (slot 600, 46,875 Hz), 20 trials, first non-zero sample (trial 1 starts from digital silence) | 2.96 ms |
| Same, burst crossing -20 dBFS, per trial | 3.08-4.21 ms, **mean 3.62 ms, std 0.28 ms, peak-to-peak 1.12 ms** |
| Pad C "0" (slot 601, 26,250 Hz), 10 trials, 2 ms-frame detector | all in the 2-4 ms frame, same as pad "." |
| Full-duplex stream (silence out), 10 trials | input works identically; its clock origin differs by a few ms (onsets read as -2..0 ms), so numbers from a duplex stream are not comparable. **Full duplex is not needed**: the input-only stream delivers audio |
| PortAudio-reported input latency | 88 ms (CoreAudio's buffer estimate; the ADC timestamps are compensated for it) |
| Cross-check of the audio clock base | during pattern playback the device's own MIDI note-out arrives 1.2-2.2 ms after the grid and the audio onset 1.6-2.0 ms after that MIDI arrival, measured on one clock; consistent with the 3.5 ms figure |
| Idle noise floor | digital zero (-240 dBFS); hits peak at -0.03 dBFS without clipping at amplitude 100, velocity 110 |

So the EP-133 renders a MIDI-triggered pad in about 3.5 ms with a third of a millisecond of jitter, about half the
8 ms measured on the OP-1 Field with the same method.

## Pattern timing (4 bars, two pads, one fader lane, Mac clock)

Project written to P9: pads C "." ← slot 600 (root 60) and C "0" ← slot 601 (root 61); pattern C01, 4 bars, 10 note
events (ticks 0, 192, 384, 576, 768, 864, 960, 1152, 1344, 1440 at 96 PPQN, notes 60/61 = pad roots, durations 24)
plus three low-pass fader records (parameter 3: 32767 at tick 0, 14000 at 384, 32767 at 768); empty B01 and D01; scene 1 =
[1,1,1,1] 4/4 with the cursor on scene 1; settings BPM patched to 120 on P2's genuine settings member. Readback identical,
reload OK, 6 s settle, then MIDI START plus 24 PPQN clock for 5 bars at 120 BPM from the Mac (clock send jitter std 0.12 ms).

| Measurement | Result |
|---|---|
| Expected hits found | **12 of 12**, including the two in bar 5 after the pattern looped |
| Audio onset after the grid time, 10 clean fine measurements | 2.96-4.21 ms, mean 3.54 ms, std 0.36 ms (the two low-passed hits fell below the fine detector's threshold; the 2 ms detector found them on time) |
| Jitter relative to the first hit | std 0.36 ms, peak-to-peak 1.25 ms |
| Device MIDI note-out after the grid time | 1.16-2.18 ms, mean 1.53 ms, channel 1, notes 60/61 as authored, velocities as authored |
| Pitch | pad "." hits are 880 Hz, pad "0" hits 440 Hz: both samples play at native pitch with root = event note, and the 26,250 Hz sample is resampled correctly |
| Low-pass lane | hits at 576 and 768 have 30-40 dB less energy above 2.5 kHz than the others; the hit at tick 384 (same tick as the 14000 record) is unfiltered and the hit at 768 (same tick as the re-open record) is still filtered: **the group fader latches but slews**, so a change is not complete within the first ~60 ms and is complete 500 ms later. Place automation ahead of the hit it should shape |
| Silence | none between start and the stop message (0.0-9.99 s of continuous programme) |

## Writes (each with checkpoint, readback and diff)

| # | Target | Bytes | Result |
|---|---|---:|---|
| 1 | P9 ← its own 6-directory skeleton | 4,096 | identical |
| 2 | P9 ← copy of P1's archive, then reload | 81,920 | identical; device switched to P9 |
| 3 | sound slot 600 ← kob-test-46875 PCM + metadata | 187,500 | identical; device filled `sound.*` defaults (amplitude 100, oneshot, root 60, release 255, `sample.end` 93750, crc) |
| 4 | sound slot 601 ← kob-test-26250 PCM + metadata | 157,500 | identical; `samplerate` 26250 kept |
| 5 | P9 ← P2's archive with pads C p10/p11 reassigned (compiler base-archive mode), reload | 57,856 | identical; pad records decode to slots 600/601, trim 93750/78750 frames, roots 60/61 |
| 6 | P9 ← same plus pattern C01, empty B01/D01, scene, BPM 120, reload | 60,928 | identical; played as authored |
| — | fast-live write during playback | — | skipped (Tyler) |

Failed writes: none. Debug frames: none. Status rejections: none. The device was left with P9 holding the test project
and P2 active again (`scripts/restore_active.py 2`); slots 600 and 601 hold the two test samples (delete with
`epsysex` `delete_sound` and the acknowledgement string if unwanted).

## Protocol facts checked against this device

Confirmed (source in parentheses):
- Product byte 0x33, request-id framing, packed7, FILE INIT/GET/PUT/LIST/STAT/METADATA shapes: `epsysex.FileClient` worked unmodified for every operation (sysex-protocol.md, file-protocol.md).
- Identity Reply carries zeroed version bytes on OS 2.5.x (sysex-protocol.md).
- Project write flags 0x06, parent 2000, name "09"; reload by `{"active": fid}` on node 2000 plus group/pad re-selection; a never-activated slot reports no active group or pad (file-protocol.md).
- Native-shaped archives read back byte-exact after a write, including compiler-built archives in base-archive mode (file-protocol.md normalization note).
- The compiler's TAR dialect: zero-change rebuilds of P1 and P2 are byte-identical to the device's archives (project-format.md, tar.py).
- Pattern header `[0, bars, count, 0]` on all 24 device-written members of P1 (project-format.md).
- Scenes: 712 B, header `[0,0,0,0,0,4,4]`, trailer byte 3 = cursor (2 on P1, 1 on P2), song length at byte 11 (project-format.md). External MIDI START plays the cursor scene (coverage.md).
- fx_settings 160 B; sidechain routing words on P1/P2: A = 0x0600 (source pads 10, 11), B/C/D = 0x8000 (destinations), matching the documented bit layout (project-format.md).
- MIDI note → pad map: notes 60/61 trigger `pads/c/p10` (".") and `pads/c/p11` ("0"); event pad bytes `(pad-1)*8` played the right pads (midi-reference.md).
- Event note is emitted as the MIDI note and sets pitch relative to the pad root; root = note gives native pitch (measured-curves.md).
- Device sends on channel 1 (TE guide setting 110; observed).
- MIDI clock in (setting 101) + START + clock from the Mac drives the sequencer; MIDI stop ends playback (midi-reference.md, TE guide).
- Sound upload flags 0x05 with `samplerate`/`channels`/`format`/`name` metadata; byte-exact readback; 26,250 Hz content accepted and played correctly (file-protocol.md).
- All FILE I/O serialized through `/tmp/ep-device.lock`; one process at a time throughout (file-protocol.md).

Contradicted or extended:
- "Slots 600-999 appear free" (ko-bridge brief, memory): **false on this unit**, 700-792, 800-870 and 900-978 held user content. LIST first, always.
- Pad records are **27 bytes** on every device-made archive here (P1, P2, P3 pads), where project-format.md calls 26 B native and 27 B a Sample-Tool artefact. Either OS 2.5.x writes 27 B natively or these projects passed through the Sample Tool; 27 B is what the compiler emits and the device accepted and played.
- Settings member is **224 B** on both P1 and P2 with bytes 220-223 = `[0, 4, 3, 0]`, not the `[0, 1]` tail the docs report for 2.5.1. TE's OS 2.5 notes say scale and key are **saved per project** (whats-new), which answers coverage.md's open "scale/root scope" question in favour of the project; bytes 220-223 are the candidates, not decoded here.
- Fader automation **slews** rather than switching at its tick (see Pattern timing); the docs describe latching only.
- Setting 101 applied as soon as "1 0 1" was typed after SHIFT+ERASE ("IN" flashed); TE's guide says to press ENTER after the code. MAIN returns to the normal screen.
- Memory size is not visible to the protocol (STAT sizes are 0 on directories); TE's guide gives only the 128 MB ceiling.
- The device populates a fresh slot's `sound.*` metadata itself after an upload (amplitude, playmode, root, release, sample.end, crc); the client only needs the stream fields.
- The identity family/member pair is 0x0020 / 2 (not in the docs).

Not tested: Program Change, bank select, CC 12/13, keys mode, song-mode playback, the EP Sample Tool's acceptance of the generated .pak, deleting a slot, the ~6 s post-reload settle window the docs recommend (honoured, not probed).

## Human steps

Only one setting had to be changed at the device; everything else (project switch, pad assignment, sample upload,
playback start/stop) was done from the Mac.

1. Before the pattern test, Tyler set MIDI clock to receive: **SHIFT+ERASE**, then typed **1 0 1** on the pads; the display showed **IN** briefly and returned to the system-settings entry screen (no ENTER was needed). **MAIN** returned the device to its normal screen. The previous value was not recorded; TE's default is 100 (off).
2. Tyler chose to leave MIDI clock on receive (101) after the spike; to restore: SHIFT+ERASE, type the previous code (1 0 0 for off), MAIN.
3. The device must be on its normal screen (MAIN) for MIDI notes and clock to play pads; it was.

## Recommended device-layer API (MCP tools)

1. `get_status` — ports, audio device, active project/group/pad, library count and bytes, lock holder, last backup.
2. `backup_device` — full projects + sounds + metadata backup as a job (about 30 min for 52 MB), or projects only in seconds.
3. `list_sounds` — occupied slots with size, name, rate, channels; the only way to find a free slot.
4. `upload_sound` — WAV in, native PCM out, free slot (or explicit overwrite acknowledgement), byte-verified, returns frames for the pad record.
5. `describe_project` — decoded view of a slot: pads with slots and trims, patterns, scenes, cursor, BPM, FX, sidechain.
6. `write_project` — JSON (pads, patterns, fader lanes, scenes, BPM, FX) compiled over the slot's current archive; checkpoint, write, readback-diff, reload; returns timings.
7. `assign_pads` — slots to group/pad with trim, root, play mode and release (a thin `write_project`).
8. `write_pattern` — one group pattern with notes and fader lanes, plus the scene cursor (a thin `write_project`).
9. `select_project` — the activation dance, later the double-buffer deploy (write an inactive slot, switch at a bar).
10. `play_pattern` — MIDI START + clock at a BPM for N bars (device in clock-in mode), optional USB-audio recording with marks.
11. `play_notes` — timed note events on the pad map for live auditioning and finger-drumming from the model.
12. `record` — input-only USB capture for N seconds to WAV with marks; levels and onsets.
13. `measure_timing` — onsets against an expected grid (the oracle used here) for latency and jitter reports.
14. `human_steps` — exact key presses for the settings the protocol cannot reach (clock 100-102, channel 110-127, USB audio 510/511, sample rate 500-502).
15. `restore_checkpoint` — write a saved TAR back to its slot (undo for 6).
16. `job_status` — progress and result of long operations (backup, crate upload).

## Addendum: server build (same day)

The MCP server (27 tools, `src/ko_bridge/server.py`) and 11 functional tests were written after the spike. Two findings
from running the tests against the device:

- **Playback right after FILE traffic misbehaves.** Twice, when a note or MIDI START followed a FILE session (a project
  GET, or a project switch) within about 3 s, the USB audio output froze on one repeating 49-sample packet (a constant
  -8 dBFS buzz at 48,000/49 ≈ 980 Hz with a DC offset) and, in the START case, the sequencer fired three note-ons at
  t = 0 and nothing more; the following metadata session then timed out. A passive listen showed no debug frames, the
  idle stream returned to digital silence, and a power cycle (Tyler) cleared the state. The epsysex docs already advise a
  ~6 s settle between write/reload and transport; ko-bridge now waits 6 s after **any** FILE session before playing or
  recording, describes the project before that wait, warns when a take's level is constant for 1.5 s, and warns when the
  device's notes all arrive within 50 ms of START. With that rule all tests pass.
- **Latency re-measured by the server's live-note path:** hits arrive 1.8-3.3 ms after the note (high-passed onset above
  -30 dBFS), consistent with the spike's 3.0-4.2 ms burst-crossing figure.

Human steps added: one power cycle (hold the power switch off, wait, power on) after the timed-out session.

## Addendum: first songs, the sample library, song mode (same day, later)

- **A jungle track through the server.** `scripts/jungle.py` (170 BPM, E minor, factory sounds, one 4-bar scene) and
  `scripts/jungle_song.py` (four scenes: intro, drop, breakdown with a melody, second drop; song order 1-2-3-4-2) were
  written to P9 with the normal checkpoint, write, readback, reload cycle (61,440 and 71,680 bytes, identical). Under the
  Mac's clock the device returned exactly the expected note-ons (282 for eight bars of the drop, 78 for four bars of the
  breakdown), transpositions included. Sounds were chosen from TE's display names (read from the backup's slot metadata,
  since FILE LIST only carries node names) and the key was set by measuring the samples' pitch.
- **Companion sample library** (`ko_bridge.library`): 613 sounds indexed in about a second from the backup PCM, with
  measured length, attack, decay, sustain, brightness, pitch and confidence, low/high energy shares; a category from the
  name where TE's names say something (283 sounds), otherwise guessed from features and bank (327, marked "guessed");
  descriptors; free tags; search by words, labels, length, pitch class, bank. Pitch estimation agrees with TE's
  `sound.rootnote` metadata on pitch class for the bass and melodic banks once the autocorrelation search starts past the
  first dip (the first version read 60 Hz subs as 2 kHz and, with an "octave rule", the reese an octave low; for a
  periodic tone the autocorrelation at twice the period is as high as at the period, so no such rule is safe). Events
  can now be written in concert pitch (`concert: "E4"`) and the bridge computes the note byte from the sample's pitch.
- **Fader slew confirmed in use:** the low-pass lane shaped the drop as written; automation placed on the same tick as a
  hit does not reach that hit.
- **Song mode plays the written order.** With MIDI clock set back to internal (SHIFT+ERASE, 1 0 0, MAIN), hold MAIN +
  ENTER opens the song list editor and SHIFT+PLAY plays the song from the start; the device ran the five positions at
  169.1 BPM by its note stream (170 nominal), stopped by itself after the last position, and the recorded MIDI shows
  intro-like, drop-like, breakdown-like and drop-like positions in the written order. The Mac's START still plays only
  the cursor scene, as the epsysex docs say. **Start burst:** at the first SHIFT+PLAY the device emitted about 140
  note-ons within 50 ms (several bars' worth of hats and shakers) and the intro position came out two bars short; the
  rest played normally. The epsysex docs report note floods when transport starts soon after a write; here the last FILE
  session was minutes earlier, so the cause is open (a second start was not taken). `scripts/record_song.py` records a
  song-mode run from its first note, measures the tempo from the hats, and flags such a burst.
- **Mix:** per-band RMS over the recording put the sub 25-30 dB above the hats in the first mix; raising the hats and
  shaker and lowering the reese and sub by 6-18 amplitude units brought the high band up 6 dB. A per-group mix report
  and MIDI-based grading for dense beats are proposed as follow-ups.
- Device state at the end: P9 holds the four-scene song (cursor on the drop), slots 600/601 hold the spike's test tones,
  MIDI clock is internal (100) after the song-mode test, P2 was the active project before the day's work.
