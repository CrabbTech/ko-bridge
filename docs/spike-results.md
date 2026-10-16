# EP-133 K.O. II write-path spike (2026-10-16)

Status: **in progress** — device backup running; numbers below are filled in as each step completes.

Goal: prove, from this Mac, that an EP-133 K.O. II can be driven the way ko-bridge needs before any UI work:
project round trip over SysEx, sample upload, note-to-audio latency over USB audio, a clocked pattern, one write during
playback. Tools: [epsysex](https://github.com/kmorrill/ep-series-sysex) at commit `891a905c` (MIT) for the FILE
protocol, `ko_bridge.*` for MIDI, audio and the checkpoint-before-write rule.

## Device

| Item | Value | Source |
|---|---|---|
| MIDI ports | input `EP-133`, output `EP-133` | mido port list |
| USB audio | `EP-133`, 2 in / 2 out, CoreAudio default 48,000 Hz | sounddevice |
| Identity reply | `f0 7e 33 06 02 00 20 76 20 00 02 00 00 00 00 00 f7` → TE, family 0x0020, member 2, **version bytes all zero** | MIDI Universal Identity Request |
| Firmware | OS 2.5.x by behaviour (USB audio present, identity version zeroed as the docs predict for 2.5.x); exact number attested by Tyler: _pending_ | device UI |
| Sample memory | TE guide: "up to 999 samples, or 128 MB"; this unit: _pending Tyler_ (protocol exposes no total; STAT on `/sounds` reports size 0) | TE guide, STAT |
| Sound library | 610 occupied slots, 52.26 MB; 1-599 factory banks, **700-792, 800-870, 900-978 also occupied**; free: 600-699, 793-799, 871-899, 979-999 | FILE LIST |
| Projects | P1 81,920 B (48 pads, 24 pattern members, 17 non-empty, scenes, 126 BPM); P2 57,856 B (48 pads, empty a01, scenes, 114 BPM); P3-P5 40,960 B (36 pads, no patterns, no settings); P6-P9 4,096 B (directory skeleton only) | FILE GET |
| Scratch slot | **P9** (Tyler: "not concerned with the existing projects") | Tyler |

## Backup

`~/Music/ko-bridge/backups/20261016-132738/`: `identity.json`, `projects/P01-P09.tar`, `sounds-list.json`,
`sounds/NNN.pcm` + `NNN.json` (raw PCM + the 19-field slot metadata), `manifest.json`, `stat.json`,
`sounds-progress.jsonl`, and `ep133-backup.pak` (ZIP in the Sample Tool layout, built from the PCM). Checkpoints of
every project write land in `backups/checkpoints/`.

## Throughput

_pending_

## Latency

_pending_

## Pattern timing

_pending_

## Writes

_pending_

## Protocol facts checked against this device

_pending_

## Human steps

_pending_

## Recommended device-layer API (MCP tools)

_pending_
