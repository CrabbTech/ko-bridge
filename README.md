# ko-bridge

An MCP server that lets an AI model drive a Teenage Engineering **EP-133 K.O. II** from a Mac: the sample library and
project slots over the SysEx FILE protocol, pads and transport over USB MIDI, and the OS 2.5 USB audio stream for
listening back and grading timing.

ko-bridge is a **sibling of [op-bridge](../op-bridge), not a replacement** for it. op-bridge targets the OP-1 Field and
continues unchanged; this project copies the generic pieces it needs (recorder with sample-accurate marks, jobs, onset
analysis, server conventions) and leaves the Field code alone. The two servers bind only their own MIDI ports and audio
device so they can run side by side.

## What it does

| Area | Tools |
|---|---|
| Orientation | `get_status`, `get_guide` (quickstart, device, project, human), `human_steps`, `set_config` |
| Sample library (slots 1-999, 128 MB) | `list_sounds`, `describe_sound`, `audition_sound`, `upload_sound` (any WAV, converted and byte-verified), `upload_sounds` (a crate, as a job), `delete_sound` (acknowledgement string, factory banks protected) |
| Projects (slots 1-9) | `describe_project`, `write_project` (pads, patterns, fader lanes, scene, BPM, FX from a typed spec, compiled over the slot's current archive), `assign_pads`, `write_pattern`, `select_project`, `list_checkpoints`, `restore_checkpoint`, `backup_device` |
| Playing and listening | `play_pattern` (MIDI START + clock, records and grades onsets against the grid), `play_notes` (live pads with measured latency), `record_audio`, `list_takes`, `get_take`, `measure_take` |
| Chopping, groove and producer tools | `chop_sample` (slice a loop across 12 pads of one group from a single shared slot via trim start + length, choke, the turnaround reversed; verified on the device), `generate_lyria_chop_group` (an instrumental loop from Lyria 3.5, a paid call), `compile_beat_with_gemini`, `critique_take` (Gemini 3.1 Pro when `GEMINI_API_KEY` is configured, otherwise a result labelled as heuristic / DSP-only). Nothing is synthesized in place of a sample: without Lyria the tool reports the error |
| Jobs | `job_status`, `list_jobs`, `cancel_job` |

Every project write is checkpointed, read back and compared byte for byte, then reloaded on the device. Playback keeps a
6 s distance from FILE traffic (see [docs/spike-results.md](docs/spike-results.md) for why). System settings cannot be
reached over USB; `human_steps` gives the exact key presses (MIDI clock receive is `SHIFT+ERASE`, `1 0 1`, `MAIN`).

## Running

```bash
uv sync
uv run ko-bridge status
uv run ko-bridge serve            # stdio MCP server
uv run ko-bridge register         # prints the Claude Desktop entry; --write merges it into claude_desktop_config.json
uv run ko-bridge backup           # projects + every sound + a .pak into ~/Music/ko-bridge/backups/<timestamp>/
uv sync --extra gemini            # google-genai for the Lyria 3.5 / Gemini 3.1 Pro tools (imported lazily)
```

Files live under `~/Music/ko-bridge` (`KO_BRIDGE_HOME`): `backups/`, `backups/checkpoints/`, `takes/`, `library.json`,
`config.json`. Secrets go in `secrets.env` there (KEY=VALUE lines, read into the environment at server start without
overriding what is already set; `~/Music/op-bridge/secrets.env` is read when ko-bridge has none), never in the repo.

## Tests

Functional tests only, on the real device; they skip when no EP-133 is connected. Writes go to the scratch slot
(`KO_BRIDGE_SCRATCH_PROJECT`, default 9) and to free sound slots from 600 up, and the scratch slot is restored from its
checkpoint afterwards. The playback tests need MIDI clock set to receive on the device. `tests/test_chop.py` reads a
factory sample from the device, chops it into the scratch project from one shared slot and checks the trims on the
readback, the reversed pad and the choke cut in a recording, and the clocked chops' timing on high-band onsets.

```bash
uv run pytest
```

## Spike (2026-10-16)

Project round trip, sample upload (~24 KB/s), 3.5 ms note-to-audio latency and a clocked 4-bar pattern with a fader lane
were all verified on hardware before the server was written; numbers, protocol facts and the human steps are in
[docs/spike-results.md](docs/spike-results.md), and the scripts in `scripts/` reproduce every step.

## Protocol credit

Project and sample I/O use [epsysex](https://github.com/kmorrill/ep-series-sysex) (MIT), the community
reverse-engineering of TE's EP-series SysEx FILE protocol.
