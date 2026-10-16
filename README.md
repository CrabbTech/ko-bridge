# ko-bridge

An MCP server that lets an AI model drive a Teenage Engineering **EP-133 K.O. II** from a Mac:
MIDI notes over USB, the device's project and sample library over SysEx, and the OS 2.5
class-compliant USB audio stream for listening back.

ko-bridge is a **sibling of [op-bridge](../op-bridge), not a replacement** for it. op-bridge
targets the OP-1 Field and continues unchanged; this project copies the generic pieces it needs
(MIDI port handling, audio capture, latency probing, background jobs) and leaves the Field code
alone. The two servers bind only their own MIDI ports and audio device so they can run side by
side. A shared core is a later extraction, once both are stable.

## Status

Spike complete (2026-10-16): project round trip, sample upload, 3.5 ms note-to-audio latency and a clocked
4-bar pattern with a fader lane all verified on real hardware. Results, measured facts and the recommended MCP tool list
are in [docs/spike-results.md](docs/spike-results.md); the scripts in `scripts/` reproduce every step.

## Ground rules

- Back up the device before every write; never write settings; never format.
- All device I/O is serialized through one lock.
- Only real-device functional tests; no unit tests.
- Secrets live in `~/Music/ko-bridge/secrets.env` (or `~/Music/op-bridge/secrets.env`), never in the repo.
- Documentation first: TE's guides and the [kmorrill/ep-series-sysex](https://github.com/kmorrill/ep-series-sysex) docs are read before an exploratory device test.

## Protocol credit

Project and sample I/O use [epsysex](https://github.com/kmorrill/ep-series-sysex) (MIT), the community
reverse-engineering of TE's EP-series SysEx FILE protocol.
