"""Clock-aligned micro-timing, MPC/SP-1200 swing, Houston Chop & Screw stutters, and fader slew compensation.

Why this module exists:
1. **96-PPQN vs. 24-PPQN MIDI Clock Quantization**:
   The EP-133 sequencer stores ticks at 96 PPQN (`24` ticks per 16th step), but external MIDI clock runs at
   24 PPQN (`6` clocks per 16th step = `4` internal ticks per MIDI clock pulse). Placing events on arbitrary
   1-tick offsets (like `+1/24` step chord staggers) forces the EP-133 clock follower to interpolate between
   `0xF8` pulses, causing audible flamming and timing jitter. All micro-timing here snaps to `CLOCK_TICKS = 4`.

2. **Group Fader Latch Slew (~100 ms)**:
   The EP-133's internal group fader slews over ~60-150 ms (`docs/spike-results.md`). Step-jumps placed *on*
   a beat miss the attack of that beat, and 1/16th LFO wobbles flatten out. `compensate_fader_slew` pre-leads
   step jumps by `0.5` steps (`12` ticks) and thins unplayable sub-step LFO spam.
"""
from __future__ import annotations

from typing import Literal

from .project import BAR_TICKS, FaderParameter, Lane, LanePoint, NoteEvent, PadLabel, STEP_TICKS

CLOCK_TICKS = 4  # 1 external 24-PPQN MIDI clock pulse = 4 internal 96-PPQN ticks


def snap_tick_to_clock(tick: int, max_ticks: int | None = None) -> int:
    """Snap a 96-PPQN tick to the nearest 4-tick (24-PPQN MIDI clock) boundary."""
    snapped = int(round(tick / CLOCK_TICKS)) * CLOCK_TICKS
    if max_ticks is not None:
        snapped = min(snapped, max_ticks - CLOCK_TICKS)
    return max(0, snapped)


def snap_step_to_clock(step: float) -> float:
    """Snap a 16th-step float to the nearest 1/6th step (4-tick MIDI clock) boundary."""
    return snap_tick_to_clock(int(round(step * STEP_TICKS))) / STEP_TICKS


def align_events_to_clock(events: list[NoteEvent], bars: int = 4) -> list[NoteEvent]:
    """Return a copy of `events` with all timestamps snapped to 4-tick MIDI clock boundaries."""
    max_ticks = bars * BAR_TICKS
    out: list[NoteEvent] = []
    for ev in events:
        t = snap_tick_to_clock(ev.tick_value(), max_ticks=max_ticks)
        out.append(ev.model_copy(update={"tick": t, "step": None}))
    return out


def swing_offset_ticks(swing: float = 0.58, snap_clock: bool = True) -> int:
    """Convert a swing ratio in [0.50, 0.75] (e.g. 0.58 = 58% MPC swing, 0.667 = triplet swing) to 96-PPQN ticks."""
    clamped = max(0.50, min(0.75, float(swing)))
    raw = int(round((clamped - 0.50) * 2.0 * STEP_TICKS))
    return snap_tick_to_clock(raw) if snap_clock else raw


def apply_mpc_swing(
    events: list[NoteEvent],
    swing: float = 0.58,
    bars: int = 4,
    snap_clock: bool = True,
    offbeat_velocity_scale: float = 0.92,
) -> list[NoteEvent]:
    """Apply classic MPC / SP-1200 16th-note swing to odd 16th steps (`step % 2 == 1`).

    With `snap_clock=True`:
    - `swing=0.58` -> `+4` ticks (1 MIDI clock pulse = 58.3% swing)
    - `swing=0.66` -> `+8` ticks (2 MIDI clock pulses = 66.7% triplet swing)
    """
    offset = swing_offset_ticks(swing, snap_clock=snap_clock)
    max_ticks = bars * BAR_TICKS
    out: list[NoteEvent] = []
    for ev in events:
        t = ev.tick_value()
        step_idx = t // STEP_TICKS
        rem = t % STEP_TICKS
        vel = ev.velocity
        if rem == 0 and (step_idx % 2 == 1):
            t = min(max_ticks - CLOCK_TICKS, t + offset)
            vel = max(1, min(127, int(round(vel * offbeat_velocity_scale))))
        elif snap_clock:
            t = snap_tick_to_clock(t, max_ticks=max_ticks)
        out.append(ev.model_copy(update={"tick": t, "step": None, "velocity": vel}))
    return out


def apply_lazy_pocket(
    events: list[NoteEvent],
    snare_pads: tuple[PadLabel, ...] = ("0", "2"),
    hat_pads: tuple[PadLabel, ...] = ("1", "3"),
    snare_delay_ticks: int = 4,
    hat_delay_ticks: int = 0,
    bars: int = 4,
) -> list[NoteEvent]:
    """Push backbeat snares (`step % 8 == 4`) or hats slightly behind the grid on exact MIDI clock boundaries."""
    max_ticks = bars * BAR_TICKS
    s_delay = snap_tick_to_clock(snare_delay_ticks)
    h_delay = snap_tick_to_clock(hat_delay_ticks)
    out: list[NoteEvent] = []
    for ev in events:
        t = ev.tick_value()
        if ev.pad in snare_pads and (t // STEP_TICKS) % 8 == 4:
            t = min(max_ticks - CLOCK_TICKS, t + s_delay)
        elif ev.pad in hat_pads and h_delay > 0:
            t = min(max_ticks - CLOCK_TICKS, t + h_delay)
        out.append(ev.model_copy(update={"tick": t, "step": None}))
    return out


def make_screw_double_up(
    pad: PadLabel,
    step: float,
    velocity: int = 110,
    style: Literal["double_up", "triplet_roll", "chop_echo"] = "double_up",
    semitones: int | None = None,
    concert: str | None = None,
    duration_steps: float = 2.0,
) -> list[NoteEvent]:
    """Generate an authentic Houston DJ Screw stutter / double-up figure around `step`.

    Styles:
    - `"double_up"`: Classic crossfader pre-hit 1/32nd (`0.5` step = `12` ticks = 3 MIDI clocks) before `step`
      at 78% velocity, followed by the full accent hit at `step`.
    - `"triplet_roll"`: Three clock-aligned hits at `step - 2/3`, `step - 1/3`, and `step` (`-16`, `-8`, `0` ticks)
      with rising velocity (`68%`, `84%`, `100%`).
    - `"chop_echo"`: Main hit at `step` followed by a pitched-down ghost echo at `step + 1.0`.
    """
    center_tick = snap_tick_to_clock(int(round(step * STEP_TICKS)))
    main_dur = max(8, int(round(duration_steps * STEP_TICKS)))
    out: list[NoteEvent] = []

    if style == "double_up":
        pre_tick = max(0, center_tick - 12)  # 12 ticks = 0.5 step = 3 MIDI clocks
        if pre_tick < center_tick:
            out.append(
                NoteEvent(
                    pad=pad,
                    tick=pre_tick,
                    velocity=max(40, int(round(velocity * 0.78))),
                    semitones=semitones,
                    concert=concert,
                    duration=10,
                )
            )
        out.append(
            NoteEvent(
                pad=pad,
                tick=center_tick,
                velocity=velocity,
                semitones=semitones,
                concert=concert,
                duration=main_dur,
            )
        )

    elif style == "triplet_roll":
        for dt, scale in ((-16, 0.68), (-8, 0.84), (0, 1.0)):
            t = max(0, center_tick + dt)
            out.append(
                NoteEvent(
                    pad=pad,
                    tick=t,
                    velocity=max(35, int(round(velocity * scale))),
                    semitones=semitones,
                    concert=concert,
                    duration=6 if dt < 0 else main_dur,
                )
            )

    else:  # "chop_echo"
        out.append(
            NoteEvent(
                pad=pad,
                tick=center_tick,
                velocity=velocity,
                semitones=semitones,
                concert=concert,
                duration=min(20, main_dur),
            )
        )
        echo_semitones = (semitones - 2) if semitones is not None else -2
        out.append(
            NoteEvent(
                pad=pad,
                tick=center_tick + STEP_TICKS,
                velocity=max(40, int(round(velocity * 0.72))),
                semitones=echo_semitones if concert is None else None,
                concert=concert,
                duration=main_dur,
            )
        )
    return out


def strum_chord(
    pad: PadLabel,
    step: float,
    notes: list[str],
    duration_steps: float = 7.5,
    velocity: int = 92,
    strum_clocks: int = 0,
) -> list[NoteEvent]:
    """Place a polyphonic chord on `pad` without 1-tick MIDI clock interpolation jitter.

    By default `strum_clocks=0` places all chord tones on the exact same tick (cleanest attack on EP-133).
    Setting `strum_clocks=1` separates chord tones by `4` ticks (1 full 24-PPQN MIDI clock pulse).
    """
    base_tick = snap_tick_to_clock(int(round(step * STEP_TICKS)))
    dur_ticks = max(8, int(round(duration_steps * STEP_TICKS)))
    out: list[NoteEvent] = []
    for i, n in enumerate(notes):
        t = base_tick + i * max(0, int(strum_clocks)) * CLOCK_TICKS
        vel = max(40, min(127, velocity - i * 3))
        out.append(NoteEvent(pad=pad, tick=t, concert=n, velocity=vel, duration=dur_ticks))
    return out


def compensate_fader_slew(
    points: list[LanePoint],
    bars: int = 4,
    lead_steps: float = 0.5,
    min_step_gap: float = 1.0,
    bpm: float | None = None,
    lead_ms: float = 120.0,
) -> list[LanePoint]:
    """Compensate for the EP-133's group-fader slew and remove unplayable LFO spam.

    Measured (docs/spike-results.md): a fader change is not complete within ~60 ms and is complete within ~200 ms, so
    the lead is a time, not a step count: 0.5 step is 110 ms at 68 BPM but only 43 ms at 174 BPM. With `bpm` given
    the lead is `lead_ms` converted to ticks (snapped to the 4-tick clock grid); without it `lead_steps` is used.

    1. Thins points closer than `min_step_gap` steps (since the hardware fader cannot track 1/16th LFOs).
    2. Pre-leads large step jumps (`|dv| >= 0.15` after `tick > 0`) by the lead so the fader reaches its target value
       *before* the beat transient fires.
    """
    if not points:
        return []
    max_ticks = bars * BAR_TICKS
    if bpm:
        lead_ticks = snap_tick_to_clock(int(round(lead_ms / 1000.0 * float(bpm) / 60.0 * 4 * STEP_TICKS)))
    else:
        lead_ticks = snap_tick_to_clock(int(round(lead_steps * STEP_TICKS)))
    lead_ticks = max(CLOCK_TICKS, lead_ticks)
    min_gap_ticks = max(CLOCK_TICKS, int(round(min_step_gap * STEP_TICKS)))

    ordered = sorted(points, key=lambda p: p.tick_value())
    thinned: list[tuple[int, float]] = []
    for pt in ordered:
        t = min(max_ticks - 1, max(0, pt.tick_value()))
        v = max(0.0, min(1.0, float(pt.value)))
        if thinned and (t - thinned[-1][0]) < min_gap_ticks:
            # Drop unplayable sub-step LFO points that fall inside the hardware fader slew window
            continue
        thinned.append((t, v))

    compensated: list[LanePoint] = []
    for idx, (t, v) in enumerate(thinned):
        if idx > 0 and t > lead_ticks:
            prev_t, prev_v = thinned[idx - 1]
            if abs(v - prev_v) >= 0.15 and (t - prev_t) >= lead_ticks * 2:
                t = max(prev_t + CLOCK_TICKS, t - lead_ticks)
        compensated.append(LanePoint(tick=t, value=round(v, 4)))
    return compensated


def make_filter_swell_lane(
    bars: int = 4,
    parameter: FaderParameter = "lowpass",
    low_val: float = 0.38,
    high_val: float = 0.95,
    peak_bar: int = 3,
    bpm: float | None = None,
) -> Lane:
    """Create a musical phrase-level filter or level swell lane that respects the EP-133's fader slew (lead in ms when `bpm` is given)."""
    raw = [
        LanePoint(step=0.0, value=low_val),
        LanePoint(step=max(2.0, (peak_bar - 1) * 16.0), value=round((low_val + high_val) * 0.5, 3)),
        LanePoint(step=max(4.0, peak_bar * 16.0 - 4.0), value=high_val),
        LanePoint(step=bars * 16.0 - 2.0, value=low_val),
    ]
    return Lane(parameter=parameter, points=compensate_fader_slew(raw, bars=bars, bpm=bpm))
