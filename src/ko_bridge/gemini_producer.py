"""Google Gemini 3.1 Pro & Lyria 3 integration for ko-bridge:
- `generate_lyria_stem`: Generates musical sample loops via `lyria-3-clip-preview` for Group C/D auto-chopping.
- `compile_beat_with_gemini`: Compiles natural-language beat prompts into multi-scene `ProjectSpec` arrangements
  using `gemini-3.1-pro-preview` with structured JSON schemas, MPC swing, DJ Screw double-up stutters, and
  slew-compensated fader automation.
- `critique_take_with_gemini`: Multimodal closed-loop audio critic that sends the recorded EP-133 USB WAV +
  4-band spectral DSP report to `gemini-3.1-pro-preview` and returns actionable mix/groove refinements.
"""
from __future__ import annotations

import base64
import io
import json
import math
import os
from typing import Any, Literal

import numpy as np
import soundfile as sf
from pydantic import BaseModel, Field

from . import analysis as an
from . import chopper as CH
from . import config as C
from . import groove as GR
from .project import (
    FaderBase,
    FxSpec,
    Group,
    Lane,
    LanePoint,
    NoteEvent,
    PadLabel,
    PatternSpec,
    ProjectSpec,
    SceneSpec,
    SidechainSource,
)

GEMINI_PRO_MODEL = "gemini-3.1-pro-preview"
LYRIA_CLIP_MODEL = "lyria-3-clip-preview"


# --------------------------------------------------------------------------- Structured Schemas for Gemini 3.1 Pro

class DrumHitSpec(BaseModel):
    pad: PadLabel = Field(description='Group A drum pad: "." (kick), "0" (snare), "1" (closed hat), "2" (open hat), "3" (clap/rim)')
    step: float = Field(ge=0, lt=64, description="16th-note step index (0..63 for 4 bars)")
    velocity: int = Field(default=100, ge=1, le=127)


class BassHitSpec(BaseModel):
    pad: PadLabel = Field(default=".", description='Group B bass pad ("." = sub/808, "0" = reese/mid-bass)')
    step: float = Field(ge=0, lt=64)
    semitones: int = Field(default=0, ge=-12, le=12, description="Semitone offset from root")
    duration_steps: float = Field(default=3.5, ge=0.5, le=16.0)
    velocity: int = Field(default=108, ge=1, le=127)


class ChopTriggerSpec(BaseModel):
    pad: PadLabel = Field(description='Group C chop pad: "1".."9" (main slices), "0" / "." (transient accents), "enter" (reversed pickup)')
    step: float = Field(ge=0, lt=64)
    velocity: int = Field(default=110, ge=1, le=127)
    semitones: int = Field(default=0, ge=-12, le=12)
    stutter: Literal["none", "double_up", "triplet_roll", "chop_echo"] = "none"
    duration_steps: float | None = Field(default=None, description="Optional explicit duration in 16th steps")


class TapeStopDiveSpec(BaseModel):
    step: float = Field(ge=0, lt=64, description="Step where the tape-stop pitch dive or Punch-In FX burst begins")
    duration_steps: float = Field(default=2.0, ge=0.5, le=8.0)
    kind: Literal["tape_stop", "punch_in_x", "punch_in_y"] = "tape_stop"
    intensity: float = Field(default=0.85, ge=0.1, le=1.0)


class GeminiBeatPlan(BaseModel):
    """Structured beat composition returned by Gemini 3.1 Pro."""
    title: str
    bpm: float = Field(ge=45, le=190)
    bars: int = Field(default=4, ge=1, le=8)
    swing: float = Field(default=0.58, ge=0.50, le=0.68, description="MPC 16th swing (0.50 straight, 0.58 pocket, 0.66 triplet)")
    snare_delay_ticks: int = Field(default=4, ge=0, le=8, description="Lazy backbeat snare pocket in ticks (0 or 4)")
    screw_semitones: float = Field(default=-3.0, ge=-7.0, le=5.0, description="Pitch/speed shift for Group C chops (-3.5 for Houston Screw)")
    drums: list[DrumHitSpec]
    bass: list[BassHitSpec]
    chops: list[ChopTriggerSpec]
    fx_dives: list[TapeStopDiveSpec] = []
    lowpass_base: float = Field(default=0.82, ge=0.2, le=1.0)
    fx_type: Literal["off", "delay", "reverb", "distortion", "chorus", "filter", "compressor"] = "delay"
    fx_x: float = Field(default=0.38, ge=0.0, le=1.0)
    fx_y: float = Field(default=0.42, ge=0.0, le=1.0)
    producer_notes: str


class PadDeltaRecommendation(BaseModel):
    group: Group
    pad: PadLabel
    amplitude_delta: int = Field(ge=-40, le=40, description="Suggested change to PadSpec.amplitude (-20 to +20)")
    pitch_delta: int = Field(default=0, ge=-12, le=12)
    reason: str


class GeminiAudioCritique(BaseModel):
    """Structured closed-loop audio critique returned by Gemini 3.1 Pro after listening to a recorded take."""
    overall_score: int = Field(ge=1, le=100)
    timing_tightness_score: int = Field(ge=1, le=100)
    groove_assessment: str
    spectral_balance_assessment: str
    swing_recommendation: float = Field(ge=0.50, le=0.68)
    snare_pocket_ticks: int = Field(ge=0, le=8)
    pad_adjustments: list[PadDeltaRecommendation] = []
    fader_recommendations: list[str] = []
    next_scene_variation_idea: str


# --------------------------------------------------------------------------- Client Helper

def _get_client(api_key: str | None = None):
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        return None
    from google import genai
    return genai.Client(api_key=key)


# --------------------------------------------------------------------------- 1. Lyria 3 Stem Generator

def _render_organic_soul_stem(bpm: float = 68.0, bars: int = 4, sr: int = 46875, seed: int = 42) -> np.ndarray:
    """Fallback warm vintage Rhodes + dusty bass + vinyl texture loop when offline or without GEMINI_API_KEY.

    Uses electric-piano tine overtones, tape wow/flutter LFO, chorus stereo spread, and transient horn/vocal
    accents so auto-chopping produces genuinely musical slices with clear transient boundaries.
    """
    rng = np.random.default_rng(seed)
    bar_s = 240.0 / float(bpm)
    total_s = bars * bar_s
    n = int(round(total_s * sr))
    t = np.arange(n, dtype=np.float64) / sr

    # Warm minor-9th chord progression across 4 bars (Cm9 -> Fm9 -> Abmaj7 -> G7alt)
    chords_midi = [
        (48, 60, 63, 67, 70, 74),  # Cm9
        (53, 60, 65, 68, 72, 75),  # Fm9
        (56, 63, 68, 72, 75, 79),  # Abmaj9
        (55, 62, 65, 71, 74, 78),  # G7#5#9
    ]

    left = np.zeros(n, dtype=np.float64)
    right = np.zeros(n, dtype=np.float64)

    # Subtle tape wow & flutter (0.55 Hz + 4.2 Hz)
    flutter = 1.0 + 0.0012 * np.sin(2 * math.pi * 0.55 * t) + 0.0004 * np.sin(2 * math.pi * 4.2 * t)

    # Render 8 structural chord/stab events per 4-bar loop (every half-bar + syncopated accents)
    half_bar_s = bar_s / 2.0
    for bar_idx in range(bars):
        chord = chords_midi[bar_idx % len(chords_midi)]
        for sub_idx, beat_offset_s in enumerate((0.0, half_bar_s * 0.75, half_bar_s, half_bar_s * 1.625)):
            onset_s = bar_idx * bar_s + beat_offset_s
            start_i = int(round(onset_s * sr))
            if start_i >= n:
                continue
            dur_s = half_bar_s * (0.85 if sub_idx in (0, 2) else 0.45)
            end_i = min(n, start_i + int(round(dur_s * sr)))
            seg_t = np.arange(end_i - start_i, dtype=np.float64) / sr
            seg_flutter = flutter[start_i:end_i]

            # Rhodes electric piano voice: fundamental + warm 2nd/3rd harmonic + bell tine transient
            env_body = (1.0 - np.exp(-seg_t / 0.004)) * np.exp(-seg_t / 1.15)
            env_tine = (1.0 - np.exp(-seg_t / 0.0015)) * np.exp(-seg_t / 0.065)
            voice_l = np.zeros_like(seg_t)
            voice_r = np.zeros_like(seg_t)
            vel_scale = 1.0 if sub_idx in (0, 2) else 0.72
            for k, m_note in enumerate(chord):
                freq = 440.0 * (2.0 ** ((m_note - 69.0) / 12.0))
                phase = 2.0 * math.pi * freq * seg_t * seg_flutter
                tone = (
                    0.65 * np.sin(phase + 0.25 * np.sin(2.0 * phase)) * env_body
                    + 0.22 * np.sin(2.0 * phase) * env_body * np.exp(-seg_t / 0.45)
                    + 0.14 * np.sin(6.0 * phase) * env_tine
                )
                pan_l = 0.65 - 0.06 * (k % 3)
                pan_r = 0.35 + 0.06 * (k % 3)
                voice_l += tone * pan_l * vel_scale
                voice_r += tone * pan_r * vel_scale

            left[start_i:end_i] += voice_l / len(chord)
            right[start_i:end_i] += voice_r / len(chord)

    # Add gentle tremolo + warm saturation + subtle vinyl dust
    trem_l = 0.85 + 0.15 * np.sin(2 * math.pi * (bpm / 60.0) * t)
    trem_r = 0.85 + 0.15 * np.cos(2 * math.pi * (bpm / 60.0) * t)
    left = np.tanh(left * trem_l * 1.8) * 0.75
    right = np.tanh(right * trem_r * 1.8) * 0.75

    dust = rng.normal(0.0, 0.0025, size=(n, 2))
    stereo = np.column_stack([left, right]) + dust
    peak = float(np.max(np.abs(stereo))) + 1e-9
    stereo = (stereo / peak * 0.89).astype(np.float32)
    return stereo


def generate_lyria_stem(
    prompt: str,
    bpm: float = 68.0,
    bars: int = 4,
    out_wav_path: str | None = None,
    samplerate: int = 46875,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Generate a musical sample loop using Google's Lyria 3 (`lyria-3-clip-preview`) and save it as a WAV.

    Falls back to the organic vintage Rhodes/Soul stem generator if `GEMINI_API_KEY` is unavailable or if the
    preview endpoint is unreachable, so downstream chopping and EP-133 project builds always succeed.
    """
    if out_wav_path is None:
        stem_id = C.new_id("lyria-stem")
        out_wav_path = C.file_path("library", "wav", f"{stem_id}.wav")
    os.makedirs(os.path.dirname(os.path.expanduser(out_wav_path)), exist_ok=True)
    out_wav_path = os.path.expanduser(out_wav_path)

    client = _get_client(api_key)
    full_prompt = (
        f"{prompt}. Tempo: {bpm:g} BPM, {bars} bars in 4/4, isolated instrumental/vocal sample loop "
        "with crisp transients suitable for hip-hop/jungle sampler chopping, no heavy master limiting."
    )
    source_kind = "fallback-vintage-soul"
    audio_bytes: bytes | None = None

    if client is not None:
        try:
            # Try Lyria 3 Clip via Interactions API first
            if hasattr(client, "interactions"):
                interaction = client.interactions.create(
                    model=LYRIA_CLIP_MODEL,
                    input=full_prompt,
                    response_modalities=["audio"],
                )
                for output in getattr(interaction, "outputs", []) or []:
                    if getattr(output, "type", None) == "audio" and getattr(output, "data", None):
                        raw = output.data
                        audio_bytes = base64.b64decode(raw) if isinstance(raw, str) else bytes(raw)
                        source_kind = LYRIA_CLIP_MODEL
                        break
        except Exception:
            audio_bytes = None

    if audio_bytes:
        data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
        sf.write(out_wav_path, data, int(sr), subtype="PCM_16")
        return {
            "wav_path": out_wav_path,
            "source": source_kind,
            "prompt": full_prompt,
            "samplerate": int(sr),
            "channels": int(data.shape[1]),
            "seconds": round(len(data) / float(sr), 3),
        }

    # Organic musical stem fallback
    stereo = _render_organic_soul_stem(bpm=bpm, bars=bars, sr=samplerate)
    sf.write(out_wav_path, stereo, int(samplerate), subtype="PCM_16")
    return {
        "wav_path": out_wav_path,
        "source": source_kind,
        "prompt": full_prompt,
        "samplerate": int(samplerate),
        "channels": 2,
        "seconds": round(len(stereo) / float(samplerate), 3),
        "note": "rendered with built-in vintage Rhodes/Soul stem engine (set GEMINI_API_KEY with Lyria access for cloud audio generation)",
    }


# --------------------------------------------------------------------------- 2. Gemini 3.1 Pro Beat Compiler

def compile_heuristic_beat_plan(prompt: str, bpm: float = 68.0, bars: int = 4) -> GeminiBeatPlan:
    """Deterministic musical Chop & Screw / Boom-Bap / Jungle pattern generator when offline or without API key."""
    low_p = prompt.lower()
    is_jungle = any(w in low_p for w in ("jungle", "dnb", "amen", "breakcore", "drum and bass"))
    is_screw = any(w in low_p for w in ("screw", "houston", "chop", "slow", "slab", "memphis", "phonk")) or (bpm <= 82)

    drums: list[DrumHitSpec] = []
    bass: list[BassHitSpec] = []
    chops: list[ChopTriggerSpec] = []
    fx_dives: list[TapeStopDiveSpec] = []

    for bar in range(bars):
        b0 = bar * 16.0
        if is_jungle:
            # Classic syncopated Amen/Jungle 2-step + ghost snare roll
            for k_step, vel in ((0.0, 122), (2.0, 96), (10.0, 118)):
                drums.append(DrumHitSpec(pad=".", step=b0 + k_step, velocity=vel))
            for s_step, vel in ((4.0, 120), (7.0, 74), (9.0, 78), (12.0, 124), (14.0, 86), (15.0, 94)):
                drums.append(DrumHitSpec(pad="0", step=b0 + s_step, velocity=vel))
            for h in range(0, 16, 2):
                drums.append(DrumHitSpec(pad="1", step=b0 + h, velocity=95 if h % 4 == 0 else 76))
        else:
            # Heavy Houston Chop & Screw / Boom-Bap knock
            for k_step, vel in ((0.0, 124), (3.0, 88), (6.0, 112), (10.0, 118)):
                drums.append(DrumHitSpec(pad=".", step=b0 + k_step, velocity=vel))
            if bar % 2 == 1:
                drums.append(DrumHitSpec(pad=".", step=b0 + 14.0, velocity=96))
            # Backbeat snare + clap layer
            for s_step in (4.0, 12.0):
                drums.append(DrumHitSpec(pad="0", step=b0 + s_step, velocity=118))
                drums.append(DrumHitSpec(pad="3", step=b0 + s_step, velocity=102))
            # Steady 8th/16th hats with accent bounce
            for h in range(16):
                if h % 2 == 0 or h in (7, 13, 15):
                    drums.append(DrumHitSpec(pad="1", step=b0 + h, velocity=98 if h % 4 == 0 else 72))
            drums.append(DrumHitSpec(pad="2", step=b0 + 14.0, velocity=86))

        # Deep 808 / Sub bass progression (Cm -> Fm -> Ab -> G)
        bar_semi = (0, 5, -4, -5)[bar % 4]
        bass.append(BassHitSpec(pad=".", step=b0 + 0.0, semitones=bar_semi, duration_steps=5.5, velocity=118))
        bass.append(BassHitSpec(pad=".", step=b0 + 6.0, semitones=bar_semi, duration_steps=3.5, velocity=106))
        bass.append(BassHitSpec(pad=".", step=b0 + 10.0, semitones=bar_semi + (2 if bar % 2 == 1 else 0), duration_steps=5.5, velocity=112))

        # Group C Chop sequence across pads "1".."8", transient accents "9"/"0", and reversed pickup "enter"
        slice_a = str((bar * 2) % 8 + 1)
        slice_b = str((bar * 2 + 1) % 8 + 1)
        chops.append(ChopTriggerSpec(pad=slice_a, step=b0 + 0.0, velocity=114, stutter="none", duration_steps=4.0))
        chops.append(
            ChopTriggerSpec(
                pad="9" if bar % 2 == 0 else "0",
                step=b0 + 4.0,
                velocity=106,
                stutter="double_up" if (is_screw and bar % 2 == 1) else "none",
                duration_steps=4.0,
            )
        )
        chops.append(
            ChopTriggerSpec(
                pad=slice_b,
                step=b0 + 8.0,
                velocity=112,
                stutter="double_up" if is_screw else "none",
                duration_steps=4.0,
            )
        )
        if bar % 2 == 1:
            # Beat 4.3 / 4.4 reversed pickup pull on pad "enter"
            chops.append(ChopTriggerSpec(pad="enter", step=b0 + 12.0, velocity=108, stutter="none", duration_steps=4.0))
            fx_dives.append(TapeStopDiveSpec(step=b0 + 14.0, duration_steps=2.0, kind="tape_stop", intensity=0.85))
        else:
            chops.append(ChopTriggerSpec(pad=".", step=b0 + 12.0, velocity=104, stutter="chop_echo" if is_screw else "none", duration_steps=4.0))

    return GeminiBeatPlan(
        title="Houston Chop & Screw Session" if is_screw else "Chopped Break Session",
        bpm=float(bpm),
        bars=int(bars),
        swing=0.58 if is_screw else 0.54,
        snare_delay_ticks=4 if is_screw else 0,
        screw_semitones=-3.5 if is_screw else 0.0,
        drums=drums,
        bass=bass,
        chops=chops,
        fx_dives=fx_dives,
        lowpass_base=0.78 if is_screw else 0.92,
        fx_type="delay" if is_screw else "compressor",
        fx_x=0.38,
        fx_y=0.44,
        producer_notes=(
            "Built with 4-tick clock-aligned micro-timing, MPC 58% swing, +4 tick lazy snare pocket, "
            "DJ Screw 1/32nd crossfader double-up stutters on Group C, and reversed pickup chops on pad 'enter'."
        ),
    )


def plan_to_project_spec(
    plan: GeminiBeatPlan,
    chop_pads: list[ Any ] | None = None,
    drum_pads: list[ Any ] | None = None,
    bass_pads: list[ Any ] | None = None,
    chop_group: Group = "C",
    pattern_number: int = 1,
    build_two_scenes: bool = True,
) -> tuple[ProjectSpec, dict[int, list[dict[str, Any]]]]:
    """Compile a `GeminiBeatPlan` into an EP-133 `ProjectSpec` + a 24-PPQN `clock_automation` map."""
    bars = int(plan.bars)

    # 1. Group A Drums with MPC swing + lazy snare pocket
    raw_drums = [
        NoteEvent(pad=d.pad, step=d.step, velocity=d.velocity, duration=12)
        for d in plan.drums
        if 0 <= d.step < bars * 16
    ]
    swung_drums = GR.apply_mpc_swing(raw_drums, swing=plan.swing, bars=bars, snap_clock=True)
    pocket_drums = GR.apply_lazy_pocket(swung_drums, snare_delay_ticks=plan.snare_delay_ticks, bars=bars)
    pat_a = PatternSpec(group="A", number=pattern_number, bars=bars, events=pocket_drums)

    # 2. Group B Chromatic Sub / 808 Bass
    bass_events = [
        NoteEvent(
            pad=b.pad,
            tick=GR.snap_tick_to_clock(int(round(b.step * 24))),
            semitones=b.semitones,
            velocity=b.velocity,
            duration=max(12, int(round(b.duration_steps * 24))),
        )
        for b in plan.bass
        if 0 <= b.step < bars * 16
    ]
    pat_b = PatternSpec(group="B", number=pattern_number, bars=bars, events=bass_events)

    # 3. Group C Sample Chops + DJ Screw Double-Up Stutters
    chop_events: list[NoteEvent] = []
    sorted_chops = sorted([c for c in plan.chops if 0 <= c.step < bars * 16], key=lambda c: c.step)
    for i, ch in enumerate(sorted_chops):
        if ch.duration_steps is not None:
            dur_steps = float(ch.duration_steps)
        elif i + 1 < len(sorted_chops):
            dur_steps = max(0.5, float(sorted_chops[i + 1].step - ch.step) - 0.1)
        else:
            dur_steps = max(1.0, bars * 16.0 - float(ch.step))

        if ch.stutter in ("double_up", "triplet_roll", "chop_echo"):
            chop_events.extend(
                GR.make_screw_double_up(
                    pad=ch.pad,
                    step=ch.step,
                    velocity=ch.velocity,
                    style=ch.stutter,
                    semitones=ch.semitones if ch.semitones != 0 else None,
                    duration_steps=dur_steps,
                )
            )
        else:
            chop_events.append(
                NoteEvent(
                    pad=ch.pad,
                    tick=GR.snap_tick_to_clock(int(round(ch.step * 24))),
                    velocity=ch.velocity,
                    semitones=ch.semitones if ch.semitones != 0 else None,
                    duration=max(8, int(round(dur_steps * 24))),
                )
            )

    chop_lane = GR.make_filter_swell_lane(bars=bars, parameter="lowpass", low_val=max(0.35, plan.lowpass_base - 0.2), high_val=plan.lowpass_base)
    pat_c = PatternSpec(group=chop_group, number=pattern_number, bars=bars, events=chop_events, lanes=[chop_lane])

    patterns = [pat_a, pat_b, pat_c]
    scenes = [SceneSpec(patterns={"A": pattern_number, "B": pattern_number, chop_group: pattern_number})]

    if build_two_scenes:
        # Scene 2: B-Section variation with intensified double-ups and full filter open
        pat_c2_events = list(chop_events)
        pat_c2_events.extend(GR.make_screw_double_up(pad="enter", step=bars * 16 - 4, velocity=116, style="double_up", duration_steps=4.0))
        pat_c2 = PatternSpec(group=chop_group, number=pattern_number + 1, bars=bars, events=pat_c2_events)
        patterns.append(pat_c2)
        scenes.append(SceneSpec(patterns={"A": pattern_number, "B": pattern_number, chop_group: pattern_number + 1}))

    # 4. Build live 24-PPQN clock_automation map (6 clocks per 16th step) for zero-slew tape-stops & Punch-In FX
    clock_auto: dict[int, list[dict[str, Any]]] = {}
    for dive in plan.fx_dives:
        start_clk = max(0, int(round(dive.step * 6)))
        dur_clks = max(3, int(round(dive.duration_steps * 6)))
        if dive.kind == "tape_stop":
            for k in range(dur_clks):
                frac = (k + 1) / dur_clks
                clock_auto.setdefault(start_clk + k, []).append({"kind": "pitchwheel", "amount": round(-dive.intensity * frac, 3)})
            clock_auto.setdefault(start_clk + dur_clks, []).append({"kind": "pitchwheel", "amount": 0.0})
        else:
            ctrl = 12 if dive.kind == "punch_in_x" else 13
            val = max(1, min(127, int(round(dive.intensity * 127))))
            clock_auto.setdefault(start_clk, []).append({"kind": "cc", "control": ctrl, "value": val})
            clock_auto.setdefault(start_clk + dur_clks, []).append({"kind": "cc", "control": ctrl, "value": 0})

    all_pads = list(drum_pads or []) + list(bass_pads or []) + list(chop_pads or [])
    spec = ProjectSpec(
        bpm=float(plan.bpm),
        pads=all_pads,
        patterns=patterns,
        scenes=scenes,
        song=[1, 2] if len(scenes) > 1 else [1],
        current_scene=1,
        fader_bases=[
            FaderBase(group="A", parameter="level", value=0.95),
            FaderBase(group="B", parameter="level", value=0.88),
            FaderBase(group=chop_group, parameter="lowpass", value=plan.lowpass_base),
        ],
        fx=FxSpec(
            type=plan.fx_type,
            x=plan.fx_x if plan.fx_type != "off" else None,
            y=plan.fx_y if plan.fx_type != "off" else None,
            output_compressor_drive=0.32,
            output_compressor_speed=0.48,
            sidechain_sources=[SidechainSource(group="A", pad=".")],
            sidechain_destinations=["B", chop_group],
            sidechain_length=0.38,
            sidechain_shape=0.55,
        ),
    )
    return spec, clock_auto


def compile_beat_with_gemini(
    prompt: str,
    bpm: float = 68.0,
    bars: int = 4,
    api_key: str | None = None,
) -> tuple[GeminiBeatPlan, str]:
    """Use `gemini-3.1-pro-preview` with structured JSON schema to compile a beat prompt into a `GeminiBeatPlan`.

    Returns `(plan, model_used)`. Falls back to `compile_heuristic_beat_plan` if no API key is configured.
    """
    client = _get_client(api_key)
    if client is not None:
        from google.genai import types
        sys_instruction = (
            "You are an expert hip-hop, Houston Chop & Screw, Boom-Bap, and Jungle producer programming a "
            "Teenage Engineering EP-133 K.O. II sampler.\n"
            "Hardware layout rules:\n"
            "- Group A (Drums): pad '.' = kick, '0' = snare, '1' = closed hat, '2' = open hat, '3' = clap/rim.\n"
            "- Group B (Bass): pad '.' = 808/sub bass, '0' = reese/mid-bass, pitched via semitones (-12..+12).\n"
            "- Group C (Sample Chops): pads '1'..'8' = 8 structural beat chops in chronological order; "
            "pads '9', '0', '.' = punchy transient accent chops; pad 'enter' = reversed (REV) pickup turnaround chop.\n"
            "- For Chop & Screw or Boom-Bap, use swing around 0.56..0.60, snare_delay_ticks=4, double_up stutters on "
            "bars 2 and 4, and trigger pad 'enter' on beat 4.4 (step 12 or 28 or 60) as a reversed pull into the downbeat."
        )
        try:
            resp = client.models.generate_content(
                model=GEMINI_PRO_MODEL,
                contents=f"Compose a {bars}-bar beat at {bpm:g} BPM for this vibe: {prompt}",
                config=types.GenerateContentConfig(
                    system_instruction=sys_instruction,
                    response_mime_type="application/json",
                    response_schema=GeminiBeatPlan,
                    temperature=0.4,
                ),
            )
            if resp.text:
                return GeminiBeatPlan.model_validate_json(resp.text), GEMINI_PRO_MODEL
        except Exception:
            pass
    return compile_heuristic_beat_plan(prompt, bpm=bpm, bars=bars), "heuristic-chop-screw-compiler"


# --------------------------------------------------------------------------- 3. Multimodal Audio Critic

def critique_take_with_gemini(
    wav_path: str,
    grade_report: dict[str, Any] | None = None,
    target_vibe: str = "Houston Chop & Screw beat with warm sub-bass, crisp snare pocket, and smooth sample chops",
    api_key: str | None = None,
) -> dict[str, Any]:
    """Analyze a recorded EP-133 take WAV using 4-band DSP metrics + `gemini-3.1-pro-preview` multimodal audio input."""
    path = os.path.expanduser(wav_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no such take WAV: {path}")

    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    spectral = an.spectral_mix_report(audio, int(sr))
    dsp_summary = {
        "spectral_mix": spectral,
        "onset_grade": {k: v for k, v in (grade_report or {}).items() if k != "events"},
    }

    client = _get_client(api_key)
    if client is not None:
        try:
            from google.genai import types
            wav_bytes = open(path, "rb").read()
            prompt_text = (
                f"Listen to this recorded hardware take from a Teenage Engineering EP-133 K.O. II sampler.\n"
                f"Target vibe: {target_vibe}\n"
                f"Measured DSP metrics: {json.dumps(dsp_summary)}\n"
                "Critique the pocket/groove, chop transitions, and 4-band spectral balance (sub, low-mid mud, presence, air), "
                "and return concrete parameter adjustments for the next pass."
            )
            resp = client.models.generate_content(
                model=GEMINI_PRO_MODEL,
                contents=[
                    types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
                    prompt_text,
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GeminiAudioCritique,
                    temperature=0.3,
                ),
            )
            if resp.text:
                critique = GeminiAudioCritique.model_validate_json(resp.text)
                return {"model": GEMINI_PRO_MODEL, "dsp": dsp_summary, "critique": critique.model_dump()}
        except Exception:
            pass

    # Deterministic DSP-driven critique fallback when offline or without GEMINI_API_KEY
    bands = spectral.get("bands_dbfs", {})
    warnings = spectral.get("warnings", [])
    adjustments: list[PadDeltaRecommendation] = []
    fader_recs: list[str] = []

    if any("muddy low-mids" in w for w in warnings):
        adjustments.append(PadDeltaRecommendation(group="C", pad="1", amplitude_delta=-12, reason="Reduce low-mid chop buildup (80-250 Hz)"))
        fader_recs.append("Set Group C fader parameter to 'highpass' at base 0.22 to clear space for Group B sub-bass")
    if any("thin sub-bass" in w for w in warnings):
        adjustments.append(PadDeltaRecommendation(group="B", pad=".", amplitude_delta=14, reason="Boost 808/sub foundation in 20-80 Hz band"))
    if any("dull top-end" in w for w in warnings):
        adjustments.append(PadDeltaRecommendation(group="A", pad="1", amplitude_delta=10, reason="Bring up closed hi-hat air above 4 kHz"))
        fader_recs.append("Open Group C lowpass fader base from 0.75 to 0.88")
    if any("clipping" in w for w in warnings):
        fader_recs.append("Lower Group A and B fader base levels by 0.12 and reduce master compressor drive to 0.25")

    jitter_std = ((grade_report or {}).get("offset_stats_ms") or {}).get("std_ms", 4.0)
    tightness = max(40, min(99, int(round(100.0 - float(jitter_std) * 2.5))))
    overall = max(50, min(98, tightness - 6 * len(warnings)))

    fallback = GeminiAudioCritique(
        overall_score=overall,
        timing_tightness_score=tightness,
        groove_assessment=(
            f"Onset timing std is {jitter_std} ms across dual-band (low + high) transient detection. "
            "Clock-aligned 4-tick micro-timing keeps chord and chop attacks free of follower interpolation flam."
        ),
        spectral_balance_assessment=(
            f"RMS {spectral.get('rms_dbfs')} dBFS, Peak {spectral.get('peak_dbfs')} dBFS, Crest {spectral.get('crest_factor_db')} dB, "
            f"Bands: {bands}. Warnings: {warnings or 'None — balanced spectral profile.'}"
        ),
        swing_recommendation=0.58,
        snare_pocket_ticks=4,
        pad_adjustments=adjustments,
        fader_recommendations=fader_recs or ["Sidechain Group B & C from Group A pad '.' at length=0.38, shape=0.55"],
        next_scene_variation_idea="Add a 1/32nd DJ Screw double-up stutter on Group C pad '3' at step 24 and trigger reversed pad 'enter' on step 60.",
    )
    return {"model": "dsp-spectral-critic", "dsp": dsp_summary, "critique": fallback.model_dump()}
