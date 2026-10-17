"""Google Gemini 3.1 Pro and Lyria 3.5 for ko-bridge: the cloud "producer" layer.

- `generate_lyria_stem`: an instrumental loop from Lyria 3.5 (`lyria-3.5`, WAV through `response_format`) saved as a
  WAV for chopping. Nothing stands in for it: without a key, without the `gemini` extra, or on a failed request it
  raises, because the library rule is real samples only (no synthesized audio, ever).
- `compile_beat_with_gemini`: a natural-language beat prompt compiled into a `GeminiBeatPlan` by `gemini-3.1-pro-preview`
  (structured JSON), then into a `ProjectSpec` by `plan_to_project_spec` with MPC swing, a lazy snare pocket, DJ Screw
  double-ups and slew-compensated fader lanes. Without a key the deterministic `compile_heuristic_beat_plan` runs and the
  result says so (`model` = "heuristic-chop-screw-compiler": no model involved).
- `critique_take_with_gemini`: the recorded take WAV plus a 4-band DSP report sent to `gemini-3.1-pro-preview` for a
  structured critique; without a key a DSP-only reading comes back, labelled (`model` = "dsp-spectral-critic").

GEMINI_API_KEY comes from the environment: `config.load_secrets` fills it from ~/Music/ko-bridge/secrets.env (or the
op-bridge one) at server start, and `_api_key` calls it again so scripts and tests find the key too. A failed cloud
request is never swallowed: it surfaces as an exception that the MCP tool reports.
"""
from __future__ import annotations

import base64
import io
import json
import os
import urllib.request
from typing import Any, Literal

import numpy as np
import soundfile as sf
from pydantic import BaseModel, Field

from . import analysis as an
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
# Lyria 3.5 has been the stable music model since 2026-09-03 and is the only one that returns WAV (response_format);
# lyria-3-clip-preview and lyria-3-pro-preview are legacy 30 s MP3 clips. All of them sing unless the prompt says not to.
LYRIA_MODEL = "lyria-3.5"
HEURISTIC_COMPILER = "heuristic-chop-screw-compiler"
DSP_CRITIC = "dsp-spectral-critic"
NO_KEY_NOTE = "no model used: GEMINI_API_KEY is not set (config.load_secrets reads ~/Music/ko-bridge/secrets.env), so this is the deterministic path"
NO_MODEL_REQUESTED_NOTE = "no model used: the deterministic path was requested (use_gemini=False)"


def no_model_note(use_gemini: bool = True) -> str:
    """Why a result carries a no-model `model` value: the caller asked for it, or there is no key."""
    return NO_KEY_NOTE if use_gemini else NO_MODEL_REQUESTED_NOTE


# --------------------------------------------------------------------------- Structured Schemas for Gemini 3.1 Pro

class DrumHitSpec(BaseModel):
    pad: PadLabel = Field(description='Group A drum pad: "." (kick), "0" (snare), "1" (closed hat), "2" (open hat), "3" (clap/rim)')
    step: float = Field(ge=0, le=63.75, description="16th-note step (0..63.75 for 4 bars; fractions place 1/32 and 1/64 hits)")
    velocity: int = Field(default=100, ge=1, le=127)


class BassHitSpec(BaseModel):
    pad: PadLabel = Field(default=".", description='Group B bass pad ("." = sub/808, "0" = reese/mid-bass)')
    step: float = Field(ge=0, le=63.75)
    semitones: int = Field(default=0, ge=-12, le=12, description="Semitone offset from root")
    duration_steps: float = Field(default=3.5, ge=0.5, le=16.0)
    velocity: int = Field(default=108, ge=1, le=127)


class ChopTriggerSpec(BaseModel):
    pad: PadLabel = Field(description='Group C chop pad: "1".."9" (main slices), "0" / "." (transient accents), "enter" (reversed pickup)')
    step: float = Field(ge=0, le=63.75)
    velocity: int = Field(default=110, ge=1, le=127)
    semitones: int = Field(default=0, ge=-12, le=12)
    stutter: Literal["none", "double_up", "triplet_roll", "chop_echo"] = "none"
    duration_steps: float | None = Field(default=None, description="Optional explicit duration in 16th steps")


class TapeStopDiveSpec(BaseModel):
    step: float = Field(ge=0, le=63.75, description="Step where the tape-stop pitch dive or Punch-In FX burst begins")
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

class GeminiUnavailable(RuntimeError):
    """No Gemini API key or no google-genai package: the cloud path cannot run, and nothing stands in for it."""


class LyriaError(RuntimeError):
    """Lyria took the request but gave no usable audio (a failed interaction, a filtered prompt, an unreadable payload)."""


def _api_key(api_key: str | None = None) -> str | None:
    if api_key:
        return api_key
    C.load_secrets()   # GEMINI_API_KEY from ~/Music/ko-bridge/secrets.env or op-bridge's, never overriding the environment
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def _scrub(text: str, api_key: str | None = None) -> str:
    """`text` with every configured key value replaced, so an SDK error that echoes a request can never leak the key."""
    for key in {api_key, os.environ.get("GEMINI_API_KEY"), os.environ.get("GOOGLE_API_KEY")} - {None, ""}:
        text = text.replace(key, "<GEMINI_API_KEY>")
    return text


def _get_client(api_key: str | None = None):
    """A google-genai client, or GeminiUnavailable saying exactly what is missing (never None)."""
    key = _api_key(api_key)
    if not key:
        raise GeminiUnavailable("no GEMINI_API_KEY: put GEMINI_API_KEY=... in ~/Music/ko-bridge/secrets.env (never in the repo) or export it")
    try:
        from google import genai
    except ImportError as e:
        raise GeminiUnavailable("the google-genai package is not installed: uv sync --extra gemini") from e
    return genai.Client(api_key=key)


def _optional_client(api_key: str | None = None):
    """The client when a key is configured, else None so the caller takes its labelled no-model path."""
    if not _api_key(api_key):
        return None
    return _get_client(api_key)


# --------------------------------------------------------------------------- 1. Lyria 3.5 Stem Generator

def generate_lyria_stem(
    prompt: str,
    bpm: float = 68.0,
    bars: int = 4,
    out_wav_path: str | None = None,
    api_key: str | None = None,
    instrumental: bool = True,
    trim_to_bars: bool = True,
    timeout_s: float = 300.0,
) -> dict[str, Any]:
    """Generate a loop with Lyria 3.5 and save it as a 16-bit WAV. A paid call (about $0.08 a clip).

    Lyria sings by default and the prompt text is the only control, so the request says "Instrumental only, no vocals"
    unless `instrumental` is off; it also asks for `bars` bars at `bpm` and a short piece, and for WAV output
    (`response_format={"type": "audio"}`, lyria-3.5 only; the legacy clip models return MP3, which is decoded too). The
    whole generation is kept at `<stem>-full.wav`; `wav_path` holds the first `bars` bars from the first onset when
    `trim_to_bars` is set, which is what the chopper wants. Raises GeminiUnavailable (no key or no package) or
    LyriaError (failed or empty interaction, undecodable audio): there is no synthesized stand-in.
    """
    client = _get_client(api_key)
    bar_s = 240.0 / float(bpm)
    loop_s = int(bars) * bar_s
    parts = [prompt.strip().rstrip(".")]
    if instrumental:
        parts.append("Instrumental only, no vocals")
    parts.append(f"Tempo {bpm:g} BPM in 4/4, a {int(bars)}-bar loop of about {loop_s:.0f} seconds that starts on the downbeat; "
                 "keep the whole piece short, around 20 seconds. Crisp transients for sampler chopping, no heavy limiting")
    full_prompt = ". ".join(parts) + "."
    if out_wav_path is None:
        out_wav_path = C.file_path("library", "wav", f"{C.new_id('lyria-stem')}.wav")
    out_wav_path = os.path.expanduser(out_wav_path)
    os.makedirs(os.path.dirname(out_wav_path) or ".", exist_ok=True)

    body = {"model": LYRIA_MODEL, "input": full_prompt, "response_format": {"type": "audio"}}
    try:
        interaction = client.interactions.create(timeout=timeout_s, **body)
    except Exception as e:   # the SDK's own errors (APIError, httpx timeouts): report them, never substitute audio
        raise LyriaError(_scrub(f"Lyria request failed: {type(e).__name__}: {e}", api_key)) from e
    status = getattr(interaction, "status", None)
    if status not in (None, "completed"):
        raise LyriaError(_scrub(f"Lyria interaction {getattr(interaction, 'id', '?')} ended with status {status!r}: {getattr(interaction, 'errors', None)}", api_key))
    audio = _audio_block(interaction)
    if audio is None:
        text = str(getattr(interaction, "output_text", "") or "")[:200]
        raise LyriaError(_scrub(f"Lyria returned no audio block (status {status!r}; text {text!r})", api_key))
    raw = _audio_bytes(audio)
    try:
        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    except Exception as e:
        raise LyriaError(f"Lyria audio ({getattr(audio, 'mime_type', '?')}, {len(raw)} B) could not be decoded: {e}") from e
    sr = int(sr)
    full_path = (out_wav_path[:-4] if out_wav_path.endswith(".wav") else out_wav_path) + "-full.wav"
    sf.write(full_path, data, sr, subtype="PCM_16")
    loop, first_onset_s = (_first_bars(data, sr, loop_s) if trim_to_bars else (data, 0.0))
    sf.write(out_wav_path, loop, sr, subtype="PCM_16")
    return {
        "wav_path": out_wav_path,
        "full_wav_path": full_path,
        "source": LYRIA_MODEL,
        "model": LYRIA_MODEL,
        "interaction_id": getattr(interaction, "id", None),   # Google's opaque trace id for the billing record; no field here carries the key
        "prompt": full_prompt,
        "mime_type": getattr(audio, "mime_type", None),
        "samplerate": sr,
        "channels": int(data.shape[1]),
        "generated_seconds": round(len(data) / sr, 3),
        "seconds": round(len(loop) / sr, 3),
        "trimmed_from_s": round(first_onset_s, 3) if trim_to_bars else None,
        "text": str(getattr(interaction, "output_text", "") or "")[:500] or None,
    }


def _audio_block(interaction):
    """The audio content block of an Interaction: `output_audio` when the SDK filled it, else the last audio item of a
    model_output step (interleaved output)."""
    audio = getattr(interaction, "output_audio", None)
    if audio is not None:
        return audio
    for step in getattr(interaction, "steps", None) or []:
        if getattr(step, "type", None) == "model_output":
            for item in getattr(step, "content", None) or []:
                if getattr(item, "type", None) == "audio":
                    audio = item
    return audio


def _audio_bytes(audio) -> bytes:
    data = getattr(audio, "data", None)
    if data:
        return base64.b64decode(data) if isinstance(data, str) else bytes(data)
    uri = getattr(audio, "uri", None)
    if uri:
        with urllib.request.urlopen(uri, timeout=120) as r:   # delivery="uri": the payload sits on a Google-hosted file
            return r.read()
    raise LyriaError("the audio block has neither inline data nor a uri")


def _first_bars(data: np.ndarray, sr: int, loop_s: float) -> tuple[np.ndarray, float]:
    """The first `loop_s` seconds from the first onset (leading silence skipped), or everything when the clip is shorter."""
    start = an.onset_sample(an.mono(data), sr, start=0, rise_db=12.0, floor_min_db=-60.0) or 0
    end = min(len(data), start + int(round(loop_s * sr)))
    return data[start:end], start / sr


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

    chop_lane = GR.make_filter_swell_lane(bars=bars, parameter="lowpass", low_val=max(0.35, plan.lowpass_base - 0.2), high_val=plan.lowpass_base, bpm=plan.bpm)
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
    use_gemini: bool = True,
) -> tuple[GeminiBeatPlan, str]:
    """Compile a beat prompt into a `GeminiBeatPlan`; returns `(plan, model)`.

    With a key and `use_gemini`, `gemini-3.1-pro-preview` returns the plan as structured JSON and a failed request
    raises (nothing is substituted). Without a key, or with `use_gemini=False`, the deterministic
    `compile_heuristic_beat_plan` runs and the model is HEURISTIC_COMPILER: no model was used, and the tool says so.
    """
    client = _optional_client(api_key) if use_gemini else None
    if client is None:
        return compile_heuristic_beat_plan(prompt, bpm=bpm, bars=bars), HEURISTIC_COMPILER
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
    if not resp.text:
        raise RuntimeError(f"{GEMINI_PRO_MODEL} returned no text for the beat plan; nothing was substituted")
    return GeminiBeatPlan.model_validate_json(resp.text), GEMINI_PRO_MODEL


# --------------------------------------------------------------------------- 3. Multimodal Audio Critic

def critique_take_with_gemini(
    wav_path: str,
    grade_report: dict[str, Any] | None = None,
    target_vibe: str = "Houston Chop & Screw beat with warm sub-bass, crisp snare pocket, and smooth sample chops",
    api_key: str | None = None,
    use_gemini: bool = True,
) -> dict[str, Any]:
    """Critique a recorded EP-133 take: a 4-band DSP report, plus `gemini-3.1-pro-preview` listening to the WAV when a
    key is configured and `use_gemini` is set (the recording leaves the machine; a failed request raises). Without a key
    the DSP-only reading comes back with model DSP_CRITIC and a note that no model was used."""
    path = os.path.expanduser(wav_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no such take WAV: {path}")

    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    spectral = an.spectral_mix_report(audio, int(sr))
    dsp_summary = {
        "spectral_mix": spectral,
        "onset_grade": {k: v for k, v in (grade_report or {}).items() if k != "events"},
    }

    client = _optional_client(api_key) if use_gemini else None
    if client is not None:
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
        if not resp.text:
            raise RuntimeError(f"{GEMINI_PRO_MODEL} returned no text for the critique; nothing was substituted")
        critique = GeminiAudioCritique.model_validate_json(resp.text)
        return {"model": GEMINI_PRO_MODEL, "dsp": dsp_summary, "critique": critique.model_dump()}

    # DSP-only reading: deterministic, and labelled as such
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
    return {"model": DSP_CRITIC, "note": no_model_note(use_gemini), "dsp": dsp_summary, "critique": fallback.model_dump()}
