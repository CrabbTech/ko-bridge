"""Tests for the musical upgrades: Auto-Chopper, 24-PPQN Groove Engine, Dual-Band Spectral Grading,
Harmonic Pitch Disambiguation, and Gemini/Lyria Producer tools (no hardware required)."""
from __future__ import annotations

import math
import numpy as np
import soundfile as sf

from ko_bridge import analysis as an
from ko_bridge import chopper as CH
from ko_bridge import gemini_producer as GP
from ko_bridge import groove as GR
from ko_bridge import library as L
from ko_bridge import player
from ko_bridge import server as srv
from ko_bridge.audio import Recording
from ko_bridge.project import LanePoint, NoteEvent


def test_chopper_shared_slot_and_screw_resample(tmp_path):
    sr = 46875
    stem = GP._render_organic_soul_stem(bpm=72.0, bars=2, sr=sr)
    wav_path = tmp_path / "soul_loop.wav"
    sf.write(wav_path, stem, sr, subtype="PCM_16")

    audio, pcm, info = CH.prepare_chopped_audio(
        str(wav_path),
        samplerate=sr,
        mono=False,
        screw_semitones=-3.5,
        source_bpm=72.0,
    )
    # -3.5 semitones slows the loop down -> more frames than the original
    assert info["effective_semitones"] == -3.5
    assert info["speed_ratio"] < 1.0
    assert len(audio) > len(stem)

    slices = CH.detect_chop_slices(audio, sr, num_slices=12, mode="hybrid", reverse_last_pad=True)
    assert len(slices) == 12
    assert [s.pad for s in slices] == list(CH.CHOP_PAD_ORDER)
    assert slices[-1].pad == "enter" and slices[-1].reverse is True

    pads = CH.build_shared_slot_chop_pads(slices, slot=610, group="C", sample_bpm=72.0)
    assert len(pads) == 12
    assert all(p.slot == 610 and p.choke is True for p in pads)
    assert pads[-1].time_mode == "reverse"
    assert all(0 <= p.trim_start < p.trim_end <= len(audio) for p in pads)


def test_groove_clock_alignment_swing_stutters_and_fader_slew():
    # 1. Sub-step events on 1-tick offsets get snapped to 4-tick (24-PPQN MIDI clock) boundaries
    raw = [NoteEvent(pad="1", tick=1), NoteEvent(pad="2", tick=5), NoteEvent(pad="3", tick=24)]
    aligned = GR.align_events_to_clock(raw, bars=1)
    assert [e.tick_value() for e in aligned] == [0, 4, 24]

    # 2. MPC swing delays odd 16th steps (step 1 = tick 24 -> tick 28 at 58% swing)
    grid = [NoteEvent(pad="1", step=0), NoteEvent(pad="1", step=1), NoteEvent(pad="1", step=2), NoteEvent(pad="1", step=3)]
    swung = GR.apply_mpc_swing(grid, swing=0.58, bars=1, snap_clock=True)
    assert [e.tick_value() for e in swung] == [0, 28, 48, 76]
    assert swung[1].velocity < swung[0].velocity

    # 3. DJ Screw double-up creates a 1/32nd (-12 tick = 3 MIDI clocks) ghost pre-hit before the main accent
    stutter = GR.make_screw_double_up(pad="3", step=8.0, velocity=110, style="double_up")
    assert len(stutter) == 2
    assert [e.tick_value() for e in stutter] == [8 * 24 - 12, 8 * 24]
    assert stutter[0].velocity < stutter[1].velocity

    # 4. Fader slew compensation thins 1/16th LFO spam and pre-leads large step jumps by 12 ticks (0.5 step)
    lfo_spam = [LanePoint(step=0.0, value=0.2), LanePoint(step=0.25, value=0.9), LanePoint(step=8.0, value=0.95)]
    comp = GR.compensate_fader_slew(lfo_spam, bars=1, lead_steps=0.5, min_step_gap=1.0)
    assert len(comp) == 2
    assert comp[1].tick_value() == 8 * 24 - 12


def test_dual_band_grading_detects_sub_bass_808_hits():
    # A pure 55 Hz sub-bass 808 hit at t=0.50 s is invisible to a 2.5 kHz high-pass filter,
    # but must be matched by the upgraded dual-band grade() in player.py
    sr = 48000
    dur_s = 1.5
    t = np.arange(int(dur_s * sr)) / sr
    audio = np.zeros((len(t), 2), dtype=np.float32)
    onset_idx = int(0.50 * sr)
    seg_t = np.arange(int(0.4 * sr)) / sr
    sub_hit = (0.8 * np.sin(2 * math.pi * 55.0 * seg_t) * np.exp(-seg_t / 0.18)).astype(np.float32)
    audio[onset_idx: onset_idx + len(sub_hit), 0] = sub_hit
    audio[onset_idx: onset_idx + len(sub_hit), 1] = sub_hit

    rec = Recording(audio=audio, samplerate=sr)
    res = player.grade(rec, start_sample=0, expected_times_s=[0.50])
    assert res["matched"] == 1
    assert res["events"][0]["band"] == "low"
    assert abs(res["events"][0]["offset_ms"]) < 25.0
    assert "sub_20_80hz" in res["spectral_mix"]["bands_dbfs"]
    assert res["spectral_mix"]["bands_dbfs"]["sub_20_80hz"] > -25.0


def test_library_filename_note_hint_and_3rd_harmonic_disambiguation():
    sr = 46875
    t = np.arange(int(0.35 * sr)) / sr
    # Synthesize an E2 (82.41 Hz) bass tone whose 3rd harmonic (B3, 247.2 Hz) dominates so raw autocorrelation
    # peaks on B (+7 semitones), while the name states "Cymatics - Dubstep Bass E.wav"
    f_fund = 82.4069
    sig = (0.25 * np.sin(2 * math.pi * f_fund * t) + 0.85 * np.sin(2 * math.pi * (3 * f_fund) * t)).astype(np.float32)
    feats = L.features(sig, sr, channels=1, name="Cymatics - Dubstep Bass E.wav")
    assert feats["name_pitch_hint"] == "E"
    assert feats["pitch_class"] == "E"
    assert feats["pitch_stable"] is True


def test_gemini_producer_and_mcp_tools_offline(tmp_path):
    res = srv.generate_lyria_chop_group(
        prompt="Houston DJ Screw soulful Rhodes loop at 68 BPM",
        bpm=68.0,
        bars=2,
        group="C",
        screw_semitones=-3.0,
        upload=False,
    )
    assert res["chop"]["uploaded"] is False
    assert len(res["chop"]["slices"]) == 12
    assert len(res["chop"]["pads"]) == 12

    comp = srv.compile_beat_with_gemini(
        prompt="Houston DJ Screw chopped soul beat at 68 BPM with heavy double-up stutters on bar 2 and 4",
        bpm=68.0,
        bars=4,
        write_to_device=False,
    )
    assert comp["project_spec"]["bpm"] == 68.0
    assert len(comp["project_spec"]["patterns"]) >= 3
    assert comp["clock_automation_pulses"] > 0

    crit = GP.critique_take_with_gemini(res["stem"]["wav_path"])
    assert "critique" in crit and 1 <= crit["critique"]["overall_score"] <= 100
