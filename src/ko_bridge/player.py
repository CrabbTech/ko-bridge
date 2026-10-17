"""Playing the EP-133 from the Mac and listening back: clocked pattern playback, timed notes, recordings and grading."""
from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from . import analysis as an
from .audio import Recorder, Recording
from .jobs import check
from .midi import ClockDriver, Midi


def _recorder(audio_device: int | None, samplerate: int = 48000) -> Recorder | None:
    if audio_device is None:
        return None
    return Recorder(audio_device, channels=2, samplerate=samplerate, blocksize=128)


def play_pattern(midi: Midi, bpm: float, bars: int, audio_device: int | None = None, pre_roll: float = 0.7, tail: float = 1.0,
                 stage: Callable[[str], None] | None = None, cancel=None,
                 clock_automation: dict[int, list[dict]] | None = None) -> dict[str, Any]:
    """MIDI START then 24 PPQN clock for `bars` 4/4 bars, STOP at the end; captures the device's MIDI note output and,
    with an audio device, the USB audio. The device must be set to receive MIDI clock (system setting 101)."""
    rec = _recorder(audio_device)
    marks: list[tuple[int, int, int]] = []

    def on_midi(_t, msg):
        if msg.type == "note_on" and msg.velocity > 0 and rec is not None and rec._stream is not None:
            marks.append((rec.mark(f"midi{len(marks)}"), msg.note, msg.velocity))

    midi.on_message_hook = on_midi
    midi.capture_start()
    if rec is not None:
        rec.start()
    try:
        time.sleep(pre_roll)
        check(cancel)
        driver = ClockDriver(midi, bpm, clocks=int(bars) * 96, send_stop=True,
                             on_start=(lambda: rec.mark("start")) if rec else None,
                             clock_automation=clock_automation)
        if stage:
            stage(f"playing {bars} bars at {bpm:g} BPM")
        driver.start()
        while driver.is_alive():
            driver.join(0.2)
            if cancel is not None and cancel.is_set():
                driver.halt()
                driver.join()
                break
        time.sleep(tail)
    finally:
        recording = rec.stop() if rec is not None else None
        events = midi.capture_stop()
        midi.on_message_hook = None
    t0 = driver.started_at or time.perf_counter()
    note_ons = [{"time_s": round(t - t0, 4), "note": m.note, "velocity": m.velocity, "channel": m.channel + 1}
                for t, m in events if m.type == "note_on" and m.velocity > 0]
    out: dict[str, Any] = {"bpm": bpm, "bars": bars, "clocks_sent": len(driver.clock_times), "midi_note_ons": note_ons,
                           "clock_jitter_ms": an.stats_ms([(b - a) * 1000.0 - 60000.0 / (bpm * 24) for a, b in zip(driver.clock_times, driver.clock_times[1:])]),
                           "cancelled": bool(cancel is not None and cancel.is_set())}
    if not note_ons:
        out["note"] = ("the device sent no notes while clocked: it is probably not set to receive MIDI clock "
                       "(human_steps('midi_clock_in'): SHIFT+ERASE, 1 0 1, MAIN), or the active project's scene has no events")
    if recording is not None:
        out["recording"] = recording
        out["start_sample"] = recording.mark_sample("start")
        out["midi_marks"] = marks
    return out


def play_notes(midi: Midi, events: list[dict[str, Any]], audio_device: int | None = None, pre_roll: float = 0.5, tail: float = 1.0,
               cancel=None) -> dict[str, Any]:
    """Send note events at their scheduled times: each has time_s, note, velocity, duration_s (and optional channel)."""
    rec = _recorder(audio_device)
    if rec is not None:
        rec.start()
    sent = []
    try:
        time.sleep(pre_roll)
        t0 = time.perf_counter()
        if rec is not None:
            rec.mark("start")
        pending = sorted(({**e, "off": e["time_s"] + e.get("duration_s", 0.25)} for e in events), key=lambda e: e["time_s"])
        offs: list[tuple[float, int, int]] = []
        i = 0
        while i < len(pending) or offs:
            check(cancel)
            now = time.perf_counter() - t0
            next_on = pending[i]["time_s"] if i < len(pending) else None
            offs.sort()
            next_off = offs[0][0] if offs else None
            if next_off is not None and (next_on is None or next_off <= next_on):
                if next_off > now:
                    time.sleep(min(0.0005, next_off - now))
                    continue
                _t, note, ch = offs.pop(0)
                midi.note_off(note, ch)
                continue
            if next_on is None:
                break
            if next_on > now:
                time.sleep(min(0.0005, next_on - now))
                continue
            e = pending[i]
            i += 1
            ch = int(e.get("channel", 1)) - 1 if e.get("channel") else midi.channel
            if rec is not None:
                rec.mark(f"n{len(sent)}")
            t_sent = midi.note_on(int(e["note"]), int(e.get("velocity", 100)), ch)
            sent.append({"time_s": round(t_sent - t0, 4), "scheduled_s": e["time_s"], "note": int(e["note"]), "velocity": int(e.get("velocity", 100))})
            offs.append((e["off"], int(e["note"]), ch))
        time.sleep(tail)
    finally:
        recording = rec.stop() if rec is not None else None
    out: dict[str, Any] = {"sent": sent}
    if recording is not None:
        out["recording"] = recording
        out["start_sample"] = recording.mark_sample("start")
        sr = recording.samplerate
        hp = an.transients(an.mono(recording.audio), sr) if len(recording.audio) else np.zeros(0)
        latencies = []
        for k in range(len(sent)):
            m = recording.mark_sample(f"n{k}")
            if m is None:
                continue
            sent[k]["onset_latency_ms"] = fine_onset_ms(hp, sr, m)
            if sent[k]["onset_latency_ms"] is not None:
                latencies.append(sent[k]["onset_latency_ms"])
        out["latency_stats_ms"] = an.stats_ms(latencies)
        stuck = stuck_audio(recording)
        if stuck:
            out["warning"] = stuck
    return out


def fine_onset_ms(hp: np.ndarray, sr: int, mark: int, threshold: float = 0.0316, window_s: float = 0.3) -> float | None:
    """Milliseconds from `mark` to the first high-passed sample above -30 dBFS (a broadband attack), or None."""
    seg = np.abs(hp[max(0, mark - int(0.005 * sr)): mark + int(window_s * sr)])
    hits = np.nonzero(seg > threshold)[0]
    if not len(hits):
        return None
    return round((hits[0] - min(mark, int(0.005 * sr))) / sr * 1000.0, 2)


def record(audio_device: int, seconds: float) -> Recording:
    with Recorder(audio_device, channels=2, samplerate=48000, blocksize=256) as rec:
        rec.mark("start")
        time.sleep(seconds)
    return rec.stop()


def grade(recording: Recording, start_sample: int, expected_times_s: list[float], before_ms: float = 40.0, after_ms: float = 160.0) -> dict[str, Any]:
    """Onsets near each expected time across both high-pass (percussive) and low-band (808/kick/warm chop) signals,
    plus 4-band spectral mix balance."""
    sr = recording.samplerate
    x = an.mono(recording.audio)
    hp = an.transients(x, sr) if len(x) else np.zeros(0)
    lp = an.low_transients(x, sr) if len(x) else np.zeros(0)
    expected_samples = [start_sample + int(round(t * sr)) for t in expected_times_s]
    found_hp = an.onsets_near(hp, sr, expected_samples, before_ms=before_ms, after_ms=after_ms) if len(hp) else [None] * len(expected_samples)
    found_lp = an.onsets_near(lp, sr, expected_samples, before_ms=before_ms, after_ms=after_ms, rise_db=9.0) if len(lp) else [None] * len(expected_samples)
    abs_hp = np.abs(hp)
    abs_lp = np.abs(lp)
    offsets = []
    rows = []
    for t, es, fh, fl in zip(expected_times_s, expected_samples, found_hp, found_lp):
        # A genuine high-band transient reaches at least -30 dBFS (0.0316) in the >2.5 kHz high-passed signal
        hp_peak = float(np.max(abs_hp[max(0, fh - 288): fh + int(0.02 * sr)])) if fh is not None else 0.0
        lp_peak = float(np.max(abs_lp[max(0, fl - 288): fl + int(0.04 * sr)])) if fl is not None else 0.0
        use_high = fh is not None and (hp_peak >= 0.0316 or (fl is None) or (hp_peak >= lp_peak * 0.5))
        f = fh if use_high else fl
        band = "high" if use_high else ("low" if fl is not None else None)
        fine = None
        if f is not None:
            if use_high:
                seg = abs_hp[max(0, f - 288): f + 288]
                hits = np.nonzero(seg > 0.0316)[0]
                fine = (max(0, f - 288) + int(hits[0])) if len(hits) else f
            else:
                seg = abs_lp[max(0, f - 480): f + 480]
                hits = np.nonzero(seg > max(0.02, lp_peak * 0.25))[0]
                fine = (max(0, f - 480) + int(hits[0])) if len(hits) else f
            offsets.append((fine - es) / sr * 1000.0)
        rows.append({"expected_s": round(t, 4), "offset_ms": None if f is None else round((fine - es) / sr * 1000.0, 2),
                     "band": band})
    hp_onsets = an.all_onsets(hp, sr) if len(hp) else []
    lp_onsets = an.all_onsets(lp, sr, rise_db=10.0) if len(lp) else []
    merged_onsets = sorted(set(hp_onsets + lp_onsets))
    return {"expected": len(expected_times_s), "matched": len(offsets), "offset_stats_ms": an.stats_ms(offsets),
            "jitter_relative_to_first_ms": an.stats_ms([o - offsets[0] for o in offsets]) if offsets else {"n": 0},
            "events": rows,
            "all_onsets_s": [round((s - start_sample) / sr, 3) for s in merged_onsets],
            "silent_spans_s": [(round(a - start_sample / sr, 2), round(b - start_sample / sr, 2)) for a, b in an.silent_spans(x, sr)] if len(x) else [],
            "peak_dbfs": round(an.peak_db(x), 1) if len(x) else None,
            "spectral_mix": an.spectral_mix_report(recording.audio, sr) if len(x) else {}}


def levels(recording: Recording) -> dict[str, Any]:
    x = recording.audio
    out: dict[str, Any] = {"seconds": round(recording.seconds, 3), "peak_dbfs": round(an.peak_db(x), 1) if len(x) else None,
                           "rms_dbfs": [round(v, 1) for v in an.channel_levels(x)] if len(x) else None, "statuses": recording.statuses[:5]}
    stuck = stuck_audio(recording)
    if stuck:
        out["warning"] = stuck
    return out


def stuck_audio(recording: Recording, window_s: float = 0.1, span_s: float = 1.5, tolerance_db: float = 0.3, floor_dbfs: float = -40.0) -> str | None:
    """A capture whose level stays within `tolerance_db` for `span_s` while loud is a repeating USB packet, not music
    (seen once, 3 s after a project switch: a 49-sample period at constant -9 dBFS)."""
    x = an.mono(recording.audio)
    sr = recording.samplerate
    win = int(window_s * sr)
    if len(x) < win * 2:
        return None
    lev = np.array([an.rms_db(x[i:i + win]) for i in range(0, len(x) - win, win)])
    need = int(span_s / window_s)
    for i in range(0, len(lev) - need):
        seg = lev[i:i + need]
        if seg.min() > floor_dbfs and seg.max() - seg.min() < tolerance_db:
            return (f"audio level constant at {seg.mean():.1f} dBFS for {span_s:.1f} s from {i * window_s:.1f} s: the USB audio stream "
                    "looks stuck on a repeating packet (seen after a project switch); wait a few seconds and retry")
    return None
