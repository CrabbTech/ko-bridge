"""A real factory sample read from the device, chopped across the 12 pads of one group in the scratch project from ONE
shared slot, and what the device then does with it: the trims (start + LENGTH) survive the write, the reversed pickup
pad plays backwards, the shared mute group (choke) cuts the previous chop, and clocked chops land on the grid in the
high band. The chopper's slice arithmetic and the groove helpers' swing arithmetic are asserted on the very slices and
events that are written and played, not on synthetic input."""
import numpy as np
import pytest
import soundfile as sf

from ko_bridge import analysis as an
from ko_bridge import groove as GR
from ko_bridge import server as srv
from ko_bridge.ep133 import DELETE_ACK, EP133
from ko_bridge.project import STEP_TICKS, NoteEvent, PatternSpec, ProjectSpec, SceneSpec

from .conftest import needs_device

pytestmark = needs_device

# Factory "sitar one shot" (docs/library-catalog.md): 3.3 s stereo with a bright attack and a long decay, so the grid chops
# start with high-band transients, the loudest and quietest grid slices sit ~36 dB apart (the choke check) and the pickup
# slice decays ~9 dB within itself (the reverse check). The spike's test tone in slot 600 stands in when it is missing.
SOURCE_SLOT = 549
FALLBACK_SLOT = 600
GROUP = "C"
BPM = 90.0
BARS = 2
CHOP_PADS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", ".", "enter"]


@pytest.fixture(scope="module")
def chopped(scratch, tmp_path_factory):
    """The source sample read from the device and chopped with chop_sample into the scratch project (one shared slot), then a
    two-bar swung chop pattern written over it. Afterwards the project is written back from the archive read before and the
    uploaded slot is deleted."""
    ep = EP133(lock_owner="ko-bridge tests")
    active_before = ep.active_project().get("project")
    _path, before, _s = ep.checkpoint(scratch, "before-chop-tests")
    occupied = {int(n["id"]) for n in ep.list_sounds()}
    slot = SOURCE_SLOT if SOURCE_SLOT in occupied else FALLBACK_SLOT
    if slot not in occupied:
        pytest.skip(f"neither slot {SOURCE_SLOT} nor {FALLBACK_SLOT} holds a sample")
    upload_slot = ep.free_slots(occupied)[0]
    info = ep.sound_info(slot)
    pcm, _meta = ep.client.read_sound(slot)
    source = np.frombuffer(pcm, dtype="<i2").reshape(-1, info["channels"]).astype(np.float32) / 32768.0
    wav = tmp_path_factory.mktemp("chop") / f"factory-{slot}.wav"
    sf.write(wav, source, info["samplerate"], subtype="PCM_16")
    state: dict = {"slot": slot, "source": source, "source_info": info}
    try:
        state["chop"] = srv.chop_sample(str(wav), group=GROUP, mode="hybrid", num_slices=12, source_bpm=BPM, samplerate=info["samplerate"],
                                        mono=False, reverse_last_pad=True, shared_slot=True, project=scratch, wait=True)
        # the chop pattern: pads 1..8 on the odd 16ths of two bars, swung MPC-style (58 % = one MIDI clock late)
        raw = [NoteEvent(pad=str(i + 1), step=1 + 4 * i, velocity=110) for i in range(8)]
        state["events"] = GR.apply_mpc_swing(raw, swing=0.58, bars=BARS, snap_clock=True)
        spec = ProjectSpec(bpm=BPM,
                           patterns=[PatternSpec(group=GROUP, number=1, bars=BARS, events=state["events"]),
                                     PatternSpec(group="A", number=1, bars=1), PatternSpec(group="B", number=1, bars=1),
                                     PatternSpec(group="D", number=1, bars=1)],
                           scene=SceneSpec(patterns={"A": 1, "B": 1, GROUP: 1, "D": 1}))
        state["pattern_write"] = srv.write_project(scratch, spec, reload=True, wait=True)
        yield state
    finally:
        ep.write_project(scratch, before, "after-chop-tests", reload=True)
        now = {int(n["id"]) for n in ep.list_sounds()}
        if upload_slot in now and upload_slot not in occupied:
            srv.delete_sound(upload_slot, acknowledge=DELETE_ACK.format(slot=upload_slot))
        if active_before and active_before != scratch:
            ep.select_project(active_before)


def test_chop_pads_share_one_slot_and_trims_stay_inside_the_sample(chopped, scratch):
    res = chopped["chop"]
    frames = res["audio_info"]["frames"]
    assert res["uploaded"] is True and res["shared_slot"] is True and res["upload"]["identical"] is True
    assert res["slot"] >= 600 and res["slot"] != chopped["slot"]
    assert srv.describe_sound(res["slot"])["frames"] == frames
    assert frames == chopped["source_info"]["frames"], "the source is already at the device rate, so no frame was added or lost"
    slices, pads = res["slices"], res["pads"]
    assert [s["pad"] for s in slices] == CHOP_PADS and [p["pad"] for p in pads] == CHOP_PADS
    grid = slices[:8]
    assert grid[0]["start_frame"] == 0 and grid[-1]["end_frame"] == frames
    assert all(a["end_frame"] == b["start_frame"] for a, b in zip(grid, grid[1:])), "grid chops tile the loop"
    assert slices[-1]["kind"] == "pickup" and slices[-1]["reverse"] is True and slices[-1]["end_frame"] == frames
    assert all(p["slot"] == res["slot"] and p["choke"] is True for p in pads)
    assert all(p["trim_start"] == s["start_frame"] and p["trim_end"] == s["end_frame"] - s["start_frame"] for p, s in zip(pads, slices))
    assert all(1 <= p["trim_end"] and p["trim_start"] + p["trim_end"] <= frames for p in pads), "trim_end is a LENGTH from trim_start"
    assert max(p["trim_start"] + p["trim_end"] for p in pads) == frames
    assert pads[-1]["time_mode"] == "reverse" and all(p["time_mode"] == "off" for p in pads[:-1])
    # the device kept every pad as written: an over-long trim would have been reset to the whole sample and the readback would differ
    assert res["project_write"]["identical"] is True, res["project_write"]
    desc = srv.describe_project(scratch)
    for p in pads:
        dev = desc["pads"][f"{GROUP}:{p['pad']}"]
        assert (dev["slot"], dev["trim_start"], dev["trim_end"], dev["time_mode"], dev["choke"]) == \
            (res["slot"], p["trim_start"], p["trim_end"], p["time_mode"], True), (p["pad"], dev)


def _rms_db(x: np.ndarray, sr: int, origin: int, start_s: float, end_s: float) -> float:
    return an.rms_db(x[origin + int(round(start_s * sr)): origin + int(round(end_s * sr))])


def test_reversed_pad_sounds_and_choke_cuts_the_previous_chop(chopped):
    slices = {s["pad"]: s for s in chopped["chop"]["slices"]}
    grid = [slices[p] for p in CHOP_PADS[:8]]
    loud = max(grid, key=lambda s: s["rms_db"])
    # the cutter: the loudest grid slice that still sits 12 dB under the loud one, so the drop is unmistakable yet well above the floor
    quieter = [s for s in grid if loud["rms_db"] - s["rms_db"] >= 12]
    assert quieter, "the source's grid slices are too even for a choke level check"
    quiet = max(quieter, key=lambda s: s["rms_db"])
    assert loud["duration_s"] >= 0.32 and quiet["duration_s"] >= 0.25
    pickup = slices["enter"]
    # loud chop, cut 80 ms later by the quiet one (choke); the loud chop alone; the reversed pickup alone
    hits = [(0.0, loud["pad"]), (0.08, quiet["pad"]), (1.4, loud["pad"]), (2.8, "enter")]
    notes = [{"time_s": t, "group": GROUP, "pad": p, "velocity": 127, "duration_s": 0.2} for t, p in hits]
    r = srv.play_notes(notes, record=True, name="test-chops", wait=True)
    assert len(r["sent"]) == 4
    take = r["take"]
    assert "warning" not in take, take.get("warning")
    audio, sr = sf.read(take["wav"], dtype="float32", always_2d=True)
    x = an.mono(audio)
    t0 = int(take["start_sample"])
    at = [t0 + int(round(s["time_s"] * sr)) for s in r["sent"]]
    floor = an.rms_db(x[max(0, t0 - int(0.4 * sr)): t0 - int(0.05 * sr)])
    for k, (_t, pad) in enumerate(hits):
        level = _rms_db(x, sr, at[k], 0.01, min(0.1, slices[pad]["duration_s"]))
        assert level > floor + 12, f"pad {pad} did not sound: {level:.1f} dBFS against a floor of {floor:.1f} dBFS"
    # choke: 120-300 ms after the first onset only the quiet chop may sound; the same window after the loud chop alone is loud
    choked = _rms_db(x, sr, at[0], 0.12, 0.30)
    alone = _rms_db(x, sr, at[2], 0.12, 0.30)
    assert alone - choked >= 8, f"choke did not cut pad {loud['pad']}: {choked:.1f} dBFS with pad {quiet['pad']} fired vs {alone:.1f} dBFS alone"
    # reverse: the pickup slice decays within itself in the source, so played backwards its second half is the louder one
    src = an.mono(chopped["source"])[pickup["start_frame"]: pickup["end_frame"]]
    half = len(src) // 2
    source_decay = an.rms_db(src[:half]) - an.rms_db(src[half:])
    dur = pickup["duration_s"]
    first, second = _rms_db(x, sr, at[3], 0.005, dur / 2), _rms_db(x, sr, at[3], dur / 2, dur - 0.005)
    assert source_decay >= 3, f"the pickup slice of slot {chopped['slot']} does not decay enough ({source_decay:.1f} dB) to tell reverse from forward"
    assert second - first >= min(3.0, source_decay / 2), \
        f"pad enter did not play backwards: first half {first:.1f}, second half {second:.1f} dBFS (the source slice decays {source_decay:.1f} dB)"
    print(f"\nchoke: {alone:.1f} -> {choked:.1f} dBFS; reverse: {first:.1f} -> {second:.1f} dBFS (source decay {source_decay:.1f} dB); floor {floor:.1f} dBFS")


def test_clocked_chops_grade_on_the_high_band(chopped):
    events = chopped["events"]
    # the swing arithmetic on the events that were written: 58 % = one MIDI clock (4 ticks) late on the odd 16ths, 8 % softer
    assert [e.tick_value() for e in events] == [STEP_TICKS * (1 + 4 * i) + 4 for i in range(8)]
    assert all(e.velocity == 101 for e in events)
    assert chopped["pattern_write"]["identical"] is True
    r = srv.play_pattern(bpm=BPM, bars=BARS, record=True, name="test-chop-pattern", wait=True)
    assert r["midi_note_ons"] >= 8, "no notes came back: set MIDI clock to receive (SHIFT+ERASE, 1 0 1, MAIN) and retry"
    assert "warning" not in r and "warning" not in r["take"], (r.get("warning"), r["take"].get("warning"))
    grade = r["grade"]
    assert grade["expected"] == 8, grade
    high = [e for e in grade["events"] if e["band"] == "high"]
    low = [e for e in grade["events"] if e["band"] == "low"]
    # the low band's causal filter smears onsets by tens of ms (docs/review-gemini-musical-upgrades.md): timing claims use the high band only
    min_high = 3 if chopped["slot"] == SOURCE_SLOT else 1
    assert len(high) >= min_high, f"only {len(high)} high-band onsets ({len(low)} low-band): {grade['events']}"
    offsets = [e["offset_ms"] for e in high]
    assert max(abs(o) for o in offsets) < 12 and abs(sum(offsets) / len(offsets)) < 8, offsets
    print(f"\nhigh-band offsets ms: {offsets}; low-band events: {len(low)}; matched {grade['matched']}/{grade['expected']}")
