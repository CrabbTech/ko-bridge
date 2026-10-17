"""Playing the scratch project from the Mac and listening over USB audio: live notes with latency, clocked playback."""
import time

import pytest

from ko_bridge import server as srv
from ko_bridge.ep133 import EP133
from ko_bridge.project import Lane, LanePoint, NoteEvent, PadSpec, PatternSpec, ProjectSpec, SceneSpec

from .conftest import needs_device

pytestmark = needs_device

# The spike's test pattern: the two test tones on C "." and "0", ten hits over four bars, a low-pass lane closing for bar 2.
TEST_TONES = {".": 600, "0": 601}
HITS = [(0, ".", 120), (192, "0", 100), (384, ".", 120), (576, "0", 100), (768, ".", 120), (864, ".", 90), (960, "0", 100),
        (1152, ".", 120), (1344, "0", 100), (1440, "0", 80)]


@pytest.fixture(scope="module")
def playing(scratch):
    """The scratch slot loaded with the spike's two-pad test pattern, whatever it held before (restored from the write's
    checkpoint afterwards): the tests must not depend on what the last song left in the slot."""
    ep = EP133(lock_owner="ko-bridge tests")
    active_before = ep.active_project().get("project")
    occupied = {int(n["id"]) for n in ep.list_sounds()}
    if not set(TEST_TONES.values()) <= occupied:
        pytest.skip("the test tones are missing from slots 600/601; run scripts/upload_sounds.py first")
    spec = ProjectSpec(
        bpm=120,
        pads=[PadSpec(group="C", pad=p, slot=s) for p, s in TEST_TONES.items()],
        patterns=[PatternSpec(group="C", number=1, bars=4, events=[NoteEvent(pad=p, tick=t, velocity=v) for t, p, v in HITS],
                              lanes=[Lane(parameter="lowpass", points=[LanePoint(tick=0, value=1.0), LanePoint(tick=384, value=0.43),
                                                                         LanePoint(tick=768, value=1.0)])]),
                  PatternSpec(group="A", number=1, bars=1), PatternSpec(group="B", number=1, bars=1), PatternSpec(group="D", number=1, bars=1)],
        scene=SceneSpec(patterns={"A": 1, "B": 1, "C": 1, "D": 1}))
    srv.write_project(scratch, spec, reload=True, wait=True)
    time.sleep(srv.SETTLE_SECONDS)
    yield scratch
    srv.restore_checkpoint(scratch)
    if active_before and active_before != scratch:
        ep.select_project(active_before)


def test_play_notes_measures_latency(playing):
    notes = [{"time_s": i * 0.6, "group": "C", "pad": "." if i % 2 == 0 else "0", "velocity": 110, "duration_s": 0.2} for i in range(6)]
    r = srv.play_notes(notes, record=True, name="test-notes", wait=True)
    assert len(r["sent"]) == 6
    lat = r["latency_stats_ms"]
    assert lat["n"] >= 5, f"onsets missing: {r}"
    assert 0.5 < lat["mean_ms"] < 12.0 and lat["std_ms"] < 2.0, lat
    assert "warning" not in r, r.get("warning")
    take = srv.get_take(r["take"]["take_id"])
    assert take["peak_dbfs"] > -20


def test_play_pattern_lands_on_the_grid(playing):
    desc = srv.describe_project(playing)
    assert desc.get("scenes"), "the scratch project needs a scene with a pattern (scripts/pattern_play.py wrote one)"
    r = srv.play_pattern(bpm=120, bars=2, record=True, name="test-pattern", wait=True)
    assert r["midi_note_ons"] > 0, "no notes came back: set MIDI clock to receive (SHIFT+ERASE, 1 0 1, MAIN) and retry"
    assert "warning" not in r and "warning" not in r["take"], (r.get("warning"), r["take"].get("warning"))
    grade = r["grade"]
    assert grade["expected"] > 0 and grade["matched"] >= grade["expected"] - 1, grade
    assert grade["offset_stats_ms"]["mean_ms"] < 10 and grade["jitter_relative_to_first_ms"]["p2p_ms"] < 5, grade
    assert srv.get_status()["settings_per_human"]["midi_clock"] == "in"
