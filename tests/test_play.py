"""Playing the scratch project from the Mac and listening over USB audio: live notes with latency, clocked playback."""
import time

import pytest

from ko_bridge import server as srv
from ko_bridge.ep133 import EP133

from .conftest import needs_device

pytestmark = needs_device


@pytest.fixture(scope="module")
def playing(scratch):
    """The scratch slot active with the spike's pattern project (pads C "." and "0" assigned); restored afterwards."""
    ep = EP133(lock_owner="ko-bridge tests")
    active_before = ep.active_project().get("project")
    _p, data, _s = ep.checkpoint(scratch, "before-play-tests")
    desc = srv.describe_project(scratch)
    if "C:." not in desc["pads"]:
        pytest.skip("scratch project has no sample on pad C '.'; run scripts/upload_sounds.py first")
    srv.select_project(scratch)
    time.sleep(srv.SETTLE_SECONDS)
    yield scratch
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
