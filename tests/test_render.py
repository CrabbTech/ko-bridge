"""render_song: a whole song recorded from the Mac, position by position, with nobody at the device."""
import pytest

from ko_bridge import server as srv

from .conftest import needs_device

pytestmark = needs_device


def test_render_one_position(scratch):
    desc = srv.describe_project(scratch)
    if not desc.get("scenes"):
        pytest.skip("scratch project has no scenes")
    cursor = desc.get("scene_cursor") or 1
    try:
        r = srv.render_song(scratch, positions=[cursor], name="test-render", wait=True)
    except Exception as exc:
        if "set MIDI clock to receive" in str(exc):
            pytest.skip(f"device is on internal clock: {exc}")
        raise
    assert r["positions"] and r["positions"][0]["midi_note_ons"] > 0
    assert r["song"]["seconds"] == pytest.approx(r["positions"][0]["bars"] * 240.0 / r["bpm"], abs=0.05)
    assert srv.get_take(r["song"]["take_id"])["kind"] == "song"
