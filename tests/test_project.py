"""Project writes on the scratch slot: describe, write a pattern with a fader lane, read back, restore the checkpoint."""
import pytest

from ko_bridge import server as srv
from ko_bridge.ep133 import EP133

from .conftest import needs_device

pytestmark = needs_device


@pytest.fixture(scope="module")
def before(scratch):
    """The scratch slot's archive before the module, restored afterwards."""
    ep = EP133(lock_owner="ko-bridge tests")
    path, data, _s = ep.checkpoint(scratch, "before-tests")
    yield data
    ep.write_project(scratch, data, "after-tests", reload=True)


def test_describe_scratch(scratch, before):
    desc = srv.describe_project(scratch)
    assert desc["project"] == scratch and desc["bytes"] == len(before)
    assert "pads" in desc and "patterns" in desc


def test_write_pattern_and_restore(scratch, before):
    pattern = {"group": "C", "number": 2, "bars": 2,
               "events": [{"pad": ".", "step": 0, "velocity": 120}, {"pad": "0", "step": 4}, {"pad": ".", "step": 8}, {"pad": "0", "step": 12, "velocity": 80},
                          {"pad": ".", "step": 16}, {"pad": "0", "step": 20}, {"pad": ".", "step": 24}, {"pad": "0", "step": 28}],
               "lanes": [{"parameter": "highpass", "points": [{"step": 0, "value": 0.0}, {"step": 16, "value": 0.6}]}]}
    r = srv.write_pattern(scratch, pattern, bpm=110, wait=True)
    assert r["identical"] is True and r["project"] == scratch
    after = r["project_after"]
    assert after["bpm"] == 110.0 and after["scenes"][0]["patterns"]["C"] == 2 and after["scene_cursor"] == 1
    c02 = after["patterns"]["C02"]
    assert len(c02["events"]) == 8 and c02["lanes"]["highpass"][1]["value"] == pytest.approx(0.6, abs=1e-3)
    desc = srv.describe_project(scratch)
    assert desc["patterns"]["C02"]["events"][0]["pad"] == "." and desc["bpm"] == 110.0
    cps = srv.list_checkpoints(scratch)
    assert cps and cps[-1]["label"].startswith("before-C02")
    restored = srv.restore_checkpoint(scratch)
    assert restored["identical"] is True
    assert EP133().read_project(scratch)[0] == before or srv.describe_project(scratch)["bpm"] != 110.0


def test_write_refuses_empty_sound_slot(scratch, before):
    lib = srv.list_sounds(refresh=False)
    free = lib["free_slots_600_plus"][-1]
    with pytest.raises(Exception, match="empty on the device"):
        srv.assign_pads(scratch, [{"group": "D", "pad": "9", "slot": free}], wait=True)
