"""The companion sample library: index from the backup (no device), search and labels; load/delete on the device."""
import numpy as np
import pytest
import soundfile as sf

from ko_bridge import library as L
from ko_bridge import server as srv
from ko_bridge.ep133 import DELETE_ACK

from .conftest import needs_device


@pytest.fixture(scope="module")
def lib():
    lib = L.Library()
    if not lib.records:
        lib.rebuild_from_backup()
    from .conftest import device_allowed
    if device_allowed() and lib.get(600) is None:
        srv.sync_library(wait=True)
        lib = L.Library()
    return lib


def test_index_covers_the_device_and_labels_make_sense(lib):
    st = lib.status()
    assert st["records"] >= 600 and st["on_device"] >= 600
    kicks = lib.search(labels=["kick"], bank="kick", limit=200)
    assert len(kicks) >= 25 and all(k["category"] == "kick" for k in kicks)
    hats = lib.search(labels=["hat_closed"], limit=100)
    assert any(h["name"] == "closed hat mid" for h in hats)
    reese = lib.search(query="reese")
    assert reese and reese[0]["slot"] == 429 and reese[0]["pitch"].startswith("G")
    pitched_e = lib.search(pitched=True, pitch_class="E", limit=80)
    assert any(r["slot"] == 542 for r in pitched_e), "liquid chord em7 should read as E"
    labels = lib.labels()["labels"]
    assert labels["kick"]["count"] >= 25 and "bank:kick" in labels


def test_search_and_tags_over_the_server(lib):
    rows = srv.search_sounds(query="snare", max_seconds=0.3, limit=5)
    assert rows and all(r["seconds"] <= 0.3 for r in rows)
    srv.tag_sound("111", add=["jungle-snare"])
    assert srv.search_sounds(labels=["jungle-snare"])[0]["slot"] == 111
    srv.tag_sound("111", remove=["jungle-snare"])
    assert not srv.search_sounds(labels=["jungle-snare"])
    if lib.get(600) is not None:
        assert srv.describe_library_sound("600")["features"]["pitch_class"] == "A", "the spike's 880 Hz test tone"


def test_import_then_load_and_delete(tmp_path):
    sr = 48000
    t = np.arange(int(0.3 * sr)) / sr
    x = (np.random.default_rng(3).uniform(-0.8, 0.8, len(t)) * np.exp(-t / 0.03)).astype(np.float32)   # a noisy, decaying hit
    wav = tmp_path / "ko-lib-hit.wav"
    sf.write(wav, x, sr, subtype="PCM_16")
    rows = srv.import_sounds([str(wav)], tags=["test"], origin="synthesised in tests", license_note="none needed")
    assert rows and rows[0]["slot"] is None and "test" in rows[0]["tags"]
    rid = rows[0]["id"]
    assert srv.search_sounds(labels=["test"], on_device=False)[0]["id"] == rid
    from .conftest import device_allowed
    if not device_allowed():
        pytest.skip("device not allowed for this run (KO_BRIDGE_DEVICE_TESTS=1 and a connected EP-133): load/delete not exercised")
    r = srv.load_sound(rid, wait=True)
    assert r["identical"] and r["slot"] >= 602
    assert srv.describe_library_sound(str(r["slot"]))["category"] in ("hat_closed", "snare", "perc", "hat_open")
    d = srv.delete_sound(r["slot"], acknowledge=DELETE_ACK.format(slot=r["slot"]))
    assert d["deleted"]
    assert L.Library().get(r["slot"]) is None
