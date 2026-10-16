"""The sample library on the real device: list, describe, upload a short WAV into a free slot, delete it again."""
import numpy as np
import pytest
import soundfile as sf

from ko_bridge import server as srv
from ko_bridge.ep133 import DELETE_ACK

from .conftest import needs_device

pytestmark = needs_device


def test_list_and_describe_sounds():
    lib = srv.list_sounds(refresh=True)
    assert lib["occupied"] >= 600 and lib["total_mb"] > 50
    slots = {s["slot"] for s in lib["sounds"]}
    assert 1 in slots and 600 in slots, "slot 600 holds the spike's test sample"
    assert all(s >= 600 and s not in slots for s in lib["free_slots_600_plus"])
    info = srv.describe_sound(600)
    assert info["samplerate"] == 46875 and info["channels"] == 1 and info["frames"] == 93750 and info["seconds"] == pytest.approx(2.0)
    srv.audition_sound(600)


def test_upload_verify_and_delete(tmp_path):
    sr = 44100
    t = np.arange(int(0.4 * sr)) / sr
    x = (0.5 * np.sin(2 * np.pi * 660 * t) * np.exp(-t / 0.1)).astype(np.float32)
    wav = tmp_path / "blip.wav"
    sf.write(wav, x, sr, subtype="PCM_16")
    r = srv.upload_sound(str(wav), name="ko test blip", samplerate=46875, wait=True)
    assert r["identical"] is True and r["slot"] >= 602 and r["frames"] == round(0.4 * 46875)
    assert r["source"]["source_samplerate"] == 44100 and r["upload_kbps"] > 10
    info = srv.describe_sound(r["slot"])
    assert info["name"] == "ko test blip" and info["frames"] == r["frames"]
    with pytest.raises(Exception):
        srv.delete_sound(r["slot"], acknowledge="please")
    d = srv.delete_sound(r["slot"], acknowledge=DELETE_ACK.format(slot=r["slot"]))
    assert d["deleted"] is True
    assert r["slot"] in srv.list_sounds(refresh=True)["free_slots_600_plus"]
