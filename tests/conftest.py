"""Functional tests only: everything that needs the EP-133 skips cleanly when it is not connected over USB.

Writes go to the scratch project (KO_BRIDGE_SCRATCH_PROJECT, default 9, the slot Tyler declared expendable) and to
free sound slots from 600 up; the scratch slot is restored from its checkpoint when the module finishes. The real
KO_BRIDGE_HOME is used so checkpoints of real writes are kept.
"""
from __future__ import annotations

import os

import mido
import pytest

from ko_bridge.audio import find_audio_input
from ko_bridge.midi import find_port

SCRATCH = int(os.environ.get("KO_BRIDGE_SCRATCH_PROJECT", "9"))


def device_present() -> bool:
    return find_port(mido.get_output_names(), "EP-133") is not None and find_audio_input("EP-133") is not None


needs_device = pytest.mark.skipif(not device_present(), reason="EP-133 not connected over USB (MIDI and audio)")


@pytest.fixture(scope="session")
def scratch() -> int:
    return SCRATCH
