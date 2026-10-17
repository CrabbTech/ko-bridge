"""Functional tests only: everything that needs the EP-133 skips cleanly when it is not connected over USB, and also when
KO_BRIDGE_DEVICE_TESTS=1 is not set, because several sessions and apps drive the same hardware at once and a device run
must be an explicit choice. No test may spend money: without KO_BRIDGE_PAID_TESTS=1 the Gemini key is removed from the
test process and the secrets files are not consulted, so any cloud path raises GeminiUnavailable instead of calling out.

Writes go to the scratch project (KO_BRIDGE_SCRATCH_PROJECT, default 9, the slot Tyler declared expendable) and to
free sound slots from 600 up; the scratch slot is restored from its checkpoint when the module finishes. The real
KO_BRIDGE_HOME is used so checkpoints of real writes are kept.
"""
from __future__ import annotations

import os

import mido
import pytest

from ko_bridge import config as C
from ko_bridge.audio import find_audio_input
from ko_bridge.midi import find_port

SCRATCH = int(os.environ.get("KO_BRIDGE_SCRATCH_PROJECT", "9"))
DEVICE_OPT_IN = "KO_BRIDGE_DEVICE_TESTS"
PAID_OPT_IN = "KO_BRIDGE_PAID_TESTS"


def device_present() -> bool:
    return find_port(mido.get_output_names(), "EP-133") is not None and find_audio_input("EP-133") is not None


def device_allowed() -> bool:
    """The device may be used by this test run: opted in AND connected."""
    return os.environ.get(DEVICE_OPT_IN) == "1" and device_present()


def _device_skip_reason() -> str:
    if os.environ.get(DEVICE_OPT_IN) != "1":
        return f"device tests run only with {DEVICE_OPT_IN}=1 (several bridge processes share the hardware)"
    return "EP-133 not connected over USB (MIDI and audio)"


needs_device = pytest.mark.skipif(not device_allowed(), reason=_device_skip_reason())


@pytest.fixture(scope="session", autouse=True)
def no_paid_calls():
    """Without KO_BRIDGE_PAID_TESTS=1 no test can reach Gemini or Lyria: the key leaves the environment for the whole run
    and config.load_secrets answers from a report that consulted no file."""
    if os.environ.get(PAID_OPT_IN) == "1":
        yield
        return
    saved = {k: os.environ.pop(k) for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY") if k in os.environ}
    report_before = C._SECRETS_REPORT
    C._SECRETS_REPORT = {"file": None, "loaded": [], "already_set": [], "skipped_lines": 0,
                         "note": f"secrets not consulted by the tests: set {PAID_OPT_IN}=1 to allow paid calls"}
    try:
        yield
    finally:
        C._SECRETS_REPORT = report_before
        os.environ.update(saved)


@pytest.fixture(scope="session")
def scratch() -> int:
    return SCRATCH
