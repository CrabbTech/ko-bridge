"""USB audio capture from the EP-133 (class-compliant since OS 2.5), with sample-accurate marks.

Copied in shape from op-bridge's device.Recorder: the first callback pins the stream's ADC clock to a sample index, and
`mark(label)` stores the sample that corresponds to "now", so a MIDI message sent right after the mark can be lined up
with the audio to well under a block. `duplex=True` opens a full-duplex stream (silence out) for devices that only
deliver input while an output stream is open.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np
import sounddevice as sd

DEFAULT_NAME = "EP-133"


def audio_devices() -> list[dict]:
    out = []
    for i, d in enumerate(sd.query_devices()):
        out.append({"index": i, "name": d["name"], "inputs": d["max_input_channels"], "outputs": d["max_output_channels"],
                    "samplerate": d["default_samplerate"]})
    return out


def find_audio_input(hint: str = DEFAULT_NAME) -> dict | None:
    for d in audio_devices():
        if hint.lower() in d["name"].lower() and d["inputs"] > 0:
            return d
    return None


@dataclass
class Recording:
    audio: np.ndarray  # frames x channels, float32 in [-1, 1]
    samplerate: int
    marks: list[tuple[str, int]] = field(default_factory=list)
    statuses: list[str] = field(default_factory=list)

    @property
    def seconds(self) -> float:
        return len(self.audio) / self.samplerate

    def mark_sample(self, label: str) -> int | None:
        for name, idx in self.marks:
            if name == label:
                return idx
        return None

    def save(self, path: str) -> str:
        import soundfile as sf

        sf.write(path, self.audio, self.samplerate, subtype="PCM_24")
        return path


class Recorder:
    def __init__(self, device: int, channels: int = 2, samplerate: int = 48000, blocksize: int = 128, duplex: bool = False):
        self.device = device
        self.channels = channels
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.duplex = duplex
        self._blocks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._stream = None
        self._t0_adc: float | None = None
        self._use_perf = False
        self.marks: list[tuple[str, int]] = []
        self.statuses: list[str] = []

    def _ingest(self, indata, time_info, status):
        if status:
            self.statuses.append(str(status))
        with self._lock:
            if self._t0_adc is None:
                adc = float(time_info.inputBufferAdcTime) if time_info is not None else 0.0
                if adc > 0:
                    self._t0_adc = adc
                else:
                    self._use_perf = True
                    self._t0_adc = time.perf_counter()
            self._blocks.append(indata.copy())

    def _callback(self, indata, frames, time_info, status):
        self._ingest(indata, time_info, status)

    def _duplex_callback(self, indata, outdata, frames, time_info, status):
        outdata.fill(0)
        self._ingest(indata, time_info, status)

    def start(self) -> "Recorder":
        if self.duplex:
            self._stream = sd.Stream(device=(self.device, self.device), channels=(self.channels, 2), samplerate=self.samplerate,
                                     blocksize=self.blocksize, dtype="float32", callback=self._duplex_callback)
        else:
            self._stream = sd.InputStream(device=self.device, channels=self.channels, samplerate=self.samplerate,
                                          blocksize=self.blocksize, dtype="float32", callback=self._callback)
        self._stream.start()
        deadline = time.perf_counter() + 2.0
        while time.perf_counter() < deadline:
            with self._lock:
                if self._t0_adc is not None:
                    return self
            time.sleep(0.002)
        self.stop()
        raise RuntimeError("audio input never delivered a block")

    def now(self) -> float:
        if self._use_perf or self._stream is None:
            return time.perf_counter()
        return float(self._stream.time)

    def sample_at(self, t: float) -> int:
        assert self._t0_adc is not None
        return int(round((t - self._t0_adc) * self.samplerate))

    def mark(self, label: str) -> int:
        idx = self.sample_at(self.now())
        self.marks.append((label, idx))
        return idx

    def stop(self) -> Recording:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()
        with self._lock:
            audio = np.concatenate(self._blocks, axis=0) if self._blocks else np.zeros((0, self.channels), np.float32)
        return Recording(audio=audio, samplerate=self.samplerate, marks=list(self.marks), statuses=list(self.statuses))

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
