"""Plain MIDI to and from the EP-133: notes, transport, a clock driver, and a timestamped input capture.

SysEx FILE traffic lives in epsysex.FileClient, which opens its own port handles; CoreMIDI delivers to every client, so a
clock driver here can run while a FILE session is open on the same endpoint.
"""
from __future__ import annotations

import threading
import time

import mido

DEFAULT_NAME = "EP-133"


def find_port(names: list[str], hint: str = DEFAULT_NAME) -> str | None:
    for n in names:
        if hint.lower() in n.lower():
            return n
    return None


class Midi:
    def __init__(self, hint: str = DEFAULT_NAME, channel: int = 0, on_message=None):
        self.on_message_hook = on_message  # called from the input thread with (perf_counter, message)
        self.out_name = find_port(mido.get_output_names(), hint)
        self.in_name = find_port(mido.get_input_names(), hint)
        if self.out_name is None or self.in_name is None:
            raise RuntimeError(f"no MIDI port matching {hint!r}: in={mido.get_input_names()} out={mido.get_output_names()}")
        self.channel = channel
        self._out = None
        self._in = None
        self._capture: list[tuple[float, mido.Message]] | None = None
        self._lock = threading.Lock()

    def open(self) -> "Midi":
        self._out = mido.open_output(self.out_name)
        self._in = mido.open_input(self.in_name, callback=self._on_message)
        return self

    def close(self) -> None:
        if self._in is not None:
            self._in.close()
            self._in = None
        if self._out is not None:
            self._out.close()
            self._out = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    def _on_message(self, msg: mido.Message) -> None:
        if msg.type == "sysex":
            return
        now = time.perf_counter()
        with self._lock:
            if self._capture is not None:
                self._capture.append((now, msg))
        if self.on_message_hook is not None:
            self.on_message_hook(now, msg)

    def capture_start(self) -> None:
        with self._lock:
            self._capture = []

    def capture_stop(self) -> list[tuple[float, mido.Message]]:
        with self._lock:
            got, self._capture = self._capture or [], None
        return got

    def send(self, msg: mido.Message) -> float:
        t = time.perf_counter()
        self._out.send(msg)
        return t

    def note_on(self, note: int, velocity: int = 100, channel: int | None = None) -> float:
        return self.send(mido.Message("note_on", note=note, velocity=velocity, channel=self.channel if channel is None else channel))

    def note_off(self, note: int, channel: int | None = None) -> float:
        return self.send(mido.Message("note_off", note=note, velocity=0, channel=self.channel if channel is None else channel))

    def start(self) -> float:
        return self.send(mido.Message("start"))

    def stop(self) -> float:
        return self.send(mido.Message("stop"))

    def clock(self) -> float:
        return self.send(mido.Message("clock"))

    def all_notes_off(self) -> None:
        for ch in range(16):
            self.send(mido.Message("control_change", control=123, value=0, channel=ch))


class ClockDriver(threading.Thread):
    """Sends MIDI start, then 24-PPQN clocks at `bpm`, until `clocks` have gone out or `halt()` is called.

    `started_at` is the perf_counter time of the start message; `clock_times` are the send times of every clock, so
    sequencer events can be placed against the clock the Mac actually delivered. `send_stop` controls whether a stop
    message follows the last clock (the EP chokes ringing voices on stop).
    """

    def __init__(self, midi: Midi, bpm: float, clocks: int | None = None, send_stop: bool = True, on_start=None):
        super().__init__(name="ko-bridge-clock", daemon=True)
        self.midi = midi
        self.bpm = float(bpm)
        self.clocks = clocks
        self.send_stop = send_stop
        self.on_start = on_start
        self.started_at: float | None = None
        self.stopped_at: float | None = None
        self.clock_times: list[float] = []
        self._halt = threading.Event()
        self._ready = threading.Event()

    def halt(self) -> None:
        self._halt.set()

    def wait_started(self, timeout: float = 5.0) -> bool:
        return self._ready.wait(timeout)

    def run(self) -> None:
        interval = 60.0 / (self.bpm * 24.0)
        if self.on_start is not None:
            self.on_start()
        self.started_at = self.midi.start()
        self._ready.set()
        next_clock = self.started_at
        sent = 0
        while not self._halt.is_set() and (self.clocks is None or sent < self.clocks):
            while True:
                remaining = next_clock - time.perf_counter()
                if remaining <= 0:
                    break
                time.sleep(min(0.0005, remaining))
            self.clock_times.append(self.midi.clock())
            sent += 1
            next_clock += interval
        if self.send_stop:
            self.stopped_at = self.midi.stop()
