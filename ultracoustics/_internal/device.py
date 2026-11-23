"""The box under a Controller: its USB stream, every command, and the manual session.

A :class:`~ultracoustics.Controller` keeps one :class:`Box` private and offers only
the customer's calls on top of it. The SDK's own procedures (the lock, the
self-test, the signal level, a firmware update) drive the Box.
"""
from typing import Optional

import numpy as np

from .box import StreamCommands
from .comms import USBBulkConnection, USBStream
from .control import DAC_CAPS
from .session import SessionCommands


class Box(StreamCommands, SessionCommands):
    """One box: the stream read by its own process into a shared-memory ring, and the
    commands that go out through that process, which alone holds the USB device."""

    def __init__(self, verbose=False, ring_samples: int = 12_000_000,
                 packet_diagnostics_path: Optional[str] = None,
                 packet_diagnostics_attach_s: float = 2.0):
        self.verbose = verbose
        self._stream: Optional[USBStream] = None
        self.device_serial = None
        self._running = False
        self._streaming = False
        self._connected = False
        self._packet_diagnostics_path = packet_diagnostics_path
        self._packet_diagnostics_attach_s = packet_diagnostics_attach_s
        self._manual_transaction = 0
        self._last_manual_rail_off_ns = None
        # Host-side laser DAC ceilings. The 638 value is replaced by the
        # board's own cap (manual channel 7) once read_optical_cap_638() has
        # run; the 1550 exposes no cap, so its compiled limit stands.
        self._dac_caps = dict(DAC_CAPS)
        # Ring capacity in samples. Lives in shared memory once the stream
        # subprocess is spawned (allocated by USBStream), so the parent and
        # reader share it zero-copy.
        self._buf_len = int(ring_samples)

    # -- Connection ----------------------------------------------------------------

    def connect(self):
        """Find the box, hand its USB to the reader process and start the stream."""
        probe = USBBulkConnection(verbose=self.verbose)
        self.device_serial = probe.dev.serial_number
        probe.close()  # release the device so the reader subprocess can claim it
        self._stream = USBStream(
            ring_capacity_samples=self._buf_len,
            verbose=self.verbose,
            packet_diagnostics_path=self._packet_diagnostics_path,
            packet_diagnostics_attach_s=self._packet_diagnostics_attach_s,
        )
        self._connected = True
        try:
            self.begin_stream()
        except Exception:
            self.close()
            raise

    def close(self):
        """Stop the box (confirmed) if it runs, end the stream and release the USB device."""
        shutdown_error = None
        try:
            if self._running or self.system_manual_active:
                self.stop_system_confirmed(timeout_s=1.0)
        except Exception as exc:
            shutdown_error = exc
        finally:
            self.end_stream()
            # Cleared first: a thread still polling sees "not connected", not
            # a stream that is being released under it.
            stream, self._stream = self._stream, None
            if stream is not None:
                stream.close()
            self._connected = False
            self._system_manual_active = False
            self._system_manual_owned = set()
        if shutdown_error is not None:
            raise RuntimeError(f"USB closed; physical shutdown was not confirmed: {shutdown_error}. The lasers may still "
                               "be on: use the box's button or power switch.") from shutdown_error

    @property
    def connected(self) -> bool:
        return self._connected and self._stream is not None

    # -- The ring and the stream ---------------------------------------------------

    @property
    def buffer(self) -> np.ndarray:
        if self._stream is None:
            return np.zeros(0, dtype=np.uint16)   # before connect: an empty placeholder
        return self._stream.buffer

    @property
    def buffer_head(self) -> int:
        return 0 if self._stream is None else self._stream.ring_head

    @property
    def buffer_capacity(self) -> int:
        return self._buf_len

    @property
    def samples_received(self) -> int:
        return 0 if self._stream is None else self._stream.ring_total

    @property
    def running(self) -> bool:
        return self._running

    @property
    def streaming(self) -> bool:
        return self._streaming and (self._stream is None or self._stream.running)

    @property
    def stream_stats(self) -> Optional[dict]:
        return None if self._stream is None else self._stream.get_stream_stats()

    @property
    def telemetry(self):
        return None if self._stream is None else self._stream.get_telemetry()

    # -- Sequences the Controller offers --------------------------------------------

    def request_telemetry(self, timeout_s: float = 1.0) -> bool:
        """Telemetry from a stopped box: True once it arrives (or already was), False if
        its firmware sends none. Raises if the box doesn't answer."""
        self._ensure_connected()
        self.begin_stream()
        snapshot = self.telemetry
        if snapshot is not None and not snapshot.host_stale:
            return True
        return self.try_optical_diagnostics(timeout_s=timeout_s)

    def restart(self, timeout_s: float = 4.0) -> None:
        """Confirmed STOP, optical diagnostics selected, then a confirmed START."""
        self._ensure_connected()
        self.stop_system_confirmed(timeout_s=timeout_s)
        self.try_optical_diagnostics()
        self.start_confirmed(timeout_s=timeout_s)
