"""
High-level controller for the Ultracoustics BROADSONIC System.

This module provides the :class:`Controller` class, which is the primary
user-facing interface for the Ultracoustics SDK. It manages:

- **System state management** (BOOT / IDLE / WARM modes)
- **USB Bulk connection** to the Master Board for streaming measurement data
- **Data acquisition** with read access to a rolling circular buffer and snapshot-based saving

Internally uses a :class:`comms.USBBulkConnection` and
:class:`comms.USBStream` for low-level I/O, exposing a simple
connect / start / stop / save interface.

For diagnostics, override mode, firmware queries, laser serial control, and
firmware flashing, see :mod:`ultracoustics._internal.maintenance`.

Typical usage::

    from ultracoustics.controller import Controller

    ctrl = Controller(verbose=True)
    ctrl.connect()             # USB bulk to master board
    ctrl.begin_stream()        # start reading USB data into buffer (laser still off)
    ctrl.start()               # send BOOT command (turn on laser measurement system)
    import time; time.sleep(2) # wait for system to lock, stabilize, and for buffer to fill
    data = ctrl.save(1.0)      # snapshot 1 second of samples
    ctrl.stop()                # send IDLE (turn off laser measurement system, stream keeps running)
    ctrl.end_stream()          # stop reading USB data
    ctrl.close()               # release all resources
"""

import time
from pathlib import Path
from typing import Optional, Union

import numpy as np

from ._internal.comms import USBBulkConnection, USBStream
from ._internal.protocol import CMD_BOOT, CMD_IDLE, CMD_WARM
from .config import SAMPLE_RATE


class Controller:
    """
    Top-level interface to the Ultracoustics system.

    Usage::

    ctrl = Controller(verbose=True)
        ctrl.connect()             # USB bulk to master board
        ctrl.begin_stream()        # start reading USB data into buffer (laser still off)
        ctrl.start()               # send BOOT command (turn on laser measurement system)
        import time; time.sleep(2) # wait for system to lock, stabilize, and for buffer to fill
        data = ctrl.save(1.0)      # snapshot 1 second of samples
        ctrl.stop()                # send IDLE (turn off laser, stream keeps running)
        ctrl.end_stream()          # stop reading USB data
        ctrl.close()               # release all resources

        Real-time buffer access
        -----------------------
        For GUI or live-analysis consumers that need zero-copy access to
        the streaming data at high refresh rates (e.g. 20 Hz), the
        following read-only properties expose the internal circular buffer
        without allocating a copy on every call (unlike :meth:`save`)::

            ctrl.buffer           # np.ndarray – the live uint16 sample ring
            ctrl.buffer_head      # int        – current write-head index
            ctrl.buffer_capacity  # int        – ring length in samples
            ctrl.samples_received # int        – cumulative sample count
            ctrl.running          # bool       – True while streaming

        The buffer is written by a background thread; readers should
        treat it as a lock-free snapshot (read the head index first,
        then slice the array).
    """

    def __init__(self, verbose=False):
        """Initialise the controller (no hardware interaction yet).

        Parameters
        ----------
        verbose : bool, optional
            If ``True``, print diagnostic messages during USB I/O
            and data capture operations. Defaults to ``False``.

        Attributes
        ----------
        _usb : USBBulkConnection or None
            Low-level USB bulk endpoint wrapper (set by :meth:`connect`).
        _stream : USBStream or None
            Background reader that feeds samples to the circular buffer.
        _running : bool
            ``True`` while acquisition is active (between :meth:`start`
            and :meth:`stop`).
        _buf : numpy.ndarray
            Rolling circular buffer of raw ``uint16`` ADC samples
            (default capacity ~1.2 s at the configured sample rate).
        """
        self.verbose = verbose
        self._stream: Optional[USBStream] = None
        self._running = False
        self._streaming = False
        self._connected = False

        # Default ring capacity: ~1.2 s at the configured sample rate.
        # Lives in shared memory once the stream subprocess is spawned
        # (allocated by USBStream) so the parent and reader share it
        # zero-copy.
        self._buf_len = int(SAMPLE_RATE * 1.2)

    # -- Connection lifecycle -------------------------------------------------

    def connect(self):
        """Probe for the Master Board and prepare a streaming session.

        Performs a quick :class:`USBBulkConnection` open/close to fail fast
        if the device is missing, then constructs a :class:`USBStream`.  The
        stream's reader subprocess is *not* spawned here — that happens in
        :meth:`begin_stream` — which avoids holding the USB device claim
        between :meth:`connect` and :meth:`begin_stream`.

        Raises
        ------
        RuntimeError
            If the Master Board is not found or the USB claim fails.
        """
        probe = USBBulkConnection(verbose=self.verbose)
        probe.close()  # release the device so the reader subprocess can claim it
        self._stream = USBStream(
            ring_capacity_samples=self._buf_len,
            verbose=self.verbose,
        )
        self._connected = True

    @property
    def connected(self) -> bool:
        """``True`` once :meth:`connect` has succeeded and resources are live."""
        return self._connected and self._stream is not None

    # -- Live buffer access (read-only) ---------------------------------------

    @property
    def buffer(self) -> np.ndarray:
        """Read-only view of the live circular sample buffer (uint16).

        Backed by shared memory written by the reader subprocess.
        """
        if self._stream is None:
            # Pre-connect: return an empty placeholder so callers don't
            # crash on attribute access.
            return np.zeros(0, dtype=np.uint16)
        return self._stream.buffer

    @property
    def buffer_head(self) -> int:
        """Current write-head index in the circular buffer."""
        if self._stream is None:
            return 0
        return self._stream.ring_head

    @property
    def buffer_capacity(self) -> int:
        """Total capacity of the circular buffer in samples."""
        return self._buf_len

    @property
    def samples_received(self) -> int:
        """Cumulative number of samples received since streaming started."""
        if self._stream is None:
            return 0
        return self._stream.ring_total

    @property
    def running(self) -> bool:
        """``True`` while the laser measurement system is active (between :meth:`start` and :meth:`stop`)."""
        return self._running

    @property
    def streaming(self) -> bool:
        """``True`` while USB data is being read into the buffer (between :meth:`begin_stream` and :meth:`end_stream`)."""
        return self._streaming

    @property
    def stream_stats(self) -> Optional[dict]:
        """Packet-level stream diagnostics when supported by firmware.

        Returns a dict with ``packets``, ``drops_host``, ``malformed``, and
        ``last_seq`` while streaming; returns ``None`` when not connected.
        """
        if self._stream is None:
            return None
        return self._stream.get_stream_stats()

    # -- State management -----------------------------------------------------

    def begin_stream(self):
        """Start reading USB bulk data into the circular buffer without turning the laser on.

        Call this after :meth:`connect` to begin collecting data immediately.
        The laser measurement system remains off until :meth:`start` is called.
        """
        self._ensure_connected()
        if not self._streaming:
            self._stream.start()
            self._streaming = True

    def end_stream(self):
        """Stop reading USB bulk data from the device.

        This does *not* send an IDLE command — use :meth:`stop` first if the
        laser is currently active.
        """
        self._streaming = False
        if self._stream:
            self._stream.stop()

    def start(self):
        """Send BOOT command to turn the laser measurement system on.

        If :meth:`begin_stream` has not been called yet, the USB data stream
        is started automatically so that samples begin arriving in the buffer.
        """
        self._ensure_connected()
        if not self._streaming:
            self.begin_stream()
        self._send(CMD_BOOT)
        self._running = True

    def stop(self):
        """Send IDLE command to turn off the laser measurement system.

        The USB data stream continues running so the buffer stays live and
        the GUI can keep displaying data. Call :meth:`end_stream` (or
        :meth:`close`) to fully halt data collection.
        """
        self._running = False
        try:
            self._send(CMD_IDLE)
        except Exception:
            pass

    def warm(self):
        """Enter the WARM / standby state. Lasers off but system still powered, allowing faster startup than BOOT."""
        self._ensure_connected()
        self._send(CMD_WARM)

    def close(self):
        """Release all hardware resources."""
        if self._running:
            self.stop()
        self.end_stream()
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        self._connected = False

    # -- Data capture ---------------------------------------------------------

    def save(self, duration_s: float = 1.0, path: Optional[Union[str, Path]] = None) -> np.ndarray:
        """Capture *duration_s* seconds of data and optionally write to disk.

        If *path* is given the raw uint16 samples are written as a binary
        file.  The captured numpy array is always returned.
        """
        samples_needed = int(SAMPLE_RATE * duration_s)
        total_samples = self.samples_received
        buf_idx = self.buffer_head

        if total_samples < samples_needed:
            raise RuntimeError(
                f"Buffer contains only {total_samples} samples "
                f"({samples_needed} needed for {duration_s}s)."
            )

        buf = self.buffer
        # Extract from circular buffer (shared memory — we copy on the way
        # out so callers can hold the result independently of the reader).
        if buf_idx >= samples_needed:
            data = buf[buf_idx - samples_needed : buf_idx].copy()
        else:
            data = np.concatenate([
                buf[self._buf_len - (samples_needed - buf_idx) :],
                buf[:buf_idx],
            ])

        if path is not None:
            data.tofile(str(path))
            if self.verbose:
                mb = len(data) * 2 / (1024 * 1024)
                print(f"Saved {len(data):,} samples ({mb:.2f} MB) to {path}")

        return data

    # -- Internals ------------------------------------------------------------

    def _ensure_connected(self):
        """Raise if the USB connection is not established."""
        if not self.connected:
            raise RuntimeError("Not connected. Call connect() first.")

    def _send(self, cmd_byte, wValue=0, wIndex=0, extra=b"", timeout_ms=5000):
        """Send a command packet over USB bulk to the Master Board.

        Routes through the streaming subprocess's command queue, since that
        process owns the libusb device handle exclusively.  ``timeout_ms`` is
        accepted for API compatibility; the subprocess applies its own
        bulkWrite timeout (2 s) which is sufficient for command packets.
        """
        from ._internal.protocol import pack_command
        if self._stream is None or not self._stream.running:
            raise RuntimeError(
                "Not streaming — call begin_stream() before sending commands."
            )
        pkt = pack_command(cmd_byte, wValue, wIndex) + extra
        self._stream.send_command(pkt)
