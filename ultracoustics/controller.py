"""
High-level controller for the Ultracoustics BROADSONIC System.

This module provides the :class:`Controller` class, which is the primary
user-facing interface for the Ultracoustics SDK. It manages:

- **USB Bulk connection** to the Master Board for streaming measurement data
- **System state management** (BOOT / IDLE / WARM modes)
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
    ctrl.start()               # send BOOT command (turn on laser measurment system), begin reading streamed data
    import time; time.sleep(2) # wait for system to lock, stabilize, and for buffer to fill
    data = ctrl.save(1.0)      # snapshot 1 second of samples
    ctrl.stop()                # send IDLE (turn off laser measurement system), stop reading streamed data
    ctrl.close()               # release all resources
"""

import time
from pathlib import Path

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
        ctrl.start()               # send BOOT command (turn on laser measurment system), begin reading streamed data
        import time; time.sleep(2) # wait for system to lock, stabilize, and for buffer to fill
        data = ctrl.save(1.0)      # snapshot 1 second of samples
        ctrl.stop()                # send IDLE (turn off laser measurement system), stop reading streamed data
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
        self._usb: USBBulkConnection | None = None
        self._stream: USBStream | None = None
        self._running = False

        # Rolling circular buffer (uint16, 1.2 s default)
        self._buf_len = int(SAMPLE_RATE * 1.2)
        self._buf = np.zeros(self._buf_len, dtype=np.uint16)
        self._buf_idx = 0
        self._total_samples = 0

    # -- Connection lifecycle -------------------------------------------------

    def connect(self):
        """Open the USB Bulk connection to the Master Board.

        Creates a :class:`USBBulkConnection` and wraps it in a
        :class:`USBStream` for background data ingestion.

        Raises
        ------
        usb.core.USBError
            If the Master Board is not found or the USB claim fails.
        """
        self._usb = USBBulkConnection(verbose=self.verbose)
        self._stream = USBStream(self._usb)

    @property
    def connected(self) -> bool:
        """``True`` if the USB connection to the Master Board is active."""
        return self._usb is not None and self._usb.connected

    # -- Live buffer access (read-only) ---------------------------------------

    @property
    def buffer(self) -> np.ndarray:
        """Read-only view of the live circular sample buffer (uint16)."""
        return self._buf

    @property
    def buffer_head(self) -> int:
        """Current write-head index in the circular buffer."""
        return self._buf_idx

    @property
    def buffer_capacity(self) -> int:
        """Total capacity of the circular buffer in samples."""
        return self._buf_len

    @property
    def samples_received(self) -> int:
        """Cumulative number of samples received since streaming started."""
        return self._total_samples

    @property
    def running(self) -> bool:
        """``True`` while acquisition is active (between :meth:`start` and :meth:`stop`)."""
        return self._running

    # -- State management -----------------------------------------------------

    def start(self):
        """Send BOOT command to turn laser system on (thus start measurements), and run logic to read back stream of usb data."""
        self._ensure_connected()
        self._stream.start(callback=self._on_data)
        self._send(CMD_BOOT)
        self._running = True

    def stop(self):
        """Send IDLE command to turn off lasers, and run logic to stop reading streamed data."""
        self._running = False
        if self._stream:
            self._stream.stop()
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
        self.stop()
        if self._stream:
            self._stream.stop()
            self._stream = None
        if self._usb:
            self._usb.close()
            self._usb = None

    # -- Data capture ---------------------------------------------------------

    def save(self, duration_s: float = 1.0, path: str | Path | None = None) -> np.ndarray:
        """Capture *duration_s* seconds of data and optionally write to disk.

        If *path* is given the raw uint16 samples are written as a binary
        file.  The captured numpy array is always returned.
        """
        samples_needed = int(SAMPLE_RATE * duration_s)

        if self._total_samples < samples_needed:
            raise RuntimeError(
                f"Buffer contains only {self._total_samples} samples "
                f"({samples_needed} needed for {duration_s}s)."
            )

        # Extract from circular buffer
        if self._buf_idx >= samples_needed:
            data = self._buf[self._buf_idx - samples_needed : self._buf_idx].copy()
        else:
            data = np.concatenate([
                self._buf[self._buf_len - (samples_needed - self._buf_idx) :],
                self._buf[: self._buf_idx],
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
        """Send a command packet over USB bulk to the Master Board."""
        self._usb.send_command(cmd_byte, wValue, wIndex, extra, timeout_ms)

    def _on_data(self, data_array):
        """Callback from USBStream – appends raw bytes to the circular buffer.

        Called on the USBStream reader thread each time a USB bulk packet
        arrives.  The bytes are interpreted as little-endian ``uint16``
        samples and written into ``_buf`` with wrap-around.
        """
        samples = np.frombuffer(data_array, dtype=np.uint16)
        n = len(samples)
        if self._buf_idx + n <= self._buf_len:
            self._buf[self._buf_idx : self._buf_idx + n] = samples
            self._buf_idx += n
        else:
            first = self._buf_len - self._buf_idx
            self._buf[self._buf_idx :] = samples[:first]
            self._buf[: n - first] = samples[first:]
            self._buf_idx = n - first
        self._total_samples += n
