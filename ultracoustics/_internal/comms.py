"""
Low-level communication layer for USB Bulk and USB-Serial connections.

Classes:
    USBBulkConnection -- Manages the USB Bulk link to the Master Board
        (STM32H7RS).  Handles device discovery, configuration, and
        provides raw send/receive primitives.
    USBStream -- Background-threaded reader that continuously pulls data
        from a USBBulkConnection directly into a circular buffer or callback.
    SerialConnection -- USB-Serial (VCP) link to a Slave Laser Board
        (STM32G4).  Auto-detects the COM/tty port and exposes a
        line-oriented text interface.
"""

import os
import platform
import struct
import sys
import threading
import time
import multiprocessing as mp
from multiprocessing import Process, Queue, shared_memory
from typing import Optional

import numpy as np
import usb.core
import usb.util
import usb.backend.libusb1
import serial
import serial.tools.list_ports

from . import stream_proc
from .protocol import BULK_OUT_EP, BULK_IN_EP, pack_command
from ..config import VENDOR_ID, PRODUCT_ID, SAMPLE_RATE

# Resolve bundled libusb DLL on Windows so the SDK is self-contained
_current_dir = os.path.dirname(os.path.abspath(__file__))
_bin_path = os.path.join(_current_dir, 'bin', 'libusb-1.0.dll')

_backend = None
if platform.system() == 'Windows' and os.path.exists(_bin_path):
    _backend = usb.backend.libusb1.get_backend(find_library=lambda x: _bin_path)


# Streaming wire format used by recent firmware revisions.
PSSI_HEADER_BYTES = 4
PSSI_PAYLOAD_BYTES = 16384
PSSI_PACKET_WIRE_BYTES = PSSI_HEADER_BYTES + PSSI_PAYLOAD_BYTES


# ---------------------------------------------------------------------------
# USB Bulk (Master Board) Communication
# ---------------------------------------------------------------------------

class USBBulkConnection:
    """
    Manages the USB Bulk connection to the Master Board (STM32H7RS).

    Handles device discovery, kernel-driver detachment, configuration,
    interface claiming, and provides low-level send/receive primitives.
    """

    def __init__(self, vid=VENDOR_ID, pid=PRODUCT_ID, verbose=False):
        self.vid = vid
        self.pid = pid
        self.verbose = verbose
        self.dev = None
        self.connected = False
        self._connect()

    # -- Connection lifecycle -------------------------------------------------

    def _connect(self):
        """Open USB device using PyUSB."""
        if self.verbose:
            print(f"Finding device {hex(self.vid)}:{hex(self.pid)}...")

        def find_dev():
            return usb.core.find(idVendor=self.vid, idProduct=self.pid, backend=_backend)

        self.dev = find_dev()
        if self.dev is None:
            raise RuntimeError(f"Master Board not found (VID: {hex(self.vid)})")

        # Detach kernel driver if active (Linux/macOS)
        detached = False
        try:
            if self.dev.is_kernel_driver_active(0):
                if self.verbose:
                    print("Detaching kernel driver...")
                self.dev.detach_kernel_driver(0)
                detached = True
        except Exception:
            pass

        if detached:
            if self.verbose:
                print("Re-finding device after detach...")
            self.dev = find_dev()
            if self.dev is None:
                raise RuntimeError("Device lost after detach.")

        # Set configuration
        try:
            current_cfg = None
            try:
                current_cfg = self.dev.get_active_configuration()
            except Exception:
                pass

            if current_cfg is None or current_cfg.bConfigurationValue != 1:
                if self.verbose:
                    print("Setting Configuration 1...")
                self.dev.set_configuration()
            elif self.verbose:
                print("Configuration already set.")
        except usb.core.USBError as e:
            print(f"Warning: Set Config failed: {e}")

        # Claim interface
        try:
            if self.verbose:
                print("Claiming interface 0...")
            usb.util.claim_interface(self.dev, 0)
        except usb.core.USBError as e:
            if e.errno != 16:  # 16 = Resource busy (already claimed)
                raise RuntimeError(f"Failed to claim interface: {e}")

        self.connected = True
        if self.verbose:
            print(
                f"Master Board connected "
                f"({hex(self.dev.idVendor)}:{hex(self.dev.idProduct)})"
            )

    def close(self):
        """Release USB resources."""
        if self.dev:
            try:
                usb.util.dispose_resources(self.dev)
            except Exception:
                pass
            self.dev = None
        self.connected = False

    # -- Low-level I/O -------------------------------------------------------

    def send(self, payload, timeout_ms=5000):
        """Send raw bytes to Bulk OUT endpoint."""
        if not self.connected:
            raise RuntimeError("Not connected")
        try:
            self.dev.write(BULK_OUT_EP, payload, timeout=timeout_ms)
        except usb.core.USBError as e:
            raise RuntimeError(f"Failed to send: {e}")

    def receive(self, length, timeout_ms=1000):
        """Blocking read from Bulk IN endpoint. Returns bytes or None."""
        if not self.connected:
            return None
        try:
            data_array = self.dev.read(BULK_IN_EP, length, timeout=timeout_ms)
            return bytes(data_array)
        except usb.core.USBError as e:
            if e.errno == 110:  # Timeout
                return None
            return None

    def send_command(self, cmd_byte, wValue=0, wIndex=0, extra=b"", timeout_ms=5000):
        """Pack and send a protocol command with optional trailing payload."""
        pkt = pack_command(cmd_byte, wValue, wIndex) + extra
        self.send(pkt, timeout_ms=timeout_ms)


# ---------------------------------------------------------------------------
# USB Bulk Streaming
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# USB Bulk Streaming
# ---------------------------------------------------------------------------

class USBStream:
    """
    High-throughput USB Bulk streaming via a dedicated reader subprocess.

    The reader process owns the libusb device handle exclusively, runs an
    async transfer pool against the BULK IN endpoint, and writes samples
    directly into a shared-memory ring buffer.  The parent (Controller / GUI)
    process attaches a numpy view onto the same shared memory and reads
    counters and samples without contending with the reader for the GIL.

    Why a subprocess?
        Profiling showed that the previous in-process threaded reader lost
        ~30 packets/s in the GUI because the Qt event loop and FFT code held
        the GIL while libusb completion callbacks tried to copy data into the
        ring.  With the reader in a separate process there is no shared GIL,
        and the host-side packet loss observed in the GUI drops to zero
        (residual losses are then attributable solely to the firmware FIFO
        or USB host controller).

    Public API (preserved from the previous implementation):
        - ``start()``            : launch the reader process.
        - ``stop()``             : signal the reader to terminate and join.
        - ``running``            : True between ``start()`` and ``stop()``.
        - ``ring_head``          : current write index in the ring buffer.
        - ``ring_total``         : cumulative samples received.
        - ``get_stream_stats()`` : dict of packet-level diagnostics
          (see :mod:`._internal.stream_proc` for the counter taxonomy).

    New methods:
        - ``buffer``             : numpy view onto the shared ring buffer
          (replaces the externally-supplied ``ring_buffer`` argument the
          previous threaded implementation accepted).
        - ``send_command(bytes)``: enqueue a raw command payload to be sent
          on the BULK OUT endpoint by the reader subprocess (only path while
          streaming, since the device handle lives in the subprocess).
    """

    def __init__(
        self,
        vid: int = VENDOR_ID,
        pid: int = PRODUCT_ID,
        ring_capacity_samples: Optional[int] = None,
        verbose: bool = False,
    ):
        # Default capacity matches the previous Controller default of ~1.2 s
        # at 10 MSPS, exposed here so callers can size the shared ring.
        if ring_capacity_samples is None:
            ring_capacity_samples = int(SAMPLE_RATE * 1.2)

        self._vid = vid
        self._pid = pid
        self._capacity = int(ring_capacity_samples)
        self.verbose = verbose

        # Allocate shared memory for the ring buffer (uint16) and counters.
        # The shared-memory blocks live for the lifetime of this object;
        # they are unlinked in close()/__del__.
        self._ring_bytes = self._capacity * 2  # uint16
        self._ring_shm = shared_memory.SharedMemory(
            create=True, size=self._ring_bytes,
        )
        self._counter_shm = shared_memory.SharedMemory(
            create=True, size=stream_proc._COUNTER_BYTES,
        )
        self._buffer = np.ndarray(
            (self._capacity,), dtype=np.uint16, buffer=self._ring_shm.buf,
        )
        self._buffer.fill(0)

        # Counters view (read-only from parent perspective; the subprocess
        # is the sole writer).  We snapshot via :func:`stream_proc.read_counters`.
        self._counters_view = stream_proc._counter_view(self._counter_shm)
        self._counters_view[:] = 0

        # Inter-process control primitives.
        self._cmd_queue: Optional[Queue] = None
        self._terminate_event = None
        self._ready_event = None
        self._proc: Optional[Process] = None

        self._running = False
        self.disconnected = False
        self.last_error: Optional[Exception] = None

    # -- Public API -----------------------------------------------------------

    def start(self, callback=None, ready_timeout_s: float = 5.0):
        """Spawn the reader subprocess.

        Parameters
        ----------
        callback : ignored
            Present only for backward compatibility with the threaded
            implementation; the new architecture writes directly into the
            shared ring buffer and does not invoke per-chunk callbacks.
        ready_timeout_s : float
            How long to wait for the subprocess to claim the device and
            signal readiness before raising.
        """
        if self._running:
            return

        if callback is not None:
            # The new path is shared-memory only; we do not run user code on
            # the libusb event thread.  Callers that previously relied on a
            # callback must instead read from ``self.buffer``.
            raise NotImplementedError(
                "USBStream callback path was removed in the multiprocessing "
                "rewrite — read samples from .buffer / .ring_head instead."
            )

        # Reset counters before launch so stats are 0-based per session.
        self._counters_view[:] = 0
        self.disconnected = False
        self.last_error = None

        # Use 'spawn' so the subprocess does not inherit any libusb state or
        # Qt resources from the parent (especially relevant on macOS/Linux
        # where 'fork' would copy them).
        ctx = mp.get_context('spawn')
        self._cmd_queue = ctx.Queue()
        self._terminate_event = ctx.Event()
        self._ready_event = ctx.Event()

        self._proc = ctx.Process(
            target=stream_proc.reader_main,
            kwargs=dict(
                vid=self._vid,
                pid=self._pid,
                ring_shm_name=self._ring_shm.name,
                counter_shm_name=self._counter_shm.name,
                ring_capacity=self._capacity,
                cmd_queue=self._cmd_queue,
                terminate_event=self._terminate_event,
                ready_event=self._ready_event,
            ),
            daemon=True,
            name='UltracousticsUSBReader',
        )
        self._proc.start()

        # Block until the subprocess opens and claims the device, or fails.
        if not self._ready_event.wait(timeout=ready_timeout_s):
            self._terminate_and_join()
            raise RuntimeError(
                "USB reader subprocess did not become ready within "
                f"{ready_timeout_s:.1f}s"
            )

        snap = stream_proc.read_counters(self._counter_shm)
        if snap['fatal'] or not snap['ready']:
            self._terminate_and_join()
            raise RuntimeError(
                "USB reader subprocess failed to claim the device "
                f"(flags={snap['flags']:#x})"
            )

        self._running = True
        if self.verbose:
            print(
                f"[USBStream] subprocess ready: {stream_proc.NUM_TRANSFERS} "
                f"transfers x {stream_proc.TRANSFER_SIZE} bytes in flight"
            )

    def stop(self):
        """Signal the reader subprocess to shut down and join it."""
        if not self._running and self._proc is None:
            return
        self._running = False
        self._terminate_and_join()

    def close(self):
        """Stop streaming and release shared memory.

        Idempotent.  After ``close()`` the object is no longer usable.
        """
        self.stop()
        for shm in (getattr(self, '_ring_shm', None), getattr(self, '_counter_shm', None)):
            if shm is None:
                continue
            try:
                shm.close()
            except Exception:  # noqa: BLE001
                pass
            try:
                shm.unlink()
            except Exception:  # noqa: BLE001
                pass
        self._ring_shm = None
        self._counter_shm = None

    def __del__(self):  # noqa: D401
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    # -- Buffer / counter accessors (parent-process consumers) ----------------

    @property
    def buffer(self) -> np.ndarray:
        """Read-only-by-convention numpy view onto the shared ring buffer."""
        return self._buffer

    @property
    def buffer_capacity(self) -> int:
        return self._capacity

    @property
    def running(self) -> bool:
        return self._running and self._proc is not None and self._proc.is_alive()

    @property
    def ring_head(self) -> int:
        # Single 64-bit aligned read — atomic on x86-64.
        return int(self._counters_view[0])

    @property
    def ring_total(self) -> int:
        return int(self._counters_view[1])

    def get_stream_stats(self) -> dict:
        """Return packet-level diagnostics with the new counter taxonomy.

        Returns a dict with keys:

        - ``packets``           : valid packets received.
        - ``drops_seq``         : packets missing per firmware sequence header.
        - ``transfer_errors``   : libusb non-OK transfer completions.
        - ``transfer_timeouts`` : libusb timeout completions.
        - ``malformed``         : packets with invalid byte counts.
        - ``ring_head`` / ``total_samples`` : current head index / cumulative.
        - ``ready`` / ``fatal`` / ``device_lost`` : subprocess status flags.

        For backward compatibility a ``drops_host`` key is also present and
        is an alias for ``drops_seq`` (which is the same thing the old name
        was actually measuring).
        """
        snap = stream_proc.read_counters(self._counter_shm)
        snap['drops_host'] = snap['drops_seq']  # legacy alias
        return snap

    # -- Outbound command path (subprocess owns the device) -------------------

    def send_command(self, payload: bytes) -> None:
        """Enqueue a raw command payload for the BULK OUT endpoint.

        Used by :class:`Controller` to issue BOOT/IDLE/WARM packets while
        streaming, since the device handle is held by the subprocess.
        """
        if not self._running or self._cmd_queue is None:
            raise RuntimeError("USBStream is not running; cannot send command")
        self._cmd_queue.put(payload)

    # -- Internal -------------------------------------------------------------

    def _terminate_and_join(self) -> None:
        """Best-effort shutdown of the reader subprocess."""
        if self._terminate_event is not None:
            self._terminate_event.set()
        if self._cmd_queue is not None:
            try:
                # Sentinel — handled as a redundant terminate hint by the
                # subprocess in case event polling lags.
                self._cmd_queue.put_nowait(None)
            except Exception:  # noqa: BLE001
                pass
        if self._proc is not None:
            self._proc.join(timeout=2.0)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=1.0)
            self._proc = None
        self._cmd_queue = None
        self._terminate_event = None
        self._ready_event = None
        # Update local "device lost" mirror from the final counter snapshot.
        snap = stream_proc.read_counters(self._counter_shm) if self._counter_shm else {}
        if snap.get('device_lost'):
            self.disconnected = True


# ---------------------------------------------------------------------------
# USB-Serial (Slave / Laser Board)
# ---------------------------------------------------------------------------

class SerialConnection:
    """
    USB-Serial link to a Slave Laser Board (STM32G4 VCP).

    Auto-detects the correct COM / tty port by VID/PID or description
    string, then exposes a line-oriented send/receive interface.
    """

    def __init__(self, port=None, baudrate=115200, verbose=False):
        self.verbose = verbose
        self.ser = None

        if port is None:
            port = self._find_port()
        if port is None:
            raise RuntimeError("Laser Board serial port not found.")
        self._open(port, baudrate)

    # -- Connection -----------------------------------------------------------

    def _find_port(self):
        """Auto-detect the laser board serial port."""
        ports = list(serial.tools.list_ports.comports())
        candidates = []
        for p in ports:
            if "STLINK" in str(p.description).upper():
                continue
            if p.vid == 0x0483 and p.pid == 0x5740:
                candidates.insert(0, p)
                continue
            desc = str(p.description).lower()
            if "broadsonic" in desc or "ultracoustics" in desc:
                candidates.insert(0, p)
            elif "tty.usbmodem" in p.device:
                candidates.append(p)
        if not candidates:
            return None
        target = candidates[0]
        if self.verbose and len(candidates) > 1:
            print(f"Multiple candidates, selecting: {target.device}")
        return target.device

    def _open(self, port, baudrate):
        try:
            self.ser = serial.Serial(port, baudrate=baudrate, timeout=0.1)
            time.sleep(0.1)
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            if self.verbose:
                print(f"Laser board connected ({port})")
        except Exception as e:
            raise RuntimeError(f"Failed to connect at {port}: {e}")

    def close(self):
        if self.ser:
            self.ser.close()
            self.ser = None

    # -- I/O ------------------------------------------------------------------

    def send(self, cmd: str):
        """Send a string command (appends CRLF)."""
        if not self.ser:
            raise RuntimeError("Not connected")
        self.ser.write(f"{cmd}\r\n".encode())
        if self.verbose:
            print(f"  -> TX: {cmd}")

    def read_line(self):
        """Non-blocking readline.  Returns stripped string or None."""
        if not self.ser or not self.ser.in_waiting:
            return None
        try:
            return self.ser.readline().decode("utf-8", errors="ignore").strip()
        except Exception:
            return None
