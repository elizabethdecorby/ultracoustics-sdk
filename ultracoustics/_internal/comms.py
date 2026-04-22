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
import threading
import time
from typing import Optional

import numpy as np
import usb.core
import usb.util
import usb.backend.libusb1
import serial
import serial.tools.list_ports

from .protocol import BULK_OUT_EP, BULK_IN_EP, pack_command
from ..config import VENDOR_ID, PRODUCT_ID, SAMPLE_RATE

# Resolve bundled libusb DLL on Windows so the SDK is self-contained
_current_dir = os.path.dirname(os.path.abspath(__file__))
_bin_path = os.path.join(_current_dir, 'bin', 'libusb-1.0.dll')

_backend = None
if platform.system() == 'Windows' and os.path.exists(_bin_path):
    _backend = usb.backend.libusb1.get_backend(find_library=lambda x: _bin_path)


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

class USBStream:
    """
    High-throughput threaded USB Bulk streaming.  (Internal)

    Wraps a ``USBBulkConnection`` and pumps data from the IN endpoint
    directly into either:
    - a pre-allocated circular buffer (normal path, zero-copy), or
    - a user callback.

    The controller binds the circular-buffer sink at construction time to
    avoid intermediate allocations and per-chunk callback overhead.
    """

    def __init__(
        self,
        connection: USBBulkConnection,
        buffer_size=65536,
        ring_buffer: Optional[np.ndarray] = None,
    ):
        self._conn = connection
        self._buffer_size = buffer_size
        self._running = False
        self._thread = None
        self._callback = None
        self._ring_buffer: Optional[np.ndarray] = None
        self._ring_head = 0
        self._ring_total = 0
        self.disconnected = False
        self.last_error: Optional[Exception] = None
        if ring_buffer is not None:
            self._set_ring_buffer(ring_buffer, reset=True)

    # -- Public API -----------------------------------------------------------

    def start(self, callback=None):
        """Begin streaming.

        Args:
            callback: Optional ``callback(data_bytes)`` invoked for every
                chunk.  If *None*, data is written into the attached ring buffer.
        """
        if self._running:
            return
        self._running = True
        self._callback = callback
        self.disconnected = False
        self.last_error = None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _set_ring_buffer(self, ring_buffer: np.ndarray, reset=True):
        """Bind a pre-allocated uint16 circular buffer as stream sink."""
        if ring_buffer.dtype != np.uint16:
            raise ValueError("ring_buffer must be dtype uint16")
        if ring_buffer.ndim != 1:
            raise ValueError("ring_buffer must be a 1D array")
        self._ring_buffer = ring_buffer
        if reset:
            self._ring_head = 0
            self._ring_total = 0

    def stop(self):
        """Stop streaming and join the background thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    @property
    def running(self):
        return self._running

    @property
    def ring_head(self) -> int:
        return self._ring_head

    @property
    def ring_total(self) -> int:
        return self._ring_total

    # -- Internal -------------------------------------------------------------

    def _loop(self):
        while self._running and self._conn.connected:
            try:
                data_array = self._conn.dev.read(
                    BULK_IN_EP, self._buffer_size, timeout=100
                )
                if len(data_array) == 0:
                    continue

                if self._callback:
                    self._callback(data_array)
                elif self._ring_buffer is not None:
                    samples = np.frombuffer(data_array, dtype=np.uint16)
                    n = len(samples)
                    buf = self._ring_buffer
                    head = self._ring_head
                    buf_len = len(buf)

                    if head + n <= buf_len:
                        buf[head : head + n] = samples
                        head += n
                    else:
                        first = buf_len - head
                        buf[head:] = samples[:first]
                        buf[: n - first] = samples[first:]
                        head = n - first

                    self._ring_head = head
                    self._ring_total += n

            except usb.core.USBError as e:
                if e.errno == 110:  # timeout – normal when idle
                    continue
                if e.errno in (60, 19):  # device gone
                    self._conn.connected = False
                    self.disconnected = True
                    break
            except Exception as e:
                self.last_error = e
                self._running = False
                if getattr(self._conn, "verbose", False):
                    print(f"USB stream loop error: {e}")
                break


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
