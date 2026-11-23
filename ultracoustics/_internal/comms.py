"""
Low-level communication layer for the USB Bulk connection.

Classes:
    USBBulkConnection -- Manages the USB Bulk link to the Master Board
        (STM32H7RS).  Handles device discovery, configuration, and
        provides raw send/receive primitives.
    USBStream -- Streams the ADC through a reader subprocess into a
        shared-memory ring buffer.
"""

import os
import platform
import time
import multiprocessing as mp
import queue
from multiprocessing import Process, Queue, shared_memory
from typing import Optional

import numpy as np
import usb.core
import usb.util
import usb.backend.libusb1

from . import stream_proc
from . import telemetry as telemetry_wire
from .protocol import BULK_OUT_EP, BULK_IN_EP, pack_command
from ..config import VENDOR_ID, PRODUCT_ID, SAMPLE_RATE

#: Why a reader subprocess may fail before it runs any SDK code.
MAIN_GUARD_HINT = ("A script that uses Controller must keep its code under "
                   "'if __name__ == \"__main__\":', because the USB reader starts a "
                   "new Python process that imports the script.")

# Resolve bundled libusb DLL on Windows so the SDK is self-contained
_current_dir = os.path.dirname(os.path.abspath(__file__))
_bin_path = os.path.join(_current_dir, 'bin', 'libusb-1.0.dll')

_backend = None
if platform.system() == 'Windows' and os.path.exists(_bin_path):
    _backend = usb.backend.libusb1.get_backend(find_library=lambda x: _bin_path)


# ---------------------------------------------------------------------------
# USB Bulk (Master Board) Communication
# ---------------------------------------------------------------------------

class TransportUncertainError(RuntimeError):
    """A nonempty OUT request may have been partly or fully delivered."""

    def __init__(self, message, *, halt_cleared=False):
        super().__init__(message)
        self.transport_status = "uncertain"
        self.halt_cleared = halt_cleared


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
            transferred = self.dev.write(BULK_OUT_EP, payload, timeout=timeout_ms)
            if transferred != len(payload):
                raise TransportUncertainError(
                    f"short BULK OUT transfer ({transferred}/{len(payload)} bytes); "
                    "request application state is uncertain"
                )
            return {"transport_status": "delivered", "transferred_bytes": transferred}
        except usb.core.USBError as e:
            is_pipe = e.errno == 32 or getattr(e, "backend_error_code", None) == -9
            if is_pipe:
                clear_error = None
                try:
                    self.dev.clear_halt(BULK_OUT_EP)
                except Exception as exc:  # preserve original rejection context
                    clear_error = exc
                if clear_error is not None:
                    raise TransportUncertainError(
                        f"BULK OUT PIPE; halt clear failed ({clear_error}); "
                        "request application state is uncertain",
                        halt_cleared=False,
                    ) from e
                # PIPE does not expose a trustworthy transferred-byte count.
                # Clearing halt restores transport availability but cannot
                # identify the rejected request or prove this payload unsent.
                raise TransportUncertainError(
                    "BULK OUT PIPE; halt cleared without replay; request "
                    "application state is uncertain",
                    halt_cleared=True,
                ) from e
            raise TransportUncertainError(
                f"BULK OUT failed ({e}); request application state is uncertain"
            ) from e

    def receive(self, length, timeout_ms=1000):
        """Blocking read from Bulk IN endpoint. Returns bytes, or None on a timeout.

        Any other USB failure raises, so a lost device is not mistaken for
        a slow one.
        """
        if not self.connected:
            return None
        try:
            data_array = self.dev.read(BULK_IN_EP, length, timeout=timeout_ms)
            return bytes(data_array)
        except usb.core.USBTimeoutError:
            return None
        except usb.core.USBError as e:
            if e.errno == 110:  # ETIMEDOUT, on backends without USBTimeoutError
                return None
            raise RuntimeError(f"USB read failed: {e}") from e

    def send_command(self, cmd_byte, wValue=0, wIndex=0, extra=b"", timeout_ms=5000):
        """Pack and send a protocol command with optional trailing payload."""
        pkt = pack_command(cmd_byte, wValue, wIndex) + extra
        self.send(pkt, timeout_ms=timeout_ms)


# ---------------------------------------------------------------------------
# USB Bulk Streaming
# ---------------------------------------------------------------------------

#: Shared memory still mapped because a caller kept a view of it past close().
_retained_segments = []


class USBStream:
    """
    High-throughput USB Bulk streaming via a dedicated reader subprocess.

    The reader process owns the libusb device handle exclusively, runs an
    async transfer pool against the BULK IN endpoint, and writes samples
    directly into a shared-memory ring buffer.  The parent (Controller / GUI)
    process attaches a numpy view onto the same shared memory and reads
    counters and samples without contending with the reader for the GIL.

    A separate process means the reader shares no GIL with the caller, so a
    busy GUI or analysis thread cannot make it lose packets.

    - ``start()``            : launch the reader process.
    - ``stop()``             : signal the reader to terminate and join.
    - ``running``            : True while the reader is alive.
    - ``buffer``             : numpy view onto the shared ring buffer.
    - ``ring_head``          : current write index in the ring buffer.
    - ``ring_total``         : cumulative samples received.
    - ``get_stream_stats()`` : dict of packet-level diagnostics
      (see :mod:`._internal.stream_proc` for the counter taxonomy).
    - ``send_command(bytes)``: enqueue a raw command payload for the BULK OUT
      endpoint (the only path while streaming, since the reader owns the
      device handle).
    """

    def __init__(
        self,
        vid: int = VENDOR_ID,
        pid: int = PRODUCT_ID,
        ring_capacity_samples: Optional[int] = None,
        verbose: bool = False,
        packet_diagnostics_path: Optional[str] = None,
        packet_diagnostics_attach_s: float = 2.0,
    ):
        if ring_capacity_samples is None:
            ring_capacity_samples = int(SAMPLE_RATE * 1.2)

        self._vid = vid
        self._pid = pid
        self._capacity = int(ring_capacity_samples)
        self.verbose = verbose
        self._packet_diagnostics_path = packet_diagnostics_path
        self._packet_diagnostics_attach_s = packet_diagnostics_attach_s

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
        self._telemetry_shm = shared_memory.SharedMemory(
            create=True, size=telemetry_wire.SIDECAR_BYTES,
        )
        telemetry_wire.initialise_sidecar(self._telemetry_shm)
        # frombuffer holds a buffer export, so a sample view kept past close()
        # stops the mapping being released instead of dangling (close() below).
        self._buffer = np.frombuffer(
            self._ring_shm.buf, dtype=np.uint16, count=self._capacity,
        )
        self._buffer.fill(0)

        # Counters view (read-only from parent perspective; the subprocess
        # is the sole writer).  We snapshot via :func:`stream_proc.read_counters`.
        self._counters_view = stream_proc._counter_view(self._counter_shm)
        self._counters_view[:] = 0

        # Inter-process control primitives.
        self._cmd_queue: Optional[Queue] = None
        self._command_ack_queue: Optional[Queue] = None
        self._error_queue: Optional[Queue] = None
        self._next_command_request_id = 1
        self._terminate_event = None
        self._ready_event = None
        self._proc: Optional[Process] = None

        self._running = False
        self._selected_stream_format = 0
        self._stream_format = 0      # last coherent read of the telemetry sidecar
        self._reader_error = ""

    # -- Public API -----------------------------------------------------------

    def start(self, ready_timeout_s: float = 5.0):
        """Spawn the reader subprocess.

        Parameters
        ----------
        ready_timeout_s : float
            How long to wait for the subprocess to claim the device and
            signal readiness before raising.
        """
        if self._running:
            return

        # Reset counters before launch so stats are 0-based per session.
        self._counters_view[:] = 0
        telemetry_wire.initialise_sidecar(self._telemetry_shm)
        self._selected_stream_format = 0
        self._stream_format = 0
        self._reader_error = ""

        # Use 'spawn' so the subprocess does not inherit any libusb state or
        # Qt resources from the parent (especially relevant on macOS/Linux
        # where 'fork' would copy them).
        ctx = mp.get_context('spawn')
        self._cmd_queue = ctx.Queue()
        self._command_ack_queue = ctx.Queue()
        self._error_queue = ctx.Queue()
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
                command_ack_queue=self._command_ack_queue,
                packet_diagnostics_path=self._packet_diagnostics_path,
                packet_diagnostics_attach_s=self._packet_diagnostics_attach_s,
                telemetry_shm_name=self._telemetry_shm.name,
                error_queue=self._error_queue,
            ),
            daemon=True,
            name='UltracousticsUSBReader',
        )
        self._proc.start()

        # Block until the subprocess opens and claims the device, or fails.
        deadline = time.monotonic() + ready_timeout_s
        while not self._ready_event.wait(timeout=0.05):
            if not self._proc.is_alive():
                detail = self._read_reader_error(0.2) or MAIN_GUARD_HINT
                self._terminate_and_join()
                raise RuntimeError(f"USB reader process exited during start-up. {detail}")
            if time.monotonic() >= deadline:
                self._terminate_and_join()
                raise RuntimeError(
                    f"USB reader process did not become ready within {ready_timeout_s:.1f} s. "
                    f"{MAIN_GUARD_HINT}"
                )

        snap = stream_proc.read_counters(self._counter_shm)
        if snap['fatal'] or not snap['ready']:
            detail = self._read_reader_error(0.2) or f"flags={snap['flags']:#x}"
            self._terminate_and_join()
            raise RuntimeError(f"USB reader process could not open the device: {detail}")

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
        self._buffer = None
        self._counters_view = None
        for shm in (getattr(self, '_ring_shm', None), getattr(self, '_counter_shm', None),
                    getattr(self, '_telemetry_shm', None)):
            if shm is None:
                continue
            try:
                shm.close()
            except BufferError:
                # The caller still holds a view of the samples. It stays
                # readable, and the mapping goes when that view does.
                _retained_segments.append(shm)
            except Exception:  # noqa: BLE001
                pass
            try:
                shm.unlink()
            except Exception:  # noqa: BLE001
                pass
        self._ring_shm = None
        self._counter_shm = None
        self._telemetry_shm = None

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
        view = self._counters_view
        return 0 if view is None else int(view[0])

    @property
    def ring_total(self) -> int:
        view = self._counters_view
        return 0 if view is None else int(view[1])

    def get_stream_stats(self) -> dict:
        """Return packet-level diagnostics with the new counter taxonomy.

        Returns a dict with keys:

        - ``packets``           : valid packets received.
        - ``drops_seq``         : packets missing per firmware sequence header.
                                  End-to-end loss indicator (firmware ring
                                  overflow OR host/kernel/USB-stack loss).
        - ``drops_fw``          : firmware-reported ring-full drop counter,
                                  delivered in-band in each packet's header.
                                  This separately sampled lifetime counter must
                                  not be subtracted from ``drops_seq`` to claim
                                  host-only loss.
        - ``transfer_errors``   : libusb non-OK transfer completions.
        - ``transfer_timeouts`` : libusb timeout completions.
        - ``malformed``         : packets with invalid byte counts.
        - ``ring_head`` / ``total_samples`` : current head index / cumulative.
        - ``ready`` / ``fatal`` / ``device_lost`` : subprocess status flags.
        - ``reader_error`` : why the reader failed, or an empty string.
        - ``drops_host``   : the same count as ``drops_seq``.
        """
        snap = stream_proc.read_counters(self._counter_shm)
        snap['drops_host'] = snap['drops_seq']
        _, telemetry_stats = telemetry_wire.read_sidecar(self._telemetry_shm)
        # A read that loses to the writer says nothing about the format.
        if telemetry_stats.get('coherent'):
            self._stream_format = telemetry_stats.get('active_format', 0)
        snap['stream_format'] = self._stream_format
        snap['telemetry'] = telemetry_stats
        snap['reader_error'] = self._read_reader_error() if snap['fatal'] else self._reader_error
        return snap

    def get_telemetry(self):
        """Return the latest coherent telemetry snapshot, or ``None``.

        The sidecar is populated only after explicit format-1 negotiation.
        ``host_stale`` on the returned immutable snapshot is evaluated when
        read, independently of whether another malformed record is received.
        """
        if self._telemetry_shm is None:
            return None
        snapshot, _ = telemetry_wire.read_sidecar(self._telemetry_shm)
        return snapshot

    # -- Outbound command path (subprocess owns the device) -------------------

    def send_command(self, payload: bytes) -> None:
        """Enqueue a raw command payload for the BULK OUT endpoint.

        Used by :class:`Controller` to issue BOOT/IDLE/WARM packets while
        streaming, since the device handle is held by the subprocess.
        """
        if not self._running or self._cmd_queue is None:
            raise RuntimeError("USBStream is not running; cannot send command")
        self._cmd_queue.put(payload)

    def send_command_confirmed(self, payload: bytes, timeout_s: float = 0.5) -> dict:
        """Send once and wait for the reader's bounded bulkWrite completion.

        This reports transport-delivered/failed/uncertain only. It never claims
        firmware receipt or application. PIPE is cleared but the payload is not
        replayed because transferred length is unknown.
        """
        if not self._running or self._cmd_queue is None or self._command_ack_queue is None:
            raise RuntimeError("USBStream is not running; cannot send command")
        request_id = self._next_command_request_id
        self._next_command_request_id += 1
        enqueued_ns = time.monotonic_ns()
        self._cmd_queue.put((request_id, payload))
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("no bounded OUT completion from reader")
            try:
                ack_id, transport, completed_ns = self._command_ack_queue.get(
                    timeout=remaining
                )
            except queue.Empty as exc:
                raise TimeoutError("no bounded OUT completion from reader") from exc
            if ack_id != request_id:
                continue
            result = {
                "enqueued_monotonic_ns": enqueued_ns,
                "bulk_write_completed_monotonic_ns": completed_ns,
                "host_delivery_bound_ns": completed_ns - enqueued_ns,
                **transport,
            }
            if transport["transport_status"] == "uncertain":
                raise TransportUncertainError(
                    f"confirmed command transport uncertain ({transport['error']}); "
                    "payload was not replayed",
                    halt_cleared=transport.get("halt_cleared", False),
                )
            if transport["transport_status"] != "delivered":
                raise RuntimeError(
                    f"confirmed command transport failed ({transport['error']})"
                )
            return result

    def _request_stream_control(self, payload: bytes, response_kind: str,
                                stream_format: Optional[int] = None,
                                timeout_s: float = 1.0,
                                response_timeout_s: Optional[float] = None) -> dict:
        """Send one request and wait for its validated firmware IN response.

        The reader waits 0.75 s for the response unless *response_timeout_s*
        says otherwise (board record commands take seconds).
        """
        if not self._running or self._cmd_queue is None or self._command_ack_queue is None:
            raise RuntimeError("USBStream is not running; cannot request stream control")
        request_id = self._next_command_request_id
        self._next_command_request_id += 1
        enqueued_ns = time.monotonic_ns()
        item = (request_id, payload, response_kind, stream_format)
        if response_timeout_s is not None:
            item += (response_timeout_s,)
        self._cmd_queue.put(item)
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("no validated firmware stream-control response")
            try:
                ack_id, transport, completed_ns = self._command_ack_queue.get(timeout=remaining)
            except queue.Empty as exc:
                raise TimeoutError("no validated firmware stream-control response") from exc
            if ack_id != request_id:
                continue
            if transport["transport_status"] == "uncertain":
                raise TransportUncertainError(
                    f"stream-control transport uncertain ({transport['error']}); "
                    "firmware application is unknown",
                    halt_cleared=transport.get("halt_cleared", False),
                )
            if transport["transport_status"] != "delivered":
                raise RuntimeError(f"stream-control request failed ({transport['error']})")
            return {
                "enqueued_monotonic_ns": enqueued_ns,
                "firmware_response_monotonic_ns": completed_ns,
                "host_round_trip_bound_ns": completed_ns - enqueued_ns,
                **transport,
            }

    def query_stream_capabilities(self, payload: bytes, timeout_s: float = 1.0):
        result = self._request_stream_control(payload, "capabilities", timeout_s=timeout_s)
        return result["firmware_response"]

    def select_stream_format_confirmed(self, payload: bytes, stream_format: int,
                                       timeout_s: float = 1.0) -> dict:
        """Deliver a negotiated format command and atomically switch parsing.

        The subprocess changes its expected record length only after receiving
        and validating the firmware's dedicated ``UTFM`` acknowledgement.
        The result proves format acceptance, not that any later telemetry
        record or board measurement is valid.
        """
        if stream_format not in (0, 1, 2):
            raise ValueError("stream format must be 0, 1, or 2")
        result = self._request_stream_control(
            payload, "format", stream_format=stream_format, timeout_s=timeout_s,
        )
        self._selected_stream_format = stream_format
        return result

    # -- Internal -------------------------------------------------------------

    def _read_reader_error(self, timeout_s: float = 0.0) -> str:
        """The text of the reader's last failure, once it has arrived."""
        if self._error_queue is not None and not self._reader_error:
            try:
                self._reader_error = self._error_queue.get(timeout=timeout_s)
            except (queue.Empty, OSError, ValueError):
                pass
        return self._reader_error

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
        self._command_ack_queue = None
        self._error_queue = None
        self._terminate_event = None
        self._ready_event = None


# ---------------------------------------------------------------------------
# USB-Serial (Slave / Laser Board)
# ---------------------------------------------------------------------------
