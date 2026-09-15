"""
Multiprocessing USB streaming reader.

Runs in a separate process and owns the libusb-1.0 device handle exclusively.
Pulls data from the BULK IN endpoint via libusb async transfers (no blocking
``read()`` calls, no intermediate Python queues), and writes samples directly
into a shared-memory ring buffer with shared-memory counters that the parent
process can read lock-free.

Design rationale
----------------
The previous threaded implementation suffered loss in the
GUI process because the USB reader threads competed with the Qt event loop
and FFT computation for the GIL.  When the GIL was held by Qt or numpy, the
reader's post-read code (sequence check + ring write) stalled long enough for
the kernel-side libusb buffer to overflow.

This module eliminates that contention:

1. Reader runs in its own process — no shared GIL with the GUI.
2. Async libusb transfers — N transfers always in flight, kernel never idle.
3. Direct ring write in the libusb event-handling loop — no intermediate
   Python queue, no dispatcher thread, no ``put_nowait`` silent drops.
4. Shared memory — counters are atomic 64-bit aligned writes (single
   instruction on x86-64); the GUI just reads them.

Drop counter taxonomy
---------------------
The single ``drops_host`` of the previous implementation conflated several
distinct failure modes.  This module separates them:

- ``drops_seq``    : packets missing per the firmware sequence header.
                    End-to-end loss indicator — could be firmware ring
                    overflow OR host/kernel/USB-stack loss.
- ``drops_fw``     : latest value of the firmware-side ``usb_dropped_buffers``
                    counter observed in a delivered packet's header. Increments
                    when the firmware pending ring rejects a packet. It is a
                    separate, asynchronously sampled counter; do not subtract it
                    from ``drops_seq`` to invent host-only loss.
- ``transfer_errors``   : libusb reported ``TRANSFER_ERROR`` / ``STALL`` /
                          ``OVERFLOW`` / ``NO_DEVICE`` on a completed transfer.
- ``transfer_timeouts`` : libusb reported ``TRANSFER_TIMED_OUT``.  Normal in
                          short bursts when the device pauses; sustained
                          values indicate the device stopped sending.
- ``malformed``         : transfer completed but the byte count is invalid
                          (odd length, or smaller than packet header).

There is no ``queue_full`` counter — the new architecture has no host-side
intermediate queue between USB and the ring buffer.
"""

import ctypes
import json
import os
import platform
import queue
import struct
import sys
import threading
import time
from multiprocessing import Event, Process, Queue
from multiprocessing.shared_memory import SharedMemory
from typing import Optional

import numpy as np

from .telemetry import (
    TelemetryFormatError, TelemetrySidecarWriter, parse_capabilities,
    parse_format_ack, parse_record, FORMAT1_RECORD_BYTES, LEGACY_RECORD_BYTES,
)
from .control import (ControlProtocolError, parse_manual_response,
                      parse_runtime_metrics)


def parse_attached_record(raw, current_format, received_monotonic_ns):
    """Attach to either existing wire format without changing the device.

    Reopening the host does not reset firmware negotiation. Length selects a
    candidate only; the normal strict parser must validate it before adoption.
    """
    candidate = ((2 if raw[LEGACY_RECORD_BYTES + 4] == 2 else 1)
                 if len(raw) == FORMAT1_RECORD_BYTES else
                 0 if len(raw) == LEGACY_RECORD_BYTES else current_format)
    parsed = parse_record(raw, candidate,
                          received_monotonic_ns=received_monotonic_ns)
    return parsed, candidate


# ---------------------------------------------------------------------------
# Shared memory layout
# ---------------------------------------------------------------------------

# Counter block: 9 × uint64 = 72 bytes, naturally aligned for atomic writes
# on x86-64.
_COUNTER_DTYPE = np.uint64
_COUNTER_BYTES = 72
_COUNTER_FIELDS = (
    'head_samples',      # 0  — write index in ring buffer (mod capacity)
    'total_samples',     # 1  — cumulative samples written
    'packets',           # 2  — cumulative valid packets received
    'drops_seq',         # 3  — cumulative packets missed (sequence gaps)
    'transfer_errors',   # 4  — libusb non-OK / non-timeout statuses
    'transfer_timeouts', # 5  — libusb timeout count
    'malformed',         # 6  — packets with invalid byte count
    'flags',             # 7  — bit0: ready, bit1: fatal_error, bit2: device_lost
    'drops_fw',          # 8  — latest firmware-reported ring-full drop count
)

FLAG_READY        = 1 << 0
FLAG_FATAL        = 1 << 1
FLAG_DEVICE_LOST  = 1 << 2


def _counter_view(shm: SharedMemory) -> np.ndarray:
    """Return a uint64 numpy view onto a counter shared memory block."""
    return np.ndarray((len(_COUNTER_FIELDS),), dtype=_COUNTER_DTYPE, buffer=shm.buf)


def read_counters(shm: SharedMemory) -> dict:
    """Snapshot the counter block as a plain dict (parent-process helper)."""
    arr = _counter_view(shm)
    # Read each field once — values may update mid-call but each individual
    # 64-bit read is atomic on x86-64.
    snap = {name: int(arr[i]) for i, name in enumerate(_COUNTER_FIELDS)}
    flags = snap['flags']
    snap['ready']        = bool(flags & FLAG_READY)
    snap['fatal']        = bool(flags & FLAG_FATAL)
    snap['device_lost']  = bool(flags & FLAG_DEVICE_LOST)
    return snap


# ---------------------------------------------------------------------------
# Subprocess entry point
# ---------------------------------------------------------------------------

# Wire format constants (mirrored from comms.py to avoid importing it in the
# subprocess — keeps the import surface small for ``spawn`` start method).
# 8-byte header = uint32 sequence + uint32 drops_fw (in-band firmware drop
# counter). MUST stay in sync with comms.py and the firmware
# pssi_packet_t in pssi_packet.h.
PSSI_HEADER_BYTES = 8
PSSI_PAYLOAD_BYTES = 16384
PSSI_PACKET_WIRE_BYTES = PSSI_HEADER_BYTES + PSSI_PAYLOAD_BYTES

# Async transfer pool — see module docstring for sizing rationale.
#
# Sizing note: USB bulk transfers terminate on a short packet, and our
# 16392-byte wire payload ends with an 8-byte short packet (16392 mod
# wMaxPacketSize=512 == 8).  Therefore each completed transfer carries
# **exactly one** logical packet, regardless of the buffer size we hand
# libusb.  Buffering in flight = NUM_TRANSFERS packets, NOT
# NUM_TRANSFERS * TRANSFER_SIZE / packet_size.
#
# At 10 MSPS that's ~1219 packets/s, so 256 transfers ≈ 210 ms of host-side
# buffering — enough to ride out routine Windows scheduler stalls without
# letting the firmware FIFO overflow.  Bump higher (env override) if you
# still see drops_seq incrementing while xerr/xto stay at 0.
NUM_TRANSFERS = int(os.environ.get('UC_NUM_TRANSFERS', '256'))
TRANSFER_SIZE = int(os.environ.get('UC_TRANSFER_SIZE', str(64 * 1024)))  # >= 16392
TRANSFER_TIMEOUT_MS = 1000

# Bulk endpoint addresses (device-specific; mirrored from protocol.py).
BULK_IN_EP = 0x81
BULK_OUT_EP = 0x01

# How aggressively to drain the command queue inside the event loop.
COMMAND_POLL_INTERVAL_S = 0.005
_SMALL_FORWARD_GAP_MAX = 1000


def _write_outbound_once(handle, payload, timeout_ms: int) -> dict:
    """Attempt one nonempty OUT request, clearing PIPE without replay."""
    try:
        transferred = handle.bulkWrite(BULK_OUT_EP, payload, timeout=timeout_ms)
        if transferred != len(payload):
            return {"transport_status": "uncertain", "transferred_bytes": transferred,
                    "halt_cleared": False,
                    "error": f"short transfer {transferred}/{len(payload)}"}
        return {"transport_status": "delivered", "transferred_bytes": transferred,
                "halt_cleared": False, "error": None}
    except Exception as exc:
        # Local import preserves the small spawn-process import surface.
        import usb1
        if not isinstance(exc, usb1.USBErrorPipe):
            return {"transport_status": "uncertain", "transferred_bytes": None,
                    "halt_cleared": False, "error": repr(exc)}
        try:
            handle.clearHalt(BULK_OUT_EP)
        except Exception as clear_exc:
            return {"transport_status": "uncertain", "transferred_bytes": None,
                    "halt_cleared": False,
                    "error": f"{exc!r}; halt clear failed: {clear_exc!r}"}
        return {"transport_status": "uncertain", "transferred_bytes": None,
                "halt_cleared": True,
                "error": f"{exc!r}; halt cleared without replay"}


def classify_sequence_transition(last_seq: Optional[int], seq: int) -> tuple[str, int, bool]:
    """Classify one uint32 sequence transition.

    Returns ``(classification, missing_packets, starts_new_epoch)``. A backward
    transition starts a new attach/reset epoch and is never converted into an
    enormous loss count. Forward gaps, including gaps larger than the old 1000
    sanity threshold, retain their explicit missing-packet count.
    """
    seq &= 0xFFFFFFFF
    if last_seq is None:
        return "first", 0, False
    last_seq &= 0xFFFFFFFF
    if seq == last_seq:
        return "duplicate", 0, False
    if seq > last_seq:
        missing = seq - last_seq - 1
        if missing == 0:
            return "next", 0, False
        kind = "small_forward_gap" if missing <= _SMALL_FORWARD_GAP_MAX else "large_forward_gap"
        return kind, missing, False
    if last_seq >= 0xFFFF0000 and seq <= 0x0000FFFF:
        delta = (seq - last_seq) & 0xFFFFFFFF
        if delta == 1:
            return "wrap", 0, False
        return "wrap_gap", delta - 1, False
    return "backwards_new_epoch", 0, True


class PacketHeaderDiagnostics:
    """Bounded, default-off recorder for attach headers and later anomalies."""

    def __init__(self, output_path: Optional[str], *, attach_duration_s: float = 2.0,
                 max_attach_records: int = 3000, max_anomaly_records: int = 512):
        self.output_path = output_path
        self.attach_duration_ns = int(attach_duration_s * 1_000_000_000)
        self.max_attach_records = max_attach_records
        self.max_anomaly_records = max_anomaly_records
        self.started_ns: Optional[int] = None
        self.attach_records: list[dict] = []
        self.anomaly_records: list[dict] = []
        self.omitted_attach = 0
        self.omitted_anomalies = 0
        self.epoch = 0

    @property
    def enabled(self) -> bool:
        return bool(self.output_path)

    def record(self, *, receive_monotonic_ns: int, length: int,
               seq: Optional[int], drops_fw: Optional[int],
               classification: str, missing_packets: int = 0,
               starts_new_epoch: bool = False) -> None:
        if not self.enabled:
            return
        if self.started_ns is None:
            self.started_ns = receive_monotonic_ns
        if starts_new_epoch:
            self.epoch += 1
        row = {
            "receive_monotonic_ns": receive_monotonic_ns,
            "since_reader_first_receive_ns": receive_monotonic_ns - self.started_ns,
            "length_bytes": length,
            "sequence": seq,
            "drops_fw": drops_fw,
            "classification": classification,
            "missing_packets": missing_packets,
            "epoch": self.epoch,
        }
        in_attach = row["since_reader_first_receive_ns"] <= self.attach_duration_ns
        if in_attach:
            if len(self.attach_records) < self.max_attach_records:
                self.attach_records.append(row)
            else:
                self.omitted_attach += 1
        elif classification != "next":
            if len(self.anomaly_records) < self.max_anomaly_records:
                self.anomaly_records.append(row)
            else:
                self.omitted_anomalies += 1

    def write(self) -> None:
        if not self.enabled:
            return
        header = {
            "type": "packet_header_diagnostics",
            "schema_version": 1,
            "attach_duration_s": self.attach_duration_ns / 1_000_000_000,
            "max_attach_records": self.max_attach_records,
            "max_anomaly_records": self.max_anomaly_records,
            "omitted_attach_records": self.omitted_attach,
            "omitted_anomaly_records": self.omitted_anomalies,
            "epoch_count": self.epoch + 1,
        }
        with open(self.output_path, "w", encoding="utf-8") as output:
            output.write(json.dumps(header, sort_keys=True) + "\n")
            for row in self.attach_records + self.anomaly_records:
                output.write(json.dumps(row, sort_keys=True) + "\n")


def _prepare_libusb_dll() -> None:
    """Make the bundled ``libusb-1.0.dll`` discoverable on Windows.

    The SDK ships the Windows DLL under ``_internal/bin/`` so the package is
    self-contained.  ``python-libusb1`` uses ``ctypes.util.find_library`` which
    on Windows searches ``PATH`` and the per-process DLL directories, so we
    extend both before importing ``usb1``.
    """
    if platform.system() != 'Windows':
        return
    here = os.path.dirname(os.path.abspath(__file__))
    bin_dir = os.path.join(here, 'bin')
    dll_path = os.path.join(bin_dir, 'libusb-1.0.dll')
    if not os.path.exists(dll_path):
        return
    # Prepend bin dir to PATH and add to DLL search dirs (Py 3.8+).
    os.environ['PATH'] = bin_dir + os.pathsep + os.environ.get('PATH', '')
    if hasattr(os, 'add_dll_directory'):
        try:
            os.add_dll_directory(bin_dir)
        except OSError:
            pass
    # Best-effort preload — keeps the DLL pinned even if PATH lookup
    # fails inside libusb1's own loader.
    try:
        ctypes.CDLL(dll_path)
    except OSError:
        pass


def reader_main(
    vid: int,
    pid: int,
    ring_shm_name: str,
    counter_shm_name: str,
    ring_capacity: int,
    cmd_queue: Queue,
    terminate_event,
    ready_event,
    command_ack_queue=None,
    packet_diagnostics_path: Optional[str] = None,
    packet_diagnostics_attach_s: float = 2.0,
    telemetry_shm_name: Optional[str] = None,
) -> None:
    """Subprocess entry point.

    Runs the libusb async event loop until ``terminate_event`` is set, writing
    received sample payloads into the shared ring buffer and tracking
    diagnostics in the shared counter block.
    """
    _prepare_libusb_dll()

    # Imports done here (post-DLL prep) so the parent process never loads
    # libusb in case the subprocess is the only USB owner.
    import usb1  # type: ignore

    # Attach to shared memory created by the parent.
    ring_shm = SharedMemory(name=ring_shm_name)
    counter_shm = SharedMemory(name=counter_shm_name)
    telemetry_shm = SharedMemory(name=telemetry_shm_name) if telemetry_shm_name else None
    telemetry_writer = TelemetrySidecarWriter(telemetry_shm) if telemetry_shm else None
    ring = np.ndarray((ring_capacity,), dtype=np.uint16, buffer=ring_shm.buf)
    counters = _counter_view(counter_shm)

    # Local mirror of head/total — the subprocess is the sole writer, so we
    # avoid round-tripping through shared memory on the hot path and only
    # publish the running totals after each packet.
    head = 0
    total = 0
    last_seq: Optional[int] = None
    stream_format = 0
    pending_control = None
    diagnostics = PacketHeaderDiagnostics(
        packet_diagnostics_path, attach_duration_s=packet_diagnostics_attach_s,
    )

    def _set_flag(bit: int) -> None:
        counters[7] = int(counters[7]) | bit

    ctx: Optional['usb1.USBContext'] = None
    handle: Optional['usb1.USBDeviceHandle'] = None
    transfers: list = []

    try:
        ctx = usb1.USBContext()
        ctx.open()
        handle = ctx.openByVendorIDAndProductID(vid, pid, skip_on_error=False)
        if handle is None:
            _set_flag(FLAG_FATAL)
            ready_event.set()  # wake parent so it can fail fast
            return

        # Detach kernel driver where applicable, claim interface 0.
        try:
            if handle.kernelDriverActive(0):
                handle.detachKernelDriver(0)
        except (usb1.USBError, NotImplementedError):
            pass
        handle.claimInterface(0)

        def _send_outbound(item, timeout_ms: int) -> None:
            nonlocal pending_control
            request_id = None
            payload = item
            is_control = False
            if isinstance(item, tuple):
                if len(item) == 2:
                    request_id, payload = item
                elif len(item) == 4:
                    request_id, payload, response_kind, requested_format = item
                    is_control = True
                else:
                    raise ValueError("invalid outbound queue item")

            # libusb synchronous bulkWrite may dispatch already-submitted
            # asynchronous IN callbacks while waiting for OUT completion. Arm
            # response recognition first so a fast UTCP/UTFM cannot be
            # consumed as a malformed ADC record before pending_control exists.
            if is_control:
                if pending_control is not None:
                    if command_ack_queue is not None:
                        command_ack_queue.put((request_id, {
                            "transport_status": "failed",
                            "transferred_bytes": None,
                            "halt_cleared": False,
                            "error": "another control response is pending",
                        }, time.monotonic_ns()))
                    return
                pending_control = (
                    request_id, response_kind, requested_format,
                    {"transport_status": "delivered",
                     "transferred_bytes": len(payload),
                     "halt_cleared": False, "error": None},
                    time.monotonic() + 0.75,
                )
            result = _write_outbound_once(handle, payload, timeout_ms)
            if is_control and pending_control is None:
                # A validated firmware response arrived reentrantly during
                # bulkWrite. It is stronger evidence than a late synchronous
                # OUT exception, so do not downgrade it or count a false
                # transport error.
                return
            if result["transport_status"] != "delivered":
                counters[4] += 1
            if is_control:
                if result["transport_status"] == "delivered":
                    pending_control = (
                        request_id, response_kind, requested_format, result,
                        pending_control[4],
                    )
                    return
                pending_control = None
            if request_id is not None and command_ack_queue is not None:
                command_ack_queue.put(
                    (request_id, result, time.monotonic_ns())
                )

        # ------------------------------------------------------------------
        # Per-transfer completion callback — called on libusb's event thread
        # (here, the main subprocess thread inside handleEvents()).
        # ------------------------------------------------------------------
        def _on_complete(transfer):
            nonlocal head, total, last_seq, stream_format, pending_control

            status = transfer.getStatus()

            if status == usb1.TRANSFER_COMPLETED:
                length = transfer.getActualLength()
                raw = transfer.getBuffer()[:length]
                if pending_control is not None and length in (16, 24, 32, 72):
                    request_id, response_kind, requested_format, delivery, _deadline = pending_control
                    try:
                        if response_kind == "capabilities":
                            response = parse_capabilities(bytes(raw))
                        elif response_kind == "format":
                            epoch = parse_format_ack(bytes(raw), requested_format)
                            response = {"accepted_format": requested_format,
                                        "stream_epoch": epoch}
                            stream_format = requested_format
                            last_seq = None
                            if telemetry_writer is not None:
                                if requested_format == 0:
                                    telemetry_writer.reset_to_legacy()
                                else:
                                    telemetry_writer.select_format(requested_format)
                        elif response_kind == "manual":
                            response = parse_manual_response(bytes(raw))
                        elif response_kind == "runtime":
                            response = parse_runtime_metrics(bytes(raw))
                        else:
                            raise TelemetryFormatError("unknown pending control response")
                    except (TelemetryFormatError, ControlProtocolError) as exc:
                        delivery = {**delivery, "transport_status": "failed",
                                    "error": str(exc)}
                    else:
                        delivery = {**delivery, "firmware_response": response}
                    if command_ack_queue is not None:
                        command_ack_queue.put((request_id, delivery, time.monotonic_ns()))
                    pending_control = None
                    if not terminate_event.is_set():
                        transfer.submit()
                    return
                try:
                    parsed, received_format = parse_attached_record(
                        raw, stream_format, time.monotonic_ns())
                except TelemetryFormatError:
                    counters[6] += 1  # malformed
                    if telemetry_writer is not None and stream_format in (1, 2):
                        telemetry_writer.record_error()
                    diagnostics.record(
                        receive_monotonic_ns=time.monotonic_ns(), length=length,
                        seq=None, drops_fw=None, classification="missing_or_truncated_frame",
                    )
                else:
                    if received_format != stream_format:
                        stream_format = received_format
                        last_seq = None
                        if telemetry_writer is not None:
                            if stream_format == 0:
                                telemetry_writer.reset_to_legacy()
                            else:
                                telemetry_writer.select_format(stream_format)
                    seq, drops_fw = parsed.sequence, parsed.drops_fw

                    classification, missing_packets, starts_new_epoch = (
                        classify_sequence_transition(last_seq, seq)
                    )
                    if missing_packets:
                        counters[3] += missing_packets
                    diagnostics.record(
                        receive_monotonic_ns=time.monotonic_ns(), length=length,
                        seq=seq, drops_fw=drops_fw, classification=classification,
                        missing_packets=missing_packets,
                        starts_new_epoch=starts_new_epoch,
                    )
                    last_seq = seq

                    # drops_fw is a snapshot of the firmware-side cumulative
                    # ring-full drop counter at the moment this packet was
                    # stamped. It is monotonic across a streaming session
                    # (firmware zeroes it only on Boot_TriggerStart).
                    # Publish the latest observed value; consumers compute
                    # deltas. We don't take max() because a wraparound or
                    # firmware reset should be reflected verbatim.
                    counters[8] = drops_fw

                    # Convert payload to uint16 sample view (zero-copy slice
                    # of the libusb-owned buffer).
                    samples = parsed.samples
                    n = samples.shape[0]

                    if head + n <= ring_capacity:
                        ring[head:head + n] = samples
                        head += n
                        if head == ring_capacity:
                            head = 0
                    else:
                        first = ring_capacity - head
                        ring[head:] = samples[:first]
                        ring[:n - first] = samples[first:]
                        head = n - first

                    total += n
                    counters[0] = head
                    counters[1] = total
                    counters[2] += 1  # packets
                    if parsed.telemetry is not None and telemetry_writer is not None:
                        telemetry_writer.publish(parsed.telemetry)

            elif status == usb1.TRANSFER_TIMED_OUT:
                counters[5] += 1
            elif status in (usb1.TRANSFER_NO_DEVICE,):
                counters[4] += 1
                _set_flag(FLAG_DEVICE_LOST | FLAG_FATAL)
                # Don't resubmit; main loop will exit on the flag.
                return
            elif status == usb1.TRANSFER_CANCELLED:
                # Quiet — happens during shutdown.
                return
            else:
                counters[4] += 1  # transfer_errors (ERROR / STALL / OVERFLOW)

            # Resubmit unless we're shutting down.
            if not terminate_event.is_set():
                try:
                    transfer.submit()
                except usb1.USBError:
                    counters[4] += 1
                    _set_flag(FLAG_DEVICE_LOST | FLAG_FATAL)

        # Build and submit the transfer pool.
        for _ in range(NUM_TRANSFERS):
            t = handle.getTransfer()
            t.setBulk(
                BULK_IN_EP,
                TRANSFER_SIZE,
                callback=_on_complete,
                timeout=TRANSFER_TIMEOUT_MS,
            )
            t.submit()
            transfers.append(t)

        # Signal parent we're up and running.
        _set_flag(FLAG_READY)
        ready_event.set()

        # Event loop: pump libusb events and drain the outbound command queue.
        while not terminate_event.is_set():
            if pending_control is not None and time.monotonic() > pending_control[4]:
                request_id, _kind, _format, delivery, _deadline = pending_control
                if command_ack_queue is not None:
                    command_ack_queue.put((request_id, {**delivery,
                        "transport_status": "failed",
                        "error": "firmware control response timed out"}, time.monotonic_ns()))
                pending_control = None
            # Handle at most one command per event-loop turn so OUT traffic
            # cannot starve IN completion processing. Accepted diagnostic
            # commands normally complete immediately; 100 ms is the hard host
            # bound for a NAKed/stalled OUT request.
            try:
                cmd_item = cmd_queue.get_nowait()
                if cmd_item is None:
                    terminate_event.set()
                else:
                    _send_outbound(cmd_item, timeout_ms=100)
            except queue.Empty:
                pass

            # Pump completions for up to 50 ms, then loop.
            try:
                ctx.handleEventsTimeout(0.05)
            except usb1.USBError:
                counters[4] += 1
                if int(counters[7]) & FLAG_DEVICE_LOST:
                    break

    except Exception:  # noqa: BLE001
        _set_flag(FLAG_FATAL)
        # Make sure the parent doesn't block forever if the failure happened
        # before we had a chance to signal ready.
        if not ready_event.is_set():
            ready_event.set()
    finally:
        try:
            diagnostics.write()
        except Exception:  # diagnostics must never break reader cleanup
            pass
        # Final drain of the outbound command queue — ensures late-enqueued
        # commands (e.g. IDLE sent immediately before terminate_event) reach
        # the device before we tear down the handle. Without this, ctrl.stop()
        # followed promptly by ctrl.end_stream()/close() can race the loop
        # and leave the laser on.
        if handle is not None and cmd_queue is not None:
            try:
                while True:
                    cmd_item = cmd_queue.get_nowait()
                    if cmd_item is None:
                        continue
                    try:
                        _send_outbound(cmd_item, timeout_ms=100)
                    except Exception:  # noqa: BLE001
                        counters[4] += 1
            except (queue.Empty, ValueError, OSError):
                pass

        # Cancel pending transfers and wait briefly for callbacks to fire.
        for t in transfers:
            try:
                t.cancel()
            except Exception:  # noqa: BLE001
                pass
        if ctx is not None:
            deadline = time.monotonic() + 0.5
            while time.monotonic() < deadline:
                try:
                    ctx.handleEventsTimeout(0.01)
                except Exception:  # noqa: BLE001
                    break
        if handle is not None:
            try:
                handle.releaseInterface(0)
            except Exception:  # noqa: BLE001
                pass
            try:
                handle.close()
            except Exception:  # noqa: BLE001
                pass
        if telemetry_shm is not None:
            telemetry_shm.close()
        if ctx is not None:
            try:
                ctx.close()
            except Exception:  # noqa: BLE001
                pass
        # Detach numpy views before the parent unlinks shared memory.
        del ring
        del counters
        try:
            ring_shm.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            counter_shm.close()
        except Exception:  # noqa: BLE001
            pass
