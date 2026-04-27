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

- ``drops_seq``    : packets missing per the firmware sequence header.  Data
                    was lost between firmware FIFO write and host receive —
                    could be firmware FIFO overflow OR USB stack loss.
                    Distinguishing the two requires firmware-side counters.
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
import os
import platform
import struct
import sys
import threading
import time
from multiprocessing import Event, Process, Queue
from multiprocessing.shared_memory import SharedMemory
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Shared memory layout
# ---------------------------------------------------------------------------

# Counter block: 8 × uint64 = 64 bytes, naturally aligned for atomic writes
# on x86-64.
_COUNTER_DTYPE = np.uint64
_COUNTER_BYTES = 64
_COUNTER_FIELDS = (
    'head_samples',      # 0  — write index in ring buffer (mod capacity)
    'total_samples',     # 1  — cumulative samples written
    'packets',           # 2  — cumulative valid packets received
    'drops_seq',         # 3  — cumulative packets missed (sequence gaps)
    'transfer_errors',   # 4  — libusb non-OK / non-timeout statuses
    'transfer_timeouts', # 5  — libusb timeout count
    'malformed',         # 6  — packets with invalid byte count
    'flags',             # 7  — bit0: ready, bit1: fatal_error, bit2: device_lost
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
PSSI_HEADER_BYTES = 4
PSSI_PAYLOAD_BYTES = 16384
PSSI_PACKET_WIRE_BYTES = PSSI_HEADER_BYTES + PSSI_PAYLOAD_BYTES

# Async transfer pool — see module docstring for sizing rationale.
#
# Sizing note: USB bulk transfers terminate on a short packet, and our
# 16388-byte wire payload ends with a 4-byte short packet (16388 mod
# wMaxPacketSize=512 == 4).  Therefore each completed transfer carries
# **exactly one** logical packet, regardless of the buffer size we hand
# libusb.  Buffering in flight = NUM_TRANSFERS packets, NOT
# NUM_TRANSFERS * TRANSFER_SIZE / packet_size.
#
# At 10 MSPS that's ~1219 packets/s, so 256 transfers ≈ 210 ms of host-side
# buffering — enough to ride out routine Windows scheduler stalls without
# letting the firmware FIFO overflow.  Bump higher (env override) if you
# still see drops_seq incrementing while xerr/xto stay at 0.
NUM_TRANSFERS = int(os.environ.get('UC_NUM_TRANSFERS', '256'))
TRANSFER_SIZE = int(os.environ.get('UC_TRANSFER_SIZE', str(64 * 1024)))  # >= 16388
TRANSFER_TIMEOUT_MS = 1000

# Bulk endpoint addresses (device-specific; mirrored from protocol.py).
BULK_IN_EP = 0x81
BULK_OUT_EP = 0x01

# How aggressively to drain the command queue inside the event loop.
COMMAND_POLL_INTERVAL_S = 0.005


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
    ring = np.ndarray((ring_capacity,), dtype=np.uint16, buffer=ring_shm.buf)
    counters = _counter_view(counter_shm)

    # Local mirror of head/total — the subprocess is the sole writer, so we
    # avoid round-tripping through shared memory on the hot path and only
    # publish the running totals after each packet.
    head = 0
    total = 0
    last_seq: Optional[int] = None

    # Sequence-gap sanity bound: gaps larger than this are treated as
    # device reset / wraparound noise rather than real loss.
    _MAX_SANE_GAP = 1000

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

        # ------------------------------------------------------------------
        # Per-transfer completion callback — called on libusb's event thread
        # (here, the main subprocess thread inside handleEvents()).
        # ------------------------------------------------------------------
        def _on_complete(transfer):
            nonlocal head, total, last_seq

            status = transfer.getStatus()

            if status == usb1.TRANSFER_COMPLETED:
                length = transfer.getActualLength()
                if length < PSSI_PACKET_WIRE_BYTES or (length - PSSI_HEADER_BYTES) % 2:
                    counters[6] += 1  # malformed
                else:
                    raw = transfer.getBuffer()[:length]
                    seq = struct.unpack_from('<I', raw, 0)[0]

                    if last_seq is not None:
                        delta = (seq - last_seq) & 0xFFFFFFFF
                        if 1 < delta <= _MAX_SANE_GAP:
                            counters[3] += (delta - 1)  # drops_seq
                    last_seq = seq

                    # Convert payload to uint16 sample view (zero-copy slice
                    # of the libusb-owned buffer).
                    samples = np.frombuffer(
                        raw, dtype=np.uint16, offset=PSSI_HEADER_BYTES,
                        count=(length - PSSI_HEADER_BYTES) // 2,
                    )
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
            # Drain any pending outbound commands (BOOT/IDLE/WARM bytes etc).
            try:
                while True:
                    cmd_bytes = cmd_queue.get_nowait()
                    if cmd_bytes is None:
                        # Sentinel — used as an alternative shutdown signal.
                        terminate_event.set()
                        break
                    try:
                        handle.bulkWrite(BULK_OUT_EP, cmd_bytes, timeout=2000)
                    except usb1.USBError:
                        counters[4] += 1
            except Exception:  # noqa: BLE001 — queue.Empty path
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
