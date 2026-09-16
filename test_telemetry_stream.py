import struct
import queue
import sys
import threading
import time
import unittest
import zlib
from multiprocessing.shared_memory import SharedMemory
from types import SimpleNamespace
from unittest import mock

import numpy as np

from ultracoustics.controller import Controller
from ultracoustics._internal import stream_proc
from ultracoustics._internal.protocol import pack_command, CMD_STREAM_CAPABILITIES, CMD_STREAM_FORMAT
from ultracoustics._internal.telemetry import (
    BoardTelemetry, FORMAT1_RECORD_BYTES, LEGACY_RECORD_BYTES,
    SIDECAR_BYTES, TelemetryFormatError, TelemetrySidecarWriter,
    initialise_sidecar, parse_capabilities, parse_format1_trailer,
    parse_format_ack, parse_record, read_sidecar,
)
from ultracoustics._internal.optical_diagnostics import crc16_ccitt


def board_blob(board, sequence=7):
    return struct.pack(
        "<HHIIiIIHHHHIIHhHbBHH",
        1, board, sequence, 1234, 25123, 1200, 34, 1, 0,
        2000, 4095, 1201, 33, 3300, -4, 0, 1, 0, 1, 0,
    )


def format1_record(seq=12, epoch=3, tlvs=None):
    legacy = struct.pack("<II", seq, 2) + np.arange(8192, dtype="<u2").tobytes()
    if tlvs is None:
        tlvs = [(1, board_blob(638) + b"\0\0"),
                (2, board_blob(1550) + b"\0\0")]
    trailer = bytearray(struct.pack("<4sBBHIHH", b"UTL1", 1, 16, 140, epoch, 0, 3))
    for kind, payload in tlvs:
        trailer += struct.pack("<HH", kind, len(payload)) + payload
    trailer += struct.pack("<HHI", 0x7FFF, 4, zlib.crc32(trailer) & 0xFFFFFFFF)
    assert len(legacy + trailer) == FORMAT1_RECORD_BYTES
    return legacy + trailer


def format2_record(kind=3, page_kind=2, seq=12, epoch=3, page_age_ms=0):
    if kind == 1:
        rotating_payload = board_blob(638) + b'\0\0'
    else:
        page = bytearray(52)
        struct.pack_into('<BBH', page, 0, 1, page_kind, 52)
        if page_kind == 5:
            struct.pack_into('<IHHIHBB', page, 4, 9, 0, 4096,
                             144_000_000, 1, 2, 12)
            struct.pack_into('<IHHhH', page, 20, 100, 6000, 20000, 0, 1)
            struct.pack_into('<IHHhH', page, 32, 200, 6001, 20001, 0, 1)
            struct.pack_into('<I', page, 44, 1234)
        struct.pack_into('<H', page, 50, crc16_ccitt(page[:50]))
        rotating_payload = bytes(page) + struct.pack('<H', page_age_ms)
    tlvs = [(kind, rotating_payload),
            (2, board_blob(1550) + b'\0\0')]
    legacy = struct.pack('<II', seq, 2) + np.arange(8192, dtype='<u2').tobytes()
    trailer = bytearray(struct.pack('<4sBBHIHH', b'UTL1', 2, 16, 140, epoch, 0, 3))
    for tlv_type, payload in tlvs:
        trailer += struct.pack('<HH', tlv_type, len(payload)) + payload
    trailer += struct.pack('<HHI', 0x7FFF, 4, zlib.crc32(trailer) & 0xFFFFFFFF)
    return legacy + trailer


class TelemetryParserTests(unittest.TestCase):
    def test_attach_to_retained_format_without_control_commands(self):
        record, mode = stream_proc.parse_attached_record(format1_record(), 0, 99)
        self.assertEqual(mode, 1)
        self.assertEqual(record.samples.shape, (8192,))
        self.assertEqual(record.telemetry.received_monotonic_ns, 99)
        legacy, mode = stream_proc.parse_attached_record(
            format1_record()[:LEGACY_RECORD_BYTES], mode, 100)
        self.assertEqual(mode, 0)
        self.assertIsNone(legacy.telemetry)

    def test_attach_rejects_corrupt_format_candidate(self):
        raw = bytearray(format1_record())
        raw[-1] ^= 1
        with self.assertRaises(TelemetryFormatError):
            stream_proc.parse_attached_record(raw, 0, 99)

    def test_control_responses_are_exact_and_validated(self):
        caps = struct.pack("<4sHHIHHHBBI", b"UTCP", 1, 24, 1,
                           16392, 16532, 503, 0, 0, 9)
        parsed = parse_capabilities(caps)
        self.assertTrue(parsed.supports_format1)
        self.assertEqual(parsed.stream_epoch, 9)
        ack = struct.pack("<4sHHB3sI", b"UTFM", 1, 16, 1, b"\0\0\0", 10)
        self.assertEqual(parse_format_ack(ack, 1), 10)
        with self.assertRaises(TelemetryFormatError):
            parse_format_ack(ack, 0)
        with self.assertRaises(TelemetryFormatError):
            parse_capabilities(caps + b"\0")

    def test_legacy_is_exact_and_always_8192_samples(self):
        raw = struct.pack("<II", 1, 2) + np.arange(8192, dtype="<u2").tobytes()
        parsed = parse_record(raw, 0)
        self.assertEqual(parsed.samples.shape, (8192,))
        self.assertIsNone(parsed.telemetry)
        with self.assertRaises(TelemetryFormatError):
            parse_record(raw + b"xx", 0)

    def test_format1_parses_sidecar_without_polluting_samples(self):
        parsed = parse_record(format1_record(), 1, received_monotonic_ns=99)
        self.assertEqual(parsed.samples.shape, (8192,))
        self.assertEqual(int(parsed.samples[-1]), 8191)
        self.assertEqual(parsed.telemetry.stream_epoch, 3)
        self.assertEqual(parsed.telemetry.board_638.board, 638)
        self.assertEqual(parsed.telemetry.board_1550.temperature_mdegc, 25123)
        self.assertEqual(parsed.telemetry.received_monotonic_ns, 99)

    def test_format_must_be_explicit(self):
        with self.assertRaises(TelemetryFormatError):
            parse_record(format1_record(), 0)
        legacy = format1_record()[:LEGACY_RECORD_BYTES]
        with self.assertRaises(TelemetryFormatError):
            parse_record(legacy, 1)

    def test_crc_and_integrity_position_enforced(self):
        corrupted = bytearray(format1_record())
        corrupted[LEGACY_RECORD_BYTES + 32] ^= 1
        with self.assertRaisesRegex(TelemetryFormatError, "CRC"):
            parse_record(corrupted, 1)
        tlvs = [(1, board_blob(638) + b"\0\0"),
                (0x8001, board_blob(1550) + b"\0\0")]
        with self.assertRaises(TelemetryFormatError):
            parse_record(format1_record(tlvs=tlvs), 1)

    def test_unknown_required_fails_and_link_flags_are_preserved(self):
        tlvs = [(1, board_blob(638) + b"\0\0"),
                (3, board_blob(1550) + b"\0\0")]
        with self.assertRaisesRegex(TelemetryFormatError, "unknown required"):
            parse_record(format1_record(tlvs=tlvs), 1)
        tlvs = [(1, board_blob(638) + struct.pack("<H", 0x8001)),
                (2, board_blob(1550) + b"\0\0")]
        parsed = parse_record(format1_record(tlvs=tlvs), 1)
        self.assertEqual(parsed.telemetry.link_flags_638, 0x8001)
        self.assertTrue(parsed.telemetry.link_638_stale)

    def test_bounded_future_trailer_skips_unknown_optional_tlv(self):
        trailer = bytearray(struct.pack("<4sBBHIHH", b"UTL1", 1, 16, 147,
                                        5, 0, 4))
        for kind, payload in ((1, board_blob(638) + b"\0\0"),
                              (0x8003, b"abc"),
                              (2, board_blob(1550) + b"\0\0")):
            trailer += struct.pack("<HH", kind, len(payload)) + payload
        trailer += struct.pack("<HHI", 0x7FFF, 4,
                               zlib.crc32(trailer) & 0xFFFFFFFF)
        epoch, board638, board1550, link638, link1550 = parse_format1_trailer(trailer)
        self.assertEqual((epoch, board638.board, board1550.board), (5, 638, 1550))
        self.assertEqual((link638, link1550), (0, 0))

    def test_shared_sidecar_is_coherent_and_tracks_epoch(self):
        shm = SharedMemory(create=True, size=SIDECAR_BYTES)
        try:
            initialise_sidecar(shm)
            writer = TelemetrySidecarWriter(shm)
            first = parse_record(format1_record(epoch=3), 1).telemetry
            second = parse_record(format1_record(seq=13, epoch=4), 1).telemetry
            writer.publish(first)
            writer.publish(second)
            snapshot, stats = read_sidecar(shm)
            self.assertEqual(snapshot.stream_epoch, 4)
            self.assertEqual(snapshot.record_sequence, 13)
            self.assertEqual(snapshot.link_flags_638, 0)
            self.assertEqual(stats["records"], 2)
            self.assertEqual(stats["epoch_changes"], 1)
            self.assertEqual(stats["active_format"], 1)
            writer.record_error()
            self.assertEqual(read_sidecar(shm)[1]["parse_errors"], 1)
            writer.reset_to_legacy()
            self.assertIsNone(read_sidecar(shm)[0])
            self.assertEqual(read_sidecar(shm)[1]["active_format"], 0)
        finally:
            shm.close()
            shm.unlink()

    def test_host_stale_ages_without_new_publication(self):
        parsed = parse_record(format1_record(), 1,
                              received_monotonic_ns=time.monotonic_ns() - 600_000_000)
        self.assertTrue(parsed.telemetry.host_stale)

    def test_format2_rotating_pages_publish_independently(self):
        shm = SharedMemory(create=True, size=SIDECAR_BYTES)
        try:
            initialise_sidecar(shm)
            writer = TelemetrySidecarWriter(shm)
            # Seed the independently rotating 638 sensor cache.
            writer.publish(parse_record(format1_record(epoch=8), 1).telemetry)
            writer.publish(parse_record(format2_record(3, 2, 20, 8), 2,
                                        received_monotonic_ns=time.monotonic_ns()).telemetry)
            writer.publish(parse_record(format2_record(4, 3, 21, 8), 2,
                                        received_monotonic_ns=time.monotonic_ns()).telemetry)
            snapshot, stats = read_sidecar(shm)
            self.assertEqual(stats['active_format'], 2)
            self.assertIsNotNone(snapshot.optical_live_638)
            self.assertIsNotNone(snapshot.optical_acquisition_638)
            self.assertIsNone(snapshot.optical_abba_638)
            self.assertEqual(snapshot.board_638.board, 638)
            writer.publish(parse_record(format1_record(seq=22, epoch=9), 1).telemetry)
            writer.publish(parse_record(format2_record(5, 4, 23, 9), 2).telemetry)
            snapshot, _ = read_sidecar(shm)
            self.assertIsNone(snapshot.optical_live_638)
            self.assertIsNone(snapshot.optical_acquisition_638)
            self.assertIsNotNone(snapshot.optical_abba_638)
            writer.publish(parse_record(format2_record(6, 5, 24, 9), 2).telemetry)
            snapshot, _ = read_sidecar(shm)
            self.assertIsNotNone(snapshot.optical_trace_638)
            self.assertEqual(snapshot.optical_trace_638.page.samples[1].feedback, 6001)
        finally:
            shm.close(); shm.unlink()

    def test_format2_sensor_records_keep_selected_format(self):
        shm = SharedMemory(create=True, size=SIDECAR_BYTES)
        try:
            initialise_sidecar(shm); writer = TelemetrySidecarWriter(shm)
            writer.publish(parse_record(format2_record(1, seq=30), 2).telemetry)
            self.assertEqual(read_sidecar(shm)[1]['active_format'], 2)
            writer.publish(parse_record(format2_record(3, 2, 31), 2).telemetry)
            self.assertEqual(read_sidecar(shm)[1]['active_format'], 2)
            with self.assertRaisesRegex(TelemetryFormatError, 'trailer version'):
                parse_record(format2_record(1), 1)
        finally:
            shm.close(); shm.unlink()

    def test_optical_master_age_is_included_in_host_staleness(self):
        shm = SharedMemory(create=True, size=SIDECAR_BYTES)
        try:
            initialise_sidecar(shm); writer = TelemetrySidecarWriter(shm)
            now = time.monotonic_ns()
            writer.publish(parse_record(format2_record(1, epoch=7), 2,
                                        received_monotonic_ns=now).telemetry)
            writer.publish(parse_record(format2_record(3, 2, epoch=7,
                                                       page_age_ms=600), 2,
                                        received_monotonic_ns=now).telemetry)
            snapshot, _ = read_sidecar(shm)
            self.assertTrue(snapshot.optical_live_638.host_stale)
            self.assertEqual(snapshot.link_flags_638, 0)
        finally:
            shm.close(); shm.unlink()


class FakeNegotiatedStream:
    running = True

    def __init__(self, snapshot, capability_flags=1):
        self.snapshot = snapshot
        self.calls = []
        self.capability_flags = capability_flags

    def query_stream_capabilities(self, payload, timeout_s):
        self.calls.append(("caps", payload, timeout_s))
        raw = struct.pack("<4sHHIHHHBBI", b"UTCP", 1, 24, self.capability_flags,
                          16392, 16532, 503, 0, 0, self.snapshot.stream_epoch - 1)
        return parse_capabilities(raw)

    def select_stream_format_confirmed(self, payload, stream_format, timeout_s):
        self.calls.append(("format", payload, stream_format, timeout_s))
        return {"firmware_response": {"accepted_format": stream_format,
                                       "stream_epoch": self.snapshot.stream_epoch}}

    def get_telemetry(self):
        return self.snapshot


class ControllerTelemetryTests(unittest.TestCase):
    def controller(self):
        snapshot = parse_record(format1_record(epoch=10), 1).telemetry
        controller = Controller.__new__(Controller)
        controller._running = False
        controller._stream = FakeNegotiatedStream(snapshot)
        return controller

    def test_enable_requires_caps_ack_and_matching_record(self):
        controller = self.controller()
        result = controller.enable_telemetry(timeout_s=0.1)
        self.assertEqual(result.stream_epoch, 10)
        self.assertEqual(controller._stream.calls[0][1],
                         pack_command(CMD_STREAM_CAPABILITIES, 0, 0))
        self.assertEqual(controller._stream.calls[1][1],
                         pack_command(CMD_STREAM_FORMAT, 1, 0))

    def test_negotiation_rejected_while_running(self):
        controller = self.controller()
        controller._running = True
        with self.assertRaisesRegex(RuntimeError, "IDLE-only"):
            controller.telemetry_capabilities()

    def test_disable_requests_exact_legacy_format(self):
        controller = self.controller()
        controller.disable_telemetry(timeout_s=0.1)
        self.assertEqual(controller._stream.calls[-1][1],
                         pack_command(CMD_STREAM_FORMAT, 0, 0))
        self.assertEqual(controller._stream.calls[-1][2], 0)

    def test_optical_diagnostics_requires_capability_bit_and_selects_two(self):
        controller = self.controller()
        with self.assertRaisesRegex(RuntimeError, 'does not advertise'):
            controller.enable_optical_diagnostics(.1)
        controller._stream = FakeNegotiatedStream(controller.telemetry,
                                                   capability_flags=5)
        controller.enable_optical_diagnostics(.1)
        self.assertEqual(controller._stream.calls[-1][2], 2)


class ReentrantControlResponseTests(unittest.TestCase):
    def test_capability_and_ack_arriving_inside_bulk_write_are_recognized(self):
        class FakeUSBError(Exception):
            pass

        fake_usb = SimpleNamespace(
            TRANSFER_COMPLETED=0, TRANSFER_TIMED_OUT=1,
            TRANSFER_NO_DEVICE=2, TRANSFER_CANCELLED=3,
            USBError=FakeUSBError, USBErrorPipe=type("USBErrorPipe", (FakeUSBError,), {}),
        )

        class Transfer:
            def setBulk(self, _endpoint, _size, callback, timeout):
                self.callback = callback
                self.timeout = timeout

            def submit(self):
                pass

            def cancel(self):
                pass

            def getStatus(self):
                return fake_usb.TRANSFER_COMPLETED

            def getActualLength(self):
                return len(self.buffer)

            def getBuffer(self):
                return self.buffer

        class Handle:
            def __init__(self):
                self.transfers = []

            def kernelDriverActive(self, _interface):
                return False

            def claimInterface(self, _interface):
                pass

            def releaseInterface(self, _interface):
                pass

            def close(self):
                pass

            def getTransfer(self):
                transfer = Transfer()
                self.transfers.append(transfer)
                return transfer

            def bulkWrite(self, _endpoint, payload, timeout):
                self.timeout = timeout
                if payload[0] == ord("c"):
                    response = struct.pack("<4sHHIHHHBBI", b"UTCP", 1, 24, 1,
                                           16392, 16532, 503, 0, 0, 9)
                else:
                    response = struct.pack("<4sHHB3sI", b"UTFM", 1, 16, 1,
                                           b"\0\0\0", 10)
                # This is the regression: synchronous libusb I/O may pump a
                # previously submitted async IN callback before returning.
                transfer = self.transfers[0]
                transfer.buffer = response
                transfer.callback(transfer)
                if payload[0] == ord("x"):
                    raise FakeUSBError("late OUT exception after valid UTFM")
                return len(payload)

        handle = Handle()

        class Context:
            def open(self):
                pass

            def openByVendorIDAndProductID(self, *_args, **_kwargs):
                return handle

            def handleEventsTimeout(self, _timeout):
                pass

            def close(self):
                pass

        fake_usb.USBContext = Context
        ring = SharedMemory(create=True, size=64)
        counters = SharedMemory(create=True, size=stream_proc._COUNTER_BYTES)
        sidecar = SharedMemory(create=True, size=SIDECAR_BYTES)
        ring.buf[:] = b"\0" * len(ring.buf)
        counters.buf[:] = b"\0" * len(counters.buf)
        initialise_sidecar(sidecar)
        commands = queue.Queue()
        acknowledgements = queue.Queue()
        commands.put((1, pack_command(CMD_STREAM_CAPABILITIES), "capabilities", None))
        commands.put((2, pack_command(CMD_STREAM_FORMAT, 1), "format", 1))
        commands.put(None)
        try:
            with mock.patch.dict(sys.modules, {"usb1": fake_usb}):
                stream_proc.reader_main(
                    0x2E9D, 0x000A, ring.name, counters.name, 32, commands,
                    threading.Event(), threading.Event(), acknowledgements,
                    telemetry_shm_name=sidecar.name,
                )
            first = acknowledgements.get_nowait()
            second = acknowledgements.get_nowait()
            self.assertEqual(first[1]["firmware_response"].stream_epoch, 9)
            self.assertEqual(second[1]["firmware_response"],
                             {"accepted_format": 1, "stream_epoch": 10})
            self.assertEqual(read_sidecar(sidecar)[1]["active_format"], 1)
            counter_values = stream_proc.read_counters(counters)
            self.assertEqual(counter_values["transfer_errors"], 0)
            self.assertTrue(acknowledgements.empty())
        finally:
            for shm in (ring, counters, sidecar):
                shm.close()
                shm.unlink()


if __name__ == "__main__":
    unittest.main()
