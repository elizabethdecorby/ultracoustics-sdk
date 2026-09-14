import struct
import time
import unittest
import zlib
from multiprocessing.shared_memory import SharedMemory

import numpy as np

from ultracoustics.controller import Controller
from ultracoustics._internal.protocol import pack_command, CMD_STREAM_CAPABILITIES, CMD_STREAM_FORMAT
from ultracoustics._internal.telemetry import (
    BoardTelemetry, FORMAT1_RECORD_BYTES, LEGACY_RECORD_BYTES,
    SIDECAR_BYTES, TelemetryFormatError, TelemetrySidecarWriter,
    initialise_sidecar, parse_capabilities, parse_format1_trailer,
    parse_format_ack, parse_record, read_sidecar,
)


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


class TelemetryParserTests(unittest.TestCase):
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

    def test_unknown_required_and_reserved_bytes_fail(self):
        tlvs = [(1, board_blob(638) + b"\0\0"),
                (3, board_blob(1550) + b"\0\0")]
        with self.assertRaisesRegex(TelemetryFormatError, "unknown required"):
            parse_record(format1_record(tlvs=tlvs), 1)
        tlvs = [(1, board_blob(638) + b"\0\1"),
                (2, board_blob(1550) + b"\0\0")]
        with self.assertRaisesRegex(TelemetryFormatError, "reserved"):
            parse_record(format1_record(tlvs=tlvs), 1)

    def test_bounded_future_trailer_skips_unknown_optional_tlv(self):
        trailer = bytearray(struct.pack("<4sBBHIHH", b"UTL1", 1, 16, 147,
                                        5, 0, 4))
        for kind, payload in ((1, board_blob(638) + b"\0\0"),
                              (0x8003, b"abc"),
                              (2, board_blob(1550) + b"\0\0")):
            trailer += struct.pack("<HH", kind, len(payload)) + payload
        trailer += struct.pack("<HHI", 0x7FFF, 4,
                               zlib.crc32(trailer) & 0xFFFFFFFF)
        epoch, board638, board1550 = parse_format1_trailer(trailer)
        self.assertEqual((epoch, board638.board, board1550.board), (5, 638, 1550))

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


class FakeNegotiatedStream:
    running = True

    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.calls = []

    def query_stream_capabilities(self, payload, timeout_s):
        self.calls.append(("caps", payload, timeout_s))
        raw = struct.pack("<4sHHIHHHBBI", b"UTCP", 1, 24, 1,
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


if __name__ == "__main__":
    unittest.main()
