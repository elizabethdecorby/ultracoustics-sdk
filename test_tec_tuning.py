import math
import struct
import unittest

from ultracoustics import identity as ident
from ultracoustics import tec_tuning as tn
from ultracoustics._internal import protocol
from ultracoustics._internal.maintenance import DiagnosticsManager


def reply(op, target, status, record=b"\xFF" * tn.RECORD_SIZE, kind=tn.KIND,
          slots_free=255):
    return (ident.REPLY_MAGIC
            + struct.pack("<BBHBBBB", ident.REPLY_SCHEMA, op, target, status,
                          slots_free, kind, 0)
            + record)


def raw_record(board, payload, seq=1):
    """A stored record with any payload bytes, bypassing the encoder's checks."""
    header = tn.MAGIC + struct.pack("<BBHI", 1, board, len(payload), seq)
    raw = header + struct.pack("<I", ident.crc32_mpeg2(header + payload)) + payload
    return raw + b"\xFF" * (tn.RECORD_SIZE - len(raw))


def tuning_638(**kw):
    values = dict(board=tn.BOARD_638, kp=0.07, ki=0.22, integrator_limit=1.36,
                  setpoint_filter=True, conditional_integration=True,
                  tuned_at=1_790_000_000, report_id="TUNE_X")
    values.update(kw)
    return tn.TecTuning(**values)


# Shared with the C tests in 1550_boot, 1550 and 638-Laser-Firmware: the
# record the firmware must accept (kp 0.07, ki 0.22, limit 1.36, both flags).
FIXTURE_638_SEQ3 = bytes.fromhex(
    "5543544e0103300003000000c9212ff2"  # header, CRC 0xF22F21C9
    "295c8f3dae47613effffffff7b14ae3f"  # kp, ki, kd unset, limit
    "0101ffff803bb16a54554e455f58ffff"  # flags, reserved, tuned_at, "TUNE_X"
    "ffffffffffffffffffffffffffffffff"
)


class CodecTests(unittest.TestCase):
    def test_round_trip_keeps_float32_values(self):
        stored = tn.parse_record(tn.build_record(tuning_638(), seq=3))
        self.assertEqual((stored.board, stored.seq), (tn.BOARD_638, 3))
        self.assertAlmostEqual(stored.tuning.kp, 0.07, places=6)
        self.assertAlmostEqual(stored.tuning.ki, 0.22, places=6)
        self.assertIsNone(stored.tuning.kd)
        self.assertTrue(stored.tuning.setpoint_filter)
        self.assertEqual(stored.tuning.report_id, "TUNE_X")
        self.assertTrue(tn.same_record(stored.tuning, tuning_638()))

    def test_fixture_header_and_crc(self):
        raw = tn.build_record(tuning_638(), seq=3)
        self.assertEqual(raw[:64], FIXTURE_638_SEQ3)
        self.assertEqual(raw[64:], b"\xFF" * 64)

    def test_unset_fields_are_erased_bytes(self):
        raw = tn.build_record(tn.TecTuning(board=tn.BOARD_1550, ki=0.091))
        payload = raw[16:16 + 48]
        self.assertEqual(payload[0:4], b"\xFF" * 4)            # kp
        self.assertEqual(struct.unpack("<f", payload[4:8])[0], struct.unpack(
            "<f", struct.pack("<f", 0.091))[0])                  # ki
        self.assertEqual(payload[8:20], b"\xFF" * 12)           # kd, limit, flags, reserved
        decoded = tn.parse_record(raw).tuning
        self.assertEqual(set(decoded.settings()), {"ki"})

    def test_empty_payload_sets_nothing(self):
        header = tn.MAGIC + struct.pack("<BBHI", 1, tn.BOARD_1550, 0, 7)
        raw = header + struct.pack("<I", ident.crc32_mpeg2(header)) + b"\xFF" * 112
        self.assertEqual(tn.parse_record(raw).tuning.settings(), {})

    def test_short_payload_drops_partial_fields(self):
        payload = struct.pack("<f", 0.05) + b"\x00\x00"  # kp, then half of ki
        header = tn.MAGIC + struct.pack("<BBHI", 1, tn.BOARD_1550, len(payload), 1)
        raw = header + struct.pack("<I", ident.crc32_mpeg2(header + payload)) + payload
        raw += b"\xFF" * (128 - len(raw))
        self.assertEqual(set(tn.parse_record(raw).tuning.settings()), {"kp"})

    def test_bad_crc_and_magic_rejected(self):
        raw = bytearray(tn.build_record(tuning_638()))
        raw[20] ^= 1
        with self.assertRaises(tn.TuningError):
            tn.parse_record(bytes(raw))
        with self.assertRaises(tn.TuningError):
            tn.parse_record(ident.build_record(ident.BoardIdentity(board=2, serial="A")))

    def test_erased_record_is_none(self):
        self.assertIsNone(tn.parse_record(b"\xFF" * 128))

    def test_record_the_firmware_rejects_still_reads(self):
        bad_flag = bytearray(tn.encode_payload(tuning_638()))
        bad_flag[17] = 2
        stored = tn.parse_record(raw_record(tn.BOARD_638, bytes(bad_flag)))
        self.assertIn("conditional_integration", stored.rejected)
        self.assertTrue(stored.tuning.setpoint_filter)
        too_big = struct.pack("<f", 90.0) + b"\xFF" * 44
        stored = tn.parse_record(raw_record(tn.BOARD_1550, too_big))
        self.assertEqual(stored.tuning.kp, 90.0)
        self.assertIn("kp", stored.rejected)
        self.assertIsNone(tn.parse_record(tn.build_record(tuning_638())).rejected)

    def test_report_id_only_when_inside_payload(self):
        payload = tn.encode_payload(tuning_638())[:40]  # report_id cut short
        self.assertIsNone(tn.parse_record(raw_record(tn.BOARD_638, payload)).tuning.report_id)

    def test_effective_integrator_limit(self):
        self.assertEqual(tn.TecTuning(board=2).effective_integrator_limit(3.3), 3.3)
        self.assertAlmostEqual(
            tn.TecTuning(board=2, ki=0.1).effective_integrator_limit(3.3), 3.0, places=5)
        self.assertEqual(tn.TecTuning(board=2, ki=0.1, integrator_limit=2.0)
                         .effective_integrator_limit(3.3), 2.0)


class BoundsTests(unittest.TestCase):
    def check(self, ok, **kw):
        tuning = tn.TecTuning(board=tn.BOARD_638, **kw)
        if ok:
            tn.validate(tuning)
        else:
            with self.assertRaises(tn.TuningError):
                tn.validate(tuning)

    def test_edges(self):
        self.check(True, kp=0.0, ki=2.0, kd=0.5, integrator_limit=20.0)
        self.check(True, kp=1.0)
        self.check(False, kp=1.0001)
        self.check(False, kp=-0.01)
        self.check(False, ki=0.0)
        self.check(True, ki=0.001)
        self.check(False, ki=0.0009)
        self.check(False, ki=2.01)
        self.check(False, kd=0.6)
        self.check(False, integrator_limit=0.0)
        self.check(False, integrator_limit=25.0)

    def test_non_finite_rejected(self):
        self.check(False, kp=math.inf)
        self.check(False, ki=math.nan)
        self.check(False, kp=1e39)  # beyond float32 range

    def test_unit_mistake_rejected(self):
        self.check(False, kp=90.0)  # the console prints milli-units

    def test_flags_only_on_638(self):
        tn.validate(tn.TecTuning(board=tn.BOARD_1550, setpoint_filter=False))
        with self.assertRaises(tn.TuningError):
            tn.validate(tn.TecTuning(board=tn.BOARD_1550, setpoint_filter=True))
        with self.assertRaises(tn.TuningError):
            tn.validate(tn.TecTuning(board=tn.BOARD_1550, conditional_integration=True))
        tn.validate(tuning_638())

    def test_main_board_and_long_report_id_rejected(self):
        with self.assertRaises(tn.TuningError):
            tn.validate(tn.TecTuning(board=ident.BOARD_MAIN, kp=0.1))
        with self.assertRaises(tn.TuningError):
            tn.validate(tuning_638(report_id="x" * 25))


class ReplyTests(unittest.TestCase):
    def test_kind_one_reply_parses(self):
        record = tn.build_record(tuning_638(), seq=3)
        result = tn.parse_reply(reply(ident.OP_READ, 638, ident.STATUS_OK, record))
        self.assertTrue(result.ok)
        self.assertEqual(result.slots_free, 255)
        self.assertEqual(result.record.seq, 3)

    def test_identity_reply_means_old_mainboard(self):
        with self.assertRaises(tn.TuningError):
            tn.parse_reply(reply(ident.OP_READ, 638, ident.STATUS_EMPTY,
                                 kind=ident.KIND_IDENTITY))

    def test_identity_parser_refuses_tuning_reply(self):
        with self.assertRaises(ident.IdentityError):
            ident.parse_reply(reply(ident.OP_READ, 638, ident.STATUS_EMPTY))

    def test_board_type_must_match_target(self):
        record = tn.build_record(tuning_638(setpoint_filter=None,
                                            conditional_integration=None))
        result = tn.parse_reply(reply(ident.OP_READ, protocol.TARGET_1550,
                                      ident.STATUS_OK, record))
        self.assertIn("board type 3", result.record.rejected)

    def test_bad_kind_status_name(self):
        result = tn.parse_reply(reply(ident.OP_READ, 638, ident.STATUS_BAD_KIND))
        self.assertEqual(result.status_name, "BAD_KIND")


class FakeConnection:
    def __init__(self, replies):
        self.replies = list(replies)
        self.sent = []

    def send_command(self, cmd, wValue=0, wIndex=0, extra=b"", timeout_ms=5000):
        self.sent.append((cmd, wValue, wIndex, bytes(extra)))

    def receive(self, size, timeout_ms=500):
        return self.replies.pop(0) if self.replies else None


class TransportTests(unittest.TestCase):
    def test_read_sends_kind_and_target(self):
        conn = FakeConnection([
            reply(ident.OP_READ, protocol.TARGET_1550, ident.STATUS_EMPTY),  # other target
            reply(ident.OP_READ, protocol.TARGET_638, ident.STATUS_OK,
                  tn.build_record(tuning_638(), seq=2)),
        ])
        result = DiagnosticsManager(conn).read_tec_tuning(protocol.TARGET_638)
        self.assertEqual(conn.sent, [(protocol.CMD_IDENTITY_READ, 1, 638, b"")])
        self.assertEqual(result.record.seq, 2)

    def test_write_sends_record_and_verifies_readback(self):
        tuning = tuning_638()
        conn = FakeConnection([reply(ident.OP_WRITE, protocol.TARGET_638, ident.STATUS_OK,
                                     tn.build_record(tuning, seq=9))])
        result = DiagnosticsManager(conn).write_tec_tuning(protocol.TARGET_638, tuning)
        cmd, value, target, payload = conn.sent[0]
        self.assertEqual((cmd, value, target, len(payload)),
                         (protocol.CMD_IDENTITY_WRITE, 1, 638, 128))
        self.assertEqual(payload[:4], b"UCTN")
        self.assertTrue(result.ok, result.status_name)

    def test_readback_mismatch_becomes_verify_fail(self):
        conn = FakeConnection([reply(ident.OP_WRITE, protocol.TARGET_638, ident.STATUS_OK,
                                     tn.build_record(tuning_638(kp=0.08), seq=9))])
        result = DiagnosticsManager(conn).write_tec_tuning(protocol.TARGET_638, tuning_638())
        self.assertEqual(result.status_name, "VERIFY_FAIL")

    def test_out_of_bounds_rejected_before_sending(self):
        conn = FakeConnection([])
        with self.assertRaises(tn.TuningError):
            DiagnosticsManager(conn).write_tec_tuning(protocol.TARGET_638,
                                                      tuning_638(ki=5.0))
        self.assertEqual(conn.sent, [])

    def test_board_and_target_checks(self):
        diag = DiagnosticsManager(FakeConnection([]))
        with self.assertRaises(ValueError):
            diag.write_tec_tuning(protocol.TARGET_1550, tuning_638())
        with self.assertRaises(ValueError):
            diag.read_tec_tuning(protocol.TARGET_MAIN)

    def test_clear_writes_a_record_that_sets_nothing(self):
        conn = FakeConnection([reply(ident.OP_WRITE, protocol.TARGET_1550,
                                     ident.STATUS_WRITE_FAIL)])
        DiagnosticsManager(conn).clear_tec_tuning(protocol.TARGET_1550)
        sent = tn.parse_record(conn.sent[0][3]).tuning
        self.assertEqual(sent.settings(), {})
        self.assertIsNotNone(sent.tuned_at)

    def test_old_mainboard_raises(self):
        conn = FakeConnection([reply(ident.OP_WRITE, protocol.TARGET_638,
                                     ident.STATUS_BAD_RECORD, kind=ident.KIND_IDENTITY)])
        with self.assertRaises(tn.TuningError):
            DiagnosticsManager(conn).write_tec_tuning(protocol.TARGET_638, tuning_638())


if __name__ == "__main__":
    unittest.main()
