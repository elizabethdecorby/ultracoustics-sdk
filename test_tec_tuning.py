import math
import struct
import unittest
from unittest import mock

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

# The same 638 record with lock PI fields (Kp 0.6, Ki 1925/s, zero 20 Hz),
# sequence 4: payload length 60, lock fields at payload offsets 48-59.
FIXTURE_638_LOCK_SEQ4 = bytes.fromhex(
    "5543544e01033c000400000075268cbb"  # header, CRC 0xBB8C2675
    "295c8f3dae47613effffffff7b14ae3f"
    "0101ffff803bb16a54554e455f58ffff"
    "ffffffffffffffffffffffffffffffff"
    "9a99193f00a0f04414ffffffffffffff"  # lock_kp, lock_ki_per_s, lock_zero_hz, reserved
)


def lock_638(**kw):
    values = dict(lock_kp=0.6, lock_ki_per_s=1925.0, lock_zero_hz=20)
    values.update(kw)
    return tuning_638(**values)


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

    def test_lock_fields_extend_the_payload_to_60_bytes(self):
        raw = tn.build_record(lock_638(), seq=4)
        self.assertEqual(raw[:80], FIXTURE_638_LOCK_SEQ4)
        self.assertEqual(raw[80:], b"\xFF" * 48)
        stored = tn.parse_record(raw)
        self.assertEqual((stored.tuning.lock_kp, stored.tuning.lock_ki_per_s,
                          stored.tuning.lock_zero_hz), (0.6, 1925.0, 20))  # 0.6, not 0.6000000238
        self.assertTrue(tn.same_record(stored.tuning, lock_638()))
        self.assertFalse(tn.same_record(stored.tuning, lock_638(lock_ki_per_s=1924.0)))
        self.assertIsNone(stored.rejected)
        self.assertIsNone(stored.lock_rejected)
        self.assertEqual(set(stored.tuning.settings()),  # TEC fields only
                         {"kp", "ki", "integrator_limit", "setpoint_filter",
                          "conditional_integration"})

    def test_lock_fields_alone_and_unset_encodings(self):
        raw = tn.build_record(tn.TecTuning(board=tn.BOARD_638, lock_kp=0.0, lock_ki_per_s=0.0,
                                           lock_zero_hz=0))
        self.assertEqual(raw[6], 60)
        self.assertEqual(raw[16:16 + 48], b"\xFF" * 48)
        decoded = tn.parse_record(raw).tuning
        self.assertEqual((decoded.lock_kp, decoded.lock_ki_per_s, decoded.lock_zero_hz), (0.0, 0.0, 0))
        self.assertEqual(decoded.settings(), {})
        nan = struct.pack("<f", math.nan)
        for lock in (b"\xFF" * 12, nan + nan + b"\xFF" * 4):
            decoded = tn.parse_record(raw_record(tn.BOARD_638, b"\xFF" * 48 + lock)).tuning
            self.assertEqual((decoded.lock_kp, decoded.lock_ki_per_s, decoded.lock_zero_hz),
                             (None, None, None))
        self.assertEqual(tn.build_record(tuning_638())[6], 48)  # no lock field: layout v1 length

    def test_lock_fields_past_the_payload_end_are_unset(self):
        payload = tn.encode_payload(lock_638())
        for length, expected in ((48, (None, None, None)), (52, (0.6, None, None)),
                                 (56, (0.6, 1925.0, None)), (57, (0.6, 1925.0, 20))):
            decoded = tn.parse_record(raw_record(tn.BOARD_638, payload[:length])).tuning
            self.assertEqual((decoded.lock_kp, decoded.lock_ki_per_s, decoded.lock_zero_hz), expected)

    def test_tec_and_lock_groups_are_rejected_independently(self):
        tec = tn.encode_payload(tuning_638())
        lock = tn.encode_payload(lock_638())[48:]
        bad_tec = struct.pack("<f", 90.0) + tec[4:]
        stored = tn.parse_record(raw_record(tn.BOARD_638, bad_tec + lock))
        self.assertIn("kp", stored.rejected)
        self.assertIsNone(stored.lock_rejected)
        for bad_lock, field in ((struct.pack("<f", 6.0) + lock[4:], "lock_kp"),
                                (lock[:4] + struct.pack("<f", math.inf) + lock[8:], "lock_ki_per_s"),
                                (lock[:8] + b"\x40" + lock[9:], "lock_zero_hz")):
            stored = tn.parse_record(raw_record(tn.BOARD_638, tec + bad_lock))
            self.assertIsNone(stored.rejected)
            self.assertIn(field, stored.lock_rejected)
        stored = tn.parse_record(raw_record(tn.BOARD_1550, b"\xFF" * 48 + lock))
        self.assertIsNone(stored.rejected)
        self.assertIn("638 only", stored.lock_rejected)

    def test_keep_stored_takes_the_other_group_from_the_board(self):
        stored = tn.parse_record(tn.build_record(lock_638(), seq=4))
        tec_write = tn.keep_stored(tn.TecTuning(board=tn.BOARD_638, kp=0.09, report_id="NEW"),
                                   stored, "lock")
        self.assertEqual((tec_write.kp, tec_write.ki, tec_write.report_id), (0.09, None, "NEW"))
        self.assertEqual((tec_write.lock_kp, tec_write.lock_ki_per_s, tec_write.lock_zero_hz),
                         (0.6, 1925.0, 20))
        lock_write = tn.keep_stored(tn.TecTuning(board=tn.BOARD_638, lock_kp=1.0), stored, "tec")
        self.assertTrue(lock_write.setpoint_filter)
        self.assertAlmostEqual(lock_write.ki, 0.22, places=6)
        self.assertEqual((lock_write.lock_kp, lock_write.lock_ki_per_s), (1.0, None))
        cleared = tn.keep_stored(tn.TecTuning(board=tn.BOARD_638), None, "tec")
        self.assertEqual(cleared, tn.TecTuning(board=tn.BOARD_638))
        lock = tn.encode_payload(lock_638())[48:]
        rejected = tn.parse_record(raw_record(tn.BOARD_638, struct.pack("<f", 90.0)
                                              + b"\xFF" * 44 + lock))
        dropped = tn.keep_stored(tn.TecTuning(board=tn.BOARD_638, lock_kp=1.0), rejected, "tec")
        self.assertEqual(dropped.settings(), {})  # the firmware ignores a rejected group anyway

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

    def test_lock_edges(self):
        self.check(True, lock_kp=0.0, lock_ki_per_s=0.0, lock_zero_hz=0)
        self.check(True, lock_kp=5.115, lock_ki_per_s=16383.0, lock_zero_hz=63)
        self.check(False, lock_kp=5.12)
        self.check(False, lock_kp=-0.001)
        self.check(False, lock_ki_per_s=16384.0)
        self.check(False, lock_ki_per_s=math.inf)
        self.check(False, lock_kp=math.nan)
        self.check(False, lock_zero_hz=64)
        self.check(False, lock_zero_hz=-1)
        self.check(False, lock_zero_hz=True)
        self.check(False, lock_zero_hz=1.0)

    def test_lock_fields_only_on_638(self):
        with self.assertRaises(tn.TuningError):
            tn.validate(tn.TecTuning(board=tn.BOARD_1550, lock_kp=0.6))

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
        record = tn.build_record(lock_638(setpoint_filter=None,
                                          conditional_integration=None))
        result = tn.parse_reply(reply(ident.OP_READ, protocol.TARGET_1550,
                                      ident.STATUS_OK, record))
        self.assertIn("board type 3", result.record.rejected)
        self.assertIn("board type 3", result.record.lock_rejected)

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
    def setUp(self):
        patcher = mock.patch("ultracoustics._internal.maintenance.time.sleep")
        self.sleep = patcher.start()
        self.addCleanup(patcher.stop)

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
        conn = FakeConnection([reply(ident.OP_READ, protocol.TARGET_638, ident.STATUS_EMPTY),
                               reply(ident.OP_WRITE, protocol.TARGET_638, ident.STATUS_OK,
                                     tn.build_record(tuning, seq=9))])
        result = DiagnosticsManager(conn).write_tec_tuning(protocol.TARGET_638, tuning)
        self.assertEqual(conn.sent[0], (protocol.CMD_IDENTITY_READ, 1, 638, b""))
        self.sleep.assert_called_once_with(tn.RMW_GAP_S)
        cmd, value, target, payload = conn.sent[1]
        self.assertEqual((cmd, value, target, len(payload)),
                         (protocol.CMD_IDENTITY_WRITE, 1, 638, 128))
        self.assertEqual(payload[:4], b"UCTN")
        self.assertTrue(result.ok, result.status_name)

    def test_readback_mismatch_becomes_verify_fail(self):
        conn = FakeConnection([reply(ident.OP_READ, protocol.TARGET_638, ident.STATUS_EMPTY),
                               reply(ident.OP_WRITE, protocol.TARGET_638, ident.STATUS_OK,
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
        sent = tn.parse_record(conn.sent[0][3]).tuning  # the 1550 has no lock fields: no read
        self.assertEqual(sent.settings(), {})
        self.assertIsNotNone(sent.tuned_at)

    def test_old_mainboard_raises(self):
        conn = FakeConnection([reply(ident.OP_READ, protocol.TARGET_638,
                                     ident.STATUS_BAD_RECORD, kind=ident.KIND_IDENTITY)])
        with self.assertRaises(tn.TuningError):
            DiagnosticsManager(conn).write_tec_tuning(protocol.TARGET_638, tuning_638())
        self.assertEqual([sent[0] for sent in conn.sent], [protocol.CMD_IDENTITY_READ])

    def test_638_tec_write_and_clear_keep_the_stored_lock_fields(self):
        stored = tn.build_record(lock_638(), seq=4)
        for write in (lambda diag: diag.write_tec_tuning(protocol.TARGET_638, tuning_638(kp=0.09)),
                      lambda diag: diag.clear_tec_tuning(protocol.TARGET_638, report_id="CLR")):
            conn = FakeConnection([reply(ident.OP_READ, protocol.TARGET_638, ident.STATUS_OK, stored)])
            conn.receive = lambda size, timeout_ms=500, conn=conn: (
                conn.replies.pop(0) if conn.replies else
                reply(ident.OP_WRITE, protocol.TARGET_638, ident.STATUS_OK,
                      tn.build_record(tn.parse_record(conn.sent[-1][3]).tuning, seq=5)))
            result = write(DiagnosticsManager(conn))
            self.assertTrue(result.ok, result.status_name)
            sent = tn.parse_record(conn.sent[1][3]).tuning
            self.assertEqual((sent.lock_kp, sent.lock_ki_per_s, sent.lock_zero_hz), (0.6, 1925.0, 20))
            self.assertEqual(conn.sent[1][3][6], 60)
        self.assertEqual(sent.settings(), {})  # the clear: no TEC field, lock fields kept
        self.assertEqual(sent.report_id, "CLR")

    def test_638_write_returns_a_failed_read_and_writes_nothing(self):
        conn = FakeConnection([reply(ident.OP_READ, protocol.TARGET_638, ident.STATUS_NO_RESPONSE)])
        result = DiagnosticsManager(conn).write_tec_tuning(protocol.TARGET_638, tuning_638())
        self.assertEqual((result.op, result.status_name), (ident.OP_READ, "NO_RESPONSE"))
        self.assertEqual(len(conn.sent), 1)

    def test_tec_write_refuses_lock_fields(self):
        conn = FakeConnection([])
        with self.assertRaisesRegex(ValueError, "keeps the stored lock PI fields"):
            DiagnosticsManager(conn).write_tec_tuning(protocol.TARGET_638, lock_638())
        self.assertEqual(conn.sent, [])


if __name__ == "__main__":
    unittest.main()
