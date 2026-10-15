import struct
import unittest
from unittest import mock

from ultracoustics import identity as ident
from ultracoustics._internal import protocol
from ultracoustics._internal.maintenance import DiagnosticsManager, Programmer


def reply(op, target, status, record=b"\xFF" * ident.RECORD_SIZE, slots_free=63):
    return (ident.REPLY_MAGIC
            + struct.pack("<BBHBBH", ident.REPLY_SCHEMA, op, target, status, slots_free, 0)
            + record)


def stored(identity, seq=1):
    """What a board returns: the sent record re-stamped with seq and CRC."""
    return ident.build_record(identity, seq=seq)


class FakeConnection:
    def __init__(self, replies):
        self.replies = list(replies)
        self.sent = []

    def send_command(self, cmd, wValue=0, wIndex=0, extra=b"", timeout_ms=5000):
        self.sent.append((cmd, wValue, wIndex, bytes(extra)))

    def receive(self, size, timeout_ms=500):
        return self.replies.pop(0) if self.replies else None


def laser_identity(**kw):
    values = dict(board=ident.BOARD_1550, serial="2507180076", pcb_rev="R202510",
                  driver_pot_ohm=1234.56, lpf_pot_ohm=4700.1, pd_pot_ohm=10000.0,
                  programmed_at=1_790_000_000, report_id="DIAG_X")
    values.update(kw)
    return ident.BoardIdentity(**values)


class ReadIdentityTests(unittest.TestCase):
    def test_read_sends_target_and_skips_unrelated_chunks(self):
        main = ident.BoardIdentity(board=ident.BOARD_MAIN, serial="A1")
        conn = FakeConnection([
            b"\x00" * 16392,                                     # stream packet
            reply(ident.OP_READ, protocol.TARGET_638, ident.STATUS_EMPTY),  # other target
            reply(ident.OP_READ, protocol.TARGET_MAIN, ident.STATUS_OK, stored(main, 4)),
        ])
        result = DiagnosticsManager(conn).read_identity(protocol.TARGET_MAIN)
        self.assertEqual(conn.sent, [(protocol.CMD_IDENTITY_READ, 0, 0, b"")])
        self.assertTrue(result.ok)
        self.assertEqual(result.record.seq, 4)
        self.assertEqual(result.record.identity.serial, "A1")

    def test_no_response_status_is_returned_not_raised(self):
        conn = FakeConnection([reply(ident.OP_READ, protocol.TARGET_1550,
                                     ident.STATUS_NO_RESPONSE)])
        result = DiagnosticsManager(conn).read_identity(protocol.TARGET_1550)
        self.assertEqual(result.status_name, "NO_RESPONSE")

    @mock.patch("ultracoustics._internal.maintenance.time.time",
                side_effect=[0.0] + [0.0, 0.0] * 3 + [100.0] * 10)
    def test_missing_reply_times_out(self, _time):
        with self.assertRaises(TimeoutError):
            DiagnosticsManager(FakeConnection([])).read_identity(protocol.TARGET_638)

    def test_unknown_target_rejected(self):
        with self.assertRaises(ValueError):
            DiagnosticsManager(FakeConnection([])).read_identity(1234)


class WriteIdentityTests(unittest.TestCase):
    def test_write_sends_record_and_accepts_float32_readback(self):
        identity = laser_identity()
        conn = FakeConnection([reply(ident.OP_WRITE, protocol.TARGET_1550,
                                     ident.STATUS_OK, stored(identity, 9))])
        result = DiagnosticsManager(conn).write_identity(protocol.TARGET_1550, identity)
        cmd, _value, target, payload = conn.sent[0]
        self.assertEqual((cmd, target, len(payload)),
                         (protocol.CMD_IDENTITY_WRITE, protocol.TARGET_1550, 128))
        self.assertTrue(result.ok, result.status_name)

    def test_readback_mismatch_becomes_verify_fail(self):
        identity = laser_identity()
        other = laser_identity(serial="SOMEONE_ELSE")
        conn = FakeConnection([reply(ident.OP_WRITE, protocol.TARGET_1550,
                                     ident.STATUS_OK, stored(other, 9))])
        result = DiagnosticsManager(conn).write_identity(protocol.TARGET_1550, identity)
        self.assertEqual(result.status_name, "VERIFY_FAIL")

    def test_board_target_mismatch_rejected_before_sending(self):
        conn = FakeConnection([])
        with self.assertRaises(ValueError):
            DiagnosticsManager(conn).write_identity(protocol.TARGET_638, laser_identity())
        self.assertEqual(conn.sent, [])

    def test_programmed_at_filled_when_missing(self):
        identity = laser_identity(programmed_at=None)
        conn = FakeConnection([reply(ident.OP_WRITE, protocol.TARGET_1550,
                                     ident.STATUS_WRITE_FAIL)])
        DiagnosticsManager(conn).write_identity(protocol.TARGET_1550, identity)
        self.assertIsNotNone(identity.programmed_at)


class SystemInfoTests(unittest.TestCase):
    def test_failures_are_collected_per_item(self):
        diag = DiagnosticsManager(FakeConnection([]))
        ok = ident.IdentityResult(op=0, target=0, status=0, slots_free=1, record=None, raw=b"")
        with mock.patch.object(diag, "query_firmware_info", return_value={"version": "abc"}), \
             mock.patch.object(diag, "query_slave_version", side_effect=TimeoutError("x")), \
             mock.patch.object(diag, "read_identity", return_value=ok):
            info = diag.read_system_info()
        self.assertEqual(info["master_fw"], {"version": "abc"})
        self.assertEqual(set(info["errors"]), {"fw_1550", "fw_638"})
        self.assertIs(info["identity"]["638"], ok)


class MasterSizeGuardTests(unittest.TestCase):
    def test_master_image_limit_leaves_sector_7_free(self):
        self.assertEqual(Programmer.MASTER_MAX_FW, 0x0800E000 - 0x08000000)


if __name__ == "__main__":
    unittest.main()
