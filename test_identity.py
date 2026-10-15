import struct
import unittest

from ultracoustics import identity as ident
from ultracoustics.identity import BoardIdentity, IdentityError


def laser(**overrides):
    values = dict(board=ident.BOARD_638, serial="2630054005", pcb_rev="R202510",
                  driver_pot_ohm=1234.5, lpf_pot_ohm=47000.0, pd_pot_ohm=10000.0,
                  programmed_at=1_790_000_000, report_id="DIAG_20261015_120000_ABCD1234")
    values.update(overrides)
    return BoardIdentity(**values)


def reply(status, record=b"\xFF" * ident.RECORD_SIZE, op=ident.OP_READ,
          target=ident.TARGET_638, slots_free=127):
    return (ident.REPLY_MAGIC + struct.pack("<BBHBBH", ident.REPLY_SCHEMA, op, target,
                                            status, slots_free, 0) + record)


class CrcTests(unittest.TestCase):
    def test_matches_mpeg2_check_value(self):
        self.assertEqual(ident.crc32_mpeg2(b"123456789"), 0x0376E6E7)


class RecordTests(unittest.TestCase):
    def test_laser_round_trip(self):
        source = laser()
        record = ident.build_record(source, seq=7)
        self.assertEqual(len(record), ident.RECORD_SIZE)
        stored = ident.parse_record(record)
        self.assertEqual(stored.seq, 7)
        self.assertEqual(stored.board, ident.BOARD_638)
        got = stored.identity
        self.assertEqual(got.serial, source.serial)
        self.assertEqual(got.pcb_rev, source.pcb_rev)
        self.assertAlmostEqual(got.driver_pot_ohm, 1234.5)
        self.assertAlmostEqual(got.lpf_pot_ohm, 47000.0)
        self.assertAlmostEqual(got.pd_pot_ohm, 10000.0)
        self.assertEqual(got.programmed_at, source.programmed_at)
        self.assertEqual(got.report_id, source.report_id)

    def test_mainboard_round_trip_and_unused_bytes_erased(self):
        source = BoardIdentity(board=ident.BOARD_MAIN, serial="2507180076",
                               pcb_rev="R202510", pd_pcb_rev="R202509")
        record = ident.build_record(source)
        length = struct.unpack_from("<H", record, 6)[0]
        self.assertTrue(all(b == 0xFF for b in record[ident.HEADER_SIZE + length:]))
        got = ident.parse_record(record).identity
        self.assertEqual(got.details(), {"serial": "2507180076", "pcb_rev": "R202510",
                                         "pd_pcb_rev": "R202509"})

    def test_erased_record_is_none(self):
        self.assertIsNone(ident.parse_record(b"\xFF" * ident.RECORD_SIZE))

    def test_every_field_at_max_length_fits(self):
        for board in (ident.BOARD_1550, ident.BOARD_638):
            ident.build_record(laser(board=board, serial="S" * 24, pcb_rev="P" * 12,
                                     report_id="R" * 36))
        ident.build_record(BoardIdentity(board=ident.BOARD_MAIN, serial="S" * 24,
                                         pcb_rev="P" * 12, pd_pcb_rev="Q" * 12,
                                         programmed_at=1, report_id="R" * 36))

    def test_oversize_payload_rejected(self):
        with self.assertRaises(IdentityError):
            ident.build_record(laser(unknown={0x7F: b"x" * 60}))

    def test_string_limit_and_board_specific_fields(self):
        with self.assertRaises(IdentityError):
            ident.build_record(laser(serial="x" * 25))
        with self.assertRaises(IdentityError):
            ident.build_record(laser(pd_pcb_rev="R1"))
        with self.assertRaises(IdentityError):
            ident.build_record(BoardIdentity(board=ident.BOARD_MAIN, lpf_pot_ohm=1.0))

    def test_unknown_tags_survive_round_trip(self):
        record = ident.build_record(laser(unknown={0x7E: b"\x01\x02"}))
        self.assertEqual(ident.parse_record(record).identity.unknown, {0x7E: b"\x01\x02"})

    def test_corrupt_crc_rejected(self):
        record = bytearray(ident.build_record(laser()))
        record[20] ^= 0x01
        with self.assertRaises(IdentityError):
            ident.parse_record(bytes(record))

    def test_firmware_restamp_validates(self):
        # Firmware rewrites board type and seq, then recomputes the CRC over header+payload.
        record = bytearray(ident.build_record(laser(board=ident.BOARD_1550)))
        struct.pack_into("<BBHI", record, 4, 1, ident.BOARD_1550,
                         struct.unpack_from("<H", record, 6)[0], 42)
        length = struct.unpack_from("<H", record, 6)[0]
        crc = ident.crc32_mpeg2(bytes(record[:12]) + bytes(record[16:16 + length]))
        struct.pack_into("<I", record, 12, crc)
        self.assertEqual(ident.parse_record(bytes(record)).seq, 42)


class ReplyTests(unittest.TestCase):
    def test_ok_reply(self):
        result = ident.parse_reply(reply(ident.STATUS_OK, ident.build_record(laser(), seq=3)))
        self.assertTrue(result.ok)
        self.assertEqual(result.target, ident.TARGET_638)
        self.assertEqual(result.slots_free, 127)
        self.assertEqual(result.record.identity.serial, "2630054005")

    def test_empty_and_no_response_have_no_record(self):
        for status in (ident.STATUS_EMPTY, ident.STATUS_NO_RESPONSE):
            result = ident.parse_reply(reply(status))
            self.assertFalse(result.ok)
            self.assertIsNone(result.record)
        self.assertEqual(ident.parse_reply(reply(ident.STATUS_NO_RESPONSE)).status_name,
                         "NO_RESPONSE")

    def test_verify_fail_with_garbage_record_does_not_raise(self):
        result = ident.parse_reply(reply(ident.STATUS_VERIFY_FAIL, b"\x00" * ident.RECORD_SIZE,
                                         op=ident.OP_WRITE))
        self.assertEqual(result.status_name, "VERIFY_FAIL")
        self.assertIsNone(result.record)

    def test_wrong_length_or_magic_rejected(self):
        with self.assertRaises(IdentityError):
            ident.parse_reply(b"\x00" * 40)
        with self.assertRaises(IdentityError):
            ident.parse_reply(b"XXXX" + reply(ident.STATUS_OK)[4:])


if __name__ == "__main__":
    unittest.main()
