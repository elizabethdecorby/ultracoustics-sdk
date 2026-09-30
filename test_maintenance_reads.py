import struct
import unittest
from unittest import mock

from ultracoustics._internal.maintenance import DiagnosticsManager
from ultracoustics._internal.protocol import (
    CMD_BOOT_VERSION, CMD_BOOT_VERSION_638, CMD_FWINFO, TARGET_638,
)


def firmware_info(version, timestamp, hardware_rev=1):
    return version.encode("ascii").ljust(32, b"\0") + struct.pack(
        "<IHH", timestamp, hardware_rev, 0
    )


class ScriptedConnection:
    def __init__(self, responses):
        self.responses = list(responses)
        self.commands = []
        self.read_lengths = []

    def send_command(self, command):
        self.commands.append(command)

    def receive(self, length, timeout_ms=1000):
        self.read_lengths.append(length)
        if not self.responses:
            return None
        response = self.responses.pop(0)
        if len(response) > length:
            self.responses.insert(0, response[length:])
            return response[:length]
        return response


class MaintenanceReadTests(unittest.TestCase):
    @mock.patch("ultracoustics._internal.maintenance.time.sleep")
    def test_master_info_after_format2_record(self, sleep):
        expected = firmware_info("ecaff79", 1790723127)
        connection = ScriptedConnection([b"S" * 16532, expected])

        result = DiagnosticsManager(connection).query_firmware_info(timeout_ms=100)

        self.assertEqual(connection.commands, [CMD_FWINFO])
        self.assertEqual(result["version"], "ecaff79")
        self.assertEqual(int(result["build_date"].timestamp()), 1790723127)
        self.assertTrue(all(length >= 16532 for length in connection.read_lengths))
        self.assertEqual(len(connection.read_lengths), 2)
        sleep.assert_not_called()

    @mock.patch("ultracoustics._internal.maintenance.time.sleep")
    def test_slave_info_after_format2_record(self, sleep):
        expected = firmware_info("0753a00", 1773763750) + firmware_info(
            "06d8236", 1790722465
        )
        for target, command in ((None, CMD_BOOT_VERSION),
                                (TARGET_638, CMD_BOOT_VERSION_638)):
            with self.subTest(target=target):
                connection = ScriptedConnection([b"S" * 16532, expected])
                manager = DiagnosticsManager(connection)
                result = (manager.query_slave_version(timeout_ms=100)
                          if target is None else
                          manager.query_slave_version(target, timeout_ms=100))
                self.assertEqual(connection.commands, [command])
                self.assertEqual(result["boot"]["version"], "0753a00")
                self.assertEqual(result["app"]["version"], "06d8236")
                self.assertEqual(len(connection.read_lengths), 2)
                self.assertTrue(all(length >= 16532
                                    for length in connection.read_lengths))
        sleep.assert_not_called()

    def test_missing_reply_stays_bounded(self):
        connection = ScriptedConnection([])
        with self.assertRaisesRegex(TimeoutError, "No firmware info response"):
            DiagnosticsManager(connection).query_firmware_info(timeout_ms=5)
        self.assertEqual(connection.commands, [CMD_FWINFO])
        self.assertTrue(connection.read_lengths)
        self.assertTrue(all(length >= 16532 for length in connection.read_lengths))


if __name__ == "__main__":
    unittest.main()
