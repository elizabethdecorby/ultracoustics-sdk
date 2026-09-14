import struct
import unittest
from unittest import mock

from ultracoustics._internal.comms import TransportUncertainError
from ultracoustics._internal.maintenance import Programmer


class ScriptedConnection:
    def __init__(self, fail_at):
        self.fail_at = fail_at
        self.calls = []

    def send_command(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if len(self.calls) == self.fail_at:
            raise TransportUncertainError("scripted uncertainty")


class ProgrammerUncertaintyTests(unittest.TestCase):
    @mock.patch("ultracoustics._internal.maintenance.time.sleep")
    def test_slave_page_uncertainty_stops_without_replay(self, _sleep):
        connection = ScriptedConnection(fail_at=2)  # BEGIN then first page
        with self.assertRaises(TransportUncertainError):
            Programmer(connection).flash_638(b"x" * 300)
        self.assertEqual(len(connection.calls), 2)

    @mock.patch("ultracoustics._internal.maintenance.time.sleep")
    def test_master_commit_uncertainty_is_not_swallowed_or_replayed(self, _sleep):
        image = b"UCBT" + struct.pack("<HHII", 1, 1, 16, 0)
        connection = ScriptedConnection(fail_at=4)  # IDLE, BEGIN, page, COMMIT
        with self.assertRaises(TransportUncertainError):
            Programmer(connection).flash_master(image)
        self.assertEqual(len(connection.calls), 4)


if __name__ == "__main__":
    unittest.main()
