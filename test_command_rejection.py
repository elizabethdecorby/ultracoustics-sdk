import unittest

import usb.core

from ultracoustics._internal.comms import (
    BULK_OUT_EP,
    CommandRejectedError,
    USBBulkConnection,
)


class FakeDevice:
    def __init__(self, error):
        self.error = error
        self.write_calls = 0
        self.clear_calls = []

    def write(self, endpoint, payload, timeout):
        self.write_calls += 1
        raise self.error

    def clear_halt(self, endpoint):
        self.clear_calls.append(endpoint)


class CommandRejectionTests(unittest.TestCase):
    def connection(self, device):
        connection = USBBulkConnection.__new__(USBBulkConnection)
        connection.connected = True
        connection.dev = device
        return connection

    def test_pipe_rejection_clears_halt_reports_and_never_retries(self):
        error = usb.core.USBError("pipe", errno=32)
        device = FakeDevice(error)
        with self.assertRaisesRegex(CommandRejectedError, "not retried"):
            self.connection(device).send(b"unknown")
        self.assertEqual(device.write_calls, 1)
        self.assertEqual(device.clear_calls, [BULK_OUT_EP])

    def test_non_pipe_error_does_not_clear_halt(self):
        device = FakeDevice(usb.core.USBError("timeout", errno=110))
        with self.assertRaisesRegex(RuntimeError, "Failed to send"):
            self.connection(device).send(b"x")
        self.assertEqual(device.write_calls, 1)
        self.assertEqual(device.clear_calls, [])


if __name__ == "__main__":
    unittest.main()
