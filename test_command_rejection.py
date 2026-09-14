import unittest

import usb.core

from ultracoustics._internal.comms import (
    BULK_OUT_EP,
    CommandRejectedError,
    USBBulkConnection,
)


class FakeDevice:
    def __init__(self, errors):
        self.errors = list(errors)
        self.write_calls = 0
        self.clear_calls = []

    def write(self, endpoint, payload, timeout):
        self.write_calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return len(payload)

    def clear_halt(self, endpoint):
        self.clear_calls.append(endpoint)


class CommandRejectionTests(unittest.TestCase):
    def connection(self, device):
        connection = USBBulkConnection.__new__(USBBulkConnection)
        connection.connected = True
        connection.dev = device
        return connection

    def test_deferred_pipe_rejection_clears_halt_and_retries_only_unsent_transfer(self):
        error = usb.core.USBError("pipe", errno=32)
        device = FakeDevice([error])
        self.connection(device).send(b"current-valid-command")
        self.assertEqual(device.write_calls, 2)
        self.assertEqual(device.clear_calls, [BULK_OUT_EP])

    def test_persistent_pipe_is_bounded_to_one_retry(self):
        error1 = usb.core.USBError("pipe", errno=32)
        error2 = usb.core.USBError("pipe", errno=32)
        device = FakeDevice([error1, error2])
        with self.assertRaisesRegex(CommandRejectedError, "remained halted"):
            self.connection(device).send(b"x")
        self.assertEqual(device.write_calls, 2)

    def test_non_pipe_error_does_not_clear_halt(self):
        device = FakeDevice([usb.core.USBError("timeout", errno=110)])
        with self.assertRaisesRegex(RuntimeError, "Failed to send"):
            self.connection(device).send(b"x")
        self.assertEqual(device.write_calls, 1)
        self.assertEqual(device.clear_calls, [])


if __name__ == "__main__":
    unittest.main()
