import unittest

import usb.core

from ultracoustics._internal.comms import (
    BULK_OUT_EP,
    CommandRejectedError,
    TransportUncertainError,
    USBBulkConnection,
)


class FakeDevice:
    def __init__(self, outcomes, clear_error=None):
        self.outcomes = list(outcomes)
        self.clear_error = clear_error
        self.write_calls = 0
        self.clear_calls = []

    def write(self, endpoint, payload, timeout):
        self.write_calls += 1
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        return len(payload)

    def clear_halt(self, endpoint):
        self.clear_calls.append(endpoint)
        if self.clear_error:
            raise self.clear_error


class CommandRejectionTests(unittest.TestCase):
    def connection(self, device):
        connection = USBBulkConnection.__new__(USBBulkConnection)
        connection.connected = True
        connection.dev = device
        return connection

    def test_pipe_clears_halt_but_does_not_replay_uncertain_payload(self):
        error = usb.core.USBError("pipe", errno=32)
        device = FakeDevice([error])
        with self.assertRaises(TransportUncertainError) as caught:
            self.connection(device).send(b"possibly-partial-command")
        self.assertTrue(caught.exception.halt_cleared)
        self.assertEqual(device.write_calls, 1)
        self.assertEqual(device.clear_calls, [BULK_OUT_EP])

    def test_next_valid_command_is_a_new_explicit_send(self):
        device = FakeDevice([usb.core.USBError("pipe", errno=32), 4])
        connection = self.connection(device)
        with self.assertRaises(CommandRejectedError):
            connection.send(b"bad?")
        result = connection.send(b"next")
        self.assertEqual(device.write_calls, 2)
        self.assertEqual(result["transport_status"], "delivered")

    def test_short_success_is_uncertain_and_not_replayed(self):
        device = FakeDevice([2])
        with self.assertRaisesRegex(TransportUncertainError, "short BULK OUT"):
            self.connection(device).send(b"four")
        self.assertEqual(device.write_calls, 1)

    def test_halt_clear_failure_preserves_uncertainty(self):
        device = FakeDevice([usb.core.USBError("pipe", errno=32)],
                            clear_error=RuntimeError("clear failed"))
        with self.assertRaises(TransportUncertainError) as caught:
            self.connection(device).send(b"x")
        self.assertFalse(caught.exception.halt_cleared)
        self.assertEqual(device.write_calls, 1)

    def test_timeout_is_uncertain_and_does_not_clear_halt(self):
        device = FakeDevice([usb.core.USBError("timeout", errno=110)])
        with self.assertRaisesRegex(TransportUncertainError, "state is uncertain"):
            self.connection(device).send(b"x")
        self.assertEqual(device.write_calls, 1)
        self.assertEqual(device.clear_calls, [])

    def test_zero_length_packet_is_a_real_success_not_failed_transfer_evidence(self):
        device = FakeDevice([0])
        result = self.connection(device).send(b"")
        self.assertEqual(result["transferred_bytes"], 0)
        self.assertEqual(result["transport_status"], "delivered")


if __name__ == "__main__":
    unittest.main()
