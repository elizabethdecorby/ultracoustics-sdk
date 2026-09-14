import queue
import unittest
from unittest import mock

from ultracoustics._internal.comms import CommandRejectedError, USBStream


class FakeCommandQueue:
    def __init__(self, ack_queue, *, status="delivered"):
        self.ack_queue = ack_queue
        self.status = status
        self.items = []

    def put(self, item):
        self.items.append(item)
        request_id, _payload = item
        result = {"transport_status": self.status,
                  "transferred_bytes": 3 if self.status == "delivered" else None,
                  "halt_cleared": self.status == "uncertain",
                  "error": None if self.status == "delivered" else self.status}
        self.ack_queue.put((request_id, result, 120))


class ConfirmedCommandTests(unittest.TestCase):
    def stream(self, *, status="delivered"):
        stream = USBStream.__new__(USBStream)
        stream._running = True
        stream._proc = mock.Mock()
        stream._proc.is_alive.return_value = True
        stream._command_ack_queue = queue.Queue()
        stream._cmd_queue = FakeCommandQueue(stream._command_ack_queue, status=status)
        stream._next_command_request_id = 1
        return stream

    @mock.patch("ultracoustics._internal.comms.time.monotonic_ns", return_value=100)
    def test_confirmed_command_is_enqueued_once(self, _clock):
        stream = self.stream()
        result = stream.send_command_confirmed(b"abc")
        self.assertEqual(len(stream._cmd_queue.items), 1)
        self.assertEqual(result["host_delivery_bound_ns"], 20)
        self.assertEqual(result["transport_status"], "delivered")

    @mock.patch("ultracoustics._internal.comms.time.monotonic_ns", return_value=100)
    def test_failed_confirmed_delivery_is_reported(self, _clock):
        stream = self.stream(status="uncertain")
        with self.assertRaisesRegex(CommandRejectedError, "transport uncertain"):
            stream.send_command_confirmed(b"abc")
        self.assertEqual(len(stream._cmd_queue.items), 1)

    @mock.patch("ultracoustics._internal.comms.time.monotonic_ns", return_value=100)
    def test_failed_transport_is_distinct_from_uncertain(self, _clock):
        stream = self.stream(status="failed")
        with self.assertRaisesRegex(RuntimeError, "transport failed"):
            stream.send_command_confirmed(b"abc")


if __name__ == "__main__":
    unittest.main()
