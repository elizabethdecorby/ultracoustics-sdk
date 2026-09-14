import queue
import unittest
from unittest import mock

from ultracoustics._internal.comms import CommandRejectedError, USBStream


class FakeCommandQueue:
    def __init__(self, ack_queue, *, ok=True):
        self.ack_queue = ack_queue
        self.ok = ok
        self.items = []

    def put(self, item):
        self.items.append(item)
        request_id, _payload = item
        self.ack_queue.put((request_id, self.ok, None if self.ok else "pipe", 120))


class ConfirmedCommandTests(unittest.TestCase):
    def stream(self, *, ok=True):
        stream = USBStream.__new__(USBStream)
        stream._running = True
        stream._proc = mock.Mock()
        stream._proc.is_alive.return_value = True
        stream._command_ack_queue = queue.Queue()
        stream._cmd_queue = FakeCommandQueue(stream._command_ack_queue, ok=ok)
        stream._next_command_request_id = 1
        return stream

    @mock.patch("ultracoustics._internal.comms.time.monotonic_ns", return_value=100)
    def test_confirmed_command_is_enqueued_once(self, _clock):
        stream = self.stream()
        result = stream.send_command_confirmed(b"abc")
        self.assertEqual(len(stream._cmd_queue.items), 1)
        self.assertEqual(result["host_delivery_bound_ns"], 20)

    @mock.patch("ultracoustics._internal.comms.time.monotonic_ns", return_value=100)
    def test_rejected_confirmed_command_is_not_retried(self, _clock):
        stream = self.stream(ok=False)
        with self.assertRaisesRegex(CommandRejectedError, "not retried"):
            stream.send_command_confirmed(b"abc")
        self.assertEqual(len(stream._cmd_queue.items), 1)


if __name__ == "__main__":
    unittest.main()
