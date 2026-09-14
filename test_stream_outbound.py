import unittest

import usb1

from ultracoustics._internal.stream_proc import BULK_OUT_EP, _write_outbound_once


class FakeHandle:
    def __init__(self, outcomes, clear_error=None):
        self.outcomes = list(outcomes)
        self.clear_error = clear_error
        self.write_calls = 0
        self.clear_calls = 0

    def bulkWrite(self, endpoint, payload, timeout):
        self.write_calls += 1
        self.asserted = (endpoint, payload, timeout)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def clearHalt(self, endpoint):
        self.clear_calls += 1
        self.clear_endpoint = endpoint
        if self.clear_error:
            raise self.clear_error


class StreamOutboundTests(unittest.TestCase):
    def test_delivered(self):
        handle = FakeHandle([3])
        result = _write_outbound_once(handle, b"abc", 100)
        self.assertEqual(result["transport_status"], "delivered")
        self.assertEqual(result["transferred_bytes"], 3)

    def test_pipe_is_uncertain_cleared_and_never_replayed(self):
        handle = FakeHandle([usb1.USBErrorPipe(), 3])
        result = _write_outbound_once(handle, b"abc", 100)
        self.assertEqual(result["transport_status"], "uncertain")
        self.assertTrue(result["halt_cleared"])
        self.assertEqual(handle.write_calls, 1)
        self.assertEqual(handle.clear_calls, 1)
        self.assertEqual(handle.clear_endpoint, BULK_OUT_EP)

    def test_pipe_with_halt_clear_failure_is_uncertain(self):
        handle = FakeHandle([usb1.USBErrorPipe()], RuntimeError("clear"))
        result = _write_outbound_once(handle, b"abc", 100)
        self.assertEqual(result["transport_status"], "uncertain")
        self.assertFalse(result["halt_cleared"])
        self.assertEqual(handle.write_calls, 1)

    def test_timeout_is_uncertain_and_not_replayed(self):
        handle = FakeHandle([usb1.USBErrorTimeout()])
        result = _write_outbound_once(handle, b"abc", 100)
        self.assertEqual(result["transport_status"], "uncertain")
        self.assertEqual(handle.write_calls, 1)
        self.assertEqual(handle.clear_calls, 0)

    def test_short_transfer_is_uncertain_and_not_replayed(self):
        handle = FakeHandle([1])
        result = _write_outbound_once(handle, b"abc", 100)
        self.assertEqual(result["transport_status"], "uncertain")
        self.assertEqual(result["transferred_bytes"], 1)
        self.assertEqual(handle.write_calls, 1)


if __name__ == "__main__":
    unittest.main()
