import unittest
from unittest import mock

from ultracoustics.controller import Controller
from ultracoustics._internal.protocol import CMD_IDLE


class StopConfirmedTests(unittest.TestCase):
    def controller(self):
        controller = Controller.__new__(Controller)
        controller._running = True
        controller._send_confirmed = mock.Mock()
        return controller

    def test_delivered_idle_clears_running_and_returns_provenance(self):
        controller = self.controller()
        result = {
            "transport_status": "delivered",
            "transferred_bytes": 12,
            "host_delivery_bound_ns": 42,
        }
        controller._send_confirmed.return_value = result

        self.assertIs(controller.stop_confirmed(timeout_s=0.25), result)
        self.assertFalse(controller._running)
        controller._send_confirmed.assert_called_once_with(
            CMD_IDLE, timeout_s=0.25
        )

    def test_failed_delivery_preserves_running(self):
        controller = self.controller()
        controller._send_confirmed.side_effect = RuntimeError(
            "confirmed command transport failed"
        )

        with self.assertRaisesRegex(RuntimeError, "transport failed"):
            controller.stop_confirmed()

        self.assertTrue(controller._running)

    def test_non_delivered_result_preserves_running(self):
        controller = self.controller()
        controller._send_confirmed.return_value = {
            "transport_status": "uncertain"
        }

        with self.assertRaisesRegex(RuntimeError, "not confirmed delivered"):
            controller.stop_confirmed()

        self.assertTrue(controller._running)


if __name__ == "__main__":
    unittest.main()
