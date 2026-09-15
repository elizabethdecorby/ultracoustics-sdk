import struct
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from ultracoustics.controller import Controller
from ultracoustics._internal.control import (
    CHANNEL_LASER_DAC, ControlProtocolError, MANUAL_SET,
    ManualTransportError, crc8_poly31, pack_manual_request,
    parse_manual_response, parse_runtime_metrics,
)
from examples.manual_laser_sweep import write_results
from ultracoustics.manual_sweep import (ManualSweepError, pd_is_fresh,
                                        run_manual_sweep, take_laser_zero)


def manual_response(transport=0, target=638, txn=7, opcode=MANUAL_SET,
                    channel=CHANNEL_LASER_DAC, applied=33000):
    raw = bytearray(32)
    struct.pack_into("<4sHHHBBBBBB", raw, 0, b"UTMC", 1, 32, target,
                     transport, 0, txn, opcode, channel, 0)
    if transport == 0:
        ack = bytearray(b"\xBB\xBB\xA5" + bytes((txn, 0, 3)))
        ack += applied.to_bytes(3, "big", signed=True)
        ack.append(crc8_poly31(ack))
        raw[16:26] = ack
    return bytes(raw)


class ControlParserTests(unittest.TestCase):
    def test_manual_request_and_response_are_exact(self):
        request = pack_manual_request(MANUAL_SET, CHANNEL_LASER_DAC, 33000, 7)
        self.assertEqual(len(request), 10)
        self.assertEqual(request[:4], b"\xAA\x43\x02\0")
        self.assertEqual(struct.unpack_from(">i", request, 4)[0], 33000)
        self.assertEqual(request[-1], crc8_poly31(request[:-1]))
        reply = parse_manual_response(manual_response())
        self.assertEqual((reply.target, reply.transaction, reply.applied_value),
                         (638, 7, 33000))

    def test_transport_failure_requires_zero_ack(self):
        reply = parse_manual_response(manual_response(1))
        with self.assertRaises(ManualTransportError):
            reply.require_transport()
        damaged = bytearray(manual_response(1))
        damaged[16] = 1
        with self.assertRaises(ControlProtocolError):
            parse_manual_response(bytes(damaged))

    def test_reserved_identity_and_ack_crc_are_strict(self):
        for index in (11, 15, 26, 25):
            damaged = bytearray(manual_response())
            damaged[index] ^= 1
            with self.assertRaises(ControlProtocolError):
                parse_manual_response(bytes(damaged))

    def test_runtime_percent_requires_validity_bit(self):
        raw = bytearray(72)
        raw[:8] = struct.pack("<4sHH", b"URTM", 1, 72)
        struct.pack_into("<7IQ4IHHBBBBI", raw, 8,
                         0, 90, 10, 80, 10, 70, 30, 123,
                         4, 5, 0, 0, 6000, 4000, 0, 5, 5, 0, 1)
        metrics = parse_runtime_metrics(bytes(raw))
        self.assertIsNone(metrics.active_percent)
        raw[8] |= 1 << 3
        metrics = parse_runtime_metrics(bytes(raw))
        self.assertEqual(metrics.active_percent, 60.0)


class FakeStream:
    running = True
    _selected_stream_format = 0

    def __init__(self, reply):
        self.reply = reply
        self.request = None

    def _request_stream_control(self, payload, kind, timeout_s=1.0):
        self.request = (payload, kind, timeout_s)
        return {"firmware_response": self.reply}

    def get_telemetry(self):
        return None


class ControllerControlTests(unittest.TestCase):
    def test_runtime_metrics_waits_for_post_response_adc_packet(self):
        reply = SimpleNamespace(current_state=0)

        class RuntimeStream:
            running = True
            def __init__(self):
                self.counts = iter((12, 12, 13))
                self.request_complete = False
            def _request_stream_control(self, payload, kind, timeout_s=1.0):
                self.request_complete = True
                return {"firmware_response": reply}
            def get_stream_stats(self):
                self.assert_after_response()
                return {"packets": next(self.counts)}
            def assert_after_response(self):
                if not self.request_complete:
                    raise AssertionError("packet baseline read before response")

        ctrl = Controller()
        ctrl._stream = RuntimeStream()
        ctrl._running = False
        with mock.patch("ultracoustics.controller.time.sleep") as sleep:
            self.assertIs(ctrl.runtime_metrics(timeout_s=.1), reply)
        self.assertEqual(sleep.call_count, 1)

    def test_manual_command_checks_response_identity_and_cap(self):
        ctrl = Controller()
        reply = parse_manual_response(manual_response(txn=1))
        ctrl._stream = FakeStream(reply)
        result = ctrl.manual_command(638, MANUAL_SET, CHANNEL_LASER_DAC, 33000)
        self.assertEqual(result.applied_value, 33000)
        self.assertEqual(ctrl._stream.request[1], "manual")
        with self.assertRaises(ValueError):
            ctrl.manual_command(638, MANUAL_SET, CHANNEL_LASER_DAC, 33001)

    def test_begin_manual_legacy_settle_covers_bootloader_and_poll_gate(self):
        events = []
        ctrl = Controller()
        ctrl._stream = FakeStream(None)
        ctrl._running = False
        metrics = SimpleNamespace(current_state=0)
        ctrl.stop_confirmed = mock.Mock()
        ctrl.runtime_metrics = mock.Mock(
            side_effect=lambda timeout_s: events.append("runtime") or metrics)
        ctrl._send_confirmed = mock.Mock(side_effect=lambda *args, **kwargs: (
            events.append("command") or {"bulk_write_completed_monotonic_ns": 1}))
        with mock.patch("ultracoustics.controller.time.sleep",
                        side_effect=lambda seconds: events.append(("sleep", seconds))) as sleep:
            ctrl.begin_manual(1550)
        self.assertEqual(sleep.call_args_list, [mock.call(0.2), mock.call(1.25)])
        self.assertEqual(events[:3], ["runtime", ("sleep", 0.2), "command"])

    def test_begin_manual_format1_still_observes_minimum_settle(self):
        ctrl = Controller()
        stream = FakeStream(None)
        stream._selected_stream_format = 1
        ctrl._stream = stream
        ctrl._running = False
        baseline = SimpleNamespace(
            board_638=SimpleNamespace(temperature_sample_tick_ms=10))
        fresh_board = SimpleNamespace(
            temperature_sample_tick_ms=11, temperature_age_ms=20,
            temperature_valid=True, temperature_stale=False,
            temperature_fault=0)
        fresh = SimpleNamespace(
            board_638=fresh_board, link_flags_638=0,
            received_monotonic_ns=2_000_000_000, host_age_s=0.01)
        ctrl.stop_confirmed = mock.Mock()
        ctrl.runtime_metrics = mock.Mock(
            return_value=SimpleNamespace(current_state=0))
        ctrl._send_confirmed = mock.Mock(return_value={
            "bulk_write_completed_monotonic_ns": 1_000_000_000})
        with mock.patch.object(type(ctrl), "telemetry",
                               new_callable=mock.PropertyMock,
                               side_effect=[baseline, fresh]), \
             mock.patch("ultracoustics.controller.time.sleep") as sleep, \
             mock.patch("ultracoustics.controller.time.monotonic_ns",
                        return_value=2_000_000_000):
            ctrl.begin_manual(638)
        self.assertEqual(sleep.call_args_list[:2], [mock.call(0.2), mock.call(1.25)])

    def test_pd_freshness_includes_host_age_and_faults(self):
        board = SimpleNamespace(pd_sample_tick_ms=2, pd_valid=True,
                                pd_stale=False, pd_backend_disabled=False,
                                pd_fault=0, pd_age_ms=150,
                                temperature_valid=True,
                                temperature_stale=False, temperature_fault=0)
        snapshot = SimpleNamespace(host_age_s=0.1, host_stale=False)
        self.assertTrue(pd_is_fresh(snapshot, board, 0, 1, 300))
        board.pd_fault = 1
        self.assertFalse(pd_is_fresh(snapshot, board, 0, 1, 300))

    def test_partial_results_record_primary_and_cleanup_errors(self):
        row = {field: 0 for field in (
            "target", "dac_requested", "dac_applied", "pd_raw_counts",
            "pd_age_ms", "pd_flags", "pd_fault", "link_flags",
            "temperature_target_c", "temperature_measured_c")}
        with TemporaryDirectory() as directory:
            csv_path, json_path = write_results(
                __import__("pathlib").Path(directory), 638, [row],
                RuntimeError("primary"), RuntimeError("cleanup"))
            self.assertIn("pd_raw_counts", csv_path.read_text())
            text = json_path.read_text()
            self.assertIn("primary", text)
            self.assertIn("cleanup", text)

    def test_first_laser_ownership_request_is_zero(self):
        calls = []

        class FakeController:
            def manual_command(self, target, opcode, channel, value=0):
                calls.append((target, opcode, channel, value))
                return SimpleNamespace(status=0)

        take_laser_zero(FakeController(), 1550)
        self.assertEqual(calls, [(1550, 0x42, CHANNEL_LASER_DAC, 0)])

    def test_public_sweep_emits_each_accepted_point(self):
        board = SimpleNamespace(pd_sample_tick_ms=1, pd_raw_average=100,
                                pd_age_ms=10, pd_flags=1, pd_fault=0,
                                pd_full_scale=4095, temperature_mdegc=25000)

        class SweepController:
            connected = True
            streaming = True

            def stop_confirmed(self): pass
            def enable_telemetry(self): pass
            def begin_manual(self, target): pass
            def temperature_target_c(self, target): return 25.0
            def finish_manual(self, target): self.finished = True
            def manual_command(self, target, opcode, channel, value=0):
                return SimpleNamespace(status=0, applied_value=value)

        ctrl = SweepController()
        emitted = []
        with mock.patch("ultracoustics.manual_sweep.wait_telemetry_ready"), \
             mock.patch("ultracoustics.manual_sweep.take_laser_zero"), \
             mock.patch("ultracoustics.manual_sweep.wait_thermal_locked"), \
             mock.patch("ultracoustics.manual_sweep.wait_fresh_pd",
                        return_value=(board, 0)):
            rows = run_manual_sweep(ctrl, 638, points=2,
                                    on_point=emitted.append)
        self.assertEqual(rows, emitted)
        self.assertEqual([row["dac_requested"] for row in rows], [0, 33000])
        self.assertEqual(rows[0]["pd_full_scale_counts"], 4095)
        self.assertFalse(rows[0]["saturated"])
        self.assertEqual(rows[0]["sample_tick_ms"], 1)
        self.assertRegex(rows[0]["captured_at_utc"], r"\+00:00$")
        self.assertTrue(ctrl.finished)

    def test_public_sweep_reports_board_specific_saturation_before_raising(self):
        board = SimpleNamespace(pd_sample_tick_ms=7, pd_raw_average=2042,
                                pd_age_ms=10, pd_flags=1, pd_fault=0,
                                pd_full_scale=2047, temperature_mdegc=25000)

        class SweepController:
            connected = True
            streaming = True
            def stop_confirmed(self): pass
            def enable_telemetry(self): pass
            def begin_manual(self, target): pass
            def temperature_target_c(self, target): return 25.0
            def finish_manual(self, target): pass
            def manual_command(self, target, opcode, channel, value=0):
                return SimpleNamespace(status=0, applied_value=value)

        emitted = []
        with mock.patch("ultracoustics.manual_sweep.wait_telemetry_ready"), \
             mock.patch("ultracoustics.manual_sweep.take_laser_zero"), \
             mock.patch("ultracoustics.manual_sweep.wait_thermal_locked"), \
             mock.patch("ultracoustics.manual_sweep.wait_fresh_pd",
                        return_value=(board, 0)):
            with self.assertRaises(ManualSweepError) as caught:
                run_manual_sweep(SweepController(), 638, points=2,
                                 on_point=emitted.append)
        self.assertTrue(caught.exception.rows[0]["saturated"])
        self.assertEqual(caught.exception.rows[0]["pd_full_scale_counts"], 2047)
        self.assertEqual(emitted, caught.exception.rows)

    def test_public_sweep_cancel_still_finishes_manual(self):
        class SweepController:
            connected = True
            streaming = True
            finished = False
            def stop_confirmed(self): pass
            def enable_telemetry(self): pass
            def begin_manual(self, target): pass
            def finish_manual(self, target): self.finished = True

        ctrl = SweepController()
        with self.assertRaises(ManualSweepError) as caught:
            run_manual_sweep(ctrl, 638, cancel=lambda: True)
        self.assertIn("canceled", str(caught.exception))
        self.assertTrue(ctrl.finished)


if __name__ == "__main__":
    unittest.main()
