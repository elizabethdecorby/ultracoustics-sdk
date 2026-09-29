"""No-hardware refusal and cancellation checks for PI characterization."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest
from unittest.mock import patch

from ultracoustics import run_pi_characterization
from ultracoustics import pi_characterization
from ultracoustics._internal.control import MANUAL_GET, MANUAL_SET
from ultracoustics.system_control import SlaveCommandRejected


class FakeController:
    device_serial = "board-test"

    def __init__(self, *, stale=False, format=2):
        live = SimpleNamespace(state=3, slope=-6.5, dac=45000, gain_law=1,
                               sequence=1, target=3500, feedback=3500)
        self.telemetry = SimpleNamespace(optical_live_638=SimpleNamespace(page=live, host_stale=stale))
        self.stream_stats = {"stream_format": format}
        self.commands = []

    def read_profile_638(self):
        self.commands.append("profile")
        return {"id": 1, "name": "638-belycomm-v1"}

    def read_normalized_pi_638(self):
        self.commands.append("pi")
        return {"active": True, "slope_valid": True, "kp": .2, "ki_per_s": 750}

    def read_optical_cap_638(self):
        return {"max_dac": 52400}

    def read_controller_timing_638(self):
        return {"divider": 1, "held": False}

    def read_control_trace_config_638(self):
        return {"amplitude_dac": 2, "hold_updates": 1, "decimation": 1}

    def configure_control_trace_638(self, *args, timeout_s=1.0):
        self.commands.append("configure")
        return {"amplitude_dac": args[0], "hold_updates": args[1], "decimation": args[2]}

    def control_638(self, action):
        self.commands.append(action)

    def retained_control_trace(self):
        return None


class CharacterizationRefusalTests(unittest.TestCase):
    def run_case(self, controller, **kwargs):
        with TemporaryDirectory() as directory:
            result = run_pi_characterization(controller, directory, **kwargs)
            self.assertEqual(json.loads((Path(directory)/"summary.json").read_text())["status"], result["status"])
            self.assertTrue((Path(directory)/"report.md").exists())
            return result

    def test_stale_lock_never_configures_or_excites(self):
        controller = FakeController(stale=True)
        tick = [0.]
        def advance():
            tick[0] += .25
            return tick[0]
        with patch.object(pi_characterization.time, "monotonic", side_effect=advance), \
             patch.object(pi_characterization.time, "sleep"):
            result = self.run_case(controller)
        self.assertEqual(result["status"], "unqualified")
        self.assertIn("fresh healthy", result["reason"])
        self.assertEqual(controller.commands, [])

    def test_unsupported_diagnostics_never_queries_slave(self):
        controller = FakeController(format=1)
        result = self.run_case(controller)
        self.assertEqual(result["status"], "unqualified")
        self.assertIn("format 2", result["reason"])
        self.assertEqual(controller.commands, [])

    def test_preflight_firmware_rejection_keeps_partial_report(self):
        controller = FakeController()
        controller.read_control_trace_config_638 = lambda: (_ for _ in ()).throw(RuntimeError("unsupported firmware"))
        result = self.run_case(controller)
        self.assertEqual(result["status"], "unqualified")
        self.assertIn("unsupported firmware", result["reason"])
        self.assertNotIn("identify", controller.commands)

    def test_cancel_is_fast_and_does_not_touch_hardware(self):
        controller = FakeController()
        result = self.run_case(controller, cancel=lambda: True)
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(controller.commands, [])

    def test_invalid_driver_prior_rejected_before_hardware(self):
        controller = FakeController()
        result = self.run_case(controller, driver_pole_hz=800,
                               driver_pole_range_hz=[1000, 1600],
                               driver_pole_source="inconsistent estimate")
        self.assertEqual(result["status"], "unqualified")
        self.assertIn("range", result["reason"])
        self.assertEqual(controller.commands, [])

    def test_settling_progress_with_elapsed_detail_can_cancel(self):
        controller = FakeController()
        sequence = [0]
        class FreshTelemetry:
            @property
            def optical_live_638(self):
                sequence[0] += 1
                live = SimpleNamespace(state=3, slope=-6.5, dac=45000+sequence[0],
                                       gain_law=1, sequence=sequence[0], target=3500,
                                       feedback=3500)
                return SimpleNamespace(page=live, host_stale=False)
        controller.telemetry = FreshTelemetry()
        tick, events, stopped = [0.], [], [False]
        def advance():
            tick[0] += .04
            return tick[0]
        def progress(event):
            events.append(event)
            if event["stage"] == "settling" and "drift_dac_per_s" in event:
                stopped[0] = True
        with patch.object(pi_characterization.time, "monotonic", side_effect=advance), \
             patch.object(pi_characterization.time, "sleep"):
            result = self.run_case(controller, progress=progress, cancel=lambda: stopped[0])
        self.assertEqual(result["status"], "cancelled")
        self.assertTrue(any("drift_dac_per_s" in item for item in events))
        self.assertNotIn("configure", controller.commands)

    def test_brief_slope_measurement_resets_settling_then_reaches_capture_gate(self):
        controller = FakeController()
        tick, sequence, cancel_now = [0.], [0], [False]
        restore_calls = [0]
        configure = controller.configure_control_trace_638
        def busy_during_cancel(*args, timeout_s=1.0):
            restore_calls[0] += 1
            if restore_calls[0] == 2:
                raise SlaveCommandRejected(638, 13, 10, MANUAL_SET)
            return configure(*args, timeout_s=timeout_s)
        controller.configure_control_trace_638 = busy_during_cancel
        class FreshTelemetry:
            @property
            def optical_live_638(self):
                sequence[0] += 1
                state = 5 if 3 < tick[0] < 4 else 3
                live = SimpleNamespace(state=state, phase=1 if state == 5 else 0,
                                       slope=0 if state == 5 else -6.5,
                                       dac=45000, gain_law=0 if state == 5 else 1,
                                       sequence=sequence[0], target=3500, feedback=3500)
                return SimpleNamespace(page=live, host_stale=False)
        controller.telemetry = FreshTelemetry()
        def advance():
            tick[0] += .04
            return tick[0]
        def progress(event):
            if event["stage"] == "capture":
                cancel_now[0] = True
        with patch.object(pi_characterization.time, "monotonic", side_effect=advance), \
             patch.object(pi_characterization.time, "sleep"):
            result = self.run_case(controller, progress=progress,
                                   cancel=lambda: cancel_now[0])
        self.assertEqual(result["status"], "cancelled")
        self.assertTrue(result["settling"]["passed"])
        self.assertEqual(result["settling"]["slope_measurement_events"], 1)
        self.assertGreaterEqual(result["settling"]["stable_elapsed_s"], 20)
        self.assertNotIn("identify", controller.commands)
        self.assertEqual(result["reason"], "Cancelled by user")
        self.assertFalse(result["trace_config_restored"])
        self.assertEqual(result["trace_config_restore_attempts"], 1)
        self.assertEqual(restore_calls[0], 2)

    def test_stuck_slope_measurement_fails_without_excitation(self):
        controller = FakeController()
        tick, sequence = [0.], [0]
        class FreshTelemetry:
            @property
            def optical_live_638(self):
                sequence[0] += 1
                state = 5 if tick[0] > 3 else 3
                live = SimpleNamespace(state=state, phase=1 if state == 5 else 0,
                                       slope=0 if state == 5 else -6.5,
                                       dac=45000, gain_law=0 if state == 5 else 1,
                                       sequence=sequence[0], target=3500, feedback=3500)
                return SimpleNamespace(page=live, host_stale=False)
        controller.telemetry = FreshTelemetry()
        def advance():
            tick[0] += .04
            return tick[0]
        with patch.object(pi_characterization.time, "monotonic", side_effect=advance), \
             patch.object(pi_characterization.time, "sleep"):
            result = self.run_case(controller)
        self.assertEqual(result["status"], "unqualified")
        self.assertIn("exceeded five seconds", result["reason"])
        self.assertNotIn("identify", controller.commands)

    def test_non_slope_transition_during_settling_still_rejects(self):
        controller = FakeController()
        tick, sequence = [0.], [0]
        class FreshTelemetry:
            @property
            def optical_live_638(self):
                sequence[0] += 1
                state = 4 if tick[0] > 3 else 3
                live = SimpleNamespace(state=state, phase=0, slope=-6.5,
                                       dac=45000, gain_law=1, sequence=sequence[0],
                                       target=3500, feedback=3500)
                return SimpleNamespace(page=live, host_stale=False)
        controller.telemetry = FreshTelemetry()
        def advance():
            tick[0] += .04
            return tick[0]
        with patch.object(pi_characterization.time, "monotonic", side_effect=advance), \
             patch.object(pi_characterization.time, "sleep"):
            result = self.run_case(controller)
        self.assertEqual(result["status"], "unqualified")
        self.assertIn("left locked state during settling", result["reason"])
        self.assertNotIn("identify", controller.commands)


class TraceRestoreTests(unittest.TestCase):
    original = {"amplitude_dac": 2, "hold_updates": 1, "decimation": 1}

    def run_restore(self, replies, *, timeout_s=3, cancel=None):
        calls, events, clock = [], [], [0.]

        class Controller:
            def configure_control_trace_638(self, *args, timeout_s):
                calls.append((args, timeout_s))
                reply = replies[min(len(calls)-1, len(replies)-1)]
                if isinstance(reply, Exception):
                    raise reply
                return reply

        def sleep(seconds):
            clock[0] += seconds

        state = {"attempts": 0}
        with patch.object(pi_characterization.time, "monotonic", side_effect=lambda: clock[0]), \
             patch.object(pi_characterization.time, "sleep", side_effect=sleep):
            try:
                pi_characterization._restore_trace_config(
                    Controller(), self.original, timeout_s=timeout_s,
                    emit=lambda *args, **kwargs: events.append((args, kwargs)),
                    cancel=cancel, state=state)
            except Exception as exc:
                return calls, events, state, exc
        return calls, events, state, None

    def test_busy_then_exact_readback_succeeds(self):
        calls, events, state, error = self.run_restore(
            [SlaveCommandRejected(638, 13, 10, MANUAL_SET), self.original])
        self.assertIsNone(error)
        self.assertEqual(state["attempts"], 2)
        self.assertEqual([call[0] for call in calls], [(2, 1, 1), (2, 1, 1)])
        self.assertTrue(any("confirmed" in args[1] for args, _ in events))

    def test_busy_deadline_is_finite(self):
        calls, _, state, error = self.run_restore(
            [SlaveCommandRejected(638, 13, 10, MANUAL_SET)], timeout_s=.2)
        self.assertIsInstance(error, TimeoutError)
        self.assertEqual(len(calls), 1)
        self.assertEqual(state["attempts"], 1)

    def test_wrong_readback_stops_without_retry(self):
        wrong = dict(self.original, decimation=4)
        calls, _, _, error = self.run_restore([wrong])
        self.assertIsInstance(error, RuntimeError)
        self.assertIn("readback differs", str(error))
        self.assertEqual(len(calls), 1)

    def test_transport_failure_is_not_retried(self):
        calls, _, _, error = self.run_restore([TimeoutError("transport uncertain")])
        self.assertIsInstance(error, TimeoutError)
        self.assertEqual(len(calls), 1)

    def test_other_status_is_not_retried(self):
        calls, _, _, error = self.run_restore([SlaveCommandRejected(638, 13, 6, MANUAL_SET)])
        self.assertIsInstance(error, SlaveCommandRejected)
        self.assertEqual(len(calls), 1)

    def test_busy_from_get_is_not_retried(self):
        calls, _, _, error = self.run_restore([SlaveCommandRejected(638, 13, 10, MANUAL_GET)])
        self.assertIsInstance(error, SlaveCommandRejected)
        self.assertEqual(len(calls), 1)

    def test_cancel_stops_after_cleanup_attempt(self):
        calls, _, _, error = self.run_restore(
            [SlaveCommandRejected(638, 13, 10, MANUAL_SET)], cancel=lambda: True)
        self.assertIsInstance(error, pi_characterization._Cancelled)
        self.assertEqual(len(calls), 1)

    def test_complete_trace_waits_for_restore_before_done(self):
        controller = FakeController()
        tick, sequence, config_calls, done = [0.], [0], [], []

        class FreshTelemetry:
            @property
            def optical_live_638(self):
                sequence[0] += 1
                live = SimpleNamespace(state=3, phase=0, slope=-6.5, dac=45000,
                                       gain_law=1, sequence=sequence[0], target=3500,
                                       feedback=3500)
                return SimpleNamespace(page=live, host_stale=False)

        def configure(*args, timeout_s=1.0):
            config_calls.append(args)
            if len(config_calls) == 2:
                raise SlaveCommandRejected(638, 13, 10, MANUAL_SET)
            return dict(zip(pi_characterization.TRACE_CONFIG_FIELDS, args))

        def control(action):
            controller.commands.append(action)
            controller.identify_at = tick[0]

        samples = {i: {"cycles": i, "feedback": 3500, "actual_dac": 45000,
                       "injection_dac": 0, "flags": 0} for i in range(4096)}
        trace = {"stream_epoch": 1, "capture_id": 1, "total_samples": 4096,
                 "samples": samples, "missing_indices": [], "flags": 0,
                 "conflict": False, "complete": True}

        def retained():
            return trace if getattr(controller, "identify_at", None) is not None \
                and tick[0]-controller.identify_at >= 2.5 else None

        def advance():
            tick[0] += .04
            return tick[0]

        controller.telemetry = FreshTelemetry()
        controller.configure_control_trace_638 = configure
        controller.control_638 = control
        controller.retained_control_trace = retained
        with TemporaryDirectory() as directory, \
             patch.object(pi_characterization.time, "monotonic", side_effect=advance), \
             patch.object(pi_characterization.time, "sleep"), \
             patch.object(pi_characterization, "review", return_value={"accepted": True}):
            def progress(event):
                if event["stage"] == "done":
                    saved = json.loads((Path(directory)/"summary.json").read_text())
                    done.append((event["status"], saved["status"],
                                 saved["trace_config_restored"], len(config_calls)))
            result = run_pi_characterization(controller, directory, progress=progress,
                                             verified_master_filter="single_sample")
        self.assertEqual(result["status"], "complete", result.get("reason"))
        self.assertEqual(config_calls, [(8, 4, 4), (2, 1, 1), (2, 1, 1)])
        self.assertEqual(result["trace_config_restore_attempts"], 2)
        self.assertEqual(done, [("complete", "complete", True, 3)])

if __name__ == "__main__":
    unittest.main()
