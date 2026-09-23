"""No-hardware refusal and cancellation checks for PI characterization."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest
from unittest.mock import patch

from ultracoustics import run_pi_characterization
from ultracoustics import pi_characterization


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

    def configure_control_trace_638(self, *args):
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


if __name__ == "__main__":
    unittest.main()
