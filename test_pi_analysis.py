"""Small offline regression checks; no hardware and no retained-trace fixtures."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from ultracoustics import pi_analysis as paired
from ultracoustics import review_saved_pi_characterization
from ultracoustics._internal import pi_model


def synthetic_capture():
    """Closed-loop, negative-slope RC + thermal plant sampled at 10 kHz."""
    rng = np.random.default_rng(11)
    n = 4096 * 4
    injection = np.repeat(rng.choice([-16., 16.], size=n // 4 + 1), 4)[:n]
    command = np.zeros(n)
    driver = np.zeros(n)
    feedback = np.zeros(n)
    for k in range(1, n):
        command[k] = injection[k] + .15 * feedback[k - 1]
        driver[k] = driver[k - 1] + .63 * (command[k - 1] - driver[k - 1])
        feedback[k] = feedback[k - 1] + .142 * (-7 * driver[k - 1] - feedback[k - 1])
        feedback[k] += rng.normal(0, .35)

    def recorder(signal):
        for _ in range(4):
            signal = np.convolve(signal, np.ones(4) / 4, "full")[:n]
        return signal[3::4][:4096]

    y, u, d = map(recorder, (feedback, command, injection))
    samples = {
        str(i): {"cycles": (12345 + i * 100000) & 0xffffffff,
                 "feedback": round(3500 + y[i]),
                 "actual_dac": round(45000 + u[i]),
                 "injection_dac": round(d[i]), "flags": 1}
        for i in range(4096)
    }
    live = {"state": 3, "gain_law": 1, "target": 3500,
            "dac": 45000, "slope": -7.0}
    return {"trace": {"total_samples": 4096, "complete": True,
                      "conflict": False, "missing_indices": [], "flags": 5,
                      "decimation": 4, "amplitude_dac": 16,
                      "hold_updates": 4, "recorder_filtered": True,
                      "clock_hz": 250000000, "samples": samples},
            "config": {"dac_cap": 52400, "master_feedback_filter": "single_sample",
                       "control_rate_hz": 10000, "amplitude_dac": 16,
                       "hold_updates": 4, "decimation": 4,
                       "driver_pole_hz": 1600,
                       "driver_pole_source": "synthetic 1600 Hz first-order driver"},
            "live": live, "capture_live_after": live}


class PairedReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.capture = synthetic_capture()
        cls.positive = paired.review(cls.capture)

    def test_closed_loop_plant_and_actual_support(self):
        result = self.positive
        self.assertTrue(result["accepted"], result.get("reason"))
        self.assertAlmostEqual(result["plant_fit"]["gain_adc_per_dac"], -7, delta=.5)
        self.assertAlmostEqual(result["plant_fit"]["thermal_tau_s"], .00065, delta=.0002)
        self.assertLess(result["plant_fit"]["heldout_rms_sigma"], 1.5)
        first = {row["hz"] for row in result["first_half"]["frf"] if 10 <= row["hz"] <= 400}
        second = {row["hz"] for row in result["second_half"]["frf"] if 10 <= row["hz"] <= 400}
        common = sorted(first & second)
        self.assertEqual(result["empirical_screen"]["measured_support_hz"],
                         [common[0], common[-1]])
        self.assertFalse(result["empirical_screen"]["candidate_margin_is_confidence_bound"])
        self.assertEqual(len(result["empirical_screen"]["candidate_grid"]),
                         result["empirical_screen"]["candidate_count"])

    def test_missing_row_rejected(self):
        capture = copy.deepcopy(self.capture)
        capture["trace"]["samples"].pop("3000")
        self.assertIn("missing or duplicated", paired.review(capture)["reason"])

    def test_mismatched_heldout_plant_rejected(self):
        capture = copy.deepcopy(self.capture)
        for i in range(2048, 4096):
            row = capture["trace"]["samples"][str(i)]
            row["feedback"] = 3500 + 2 * (row["feedback"] - 3500)
        result = paired.review(capture)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "adjacent empirical FRFs disagree")

    def test_strict_and_crossover_only_have_distinct_band_gates(self):
        first = [r for r in self.positive["first_half"]["frf"] if r["hz"] >= 48]
        second = [r for r in self.positive["second_half"]["frf"] if r["hz"] >= 48]
        strict = paired.empirical_screen(first, second, -7, 10000, 1, 1000)
        crossover = paired.empirical_screen(first, second, -7, 10000, 1, 1000,
                                             crossover_only=True)
        self.assertFalse(strict["accepted"])
        self.assertIn("20-300 Hz", strict["reason"])
        self.assertTrue(crossover["accepted"], crossover.get("reason"))
        self.assertGreaterEqual(crossover["measured_support_hz"][0], 48)

    def test_screen_baseline_comes_from_measured_runtime_pair(self):
        capture = copy.deepcopy(self.capture)
        capture["pi"] = {"kp": .483, "ki_per_s": 1178.7}
        capture["runtime_caps"] = {"kp_max": 1, "ki_max_per_s": 1178.7}
        result = paired.review(capture, crossover_only=True)
        self.assertEqual(result["empirical_screen"]["baseline"]["kp"], .483)
        self.assertEqual(result["empirical_screen"]["baseline"]["ki_per_s"], 1178.7)

    def test_below_band_baseline_still_supplies_measured_sensitivity_comparator(self):
        capture = copy.deepcopy(self.capture)
        capture["pi"] = {"kp": .2, "ki_per_s": 90}
        result = paired.review(capture, crossover_only=True)
        baseline = result["empirical_screen"]["baseline"]
        self.assertFalse(baseline["crossover_in_measured_band"])
        self.assertIsNone(baseline["worst_phase_margin_deg"])
        self.assertIsInstance(baseline["worst_lowband_sensitivity_db"], float)
        self.assertTrue(result["accepted"], result.get("reason"))
        self.assertTrue(all(item["crossover_in_measured_band"]
                            for item in result["empirical_screen"]["best_supported"]))

    def test_missing_driver_prior_keeps_empirical_evidence(self):
        capture = copy.deepcopy(self.capture)
        capture["config"].pop("driver_pole_hz")
        capture["config"].pop("driver_pole_source")
        result = paired.review(capture, crossover_only=True)
        self.assertNotIn("plant_fit", result)
        self.assertFalse(result["driver_model"]["available"])
        self.assertIn("empirical_screen", result)
        self.assertEqual(result["basis"], "empirical halves only")

    def test_driver_prior_is_explicit_and_bounded(self):
        nominal, bounds, source = pi_model.driver_pole_config({
            "driver_pole_hz": 800, "driver_pole_range_hz": [600, 1200],
            "driver_pole_source": "board revision estimate"})
        self.assertEqual((nominal, bounds, source), (800, (600, 1200), "board revision estimate"))
        with self.assertRaisesRegex(ValueError, "provenance"):
            pi_model.driver_pole_config({"driver_pole_hz": 800})
        with self.assertRaisesRegex(ValueError, "range"):
            pi_model.driver_pole_config({"driver_pole_hz": 800,
                                         "driver_pole_range_hz": [900, 1400],
                                         "driver_pole_source": "estimate"})
        f = np.asarray([800.])
        slow = pi_model.model(f, -7, .0005, .00005, np.ones_like(f), 800)
        fast = pi_model.model(f, -7, .0005, .00005, np.ones_like(f), 1600)
        self.assertLess(abs(slow[0]), abs(fast[0]))

    def test_rejected_band_reports_actual_coverage(self):
        first = [r for r in self.positive["first_half"]["frf"] if r["hz"] >= 60]
        second = [r for r in self.positive["second_half"]["frf"] if r["hz"] >= 60]
        result = paired.empirical_screen(first, second, -7, 10000, 1, 1000,
                                         crossover_only=True)
        self.assertFalse(result["accepted"])
        self.assertFalse(result["coverage"]["has_low_coverage"])
        self.assertGreaterEqual(result["coverage"]["common_band_hz"][0], 60)

    def test_sparse_comparison_band_cannot_pass(self):
        first = [r for r in self.positive["first_half"]["frf"]
                 if r["hz"] <= 50 or r["hz"] >= 120]
        second = [r for r in self.positive["second_half"]["frf"]
                  if r["hz"] <= 50 or r["hz"] >= 120]
        result = paired.empirical_screen(first, second, -7, 10000, 1, 1000,
                                         crossover_only=True)
        self.assertFalse(result["accepted"])
        self.assertLess(result["coverage"]["comparison_bin_count"], 3)

    def test_dac_shift_withholds_offline_candidates(self):
        capture = copy.deepcopy(self.capture)
        capture["capture_live_after"] = dict(capture["capture_live_after"])
        capture["capture_live_after"]["dac"] += 150
        result = paired.review(capture, crossover_only=True)
        self.assertFalse(result["accepted"])
        self.assertIn("DAC operating point", result["reason"])

    def test_saved_trace_can_be_reanalyzed_with_later_pole_evidence(self):
        capture = self.capture
        with TemporaryDirectory() as directory:
            target = Path(directory)
            metadata = {k: v for k, v in capture["trace"].items() if k != "samples"}
            report = {"board_serial": "synthetic", "profile": {"id": 1},
                      "trace_metadata": metadata, "trace_file": "trace.npz",
                      "capture_counter_deltas": {k: 0 for k in
                                                 ("drops_seq", "drops_fw", "transfer_errors",
                                                  "transfer_timeouts", "malformed")},
                      "capture_live_after_host_delay_s": 2.1,
                      "configuration": {k: v for k, v in capture["config"].items()
                                        if not k.startswith("driver_pole")},
                      "pi_before": {"kp": .2, "ki_per_s": 750},
                      "live_before": capture["live"],
                      "capture_live_after": capture["capture_live_after"]}
            original = json.dumps(report)
            (target/"summary.json").write_text(original)
            keys = ("cycles", "feedback", "actual_dac", "injection_dac", "flags")
            rows = np.asarray([[capture["trace"]["samples"][str(i)][k] for k in keys]
                               for i in range(4096)], dtype=np.int64)
            np.savez_compressed(target/"trace.npz", rows=rows)
            without = review_saved_pi_characterization(target)
            self.assertFalse(without["analysis"]["driver_model"]["available"])
            with_prior = review_saved_pi_characterization(
                target, driver_pole_hz=800,
                driver_pole_range_hz=[600, 1200],
                driver_pole_source="independent board estimate")
            self.assertEqual(with_prior["analysis"]["plant_fit"]["driver_pole_hz"], 800)
            self.assertEqual((target/"summary.json").read_text(), original)
            with self.assertRaisesRegex(ValueError, "new provenance"):
                review_saved_pi_characterization(target, driver_pole_hz=800)
            with self.assertRaisesRegex(ValueError, "range"):
                review_saved_pi_characterization(
                    target, driver_pole_hz=800,
                    driver_pole_range_hz=[900, 1200],
                    driver_pole_source="invalid replay estimate")


if __name__ == "__main__":
    unittest.main()
