"""Small offline regression checks; no hardware and no retained-trace fixtures."""
import copy
import unittest

import numpy as np

from ultracoustics import pi_analysis as paired


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
                       "hold_updates": 4, "decimation": 4},
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


if __name__ == "__main__":
    unittest.main()
