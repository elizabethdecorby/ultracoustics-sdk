import math
import tempfile
import unittest
from pathlib import Path

from ultracoustics.power_calibration import CalibrationPoint, PowerCalibration, PowerCalibrationError


def calibration(**changes):
    points = [CalibrationPoint(i, i, i * 100, float(i), raw={"vendor_field": i})
              for i in range(15)]
    values = dict(board_id=1550, board_serial="ABC123", points=points,
                  hardware_metadata={"feedback_resistance_ohm": 5000})
    values.update(changes)
    return PowerCalibration(**values)


class PowerCalibrationTests(unittest.TestCase):
    def test_interpolation_range_and_round_trip(self):
        value = calibration()
        self.assertAlmostEqual(value.power_mw(250), 2.5)
        with self.assertRaisesRegex(PowerCalibrationError, "outside"):
            value.power_mw(-1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "curve.json"
            value.save(path)
            loaded = PowerCalibration.load(path, 1550, "ABC123")
            self.assertTrue(loaded.complete)
            self.assertEqual(loaded.points[4].raw["vendor_field"], 4)

    def test_dac_nonmonotonic_and_duplicate_pd_collapse(self):
        value = calibration()
        value.points[5].dac_applied = 1
        value.points[1].pd_raw_counts = value.points[0].pd_raw_counts
        value.points[1].measured_mw = value.points[0].measured_mw
        self.assertTrue(value.complete)
        value.points[1].measured_mw = 99
        self.assertFalse(value.complete)
        with self.assertRaisesRegex(PowerCalibrationError, "ambiguous"):
            value.power_mw(0)

    def test_incomplete_or_unsafe_points_never_make_curve(self):
        mutations = (lambda p: setattr(p[0], "measured_mw", None),
                     lambda p: setattr(p[0], "measured_mw", math.nan),
                     lambda p: setattr(p[0], "saturated", True), lambda p: p.pop())
        for mutation in mutations:
            value = calibration(); mutation(value.points)
            self.assertFalse(value.complete)
            with self.assertRaises(PowerCalibrationError): value.power_mw(100)

    def test_identity_hardware_scope_and_resistor_not_global(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "curve.json"; calibration().save(path)
            with self.assertRaisesRegex(PowerCalibrationError, "serial"):
                PowerCalibration.load(path, 1550, "OTHER")
            with self.assertRaisesRegex(PowerCalibrationError, "metadata"):
                PowerCalibration.load(path, 1550, "ABC123", {"feedback_resistance_ohm": 10000})
        self.assertTrue(calibration(hardware_metadata={"feedback_resistance_ohm": 10000}).complete)


if __name__ == "__main__": unittest.main()
