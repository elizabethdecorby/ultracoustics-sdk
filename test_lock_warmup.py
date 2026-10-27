"""No-hardware tests for the lock warm-up: settle, check, retune, store and confirm on the simulated box of
test_lock_check (a 1 us closed-loop stream behind the calibration tests' Fake controller).

Run: /Users/ayden/Ultracoustics/.venv-638/bin/python -I -c "import sys,unittest; sys.path.insert(0,'.'); unittest.main(module='test_lock_warmup')"
"""
import dataclasses
import json
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ultracoustics import lock_calibration, lock_check, lock_warmup, warm_up_lock_638
from ultracoustics._internal import lock_model as lm
from test_lock_calibration import Clock
from test_lock_check import StreamFake, onset_constant, sampled_plant


class WarmFake(StreamFake):
    """StreamFake that can STOP, START and store a tuning record, like the master and 638 do."""

    def __init__(self, clock, const, ring_after_apply=False, noisy_dac=False, silent_s=0.0, **kw):
        super().__init__(clock, const, **kw)
        self.silent_until = clock() + silent_s                 # booting: no 638 pages yet
        self.ring_after_apply, self.noisy_dac = ring_after_apply, noisy_dac
        self.record, self.ringing = None, False
        self.noise = np.random.default_rng(5)

    @property
    def running(self):
        return self.state != 0

    @property
    def telemetry(self):
        if self.clock() < self.silent_until:
            return None
        snap = super().telemetry
        if self.noisy_dac:
            page = snap.optical_live_638.page
            snap.optical_live_638.page = dataclasses.replace(page, dac=20000 + int(self.noise.integers(-400, 400)))
        return snap

    @property
    def buffer(self):
        if not hasattr(self, "_ring"):
            return np.zeros(self.RATE, dtype=np.uint16)
        now = self.samples_received
        if now > self._filled:
            idx = np.arange(max(self._filled, now - self.buffer_capacity), now)
            values = 6000.0 + self.noise.normal(0, 2, len(idx))
            if self.ringing:                                   # a loop ringing at 400 Hz
                values += 60 * np.sin(2 * np.pi * 400 * idx / self.RATE)
            if self._burst is not None:
                offset = idx - self._burst[1]
                inside = (offset >= 0) & (offset < len(self._burst[0]))
                values[inside] = self._burst[0][offset[inside]]
            self._ring[idx % self.buffer_capacity] = np.round(values).astype(np.uint16)
            self._filled = now
        return self._ring

    @buffer.setter
    def buffer(self, _value):
        pass

    def set_normalized_pi_638(self, kp, ki, zero_hz=0, timeout_s=1):
        result = super().set_normalized_pi_638(kp, ki, zero_hz=zero_hz, timeout_s=timeout_s)
        self.ringing = self.ring_after_apply
        return result

    def restore_normalized_pi_638(self, timeout_s=1):
        self.ringing = False
        return super().restore_normalized_pi_638(timeout_s)

    def stop_system_confirmed(self, timeout_s=1):
        self.calls.append("stop")
        self.state = 0
        return SimpleNamespace(current_state=0)

    def enable_optical_diagnostics(self, timeout_s=1):
        self.calls.append("diagnostics")

    def write_lock_tuning_638(self, kp, ki_per_s, zero_hz=0, report_id=None, timeout_s=15):
        assert self.state == 0, "records are written in IDLE"
        self.calls.append(("write", kp, ki_per_s, zero_hz))
        self.record = dict(kp=kp, ki_per_s=ki_per_s, zero_hz=zero_hz)
        return dict(lock_kp=kp, lock_ki_per_s=ki_per_s, lock_zero_hz=zero_hz, report_id=report_id, stored=True)

    def start_confirmed(self, timeout_s=1):
        self.calls.append("start")
        if self.record is not None:                           # the 638 boots with the record's gains
            self.stored = dict(self.record)
            self.defaults = dict(self.record)
        self.pi.update(overridden=False, **self.defaults)
        self.state, self.busy_until, self.ringing = 3, self.clock(), False   # power-cycled: no pending readout
        return SimpleNamespace(current_state=5)


def warm(fake, clock, **kwargs):
    events = []
    fake_time = SimpleNamespace(monotonic=clock, sleep=clock.sleep, time=time.time)
    with TemporaryDirectory() as tmp, patch.object(lock_calibration, "time", fake_time), \
            patch.object(lock_check, "time", fake_time), patch.object(lock_warmup, "time", fake_time), \
            patch.object(lock_check, "SAMPLE_RATE", StreamFake.RATE), \
            patch.object(lock_calibration, "SAMPLE_RATE", StreamFake.RATE), \
            patch.object(lock_calibration, "SAMPLES_PER_BIN", StreamFake.RATE // 10_000):
        report = warm_up_lock_638(fake, Path(tmp) / "warm", progress=events.append,
                                  onset_after_sample_us=fake.const, **kwargs)
        saved = json.loads((Path(tmp) / "warm" / "warm_up.json").read_text())
    return report, events, saved


class WarmUpTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.const = onset_constant(np.random.default_rng(10))
        f, P = sampled_plant()
        band = (f > 60) & (f < 3000)
        cls.params = lm.fit({"f": f, "H": P, "relerr": np.where(band, 0.01, np.inf), "fs": 1e4})["params"]
        cls.design = lm.migo([cls.params], 1.4, kp_limit=2.0, ki_limit=16383.0, contract=2)

    def fake(self, kp, ki, **kw):
        clock = Clock()
        return WarmFake(clock, self.const, stored=dict(kp=kp, ki_per_s=ki, zero_hz=0), **kw), clock

    def writes(self, fake):
        return [c for c in fake.calls if isinstance(c, tuple) and c[0] == "write"]

    def test_good_gains_are_left_alone(self):
        fake, clock = self.fake(*self.design)
        report, events, saved = warm(fake, clock)
        self.assertEqual(report["outcome"], "ready", report["message"])
        self.assertFalse(report["changed"])
        self.assertEqual(report["check"]["status"], "ok")
        self.assertEqual(self.writes(fake), [])
        self.assertNotIn("stop", fake.calls)
        self.assertEqual({e["message"] for e in events}, {"Warming up", "Ready"})   # nothing else for users
        fractions = [e["fraction"] for e in events]
        self.assertEqual(fractions, sorted(fractions))
        self.assertEqual(events[-1]["fraction"], 1.0)
        self.assertGreaterEqual(report["settle"]["locked_s"], 10)
        self.assertEqual(saved["outcome"], "ready")

    def test_hot_gains_are_retuned_stored_and_confirmed(self):
        kp, ki = lm.migo([self.params], 1.9, kp_limit=2.0, ki_limit=16383.0, contract=2)
        fake, clock = self.fake(kp, ki)
        report, events, _ = warm(fake, clock)
        self.assertEqual(report["outcome"], "retuned", report["message"])
        self.assertEqual(report["check"]["status"], "rerun")
        self.assertTrue(report["apply"]["verified"])
        rec = report["gains_after"]
        self.assertEqual(self.writes(fake), [("write", rec["kp"], rec["ki_per_s"], 0)])
        self.assertTrue(report["store"]["verified"])
        self.assertEqual(report["confirm"]["status"], "ok")
        self.assertEqual(report["confirm"]["gains"]["ki_per_s"], rec["ki_per_s"])
        self.assertEqual(fake.calls.count("identify"), 2)        # the restart cleared the 638's busy readout
        self.assertEqual({e["message"] for e in events}, {"Warming up", "Ready"})

    def test_slow_gains_are_retuned(self):
        kp, ki = self.design
        fake, clock = self.fake(kp, round(ki / 2))
        report, *_ = warm(fake, clock)
        self.assertEqual(report["outcome"], "retuned", report["message"])
        self.assertEqual(report["check"]["status"], "improvable")
        self.assertGreater(report["gains_after"]["ki_per_s"], 1.3 * round(ki / 2))

    def test_runtime_only_without_store(self):
        kp, ki = self.design
        fake, clock = self.fake(kp, round(ki / 2))
        report, *_ = warm(fake, clock, store=False)
        self.assertEqual(report["outcome"], "retuned")
        self.assertEqual(self.writes(fake), [])
        self.assertTrue(fake.pi["overridden"])

    def test_ringing_after_apply_puts_the_previous_gains_back(self):
        kp, ki = self.design
        fake, clock = self.fake(kp, round(ki / 2), ring_after_apply=True)
        report, *_ = warm(fake, clock)
        self.assertEqual(report["outcome"], "ready", report["message"])
        self.assertFalse(report["apply"]["verified"])
        self.assertIn("residual", report["apply"]["message"])
        self.assertTrue(report["apply"]["details"]["previous_gains_restored"])
        self.assertFalse(fake.pi["overridden"])                 # back on the stored defaults
        self.assertEqual(self.writes(fake), [])

    def test_inconclusive_check_changes_nothing(self):
        fake, clock = self.fake(*self.design, drop=True)
        report, *_ = warm(fake, clock)
        self.assertEqual(report["outcome"], "ready")
        self.assertEqual(report["check"]["status"], "inconclusive")
        self.assertIsNone(report["apply"])

    def test_a_lock_that_never_settles_skips_the_check(self):
        fake, clock = self.fake(*self.design, noisy_dac=True)
        report, *_ = warm(fake, clock)
        self.assertEqual(report["outcome"], "ready")
        self.assertFalse(report["settle"]["settled"])
        self.assertNotIn("identify", fake.calls)

    def test_a_638_in_error_is_not_waited_on(self):
        fake, clock = self.fake(*self.design)
        fake.state = 4                                         # ERROR: acquisition found no dip
        start = clock()
        report, *_ = warm(fake, clock)
        self.assertFalse(report["settle"]["settled"])
        self.assertIn("ERROR", report["settle"]["reason"])
        self.assertLess(clock() - start, 10)
        self.assertNotIn("identify", fake.calls)

    def test_a_long_boot_is_waited_out(self):
        fake, clock = self.fake(*self.design, silent_s=25.0)   # the 1550 warming up from cold
        report, *_ = warm(fake, clock)
        self.assertEqual(report["outcome"], "ready", report["message"])
        self.assertTrue(report["settle"]["settled"])

    def test_cancel_while_settling(self):
        fake, clock = self.fake(*self.design)
        start = clock()
        report, *_ = warm(fake, clock, cancel=lambda: clock() - start > 3)
        self.assertEqual(report["outcome"], "cancelled")
        self.assertNotIn("identify", fake.calls)


if __name__ == "__main__":
    unittest.main()
