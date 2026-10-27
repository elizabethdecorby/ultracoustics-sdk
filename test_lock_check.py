"""No-hardware tests for the quick lock check: the stream analysis on a closed loop simulated at 1 us, and
check_lock_638 against the calibration tests' Fake controller serving that simulated stream.

Run: /Users/ayden/Ultracoustics/.venv-638/bin/python -I -c "import sys,unittest; sys.path.insert(0,'.'); unittest.main(module='test_lock_check')"
"""
import json
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.signal import lfilter

from ultracoustics import check_lock_638, lock_calibration, lock_check
from ultracoustics._internal import lock_model as lm
from ultracoustics._internal import stream_lock as sl
from ultracoustics._internal.control import MANUAL_SET
from ultracoustics.system_control import SlaveCommandRejected
from test_lock_calibration import Clock, Fake

PLANT = dict(g=0.9, tau=0.42e-3, driver_hz=1200.0, delay_us=20, slope=-18.0)
WRITE_US = 12           # controller sample -> DAC write
HOLD, AMP = 4, 24
BURST_UPDATES = 4096 * 4


def firmware_prbs(n_updates, hold, amplitude):
    """Per-update transcription of ControlTrace_NextInjection, independent of stream_lock.prbs9."""
    state, count, out = 0x1FF, 0, []
    for _ in range(n_updates):
        out.append(amplitude if state & 1 else -amplitude)
        count += 1
        if count == hold:
            count = 0
            state = (state >> 1) | (((state ^ (state >> 4)) & 1) << 8)
    return np.array(out, dtype=float)


def _matrices():
    ad = np.exp(-2 * np.pi * PLANT["driver_hz"] * 1e-6)
    at = np.exp(-1e-6 / PLANT["tau"])
    A = np.array([[ad, 0.0], [1 - at, at]])
    B = np.array([1 - ad, 0.0])
    return ad, at, A, B


def _step(A, B, length):
    """x -> A^L x + F u for L 1-us steps at constant input."""
    AL = np.linalg.matrix_power(A, length)
    return AL, np.linalg.solve(np.eye(2) - A, (np.eye(2) - AL) @ B)


def simulate_stream(kp, ki, rng, phase_us=37, pre_updates=3000, post_updates=3000, noise=3.0, burst=True):
    """Closed loop with the LockedPi law at 10 kHz; the plant (driver pole, delay, thermal lag) at 1 us.
    Returns the 1 us stream and the stream index of the controller sample of the burst's first update."""
    ad, at, A, B = _matrices()
    lag = WRITE_US + PLANT["delay_us"]                     # the DAC change reaches the plant this long after a sample
    A1, F1 = _step(A, B, lag)
    A2, F2 = _step(A, B, 100 - lag)
    n = pre_updates + BURST_UPDATES + post_updates
    r = np.zeros(n)
    if burst:
        r[pre_updates:pre_updates + BURST_UPDATES] = firmware_prbs(BURST_UPDATES, HOLD, AMP)
    k_est, k_true = PLANT["slope"], PLANT["slope"] * PLANT["g"]
    y0, x0 = 6000.0, 20000.0
    total_us = n * 100 + phase_us + 200
    noise_us = rng.normal(0, noise, total_us)
    walk = np.cumsum(rng.normal(0, 0.03, n))
    x, u_prev, integ = np.zeros(2), 0.0, x0
    u = np.empty(n)
    for j in range(n):
        y = y0 + k_true * (x[1] + walk[j]) + noise_us[phase_us + 100 * j]
        e = (y0 - y) / k_est
        integ += ki * e * lm.DT
        u[j] = integ + kp * e + r[j]
        # from this sample to the next: the previous command until the new one arrives, then the new one
        x = A1 @ x + F1 * u_prev
        x = A2 @ x + F2 * (u[j] - x0)
        u_prev = u[j] - x0
    staircase = np.zeros(total_us)
    for j in range(n):
        staircase[phase_us + 100 * j + lag: phase_us + 100 * (j + 1) + lag] = u[j] - x0
    xd = lfilter([0, 1 - ad], [1, -ad], staircase)
    xt = lfilter([0, 1 - at], [1, -at], xd)
    walk_us = np.repeat(walk, 100)
    walk_us = np.concatenate((np.full(phase_us, walk[0]), walk_us, np.full(total_us, walk[-1])))[:total_us]
    stream = y0 + k_true * (xt + walk_us) + noise_us
    return stream, phase_us + 100 * pre_updates


def sampled_plant():
    """Normalized plant H of the simulated loop as the controller sees it: its sampled pulse response."""
    ad, at, A, B = _matrices()
    lag = WRITE_US + PLANT["delay_us"]
    stair = np.zeros(100 * 400)
    stair[lag:100 + lag] = 1.0
    xt = lfilter([0, 1 - at], [1, -at], lfilter([0, 1 - ad], [1, -ad], stair))
    p = PLANT["g"] * xt[::100]                              # sampled response, per DAC of a one-update pulse
    f = np.arange(1, 4000) * 0.25
    return f, np.array([np.sum(p * np.exp(-2j * np.pi * fi * lm.DT * np.arange(len(p)))) for fi in f])


def true_peak_S(kp, ki, fmin=60.0, fmax=sl.FMAX_HZ):
    """Peak |S| of the simulated loop, 1 / (1 + C H) on the exact sampled plant."""
    f, P = sampled_plant()
    S = np.abs(1 / (1 + lm.controller(f, kp, ki) * P))
    band = (f > fmin) & (f < fmax)
    return float(S[band].max())


def onset_constant(rng):
    """The simulated loop's own onset-after-sample constant (the hardware's is ~87 us)."""
    stream, start = simulate_stream(0.4, 900.0, rng, phase_us=11)
    a = sl.analyse(stream, AMP, HOLD, PLANT["slope"], 0.4, 900.0, onset_after_sample_us=0.0)
    return a["start_us"] - start


class StreamAnalysisTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.const = onset_constant(np.random.default_rng(10))

    def test_prbs9_is_the_firmware_sequence(self):
        one = sl.prbs9(HOLD, AMP)
        self.assertEqual(len(one), 511 * HOLD)
        np.testing.assert_array_equal(one, firmware_prbs(511 * HOLD, HOLD, AMP))
        np.testing.assert_array_equal(np.tile(one, 2), firmware_prbs(2 * 511 * HOLD, HOLD, AMP))  # period 511 chips
        self.assertEqual(int((one[::HOLD] > 0).sum()), 256)

    def test_timing_and_peak_S_match_the_simulated_loop(self):
        for kp, ki, phase in ((0.4, 900.0, 63), (0.7, 2600.0, 5)):
            stream, start = simulate_stream(kp, ki, np.random.default_rng(int(ki)), phase_us=phase)
            a = sl.analyse(stream, AMP, HOLD, PLANT["slope"], kp, ki, onset_after_sample_us=self.const)
            self.assertLessEqual(abs(a["start_us"] - start), 3, (kp, ki))
            self.assertEqual(a["periods"], sl.MAX_PERIODS)
            self.assertAlmostEqual(a["peak_S"], true_peak_S(kp, ki), delta=0.03)
            self.assertLess(a["fit"]["median_rel_resid"], 0.05)

    def test_no_burst_is_refused(self):
        stream, _ = simulate_stream(0.4, 900.0, np.random.default_rng(3), burst=False, post_updates=0)
        with self.assertRaises(ValueError):
            sl.analyse(stream[:200_000], AMP, HOLD, PLANT["slope"], 0.4, 900.0)


class StreamFake(Fake):
    """The calibration tests' Fake, serving the simulated stream at 1 MSPS (SAMPLE_RATE patched to match)."""
    RATE = 1_000_000

    def __init__(self, clock, const, drop=False, **kw):
        super().__init__(clock, **kw)
        self.const, self.drop = const, drop
        self.buffer_capacity = 1_000_000
        self._ring = np.full(self.buffer_capacity, 6000, dtype=np.uint16)
        self._filled = self.samples_received
        self._burst = None

    @property
    def samples_received(self):
        return 50_000_000 + int((self.clock() - self.t_start) * self.RATE)

    @property
    def buffer(self):
        if not hasattr(self, "_ring"):           # Fake.__init__ sizes its ring before ours exists
            return np.zeros(self.RATE, dtype=np.uint16)
        now = self.samples_received
        if now > self._filled:
            idx = np.arange(max(self._filled, now - self.buffer_capacity), now)
            values = np.full(len(idx), 6000.0)
            if self._burst is not None:
                offset = idx - self._burst[1]
                inside = (offset >= 0) & (offset < len(self._burst[0]))
                values[inside] = self._burst[0][offset[inside]]
            self._ring[idx % self.buffer_capacity] = np.round(values).astype(np.uint16)
            self._filled = now
        return self._ring

    @buffer.setter
    def buffer(self, _value):
        pass  # Fake.__init__ assigns a static noise buffer; this one is filled from the simulated stream

    def control_638(self, action, timeout_s=1):
        if action != "identify":
            return super().control_638(action, timeout_s)
        if self.clock() < self.busy_until:             # still streaming the previous trace's rows
            self.calls.append("identify_busy")
            raise SlaveCommandRejected(638, 3, 10, MANUAL_SET)
        self.calls.append(action)
        self.busy_until = self.clock() + 2 + 57
        pi = self.pi
        stream, start = simulate_stream(pi["kp"], pi["ki_per_s"], self.rng, pre_updates=500)
        # place the burst so its first update is ~2 ms after the command, like the 638
        self._burst = (stream, self.samples_received + 2000 - start)
        if self.drop:
            self.stream_stats["drops_seq"] += 1


def run_check(fake, clock, **kwargs):
    events = []
    fake_time = SimpleNamespace(monotonic=clock, sleep=clock.sleep, time=time.time)
    with TemporaryDirectory() as tmp, patch.object(lock_calibration, "time", fake_time), \
            patch.object(lock_check, "time", fake_time), patch.object(lock_check, "SAMPLE_RATE", StreamFake.RATE):
        report = check_lock_638(fake, Path(tmp) / "check", progress=events.append,
                                onset_after_sample_us=fake.const, **kwargs)
        files = sorted(p.name for p in (Path(tmp) / "check").iterdir())
        saved = json.loads((Path(tmp) / "check" / "lock_check.json").read_text())
    return report, events, files, saved


class CheckLockTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.const = onset_constant(np.random.default_rng(10))
        f, P = sampled_plant()
        band = (f > 60) & (f < 3000)
        cls.params = lm.fit({"f": f, "H": P, "relerr": np.where(band, 0.01, np.inf), "fs": 1e4})["params"]
        cls.design = lm.migo([cls.params], 1.4, kp_limit=2.0, ki_limit=16383.0, contract=2)

    def fake(self, kp, ki, **kw):
        clock = Clock()
        fake = StreamFake(clock, self.const, previous=dict(kp=kp, ki_per_s=ki, zero_hz=0), **kw)
        return fake, clock

    def test_well_tuned_gains_are_ok_and_nothing_is_changed(self):
        kp, ki = self.design
        fake, clock = self.fake(kp, ki)
        report, events, files, saved = run_check(fake, clock)
        self.assertEqual(report["status"], "ok", report["message"])
        self.assertEqual(set(lock_check.REPORT_KEYS) - set(report), set())
        self.assertEqual(report["gains"], {"kp": kp, "ki_per_s": ki, "zero_hz": 0})
        self.assertLessEqual(report["measured_peak_S"], 1.5)
        self.assertAlmostEqual(report["recommended"]["ki_per_s"], ki, delta=0.15 * ki)
        self.assertFalse([c for c in fake.calls if isinstance(c, tuple) and c[0] == "set_pi"])
        configures = [c for c in fake.calls if isinstance(c, tuple) and c[0] == "configure"]
        self.assertEqual(configures, [("configure", 24, 4, 4)])          # left configured, never restored
        self.assertEqual(fake.calls.count("identify"), 1)
        self.assertEqual(report["trace_config"]["before"]["amplitude_dac"], 2)
        self.assertEqual([e["stage"] for e in events if e["stage"] != events[0]["stage"]][:1], ["arming"])
        self.assertEqual(files, ["lock_check.json", "lock_check.npz"])
        self.assertEqual(saved["status"], "ok")
        self.assertLess(report["duration_s"], 10)

    def test_a_second_check_reuses_the_configuration(self):
        fake, clock = self.fake(*self.design)
        run_check(fake, clock)
        report, *_ = run_check(fake, clock)          # the 638 is still streaming the first trace (BUSY)
        self.assertEqual(report["status"], "ok", report["message"])
        configures = [c for c in fake.calls if isinstance(c, tuple) and c[0] == "configure"]
        self.assertEqual(len(configures), 1)          # the configuration left by the first check is reused
        self.assertEqual(fake.calls.count("identify"), 2)
        self.assertGreater(fake.calls.count("identify_busy"), 10)
        self.assertGreater(report["details"]["busy_wait_s"], 30)

    def test_hot_gains_ask_for_a_rerun(self):
        kp, ki = lm.migo([self.params], 1.9, kp_limit=2.0, ki_limit=16383.0, contract=2)
        fake, clock = self.fake(kp, ki)
        report, *_ = run_check(fake, clock)
        self.assertEqual(report["status"], "rerun", report["message"])
        self.assertTrue(report["reasons"])
        self.assertLess(report["recommended"]["ki_per_s"], ki)

    def test_slow_gains_are_improvable(self):
        kp, ki = self.design
        fake, clock = self.fake(kp, round(ki / 2))
        report, *_ = run_check(fake, clock)
        self.assertEqual(report["status"], "improvable", report["message"])
        self.assertGreater(report["recommended"]["ki_per_s"], 1.3 * round(ki / 2))

    def test_stream_drop_is_inconclusive(self):
        kp, ki = self.design
        fake, clock = self.fake(kp, ki, drop=True)
        report, *_ = run_check(fake, clock)
        self.assertEqual(report["status"], "inconclusive")
        self.assertIn("stream counters", report["message"])

    def test_not_locked_fails_before_the_burst(self):
        fake, clock = self.fake(*self.design)
        fake.state = 2
        report, *_ = run_check(fake, clock)
        self.assertEqual(report["status"], "failed")
        self.assertNotIn("identify", fake.calls)

    def test_cancel_during_the_burst(self):
        fake, clock = self.fake(*self.design)
        report, *_ = run_check(fake, clock, cancel=lambda: "identify" in fake.calls)
        self.assertEqual(report["status"], "cancelled")


if __name__ == "__main__":
    unittest.main()
