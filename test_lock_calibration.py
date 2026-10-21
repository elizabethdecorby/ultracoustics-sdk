"""No-hardware tests for lock calibration: model parity with unitcal, gap tolerance, and
calibrate_lock_638 / apply_saved_lock_gains_638 on a simulated closed loop.

Run: /Users/ayden/Ultracoustics/.venv-638/bin/python -I -c "import sys,unittest; sys.path.insert(0,'.'); unittest.main(module='test_lock_calibration')"
"""
import dataclasses
import glob
import json
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ultracoustics import apply_saved_lock_gains_638, calibrate_lock_638, lock_calibration, pi_characterization
from ultracoustics._internal import lock_model as lm
from ultracoustics._internal.control import MANUAL_SET
from ultracoustics.system_control import SlaveCommandRejected, SystemControlMixin as SDK

RUN = Path(__file__).resolve().parents[2] / "runs/20261020-UNIT-CALIBRATION"
TRUE = dict(g=0.8, tau=0.42e-3, delay_samples=1, driver_hz=1200.0, slope=-18.0)
API_KEYS = {"schema", "status", "message", "box_serial", "contract", "firmware_limits", "previous_gains", "gains",
            "capped_by_firmware", "ms_limit", "predicted_worst_peak", "fit", "residual_before_adc",
            "residual_after_adc", "relock", "check", "verified", "applied", "duration_s", "report_dir"}
STAGES = ["checking", "settling", "residual_before", "measuring", "designing", "applying", "residual_after",
          "relock_test", "restoring_trace"]


def prbs9(hold, n):
    state, out = 0x1FF, []
    while len(out) < n:
        new = ((state >> 8) ^ (state >> 4)) & 1
        state = ((state << 1) | new) & 0x1FF
        out.extend([1 if new else -1] * hold)
    return np.asarray(out[:n], dtype=float)


def simulate(kp, ki, amplitude, hold, decimation, rng, noise=3.0, rows=4096, zero_hz=0):
    """10 kHz closed loop with the LockedPi law; recorder rows like the 638 (RUN/test_collect_unit.py)."""
    n = rows * decimation + 4 * (decimation - 1) + 2 * 511 * hold
    r = np.round(amplitude * prbs9(hold, n))
    a_d = np.exp(-2 * np.pi * TRUE["driver_hz"] * lm.DT)
    a_t = np.exp(-lm.DT / TRUE["tau"])
    k, y0, x0 = TRUE["slope"], 6000.0, 20000.0
    xd = xt = 0.0
    integ, wz, xz = x0, 2 * np.pi * zero_hz, 0.0
    walk = np.cumsum(rng.normal(0, 0.05, n))
    hist = [0.0] * (TRUE["delay_samples"] + 1)
    ys, us = np.empty(n), np.empty(n)
    for i in range(n):
        y = y0 + k * (TRUE["g"] * hist[0] + walk[i]) + rng.normal(0, noise)
        e = (y0 - y) / k
        if wz:
            xz += e * lm.DT
            e = e + wz * xz
        integ += ki * e * lm.DT
        cmd = integ + kp * e + r[i]
        ys[i], us[i] = y, cmd
        xd = a_d * xd + (1 - a_d) * (cmd - x0)
        xt = a_t * xt + (1 - a_t) * xd
        hist = hist[1:] + [xt]
    cycles = np.cumsum(np.full(n, 14400.0)) % 2**32

    def rec(x):
        for _ in range(4):
            if decimation > 1:
                x = np.convolve(x, np.ones(decimation) / decimation)[: len(x)]
        return x[4 * (decimation - 1):][::decimation][:rows]

    start = 2 * 511 * hold
    out = np.column_stack([cycles[start:][4 * (decimation - 1):][::decimation][:rows] if decimation > 1 else rec(cycles[start:]),
                           rec(ys[start:]), rec(us[start:]), rec(r[start:]), np.ones(rows)])
    return np.round(out).astype(np.int64)


def drop_mask(n, rng, fraction=0.01, max_run=8, runs=True):
    """Present-row mask with ~fraction missing: isolated single rows, or runs of 1..max_run."""
    present = np.ones(n, dtype=bool)
    target = int(np.ceil(fraction * n))
    while (~present).sum() < target:
        length = int(rng.integers(1, max_run + 1)) if runs else 1
        length = min(length, target - int((~present).sum()))
        start = int(rng.integers(1, n - length - 1))
        if not present[start - 1:start + length + 1].all():
            continue  # keep runs separate so no gap exceeds max_run
        present[start:start + length] = False
    return present


# ---------------------------------------------------------------- model parity and gap tolerance
class LockModelTests(unittest.TestCase):
    @unittest.skipUnless((RUN / "unitcal.py").exists(), "run folder with unitcal.py not present")
    def test_identical_to_unitcal_on_recorded_traces(self):
        sys.path.insert(0, str(RUN))
        try:
            import unitcal as u
        finally:
            sys.path.remove(str(RUN))
        files = sorted(glob.glob(str(RUN / "data/*/*/trace_defaults_hi.npz"))
                       + glob.glob(str(RUN / "data_v2/0056/*/trace_*_hi.npz")))
        files = [f for f in files if Path(f).with_suffix(".json").exists()]
        self.assertGreaterEqual(len(files), 20)
        for f in files:
            rows = np.load(f)["rows"]
            slope = json.loads(Path(f).with_suffix(".json").read_text())["live_before"]["slope"]
            a = u.frf(rows, 4, 4, slope)
            for present in (None, np.ones(len(rows), dtype=bool)):
                b = lm.frf(rows, 4, 4, slope, present)
                for key in ("f", "P", "H", "S", "relerr", "fs", "periods"):
                    self.assertTrue(np.array_equal(a[key], b[key], equal_nan=True), (f, key))
            fa, fb = u.fit(a), lm.fit(b)
            self.assertEqual(fa, fb, f)
            p = fa["params"]
            self.assertEqual(u.consistency(a, 0.483, 449.8), lm.consistency(b, 0.483, 449.8), f)
            for contract, kp_limit in ((1, 1.0), (2, 2.0)):
                da = u.migo([p], 1.4, kp_limit=kp_limit, contract=contract)
                db = lm.migo([p], 1.4, kp_limit=kp_limit, contract=contract)
                self.assertEqual(da, db, (f, contract))
                self.assertEqual(u.predict(p, *da), lm.predict(p, *db), f)
                self.assertEqual(u.compare_measured(a, p, *da), lm.compare_measured(b, p, *db), f)

    def test_masked_frf_tolerates_one_percent_missing_rows(self):
        rng = np.random.default_rng(5)
        rows = simulate(0.483, 449.8, 24, 4, 4, rng)
        full = lm.frf(rows, 4, 4, TRUE["slope"])
        ref = lm.fit(full)
        ref_design = lm.migo([ref["params"]], 1.4, kp_limit=2.0, contract=2)
        self.assertLess(abs(ref["g"] - TRUE["g"]), 0.08)
        for runs in (False, True):
            for seed in range(3):
                present = drop_mask(len(rows), np.random.default_rng(seed), runs=runs)
                self.assertLessEqual(lm.max_gap(present), 8)
                garbage = rows.copy()
                garbage[~present] = 123456  # missing rows carry no information
                fr = lm.frf(garbage, 4, 4, TRUE["slope"], present)
                ft = lm.fit(fr)
                for k, tol in (("g", 0.02), ("tau_ms", 0.05), ("driver_pole_hz", 0.1)):
                    self.assertLess(abs(ft[k] / ref[k] - 1), tol, (runs, seed, k))
                self.assertLess(abs(ft["delay_us"] - ref["delay_us"]), 5)
                kp, ki = lm.migo([ft["params"]], 1.4, kp_limit=2.0, contract=2)
                self.assertLessEqual(abs(kp - ref_design[0]), 0.025 + 1e-9)  # within one grid step
                self.assertLessEqual(abs(ki - ref_design[1]), 13)

    def test_trace_rows_mask_and_gap(self):
        trace = {"total_samples": 6, "samples": {0: dict(cycles=1, feedback=2, actual_dac=3, injection_dac=4, flags=1),
                                                 3: dict(cycles=5, feedback=6, actual_dac=7, injection_dac=8, flags=1)}}
        rows, present = lm.trace_rows(trace)
        self.assertEqual(present.tolist(), [True, False, False, True, False, False])
        self.assertEqual(rows[3].tolist(), [5, 6, 7, 8, 1])
        self.assertEqual(lm.max_gap(present), 2)


# ---------------------------------------------------------------- fake controller
@dataclasses.dataclass(frozen=True)
class Live:
    sequence: int
    state: int
    dac: int
    feedback: int
    target: int
    slope: float
    tick_ms: int = 0


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += max(s, 0.001)


class Fake:
    """Minimal ultracoustics.Controller already LOCKED in automatic RUN (RUN/test_collect_unit.py Fake)."""
    TRACE_COLUMNS = ("cycles", "feedback", "actual_dac", "injection_dac", "flags")

    def __init__(self, clock, contract=2, previous=None, drop_rows=25, corrupt_capture=False, config_busy_s=0,
                 stored=None):
        self.clock, self.contract, self.device_serial = clock, contract, "0056004B3033511438323634"
        self.stored = stored  # on-box record (STORAGE_SPEC.md): the box defaults instead of the compiled profile
        self.defaults = dict(stored or dict(kp=0.483, ki_per_s=449.8, zero_hz=0))
        self.pi = dict(active=True, overridden=False, slope_valid=True, **self.defaults)
        if previous:
            self.pi.update(overridden=True, **previous)
        self.state, self.seq, self.relock_t, self.pending_unlock_t = 3, 0, None, None
        self.acq = SimpleNamespace(page=dict(event_id=1, baseline=9000, minimum=4000, target=5500, result=1,
                                             end_tick_ms=int((clock() - 30) * 1000)), stream_epoch=7)
        self.trace_cfg = dict(amplitude_dac=2, hold_updates=1, decimation=1, packed=0)
        self.trace = None
        self.capture_id, self.calls, self.rng = 0, [], np.random.default_rng(2)
        self.drop_rows, self.corrupt_capture = drop_rows, corrupt_capture
        self.busy_until = clock() + config_busy_s
        self.stream_stats = {"stream_format": 2, "drops_seq": 0, "drops_fw": 0, "transfer_errors": 0,
                             "transfer_timeouts": 0, "malformed": 0}
        self.buffer = np.random.default_rng(3).normal(6000, 4, 4_000_000).astype(np.uint16)
        self.buffer_capacity = len(self.buffer)  # ring smaller than one 2 s residual record
        self.t_start = clock()

    _running = True
    system_manual_active = False

    @property
    def samples_received(self):
        return 50_000_000 + int((self.clock() - self.t_start) * 1e7)

    def _tick(self):
        now = self.clock()
        if self.pending_unlock_t is not None and now >= self.pending_unlock_t:
            self.pending_unlock_t, self.state = None, 2
        if self.relock_t is not None and now >= self.relock_t:
            self.relock_t, self.state = None, 3
            old = self.acq.page
            self.acq = SimpleNamespace(page=dict(old, event_id=old["event_id"] + 1, end_tick_ms=int(now * 1000)),
                                       stream_epoch=7)
        self.seq += 1

    @property
    def telemetry(self):
        self._tick()
        page = Live(self.seq, self.state, 20000, 5500, 5500, -18.0, int(self.clock() * 1000))
        live = SimpleNamespace(page=page, host_stale=False, stream_epoch=7)
        return SimpleNamespace(stream_epoch=7, host_stale=False, optical_live_638=live, optical_acquisition_638=self.acq)

    def retained_control_trace(self):
        """Rows replay over a 57 s page pass; dropped rows only return 50 s later (0056 behaviour)."""
        if self.trace is None:
            return None
        t = self.clock()
        samples = {i: self.trace["rows"][i] for i in np.flatnonzero(self.trace["arrival"] <= t)}
        if not samples:
            return self.previous_trace
        return dict(self.trace["meta"], samples=samples, complete=len(samples) == 4096,
                    missing_indices=tuple(i for i in range(4096) if i not in samples))

    previous_trace = None

    def read_profile_638(self, timeout_s=1): return {"id": 1, "name": "638-belycomm-v1"}
    def read_optical_cap_638(self, timeout_s=1): return {"max_dac": 44000}
    def read_controller_timing_638(self, timeout_s=1): return {"divider": 1, "held": False, "raw": 256}
    def read_control_trace_config_638(self, timeout_s=1): return dict(self.trace_cfg)

    _normalized_pi_command_638 = SDK._normalized_pi_command_638
    _normalized_pi_info_638 = SDK._normalized_pi_info_638
    read_normalized_pi_638 = SDK.read_normalized_pi_638

    def set_normalized_pi_638(self, kp, ki, zero_hz=0, timeout_s=1):
        self.calls.append(("set_pi", kp, ki, zero_hz))
        return SDK.set_normalized_pi_638(self, kp, ki, zero_hz=zero_hz, timeout_s=timeout_s)

    def restore_normalized_pi_638(self, timeout_s=1):
        self.calls.append("restore_pi")
        return SDK.restore_normalized_pi_638(self, timeout_s=timeout_s)

    def manual_command(self, target, opcode, channel, value=0, timeout_s=1):
        assert target == 638 and channel in (10, 11)
        pi, status, v2 = self.pi, 0, self.contract == 2
        self.calls.append(("wire", opcode, channel, value))
        if opcode == MANUAL_SET and channel == 11:
            pi.update(overridden=False, **self.defaults)
        elif opcode == MANUAL_SET and channel == 10:
            if v2 and value >= 0 and value >> 30 == 1:
                pi.update(overridden=True, kp=((value >> 14) & 1023) * 0.005, ki_per_s=value & 0x3fff,
                          zero_hz=(value >> 24) & 63)
            elif not v2 and 0 <= value < 1 << 24 and value >> 14 <= 1000:
                pi.update(overridden=True, kp=(value >> 14) / 1000, ki_per_s=(value & 0x3fff) / 10)
            else:
                status = 6
        if channel == 11:
            raw = self.contract | pi["active"] << 8 | pi["overridden"] << 9 | pi["slope_valid"] << 10
            raw |= pi["zero_hz"] << 11 if v2 else 0
            raw |= 1 << 17 | 1 << 18 if self.stored else 0  # storage supported, stored gains active
        else:
            raw = (round(pi["kp"] / 0.005) << 14 | round(pi["ki_per_s"]) if v2
                   else round(pi["kp"] * 1000) << 14 | round(pi["ki_per_s"] * 10))
            raw -= (1 << 24) if raw & (1 << 23) else 0
        return SimpleNamespace(target=638, status=status, channel=channel, opcode=opcode, owner=0, applied_value=raw)

    def configure_control_trace_638(self, a, h, d, timeout_s=1):
        self.calls.append(("configure", a, h, d))
        if self.clock() < self.busy_until:
            raise SlaveCommandRejected(638, 13, 10, MANUAL_SET)
        self.trace_cfg = dict(amplitude_dac=a, hold_updates=h, decimation=d, packed=a + 1000 * h + 10 ** 6 * d)
        return dict(self.trace_cfg)

    def control_638(self, action, timeout_s=1):
        self.calls.append(action)
        now = self.clock()
        if action == "identify":
            c = self.trace_cfg
            rows = simulate(self.pi["kp"], self.pi["ki_per_s"], c["amplitude_dac"], c["hold_updates"],
                            c["decimation"], self.rng, zero_hz=self.pi["zero_hz"])
            self.capture_id += 1
            arrival = now + 2 + 57 * np.arange(4096) / 4096
            dropped = drop_mask(4096, self.rng, fraction=self.drop_rows / 4096) if self.drop_rows else np.ones(4096, bool)
            arrival[~dropped] = now + 2 + 57 + 50
            self.trace = {"rows": {i: dict(zip(self.TRACE_COLUMNS, map(int, row))) for i, row in enumerate(rows)},
                          "arrival": arrival,
                          "meta": dict(stream_epoch=7, capture_id=self.capture_id, total_samples=4096, flags=1,
                                       conflict=False, amplitude_dac=c["amplitude_dac"], start_tick_ms=int(now * 1000),
                                       hold_updates=c["hold_updates"], decimation=c["decimation"])}
            self.busy_until = now + 2 + 57 * (1 if self.contract == 2 else 2)  # v2 releases BUSY after one pass
            if self.corrupt_capture:
                self.stream_stats["drops_seq"] += 1
        elif action == "reacquire":
            self.relock_t, self.pending_unlock_t = now + 0.8, now + 0.2


def run(fake, clock, **kwargs):
    events = []
    fake_time = SimpleNamespace(monotonic=clock, sleep=clock.sleep, time=time.time)
    with TemporaryDirectory() as tmp, patch.object(lock_calibration, "time", fake_time), \
            patch.object(pi_characterization, "time", fake_time):  # trace-restore BUSY waits
        report = calibrate_lock_638(fake, Path(tmp) / "cal", progress=events.append, **kwargs)
        files = sorted(p.name for p in (Path(tmp) / "cal").iterdir())
        saved = json.loads((Path(tmp) / "cal" / "lock_calibration.json").read_text())
    return report, events, files, saved


def set_calls(fake):
    return [c for c in fake.calls if isinstance(c, tuple) and c[0] == "set_pi"]


class CalibrateLockTests(unittest.TestCase):
    def test_success_on_contract_2(self):
        clock = Clock()
        fake = Fake(clock, contract=2)
        report, events, files, saved = run(fake, clock)
        self.assertEqual(report["status"], "complete", report["message"])
        self.assertTrue(API_KEYS <= set(report))
        self.assertEqual(saved, report)
        json.dumps(report, allow_nan=False)
        self.assertTrue(report["verified"] and report["applied"], report["message"])
        g = report["gains"]
        now = fake.read_normalized_pi_638()
        self.assertEqual((now["kp"], now["ki_per_s"], now["zero_hz"], now["overridden"]), (g["kp"], g["ki_per_s"], 0, True))
        self.assertEqual(g["ki_per_s"], int(g["ki_per_s"]))
        self.assertGreater(g["ki_per_s"], lm.KI_LIMIT)  # past the v1 cap for this plant
        self.assertLessEqual(g["kp"], lock_calibration.KP_CAP)
        self.assertLessEqual(report["predicted_worst_peak"], 1.4 + 1e-9)
        self.assertFalse(report["capped_by_firmware"])
        self.assertEqual((report["contract"], report["firmware_limits"]), (2, {"kp": 5.115, "ki_per_s": 16383.0}))
        self.assertEqual(report["previous_gains"], {"kp": 0.485, "ki_per_s": 450.0, "zero_hz": 0, "overridden": False})
        self.assertLess(abs(report["fit"]["g"] - TRUE["g"]), 0.1)
        self.assertEqual(set(report["fit"]), {"g", "tau_ms", "delay_us", "driver_pole_hz", "median_rel_resid", "bins"})
        self.assertTrue(report["relock"]["relocked"] and report["relock"]["override_kept"])
        self.assertLess(report["relock"]["seconds"], 2)
        self.assertGreater(report["residual_before_adc"], 0)
        self.assertLessEqual(report["residual_after_adc"], 1.25 * report["residual_before_adc"])
        self.assertIsNone(report["check"])
        trace = report["details"]["trace"]
        self.assertTrue(trace["valid"] and not trace["complete"] and trace["max_gap_rows"] <= 8)
        self.assertGreaterEqual(trace["rows_present"], 0.99 * 4096)
        self.assertLess(trace["readout_s"], 70)  # did not wait for the replay of the dropped rows
        self.assertEqual(("configure", 24, 4, 4), next(c for c in fake.calls if c[0] == "configure"))
        self.assertEqual(fake.trace_cfg["hold_updates"], 1)  # original trace configuration restored
        self.assertEqual(fake.calls.count("reacquire"), 1)
        self.assertEqual(files, ["lock_calibration.json", "residual_after.npz", "residual_before.npz", "trace.npz"])
        stages = [e["stage"] for e in events]
        self.assertEqual(list(dict.fromkeys(stages)), STAGES)
        fractions = [e["fraction"] for e in events]
        self.assertEqual(fractions, sorted(fractions))
        self.assertEqual(fractions[-1], 1.0)
        self.assertTrue(all({"stage", "message", "elapsed_s", "fraction"} <= set(e) for e in events))
        self.assertLess(report["duration_s"], 110)

    def test_full_check_adds_second_trace(self):
        clock = Clock()
        fake = Fake(clock, contract=2)
        report, events, files, _ = run(fake, clock, full_check=True)
        self.assertEqual(report["status"], "complete", report["message"])
        self.assertTrue(report["check"]["ok"] and report["verified"], report["check"])
        self.assertLess(report["check"]["median_abs_error"], 0.05)
        self.assertIn("checking_design", [e["stage"] for e in events])
        self.assertIn("trace_check.npz", files)
        self.assertEqual(fake.calls.count("identify"), 2)

    def test_cancel_mid_measurement_leaves_gains_and_restores_trace(self):
        clock = Clock()
        fake = Fake(clock, contract=2, previous={"kp": 0.3, "ki_per_s": 900})
        started = {}

        def cancel():
            if "identify" in fake.calls:
                started.setdefault("t", clock())
                return clock() - started["t"] > 10
            return False

        report, events, _, _ = run(fake, clock, cancel=cancel)
        self.assertEqual(report["status"], "cancelled")
        self.assertEqual(set_calls(fake), [])
        self.assertEqual((fake.pi["kp"], fake.pi["ki_per_s"], fake.pi["overridden"]), (0.3, 900, True))
        self.assertEqual(fake.trace_cfg["hold_updates"], 1)
        self.assertFalse(report["applied"])
        self.assertEqual(events[-1]["stage"], "restoring_trace")

    def test_cancel_after_apply_puts_previous_override_back_exactly(self):
        clock = Clock()
        fake = Fake(clock, contract=2, previous={"kp": 0.3, "ki_per_s": 900})
        before = fake.read_normalized_pi_638()
        cancel = lambda: "reacquire" in fake.calls and fake.state == 3 and fake.relock_t is None
        report, _, _, _ = run(fake, clock, cancel=cancel)
        self.assertEqual(report["status"], "cancelled", report["message"])
        self.assertEqual(len(set_calls(fake)), 2)
        self.assertEqual(set_calls(fake)[-1], ("set_pi", 0.3, 900.0, 0))
        self.assertEqual(fake.read_normalized_pi_638(), before)
        self.assertTrue(report["details"]["previous_gains_restored"] and report["details"]["trace_config_restored"])
        self.assertFalse(report["applied"])
        self.assertEqual(fake.trace_cfg["hold_updates"], 1)

    def test_failure_after_apply_restores_profile_defaults(self):
        clock = Clock()
        fake = Fake(clock, contract=2)
        original = fake.control_638

        def slow_relock(action, timeout_s=1):
            original(action, timeout_s)
            if action == "reacquire":
                fake.relock_t = clock() + 3.0  # relocks only after the 1 s reacquire timeout

        fake.control_638 = slow_relock
        with patch.dict(lock_calibration.TIMING, reacquire_timeout_s=1.0):
            report, _, _, _ = run(fake, clock)
        self.assertEqual(report["status"], "failed")
        self.assertIn("did not relock", report["message"])
        self.assertIn("restore_pi", fake.calls)  # defaults were active at the start -> restore, not set
        self.assertFalse(fake.pi["overridden"])
        self.assertTrue(report["details"]["previous_gains_restored"])
        self.assertFalse(report["relock"]["relocked"])

    def test_failure_after_apply_restores_stored_box_defaults(self):
        clock = Clock()
        stored = {"kp": 0.6, "ki_per_s": 1925, "zero_hz": 20}  # box defaults from an on-box record
        fake = Fake(clock, contract=2, stored=stored)
        original = fake.control_638

        def slow_relock(action, timeout_s=1):
            original(action, timeout_s)
            if action == "reacquire":
                fake.relock_t = clock() + 3.0

        fake.control_638 = slow_relock
        with patch.dict(lock_calibration.TIMING, reacquire_timeout_s=1.0):
            report, _, _, _ = run(fake, clock)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["previous_gains"], {"kp": 0.6, "ki_per_s": 1925.0, "zero_hz": 20, "overridden": False})
        self.assertIn("restore_pi", fake.calls)
        self.assertTrue(report["details"]["previous_gains_restored"], report["message"])
        rb = fake.read_normalized_pi_638()
        self.assertEqual((rb["kp"], rb["ki_per_s"], rb["zero_hz"], rb["overridden"], rb["stored_active"]),
                         (0.6, 1925.0, 20, False, True))

    def test_gate_rejection_leaves_gains(self):
        clock = Clock()
        fake = Fake(clock, contract=2, corrupt_capture=True)
        report, events, _, _ = run(fake, clock)
        self.assertEqual(report["status"], "rejected")
        self.assertIn("fit gate", report["message"])
        self.assertFalse(report["details"]["gate"]["passed"] or report["details"]["gate"]["valid_capture"])
        self.assertEqual(set_calls(fake), [])
        self.assertNotIn("restore_pi", fake.calls)
        self.assertIsNone(report["gains"])
        self.assertFalse(report["verified"] or report["applied"])
        self.assertEqual(fake.trace_cfg["hold_updates"], 1)
        self.assertNotIn("applying", [e["stage"] for e in events])

    def test_contract_1_uses_12_dac_and_v1_caps(self):
        clock = Clock()
        fake = Fake(clock, contract=1, config_busy_s=25)  # a previous trace still replaying when measuring starts
        report, _, _, _ = run(fake, clock)
        self.assertEqual(report["details"]["busy_waits"][0], {"stage": "measuring", "channel": 13})
        self.assertEqual(report["status"], "complete", report["message"])
        self.assertEqual((report["contract"], report["firmware_limits"]), (1, {"kp": 1.0, "ki_per_s": 1638.3}))
        configs = [c for c in fake.calls if isinstance(c, tuple) and c[0] == "configure"]
        self.assertEqual(configs[0], ("configure", 12, 4, 4))
        self.assertGreater(len(configs), 3)  # waited out BUSY before the capture and before the restore
        g = report["gains"]
        self.assertLessEqual(g["kp"], 1.0)
        self.assertLessEqual(g["ki_per_s"], 1638.3)
        self.assertTrue(report["capped_by_firmware"])  # this plant wants more Ki than v1 allows
        words = [c[3] for c in fake.calls if isinstance(c, tuple) and c[:3] == ("wire", MANUAL_SET, 10)]
        self.assertTrue(words and all(w < 1 << 24 for w in words))
        self.assertTrue(report["applied"] and fake.pi["overridden"])

    def test_refuses_before_hardware_when_not_locked_or_bad_arguments(self):
        clock = Clock()
        fake = Fake(clock, contract=1)
        report, _, _, _ = run(fake, clock, amplitude_dac=24)  # > 12 on contract 1
        self.assertEqual(report["status"], "failed")
        self.assertIn("amplitude_dac", report["message"])
        self.assertNotIn("identify", fake.calls)
        fake = Fake(clock, contract=2)
        fake.state = 2
        report, _, _, _ = run(fake, clock)
        self.assertEqual(report["status"], "failed")
        self.assertFalse(any(isinstance(c, tuple) and c[0] in ("configure", "set_pi") for c in fake.calls))


class ApplySavedGainsTests(unittest.TestCase):
    def assert_refused(self, fake, gains, text):
        with self.assertRaises(ValueError) as caught:
            apply_saved_lock_gains_638(fake, gains)
        self.assertIn(text, str(caught.exception))
        self.assertFalse([c for c in fake.calls if isinstance(c, tuple) and (c[0] == "set_pi" or c[:2] == ("wire", MANUAL_SET))])

    def test_out_of_contract_gains_send_nothing(self):
        clock = Clock()
        v1 = Fake(clock, contract=1)
        self.assert_refused(v1, {"kp": 0.6, "ki_per_s": 2000, "zero_hz": 0}, "firmware contract v1 caps Ki at 1638.3")
        self.assert_refused(v1, {"kp": 1.2, "ki_per_s": 1000, "zero_hz": 0}, "caps Kp at 1")
        self.assert_refused(v1, {"kp": 0.5, "ki_per_s": 1000, "zero_hz": 20}, "no second integrator")
        v2 = Fake(clock, contract=2)
        self.assert_refused(v2, {"kp": 6.0, "ki_per_s": 1000, "zero_hz": 0}, "caps Kp at 5.115")
        self.assert_refused(v2, {"kp": 0.5, "ki_per_s": 20000, "zero_hz": 0}, "caps Ki at 16383")
        self.assert_refused(v2, {"kp": 0.525, "ki_per_s": 1637.5, "zero_hz": 0}, "1/s steps")
        self.assert_refused(v2, {"kp": float("nan"), "ki_per_s": 1000, "zero_hz": 0}, "finite")

    def test_in_contract_gains_are_set_and_read_back(self):
        clock = Clock()
        v2 = Fake(clock, contract=2)
        rb = apply_saved_lock_gains_638(v2, {"kp": np.float64(0.625), "ki_per_s": 2037.0, "zero_hz": 0})
        self.assertEqual((rb["kp"], rb["ki_per_s"], rb["overridden"]), (0.625, 2037.0, True))
        v1 = Fake(clock, contract=1)
        rb = apply_saved_lock_gains_638(v1, {"kp": 0.525, "ki_per_s": 1637.5, "zero_hz": 0, "overridden": True})
        self.assertEqual((rb["kp"], rb["ki_per_s"]), (0.525, 1637.5))


if __name__ == "__main__":
    unittest.main()
