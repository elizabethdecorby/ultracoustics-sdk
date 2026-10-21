"""Per-probe 638 lock calibration on a live automatic RUN session.

One D4/hold-4 PRBS trace at the current gains -> plant fit -> the largest Ki whose
worst-case peak |S| <= ms_limit over slope error 0.7..1.4 -> applied as a runtime
override -> one reacquire plus 2 s in-lock residual records before and after.
Fit and design are runs/20261020-UNIT-CALIBRATION/unitcal.py (``_internal.lock_model``).
Never STARTs or STOPs. No other thread may issue Controller commands during a call.
Contract: runs/20261020-UNIT-CALIBRATION/LOCK_CALIBRATION_API.md.
"""
from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np

from ._internal import lock_model as lm
from ._internal.control import MANUAL_SET
from .pi_characterization import COUNTERS, TRACE_CONFIG_FIELDS, _Cancelled, _plain, _restore_trace_config
from .system_control import CHANNEL_OPTICAL, CHANNEL_TRACE_CONFIG, SlaveCommandRejected

LOCKED, ERROR, MEASURE_SLOPE = 3, 4, 5
HOLD, DECIMATION = 4, 4                # the D4 band the fit was validated on
INJECTION_DAC = {1: 12, 2: 24}         # contract -> default PRBS amplitude
MAX_INJECTION_DAC = {1: 12, 2: 32}     # v1 slews against the injected DAC, v2 against the PI command
KP_CAP = 2.0                           # design search cap (the contract 2 wire allows 5.115)
MS_RANGE = (1.2, 2.0)                  # hardware-validated design targets
ROWS_FRACTION, MAX_GAP_ROWS = 0.99, 8  # gap-tolerant trace readout
GATE = dict(min_bins=60, max_consistency_error=0.08, max_fit_resid=0.15)  # collect_unit.design
CHECK = dict(min_bins=40, max_error=0.08, peak_margin=0.1)               # calibrate_quick check capture
# 2 s records swing with the room, so before/after totals are reported, not gated; only gross
# growth at 200-1000 Hz (where a too-hot loop rings) fails verification. Ms 2.0 on 0056/Annie: 2.1x.
RINGING_BAND, RINGING_RATIO = "200-1000", 2.5
TIMING = dict(settle_since_lock_s=20, settle_window_s=15, settle_max_s=20, residual_s=2.0,
              residual_attempts=2, guard_s=3.0, trace_timeout_s=240, busy_s=120,
              reacquire_timeout_s=45, stale_grace_s=1.5, lock_wait_s=10, trace_restore_s=120)
EXPECTED_S = dict(checking=1, settling=15, residual_before=2.5, measuring=65, designing=1,
                  applying=3, residual_after=2.5, relock_test=4, checking_design=68,
                  restoring_trace=1)  # progress fraction weights (0056, 10-21 logs)
GUARD_SAMPLES = 8192
SAMPLE_RATE = 10_000_000
SAMPLES_PER_BIN = 1000                 # residual records are 100 us means


class _Rejected(Exception):
    """The identification failed the fit gate; the gains were not changed."""


def _gains(pi):
    return {"kp": pi["kp"], "ki_per_s": pi["ki_per_s"], "zero_hz": pi.get("zero_hz", 0)}


class _Calibration:
    def __init__(self, controller, report_dir, progress, cancel, ms_limit, amplitude_dac, full_check):
        self.c, self.progress, self.cancel = controller, progress, cancel
        self.ms_limit, self.amplitude_dac, self.full_check = ms_limit, amplitude_dac, full_check
        self.out = Path(report_dir).expanduser().resolve() if report_dir is not None else None
        self.t0 = time.monotonic()
        self.stages = [s for s in EXPECTED_S if full_check or s != "checking_design"]
        self.total_s = sum(EXPECTED_S[s] for s in self.stages)
        self.stage, self.stage_t0, self.stage_seconds = None, self.t0, {}
        self.message, self.part, self.extra, self.last_emit, self.fraction = "", None, {}, -math.inf, 0.0
        self.epoch, self.cap, self.depth, self.last_trace_id = None, None, None, None
        self.previous = self.new = self.original_trace = None
        self.configured = self.changed = False
        self.r = {"schema": 1, "status": "failed", "message": "",
                  "box_serial": getattr(controller, "device_serial", None), "contract": None,
                  "firmware_limits": None, "previous_gains": None, "gains": None,
                  "capped_by_firmware": False, "ms_limit": ms_limit, "predicted_worst_peak": None,
                  "fit": None, "residual_before_adc": None, "residual_after_adc": None, "relock": None,
                  "check": None, "verified": False, "applied": False, "duration_s": 0.0,
                  "report_dir": str(self.out) if self.out else None}
        self.d = self.r["details"] = {"created_utc": datetime.now(timezone.utc).isoformat(),
                                      "stage_seconds": self.stage_seconds}

    # ---------- progress / cancel ----------
    def enter(self, stage, message):
        now = time.monotonic()
        if self.stage is not None:
            self.stage_seconds[self.stage] = round(self.stage_seconds.get(self.stage, 0) + now - self.stage_t0, 2)
        self.stage, self.stage_t0, self.part, self.extra = stage, now, None, {}
        self.emit(message)

    def emit(self, message=None, **extra):
        self.message = self.message if message is None else message
        self.last_emit = time.monotonic()
        if not self.progress:
            return
        expected = EXPECTED_S[self.stage]
        done = sum(EXPECTED_S[s] for s in self.stages[:self.stages.index(self.stage)])
        within = expected * self.part if self.part is not None else time.monotonic() - self.stage_t0
        self.fraction = max(self.fraction, min(1.0, (done + min(within, expected)) / self.total_s))
        event = {"stage": self.stage, "message": self.message, "elapsed_s": round(time.monotonic() - self.t0, 2),
                 "fraction": round(self.fraction, 3)}
        event.update(self.extra, **extra)
        self.progress(event)

    def heartbeat(self):
        if time.monotonic() - self.last_emit >= 1.0:
            self.emit()

    def check_cancel(self):
        if self.cancel and self.cancel():
            raise _Cancelled("Cancelled by user")

    # ---------- telemetry (collect_unit.Session) ----------
    def live(self):
        snap = self.c.telemetry
        if snap is None or snap.host_stale or snap.stream_epoch != self.epoch:
            return None
        entry = getattr(snap, "optical_live_638", None)
        if entry is None or entry.host_stale or entry.stream_epoch != self.epoch:
            return None
        return entry.page

    def fresh_page(self, timeout=None):
        """Newest fresh live page within a bounded grace; a single read can miss a seqlock window."""
        deadline = time.monotonic() + (TIMING["stale_grace_s"] if timeout is None else timeout)
        while True:
            page = self.live()
            if page is not None or time.monotonic() >= deadline:
                return page
            time.sleep(.025)

    def acquisition(self):
        """Latest acquisition page of this epoch; pages are re-sent, so compare event_id for freshness."""
        snap = self.c.telemetry
        acq = getattr(snap, "optical_acquisition_638", None) if snap is not None else None
        if acq is None or acq.stream_epoch != self.epoch:
            return None
        return dataclasses.asdict(acq.page) if dataclasses.is_dataclass(acq.page) else dict(acq.page)

    def current_acquisition(self):
        deadline = time.monotonic() + TIMING["stale_grace_s"]
        while True:
            acq = self.acquisition()
            if acq is not None or time.monotonic() >= deadline:
                return acq
            time.sleep(.025)

    def stats(self):
        stats = dict(self.c.stream_stats or {})
        if stats.get("fatal") or stats.get("device_lost"):
            raise RuntimeError("ADC stream ended")
        return stats

    def watch(self, seconds, *, require_lock=True, guard_limit=None):
        """Wait while checking cancel, lock freshness, the stream and optionally the oscillation guard."""
        deadline = time.monotonic() + seconds
        fresh = time.monotonic()
        next_guard, strikes = fresh + .25, 0
        while True:
            self.check_cancel()
            self.heartbeat()
            page, now = self.live(), time.monotonic()
            if page is not None:
                fresh = now
                if require_lock and page.state not in (LOCKED, MEASURE_SLOPE):
                    raise RuntimeError(f"638 left LOCKED (state {page.state})")
            elif require_lock and now - fresh > TIMING["stale_grace_s"]:
                raise RuntimeError("Lock telemetry stale beyond grace")
            self.stats()
            if guard_limit is not None and now >= next_guard:
                next_guard = now + .25
                rms = self.recent_hp_rms()
                strikes = strikes + 1 if rms is not None and rms > guard_limit else 0
                if strikes >= 2:
                    raise RuntimeError(f"Oscillation guard: high-passed error RMS {rms:.1f} ADC > {guard_limit:.1f}")
            if now >= deadline:
                return
            time.sleep(min(.025, deadline - now))

    def recent_hp_rms(self, n=2_500_000):
        c = self.c
        cap = int(c.buffer_capacity)
        head = int(c.samples_received) - GUARD_SAMPLES
        start = head - n
        if start < 0 or n > cap - 2 * GUARD_SAMPLES:
            return None
        off = start % cap
        raw = c.buffer[off:off + n] if off + n <= cap else np.concatenate((c.buffer[off:], c.buffer[:off + n - cap]))
        binned = np.asarray(raw, dtype=np.float64).reshape(-1, SAMPLES_PER_BIN).mean(axis=1)
        if int(c.samples_received) - start >= cap - GUARD_SAMPLES:
            return None  # overwritten while copying
        return lm.highpass_rms(binned)

    def guard_limit(self):
        return max(4 * (self.d.get("residual_before", {}).get("hp_rms_adc") or 0), 0.01 * self.depth)

    def busy_retry(self, send, channel, guard_limit=None):
        """Retry only the acknowledged 638 'busy' rejection (trace replay), within a bound."""
        deadline = time.monotonic() + TIMING["busy_s"]
        while True:
            try:
                return send()
            except SlaveCommandRejected as exc:
                if (exc.target, exc.channel, exc.status, exc.opcode) != (638, channel, 10, MANUAL_SET) \
                        or time.monotonic() >= deadline:
                    raise
                self.d.setdefault("busy_waits", []).append({"stage": self.stage, "channel": channel})
                self.watch(1.0, guard_limit=guard_limit)

    # ---------- steps ----------
    def check(self):
        c = self.c
        if not 0 < self.ms_limit < math.inf or not MS_RANGE[0] <= self.ms_limit <= MS_RANGE[1]:
            raise ValueError(f"ms_limit must be within {MS_RANGE[0]}..{MS_RANGE[1]}")
        if getattr(c, "system_manual_active", False) or not getattr(c, "_running", True):
            raise RuntimeError("Lock calibration requires automatic RUN, not manual or IDLE")
        deadline = time.monotonic() + 2
        while (c.stream_stats or {}).get("stream_format") != 2:
            if time.monotonic() >= deadline:
                raise RuntimeError("Optical diagnostic stream format 2 is required")
            time.sleep(.05)
        snap = c.telemetry
        self.epoch = snap.stream_epoch if snap is not None else None
        page = self.fresh_page()
        if page is None or page.state != LOCKED:
            raise RuntimeError("The 638 is not LOCKED with fresh telemetry")
        if abs(page.slope) < 1.5:
            raise RuntimeError("A valid signed slope is required")
        pi = c.read_normalized_pi_638(timeout_s=2)
        self.previous = pi
        self.r["previous_gains"] = dict(_gains(pi), overridden=pi["overridden"])
        self.d.update(pi_before=pi, profile=c.read_profile_638(timeout_s=2),
                      timing=c.read_controller_timing_638(timeout_s=2), live_before=page)
        self.cap = c.read_optical_cap_638(timeout_s=2)["max_dac"]
        self.original_trace = c.read_control_trace_config_638(timeout_s=2)
        self.d.update(optical_cap=self.cap, trace_config_before=self.original_trace)
        if not pi["active"] or not pi["slope_valid"]:
            raise RuntimeError("Normalized PI is inactive or its slope is invalid")
        contract = pi.get("version", 1)
        if contract not in lm.WIRE_LIMITS:
            raise RuntimeError(f"Unsupported normalized PI contract v{contract}")
        kp_lim, ki_lim = lm.WIRE_LIMITS[contract]
        self.r.update(contract=contract, firmware_limits={"kp": kp_lim, "ki_per_s": ki_lim})
        if self.d["timing"]["divider"] != 1 or self.d["timing"]["held"]:
            raise RuntimeError("Lock calibration requires 10 kHz unheld control")
        amp = INJECTION_DAC[contract] if self.amplitude_dac is None else self.amplitude_dac
        if type(amp) is not int or not 4 <= amp <= MAX_INJECTION_DAC[contract]:
            raise ValueError(f"amplitude_dac must be an integer 4..{MAX_INJECTION_DAC[contract]} on contract v{contract}")
        self.amp = self.d["amplitude_dac"] = amp
        if not 500 + amp < page.dac < self.cap - 500 - amp:
            raise RuntimeError(f"Insufficient DAC headroom at DAC {page.dac} (cap {self.cap})")
        acq = self.current_acquisition()
        if acq is None or acq["baseline"] <= acq["minimum"]:
            raise RuntimeError("No current acquisition event with a dip")
        self.acq, self.depth = acq, float(acq["baseline"] - acq["minimum"])
        self.d.update(acquisition=acq, dip_depth_adc=self.depth)

    def settle(self):
        """collect_unit.settle: >= 20 s since the lock, a 15 s window of drift <= 3 DAC/s, span <= 100 DAC
        and median |error| <= max(60, 1.5 % of the dip); not reached within 20 s -> continue flagged."""
        t, limit_err = TIMING, max(60.0, 0.015 * self.depth)
        hist, last_seq, t0 = [], None, time.monotonic()
        result = {"passed": False, "limits": {"since_lock_s": t["settle_since_lock_s"], "window_s": t["settle_window_s"],
                                              "drift_dac_per_s": 3, "span_dac": 100, "median_abs_error_adc": limit_err}}
        while time.monotonic() - t0 < t["settle_max_s"]:
            self.watch(.2)
            page = self.live()
            if page is None or page.state != LOCKED or page.sequence == last_seq:
                continue
            last_seq, now = page.sequence, time.monotonic()
            hist.append((now, page.dac, page.feedback - page.target))
            self.part = min(1.0, (now - t0) / t["settle_window_s"])
            since_lock = ((page.tick_ms - self.acq["end_tick_ms"]) % 2**32) / 1000
            recent = np.asarray([h for h in hist if h[0] >= now - t["settle_window_s"]])
            if (since_lock < t["settle_since_lock_s"] or len(recent) < 10
                    or recent[-1, 0] - recent[0, 0] < t["settle_window_s"] - 1):
                continue
            drift = float(np.polyfit(recent[:, 0] - recent[0, 0], recent[:, 1], 1)[0])
            result.update(elapsed_s=round(now - t0, 1), drift_dac_per_s=round(drift, 2),
                          span_dac=float(np.ptp(recent[:, 1])),
                          median_abs_error_adc=float(np.median(np.abs(recent[:, 2]))))
            if abs(drift) <= 3 and result["span_dac"] <= 100 and result["median_abs_error_adc"] <= limit_err:
                result["passed"] = True
                break
        self.d["settling"] = result

    def record(self, seconds):
        """In-lock ADC record reduced to 100 us means while it streams (works with any ring size).
        Returns (means, None) or (None, reason)."""
        c = self.c
        cap = int(c.buffer_capacity)
        before = self.stats()
        cursor = int(c.samples_received) + GUARD_SAMPLES
        end = cursor + int(round(seconds * SAMPLE_RATE))
        bins, rest, fresh = [], np.empty(0), time.monotonic()
        while True:
            self.check_cancel()
            self.heartbeat()
            page, now = self.live(), time.monotonic()
            if page is not None:
                fresh = now
                if page.state != LOCKED:
                    return None, f"638 state {page.state} during the record"
            elif now - fresh > TIMING["stale_grace_s"]:
                raise RuntimeError("Lock telemetry stale beyond grace")
            self.stats()
            head = int(c.samples_received) - GUARD_SAMPLES
            if head - cursor > cap - 2 * GUARD_SAMPLES:
                return None, "ADC ring overrun"
            upto = min(head, end)
            if upto > cursor:
                i, k = cursor % cap, upto - cursor
                raw = c.buffer[i:i + k] if i + k <= cap else np.concatenate((c.buffer[i:], c.buffer[:i + k - cap]))
                raw = np.asarray(raw, dtype=np.float64)
                if int(c.samples_received) - cursor > cap - GUARD_SAMPLES:
                    return None, "ADC ring overwritten while copying"
                rest = np.concatenate((rest, raw))
                nb = len(rest) // SAMPLES_PER_BIN
                bins.append(rest[:nb * SAMPLES_PER_BIN].reshape(nb, SAMPLES_PER_BIN).mean(axis=1))
                rest, cursor = rest[nb * SAMPLES_PER_BIN:], upto
            if cursor >= end:
                break
            time.sleep(.01)
        after = self.stats()
        deltas = {k: int(after.get(k, 0)) - int(before.get(k, 0)) for k in COUNTERS}
        if any(deltas.values()):
            return None, f"ADC/USB counters changed during the record: {deltas}"
        return np.concatenate(bins), None

    def residual(self, name):
        """Total RMS 1 Hz-4.5 kHz of one 2 s in-lock record (ab_rest.band_rms); one retry if rejected."""
        for attempt in range(1, TIMING["residual_attempts"] + 1):
            means, reason = self.record(TIMING["residual_s"])
            if means is not None:
                break
            self.d.setdefault("residual_rejections", []).append({"record": name, "attempt": attempt, "reason": reason})
        else:
            return None
        bands, total = lm.band_rms([means])
        self.d[name] = {"total_adc": total, "bands_adc": bands, "hp_rms_adc": lm.highpass_rms(means),
                        "mean_adc": float(means.mean()), "attempts": attempt}
        if self.out:
            self.d[name]["file"] = f"{name}.npz"
            np.savez_compressed(self.out / f"{name}.npz", mean_100us=means.astype(np.float32))
        return total

    def configure_trace(self, amplitude, guard_limit=None):
        wanted = (amplitude, HOLD, DECIMATION)
        self.configured = self.configured or tuple(self.original_trace[k] for k in TRACE_CONFIG_FIELDS) != wanted
        return self.busy_retry(lambda: self.c.configure_control_trace_638(*wanted, timeout_s=1.0),
                               CHANNEL_TRACE_CONFIG, guard_limit)

    def measure(self, name, guard_limit=None):
        """One PRBS-9 trace; readout stops at >= 99 % of rows with no gap > 8 rows (collect_unit.identify)."""
        c, amplitude = self.c, self.amp
        page = self.fresh_page()
        if page is None or page.state != LOCKED:
            raise RuntimeError("Not LOCKED before identification")
        if not 500 + amplitude < page.dac < self.cap - 500 - amplitude:
            raise RuntimeError(f"Insufficient DAC headroom for identification at DAC {page.dac}")
        config = self.configure_trace(amplitude, guard_limit)
        self.watch(.3, guard_limit=guard_limit)
        old = None
        for _ in range(40):  # the shared-memory read can transiently return None
            old = c.retained_control_trace()
            if old is not None:
                break
            time.sleep(.025)
        old_id = (old["stream_epoch"], old["capture_id"]) if old else self.last_trace_id
        wanted = (amplitude, HOLD, DECIMATION)
        before, stats_before = self.fresh_page(), self.stats()
        if before is None or before.state != LOCKED:
            raise RuntimeError("No fresh LOCKED page before the identification trigger")
        duration = 4096 * DECIMATION / 10_000
        self.check_cancel()
        c.control_638("identify", timeout_s=1.0)
        t0 = time.monotonic()
        post = post_stats = post_delay = None
        while True:
            if time.monotonic() - t0 >= TIMING["trace_timeout_s"]:
                raise TimeoutError(f"Trace {name}: {ROWS_FRACTION:.0%} of rows with no gap > {MAX_GAP_ROWS} "
                                   f"not received within {TIMING['trace_timeout_s']} s")
            self.watch(.5, guard_limit=guard_limit)
            if post is None and time.monotonic() - t0 >= duration + .4:
                post, post_stats, post_delay = self.fresh_page(.5), self.stats(), time.monotonic() - t0
            trace = c.retained_control_trace()
            # New = armed after the pre-trigger 638 tick, with the requested settings, not the old capture.
            if not (trace is not None and trace.get("start_tick_ms", -1) >= before.tick_ms - 100
                    and (trace.get("amplitude_dac"), trace.get("hold_updates"), trace.get("decimation")) == wanted
                    and (old_id is None or (trace["stream_epoch"], trace["capture_id"]) != old_id)):
                continue
            if trace["conflict"] or trace["flags"] & 2:
                raise RuntimeError(f"Trace {name} aborted or conflicted")
            total, got = trace["total_samples"], len(trace["samples"])
            self.part, self.extra = got / total, {"received_rows": got, "total_rows": total}
            if post is not None and trace["flags"] & 1 and got >= ROWS_FRACTION * total:
                rows, present = lm.trace_rows(trace)
                if lm.max_gap(present) <= MAX_GAP_ROWS:
                    break
        self.last_trace_id = (trace["stream_epoch"], trace["capture_id"])
        readout_s = time.monotonic() - t0
        deltas = {k: int(post_stats.get(k, 0)) - int(stats_before.get(k, 0)) for k in COUNTERS}
        pi = c.read_normalized_pi_638(timeout_s=2)
        fr = lm.frf(rows, HOLD, DECIMATION, before.slope, present)
        cons = lm.consistency(fr, pi["kp"], pi["ki_per_s"], fmin=60, zero_hz=pi.get("zero_hz", 0))
        meta = {"config": config, "amplitude_dac": amplitude, "hold": HOLD, "decimation": DECIMATION,
                "readout_s": round(readout_s, 2), "rows_present": int(present.sum()),
                "max_gap_rows": lm.max_gap(present), "complete": bool(present.all()),
                "live_before": before, "live_after": post, "post_delay_s": post_delay,
                "dac_shift": post.dac - before.dac, "capture_counter_deltas": deltas,
                "valid": not any(deltas.values()), "pi_at_capture": pi, "slope": before.slope,
                "consistency": cons,
                "trace_metadata": {k: v for k, v in trace.items() if k not in ("samples", "missing_indices")}}
        if self.out:
            meta["file"] = f"trace{'' if name == 'trace' else '_' + name}.npz"
            np.savez_compressed(self.out / meta["file"], rows=rows, present=present)
        self.d[name] = meta
        return fr, meta

    def design(self, fr, meta):
        cons, ft = meta["consistency"], None
        if cons["bins"] >= 20:  # fewer bins cannot be fitted (collect_unit.design)
            ft = lm.fit(fr)
            self.r["fit"] = {k: ft[k] for k in ("g", "tau_ms", "delay_us", "driver_pole_hz", "median_rel_resid", "bins")}
            self.d["fit_params"] = ft["params"]
        gate = (ft is not None and meta["valid"] and cons["bins"] >= GATE["min_bins"]
                and cons["median_abs_error"] is not None
                and cons["median_abs_error"] <= GATE["max_consistency_error"]
                and ft["median_rel_resid"] <= GATE["max_fit_resid"])
        self.d["gate"] = {"passed": bool(gate), "limits": GATE, "valid_capture": meta["valid"],
                          "bins": cons["bins"], "consistency_error": cons["median_abs_error"],
                          "fit_resid": ft["median_rel_resid"] if ft else None}
        if not gate:
            raise _Rejected(f"identification failed the fit gate ({self.d['gate']}); gains unchanged")
        contract = self.r["contract"]
        kp_lim, ki_lim = lm.WIRE_LIMITS[contract]
        kp, ki = lm.migo([ft["params"]], self.ms_limit, kp_limit=min(KP_CAP, kp_lim), ki_limit=ki_lim, contract=contract)
        if ki <= 0:
            raise _Rejected(f"no PI design reaches worst-case peak |S| <= {self.ms_limit}; gains unchanged")
        grid = lm.ki_grid(ki_lim)
        top = grid[grid <= ki_lim].max()
        top = float(np.floor(top)) if contract == 2 else float(top)
        predicted = lm.predict(ft["params"], kp, ki)
        self.r.update(gains={"kp": kp, "ki_per_s": ki, "zero_hz": 0}, predicted_worst_peak=predicted["worst_ms"],
                      capped_by_firmware=bool(ki >= top or (kp_lim < KP_CAP and kp >= kp_lim)))
        self.d["predicted"] = predicted
        return ft["params"], kp, ki

    def relock(self, limit):
        before = self.current_acquisition()
        prev_id = before.get("event_id") if before else None
        self.busy_retry(lambda: self.c.control_638("reacquire", timeout_s=1.0), CHANNEL_OPTICAL, limit)
        t0 = time.monotonic()
        acq = page = None
        left = ok = False
        while time.monotonic() - t0 < TIMING["reacquire_timeout_s"]:  # not cancellable: wait for the outcome
            self.heartbeat()
            page = self.live()
            if page is not None and page.state != LOCKED:
                left = True
                if page.state == ERROR:
                    break
            acq = self.acquisition()
            if (acq is not None and acq.get("event_id") != prev_id and acq.get("result") == 1
                    and page is not None and page.state == LOCKED and page.tick_ms >= acq.get("end_tick_ms", 0)):
                ok = True
                break
            self.stats()
            time.sleep(.025)
        rel = self.r["relock"] = {"relocked": ok, "seconds": round(time.monotonic() - t0, 2), "override_kept": False}
        self.d["relock"] = {"previous_event_id": prev_id, "acquisition": acq, "page": page, "left_locked_seen": left}
        if not ok:
            raise RuntimeError(f"Reacquire at the new gains did not relock within {TIMING['reacquire_timeout_s']} s")
        pi = self.d["relock"]["pi_after"] = self.c.read_normalized_pi_638(timeout_s=2)
        rel["override_kept"] = bool(pi["overridden"] and pi["packed"] == self.new["packed"]
                                    and pi.get("zero_hz", 0) == self.new.get("zero_hz", 0))
        if acq["baseline"] > acq["minimum"]:
            self.depth = float(acq["baseline"] - acq["minimum"])

    def wait_locked(self):
        deadline = time.monotonic() + TIMING["lock_wait_s"]
        while True:
            page = self.fresh_page()
            if page is not None and page.state == LOCKED:
                return
            if time.monotonic() >= deadline:
                raise RuntimeError("638 not LOCKED")
            time.sleep(.1)

    def restore_previous(self):
        """Put the gains active at the start back exactly (defaults -> restore, override -> set)."""
        prev = self.previous
        self.wait_locked()
        if prev["overridden"]:
            rb = self.c.set_normalized_pi_638(prev["kp"], prev["ki_per_s"], zero_hz=prev.get("zero_hz", 0), timeout_s=2)
        else:
            rb = self.c.restore_normalized_pi_638(timeout_s=2)
        if (rb["packed"], rb["overridden"], rb.get("zero_hz", 0)) != (prev["packed"], prev["overridden"], prev.get("zero_hz", 0)):
            raise RuntimeError(f"previous gains readback differs: {rb}")

    # ---------- orchestration ----------
    def run(self):
        r = self.r
        try:
            if self.out:
                self.out.mkdir(parents=True, exist_ok=True)
            self.enter("checking", "Checking lock, firmware contract and current gains")
            self.check_cancel()
            self.check()
            self.enter("settling", "Waiting for a stable operating point")
            self.settle()
            self.enter("residual_before", "Recording 2 s in-lock residual at the current gains")
            r["residual_before_adc"] = self.residual("residual_before")
            self.enter("measuring", f"Identification trace (PRBS-9 hold {HOLD} D {DECIMATION} +/-{self.amp} DAC)")
            fr, meta = self.measure("trace")
            self.enter("designing", "Fitting the plant and designing the gains")
            params, kp, ki = self.design(fr, meta)
            self.enter("applying", f"Applying Kp {kp} Ki {ki}/s; {TIMING['guard_s']:g} s guard")
            page = self.fresh_page()
            if page is None or page.state != LOCKED or not 500 < page.dac < self.cap - 500:
                raise RuntimeError("Lock or DAC headroom not healthy before applying the design")
            self.check_cancel()
            self.changed = True  # set before the SET: it can apply even if its readback fails
            self.new = self.c.set_normalized_pi_638(kp, ki, zero_hz=0, timeout_s=2)
            self.watch(TIMING["guard_s"], guard_limit=self.guard_limit())
            # Before the reacquire: same lock point as residual_before, so the two compare.
            self.enter("residual_after", "Recording 2 s in-lock residual at the new gains")
            r["residual_after_adc"] = self.residual("residual_after")
            self.enter("relock_test", "Reacquiring once at the new gains")
            self.check_cancel()
            self.relock(self.guard_limit())
            self.watch(TIMING["guard_s"], guard_limit=self.guard_limit())
            if self.full_check:
                self.enter("checking_design", "Second trace at the new gains")
                fr2, meta2 = self.measure("check", guard_limit=0.08 * self.depth)
                v = lm.compare_measured(fr2, params, kp, ki)
                v["ok"] = bool(meta2["valid"] and v.get("bins", 0) >= CHECK["min_bins"]
                               and v["median_abs_error"] <= CHECK["max_error"]
                               and v["measured_peak_S"] <= self.ms_limit + CHECK["peak_margin"])
                r["check"] = v
            bands = [(self.d.get(k) or {}).get("bands_adc", {}).get(RINGING_BAND)
                     for k in ("residual_before", "residual_after")]
            failed = [why for ok, why in (
                (r["relock"]["override_kept"], "override not kept after the reacquire"),
                (None not in bands, "residual record rejected"),
                (None in bands or bands[1] <= RINGING_RATIO * bands[0],
                 f"{RINGING_BAND} Hz residual {bands[1]} ADC > {RINGING_RATIO} x {bands[0]} ADC (ringing)"),
                (not self.full_check or r["check"]["ok"], "check capture outside limits")) if not ok]
            r["verified"] = not failed
            r["status"] = "complete"
            r["message"] = (f"Kp {kp} Ki {ki}/s applied as a runtime override; "
                            + ("verified" if not failed else "NOT verified: " + "; ".join(failed)))
        except _Rejected as exc:
            r.update(status="rejected", message=str(exc))
        except _Cancelled as exc:
            r.update(status="cancelled", message=str(exc))
        except Exception as exc:
            r.update(status="failed", message=f"{type(exc).__name__}: {exc}")
        finally:
            self.cleanup()
        return r

    def cleanup(self):
        r = self.r
        self.enter("restoring_trace", "Restoring the previous gains" if self.changed and r["status"] != "complete"
                   else "Restoring the trace configuration")
        if self.changed and r["status"] != "complete":
            try:
                self.restore_previous()
                self.d["previous_gains_restored"] = True
            except BaseException as exc:
                self.d["previous_gains_restored"] = False
                r["message"] += f"; previous gains NOT restored ({type(exc).__name__}: {exc})"
        if self.configured:
            state = {"attempts": 0}
            try:
                _restore_trace_config(self.c, self.original_trace, timeout_s=TIMING["trace_restore_s"],
                                      emit=lambda _stage, message, **kw: self.emit(message, **kw),
                                      cancel=None, state=state)
                self.d["trace_config_restored"] = True
            except BaseException as exc:
                self.d["trace_config_restored"] = False
                r["message"] += f"; trace configuration NOT restored ({type(exc).__name__}: {exc})"
            self.d["trace_restore_attempts"] = state["attempts"]
        if self.changed:
            try:
                pi = self.d["pi_final"] = self.c.read_normalized_pi_638(timeout_s=2)
                r["applied"] = bool(self.new is not None and pi["overridden"] and pi["packed"] == self.new["packed"]
                                    and pi.get("zero_hz", 0) == self.new.get("zero_hz", 0))
            except Exception as exc:
                self.d["pi_final_error"] = str(exc)
                r["applied"] = bool(self.new is not None and not self.d.get("previous_gains_restored", r["status"] != "complete"))
        if r["status"] == "complete" and not r["applied"]:
            r["verified"] = False
            r["message"] += "; the new gains are not active at the end"
        now = time.monotonic()
        self.stage_seconds[self.stage] = round(self.stage_seconds.get(self.stage, 0) + now - self.stage_t0, 2)
        r["duration_s"] = round(now - self.t0, 2)
        plain = _plain(r)
        if self.out:
            try:
                (self.out / "lock_calibration.json").write_text(json.dumps(plain, indent=1, allow_nan=False) + "\n")
            except Exception as exc:
                r["message"] += f"; report not written ({exc})"
                plain = _plain(r)
        r.clear()
        r.update(plain)
        self.part = 1.0
        self.emit(f"{r['status']}: {r['message']}")


def calibrate_lock_638(controller, report_dir=None, *, progress=None, cancel=None,
                       ms_limit=1.4, amplitude_dac=None, full_check=False):
    """Measure this probe's lock plant and apply the max-Ki PI meeting ``ms_limit`` (runtime override).

    Preconditions: connected, streaming optical diagnostics (format 2), LOCKED in
    automatic RUN with normalized PI active and a valid slope; no other thread
    issues Controller commands during the call. Never STARTs or STOPs.
    ``progress(event)`` receives ``{"stage", "message", "elapsed_s", "fraction"}``;
    ``cancel`` is a zero-argument predicate. On cancel or failure the gains active
    at the start are put back exactly and the trace configuration is restored; on
    success the new gains stay applied. Returns the JSON-serializable report of
    LOCK_CALIBRATION_API.md (plus ``details``); written to
    ``report_dir/lock_calibration.json`` with the trace and residual records.
    """
    return _Calibration(controller, report_dir, progress, cancel, ms_limit, amplitude_dac, full_check).run()


def _on_grid(value, step):
    return abs(value / step - round(value / step)) <= 1e-6


def apply_saved_lock_gains_638(controller, gains, timeout_s=2.0):
    """Set saved ``{"kp", "ki_per_s", "zero_hz"}`` with ``set_normalized_pi_638`` and return its readback.

    The gains are checked against the firmware's normalized PI contract (read
    with one GET) first; a ValueError says why and nothing is set when they do
    not fit. Values must be exactly representable on the contract's wire grid,
    so the applied gains equal the saved ones. Requires normalized PI active with
    a valid slope (LOCKED).
    """
    kp, ki, zero = gains["kp"], gains["ki_per_s"], gains.get("zero_hz", 0)
    for name, value in (("kp", kp), ("ki_per_s", ki)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite number >= 0")
    if isinstance(zero, float) and zero.is_integer():
        zero = int(zero)
    if isinstance(zero, bool) or not isinstance(zero, int) or zero < 0:
        raise ValueError("zero_hz must be an integer >= 0")
    contract = controller.read_normalized_pi_638(timeout_s=timeout_s).get("version", 1)
    if contract == 1:
        if zero:
            raise ValueError("firmware contract v1 has no second integrator (zero_hz must be 0)")
        if kp > lm.WIRE_LIMITS[1][0]:
            raise ValueError(f"firmware contract v1 caps Kp at {lm.WIRE_LIMITS[1][0]:g}")
        if ki > lm.WIRE_LIMITS[1][1]:
            raise ValueError(f"firmware contract v1 caps Ki at {lm.WIRE_LIMITS[1][1]:g}")
        if not _on_grid(kp, 0.001) or not _on_grid(ki, 0.1):
            raise ValueError("firmware contract v1 sets Kp in 0.001 and Ki in 0.1/s steps; "
                             f"Kp {kp} / Ki {ki} would not be applied exactly")
    elif contract == 2:
        if kp > lm.WIRE_LIMITS[2][0]:
            raise ValueError(f"firmware contract v2 caps Kp at {lm.WIRE_LIMITS[2][0]:g}")
        if ki > lm.WIRE_LIMITS[2][1]:
            raise ValueError(f"firmware contract v2 caps Ki at {lm.WIRE_LIMITS[2][1]:g}")
        if zero > 63:
            raise ValueError("firmware contract v2 caps zero_hz at 63")
        if not _on_grid(kp, 0.005) or not _on_grid(ki, 1.0):
            raise ValueError("firmware contract v2 sets Kp in 0.005 and Ki in 1/s steps; "
                             f"Kp {kp} / Ki {ki} would not be applied exactly")
    else:
        raise ValueError(f"unsupported normalized PI contract v{contract}")
    return controller.set_normalized_pi_638(float(kp), float(ki), zero_hz=zero, timeout_s=timeout_s)
