"""Quick check of the 638's lock gains (a few seconds), for prompting a rerun of calibrate_lock_638.

The lock calibration's bounded identification burst (PRBS-9, +/-24 DAC on contract v2, 1.64 s) is analysed from
the 10 MSPS photodetector stream instead of the 638's trace readout, which takes about a minute
(``_internal.stream_lock``). From one burst at the running gains it measures peak |S| and fits the plant, so it can
also say what a recalibration would choose. The gains are never changed.

Verdict (``status``):
  ok            peak |S| and the gains' worst case on the measured plant are within ms_limit + 0.1
  rerun         either exceeds ms_limit + 0.1 (the margin the calibration's own check capture allows)
  improvable    within limits, but a recalibration would raise Ki by 30 % or more
  inconclusive  the burst could not be measured well (stream drops, too few bins, poor plant fit)
  failed / cancelled

The burst modulates the laser for 1.64 s and shows in the signal: run it after the lock has settled and never
while recording. After a burst the 638 streams its trace rows for about a minute and refuses another burst
meanwhile, so a second check (or a calibration) within that minute waits; the first check after a START does not. The 638's trace configuration is left at the identification setting (it cannot change while the
638 streams the trace rows, about a minute; the 638 is power-cycled at the next STOP anyway); the previous setting
is in the report.

Validated on 00560032 only (firmware-analysis runs/20261027-QUICK-LOCK-CHECK): 0.6 s of the burst reproduced the
trace method's peak |S| within 0.01 and the calibration's gains within 1 %.
"""
import json
import time
from pathlib import Path

import numpy as np

from . import lock_calibration as _cal
from ._internal import lock_model as lm
from ._internal import stream_lock as sl
from .pi_characterization import COUNTERS, TRACE_CONFIG_FIELDS, _Cancelled, _plain
from .system_control import CHANNEL_OPTICAL, SlaveCommandRejected

SAMPLE_RATE = 10_000_000
PRE_S, POST_S = 0.3, 0.25
BURST_S = 4096 * _cal.DECIMATION / 10_000           # the 638 records 4096 rows, one per DECIMATION updates
RERUN_MARGIN = _cal.CHECK["peak_margin"]
FASTER_KI_RATIO = 1.3
EXPECTED_S = dict(checking=1, arming=0.5, exciting=2.2, analysing=1)
REPORT_KEYS = ("schema", "status", "message", "box_serial", "contract", "gains", "ms_limit", "threshold",
               "measured_peak_S", "worst_peak_current", "recommended", "fit", "reasons", "timing", "periods",
               "bins", "trace_config", "duration_s", "report_dir")


class _Check(_cal._Calibration):
    expected = EXPECTED_S

    def __init__(self, controller, report_dir, progress, cancel, ms_limit, amplitude_dac, onset_after_sample_us):
        super().__init__(controller, report_dir, progress, cancel, ms_limit, amplitude_dac, False)
        self.onset_after_sample_us = onset_after_sample_us
        self.r = {k: None for k in REPORT_KEYS}
        self.r.update(schema=1, status="failed", message="", box_serial=getattr(controller, "device_serial", None),
                      ms_limit=ms_limit, threshold=ms_limit + RERUN_MARGIN, reasons=[], duration_s=0.0,
                      report_dir=str(self.out) if self.out else None)
        self.d = self.r["details"] = {"stage_seconds": self.stage_seconds}

    def capture(self, guard_limit):
        """Trigger the burst and copy [trigger - PRE_S, burst end + POST_S] out of the ring as 1 us means."""
        c, per_us = self.c, SAMPLE_RATE // 1_000_000
        cap = int(c.buffer_capacity)
        if int(PRE_S * SAMPLE_RATE) > cap - 3 * _cal.GUARD_SAMPLES:
            raise RuntimeError("ADC ring too small for the pre-trigger record")
        deadline, waited = time.monotonic() + _cal.TIMING["busy_s"], time.monotonic()
        while True:
            self.check_cancel()
            before = self.stats()
            trigger = int(c.samples_received)              # re-taken per attempt: the burst follows the accepted one
            try:
                c.control_638("identify", timeout_s=1.0)
                break
            except SlaveCommandRejected as exc:
                if (exc.target, exc.channel, exc.status) != (638, CHANNEL_OPTICAL, 10) or time.monotonic() >= deadline:
                    raise
                self.emit("Waiting for the previous burst's trace readout to finish (up to a minute)")
                self.watch(1.0, guard_limit=guard_limit)
        self.d["busy_wait_s"] = round(time.monotonic() - waited, 2)
        start = trigger - int(PRE_S * SAMPLE_RATE)
        end = int(c.samples_received) + int((BURST_S + POST_S) * SAMPLE_RATE)
        cursor, parts, rest = start, [], np.empty(0)
        while cursor < end:
            self.watch(0.05, guard_limit=guard_limit)          # lock, telemetry, stream, cancel, guard
            head = int(c.samples_received) - _cal.GUARD_SAMPLES
            if head - cursor > cap - 2 * _cal.GUARD_SAMPLES:
                raise RuntimeError("ADC ring overrun")
            upto = min(head, end)
            if upto > cursor:
                i, k = cursor % cap, upto - cursor
                raw = c.buffer[i:i + k] if i + k <= cap else np.concatenate((c.buffer[i:], c.buffer[:i + k - cap]))
                raw = np.asarray(raw, dtype=np.float64)
                if int(c.samples_received) - cursor > cap - _cal.GUARD_SAMPLES:
                    raise RuntimeError("ADC ring overwritten while copying")
                rest = np.concatenate((rest, raw))
                n = len(rest) // per_us
                parts.append(rest[:n * per_us].reshape(n, per_us).mean(axis=1))
                rest, cursor = rest[n * per_us:], upto
                self.part = (cursor - start) / (end - start)
        after = self.stats()
        deltas = {k: int(after.get(k, 0)) - int(before.get(k, 0)) for k in COUNTERS}
        return np.concatenate(parts), deltas

    def run(self):
        r = self.r
        try:
            if self.out:
                self.out.mkdir(parents=True, exist_ok=True)
            self.enter("checking", "Checking lock, firmware contract and current gains")
            self.check_cancel()
            self.check()
            pi = self.previous
            r["gains"] = _cal._gains(pi)
            self.enter("arming", f"Configuring the identification burst (+/-{self.amp} DAC)")
            if tuple(self.original_trace[k] for k in TRACE_CONFIG_FIELDS) == (self.amp, _cal.HOLD, _cal.DECIMATION):
                left = self.original_trace  # left by an earlier check; a SET would wait out its readout (~50 s)
            else:
                left = self.configure_trace(self.amp)
            r["trace_config"] = {"before": self.original_trace, "left": left}
            self.watch(0.3)
            page = self.fresh_page()
            if page is None or page.state != _cal.LOCKED:
                raise RuntimeError("Not LOCKED before the burst")
            if not 500 + self.amp < page.dac < self.cap - 500 - self.amp:
                raise RuntimeError(f"Insufficient DAC headroom for the burst at DAC {page.dac}")
            self.enter("exciting", f"Identification burst ({BURST_S:.2f} s, +/-{self.amp} DAC)")
            stream, deltas = self.capture(guard_limit=0.08 * self.depth)
            self.d["stream_counter_deltas"] = deltas
            self.enter("analysing", "Measuring |S| and fitting the plant from the stream")
            a = sl.analyse(stream, self.amp, _cal.HOLD, page.slope, pi["kp"], pi["ki_per_s"], pi.get("zero_hz", 0),
                           self.onset_after_sample_us)
            r.update(measured_peak_S=a["peak_S"], periods=a["periods"], bins=a["bins"],
                     timing={k: a[k] for k in ("coarse_us", "onset_us", "start_us", "onset_after_sample_us",
                                                "correlation")})
            self.d.update(slope=page.slope, band_hz=a["band_hz"])
            if a["fit"] is not None:
                ft = a["fit"]
                r["fit"] = {k: ft[k] for k in ("g", "tau_ms", "delay_us", "driver_pole_hz", "median_rel_resid", "bins")}
                self.d["fit_params"] = ft["params"]
            if self.out:
                fr = a["frf"]
                np.savez_compressed(self.out / "lock_check.npz", stream_1us=stream.astype(np.float32),
                                    f=fr["f"], S=fr["S"], relerr=fr["relerr"])
            self.verdict(a, pi, deltas)
        except _Cancelled as exc:
            r.update(status="cancelled", message=str(exc))
        except Exception as exc:
            r.update(status="failed", message=f"{type(exc).__name__}: {exc}")
        finally:
            self.finish()
        return r

    def verdict(self, a, pi, deltas):
        r, threshold = self.r, self.r["threshold"]
        ft = a["fit"]
        problems = [why for bad, why in (
            (any(deltas.values()), f"stream counters changed during the burst: {deltas}"),
            (a["peak_S"] is None or a["bins"] < _cal.CHECK["min_bins"], f"only {a['bins']} usable bins"),
            (ft is None or ft["median_rel_resid"] > _cal.GATE["max_fit_resid"], "the plant fit is poor")) if bad]
        if problems:
            r.update(status="inconclusive", message="; ".join(problems), reasons=problems)
            return
        kp, ki, zero = pi["kp"], pi["ki_per_s"], pi.get("zero_hz", 0)
        r["worst_peak_current"] = lm.worst_ms([ft["params"]], kp, ki, zero_hz=zero)
        kp_lim, ki_lim = lm.WIRE_LIMITS[r["contract"]]
        rec_kp, rec_ki = lm.migo([ft["params"]], self.ms_limit, kp_limit=min(_cal.KP_CAP, kp_lim), ki_limit=ki_lim,
                                 contract=r["contract"])
        if rec_ki > 0:
            r["recommended"] = {"kp": rec_kp, "ki_per_s": rec_ki, "zero_hz": 0,
                                "predicted_worst_peak": lm.worst_ms([ft["params"]], rec_kp, rec_ki)}
        reasons = []
        if a["peak_S"] > threshold:
            reasons.append(f"measured peak |S| {a['peak_S']:.2f} is above {threshold:.2f}")
        if r["worst_peak_current"] > threshold:
            reasons.append(f"worst case of the current gains on the measured plant is "
                           f"{r['worst_peak_current']:.2f}, above {threshold:.2f}")
        gains = f"Kp {kp:g} Ki {ki:g}/s"
        if reasons:
            r.update(status="rerun", reasons=reasons, message=f"{gains}: rerun the lock calibration ("
                     + "; ".join(reasons) + ")")
        elif r["recommended"] and rec_ki >= FASTER_KI_RATIO * ki:
            reasons.append(f"a recalibration would raise Ki from {ki:g} to {rec_ki:g}/s")
            r.update(status="improvable", reasons=reasons, message=f"{gains} are within limits; {reasons[0]}")
        else:
            r.update(status="ok", message=f"{gains} are within limits (measured peak |S| {a['peak_S']:.2f}, "
                     f"worst case {r['worst_peak_current']:.2f}, limit {threshold:.2f})")

    def finish(self):
        r, now = self.r, time.monotonic()
        self.stage_seconds[self.stage] = round(self.stage_seconds.get(self.stage, 0) + now - self.stage_t0, 2)
        r["duration_s"] = round(now - self.t0, 2)
        plain = _plain(r)
        if self.out:
            try:
                (self.out / "lock_check.json").write_text(json.dumps(plain, indent=1, allow_nan=False) + "\n")
            except Exception as exc:
                plain["message"] += f"; report not written ({exc})"
        r.clear()
        r.update(plain)
        self.part = 1.0
        self.emit(f"{r['status']}: {r['message']}")


def check_lock_638(controller, report_dir=None, *, progress=None, cancel=None, ms_limit=1.4, amplitude_dac=None,
                   onset_after_sample_us=sl.ONSET_AFTER_SAMPLE_US):
    """Measure the running 638 lock gains in a few seconds and say whether to rerun calibrate_lock_638.

    Same preconditions as calibrate_lock_638: automatic RUN, optical diagnostics (stream format 2), the 638
    LOCKED with normalized PI active and a valid slope, 10 kHz unheld control, DAC headroom for the burst. Best
    run at least ~20 s after the lock acquired, and never while recording (the burst shows in the signal).
    ``progress(event)`` receives ``{"stage", "message", "elapsed_s", "fraction"}``; ``cancel()`` returning True
    stops it before or during the burst. Returns the report (see the module docstring for ``status``); with
    ``report_dir`` it also writes lock_check.json and lock_check.npz (stream, f, S) there. Never changes the gains.
    """
    return _Check(controller, report_dir, progress, cancel, ms_limit, amplitude_dac, onset_after_sample_us).run()
