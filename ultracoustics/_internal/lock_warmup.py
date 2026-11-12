"""Warm-up of the 638 lock after START: one call a GUI can show as "Warming up" and never ask the user about.

warm_up_lock_638 waits for a settled lock, measures the running gains (check_lock_638, ~3 s) and, only when they
exceed the robustness limit or a recalibration would raise Ki by 30 % or more, applies the check's recommended
gains, verifies them (oscillation guard, ringing, one reacquire; ~11 s, the previous gains come back on any
failure) and stores them on the box (STOP, tuning record, START), then confirms with a second check. Typically it
finishes ~15 s after the lock; a START that retunes takes about a minute. A check that cannot measure (the probe
being handled, bumps) leaves the gains as they are; the next START tries again. A lock that never settles, or is
gone by the end, is reported as "not_ready", so a caller can tell a measured lock from one it must not use.

The building blocks are public too: wait_settled_638, apply_recommended_lock_gains_638, store_lock_gains_638.
"""
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from . import lock_calibration as _cal
from .lock_check import check_lock_638
from .lock_common import Cancelled as _Cancelled, plain as _plain

# Quiet enough for a 1.6 s burst: the span limit catches handling and bumps; a slow thermal drift is detrended
# by the check, so it may be faster than the calibration's 3 DAC/s (its trace takes a minute), which right after
# lock kept warm-ups waiting 6-10 s longer.
SETTLE = dict(min_locked_s=10.0, quiet_s=5.0, span_dac=100, drift_dac_per_s=10.0, timeout_s=90.0)
# The 638 in ERROR, or silent after it had been publishing (the master stopped the box): stop waiting. Silence
# before its first page is the boot (the 1550 warming up, the 638 not yet powered) and is waited out, unless the
# stream is not carrying optical telemetry at all, when no page will ever come.
GIVE_UP = dict(error_s=5.0, stale_s=10.0, no_telemetry_s=3.0)
LOCK_TIMEOUT_S = 60.0
APPLY_EXPECTED_S = dict(checking=0.5, residual_before=2.0, applying=3.0, residual_after=2.0, relock_test=4.0)
STEP_WEIGHTS = dict(settling=12, checking=3.5, applying=11.5, storing=20, confirming=15.5)


def _live(controller):
    snap = controller.telemetry
    entry = getattr(snap, "optical_live_638", None) if snap is not None else None
    if snap is None or getattr(snap, "host_stale", False) or entry is None or getattr(entry, "host_stale", False):
        return None
    return entry.page


def wait_settled_638(controller, *, progress=None, cancel=None, min_locked_s=SETTLE["min_locked_s"],
                     quiet_s=SETTLE["quiet_s"], span_dac=SETTLE["span_dac"],
                     drift_dac_per_s=SETTLE["drift_dac_per_s"], timeout_s=SETTLE["timeout_s"]):
    """Wait until the 638 has been LOCKED for ``min_locked_s`` and its DAC was quiet over the last ``quiet_s``
    (span <= ``span_dac``, |drift| <= ``drift_dac_per_s``; see SETTLE).
    Returns ``{"settled", "locked_s", "span_dac", "drift_dac_per_s", "waited_s", "reason"}``; never raises on
    a noisy lock, only on cancel. A drop restarts the clock. Gives up early when the 638 stays in ERROR or
    telemetry stops (the master stopped the box), so a box that cannot lock is not waited on."""
    t0 = time.monotonic()
    locked_since, last_seq, hist, error_since, fresh_at = None, None, [], None, None
    out = {"settled": False, "locked_s": 0.0, "span_dac": None, "drift_dac_per_s": None, "reason": ""}
    while time.monotonic() - t0 < timeout_s:
        if cancel and cancel():
            raise _Cancelled("Cancelled while waiting for a settled lock")
        page, now = _live(controller), time.monotonic()
        if page is None and fresh_at is not None and now - fresh_at > GIVE_UP["stale_s"]:
            out["reason"] = "638 telemetry stopped (the box stopped?)"
            break
        if (page is None and fresh_at is None and now - t0 > GIVE_UP["no_telemetry_s"]
                and (controller.stream_stats or {}).get("stream_format") != 2):
            out["reason"] = "optical telemetry is off: call enable_optical_diagnostics() before START"
            break
        if page is not None and page.sequence != last_seq:
            last_seq, fresh_at = page.sequence, now
            error_since = (error_since or now) if page.state == _cal.ERROR else None
            if error_since is not None and now - error_since >= GIVE_UP["error_s"]:
                out["reason"] = "the 638 reported ERROR (no lock)"
                break
            if page.state != _cal.LOCKED:
                locked_since, hist = None, []
            else:
                locked_since = now if locked_since is None else locked_since
                hist.append((now, float(page.dac)))
                hist = [h for h in hist if h[0] >= now - quiet_s]
                out["locked_s"] = round(now - locked_since, 2)
                if now - locked_since >= min_locked_s and len(hist) >= 5 and hist[-1][0] - hist[0][0] >= quiet_s - 1:
                    t, d = np.asarray(hist).T
                    out.update(span_dac=float(np.ptp(d)), drift_dac_per_s=float(np.polyfit(t - t[0], d, 1)[0]))
                    if out["span_dac"] <= span_dac and abs(out["drift_dac_per_s"]) <= drift_dac_per_s:
                        out["settled"] = True
                        break
        if progress:
            progress({"stage": "settling", "fraction": min(1.0, out["locked_s"] / min_locked_s),
                      "message": "Waiting for a settled lock", "elapsed_s": round(now - t0, 2)})
        time.sleep(0.05)
    if not out["settled"] and not out["reason"]:
        out["reason"] = "the lock was not quiet enough within the time allowed"
    out["waited_s"] = round(time.monotonic() - t0, 2)
    return out


def locked_638(controller):
    """Fresh telemetry shows the 638 LOCKED. Reads only the cached telemetry. The 638's own
    page decides: the snapshot as a whole also covers pages this may not need."""
    entry = getattr(controller.telemetry, "optical_live_638", None)
    if entry is None or getattr(entry, "host_stale", False):
        return False
    page = getattr(entry, "page", entry)
    return getattr(page, "state", None) == _cal.LOCKED and not getattr(page, "stale", False)


def _lock_outcome(controller):
    """"ready" while fresh telemetry shows LOCKED, else "not_ready"."""
    return "ready" if locked_638(controller) else "not_ready"


class _Apply(_cal._Calibration):
    expected = APPLY_EXPECTED_S

    def __init__(self, controller, report_dir, progress, cancel, gains, ms_limit):
        super().__init__(controller, report_dir, progress, cancel, ms_limit, None, False)
        self.gains = gains
        self.r = {"schema": 1, "status": "failed", "message": "", "gains": dict(gains), "previous_gains": None,
                  "verified": False, "applied": False, "relock": None, "residual_before_adc": None,
                  "residual_after_adc": None, "duration_s": 0.0}
        self.d = self.r["details"] = {"stage_seconds": self.stage_seconds}

    def run(self):
        r = self.r
        try:
            self.enter("checking", "Checking lock and current gains")
            self.check()
            r["previous_gains"] = _cal._gains(self.previous)
            self.enter("residual_before", "Recording 2 s at the current gains")
            r["residual_before_adc"] = self.residual("residual_before")
            self.enter("applying", "Applying the recommended gains")
            self.check_cancel()
            self.changed = True
            self.new = self.c.set_normalized_pi_638(self.gains["kp"], self.gains["ki_per_s"],
                                                    zero_hz=self.gains.get("zero_hz", 0), timeout_s=2)
            self.watch(_cal.TIMING["guard_s"], guard_limit=self.guard_limit())
            self.enter("residual_after", "Recording 2 s at the new gains")
            r["residual_after_adc"] = self.residual("residual_after")
            self.enter("relock_test", "Reacquiring once at the new gains")
            self.check_cancel()
            self.relock(self.guard_limit())
            bands = [(self.d.get(k) or {}).get("bands_adc", {}).get(_cal.RINGING_BAND)
                     for k in ("residual_before", "residual_after")]
            failed = [why for ok, why in (
                (r["relock"]["override_kept"], "override not kept after the reacquire"),
                (None not in bands, "residual record rejected"),
                (None in bands or bands[1] <= _cal.RINGING_RATIO * bands[0],
                 f"{_cal.RINGING_BAND} Hz residual {bands[1]} ADC > {_cal.RINGING_RATIO} x {bands[0]} ADC")) if not ok]
            r.update(verified=not failed, status="complete" if not failed else "failed",
                     message="verified" if not failed else "; ".join(failed))
        except _Cancelled as exc:
            r.update(status="cancelled", message=str(exc))
        except Exception as exc:
            r.update(status="failed", message=f"{type(exc).__name__}: {exc}")
        finally:
            if self.changed and not r["verified"]:
                try:
                    self.restore_previous()
                    self.d["previous_gains_restored"] = True
                except BaseException as exc:
                    self.d["previous_gains_restored"] = False
                    r["message"] += f"; previous gains NOT restored ({type(exc).__name__}: {exc})"
            r["applied"] = bool(self.changed and r["verified"])
            r["duration_s"] = round(time.monotonic() - self.t0, 2)
            if self.stage is not None:
                self.stage_seconds[self.stage] = round(time.monotonic() - self.stage_t0, 2)
            plain = _plain(r)
            r.clear()
            r.update(plain)
        return r


def apply_recommended_lock_gains_638(controller, gains, *, progress=None, cancel=None, ms_limit=1.4):
    """Apply ``gains`` (kp, ki_per_s, zero_hz) as a runtime override and verify them the way the calibration does
    after its design: 3 s oscillation guard, 2 s residual records before/after (200-1000 Hz ringing <= 2.5x) and
    one reacquire. About 11 s. On any failure or cancel the previous gains are put back. Returns
    ``{"status", "verified", "applied", "message", ...}``; nothing is stored on the box."""
    return _Apply(controller, None, progress, cancel, gains, ms_limit).run()


def _wait_locked(controller, timeout_s, cancel):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if cancel and cancel():
            raise _Cancelled("Cancelled while waiting for the lock")
        try:
            pi = controller.read_normalized_pi_638(timeout_s=2)
            if pi["active"] and pi["slope_valid"]:
                return pi
        except RuntimeError:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"no lock with normalized PI active within {timeout_s:g} s after START")


#: The 638 reads its gains back on its wire grid (contract v2: Kp in 0.005, Ki in 1/s): half a step either
#: way is the same gain.
KP_TOLERANCE = 0.0025 + 1e-9
KI_TOLERANCE = 0.5 + 1e-9


def stored_gains_problem(pi, gains):
    """Why a normalized PI readback after START does not show ``gains`` running from the tuning record, or None."""
    if pi.get("record_rejected"):
        return "the 638 rejected the stored lock gains"
    if not pi.get("record_active"):
        return "the 638 reports no stored lock gains after the restart"
    if pi.get("overridden"):
        return "the 638 runs runtime gains, not its stored ones"
    if not (int(pi.get("zero_hz") or 0) == int(gains.get("zero_hz") or 0)
            and abs(float(pi["kp"]) - float(gains["kp"])) <= KP_TOLERANCE
            and abs(float(pi["ki_per_s"]) - float(gains["ki_per_s"])) <= KI_TOLERANCE):
        return f"the 638 runs Kp {float(pi['kp']):.3f}, Ki {float(pi['ki_per_s']):g}/s, not the stored gains"
    return None


def store_lock_gains_638(controller, gains, report_id, *, cancel=None, progress=None,
                         lock_timeout_s=LOCK_TIMEOUT_S):
    """Store ``gains`` in the 638's tuning record and restart so it runs them: confirmed STOP, record write
    (TEC fields kept), optical diagnostics, confirmed START, wait up to ``lock_timeout_s`` for the lock, then
    check the 638 runs the record's gains as its box defaults (:func:`stored_gains_problem`).

    Returns ``{"stored", "verified", "problem", "gains", "report_id", "record", "readback"}``: ``problem`` says
    why it is not verified. A STOP that is not confirmed raises, with nothing written. A write that fails comes
    back ``stored`` False and leaves the box stopped; a restart that fails comes back ``stored`` True.
    ``progress(event)`` gets ``{"stage", "message"}`` at each step.

    ``cancel`` is polled before the STOP, after the write and while waiting for the lock. The write itself runs to
    the end, so a cancel after it leaves the gains stored, with the box stopped if START had not been sent."""
    gains = {"kp": float(gains["kp"]), "ki_per_s": float(gains["ki_per_s"]), "zero_hz": int(gains.get("zero_hz") or 0)}
    out = {"stored": False, "verified": False, "problem": None, "gains": gains, "report_id": report_id,
           "record": None, "readback": None}

    def say(stage, message):
        if progress:
            progress({"stage": stage, "message": message})
    if cancel and cancel():
        raise _Cancelled("Cancelled before storing the gains")
    say("stopping", "Stopping the box to store the gains...")
    controller.stop_system_confirmed(timeout_s=4.0)
    say("writing", "Writing the lock gains into the 638 tuning record (the 638 restarts into its bootloader)...")
    try:
        record = controller.write_lock_tuning_638(gains["kp"], gains["ki_per_s"], zero_hz=gains["zero_hz"],
                                                  report_id=report_id)
    except (RuntimeError, TimeoutError, ValueError) as exc:
        out["problem"] = f"the write failed: {exc}"
        return out
    out.update(stored=True, record=record)
    if cancel and cancel():
        raise _Cancelled("Cancelled with the new gains stored; the box stays stopped")
    sequence = record.get("sequence") if isinstance(record, dict) else None
    say("verifying", f"Stored as record #{sequence}. Restarting so the 638 loads it...")
    try:
        controller.enable_optical_diagnostics(timeout_s=1.0)
        controller.start_confirmed(timeout_s=4.0)
        pi = _wait_locked(controller, lock_timeout_s, cancel)
    except (RuntimeError, TimeoutError) as exc:
        out["problem"] = f"the restart failed: {exc}"
        return out
    out["readback"] = pi
    out["problem"] = stored_gains_problem(pi, gains)
    out["verified"] = out["problem"] is None
    return out


def warm_up_lock_638(controller, report_dir=None, *, progress=None, cancel=None, store=True, ms_limit=1.4,
                     onset_after_sample_us=None):
    """After START: settle, check and, when needed, retune and store the lock gains. See the module docstring.

    ``progress(event)`` receives ``{"stage", "fraction", "message", "detail"}``; ``fraction`` covers the whole
    warm-up, ``message`` is always "Warming up" (for users) and ``detail`` says what is happening (for
    developers). ``cancel()`` stops it; gains being verified are put back. Returns
    ``{"outcome", "changed", "gains_before", "gains_after", "settle", "check", "apply", "store", "confirm",
    "duration_s"}``. ``outcome`` is one of:

    - "ready": locked, gains unchanged (``check`` says whether they were measured).
    - "retuned": locked on new gains, stored (runtime only with ``store=False``).
    - "not_ready": no settled lock, so do not measure. ``changed`` says if new gains were stored first.
    - "cancelled" / "failed": the box keeps the gains it had unless ``changed`` is true and the store had begun;
      ``message`` says so, and a failed or cancelled store can leave the box stopped."""
    t0 = time.monotonic()
    out = {"schema": 1, "outcome": "failed", "changed": False, "message": "", "gains_before": None,
           "gains_after": None, "settle": None, "check": None, "apply": None, "store": None, "confirm": None,
           "duration_s": 0.0}
    plan = ["settling", "checking"]
    out_dir = Path(report_dir).expanduser() if report_dir is not None else None
    check_kwargs = {} if onset_after_sample_us is None else {"onset_after_sample_us": onset_after_sample_us}

    def reporter(step):
        def report(event):
            if not progress:
                return
            done = sum(STEP_WEIGHTS[s] for s in plan[:plan.index(step)])
            total = sum(STEP_WEIGHTS[s] for s in plan)
            within = float(event.get("fraction") or 0.0) * STEP_WEIGHTS[step]
            progress({"stage": step, "fraction": round(min(1.0, (done + within) / total), 3),
                      "message": "Warming up", "detail": event.get("message", "")})
        return report

    def check(name):
        return check_lock_638(controller, out_dir / name if out_dir else None, progress=reporter(plan[-1]),
                              cancel=cancel, ms_limit=ms_limit, **check_kwargs)

    try:
        out["settle"] = wait_settled_638(controller, progress=reporter("settling"), cancel=cancel)
        if not out["settle"]["settled"]:
            out.update(outcome="not_ready", message=f"not settled ({out['settle']['reason']}); gains left as they are")
            return out
        out["check"] = c1 = check("check")
        out["gains_before"] = out["gains_after"] = c1.get("gains")
        if c1["status"] == "cancelled":
            raise _Cancelled(c1["message"])
        if c1["status"] not in ("rerun", "improvable") or not c1.get("recommended"):
            out.update(outcome=_lock_outcome(controller), message=f"check {c1['status']}: gains left as they are")
            return out
        plan.append("applying")
        rec = {k: c1["recommended"][k] for k in ("kp", "ki_per_s", "zero_hz")}
        out["apply"] = a = apply_recommended_lock_gains_638(controller, rec, progress=reporter("applying"),
                                                            cancel=cancel, ms_limit=ms_limit)
        if a["status"] == "cancelled":
            raise _Cancelled(a["message"])
        if not a["verified"]:
            out.update(outcome=_lock_outcome(controller),
                       message=f"recommended gains not verified ({a['message']}); previous gains kept")
            return out
        out.update(changed=True, gains_after=rec)
        if not store:
            out.update(outcome="retuned", message="retuned for this session (not stored)")
            return out
        plan.extend(["storing", "confirming"])
        reporter("storing")({"fraction": 0.0, "message": "Storing the new gains on the box and restarting"})
        report_id = datetime.now().strftime("warmup_%Y%m%d_%H%M%S")
        out["store"] = st = store_lock_gains_638(controller, rec, report_id, cancel=cancel)
        if st["readback"] is None:
            raise RuntimeError(st["problem"])
        reporter("confirming")({"fraction": 0.0, "message": "Waiting for a settled lock"})
        settle2 = wait_settled_638(controller, cancel=cancel)
        if not settle2["settled"]:
            out["confirm"] = {"status": "skipped", "settle": settle2}
            out.update(outcome="not_ready", message=f"retuned to Kp {rec['kp']:g} Ki {rec['ki_per_s']:g}/s and stored, "
                       f"but the lock did not settle again ({settle2['reason']})")
            return out
        out["confirm"] = check("confirm")
        out.update(outcome="retuned", message=f"retuned to Kp {rec['kp']:g} Ki {rec['ki_per_s']:g}/s "
                   f"(stored {'and verified' if st['verified'] else 'but the restart readback differs'}; "
                   f"confirm check {out['confirm']['status']})")
    except _Cancelled as exc:
        out.update(outcome="cancelled", message=str(exc))
    except Exception as exc:
        out.update(outcome="failed", message=f"{type(exc).__name__}: {exc}")
    finally:
        out["duration_s"] = round(time.monotonic() - t0, 2)
        if out_dir:
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "warm_up.json").write_text(json.dumps(_plain(out), indent=1, allow_nan=False) + "\n")
            except Exception as exc:
                out["message"] += f"; report not written ({exc})"
        if progress:
            ready = out["outcome"] in ("ready", "retuned")
            progress({"stage": "done", "fraction": 1.0, "message": "Ready" if ready else "Not ready",
                      "detail": out["message"]})
    return out
