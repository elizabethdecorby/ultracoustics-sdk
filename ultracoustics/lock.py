"""Checking and calibrating the lock gains.

The 638 laser is held on the probe by a control loop with two gains, Kp and Ki.
The right gains depend on the probe and can change from one START to the next.

Run ``ctrl.check_lock(retune=True)`` after every START where possible. It
confirms the gains suit this probe and this start, and corrects them if they
don't. Keep the probe still and away from noise while it runs, ideally before
the monitored machine starts. It disturbs the signal for about 1.6 s, so it won't
run during a recording or while a monitor is on. If conditions are too noisy it reports that the gains
couldn't be measured, leaves them unchanged, and you can run it again at the
next START.

``ctrl.calibrate_lock()`` is the full calibration: it measures the loop for
about a minute and designs the gains from it. Run it when a probe is new, or
when the check keeps asking for a retune. Choosing Kp and Ki by hand is a
service task.

The steps of a check, and what each ``LockReport.status`` means:

=============  ===============================================================
ok             The gains are good.
improvable     The gains are good, and a retune would make the lock faster.
needs_retune   The gains are not safe for this probe: retune them.
retuned        ``retune=True`` found better gains, verified them and stored them.
calibrated     The full calibration designed gains, verified them and kept them.
not_verified   The calibration's gains failed their checks; the old ones are back.
rejected       The calibration's measurement was not good enough; nothing changed.
inconclusive   Too noisy to measure. The gains are unchanged.
not_ready      No settled lock to measure. Do not take data yet.
cancelled      ``cancel()`` stopped it.
failed         It could not run. ``message`` says why.
=============  ===============================================================
"""
from dataclasses import dataclass, field
from datetime import datetime

from ._internal.lock_calibration import calibrate_lock_638, restore_lock_gains_638
from ._internal.lock_check import check_lock_638
from ._internal.lock_warmup import locked_638, store_lock_gains_638, wait_settled_638, warm_up_lock_638
from ._internal.lock_common import Cancelled

__all__ = ["LockError", "LockGains", "LockReport", "calibrate_lock", "check_lock", "clear_stored_lock_gains",
           "lock_gains", "locked_638", "restore_lock_gains", "wait_ready"]

_STATUS = {"rerun": "needs_retune"}


class LockError(RuntimeError):
    """The lock did not reach the state asked for."""


@dataclass(frozen=True)
class LockGains:
    kp: float
    ki_per_s: float
    zero_hz: int = 0
    source: str = ""    # on the box: "live", "stored" or "default" (the factory gains)


@dataclass(frozen=True)
class LockReport:
    """What a lock check or calibration found.

    ``status`` is one of:

    =============  ===============================================================
    ok             The gains are good.
    improvable     The gains are good, and a retune would make the lock faster.
    needs_retune   The gains are not safe for this probe: retune them.
    retuned        ``retune=True`` found better gains, verified them and stored them.
    calibrated     The full calibration designed gains, verified them and kept them.
    not_verified   The calibration's gains failed their checks; the old ones are back.
    rejected       The calibration's measurement was not good enough; nothing changed.
    inconclusive   Too noisy to measure. The gains are unchanged.
    not_ready      No settled lock to measure. Do not take data yet.
    cancelled      ``cancel()`` stopped it.
    failed         It could not run. ``message`` says why.
    =============  ===============================================================
    """
    status: str
    message: str
    gains: LockGains = None         # the gains running when it finished
    recommended: LockGains = None   # what the measurement says the gains should be
    reasons: tuple = ()
    details: dict = field(default_factory=dict)
    ready: bool = False             # a settled lock to take data on, measured or not

    @property
    def ok(self):
        """The lock is usable, on gains that were measured to be good."""
        return self.status in ("ok", "improvable", "retuned", "calibrated")


def _gains(raw):
    return None if not raw else LockGains(raw["kp"], raw["ki_per_s"], raw.get("zero_hz", 0))


def _from_check(raw, ready):
    return LockReport(_STATUS.get(raw["status"], raw["status"]), raw["message"], _gains(raw.get("gains")),
                      _gains(raw.get("recommended")), tuple(raw.get("reasons") or ()), raw, ready)


def _from_warm_up(raw):
    check = raw.get("check") or {}
    ready = raw["outcome"] in ("ready", "retuned") and bool((raw.get("settle") or {}).get("settled"))
    if raw["outcome"] != "ready":
        return LockReport(raw["outcome"], raw["message"], _gains(raw.get("gains_after")), None,
                          tuple(check.get("reasons") or ()), raw, ready)
    # A lock that is "ready" may still be unmeasured, or on gains that could not be improved.
    return LockReport(_STATUS.get(check["status"], check["status"]), raw["message"], _gains(raw.get("gains_after")),
                      _gains(check.get("recommended")), tuple(check.get("reasons") or ()), raw, ready)


def _require_quiet(controller, what):
    if controller._recorders or controller._monitors:
        raise LockError(f"{what} disturbs the signal: finish the recording or monitor first")
    _require_running(controller)


def _require_running(controller):
    if not controller.running:
        raise LockError("the box is not running: call start() first")


def check_lock(controller, retune=False, *, progress=None, cancel=None, report_dir=None):
    """Wait for a settled lock and measure the gains, in a few seconds. Returns a :class:`LockReport`.

    This is the check the GUI runs at START. It adds a 1.64 s burst of ±24 laser
    DAC counts to the lock and reads the response from the data stream. It takes
    no manual control of the box and never changes the gains.

    ``retune=True`` goes on, as the GUI does: when the gains need it, it applies
    the recommended gains, verifies them, stores them on the box and restarts it,
    which takes about a minute. Failed gains are taken back, so a run that
    cannot improve the lock leaves it as it was.

    ``progress(event)`` is called with ``{"stage", "fraction", ...}``, ``cancel()``
    returning True stops it, and ``report_dir`` keeps the measurement there for
    support. LockError during a recording or monitor, or if the box is not running.
    """
    _require_quiet(controller, "The lock check")
    if retune:
        return _from_warm_up(warm_up_lock_638(controller._box, report_dir, progress=progress, cancel=cancel))
    try:
        settle = wait_settled_638(controller._box, progress=progress, cancel=cancel)
    except Cancelled as exc:
        return LockReport("cancelled", str(exc))
    if not settle["settled"]:
        return LockReport("not_ready", f"not settled: {settle['reason']}", details=settle)
    raw = check_lock_638(controller._box, report_dir, progress=progress, cancel=cancel)
    return _from_check(raw, raw["status"] not in ("cancelled", "failed") and locked_638(controller._box))


def wait_ready(controller, timeout_s=90.0):
    """Wait for a settled lock: 10 s locked and 5 s quiet. LockError if none came.

    This does not measure the gains: :func:`check_lock` does.
    """
    _require_running(controller)
    settle = wait_settled_638(controller._box, timeout_s=timeout_s)
    if not settle["settled"]:
        raise LockError(f"no settled lock: {settle['reason']}")


def lock_gains(controller):
    """The gains the 638 is running. ``source`` says where they came from."""
    pi = controller._box.read_normalized_pi_638()
    source = "live" if pi["overridden"] else "stored" if pi.get("record_active") else "default"
    return LockGains(pi["kp"], pi["ki_per_s"], pi["zero_hz"], source)


def _gains_now(controller):
    """The running gains, or None if the box cannot say (after a failure, say)."""
    try:
        return lock_gains(controller)
    except Exception:
        return None


def calibrate_lock(controller, store=True, *, progress=None, cancel=None, report_dir=None):
    """The full lock calibration: measure this probe's lock loop and design its gains. Returns a :class:`LockReport`.

    More thorough than ``check_lock(retune=True)``, and longer: about two minutes.
    It records the loop's response to a small burst for about a minute, fits the
    loop to it, and designs the fastest gains that keep a safe stability margin.
    The new gains then go through the checks of a retune: an oscillation guard, a
    ringing limit and one reacquire. Keep the probe still and the room quiet.

    Gains that pass are kept and, with ``store=True``, stored on the box, which
    restarts on them to prove it: about another minute. With ``store=False`` they
    last until the next STOP. Gains that fail are taken back, and so are the old
    ones after a cancel or an error, so a calibration never leaves worse gains.

    ``status`` is ``calibrated``, ``not_verified`` (the new gains failed their
    checks and the old ones are back), ``rejected`` (the measurement was not good
    enough to design from; nothing changed), ``cancelled`` or ``failed``.
    ``details`` holds the whole measurement, and ``report_dir`` keeps it for
    support. LockError during a recording or monitor, if the box is not running,
    or with ``store=True`` on firmware that cannot store gains.
    """
    _require_quiet(controller, "The lock calibration")
    if store and not controller._box.read_normalized_pi_638().get("record_supported"):
        raise LockError("this 638 firmware cannot store lock gains on the box: use store=False")
    raw = calibrate_lock_638(controller._box, report_dir, progress=progress, cancel=cancel)
    designed = _gains(raw.get("gains"))
    if raw["status"] != "complete":    # the old gains are already back
        return LockReport(raw["status"], raw["message"], _gains_now(controller), designed, details=raw,
                          ready=locked_638(controller._box))
    if not raw["verified"]:
        restore_lock_gains_638(controller._box, raw.get("previous_gains"))
        return LockReport("not_verified", f"{raw['message']}. The previous gains are back.", lock_gains(controller),
                          designed, details=raw, ready=locked_638(controller._box))
    if store:
        try:
            stored = raw["store"] = store_lock_gains_638(controller._box, raw["gains"],
                                                         datetime.now().strftime("sdk_%Y%m%d_%H%M%S"), cancel=cancel)
        except Cancelled as exc:    # says whether the gains were stored, and if the box stopped
            return LockReport("cancelled", str(exc), _gains_now(controller), designed, details=raw,
                              ready=locked_638(controller._box))
        if not stored["verified"]:
            return LockReport("failed", store_failure(stored), _gains_now(controller), designed, details=raw,
                              ready=locked_638(controller._box))
    return LockReport("calibrated", raw["message"], lock_gains(controller), designed, details=raw,
                      ready=locked_638(controller._box))


def store_failure(stored):
    """What went wrong in ``store_lock_gains_638``'s result, in words."""
    problem = stored.get("problem")
    if not stored.get("stored", True):
        return f"the gains were not stored: {problem}"
    return "the gains were stored, but the box did not start on them" + (f": {problem}" if problem else "")


def restore_lock_gains(controller):
    """Drop live gains and go back to the box's defaults. Returns the :class:`LockGains`."""
    _require_quiet(controller, "Changing the lock gains")
    controller._box.restore_normalized_pi_638()
    return lock_gains(controller)


def clear_stored_lock_gains(controller):
    """Remove the gains stored on the box, so its next START runs the factory
    gains. The box's TEC settings stay. Needs the box stopped."""
    return controller._box.clear_lock_tuning_638()
