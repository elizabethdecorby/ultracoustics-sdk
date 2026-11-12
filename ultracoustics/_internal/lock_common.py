"""Helpers shared by the 638 lock routines (calibration, check, warm-up)."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import time

import numpy as np

from .control import MANUAL_SET
from .box import CHANNEL_TRACE_CONFIG, SlaveCommandRejected

#: Stream counters that must not move during a capture.
COUNTERS = ("drops_seq", "drops_fw", "transfer_errors", "transfer_timeouts", "malformed")
TRACE_CONFIG_FIELDS = ("amplitude_dac", "hold_updates", "decimation")
#: The 638's "busy" status: it is sending the last trace and refuses a change.
STATUS_BUSY = 10


class Cancelled(Exception):
    """The caller's ``cancel()`` returned True."""


def plain(value):
    """``value`` as plain JSON-ready types; non-finite floats become None."""
    if is_dataclass(value):
        return {k: plain(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return [plain(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if hasattr(value, "__dict__"):
        return plain(vars(value))
    return value


def restore_trace_config(controller, original, *, timeout_s, emit, cancel, state):
    """Put the 638's trace settings back, retrying only its "busy" reply, then verify them."""
    expected = tuple(original[key] for key in TRACE_CONFIG_FIELDS)
    deadline = time.monotonic() + timeout_s
    while True:
        if state["attempts"] and time.monotonic() >= deadline:
            raise TimeoutError("Original trace configuration remained busy for restore deadline")
        state["attempts"] += 1  # Cleanup gets one attempt even after cancellation.
        emit("restore", "Restoring original trace configuration",
             attempt=state["attempts"], timeout_s=timeout_s)
        remaining = deadline - time.monotonic()
        try:
            # configure_control_trace_638 performs SET, GET and packed-word
            # verification. Also compare the returned fields to the saved values.
            restored = controller.configure_control_trace_638(
                *expected, timeout_s=max(.05, min(1.0, remaining / 3)))
            actual = tuple(restored[key] for key in TRACE_CONFIG_FIELDS)
            if actual != expected:
                raise RuntimeError(f"638 trace configuration restore readback differs: {actual} != {expected}")
            emit("restore", "Original trace configuration confirmed",
                 attempt=state["attempts"])
            return
        except SlaveCommandRejected as exc:
            if (exc.target, exc.channel, exc.status, exc.opcode) != (638, CHANNEL_TRACE_CONFIG, STATUS_BUSY,
                                                                     MANUAL_SET):
                raise
            if cancel and cancel():
                raise Cancelled("Cancelled while waiting to restore original trace configuration") from exc
            if time.monotonic() >= deadline:
                raise TimeoutError("Original trace configuration remained busy for restore deadline") from exc
            emit("restore", "Trace configuration still busy; waiting to retry",
                 attempt=state["attempts"], remaining_s=round(deadline-time.monotonic(), 2))
            until = min(deadline, time.monotonic() + 1.0)
            while time.monotonic() < until:
                if cancel and cancel():
                    raise Cancelled("Cancelled while waiting to restore original trace configuration") from exc
                time.sleep(min(.05, max(0, until-time.monotonic())))
