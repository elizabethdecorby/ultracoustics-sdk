"""Reusable, safety-bounded master-USB laser sweep policy."""

import time

from ._internal.control import (
    CHANNEL_LASER_DAC, MANUAL_RENEW, MANUAL_SET, MANUAL_TAKE,
)

CAPS = {638: 33000, 1550: 43253}
STATUS_NAMES = {
    1: "bad start", 2: "bad CRC", 3: "bad opcode", 4: "bad channel",
    5: "bad reserved byte", 6: "bad value or thermal not LOCKED",
    7: "not owner", 8: "already owned", 9: "transaction conflict",
}


class ManualSweepError(RuntimeError):
    """Sweep failure carrying accepted partial rows and cleanup provenance."""

    def __init__(self, primary, rows, cleanup_error=None):
        message = str(primary)
        if cleanup_error is not None:
            message += f"; cleanup also failed: {cleanup_error}"
        super().__init__(message)
        self.primary_error = primary
        self.cleanup_error = cleanup_error
        self.rows = list(rows)


def require_ok(reply, operation):
    if reply.status != 0:
        reason = STATUS_NAMES.get(reply.status, "unknown status")
        raise RuntimeError(
            f"{operation} rejected by slave: {reason} (status {reply.status})")
    return reply


def selected_board(snapshot, target):
    if target == 638:
        return snapshot.board_638, snapshot.link_flags_638
    return snapshot.board_1550, snapshot.link_flags_1550


def wait_telemetry_ready(ctrl, target, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        snapshot = ctrl.telemetry
        if snapshot is not None:
            board, link = selected_board(snapshot, target)
            physical_age = board.temperature_age_ms + snapshot.host_age_s * 1000.0
            if (link == 0 and board.temperature_valid and
                    not board.temperature_stale and board.temperature_fault == 0 and
                    physical_age < 500.0 and not snapshot.host_stale):
                return board
        time.sleep(0.02)
    raise TimeoutError("target telemetry did not become ready after power-on")


def take_laser_zero(ctrl, target):
    return require_ok(ctrl.manual_command(
        target, MANUAL_TAKE, CHANNEL_LASER_DAC, 0), "TAKE zero")


def wait_thermal_locked(ctrl, target, target_c, timeout_s=30.0):
    deadline = time.monotonic() + timeout_s
    started = time.monotonic()
    next_renew_s = 0.7
    first_stable_tick = None
    last_tick = None
    last_good_at = None
    stable_samples = 0
    while time.monotonic() < deadline:
        elapsed = time.monotonic() - started
        if elapsed >= next_renew_s:
            require_ok(ctrl.manual_command(target, MANUAL_RENEW,
                                           CHANNEL_LASER_DAC), "warm-up RENEW")
            next_renew_s += 0.6
        snapshot = ctrl.telemetry
        if snapshot is not None:
            board, link = selected_board(snapshot, target)
            physical_age = board.temperature_age_ms + snapshot.host_age_s * 1000.0
            healthy = (link == 0 and board.temperature_valid and
                       not board.temperature_stale and board.temperature_fault == 0 and
                       physical_age < 500.0 and not snapshot.host_stale)
            in_band = abs(board.temperature_mdegc / 1000.0 - target_c) <= 0.3
            if healthy and in_band:
                if first_stable_tick is None:
                    first_stable_tick = board.temperature_sample_tick_ms
                    stable_samples = 0
                    last_tick = None
                if board.temperature_sample_tick_ms != last_tick:
                    stable_samples += 1
                    last_tick = board.temperature_sample_tick_ms
                    last_good_at = time.monotonic()
                sensor_span_ms = ((board.temperature_sample_tick_ms -
                                   first_stable_tick) & 0xFFFFFFFF)
                if stable_samples >= 2 and sensor_span_ms >= 300:
                    return board
            elif (board.temperature_fault != 0 or
                  (board.temperature_valid and not in_band)):
                first_stable_tick = None
                stable_samples = 0
                last_tick = None
                last_good_at = None
        if last_good_at is not None and time.monotonic() - last_good_at > 1.0:
            first_stable_tick = None
            stable_samples = 0
            last_tick = None
            last_good_at = None
        time.sleep(0.02)
    raise TimeoutError("thermal control did not reach the LOCKED entry condition")


def pd_is_fresh(snapshot, board, link_flags, prior_tick, elapsed_ms):
    physical_age_ms = board.pd_age_ms + snapshot.host_age_s * 1000.0
    return (board.pd_sample_tick_ms != prior_tick and board.pd_valid and
            not board.pd_stale and not board.pd_backend_disabled and
            board.pd_fault == 0 and link_flags == 0 and
            board.temperature_valid and not board.temperature_stale and
            board.temperature_fault == 0 and
            physical_age_ms < elapsed_ms and not snapshot.host_stale)


def wait_fresh_pd(ctrl, target, prior_tick, set_ack_at, timeout_s=2.0):
    deadline = time.monotonic() + timeout_s
    next_renew_s = 0.7
    while time.monotonic() < deadline:
        elapsed = time.monotonic() - set_ack_at
        if elapsed >= next_renew_s:
            require_ok(ctrl.manual_command(target, MANUAL_RENEW,
                                           CHANNEL_LASER_DAC), "RENEW")
            next_renew_s += 0.6
        snapshot = ctrl.telemetry
        if snapshot is not None:
            board, link_flags = selected_board(snapshot, target)
            if elapsed >= 0.6 and pd_is_fresh(
                    snapshot, board, link_flags, prior_tick, int(elapsed * 1000)):
                return board, link_flags
        time.sleep(0.02)
    raise TimeoutError("no new, bounded-age, link-healthy 4 Hz PD sample")


def run_manual_sweep(ctrl, target, points=6, on_point=None):
    """Run one bounded sweep and return accepted raw-count row dictionaries."""
    if target not in CAPS:
        raise ValueError("target must be 638 or 1550")
    if not 2 <= points <= 8:
        raise ValueError("points must be between 2 and 8")
    if not ctrl.connected:
        raise RuntimeError("Controller must be connected")
    cap = CAPS[target]
    values = [round(cap * index / (points - 1)) for index in range(points)]
    rows = []
    primary = None
    cleanup = None
    try:
        if not ctrl.streaming:
            ctrl.begin_stream()
        ctrl.stop_confirmed()
        ctrl.enable_telemetry()
        ctrl.begin_manual(target)
        wait_telemetry_ready(ctrl, target)
        take_laser_zero(ctrl, target)
        target_c = ctrl.temperature_target_c(target)
        wait_thermal_locked(ctrl, target, target_c)
        prior_tick = None
        for dac in values:
            reply = require_ok(ctrl.manual_command(
                target, MANUAL_SET, CHANNEL_LASER_DAC, dac), "SET")
            set_ack_at = time.monotonic()
            board, link_flags = wait_fresh_pd(
                ctrl, target, prior_tick, set_ack_at)
            prior_tick = board.pd_sample_tick_ms
            row = {
                "target": target, "dac_requested": dac,
                "dac_applied": reply.applied_value,
                "pd_raw_counts": board.pd_raw_average,
                "pd_age_ms": board.pd_age_ms, "pd_flags": board.pd_flags,
                "pd_fault": board.pd_fault, "link_flags": link_flags,
                "temperature_target_c": target_c,
                "temperature_measured_c": board.temperature_mdegc / 1000.0,
            }
            rows.append(row)
            if on_point is not None:
                on_point(dict(row))
            saturation_limit = min(4090, max(0, board.pd_full_scale - 5))
            if board.pd_raw_average >= saturation_limit:
                raise RuntimeError(
                    f"photodiode saturated at {board.pd_raw_average} counts")
    except Exception as exc:
        primary = exc
    finally:
        try:
            ctrl.finish_manual(target)
        except Exception as exc:
            cleanup = exc
    if primary is not None:
        raise ManualSweepError(primary, rows, cleanup) from primary
    if cleanup is not None:
        raise ManualSweepError(cleanup, rows) from cleanup
    return rows
