"""Bounded, user-triggered 638 PI characterization on an existing RUN session.

This records one local plant trace and produces evidence for human review. It
never starts/stops the system, changes PI gains, or controls a robot.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from .pi_analysis import review

COUNTERS = ("drops_seq", "drops_fw", "transfer_errors", "transfer_timeouts", "malformed")


class _Cancelled(Exception):
    pass


def _plain(value):
    if is_dataclass(value):
        return {k: _plain(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if hasattr(value, "__dict__"):
        return _plain(vars(value))
    return value


def _live(controller):
    telemetry = controller.telemetry
    cached = getattr(telemetry, "optical_live_638", None) if telemetry is not None else None
    return cached.page if cached is not None and not cached.host_stale else None


def _counter_deltas(before, after):
    return {key: int(after.get(key, 0)) - int(before.get(key, 0)) for key in COUNTERS}


def _trace_array(trace):
    rows = trace["samples"]
    ordered = [rows[i] if i in rows else rows[str(i)] for i in range(trace["total_samples"])]
    return np.asarray([[r.cycles, r.feedback, r.actual_dac, r.injection_dac, r.flags]
                       if is_dataclass(r) else
                       [r[k] for k in ("cycles", "feedback", "actual_dac", "injection_dac", "flags")]
                       for r in ordered], dtype=np.int64)


def run_pi_characterization(controller, report_dir, progress=None, cancel=None,
                            verified_master_filter=None):
    """Capture one 4096-row D4/hold4/8-DAC trace and write a reviewable report.

    ``controller`` must already be connected, streaming format-2 diagnostics,
    and locked in automatic RUN. ``progress`` receives a stage dictionary;
    ``cancel`` is a zero-argument predicate. Routine rejection/cancellation is
    returned as a report, with partial evidence retained in ``report_dir``.
    ``verified_master_filter='single_sample'`` is an explicit caller
    assertion from installation evidence; absent that, measured FRFs remain
    available but gain screening is unqualified. No other thread should issue
    Controller commands during this call.
    """
    output = Path(report_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    last_fresh_lock = started
    partial_trace = None
    configured = False
    report = {"schema": 1, "status": "failed", "created_utc": datetime.now(timezone.utc).isoformat(),
              "report_dir": str(output), "board_serial": getattr(controller, "device_serial", None),
              "limitations": ["One local 1.64 s identification record is insufficient to deploy gains.",
                              "SPI5 DAC commands are not analog output measurements.",
                              "Only 10–400 Hz may be supported; low-frequency integral and high-frequency stability are not directly measured.",
                              "Master feedback filter has no active-session readback; the caller must verify it independently before gain screening."],
              "configuration": {"amplitude_dac": 8, "hold_updates": 4, "decimation": 4,
                                "master_feedback_filter": verified_master_filter or "unverified"}}

    def emit(stage, message, **extra):
        if progress:
            event = dict(stage=stage, message=message, elapsed_s=round(time.monotonic()-started, 2))
            event.update(extra)
            progress(event)

    def check_cancel():
        if cancel and cancel():
            raise _Cancelled("Cancelled by user")

    def sleep(seconds, require_lock=True):
        nonlocal last_fresh_lock
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            check_cancel()
            live = _live(controller)
            if require_lock:
                if live is not None:
                    if live.state != 3:
                        raise RuntimeError(f"638 left locked state: {live.state}")
                    last_fresh_lock = time.monotonic()
                elif time.monotonic() - last_fresh_lock > 2:
                    raise RuntimeError("No fresh locked telemetry for two seconds")
            stats = controller.stream_stats or {}
            if stats.get("device_lost") or stats.get("fatal"):
                raise RuntimeError("ADC stream ended")
            time.sleep(min(.05, max(0, until-time.monotonic())))

    try:
        check_cancel()
        emit("preflight", "Checking board, firmware, and live lock")
        if getattr(controller, "system_manual_active", False) or not getattr(controller, "_running", True):
            raise RuntimeError("Characterization requires automatic RUN, not manual or IDLE")
        stats = controller.stream_stats or {}
        stream_deadline = time.monotonic()+2
        while stats.get("stream_format") != 2 and time.monotonic() < stream_deadline:
            sleep(.05, require_lock=False)
            stats = controller.stream_stats or {}
        if stats.get("stream_format") != 2:
            raise RuntimeError("Optical diagnostic stream format 2 is required")
        lock_deadline = time.monotonic()+45
        live = _live(controller)
        while (live is None or live.state != 3) and time.monotonic() < lock_deadline:
            sleep(.1, require_lock=False)
            live = _live(controller)
        if live is None or live.state != 3:
            raise RuntimeError("A fresh healthy 638 lock was not reached within 45 seconds")
        if abs(live.slope) < 1.5:
            raise RuntimeError("A valid signed slope is required")
        report["profile"] = controller.read_profile_638()
        report["pi_before"] = controller.read_normalized_pi_638()
        report["optical_cap"] = controller.read_optical_cap_638()
        report["timing"] = controller.read_controller_timing_638()
        report["trace_config_before"] = controller.read_control_trace_config_638()
        original_config = report["trace_config_before"]
        configuration_differs = tuple(original_config[k] for k in ("amplitude_dac", "hold_updates", "decimation")) != (8, 4, 4)
        report["firmware_provenance"] = {
            "normalized_pi_contract_version": report["pi_before"].get("version"),
            "master_image": "unavailable from active streaming session",
            "638_image": "unavailable from active streaming session"}
        if report["profile"]["id"] not in (1, 2):
            raise RuntimeError("Unsupported 638 board profile")
        if not report["pi_before"]["active"] or not report["pi_before"]["slope_valid"]:
            raise RuntimeError("Normalized PI is inactive or its slope is invalid")
        if report["timing"]["divider"] != 1 or report["timing"]["held"]:
            raise RuntimeError("Characterization requires 10 kHz unheld control")
        cap = report["optical_cap"]["max_dac"]
        if not 500 < live.dac < cap - 500 or abs(live.slope) < 1.5:
            raise RuntimeError("Insufficient DAC headroom or slope")
        report["live_preflight"] = _plain(live)
        # Distinct live sequences over a recent 15 s window; 120 s maximum.
        emit("settling", "Waiting for a stable operating point")
        settle_started = time.monotonic()
        history = []
        last_sequence = None
        last_settle_notice = 0
        settled = False
        while time.monotonic() - settle_started < 120:
            sleep(.2)
            live = _live(controller)
            if live is None or live.sequence == last_sequence:
                continue
            last_sequence = live.sequence
            now = time.monotonic()
            history.append((now, live.dac, live.feedback-live.target, live.slope))
            recent = np.asarray([row for row in history if row[0] >= now-15], dtype=float)
            if len(recent) < 15 or recent[-1, 0]-recent[0, 0] < 14:
                continue
            drift = float(np.polyfit(recent[:, 0]-recent[0, 0], recent[:, 1], 1)[0])
            settling = {"elapsed_s": round(now-settle_started, 2),
                        "drift_dac_per_s": round(drift, 3), "dac_span": float(np.ptp(recent[:, 1])),
                        "median_abs_error": float(np.median(abs(recent[:, 2]))),
                        "slope_span": float(np.ptp(recent[:, 3]))}
            if now-settle_started >= 20 and abs(drift) <= 3 and settling["dac_span"] <= 100 and settling["median_abs_error"] <= 60 and settling["slope_span"] <= .05:
                report["settling"] = settling
                settled = True
                break
            notice = int(now-settle_started)//10
            if notice > last_settle_notice:
                last_settle_notice = notice
                emit("settling", "Operating point still settling", **settling)
        if not settled:
            raise RuntimeError("Settling gate not reached within 120 seconds")
        # A bounded configuration write; no PI or laser operating-point change.
        deadline = time.monotonic()+30
        while True:
            check_cancel()
            try:
                configured = configuration_differs  # SET may succeed even if its readback fails.
                config = controller.configure_control_trace_638(8, 4, 4)
                break
            except RuntimeError as exc:
                if "status 10" not in str(exc) or time.monotonic() >= deadline:
                    raise
                sleep(1)
        config.update(dac_cap=cap, master_feedback_filter=verified_master_filter or "unverified",
                      control_rate_hz=10000)
        report["configuration"] = config
        sleep(.3)
        live = _live(controller)
        if live is None or live.state != 3 or not 500 < live.dac < cap-500:
            raise RuntimeError("Lock or DAC headroom changed before capture")
        old = controller.retained_control_trace()
        old_id = (old["stream_epoch"], old["capture_id"]) if old else None
        report["live_before"] = _plain(live)
        report["stats_before"] = dict(controller.stream_stats or {})
        emit("capture", "Recording bounded 638 excitation")
        check_cancel()
        controller.control_638("identify")
        triggered = time.monotonic()
        deadline = triggered+240
        trace = None
        post_taken = False
        last_replay_notice = 0
        while time.monotonic() < deadline:
            sleep(.5)
            if not post_taken and time.monotonic()-triggered >= 2:
                post = _live(controller)
                if post is not None:
                    report["capture_live_after"] = _plain(post)
                    report["capture_after_stats"] = dict(controller.stream_stats or {})
                    report["capture_live_after_host_delay_s"] = round(time.monotonic()-triggered, 3)
                    post_taken = True
            trace = controller.retained_control_trace()
            identity = (trace["stream_epoch"], trace["capture_id"]) if trace else None
            if trace and identity != old_id:
                partial_trace = trace
                report["trace_progress"] = {"capture_id": trace["capture_id"],
                                            "received_rows": len(trace["samples"]),
                                            "total_rows": trace["total_samples"]}
                if trace["conflict"] or trace["flags"] & 2:
                    raise RuntimeError("Trace was aborted or conflicted")
                if trace["complete"]:
                    break
            notice = int(time.monotonic()-triggered)//10
            if notice > last_replay_notice:
                last_replay_notice = notice
                emit("replay", "Receiving indexed trace", received_rows=len(trace["samples"]) if trace and identity != old_id else 0)
        if not trace or (trace["stream_epoch"], trace["capture_id"]) == old_id or not trace["complete"]:
            raise TimeoutError("Complete trace not received within 240 seconds")
        report["stats_after"] = dict(controller.stream_stats or {})
        report["live_after_replay"] = _plain(_live(controller))
        if not post_taken:
            raise RuntimeError("No fresh lock snapshot near the capture interval")
        if report["capture_live_after_host_delay_s"] > 3:
            raise RuntimeError("Post-capture live snapshot arrived too late for operating-point check")
        rows = _trace_array(trace)
        np.savez_compressed(output/"trace.npz", rows=rows)
        report["trace_file"] = "trace.npz"
        report["trace_metadata"] = {key: value for key, value in trace.items() if key not in ("samples", "missing_indices")}
        capture_deltas = _counter_deltas(report["stats_before"], report["capture_after_stats"])
        report["capture_counter_deltas"] = capture_deltas
        report["replay_counter_deltas"] = _counter_deltas(report["capture_after_stats"], report["stats_after"])
        if any(value != 0 for value in capture_deltas.values()):
            raise RuntimeError("ADC/USB integrity counters changed during capture")
        emit("analysis", "Estimating measured response and diagnostic PI screen")
        capture = {"trace": dict(trace, samples={i: dict(zip(("cycles", "feedback", "actual_dac", "injection_dac", "flags"), map(int, row))) for i, row in enumerate(rows)}),
                   "config": config, "pi": report["pi_before"], "live": report["live_before"],
                   "capture_live_after": report["capture_live_after"],
                   "runtime_caps": {"kp_max": 1, "ki_max_per_s": max(1000, report["pi_before"]["ki_per_s"])}}
        report["runtime_cap_inference"] = "Ki support is conservatively bounded by 1000/s or the current readback, whichever is greater; this protocol does not expose the firmware maximum."
        # The spectral FRF can be inspected without the filter assertion, but
        # the fitted phase and candidate screen require known master timing.
        if verified_master_filter == "single_sample":
            analysis = review(capture, crossover_only=True)
        else:
            from ._internal import pi_frf
            y, u, d, fs, *_ = pi_frf.arrays_from_capture(capture)
            first, quality_first = pi_frf.estimate(y[:2048], u[:2048], d[:2048], fs, nfft=256)
            second, quality_second = pi_frf.estimate(y[2048:], u[2048:], d[2048:], fs, nfft=256)
            analysis = {"accepted": False, "reason": "Master feedback filter has not been independently verified",
                        "first_half": {"quality": quality_first, "frf": first},
                        "second_half": {"quality": quality_second, "frf": second},
                        "screen": None}
        if analysis.get("operating_point_warning"):
            analysis["accepted"] = False
            analysis["reason"] = "Pre/post DAC operating point shifted by more than 100 codes; gain candidates withheld"
        report["analysis"] = analysis
        report["status"] = "complete" if analysis.get("accepted") else "unqualified"
        report["reason"] = analysis.get("reason")
        emit("done", "Characterization report written", status=report["status"])
    except _Cancelled as exc:
        report.update(status="cancelled", reason=str(exc))
    except Exception as exc:
        report.update(status="unqualified", reason=f"{type(exc).__name__}: {exc}")
    finally:
        if configured:
            old_config = report.get("trace_config_before") or {}
            try:
                controller.configure_control_trace_638(old_config["amplitude_dac"],
                                                       old_config["hold_updates"],
                                                       old_config["decimation"])
                report["trace_config_restored"] = True
            except Exception as exc:
                report["trace_config_restored"] = False
                report["trace_config_restore_error"] = str(exc)
                if report["status"] == "complete":
                    report["status"] = "unqualified"
                    report["reason"] = "Original trace configuration could not be restored"
        if partial_trace is not None and "trace_file" not in report:
            try:
                # A failed or cancelled replay can still leave useful indexed
                # rows. Preserve them without claiming a complete capture.
                indices = sorted(partial_trace["samples"])
                partial = [partial_trace["samples"][i] for i in indices]
                values = np.asarray([[r.cycles, r.feedback, r.actual_dac, r.injection_dac, r.flags]
                                     if is_dataclass(r) else
                                     [r[k] for k in ("cycles", "feedback", "actual_dac", "injection_dac", "flags")]
                                     for r in partial], dtype=np.int64)
                np.savez_compressed(output/"trace-partial.npz", indices=np.asarray(indices), rows=values)
                report["partial_trace_file"] = "trace-partial.npz"
            except Exception as exc:
                report["partial_trace_save_error"] = str(exc)
        report["elapsed_s"] = round(time.monotonic()-started, 2)
        report["summary_file"] = "summary.json"
        report["report_file"] = "report.md"
        (output/"summary.json").write_text(json.dumps(_plain(report), indent=2, allow_nan=False)+"\n")
        status = report["status"]
        fit = (report.get("analysis") or {}).get("plant_fit") or {}
        screen = (report.get("analysis") or {}).get("screen") or {}
        candidates = (screen.get("best_supported") or []) if status == "complete" else []
        lines = ["# 638 PI characterization", "", f"Status: **{status}**", f"Board: {report.get('board_serial') or 'unknown'}", f"Profile: {(report.get('profile') or {}).get('name', 'unknown')}", f"Reason: {report.get('reason') or 'See measured evidence below.'}", f"Master feedback filter assertion: {report['configuration'].get('master_feedback_filter')}", f"Original trace configuration restored: {report.get('trace_config_restored', 'not changed')}", f"Trace configuration restore error: {report.get('trace_config_restore_error', 'none')}", "", "## Measured evidence", "", f"Current gains: {report.get('pi_before')}", f"Trace: {report.get('trace_progress')}", f"Capture counter changes: {report.get('capture_counter_deltas')}", f"Replay counter changes: {report.get('replay_counter_deltas')}", f"Diagnostic fit: {fit}", "", "## Exploratory candidates", ""]
        lines.extend(f"- Kp {item['kp']}, Ki {item['ki_per_s']}/s; worst estimated phase margin {item['worst_phase_margin_deg']}°" for item in candidates)
        if not candidates:
            lines.append("No candidate qualified for display from this record.")
        lines += ["", "## Limits", ""] + [f"- {item}" for item in report["limitations"]]
        lines += ["", "Detailed frequency response, quality diagnostics, and screening evidence: `summary.json`. Bounded indexed trace: `trace.npz` when capture completed.", "No gains were applied by this run.", ""]
        (output/"report.md").write_text("\n".join(lines))
    return report
