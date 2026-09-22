#!/usr/bin/env python3
"""Offline instrumental FRF from a retained, complete 638 identification trace.

Usage: .venv-638/bin/python analyze_identification.py capture.json [-o result.json]
       .venv-638/bin/python analyze_identification.py --self-test
Input is {'trace': read_retained_trace(...), 'config': ..., 'pi': ..., 'live': ...}.
No controller gains are selected by this script.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import signal

NROWS = 4096
NFFT = 512


def reject(reason):
    return {"accepted": False, "reason": reason, "gain_design": {"status": "withheld"}}


def arrays_from_capture(capture):
    trace = capture["trace"]
    rows = trace.get("samples", {})
    if (trace.get("total_samples") != NROWS or not trace.get("complete") or
            trace.get("conflict") or trace.get("missing_indices") or
            trace.get("flags", 0) & 3 != 1 or not trace.get("flags", 0) & 4):
        raise ValueError("trace must be a complete, excited, nonaborted 4096-row capture")
    if len(rows) != NROWS or set(map(int, rows)) != set(range(NROWS)):
        raise ValueError("trace indices are missing or duplicated")
    d = int(trace.get("decimation", 1))
    amp = int(trace.get("amplitude_dac", 2))
    hold = trace.get("hold_updates")
    if not 1 <= d <= 64 or not 1 <= amp <= 64 or hold is None or not 1 <= int(hold) <= 255:
        raise ValueError("decimation/amplitude/hold metadata missing or invalid")
    if bool(trace.get("recorder_filtered")) != (d > 1):
        raise ValueError("recorder filter metadata disagrees with decimation")
    config = capture.get("config") or {}
    if isinstance(config, dict):
        for key, observed in (("amplitude_dac", amp), ("hold_updates", int(hold)), ("decimation", d)):
            if key in config and config[key] != observed:
                raise ValueError(f"trace/config {key} mismatch")
    ordered = [rows[str(i)] if str(i) in rows else rows[i] for i in range(NROWS)]
    if any(row["flags"] != 1 for row in ordered):
        raise ValueError("invalid record flags")
    cycles = np.asarray([row["cycles"] for row in ordered], dtype=np.uint64)
    diffs = ((cycles[1:] - cycles[:-1]) & 0xffffffff).astype(np.float64)
    if not np.all((diffs > 0) & (diffs < 0x80000000)):
        raise ValueError("timestamp reversal or ambiguous wrap")
    dt = float(np.median(diffs))
    if np.any(np.abs(diffs-dt) > .5*dt):
        raise ValueError("record timing gap/jitter exceeds 50% of median")
    clock = float(trace["clock_hz"])
    if clock <= 0:
        raise ValueError("invalid trace clock")
    fs = clock/dt
    expected = config.get("control_rate_hz") if isinstance(config, dict) else None
    if expected is not None:
        expected = float(expected)
        if expected <= 0 or abs(fs*d-expected)/expected > .05:
            raise ValueError("measured control rate differs from configured rate by >5%")
    y = np.asarray([row["feedback"] for row in ordered], dtype=np.float64)
    u = np.asarray([row["actual_dac"] for row in ordered], dtype=np.float64)
    inj = np.asarray([row["injection_dac"] for row in ordered], dtype=np.float64)
    if np.any(np.abs(inj) > amp) or np.std(inj) < 1.0:
        raise ValueError("excitation absent, clipped, or too small after recorder filtering")
    if np.ptp(u) < 6.0 or np.std(u) < 2.0:
        raise ValueError("actual DAC command variation under several codes")
    cap = config.get("dac_cap") if isinstance(config, dict) else None
    if cap is None or not 1 <= float(cap) <= 65535:
        raise ValueError("config.dac_cap is required to check actuator clipping")
    if np.min(u) <= 2 or np.max(u) >= float(cap)-2:
        raise ValueError("DAC command reached an actuator rail")
    if np.min(y) <= 2 or np.max(y) >= 16381:
        raise ValueError("feedback reached an ADC rail")
    return y, u, inj, fs, d, amp, int(hold), dt, diffs, int(np.sum(cycles[1:] < cycles[:-1]))


def estimate(y, u, inj, fs, nfft=NFFT):
    # scipy.signal.csd(x,y) is conj(X)*Y. Both spectra share D*, so the
    # closed-loop instrumental ratio CSD(d,y)/CSD(d,u) estimates Y/U.
    kwargs = dict(fs=fs, window="hann", nperseg=nfft, noverlap=0,
                  detrend="constant", scaling="spectrum")
    f, pdy = signal.csd(inj, y, **kwargs)
    _, pdu = signal.csd(inj, u, **kwargs)
    _, pdd = signal.welch(inj, **kwargs)
    _, cdy = signal.coherence(inj, y, fs=fs, window="hann", nperseg=nfft,
                             noverlap=0, detrend="constant")
    _, cdu = signal.coherence(inj, u, fs=fs, window="hann", nperseg=nfft,
                             noverlap=0, detrend="constant")
    # Nonoverlapping segment jackknife for frequency-dependent uncertainty.
    nseg = len(inj)//nfft
    win = signal.windows.hann(nfft, sym=False)
    dy, du = [], []
    for k in range(nseg):
        sl = slice(k*nfft, (k+1)*nfft)
        df = np.fft.rfft(signal.detrend(inj[sl], type="constant")*win)
        dy.append(np.conj(df)*np.fft.rfft(signal.detrend(y[sl], type="constant")*win))
        du.append(np.conj(df)*np.fft.rfft(signal.detrend(u[sl], type="constant")*win))
    dy, du = np.asarray(dy), np.asarray(du)
    g = pdy/pdu
    loo = np.asarray([(np.sum(np.delete(dy, k, axis=0), axis=0) /
                       np.sum(np.delete(du, k, axis=0), axis=0)) for k in range(nseg)])
    with np.errstate(divide="ignore", invalid="ignore"):
        log_dev = np.log(np.abs(loo)/np.abs(g))
        phase_dev = np.angle(loo/g)
    scale = np.sqrt((nseg-1)/nseg)
    mag95 = 1.96*scale*np.sqrt(np.sum(log_dev**2, axis=0))
    phase95 = 1.96*scale*np.sqrt(np.sum(phase_dev**2, axis=0))
    # Report only the conservative band; 4 boxcars give ~-15.7 dB at
    # output Nyquist, so possible out-of-band aliases remain.
    power_gate = max(float(np.max(pdd[1:]))*.005, 1e-9)
    eligible = ((f > 0) & (f < .2*fs) & (pdd > power_gate) &
                (cdu >= .6) & (cdy >= .5) & np.isfinite(mag95) &
                (mag95 < .7) & (phase95 < .8))
    bins = [{"hz": round(float(f[i]), 3), "gain_adc_per_dac": round(float(abs(g[i])), 4),
             "phase_deg": round(float(np.degrees(np.angle(g[i]))), 2),
             "coherence_dy": round(float(cdy[i]), 3),
             "coherence_du": round(float(cdu[i]), 3),
             "magnitude_95pct_factor": round(float(np.exp(mag95[i])), 3),
             "phase_95pct_deg": round(float(np.degrees(phase95[i])), 2)}
            for i in np.flatnonzero(eligible)]
    return bins, {"candidate_bins": int(np.sum((f > 0) & (f < .2*fs))),
                  "accepted_bins": len(bins), "segment_count": nseg,
                  "frequency_resolution_hz": round(float(fs/nfft), 4)}


def analyze(capture):
    try:
        y, u, inj, fs, d, amp, hold, dt, diffs, wraps = arrays_from_capture(capture)
        bins, quality = estimate(y, u, inj, fs)
    except (KeyError, TypeError, ValueError) as exc:
        return reject(str(exc))
    enough = len(bins) >= 8 and quality["segment_count"] >= 8
    return {"accepted": enough, "reason": None if enough else "too few coherent, precise frequencies",
            "master_feedback_filter": (capture.get("config") or {}).get("master_feedback_filter"),
            "sample_rate_hz": round(fs, 4), "control_rate_hz_estimate": round(fs*d, 4),
            "timestamp_wraps": wraps, "capture_duration_s": round(float(np.sum(diffs)/capture["trace"]["clock_hz"]), 6),
            "median_record_period_cycles": round(dt, 2),
            "record_period_jitter_p95_fraction": round(float(np.percentile(np.abs(diffs-dt), 95)/dt), 5),
            "record_period_jitter_max_fraction": round(float(np.max(np.abs(diffs-dt))/dt), 5),
            "max_record_gap_cycles": int(max(diffs)),
            "record_count": NROWS, "decimation": d, "amplitude_dac": amp,
            "hold_updates": hold, "recorder_filter": "four cascaded length-D boxcars" if d > 1 else "none",
            "recorder_group_delay_s": round(2*(d-1)/(fs*d), 8),
            "fit_limit_hz": round(.2*fs, 4),
            "alias_limit": "Four boxcars attenuate output Nyquist by only ~12 dB at D=2, approaching ~15.7 dB for large D; aliases remain possible.",
            "dac_meaning": "SPI5 DAC launch command, not analog readback; recorded rows cannot exclude subrecord rail clips or intervening saturation",
            "frf_method": "instrumental CSD(injection,feedback)/CSD(injection,DAC command); 8 nonoverlapping Hann segments, leave-one-segment-out uncertainty",
            "quality": quality, "frf": bins,
            "gain_design": {"status": "withheld", "reason": "Requires independent plant validation, stability/delay and saturation margins, and disturbance trials before joint Kp/Ki selection."}}


def self_test():
    rng = np.random.default_rng(7)
    n = 4096
    d = rng.choice([-32., 32.], n)
    y = np.zeros(n)
    u = np.zeros(n)
    a, plant, controller = .75, .6, .45
    for i in range(1, n):
        u[i] = d[i] - controller*y[i-1]
        y[i] = a*y[i-1] + (1-a)*plant*u[i-1] + rng.normal(0, .05)
    bins, q = estimate(y, u, d, 1000.)
    assert q["accepted_bins"] >= 8
    for item in bins[:8]:
        omega = 2*np.pi*item["hz"]/1000.
        truth = (1-a)*plant*np.exp(-1j*omega)/(1-a*np.exp(-1j*omega))
        assert abs(item["gain_adc_per_dac"]-abs(truth))/abs(truth) < .25
    trace = {"total_samples": n, "complete": True, "conflict": False,
             "missing_indices": [], "flags": 5, "decimation": 1,
             "amplitude_dac": 32, "hold_updates": 1,
             "recorder_filtered": False, "clock_hz": 250000000,
             "samples": {str(i): {"cycles": (0xfffff000+i*250000) & 0xffffffff,
                                  "feedback": int(round(6000+y[i])),
                                  "actual_dac": int(round(40000+u[i])),
                                  "injection_dac": int(d[i]), "flags": 1}
                         for i in range(n)}}
    result = analyze({"trace": trace, "config": {"dac_cap": 52400,
                      "amplitude_dac": 32, "hold_updates": 1, "decimation": 1,
                      "control_rate_hz": 1000}})
    assert result["accepted"] and result["quality"]["accepted_bins"] >= 8
    assert result["timestamp_wraps"] > 0
    bad = dict(trace, samples={k: v for k, v in trace["samples"].items() if k != "2000"})
    assert not analyze({"trace": bad, "config": {"dac_cap": 52400}})["accepted"]
    return {"passed": True, "accepted_bins": q["accepted_bins"],
            "model": "closed-loop first-order plant with external injection and wrapped timestamps"}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("capture", nargs="?", type=Path)
    p.add_argument("--output", "-o", type=Path)
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        print(json.dumps(self_test(), indent=2))
    else:
        if a.capture is None:
            p.error("capture.json is required unless --self-test")
        result = analyze(json.loads(a.capture.read_text()))
        target = a.output or a.capture.with_name(a.capture.stem + "-frf.json")
        target.write_text(json.dumps(result, indent=2) + "\n")
        print(f"{target}: {'accepted' if result['accepted'] else 'rejected'}")
