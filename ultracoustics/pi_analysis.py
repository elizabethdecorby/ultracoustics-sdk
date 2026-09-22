#!/usr/bin/env python3
"""Offline adjacent-half FRF, diagnostic fit, and exploratory PI screening.

The capture is D=4, hold=4, amplitude 8-64. No device writes occur. Call
``review(capture, crossover_only=True)`` with a complete indexed trace.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from ._internal import pi_frf as frf
from ._internal import pi_model as pi


def robust_screen(fit, control_rate, kp_cap, ki_cap, empirical_pairs, baseline_kp=.2, baseline_ki=750.):
    """Joint search ranked by the slowest member of the sensitivity family."""
    rate = float(control_rate)
    if not 1000 <= rate <= 10000:
        raise ValueError("control rate outside evaluated 1-10 kHz range")
    period = 1 / rate
    measured = min(400., fit["overlap_max_hz"])
    freq = np.geomspace(.4, .95 * rate / 2, 1000)
    zinv = np.exp(-2j * np.pi * freq * period)
    family = [pi.model(freq, fit["gain_adc_per_dac"] * gs,
                       fit["thermal_tau_s"] * ts,
                       fit["extra_delay_s"] + ticks * period,
                       np.ones_like(freq)) / fit["slope_adc_per_dac"]
              for gs in (.8, 1., 1.2) for ts in (.5, 1., 2.)
              for ticks in (0, 1, 2)]
    def assess(kp, ki):
        c = kp + ki * period / (1 - zinv)
        margins = [pi.margins(plant * c, freq) for plant in family]
        if any(m is None for m in margins):
            return None
        gm = [m[2] for m in margins if m[2] is not None]
        return {"kp": round(float(kp), 3), "ki_per_s": round(float(ki), 1),
                "min_crossover_hz": round(min(m[0] for m in margins), 2),
                "max_crossover_hz": round(max(m[0] for m in margins), 2),
                "worst_phase_margin_deg": round(min(m[1] for m in margins), 1),
                "worst_gain_margin_db": round(min(gm), 1) if gm else None,
                "supported_by_current_runtime": bool(kp <= kp_cap and ki <= ki_cap)}
    baseline = assess(baseline_kp, baseline_ki)
    candidates = []
    for kp in np.linspace(.05, 2., 28):
        for ki in np.geomspace(25., 5000., 34):
            kp, ki = round(float(kp), 3), round(float(ki), 1)
            if (kp, ki) not in empirical_pairs:
                continue
            item = assess(kp, ki)
            if item is None or item["worst_phase_margin_deg"] < 45:
                continue
            gm = item["worst_gain_margin_db"]
            if gm is not None and gm < 6:
                continue
            if item["max_crossover_hz"] > .7 * measured:
                continue
            candidates.append(item)
    candidates.sort(key=lambda v: (-v["min_crossover_hz"],
                                   -v["worst_phase_margin_deg"]))
    return {"baseline": baseline, "candidate_count": len(candidates),
            "best_unrestricted": candidates[:6],
            "best_supported": [v for v in candidates if v["supported_by_current_runtime"]][:6],
            "ranking": "largest minimum crossover across 27 gain/pole/delay sensitivity cases",
            "measured_support_hz": [10, measured],
            "crossover_limit_hz": .7 * measured,
            "margin_status": "phase/gain margins come from extrapolating the fitted model beyond measured support; they are screening estimates, not measured margins",
            "status": "exploratory offline candidates; no gain applied"}


def empirical_screen(first, second, slope, rate, kp_cap, ki_cap, crossover_only=False,
                     baseline_kp=.2, baseline_ki=750.):
    """Compare independent halves and screen PI only inside their common band."""
    a = {r["hz"]: r for r in first if 10 <= r["hz"] <= 400}
    b = {r["hz"]: r for r in second if 10 <= r["hz"] <= 400}
    hz = np.asarray(sorted(set(a) & set(b)), dtype=float)
    floor = 50 if crossover_only else 20
    if len(hz) < 20 or hz[0] > floor or hz[-1] < 300:
        return {"accepted": False, "reason": f"common coherent band lacks 20 bins or {floor}-300 Hz coverage"}
    def complex_rows(table):
        r = [table[f] for f in hz]
        g = np.asarray([v["gain_adc_per_dac"] *
                        np.exp(1j*np.deg2rad(v["phase_deg"])) for v in r])
        sm = np.asarray([max(.07, math.log(v["magnitude_95pct_factor"])/1.96) for v in r])
        sp = np.asarray([max(.07, math.radians(v["phase_95pct_deg"])/1.96) for v in r])
        return g, sm, sp
    ga, ma, pa = complex_rows(a)
    gb, mb, pb = complex_rows(b)
    mismatch = np.r_[np.log(abs(ga/gb))/np.hypot(ma, mb),
                     np.angle(ga/gb)/np.hypot(pa, pb)]
    mismatch_rms = float(np.sqrt(np.mean(mismatch**2)))
    agreement = {"common_bin_count": len(hz), "band_hz": [float(hz[0]), float(hz[-1])],
                 "half_difference_rms_sigma": mismatch_rms,
                 "gate_rms_sigma": 1.5}
    if mismatch_rms > 1.5:
        return {"accepted": False, "reason": "adjacent empirical FRFs disagree", "agreement": agreement}
    period = 1/float(rate)
    zinv = np.exp(-2j*np.pi*hz*period)
    plants = [ga/slope, gb/slope]
    band = (hz >= (50 if crossover_only else 20)) & (hz <= (100 if crossover_only else 60))
    def assess(kp, ki):
        controller = kp + ki*period/(1-zinv)
        checks = []
        for plant in plants:
            loop = plant*controller
            mag = abs(loop)
            crossings = np.flatnonzero((mag[:-1]-1)*(mag[1:]-1) <= 0)
            if not len(crossings):
                return None
            phase = np.unwrap(np.angle(loop))
            margins = []
            for i in crossings:
                t = (1-mag[i])/(mag[i+1]-mag[i]) if mag[i+1] != mag[i] else 0
                fc = float(hz[i]+t*(hz[i+1]-hz[i]))
                pm = float(180+np.rad2deg(phase[i]+t*(phase[i+1]-phase[i])))
                margins.append((fc, pm))
            s = abs(1/(1+loop))
            checks.append((min(v[0] for v in margins), max(v[0] for v in margins),
                           min(v[1] for v in margins),
                           float(np.max(s)), float(np.min(abs(1+loop))),
                           float(20*np.log10(np.mean(s[band])))))
        return {"kp": round(float(kp), 3), "ki_per_s": round(float(ki), 1),
                "min_crossover_hz": round(min(v[0] for v in checks), 2),
                "max_crossover_hz": round(max(v[1] for v in checks), 2),
                "worst_phase_margin_deg": round(min(v[2] for v in checks), 1),
                "max_inband_sensitivity": round(max(v[3] for v in checks), 3),
                "min_inband_return_distance": round(min(v[4] for v in checks), 3),
                "worst_lowband_sensitivity_db": round(max(v[5] for v in checks), 2),
                "supported_by_current_runtime": bool(kp <= kp_cap and ki <= ki_cap)}
    baseline = assess(baseline_kp, baseline_ki)
    if baseline is None:
        return {"accepted": False, "reason": "baseline crossover is outside or unresolved inside common measured band",
                "agreement": agreement}
    candidates = []
    for kp in np.linspace(.05, 2., 28):
        for ki in np.geomspace(25., 5000., 34):
            kp, ki = round(float(kp), 3), round(float(ki), 1)
            item = assess(kp, ki)
            if item is None or item["worst_phase_margin_deg"] < 45:
                continue
            if item["max_crossover_hz"] > .7*hz[-1]:
                continue
            if item["max_inband_sensitivity"] > 2 or item["min_inband_return_distance"] < .5:
                continue
            if (item["worst_lowband_sensitivity_db"] >
                    baseline["worst_lowband_sensitivity_db"]-.5 or
                    item["max_inband_sensitivity"] > baseline["max_inband_sensitivity"]+.1):
                continue
            candidates.append(item)
    candidates.sort(key=lambda v: (v["worst_lowband_sensitivity_db"],
                                   -v["min_crossover_hz"]))
    return {"accepted": bool(candidates), "agreement": agreement,
            "baseline": baseline, "candidate_count": len(candidates),
            "_passing_candidates": candidates,
            "best_unrestricted": candidates[:6],
            "best_supported": [v for v in candidates if v["supported_by_current_runtime"]][:6],
            "measured_support_hz": [float(hz[0]), float(hz[-1])],
            "screen_mode": "crossover-only, 50-100 Hz comparison" if crossover_only else "full 20-60 Hz comparison",
            "low_frequency_limit": "DC and integral behavior below first qualified bin inferred from static slope and model, not measured dynamically" if crossover_only else None,
            "status": "in-band exploratory screen only; no high-frequency stability qualification or gain application"}


def review(capture, crossover_only=False):
    try:
        y, u, d, fs, dec, amp, hold, _, diffs, _ = frf.arrays_from_capture(capture)
        if (dec, hold) != (4, 4):
            raise ValueError("paired check requires decimation=4 and hold_updates=4")
        if capture.get("config", {}).get("master_feedback_filter") != "single_sample":
            raise ValueError("master feedback filter must be explicitly single_sample")
        # live_after follows slow USB page replay; only this immediate snapshot
        # is close enough to the excitation interval for an operating-point gate.
        pre, post = capture["live"], capture["capture_live_after"]
        if pre["state"] != 3 or post["state"] != 3:
            raise ValueError("pre/post device state must remain locked (3)")
        if pre["gain_law"] != post["gain_law"] or pre["target"] != post["target"]:
            raise ValueError("gain law or target changed across capture")
        slope0, slope1 = float(pre["slope"]), float(post["slope"])
        if slope0 * slope1 <= 0 or min(abs(slope0), abs(slope1)) < 1.5:
            raise ValueError("pre/post signed slope absent, weak, or sign-changing")
        slope_change = abs(slope1 / slope0 - 1)
        if slope_change > .2:
            raise ValueError("pre/post slope changed by more than 20%")
        dac_live_delta = float(post["dac"])-float(pre["dac"])
        dac_half_delta = float(np.mean(u[2048:])-np.mean(u[:2048]))
        if abs(dac_half_delta) > 100:
            raise ValueError("recorded DAC half means moved by more than 100 codes")
        if np.percentile(np.abs(diffs-np.median(diffs)), 95) / np.median(diffs) > .05:
            raise ValueError("record timing jitter exceeds 5% at p95")
        first, q1 = frf.estimate(y[:2048], u[:2048], d[:2048], fs, nfft=256)
        second, q2 = frf.estimate(y[2048:], u[2048:], d[2048:], fs, nfft=256)
        for label, rows, q in (("first", first, q1), ("second", second, q2)):
            supported = [v for v in rows if 10 <= v["hz"] <= 400]
            if q["segment_count"] != 8 or len(supported) < 20:
                raise ValueError(f"{label} half lacks 8 segments and 20 qualified 10-400 Hz bins")
        def pack(rows):
            return {"accepted": True, "master_feedback_filter": "single_sample",
                    "frf": [r for r in rows if 10 <= r["hz"] <= 400]}
        fitted, taps = pi.fit({"low": pack(first), "high": pack(second),
                               "live": {"slope": slope0},
                               "config": {"master_feedback_filter": "single_sample"}})
        output = {"accepted": False, "capture_duration_s": float(np.sum(diffs)/capture["trace"]["clock_hz"]),
                  "sample_rate_hz": fs, "amplitude_dac": amp,
                  "first_half": {"quality": q1, "dac_mean": float(np.mean(u[:2048])),
                                 "feedback_mean": float(np.mean(y[:2048])), "frf": first},
                  "second_half": {"quality": q2, "dac_mean": float(np.mean(u[2048:])),
                                  "feedback_mean": float(np.mean(y[2048:])), "frf": second},
                  "pre_post_slope_change_fraction": slope_change,
                  "pre_post_live_dac_delta": dac_live_delta,
                  "recorded_half_dac_delta": dac_half_delta,
                  "operating_point_warning": bool(abs(dac_live_delta) > 100),
                  "plant_fit": fitted,
                  "limitations": ["Adjacent frequency bins and jackknife intervals are approximate, not independent confidence tests.",
                                  "SPI5 DAC command is not an analog output measurement.",
                                  "One 1.64 s record tests a local plant; repeat and motion trials remain necessary before gain deployment."]}
        runtime = capture.get("runtime_caps", {})
        current = capture.get("pi") or {}
        baseline_kp = float(current.get("kp", .2))
        baseline_ki = float(current.get("ki_per_s", 750.))
        empirical = empirical_screen(first, second, slope0,
                                     float(capture["config"]["control_rate_hz"]),
                                     float(runtime.get("kp_max", 1)),
                                     float(runtime.get("ki_max_per_s", 1000)),
                                     crossover_only=crossover_only,
                                     baseline_kp=baseline_kp, baseline_ki=baseline_ki)
        empirical_candidates = empirical.pop("_passing_candidates", [])
        empirical_by_pair = {(v["kp"], v["ki_per_s"]): v for v in empirical_candidates}
        empirical_pairs = set(empirical_by_pair)
        output["empirical_screen"] = empirical
        if not empirical["accepted"]:
            output["reason"] = empirical.get("reason", "no empirical PI pair passed the in-band screen")
            return output
        if not fitted["validated_for_screening"]:
            output.update(accepted=True, basis="empirical halves only",
                          reason="one-pole model failed; empirical halves agreed",
                          screen=empirical)
            return output
        screen = robust_screen(fitted, float(capture["config"]["control_rate_hz"]),
                               float(runtime.get("kp_max", 1)),
                               float(runtime.get("ki_max_per_s", 1000)), empirical_pairs,
                               baseline_kp=baseline_kp, baseline_ki=baseline_ki)
        screen["measured_support_hz"] = empirical["measured_support_hz"]
        for category in ("best_supported", "best_unrestricted"):
            for item in screen[category]:
                same = empirical_by_pair[(item["kp"], item["ki_per_s"])]
                item["empirical_half_worst"] = {
                    key: same[key] for key in ("min_crossover_hz", "max_crossover_hz",
                                                "worst_phase_margin_deg", "max_inband_sensitivity",
                                                "worst_lowband_sensitivity_db")}
        output.update(accepted=bool(screen["candidate_count"]),
                      basis="model and empirical halves",
                      reason=None if screen["candidate_count"] else "no joint PI pair passed both model and empirical gates",
                      screen=screen)
        if crossover_only:
            band = empirical["measured_support_hz"]
            output["assessment_scope"] = (f"exploratory crossover screen only: measured {band[0]:g}-{band[1]:g} Hz; "
                                          "low-frequency Ki and high-frequency stability unqualified")
        return output
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        return {"accepted": False, "reason": str(exc), "screen": None}


def plot_review(result, target):
    """Plot measured halves and nominal model within the measured band."""
    if "plant_fit" not in result:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fit = result["plant_fit"]
    support_min = max(min(r["hz"] for r in result["first_half"]["frf"]),
                      min(r["hz"] for r in result["second_half"]["frf"]))
    support_max = min(400, fit["overlap_max_hz"])
    f = np.geomspace(support_min, support_max, 300)
    plant = pi.model(f, fit["gain_adc_per_dac"], fit["thermal_tau_s"],
                     fit["extra_delay_s"], np.ones_like(f))
    fig, ax = plt.subplots(3, 1, figsize=(8, 9), constrained_layout=True)
    for key, label, color in (("first_half", "Fit half", "#27649a"),
                               ("second_half", "Held-out half", "#d27720")):
        rows = [r for r in result[key]["frf"] if 10 <= r["hz"] <= 400]
        hz = [r["hz"] for r in rows]
        gain = [r["gain_adc_per_dac"] for r in rows]
        phase = [r["phase_deg"] for r in rows]
        ax[0].scatter(hz, gain, s=10, label=label, color=color, alpha=.75)
        ax[1].scatter(hz, phase, s=10, label=label, color=color, alpha=.75)
    ax[0].plot(f, abs(plant), color="black", label="Fitted plant")
    ax[1].plot(f, np.rad2deg(np.angle(plant)), color="black")
    ax[0].set_ylabel("ADC / DAC magnitude")
    ax[1].set_ylabel("Phase (degrees)")
    ax[0].legend()
    for a in ax[:2]:
        a.set_xscale("log")
        a.grid(True, alpha=.25)
    slope = fit["slope_adc_per_dac"]
    normalized = plant / slope
    period = 1 / 10000
    zinv = np.exp(-2j * np.pi * f * period)
    pairs = [("Current PI", .2, 750., "#666666")]
    screened = result.get("screen") or {}
    supported = screened.get("best_supported") or []
    if supported:
        top = supported[0]
        pairs.append((f"Top screen {top['kp']:.3f}/{top['ki_per_s']:.1f}",
                      top["kp"], top["ki_per_s"], "#008268"))
    for label, kp, ki, color in pairs:
        loop = normalized * (kp + ki * period / (1-zinv))
        ax[2].plot(f, 20*np.log10(abs(1/(1+loop))), label=label, color=color)
    ax[2].set_xscale("log")
    ax[2].set_ylabel("Nominal sensitivity (dB)")
    ax[2].set_xlabel(f"Frequency (Hz); qualified overlap {support_min:.1f}–{support_max:.1f} Hz")
    ax[2].grid(True, alpha=.25)
    ax[2].legend()
    fig.suptitle("Adjacent-half identification (model curves are screening estimates)")
    fig.savefig(target, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--crossover-only", action="store_true",
                        help="separate exploratory 50-390 Hz crossover assessment")
    args = parser.parse_args()
    result = review(json.loads(args.capture.read_text()), crossover_only=args.crossover_only)
    output = args.output or args.capture.with_name(args.capture.stem + "-paired-review.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    plot_review(result, output.with_suffix(".png"))
    print(f"{output}: {'accepted' if result['accepted'] else 'withheld'}: {result.get('reason')}")
