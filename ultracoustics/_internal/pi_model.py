#!/usr/bin/env python3
"""Offline plant fit and joint normalized-PI screening; never applies gains.

Input JSON: {"low": <analyze_identification result>, "high": <result>,
             "live": {"slope": signed ADC/DAC},
             "config": {"control_rate_hz": 10000,
                        "master_feedback_filter": "single_sample" or "hann2000",
                        "runtime_kp_max": 1,
                        "runtime_ki_max_per_s": 1000}}.
CLI: .venv-638/bin/python evaluate_joint_pi.py bundle.json [-o review.json]
"""
import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ADC_HZ = 10_000_000.0

def driver_pole_config(config):
    """Validate an explicit first-order driver-pole prior and its provenance.

    The FRF cannot distinguish the driver pole from the fitted thermal pole:
    two first-order factors commute. A point prior gets a ±25% sensitivity
    bracket; a supplied range must contain the point prior.
    """
    source = config.get("driver_pole_source")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("driver pole requires explicit hardware/measurement provenance")
    nominal = float(config["driver_pole_hz"])
    if not np.isfinite(nominal) or not 300 <= nominal <= 5000:
        raise ValueError("driver pole must be finite and within 300–5000 Hz")
    bounds = config.get("driver_pole_range_hz")
    low, high = ((max(300., nominal*.75), min(5000., nominal*1.25))
                 if bounds is None else map(float, bounds))
    if (not np.isfinite(low) or not np.isfinite(high) or
            not 300 <= low <= nominal <= high <= 5000):
        raise ValueError("driver pole range must be finite, ordered, contain nominal, and stay within 300–5000 Hz")
    return nominal, (low, high), source.strip()


def fir_taps():
    raise ValueError("Hann2000 mode is not qualified by the packaged characterization workflow")


def fir_response(freq, taps):
    if taps is None: return np.ones_like(freq, dtype=complex)
    return np.exp(-2j*np.pi*np.outer(freq, np.arange(len(taps)))/ADC_HZ) @ taps


def rows(data):
    if not data.get("accepted") or len(data.get("frf", [])) < 8:
        raise ValueError("both captures need accepted, coherence-gated FRF rows")
    r = data["frf"]
    f = np.asarray([v["hz"] for v in r], dtype=float)
    g = np.asarray([v["gain_adc_per_dac"]*np.exp(1j*np.deg2rad(v["phase_deg"])) for v in r])
    lm = np.asarray([max(.07, math.log(v["magnitude_95pct_factor"])/1.96) for v in r])
    ph = np.asarray([max(.07, math.radians(v["phase_95pct_deg"])/1.96) for v in r])
    if np.any(~np.isfinite(g)) or np.any(np.diff(f) <= 0) or np.any(f <= 0):
        raise ValueError("FRF rows must have finite gains and increasing positive frequency")
    return f, g, lm, ph


def model(freq, gain, tau, delay, taps_response, driver_pole_hz):
    w = 2*np.pi*freq
    rc = 1/(1+1j*freq/driver_pole_hz)
    return gain*rc*taps_response/(1+1j*w*tau)*np.exp(-1j*w*delay)


def residual(observed, predicted, sigma_mag, sigma_phase):
    logmag = np.log(np.abs(predicted)/np.abs(observed))/sigma_mag
    phase = np.angle(predicted/observed)/sigma_phase
    return np.r_[logmag, phase]


def fit(bundle):
    driver_pole_hz, driver_pole_range_hz, driver_pole_source = driver_pole_config(bundle.get("config", {}))
    mode = bundle.get("config", {}).get("master_feedback_filter")
    if mode not in ("single_sample", "hann2000"):
        raise ValueError("config.master_feedback_filter must explicitly be single_sample or hann2000")
    for key in ("low", "high"):
        observed_mode = bundle[key].get("master_feedback_filter")
        if observed_mode is not None and observed_mode != mode:
            raise ValueError(f"{key} capture master feedback filter differs from bundle config")
    low_f, low_g, low_m, low_p = rows(bundle["low"])
    high_f, high_g, high_m, high_p = rows(bundle["high"])
    slope = float(bundle["live"]["slope"])
    if not np.isfinite(slope) or abs(slope) < 1.5:
        raise ValueError("live signed slope is absent or too weak")
    overlap_max = min(float(max(low_f)), float(max(high_f)))
    if overlap_max < 30 or sum(high_f <= overlap_max) < 4:
        raise ValueError("low/high captures have too little overlapping qualified bandwidth")
    take_low = low_f <= overlap_max
    take_high = high_f <= overlap_max
    lf, lg, lm, lp = low_f[take_low], low_g[take_low], low_m[take_low], low_p[take_low]
    hf, hg, hm, hp = high_f[take_high], high_g[take_high], high_m[take_high], high_p[take_high]
    if len(lf) < 8 or len(hf) < 4:
        raise ValueError("insufficient qualified overlap bins")
    taps = fir_taps() if mode == "hann2000" else None
    hlow, hhigh = fir_response(lf, taps), fir_response(hf, taps)
    sign = math.copysign(1, slope)
    def unpack(v): return sign*np.exp(v[0]), v[1], v[2]
    def objective(v):
        return residual(lg, model(lf, *unpack(v), hlow, driver_pole_hz), lm, lp)
    bounds = ([math.log(abs(slope)*.25), 0., 0.],
              [math.log(abs(slope)*4.), .05, .005])
    best = None
    for tau0 in (.0003, .001, .004, .02):
        trial = least_squares(objective, [math.log(abs(slope)), tau0, .0001],
                              bounds=bounds, max_nfev=200)
        if best is None or np.linalg.norm(trial.fun) < np.linalg.norm(best.fun):
            best = trial
    gain, tau, delay = unpack(best.x)
    fit_rms = float(np.sqrt(np.mean(best.fun**2)))
    held = residual(hg, model(hf, gain, tau, delay, hhigh, driver_pole_hz), hm, hp)
    held_rms = float(np.sqrt(np.mean(held**2)))
    # Keep the diagnostic fit but do not screen gains when held-out evidence
    # differs materially from its stated uncertainty.
    validated = bool(best.success and fit_rms <= 1.5 and held_rms <= 1.5)
    return {"gain_adc_per_dac": gain, "master_feedback_filter": mode,
            "driver_pole_hz": driver_pole_hz,
            "driver_pole_range_hz": list(driver_pole_range_hz),
            "driver_pole_source": driver_pole_source,
            "thermal_tau_s": tau,
            "extra_delay_s": delay, "fit_rms_sigma": fit_rms,
            "heldout_rms_sigma": held_rms, "overlap_max_hz": overlap_max,
            "fit_bin_count": len(lf), "heldout_bin_count": len(hf),
            "slope_adc_per_dac": slope, "validated_for_screening": validated}, taps


def margins(loop, freq):
    mag = np.abs(loop)
    crossings = np.flatnonzero((mag[:-1]-1)*(mag[1:]-1) <= 0)
    if not len(crossings):
        return None
    i = crossings[-1]
    t = (1-mag[i])/(mag[i+1]-mag[i]) if mag[i+1] != mag[i] else 0.
    fc = float(freq[i]+t*(freq[i+1]-freq[i]))
    phase = np.unwrap(np.angle(loop))
    pm = float(180+np.degrees(phase[i]+t*(phase[i+1]-phase[i])))
    phase_cross = np.flatnonzero((phase[:-1]+np.pi)*(phase[1:]+np.pi) <= 0)
    gm = float(-20*np.log10(mag[phase_cross[0]])) if len(phase_cross) else None
    return fc, pm, gm


def screen(fit_result, taps, control_rate, kp_cap, ki_cap,
           baseline_kp=.2, baseline_ki=750.):
    slope = fit_result["slope_adc_per_dac"]
    tau = fit_result["thermal_tau_s"]
    delay = fit_result["extra_delay_s"]
    gain = fit_result["gain_adc_per_dac"]
    driver_poles = (fit_result["driver_pole_range_hz"][0],
                    fit_result["driver_pole_hz"],
                    fit_result["driver_pole_range_hz"][1])
    T = 1/control_rate
    measured_limit = fit_result["overlap_max_hz"]
    freq = np.geomspace(max(.2, measured_limit/1000), control_rate/2*.95, 1400)
    h = fir_response(freq, taps)
    w = 2*np.pi*freq
    zinv = np.exp(-1j*w*T)
    family = []
    # This bracket is a sensitivity exercise, not a confidence interval.
    for gain_scale in (.8, 1., 1.2):
        for tau_scale in (.5, 1., 2.):
            for extra_ticks in (0., 1., 2.):
                for pole in driver_poles:
                    family.append(model(freq, gain*gain_scale, tau*tau_scale,
                                        delay+extra_ticks*T, h, pole)/slope)
    def assess(kp, ki):
        c = kp + ki*T/(1-zinv)
        checks = [margins(p*c, freq) for p in family]
        if any(v is None for v in checks): return None
        gm_values = [v[2] for v in checks if v[2] is not None]
        return {"kp": round(float(kp), 4), "ki_per_s": round(float(ki), 2),
                "worst_phase_margin_deg": round(min(v[1] for v in checks), 1),
                "worst_gain_margin_db": round(min(gm_values), 1) if gm_values else None,
                "max_crossover_hz": round(max(v[0] for v in checks), 2),
                "supported_by_current_runtime": bool(kp <= kp_cap and ki <= ki_cap)}
    baseline = assess(baseline_kp, baseline_ki)
    candidates = []
    for kp in np.linspace(.05, 2., 28):
        for ki in np.geomspace(25., 5000., 34):
            item = assess(kp, ki)
            if item is None: continue
            worst_pm = item["worst_phase_margin_deg"]
            worst_gm = item["worst_gain_margin_db"]
            max_fc = item["max_crossover_hz"]
            if worst_pm < 45 or (worst_gm is not None and worst_gm < 6):
                continue
            # Never claim margin at a crossover beyond qualified overlap.
            if max_fc > .7*measured_limit:
                continue
            candidates.append(item)
    candidates.sort(key=lambda x: (-x["max_crossover_hz"],
                                   -x["worst_phase_margin_deg"]))
    supported = [v for v in candidates if v["supported_by_current_runtime"]]
    return {"baseline": baseline,
            "best_unrestricted": candidates[:6], "best_supported": supported[:6],
            "candidate_count": len(candidates), "supported_count": len(supported)}


def evaluate(bundle):
    try:
        fitted, taps = fit(bundle)
        if not fitted["validated_for_screening"]:
            return {"accepted": False, "reason": "held-out or fitted response exceeds 1.5x reported uncertainty RMS",
                    "plant_fit_diagnostic_only": fitted, "screen": None,
                    "status": "PI screening withheld; no gain applied"}
        config = bundle.get("config", {})
        rate = float(config.get("control_rate_hz", 10000))
        if not 1000 <= rate <= 10000:
            raise ValueError("control rate outside evaluated 1–10 kHz range")
        cap_kp = float(config.get("runtime_kp_max", 1.))
        cap_ki = float(config.get("runtime_ki_max_per_s", 1000.))
        current = bundle.get("current_pi") or {}
        screened = screen(fitted, taps, rate, cap_kp, cap_ki,
                          baseline_kp=float(current.get("kp", .2)),
                          baseline_ki=float(current.get("ki_per_s", 750.)))
    except (KeyError, TypeError, ValueError, OSError) as exc:
        return {"accepted": False, "reason": str(exc), "candidates": []}
    return {"accepted": True, "plant_fit": fitted, "screen": screened,
            "status": "offline review candidates only; no gain applied",
            "limitations": ["First-order thermal pole plus delay is a model assumption.",
                            "Driver pole and range come from explicit provenance; the master filter follows the explicit config.",
                            "Gain/delay/tau sensitivity grid is not a statistical confidence region.",
                            "Crossover is restricted below 70% of measured overlap bandwidth.",
                            "Two captures and live slope must represent the same operating point; this is not proven by FRF files alone.",
                            "Recorder filters cancel ideally in Y/U; no recorder delay or fixed ZOH is added to the empirical plant.",
                            "Slew, saturation, reacquisition, and low-coherence frequency bands are not certified."]}


def self_test():
    taps = None
    slope = -4.7
    def fake(freq):
        g = model(freq, slope, .0012, .00018, fir_response(freq, taps), 1600)
        return {"accepted": True, "frf": [{"hz": float(f),
                "gain_adc_per_dac": float(abs(v)), "phase_deg": float(np.degrees(np.angle(v))),
                "magnitude_95pct_factor": 1.2, "phase_95pct_deg": 8.}
                for f, v in zip(freq, g)]}
    b = {"low": fake(np.arange(2., 181., 2.)),
         "high": fake(np.arange(20., 181., 20.)),
         "live": {"slope": slope}, "config": {"control_rate_hz": 10000,
                                               "master_feedback_filter": "single_sample",
                                               "driver_pole_hz": 1600,
                                               "driver_pole_source": "synthetic test fixture"}}
    result = evaluate(b)
    assert result["accepted"], result
    assert abs(result["plant_fit"]["thermal_tau_s"]-.0012) < .0003
    assert not evaluate({**b, "high": {"accepted": False}})["accepted"]
    incompatible = fake(np.arange(20., 181., 20.))
    for row in incompatible["frf"]: row["gain_adc_per_dac"] *= 3
    assert not evaluate({**b, "high": incompatible})["accepted"]
    assert not evaluate({**b, "config": {"control_rate_hz": 10000}})["accepted"]
    return {"passed": True, "candidate_count": result["screen"]["candidate_count"]}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle", nargs="?", type=Path)
    p.add_argument("--output", "-o", type=Path)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
    else:
        if args.bundle is None: p.error("bundle JSON required")
        result = evaluate(json.loads(args.bundle.read_text()))
        target = args.output or args.bundle.with_name(args.bundle.stem+"-pi-review.json")
        target.write_text(json.dumps(result, indent=2)+"\n")
        print(f"{target}: {'accepted' if result['accepted'] else 'withheld'}")
