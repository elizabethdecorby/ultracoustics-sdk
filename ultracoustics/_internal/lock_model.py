"""Pure 638 lock-calibration model: FRF, plant fit and max-Ki PI design. No device I/O.

Ported from runs/20261020-UNIT-CALIBRATION/unitcal.py (validated on hardware 10-20/10-21);
numerically identical for complete traces. ``frf`` additionally accepts a row mask so a
trace with a few rows still missing can be used: the period-synchronous average then
uses only the rows that are present.

Conventions (match the 638 LockedPi): error_dac = (target - feedback) / k,
u = Kp*e + integrator, integrator += Ki*e*dt at dt = 100 us. The trace records
feedback y[n], the DAC command u[n] (PI + injection) and the injection r[n].
Contract v2 firmware adds an optional second integrator with its zero at zero_hz:
C = (Kp + Ki*dt/(1 - 1/z)) * (1 + 2*pi*zero_hz*dt/(1 - 1/z)).
"""
import dataclasses

import numpy as np
from scipy.optimize import least_squares

CORE_HZ = 144e6          # 638 DWT clock used for trace timestamps
DT = 1e-4                # LockedPi update period at divider 1
GAMMAS = (0.7, 1.0, 1.4) # slope-normalization error the design must tolerate
KP_LIMIT = 1.0           # set_normalized_pi_638 range, contract v1
KI_LIMIT = 1638.3
WIRE_LIMITS = {1: (KP_LIMIT, KI_LIMIT), 2: (5.115, 16383.0)}  # contract -> (Kp, Ki) limits
GOOD_RELERR = 0.15
FGRID = np.logspace(0, np.log10(4900), 1500)
NYQUIST_FRACTION = 0.7  # recorder aliasing corrupts the ratio near Nyquist (seen on 004E 10-20)
TRACE_COLUMNS = ("cycles", "feedback", "actual_dac", "injection_dac", "flags")
RESIDUAL_BANDS_HZ = ((1, 5), (5, 20), (20, 50), (50, 200), (200, 1000), (1000, 4500))


def controller(f, kp, ki, zero_hz=0):
    z = np.exp(1j * 2 * np.pi * np.asarray(f, dtype=float) * DT)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = kp + ki * DT / (1 - 1 / z)
        if zero_hz:
            c = c * (1 + 2 * np.pi * zero_hz * DT / (1 - 1 / z))
        return c


def model(p, f):
    g, tau, td, fd = p
    s = 1j * 2 * np.pi * np.asarray(f, dtype=float)
    return g * np.exp(-s * td) / (1 + s * tau) / (1 + s / (2 * np.pi * fd))


def trace_rows(trace):
    """(rows, present) from a retained trace; missing rows are zero with present False."""
    total = trace["total_samples"]
    samples = trace["samples"]
    rows = np.zeros((total, len(TRACE_COLUMNS)), dtype=np.int64)
    present = np.zeros(total, dtype=bool)
    for i in range(total):
        r = samples.get(i, samples.get(str(i)))
        if r is None:
            continue
        rows[i] = [getattr(r, k) if dataclasses.is_dataclass(r) else r[k] for k in TRACE_COLUMNS]
        present[i] = True
    return rows, present


def max_gap(present):
    """Longest run of missing rows."""
    longest = run = 0
    for ok in present:
        run = 0 if ok else run + 1
        longest = max(longest, run)
    return longest


def frf(rows, hold, decimation, slope, present=None):
    """Period-synchronous FRF of one PRBS-9 trace.

    Returns f, plant P = Y/U (ADC/DAC), normalized H = P/slope, input
    sensitivity S = U/R and the per-bin relative uncertainty of P.
    ``present`` (bool per row, optional) marks rows that were received: the
    detrend is fitted to present rows and each missing row is replaced by the
    mean of the same PRBS phase over the other averaged periods, so the
    period average uses only present rows. Without missing rows the result is
    identical to unitcal.frf.
    """
    rows = np.asarray(rows, dtype=float)
    period = 511 * hold / decimation
    if period != int(period):
        raise ValueError("PRBS period is not an integer number of recorded rows")
    period = int(period)
    nper = len(rows) // period
    if nper < 3:
        raise ValueError("fewer than three PRBS periods recorded")
    keep = None
    if present is not None and not np.all(present):
        present = np.asarray(present, dtype=bool)
        if len(present) != len(rows):
            raise ValueError("row mask length differs from the trace")
        keep = present[: nper * period]
    if keep is None:
        dt = np.median(np.diff(rows[:, 0]) % 2**32) / CORE_HZ
    else:
        pairs = present[1:] & present[:-1]  # timestamp steps between adjacent received rows only
        dt = np.median((np.diff(rows[:, 0]) % 2**32)[pairs]) / CORE_HZ

    used = np.arange(nper * period)

    def blocks(col):
        x = rows[: nper * period, col]
        if keep is None:
            x = x - np.polyval(np.polyfit(used, x, 1), used)  # remove slow operating-point drift
            return np.fft.rfft(x.reshape(nper, period)[1:], axis=1)  # drop first period
        x = x - np.polyval(np.polyfit(used[keep], x[keep], 1), used)
        b, m = x.reshape(nper, period)[1:], keep.reshape(nper, period)[1:]
        count = m.sum(axis=0)
        if not count.all():
            raise ValueError("a PRBS phase is missing from every averaged period")
        phase_mean = np.where(m, b, 0.0).sum(axis=0) / count
        return np.fft.rfft(np.where(m, b, phase_mean), axis=1)

    y, u, r = blocks(1), blocks(2), blocks(3)
    n = y.shape[0]
    ym, um, rm = y.mean(0), u.mean(0), r.mean(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        relerr = np.hypot(y.std(0, ddof=1) / np.sqrt(n) / np.abs(ym),
                          u.std(0, ddof=1) / np.sqrt(n) / np.abs(um))
        p = ym / um
        s = um / rm
    f = np.fft.rfftfreq(period, dt)
    relerr[0] = np.inf
    return {"f": f, "P": p, "H": p / slope, "S": s, "relerr": relerr,
            "fs": 1 / dt, "periods": n, "slope": slope}


def good(fr, fmin=0.0, fmax=None):
    f = fr["f"]
    if fmax is None:
        fmax = NYQUIST_FRACTION * fr["fs"] / 2
    return (f > fmin) & (f < fmax) & np.isfinite(fr["relerr"]) & (fr["relerr"] < GOOD_RELERR)


def smoothed_peak(f, values, width=9):
    """Max of a running median over adjacent bins; a raw max over noisy bins is biased high."""
    order = np.argsort(f)
    v = np.asarray(values)[order]
    if len(v) < width:
        return float(np.median(v)) if len(v) else None
    half = width // 2
    return float(max(np.median(v[i - half:i + half + 1]) for i in range(half, len(v) - half)))


def consistency(fr, kp, ki, fmin=60.0, zero_hz=0):
    """Measured |S| vs 1/|1 + C H| predicted from the same capture's H."""
    m = good(fr, fmin)
    if not m.any():
        return {"bins": 0, "median_abs_error": None, "measured_max_S": None}
    pred = 1 / (1 + controller(fr["f"][m], kp, ki, zero_hz) * fr["H"][m])
    return {"bins": int(m.sum()),
            "band_hz": [float(fr["f"][m].min()), float(fr["f"][m].max())],
            "median_abs_error": float(np.median(np.abs(np.abs(pred) - np.abs(fr["S"][m])))),
            "measured_max_S": float(np.abs(fr["S"][m]).max()),
            "measured_peak_S": smoothed_peak(fr["f"][m], np.abs(fr["S"][m]))}


def fit(fr, fmin=60.0):
    m = good(fr, fmin)
    f, h, w = fr["f"][m], fr["H"][m], 1 / np.maximum(fr["relerr"][m], 0.03)

    def resid(p):
        r = (model(p, f) - h) / np.abs(h) * w
        return np.concatenate([r.real, r.imag])

    sol = least_squares(resid, [0.9, 5e-4, 1e-4, 1500],
                        bounds=([0.2, 5e-5, 0, 300], [3, 5e-3, 5e-4, 2e4]))
    rel = np.abs((model(sol.x, f) - h) / h)
    return {"params": [float(v) for v in sol.x], "g": float(sol.x[0]),
            "tau_ms": float(1e3 * sol.x[1]), "delay_us": float(1e6 * sol.x[2]),
            "driver_pole_hz": float(sol.x[3]), "median_rel_resid": float(np.median(rel)),
            "bins": int(m.sum())}


def s_matrix(params_list, kp, ki_values, gammas=GAMMAS, f=FGRID, zero_hz=0):
    """Worst |S| over frequency, gamma and units for each Ki in ki_values."""
    c = kp + np.outer(np.asarray(ki_values, dtype=float), DT / (1 - np.exp(-1j * 2 * np.pi * f * DT)))
    if zero_hz:
        c = c * (1 + 2 * np.pi * zero_hz * DT / (1 - np.exp(-1j * 2 * np.pi * f * DT)))
    worst = np.zeros(len(ki_values))
    for p in params_list:
        h = model(p, f)
        for g in gammas:
            worst = np.maximum(worst, np.abs(1 / (1 + g * c * h)).max(axis=1))
    return worst


def worst_ms(params_list, kp, ki, gammas=GAMMAS, zero_hz=0):
    return float(s_matrix(params_list, kp, [ki], gammas, zero_hz=zero_hz)[0])


def ki_grid(ki_limit):
    return np.arange(100.0, max(4000.0, ki_limit) + 1e-9, 12.5)  # == unitcal.KI_GRID up to 4000


def migo(params_list, ms_max, gammas=GAMMAS, ki_limit=None, kp_limit=None, zero_hz=0, contract=1):
    """Largest Ki (any Kp in range) whose worst-case sensitivity peak <= ms_max.

    When several Kp reach the same Ki (e.g. at the Ki limit), take the Kp with
    the lowest worst-case peak at that Ki. Limits default to the contract's wire
    limits (WIRE_LIMITS); contract 2 designs are put on its wire grid (Kp in
    0.005 steps, Ki floored to an integer 1/s).
    """
    kp_limit = WIRE_LIMITS[contract][0] if kp_limit is None else kp_limit
    ki_limit = WIRE_LIMITS[contract][1] if ki_limit is None else ki_limit
    kp_grid = np.round(np.arange(0.0, kp_limit + 1e-9, 0.025), 3)  # == KP_GRID for kp_limit 1
    grid = ki_grid(ki_limit)
    ki = grid[grid <= ki_limit]
    reach = []
    for kp in kp_grid[kp_grid <= kp_limit]:
        last = len(ki) - 1
        for start in range(0, len(ki), 128):  # chunks: stop at the first violation (same result, faster)
            ok = s_matrix(params_list, kp, ki[start:start + 128], gammas, zero_hz=zero_hz) <= ms_max
            if not ok.all():
                last = start + int(np.argmin(ok)) - 1  # before first violation
                break
        if last >= 0:
            reach.append((float(kp), float(ki[last])))
    if not reach:
        return (0.0, 0.0)
    top = max(k for _, k in reach)
    ties = [kp for kp, k in reach if k == top]
    kp = min(ties, key=lambda value: worst_ms(params_list, value, top, gammas, zero_hz))
    if contract == 2:
        return (round(round(kp / 0.005) * 0.005, 3), float(np.floor(top)))
    return (kp, top)


def predict(params, kp, ki, zero_hz=0):
    """Model-based margins and disturbance metrics for one PI (+ optional second integrator)."""
    loop = controller(FGRID, kp, ki, zero_hz) * model(params, FGRID)
    i = int(np.argmin(np.abs(np.abs(loop) - 1)))
    s = lambda fq: float(np.abs(1 / (1 + controller(fq, kp, ki, zero_hz) * model(params, fq))))
    out = {"kp": kp, "ki_per_s": ki, "worst_ms": worst_ms([params], kp, ki, zero_hz=zero_hz),
           "nominal_ms": worst_ms([params], kp, ki, (1.0,), zero_hz),
           "crossover_hz": float(FGRID[i]), "phase_margin_deg": float(180 + np.degrees(np.angle(loop[i]))),
           "S_2hz": s(2.0), "S_10hz": s(10.0), "S_40hz": s(40.0),
           "ramp_lag_dac_per_1000_dac_per_s": 1000.0 / (ki * params[0]) if ki > 0 else None}
    if zero_hz:  # type-2 loop: no steady ramp lag
        out.update(zero_hz=zero_hz, ramp_lag_dac_per_1000_dac_per_s=0.0 if ki > 0 else None)
    return out


def compare_measured(fr, params, kp, ki, fmin=60.0, zero_hz=0):
    """Measured |S| at the applied gains vs the model designed from other data."""
    m = good(fr, fmin)
    if not m.any():
        return {"bins": 0}
    pred = np.abs(1 / (1 + controller(fr["f"][m], kp, ki, zero_hz) * model(params, fr["f"][m])))
    meas = np.abs(fr["S"][m])
    return {"bins": int(m.sum()), "median_abs_error": float(np.median(np.abs(pred - meas))),
            "measured_max_S": float(meas.max()), "measured_peak_S": smoothed_peak(fr["f"][m], meas),
            "predicted_max_S_in_band": float(pred.max())}


def highpass_rms(binned, window=200):
    """RMS after removing a moving average (200 x 100 us = 20 ms)."""
    x = np.asarray(binned, dtype=float)
    if len(x) <= window:
        return float(np.std(x))
    kernel = np.ones(window) / window
    trend = np.convolve(x, kernel, mode="same")
    core = slice(window, len(x) - window) if len(x) > 3 * window else slice(None)
    return float(np.std((x - trend)[core]))


def band_rms(records, fs=1e4):
    """Mean Welch PSD over 2 s records of 100 us means (1 s segments, linear detrend) -> RMS per band.

    As runs/20261020-UNIT-CALIBRATION/ab_rest.band_rms; the total is the root-sum-square of the bands.
    """
    from scipy.signal import welch
    psd = np.mean([welch(x, fs, nperseg=10000, detrend="linear")[1] for x in records], axis=0)
    f = np.fft.rfftfreq(10000, 1 / fs)
    df = f[1] - f[0]
    bands = {f"{lo}-{hi}": float(np.sqrt(psd[(f >= lo) & (f < hi)].sum() * df)) for lo, hi in RESIDUAL_BANDS_HZ}
    return bands, float(np.sqrt(sum(b * b for b in bands.values())))
