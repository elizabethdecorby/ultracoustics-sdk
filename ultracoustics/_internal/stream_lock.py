"""Pure stream-side analysis for the 638 quick lock check. No device I/O.

The lock calibration's identification burst (PRBS-9, one chip = ``hold`` updates, +/-A DAC added to the PI
command) shows in the 10 MSPS photodetector stream, which is the loop's own feedback: on 00560032 the stream
reproduced the 638's recorded feedback with correlation 0.9996 (firmware-analysis runs/20261027-QUICK-LOCK-CHECK).
With the injection R regenerated exactly and the stream Y taken on the controller's sample instants,
S = 1 - C * (Y/R) / k, the same input sensitivity as the trace's U/R.

Timing comes from the stream too. Deconvolving the period-averaged stream by the chip impulse train gives the
loop's response to one chip, which is flat until the DAC write; its onset sits a fixed time after the controller's
sample (ONSET_AFTER_SAMPLE_US). Near the sensitivity peak |S| moves about 0.0016 per us of misalignment.

Conventions follow lock_model: DT = 100 us updates, error = (target - feedback) / k. ``stream_us`` is the ADC
stream reduced to 1 us means.
"""
import numpy as np

from . import lock_model as lm

US_PER_UPDATE = 100
PRBS_LENGTH = 511
#: Onset of the single-chip response after the controller's sample (00560032, 2026-10-27: 86.5 and 88.3 us at two
#: gain sets). Firmware timing plus the driver filter; check it on other boxes.
ONSET_AFTER_SAMPLE_US = 87.4
#: The stream/PRBS correlation peaks about this many updates after the PRBS start (00560032: 3 in both captures);
#: only a starting point, the onset search spans -1000..+1500 us around it.
LEAD_UPDATES = 3
CORRELATION_FRACTION = 0.8
MAX_PERIODS = 8          # the 638's 4096-row capture lasts 8.02 PRBS periods at hold 4
FMAX_HZ = 875.0          # the trace method's band (0.7 x the Nyquist of its 2.5 kHz rows), for parity


def prbs9(hold, amplitude):
    """One period (511 * hold updates) of ControlTrace_NextInjection: state 0x1FF, +A when bit 0 is set,
    shift every ``hold`` updates with feedback bit0 ^ bit4. Matches the firmware exactly (00560032 traces)."""
    state, out = (1 << 9) - 1, np.empty(PRBS_LENGTH * hold)
    for chip in range(PRBS_LENGTH):
        out[chip * hold:(chip + 1) * hold] = amplitude if state & 1 else -amplitude
        state = (state >> 1) | (((state ^ (state >> 4)) & 1) << 8)
    return out


def norm_xcorr(a, b):
    """Normalized cross-correlation of ``a`` against ``b`` at every lag where ``a`` fits inside ``b``."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = (a - a.mean()) / (a.std() * len(a))
    n = len(b) + len(a)
    full = np.fft.irfft(np.fft.rfft(b, n) * np.conj(np.fft.rfft(a, n)), n)[:len(b) - len(a) + 1]
    c1, c2 = np.cumsum(np.concatenate(([0.0], b))), np.cumsum(np.concatenate(([0.0], b * b)))
    m = len(a)
    mean = (c1[m:] - c1[:-m]) / m
    std = np.sqrt(np.maximum((c2[m:] - c2[:-m]) / m - mean ** 2, 1e-12))
    return full / std


def chip_response(stream_us, start_us, chips, hold, nper):
    """Response to one +1 chip at 1 us resolution, t = 0 at the assumed chip start; the first period is dropped."""
    period_us = PRBS_LENGTH * hold * US_PER_UPDATE
    seg = np.asarray(stream_us[start_us:start_us + nper * period_us], dtype=float)
    if start_us < 0 or len(seg) < nper * period_us:
        raise ValueError("the stream does not cover the requested periods")
    y = seg.reshape(nper, period_us)[1:].mean(axis=0)
    impulses = np.zeros(period_us)
    impulses[::hold * US_PER_UPDATE] = chips
    spectrum = np.fft.fft(impulses)
    spectrum[0] = np.inf                       # the mean is not identifiable
    return np.real(np.fft.ifft(np.fft.fft(y - y.mean()) / spectrum))


def onset_us(response, slope):
    """Onset of the single-chip response (us after t = 0): line through its 10-40 % rise towards the first
    extremum, extrapolated to the pre-onset baseline. A +DAC chip moves the detector by sign(slope)."""
    x = np.sign(slope) * np.roll(response, 1000)[:2500]          # t = -1000 .. +1500 us
    i_ext = int(np.argmax(x))
    base = float(np.median(x[:800]))
    rise = x[i_ext] - base
    below10 = np.flatnonzero(x[:i_ext] < base + 0.1 * rise)
    below40 = np.flatnonzero(x[:i_ext] < base + 0.4 * rise)
    j10 = int(below10[-1]) + 1 if len(below10) else 0
    j40 = int(below40[-1]) + 1 if len(below40) else i_ext
    t = np.arange(j10, max(j40, j10 + 3))
    a, b = np.polyfit(t, x[t], 1)
    return float((base - b) / a) - 1000.0


def sensitivity(stream_us, start_us, period, nper, kp, ki, zero_hz, slope):
    """Period-synchronous S = 1 - C*(Y/R)/k on the 10 kHz update grid, first period dropped.
    Returns f, S, Y/R (ADC per DAC), the plant H = (Y/R)/S/k and the per-bin relative error of Y."""
    phase, j0 = int(start_us) % US_PER_UPDATE, int(start_us) // US_PER_UPDATE
    n = nper * len(period)
    y = np.asarray(stream_us[phase::US_PER_UPDATE][j0:j0 + n], dtype=float)
    if j0 < 0 or len(y) < n:
        raise ValueError("the stream does not cover the requested periods")
    idx = np.arange(n)
    y = y - np.polyval(np.polyfit(idx, y, 1), idx)            # slow operating-point drift
    Y = np.fft.rfft(y.reshape(nper, len(period))[1:], axis=1)
    R = np.fft.rfft(period)
    f = np.fft.rfftfreq(len(period), lm.DT)
    with np.errstate(divide="ignore", invalid="ignore"):
        yr = Y.mean(axis=0) / R
        s = 1 - lm.controller(f, kp, ki, zero_hz) * yr / slope
        relerr = np.abs(Y.std(axis=0, ddof=1) / np.sqrt(len(Y)) / np.abs(Y.mean(axis=0)))
        h = yr / s / slope
    relerr[0] = np.inf
    return {"f": f, "S": s, "YR": yr, "H": h, "relerr": relerr}


def analyse(stream_us, amplitude, hold, slope, kp, ki, zero_hz=0, onset_after_sample_us=ONSET_AFTER_SAMPLE_US,
            fmin=60.0, fmax=FMAX_HZ):
    """Find the burst, time it, measure |S| at the running gains and fit the plant.

    Returns start_us (stream index of the controller sample of the PRBS's first update), the timing details,
    periods used, bins, measured peak |S| (smoothed, fmin..fmax), the plant fit (lock_model.fit) and the
    sensitivity arrays. Raises ValueError when no burst or too few periods are found.
    """
    period = prbs9(hold, amplitude)
    grid = np.asarray(stream_us[0::US_PER_UPDATE], dtype=float)
    if len(grid) < 3 * len(period):
        raise ValueError("stream shorter than three PRBS periods")
    cc = np.abs(norm_xcorr(np.tile(period, 2), grid))
    high = np.flatnonzero(cc >= CORRELATION_FRACTION * cc.max())
    first, last = int(high[0]), int(high[-1])
    coarse_us = US_PER_UPDATE * (first - LEAD_UPDATES)
    burst_periods = (last - first) // len(period) + 2      # the 2-period template fits from first to last
    room = (len(stream_us) - max(coarse_us, 0) - 2000) // (len(period) * US_PER_UPDATE)
    nper = int(min(MAX_PERIODS, burst_periods, room))
    if nper < 3:
        raise ValueError(f"only {nper} PRBS periods in the stream (need 3)")
    response = chip_response(stream_us, coarse_us, period[::hold] / amplitude, hold, nper)
    onset = onset_us(response, slope)
    start_us = int(round(coarse_us + onset - onset_after_sample_us))
    fr = sensitivity(stream_us, start_us, period, nper, kp, ki, zero_hz, slope)
    good = (fr["f"] > fmin) & (fr["f"] < fmax) & np.isfinite(fr["relerr"]) & (fr["relerr"] < lm.GOOD_RELERR)
    out = {"start_us": start_us, "coarse_us": coarse_us, "onset_us": round(onset, 2),
           "onset_after_sample_us": onset_after_sample_us, "correlation": float(cc.max()), "periods": nper,
           "bins": int(good.sum()), "band_hz": [float(fmin), float(fmax)], "peak_S": None, "fit": None, "frf": fr}
    if good.sum() >= 9:
        out["peak_S"] = lm.smoothed_peak(fr["f"][good], np.abs(fr["S"][good]))
    if good.sum() >= 20:
        out["fit"] = lm.fit({"f": fr["f"], "H": fr["H"], "relerr": np.where(good, fr["relerr"], np.inf),
                             "fs": 1 / lm.DT}, fmin)
    return out
