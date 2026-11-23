"""Thermomechanical-noise pressure calibration.

The sensor's own thermal peak is the pressure reference:

    |H(f)|^2 = S_TH(f) / NEP_1^2     [W^2/Hz per Pa^2/Hz]
    S_pp(f)  = S_TOT / |H(f)|^2       [Pa^2/Hz, noise included]

S_TH is the sum of the sensor modes in the pressure scale, so the second mode is
divided out along with the first and the thermal floor reads flat at NEP_1^2
across both peaks. NEP comes from the fundamental (mode 1) only.

The acquisition is anti-aliased, so every narrow peak in the band is a real
sensor mode. The fit also finds broad features of the readout's response; they
stay in the fitted model but not in the pressure scale (see scale_modes). The
anti-alias filter's response is fitted too (see filter_gain): it shapes the
optical terms (and sound) but not the dark, which was recorded through it. The
calibrated band stops at ANTI_ALIAS_CUTOFF_HZ; the fit runs on to FIT_TOP_HZ.
"""
from dataclasses import dataclass, field, replace
from functools import cached_property
import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks, peak_widths

from .. import nep as ucnep
from ..config import SAMPLE_RATE
from .spectrum import POWER_DRIFT_WARNING, Spectrum, _now


# Version 3: fitted through the anti-alias filter. Earlier files modelled
# aliases and a dark reference taken without the filter, so they are refused.
FILE_VERSION = 3
DARK_SUMMARY_BINS = 512   # of the dark spectrum, as stored in a recording
# Calibration averages far more than one display frame: 0.5 s at 16384 points
# is ~300 Welch segments, which is what keeps repeated fits consistent.
CAPTURE_SECONDS = 0.5
FIT_FLOOR_MAX_DB = 3.0       # an automatic fit's white floor over dark + shot noise (clean: ~0 dB)
CAPTURE_FFT_SIZE = 16384
MIN_AVERAGES = 16
# The fit and the mode search start here. Below it, 1/f-like excess noise
# (+6 dB at 150 kHz, +1 dB at 550 kHz, gone by ~0.9 MHz on the bench) rises in
# a way the white floor C can't follow, and bent the fit. Sensor modes sit above
# 2 MHz on every probe so far.
MIN_MODE_FREQ_HZ = 1.0e6
MIN_MODE_Q = 3.0
MAX_MODE_Q = 3000.0  # the fit bounds Q at ~3160; a pinned fit is not a result
MAX_FIT_LOG_Q = 3.5
MIN_MODE_WIDTH_BINS = 3

# Autofit. Every resolved resonance in the band (1 MHz to the anti-alias
# cutoff) is fitted, however weak, so the model follows the spectrum; the sensor
# modes among them are divided out so the thermal floor reads flat in pressure
# units. Single-bin pickup lines are masked. After each fit the largest
# leftover bump (e.g. a weak feature on a strong mode's skirt) is added if it
# fits as a resonance. NEP comes from the fundamental: the lowest-frequency
# mode clear of the filter edge within MODE_STRENGTH_WINDOW_DB of the tallest.
PEAK_PROMINENCE_DB = 2.0
MODE_STRENGTH_WINDOW_DB = 10.0
# The anti-alias filter. The views and the calibrated band stop at its cutoff,
# but its response bends the spectrum from ~4.2 MHz (on one box ~+1 dB of
# peaking near 4.35 MHz, then -3 dB near 4.46 and -13 dB by 4.6; another box
# droops from 4.2 MHz with no peaking). Left out of the model, the roll-off
# pulled the fit and the peaking was fitted as a mode, so every calibration fits
# the filter (filter_gain) from the white optical floor, running on to
# FIT_TOP_HZ to see the roll-off.
ANTI_ALIAS_CUTOFF_HZ = 4.5e6
FIT_TOP_HZ = 4.6e6
# filter_gain parameters: corner (MHz), order, peaking, peaking centre (MHz),
# peaking width (MHz). The start is between the boxes measured so far.
FILTER_START = (4.43, 38.0, 0.5, 4.39, 0.07)
FILTER_LOWER = (4.0, 5.0, 0.0, 4.0, 0.03)
FILTER_UPPER = (5.0, 80.0, 3.0, 4.6, 0.2)
PEAK_SMOOTH_BINS = 5
SPIKE_MASK_BINS = 3
RESIDUAL_PROMINENCE_DB = 0.5
RESIDUAL_SMOOTH_BINS = 11
RESIDUAL_SEARCH_ROUNDS = 3
LEFTOVER_MIN_PEAK_FRACTION = 0.25  # of the local noise floor
RESIDUAL_ROUND_MAX_NFEV = 200  # warm-started rounds converge in tens of steps
MODE_FREQ_TOLERANCE_WIDTHS = 0.5
PEAK_CORE_SIGMA = 0.5      # bins within ±FWHM of a sensor mode count 4x in the fit
FIT_LOSS_SCALE = 0.2       # log10 units (~2 dB); only larger residuals are down-weighted
# The fundamental's f0 and Q (and so the NEP) come from a refit over this many
# linewidths either side of it (see _refit_fundamental). From 3 to 20 they
# agreed within 3% on the bench, and with the half-max width.
LOCAL_FIT_FWHM = 5.0

# Pressure scale. Only sensor modes set it. A resonance much broader than the
# fundamental is the readout's response shaping the noise floor (the dark
# spectrum peaks near 3.9 MHz, where a Q~5 "mode" gets fitted) or a bump on a
# skirt, not the sensor: checked against a 10 MHz detector, real modes sat at
# >= 0.66x the fundamental's Q and these at <= 0.3x.
SCALE_MIN_Q_FRACTION = 0.5
# A mode within this many linewidths of the cutoff is cut by the filter: its
# lower skirt is fitted, but its peak height is extrapolated past the cutoff
# (Q can run to the bound), so it is never taken for the fundamental.
EDGE_GUARD_FWHM = 2.0

# Acoustic calibrator (absolute scale). 2^19 points gives 19 Hz bins; the ring
# holds enough for ~9 averages in CAPTURE_SECONDS.
ACOUSTIC_LEVEL_DB = 114.0
ACOUSTIC_FREQ_HZ = 1000.0
TONE_FFT_SIZE = 2 ** 19
TONE_SEARCH = 0.05        # calibrators are specified to ~1 %; search ±5 %
TONE_HALF_WIDTH_BINS = 3  # Hann main lobe is ±2 bins
TONE_MIN_SNR = 100.0      # peak must sit 20 dB over the local floor
MAX_TONE_THD = 0.03

P_REF_PA = 20e-6

class CalibrationError(Exception):
    """Calibration could not be produced; the message is shown to the user."""


@dataclass
class ProbeParameters:
    """Sensor constants that set the absolute NEP scale (NEP^2 is linear in m_eff).

    ``acoustic_correction`` multiplies the thermal NEP^2 once an acoustic
    calibrator has measured the true scale. It belongs to the probe, not to one
    fit, so it survives re-fits (f0 and Q may drift; the sensor's mass/area don't).
    """
    m_eff_kg: float = ucnep.DEFAULT_M_EFF
    radius_m: float = ucnep.DEFAULT_DOME_RADIUS
    temperature_k: float = ucnep.DEFAULT_TEMPERATURE
    acoustic_correction: float = 1.0
    acoustic_note: str = ''
    acoustic_date: str = ''
    name: str = 'Probe 1'

    def nep2(self, omega_n, q):
        """The thermal noise-equivalent pressure², Pa²/Hz, of a sensor mode at angular
        frequency *omega_n* (rad/s) with quality factor *q*."""
        return self.acoustic_correction * float(ucnep.nep_th_squared(
            omega_n, q, m_eff=self.m_eff_kg, a=self.radius_m, T=self.temperature_k))

    @classmethod
    def from_stored(cls, d):
        """Ignore keys this version no longer uses (e.g. the real-mode lists
        kept before the anti-alias filter, when other peaks were aliases)."""
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Mode:
    A: float
    omega_n: float
    Q: float

    @property
    def peak_w2hz(self):
        return self.A * self.Q ** 2 / self.omega_n ** 4

    @property
    def f0_hz(self):
        return self.omega_n / (2.0 * np.pi)


@dataclass
class Calibration:
    """One START's pressure calibration: what turns optical power into sound pressure.

    :meth:`Controller.calibrate_pressure` makes one, recordings keep theirs as
    ``recording.calibration``, and :func:`load_calibration` opens one the GUI saved.
    The calls to use are :meth:`to_pressure` (a spectrum in Pa²/Hz),
    :meth:`sensitivity`, :meth:`noise_budget`, :meth:`pressure_waveform` and
    :meth:`pa_per_uw`. ``nep2`` is the quietest sound the probe resolves, in Pa²/Hz,
    and ``fundamental`` the sensor's first resonance. The other fields are the fit
    itself, kept so a recording can be analysed again later.

    A calibration belongs to one probe at one START: do not carry it to another.
    """
    modes: list               # every fitted mode, sorted by frequency
    probe: ProbeParameters
    dark: Spectrum
    photocurrent_counts: float
    shot_psd_w2hz: float
    floor_w2hz: float         # fitted white noise above dark + shot (laser noise etc.)
    nep2: float               # fundamental's thermal NEP^2, Pa^2/Hz (see fundamental())
    fft_size: int
    num_averages: int
    sample_rate: float
    w_per_count: float
    signal: Spectrum = None   # the spectrum that was fit, kept for diagnosis/refits
    filter_params: tuple = None  # fitted anti-alias filter (filter_gain); None: flat
    created_at: str = field(default_factory=_now)

    @property
    def floor_pa2hz(self):
        """The thermal noise floor in Pa²/Hz: the same number as ``nep2``."""
        return self.nep2

    @property
    def fundamental(self):
        """The sensor's first resonance, as a fitted mode: ``f0_hz`` and ``Q``."""
        return fundamental(self.modes, band_top_hz(self.sample_rate))

    @cached_property
    def scale_modes(self):
        return scale_modes(self.modes, band_top_hz(self.sample_rate))

    def with_probe(self, probe):
        """NEP depends only on the fundamental's (omega_n, Q) and the probe."""
        ref = self.fundamental
        return replace(self, probe=probe, nep2=probe.nep2(ref.omega_n, ref.Q))

    def sensitivity(self, freqs):
        """W²/Hz of optical power per Pa²/Hz of sound, at ``freqs`` in Hz."""
        return transfer_w2_per_pa2(self, freqs)

    def to_pressure(self, spectrum, check_power=True):
        """``spectrum`` in pressure units: see :func:`.pressure.to_pressure`."""
        from .pressure import to_pressure
        return to_pressure(self, spectrum, check_power)

    def noise_budget(self, spectrum):
        """Each noise term of ``spectrum``: see :func:`.noise.noise_budget`."""
        from .noise import noise_budget
        return noise_budget(self, spectrum)

    def pa_per_uw(self, band):
        """Pa per µW over ``band``: see :func:`.pressure.pa_per_uw`."""
        from .pressure import pa_per_uw
        return pa_per_uw(self, band)

    def pressure_waveform(self, power_uw, band, sample_rate=SAMPLE_RATE):
        """A waveform in Pa from µW: see :func:`.pressure.pressure_waveform`."""
        from .pressure import pressure_waveform
        return pressure_waveform(self, power_uw, band, sample_rate)


def capture_spectrum(samples, sample_rate, fft_size=CAPTURE_FFT_SIZE,
                     min_averages=MIN_AVERAGES):
    """Welch PSD over every whole segment in ``samples``."""
    samples = np.asarray(samples, dtype=np.float64)
    num_averages = samples.size // fft_size
    if num_averages < min_averages:
        raise CalibrationError(
            f'Need at least {min_averages * fft_size / sample_rate * 1e3:.0f} ms of '
            'streamed data. Let the stream run and try again.')
    used = samples[:num_averages * fft_size]
    freqs, psd = ucnep.compute_psd_w2hz(used, fft_size=fft_size,
                                        num_averages=num_averages,
                                        sample_rate=sample_rate)
    return Spectrum(freqs, np.maximum(psd, 1e-40), float(used.mean()),
                    fft_size, num_averages)


def photocurrent_counts(mean_counts, dark):
    offset = dark.mean_counts if dark is not None else 0.0
    return max(float(mean_counts) - offset, 0.0)


def power_drift(cal, mean_counts):
    """Relative optical-power change since calibration, or None if unknown."""
    if cal is None or cal.photocurrent_counts <= 0:
        return None
    return photocurrent_counts(mean_counts, cal.dark) / cal.photocurrent_counts - 1.0


def fit_max_hz(sample_rate):
    """Top of the fitted band: past the cutoff, into the filter's roll-off."""
    return min(FIT_TOP_HZ, sample_rate / 2)


def band_top_hz(sample_rate):
    """Top of the calibrated band: the anti-alias cutoff."""
    return min(ANTI_ALIAS_CUTOFF_HZ, sample_rate / 2)


def calibrated_band(cal, freqs):
    """Mask of the bins the calibration covers (below the anti-alias cutoff)."""
    return np.asarray(freqs) <= band_top_hz(cal.sample_rate)


def filter_gain(params, freqs):
    """The anti-alias filter's power gain |H|^2 at ``freqs`` (1 when ``params``
    is None): a steep Butterworth-like low-pass times a Gaussian peaking term.
    Fitted to the optical floor it followed each box measured so far within
    0.2 dB from 4.25 to 4.55 MHz, peaking or not."""
    f = np.asarray(freqs, dtype=np.float64) / 1e6
    if params is None:
        return np.ones_like(f)
    corner, order, peak, centre, width = params
    return ((1.0 + peak * np.exp(-0.5 * ((f - centre) / width) ** 2))
            / (1.0 + (f / corner) ** (2.0 * order)))


def _filter_gain_jacobian(params, freqs):
    """filter_gain and its derivatives by each of its parameters."""
    f = np.asarray(freqs, dtype=np.float64) / 1e6
    corner, order, peak, centre, width = params
    u = (f - centre) / width
    bump = np.exp(-0.5 * u ** 2)
    top = 1.0 + peak * bump
    rise = (f / corner) ** (2.0 * order)
    below = 1.0 + rise
    gain = top / below
    return gain, (top * rise * 2.0 * order / (corner * below ** 2),
                  -top * rise * 2.0 * np.log(f / corner) / below ** 2,
                  bump / below,
                  peak * bump * u / width / below,
                  peak * bump * u ** 2 / width / below)


def filter_corner_hz(cal):
    """Where the fitted filter is 3 dB down, or None if none was fitted."""
    if cal.filter_params is None:
        return None
    f = np.linspace(3.5e6, cal.sample_rate / 2, 3001)
    below = np.nonzero(filter_gain(cal.filter_params, f) < 0.5)[0]
    return float(f[below[0]]) if below.size else None


def at_filter_edge(mode, top_hz):
    """Within EDGE_GUARD_FWHM linewidths of the band top, where the anti-alias
    filter cuts the mode's upper side."""
    return mode.f0_hz + EDGE_GUARD_FWHM * mode.f0_hz / mode.Q > top_hz


def fundamental(modes, top_hz):
    """The lowest-frequency mode within MODE_STRENGTH_WINDOW_DB of the tallest,
    among the modes clear of the filter edge.

    The fundamental is the sensor's lowest mode and normally its strongest; the
    strength window keeps a weak low-frequency peak from being taken for it.
    """
    clear = [m for m in modes if not at_filter_edge(m, top_hz)]
    if not clear:
        raise CalibrationError('No sensor mode was fitted below the anti-alias filter '
                               'edge, so there is no fundamental.')
    tallest = max(m.peak_w2hz for m in clear)
    strong = [m for m in clear
              if m.peak_w2hz >= tallest * 10.0 ** (-MODE_STRENGTH_WINDOW_DB / 10.0)]
    return min(strong, key=lambda m: m.omega_n)


def scale_modes(modes, top_hz):
    """The modes that set the pressure scale: the sensor modes.

    Resonances broader than SCALE_MIN_Q_FRACTION of the fundamental's Q stay in
    the fitted model (they shape the floor) but are not sensitivity.
    """
    ref = fundamental(modes, top_hz)
    return [m for m in modes if m.Q >= SCALE_MIN_Q_FRACTION * ref.Q]


def calibrate(signal, dark, probe, sample_rate):
    """Fit every resolved resonance in the band and return a Calibration.

    Forward model, fit in log space from MIN_MODE_FREQ_HZ to FIT_TOP_HZ:

        S(f) = S_DARK(f) + G(f) * (S_SHOT + C + sum_modes SHO(f))

    with a free white floor C >= 0 and the anti-alias filter's gain G
    (filter_gain), fitted along with the modes. The dark was recorded through
    the filter, so G applies to the optical terms only. The input is
    anti-aliased, so every narrow peak is a sensor mode; the sensor modes are
    divided out of the pressure view (see scale_modes).
    """
    w_pc = ucnep.w_per_count()
    freqs = signal.freqs
    psd_total = np.maximum(signal.psd_w2hz, 1e-35)
    current = photocurrent_counts(signal.mean_counts, dark)
    shot = float(ucnep.shot_noise_psd_w2hz(current, w_per_count_value=w_pc))
    dark_floor = dark.on_grid(freqs) if dark is not None else np.zeros_like(freqs)
    known_floor = shot + dark_floor
    bin_hz = float(freqs[1] - freqs[0])
    top = fit_max_hz(sample_rate)
    band_top = band_top_hz(sample_rate)

    peaks = _find_peaks(freqs, psd_total, known_floor, top)
    # Widths come from the smoothed spectrum, where a one-bin spike already
    # looks PEAK_SMOOTH_BINS wide; a resonance must be clearly wider than that.
    min_width = max(MIN_MODE_WIDTH_BINS, 1.5 * PEAK_SMOOTH_BINS) * bin_hz
    seeds = [p for p in peaks if p['width_hz'] >= min_width]
    if not seeds:
        raise CalibrationError(
            'No thermomechanical peak was found. Check that the system is '
            'locked (START) and that the dark reference was taken with the '
            'lasers off.')
    masked = _merge(_spike_region(p, bin_hz) for p in peaks if p['width_hz'] < min_width)

    # A detected peak that won't fit as a resolved resonance (Q pinned, or too
    # broad to be mechanical) is dropped and the rest refitted.
    while seeds:
        try:
            modes, floor, model, filt = _fit(freqs, psd_total, dark_floor, shot, seeds,
                                             masked, top)
        except CalibrationError:
            modes = None
        if modes is not None and all(_is_resonance(m, bin_hz) for m in modes):
            break
        # Drop the offending peak, or the weakest if the fit failed outright.
        bad = (min(range(len(seeds)), key=lambda i: seeds[i]['height']) if modes is None
               else next(i for i, m in enumerate(modes) if not _is_resonance(m, bin_hz)))
        seeds = seeds[:bad] + seeds[bad + 1:]
    else:
        raise CalibrationError(
            'The thermal peak could not be fit. Check that the system is locked '
            'before calibrating.')
    modes, floor, filt = _add_leftover_modes(freqs, psd_total, dark_floor, shot, modes, floor,
                                             model, filt, masked, top, band_top)
    modes.sort(key=lambda m: m.omega_n)
    modes = _refit_fundamental(freqs, psd_total, dark_floor, shot, modes, floor, filt, masked,
                               band_top)
    ref = fundamental(modes, band_top)
    return Calibration(modes, probe, dark, current, shot, floor,
                       probe.nep2(ref.omega_n, ref.Q), signal.fft_size,
                       signal.num_averages, float(sample_rate), w_pc, signal=signal,
                       filter_params=tuple(float(v) for v in filt))


def _refit_fundamental(freqs, psd_total, dark_floor, shot, modes, floor, filt, masked, top_hz):
    """Refit the fundamental alone over ±LOCAL_FIT_FWHM linewidths.

    Across the band the white floor C is not flat (the readout's response
    rises with frequency), and the leftover search patches the fundamental's
    skirts with ghost modes; both pulled its Q to 63-73 on the bench where its
    half-max width says 61. Locally the floor is flat, so the peak sets Q. The
    other modes add their fixed tails; ones inside the window are skirt
    patches, now described by the refit, and are dropped. If the refit fails
    the band fit stands.
    """
    ref = fundamental(modes, top_hz)
    half = LOCAL_FIT_FWHM * ref.f0_hz / ref.Q
    window = ((np.abs(freqs - ref.f0_hz) < half)
              & (freqs >= MIN_MODE_FREQ_HZ) & (freqs <= top_hz))
    for lo, hi in masked:
        window &= ~((freqs >= lo) & (freqs <= hi))
    others = [m for m in modes if m is not ref]
    outside = [m for m in others if abs(m.f0_hz - ref.f0_hz) >= half]
    omega = 2.0 * np.pi * freqs[window]
    gain = filter_gain(filt, freqs[window])
    tails = sum((ucnep.sho_psd(omega, m.A, m.omega_n, m.Q) for m in outside), np.zeros_like(omega))
    base = dark_floor[window] + gain * (shot + tails)
    target = np.log10(psd_total[window])
    scale = float(np.median(shot + dark_floor[window]))
    width = ref.f0_hz / ref.Q

    def model(p):
        return base + gain * (10.0 ** p[3] + ucnep.sho_psd(omega, 10.0 ** p[0],
                                                           2 * np.pi * p[1] * 1e6, 10.0 ** p[2]))
    lower = [np.log10(ref.A) - 3, (ref.f0_hz - 0.5 * width) / 1e6, 0.0, np.log10(scale * 1e-4)]
    upper = [np.log10(ref.A) + 3, (ref.f0_hz + 0.5 * width) / 1e6, MAX_FIT_LOG_Q,
             np.log10(scale * 1e2)]
    p0 = np.clip([np.log10(ref.A), ref.f0_hz / 1e6, np.log10(ref.Q),
                  np.log10(max(floor, scale * 1e-3))], lower, upper)
    result = least_squares(lambda p: np.log10(model(p)) - target, p0,
                           bounds=(lower, upper), x_scale='jac')
    a, f0, log_q, _ = result.x
    refit = Mode(10.0 ** a, 2 * np.pi * f0 * 1e6, 10.0 ** log_q)
    if not result.success or not _is_resonance(refit, float(freqs[1] - freqs[0])):
        return modes
    return sorted(outside + [refit], key=lambda m: m.omega_n)


def _is_resonance(mode, bin_hz):
    """Damped enough to be mechanical, and wide enough to be resolved.

    A single-bin noise spike otherwise fits as a Q-at-the-bound "mode".
    """
    width = mode.f0_hz / mode.Q
    return (MIN_MODE_Q <= mode.Q <= MAX_MODE_Q
            and width >= MIN_MODE_WIDTH_BINS * bin_hz)


def _find_peaks(freqs, psd_total, known_floor, top_hz):
    """Every peak above PEAK_PROMINENCE_DB between MIN_MODE_FREQ_HZ and ``top_hz``.

    Heights and widths are taken on a lightly smoothed spectrum so bin-to-bin
    scatter neither creates peaks nor fragments the half-max crossing.
    """
    lo = int(np.searchsorted(freqs, MIN_MODE_FREQ_HZ))
    hi = int(np.searchsorted(freqs, top_hz))
    kernel = np.ones(PEAK_SMOOTH_BINS) / PEAK_SMOOTH_BINS
    smooth = np.convolve(psd_total, kernel, mode='same')
    index, _ = find_peaks(10.0 * np.log10(smooth[lo:hi]), prominence=PEAK_PROMINENCE_DB)
    index = index + lo
    if index.size == 0:
        return []
    widths = peak_widths(smooth, index, rel_height=0.5)[0]
    bin_hz = float(freqs[1] - freqs[0])
    return [dict(freq=float(freqs[i]), height=float(smooth[i]),
                 excess=float(max(smooth[i] - known_floor[i], smooth[i] * 1e-3)),
                 width_hz=float(w * bin_hz))
            for i, w in zip(index, widths)]


def _spike_region(peak, bin_hz):
    pad = SPIKE_MASK_BINS * bin_hz
    return peak['freq'] - pad, peak['freq'] + pad


def _merge(regions):
    merged = []
    for lo, hi in sorted(regions):
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def _seed_from_mode(mode):
    """Turn a fitted mode back into a seed so a refit starts from it."""
    return dict(freq=mode.f0_hz, width_hz=mode.f0_hz / mode.Q, excess=mode.peak_w2hz)


def _add_leftover_modes(freqs, psd_total, dark_floor, shot, modes, floor, model, filt, masked,
                        top_hz, band_top_hz):
    """Add the largest leftover bump as another mode and refit, a few times.

    This finds weak modes too small to register as peaks on their own, such as
    one on a strong mode's skirt. A bump that doesn't fit as a resolved
    resonance (a broad floor hump, a line) is not a mode: it is skipped and the
    search moves on to the next one. Bumps are looked for in the calibrated
    band only: in the roll-off above it, a slight misfit of the filter would
    pass for a mode whose peak is all extrapolation.
    """
    bin_hz = float(freqs[1] - freqs[0])
    skipped = []
    added = 0
    for _ in range(2 * RESIDUAL_SEARCH_ROUNDS):
        if added == RESIDUAL_SEARCH_ROUNDS:
            break
        bump = _largest_residual_bump(freqs, psd_total, model, modes, masked + skipped,
                                      band_top_hz)
        if bump is None:
            break
        seeds = [_seed_from_mode(m) for m in modes] + [bump]
        try:
            trial = _fit(freqs, psd_total, dark_floor, shot, seeds, masked, top_hz,
                         max_nfev=RESIDUAL_ROUND_MAX_NFEV, filter0=filt)
        except CalibrationError:
            trial = None
        if trial is not None:
            new = trial[0][-1]
            local_floor = np.interp(new.f0_hz, freqs, dark_floor) + shot + trial[1]
        # A mode must be a resolved resonance that stands out of the floor;
        # smaller bumps are scatter, and dividing them out would do nothing.
        if (trial is None or not all(_is_resonance(m, bin_hz) for m in trial[0])
                or new.peak_w2hz < LEFTOVER_MIN_PEAK_FRACTION * local_floor):
            skipped.append((bump['freq'] - bump['width_hz'], bump['freq'] + bump['width_hz']))
            continue
        modes, floor, model, filt = trial
        added += 1
    return modes, floor, filt


def _largest_residual_bump(freqs, psd_total, model, fitted, masked, top_hz):
    """The most prominent resolved bump left in data/model, clear of mode tops."""
    bin_hz = float(freqs[1] - freqs[0])
    raw = 10.0 * np.log10(psd_total / model)
    in_mask = np.zeros(freqs.shape, dtype=bool)
    for lo, hi in masked:
        in_mask |= (freqs >= lo) & (freqs <= hi)
    # Zero masked bins before smoothing so a masked line can't leak out as a bump.
    kernel = np.ones(RESIDUAL_SMOOTH_BINS) / RESIDUAL_SMOOTH_BINS
    resid = np.convolve(np.where(in_mask, 0.0, raw), kernel, mode='same')
    allowed = (freqs >= 2 * MIN_MODE_FREQ_HZ) & (freqs <= top_hz) & ~in_mask
    for mode in fitted:
        allowed &= np.abs(freqs - mode.f0_hz) > mode.f0_hz / mode.Q
    # Anything narrower than the smoothing window is a line, not a resonance.
    index, props = find_peaks(np.where(allowed, resid, 0.0),
                              prominence=RESIDUAL_PROMINENCE_DB,
                              width=1.5 * RESIDUAL_SMOOTH_BINS)
    if index.size == 0:
        return None
    k = int(np.argmax(props['prominences']))
    i = int(index[k])
    return dict(freq=float(freqs[i]), width_hz=float(props['widths'][k] * bin_hz),
                excess=float(model[i] * (10.0 ** (resid[i] / 10.0) - 1.0)))


def _fit(freqs, psd_total, dark_floor, shot, seeds, masked, top_hz, max_nfev=1000,
         filter0=FILTER_START):
    """Robust log-space fit of the forward model (see calibrate).

    Parameters are (log10 A, f0 in MHz, log10 Q) per peak, the five of
    filter_gain, then log10 C, which keeps them within a few decades of each
    other for the optimiser. Each mode's top (±1 FWHM) is weighted up. Returns
    (fitted modes in seed order, floor_w2hz, model on the full grid, filter
    parameters).
    """
    keep = (freqs >= MIN_MODE_FREQ_HZ) & (freqs <= top_hz)
    for lo, hi in masked:
        keep &= ~((freqs >= lo) & (freqs <= hi))
    if keep.sum() < 20:
        raise CalibrationError('Not enough unmasked spectrum to fit.')
    omega_all = 2.0 * np.pi * freqs
    target = np.log10(psd_total[keep])
    sigma = np.ones_like(target)

    p0, lower, upper = [], [], []
    for seed in seeds:
        f0 = seed['freq']
        q0 = float(np.clip(f0 / seed['width_hz'], MIN_MODE_Q, MAX_MODE_Q))
        a0 = np.log10(max(seed['excess'], 1e-40) * ((2 * np.pi * f0) ** 2 / q0) ** 2)
        p0 += [a0, f0 / 1e6, np.log10(q0)]
        # A mode may move only within half its width of where it was found, so
        # it can't slide onto a neighbouring mode.
        shift = max(MODE_FREQ_TOLERANCE_WIDTHS * seed['width_hz'], 2 * (freqs[1] - freqs[0]))
        lower += [a0 - 3, (f0 - shift) / 1e6, 0.0]
        upper += [a0 + 3, (f0 + shift) / 1e6, MAX_FIT_LOG_Q]
        sigma[np.abs(freqs[keep] - f0) < seed['width_hz']] = PEAK_CORE_SIGMA
    n = len(seeds)
    p0 += list(filter0)
    lower += list(FILTER_LOWER)
    upper += list(FILTER_UPPER)
    dark = dark_floor[keep]
    scale = float(np.median(dark + shot)) or float(np.median(psd_total[keep])) * 1e-3
    excess = float(np.median(psd_total[keep] - dark - shot))
    p0.append(np.log10(max(excess, scale * 1e-3)))
    lower.append(np.log10(scale * 1e-4))
    upper.append(np.log10(scale * 1e2))
    p0 = np.clip(p0, lower, upper)

    def optical(p, omega):
        total = shot + 10.0 ** p[-1]
        for i in range(n):
            a, wn, q = 10.0 ** p[3 * i], 2 * np.pi * p[3 * i + 1] * 1e6, 10.0 ** p[3 * i + 2]
            total = total + ucnep.sho_psd(omega, a, wn, q)
        return total

    def model(p, f, base):
        return base + filter_gain(p[3 * n:3 * n + 5], f) * optical(p, 2 * np.pi * f)

    f_fit = freqs[keep]
    omega_fit = omega_all[keep]

    def jacobian(p):
        """Analytic d(residual)/d(param); finite differences cost a model
        evaluation per parameter per step, which dominated the run time."""
        gain, d_gain = _filter_gain_jacobian(p[3 * n:3 * n + 5], f_fit)
        light = optical(p, omega_fit)
        total = dark + gain * light
        jac = np.empty((omega_fit.size, len(p)))
        w = omega_fit
        for i in range(n):
            a, wn, q = 10.0 ** p[3 * i], 2 * np.pi * p[3 * i + 1] * 1e6, 10.0 ** p[3 * i + 2]
            detune = wn ** 2 - w ** 2
            denom = detune ** 2 + (wn * w / q) ** 2
            term = gain * a / denom
            jac[:, 3 * i] = term * np.log(10.0)
            jac[:, 3 * i + 1] = (-term / denom * (4 * wn * detune + 2 * wn * w ** 2 / q ** 2)
                                 * 2 * np.pi * 1e6)
            jac[:, 3 * i + 2] = term / denom * 2 * wn ** 2 * w ** 2 / q ** 3 * q * np.log(10.0)
        for j, d in enumerate(d_gain):
            jac[:, 3 * n + j] = d * light
        jac[:, -1] = gain * 10.0 ** p[-1] * np.log(10.0)
        return jac / (total * np.log(10.0) * sigma)[:, None]

    result = least_squares(
        lambda p: (np.log10(model(p, f_fit, dark)) - target) / sigma, p0,
        jac=jacobian, bounds=(lower, upper), loss='soft_l1', f_scale=FIT_LOSS_SCALE,
        x_scale='jac', max_nfev=max_nfev)
    if not result.success:
        raise CalibrationError(f'Peak fit did not converge: {result.message}')
    p = result.x
    modes = [Mode(10.0 ** p[3 * i], 2 * np.pi * p[3 * i + 1] * 1e6, 10.0 ** p[3 * i + 2])
             for i in range(n)]
    return (modes, float(10.0 ** p[-1]), model(p, freqs, dark_floor),
            tuple(p[3 * n:3 * n + 5]))


def thermal_psd_w2hz(cal, freqs, modes=None):
    """Sum of ``modes`` (default: every fitted mode), no floors, W^2/Hz."""
    omega = 2.0 * np.pi * np.asarray(freqs, dtype=np.float64)
    return sum((ucnep.sho_psd(omega, m.A, m.omega_n, m.Q)
                for m in (cal.modes if modes is None else modes)), np.zeros_like(omega))


def transfer_w2_per_pa2(cal, freqs):
    """Sensitivity used to convert to pressure, W^2/Hz per Pa^2/Hz: the sensor
    modes in the pressure scale over the fundamental's NEP^2, through the
    anti-alias filter (sound is filtered like the thermal noise is)."""
    return (filter_gain(cal.filter_params, freqs)
            * thermal_psd_w2hz(cal, freqs, cal.scale_modes) / cal.nep2)


def live_shot_w2hz(cal, mean_counts):
    return float(ucnep.shot_noise_psd_w2hz(
        photocurrent_counts(mean_counts, cal.dark), w_per_count_value=cal.w_per_count))


def pressure_psd(cal, freqs, psd_w2hz):
    """Calibrated pressure PSD in Pa^2/Hz, noise included.

    Dark and shot noise are not subtracted: the result is the pressure-referred
    noise, so off resonance it rises above NEP^2 where the detector, not the
    sensor's thermal motion, limits detection.
    """
    return (np.asarray(psd_w2hz, dtype=np.float64)
            / np.maximum(transfer_w2_per_pa2(cal, freqs), 1e-300))


def noise_terms(cal, freqs, mean_counts):
    """Each term of the fitted spectrum at ``freqs``, W^2/Hz, as recorded: the optical
    ones through the filter. See :mod:`.noise` for what each one is."""
    freqs = np.asarray(freqs, dtype=np.float64)
    gain = filter_gain(cal.filter_params, freqs)
    scale = cal.scale_modes
    broad = [mode for mode in cal.modes if not any(mode is kept for kept in scale)]
    return dict(dark=np.zeros_like(freqs) if cal.dark is None else np.maximum(cal.dark.on_grid(freqs), 0.0),
                shot=gain * live_shot_w2hz(cal, mean_counts), white=gain * cal.floor_w2hz,
                thermal=gain * thermal_psd_w2hz(cal, freqs, scale),
                broad=gain * thermal_psd_w2hz(cal, freqs, broad))


def fit_overlay_db(cal, freqs, mean_counts):
    """The fitted spectrum, dB(W^2/Hz): every term of :func:`noise_terms`."""
    return 10.0 * np.log10(np.maximum(sum(noise_terms(cal, freqs, mean_counts).values()), 1e-40))


# -- Acoustic calibrator ----------------------------------------------------

@dataclass
class ToneMeasurement:
    freq_hz: float
    p_rms_pa: float     # as read through the current calibration
    thd: float          # harmonics 2-5, amplitude ratio
    level_db: float     # calibrator's stated SPL
    correction: float   # multiply the current NEP^2 by this to read level_db

    @property
    def measured_db(self):
        return 20.0 * np.log10(self.p_rms_pa / P_REF_PA)


def measure_tone(cal, samples, sample_rate, level_db=ACOUSTIC_LEVEL_DB,
                 freq_hz=ACOUSTIC_FREQ_HZ):
    """Read a calibrator tone through ``cal`` and derive the NEP^2 correction.

    Long segments are needed here: the 16384-point calibration grid has 610 Hz
    bins, far too coarse to separate a 1 kHz tone from DC.
    """
    spec = capture_spectrum(samples, sample_rate, fft_size=TONE_FFT_SIZE,
                            min_averages=2)
    f = spec.freqs
    df = float(f[1] - f[0])
    pa2 = np.maximum(pressure_psd(cal, f, spec.psd_w2hz), 0.0)

    lo, hi = np.searchsorted(f, [freq_hz * (1 - TONE_SEARCH), freq_hz * (1 + TONE_SEARCH)])
    peak = int(lo + np.argmax(pa2[lo:hi]))
    fundamental, floor = tone_power(pa2, peak, df)
    if fundamental <= 0 or pa2[peak] < TONE_MIN_SNR * floor:
        raise CalibrationError(
            f'No calibrator tone found near {freq_hz:g} Hz. Check that the '
            'calibrator is seated on the probe, switched on, and the system is locked.')
    harmonics = 0.0
    for h in range(2, 6):
        centre = int(round(h * peak))
        if centre + TONE_HALF_WIDTH_BINS * 8 >= f.size:
            break
        window = slice(centre - 2, centre + 3)
        power, _ = tone_power(pa2, centre - 2 + int(np.argmax(pa2[window])), df)
        harmonics += max(power, 0.0)

    p_rms = float(np.sqrt(fundamental))
    target = P_REF_PA * 10.0 ** (level_db / 20.0)
    lobe = slice(peak - 2, peak + 3)
    centroid = float(np.sum(f[lobe] * pa2[lobe]) / np.sum(pa2[lobe]))
    return ToneMeasurement(centroid, p_rms, float(np.sqrt(harmonics / fundamental)),
                           float(level_db), float((target / p_rms) ** 2))


def tone_power(pa2, peak, df):
    """Hann main lobe power (Pa^2) above the local floor, and that floor (Pa^2/Hz)."""
    k = TONE_HALF_WIDTH_BINS
    near = np.r_[pa2[max(peak - 8 * k, 0):max(peak - 2 * k, 0)],
                 pa2[peak + 2 * k + 1:peak + 8 * k + 1]]
    floor = float(np.median(near)) if near.size else 0.0
    lobe = pa2[max(peak - k, 0):peak + k + 1]
    return float((lobe.sum() - floor * lobe.size) * df), floor


def apply_acoustic_correction(probe, tone):
    return replace(probe, acoustic_correction=probe.acoustic_correction * tone.correction,
                   acoustic_note=f'{tone.level_db:.1f} dB @ {tone.freq_hz:.0f} Hz',
                   acoustic_date=_now())


# -- Persistence ------------------------------------------------------------

def _spectrum_dict(s):
    return dict(freqs=s.freqs.tolist(), psd_w2hz=s.psd_w2hz.tolist(),
                mean_counts=s.mean_counts, fft_size=s.fft_size,
                num_averages=s.num_averages, captured_at=s.captured_at)


def _spectrum_from(d):
    return Spectrum(np.asarray(d['freqs'], dtype=np.float64),
                    np.asarray(d['psd_w2hz'], dtype=np.float64),
                    float(d['mean_counts']), int(d['fft_size']),
                    int(d['num_averages']), d['captured_at'])


def to_dict(cal):
    return dict(
        version=FILE_VERSION, created_at=cal.created_at,
        modes=[dict(A=m.A, omega_n=m.omega_n, Q=m.Q) for m in cal.modes],
        nep2=cal.nep2, floor_w2hz=cal.floor_w2hz,
        probe=vars(cal.probe).copy(),
        dark=_spectrum_dict(cal.dark) if cal.dark is not None else None,
        signal=_spectrum_dict(cal.signal) if cal.signal is not None else None,
        photocurrent_counts=cal.photocurrent_counts,
        shot_psd_w2hz=cal.shot_psd_w2hz, fft_size=cal.fft_size,
        num_averages=cal.num_averages, sample_rate=cal.sample_rate,
        w_per_count=cal.w_per_count,
        filter_params=list(cal.filter_params) if cal.filter_params is not None else None)


def to_recording_dict(cal):
    """What a recording stores of ``cal``: no fitted spectrum, and the dark reduced
    to DARK_SUMMARY_BINS, enough for its noise and photocurrent."""
    stored = to_dict(cal)
    stored.pop('signal', None)
    if cal.dark is not None:
        stored['dark'] = _spectrum_dict(_reduced(cal.dark, DARK_SUMMARY_BINS))
    return stored


def _reduced(spectrum, bins):
    """``spectrum`` averaged into at most ``bins`` equal groups of bins."""
    group = max(1, -(-spectrum.freqs.size // bins))
    edges = np.arange(0, spectrum.freqs.size, group)

    def means(x):
        return np.add.reduceat(x, edges) / np.diff(np.append(edges, x.size))

    return replace(spectrum, freqs=means(spectrum.freqs), psd_w2hz=means(spectrum.psd_w2hz))


def from_dict(d):
    version = d.get('version')
    if version in (1, 2):
        raise CalibrationError('The stored calibration was made before the anti-alias '
                               'filter was fitted. Capture dark and calibrate again.')
    if version != FILE_VERSION:
        raise CalibrationError(f'Unsupported calibration file version {version}')
    modes = d['modes']
    return Calibration(
        modes=[Mode(m['A'], m['omega_n'], m['Q']) for m in modes],
        probe=ProbeParameters.from_stored(d['probe']),
        dark=_spectrum_from(d['dark']) if d.get('dark') else None,
        photocurrent_counts=float(d['photocurrent_counts']),
        shot_psd_w2hz=float(d['shot_psd_w2hz']),
        floor_w2hz=float(d['floor_w2hz']),
        nep2=float(d['nep2']),
        fft_size=int(d['fft_size']), num_averages=int(d['num_averages']),
        sample_rate=float(d['sample_rate']), w_per_count=float(d['w_per_count']),
        signal=_spectrum_from(d['signal']) if d.get('signal') else None,
        # Absent from files fitted before the filter was modelled: flat, as fitted.
        filter_params=tuple(d['filter_params']) if d.get('filter_params') else None,
        created_at=d['created_at'])


def save(cal, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(to_dict(cal)), encoding='utf-8')
    os.replace(tmp, path)


def floor_over_noise_db(cal):
    """How far the fit's white floor sits above dark and shot noise, in dB. About 0
    on a clean fit: far above FIT_FLOOR_MAX_DB, something other than the sensor's
    thermal motion was in the fit, such as sound or a disturbed lock."""
    known = cal.shot_psd_w2hz + (float(np.median(cal.dark.psd_w2hz)) if cal.dark is not None else 0.0)
    return float(10 * np.log10((known + cal.floor_w2hz) / known))


def load_calibration(path):
    """The calibration saved at ``path``. FileNotFoundError if there is none,
    CalibrationError if it cannot be read."""
    calibration = load(path)
    if calibration is None:
        raise FileNotFoundError(f'{path} not found')
    return calibration


def load(path):
    """Stored calibration, or None when no file exists."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        cal = from_dict(json.loads(path.read_text(encoding='utf-8')))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CalibrationError(f'Stored calibration could not be read: {exc}') from exc
    return cal
