"""
Noise-Equivalent Pressure (NEP) calibration utilities.

This module implements the thermomechanical-noise pressure-calibration
pipeline described in
``Photodetector-PSSIandUSBHS/docs/ThermomechanicalNoise_Writeup.md``.

The full data flow is:

    raw uint16 ADC counts
        │  (× w_per_count, ²)            -> W²/Hz   compute_psd_w2hz()
        ▼
    S_WW^TOT(ω)  ──────────────────────────────────────────────┐
                                                               │
    dark + shot subtraction (clean PSD)  ──> SHO peak fit      │
                                              │                │
                                              ▼                ▼
                                     S_WW^TH(ω; A, ω₀, Q)   NEP_TH²
                                              │                │
                                              └─► S_pp^TOT = (S_WW^TOT
                                                          / S_WW^TH)
                                                          · NEP_TH²
                                                          [Pa²/Hz]

All routines are pure NumPy / SciPy; no GUI dependency.  The optional
:mod:`ultracoustics.nep_dialog` module provides a PyQt5 interactive peak
selector that drives :func:`fit_sho_log`.

Key public functions
--------------------
w_per_count            Optical-power scaling per ADC count.
compute_psd_w2hz       Welch-style one-sided PSD in W²/Hz.
shot_noise_psd_w2hz    Frequency-independent shot-noise floor.
sho_psd                Single-mode damped-oscillator PSD model.
dual_sho_psd           Two-mode model (sum of SHOs).
find_peak              Parabolic peak + FWHM/FW10M walk-out.
fit_sho_log            Bounded log-space SHO fit (single or dual peak).
evaluate_sho_psd       Evaluate a stored fit at arbitrary frequencies.
nep_th_squared / nep_th  Thermomechanical NEP from fit + dome geometry.
psd_to_pressure_pa2hz  Calibrated pressure PSD via writeup §2.6.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from .config import SAMPLE_RATE

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

K_B = 1.380649e-23           # Boltzmann constant  [J/K]
ELEM_CHARGE = 1.602176634e-19  # elementary charge   [C]

# Default dome physical parameters from the writeup §2.1 / §2.6
DEFAULT_M_EFF = 25e-12       # effective modal mass    [kg]   (25 pg)
DEFAULT_DOME_RADIUS = 50e-6  # dome radius             [m]    (50 µm)
DEFAULT_TEMPERATURE = 300.0  # ambient                 [K]

# ADC / TIA chain (matches nep_calculator.py and the master-board schematic)
DEFAULT_R_PD = 0.9            # photodiode responsivity   [A/W]
DEFAULT_R_FEEDBACK = 20_000.0 # TIA feedback resistor     [Ω]
DEFAULT_DIFF_GAIN = 5.0 / 3.0 # differential front-end gain
DEFAULT_V_REF = 5.0           # ADC full-scale reference  [V]
DEFAULT_ADC_FULLSCALE = 16383 # 14-bit max code


# ---------------------------------------------------------------------------
# ADC → optical-power calibration
# ---------------------------------------------------------------------------

def w_per_count(R_PD: float = DEFAULT_R_PD,
                R_feedback: float = DEFAULT_R_FEEDBACK,
                diff_gain: float = DEFAULT_DIFF_GAIN,
                V_ref: float = DEFAULT_V_REF,
                adc_fullscale: int = DEFAULT_ADC_FULLSCALE) -> float:
    """Optical power (W) per ADC count.

    Composes V/count → A/V → W/A.  Mirrors the chain used in
    ``nep_calculator.py::_finish_collection`` so that dark / signal /
    pressure conversions all use a single shared definition.
    """
    v_per_count = V_ref / adc_fullscale
    return v_per_count / (R_feedback * diff_gain * R_PD)


# ---------------------------------------------------------------------------
# Welch-style PSD in physical units
# ---------------------------------------------------------------------------

def compute_psd_w2hz(samples,
                     fft_size: int = 16384,
                     num_averages: int = 1,
                     sample_rate: float = SAMPLE_RATE,
                     w_per_count_value: float | None = None,
                     subtract_mean: bool = True):
    """One-sided averaged PSD in W²/Hz (Welch / Hanning).

    Implements the writeup §3.1 normalisation::

        PSD_k = (2 / (fs · Σ wₙ²)) · |FFT{x·w}|²

    averaged over ``num_averages`` non-overlapping segments and converted
    from (counts)²/Hz into W²/Hz with ``w_per_count_value²``.

    Parameters
    ----------
    samples
        1-D array of raw ADC values; at least ``fft_size·num_averages``
        long.
    fft_size, num_averages
        Window length and number of averages.
    sample_rate
        Sampling rate in Hz (default SDK ``SAMPLE_RATE``).
    w_per_count_value
        Override for the W/count scalar.  Defaults to :func:`w_per_count`.
    subtract_mean
        If ``True``, removes the per-segment DC level before windowing
        (recommended — keeps DC bin from leaking into the noise floor).

    Returns
    -------
    freqs : ndarray, Hz
    psd   : ndarray, W²/Hz   (length ``fft_size//2 + 1``)
    """
    samples = np.asarray(samples, dtype=np.float64)
    total = fft_size * num_averages
    if samples.size < total:
        raise ValueError(
            f"Need {total} samples ({fft_size}×{num_averages}), "
            f"got {samples.size}"
        )

    seg = samples[:total].reshape(num_averages, fft_size).copy()
    if subtract_mean:
        seg -= seg.mean(axis=1, keepdims=True)
    win = np.hanning(fft_size)
    spec_pow = np.abs(np.fft.rfft(seg * win, axis=1)) ** 2

    norm = 2.0 / (float(sample_rate) * np.sum(win ** 2))
    psd_counts2 = (spec_pow * norm).mean(axis=0)

    if w_per_count_value is None:
        w_per_count_value = w_per_count()
    psd_w2hz = psd_counts2 * (w_per_count_value ** 2)

    freqs = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
    return freqs, psd_w2hz


def shot_noise_psd_w2hz(mean_counts: float,
                        w_per_count_value: float | None = None,
                        R_PD: float = DEFAULT_R_PD) -> float:
    """Shot-noise PSD floor in W²/Hz (writeup §2.5).

        S_WW^SHOT = 2·q·I_dc / R_PD²,    I_dc = ⟨P⟩·R_PD

    ``mean_counts`` should already have any dark-mean offset subtracted
    so it represents photocurrent only.
    """
    if w_per_count_value is None:
        w_per_count_value = w_per_count(R_PD=R_PD)
    mc = max(float(mean_counts), 0.0)
    P0 = mc * w_per_count_value
    I_dc = P0 * R_PD
    return (2.0 * ELEM_CHARGE * I_dc) / (R_PD ** 2)


# ---------------------------------------------------------------------------
# Damped-oscillator PSD models
# ---------------------------------------------------------------------------

def sho_psd(omega, A, omega_n, Q_n, C: float = 0.0):
    """Single damped-oscillator PSD, writeup §2.3::

        S(ω) = A / ((ω_n² − ω²)² + (ω_n·ω/Q_n)²)  +  C

    Falls off as ω⁻⁴ in the tails — *not* a symmetric Lorentzian.
    """
    omega = np.asarray(omega)
    return A / ((omega_n ** 2 - omega ** 2) ** 2
                + (omega_n * omega / Q_n) ** 2) + C


def dual_sho_psd(omega, A1, w1, Q1, A2, w2, Q2, C: float = 0.0):
    """Two-mode SHO PSD (sum of two :func:`sho_psd` terms)."""
    return (sho_psd(omega, A1, w1, Q1)
            + sho_psd(omega, A2, w2, Q2)
            + C)


def evaluate_sho_psd(freqs,
                     fit_A, fit_omega_n, fit_Q_n,
                     fit_A2=None, fit_omega_n2=None, fit_Q_n2=None,
                     fit_C: float = 0.0):
    """Evaluate the fitted SHO model at arbitrary Hz frequencies.

    Accepts the parameter dict produced by :func:`fit_sho_log` (passed as
    keyword arguments).  The optional ``..._2`` arguments add a second
    mode if present.
    """
    omega = 2.0 * np.pi * np.asarray(freqs, dtype=np.float64)
    out = sho_psd(omega, fit_A, fit_omega_n, fit_Q_n, fit_C)
    if (fit_A2 is not None and fit_omega_n2 is not None
            and fit_Q_n2 is not None):
        out = out + sho_psd(omega, fit_A2, fit_omega_n2, fit_Q_n2, 0.0)
    return out


# ---------------------------------------------------------------------------
# Peak finding (parabolic + FWHM walk-out)
# ---------------------------------------------------------------------------

def find_peak(freqs, psd, fmin: float | None = None,
              fmax: float | None = None) -> dict | None:
    """Parabolic-interpolated peak and full-width-at-half/tenth-max.

    Returns a dict with keys
    ``peak_freq, peak_psd, fwhm, fw10m, f_left, f_right, half_max``
    or ``None`` if the slice is too short.
    """
    f = np.asarray(freqs)
    m = np.asarray(psd)

    si = (np.searchsorted(f, fmin) if fmin is not None and fmin > 0 else 1)
    ci = (np.searchsorted(f, fmax) if fmax is not None and fmax > 0 else len(f))

    fs = f[si:ci]
    ms = m[si:ci]
    if fs.size < 3:
        return None

    pk = int(np.argmax(ms))
    if 0 < pk < ms.size - 1:
        a, b, c = ms[pk - 1], ms[pk], ms[pk + 1]
        denom = a - 2.0 * b + c
        if abs(denom) > 1e-12:
            p = 0.5 * (a - c) / denom
            df = fs[1] - fs[0]
            peak_f = fs[pk] + p * df
            peak_p = b - 0.25 * (a - c) * p
        else:
            peak_f, peak_p = float(fs[pk]), float(ms[pk])
    else:
        peak_f, peak_p = float(fs[pk]), float(ms[pk])

    half = peak_p / 2.0
    tenth = peak_p * 0.10

    def _walk(level: float, side: str):
        if side == 'left':
            for i in range(pk - 1, -1, -1):
                if ms[i] <= level:
                    d = ms[i + 1] - ms[i]
                    frac = (level - ms[i]) / d if d != 0 else 0.0
                    return float(fs[i] + frac * (fs[i + 1] - fs[i]))
        else:
            for i in range(pk + 1, ms.size):
                if ms[i] <= level:
                    d = ms[i] - ms[i - 1]
                    frac = (level - ms[i - 1]) / d if d != 0 else 0.0
                    return float(fs[i - 1] + frac * (fs[i] - fs[i - 1]))
        return None

    f_left, f_right = _walk(half, 'left'), _walk(half, 'right')
    f_l10, f_r10 = _walk(tenth, 'left'), _walk(tenth, 'right')

    return dict(
        peak_freq=peak_f,
        peak_psd=peak_p,
        half_max=half,
        f_left=f_left,
        f_right=f_right,
        fwhm=(f_right - f_left) if (f_left and f_right) else None,
        fw10m=(f_r10 - f_l10) if (f_l10 and f_r10) else None,
    )


# ---------------------------------------------------------------------------
# Bounded log-space SHO fit
# ---------------------------------------------------------------------------

def fit_sho_log(freqs, psd,
                peak_freq: float, fwhm: float,
                *,
                fit_lo_hz: float | None = None,
                fit_hi_hz: float | None = None,
                ignore_regions=(),
                peak_wt: float = 0.5,
                peak_psd: float | None = None,
                peak2_freq: float | None = None,
                peak2_fwhm: float | None = None,
                peak2_psd: float | None = None,
                psd_dark=None,
                shot_psd_w2hz: float = 0.0) -> dict:
    """Log-space damped-oscillator fit to a measured PSD.

    Implements the procedure in writeup §4.  Single-mode fit by default;
    pass ``peak2_*`` kwargs for the two-mode model.

    By default this routine performs a **forward-model fit**: the
    measured PSD passed in as ``psd`` is treated as the *raw* total
    PSD, and the model that is fitted is

        S_TOT(ω) = SHO(ω; A, ω_n, Q)  +  S_DARK(ω)  +  S_SHOT

    where ``psd_dark`` is the per-bin dark PSD on the same frequency
    grid as ``freqs`` (in W²/Hz) and ``shot_psd_w2hz`` is the scalar
    shot-noise floor.  Both default to zero, in which case the call
    reduces to a pure SHO fit (equivalent to the legacy
    subtraction-based behaviour when the caller passes
    ``psd = psd_clean``).

    The returned dict mirrors the keys consumed elsewhere in the codebase
    (``fit_A``, ``fit_omega_n``, ``fit_Q_n``, optional ``fit_A2``,
    ``fit_omega_n2``, ``fit_Q_n2``, ``fit_C``, ``fit_freqs``,
    ``fit_curve``, ``fit_error``).  ``fit_curve`` is the **pure SHO**
    component evaluated on ``freqs[1:]`` (DC bin skipped) so it overlays
    cleanly onto the dark/shot-subtracted PSD.  ``fit_curve_total``
    contains the full forward-model evaluation (SHO + dark + shot) on
    the same grid for overlay onto the raw PSD.
    """
    is_dual = peak2_freq is not None and peak2_freq > 0 and peak2_fwhm

    if fit_lo_hz is None or fit_lo_hz <= 0:
        fit_lo_hz = max(peak_freq - 5.0 * fwhm, 0.0)
    if fit_hi_hz is None or fit_hi_hz <= 0:
        if is_dual:
            fit_hi_hz = peak2_freq + peak2_fwhm
        else:
            fit_hi_hz = peak_freq + fwhm

    freqs = np.asarray(freqs, dtype=np.float64)
    psd = np.asarray(psd, dtype=np.float64)

    # Background components for the forward model.  Default to zero so the
    # call degrades to a pure SHO fit when the caller passes a
    # pre-cleaned PSD.
    shot_term = max(float(shot_psd_w2hz or 0.0), 0.0)
    if psd_dark is None:
        dark_full = np.zeros_like(freqs)
    else:
        dark_full = np.asarray(psd_dark, dtype=np.float64)
        if dark_full.shape != freqs.shape:
            result = dict(fit_A=None, fit_omega_n=None, fit_Q_n=None,
                          fit_C=0.0, fit_freqs=None, fit_curve=None,
                          fit_curve_total=None,
                          fit_error="psd_dark length must match freqs")
            return result

    si = np.searchsorted(freqs, fit_lo_hz, side='left')
    ci = np.searchsorted(freqs, fit_hi_hz, side='right')

    result = dict(fit_A=None, fit_omega_n=None, fit_Q_n=None,
                  fit_C=0.0, fit_freqs=None, fit_curve=None,
                  fit_curve_total=None, fit_error=None)
    if ci - si < 5:
        result["fit_error"] = "Not enough points in fit window"
        return result

    f_fit = freqs[si:ci]
    p_fit = psd[si:ci]
    dark_fit = dark_full[si:ci]
    mask = np.ones(f_fit.size, dtype=bool)
    for ilo, ihi in ignore_regions:
        mask &= ~((f_fit >= ilo) & (f_fit <= ihi))
    f_fit = f_fit[mask]
    p_fit = p_fit[mask]
    dark_fit = dark_fit[mask]
    if f_fit.size < 5:
        result["fit_error"] = "Not enough points after ignore-mask"
        return result

    omega_fit = 2.0 * np.pi * f_fit
    log_p = np.log10(np.maximum(p_fit, 1e-35))

    sigma = np.ones_like(p_fit)
    band1 = (f_fit > peak_freq - fwhm) & (f_fit < peak_freq + fwhm)
    sigma[band1] = peak_wt
    if is_dual:
        band2 = (f_fit > peak2_freq - peak2_fwhm) & (f_fit < peak2_freq + peak2_fwhm)
        sigma[band2] = peak_wt

    df = float(freqs[1] - freqs[0]) if freqs.size > 1 else 1.0

    # Peak-power estimate above the dark+shot floor for the SHO amplitude
    # initial guess.
    omega_n0 = 2.0 * np.pi * peak_freq
    Q0 = peak_freq / fwhm
    region = (freqs > peak_freq - 2 * fwhm) & (freqs < peak_freq + 2 * fwhm)
    psd_region_sho = np.maximum(psd[region] - dark_full[region] - shot_term, 0.0)
    power_est = float(np.sum(psd_region_sho) * df)
    A0 = power_est * 4.0 * (omega_n0 ** 3) / Q0
    if A0 <= 0:
        peak_p_guess = peak_psd if peak_psd is not None else 1e-30
        # peak_psd from the caller may be on the clean PSD already; if it
        # was measured on raw, subtract the local floor before seeding.
        peak_p_guess = max(peak_p_guess, 1e-35)
        A0 = peak_p_guess * (omega_n0 ** 2 / Q0) ** 2

    f_disp = freqs[1:]
    omega_disp = 2.0 * np.pi * f_disp
    dark_disp = dark_full[1:]

    try:
        if not is_dual:
            def sho_eval(o, A, wn, Q):
                return A / ((wn ** 2 - o ** 2) ** 2 + (wn * o / Q) ** 2)

            def model(o, A, wn, Q):
                return np.log10(sho_eval(o, A, wn, Q) + dark_fit + shot_term
                                + 1e-35)

            popt, _ = curve_fit(
                model, omega_fit, log_p,
                p0=[A0, omega_n0, Q0],
                bounds=([0.0, 0.8 * omega_n0, 1.0],
                        [np.inf, 1.2 * omega_n0, 1000.0]),
                sigma=sigma, maxfev=50_000,
            )
            A, wn, Q = popt
            sho_disp = sho_eval(omega_disp, A, wn, Q)
            result.update(fit_A=float(A),
                          fit_omega_n=float(wn),
                          fit_Q_n=float(Q),
                          fit_freqs=f_disp,
                          fit_curve=sho_disp,
                          fit_curve_total=sho_disp + dark_disp + shot_term)
        else:
            omega_n2_0 = 2.0 * np.pi * peak2_freq
            Q2_0 = peak2_freq / peak2_fwhm
            region2 = ((freqs > peak2_freq - 2 * peak2_fwhm)
                       & (freqs < peak2_freq + 2 * peak2_fwhm))
            psd_region2_sho = np.maximum(
                psd[region2] - dark_full[region2] - shot_term, 0.0
            )
            power_est2 = float(np.sum(psd_region2_sho) * df)
            A2_0 = power_est2 * 4.0 * (omega_n2_0 ** 3) / Q2_0
            if A2_0 <= 0:
                peak2_p_guess = peak2_psd if peak2_psd is not None else 1e-30
                A2_0 = max(peak2_p_guess, 1e-35) * (omega_n2_0 ** 2 / Q2_0) ** 2

            def sho2_eval(o, A1, w1, Q1, A2, w2, Q2):
                return (A1 / ((w1 ** 2 - o ** 2) ** 2 + (w1 * o / Q1) ** 2)
                        + A2 / ((w2 ** 2 - o ** 2) ** 2 + (w2 * o / Q2) ** 2))

            def model2(o, A1, w1, Q1, A2, w2, Q2):
                return np.log10(sho2_eval(o, A1, w1, Q1, A2, w2, Q2)
                                + dark_fit + shot_term + 1e-35)

            popt, _ = curve_fit(
                model2, omega_fit, log_p,
                p0=[A0, omega_n0, Q0, A2_0, omega_n2_0, Q2_0],
                bounds=([0.0, 0.8 * omega_n0, 1.0,
                         0.0, 0.8 * omega_n2_0, 1.0],
                        [np.inf, 1.2 * omega_n0, 1000.0,
                         np.inf, 1.2 * omega_n2_0, 1000.0]),
                sigma=sigma, maxfev=150_000,
            )
            A1, w1, Q1, A2, w2, Q2 = popt
            sho_disp = sho2_eval(omega_disp, A1, w1, Q1, A2, w2, Q2)
            result.update(fit_A=float(A1),
                          fit_omega_n=float(w1),
                          fit_Q_n=float(Q1),
                          fit_A2=float(A2),
                          fit_omega_n2=float(w2),
                          fit_Q_n2=float(Q2),
                          fit_freqs=f_disp,
                          fit_curve=sho_disp,
                          fit_curve_total=sho_disp + dark_disp + shot_term)
    except Exception as e:  # noqa: BLE001 — surface fit error to caller
        result["fit_error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Thermomechanical NEP and pressure conversion
# ---------------------------------------------------------------------------

def nep_th_squared(omega_n: float,
                   Q_n: float,
                   m_eff: float = DEFAULT_M_EFF,
                   a: float = DEFAULT_DOME_RADIUS,
                   T: float = DEFAULT_TEMPERATURE) -> float:
    """Thermomechanical NEP² in (Pa²/Hz), writeup Eq. §2.6::

        NEP_TH² = 8·k_B·T·m_eff·f₀ / (π·a⁴·Q_m)

    ``omega_n`` is the angular natural frequency from the SHO fit;
    f₀ = ω_n / 2π.
    """
    f0 = omega_n / (2.0 * np.pi)
    return (8.0 * K_B * T * m_eff * f0) / (np.pi * (a ** 4) * Q_n)


def nep_th(omega_n: float, Q_n: float, **kwargs) -> float:
    """Thermomechanical NEP in Pa/√Hz (square-root of :func:`nep_th_squared`)."""
    return float(np.sqrt(nep_th_squared(omega_n, Q_n, **kwargs)))


def psd_to_pressure_pa2hz(psd_tot_w2hz, psd_th_fit_w2hz,
                          nep_th_squared_value: float,
                          psd_dark_w2hz=None,
                          shot_psd_w2hz: float = 0.0):
    """Calibrated pressure PSD in Pa²/Hz, writeup §2.6.

    Default (legacy) behaviour::

        S_pp^TOT(ω) = (S_WW^TOT / S_WW^TH(fit)) · NEP_TH²

    When ``psd_dark_w2hz`` and/or ``shot_psd_w2hz`` are supplied, the
    known non-thermomechanical noise contributions are removed from the
    numerator first (Option 2 from the project discussion)::

        S_pp(ω) = ((S_WW^TOT − S_DARK − S_SHOT) / S_WW^TH(fit)) · NEP_TH²

    This yields a calibrated spectrum whose *expected* baseline (when
    only thermomechanical + dark + shot are present) is NEP_TH², instead
    of being inflated by the dark+shot floor.  Bin-to-bin statistical
    variance in the wings is unchanged — see project notes.

    Parameters
    ----------
    psd_tot_w2hz
        Raw measured PSD on the analysis grid (W²/Hz).
    psd_th_fit_w2hz
        Pure-SHO model PSD on the same grid (W²/Hz), typically from
        :func:`evaluate_sho_psd`.
    nep_th_squared_value
        Thermomechanical NEP² (Pa²/Hz), e.g. :func:`nep_th_squared`.
    psd_dark_w2hz
        Optional per-bin dark PSD on the same grid (W²/Hz).  ``None``
        leaves the dark contribution in the numerator (legacy).
    shot_psd_w2hz
        Optional scalar shot-noise floor (W²/Hz).  ``0.0`` leaves it in
        the numerator (legacy).
    """
    psd_th_fit_w2hz = np.maximum(np.asarray(psd_th_fit_w2hz), 1e-300)
    numerator = np.asarray(psd_tot_w2hz, dtype=np.float64).copy()
    if psd_dark_w2hz is not None:
        numerator = numerator - np.asarray(psd_dark_w2hz, dtype=np.float64)
    shot_val = float(shot_psd_w2hz or 0.0)
    if shot_val:
        numerator = numerator - shot_val
    return (numerator / psd_th_fit_w2hz) * float(nep_th_squared_value)


def spl_db_per_rthz(psd_pa2hz, p_ref: float = 20e-6):
    """Convert pressure PSD (Pa²/Hz) → SPL in dB/√Hz, writeup §2.7."""
    psd_pa2hz = np.maximum(np.asarray(psd_pa2hz), 0.0)
    return 20.0 * np.log10(np.sqrt(psd_pa2hz) / p_ref + 1e-300)


__all__ = [
    # constants
    "K_B", "ELEM_CHARGE",
    "DEFAULT_M_EFF", "DEFAULT_DOME_RADIUS", "DEFAULT_TEMPERATURE",
    "DEFAULT_R_PD", "DEFAULT_R_FEEDBACK", "DEFAULT_DIFF_GAIN",
    "DEFAULT_V_REF", "DEFAULT_ADC_FULLSCALE",
    # calibration / PSD
    "w_per_count",
    "compute_psd_w2hz",
    "shot_noise_psd_w2hz",
    # models
    "sho_psd", "dual_sho_psd", "evaluate_sho_psd",
    # peak + fit
    "find_peak", "fit_sho_log",
    # NEP / pressure
    "nep_th", "nep_th_squared",
    "psd_to_pressure_pa2hz", "spl_db_per_rthz",
]
