"""
Thermomechanical noise and pressure calibration.

The sensor's Brownian motion is always present and passes through the same
mechanical response as sound, so its peak in the photodiode spectrum is a
built-in pressure reference.

Key functions
-------------
w_per_count             Optical power per ADC count.
compute_psd_w2hz        Averaged one-sided PSD in W²/Hz.
shot_noise_psd_w2hz     Frequency-independent shot-noise floor.
sho_psd                 Damped-oscillator PSD model.
nep_th_squared, nep_th  Thermomechanical noise-equivalent pressure.
psd_to_pressure_pa2hz   Pressure PSD from a fitted thermal PSD.
spl_db_per_rthz         Pressure PSD to dB SPL/√Hz.
"""

from __future__ import annotations

import numpy as np

from .config import ADC_MAX_VALUE, SAMPLE_RATE
from .readout import ADC_VREF, DIFF_GAIN, RESPONSIVITY_1550, TRANSIMPEDANCE_1550, w_per_count

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

K_B = 1.380649e-23           # Boltzmann constant  [J/K]
ELEM_CHARGE = 1.602176634e-19  # elementary charge   [C]

# Default sensor physical parameters
DEFAULT_M_EFF = 25e-12       # effective modal mass    [kg]   (25 pg)
DEFAULT_DOME_RADIUS = 50e-6  # sensor radius           [m]    (50 µm)
DEFAULT_TEMPERATURE = 300.0  # ambient                 [K]

# The ADC / TIA chain, under the names these functions take as arguments.
DEFAULT_R_PD = RESPONSIVITY_1550
DEFAULT_R_FEEDBACK = TRANSIMPEDANCE_1550
DEFAULT_DIFF_GAIN = DIFF_GAIN
DEFAULT_V_REF = ADC_VREF
DEFAULT_ADC_FULLSCALE = ADC_MAX_VALUE


# ---------------------------------------------------------------------------
# Welch-style PSD in physical units
# ---------------------------------------------------------------------------

def compute_psd_w2hz(samples,
                     fft_size: int = 16384,
                     num_averages: int = 1,
                     sample_rate: float = SAMPLE_RATE,
                     w_per_count_value: float | None = None,
                     subtract_mean: bool = True):
    """One-sided averaged PSD in W²/Hz (Welch / Hanning)::

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
    # DC and Nyquist bins are not duplicated in the one-sided spectrum,
    # so the factor-of-2 in ``norm`` over-counts them. Halve them so the
    # integral of the PSD equals the variance of the windowed signal.
    psd_counts2[0] *= 0.5
    psd_counts2[-1] *= 0.5

    if w_per_count_value is None:
        w_per_count_value = w_per_count()
    psd_w2hz = psd_counts2 * (w_per_count_value ** 2)

    freqs = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
    return freqs, psd_w2hz


def shot_noise_psd_w2hz(mean_counts: float,
                        w_per_count_value: float | None = None,
                        R_PD: float = DEFAULT_R_PD) -> float:
    """Shot-noise PSD floor in W²/Hz.

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
    """Single damped-oscillator PSD::

        S(ω) = A / ((ω_n² − ω²)² + (ω_n·ω/Q_n)²)  +  C

    Falls off as ω⁻⁴ in the tails — *not* a symmetric Lorentzian.
    """
    omega = np.asarray(omega)
    return A / ((omega_n ** 2 - omega ** 2) ** 2
                + (omega_n * omega / Q_n) ** 2) + C


# ---------------------------------------------------------------------------
# Thermomechanical NEP and pressure conversion
# ---------------------------------------------------------------------------

def nep_th_squared(omega_n: float,
                   Q_n: float,
                   m_eff: float = DEFAULT_M_EFF,
                   a: float = DEFAULT_DOME_RADIUS,
                   T: float = DEFAULT_TEMPERATURE) -> float:
    """Thermomechanical NEP² in (Pa²/Hz)::

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
    """Calibrated pressure PSD in Pa²/Hz::

        S_pp(ω) = ((S_WW^TOT − S_DARK − S_SHOT) / S_WW^TH(fit)) · NEP_TH²

    With the dark and shot terms left out, the result still contains them,
    so the floor reads above NEP_TH² wherever they matter. Subtracting them
    leaves an expected floor of NEP_TH², but does not reduce the bin-to-bin
    scatter.

    Parameters
    ----------
    psd_tot_w2hz
        Raw measured PSD on the analysis grid (W²/Hz).
    psd_th_fit_w2hz
        Thermal model PSD on the same grid (W²/Hz).
    nep_th_squared_value
        Thermomechanical NEP² (Pa²/Hz), e.g. :func:`nep_th_squared`.
    psd_dark_w2hz
        Optional per-bin dark PSD on the same grid (W²/Hz).
    shot_psd_w2hz
        Optional scalar shot-noise floor (W²/Hz).
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
    """Convert pressure PSD (Pa²/Hz) to SPL in dB/√Hz."""
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
    "sho_psd",
    # NEP / pressure
    "nep_th", "nep_th_squared",
    "psd_to_pressure_pa2hz", "spl_db_per_rthz",
]
