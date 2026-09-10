"""
Signal processing utilities – FFT, PSD, and data conversion.

Internal note: All pure-computation logic extracted from usb_fft_viewer.py lives here
so it can be reused without any GUI dependency.

Functions
---------
load_binary(path, dtype)
    Load raw ADC samples from a binary file on disk.
adc_to_uw(samples, baseline, responsivity, transimpedance)
    Convert raw 14-bit ADC counts to optical power in µW.
compute_psd(samples, fft_size, num_averages, sample_rate)
    Compute a Hanning-windowed, averaged one-sided PSD in dB re 1 W²/Hz.
"""

import numpy as np
from pathlib import Path

from .config import SAMPLE_RATE

# ---------------------------------------------------------------------------
# Physical unit conversion constants
# ---------------------------------------------------------------------------

ADC_FULL_SCALE = 16383          # 14-bit ADC
ADC_VREF = 5.0                  # Volts
TRANSIMPEDANCE = 20_000         # 20 kΩ — the 1550 nm channel, and the default
DIFF_GAIN = 470 / 280           # ADA4940 differential driver gain
                                # (annotated 1.667× on the schematic but actual ratio is 1.679)

# Feedback resistor per readout channel. The two channels do NOT share one
# front end: the 638 nm channel uses a 10 kΩ TIA and the 1550 nm channel a
# 20 kΩ TIA, so a given ADC count corresponds to twice the photocurrent on
# the 638 channel. Pass the matching value to :func:`adc_to_uw` — the
# default is the 1550 nm value for backwards compatibility.
TRANSIMPEDANCE_1550 = 20_000    # Ω
TRANSIMPEDANCE_638 = 10_000     # Ω

# Derived
_ADC_TO_VOLTAGE = ADC_VREF / ADC_FULL_SCALE          # V / count (at ADC pin)
_VOLTAGE_TO_CURRENT = 1.0 / TRANSIMPEDANCE           # A / V (at TIA output)
# ADA4940 amplifies TIA → ADC by DIFF_GAIN, so the photocurrent that
# produced one ADC count is reduced by 1/DIFF_GAIN.
_ADC_TO_CURRENT_UA = (_ADC_TO_VOLTAGE / DIFF_GAIN) * _VOLTAGE_TO_CURRENT * 1e6  # µA / count

# Responsivity: R = η·q·λ/(hc). Spec sheet says R=0.9A/wW @ 1310nm. Solving for quuantum efficiency gives 0.9 × 1240/1310 = 0.852. 
# so then R(1550) = 0.852 × 1550/1240 = 1.065 A/W. This is approximate and assuming quantum eff is completely flat. 
# likely the quantum efficiency peaks a bit around 1550 giving an even higher responsivity

_RESPONSIVITY = 1.077  # µA / µW

ADC_TO_POWER_UW = _ADC_TO_CURRENT_UA / _RESPONSIVITY
"""µW per ADC count (full signal chain incl. ADA4940 ~1.68× gain)."""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_binary(path, dtype=np.uint16):
    """Load raw ADC samples from a binary file.

    Reads the entire file as a flat array of the given dtype.

    Args:
        path: Filesystem path to the binary file.
        dtype: NumPy dtype of each sample (default ``np.uint16`` for
            14-bit ADC values stored in 16-bit words).

    Returns:
        numpy.ndarray: 1-D array of raw sample values.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return np.fromfile(p, dtype=dtype)


# ---------------------------------------------------------------------------
# Unit conversion helpers
# ---------------------------------------------------------------------------

def adc_to_uw(samples, baseline=0.0, responsivity=None, transimpedance=None):
    """Convert raw ADC counts to optical power in µW.

    Applies the full signal chain: ADC → voltage → (÷ ADA4940 gain) →
    (÷ R_f) → current → (÷ responsivity) → optical power. The ADA4940
    differential driver amplifies the TIA output before the ADC, so the
    photocurrent is the ADC voltage divided by ``R_f · DIFF_GAIN``.

    Both the photodiode and the feedback resistor differ between the two
    readout channels, so a full conversion needs both values:

    ==========  ==================  ==================
    Channel     Responsivity (A/W)  Transimpedance (Ω)
    ==========  ==================  ==================
    1550 nm     1.077 (InGaAs)      20 000
    638 nm      0.3 (Si visible)    10 000
    ==========  ==================  ==================

    Args:
        samples: Array of uint16 ADC values (14-bit range 0–16383).
        baseline: ADC-count DC offset to subtract before conversion.
        responsivity: Detector responsivity in A/W (== µA/µW). Defaults
            to the 1550 nm InGaAs value.
        transimpedance: TIA feedback resistance in ohms. Defaults to
            :data:`TRANSIMPEDANCE` (the 1550 nm value). Pass
            :data:`TRANSIMPEDANCE_638` when converting 638 nm data.

    Returns:
        numpy.ndarray: Optical power values in µW (float64).

    Note:
        Every argument combination goes through one formula, so calling
        this with the default responsivity and transimpedance spelled out
        explicitly gives exactly the same scale as omitting them. An
        earlier revision had a second code path that hardcoded a 10 kΩ
        feedback resistor, which silently doubled every converted value
        whenever a responsivity was supplied.
    """
    if responsivity is None:
        responsivity = _RESPONSIVITY
    if transimpedance is None:
        transimpedance = TRANSIMPEDANCE
    scale = (_ADC_TO_VOLTAGE / DIFF_GAIN) / transimpedance * 1e6 / responsivity
    return (samples.astype(np.float64) - baseline) * scale


# ---------------------------------------------------------------------------
# FFT / PSD
# ---------------------------------------------------------------------------

def compute_psd(samples, fft_size=8192, num_averages=1, sample_rate=SAMPLE_RATE):
    """Compute one-sided Power Spectral Density in physical units.

    Segments the input into *num_averages* non-overlapping blocks of
    *fft_size*, removes the per-segment DC level, applies a Hanning
    window, computes the real FFT and averages the **power** spectra
    (Welch's method).  The result is converted from ADC counts through
    the transimpedance / responsivity chain into W²/Hz, then expressed
    in dB (10·log10).

    Args:
        samples: 1-D array of raw ADC values (at least
            ``fft_size * num_averages`` elements).
        fft_size: Number of points per FFT segment (default 8192).
        num_averages: Number of non-overlapping segments to average
            for noise reduction.
        sample_rate: ADC sampling frequency in Hz (default 10 MHz).

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]:
            - **freq_hz** – frequency axis in Hz (length ``fft_size//2 + 1``).
            - **psd_db** – PSD in dB re 1 W²/Hz.

    Raises:
        ValueError: If *samples* is shorter than ``fft_size * num_averages``.
    """
    samples = np.asarray(samples, dtype=np.float64)
    total_needed = fft_size * num_averages
    if len(samples) < total_needed:
        raise ValueError(
            f"Need {total_needed} samples ({fft_size}×{num_averages}), "
            f"got {len(samples)}"
        )

    window = np.hanning(fft_size)

    # Reshape into segments and detrend (remove per-segment DC).
    segments = samples[:total_needed].reshape(num_averages, fft_size)
    segments = segments - segments.mean(axis=1, keepdims=True)
    windowed = segments * window

    # Vectorised FFT, average |X|² (Welch).
    ffts = np.fft.rfft(windowed, axis=1)
    power_avg = np.mean(np.abs(ffts) ** 2, axis=0)

    # Convert ADC-count² spectrum → W²/Hz using the writeup's
    # transimpedance / responsivity chain.
    adc_to_power_fft = _ADC_TO_CURRENT_UA / _RESPONSIVITY  # µW / count
    # Welch normalisation: 2 / (fs · Σw²); DC and Nyquist halved below.
    norm = 2.0 / (sample_rate * np.sum(window ** 2))
    psd_uw2 = power_avg * (adc_to_power_fft ** 2) * norm
    psd_uw2[0] *= 0.5
    psd_uw2[-1] *= 0.5
    # µW²/Hz → W²/Hz
    psd_w = psd_uw2 * 1e-12

    psd_db = 10 * np.log10(psd_w + 1e-30)

    freq_hz = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
    return freq_hz, psd_db


# ---------------------------------------------------------------------------
# Time-domain noise metrics
# ---------------------------------------------------------------------------

def compute_noise_metrics(samples, sample_rate_hz=None, dc_guard_bins: int = 1):
    """Compute a battery of noise metrics on a single sample block.

    There is no single "the" noise number for a photodetector signal, so
    this returns several established figures side by side. All are
    expressed in raw ADC counts (or dimensionless ratios); for calibrated
    W²/Hz spectral density use :func:`compute_psd` / the :mod:`ultracoustics.nep`
    helpers instead.

    Parameters
    ----------
    samples : array-like
        1-D array of raw ADC values (uint16 or float).
    sample_rate_hz : float or None
        Sampling frequency in Hz. When given, ``rin_db`` is normalised to
        a per-Hz noise density (dB/Hz) over the analysis bandwidth
        ``fs / 2``; otherwise it is the unnormalised
        ``10·log10(var / mean²)`` in dB.
    dc_guard_bins : int
        Number of low-frequency bins (including DC) to drop from the FFT
        integral. Defaults to 1 (DC only).

    Returns
    -------
    dict
        Keys:

        * ``mean`` – DC level (ADC counts).
        * ``std_ac_rms`` – AC RMS = std of DC-removed signal (counts).
        * ``peak_to_peak`` – max − min (counts).
        * ``cv`` – coefficient of variation = std / mean (dimensionless).
        * ``fft_integral`` – sum(|FFT(x_ac)|²)/n over the AC band
          (Parseval-consistent, raw units²).
        * ``rin`` – variance / mean² (dimensionless intensity noise).
        * ``rin_db`` – ``10·log10(rin)`` dB, or dB/Hz when
          *sample_rate_hz* is given.
    """
    x = np.asarray(samples, dtype=np.float64).ravel()
    n = x.size

    mean_val = float(np.mean(x)) if n else float("nan")
    std_ac = float(np.std(x, ddof=1)) if n > 1 else float("nan")
    pkpk = float(np.max(x) - np.min(x)) if n else float("nan")
    cv = std_ac / mean_val if mean_val != 0 else float("nan")

    # Spectral noise: rfft of DC-removed signal.
    x_ac = x - mean_val
    spec = np.fft.rfft(x_ac) if n else np.zeros(0, dtype=complex)
    psd_like = (np.abs(spec) ** 2) / n if n else np.zeros(0)

    guard = max(1, int(dc_guard_bins))
    fft_integral = float(np.sum(psd_like[guard:])) if psd_like.size > guard else float("nan")

    # RIN-like ratio: variance / mean^2 (intensity noise).
    if mean_val > 0 and np.isfinite(std_ac):
        rin = (std_ac ** 2) / (mean_val ** 2)
        if sample_rate_hz and sample_rate_hz > 0:
            analysis_bw = sample_rate_hz / 2.0
            rin_db = 10.0 * np.log10(rin / analysis_bw)
        else:
            rin_db = 10.0 * np.log10(rin) if rin > 0 else float("nan")
    else:
        rin = float("nan")
        rin_db = float("nan")

    return {
        "mean": mean_val,
        "std_ac_rms": std_ac,
        "peak_to_peak": pkpk,
        "cv": cv,
        "fft_integral": fft_integral,
        "rin": rin,
        "rin_db": rin_db,
    }


