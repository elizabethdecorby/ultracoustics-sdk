"""
Signal processing utilities – FFT, PSD, and data conversion.

Internal note: All pure-computation logic extracted from usb_fft_viewer.py lives here
so it can be reused without any GUI dependency.

Functions
---------
load_binary(path, dtype)
    Load raw ADC samples from a binary file on disk.
adc_to_uw(samples, baseline)
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
TRANSIMPEDANCE = 20_000         # 20 kΩ

# Derived
_ADC_TO_VOLTAGE = ADC_VREF / ADC_FULL_SCALE          # V / count
_VOLTAGE_TO_CURRENT = 1.0 / TRANSIMPEDANCE           # A / V
_ADC_TO_CURRENT_UA = _ADC_TO_VOLTAGE * _VOLTAGE_TO_CURRENT * 1e6  # µA / count

# Responsivity: 140 µA ≙ 156 µW → 0.897 µA/µW
_RESPONSIVITY = 140.0 / 156.0   # µA / µW
_DIFFERENTIAL_FACTOR = 0.5      # differential mode sees half

ADC_TO_POWER_UW = (_ADC_TO_CURRENT_UA / _RESPONSIVITY) * _DIFFERENTIAL_FACTOR
"""µW per ADC count (with differential-mode correction)."""


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

def adc_to_uw(samples, baseline=0.0):
    """Convert raw ADC counts to optical power in µW.

    Applies the full signal chain: ADC→voltage→current→optical power,
    including the 0.5× differential-mode correction factor.

    Args:
        samples: Array of uint16 ADC values (14-bit range 0–16383).
        baseline: ADC-count DC offset to subtract before conversion.

    Returns:
        numpy.ndarray: Optical power values in µW (float64).
    """
    return (samples.astype(np.float64) - baseline) * ADC_TO_POWER_UW


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


