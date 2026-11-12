"""Time-domain filters for ADC samples."""
from functools import lru_cache

from scipy import signal as sps


def band_filter(band_hz, sample_rate):
    """4th-order Butterworth band-pass (low/high-pass when an edge is at 0 or Nyquist)."""
    nyquist = sample_rate / 2
    lo, hi = sorted(float(f) for f in band_hz)
    lo = min(lo, 0.95 * nyquist)
    hi = min(max(hi, lo * 1.05, lo + 1e3), 0.98 * nyquist)
    if lo <= 0:
        return sps.butter(4, hi, 'lowpass', fs=sample_rate, output='sos')
    if hi >= 0.98 * nyquist:
        return sps.butter(4, lo, 'highpass', fs=sample_rate, output='sos')
    return sps.butter(4, [lo, hi], 'bandpass', fs=sample_rate, output='sos')


def band_passed(x, band_hz, sample_rate):
    """``x`` band-passed with no delay (forwards and backwards through band_filter)."""
    lo, hi = band_hz
    return sps.sosfiltfilt(_band_sos(float(lo), float(hi), float(sample_rate)), x)


band_pass = band_passed


def notch(x, freq_hz, sample_rate, q=30.0):
    """``x`` with a narrow band around ``freq_hz`` removed, with no delay, for a
    pickup line. ``q``: the centre frequency over the notch's width."""
    b, a = sps.iirnotch(freq_hz, q, fs=sample_rate)
    return sps.filtfilt(b, a, x)


@lru_cache(maxsize=16)
def _band_sos(lo, hi, sample_rate):
    """band_filter, designed once per band (a live display asks every frame)."""
    return band_filter((lo, hi), sample_rate)
