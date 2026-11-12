"""Fractional-octave bands (IEC 61260-1) and the level in each."""
from dataclasses import dataclass
import math

import numpy as np

LOWEST_BAND_HZ = 10.0

# IEC 61260-1 preferred labels for 1/3-octave (and octave) mid-band frequencies.
_PREFERRED = ('1', '1.25', '1.6', '2', '2.5', '3.15', '4', '5', '6.3', '8')
_G = 10 ** 0.3   # base-ten octave ratio


def _nominal(x, fraction, exact):
    if fraction in (1, 3):
        i = x * (3 // fraction)    # index in the 1/3-octave series
        return float(f'{_PREFERRED[i % 10]}e{3 + i // 10}')
    return float(f'{exact:.3g}')


def octave_bands(f_lo, f_hi, fraction=3):
    """IEC 61260-1 base-ten 1/fraction-octave bands lying fully inside [f_lo, f_hi].

    Returns ascending (nominal_hz, lower_hz, upper_hz) tuples.
    """
    b = int(fraction)
    f_lo = max(float(f_lo), 1.0)    # the log-spaced series needs a positive floor
    half = _G ** (1 / (2 * b))
    odd = 0 if b % 2 else 1         # even fractions sit between the odd-b mid-bands
    x = math.floor(b * math.log(f_lo / 1000, _G)) - 1
    bands = []
    while True:
        fm = 1000 * _G ** ((2 * x + odd) / (2 * b))
        if fm * half > f_hi:
            return bands
        if fm / half >= f_lo:
            bands.append((_nominal(x, b, fm), fm / half, fm * half))
        x += 1


def band_levels(freqs, density, bands):
    """Integrate a linear density over each band: (power, resolved).

    Each bin covers [f - df/2, f + df/2], split proportionally at band edges.
    ``power`` is NaN where the data doesn't cover the band; ``resolved`` marks
    bands at least two bins wide.
    """
    freqs = np.asarray(freqs, np.float64)
    density = np.asarray(density, np.float64)
    df = (freqs[-1] - freqs[0]) / (len(freqs) - 1)
    edges = np.append(freqs - df / 2, freqs[-1] + df / 2)
    # Piecewise-constant density integrates to a piecewise-linear cumulative sum.
    cumulative = np.append(0.0, np.cumsum(density * df))
    lower = np.array([band[1] for band in bands], np.float64)
    upper = np.array([band[2] for band in bands], np.float64)
    power = np.interp(upper, edges, cumulative) - np.interp(lower, edges, cumulative)
    power[(lower < edges[0]) | (upper > edges[-1])] = np.nan
    return power, upper - lower >= 2 * df


@dataclass(frozen=True)
class OctaveLevels:
    """Levels in fractional-octave bands (IEC 61260-1)."""
    center_hz: np.ndarray    # nominal mid-band frequencies
    lower_hz: np.ndarray
    upper_hz: np.ndarray
    power: np.ndarray        # in each band: Pa², or W²
    level_db: np.ndarray
    unit: str                # 'dB SPL' or 'dB re 1 W²'
    resolved: np.ndarray     # False: narrower than two FFT bins, so only indicative


def octave_levels(freqs, density, fraction, lo_hz, hi_hz, reference, unit):
    """:class:`OctaveLevels` of a density, from the first band at least one bin wide.
    ValueError if no band fits."""
    freqs = np.asarray(freqs, np.float64)
    if freqs.size < 2:
        raise ValueError('a band level needs at least two frequency bins')
    bin_hz = float(freqs[1] - freqs[0])
    width = 2 ** (1 / (2 * fraction)) - 2 ** (-1 / (2 * fraction))
    lo = max(lo_hz or 0.0, float(freqs[0]), LOWEST_BAND_HZ, bin_hz / width)
    hi = min(hi_hz or math.inf, float(freqs[-1]))
    chosen = octave_bands(lo, hi, fraction)
    if not chosen:
        raise ValueError(f'no 1/{fraction}-octave band fits between {lo:g} and {hi:g} Hz')
    power, resolved = band_levels(freqs, density, chosen)
    with np.errstate(divide='ignore', invalid='ignore'):
        level = 10.0 * np.log10(power / reference)
    return OctaveLevels(*np.array(chosen).T, power, level, unit, resolved)
