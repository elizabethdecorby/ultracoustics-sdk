"""Pressure and sound pressure level from a calibrated spectrum.

Three different numbers answer "how loud is it", and each has its own name here:

- :attr:`PressureSpectrum.spl`, a density in dB SPL/√Hz, for noise and broadband
  signals. It does not depend on the FFT size.
- :meth:`PressureSpectrum.band_level` and :meth:`~PressureSpectrum.octave_levels`,
  in dB SPL, for the total sound in a band.
- :func:`tone_level`, in dB SPL, for a steady tone.

The GUI reads all of these through the same functions.
"""
from dataclasses import dataclass
import math
import warnings

import numpy as np

from . import bands
from .calibration import (P_REF_PA, POWER_DRIFT_WARNING, calibrated_band, power_drift, pressure_psd,
                          transfer_w2_per_pa2)
from .filters import band_passed
from .transients import measure_tone, samples_for_bandwidth, tone_track
from ..readout import ADC_TO_POWER_UW

#: Most lock-in windows tone_level averages, spread over the recording.
MAX_TONE_WINDOWS = 64
#: Most the sensitivity may swing across a band before one Pa scale is doubtful.
MAX_RIPPLE_DB = 1.0
#: A display's upper limit: off resonance the divisor is tiny, so what reaches it is
#: noise over almost no sensitivity, not sound.
DISPLAY_HEADROOM = 1e4
SPL_120_PA2HZ = 400.0  # (20 µPa)² × 10^(120/10)


def spl_db(pa2):
    """dB re 20 µPa of a mean-square pressure, in Pa² or Pa²/Hz; NaN below zero."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return 10.0 * np.log10(np.asarray(pa2, dtype=np.float64) / P_REF_PA ** 2)


def display_ceiling_pa2hz(cal):
    """The top of a displayed pressure density, Pa²/Hz."""
    return max(cal.nep2 * DISPLAY_HEADROOM, SPL_120_PA2HZ)


def display_pa2hz(cal, freqs, psd_w2hz):
    """Pressure density as the GUI plots it, Pa²/Hz: held between NEP², below which the
    ratio is estimator scatter, and :func:`display_ceiling_pa2hz`. For display only:
    integrating clipped bins would bias a level."""
    return np.clip(pressure_psd(cal, freqs, psd_w2hz), cal.floor_pa2hz, display_ceiling_pa2hz(cal))


@dataclass(frozen=True)
class BandLevel:
    """The sound in one band."""
    p_rms_pa: float
    spl_db: float
    lo_hz: float
    hi_hz: float
    resolved: bool = True    # False: narrower than two FFT bins, so only indicative


@dataclass(frozen=True)
class PressureSpectrum:
    """A calibrated pressure spectrum, over the calibrated band only."""
    freqs: np.ndarray
    psd_pa2hz: np.ndarray
    sigma_pa2hz: np.ndarray  # one standard deviation of each bin
    num_averages: int

    @property
    def spl(self):
        """Pressure density in dB SPL/√Hz (re 20 µPa). NaN for a bin that came out
        negative after subtracting noise."""
        return spl_db(self.psd_pa2hz)

    def significant(self, sigma=3.0):
        """Mask of the bins more than ``sigma`` standard deviations above zero."""
        return self.psd_pa2hz > sigma * self.sigma_pa2hz

    def band_level(self, lo_hz, hi_hz):
        """The total sound between two frequencies, as a :class:`BandLevel`.

        Each bin counts for its whole width, so a band reads the same power as the
        octave bands across it. Negative bins left by a subtraction are kept while
        integrating, because clipping them would bias the level upward. A band that
        integrates to zero or less has a p_rms of 0. ValueError if the band misses
        this spectrum.
        """
        if self.freqs.size < 2:
            raise ValueError('this spectrum has no calibrated bins to integrate')
        df = float(self.freqs[1] - self.freqs[0])
        lo = max(float(min(lo_hz, hi_hz)), float(self.freqs[0]) - df / 2)
        hi = min(float(max(lo_hz, hi_hz)), float(self.freqs[-1]) + df / 2)
        if hi <= lo:
            raise ValueError(f'{lo_hz:g} to {hi_hz:g} Hz is outside this spectrum')
        power = max(float(bands.band_levels(self.freqs, self.psd_pa2hz, [(None, lo, hi)])[0][0]), 0.0)
        return BandLevel(math.sqrt(power), float(spl_db(power)) if power > 0 else -math.inf, lo, hi,
                         hi - lo >= 2 * df)

    def octave_levels(self, fraction=3, lo_hz=None, hi_hz=None):
        """Levels in 1/``fraction``-octave bands, as :class:`.bands.OctaveLevels`."""
        return bands.octave_levels(self.freqs, self.psd_pa2hz, fraction, lo_hz, hi_hz, P_REF_PA ** 2, 'dB SPL')


def to_pressure(cal, spectrum, check_power=True):
    """``spectrum`` in pressure units, over the calibrated band.

    Noise stays in unless it was subtracted first, so off resonance the result
    rises above NEP² where the detector, not the dome's thermal motion, limits
    detection. A warning if the optical power moved since the calibration, as
    the scale depends on it (``check_power=False`` for a caller that shows this
    itself, as the GUI does).
    """
    if check_power:
        drift = power_drift(cal, spectrum.mean_counts)
        if drift is not None and abs(drift) > POWER_DRIFT_WARNING:
            warnings.warn(f'optical power changed {drift:+.0%} since this calibration: its scale may no longer '
                          'hold', stacklevel=3)
    keep = (spectrum.freqs > 0) & calibrated_band(cal, spectrum.freqs)
    sensitivity = np.maximum(transfer_w2_per_pa2(cal, spectrum.freqs[keep]), 1e-300)
    return PressureSpectrum(spectrum.freqs[keep], spectrum.psd_w2hz[keep] / sensitivity,
                            spectrum.sigma[keep] / sensitivity, spectrum.num_averages)


@dataclass(frozen=True)
class ToneLevel:
    """A tone's level, read by a lock-in at exactly one frequency."""
    freq_hz: float
    p_rms_pa: float        # None without a calibration
    spl_db: float          # None without a calibration
    bandwidth_hz: float    # the measurement's noise bandwidth
    power_w2: float        # mean-square optical power
    phase_rad: float = 0.0

    @property
    def power_db(self):
        return 10.0 * math.log10(self.power_w2 + 1e-300)


@dataclass(frozen=True)
class ToneSeries:
    """A tone's level through time, one reading per lock-in window."""
    times_s: np.ndarray    # window centres
    power_w2: np.ndarray
    spl_db: np.ndarray     # None without a calibration
    phase_rad: np.ndarray

    @property
    def power_db(self):
        with np.errstate(divide='ignore'):
            return 10.0 * np.log10(self.power_w2)


def _tone_pa2(cal, freq_hz, power_w2):
    return pressure_psd(cal, np.full(np.shape(power_w2), float(freq_hz)), power_w2)


def read_tone(samples, sample_rate, freq_hz, cal=None, start_index=0, uw_per_count=ADC_TO_POWER_UW):
    """One lock-in reading of all of ``samples`` at ``freq_hz``, as a :class:`ToneLevel`.

    ``start_index`` is the stream index of ``samples[0]``, so a steady tone reads
    the same phase from every block. Without ``cal`` it gives optical power only.
    """
    tone = measure_tone(samples, sample_rate, freq_hz, start_index=start_index)
    power = tone.power * (uw_per_count * 1e-6) ** 2
    if cal is None:
        return ToneLevel(float(freq_hz), None, None, tone.bandwidth_hz, power, tone.phase_rad)
    pa2 = float(_tone_pa2(cal, freq_hz, np.array([power]))[0])
    return ToneLevel(float(freq_hz), math.sqrt(pa2), float(spl_db(pa2)), tone.bandwidth_hz, power, tone.phase_rad)


def tone_series(samples, sample_rate, freq_hz, bandwidth_hz, cal=None, points=400, start_index=0,
                uw_per_count=ADC_TO_POWER_UW):
    """Lock-in readings across ``samples``, as a :class:`ToneSeries`: windows of the
    length ``bandwidth_hz`` needs, at up to ``points`` even positions."""
    times, amplitude, phase = tone_track(samples, sample_rate, freq_hz, bandwidth_hz, points=points,
                                         start_index=start_index)
    power = amplitude ** 2 / 2 * (uw_per_count * 1e-6) ** 2
    return ToneSeries(times, power, None if cal is None else spl_db(_tone_pa2(cal, freq_hz, power)), phase)


def tone_level(recording, cal, freq_hz, bandwidth_hz=10.0):
    """The level of a steady tone at ``freq_hz`` in ``recording``, at any frequency.

    Narrow ``bandwidth_hz`` rejects more noise. Windows that long are read across
    the recording and their powers averaged.
    """
    window = samples_for_bandwidth(recording.sample_rate, bandwidth_hz)
    if recording.length < window:
        raise ValueError(f'{recording.duration_s:g} s of data is too short for a {bandwidth_hz:g} Hz bandwidth')
    series = tone_series(recording.samples, recording.sample_rate, freq_hz, bandwidth_hz, cal,
                         points=min(recording.length // window, MAX_TONE_WINDOWS),
                         uw_per_count=recording.uw_per_count)
    power = float(np.mean(series.power_w2))
    pa2 = float(_tone_pa2(cal, freq_hz, np.array([power]))[0])
    return ToneLevel(float(freq_hz), math.sqrt(pa2), float(spl_db(pa2)), float(bandwidth_hz), power,
                     float(series.phase_rad[-1]))


def sensitivity_ripple_db(cal, band):
    """How far the sensitivity swings across ``band`` (lo, hi) Hz, in dB."""
    lo, hi = sorted(float(f) for f in band)
    sensitivity = transfer_w2_per_pa2(cal, np.geomspace(max(lo, 1.0), hi, 400))
    return float(10.0 * np.log10(sensitivity.max() / sensitivity.min()))


def pa_per_uw(cal, band):
    """(Pa per µW, the frequency it was taken at) for a waveform in ``band`` (lo, hi) Hz,
    from the sensitivity at the band's geometric centre."""
    lo, hi = sorted(float(f) for f in band)
    f_ref = math.sqrt(max(lo, hi / 10) * hi)
    transfer = float(transfer_w2_per_pa2(cal, np.array([f_ref]))[0])
    return 1e-6 / math.sqrt(max(transfer, 1e-300)), f_ref


def pressure_waveform(cal, power_uw, band, sample_rate):
    """A waveform in Pa from optical power in µW, band-limited to ``band`` (lo, hi) Hz.

    This is one scale factor, so it only holds where the sensitivity is flat:
    below about 500 kHz on the probes measured. A warning if the sensitivity swings
    by more than MAX_RIPPLE_DB across the band, as it does over a resonance: use a
    narrower band, or tone_level for a tone. Thermal noise stays in: it passes
    through the dome as sound does, so only a narrower band reduces it.
    """
    ripple = sensitivity_ripple_db(cal, band)
    if ripple > MAX_RIPPLE_DB:
        warnings.warn(f'the sensitivity swings {ripple:.1f} dB across {band[0]:g}-{band[1]:g} Hz, so this '
                      'waveform\'s scale is only right near the band\'s centre', stacklevel=3)
    lo, hi = sorted(float(f) for f in band)
    x = np.asarray(power_uw, dtype=np.float64)
    return band_passed(x - x.mean(), (lo, hi), sample_rate) * pa_per_uw(cal, band)[0]
