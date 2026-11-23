"""What a spectrum's noise is made of.

The calibration's fitted model gives each term on the spectrum's bins::

    S(f) = dark(f) + G(f) * (shot + white + thermal(f) + broad(f))

``G`` is the anti-alias filter's power gain and ``thermal`` the dome modes that
set the pressure scale. ``broad`` holds the broad features of the readout's
response that the fit also finds. Dividing by the sensitivity refers each term
to pressure, where the thermal noise is flat at NEP².

Thermal noise passes through the dome as sound does, so it cannot be filtered
out of a single waveform. It can only be subtracted from averaged power, reduced
by narrowing the band, or averaged down over repeated events.
"""
from dataclasses import dataclass, replace

import numpy as np

from .calibration import noise_terms, transfer_w2_per_pa2
from .pressure import spl_db
from .spectrum import Spectrum

#: Below this the measured noise rises above the model (a 1/f-like excess the
#: fit does not cover), so subtract a measured background there instead.
MODEL_LOW_HZ = 500e3
#: What can lift the floor above the thermal noise, in order of preference on a tie.
SOURCES = ('shot', 'white', 'dark', 'broad', 'excess')


@dataclass(frozen=True)
class NoiseBudget:
    """Each noise term of a spectrum, in W²/Hz on its bins."""
    spectrum: Spectrum
    dark: np.ndarray       # the electronics: zeros if the calibration has no dark spectrum
    shot: np.ndarray
    white: np.ndarray      # the fit's white floor above dark and shot, such as laser noise
    thermal: np.ndarray    # the dome modes in the pressure scale
    broad: np.ndarray      # broad features of the readout's response
    sensitivity: np.ndarray  # W²/Hz per Pa²/Hz

    @property
    def freqs(self):
        """The frequency of each bin, Hz: the spectrum's own."""
        return self.spectrum.freqs

    @property
    def model(self):
        """All the fitted noise."""
        return self.dark + self.shot + self.white + self.thermal + self.broad

    @property
    def excess(self):
        """The measured spectrum above the model: sound, or noise the fit does not cover."""
        return self.spectrum.psd_w2hz - self.model

    @property
    def limited_by(self):
        """What limits detection at each bin, by name: 'thermal' while everything else
        adds up to less than the dome's own motion, so the floor is within 3 dB of the
        NEP; otherwise the largest of ``SOURCES``."""
        others = np.vstack((self.shot, self.white, self.dark, self.broad, np.maximum(self.excess, 0.0)))
        return np.where(others.sum(axis=0) < self.thermal, 'thermal', np.array(SOURCES)[np.argmax(others, axis=0)])

    def limited_by_at(self, freq_hz):
        """What limits detection at the bin nearest ``freq_hz``."""
        return str(self.limited_by[int(np.argmin(np.abs(self.freqs - freq_hz)))])

    @property
    def noise_spl(self):
        """The fitted noise referred to pressure, in dB SPL/√Hz."""
        return spl_db(self.model / np.maximum(self.sensitivity, 1e-300))

    def residual(self):
        """The spectrum less the noise model, as a :class:`Spectrum`.

        NaN below MODEL_LOW_HZ, where the model does not hold: subtract a
        measured background there with :meth:`Spectrum.subtract`.
        """
        kept = np.where(self.freqs >= MODEL_LOW_HZ, self.excess, np.nan)
        return replace(self.spectrum, psd_w2hz=kept, sigma_w2hz=self.spectrum.sigma)


def noise_budget(cal, spectrum):
    """The :class:`NoiseBudget` of ``spectrum`` under the calibration ``cal``."""
    return NoiseBudget(spectrum, **noise_terms(cal, spectrum.freqs, spectrum.mean_counts),
                       sensitivity=transfer_w2_per_pa2(cal, spectrum.freqs))
