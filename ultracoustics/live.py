"""The spectrum of the live stream, as the GUI draws it.

For a narrow view the stream is decimated to just that band as it arrives (a
zoom FFT), so each update only filters the new samples and transforms a short
block; the bin width stays whatever ``fft_size`` asks for. A wide view (or
``zoom=False``) transforms the newest samples at the full rate.

    live = LiveSpectrum()
    live.configure(0.0, 200e3, fft_size=16384, averages=8)
    while watching:
        spectrum = live.update(ctrl)    # a Spectrum, or None until enough has arrived
"""
import dataclasses

import numpy as np

from .analysis import spectrum as spectral
from .config import SAMPLE_RATE
from .readout import ADC_TO_POWER_UW
from .recorder import copy_span


class LiveSpectrum:
    """The spectrum of the live stream, updated as samples arrive: for a live plot.

    Make one, :meth:`configure` the frequency range and resolution, then call
    :meth:`update` with a connected :class:`~ultracoustics.Controller` each time
    the plot redraws. For a spectrum of data you already have, use
    :meth:`Recording.spectrum <ultracoustics.Recording.spectrum>` instead: it
    averages every FFT, where this one shows only the newest ``averages``.

    ::

        live = uc.LiveSpectrum()
        live.configure(0.0, 200e3, fft_size=16384, averages=8)
        while watching:
            spectrum = live.update(ctrl)    # a Spectrum in W²/Hz, or None until enough has arrived
    """

    def __init__(self, fs=SAMPLE_RATE, uw_per_count=ADC_TO_POWER_UW):
        """*fs* is the sample rate in Hz and *uw_per_count* the optical power of one
        ADC count: leave both alone for a BROADSONIC."""
        self.fs = fs
        self.scale = (uw_per_count * 1e-6) ** 2
        self.plan = spectral.plan_zoom(fs, 0.0, fs / 2, enabled=False)
        self.fft_size, self.averages = 8192, 1
        self.estimator, self.zoom = 'hann', True
        self.stream = None
        self._warmup = 0
        self.mean_counts = 0.0  # of the newest data, for the calibration's shot noise

    def configure(self, f_lo, f_hi, fft_size, averages, estimator='hann', zoom=True):
        """Set the view (*f_lo* to *f_hi*, Hz) and the resolution; True if the
        frequency bins change.

        The bin width is ``sample rate / fft_size``, and *averages* is how many of
        the newest FFTs each spectrum averages. *estimator* is ``'hann'`` or
        ``'multitaper'``. With *zoom*, a narrow view is computed from the stream
        reduced to that band, which is much faster at the same bin width.
        """
        plan = self._plan_for(max(0.0, f_lo), min(self.fs / 2, f_hi), zoom)
        changed = (plan != self.plan or fft_size != self.fft_size)
        resized = changed or averages != self.averages
        self.plan, self.fft_size, self.averages = plan, fft_size, averages
        self.estimator, self.zoom = estimator, zoom
        if resized:
            self.stream = None  # rebuilt (and refilled from the ring) on the next update
        if plan.decimation > 1 and changed:
            # Input the decimation filters need before their output is valid
            self._warmup = (spectral.Decimator(plan).warmup + 1) * plan.decimation
        return changed

    def _plan_for(self, f_lo, f_hi, zoom):
        """Keep the current decimation while it still covers the view and is
        still the right factor (panning shouldn't refill the stream); a new plan
        shows all of its trusted band, not just the view."""
        plan = spectral.plan_zoom(self.fs, f_lo, f_hi, enabled=zoom)
        current = self.plan
        if (plan.decimation == current.decimation > 1 and plan.complex == current.complex
                and _covers(current, f_lo, f_hi)):
            return current
        return _widened(plan)

    @property
    def nfft(self):
        """Points in each FFT actually computed: ``fft_size``, or fewer in a zoomed view."""
        return self.plan.nfft(self.fft_size)

    def needed(self):
        """Input samples one spectrum uses."""
        return self.fft_size * self.averages

    def _spectrum(self, freqs, density):
        tapers = 1 if self.estimator == 'hann' else int(2 * spectral.MULTITAPER_NW) - 1
        return spectral.Spectrum(freqs, density * self.scale, self.mean_counts, self.fft_size,
                                 self.averages * tapers)

    def update(self, controller):
        """A :class:`~ultracoustics.Spectrum` of the newest data from *controller*, or
        None if not enough has arrived yet."""
        total = int(controller.samples_received)
        if total < self.needed():
            return None
        if self.plan.decimation == 1:
            data = copy_span(controller, total - self.needed(), total)
            return None if data is None else self.compute(data)
        if self.stream is None:
            self.stream = spectral.ZoomStream(self.plan, self.nfft * self.averages)
        stream = self.stream
        behind = stream.next_index is None or not 0 <= total - stream.next_index <= self.needed()
        if behind:
            # New stream, a restarted one, or a backlog longer than one spectrum
            # needs: start again from just what that spectrum needs (filtering
            # less than the backlog would cost)
            stream.reset()
            start = max(0, total - self.needed() - self._warmup)
        else:
            start = stream.next_index
        if total > start:
            data = copy_span(controller, start, total)
            if data is None:
                self.stream = None
                return None
            stream.feed(data, start)
            self.mean_counts = float(data[-min(data.size, 1 << 20):].mean())
        if stream.available < self.nfft:
            return None
        return self._spectrum(*stream.psd(self.fft_size, self.averages, self.estimator, spectral.MULTITAPER_NW))

    def compute(self, samples):
        """The same spectrum from a block already in hand (e.g. paused data)."""
        samples = np.asarray(samples)
        self.mean_counts = float(samples[-min(samples.size, 1 << 20):].mean())
        if self.plan.decimation == 1:
            n = min(self.fft_size, samples.size)
            averages = max(1, min(self.averages, samples.size // n))
            segments = samples[samples.size - n * averages:].reshape(averages, n)
            return self._spectrum(*spectral.segment_psd(segments.astype(np.float64), self.fs, self.estimator,
                                                        spectral.MULTITAPER_NW))
        # Just what one spectrum needs, plus the filters' warm-up
        samples = samples[-(self.needed() + self._warmup + self.plan.decimation):]
        return self._spectrum(*spectral.zoom_psd(samples, self.plan, self.fft_size, self.averages,
                                                 self.estimator, spectral.MULTITAPER_NW))

    def describe(self):
        """One line saying how the spectrum is being computed, for a display."""
        plan = self.plan
        if plan.decimation == 1:
            return f'Full rate · {self.nfft:,}-pt FFT'
        return (f'Zoom ×{plan.decimation} · {plan.out_rate / 1e3:.3g} kS/s · '
                f'{self.nfft:,}-pt FFT')


def _trusted(plan):
    """(lo, hi) of the band a plan's decimated signal shows faithfully."""
    half = spectral.PASSBAND * plan.out_rate / 2
    if plan.decimation == 1:
        return 0.0, plan.fs / 2
    if plan.complex:
        return max(0.0, plan.center_hz - half), min(plan.fs / 2, plan.center_hz + half)
    return 0.0, half


def _covers(plan, f_lo, f_hi):
    lo, hi = _trusted(plan)
    return lo <= f_lo and f_hi <= hi


def _widened(plan):
    if plan.decimation == 1:
        return plan
    lo, hi = _trusted(plan)
    return dataclasses.replace(plan, f_lo=lo, f_hi=hi)
