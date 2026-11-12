"""Power spectra of ADC samples: Welch, multitaper, zoom FFT and spectrograms.

A narrow view [f_lo, f_hi] is decimated first (a real low-pass, or a complex
mix-to-baseband plus low-pass) so a short FFT gives the same bin width as a
huge full-rate one. Every density returned here is linear in input units²/Hz
(ADC counts²/Hz); unit conversion and dB are the caller's job.

``samples`` may be anything sliceable with len() (an ndarray, a memmap, a split
recording), read a chunk or a window at a time. Times are seconds from
samples[0].
"""
from dataclasses import dataclass, field, replace
from datetime import datetime
from functools import lru_cache
import math
import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import fft as sfft
from scipy import signal as sps
from scipy.fft import next_fast_len

from .bands import octave_levels

#: Relative change in optical power past which two spectra are not comparable.
POWER_DRIFT_WARNING = 0.10


def _now():
    return datetime.now().isoformat(timespec='seconds')


@dataclass
class Spectrum:
    """An averaged one-sided power spectral density of the optical signal.

    ``freqs`` are in Hz and ``psd_w2hz`` in W²/Hz. ``mean_counts`` is the raw
    mean the data was taken at. ``num_averages`` sets the statistics: one bin
    scatters by ``psd / sqrt(num_averages)``, which :attr:`sigma` reports.
    """
    freqs: np.ndarray
    psd_w2hz: np.ndarray
    mean_counts: float
    fft_size: int
    num_averages: int
    captured_at: str = field(default_factory=_now)
    sigma_w2hz: np.ndarray = None   # set where it is not psd / sqrt(num_averages)
    each_w2hz: np.ndarray = None    # every FFT's own spectrum, (FFTs, freqs), when asked for

    @property
    def sigma(self):
        """One standard deviation of each bin, W²/Hz."""
        if self.sigma_w2hz is not None:
            return self.sigma_w2hz
        return self.psd_w2hz / math.sqrt(self.num_averages)

    def on_grid(self, freqs):
        return np.interp(freqs, self.freqs, self.psd_w2hz,
                         left=self.psd_w2hz[0], right=self.psd_w2hz[-1])

    def subtract(self, background):
        """This spectrum less a measured ``background``, bin by bin.

        Bins that come out negative stay negative: they are scatter, and
        clipping them would bias any band level upward. The result's
        :attr:`sigma` is the uncertainty of the difference, so
        :meth:`significant` says which bins truly stand above the background.

        ValueError if the frequency bins differ. A warning if the optical power
        differs by more than POWER_DRIFT_WARNING: the background's shot noise
        then no longer matches.
        """
        if not np.array_equal(self.freqs, background.freqs):
            raise ValueError('the spectra have different frequency bins: use the same fft_size for both')
        if background.mean_counts and (
                abs(self.mean_counts / background.mean_counts - 1.0) > POWER_DRIFT_WARNING):
            warnings.warn(f'mean level {self.mean_counts:.0f} counts against {background.mean_counts:.0f} '
                          'in the background: the optical power changed, so its noise does not match',
                          stacklevel=2)
        return replace(self, psd_w2hz=self.psd_w2hz - background.psd_w2hz,
                       sigma_w2hz=np.hypot(self.sigma, background.sigma))

    def significant(self, sigma=3.0):
        """Mask of the bins more than ``sigma`` standard deviations above zero."""
        return self.psd_w2hz > sigma * self.sigma

    def difference_db(self, reference):
        """This spectrum over ``reference``, in dB at each of this one's bins. A ratio, so
        the probe's sensitivity cancels. :meth:`subtract` gives the excess power instead."""
        return db_difference(self.freqs, 10.0 * np.log10(self.psd_w2hz + 1e-30), reference.freqs,
                             10.0 * np.log10(reference.psd_w2hz + 1e-30))

    def octave_levels(self, fraction=3, lo_hz=None, hi_hz=None):
        """Optical power in 1/``fraction``-octave bands, in dB re 1 W², as
        :class:`~.bands.OctaveLevels`."""
        return octave_levels(self.freqs, self.psd_w2hz, fraction, lo_hz, hi_hz, 1.0, 'dB re 1 W²')


def db_difference(freqs, db, reference_freqs, reference_db):
    """A dB spectrum less a reference's, on the first one's bins."""
    return np.asarray(db) - np.interp(freqs, reference_freqs, reference_db)


def pool_spectra(spectra):
    """Several spectra's FFTs as one spectrum, on the first's bins: a coarser one's
    FFTs are interpolated onto them. Each needs its ``each_w2hz``."""
    first = spectra[0]
    rows = [s.each_w2hz if len(s.freqs) == len(first.freqs) and np.allclose(s.freqs, first.freqs)
            else np.array([np.interp(first.freqs, s.freqs, row) for row in s.each_w2hz])
            for s in spectra]
    every = np.concatenate(rows)
    mean_counts = float(np.average([s.mean_counts for s in spectra], weights=[len(r) for r in rows]))
    return replace(first, psd_w2hz=every.mean(axis=0), mean_counts=mean_counts, each_w2hz=every,
                   num_averages=sum(s.num_averages for s in spectra), sigma_w2hz=None)


PASSBAND = 0.8   # fraction of the decimated Nyquist band that is shown/trusted
STOPBAND_DB = 100.0
# The mixing frequency sits on a fs/MIX_PERIOD grid, so the oscillator repeats
# within MIX_PERIOD samples and is a table lookup rather than np.exp per sample.
MIX_PERIOD = 2 ** 16


@dataclass(frozen=True)
class ZoomPlan:
    """How a view band is decimated before the FFT."""
    fs: float            # input sample rate
    f_lo: float          # band to show
    f_hi: float
    decimation: int      # total factor D, power of 2; 1 = no zoom
    complex: bool        # True: mixed to baseband around center_hz
    center_hz: float     # mixing frequency (0.0 when not complex)

    @property
    def out_rate(self) -> float:
        return self.fs / self.decimation

    def nfft(self, fft_size: int) -> int:
        """FFT length giving the full-rate bin width fs/fft_size."""
        return max(fft_size // self.decimation, 16)


def _pow2_at_most(x, cap):
    """Largest power of 2 not above x or cap (at least 1)."""
    d = 1
    while d * 2 <= min(x, cap):
        d *= 2
    return d


def plan_zoom(fs, f_lo, f_hi, enabled=True, max_decimation=4096):
    """Pick the largest decimation whose trusted band still covers [f_lo, f_hi]."""
    fs, f_lo, f_hi = float(fs), float(f_lo), float(f_hi)
    if not enabled or f_hi <= f_lo:
        return ZoomPlan(fs, f_lo, f_hi, 1, False, 0.0)
    edge = PASSBAND * fs / 2
    d_real = _pow2_at_most(edge / f_hi, max_decimation)
    grid = fs / MIX_PERIOD
    center = round((f_lo + f_hi) / 2 / grid) * grid
    d_cplx = _pow2_at_most(edge / max(center - f_lo, f_hi - center), max_decimation)
    # A complex sample is two real numbers, so mixing only pays at more than
    # twice the real factor; at equal data rates the real path is preferred
    # (it keeps DC removal and needs no oscillator).
    if d_cplx > 2 * d_real:
        return ZoomPlan(fs, f_lo, f_hi, d_cplx, True, center)
    return ZoomPlan(fs, f_lo, f_hi, d_real, False, 0.0)


def _stage_factors(decimation):
    """8s first, the remainder, then a final 2: fewest taps per input sample."""
    if decimation <= 1:
        return []
    factors, rest = [], decimation // 2
    while rest >= 8:
        factors.append(8)
        rest //= 8
    if rest > 1:
        factors.append(rest)
    return factors + [2]


@lru_cache(maxsize=64)
def _stage_taps(rate_in, factor, fp):
    """Kaiser low-pass: flat to fp, >= STOPBAND_DB from (rate_out - fp) up.

    Only images that would land in the final trusted band matter, so early
    full-rate stages get a very wide transition and stay short. kaiserord's
    length formula runs up to ~3 dB short for short filters, so the design
    is checked and tightened until it really meets STOPBAND_DB.
    """
    rate_out = rate_in / factor
    width = (rate_out - 2 * fp) / (rate_in / 2)
    for target in np.arange(STOPBAND_DB, STOPBAND_DB + 10, 0.5):
        numtaps, beta = sps.kaiserord(target, width)
        numtaps = -(-numtaps // factor) * factor   # whole polyphase rows
        taps = sps.firwin(numtaps, rate_out / 2, window=('kaiser', beta), fs=rate_in)
        f, h = sps.freqz(taps, worN=1 << 16, fs=rate_in)
        if np.max(np.abs(h[f >= rate_out - fp])) <= 10 ** (-STOPBAND_DB / 20):
            break
    taps.flags.writeable = False
    return taps


class Decimator:
    """Streaming multistage FIR decimator, optionally mixing to baseband first.

    Filters start from zero state, so N inputs give about N/decimation outputs
    of which the first ``warmup`` are spoiled. Output j is centred on input
    sample ``first_index + j * decimation - delay``.
    """

    def __init__(self, plan):
        self.plan = plan
        fp = PASSBAND * plan.out_rate / 2
        rate, step, self.delay, self.warmup = plan.fs, 1, 0.0, 0
        self.stages, self._kernels = [], []
        for factor in _stage_factors(plan.decimation):
            taps = _stage_taps(rate, factor, fp)
            self.stages.append((factor, taps))
            self._kernels.append(taps[::-1].copy())   # contiguous: einsum's fast path
            self.delay += (len(taps) - 1) / 2 * step
            # outputs whose window reaches the zero state or spoiled inputs
            self.warmup = -(-(self.warmup + len(taps) - 1) // factor)
            rate /= factor
            step *= factor
        if plan.complex:
            k = round(plan.center_hz * MIX_PERIOD / plan.fs)
            if not math.isclose(k * plan.fs / MIX_PERIOD, plan.center_hz):
                raise ValueError('center_hz must lie on the fs/2**16 grid')
            phase = 2 * np.pi * ((k * np.arange(MIX_PERIOD)) % MIX_PERIOD) / MIX_PERIOD
            self._lo = np.vstack((np.cos(phase), -np.sin(phase)))
        self.reset()

    def reset(self):
        """Clear filter state; the next chunk may start at any index."""
        self.next_index = None
        self._offset = None
        channels = 2 if self.plan.complex else 1
        self._tails = [np.zeros((channels, len(taps) - 1)) for _, taps in self.stages]

    def _mix(self, x, start):
        """Multiply by exp(-j·2π·fc·n/fs), n the absolute sample index, as (re, im) rows."""
        out = np.empty((2, len(x)))
        i, phase = 0, start % MIX_PERIOD
        while i < len(x):
            n = min(len(x) - i, MIX_PERIOD - phase)
            np.multiply(x[i:i + n], self._lo[:, phase:phase + n], out=out[:, i:i + n])
            i, phase = i + n, 0
        return out

    def process(self, chunk, start_index):
        """Decimate the next contiguous chunk; float64 (real) or complex128 output."""
        if self.next_index is not None and start_index != self.next_index:
            raise ValueError(f'chunk starts at {start_index}, expected {self.next_index}')
        x = np.asarray(chunk, dtype=np.float64)
        self.next_index = start_index + len(x)
        if not self.stages and not self.plan.complex:
            return x
        if not len(x):
            return np.empty(0, np.complex128 if self.plan.complex else np.float64)
        if self.plan.complex:
            # Mixing moves the ADC's large DC offset to -center_hz, where the
            # 100 dB stopband still leaves a visible spur once it aliases in.
            # A fixed reference (the stream's first sample) removes most of
            # it without making the output depend on chunking.
            if self._offset is None:
                self._offset = x[0]
            x = self._mix(x - self._offset, start_index)
        else:
            x = x[None]
        for i, ((factor, _), kernel) in enumerate(zip(self.stages, self._kernels)):
            x = np.concatenate((self._tails[i], x), axis=1)
            count = max((x.shape[1] - len(kernel)) // factor + 1, 0)
            if count:
                windows = sliding_window_view(x, len(kernel), axis=1)[:, ::factor]
                self._tails[i] = x[:, count * factor:].copy()
                x = np.einsum('cij,j->ci', windows, kernel)
            else:
                self._tails[i], x = x, x[:, :0]
        return x[0] + 1j * x[1] if self.plan.complex else x[0]


class ZoomStream:
    """Decimator plus a ring buffer of the last ``capacity`` decimated samples."""

    def __init__(self, plan, capacity):
        self.plan = plan
        self.capacity = int(capacity)
        self._decimator = Decimator(plan)
        self._ring = np.zeros(self.capacity, np.complex128 if plan.complex else np.float64)
        self.reset()

    def reset(self):
        """Forget all data; the next feed may start at any index."""
        self._decimator.reset()
        self._write = 0
        self._held = 0
        self._skip = self._decimator.warmup

    @property
    def next_index(self):
        return self._decimator.next_index

    @property
    def available(self):
        return self._held

    def feed(self, chunk, start_index):
        """Decimate the next contiguous chunk into the ring (ValueError on a gap)."""
        y = self._decimator.process(chunk, start_index)
        drop = min(self._skip, len(y))
        self._skip -= drop
        y = y[drop:][-self.capacity:]
        head = min(len(y), self.capacity - self._write)
        self._ring[self._write:self._write + head] = y[:head]
        self._ring[:len(y) - head] = y[head:]
        self._write = (self._write + len(y)) % self.capacity
        self._held = min(self._held + len(y), self.capacity)

    def latest(self, n):
        """The last n decimated samples, oldest first."""
        if n > self._held:
            raise ValueError(f'{n} samples requested, {self._held} available')
        end = self._write
        if n <= end:
            return self._ring[end - n:end].copy()
        return np.concatenate((self._ring[end - n:], self._ring[:end]))

    def psd(self, fft_size, averages, estimator='hann', nw=4.0):
        """Like zoom_psd, on the latest decimated data."""
        n = self.plan.nfft(fft_size)
        y = self.latest(min(self._held, averages * n))
        return _band_psd(self.plan, y, fft_size, averages, estimator, nw)


@lru_cache(maxsize=16)
def _tapers(estimator, n, nw):
    """Tapers as rows, each scaled to Σw² = 1 (read-only, cached)."""
    if estimator == 'hann':
        w = np.hanning(n)[None]
    elif estimator == 'multitaper':
        w = np.atleast_2d(sps.windows.dpss(n, nw, Kmax=int(2 * nw) - 1))
    else:
        raise ValueError(f'unknown estimator {estimator!r}')
    w = w / np.sqrt(np.sum(w ** 2, axis=1, keepdims=True))
    w.flags.writeable = False
    return w


def _power(spectrum):
    """|X|², without the square root that np.abs takes first."""
    return spectrum.real ** 2 + spectrum.imag ** 2


@lru_cache(maxsize=8)
def tone_bandwidth_bins(estimator, nw=4.0, n=4096):
    """Bins of segment_psd's density that make up a steady tone's power: the
    density at the tone × this × the bin width is the tone's power (1.5 for
    Hann, its equivalent noise bandwidth). Barely depends on ``n``."""
    tapers = _tapers(estimator, n, float(nw))
    return float(n / np.mean(np.sum(tapers, axis=1) ** 2))


def segment_psd(segments, fs, estimator='hann', nw=4.0, complex_input=False, center_hz=0.0,
                each=False):
    """Average PSD of the rows of ``segments``, one-sided density of the real signal
    (``each``: every row's own, shaped (rows, freqs), for their spread).

    Real rows: mean removed, rfft, DC and Nyquist halved (the SDK's compute_psd).
    Complex baseband rows: fftshifted, doubled so it equals the one-sided
    density of the real signal they were mixed from.
    """
    x = np.atleast_2d(segments)
    n = x.shape[1]
    tapers = _tapers(estimator, n, float(nw))
    scale = 2.0 / (fs * (1 if each else len(x)) * len(tapers))
    rows = (lambda p: p) if each else (lambda p: np.sum(p, axis=0))
    if complex_input:
        x = np.asarray(x, np.complex128)
        power = sum(rows(_power(sfft.fft(x * w, axis=1, workers=-1))) for w in tapers)
        density = np.fft.fftshift(power, axes=-1) * scale
        freqs = center_hz + np.fft.fftshift(np.fft.fftfreq(n, 1.0 / fs))
        return freqs, density
    x = np.asarray(x, np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    power = sum(rows(_power(sfft.rfft(x * w, axis=1, workers=-1))) for w in tapers)
    density = power * scale
    density[..., 0] *= 0.5
    if n % 2 == 0:
        density[..., -1] *= 0.5
    return np.fft.rfftfreq(n, 1.0 / fs), density


def _band_psd(plan, y, fft_size, averages, estimator, nw, each=False):
    """PSD of the last whole segments of decimated data, cropped to the view
    (``each``: every segment's, as segment_psd)."""
    n = plan.nfft(fft_size)
    count = min(averages, len(y) // n)
    if count < 1:
        raise ValueError(f'need {n} decimated samples for one segment, have {len(y)}')
    segments = y[len(y) - count * n:].reshape(count, n)
    freqs, density = segment_psd(segments, plan.out_rate, estimator, nw,
                                 plan.complex, plan.center_hz, each)
    margin = plan.out_rate / n * (1 + 1e-9)
    keep = (freqs >= plan.f_lo - margin) & (freqs <= plan.f_hi + margin)
    return freqs[keep], density[..., keep]


#: Multitaper time-bandwidth: 2·NW−1 = 5 tapers, peaks about 2·NW = 6 bins wide.
MULTITAPER_NW = 3.0
#: Samples read at a time, so a long recording is never held in memory whole.
WELCH_CHUNK = 1 << 22
#: A stretch up to this long is read at once; a longer one in blocks, or in pieces when capped.
WHOLE_READ_MAX = 20_000_000
ZOOM_PIECES, ZOOM_PIECE_FFTS = 32, 4   # a capped zoom of a long stretch: pieces spread over it
MIN_TRANSFORM = 64    # samples: less says nothing about a spectrum


def _full_rate_blocks(samples, fs, first, stop, n, limit, estimator, nw):
    """Blocks of (freqs, one row per FFT, (sum, count) of the samples): the FFTs of
    samples[first:stop], spread evenly over it, at most ``limit`` of them."""
    count = (stop - first) // n
    starts = np.linspace(first, stop - n, min(count, limit or count)).astype(int)
    whole = stop - first <= WHOLE_READ_MAX
    held = np.asarray(samples[first:stop]) if whole else samples
    offset = first if whole else 0
    tally = (float(held.sum()), held.size) if whole else None
    per_read = max(1, WELCH_CHUNK // n)
    for i in range(0, starts.size, per_read):
        segments = np.stack([np.asarray(held[a - offset:a - offset + n]) for a in starts[i:i + per_read]])
        segments = segments.astype(np.float64)
        freqs, rows = segment_psd(segments, fs, estimator, nw, each=True)
        yield freqs, rows, tally or (float(segments.sum()), segments.size)
        if whole:
            tally = (0.0, 0)


def _zoomed_blocks(samples, plan, first, stop, fft_size, limit, estimator, nw):
    """As _full_rate_blocks through a zoom ``plan``. Every FFT unless ``limit`` is
    set and the stretch is too long to read: then FFTs of pieces spread over it."""
    length = stop - first
    if length <= WHOLE_READ_MAX:
        counts = np.asarray(samples[first:stop])
        freqs, rows = zoom_psd(counts, plan, fft_size, length // fft_size, estimator, nw, each=True)
        yield freqs, rows, (float(counts.sum()), counts.size)
    elif limit:
        piece = ZOOM_PIECE_FFTS * fft_size + (Decimator(plan).warmup + 1) * plan.decimation
        for start in np.linspace(first, stop - piece, ZOOM_PIECES).astype(int):
            counts = np.asarray(samples[start:start + piece])
            freqs, rows = zoom_psd(counts, plan, fft_size, ZOOM_PIECE_FFTS, estimator, nw, each=True)
            yield freqs, rows, (float(counts.sum()), counts.size)
    else:
        yield from _streamed_zoom_blocks(samples, plan, first, stop, fft_size, estimator, nw)


def _streamed_zoom_blocks(samples, plan, first, stop, fft_size, estimator, nw):
    """Every FFT of a stretch too long to hold, decimated as it is read."""
    decimator, n = Decimator(plan), plan.nfft(fft_size)
    margin = plan.out_rate / n * (1 + 1e-9)
    skip, pending = decimator.warmup, None
    for pos in range(first, stop, WELCH_CHUNK):
        chunk = np.asarray(samples[pos:min(pos + WELCH_CHUNK, stop)], np.float64)
        y = decimator.process(chunk, pos)
        drop = min(skip, len(y))
        skip, y = skip - drop, y[drop:]
        pending = y if pending is None else np.concatenate((pending, y))
        k = len(pending) // n
        if k:
            freqs, rows = segment_psd(pending[:k * n].reshape(k, n), plan.out_rate, estimator, nw,
                                      plan.complex, plan.center_hz, each=True)
            pending = pending[k * n:]
            keep = (freqs >= plan.f_lo - margin) & (freqs <= plan.f_hi + margin)
            yield freqs[keep], rows[..., keep], (float(chunk.sum()), chunk.size)


def _one_transform(samples, fs, first, stop, estimator, nw):
    """One FFT of all of a stretch too short for two: its bins follow its length."""
    counts = np.asarray(samples[first:stop])
    data = counts[:fast_length(stop - first)].astype(np.float64)
    freqs, rows = segment_psd(data[None, :], fs, estimator, nw, each=True)
    return freqs, rows, (float(counts.sum()), counts.size)


def _span_blocks(samples, fs, first, stop, fft_size, plan, limit, estimator, nw, short_ok):
    """The FFTs of one contiguous span, as blocks. None fit: nothing, unless ``short_ok``."""
    if short_ok and stop - first < 2 * fft_size:
        if stop - first >= MIN_TRANSFORM:
            yield _one_transform(samples, fs, first, stop, estimator, nw)
        return
    try:
        if plan.decimation == 1:
            yield from _full_rate_blocks(samples, fs, first, stop, fft_size, limit, estimator, nw)
        else:
            yield from _zoomed_blocks(samples, plan, first, stop, fft_size, limit, estimator, nw)
    except ValueError:      # too short for the zoom's filters
        if short_ok:
            yield _one_transform(samples, fs, first, stop, estimator, nw)


def _setup(samples, fs, band, plan, spans, max_averages):
    fs = float(fs)
    if plan is None:
        plan = plan_zoom(fs, 0.0, fs / 2, enabled=False) if band is None else plan_zoom(fs, *band)
    spans = [(0, len(samples))] if spans is None else spans
    total = sum(stop - first for first, stop in spans)
    limits = [max(1, round(max_averages * (stop - first) / total)) if max_averages else None
              for first, stop in spans]
    return fs, plan, spans, limits


def welch(samples, fs, fft_size=16384, estimator='hann', nw=MULTITAPER_NW, band=None, spans=None, plan=None,
          max_averages=None):
    """The average PSD over the FFTs of ``samples``.

    ``spans``: the (first, stop) index pairs to use, each contiguous, so no FFT
    crosses a gap; all of ``samples`` by default. ``band``: (lo, hi) Hz, to
    decimate to that band first (a zoom FFT) with the same bin width, fs/fft_size,
    or ``plan`` for a ZoomPlan already chosen. ``max_averages``: at most this many
    FFTs, spread evenly; all of them by default.
    Returns (freqs, density in counts²/Hz, FFTs averaged, mean counts).
    ValueError if not even one FFT fits.
    """
    fs, plan, spans, limits = _setup(samples, fs, band, plan, spans, max_averages)
    total, freqs, count, summed, used = None, None, 0, 0.0, 0
    for (first, stop), limit in zip(spans, limits):
        for freqs, rows, (block_sum, block_size) in _span_blocks(samples, fs, first, stop, int(fft_size), plan,
                                                                 limit, estimator, nw, False):
            total = rows.sum(axis=0) if total is None else total + rows.sum(axis=0)
            count, summed, used = count + len(rows), summed + block_sum, used + block_size
    if count == 0:
        raise ValueError(f'no {int(fft_size)}-point FFT fits in the data')
    return freqs, total / count, count, summed / used


def every_fft(samples, fs, fft_size=16384, estimator='hann', nw=MULTITAPER_NW, band=None, spans=None, plan=None,
              max_averages=None, short_ok=False):
    """As :func:`welch`, keeping every FFT: a list with one (freqs, rows shaped (FFTs,
    freqs), mean counts) for each span an FFT fitted in. ``short_ok``: a span too
    short for two FFTs gets one transform of all of it, at the bins that gives.
    """
    fs, plan, spans, limits = _setup(samples, fs, band, plan, spans, max_averages)
    parts = []
    for (first, stop), limit in zip(spans, limits):
        blocks = list(_span_blocks(samples, fs, first, stop, int(fft_size), plan, limit, estimator, nw, short_ok))
        if blocks:
            parts.append((blocks[0][0], np.concatenate([rows for _, rows, _ in blocks]),
                          sum(b[2][0] for b in blocks) / sum(b[2][1] for b in blocks)))
    if not parts:
        raise ValueError(f'no {int(fft_size)}-point FFT fits in the data')
    return parts


def zoom_psd(samples, plan, fft_size, averages, estimator='hann', nw=4.0, each=False):
    """Batch zoom PSD of ``samples``: (freqs, density in units²/Hz) over the view
    (``each``: every segment's, shaped (segments, freqs))."""
    decimator = Decimator(plan)
    y = decimator.process(samples, 0)[decimator.warmup:]
    return _band_psd(plan, y, fft_size, averages, estimator, nw, each)


def fast_length(n):
    """The largest length <= ``n`` that the FFT handles quickly (smooth sizes)."""
    m = max(int(n), 1)
    while next_fast_len(m, real=True) != m:
        m -= 1
    return m


CHUNK = 1 << 22      # input samples per Decimator call: ~32 MB as float64
MAX_FFT = 1 << 20    # cap on the default fft_size


def _plan(n, fs, f_lo, f_hi):
    """plan_zoom's plan and a Decimator for it, decimating less when n samples
    would not leave one of the plan's shortest FFTs past the filters' start-up."""
    plan = plan_zoom(fs, f_lo, f_hi)
    decimator = Decimator(plan)
    while plan.decimation > 1 and -(-n // plan.decimation) - decimator.warmup < plan.nfft(1):
        plan = plan_zoom(fs, f_lo, f_hi, max_decimation=plan.decimation // 2)
        decimator = Decimator(plan)
    return plan, decimator


def spectrogram(samples, fs, f_lo, f_hi, rows=200, fft_size=None, estimator='hann', nw=3.0):
    """Zoom-FFT spectra of up to ``rows`` equal consecutive parts of ``samples``.

    The data are decimated once and the decimated stream, less the filters'
    start-up, is cut into rows. Each row is the Welch average of as many
    ``fft_size``-point (full-rate equivalent) FFTs as fit in it, centred in it;
    ``fft_size`` defaults to the largest power of two that fits a row, at most
    2**20. Returns (times_s, freqs_hz, density): row centres, the bins
    zoom_psd returns for the view (it and one bin either side) and density
    shaped (rows, freqs). ValueError if not even one FFT fits.
    """
    n = len(samples)
    plan, decimator = _plan(n, fs, f_lo, f_hi)
    d = plan.decimation
    valid = -(-n // d) - decimator.warmup   # decimated samples past the start-up
    if fft_size is None:
        fits = max(int(valid * d / rows), 1)
        fft_size = min(1 << (fits.bit_length() - 1), MAX_FFT)
    nfft = plan.nfft(fft_size)
    rows = min(rows, valid // nfft)
    if rows < 1:
        raise ValueError(f'{n} samples hold no {nfft * d}-sample FFT')
    share = valid // rows
    count = share // nfft
    first = np.arange(rows) * share + (share - count * nfft) // 2   # valid index of each row's first FFT
    centre = (first + decimator.warmup + (count * nfft - 1) / 2) * d - decimator.delay
    times = centre / fs

    # Each row's FFTs are summed as soon as they are whole, so neither the
    # input nor the decimated stream (full rate when the view is wide) is held.
    total = keep = freqs = None
    row = done = 0                                       # row being summed, its FFTs so far
    pending, offset = np.empty(0), -decimator.warmup     # unused output, valid index of pending[0]
    for start in range(0, n, CHUNK):
        pending = np.concatenate((pending, decimator.process(samples[start:start + CHUNK], start)))
        while row < rows:
            i = first[row] + done * nfft - offset
            take = min(count - done, (len(pending) - i) // nfft)
            if take < 1:
                break
            f, density = segment_psd(pending[i:i + take * nfft].reshape(take, nfft), plan.out_rate,
                                     estimator, nw, plan.complex, plan.center_hz)
            if total is None:
                margin = plan.out_rate / nfft * (1 + 1e-9)   # zoom_psd's crop
                keep = (f >= plan.f_lo - margin) & (f <= plan.f_hi + margin)
                freqs, total = f[keep], np.zeros((rows, np.count_nonzero(keep)))
            total[row] += density[keep] * take
            done += take
            if done == count:
                row, done = row + 1, 0
        if row == rows:
            break
        drop = min(first[row] + done * nfft - offset, len(pending))
        pending, offset = pending[drop:], offset + drop
    return times, freqs, total / count


def strip_spectrogram(samples, fs, f_lo, f_hi, columns=800, bins=96, budget=48_000_000,
                      cancelled=None):
    """All of ``samples`` as time against frequency, for the overview strip.

    Hann spectra at up to ``columns`` evenly spaced places, each over
    [f_lo, f_hi] with about ``bins`` bins, averaging as many FFTs as its share
    of ``budget`` input samples holds. Narrow bands need long FFTs, so they get
    fewer columns rather than reading more. Returns (times_s, freqs_hz,
    density shaped (columns, freqs)), or None once ``cancelled()``; ValueError
    if the samples hold no FFT.
    """
    n = len(samples)
    fft_size = 1 << max(6, int(np.ceil(np.log2(fs * bins / max(f_hi - f_lo, 1.0)))))
    plan, decimator = _plan(n, fs, f_lo, f_hi)
    d, nfft = plan.decimation, plan.nfft(fft_size)
    least = (decimator.warmup + nfft) * d          # one FFT past the filters' start-up
    if least > n:
        raise ValueError(f'{n} samples hold no {nfft * d}-sample FFT')
    per = min(n, max(least, budget // columns))
    columns = int(max(1, min(columns, budget // per)))
    averages = max(1, (per // d - decimator.warmup) // nfft)
    starts = np.linspace(0, n - per, columns).astype(np.int64)
    out = freqs = None
    for k, start in enumerate(starts):
        if cancelled is not None and cancelled():
            return None
        decimator.reset()
        y = decimator.process(samples[start:start + per], int(start))[decimator.warmup:]
        f, density = _band_psd(plan, y, fft_size, averages, 'hann', 3.0)
        if out is None:
            freqs, out = f, np.empty((columns, f.size))
        out[k] = density
    times = (starts + per / 2) / fs
    return times, freqs, out
