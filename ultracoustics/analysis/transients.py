"""Measurements on a stretch of signal: a lock-in tone reading, peak finding, a
wavelet scalogram and damped-mode (ringdown) fits.

Amplitudes are in the units of the samples passed in (ADC counts or µW); times
are seconds from the first sample unless noted.
"""
from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
from scipy.fft import next_fast_len

MAX_PENCIL_SAMPLES = 4000  # the damped-mode fit runs on at most this many samples
PENCIL_COLUMNS = 400       # pencil parameter cap: the SVD of 3600 x 401 takes ~0.3 s
NOISE_FACTOR = 3.0         # singular values this far above the noise bulk are signal
SETTLE_SAMPLES = 2         # band-limited samples skipped at the start (filter edge)


@dataclass
class Tone:
    freq_hz: float
    amplitude: float      # peak amplitude, input units
    phase_rad: float      # of cos(2π f n / fs + φ), n the absolute sample index
    power: float          # amplitude ** 2 / 2: the tone's mean square
    bandwidth_hz: float   # equivalent noise bandwidth of the measurement


def samples_for_bandwidth(fs, bandwidth_hz):
    """Hann window length whose equivalent noise bandwidth (1.5 fs / N) is ``bandwidth_hz``."""
    return max(2, int(round(1.5 * fs / bandwidth_hz)))


@lru_cache(maxsize=4)
def _tone_kernel(fs, freq_hz, n):
    # Periodic Hann, so the ENBW is exactly 1.5 fs / n
    k = np.arange(n)
    window = 0.5 - 0.5 * np.cos(2 * np.pi * k / n)
    phase = (2 * np.pi * freq_hz / fs) * k
    kernel = np.vstack((window * np.cos(phase), window * np.sin(phase)))
    kernel.flags.writeable = False
    return kernel, kernel.sum(axis=1), window.sum(), float(np.dot(window, window))


def measure_tone(samples, fs, freq_hz, start_index=0):
    """Lock-in reading of the tone at exactly ``freq_hz`` (not snapped to a bin).

    A Hann-windowed single-frequency DFT of the samples less their mean.
    ``start_index`` is the absolute index of ``samples[0]``; the phase is
    referred to index 0, so a steady tone reads the same phase from every window.
    """
    x = np.asarray(samples)
    if x.size < 2:
        raise ValueError('measure_tone needs at least 2 samples')
    fs, freq_hz = float(fs), float(freq_hz)
    kernel, kernel_sum, w_sum, w_sq = _tone_kernel(fs, freq_hz, x.size)
    # Σ w (x - mean) cos and Σ w (x - mean) sin, without a mean-removed copy
    c, s = kernel @ x - float(x.mean()) * kernel_sum
    turns = math.fmod(freq_hz * int(start_index) / fs, 1.0)
    phase = math.remainder(math.atan2(-s, c) - 2 * math.pi * turns, 2 * math.pi)
    amplitude = 2 * math.hypot(c, s) / w_sum
    return Tone(freq_hz, amplitude, phase, amplitude ** 2 / 2, fs * w_sq / w_sum ** 2)


def tone_track(samples, fs, freq_hz, bandwidth_hz, points=400, start_index=0):
    """The tone tracker's reading across ``samples``: measure_tone on windows of
    samples_for_bandwidth(fs, bandwidth_hz) samples at up to ``points`` evenly
    spaced positions from the first sample to the last (one window of all of
    it when shorter). Returns (times_s, amplitude, phase_rad): window centres,
    peak amplitude in counts and phase referred to absolute index 0, where
    ``start_index`` is that of samples[0].
    """
    n = len(samples)
    width = min(samples_for_bandwidth(fs, bandwidth_hz), n)
    starts = np.linspace(0, n - width, min(points, n - width + 1)).round().astype(np.int64)
    tones = [measure_tone(samples[s:s + width], fs, freq_hz, start_index + int(s)) for s in starts]
    times = (starts + width / 2) / fs   # a periodic Hann of N points is centred on N/2
    return (times, np.array([t.amplitude for t in tones], dtype=float),
            np.array([t.phase_rad for t in tones], dtype=float))


def strongest_peak(freqs, density, f_lo, f_hi):
    """Frequency of the largest ``density`` bin in [f_lo, f_hi], refined by a
    parabola through its dB value and its neighbours'; None if no bins.

    ``density`` is linear (e.g. W²/Hz), not dB.
    """
    freqs = np.asarray(freqs, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        db = 10 * np.log10(np.asarray(density, dtype=float))
    inside = np.flatnonzero((freqs >= f_lo) & (freqs <= f_hi) & ~np.isnan(db))
    if inside.size == 0:
        return None
    i = int(inside[np.argmax(db[inside])])
    if 0 < i < freqs.size - 1:
        a, b, c = db[i - 1:i + 2]
        curvature = a - 2 * b + c
        if np.isfinite(curvature) and curvature < 0:
            offset = min(max(0.5 * (a - c) / curvature, -0.5), 0.5)
            return float(freqs[i] + offset * (freqs[i + 1] - freqs[i - 1]) / 2)
    return float(freqs[i])


def cwt_scalogram(x, fs, f_min, f_max, n_freqs=96, w0=6.0, max_columns=1024):
    """Morlet wavelet magnitude |W(t, f)| on ``n_freqs`` log-spaced frequencies.

    Returns (times_s, freqs_hz, magnitude), magnitude shaped (n_freqs, n_times).
    Each column is the mean |W| over an equal share of ``x`` (at most
    ``max_columns`` of them) and times_s are their centres, 0 being x[0]. A
    unit-amplitude sinusoid reads 1.0 on its ridge, away from the ends.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    f_max = min(float(f_max), fs / 2)
    if n < 2 or not 0 < f_min < f_max:
        raise ValueError(f'cannot make a scalogram of {n} samples from {f_min} to {f_max} Hz')
    freqs = np.geomspace(f_min, f_max, n_freqs)
    columns = min(int(max_columns), n)
    # Zero padding past the slowest wavelet's reach keeps the ends from wrapping
    pad = min(n, int(4 * w0 / (2 * np.pi * f_min) * fs))
    m = next_fast_len(n + pad, real=True)
    spectrum = np.fft.rfft(x - x.mean(), m)
    df = fs / m
    reach = 5 / w0  # the wavelet's Gaussian (sigma f / w0) is e^-12.5 at 5 sigma
    per_column = int(np.ceil(4 * columns * m / n))
    magnitude = np.zeros((n_freqs, columns))
    for row, f in enumerate(freqs):
        a = max(1, int(f * (1 - reach) / df))
        b = min(spectrum.size, int(np.ceil(f * (1 + reach) / df)) + 1)
        if b <= a:
            continue
        # Each row needs only its own band: an inverse FFT just long enough for
        # it and for a few samples per column is an exact, alias-free rate
        # reduction. Shifting the band to 0 Hz leaves |W| unchanged.
        size = next_fast_len(max(b - a, per_column))
        band = np.zeros(size, complex)
        band[:b - a] = spectrum[a:b] * np.exp(-0.5 * (w0 * (np.arange(a, b) * df / f - 1)) ** 2)
        w = np.abs(np.fft.ifft(band)) * (2 * size / m)  # 2: the analytic signal
        t = np.arange(size) * (m / size)  # in input samples
        keep = t < n
        col = (t[keep] * (columns / n)).astype(int)
        magnitude[row] = (np.bincount(col, w[keep], columns)
                          / np.maximum(np.bincount(col, minlength=columns), 1))
    times = (np.arange(columns) + 0.5) * (n / columns) / fs
    return times, freqs, magnitude


@dataclass
class Mode:
    freq_hz: float
    tau_s: float            # amplitude decay time constant (inf if not decaying)
    q: float                # π f τ
    amplitude: float        # peak amplitude at x[0], input units
    phase_rad: float        # of A e^(-t/τ) cos(2π f t + φ), t from x[0]
    energy_fraction: float  # share of the fitted signal energy over the fit span


def modes_signal(modes, t):
    """The sum of ``modes`` at times ``t`` (s from x[0])."""
    t = np.asarray(t, dtype=float)
    out = np.zeros(t.shape)
    for mode in modes:
        decay = 0.0 if math.isinf(mode.tau_s) else 1 / mode.tau_s
        out += (mode.amplitude * np.exp(-decay * t)
                * np.cos(2 * np.pi * mode.freq_hz * t + mode.phase_rad))
    return out


def damped_modes(x, fs, f_lo=None, f_hi=None, max_modes=6):
    """Ringing modes of ``x``: a sum of damped sinusoids fitted by the matrix pencil.

    With a band, ``x`` is first band-limited to it and its rate reduced, so the
    pencil sees few samples and only that band. Returns the positive-frequency
    modes inside the band, most energetic first; [] if nothing stands above
    the noise.
    """
    x = np.asarray(x, dtype=float)
    if x.size < 32 or max_modes < 1 or not np.isfinite(x).all():
        return []
    x = x - x.mean()
    nyquist = fs / 2
    lo = min(max(0.0 if f_lo is None else float(f_lo), 0.0), nyquist)
    hi = min(max(nyquist if f_hi is None else float(f_hi), 0.0), nyquist)
    if hi <= lo:
        return []
    if f_lo is None and f_hi is None:
        y, rate, shift, start = x, float(fs), None, 0
    else:
        y, rate, shift = _band_limited(x, fs, lo, hi)
        start = SETTLE_SAMPLES
    y = y[start:]
    y = y[:_active_length(y, MAX_PENCIL_SAMPLES)]
    if y.size < 32:
        return []
    poles, amps, energy = _pencil(y, 2 * max_modes)
    total = energy.sum()
    if poles.size == 0 or not total > 0:
        return []
    amps = amps * poles ** -start  # refer to x[0]
    # A "mode" broader than the band is the fit soaking up an edge or noise
    min_tau = 1 / (math.pi * (hi - lo))
    modes = []
    for z, c, e in zip(poles, amps, energy):
        turns = np.angle(z) / (2 * np.pi)
        if shift is None:  # real samples: keep one of each conjugate pair
            if turns <= 0:
                continue
            f, amplitude, e = turns * rate, 2 * abs(c), 2 * e
        else:
            f, amplitude = shift + turns * rate, abs(c)
        decay = -math.log(abs(z)) * rate
        tau = 1 / decay if decay > 0 else math.inf
        if not (lo <= f <= hi and f > 0 and tau >= min_tau):
            continue
        modes.append(Mode(float(f), tau, math.pi * f * tau, float(amplitude),
                          float(np.angle(c)), float(e / total)))
    modes.sort(key=lambda mode: mode.energy_fraction, reverse=True)
    return modes[:max_modes]


def event_window(event, uw_per_count=None, max_samples=None):
    """A trigger event's samples from the trigger on (at most ``max_samples``),
    in µW less their mean, and its sample rate."""
    start = max(event.pre_samples, 0)
    stop = None if max_samples is None else start + max_samples
    counts = event.samples[start:stop].astype(float)  # a copy: the event is left alone
    if counts.size:
        counts -= counts.mean()
    scale = event.uw_per_count if uw_per_count is None else uw_per_count
    return counts * scale, float(event.sample_rate)


def _band_limited(x, fs, lo, hi):
    """``x`` limited to [lo, hi] (with tapered margins) at a reduced rate.

    Returns (y, rate, shift): complex baseband shifted down by ``shift`` Hz, or
    for a band near DC real samples and shift None. Done in one FFT, which is
    a complex mix, ideal low-pass and decimation in one step.
    """
    n = x.size
    margin = max(0.25 * (hi - lo), 3 * fs / n)
    # Zero padding past the taper's ringing keeps the end from wrapping onto the start
    m = next_fast_len(n + int(4 * fs / margin), real=True)
    spectrum = np.fft.rfft(x, m)
    df = fs / m
    near_dc = lo < 2 * margin
    a = 0 if near_dc else max(1, int((lo - margin) / df))
    b = min(spectrum.size, int(np.ceil((hi + margin) / df)) + 1)
    f = np.arange(a, b) * df
    edge = (hi + margin - f) if near_dc else np.minimum(f - (lo - margin), hi + margin - f)
    taper = 0.5 - 0.5 * np.cos(np.pi * np.clip(edge / margin, 0, 1))
    if near_dc:
        size = next_fast_len(2 * b, real=True)
        y = np.fft.irfft(spectrum[a:b] * taper, size) * (size / m)
        shift = None
    else:
        size = next_fast_len(b - a + 2)
        centre = (a + b) // 2
        band = np.zeros(size, complex)
        band[(np.arange(a, b) - centre) % size] = 2 * spectrum[a:b] * taper
        y = np.fft.ifft(band) * (size / m)
        shift = centre * df
    rate = fs * size / m
    return y[:int(np.ceil(n * size / m))], rate, shift


def _active_length(y, limit):
    """How much of ``y`` to fit: up to where its power sinks into the noise
    (a little past it), at most ``limit`` samples."""
    blocks = 64
    if y.size < 4 * blocks:
        return min(y.size, limit)
    width = y.size // blocks
    power = (np.abs(y[:width * blocks]) ** 2).reshape(blocks, width).mean(axis=1)
    # The quietest blocks are the noise, unless the signal never decays
    above = np.flatnonzero(power > 10 * np.percentile(power, 10))
    if above.size == 0:
        return min(y.size, limit)
    return min(max(int(1.5 * (above[-1] + 1) * width), 4 * width), y.size, limit)


def _pencil(y, max_order):
    """Poles, amplitudes (at y[0]) and energies over y of the modes in ``y``."""
    n = y.size
    columns = min(n // 3, PENCIL_COLUMNS)
    hankel = np.lib.stride_tricks.sliding_window_view(y, columns + 1)
    _, s, vh = np.linalg.svd(hankel, full_matrices=False)
    # The bulk of the singular values is noise; a fraction 1e-6 of the largest
    # is rounding on clean data
    noise = max(s[s.size // 4], s[0] * 1e-6)
    order = min(int(np.sum(s > NOISE_FACTOR * noise)), max_order)
    empty = np.zeros(0, complex)
    if order == 0:
        return empty, empty, np.zeros(0)
    v = vh[:order].T  # the signal subspace of the rows: spans [1, z, z², ...] per pole
    poles = np.linalg.eigvals(np.linalg.pinv(v[:-1]) @ v[1:])
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        growth = np.log(np.abs(poles)) * n
    poles = poles[np.isfinite(growth) & (growth < 50)]  # drop runaway noise poles
    if poles.size == 0:
        return empty, empty, np.zeros(0)
    basis = poles[None, :] ** np.arange(n)[:, None]
    norms = np.linalg.norm(basis, axis=0)
    amps = np.linalg.lstsq(basis / norms, y.astype(complex), rcond=None)[0]
    return poles, amps / norms, np.abs(amps) ** 2
