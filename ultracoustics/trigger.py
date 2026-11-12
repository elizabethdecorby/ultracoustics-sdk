"""Triggers and their detectors: the same code runs live and over a saved recording.

Every sample is checked, and nothing here runs on the box: the PC must keep up,
and samples it misses are counted. :mod:`.monitor` runs these on the live stream.
"""
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import dataclass, field, replace
from datetime import timedelta
import math
import os
import threading
import warnings

import numpy as np
from scipy import fft as sfft
from scipy import signal as sps

from .analysis.filters import band_filter
from .analysis.pressure import MAX_RIPPLE_DB, sensitivity_ripple_db
from .recording import SigMFRecording, TriggerEvent, utc_iso

#: Threads for the band FFTs. Most of a block's time is in the arrays around the FFT, so
#: blocks of frames run in parallel, not the transform alone.
FFT_THREADS = max(1, min(4, (os.cpu_count() or 2) // 2))
MIN_PARALLEL_SAMPLES = 1 << 18    # a block smaller than this isn't worth handing out
_pool = None


def _thread_pool():
    global _pool
    if _pool is None:
        _pool = ThreadPoolExecutor(FFT_THREADS, thread_name_prefix='band-fft')
    return _pool


def _frame_power(x, nfft, hop, window, first_bin, stop_bin):
    """|X|² of bins [first_bin, stop_bin) of every full frame of ``x``, shaped (frames, bins)."""
    n_frames = (x.size - nfft) // hop + 1
    if n_frames <= 0:
        return None
    view = np.lib.stride_tricks.sliding_window_view(x, nfft)[::hop][:n_frames]

    def power(rows):
        frames = (rows - rows.mean(axis=1, keepdims=True)) * window
        spectrum = sfft.rfft(frames, axis=1)[:, first_bin:stop_bin]
        return spectrum.real ** 2 + spectrum.imag ** 2

    if n_frames * nfft < MIN_PARALLEL_SAMPLES or n_frames < 2 or FFT_THREADS == 1:
        return power(view)
    step = math.ceil(n_frames / FFT_THREADS)
    return np.concatenate(list(_thread_pool().map(power, (view[i:i + step] for i in range(0, n_frames, step)))))


class _Schmitt:
    """Fires when a metric reaches ``fire`` after having been below ``rearm``.

    After each trigger nothing fires (or re-arms) until the holdoff has passed,
    so one burst gives one event and the next needs a fresh edge. With ``armed``
    it may fire from the start: a limit that doesn't depend on the signal knows a
    level above it is a fault, even with no quiet moment first.
    """

    def __init__(self, holdoff_samples, armed=False):
        self.holdoff = int(holdoff_samples)
        self.start_armed = armed
        self.armed = armed
        self.blocked_until = -1

    def reset(self):
        self.armed = self.start_armed

    def run(self, origin, step, y, fire, rearm):
        """``y[i]`` is the metric at absolute sample ``origin + i * step``."""
        fire_at = np.flatnonzero(y >= fire)
        rearm_at = np.flatnonzero(y < rearm)
        out, i = [], 0
        while True:
            i = max(i, math.ceil((self.blocked_until - origin) / step))
            pool = fire_at if self.armed else rearm_at
            k = np.searchsorted(pool, i)
            if k == pool.size:
                return out
            j = int(pool[k])
            if not self.armed:
                self.armed, i = True, j + 1
                continue
            index = origin + j * step
            out.append(index)
            self.armed, self.blocked_until, i = False, index + self.holdoff, j + 1


@dataclass
class LevelTrigger:
    """Optical power crosses a level, optionally on a band-passed copy of the signal.

    Set ``level_pa`` to give the level in pressure instead: that needs a calibration,
    and ``band_hz``, as the sensitivity depends on frequency.
    """
    level_uw: float = 1.0
    edge: str = 'rising'          # 'rising', 'falling' or 'either' (|signal| reaches |level|)
    hysteresis_uw: float = 0.1
    band_hz: tuple = None         # trigger filter: (low, high) Hz, trigger path only
    level_pa: float = None
    name: str = ''                # how events, alarms and the trend log call it

    @property
    def label(self):
        return self.name or self.describe()

    def detector(self, sample_rate, holdoff_samples, calibration=None):
        return LevelDetector(self._in_uw(calibration), sample_rate, holdoff_samples)

    def _in_uw(self, calibration):
        """This trigger with a level in Pa turned into µW, its hysteresis a tenth of it."""
        if self.level_pa is None:
            return self
        if calibration is None:
            raise ValueError('a level in Pa needs a pressure calibration')
        if not self.band_hz:
            raise ValueError('a level in Pa needs band_hz: the sensitivity depends on frequency')
        _warn_on_ripple(calibration, self.band_hz)
        level = self.level_pa / calibration.pa_per_uw(self.band_hz)[0]
        return replace(self, level_uw=level, hysteresis_uw=level / 10)

    def describe(self):
        edge = {'rising': 'rising', 'falling': 'falling', 'either': '±'}[self.edge]
        text = f'level {self.level_uw:g} µW {edge}' if self.level_pa is None else f'level {self.level_pa:g} Pa {edge}'
        if self.band_hz:
            text += f', filter {self.band_hz[0] / 1e3:g}–{self.band_hz[1] / 1e3:g} kHz'
        return text

    def to_meta(self):
        return {'type': 'level', 'level_uw': self.level_uw, 'edge': self.edge,
                'hysteresis_uw': self.hysteresis_uw, 'level_pa': self.level_pa,
                'filter_hz': list(self.band_hz) if self.band_hz else None}


@dataclass
class BandPowerTrigger:
    """Power in a frequency band rises a set number of dB above its normal level.

    What counts as normal is one of three things:

    - By default the level is learned for ``learn_seconds`` on arming, then follows
      slow changes (time constant ``track_seconds``; 0 holds it), so a drifting floor
      does not fire the trigger but an event does. A fault that grows slowly is
      followed too, and one already there when it arms is learned as normal.
    - ``baseline``, a Spectrum of the machine when healthy, is held for good. Use this
      to catch slow degradation.
    - ``level_spl`` is an absolute limit in dB SPL for the band, which needs a
      calibration. ``threshold_db`` is then not used. Each frame is compared with it,
      so it reads short-time peaks: steady noise reaches about 6 dB above its band
      level, so keep the limit at least that far above the quiet level. A loud tone's
      frames read within 0.2 dB of its level.

    The last two know what normal is, so a fault already there when it arms fires.
    """
    band_hz: tuple = (1e6, 3e6)
    threshold_db: float = 10.0
    hysteresis_db: float = 3.0
    learn_seconds: float = 1.0
    track_seconds: float = 10.0
    baseline: object = field(default=None, compare=False, repr=False)
    level_spl: float = None
    name: str = ''                # how events, alarms and the trend log call it

    @property
    def label(self):
        return self.name or self.describe()

    def detector(self, sample_rate, holdoff_samples, calibration=None):
        return BandPowerDetector(self, sample_rate, holdoff_samples, calibration)

    def describe(self):
        limit = f'{self.level_spl:g} dB SPL' if self.level_spl is not None else f'+{self.threshold_db:g} dB'
        return f'band power {self.band_hz[0] / 1e3:g}–{self.band_hz[1] / 1e3:g} kHz {limit}'

    def to_meta(self):
        return {'type': 'band_power', 'band_hz': list(self.band_hz),
                'threshold_db': self.threshold_db, 'hysteresis_db': self.hysteresis_db,
                'learn_seconds': self.learn_seconds, 'track_seconds': self.track_seconds,
                'healthy_baseline': self.baseline is not None, 'level_spl': self.level_spl}


def _warn_on_ripple(calibration, band_hz):
    ripple = sensitivity_ripple_db(calibration, band_hz)
    if ripple > MAX_RIPPLE_DB:
        warnings.warn(f'the sensitivity swings {ripple:.1f} dB across {band_hz[0]:g}-{band_hz[1]:g} Hz, so a '
                      'pressure limit is only right near the band\'s centre', stacklevel=4)


class LevelDetector:
    def __init__(self, cfg, sample_rate, holdoff_samples):
        if cfg.edge == 'either' and abs(cfg.hysteresis_uw) >= abs(cfg.level_uw):
            # |signal| would have to fall below |level| - hysteresis <= 0 to re-arm: never
            raise ValueError('the hysteresis must be smaller than the level, or the trigger '
                             'can never fire')
        self.cfg = cfg
        self.schmitt = _Schmitt(holdoff_samples)
        self.sos = self.zi = None
        if cfg.band_hz:
            self.sos = band_filter(cfg.band_hz, sample_rate)
        self.level = None  # latest block's peak trigger metric, for a live readout
        # The same in the direction it fires, with its limits: what an alarm follows
        self.peak = self.fire = self.clear = None

    def reset(self):
        self.schmitt.reset()
        self.zi = None

    def process(self, start, x):
        if self.sos is not None:
            if self.zi is None:
                self.zi = sps.sosfilt_zi(self.sos) * x[0]
            x, self.zi = sps.sosfilt(self.sos, x, zi=self.zi)
        level, hyst = self.cfg.level_uw, abs(self.cfg.hysteresis_uw)
        if self.cfg.edge == 'rising':
            y, fire, rearm = x, level, level - hyst
        elif self.cfg.edge == 'falling':
            y, fire, rearm = -x, -level, -level - hyst
        else:
            y, fire, rearm = np.abs(x), abs(level), abs(level) - hyst
        self.peak, self.fire, self.clear = float(np.max(y)), fire, rearm
        self.level = self.peak * (-1 if self.cfg.edge == 'falling' else 1)
        return self.schmitt.run(start, 1, y, fire, rearm)


# Shortest band-power frame (204.8 µs at 10 MS/s): shorter frames make the
# band's noise jump around enough to fire a +10 dB threshold on its own.
MIN_BAND_FFT = 2048


class FrameBank:
    """The short-time FFT of a stream, shared by every band that uses this FFT size."""

    def __init__(self, nfft):
        self.nfft, self.hop = nfft, nfft // 4
        self.window = np.hanning(nfft).astype(np.float32)
        self.members = []
        self.tail = np.zeros(0, np.float32)
        self.tail_start = None

    def add(self, detector):
        self.members.append(detector)
        detector.bank = self

    def reset(self):
        self.tail, self.tail_start = np.zeros(0, np.float32), None

    def process(self, start, x):
        """Feed one block: {detector: sample indices where it fired}."""
        if self.tail.size and self.tail_start + self.tail.size == start:
            x, start = np.concatenate((self.tail, x)), self.tail_start
        first = min(d.first_bin for d in self.members)
        power = _frame_power(x, self.nfft, self.hop, self.window, first, max(d.stop_bin for d in self.members))
        if power is None:
            self.tail, self.tail_start = np.asarray(x, np.float32), start
            return {d: [] for d in self.members}
        consumed = len(power) * self.hop
        self.tail = np.asarray(x[consumed:], np.float32)
        self.tail_start = start + consumed
        return {d: d.on_frames(start, consumed, 10 * np.log10(
                    power[:, d.first_bin - first:d.stop_bin - first].sum(axis=1) + 1e-30))
                for d in self.members}


class BandPowerDetector:
    """Short-time FFT power in a band, in dB above its normal level.

    The FFT size follows the band: about four bins across it, from 2048 points
    (0.2 ms frames) to 65536 (narrow, low-frequency bands). Frames overlap by
    75 %, so an event is caught whatever its alignment.
    """

    def __init__(self, cfg, sample_rate, holdoff_samples, calibration=None):
        if cfg.level_spl is None and cfg.hysteresis_db >= cfg.threshold_db:
            # The band would have to fall below its own recent level to re-arm: it never does
            raise ValueError('the hysteresis must be smaller than the threshold, or the trigger '
                             'can never fire')
        self.cfg = cfg
        self.sample_rate = sample_rate
        lo, hi = sorted(float(f) for f in cfg.band_hz)
        width = max(hi - lo, 1.0)
        self.nfft = int(np.clip(2 ** math.ceil(math.log2(4 * sample_rate / width)),
                                MIN_BAND_FFT, 65536))
        self.hop = self.nfft // 4
        self.window = np.hanning(self.nfft).astype(np.float32)
        freqs = np.fft.rfftfreq(self.nfft, 1 / sample_rate)
        self.bins = np.flatnonzero((freqs >= lo) & (freqs <= hi))
        if self.bins.size < 4:  # one or two bins' noise alone crosses +10 dB
            centre = int(np.argmin(np.abs(freqs - (lo + hi) / 2)))
            first = int(np.clip(centre - 1, 1, freqs.size - 4))
            self.bins = np.arange(first, first + 4)
        self.first_bin, self.stop_bin = int(self.bins[0]), int(self.bins[-1]) + 1
        self.learn_frames = max(1, int(cfg.learn_seconds * sample_rate / self.hop))
        FrameBank(self.nfft).add(self)
        self.baseline_db = None
        if cfg.baseline is not None:
            self.baseline_db = self._expected_db(cfg.baseline)
            self.cfg = cfg = replace(cfg, track_seconds=0.0)
        elif cfg.level_spl is not None:
            self.baseline_db = self._db_at_0_spl(cfg, calibration)
            self.cfg = cfg = replace(cfg, threshold_db=cfg.level_spl, track_seconds=0.0)
        self.schmitt = _Schmitt(holdoff_samples, armed=self.baseline_db is not None)
        self._learning = []
        self._loud_samples = 0  # since the band was last quiet
        self.level = None  # latest block's highest dB above baseline, for the meter
        self.peak = self.fire = self.clear = None   # as LevelDetector's

    @property
    def tail(self):
        return self.bank.tail

    @property
    def frame_seconds(self):
        return self.nfft / self.sample_rate

    def _frame_scale(self):
        """|X|² summed over the band, per µW²/Hz of density in each of its bins."""
        return self.sample_rate * float(np.sum(self.window.astype(np.float64) ** 2)) / 2

    def _expected_db(self, spectrum):
        """The frame level a band at ``spectrum``'s density reads, as a median: what a
        learned baseline would settle on. A Hann bin shares half its power with its
        neighbours, so the band's noise has about 2/1.5 degrees of freedom per bin."""
        freqs = np.fft.rfftfreq(self.nfft, 1 / self.sample_rate)[self.bins]
        density = np.interp(freqs, spectrum.freqs, spectrum.psd_w2hz) * 1e12    # µW²/Hz
        dof = 2 * self.bins.size / 1.5
        median_over_mean = (1 - 2 / (9 * dof)) ** 3
        return float(10 * np.log10(density.sum() * self._frame_scale() * median_over_mean + 1e-30))

    def _db_at_0_spl(self, cfg, calibration):
        """The frame level of sound at 0 dB SPL in the band, so dB above it is dB SPL."""
        if calibration is None:
            raise ValueError('a limit in dB SPL needs a pressure calibration')
        _warn_on_ripple(calibration, cfg.band_hz)
        uw_rms = 20e-6 / calibration.pa_per_uw(cfg.band_hz)[0]
        return float(10 * np.log10(uw_rms ** 2 * self.nfft / self.sample_rate * self._frame_scale() + 1e-30))

    def relearn(self):
        self.baseline_db, self._learning = None, []
        self.schmitt.reset()

    def reset(self):
        self.bank.reset()
        self.schmitt.reset()

    def _frame_db(self, x):
        """Band power (dB) of each full frame of ``x``."""
        power = _frame_power(x, self.nfft, self.hop, self.window, self.first_bin, self.stop_bin)
        return np.zeros(0) if power is None else 10 * np.log10(power.sum(axis=1) + 1e-30)

    def peak_db(self, x):
        """Highest band power in ``x``, in dB above the baseline (None if too short)."""
        db = self._frame_db(np.asarray(x, np.float32))
        if db.size == 0 or self.baseline_db is None:
            return None
        return float(db.max() - self.baseline_db)

    def process(self, start, x):
        return self.bank.process(start, x)[self]

    def on_frames(self, start, consumed, db):
        """The band's level in the frames from sample ``start``, ``consumed`` samples' worth."""
        if self.baseline_db is None:
            self._learning.extend(db.tolist())
            if len(self._learning) >= self.learn_frames:
                self.baseline_db = float(np.median(self._learning))
                self._learning = []
            return []
        rel = db - self.baseline_db
        self.level = float(rel.max())
        cfg = self.cfg
        rearm = cfg.threshold_db - abs(cfg.hysteresis_db)
        self.peak, self.fire, self.clear = self.level, cfg.threshold_db, rearm
        found = self.schmitt.run(start + self.nfft // 2, self.hop, rel, cfg.threshold_db, rearm)
        if cfg.track_seconds:
            # Follow the band's slow drift on quiet blocks. A rise that lasts
            # (half the time constant) becomes the new normal too, or the
            # trigger could never re-arm.
            quiet = self.level < rearm
            self._loud_samples = 0 if quiet else self._loud_samples + consumed
            if quiet or self._loud_samples > cfg.track_seconds * self.sample_rate / 2:
                alpha = min(1.0, consumed / (cfg.track_seconds * self.sample_rate))
                self.baseline_db += alpha * (float(np.median(db)) - self.baseline_db)
        return found


class Watch:
    """Several triggers over a stream of blocks, live or saved. Bands of one FFT size share it."""

    def __init__(self, triggers, sample_rate, holdoff_samples, calibration=None):
        self.triggers = list(triggers) if isinstance(triggers, (list, tuple)) else [triggers]
        if not self.triggers:
            raise ValueError('nothing to watch: give at least one trigger')
        self.detectors = [t.detector(sample_rate, holdoff_samples, calibration) for t in self.triggers]
        self.banks = {}
        for detector in self.detectors:
            if isinstance(detector, BandPowerDetector):
                if detector.nfft in self.banks:
                    self.banks[detector.nfft].add(detector)
                else:
                    self.banks[detector.nfft] = detector.bank

    def reset(self):
        for detector in self.detectors:
            detector.reset()

    def process(self, start, x):
        """Feed one block of µW: [(detector position, sample index)], in time order."""
        found = {}
        for bank in self.banks.values():
            found.update(bank.process(start, x))
        hits = []
        for position, detector in enumerate(self.detectors):
            indices = found[detector] if detector in found else detector.process(start, x)
            hits += [(position, int(index)) for index in indices]
        return sorted(hits, key=lambda hit: hit[1])


# Features look at most this far past the trigger (0.1 s), so long captures
# don't stall the engine
FEATURE_SAMPLES = 1 << 20


def event_features(event, band_hz=None, edge=None):
    """Peak (µW from the trigger baseline, in the edge's direction), the dominant
    frequency after the trigger, inside ``band_hz`` when given, and the shape of what
    follows it: rms, crest factor and kurtosis (3 for Gaussian noise; impacts read high)."""
    x = event.uw()
    if edge == 'rising':
        peak = int(np.argmax(x))
    elif edge == 'falling':
        peak = int(np.argmin(x))
    else:
        peak = int(np.argmax(np.abs(x)))
    post = x[event.pre_samples:event.pre_samples + FEATURE_SAMPLES]
    dominant = None
    if post.size >= 64:
        # A light taper: an event starts right at the trigger, where a Hann
        # window would weight it almost to nothing
        window = sps.windows.tukey(post.size, 0.1)
        spectrum = np.abs(np.fft.rfft((post - post.mean()) * window))
        freqs = np.fft.rfftfreq(post.size, 1 / event.sample_rate)
        spectrum[0] = 0
        if band_hz:
            inside = (freqs >= min(band_hz)) & (freqs <= max(band_hz))
            if inside.any():
                spectrum = np.where(inside, spectrum, 0)
        dominant = float(freqs[np.argmax(spectrum)])
    return {'peak_uw': float(x[peak]),
            'peak_ms': (peak - event.pre_samples) / event.sample_rate * 1e3,
            'dominant_hz': dominant, **shape_features(post)}


def shape_features(x):
    """rms, crest factor and kurtosis of ``x`` about its mean; None for too little data."""
    if x.size < 64:
        return {'rms_uw': None, 'crest': None, 'kurtosis': None}
    y = np.asarray(x, np.float64) - float(np.mean(x))
    variance = float(np.mean(y ** 2))
    if variance <= 0:
        return {'rms_uw': 0.0, 'crest': None, 'kurtosis': None}
    return {'rms_uw': math.sqrt(variance), 'crest': float(np.max(np.abs(y))) / math.sqrt(variance),
            'kurtosis': float(np.mean(y ** 4)) / variance ** 2}


def make_event(number, trigger, detector, index, first, samples, sample_rate, utc, baseline, uw_per_count):
    """A TriggerEvent with its features, the same from the live stream and from a file."""
    event = TriggerEvent(number, index, first, samples, sample_rate, utc, baseline, uw_per_count,
                         band_hz=trigger.band_hz)
    event.features = event_features(event, trigger.band_hz, getattr(trigger, 'edge', None))
    event.features['trigger'] = trigger.label
    if isinstance(detector, BandPowerDetector):
        # From a frame before the trigger, so it includes the frame that fired
        pre = event.pre_samples
        event.features['band_db'] = detector.peak_db(event.uw()[max(0, pre - detector.nfft):pre + FEATURE_SAMPLES])
    return event


class EventRecorder:
    """Appends each trigger event to one recording plus an ``events.csv`` table.

    Events are numbered 1, 2, ... in the order they are written to the file.
    """

    FIELDS = ('event', 'utc', 'trigger_index', 'file_sample', 'trigger', 'peak_uw', 'peak_ms',
              'dominant_hz', 'band_db', 'rms_uw', 'crest', 'kurtosis')

    def __init__(self, base, sample_rate, global_meta, max_events=0):
        self.recording = SigMFRecording(base, sample_rate, global_meta)
        self.csv_path = self.recording.base.with_name(self.recording.base.name + '_events.csv')
        self.max_events = int(max_events)
        self.count = 0
        self.error = None
        self._lock = threading.Lock()
        self._csv = open(self.csv_path, 'w', newline='', encoding='utf-8-sig')   # Excel: µ, ²
        self._writer = csv.DictWriter(self._csv, fieldnames=self.FIELDS)
        self._writer.writeheader()

    @property
    def full(self):
        return bool(self.max_events) and self.count >= self.max_events

    @property
    def bytes(self):
        return self.recording.bytes

    def add(self, event):
        """Append one event; False if full, closed or the disk failed (see ``error``)."""
        with self._lock:
            if self._csv.closed or self.full or self.error is not None:
                return False
            try:
                self._append(event)
            except OSError as exc:
                self.error = exc
                return False
            self.count += 1
            return True

    def _append(self, event):
        start = self.recording.samples
        segment_utc = event.utc - timedelta(seconds=event.pre_samples / event.sample_rate)
        self.recording.write(event.samples, event.start_index, segment_utc)
        f = event.features
        self.recording.annotate(
            start + event.pre_samples, event.samples.size - event.pre_samples, 'trigger',
            f'event {self.count + 1}', event=self.count + 1,
            pre_trigger_samples=event.pre_samples,
            baseline_counts=round(event.baseline_counts, 3),
            **{k: v for k, v in f.items() if v is not None})
        self._writer.writerow({'event': self.count + 1, 'utc': utc_iso(event.utc),
                               'trigger_index': event.trigger_index,
                               'file_sample': start + event.pre_samples,
                               **{k: f.get(k) for k in self.FIELDS[4:]}})
        self._csv.flush()

    def close(self):
        with self._lock:
            if not self._csv.closed:
                self._csv.close()
                try:
                    self.recording.close()
                except OSError as exc:
                    self.error = self.error or exc
