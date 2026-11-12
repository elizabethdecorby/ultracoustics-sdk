"""Recordings on disk.

The ``.sigmf-data`` file is the raw little-endian uint16 ADC stream, exactly
what ``np.fromfile(path, '<u2')`` reads, and the ``.sigmf-meta`` JSON beside it
(https://sigmf.org) carries the sample rate, scaling, calibration, trigger
settings and one annotation per trigger event.

Opening a recording (snapshots, continuous recordings split into ``name``,
``name_001``, ..., and event files) does not read the samples into memory, and
a long recording can be reduced to a min/max/mean overview for navigating it.
"""
from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from itertools import accumulate
import json
import os
from pathlib import Path
import re
import warnings

import numpy as np

from .analysis import calibration as nepcal
from .analysis import spectrum as spectral
from .config import ADC_BITS, SAMPLE_RATE, STREAM_HEALTH_KEYS
from .readout import ADC_TO_POWER_UW, counts_to_uw

SIGMF_VERSION = '1.2.0'
DATATYPE = 'ru16_le'
EXTENSION = {'name': 'broadsonic', 'version': '1.0.0', 'optional': True}


def utc_iso(when):
    return when.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')


@dataclass
class TriggerEvent:
    number: int
    trigger_index: int        # absolute sample index where the trigger fired
    start_index: int          # absolute sample index of samples[0]
    samples: np.ndarray       # uint16 ADC counts, pre-trigger included
    sample_rate: float
    utc: datetime             # when samples[trigger] was taken (host clock)
    baseline_counts: float    # DC the trigger level was measured from
    uw_per_count: float
    features: dict = field(default_factory=dict)
    band_hz: tuple = None     # the trigger's band (band power or trigger filter), if any

    @property
    def pre_samples(self):
        return self.trigger_index - self.start_index

    def uw(self):
        return (self.samples.astype(np.float32) - self.baseline_counts) * self.uw_per_count


class SigMFRecording:
    """One ``.sigmf-data`` / ``.sigmf-meta`` pair.

    Samples are appended as they arrive. A new capture segment starts whenever
    the stream index jumps (a gap, or the next triggered event); the metadata
    is written atomically when the recording closes.
    """

    def __init__(self, base, sample_rate, global_meta=None):
        self.base = Path(base)
        self.base.parent.mkdir(parents=True, exist_ok=True)
        self.data_path = self.base.with_name(self.base.name + '.sigmf-data')
        self.meta_path = self.base.with_name(self.base.name + '.sigmf-meta')
        self.sample_rate = sample_rate
        self.global_meta = dict(global_meta or {})
        self.captures, self.annotations = [], []
        self.samples = 0
        self._next_index = None
        self._file = open(self.data_path, 'wb')

    @property
    def bytes(self):
        return self.samples * 2

    def write(self, samples, global_index, when):
        """Append samples taken from absolute stream index ``global_index`` at ``when``."""
        if global_index != self._next_index:
            self.captures.append({'core:sample_start': self.samples,
                                  'core:global_index': int(global_index),
                                  'core:datetime': utc_iso(when)})
        np.asarray(samples, dtype='<u2').tofile(self._file)
        self._next_index = int(global_index) + len(samples)
        self.samples += len(samples)

    def annotate(self, sample_start, sample_count, label, comment='', **extra):
        note = {'core:sample_start': int(sample_start), 'core:label': label}
        if sample_count is not None:
            note['core:sample_count'] = int(sample_count)
        if comment:
            note['core:comment'] = comment
        note.update({f'broadsonic:{k}': v for k, v in extra.items()})
        self.annotations.append(note)

    def close(self):
        if self._file.closed:
            return
        self._file.close()
        meta = {
            'global': {'core:datatype': DATATYPE, 'core:sample_rate': self.sample_rate,
                       'core:version': SIGMF_VERSION, 'core:extensions': [EXTENSION],
                       **self.global_meta},
            'captures': self.captures,
            'annotations': sorted(self.annotations, key=lambda a: a['core:sample_start']),
        }
        tmp = self.meta_path.with_name(self.meta_path.name + '.tmp')
        tmp.write_text(json.dumps(meta, indent=2), encoding='utf-8')
        os.replace(tmp, self.meta_path)


def read_sigmf(base):
    """(samples as a uint16 memmap, metadata) for a recording written here."""
    base = Path(str(base).removesuffix('.sigmf-meta').removesuffix('.sigmf-data'))
    meta = json.loads(base.with_name(base.name + '.sigmf-meta').read_text(encoding='utf-8'))
    data = np.memmap(base.with_name(base.name + '.sigmf-data'), dtype='<u2', mode='r')
    return data, meta


def recording_base(root, label, kind, now=None):
    """``root/YYYY-MM-DD/HHMMSS_label_kind``, made unique if it already exists."""
    now = now or datetime.now()
    clean = re.sub(r'[^A-Za-z0-9._-]+', '-', label.strip()).strip('-')
    stem = '_'.join(part for part in (now.strftime('%H%M%S'), clean, kind) if part)
    folder = Path(root).expanduser() / now.strftime('%Y-%m-%d')
    base, n = folder / stem, 2
    while base.with_name(base.name + '.sigmf-meta').exists() or \
            base.with_name(base.name + '.sigmf-data').exists():
        base, n = folder / f'{stem}-{n}', n + 1
    return base


KINDS = ('paused', 'event', 'recording', 'event-file')
#: The GUI's multitaper time-bandwidth: 2·NW−1 = 5 tapers.
#: Most FFTs the GUI's Analyze view averages in a spectrum.
MAX_VIEW_AVERAGES = 200
SAVE_CHUNK = 1 << 22      # samples written at a time


@dataclass
class Recording:
    """A fixed stretch of raw ADC counts with its time axis and metadata.

    ``samples`` stays on disk for a recording opened with :func:`load_recording`
    and is read when asked for, so a long recording costs no memory.
    :attr:`calibration` is the one saved with the recording, or None, and then
    :attr:`calibration_problem` says why if it had one that cannot be used.
    """
    name: str                 # for titles: 'Paused', 'Event 4', a file name
    samples: object           # uint16 counts: ndarray, memmap, or anything sliceable with len()
    sample_rate: float
    kind: str
    t0_s: float = 0.0         # time of samples[0] on the time axis (s)
    utc_start: datetime = None
    calibration: object = None   # the recording's own calibration, if any
    gaps: list = field(default_factory=list)     # [(sample index, missing samples)]
    markers: list = field(default_factory=list)  # [(sample index, label)]
    events: list = field(default_factory=list)   # TriggerEvents, for event files
    baseline_counts: float = None   # events: the DC the trigger measured from
    uw_per_count: float = ADC_TO_POWER_UW
    breaks: list = field(default_factory=list)   # sample indices where the stream is not contiguous
    calibration_problem: str = ''
    meta: dict = field(default_factory=dict)     # the SigMF metadata, for a recording from a file

    @property
    def start_time(self):
        """When ``samples[0]`` was taken (UTC), if known."""
        return self.utc_start

    @property
    def length(self):
        return len(self.samples)

    @property
    def duration_s(self):
        return self.length / self.sample_rate

    @property
    def end_s(self):
        return self.t0_s + self.duration_s

    @property
    def navigable(self):
        """Longer than one screen: it has a time window to move along."""
        return self.kind in ('paused', 'recording')

    @property
    def is_event(self):
        return self.kind in ('event', 'event-file')

    def index(self, t_s):
        """Sample index at time ``t_s`` (s on the plot's axis), clipped."""
        return int(np.clip(round((t_s - self.t0_s) * self.sample_rate), 0, self.length))

    def time(self, index):
        return self.t0_s + index / self.sample_rate

    def _range(self, t0_s, t1_s):
        return (0 if t0_s is None else self.index(t0_s),
                self.length if t1_s is None else self.index(t1_s))

    def counts(self, t0_s=None, t1_s=None):
        """The raw ADC counts between two times (s), all of them by default."""
        first, stop = self._range(t0_s, t1_s)
        return np.asarray(self.samples[first:stop])

    def power_uw(self, t0_s=None, t1_s=None, baseline=0.0):
        """The optical power in µW between two times (s), all of it by default.
        ``baseline``: ADC counts to subtract first, such as the dark level."""
        return counts_to_uw(self.counts(t0_s, t1_s), baseline, self.uw_per_count)

    def spans(self, t0_s=None, t1_s=None):
        """The contiguous (first, stop) sample ranges between two times: a gap
        in the stream, or the end of an event, starts a new one."""
        first, stop = self._range(t0_s, t1_s)
        edges = [first, *(b for b in sorted(self.breaks) if first < b < stop), stop]
        return [(a, b) for a, b in zip(edges, edges[1:]) if b > a]

    def spectrum(self, fft_size=16384, t0_s=None, t1_s=None, estimator='hann', band=None, *, plan=None,
                 max_averages=None, keep_ffts=False, short_ok=False):
        """The average power spectrum between two times, all of it by default.

        Every whole FFT is used, and none crosses a gap. ``fft_size`` sets the
        bin width (sample_rate / fft_size). ``estimator``: 'hann', or
        'multitaper' for cleaner peaks over fewer bins. ``band``: (lo, hi) Hz
        to zoom to, which takes far less time for a narrow band at fine bins.

        The GUI's Analyze view is this call with three more settings:
        ``max_averages`` limits the FFTs, spread evenly (the GUI uses
        MAX_VIEW_AVERAGES, to stay quick on long stretches). ``keep_ffts``
        keeps each FFT's spectrum in ``each_w2hz``, for the spread and peak.
        ``short_ok`` gives data too short for two FFTs one transform of all of
        it, at the bins that gives, instead of a ValueError if no FFT fits.
        ``plan`` is a ZoomPlan to use in place of ``band``.
        """
        scale = (self.uw_per_count * 1e-6) ** 2
        tapers = 1 if estimator == 'hann' else int(2 * spectral.MULTITAPER_NW) - 1
        options = dict(estimator=estimator, nw=spectral.MULTITAPER_NW, band=band, plan=plan,
                       spans=self.spans(t0_s, t1_s), max_averages=max_averages)
        if not (keep_ffts or short_ok):
            freqs, density, ffts, mean = spectral.welch(self.samples, self.sample_rate, fft_size, **options)
            return spectral.Spectrum(freqs, density * scale, mean, int(fft_size), ffts * tapers)
        parts = spectral.every_fft(self.samples, self.sample_rate, fft_size, short_ok=short_ok, **options)
        return spectral.pool_spectra([
            spectral.Spectrum(freqs, rows.mean(axis=0) * scale, mean, int(fft_size), len(rows) * tapers,
                              each_w2hz=rows * scale)
            for freqs, rows, mean in parts])

    def save(self, path, t0_s=None, t1_s=None, meta=None):
        """Write this recording, or the part between two times, as SigMF at ``path``
        (a name, or either file's path). Returns the path of the .sigmf-meta.

        ``meta``: SigMF global fields to set on the new file. Its calibration is
        kept unless ``meta`` gives one. A part also notes where it came from.
        FileExistsError rather than overwrite a recording.
        """
        if self.kind == 'event-file':
            raise ValueError('event files keep their events in their own metadata; copy the files instead')
        first, stop = self._range(t0_s, t1_s)
        if stop <= first:
            raise ValueError(f'{self.name} has no samples between those times')
        fields = self._global_meta(meta)
        if t0_s is not None or t1_s is not None:
            fields['broadsonic:trimmed_from'] = {'source': self.name, 'start_s': self.time(first),
                                                 'end_s': self.time(stop)}
        when = self.utc_start or datetime.now(timezone.utc)
        recording = SigMFRecording(new_recording_base(path), self.sample_rate, fields)
        missing = dict(self.gaps)
        lost = sum(count for index, count in self.gaps if index < first)
        for a, b in self.spans(t0_s, t1_s):
            lost += missing.get(a, 0)
            for pos in range(a, b, SAVE_CHUNK):
                end = min(pos + SAVE_CHUNK, b)
                recording.write(self.samples[pos:end], pos + lost,
                                when + timedelta(seconds=(pos + lost) / self.sample_rate))
        recording.close()
        return recording.meta_path

    def _global_meta(self, extra=None):
        """The SigMF global fields to write this recording with."""
        known = {k: v for k, v in self.meta.get('global', self.meta).items()
                 if k not in ('core:datatype', 'core:sample_rate', 'core:version', 'core:extensions')}
        if self.calibration is not None:
            known['broadsonic:calibration'] = nepcal.to_recording_dict(self.calibration)
        for key, value in (extra or {}).items():
            if key != 'broadsonic:calibration' or value:
                known[key] = value
        known.setdefault('broadsonic:uw_per_count', self.uw_per_count)
        return known

    def span(self, t_lo_s, t_hi_s, pad=0):
        """(first index, counts) for [t_lo, t_hi], with up to ``pad`` extra
        samples before it (to settle a filter); a copy, safe to modify."""
        start, stop = self.index(t_lo_s), self.index(t_hi_s)
        first = max(0, start - pad)
        return first, np.array(self.samples[first:max(stop, first)], dtype=np.uint16)


PART = re.compile(r'(.+)_(\d{3,})')   # a continuous recording's later files: name_001...
# Annotation fields of a trigger that are not the event's features
NOT_FEATURES = {'broadsonic:event', 'broadsonic:pre_trigger_samples',
                'broadsonic:baseline_counts'}


class ChainedSamples:
    """The files of a split recording as one sequence, read only when sliced.

    Supports ``len()`` and contiguous slices: ``chain[a:b]`` is a uint16 array,
    a view of the file when it lies in one part and a copy when it spans two.
    """

    def __init__(self, parts):
        self.parts = list(parts)
        self.starts = [0, *accumulate(len(p) for p in self.parts)]  # each part's first sample

    def __len__(self):
        return self.starts[-1]

    def __getitem__(self, key):
        if not isinstance(key, slice):
            raise TypeError('recording samples are read by slices, e.g. samples[a:b]')
        start, stop, step = key.indices(len(self))
        if step != 1:
            raise ValueError('recording samples are read by contiguous slices only')
        pieces = [part[max(start - first, 0):stop - first]
                  for part, first in zip(self.parts, self.starts)
                  if first < stop and first + len(part) > start]
        if not pieces:
            return np.zeros(0, np.uint16)
        return np.asarray(pieces[0]) if len(pieces) == 1 else np.concatenate(pieces)


def base_path(path):
    return Path(str(path).removesuffix('.sigmf-meta').removesuffix('.sigmf-data'))


def new_recording_base(path):
    """``path`` as a new recording's base name. FileExistsError rather than
    overwrite a recording."""
    base = base_path(path)
    if _has_meta(base) or with_ext(base, '.sigmf-data').exists():
        raise FileExistsError(f'{base} already has a recording; choose another name')
    return base


def stream_drops(controller):
    """The stream's loss counters so far."""
    stats = getattr(controller, 'stream_stats', None) or {}
    return {key: stats.get(key, 0) for key in STREAM_HEALTH_KEYS}


def global_meta(kind, controller=None, calibration=None, probe=None, description=None,
                recorder='Ultracoustics SDK', system_state=None):
    """The SigMF global fields of a recording made now, from the SDK and the GUI alike.

    ``controller``: the box the data just came from, so its lock and stream state
    apply. Leave it out for data from anywhere else, such as a stretch of an old file.
    """
    probe = calibration.probe if probe is None and calibration is not None else probe
    running = getattr(controller, 'running', False)
    return {'core:recorder': recorder, 'core:description': description or kind, 'core:hw': 'BROADSONIC',
            'broadsonic:kind': kind, 'broadsonic:adc_bits': ADC_BITS, 'broadsonic:uw_per_count': ADC_TO_POWER_UW,
            'broadsonic:system_state': system_state or ('RUNNING' if running else 'IDLE'),
            'broadsonic:probe': None if probe is None else dict(vars(probe)),
            'broadsonic:calibration': None if calibration is None else nepcal.to_recording_dict(calibration),
            **provenance(controller)}


def provenance(controller):
    """SigMF global fields saying what made a recording: the SDK, and with
    ``controller`` the box, its lock and the stream's loss counters so far."""
    from . import __version__
    meta = {'broadsonic:sdk_version': __version__}
    if controller is not None:
        meta.update({'broadsonic:box_serial': getattr(controller, 'device_serial', None),
                     'broadsonic:locked': bool(getattr(controller, 'locked', False)),
                     'broadsonic:stream_drops': stream_drops(controller)})
    return meta


def with_ext(base, ext):
    return base.with_name(base.name + ext)


def _has_meta(base):
    return with_ext(base, '.sigmf-meta').is_file()


def _chain(base):
    """(first file, every file in order) of the recording ``base`` belongs to."""
    m = PART.fullmatch(base.name)
    root = base.with_name(m[1]) if m and _has_meta(base.with_name(m[1])) else base
    parts = [root]
    while _has_meta(nxt := root.with_name(f'{root.name}_{len(parts):03d}')):
        parts.append(nxt)
    if base in parts:
        return root, parts
    if _has_meta(base):
        raise ValueError(f'{root.name}_{len(parts):03d}.sigmf-meta is missing, so '
                         f'{base.name} cannot be joined to the start of its recording')
    return base, [base]  # missing: reading it says what is missing


def _read_part(base):
    """(samples memmap, meta) of one file pair; ValueError if missing or unreadable."""
    meta_path, data_path = with_ext(base, '.sigmf-meta'), with_ext(base, '.sigmf-data')
    for path in (meta_path, data_path):
        if not path.is_file():
            raise ValueError(f'{path.name} not found')
    try:
        if data_path.stat().st_size:
            data, meta = read_sigmf(base)
        else:  # a part closed before any data came: numpy cannot map an empty file
            data = np.zeros(0, np.uint16)
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
        datatype = meta['global'].get('core:datatype')
        float(meta['global']['core:sample_rate'])
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f'{base.name} is not a readable SigMF recording ({exc})') from exc
    if datatype != DATATYPE:
        raise ValueError(f'{base.name}: unsupported datatype {datatype!r} '
                         f'(only {DATATYPE} recordings can be opened)')
    return data, meta


def _utc(text):
    when = datetime.fromisoformat(text)
    return when.astimezone(timezone.utc) if when.tzinfo else when.replace(tzinfo=timezone.utc)


def _calibration(compact):
    """(the recording's calibration or None, why it cannot be used). The dict
    stays in ``recording.meta``."""
    if not compact:
        return None, ''
    try:
        return nepcal.from_dict(compact), ''
    except nepcal.CalibrationError as exc:
        return None, str(exc)
    except (KeyError, TypeError, ValueError) as exc:
        return None, f'its calibration could not be read ({exc!r})'


def _gap_label(missing, sample_rate):
    ms = missing / sample_rate * 1e3
    return f'gap {ms:.3g} ms' if ms < 100 else f'gap {ms:.0f} ms'


def load_recording(path, sample_rate=SAMPLE_RATE):
    """Open a saved recording, without reading its samples into memory.

    ``path``: either file of a SigMF recording, or its name without the
    extension. Any file of a split recording opens all of them. A ``.bin``
    file of raw samples also works; it carries no metadata, so ``sample_rate``
    gives its rate. ValueError if the files are missing or unreadable. A warning
    says why, if the recording has a calibration that cannot be used.
    """
    path = Path(path)
    if path.suffix == '.bin':
        if not path.is_file():
            raise ValueError(f'{path.name} not found')
        return Recording(path.stem, np.memmap(path, dtype='<u2', mode='r'), float(sample_rate), 'recording')
    recording = open_recording(path)
    if recording.calibration_problem:
        warnings.warn(f'{recording.name}: {recording.calibration_problem}', stacklevel=2)
    return recording


def open_recording(path):
    """The recording at ``path`` (meta, data or base path) as a Recording.

    Opening any file of a split continuous recording opens all of them, joined
    in order. Raises ValueError if the files are missing or unreadable.
    """
    root, parts = _chain(base_path(path))
    loaded = [_read_part(part) for part in parts]
    try:
        source = _source(root.name, loaded)
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError(f'{root.name}: malformed metadata ({exc!r})') from exc
    if not source.length:
        raise ValueError(f'{root.name} holds no samples')
    return source


def _source(name, loaded):
    meta = loaded[0][1]
    g = meta['global']
    fs = float(g['core:sample_rate'])
    event_file = g.get('broadsonic:kind') == 'events'
    samples = ChainedSamples(data for data, _ in loaded)
    # Capture segments (start, stream index, time) and annotations of every
    # part, at their sample index in the joined stream
    segments, notes = [], []
    for first, (_, m) in zip(samples.starts, loaded):
        segments += [(first + c['core:sample_start'], c.get('core:global_index'),
                      c.get('core:datetime')) for c in m['captures']]
        notes += [(first + a['core:sample_start'], a) for a in m['annotations']]

    gaps = {}
    for i, note in notes:
        if note['core:label'] == 'gap':
            gaps[i] = gaps.get(i, 0) + int(note.get('broadsonic:lost_samples', 0))
    if not event_file:  # in an event file every event starts a segment of its own
        # The stream index jumps where samples are missing: the true count,
        # whether or not the recorder noted the gap
        for (s0, g0, _), (s1, g1, _) in zip(segments, segments[1:]):
            if g0 is not None and g1 is not None and g1 - g0 > s1 - s0:
                gaps[s1] = g1 - g0 - (s1 - s0)
    gaps = sorted(gaps.items())
    markers = [(i, _gap_label(n, fs)) for i, n in gaps]

    events = []
    trig = g.get('broadsonic:trigger') or {}
    band = trig.get('band_hz') or trig.get('filter_hz')  # a level trigger's filter
    seg_starts = [s for s, _, _ in segments]
    for i, note in notes:
        if note['core:label'] != 'trigger':
            continue
        number = note['broadsonic:event']
        markers.append((i, f'event {number}'))
        if not event_file:
            continue
        pre = int(note['broadsonic:pre_trigger_samples'])
        start, stop = i - pre, i + int(note.get('core:sample_count', 0))
        # Back-to-back events share a segment, so find it rather than assume one each
        s0, g0, when = segments[bisect_right(seg_starts, start) - 1]
        event = TriggerEvent(
            number, g0 + i - s0, g0 + start - s0, samples[start:stop], fs,
            _utc(when) + timedelta(seconds=(i - s0) / fs),
            float(note['broadsonic:baseline_counts']), float(g['broadsonic:uw_per_count']),
            features={k.removeprefix('broadsonic:'): v for k, v in note.items()
                      if k.startswith('broadsonic:') and k not in NOT_FEATURES},
            band_hz=tuple(band) if band else None)
        event.from_file = name
        events.append(event)

    calibration, problem = _calibration(g.get('broadsonic:calibration'))
    return Recording(name, samples, fs, 'event-file' if event_file else 'recording',
                     utc_start=_utc(segments[0][2]) if segments and segments[0][2] else None,
                     calibration=calibration, gaps=gaps, markers=sorted(markers, key=lambda m: m[0]),
                     events=events, uw_per_count=float(g.get('broadsonic:uw_per_count', ADC_TO_POWER_UW)),
                     breaks=seg_starts[1:] if event_file else [index for index, _ in gaps],
                     calibration_problem=problem,
                     meta=meta)   # the first file's; every part has the same global fields


def overview(samples, sample_rate, columns=4000, chunk=8_000_000, progress=None,
             cancelled=None):
    """Min, max and mean counts of ``samples`` in at most ``columns`` equal blocks.

    Reads ``chunk`` samples at a time. Returns ``(times_s, lo, hi, mean)``
    (times are block centres from sample 0), or None once ``cancelled()`` is
    true; ``progress(fraction)`` is called after each chunk.
    """
    n = len(samples)
    block = max(1, -(-n // max(1, int(columns))))
    cols = -(-n // block)
    lo, hi = np.full(cols, np.inf), np.full(cols, -np.inf)
    total = np.zeros(cols)
    chunk = max(1, int(chunk))
    for pos in range(0, n, chunk):
        if cancelled is not None and cancelled():
            return None
        x = np.asarray(samples[pos:pos + chunk])
        first = pos // block
        # Where each block starts in this chunk; the first may have begun in the last one
        idx = np.arange(first * block, pos + x.size, block) - pos
        idx[0] = 0
        here = slice(first, first + idx.size)
        lo[here] = np.minimum(lo[here], np.minimum.reduceat(x, idx))
        hi[here] = np.maximum(hi[here], np.maximum.reduceat(x, idx))
        total[here] += np.add.reduceat(x, idx, dtype=np.uint64)  # exact within a chunk
        if progress is not None:
            progress((pos + x.size) / n)
    starts = np.arange(cols) * block
    stops = np.minimum(starts + block, n)
    return (starts + stops - 1) / 2 / sample_rate, lo, hi, total / (stops - starts)


def recent_label(path):
    """Short menu text for a recording, e.g. '153012 label continuous · 2026-10-14'."""
    base = base_path(path)
    m = PART.fullmatch(base.name)
    text = (m[1] if m else base.name).replace('_', ' ')
    day = base.parent.name
    return f'{text} · {day}' if re.fullmatch(r'\d{4}-\d{2}-\d{2}', day) else text
