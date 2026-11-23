"""Watching the live stream, and replaying the same triggers over a saved recording.

A monitor checks every sample against its triggers. It is built to run unattended,
so it says what it did: :meth:`Monitor.summary` gives the events, the alarms, the
samples lost and the time it was not watching, and a folder of files holds the
same for after the run.

Triggering runs on the PC, not in the box, so the PC must stay connected and keep up.
Whatever it misses is counted, never skipped silently.

- An **event** is a captured stretch of data around a trigger.
- An **alarm** is a condition that stays above its limit: it is raised once and stays
  raised until the condition clears, so a fault that persists doesn't fill the disk.
- While the lock is lost nothing is watched: events then would only be the lock
  moving. The monitor pauses, says so, and records the time.
"""
from collections import deque
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import re
import threading
import time

import numpy as np

from .recorder import POLL_SECONDS, READ_SECONDS, StreamCursor, StreamReset, copy_span
from .recording import utc_iso
from .trigger import BandPowerTrigger, EventRecorder, Watch, make_event

#: After the lock returns, wait this long before trusting the signal again.
LOCK_SETTLE_S = 1.0
#: Telemetry lags the signal: hold each event this long, so one fired by the lock
#: being lost is dropped with it.
LOCK_GUARD_S = 0.1
#: An alarm clears after the condition has been below its limit for this long.
ALARM_CLEAR_S = 1.0
EVENT_FILE_BYTES = 1 << 30


class _Baseline:
    """Raw counts to µW about a slow DC estimate (~100 ms), as triggers measure them."""

    def __init__(self, sample_rate, uw_per_count, dc_removed):
        self.sample_rate, self.uw_per_count, self.dc_removed = sample_rate, uw_per_count, dc_removed
        self.dc = None

    def __call__(self, data):
        """(µW as float32, the baseline in counts)."""
        mean = float(data.mean())
        alpha = min(1.0, data.size / (0.1 * self.sample_rate))
        self.dc = mean if self.dc is None else self.dc + alpha * (mean - self.dc)
        baseline = self.dc if self.dc_removed() else 0.0
        return (data.astype(np.float32) - np.float32(baseline)) * np.float32(self.uw_per_count), baseline


class _Alarm:
    """Raised when a trigger's level reaches its limit, cleared after it stays below."""

    def __init__(self, label):
        self.label = label
        self.active = False
        self.peak = None
        self.raised = 0
        self._quiet_s = 0.0

    def update(self, detector, seconds):
        """Follow one block: 'raised', 'cleared' or None."""
        peak = detector.peak
        if peak is None:
            return None
        if not self.active:
            if peak < detector.fire:
                return None
            self.active, self.peak, self._quiet_s = True, peak, 0.0
            self.raised += 1
            return 'raised'
        self.peak = max(self.peak, peak)
        self._quiet_s = self._quiet_s + seconds if peak < detector.clear else 0.0
        if self._quiet_s < ALARM_CLEAR_S:
            return None
        self.active = False
        return 'cleared'


@dataclass(frozen=True)
class AlarmChange:
    """An alarm raised or cleared, as ``on_alarm`` receives it."""
    trigger: str
    state: str          # 'raised' or 'cleared'
    utc: datetime
    level: float        # the trigger's level: µW, or dB above its normal level (dB SPL with level_spl)
    limit: float


@dataclass(frozen=True)
class MonitorSummary:
    """How a monitor run went. ``not_watched`` lists (UTC start, seconds) of each pause."""
    started: datetime
    seconds: float
    samples_checked: int
    samples_lost: int
    not_monitored_s: float
    not_watched: tuple
    events: int
    events_not_saved: int
    alarms_raised: int
    alarms_active: tuple
    folder: str = None
    error: str = None

    @property
    def complete(self):
        """Every second of the run was watched, and nothing went wrong."""
        return not (self.samples_lost or self.not_monitored_s or self.events_not_saved or self.error)


class EventFolder:
    """Events to numbered recordings in a folder, a new file past ``file_bytes``.

    Stops saving at ``max_bytes`` in all, and counts what it could not keep: it never
    deletes anything. The monitor goes on watching and logging either way.
    """

    def __init__(self, folder, sample_rate, global_meta, max_bytes=0, file_bytes=EVENT_FILE_BYTES):
        self.folder = Path(folder)
        self.sample_rate, self.global_meta = sample_rate, global_meta
        self.max_bytes, self.file_bytes = int(max_bytes), int(file_bytes)
        self.closed_bytes = 0
        self.count = self.not_saved = 0
        self.error = None
        self.parts = 0
        self.current = None

    @property
    def bytes(self):
        return self.closed_bytes + (self.current.bytes if self.current is not None else 0)

    @property
    def full(self):
        return bool(self.max_bytes) and self.bytes >= self.max_bytes

    def _roll(self):
        if self.current is not None:
            self.close()
        self.parts += 1
        self.current = EventRecorder(self.folder / f'events_{self.parts:04d}', self.sample_rate, self.global_meta)

    def add(self, event):
        if self.error is not None or self.full:
            self.not_saved += 1
            return False
        if self.current is None or self.current.bytes >= self.file_bytes:
            try:
                self._roll()
            except OSError as exc:
                self.error, self.not_saved = exc, self.not_saved + 1
                return False
        if not self.current.add(event):
            self.error, self.not_saved = self.current.error, self.not_saved + 1
            return False
        self.count += 1
        return True

    def close(self):
        if self.current is not None:
            self.current.close()
            self.closed_bytes += self.current.bytes
            self.error = self.error or self.current.error
            self.current = None


def _column(label):
    return re.sub(r'[^a-z0-9]+', '_', label.lower()).strip('_')


class TrendLog:
    """A row of level statistics every ``period_s`` of data, and a row for each alarm change.

    ``trend.csv`` is often the main output of a long run: a machine getting slowly
    worse shows up here before any trigger fires.
    """

    def __init__(self, folder, labels, period_s, sample_rate):
        self.period_samples = max(1, int(period_s * sample_rate))
        self.columns = [_column(label) for label in labels]
        names = ['utc', 'locked', 'dc_uw', 'rms_uw', 'peak_uw', 'crest', 'kurtosis', 'temperature_c', 'events',
                 'samples_lost', 'not_monitored_s']
        for column in self.columns:
            names += [f'{column}_level', f'{column}_alarm']
        folder = Path(folder)
        self._trend = open(folder / 'trend.csv', 'w', newline='', encoding='utf-8-sig')
        self._alarms = open(folder / 'alarms.csv', 'w', newline='', encoding='utf-8-sig')
        self.rows = csv.DictWriter(self._trend, fieldnames=names)
        self.rows.writeheader()
        self.alarm_rows = csv.writer(self._alarms)
        self.alarm_rows.writerow(('utc', 'trigger', 'state', 'level', 'limit'))
        self._alarms.flush()
        self._reset()

    def _reset(self):
        self.count = 0
        self.seen = 0
        self.sums = np.zeros(4)   # of x, x², x³, x⁴
        self.biggest = 0.0
        self.levels = [None] * len(self.columns)

    def add(self, x, detectors):
        """Count a watched block's samples and the triggers' levels."""
        y = np.asarray(x, np.float64)
        squared = y * y
        self.sums += (y.sum(), squared.sum(), (squared * y).sum(), (squared * squared).sum())
        self.biggest = max(self.biggest, float(np.max(np.abs(y))))
        self.count += y.size
        for position, detector in enumerate(detectors):
            if detector.peak is not None:
                self.levels[position] = max(detector.peak, self.levels[position] or -math.inf)

    def skip(self, samples):
        """Count samples that were not watched."""
        self.seen += samples

    @property
    def due(self):
        return self.count + self.seen >= self.period_samples

    def _statistics(self):
        n = self.count
        if not n:
            return {}
        mean, m2, m3, m4 = self.sums[0] / n, self.sums[1] / n, self.sums[2] / n, self.sums[3] / n
        variance = m2 - mean ** 2
        stats = {'rms_uw': math.sqrt(max(variance, 0.0)), 'peak_uw': self.biggest}
        if variance > 0:
            stats['crest'] = self.biggest / math.sqrt(variance)
            stats['kurtosis'] = (m4 - 4 * mean * m3 + 6 * mean ** 2 * m2 - 3 * mean ** 4) / variance ** 2
        return stats

    def write(self, utc, dc_uw, locked, temperature_c, events, lost, not_monitored_s, alarms):
        row = {'utc': utc_iso(utc), 'locked': int(locked), 'dc_uw': dc_uw, 'temperature_c': temperature_c,
               'events': events, 'samples_lost': lost, 'not_monitored_s': round(not_monitored_s, 3),
               **self._statistics()}
        for column, level, alarm in zip(self.columns, self.levels, alarms):
            row[f'{column}_level'] = level
            row[f'{column}_alarm'] = int(alarm.active)
        self.rows.writerow(row)
        self._trend.flush()
        self._reset()

    def alarm(self, change):
        self.alarm_rows.writerow((utc_iso(change.utc), change.trigger, change.state, change.level, change.limit))
        self._alarms.flush()

    def close(self):
        self._trend.close()
        self._alarms.close()


class Monitor(threading.Thread):
    """Runs triggers over every sample and hands back captured events.

    ``triggers`` is one trigger or a list. ``on_event(event)``, ``on_status(text)`` and
    ``on_alarm(change)`` are called from this thread. In single mode it stops itself
    after the first event. :meth:`Controller.monitor` is the way to make one.

    ``require_lock``: watch only while the lock holds, see the module docstring.
    ``events_during_alarm``: with False, a trigger whose alarm is raised captures no
    more events until it clears, so a persistent fault gives one event, not thousands.
    """

    def __init__(self, controller, sample_rate, uw_per_count, triggers, *, pre_samples, post_samples,
                 holdoff_samples, single=False, dc_removed=lambda: True, on_event=None, on_status=None,
                 on_alarm=None, on_finished=None, sink=None, calibration=None, require_lock=False,
                 events_during_alarm=True, trend=None, owns_sink=False, folder=None):
        super().__init__(name='monitor', daemon=True)
        self.controller = controller
        self.sample_rate = sample_rate
        self.uw_per_count = uw_per_count
        self.watch = Watch(triggers, sample_rate, holdoff_samples, calibration)
        self.window = (int(pre_samples), int(post_samples))
        self.single = single
        self.on_event = on_event or (lambda event: None)
        self.on_status = on_status or (lambda text: None)
        self.on_alarm = on_alarm or (lambda change: None)
        self.on_finished = on_finished or (lambda monitor: None)
        self.sink = sink
        self.owns_sink = owns_sink
        self.require_lock = require_lock
        self.events_during_alarm = events_during_alarm
        self.trend = trend
        self.folder = folder
        self.baseline = _Baseline(sample_rate, uw_per_count, dc_removed)
        self.alarms = [_Alarm(t.label) for t in self.watch.triggers]
        self.fired = 0
        self.lost = 0
        self.checked = 0
        self.events_held_back = 0
        self.error = None
        self.started_at = datetime.now(timezone.utc)
        self._started_clock = time.monotonic()
        self._finished_seconds = None
        self._settling = None        # samples left before a returned lock counts; None while watching
        self._not_watched = 0
        self._pauses = []            # [UTC start, samples]
        # (time, value) of what the first trigger measures, and when it fired, for a
        # live meter (monotonic seconds; read from other threads with list())
        self.history = deque(maxlen=4000)
        self.event_times = deque(maxlen=200)
        self._stop_event = threading.Event()

    # -- The one-trigger view the GUI's Trigger tab uses ---------------------------------

    @property
    def trigger(self):
        return self.watch.triggers[0]

    @property
    def detector(self):
        return self.watch.detectors[0]

    def can_update(self, trigger):
        """Whether ``trigger`` differs only in levels the running detector can take."""
        old = self.trigger
        if type(trigger) is not type(old) or trigger.band_hz != old.band_hz:
            return False
        return isinstance(trigger, BandPowerTrigger) or trigger.edge == old.edge

    def update(self, trigger, pre_samples, post_samples, holdoff_samples):
        """Change thresholds and the capture window without restarting (no re-learning).

        A new level or threshold needs a fresh edge: lowering it under a steady
        signal must not fire.
        """
        old = self.trigger
        self.watch.triggers[0] = self.detector.cfg = trigger
        self.window = (int(pre_samples), int(post_samples))
        self.detector.schmitt.holdoff = int(holdoff_samples)
        if trigger != old:
            self.detector.schmitt.reset()

    # -- State -------------------------------------------------------------------------

    @property
    def paused(self):
        """Not watching now: the lock is gone, or has not been back long enough."""
        return self._settling is not None

    @property
    def stopped(self):
        """Whether :meth:`stop` has been called."""
        return self._stop_event.is_set()

    @property
    def active_alarms(self):
        """The names of the triggers whose alarm is raised now."""
        return tuple(alarm.label for alarm in self.alarms if alarm.active)

    def stop(self, timeout=1.0):
        """Stop watching, and wait up to *timeout* seconds for the monitor to finish
        its files. :meth:`summary` then says how the run went."""
        self._stop_event.set()
        if self.is_alive() and threading.current_thread() is not self:
            self.join(timeout)

    def summary(self):
        """How the run has gone so far, as a :class:`MonitorSummary`."""
        seconds = self._finished_seconds
        saved = self.sink if self.owns_sink else None
        return MonitorSummary(
            self.started_at, time.monotonic() - self._started_clock if seconds is None else seconds, self.checked,
            self.lost, self._not_watched / self.sample_rate,
            tuple((when, count / self.sample_rate) for when, count in self._pauses), self.fired,
            getattr(saved, 'not_saved', 0) + self.events_held_back, sum(alarm.raised for alarm in self.alarms),
            self.active_alarms, None if self.folder is None else str(self.folder),
            None if self.error is None else str(self.error))

    # -- The loop ----------------------------------------------------------------------

    def run(self):
        try:
            self._run()
        except Exception as exc:  # the thread ends; the owner can start another
            self.error = exc
            self.on_status(f'Trigger error: {exc}')
        finally:
            self._finish()

    def _finish(self):
        self._finished_seconds = time.monotonic() - self._started_clock
        if self.trend is not None:
            self.trend.close()
        if self.owns_sink and self.sink is not None:
            self.sink.close()
        if self.folder is not None:
            text = json.dumps(asdict(self.summary()), indent=2, default=str)
            (Path(self.folder) / 'summary.json').write_text(text, encoding='utf-8')
        self.on_finished(self)

    def _run(self):
        cursor = StreamCursor(self.controller)
        pending = []
        chunk = int(READ_SECONDS * self.sample_rate)
        while not self._stop_event.is_set():
            try:
                start, data, lost = cursor.read(chunk)
            except StreamReset:
                self.on_status('Stream restarted; trigger re-armed')
                cursor, self.baseline.dc = StreamCursor(self.controller), None
                self.watch.reset()
                pending.clear()
                continue
            if lost:
                self.lost += lost
                self.watch.reset()
                self.on_status(f'Trigger fell behind: {lost / self.sample_rate * 1e3:.0f} ms not checked')
            if data.size and self._watching(data.size, start, pending):
                self._check(start, data, pending)
            self._emit_ready(pending)
            self._log_trend()
            if self.single and self.fired and not pending:
                break
            if data.size < chunk:
                self._stop_event.wait(POLL_SECONDS)
        self.fired -= len(pending)     # stopped before their data arrived: not events

    def _utc_of(self, index):
        lag = (int(self.controller.samples_received) - index) / self.sample_rate
        return datetime.now(timezone.utc) - timedelta(seconds=lag)

    def _watching(self, size, start, pending):
        """Whether this block is watched, following the lock."""
        if not self.require_lock:
            return True
        locked = bool(self.controller.locked)
        if locked and self._settling is None:
            return True
        if not locked:
            if self._settling is None:
                self._pauses.append([self._utc_of(start), 0])
                self.fired -= len(pending)      # those events were the lock going, not the machine
                pending.clear()
                self.watch.reset()
                self.on_status('Lock lost: not monitoring until it returns' if self.checked
                               else 'Waiting for the lock')
            self._settling = int(LOCK_SETTLE_S * self.sample_rate)
        else:
            self._settling -= size
        self._not_watched += size
        self._pauses[-1][1] += size
        if self.trend is not None:
            self.trend.skip(size)
        if locked and self._settling <= 0:
            self._settling = None
            self.on_status('Lock back: monitoring again')
        return False

    def _check(self, start, data, pending):
        x, baseline = self.baseline(data)
        was_raised = [alarm.active for alarm in self.alarms]
        hits = self.watch.process(start, x)
        self.checked += data.size
        for position, index in hits:
            if self.single and self.fired:
                break
            if not self.events_during_alarm:
                if was_raised[position]:
                    continue
                was_raised[position] = True    # this first hit raises it, at the end of the block
            self.fired += 1
            lag = (int(self.controller.samples_received) - index) / self.sample_rate
            self.event_times.append(time.monotonic() - lag)
            pending.append((self.fired, index, baseline, datetime.now(timezone.utc) - timedelta(seconds=lag),
                            self.window, position))
        pending.sort(key=lambda item: item[1])
        first = self.detector
        if first.level is not None:
            self.history.append((time.monotonic(), first.level))
        seconds = data.size / self.sample_rate
        for alarm, detector in zip(self.alarms, self.watch.detectors):
            state = alarm.update(detector, seconds)
            if state is not None:
                change = AlarmChange(alarm.label, state, self._utc_of(start), detector.peak, detector.fire)
                if self.trend is not None:
                    self.trend.alarm(change)
                self.on_alarm(change)
        if self.trend is not None:
            self.trend.add(x, self.watch.detectors)

    def _emit_ready(self, pending):
        total = int(self.controller.samples_received)
        hold = int(LOCK_GUARD_S * self.sample_rate) if self.require_lock else 0
        while pending and total >= pending[0][1] + max(pending[0][4][1], hold):
            number, index, baseline, utc, (pre, post), position = pending.pop(0)
            samples = copy_span(self.controller, index - pre, index + post)
            if samples is None:
                self.on_status('An event was lost: its data left the memory buffer')
                self.events_held_back += 1
                if self.single:
                    self.fired = 0  # wait for another
                continue
            event = make_event(number, self.watch.triggers[position], self.watch.detectors[position], index,
                               index - pre, samples, self.sample_rate, utc, baseline, self.uw_per_count)
            sink = self.sink  # the caller may clear it at any time
            if sink is not None:
                sink.add(event)
            self.on_event(event)

    def _log_trend(self):
        trend = self.trend
        if trend is None or not trend.due:
            return
        snapshot = getattr(self.controller, 'telemetry', None)
        board = getattr(snapshot, 'board_638', None)
        temperature = None if board is None else board.temperature_mdegc / 1000.0
        locked = bool(getattr(self.controller, 'locked', True))
        dc = self.baseline.dc
        trend.write(datetime.now(timezone.utc), None if dc is None else dc * self.uw_per_count, locked, temperature,
                    self.fired, self.lost, self._not_watched / self.sample_rate, self.alarms)


def find_events(recording, triggers, pre_s=0.01, post_s=0.1, holdoff_s=None, calibration=None):
    """Replay ``triggers`` over a saved recording, as a live monitor would: tune limits
    here, with no box. Returns the list of TriggerEvents. No trigger fires across a gap.

    ``holdoff_s``: how long a trigger stays off after firing, ``post_s`` by default.
    ``calibration``: for limits in pressure, the recording's own by default.
    """
    fs = recording.sample_rate
    pre, post = int(round(pre_s * fs)), max(1, int(round(post_s * fs)))
    holdoff = post if holdoff_s is None else int(round(holdoff_s * fs))
    watch = Watch(triggers, fs, holdoff, recording.calibration if calibration is None else calibration)
    baseline = _Baseline(fs, recording.uw_per_count, lambda: True)
    began = recording.start_time or datetime.now(timezone.utc)
    chunk, events = int(READ_SECONDS * fs), []
    for first, stop in recording.spans():
        watch.reset()
        baseline.dc = None
        for pos in range(first, stop, chunk):
            x, dc = baseline(np.asarray(recording.samples[pos:min(pos + chunk, stop)]))
            for position, index in watch.process(pos, x):
                a, b = max(first, index - pre), min(stop, index + post)
                events.append(make_event(len(events) + 1, watch.triggers[position], watch.detectors[position], index,
                                         a, np.array(recording.samples[a:b], dtype=np.uint16), fs,
                                         began + timedelta(seconds=index / fs), dc, recording.uw_per_count))
    return events


__all__ = ['AlarmChange', 'EventFolder', 'Monitor', 'MonitorSummary', 'TrendLog', 'find_events']
