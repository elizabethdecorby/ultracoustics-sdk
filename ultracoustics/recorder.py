"""Gap-free reading of the live ADC stream, and recording it to disk.

Everything here reads the controller's shared-memory ring by absolute sample
index: ``samples_received`` counts every sample since the stream began, and
sample ``i`` lives at ``buffer[i % capacity]``. A reader that falls more than a
ring's length behind loses the overwritten span; that loss is counted and
reported, never skipped silently.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import threading
import time

import numpy as np

from .config import RING_GUARD_SAMPLES
from .recording import SigMFRecording, load_recording, stream_drops

# One USB packet. Data within a packet of the write head may be mid-update.
GUARD_SAMPLES = RING_GUARD_SAMPLES
SPLIT_BYTES = 2 * 1024 ** 3      # continuous recordings start a new file here
# The ring keeps filling while a long block is copied out of it; leave this
# much of it unused by snapshots and events.
MARGIN_SECONDS = 0.2
READ_SECONDS = 0.1               # largest block a reader takes at once
POLL_SECONDS = 0.02


class StreamReset(RuntimeError):
    """The sample counter went backwards: the stream was restarted."""


class CaptureError(RuntimeError):
    """A capture cannot be trusted: samples were lost, or the lock dropped."""


def copy_span(controller, start, stop):
    """Samples [start, stop) by absolute index, or None if the ring overwrote any."""
    buf = controller.buffer
    capacity = len(buf)
    total = int(controller.samples_received)
    if stop > total or total - start > capacity - GUARD_SAMPLES:
        return None
    begin, count = start % capacity, stop - start
    if begin + count <= capacity:
        out = np.array(buf[begin:begin + count], dtype=np.uint16)
    else:
        out = np.concatenate((buf[begin:], buf[:count - (capacity - begin)])).astype(
            np.uint16, copy=False)
    # The writer may have lapped the copy while it ran.
    if int(controller.samples_received) - start > capacity - GUARD_SAMPLES:
        return None
    return out


class StreamCursor:
    """Walks the ring in order; each ``read`` returns the next contiguous block."""

    def __init__(self, controller, start=None):
        self.controller = controller
        self.next = int(controller.samples_received) if start is None else int(start)

    def read(self, max_samples):
        """Return ``(start, samples, lost)``; ``lost`` samples were overwritten unread."""
        total = int(self.controller.samples_received)
        if total < self.next:
            raise StreamReset('sample counter went backwards')
        oldest = total - (len(self.controller.buffer) - GUARD_SAMPLES)
        lost = 0
        if self.next < oldest:
            lost, self.next = oldest - self.next, oldest
        stop = min(total, self.next + int(max_samples))
        if stop <= self.next:
            return self.next, np.zeros(0, np.uint16), lost
        data = copy_span(self.controller, self.next, stop)
        if data is None:  # lapped mid-copy: skip to what is still there
            oldest = int(self.controller.samples_received) - (
                len(self.controller.buffer) - GUARD_SAMPLES)
            lost += max(0, oldest - self.next)
            self.next = max(self.next, oldest)
            return self.next, np.zeros(0, np.uint16), lost
        start, self.next = self.next, stop
        return start, data, lost


def capture_samples(controller, count, sample_rate, require_lock=True, stall_s=2.0):
    """The next ``count`` samples of the stream, without a gap: ``(samples, start time)``.

    Reads while the ring fills, so ``count`` may exceed the ring. CaptureError if
    any sample was lost, the stream stalled or, with ``require_lock``, the lock
    was not confirmed at any point.
    """
    def check():
        if require_lock and not controller.locked:
            raise CaptureError('the lock is not confirmed: run start() and check_lock() or wait_ready() '
                               'first, or pass require_lock=False')

    health = controller._stream_health()
    check()
    cursor, parts, got, started = StreamCursor(controller), [], 0, None
    last = time.monotonic()
    try:
        while got < count:
            check()
            start, data, lost = cursor.read(min(count - got, int(READ_SECONDS * sample_rate)))
            if lost:
                raise CaptureError(f'{lost} samples were overwritten before they were read: '
                                   'the computer fell behind the stream')
            if data.size:
                if started is None:
                    lag = (int(controller.samples_received) - start) / sample_rate
                    started = datetime.now(timezone.utc) - timedelta(seconds=lag)
                parts.append(data)
                got += data.size
                last = time.monotonic()
            elif time.monotonic() - last > stall_s:
                raise CaptureError(f'no samples arrived for {stall_s:g} s: is the stream running?')
            else:
                time.sleep(POLL_SECONDS)
    except StreamReset as exc:
        raise CaptureError('the stream restarted during the capture') from exc
    check()
    after = controller._stream_health()
    dropped = {key: after[key] - health[key] for key in after if after[key] != health[key]}
    if dropped:
        raise CaptureError(f'the stream lost data during the capture: {dropped}')
    return np.concatenate(parts), started


def take_snapshot(controller, seconds, sample_rate):
    """Copy the newest ``seconds`` out of the ring: ``(samples, start index, start time)``."""
    total = int(controller.samples_received)
    capacity = len(controller.buffer)
    count = min(int(seconds * sample_rate), total,
                capacity - GUARD_SAMPLES - int(MARGIN_SECONDS * sample_rate))
    if count <= 0:
        raise RuntimeError('No data in memory yet')
    start = total - count
    samples = copy_span(controller, start, total)
    if samples is None:
        raise RuntimeError('Data was overwritten while copying; try a shorter snapshot')
    return samples, start, datetime.now(timezone.utc) - timedelta(seconds=count / sample_rate)


def write_snapshot(snapshot, base, sample_rate, global_meta):
    samples, start, taken = snapshot
    recording = SigMFRecording(base, sample_rate, global_meta)
    recording.write(samples, start, taken)
    recording.close()
    return recording


def save_snapshot(controller, seconds, base, sample_rate, global_meta):
    """Write the newest ``seconds`` already in memory. Returns the recording."""
    return write_snapshot(take_snapshot(controller, seconds, sample_rate), base,
                          sample_rate, global_meta)


class ContinuousRecorder(threading.Thread):
    """Streams every sample to disk until stopped or a limit is reached.

    Files split at ``split_bytes`` (``name``, ``name_001``, ...). Samples the
    disk could not keep up with are recorded as a ``gap`` annotation, and the
    next block starts a new capture segment at its true stream index. Samples
    lost before the ring never reach this reader: each file's metadata ends with
    the stream's counters, and :attr:`dropped` says what rose.
    """

    def __init__(self, controller, base, sample_rate, global_meta, *, max_seconds=0,
                 max_bytes=0, split_bytes=SPLIT_BYTES, on_finished=None):
        super().__init__(name='continuous-recorder', daemon=True)
        self.controller = controller
        self.base = Path(base)
        self.sample_rate = sample_rate
        self.global_meta = global_meta
        self.max_samples = int(max_seconds * sample_rate) if max_seconds else 0
        self.max_bytes = int(max_bytes)
        self.split_bytes = int(split_bytes)
        self.on_finished = on_finished or (lambda recorder: None)
        self.files = []
        self.samples = 0
        self.lost = 0
        self.error = None
        self.reason = ''
        self.started_at = time.monotonic()
        self._drops_at_start = stream_drops(controller)
        self._stop_event = threading.Event()

    @property
    def dropped(self):
        """The stream's loss counters that rose since this recording began."""
        now = stream_drops(self.controller)
        return {key: now[key] - self._drops_at_start[key] for key in now if now[key] != self._drops_at_start[key]}

    @property
    def bytes(self):
        return self.samples * 2

    @property
    def elapsed(self):
        return time.monotonic() - self.started_at

    def stop(self):
        self._stop_event.set()

    def open(self):
        """The finished recording, read from disk."""
        return load_recording(self.files[0].meta_path)

    def _close_part(self, recording):
        recording.global_meta['broadsonic:stream_drops_end'] = stream_drops(self.controller)
        recording.close()

    def _open_part(self):
        part = len(self.files)
        base = self.base if part == 0 else self.base.with_name(f'{self.base.name}_{part:03d}')
        self.files.append(SigMFRecording(base, self.sample_rate, self.global_meta))
        return self.files[-1]

    def run(self):
        recording = None
        try:
            cursor = StreamCursor(self.controller)
            recording = self._open_part()
            while not self._stop_event.is_set():
                start, data, lost = cursor.read(int(READ_SECONDS * self.sample_rate))
                if lost:
                    self.lost += lost
                    recording.annotate(recording.samples, None, 'gap',
                                       f'{lost} samples lost (recording fell behind)',
                                       lost_samples=lost)
                if data.size:
                    if self.max_samples:
                        data = data[:self.max_samples - self.samples]
                    lag = (int(self.controller.samples_received) - start) / self.sample_rate
                    recording.write(data, start, datetime.now(timezone.utc) - timedelta(seconds=lag))
                    self.samples += data.size
                    if self.max_samples and self.samples >= self.max_samples:
                        self.reason = 'duration reached'
                        break
                    if self.max_bytes and self.bytes >= self.max_bytes:
                        self.reason = 'size limit reached'
                        break
                    if recording.bytes >= self.split_bytes:
                        self._close_part(recording)
                        recording = self._open_part()
                if data.size < READ_SECONDS * self.sample_rate:
                    self._stop_event.wait(POLL_SECONDS)
            else:
                self.reason = self.reason or 'stopped'
        except StreamReset:
            self.reason = 'stream restarted'
        except OSError as exc:
            self.error, self.reason = exc, f'write failed: {exc}'
        finally:
            try:
                if recording is not None:
                    self._close_part(recording)
            except OSError as exc:
                self.error, self.reason = exc, f'write failed: {exc}'
            self.on_finished(self)
