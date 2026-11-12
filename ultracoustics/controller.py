"""
The :class:`Controller`: the box, from connection to capture, recording, monitoring and the lock.

The USB stream is read by a process of its own into a shared-memory ring, so
nothing is lost while your script is busy. That process imports your script
when it starts: keep a script's work under ``if __name__ == "__main__":``.

Typical usage::

    from ultracoustics import Controller

    with Controller() as ctrl:              # close() always runs, with a confirmed stop
        dark = ctrl.capture_dark()          # lasers off
        ctrl.start()
        ctrl.check_lock(retune=True)        # every START, where the signal can stand it
        ctrl.calibrate_pressure(dark)
        recording = ctrl.capture(1.0)       # checks the lock and for lost samples
        recording.save("capture")           # SigMF
"""

from dataclasses import dataclass
from datetime import datetime
import time
from pathlib import Path
from typing import Optional

import numpy as np

from . import lock as _lock
from ._internal.device import Box
from ._internal.protocol import CMD_WARM
from .analysis import calibration as nepcal
from .config import RING_GUARD_SAMPLES, SAMPLE_RATE
from .monitor import EventFolder, Monitor, TrendLog
from .readout import ADC_TO_POWER_UW, DARK_MAX_COUNTS, DARK_SETTLE_S
from .recorder import CaptureError, ContinuousRecorder, MARGIN_SECONDS, capture_samples, take_snapshot
from .recording import Recording, global_meta, new_recording_base



# After a lock check the excitation needs this long to die away before a fit.
EXCITATION_DECAY_S = 0.5


@dataclass(frozen=True)
class Status:
    """What the box and the stream are doing now, from cached state."""
    connected: bool
    streaming: bool
    running: bool             # between start() and stop()
    locked: bool              # fresh telemetry shows the 638 locked
    recording: bool
    monitoring: bool
    samples_received: int
    stream_drops: dict        # the stream's loss counters since it began


@dataclass(frozen=True)
class BoxFault:
    """Why the box stopped itself: its fault code and what it means (see :meth:`Controller.stop`)."""
    code: int
    reason: str


#: What a Controller offers. Everything under them is on its private box.
PUBLIC_NAMES = ('buffer', 'buffer_capacity', 'buffer_head', 'calibrate_lock', 'calibrate_pressure', 'calibration',
                'capture', 'capture_dark', 'check_lock', 'clear_stored_lock_gains', 'close', 'connect', 'connected',
                'device_serial', 'lock_gains', 'locked', 'measure_signal_level', 'monitor', 'record',
                'request_telemetry', 'restart', 'restore_lock_gains', 'running', 'samples_received', 'self_test',
                'snapshot', 'start', 'status', 'stop', 'store_signal_level', 'stream_stats', 'streaming',
                'telemetry', 'verbose', 'wait_ready', 'warm')


class Controller:
    """
    The Ultracoustics box: one per box, from one script at a time.

    Usage::

        with Controller() as ctrl:
            ctrl.start()
            ctrl.check_lock(retune=True)    # or wait_ready() if a burst cannot be tolerated
            recording = ctrl.capture(1.0)
            ctrl.record("long_run", duration_s=60)

    For a live display, ``buffer`` is the ring itself, without a copy: read
    ``buffer_head`` first, then slice ``buffer``. ``samples_received`` counts
    every sample since the stream began. :meth:`snapshot` copies the newest
    samples out as a :class:`Recording`.
    """

    #: The box under it (the lab's service tools give their controller a box with more)
    _box_class = Box

    def __init__(self, verbose=False, ring_seconds: float = 1.2,
                 packet_diagnostics_path: Optional[str] = None,
                 packet_diagnostics_attach_s: float = 2.0):
        """Nothing is sent to the box until :meth:`connect` (or ``with``).

        Parameters
        ----------
        verbose : bool, optional
            Print messages about USB traffic and captures.
        ring_seconds : float, optional
            How many seconds of samples the live ring holds: 1.2 by default.
            A monitor's event (``pre_s + post_s``) and :meth:`snapshot` are limited
            by it. Each second takes 20 MB of memory.
        packet_diagnostics_path : str or None, optional
            For support: the reader writes a log of USB packet headers here.
        packet_diagnostics_attach_s : float, optional
            For support: how long after the stream starts to log every header.
        """
        self._box = self._box_class(verbose, int(SAMPLE_RATE * ring_seconds), packet_diagnostics_path,
                                    packet_diagnostics_attach_s)
        #: The Calibration that captures and recordings keep with their data.
        self.calibration = None
        self._recorders = set()
        self._monitors = set()

    # -- Connection lifecycle -------------------------------------------------

    def connect(self):
        """Find the box and start its stream: from here :attr:`buffer` fills with
        samples, with the lasers off until :meth:`start`. ``with Controller() as ctrl``
        calls this for you.

        Raises
        ------
        RuntimeError
            If no box is found, or another program has it open.
        """
        self._box.connect()

    def __enter__(self):
        if not self.connected:
            self.connect()
        return self

    def __exit__(self, *exc_info):
        self.close()

    @property
    def verbose(self) -> bool:
        """Print messages about USB traffic and captures."""
        return self._box.verbose

    @verbose.setter
    def verbose(self, value):
        self._box.verbose = value

    @property
    def device_serial(self) -> Optional[str]:
        """The box's USB serial number, once connected."""
        return self._box.device_serial

    @device_serial.setter
    def device_serial(self, value):
        self._box.device_serial = value

    @property
    def connected(self) -> bool:
        """``True`` once :meth:`connect` has succeeded and resources are live."""
        return self._box.connected

    # -- Live buffer access (read-only) ---------------------------------------

    @property
    def buffer(self) -> np.ndarray:
        """The live ring of samples, in ADC counts (uint16). Do not write to it."""
        return self._box.buffer

    @property
    def buffer_head(self) -> int:
        """Where the next sample will go in :attr:`buffer`."""
        return self._box.buffer_head

    @property
    def buffer_capacity(self) -> int:
        """The length of :attr:`buffer`, in samples."""
        return self._box.buffer_capacity

    @property
    def samples_received(self) -> int:
        """Cumulative number of samples received since streaming started."""
        return self._box.samples_received

    @property
    def running(self) -> bool:
        """``True`` between :meth:`start` and :meth:`stop`."""
        return self._box.running

    @property
    def streaming(self) -> bool:
        """``True`` while samples are arriving in :attr:`buffer`: from :meth:`connect`
        until :meth:`close`, unless the reader stopped (a lost device, say: see
        :attr:`stream_stats`).
        """
        return self._box.streaming

    @property
    def locked(self) -> bool:
        """``True`` while fresh telemetry shows the 638 locked. This says nothing
        about how good the lock is: for that, see :meth:`check_lock`."""
        return _lock.locked_638(self._box)

    @property
    def status(self) -> Status:
        """The state of the box and the stream. Sends nothing to the box."""
        return Status(self.connected, self.streaming, self.running, self.locked, bool(self._recorders),
                      bool(self._monitors), self.samples_received, self._box._stream_health())

    @property
    def stream_stats(self) -> Optional[dict]:
        """The stream's packet counters: ``packets``, ``drops_host``, ``malformed``
        and ``last_seq`` among them. ``None`` when not connected."""
        return self._box.stream_stats

    @property
    def telemetry(self):
        """The box's newest telemetry (lasers, temperatures, lock), or ``None``.

        It arrives alongside the samples once :meth:`start` has run, and while
        stopped after :meth:`request_telemetry`. ``host_stale`` is set when none has
        arrived for 500 ms.
        """
        return self._box.telemetry

    def request_telemetry(self, timeout_s: float = 1.0) -> bool:
        """Ask a stopped box for its :attr:`telemetry`: its lasers, temperatures and
        its own START button, without starting it. :meth:`start` does this too.

        Returns ``True`` once telemetry arrives (or already was), ``False`` if the
        box's firmware sends none.

        Raises
        ------
        RuntimeError or TimeoutError
            If the box doesn't answer. One started from its own button before
            :meth:`connect` doesn't, until it stops.
        """
        return self._box.request_telemetry(timeout_s)

    # -- State management -----------------------------------------------------

    def start(self, timeout_s: float = 4.0):
        """Turn the laser measurement system on. Does nothing if it is already on.

        Starts the stream if needed, confirms IDLE, selects optical diagnostics
        so :attr:`locked` and the lock calls can see the 638, then confirms
        START. This returns before the box has locked: follow it with
        :meth:`check_lock` or :meth:`wait_ready`.
        """
        self._box._ensure_connected()
        if not self._box._running:
            self._box.restart(timeout_s)

    def restart(self, timeout_s: float = 4.0):
        """Stop the laser measurement system if it is on, and start it again: as
        :meth:`start`, for a box that may be running."""
        self._box.restart(timeout_s)

    def stop(self, timeout_s: float = 1.0, *, end_recordings: bool = True) -> Optional[BoxFault]:
        """Turn off the laser measurement system and confirm it reached IDLE.

        A recording or monitor in progress ends first, unless ``end_recordings``
        is False: then it carries on, with the lasers off. The stream keeps
        running, so :attr:`buffer` stays live: :meth:`close` ends it.

        Returns the :class:`BoxFault` that had already stopped the box (the probe
        signal lost, say), or ``None``: this stop was the first, or the box's own
        button stopped it.

        Raises
        ------
        RuntimeError
            If the box could not be confirmed stopped. :attr:`running` stays
            ``True`` in that case, so :meth:`close` tries again.
        """
        if end_recordings:
            self._end_recordings()
        metrics = self._box.stop_system_confirmed(timeout_s=timeout_s)
        reason = metrics.fault
        return None if reason is None else BoxFault(int(metrics.fault_code), reason)

    def warm(self):
        """Standby: lasers off, temperatures held, for a faster :meth:`start`."""
        self._box._ensure_connected()
        self._box._send(CMD_WARM)

    def close(self):
        """Stop the box (confirmed), end the stream and release the USB device."""
        self._end_recordings()
        self._box.close()

    # -- Lock -----------------------------------------------------------------

    def wait_ready(self, timeout_s: float = 90.0) -> None:
        """Wait for a settled lock, without disturbing the signal.

        For when :meth:`check_lock`'s burst cannot be tolerated: this does not
        measure the gains. :class:`LockError` if no settled lock came in time.
        """
        _lock.wait_ready(self, timeout_s)

    def check_lock(self, retune: bool = False, *, progress=None, cancel=None, report_dir=None):
        """Check the lock gains suit this probe and this start. Returns a :class:`LockReport`.

        Do this after every START where possible, as the GUI does. ``retune=True``
        also corrects the gains when they need it, and stores them on the box.
        See :func:`ultracoustics.lock.check_lock`.
        """
        return _lock.check_lock(self, retune, progress=progress, cancel=cancel, report_dir=report_dir)

    def calibrate_lock(self, store: bool = True, *, progress=None, cancel=None, report_dir=None):
        """The full lock calibration: measure this probe's lock loop and design its
        gains, about two minutes, and a minute more to store them. Returns a
        :class:`LockReport`. See :func:`ultracoustics.lock.calibrate_lock`.
        """
        return _lock.calibrate_lock(self, store, progress=progress, cancel=cancel, report_dir=report_dir)

    @property
    def lock_gains(self):
        """The gains the 638 is running, as :class:`LockGains`. Asks the box."""
        return _lock.lock_gains(self)

    def restore_lock_gains(self):
        """Drop live gains, a calibration's that were not stored say, and go back to the
        box's defaults. Returns the :class:`LockGains`."""
        return _lock.restore_lock_gains(self)

    def clear_stored_lock_gains(self):
        """Remove the gains stored on the box, so its next START runs the factory
        gains. Needs the box stopped."""
        return _lock.clear_stored_lock_gains(self)

    # -- Checks of the box --------------------------------------------------------

    def self_test(self, probe=None, progress=None, cancel=None, probe_serial=None):
        """The self-test, as the GUI's Diagnostics › Self-check: about six minutes, and the
        box is left stopped. ``probe_serial`` is the probe's serial number, for the report.
        Returns a :class:`SelfTestReport`. See :func:`ultracoustics.selftest.run_self_test`.
        """
        from .selftest import run_self_test
        return run_self_test(self, probe=probe, progress=progress, cancel=cancel,
                             probe_serial=probe_serial)

    def measure_signal_level(self, progress=None, cancel=None):
        """Recommend the 1550 nm laser's setting for this probe, as the GUI's Diagnostics ›
        Adjust signal level. Changes nothing. Returns a :class:`SignalLevel`, or None if
        cancelled. See :mod:`ultracoustics.signal_level`.
        """
        from .signal_level import measure_signal_level
        return measure_signal_level(self, progress=progress, cancel=cancel)

    def store_signal_level(self, level, progress=None):
        """Store a :class:`SignalLevel`'s recommended setting and check the box comes back
        at it. Returns a :class:`StoredSignalLevel`.
        """
        from .signal_level import store_signal_level
        return store_signal_level(self, level, progress=progress)

    # -- Data capture ---------------------------------------------------------

    def capture(self, duration_s: float = 1.0, require_lock: bool = True) -> Recording:
        """The next *duration_s* seconds of samples, in memory, as a :class:`Recording`.

        Every sample is kept and in order, and the duration is not limited by
        the ring. For longer than memory comfortably holds use :meth:`record`.

        Raises
        ------
        ValueError
            If *duration_s* is not positive.
        CaptureError
            If any sample was lost or, with *require_lock*, the lock was not
            confirmed from start to finish.
        """
        count = int(round(duration_s * SAMPLE_RATE))
        if count <= 0:
            raise ValueError("duration_s must be positive")
        self._box.begin_stream()
        samples, started = capture_samples(self, count, SAMPLE_RATE, require_lock)
        return Recording("Capture", samples, SAMPLE_RATE, "recording", utc_start=started,
                         calibration=self.calibration, meta=self._recording_meta("capture"))

    def snapshot(self, duration_s: float = 1.0, meta: Optional[dict] = None) -> Recording:
        """What just happened: up to the newest *duration_s* seconds already in memory.

        :meth:`capture` takes what happens next. This one reaches back as far as
        the ring holds, so check the result's ``duration_s``. It needs no lock, and
        its metadata says whether there was one. *meta* replaces the SigMF fields.

        Raises
        ------
        RuntimeError
            If no data has arrived, or the ring overwrote it during the copy.
        """
        samples, _, taken = take_snapshot(self, duration_s, SAMPLE_RATE)
        return Recording("Snapshot", samples, SAMPLE_RATE, "recording", utc_start=taken,
                         calibration=self.calibration, meta=meta or self._recording_meta("snapshot"))

    def capture_dark(self):
        """The lasers-off reference :meth:`calibrate_pressure` needs, as a Spectrum.

        Needs the box stopped: run it before :meth:`start`. :class:`CaptureError`
        if the signal shows the lasers are not off.
        """
        if self.running:
            raise RuntimeError("capture_dark() needs the lasers off: call stop() first")
        self._box.stop_system_confirmed(timeout_s=4.0)
        time.sleep(DARK_SETTLE_S)
        samples = self.capture(nepcal.CAPTURE_SECONDS, require_lock=False).counts()
        if float(samples.mean()) >= DARK_MAX_COUNTS:
            raise CaptureError(f"the signal reads {float(samples.mean()):.0f} counts with the lasers off "
                               f"(a dark is below {DARK_MAX_COUNTS})")
        return nepcal.capture_spectrum(samples, SAMPLE_RATE)

    def calibrate_pressure(self, dark=None, probe=None):
        """Fit this START's thermal noise, keep it as :attr:`calibration` and return it.

        *dark* is from :meth:`capture_dark`; without it, the dark of the calibration
        in use. Keep the probe in quiet surroundings with a settled lock: this
        reads pressure off the dome's thermal motion.

        Raises
        ------
        CaptureError
            If there is no lock, or samples were lost.
        CalibrationError
            If there is no dark reference, no fit was possible, or something
            other than thermal noise was in it.
        """
        if dark is None and self.calibration is not None:
            dark = self.calibration.dark
        if dark is None:
            raise nepcal.CalibrationError("no dark reference: run capture_dark() before start()")
        time.sleep(EXCITATION_DECAY_S)
        samples = self.capture(nepcal.CAPTURE_SECONDS).counts()
        fit = nepcal.calibrate(nepcal.capture_spectrum(samples, SAMPLE_RATE), dark,
                               probe if probe is not None else nepcal.ProbeParameters(), SAMPLE_RATE)
        floor_db = nepcal.floor_over_noise_db(fit)
        if floor_db > nepcal.FIT_FLOOR_MAX_DB:
            raise nepcal.CalibrationError(f"the fit's noise floor is {floor_db:+.1f} dB over dark and shot noise, "
                                          "so more than thermal noise was in it. Quieten the probe and try again.")
        self.calibration = fit
        return fit

    def record(self, path, duration_s: Optional[float] = None, max_bytes: int = 0, wait: bool = True,
               require_lock: bool = True, meta: Optional[dict] = None, on_finished=None) -> ContinuousRecorder:
        """Write the stream to a SigMF recording at *path*, split into 2 GiB files.

        Ends at *duration_s* or *max_bytes*, whichever comes first, or at
        :meth:`stop`. Samples that could not be written in time are marked in
        the file as a gap and counted in the result's ``lost``. Samples lost
        before they reached the computer appear in its ``dropped``, and in the
        file's ``broadsonic:stream_drops_end``. Returns the recorder: its
        ``open()`` gives the :class:`Recording`, and ``samples``, ``lost``,
        ``dropped`` and ``reason`` say how it went. With ``wait=False`` it
        returns at once and runs on: ``stop()`` and ``join()`` end it. The lock
        is checked at the start only. *meta* replaces the SigMF fields, and
        ``on_finished(recorder)`` is called from the recorder's thread at the end.

        Raises
        ------
        FileExistsError
            If *path* already has a recording.
        CaptureError
            If *require_lock* and the lock is not confirmed.
        OSError
            If the recording could not be written (with ``wait=True``).
        """
        if wait and not duration_s and not max_bytes:
            raise ValueError("give duration_s or max_bytes, or use wait=False and stop() it yourself")
        base = new_recording_base(path)
        self._box.begin_stream()
        if require_lock and not self.locked:
            raise CaptureError("the lock is not confirmed: run start() and check_lock() or wait_ready() "
                               "first, or pass require_lock=False")
        def finished(recorder):
            self._recorders.discard(recorder)
            if on_finished is not None:
                on_finished(recorder)

        recorder = ContinuousRecorder(self, base, SAMPLE_RATE, meta or self._recording_meta("continuous"),
                                      max_seconds=duration_s or 0, max_bytes=max_bytes, on_finished=finished)
        self._recorders.add(recorder)
        recorder.start()
        if wait:
            try:
                while recorder.is_alive():
                    recorder.join(0.1)
            finally:
                recorder.stop()
                recorder.join()
            if recorder.error is not None:
                raise recorder.error
        return recorder

    def monitor(self, triggers, pre_s: float = 0.01, post_s: float = 0.1, holdoff_s: Optional[float] = None,
                folder=None, trend_s: float = 1.0, max_gb: Optional[float] = None, require_lock: bool = True,
                on_event=None, on_status=None, on_alarm=None, *, single: bool = False, sink=None,
                events_during_alarm: bool = False, meta: Optional[dict] = None, start: bool = True) -> Monitor:
        """Watch the stream for *triggers*, one or a list. Returns the running :class:`Monitor`.

        Each event keeps *pre_s* before the trigger and *post_s* after it, and a
        trigger stays off for *holdoff_s* after firing, *post_s* by default. With
        *folder* a new subfolder of it holds the events, ``trend.csv`` with a row
        every *trend_s*, ``alarms.csv`` and, at the end, ``summary.json``. Event
        files stop at *max_gb* in all: nothing is deleted, and monitoring goes on.

        ``on_event(event)``, ``on_status(text)`` and ``on_alarm(change)`` run on the
        monitor's thread. Call ``stop()`` on the monitor to end it, then ``summary()``
        says what was watched. With *require_lock* it watches only while the lock
        holds, so it can be started before the lock settles and says when it is
        not watching.

        Run :meth:`check_lock` first: its burst is refused while a monitor runs.
        *single*, *sink*, *events_during_alarm*, *meta* and *start* are what the GUI
        needs: see :class:`Monitor`.

        Raises
        ------
        ValueError
            If a trigger cannot work, or an event would not fit in the ring: then
            create the Controller with a larger ``ring_seconds``.
        """
        self._box.begin_stream()
        pre, post = int(round(pre_s * SAMPLE_RATE)), max(1, int(round(post_s * SAMPLE_RATE)))
        if pre + post > self.buffer_capacity - RING_GUARD_SAMPLES - int(MARGIN_SECONDS * SAMPLE_RATE):
            raise ValueError(f"an event of {pre_s + post_s:g} s does not fit in the ring: "
                             "create the Controller with a larger ring_seconds")
        holdoff = post if holdoff_s is None else int(round(holdoff_s * SAMPLE_RATE))

        def finished(monitor):
            self._monitors.discard(monitor)

        monitor = Monitor(self, SAMPLE_RATE, ADC_TO_POWER_UW, triggers, pre_samples=pre, post_samples=post,
                          holdoff_samples=holdoff, single=single, on_event=on_event, on_status=on_status,
                          on_alarm=on_alarm, on_finished=finished, sink=sink, calibration=self.calibration,
                          require_lock=require_lock, events_during_alarm=events_during_alarm)
        if folder is not None:
            run = Path(folder) / datetime.now().strftime("%Y%m%d_%H%M%S")
            run.mkdir(parents=True)
            fields = meta or dict(self._recording_meta("events"),
                                  **{"broadsonic:triggers": [t.to_meta() for t in monitor.watch.triggers]})
            monitor.folder = run
            monitor.sink = EventFolder(run, SAMPLE_RATE, fields, int((max_gb or 0) * 1e9))
            monitor.owns_sink = True
            monitor.trend = TrendLog(run, [t.label for t in monitor.watch.triggers], trend_s, SAMPLE_RATE)
        self._monitors.add(monitor)
        if start:
            monitor.start()
        return monitor

    def _stream_health(self):
        """The stream's loss counters now (the recorder compares them before and after)."""
        return self._box._stream_health()

    def _end_recordings(self):
        for recorder in list(self._recorders):
            recorder.stop()
            recorder.join()
        for monitor in list(self._monitors):
            monitor.stop(5.0)

    def _recording_meta(self, kind):
        """SigMF global fields for a recording made now."""
        return global_meta(kind, self, self.calibration)
