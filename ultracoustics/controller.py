"""
The :class:`Controller`: the box, from connection to capture, recording, monitoring and the lock.

The USB stream is read by a process of its own into a shared-memory ring, so
nothing is lost while your script is busy. That process imports your script
when it starts: keep a script's work under ``if __name__ == "__main__":``.

Typical usage::

    from ultracoustics import Controller

    with Controller() as ctrl:              # close() always runs, with a confirmed stop
        ctrl.start_ready()                  # lasers on, ready and calibrated: about 30 s
        recording = ctrl.capture(1.0)       # checked for lost samples
        recording.save("capture")           # SigMF
"""

from dataclasses import dataclass
import atexit
from datetime import datetime
import signal
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

from . import lock as _lock
from ._internal.device import Box
from ._internal.protocol import CMD_WARM
from .analysis import calibration as nepcal
from .config import ADC_MAX_VALUE, RING_GUARD_SAMPLES, SAMPLE_RATE
from .monitor import EventFolder, Monitor, TrendLog
from .readout import ADC_TO_POWER_UW, CLIP_FRACTION, DARK_MAX_COUNTS, DARK_SETTLE_S
from .recorder import CaptureError, ContinuousRecorder, MARGIN_SECONDS, capture_samples, take_snapshot
from .recording import Recording, global_meta, new_recording_base



# After a lock check the excitation needs this long to die away before a fit.
EXCITATION_DECAY_S = 0.5
# How long start_ready(quick=True) waits for the box to report it is ready.
QUICK_START_TIMEOUT_S = 30.0
# A box that is running but not yet ready gets this long before the quick start restarts it.
QUICK_SETTLE_GRACE_S = 5.0


#: Connected Controllers, held until close(): the exit hook must still find one its script let go of
_open = set()


def _close_at_exit():
    """At interpreter exit: stop every box its script left running (no ``with``, no close())."""
    for controller in list(_open):
        try:
            controller.close()
        except Exception:
            pass


def _sigterm(*_):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)    # a second one must not cut the stop short
    sys.exit(143)


def _exit_on_sigterm():
    """Make a plain ``kill`` (SIGTERM: a service manager, docker stop) end the script as
    sys.exit() does, so the lasers are turned off. Only from the main thread, and only when
    the script set no handler itself."""
    if (threading.current_thread() is threading.main_thread()
            and signal.getsignal(signal.SIGTERM) is signal.SIG_DFL):
        signal.signal(signal.SIGTERM, _sigterm)


atexit.register(_close_at_exit)


class StartError(RuntimeError):
    """:meth:`Controller.start_ready` could not get the box ready to measure. The
    message says what to do. The lasers are off again, unless it says otherwise.

    ``signal_too_strong`` is True when the cause is the signal level for this probe.
    ``detail`` is the technical reason, for support.
    """
    signal_too_strong = False
    detail = ""


@dataclass(frozen=True)
class StartReport:
    """What :meth:`Controller.start_ready` did. Having one means the box is ready to measure."""
    calibrated: bool          # this start's pressure calibration succeeded
    message: str
    seconds: float            # how long the start took
    lock: object = None       # the LockReport of the check; None after quick=True


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
                'snapshot', 'start', 'start_ready', 'status', 'stop', 'store_signal_level', 'stream_stats', 'streaming',
                'telemetry', 'verbose', 'wait_ready', 'warm')


class Controller:
    """
    The Ultracoustics box: one per box, from one script at a time.

    A script that creates one must keep its work under
    ``if __name__ == "__main__":``: the USB reader is a second process, which
    imports the script when it starts.

    Usage::

        with Controller() as ctrl:
            ctrl.start_ready()              # lasers on, ready and calibrated
            recording = ctrl.capture(1.0)
            ctrl.record("long_run", duration_s=60)

    For a live display, ``buffer`` is the ring itself, without a copy: read
    ``buffer_head`` first, then slice ``buffer``. ``samples_received`` counts
    every sample since the stream began. :meth:`snapshot` copies the newest
    samples out as a :class:`Recording`.
    """

    #: The class of the private box under it
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
            A monitor's event (``pre_s + post_s``) and :meth:`snapshot` can use
            the ring less 0.2 s: about 1.0 s by default. Each second takes 20 MB
            of memory.
        packet_diagnostics_path : str or None, optional
            For support: the reader writes a log of USB packet headers here.
        packet_diagnostics_attach_s : float, optional
            For support: how long after the stream starts to log every header.
        """
        self._box = self._box_class(verbose, int(SAMPLE_RATE * ring_seconds), packet_diagnostics_path,
                                    packet_diagnostics_attach_s)
        #: The Calibration that captures and recordings keep with their data.
        self.calibration = None
        self._dark = None        # the lasers-off reference of the last start_ready()
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
        _open.add(self)              # lasers off however the script ends, short of being killed outright
        _exit_on_sigterm()

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
        """The stream's packet counters, as a dict: ``packets``, ``drops_seq``,
        ``drops_fw``, ``transfer_errors``, ``transfer_timeouts`` and ``malformed``
        among them. ``None`` when not connected."""
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
        START. This returns before the box is ready to measure: most scripts
        want :meth:`start_ready`, which does the whole start-up. After this one,
        follow with :meth:`check_lock` or :meth:`wait_ready`.
        """
        self._box._ensure_connected()
        if not self._box._running:
            self._box.restart(timeout_s)

    def start_ready(self, quick: bool = False, *, progress=None, cancel=None) -> StartReport:
        """Turn the box on and get it ready to measure, in one call. Returns a :class:`StartReport`.

        This is what START does in the GUI, and takes about 30 seconds: the
        lasers-off (dark) reference, lasers on, the wait for the box to settle on
        the sensor, a check of its control settings (corrected and stored on the box
        when they need it, which adds about a minute), then this start's pressure
        calibration. Keep the probe still and the surroundings quiet until it returns.
        A start that fails usually says so within 90 seconds, or a few minutes if it
        was correcting the settings.

        If only the pressure calibration fails (surroundings too noisy, say), the box
        is still ready: ``report.calibrated`` is False and ``report.message`` says
        why. An earlier start's calibration, if this Controller has one, then stays
        attached to new data: do not trust pressure values from it.
        ``report.lock.ok`` is False when the control settings could not be checked.

        ``quick=True`` skips every check: lasers on, then wait only until the box
        reports it is on the sensor, a few seconds (it gives up after 30). Nothing
        is measured or corrected, the signal may still drift for the first seconds,
        and there is no new pressure calibration. A box that is running and ready
        is left as it is.

        ``progress(event)`` is called with ``{"stage", "fraction", ...}`` as it goes,
        and ``cancel()`` returning True stops it. An exception raised by either
        stops the start, turns the lasers off and is raised again.

        Raises
        ------
        StartError
            If the box did not become ready, with what to do next. The lasers are
            turned off first; the message says so if that could not be confirmed.
            ``signal_too_strong`` is True when the cause is the signal level, and
            ``detail`` holds the technical reason, for support. Refused, with the
            box left as it was, while a recording or monitor is running.
        """
        began = time.monotonic()
        self._box._ensure_connected()
        if self._recorders or self._monitors:
            raise StartError("starting disturbs the signal: finish the recording or monitor first")
        callers, strong = [], []     # an exception from the caller's progress or cancel; a saturated detector

        def cancelled():
            if not callers and cancel is not None:
                try:
                    return bool(cancel())
                except Exception as exc:
                    callers.append(exc)
            return bool(callers)

        def watch(event):
            if not strong and self._detector_saturated():
                strong.append(True)
            if progress is not None and not callers:
                try:
                    progress(event)
                except Exception as exc:
                    callers.append(exc)

        try:
            try:
                report = (self._start_quick(began, cancelled) if quick
                          else self._start_checked(began, watch, cancelled, strong))
            except StartError:
                raise
            except (RuntimeError, OSError) as exc:     # a command the box did not answer, say
                self._start_failed("the box could not be started: it did not answer a command. Check its power and "
                                   "the USB cable, then run ctrl.self_test(): its report says what is wrong.",
                                   detail=f"{type(exc).__name__}: {exc}")
            except BaseException:                      # Ctrl-C among them: never leave the lasers on
                self._warn_if_on(self._stop_quietly())
                raise
        except StartError:
            if callers:
                raise callers[0]
            raise
        if callers:
            self._warn_if_on(self._stop_quietly())
            raise callers[0]
        return report

    @staticmethod
    def _warn_if_on(not_stopped):
        """When another exception is on its way out, the lasers-on warning still has to be seen."""
        if not_stopped:
            warnings.warn(not_stopped.strip(), RuntimeWarning, stacklevel=3)

    def _start_quick(self, began, cancelled):
        self.start()
        settling = time.monotonic() + QUICK_SETTLE_GRACE_S
        while not self.locked and time.monotonic() < settling and not cancelled():
            time.sleep(0.1)
        if not self.locked and not cancelled():    # it had stopped itself, or was in standby: start() did nothing
            self.restart()
        while not self.locked:
            if cancelled():
                self._start_failed("start-up cancelled.")
            if time.monotonic() - began > QUICK_START_TIMEOUT_S:
                strong = self._detector_saturated()
                self._start_failed(self._not_ready(f"it was not ready after {QUICK_START_TIMEOUT_S:g} s", strong),
                                   strong)
            time.sleep(0.1)
        return StartReport(False, "ready, without checks. " + self._units_note(), time.monotonic() - began)

    def _start_checked(self, began, watch, cancelled, strong):
        self.stop(timeout_s=4.0)
        dark, no_dark = None, ""
        try:
            dark = self.capture_dark()
        except (RuntimeError, OSError, nepcal.CalibrationError) as exc:    # none: an earlier calibration's will do
            no_dark = str(exc)
        self._dark = dark
        self.start()
        lock = self.check_lock(retune=True, progress=watch, cancel=cancelled)
        if not lock.ready:
            if lock.status == "cancelled":
                self._start_failed("start-up cancelled.")
            why = ("it did not settle on the sensor" if lock.status == "not_ready"
                   else "its control settings could not be set")
            self._start_failed(self._not_ready(why, bool(strong)), bool(strong) or self._detector_saturated(),
                               lock.message)
        ready = "ready" if lock.ok else ("ready (its control settings could not be checked this start: "
                                         "the surroundings may be too noisy)")
        if dark is None and self.calibration is None:
            return StartReport(False, f"{ready}, but with no pressure calibration: the lasers-off reference could "
                               f"not be taken ({no_dark}). " + self._units_note(), time.monotonic() - began, lock)
        try:
            self.calibrate_pressure(dark)
        except (CaptureError, nepcal.CalibrationError) as exc:
            if not self.locked:
                self._start_failed(self._not_ready("it stopped being ready during the pressure calibration"),
                                   detail=str(exc))
            return StartReport(False, f"{ready}, but this start's pressure calibration did not succeed ({exc}). "
                               + self._units_note() + " To try again without restarting, keep the probe still "
                               "and the surroundings quiet, then call ctrl.calibrate_pressure().",
                               time.monotonic() - began, lock)
        return StartReport(True, f"{ready}, and calibrated for sound pressure", time.monotonic() - began, lock)

    def _units_note(self):
        return ("An earlier start's calibration is still attached to new data: do not trust pressure values from it."
                if self.calibration is not None else "Data is in optical power only, without sound pressure.")

    def _detector_saturated(self):
        """The newest 100 ms sits at the top of the detector's range."""
        try:
            counts = self.snapshot(0.1).counts()
        except RuntimeError:
            return False
        return float(counts.mean()) >= CLIP_FRACTION * ADC_MAX_VALUE

    def _not_ready(self, why, saturated=False):
        if saturated or self._detector_saturated():
            return ("the signal is too strong for the detector, so the box cannot get ready. The signal level needs "
                    "setting for this probe, once: in the GUI, Diagnostics > Adjust signal level; in a script, "
                    "level = ctrl.measure_signal_level(), then ctrl.store_signal_level(level), which changes a "
                    "setting kept on the box. Then start again.")
        return (f"the box did not become ready: {why}. Run ctrl.self_test(): its report says what is wrong. "
                "A probe that is unplugged or moving is the commonest cause.")

    def _stop_quietly(self):
        """Lasers off after a failed start. Returns what to tell the user if that could not be confirmed."""
        try:
            self.stop(timeout_s=4.0)
        except (RuntimeError, OSError) as exc:    # close() tries again
            return (f" The box could not be confirmed stopped ({exc}): the lasers may still be on. "
                    "Use the box's button or power switch.")
        return ""

    def _start_failed(self, message, signal_too_strong=False, detail=""):
        error = StartError(message + self._stop_quietly())
        error.signal_too_strong, error.detail = signal_too_strong, detail
        raise error

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
        """Standby: lasers off, temperatures held, for a faster next start.

        Sent without waiting for the box's answer. Start again with :meth:`restart`
        or :meth:`start_ready`: :meth:`start` does nothing when this Controller has
        already started the box.
        """
        self._box._ensure_connected()
        self._box._send(CMD_WARM)

    def close(self):
        """Stop the box (confirmed), end the stream and release the USB device."""
        self._end_recordings()
        try:
            self._box.close()
        finally:
            _open.discard(self)

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
        gains, about two minutes, and about a minute more to store them. Returns a
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
        """The self-test, as the GUI's Diagnostics > Self-check: about six minutes, and the
        box is left stopped. Returns a :class:`SelfTestReport`.

        Keep the probe still and the room quiet. Nothing may be recording or
        monitoring.

        Parameters
        ----------
        probe : ProbeParameters, optional
            The probe's constants, if Ultracoustics gave you some for this probe.
            Leave it out to use those of the calibration in use, or the standard ones.
        progress : callable, optional
            Called with a dict as each step runs and when it is done, for a display.
        cancel : callable, optional
            Return True from it to stop the test after the current step.
        probe_serial : str, optional
            The serial number printed on the probe. It is only written into the
            report, so support can tell which probe was tested.
        """
        from .selftest import run_self_test
        return run_self_test(self, probe=probe, progress=progress, cancel=cancel,
                             probe_serial=probe_serial)

    def measure_signal_level(self, progress=None, cancel=None):
        """Recommend the 1550 nm laser's setting for this probe, as the GUI's Diagnostics >
        Adjust signal level. Returns a :class:`SignalLevel`, or None if cancelled.

        It takes a few minutes, with the lasers on, and leaves the box stopped: call
        :meth:`start_ready` again afterwards. It changes nothing on the box. Nothing
        may be recording or monitoring. :class:`SignalLevelError` if no setting fits
        this probe. See :mod:`ultracoustics.signal_level`.
        """
        from .signal_level import measure_signal_level
        return measure_signal_level(self, progress=progress, cancel=cancel)

    def store_signal_level(self, level, progress=None):
        """Store a :class:`SignalLevel`'s recommended setting and check the box comes back
        at it. Returns a :class:`StoredSignalLevel`: check its ``verified``.

        This changes a setting kept on the box: do it by hand, once for a probe, not
        automatically in a measurement script. It takes a few minutes, with the
        lasers on, cannot be cancelled, and leaves the box stopped: call
        :meth:`start_ready` again afterwards. Nothing may be recording or monitoring.
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
            If any sample was lost, no samples arrived for 2 s (the stream
            stopped) or, with *require_lock*, the lock was not confirmed from
            start to finish.
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
        reads pressure off the sensor's thermal motion.

        Raises
        ------
        CaptureError
            If there is no lock, or samples were lost.
        CalibrationError
            If there is no dark reference, no fit was possible, or something
            other than thermal noise was in it.
        """
        if dark is None:
            dark = self._dark
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
        ``dropped``, ``not_ready_s`` and ``reason`` say how it went. With
        ``wait=False`` it returns at once and runs on: ``stop()`` and ``join()``
        end it, and its ``error`` holds what ``wait=True`` would have raised.

        The lock is required at the start only. After that the recording carries
        on if the lock is lost: ``not_ready_s`` counts the seconds recorded while
        the box was not ready, and the file has a ``not ready`` annotation where
        each such stretch begins. If no samples arrive for 2 s the recording stops.
        *meta* replaces the SigMF fields, and ``on_finished(recorder)`` is called
        from the recorder's thread at the end.

        Raises
        ------
        ValueError
            With ``wait=True`` and neither *duration_s* nor *max_bytes*.
        FileExistsError
            If *path*, or its first split part ``path_001``, already has a
            recording. Split parts never overwrite a file.
        CaptureError
            If *require_lock* and the lock is not confirmed at the start, or
            (with ``wait=True``) no samples arrived for 2 s.
        OSError
            If the recording could not be written (with ``wait=True``).
        """
        if wait and not duration_s and not max_bytes:
            raise ValueError("give duration_s or max_bytes, or use wait=False and stop() it yourself")
        base = new_recording_base(path)
        self._box.begin_stream()
        if require_lock and not self.locked:
            raise CaptureError("the box is not ready to measure (the lock is not confirmed). Before a start: run "
                               "ctrl.start_ready(). After one: the probe moved or the signal was lost. ctrl.stop() "
                               "returns why if the box stopped itself; then start_ready() again, and "
                               "ctrl.self_test() if it keeps happening.")
        def finished(recorder):
            self._recorders.discard(recorder)
            if on_finished is not None:
                on_finished(recorder)

        recorder = ContinuousRecorder(self, base, SAMPLE_RATE, meta or self._recording_meta("continuous"),
                                      max_seconds=duration_s or 0, max_bytes=max_bytes, on_finished=finished,
                                      require_lock=require_lock)
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
        not watching. Time with no samples arriving at all counts as not watched too.

        Run :meth:`start_ready` first: it is refused while a monitor runs.
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
