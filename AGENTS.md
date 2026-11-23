# AGENTS.md

For an AI coding assistant integrating the Ultracoustics SDK (`ultracoustics`,
Python 3.9+) into a user's code. The device is the BROADSONIC: an optical
ultrasound probe on a fiber and a readout box that streams one channel of
14-bit samples at 10 MS/s over USB. The box contains lasers. The desktop
program for it is the Ultracoustics GUI ("the GUI" below). Follow the hard
rules exactly. When this file and a docstring disagree, the docstring of the
installed version wins: check with `help(uc.Controller.start_ready)`.

## Canonical script

```python
import time
import ultracoustics as uc

def main():
    with uc.Controller() as ctrl:        # connects; on exit: confirmed stop, lasers off, USB released
        report = ctrl.start_ready()      # blocks about 30 s (a minute more if it corrects the box's settings), prints nothing; raises uc.StartError
        if not report.calibrated:        # ready, but no new pressure calibration this start: the message says why
            print(report.message)
        rec = ctrl.capture(1.0)          # uc.Recording; raises uc.CaptureError rather than return doubtful data
        name = "capture_" + time.strftime("%Y%m%d_%H%M%S")    # a new name each run: save() never overwrites
        print("saved", rec.save(name))   # name.sigmf-data + name.sigmf-meta; returns the .sigmf-meta path

if __name__ == "__main__":               # mandatory in any script that creates a Controller
    main()
```

Analysis of saved files needs no box, no `Controller` and no main guard:

```python
import ultracoustics as uc

rec = uc.load_recording("capture_20261116_141500.sigmf-meta")    # ValueError if it does not exist
spectrum = rec.spectrum(fft_size=16384)              # uc.Spectrum: .freqs (Hz), .psd_w2hz (W²/Hz)
pressure = rec.calibration.to_pressure(spectrum)     # uc.PressureSpectrum: .psd_pa2hz, .spl; 0 to 4.5 MHz only
level = pressure.band_level(20e3, 100e3).spl_db      # dB SPL in the band
```

## Hard rules

1. **Public API only.** Use names on `ultracoustics` (`uc.*`), methods and
   properties of `uc.Controller`, and `ultracoustics.updates`. Never import
   `ultracoustics._internal`. Never touch a name starting with an underscore
   (`ctrl._box`, ...). Never open the USB device or send raw commands. That code
   can drive lasers and overwrite firmware and stored settings without safety
   checks, and can leave the box needing service.
2. **Main guard.** All code that creates a `Controller` must run under
   `if __name__ == "__main__":`. The SDK spawns a reader process that re-imports
   the script. This applies to the entry script, including test runners and
   notebooks exported to scripts.
3. **One process per box, one box per computer.** Only one `Controller` in one
   program at a time. The GUI must be closed. Do not create a second
   `Controller`, and do not share one across processes. The SDK cannot choose
   between boxes: connect one box to a computer.
4. **Always release the box.** `uc.Controller()` itself sends nothing: the box
   is opened by `with uc.Controller() as ctrl:` (preferred) or `ctrl.connect()`,
   and the connection errors are raised there, not by the constructor.
   Without `with`: `ctrl = uc.Controller(); ctrl.connect()`, then the work in
   `try:` and `ctrl.close()` in `finally:`. As a backstop the SDK also turns
   the lasers off when Python exits without `with` or `close()`, and on a plain
   SIGTERM when the script installed no handler of its own. It cannot after a
   forced kill (Task Manager "End task", `kill -9`, `os._exit()`, a crash of
   Python, power loss): the lasers then stay on until the box's button or power
   switch is used. Do not rely on the backstop: it reports nothing if it fails.
   If `stop()` raises `RuntimeError` or `TimeoutError` (an `OSError`), or
   `close()` raises `RuntimeError`, the stop could not be confirmed: tell the
   user the lasers may still be on and to use the box's own button or power
   switch. `docs/how_it_works.md` has the table of exits.
5. **Order.** `Controller()` -> `start_ready()` -> capture / record / monitor ->
   exit. Nothing that needs a ready box works before `start_ready()` returns.
   `start_ready()`, `check_lock()`, `calibrate_lock()`, `self_test()`,
   `measure_signal_level()` and `store_signal_level()` are refused while a
   recording or a monitor is running: finish those first.
6. **Do not shorten the start-up by default.** Use `start_ready()`. Use
   `start_ready(quick=True)` only if the user asks for a fast start and accepts
   the cost: no checks, signal may drift for the first seconds, no new pressure
   calibration. Data is then in optical power only, or carries an earlier
   start's calibration that must not be trusted.
7. **During `start_ready()` and `self_test()`** the probe must be still and the
   surroundings quiet. Tell the user. Do not start the machine under test
   before `start_ready()` returns if it can be avoided.
8. **Never do these without the user explicitly asking, in the code path where
   it happens:**
   - `ctrl.store_signal_level(...)`: writes a setting kept on the box.
   - `ctrl.calibrate_lock()`, `ctrl.clear_stored_lock_gains()`,
     `ctrl.restore_lock_gains()`: change the box's control settings.
     (`start_ready()` may also correct and store those settings when its check
     finds they need it. That is its normal, intended behaviour.)
9. **Do not retry in a tight loop.** On `uc.StartError`, show the message and
   stop. It says what the user must do: run the self-test, or set the signal
   level (`err.signal_too_strong` is then `True`).
10. **Laser safety is the user's.** These calls turn lasers on:
    `start_ready()`, `start()`, `restart()`, `self_test()`,
    `measure_signal_level()`, `store_signal_level()` and `calibrate_lock()`.
    Never add code that calls one on import, at module level, or on a timer the
    user did not ask for.
11. **The self-test is the first step whenever the box does not behave as
    expected**: an error that says the box is not ready, and equally data that
    looks wrong with no error at all. `report = ctrl.self_test()`, print
    `report.text()`, save with `report.save(path)`: JSON, never overwrites, so
    put the date and time in the name. About six minutes. It turns the lasers
    on, stores nothing on the box, releases and reopens the USB, and leaves the
    box stopped. It starts with a stop that discards the box's fault: call
    `fault = ctrl.stop()` first if the fault matters. The GUI's menu calls it
    Self-check. Tell the user to run it before you guess at causes, and give
    any script you write for them a way to run it (a `selftest` argument, say).
    One exception: when a message says the signal is too strong, the fix is the
    signal level (`measure_signal_level()`, then `store_signal_level()` on the
    user's say-so), and `report.recommends_signal_adjustment` from a self-test
    says the same.

## Units

| Quantity | Unit | Source |
|---|---|---|
| Raw samples | ADC counts, `uint16`, 14 bits used, 0 to 16383 | `rec.counts()`, `ctrl.buffer` |
| Sample rate | 10 000 000 Hz | `rec.sample_rate` |
| Optical power | µW, `float64`; `counts * 0.008441` (exact factor: `rec.uw_per_count`) | `rec.power_uw()`, `uc.counts_to_uw(counts)` |
| Optical spectrum | W²/Hz, every whole FFT averaged | `rec.spectrum(fft_size).psd_w2hz` |
| Pressure spectrum | Pa²/Hz | `cal.to_pressure(spectrum).psd_pa2hz` |
| Density | dB SPL in 1 Hz | `pressure.spl` |
| Band level | dB SPL | `pressure.band_level(lo_hz, hi_hz).spl_db`, `.p_rms_pa` |
| Tone level | dB SPL | `uc.tone_level(rec, cal, freq_hz, bandwidth_hz=10).spl_db`; needs `1.5 / bandwidth_hz` seconds of data (0.15 s by default; `ValueError` on less) |

- dB SPL is re 20 µPa. Frequencies are Hz, times are seconds.
- Density, band level and tone level are different quantities. Do not compare
  one with another. A pure tone's density depends on FFT size: use `tone_level`.
- Pressure needs a calibration (`rec.calibration`, `ctrl.calibration`). It is
  `None` after `quick=True` with no earlier calibration: then only counts, µW
  and W²/Hz are available. Check for `None`.
- Pressure is valid 0 to 4.5 MHz. `to_pressure` returns only that band.
- The mean of `power_uw()` is the light on the detector, not sound. Sound is
  the fluctuation. The mean includes the detector's dark offset: with the
  lasers off it reads about 240 counts, about 2 µW. `power_uw(baseline=counts)`
  subtracts a level given in counts.
- `power_uw()` returns `float64`: 80 MB for each second of data. For a mean,
  use `rec.counts().mean() * rec.uw_per_count`, and read long recordings in
  stretches with `t0_s` and `t1_s`.
- A quiet spectrum always shows the sensor's resonance peaks, driven by thermal
  motion: around 2 to 3.5 MHz on the sample's probe. They are not a fault, and
  the calibration is read from them.
- A file is 20 MB per second (about 72 GB per hour). `record()` splits at 2 GiB.

## Errors

| Exception | Base | Means | Do |
|---|---|---|---|
| `RuntimeError("Master Board not found ...")` | | Raised by `with` / `connect()`: no BROADSONIC box on USB (not powered, or no data cable) | Tell the user: power, cable |
| `RuntimeError("USB reader process could not open the device ...")` or `("... exited during start-up ...")` | | Raised by `with` / `connect()`: usually another program (the GUI, another script) has the box, or the main guard is missing | Tell the user to close the GUI; check the main guard |
| `RuntimeError` or `TimeoutError` from `stop()` on a connected box; `RuntimeError("... shutdown was not confirmed ...")` from `close()` | | The stop could not be confirmed: the lasers may still be on | Tell the user to use the box's button or power switch |
| `RuntimeError` (other) | | "Not connected. Call connect() first."; `self_test`, `measure_signal_level` or `store_signal_level` called during a recording or monitor | Show the message |
| `uc.StartError` | `RuntimeError` | Every failure of `start_ready()`: the box did not become ready or did not answer a command (the lasers are then off, unless the message says the stop could not be confirmed), it was cancelled, or it was refused because a recording or monitor is running (the box is left as it was). `.signal_too_strong` (bool): the cause is the signal level. `.detail`: the technical reason, for support | Show the message. Do not loop. |
| `uc.CaptureError` | `RuntimeError` | Samples were lost, no samples arrived for 2 s, or the box was not ready. The "not ready" message reads: "the box is not ready to measure (the lock is not confirmed). Before a start: run ctrl.start_ready(). After one: the probe moved or the signal was lost. ctrl.stop() returns why if the box stopped itself; then start_ready() again, and ctrl.self_test() if it keeps happening." | Show the message. For lost data: no hub, less load, or use `record()` |
| `uc.CalibrationError` | `Exception` | No pressure calibration could be made (no dark, too noisy) | Only from `calibrate_pressure()`; `start_ready()` handles it and reports `calibrated=False` |
| `uc.LockError` | `RuntimeError` | From the advanced calls: the box was not running, not settled, or busy recording | See `docs/advanced_control.md` |
| `uc.SignalLevelError` | `RuntimeError` | No signal level setting suits this probe | Show the message; user contacts Ultracoustics |
| `uc.UpdateCheckError` | `Exception` | The update list could not be fetched (offline) | Not a fault of the box |
| `uc.FirmwareRefused` | `uc.FirmwareUpdateError` | Update refused before anything was written. Box unchanged. | Show the message |
| `uc.FirmwareUpdateError` | `Exception` | Update did not finish or could not be confirmed | Show the message; keep the box powered; user contacts Ultracoustics |
| `FileExistsError` | | A recording of that name exists | Choose a new name. Never delete for the user. |
| `ValueError` | | Bad argument: non-positive duration, an impossible trigger, an event longer than the ring, `record(wait=True)` with no `duration_s` or `max_bytes`, `subtract` on spectra of different `fft_size`, `tone_level` on too little data, `load_recording` on a missing file (not `FileNotFoundError`) | Fix the call |

"Lock" in an API name or message means the box holding itself in tune with the
sensor, which the user documentation calls "ready".

## Task -> call

| Task | Call |
|---|---|
| Start, full | `ctrl.start_ready()` -> `uc.StartReport(calibrated, message, seconds, lock)`. About 30 s; about a minute more when it corrects the box's control settings, which it then stores on the box. A start that fails usually raises within 90 s, or after a few minutes if it was correcting the settings. It prints nothing while it runs. `report.lock.ok` is `False` when the control settings could not be checked (`report.lock` is `None` after `quick=True`) |
| Start, fast, unchecked | `ctrl.start_ready(quick=True)`: a few seconds, gives up after 30 s |
| Progress bar / cancel during start | `ctrl.start_ready(progress=fn, cancel=fn)`; `fn(event)` gets a dict with `"stage"` and `"fraction"`; `cancel()` returns `True` to stop. An exception raised by either stops the box and is raised again as itself |
| Retry a failed pressure calibration | `ctrl.calibrate_pressure()` with no arguments: same dark, no restart. Raises `uc.CalibrationError` or `uc.CaptureError` |
| Stop lasers, keep connection | `fault = ctrl.stop()` -> `uc.BoxFault(code, reason)` or `None` |
| Standby between measurements | `ctrl.warm()`: lasers off, temperatures held, sent without waiting for an answer. Start again with `ctrl.start_ready()` or `ctrl.restart()`, not `ctrl.start()`, which does nothing when this script already started the box |
| Next N seconds in memory | `ctrl.capture(seconds)` -> `uc.Recording` |
| What just happened (up to about 1.0 s back) | `ctrl.snapshot(seconds)`; check `.duration_s`. For more, `uc.Controller(ring_seconds=...)`, at 20 MB of memory per second |
| Long recording to disk | `ctrl.record(path, duration_s=...)` -> recorder with `.samples`, `.lost`, `.dropped` (a dict), `.not_ready_s`, `.reason`, `.error`, `.open()` |
| Recording in the background | `r = ctrl.record(path, wait=False)`; later `r.stop(); r.join()`, then check `r.error` |
| Load a file | `uc.load_recording(path)` (lazy; does not read samples into memory) |
| Samples of part of a file | `rec.counts(t0_s, t1_s)`, `rec.power_uw(t0_s, t1_s)` |
| Spectrum | `rec.spectrum(fft_size=16384, t0_s=None, t1_s=None)` |
| Background subtraction | `measured.subtract(background)`; same `fft_size`; then `.significant(sigma=3)` if wanted. Negative bins are kept on purpose: do not clip them. A band at or below zero gives `spl_db = -inf`. `subtract` and `to_pressure` emit a Python warning when the optical power of the two differs |
| Band level in optical units | No call. `psd_w2hz` is one-sided: `sel = (s.freqs >= lo) & (s.freqs <= hi)`; `w2 = s.psd_w2hz[sel].sum() * (s.freqs[1] - s.freqs[0])`; µW rms is `sqrt(w2) * 1e6` |
| Change between two spectra, dB | `now.difference_db(healthy)` |
| Filter a waveform | `uc.band_pass(x, (lo, hi), fs)`, `uc.notch(x, freq_hz, fs)` |
| Waveform in Pa | `cal.pressure_waveform(rec.power_uw(), band=(lo, hi), sample_rate=fs)` |
| What limits the noise floor | `cal.noise_budget(spectrum).limited_by_at(freq_hz)` |
| Watch for events live | `ctrl.monitor(triggers, pre_s=0.01, post_s=0.1, folder=..., max_gb=..., on_event=fn, on_alarm=fn, on_status=fn)` -> `uc.Monitor`, returns at once. `pre_s + post_s` must stay under about 1.0 s unless `ring_seconds` is raised. A dated subfolder of `folder` gets the events, the level log and the alarms. `.stop()`, `.summary()` |
| Triggers | `uc.BandPowerTrigger((lo, hi), threshold_db=10, name=...)` learns normal for 1 s when armed, then follows slow drift, so a rise that lasts becomes the new normal. Add `baseline=spectrum` (a `uc.Spectrum` taken when healthy) to hold normal fixed. `uc.LevelTrigger(level_uw=..., name=...)` or `level_pa=...` with `band_hz=(lo, hi)`. See `help()` on each |
| Same triggers on a file | `uc.find_events(rec, triggers)`. Returns every firing, as a live monitor does with `events_during_alarm=True`. A live monitor by default keeps one event per alarm, so it gives fewer |
| Live samples for a display | `ctrl.buffer_head` then slice `ctrl.buffer` (do not write to it), or `uc.LiveSpectrum` |
| State of box and stream | `ctrl.status` -> `uc.Status(connected, streaming, running, locked, recording, monitoring, samples_received, stream_drops)`. `running` means started by this program and not yet stopped by it, not that the lasers are on now. `locked` is the live "ready" flag. Also `ctrl.telemetry`, `ctrl.stream_stats` |
| Diagnose anything | `ctrl.self_test(probe_serial=...)` -> `uc.SelfTestReport`: `.ok`, `.problems`, `.text()`, `.save(path)` (JSON; never overwrites). Lasers on, about six minutes, box left stopped |
| Signal too strong | `level = ctrl.measure_signal_level()`: sweeps the laser for a few minutes with the lasers on, stores nothing, leaves the box stopped, returns `None` if cancelled, raises `uc.SignalLevelError` if no setting fits. Show `level.describe()`. Only on the user's say-so: `stored = ctrl.store_signal_level(level)`, check `stored.verified`. Then `ctrl.start_ready()` again |
| Check for updates | `uc.check_for_updates()` with no `Controller` open -> `.describe()`. It opens the box itself over USB: ask first |
| Install firmware | `uc.update_firmware()`: opens and stops the box itself, needs the main guard, and only on the user's say-so. This release carries no approval key yet, so it refuses every release with `uc.FirmwareRefused` |

## Behaviour to rely on

- `capture()` returns complete, in-order data or raises. It never returns
  partial data.
- `record()` checks that the box is ready only when it starts
  (`uc.CaptureError` if not). After that it writes whatever arrives: if the box
  stops itself part-way, the file holds lasers-off data with `lost == 0`. That
  time is counted in `recorder.not_ready_s`, and the file has a "not ready"
  annotation where each such stretch begins. It does not raise on loss: check
  `recorder.lost` (samples), `recorder.dropped` (a dict, empty when clean) and
  `recorder.not_ready_s` after every recording. Gaps are marked in the file;
  `rec.gaps` lists them.
- `record()` stops if no samples arrive for 2 s (cable pulled): with
  `wait=True` the call raises `uc.CaptureError`; with `wait=False` it is in
  `recorder.error`, and `recorder.reason` says so.
- `record()` raises `FileExistsError` before it starts when `name` or
  `name_001` exists. Split parts never overwrite a file.
- For anything that fits in memory, prefer `ctrl.capture(seconds).save(path)`,
  which checks readiness throughout.
- `monitor()` pauses while the box is not ready and counts that time:
  `summary().not_monitored_s`, `summary().complete`. Time with no samples at
  all (cable pulled) counts there too, so `complete` is then `False`.
  `on_status` gets "No samples are arriving from the box: not monitoring. Check
  the cable." and later "Samples are arriving again: monitoring". A summary
  file that cannot be written is reported in `summary().error`.
- `on_alarm` gets `uc.AlarmChange(trigger, state, utc, level, limit)`, on both
  raise and clear. While an alarm stays raised its trigger saves no further
  events: one event per episode. Events are short (`pre_s + post_s`).
- `ctrl.calibration` is not cleared by `stop()`. A later `start_ready()`
  replaces it on success. If that start's calibration fails
  (`report.calibrated` is `False`), or after `quick=True`, the earlier start's
  calibration stays attached to new data and is saved with it:
  `report.message` says so, and says not to trust pressure values from it.
  Treat `not report.calibrated` as an error when pressure matters, or retry
  with `ctrl.calibrate_pressure()`. On a script's first start there is no
  earlier calibration: data is then in optical power only.
- `on_event`, `on_alarm` and `on_status` run on the monitor's thread. Keep them
  short, and catch every exception inside them: one that escapes ends the
  monitor (see `summary().error`). `progress` and `cancel` of `start_ready()`
  are called on your own thread, inside the call.
- `start_ready()` on a running box stops it first and starts again.
  `start_ready(quick=True)` leaves alone a box that is running and ready, and
  restarts one that is running but not ready (it stopped itself, or is in
  standby after `warm()`).
- Every failure of `start_ready()` is `uc.StartError`, with the lasers turned
  off. If that stop could not be confirmed, the message says the lasers may
  still be on. Ctrl-C during it turns the lasers off too.
- The pressure calibration belongs to one probe at one start. Do not copy a
  calibration between probes or sessions.
- A background (source off) can only be taken by the user's procedure. If the
  user's source can be switched off, capture a background in the same start and
  subtract spectra. Offer this; it is the main thing that makes data usable.

## Long and unattended runs

- For periodic logging, start once and stay started: every `start_ready()`
  needs a still probe and quiet surroundings and makes a new calibration.
- Ctrl-C is safe: `KeyboardInterrupt` reaches your code, and leaving the `with`
  block does a confirmed stop.
- A plain `kill` (SIGTERM, `docker stop`, a service manager) also turns the
  lasers off, when the script installed no SIGTERM handler of its own. A forced
  kill (`kill -9`, Task Manager "End task", a crash of Python, power loss)
  cannot: the lasers stay on until the box's button or power switch is used.
  Tell the user this before setting up an unattended run.
- `stop()` can raise `RuntimeError` or `TimeoutError` (an `OSError`), and
  `close()` `RuntimeError`, when the stop could not be confirmed: catch both
  in a long run, and tell the user the lasers may still be on.
- If the box's answer to START is lost, leaving the `with` block still sends
  STOP.
- If the box stops itself (the probe signal lost, say), `ctrl.running` stays
  `True`, `ctrl.locked` goes `False`, `capture()` raises `uc.CaptureError` and a
  monitor pauses. `fault = ctrl.stop()` returns the `uc.BoxFault(code, reason)`:
  log it. `start_ready()` may then be called again on the same `Controller`,
  once and not in a loop. It needs quiet again.
- If the cable drops, `ctrl.streaming` goes `False` and `ctrl.stream_stats`
  says why. Leave the `with` block and tell the user. Do not reconnect in a loop.

## Do not run on the user's hardware unasked

Anything that creates a `Controller` turns on real hardware. When testing your
integration, prefer the offline path: `examples/data/sample.sigmf-meta` is a
real recording (0.1 s long) with its calibration, and `uc.Recording(name, samples,
sample_rate, "recording", calibration=cal)` wraps your own array of counts.
Ask before running code that starts the box.

## Where things are

| Topic | File |
|---|---|
| First page: what it is, the first script | `README.md` |
| Everything to know before relying on the data | `docs/essentials.md` |
| Step-by-step first use, glossary | `docs/getting_started.md` |
| Box vs computer, USB stream (informational), start-up stages, the lasers at each kind of exit | `docs/how_it_works.md` |
| Units, spectra, pressure, the three dB SPL numbers, noise removal | `docs/analysis_guide.md` |
| File format (SigMF), metadata fields, gaps, events | `docs/saving_data.md` |
| Triggers, alarms, run folder, limits | `docs/monitoring.md` |
| Separate start-up calls, lock check and calibration, signal level, changing probes | `docs/advanced_control.md` |
| Every public name | `docs/reference.md` |
| Updates and firmware | `docs/UPDATES.md` |
| Runnable code | `examples/` (`analyze_recording.py`, `remove_background.py`, `find_events.py` need no box; `capture.py`, `monitor.py` do) |
