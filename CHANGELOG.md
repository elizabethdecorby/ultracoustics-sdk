# Changelog

## 2.0.1

- **One call to start.** `Controller.start_ready()` does the whole start-up,
  as START in the GUI: the lasers-off dark reference, lasers on, the wait for
  the box to be ready, the check of its control settings and this start's
  pressure calibration. It returns a `StartReport` and raises `StartError`,
  with the lasers off again, if the box does not become ready.
  `start_ready(quick=True)` is ready in a few seconds, with no checks and no
  new pressure calibration. The separate calls (`capture_dark()`, `start()`,
  `check_lock()`, `wait_ready()`, `calibrate_pressure()`) are unchanged.
- **Start failures.** Every failure of `start_ready()` is a `StartError`,
  including a command the box did not answer, with the lasers turned off. If
  that stop could not be confirmed, the message says the lasers may still be
  on. `StartError.signal_too_strong` says when the cause is the signal level,
  and `StartError.detail` holds the technical reason, for support. An
  exception raised by the caller's own `progress` or `cancel` stops the box and
  is raised again as itself. Ctrl-C during the call turns the lasers off.
- **Start report.** `report.message` says when the control settings could not
  be checked (`report.lock.ok` is `False`). When this start's pressure
  calibration fails, it says whether an earlier start's calibration is still
  attached to new data, which must not be trusted, and that
  `calibrate_pressure()` with no arguments tries again on the same dark
  reference without restarting.
- **Quick start.** `start_ready(quick=True)` leaves alone a box that is running
  and ready. One that is running but not ready, because it stopped itself or
  is in standby after `warm()`, is restarted.
- **Lasers off at exit.** The SDK turns the lasers off when Python exits, even
  without `with` or `close()`, and on a plain SIGTERM when the script installed
  no handler of its own. It cannot after a forced kill (Task Manager "End
  task", `kill -9`, power loss, a crash of Python): the lasers then stay on
  until the box's button or power switch is used. If the box's answer to START
  is lost, leaving `with` still sends STOP. A stop that could not be confirmed
  at `close()` says the lasers may still be on.
- **Recording.** `record()` stops with `CaptureError` if no samples arrive for
  2 s: raised with `wait=True`, kept in the recorder's `error` with
  `wait=False`. The recorder's `not_ready_s` counts the seconds recorded while
  the box was not ready, and the file has a "not ready" annotation where each
  such stretch begins. A recording refuses to start, with `FileExistsError`,
  when `name_001` exists, and split parts never overwrite a file.
- **Monitoring.** Time with no samples arriving at all now counts in
  `not_monitored_s`, so `complete` is then `False`, and `on_status` says "No
  samples are arriving from the box: not monitoring. Check the cable." and
  "Samples are arriving again: monitoring". A summary that cannot be written
  no longer blocks the controller: the summary's `error` says so.
- **Documentation.** The README is now the one page with everything a user must
  know: the minimal script, the dark and background references, units, and the
  self-test as the first step when something is wrong. New pages:
  [How it works](docs/how_it_works.md) (the box, the USB stream and the
  computer, with diagrams, and the lasers at each kind of exit),
  [Advanced control](docs/advanced_control.md) (the start-up one stage at a
  time, the lock check and calibration, the signal level, changing probes) and
  [AGENTS.md](AGENTS.md) for AI coding assistants. The snippets and
  `examples/capture.py` save under names with the date and time, so they run
  more than once. This release carries no firmware approval key yet, so
  `update_firmware()` refuses every release: see [Updates](docs/UPDATES.md).
- **Words.** The main pages say the box is "ready" where the API says "lock",
  and "sensor" for the sensing element in the probe tip, in the documentation,
  the help text and the messages. No name in the API has changed.
- **Messages.** A capture or recording refused because the box is not ready now
  says to run `start_ready()` before a start, and after one that the probe
  moved or the signal was lost, that `stop()` returns why, and to run the
  self-test if it keeps happening. Start-up messages no longer use internal
  words. "Master Board not found" now ends "no BROADSONIC box on USB. Check its
  power and that the cable carries data."
- **Self-test.** The steps run in the order the box works in, with a new one,
  "Light back from the microphone", before the 638 nm laser starts. A step
  that relies on something an earlier step found broken does not run, and
  `SelfTestReport.cause` names the first thing that really broke, with advice
  for the user. Both TECs must reach their target in the checks, and the data
  rate and stream losses are checked with the lasers off.
- **Why a START did not lock.** The lock step keeps the 638's acquisition
  pages and the detector's trace across START and reports, per attempt, where
  the dip fell against the search. A lock that landed far from where its
  search ended, or close to missing, is flagged as a near miss.
- The lock step gives up 5 s after the 638 reports an error, or 10 s after it
  falls silent, instead of waiting 100 s, and a lock lost while it is watched
  counts as lost.
- `recommends_signal_adjustment` is also true when the detector is near full
  scale with the 638 nm laser off.

## 2.0.0

The first release for customers.

- **Capture.** `Controller` captures to memory with `capture()`, takes what just
  happened with `snapshot()`, and records to disk with `record()`, all as
  SigMF recordings that carry their calibration and say how they were made.
  The stream starts at `connect()`; `stop()` says why, if the box had stopped
  itself, and `request_telemetry()` reads a stopped box's lasers and
  temperatures.
- **Pressure.** `calibrate_pressure()` calibrates each START from the probe's
  thermal noise. Spectra convert to sound pressure, with density, band level,
  octave bands and tone level kept apart.
- **Noise.** Measured background subtraction with significance, the fitted
  noise budget, band-pass and notch filters.
- **Monitoring.** `monitor()` watches the live stream with level and
  band-power triggers, healthy baselines, limits in pressure, alarms, a trend
  log and lock gating. `find_events()` replays the same triggers on saved data.
- **The lock.** `check_lock()` confirms the lock gains suit the probe at each
  START, and retunes them if not. `calibrate_lock()` is the full calibration:
  it measures the loop, designs the gains, verifies them and stores them.
- **Checks of the box.** `self_test()` runs the GUI's Self-check, and
  `measure_signal_level()` and `store_signal_level()` set the 1550 nm laser's
  power for the probe, as the GUI's Adjust signal level.
- **Updates.** `check_for_updates()` compares the SDK and the box with the
  list on ultracoustics.ca, and `update_firmware()` installs firmware releases
  Ultracoustics has signed, from the website or an unzipped download. Their
  single steps are in `ultracoustics.updates`. The code that writes an image
  checks the signature itself, so nothing reaches a board without one.
- **Customer names only.** `Controller` has the methods listed in its help
  and no others, and drives the box through a private object. The code under
  it, `ultracoustics._internal`, ships because the self-test, the signal
  level, the lock and firmware updates run on it. It is not an API and is not
  supported: see "What is supported" in the README.
- **Recordings.** `load_recording()` opens SigMF recordings, split recordings,
  event files and raw `.bin` captures, without loading the samples into memory.
