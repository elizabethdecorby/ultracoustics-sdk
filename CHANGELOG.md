# Changelog

## 1.0.0

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
