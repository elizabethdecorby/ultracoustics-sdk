# Reference

Every public name in the SDK, one line each, grouped by what you are trying to
do. All of them are on `ultracoustics` itself:

```python
import ultracoustics as uc
```

For the detail of any of them, ask Python: `help(uc.Controller.capture)`.
New to the SDK? Start with [Getting started](getting_started.md).

## The box

| Name | What it is |
|---|---|
| `uc.Controller` | The box. Connect with `with uc.Controller() as ctrl:`; everything below is a call on `ctrl`. |
| `uc.Status` | What `ctrl.status` returns: `connected`, `streaming`, `running`, `locked`, `recording`, `monitoring`, `samples_received`, `stream_drops`. |
| `uc.BoxFault` | What `ctrl.stop()` returns if the box had already stopped itself: `code` and `reason`. Otherwise `None`. |
| `uc.CaptureError` | Raised when data could not be trusted: samples were lost, or there was no lock. |

### Starting and stopping

| Call | Does |
|---|---|
| `ctrl.connect()`, `ctrl.close()` | Open and release the box. `with uc.Controller() as ctrl:` calls both for you. |
| `ctrl.start()` | Lasers on, as START. Returns before the lock: follow with `check_lock()` or `wait_ready()`. |
| `ctrl.stop()` | Lasers off, as STOP. Ends any recording or monitor first. |
| `ctrl.restart()` | Stop if running, then start. |
| `ctrl.warm()` | Standby: lasers off but temperatures held, so the next `start()` is faster. |

### Its state

| Name | Tells you |
|---|---|
| `ctrl.status` | Everything at once, as a `uc.Status`. Sends nothing to the box. |
| `ctrl.connected`, `ctrl.streaming`, `ctrl.running`, `ctrl.locked` | Each as `True` or `False`. `locked` says there is a lock, not how good it is. |
| `ctrl.device_serial` | The box's USB serial number. |
| `ctrl.telemetry` | The box's newest report of its lasers, temperatures and lock, or `None`. |
| `ctrl.request_telemetry()` | Asks a stopped box for its telemetry. |
| `ctrl.stream_stats` | The USB stream's packet counters, for diagnosing lost samples. |
| `ctrl.verbose` | Set `True` to print messages about USB traffic and captures. |

### The lock

| Name | Does |
|---|---|
| `ctrl.check_lock(retune=True)` | Waits for a lock, checks its gains suit this probe, corrects them if not. Run after every START. Disturbs the signal for 1.6 s. |
| `ctrl.wait_ready()` | Waits for a settled lock without disturbing the signal. Does not check the gains. |
| `ctrl.calibrate_lock()` | The full lock calibration, about three minutes. For a new probe. |
| `ctrl.lock_gains` | The gains running now, as a `uc.LockGains`. |
| `ctrl.restore_lock_gains()` | Drops gains that were not stored. |
| `ctrl.clear_stored_lock_gains()` | Returns the box to its factory gains from the next START. |
| `uc.LockReport` | What the lock calls return. Read `ready` (safe to take data), `ok` (and the gains are good), `status` and `message`. |
| `uc.LockGains` | `kp`, `ki_per_s`, `zero_hz` and `source`. Reported, never set by hand. |
| `uc.LockError` | Raised when the lock did not reach the state asked for. |

`check_lock()` and `calibrate_lock()` also take `progress` (a function called
as it goes, for a display), `cancel` (a function that returns `True` to stop
early) and `report_dir` (a folder to keep the measurement in, if support asks
for it).

### Taking data

| Call | Does |
|---|---|
| `ctrl.capture_dark()` | The lasers-off reference the calibration needs. Before `start()`. |
| `ctrl.calibrate_pressure(dark)` | This START's pressure calibration. Kept as `ctrl.calibration` and saved with every recording. |
| `ctrl.capture(seconds)` | The next stretch of signal, in memory, as a `uc.Recording`. |
| `ctrl.snapshot(seconds)` | What just happened, from the ring: up to 1.2 s back. |
| `ctrl.record(path, duration_s=...)` | Writes the stream straight to disk, any length. Returns a recorder (below). |
| `ctrl.monitor(triggers, ...)` | Watches the stream for triggers. Returns a `uc.Monitor`. |

`ctrl.record()` gives back a recorder. Read `samples` (how many were saved),
`lost` (how many the disk could not keep up with), `dropped` (lost before they
reached the computer) and `reason` (why it ended). `open()` gives the
`uc.Recording`. With `wait=False` the call returns at once and the recording
runs on: end it with the recorder's `stop()` then `join()`.

### The live signal

| Name | Is |
|---|---|
| `ctrl.buffer` | The ring itself: the newest samples as raw counts. Read-only. |
| `ctrl.buffer_head` | Where the next sample will be written in `buffer`. Read it first, then slice `buffer`. |
| `ctrl.buffer_capacity` | The length of `buffer`, in samples. |
| `ctrl.samples_received` | Every sample since the stream began. |
| `uc.LiveSpectrum` | A spectrum of the live stream that updates as samples arrive, for a live plot. |
| `uc.counts_to_uw(counts)` | Raw counts to optical power in µW. |

For most scripts `ctrl.snapshot()` is easier than reading `buffer` yourself.

### Checking the box

| Name | Does |
|---|---|
| `ctrl.self_test(probe_serial="...")` | The six-minute self-test. Returns a `uc.SelfTestReport`. Leaves the box stopped. |
| `uc.SelfTestReport` | `ok`, `problems`, `text()` to print it, `save(path)` to send to support, `recommends_signal_adjustment`. |
| `ctrl.measure_signal_level()` | Works out the right 1550 nm laser power for this probe. Changes nothing. |
| `ctrl.store_signal_level(level)` | Stores that setting on the box and checks it. |
| `uc.SignalLevel` | What `measure_signal_level()` found. `describe()` prints it. |
| `uc.StoredSignalLevel` | What was stored: `dac`, and `verified` if the box came back at it. |
| `uc.SignalLevelError` | Raised when no setting suits this probe. |

## Recordings

| Name | What it is |
|---|---|
| `uc.load_recording(path)` | Opens a saved recording without loading it all into memory. |
| `uc.Recording` | A stretch of signal. `duration_s`, `sample_rate`, `start_time`, `calibration`, `gaps`. |
| `rec.counts()`, `rec.power_uw()` | The samples, raw or in µW. Both take a start and end time in seconds. |
| `rec.spectrum(fft_size)` | Its averaged spectrum, as a `uc.Spectrum`. |
| `rec.save(path)` | Writes it as SigMF. Never overwrites. |
| `rec.events` | For a monitor's event file: each event, with its `features`. |

## Spectra and pressure

| Name | What it is |
|---|---|
| `uc.Spectrum` | Optical power at each frequency: `freqs` (Hz), `psd_w2hz` (W²/Hz), `num_averages`, `sigma`. |
| `spectrum.subtract(background)` | Removes a measured background. |
| `spectrum.significant(sigma=5)` | Which bins truly stand above the noise. |
| `spectrum.difference_db(other)` | The change from another spectrum, in dB. |
| `uc.Calibration` | One START's pressure calibration. `rec.calibration` gives a recording's. |
| `cal.to_pressure(spectrum)` | A `uc.PressureSpectrum`. |
| `cal.noise_budget(spectrum)` | A `uc.NoiseBudget`: what sets the noise floor at each frequency. |
| `cal.pressure_waveform(...)`, `cal.pa_per_uw(band)` | A waveform in Pa, and the scale factor used. |
| `uc.load_calibration(path)` | Opens a calibration file the app saved. |
| `uc.CalibrationError` | Raised when no calibration could be made. |
| `uc.PressureSpectrum` | Sound pressure at each frequency: `spl` (density), `band_level(lo, hi)`, `octave_levels()`. |
| `uc.tone_level(rec, cal, freq_hz)` | The level of one steady tone. |
| `uc.NoiseBudget` | `dark`, `shot`, `white`, `thermal`, `broad`, and `limited_by_at(freq_hz)`. |
| `uc.band_pass(x, band, fs)`, `uc.notch(x, freq_hz, fs)` | Filters for waveforms. They add no delay. |
| `uc.ProbeParameters` | A probe's physical constants. Only needed if Ultracoustics gives you some for a particular probe. |

`band_level()` returns an object with `spl_db`, `p_rms_pa` and `resolved`;
`tone_level()` one with `spl_db`, `p_rms_pa` and `bandwidth_hz`;
`octave_levels()` one with `center_hz` and `level_db`.

## Monitoring

| Name | What it is |
|---|---|
| `uc.BandPowerTrigger(band_hz, threshold_db=10)` | Fires when a frequency band gets louder than normal. |
| `uc.LevelTrigger(level_uw=...)` | Fires when the signal crosses a level. |
| `uc.Monitor` | A running monitor, from `ctrl.monitor()`. `stop()`, `summary()`, `active_alarms`. |
| `uc.MonitorSummary` | How the run went: `events`, `alarms_raised`, `samples_lost`, `not_monitored_s`, `complete`, `folder`. |
| `uc.AlarmChange` | What your `on_alarm` function is given: `trigger`, `state`, `utc`, `level`, `limit`. |
| `uc.find_events(rec, triggers)` | Runs the same triggers over a saved recording, to tune limits with no box. |

## Updates

| Name | What it is |
|---|---|
| `uc.check_for_updates()` | Compares this SDK and the connected box with the published list. Returns a `uc.UpdateReport`. |
| `uc.update_firmware()` | Installs signed firmware. Keep the box powered until it returns. |
| `uc.UpdateReport` | `describe()` prints it; `sdk_update` and `firmware_updates` say what is available. |
| `uc.UpdateCheckError` | The list could not be fetched, for example with no network. |
| `uc.FirmwareRefused` | An update was refused before anything was written. The box is unchanged. |
| `uc.FirmwareUpdateError` | An update did not finish or could not be confirmed. |

The single steps these are built from are in `ultracoustics.updates`, the one
module you import from directly. See [Updates](UPDATES.md).

## What is not here

Some files in the package have ordinary names but are not for scripts to import
from: `ultracoustics.config`, `ultracoustics.readout`, `ultracoustics.nep`,
`ultracoustics.recorder`, `ultracoustics.live`, `ultracoustics.lock`,
`ultracoustics.selftest`, `ultracoustics.signal_level` and the files under
`ultracoustics.analysis`. They hold the working parts behind the names above,
and the Ultracoustics app uses them. What they contain beyond those names can
change from one release to the next.

`ultracoustics._internal`, and any name that starts with an underscore, is
private and unsupported: see "What is supported" in the README.
