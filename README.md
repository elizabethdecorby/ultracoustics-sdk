# Ultracoustics SDK

Python for the Ultracoustics BROADSONIC optical ultrasound sensor. Use it to
capture the box's 10 MS/s signal, analyse saved recordings in sound pressure, or monitor for changes.
The Ultracoustics GUI is built on these same functions, so a script and the GUI give the same answers.

Analysis needs no box. Every guide and the first three examples run on the
sample recording in `examples/data`.

## Install

Python 3.9 or later. From a copy of this repository:

```bash
pip install .
```

This also installs NumPy, SciPy, PyUSB, libusb1 and pySerial. To talk to the
box, the computer also needs libusb and permission to use the device:

- **Windows.** Nothing more. The box installs its own WinUSB driver, and the
  SDK brings `libusb-1.0.dll`. If Device Manager shows BROADSONIC without a
  driver, see Troubleshooting.
- **macOS.** `brew install libusb`
- **Linux.** Install libusb (`sudo apt install libusb-1.0-0` on Debian and
  Ubuntu). Then let users open the box without root: save this line as
  `/etc/udev/rules.d/99-ultracoustics.rules`

  ```text
  SUBSYSTEM=="usb", ATTR{idVendor}=="2e9d", ATTR{idProduct}=="000a", MODE="0660", TAG+="uaccess"
  ```

  then run `sudo udevadm control --reload-rules` and plug the box in again.

Use a USB 2.0 High Speed port or faster, without a hub. The stream is 20 MB/s.

## Capture

```python
import ultracoustics as uc

def main():
    with uc.Controller() as ctrl:            # close() always runs, with a confirmed stop
        dark = ctrl.capture_dark()           # lasers-off reference, before START
        ctrl.start()
        ctrl.check_lock(retune=True)         # every START: see "The lock" below
        ctrl.calibrate_pressure(dark)        # this START's pressure calibration
        rec = ctrl.capture(1.0)              # the next second, checked for lost samples and lock
        rec.save("capture")                  # SigMF, with its calibration
        ctrl.record("long_run", duration_s=60)    # straight to disk, any length

if __name__ == "__main__":
    main()
```

Keep the `__main__` guard. The USB stream is read by a second Python process,
which imports your script when it starts, so code outside the guard runs twice.

The stream starts at `connect()`, so `ctrl.buffer` shows the signal before
START, with the lasers off. `ctrl.telemetry` has the lasers and temperatures
while running; `ctrl.request_telemetry()` asks a stopped box for them too.
`fault = ctrl.stop()` says why, if the box had already stopped itself (the probe
signal lost, say): `fault.code` and `fault.reason`, or `None`.

## Analyse a recording

```python
import ultracoustics as uc

rec = uc.load_recording("capture.sigmf-meta")
spectrum = rec.spectrum(fft_size=16384)               # W²/Hz, every FFT averaged
pressure = rec.calibration.to_pressure(spectrum)      # Pa²/Hz, and pressure.spl in dB SPL

print(pressure.band_level(20e3, 100e3).spl_db)        # total sound in a band, dB SPL
print(uc.tone_level(rec, rec.calibration, 40e3).spl_db)   # one steady tone, dB SPL
print(rec.calibration.noise_budget(spectrum).limited_by_at(1e6))   # what sets the floor there
```

A density, a band level and a tone level are three different numbers in dB SPL.
The [analysis guide](docs/analysis_guide.md) explains when to use which, and
how to remove noise.

## Watch a machine

```python
healthy = ctrl.capture(5.0).spectrum()      # the machine running normally
watch = uc.BandPowerTrigger((20e3, 100e3), threshold_db=10, baseline=healthy, name="bearing")
monitor = ctrl.monitor(watch, folder="runs", on_alarm=print)
...
monitor.stop()
print(monitor.summary())                    # events, alarms, samples lost, time not watched
```

The run's folder holds the events, a level log every second and the alarms.
`uc.find_events(recording, watch)` replays the same triggers over a saved
recording, to tune limits with no box. See [monitoring](docs/monitoring.md).

## The lock

The box holds its laser on the probe with a control loop. The loop's gains
depend on the probe and can change from one START to the next.

Run `ctrl.check_lock(retune=True)` after every START where possible. It
confirms the lock gains suit this probe and this start, and corrects them if
they don't. Keep the probe still and away from noise while it runs, ideally
before the monitored machine starts. It disturbs the signal for about 1.6 s, so
the SDK won't run it during a recording or while a monitor is on. If
conditions are too noisy it reports that the gains couldn't be measured, leaves
them unchanged, and you can run it again at the next START.

`ctrl.calibrate_lock()` is the full calibration. It measures the loop for about
a minute and designs the gains from the measurement, then verifies them and
stores them on the box. Run it for a new probe, or when the check keeps asking
for a retune. Gains that fail their checks are never kept.

```python
report = ctrl.check_lock(retune=True)
report.status, report.message      # "ok", "retuned", "inconclusive", ...: see help(uc.LockReport)
report.ready                       # a settled lock to take data on
ctrl.wait_ready()                  # waits for a settled lock without the burst
report = ctrl.calibrate_lock()     # the full calibration: about 2 minutes, and 1 more to store
ctrl.lock_gains                    # the gains running now: kp, ki_per_s and where they came from
ctrl.restore_lock_gains()          # drop gains that were not stored: back to the box's
ctrl.clear_stored_lock_gains()     # box stopped: the factory gains from the next START
```

Choosing Kp and Ki by hand is a service task.

## Check the box

These are the GUI's Diagnostics menu. Each takes a few minutes, needs the box
connected with nothing recording or monitoring, and leaves it stopped.

```python
report = ctrl.self_test(probe_serial="...")   # the Self-check: firmware, lasers, temperatures, probe, lock, data;
                                              # probe_serial: the number on your probe, for the report
print(report.text())
report.save("self_test.json")      # send it to Ultracoustics when something seems wrong

level = ctrl.measure_signal_level()     # Adjust signal level: the 1550 nm power for this probe
print(level.describe())                 # changes nothing
ctrl.store_signal_level(level)          # stores it, and checks the box comes back at it
```

`report.recommends_signal_adjustment` says when the detector saturated in the
self-test, which `measure_signal_level()` corrects.

## Updates

`check_for_updates()` compares this SDK and the connected box with the list on
ultracoustics.ca, and `update_firmware()` installs firmware Ultracoustics has
signed. Neither runs by itself, and the box never needs a network.

```python
report = uc.check_for_updates()    # close any Controller first
print(report.describe())
```

The steps behind them (reading the box's versions, one board at a time) are in
`ultracoustics.updates`: see [Updates](docs/UPDATES.md).

## Documentation

| | |
|---|---|
| [Analysis guide](docs/analysis_guide.md) | Counts to sound pressure, the three SPL numbers, removing noise, waveforms and tones |
| [Saving data](docs/saving_data.md) | The SigMF files, their metadata, gaps, split files and events |
| [Monitoring](docs/monitoring.md) | Triggers, baselines, alarms and events, the run folder, limits |
| [Updates](docs/UPDATES.md) | Checking for updates, installing firmware, with or without a network |
| [Examples](examples) | `analyze_recording.py`, `remove_background.py` and `find_events.py` run on the sample recording. `capture.py` and `monitor.py` need a box. |

Every public class and function has its own help: `help(uc.Recording)`,
`help(uc.Controller.monitor)`.

## Troubleshooting

**"Master Board not found".** Check that the box is on and the cable carries
data. Only one program can use the box at a time: close the GUI first.

**Windows shows BROADSONIC without a driver.** Run
[Zadig](https://zadig.akeo.ie), choose Options > List All Devices, select
BROADSONIC (2E9D:000A), choose WinUSB and click Replace Driver.

**Access denied on Linux.** Add the udev rule above, then plug the box in again.

**`CaptureError: ... the computer fell behind the stream`, or `the stream
lost data`.** Use a port without a hub, close other heavy programs, and save
long runs with `record()`, which writes straight to disk.

**The script starts twice, or fails at start with a message about
`__main__`.** Put the script's work under `if __name__ == "__main__":`.

## Licence

The SDK is under the BSD 3-Clause licence: see [LICENSE](LICENSE). The Windows
`libusb-1.0.dll` it includes is libusb, under the GNU LGPL 2.1 or later: see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
