# Getting started

For someone new to the BROADSONIC, to Python, or to both. It takes you from an
unopened kit to a saved, calibrated recording, and explains the words the rest
of the documentation uses.

If you only want to analyse recordings someone else made, you need no hardware:
skip to [Install the SDK](#2-install-the-sdk).

## Before you start: safety

The readout unit contains two lasers, one of them invisible. The
[User Manual](https://ultracoustics.ca/assets/User%20Manual/BROADSONIC%20Optical%20Microphone%20User%20Manual.pdf)
has the full safety section. The points that matter every time:

- **Never look into the probe tip, a fiber end or an optical port.** Invisible
  light may be present, so the absence of a red glow does not mean it is safe.
- **Never touch or clean the probe's sensing face.** Hold the probe by its
  casing, and keep the tip at least 4 mm from whatever you measure.
- **Do not kink or pull the fiber cable.** Keep every bend wider than a 30 mm
  radius.
- **Unplug the 9 VDC supply before connecting, disconnecting or cleaning** the
  optical connectors.

A script can start the lasers just as the START button in the app does. Treat
`ctrl.start()` as you would pressing START.

## 1. Set up the hardware

Follow the Quick Start Guide that came with your kit, sections 01 to 03. In
short:

1. With the power supply unplugged, clean both optical connectors and both
   ports with the supplied cleaner pen.
2. Push the paired connector straight into the ports until both are seated. It
   fits one way only.
3. Connect the 9 VDC supply, then a USB-C **data** cable from the rear port to
   your computer. A charging-only cable will not work. Use a port on the
   computer itself, not a hub.
4. Leave 5 cm clear around each ventilation opening.

The light on the unit tells you its state:

| Light | Means |
|---|---|
| Breathing white | Powered, lasers off. This is the stopped state. |
| Solid magenta | Measuring. |
| Repeated quick flashes | A fault. Count the flashes and see the User Manual, section 5.1. |

The manuals, the app and the datasheet are at
[ultracoustics.ca/downloads](https://ultracoustics.ca/downloads). For help,
write to Ultracoustics at [ultracoustics.ca/contact](https://ultracoustics.ca/contact).

**Only one program can use the box at a time.** Close the BROADSONIC app before
you run a script, and stop your script before you open the app.

## 2. Install the SDK

You need Python 3.9 or later. On Windows, install it from
[python.org](https://www.python.org/downloads/) and tick "Add python.exe to
PATH" in the installer.

Open a terminal (on Windows: PowerShell) in the folder that holds this SDK, the
one with `README.md` and `pyproject.toml` in it. Then:

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
pip install .
```

The first line makes a private Python environment in a folder called `.venv`,
so the SDK and the packages it needs stay separate from anything else on your
computer. The second switches this terminal to it: run it again each time you
open a new terminal. On macOS and Linux it is `source .venv/bin/activate`. If
PowerShell refuses to run it, use Command Prompt instead. The third installs
the SDK into it.

Check it worked:

```bash
python examples/analyze_recording.py
```

This reads the sample recording in `examples/data` and prints its levels. No
box is needed. If you see a table of octave bands, the SDK is installed.

## 3. Your first capture

Close the BROADSONIC app, check the unit shows breathing white, and run:

```bash
python examples/capture.py
```

Here is that script, step by step. Each step depends on the one before, so keep
the order.

```python
import ultracoustics as uc

def main():
    with uc.Controller() as ctrl:
        dark = ctrl.capture_dark()
        ctrl.start()
        report = ctrl.check_lock(retune=True)
        ctrl.calibrate_pressure(dark)
        rec = ctrl.capture(1.0)
        rec.save("capture")

if __name__ == "__main__":
    main()
```

| Line | What it does | If it goes wrong |
|---|---|---|
| `import ultracoustics as uc` | Loads the SDK. Everything you use is then `uc.something`. | |
| `with uc.Controller() as ctrl:` | Finds the box and connects. When the indented block ends, even after an error, the lasers are turned off and the USB is released. | `Master Board not found`: the box is off, the cable carries no data, or the app is still open. |
| `ctrl.capture_dark()` | Measures the detector with the lasers off. The calibration needs this reference, and it can only be taken before START. | `RuntimeError` if the box is already running. |
| `ctrl.start()` | The same as START in the app: the lasers come on. It returns straight away, before the box is ready. | |
| `ctrl.check_lock(retune=True)` | Waits for the box to be ready, checks it suits this probe, and corrects it if not. Keep the probe still and the room quiet. Usually a few seconds, up to a minute if it corrects. | `report.ready` is `False`: check the probe is connected and nothing is disturbing it, then run the script again. |
| `ctrl.calibrate_pressure(dark)` | Works out how to turn the optical signal into sound pressure. Needs a quiet room. | Leave it out and the recording still saves, but only in optical power, with no dB SPL. |
| `ctrl.capture(1.0)` | Takes the next second of signal into memory as a recording. | `CaptureError`: samples were lost or the box stopped being ready part-way. Nothing doubtful is ever returned. |
| `rec.save("capture")` | Writes `capture.sigmf-data` and `capture.sigmf-meta`. Keep the two files together. | `FileExistsError`: it never overwrites. Choose another name. |
| `if __name__ == "__main__":` | Required. The SDK reads the USB in a second Python process, which loads your script again when it starts. This line stops your code running twice. | Without it the script starts twice, or fails at start. |

In the app, START does the middle steps for you ("Warming up", then "Ready",
then "Calibrated"). In a script you call them yourself, which is why there are
more lines.

To record for longer than memory holds, write straight to disk instead:

```python
recorder = ctrl.record("long_run", duration_s=60)
print(recorder.samples, "samples saved,", recorder.lost, "lost")
```

## 4. Look at what you saved

```python
import ultracoustics as uc

rec = uc.load_recording("capture.sigmf-meta")
print(rec.duration_s, "seconds")

spectrum = rec.spectrum(fft_size=16384)             # how strong the signal is at each frequency
pressure = rec.calibration.to_pressure(spectrum)    # the same, as sound pressure
print(pressure.band_level(20e3, 100e3).spl_db, "dB SPL between 20 and 100 kHz")
```

This needs no box and no `__main__` line, because it never opens the USB. The
[analysis guide](analysis_guide.md) goes on from here.

## 5. Finding your way around the SDK

- **Everything you need is on `ultracoustics` itself.** After
  `import ultracoustics as uc`, write `uc.Controller`, `uc.load_recording`,
  `uc.BandPowerTrigger` and so on. You do not need to import from the files
  inside the package. The one exception is `ultracoustics.updates`, for the
  single steps of a firmware update.
- **`help()` works on all of it.** `help(uc.Controller)` lists what the box can
  do, `help(uc.Controller.capture)` explains one call, and `help(uc.Recording)`
  what you can do with a recording.
- **[The reference](reference.md) lists every name on one page**, grouped by
  what you are trying to do.
- **Calls give back objects, and you read results off them.**
  `ctrl.check_lock()` gives a report with `.ready` and `.message`;
  `ctrl.capture()` gives a recording with `.duration_s`, `.spectrum()` and
  `.save()`.
- **Errors are specific.** `uc.CaptureError` means data could not be trusted,
  `uc.LockError` that the box was not ready, `uc.CalibrationError` that the
  room was not quiet enough to calibrate. The message says what to do.
- **Numbers like `20e3` are Python for 20 000.** Frequencies are in Hz, times
  in seconds, optical power in µW, pressure in Pa or dB SPL.

## 6. Words used in this documentation

**The hardware**

| Word | Means |
|---|---|
| Box, readout unit | The BROADSONIC unit: lasers, detector and electronics. |
| Probe, microphone | The optical sensor on the end of the fiber cable. |
| Dome | The tiny membrane in the probe tip that sound moves. The box measures that movement with light. |
| START, STOP | Lasers on and measuring, or lasers off. In a script: `ctrl.start()` and `ctrl.stop()`. |
| 1550 nm laser | The invisible infrared laser whose light, reflected from the probe, is the signal. |
| 638 nm laser | The visible red laser. The box uses it to hold the lock. |
| Firmware | The software inside the box. Updated only with releases Ultracoustics has signed. |
| The app, the GUI | The BROADSONIC desktop program. It is built on this SDK. |

**Getting ready to measure**

| Word | Means |
|---|---|
| Lock | The box holding its laser at the probe's most sensitive point. Without a lock there is no measurement. "Locked" means it has one. |
| Lock gains | Settings of the loop that holds the lock. `check_lock()` chooses them; you never set them by hand. |
| Dark | A measurement with the lasers off, showing the electronics' own noise. |
| Pressure calibration | The measurement, made at each START, that lets the SDK convert optical power to sound pressure. It belongs to one probe at one START. |
| Signal level | How bright the reflected light is at the detector. Adjusted once per probe so the detector does not saturate. |
| Self-test | A six-minute check of the whole box, for when something seems wrong. |

**The data**

| Word | Means |
|---|---|
| Sample | One reading of the detector. The box takes 10 million a second (10 MS/s). |
| Counts | The raw readings, whole numbers from 0 to 16383. |
| Optical power, µW | Counts converted to the light reaching the detector. Sound appears as small fluctuations in it. |
| Stream | The continuous flow of samples from the box over USB: 20 MB every second. |
| Ring, buffer | The last 1.2 seconds of the stream, kept in memory and continuously overwritten. |
| Recording | A stretch of samples with its time, calibration and description. |
| SigMF | The file format for recordings: a `.sigmf-data` file of samples and a `.sigmf-meta` file describing them. |
| Gap | A place in a recording where samples were lost. Recordings know their gaps. |

**The analysis**

| Word | Means |
|---|---|
| Spectrum | How much signal there is at each frequency. |
| FFT size | How finely a spectrum divides frequency. Larger gives narrower bins and needs more data. |
| Bin | One frequency step of a spectrum. |
| dB SPL | Sound pressure level, the usual unit of loudness, relative to 20 µPa. |
| Density, band level, tone level | Three different answers to "how loud". The [analysis guide](analysis_guide.md) explains when to use which. |
| Calibrated band | 0 to 4.5 MHz: the range over which pressure values are valid. |
| Noise floor | The quietest signal that can be told from the system's own noise. |

**Monitoring**

| Word | Means |
|---|---|
| Trigger | A condition to watch for, such as a band getting 10 dB louder. |
| Baseline | What "normal" is for a trigger: learned on the spot, or a spectrum of the machine when healthy. |
| Event | A short stretch of data saved around the moment a trigger fired. |
| Alarm | A trigger's condition staying true. Raised once, cleared when it ends. |

## Where next

| To | Read |
|---|---|
| Understand the numbers | [Analysis guide](analysis_guide.md) |
| Save data and read the files | [Saving data](saving_data.md) |
| Watch a machine unattended | [Monitoring](monitoring.md) |
| Look a name up | [Reference](reference.md) |
| Keep the box up to date | [Updates](UPDATES.md) |
