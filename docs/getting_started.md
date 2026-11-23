# Getting started

The slow walkthrough. The [README](../README.md) is the first page: what the SDK is and the first
script. [Essentials](essentials.md) has what you must know before relying on
the data. This page takes the same ground one step at a time, for someone new to the BROADSONIC, to
Python, or to both. It takes you from an unopened kit to a saved, calibrated
recording, and explains the words the rest of the documentation uses.

If you only want to analyse recordings someone else made, you need no hardware:
skip to [Install the SDK](#2-install-the-sdk).

## Before you start: safety

The readout box contains two lasers, one of them invisible. The
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

A script can start the lasers just as the START button in the Ultracoustics
GUI does. Treat `ctrl.start_ready()` as you would pressing START.

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

The light on the box tells you its state:

| Light | Means |
|---|---|
| Breathing white | Powered, lasers off. This is the stopped state. |
| Solid magenta | Measuring. |
| Repeated quick flashes | A fault. Count the flashes and see the User Manual, section 5.1. |

The manuals, the GUI and the datasheet are at
[ultracoustics.ca/downloads](https://ultracoustics.ca/downloads). For help,
write to Ultracoustics at [ultracoustics.ca/contact](https://ultracoustics.ca/contact).

**Try the GUI first.** Before you write a script, press START in the GUI and
look at the signal. It confirms the box and probe work, and shows you what to
expect from a script.

**Only one program can use the box at a time.** Close the GUI before you run a
script, and stop your script before you open the GUI.

## 2. Install the SDK

You need Python 3.9 or later. On Windows, install it from
[python.org](https://www.python.org/downloads/) and tick "Add python.exe to
PATH" in the installer.

Open a terminal (on Windows: PowerShell) in the folder that holds this SDK, the
one with `README.md` and `pyproject.toml` in it. Then run these three lines.

On macOS and Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install .
```

The first line makes a private Python environment in a folder called `.venv`,
so the SDK and the packages it needs stay separate from anything else on your
computer. On current macOS and Linux a bare `pip install .` without it is
refused with "externally-managed-environment". The second line switches this
terminal to the environment: run it again each time you open a new terminal. If
PowerShell refuses to run it, use Command Prompt instead. The third installs
the SDK into it.

Check it worked:

```bash
python examples/analyze_recording.py
```

This reads the sample recording in `examples/data` and prints its levels. No
box is needed. If you see a table of octave bands, the SDK is installed.

`pip install .` also installs the packages the SDK needs, NumPy, SciPy, PyUSB
and libusb1 among them. To talk to the box, the computer also needs libusb and
permission to use the device:

- **Windows.** Nothing more. The box installs its own WinUSB driver, and the
  SDK brings `libusb-1.0.dll`.
- **macOS.** `brew install libusb`
- **Linux.** Install libusb (`sudo apt install libusb-1.0-0` on Debian and
  Ubuntu). Then let users open the box without root: save this line as
  `/etc/udev/rules.d/99-ultracoustics.rules`

  ```text
  SUBSYSTEM=="usb", ATTR{idVendor}=="2e9d", ATTR{idProduct}=="000a", MODE="0660", TAG+="uaccess"
  ```

  then run `sudo udevadm control --reload-rules` and plug the box in again.

### If the computer cannot open the box

- **"Master Board not found".** No box was found on USB. Check that the box is
  on and the cable carries data.
- **"USB reader process could not open the device"**, or **"exited during
  start-up".** Usually another program has the box: close the GUI and any
  other script that uses it. Or the script lacks its
  `if __name__ == "__main__":` line.
- **Windows shows BROADSONIC without a driver.** Run
  [Zadig](https://zadig.akeo.ie), choose Options > List All Devices, select
  BROADSONIC (2E9D:000A), choose WinUSB and click Replace Driver.
- **Access denied on Linux.** Add the udev rule above, then plug the box in
  again.

## 3. Your first capture

Close the GUI and check the box shows breathing white. Keep the probe still and
the surroundings quiet, ideally with the thing you want to measure switched
off. Then run:

```bash
python examples/capture.py
```

Here is the heart of that script. It is the
[first script](essentials.md#the-first-script-in-full).

```python
import time
import ultracoustics as uc

def main():
    with uc.Controller() as ctrl:     # required
        report = ctrl.start_ready()   # required: about 30 s
        if not report.calibrated:     # no new pressure calibration from this start
            print(report.message)
        rec = ctrl.capture(1.0)
        name = "capture_" + time.strftime("%Y%m%d_%H%M%S")    # a new name each run
        print("saved", rec.save(name))                        # optional

if __name__ == "__main__":            # required
    main()
```

| Line | What it does | If it goes wrong |
|---|---|---|
| `import ultracoustics as uc` | Loads the SDK. Everything you use is then `uc.something`. | |
| `with uc.Controller() as ctrl:` | Finds the box and connects. When the indented block ends, even after an error, the lasers are turned off and the USB is released. | `Master Board not found`: the box is off or the cable carries no data. `USB reader process could not open the device`: usually the GUI is still open. |
| `ctrl.start_ready()` | The same as START in the GUI. It turns the lasers on, waits until the box is ready, and calibrates for sound pressure. It prints nothing while it runs. It returns a report: `report.message` says how it went. | `uc.StartError`: the box did not become ready. The lasers are off again and the message says what to do. |
| `if not report.calibrated:` | The box is ready, but this start made no pressure calibration. The message says why, and whether data is in optical power only or carries an earlier start's calibration that must not be trusted. | |
| `ctrl.capture(1.0)` | Takes the next second of signal into memory as a recording. | `uc.CaptureError`: samples were lost or the box stopped being ready part-way. Nothing doubtful is ever returned. |
| `time.strftime(...)` | Puts the date and time in the name, such as `capture_20261116_141500`, so each run gets a new one. | |
| `rec.save(name)` | Writes `name.sigmf-data` and `name.sigmf-meta`, and returns the path. Keep the two files together. | `FileExistsError`: it never overwrites. Choose another name. |
| `if __name__ == "__main__":` | Required. The SDK reads the USB in a second Python process, which loads your script again when it starts. This line stops your code running twice. | Without it the script starts twice, or fails at start. |

### What `start_ready()` does while you wait

The call returns after about 30 seconds. In that time it:

1. Measures the detector with the lasers off. This is the **dark** reference
   that the calibration needs.
2. Turns the lasers on.
3. Waits for the box to settle on the sensor.
4. Checks that the box's control settings suit this probe. When they do not,
   it corrects them and stores them on the box, which adds about a minute.
5. Measures this start's **pressure calibration**, which turns optical power
   into sound pressure.

This is why the probe must be still and the surroundings quiet: steps 3 to 5
measure the probe itself. A start that fails usually says so within 90
seconds, or after a few minutes if step 4 was correcting the settings.

What the report tells you, and what `start_ready(quick=True)` gives up, is in
Essentials: [The first script, in full](essentials.md#the-first-script-in-full) and
[What you must know](essentials.md#what-you-must-know).
[How it works](how_it_works.md#start-up) has a diagram of these stages. To run
them one at a time, see [Advanced control](advanced_control.md).

## 4. What to do next

Each of these is a short snippet in Essentials:

- [Take a background](essentials.md#remove-the-background): the one reference
  only you can take. Do it just after `start_ready()`.
- [Record for longer than memory holds](essentials.md#record-for-a-long-time).
- [Look at what you saved](essentials.md#analyse-a-saved-recording): this needs
  no box.
- [Run the self-test](essentials.md#when-something-is-wrong) when anything
  seems wrong.

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
  `ctrl.start_ready()` gives a report with `.calibrated` and `.message`;
  `ctrl.capture()` gives a recording with `.duration_s`, `.spectrum()` and
  `.save()`.
- **Errors are specific.** `uc.StartError` means the box did not become ready,
  `uc.CaptureError` that data could not be trusted. The message says what to
  do.
- **When something is wrong, run the self-test first.** `ctrl.self_test()`
  checks the whole system in about six minutes. The GUI's Diagnostics menu
  calls it Self-check. Essentials shows
  [how to run it](essentials.md#when-something-is-wrong).
- **Numbers like `20e3` are Python for 20 000.** Frequencies are in Hz, times
  in seconds, optical power in µW, pressure in Pa or dB SPL.

## 6. Words used in this documentation

**The hardware**

| Word | Means |
|---|---|
| Box, readout box | The BROADSONIC readout unit: lasers, detector and electronics. |
| Probe, microphone | The physical object on the end of the fiber cable. |
| Sensor | The sensing element in the probe tip. Sound moves it, and the box measures that movement with light. |
| START, STOP | Lasers on and measuring, or lasers off. In a script: `ctrl.start_ready()` and `ctrl.stop()`. |
| 1550 nm laser | The invisible infrared laser whose light, reflected from the probe, is the signal. |
| 638 nm laser | The visible red laser. The box uses it to keep itself in tune with the sensor. |
| Firmware | The software inside the box. Updated only with releases Ultracoustics has signed. |
| The GUI | The Ultracoustics GUI, the desktop program for the box. It is built on this SDK. |

**Getting ready to measure**

| Word | Means |
|---|---|
| Ready | The box is holding itself in tune with the sensor, so the signal can be trusted. `start_ready()` returns when it is. The API calls this the lock: see [Advanced control](advanced_control.md). |
| Dark | A measurement with the lasers off, showing the electronics' own noise. `start_ready()` takes it. |
| Background | A capture of the surroundings with the thing you are measuring switched off. You take it, and subtract it. |
| Pressure calibration | The measurement, made at each START, that lets the SDK convert optical power to sound pressure. It belongs to one probe at one START. `start_ready()` makes it. |
| Signal level | How bright the reflected light is at the detector. Adjusted once per probe so the detector does not saturate. |
| Self-test | A six-minute check of the whole box, for when something seems wrong: `ctrl.self_test()`. The GUI's menu calls it Self-check. |

**The data**

| Word | Means |
|---|---|
| Sample | One reading of the detector. The box takes 10 million a second (10 MS/s). |
| Counts | The raw readings, whole numbers from 0 to 16383. |
| Optical power, µW | Counts converted to the light reaching the detector: one count is 0.00844 µW. Sound appears as small fluctuations in it. |
| Stream | The continuous flow of samples from the box over USB: 20 MB every second. |
| Ring, buffer | The last 1.2 seconds of the stream, kept in memory and continuously overwritten. A snapshot or an event can use about 1.0 s of it. |
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
| See how the box, the USB and your script fit together | [How it works](how_it_works.md) |
| Understand the numbers | [Analysis guide](analysis_guide.md) |
| Save data and read the files | [Saving data](saving_data.md) |
| Watch a machine unattended | [Monitoring](monitoring.md) |
| Run the start-up one stage at a time | [Advanced control](advanced_control.md) |
| Look a name up | [Reference](reference.md) |
| Keep the box up to date | [Updates](UPDATES.md) |
