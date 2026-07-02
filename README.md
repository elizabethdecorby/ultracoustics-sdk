# Ultracoustics SDK

Python SDK for the Ultracoustics BROADSONIC hardware platform.

## Project Structure

```text
ultracoustics-sdk/
├── pyproject.toml
├── README.md
├── examples/
│   ├── basic_capture.py
│   ├── offline_psd.py
│   └── probe_characterization.py
└── ultracoustics/
    ├── __init__.py
    ├── config.py
    ├── controller.py
    ├── characterization.py
    ├── processing.py
    ├── nep_dialog.py
    ├── nep.py
    └── _internal/
        ├── __init__.py
        ├── comms.py
        ├── maintenance.py
        ├── protocol.py
        ├── stream_proc.py
        └── bin/
```

## At a Glance

- The public API is exposed through the `ultracoustics` package.
- Modules under `_internal` are implementation details and are included for guided diagnostics.
- USB ingestion runs in a **separate reader process** using async `libusb1` transfers. Samples are written directly into a **shared-memory ring buffer** so the parent process (your script or GUI) reads them lock-free and zero-copy.

## Installation

Install in editable mode so local source edits are available immediately.

1. Create and activate a virtual environment.

Windows PowerShell:

```powershell
py -3.13 -m venv venv_name
.\venv_name\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python3 -m venv venv_name
source venv_name/bin/activate
```

2. From within the new venv, install the SDK from the repository root (the folder that contains pyproject.toml).

```bash
pip install -e .
```

This installs NumPy, PyUSB, libusb1, and PySerial automatically.

## Hardware Connection

The Master Board supports WCID plug-and-play on Windows.

1. Plug the board into a USB data port.
2. Confirm Windows assigns WinUSB driver automatically (see "BROADSONIC" in Device Manager)
3. If automatic driver assignment fails, use the Zadig steps in Troubleshooting.

## Quick Start

```python
import time
from ultracoustics import Controller, load_binary, compute_psd

ctrl = Controller(verbose=True)
ctrl.connect()

# Start USB data ingestion first (laser remains off)
ctrl.begin_stream()

# Enter measurement mode (laser on, probe turns red, pink indicater)
ctrl.start()

# Wait for system lock and buffer fill
time.sleep(5.1)

# Save 1 second of raw uint16 ADC data
data = ctrl.save(duration_s=1.0, path="capture.bin")

ctrl.stop()        # System to IDLE state (laser off, white indicator)
ctrl.end_stream()  # Stop USB reader thread
ctrl.close()

# Optional offline PSD analysis
samples = load_binary("capture.bin")
freq_hz, psd_db = compute_psd(samples, fft_size=8192, num_averages=20)
```

## API Reference

NEED to ADD NEP notes***

### `Controller(verbose=False)`

Top-level interface to the BROADSONIC hardware. All hardware interaction goes through this class.

```python
from ultracoustics import Controller
ctrl = Controller(verbose=True)
```

**Connection lifecycle**

| Method | Description |
|---|---|
| `connect()` | Open the USB Bulk connection to the Master Board. Must be called first. |
| `close()` | Release all USB resources. |

**State management**

| Method | Description |
|---|---|
| `begin_stream()` | Start reading USB data into the circular buffer. Laser remains off. |
| `end_stream()` | Stop the USB reader thread.|
| `start()` | Send BOOT command: turns the laser on and begins measurement. Calls `begin_stream()` automatically if needed. |
| `stop()` | Send IDLE command: turns the laser off. Stream keeps running so the data buffer stays live. |
| `warm()` | Enter WARM/standby state: lasers off but system stays regulated for faster re-start than BOOT. |

**Data capture**

| Method / Signature | Description |
|---|---|
| `save(duration_s=1.0, path=None) → np.ndarray` | Snapshot the last `duration_s` seconds from the circular buffer. Writes raw `uint16` binary to `path` if provided. Raises `RuntimeError` if the buffer has not accumulated enough samples yet. |

**Live buffer properties** (zero-copy access for GUIs / real-time consumers)

| Property | Type | Description |
|---|---|---|
| `buffer` | `np.ndarray` (uint16) | The live circular sample ring (backed by shared memory written by the reader subprocess). |
| `buffer_head` | `int` | Current write-head index in the ring. |
| `buffer_capacity` | `int` | Ring length in samples (~1.2 s at 10 MHz). |
| `samples_received` | `int` | Cumulative samples received since streaming started. |
| `running` | `bool` | `True` between `start()` and `stop()`. |
| `streaming` | `bool` | `True` between `begin_stream()` and `end_stream()`. |
| `connected` | `bool` | `True` if the USB connection is open. |

---

### `load_binary(path, dtype=np.uint16) → np.ndarray`

Load raw ADC samples from a binary file written by `save()`.

```python
from ultracoustics import load_binary
samples = load_binary("capture.bin")
```

---

### `compute_psd(samples, fft_size=8192, num_averages=1, sample_rate=10_000_000) → (freq_hz, psd_db)`

Compute a Hanning-windowed, averaged one-sided Power Spectral Density in dB re 1 W²/Hz.

```python
from ultracoustics import compute_psd
freq_hz, psd_db = compute_psd(samples, fft_size=8192, num_averages=20)
```

| Parameter | Default | Description |
|---|---|---|
| `samples` | — | 1-D uint16 array (at least `fft_size × num_averages` elements). |
| `fft_size` | `8192` | Points per FFT segment. Controls frequency resolution. |
| `num_averages` | `1` | Non-overlapping segments to average. Higher values reduce noise floor. |
| `sample_rate` | `10_000_000` | ADC sampling rate in Hz. |

Returns `(freq_hz, psd_db)` — both 1-D float64 arrays of length `fft_size // 2 + 1`.

---

### `adc_to_uw(samples, baseline=0.0) → np.ndarray`

Convert raw 14-bit ADC counts to optical power in µW, applying the full signal chain (ADC → voltage → current → optical power, with 0.5× differential-mode correction).

```python
from ultracoustics import adc_to_uw
power_uw = adc_to_uw(samples, baseline=samples.mean())
```

| Parameter | Default | Description |
|---|---|---|
| `samples` | — | uint16 array of raw ADC values (0–16383). |
| `baseline` | `0.0` | ADC-count DC offset to subtract before conversion (e.g. dark-current baseline). |

---

## Probe Characterization

`Controller.run_probe_characterization(...)` runs the bulk-only probe-characterization ramp over the master USB bulk connection (no USB-serial connection to the 638 required) and returns the binned LI curve. Override / power / trigger sequencing is handled internally and is **not** exposed.

The capture must fit in the ring buffer in one piece, so construct the `Controller` with a large enough `ring_seconds` (the method raises a clear `RuntimeError` quoting the required value if it is too small).

```python
from ultracoustics import Controller

ctrl = Controller(verbose=True, ring_seconds=12)  # >= ramp (~9 s) + margin
ctrl.connect()
ctrl.begin_stream()

result = ctrl.run_probe_characterization()
# result.current          -> [0, 100, ..., 33000]  (DAC setpoints)
# result.photodetector     -> mean ADC counts per setpoint
# result.saturated         -> True if the PD saturated mid-ramp (curve trimmed)

ctrl.end_stream()
ctrl.close()
```

| Parameter | Default | Description |
|---|---|---|
| `max_current` | `33000` | DAC setpoint the ramp tops out at (must match firmware `PROBE_RAMP_MAX`). |
| `step_size` | `100` | DAC increment per bin (must match firmware; host cannot change it). |
| `bin_seconds` | `0.025` | One ramp tick in seconds (25 ms at the firmware 40 Hz rate). |
| `start_offset_s` | `0.0` | Forward-bin offset (s) for the unknown 40 Hz tick phase; correct within ~1 bin. |
| `laser_warmup_s` | `10.0` | 1550 turn-on transient settle time before the 638 ramp begins. |
| `boot_settle_s` | `10.0` | 638 boot / rail settle time after power-on. |
| `capture_guard_s` | `0.7` | Extra capture beyond the ramp for startup + stop latency. |
| `saturation_threshold` | `0.95` | Fraction of `ADC_MAX_VALUE` treated as PD saturation; aborts + trims. |
| `verbose` | `Controller.verbose` | Per-call verbose override. |

Returns a `ProbeCharacterizationResult` (`current`, `photodetector`, `saturated`, `sample_rate`, `bin_seconds`, `step_size`, `max_current`; `.as_dict()` for a JSON-friendly view).

**Post-analysis helpers** (pure functions, no hardware needed — operate on a `ProbeCharacterizationResult` or a saved curve):

| Function | Description |
|---|---|
| `bin_ramp(ramp, ...)` | Time-bin a captured ramp array into setpoint → mean-PD pairs. |
| `load_laser_calibration(path)` | Load a laser current → optical power (mW) calibration CSV; `None` if missing. |
| `apply_laser_calibration(probe_data, laser_cal)` | Interpolate laser current → optical power (mW). |
| `detect_resonance_dips(optical_power_mw, photodetector, ...)` | Find resonance dips via `scipy.signal.find_peaks`. |
| `analyze_dips(optical_power_mw, photodetector, dip_indices)` | Summarize dips (depth %, spacing, baseline). |

---

## Examples

Run examples from the repository root after installation:


python examples/basic_capture.py
python examples/offline_psd.py
python examples/probe_characterization.py

`probe_characterization.py` produces a CSV + plot; the plot needs the optional `plot` extra:

```bash
pip install -e ".[plot]"
```

## Troubleshooting

ModuleNotFoundError:

- Ensure the virtual environment is activated.
- Ensure installation was run from the repo root with `pip install -e .`.

Access denied on Linux:

- Configure a udev rule for USB access, or run with elevated privileges.

Device not found:

- Verify the board is powered and the USB cable supports data.
- Confirm the driver assignment in Device Manager.

If automatic Windows driver setup fails:

1. Download and run Zadig: https://zadig.akeo.ie
2. Plug in the Master Board.
3. In Zadig, open Options -> List All Devices.
4. Select BROADSONIC (VID:PID 2E9D:000A).
5. Select WinUSB and click Replace Driver.

