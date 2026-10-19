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

**Negotiated telemetry**

Legacy 16,392-byte ADC records remain the default. While the system is IDLE
and the reader is active, `ctrl.enable_telemetry()` verifies capabilities,
selects format 1, validates the firmware acknowledgement and returns the first
52-byte-per-board diagnostic snapshot. `ctrl.telemetry` then returns the latest
immutable snapshot without changing the 8,192-sample `uint16` ring. Use
`ctrl.disable_telemetry()` to return explicitly to legacy framing. Transport
delivery and telemetry validity do not acknowledge actuator or setpoint
application.

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

### `adc_to_uw(samples, baseline=0.0, responsivity=None, transimpedance=None) → np.ndarray`

Convert raw 14-bit ADC counts to optical power in µW, applying the full signal chain: ADC → voltage → ÷ ADA4940 gain → ÷ R_f → photocurrent → ÷ responsivity → optical power.

The two readout channels differ in **both** the photodiode and the TIA feedback resistor, so converting 638 nm data needs both values passed:

| Channel | Responsivity | Transimpedance | Full scale (16383 counts) |
|---|---|---|---|
| 1550 nm (InGaAs) | 1.077 A/W | 20 kΩ | ≈ 138 µW |
| 638 nm (Si visible) | 0.3 A/W | 10 kΩ | ≈ 993 µW |

```python
from ultracoustics import adc_to_uw
from ultracoustics.processing import TRANSIMPEDANCE_638

# 1550 nm — the defaults already match this channel
power_uw = adc_to_uw(samples, baseline=samples.mean())

# 638 nm — pass both channel constants
power_uw = adc_to_uw(samples, baseline=samples.mean(),
                     responsivity=0.3, transimpedance=TRANSIMPEDANCE_638)
```

| Parameter | Default | Description |
|---|---|---|
| `samples` | — | uint16 array of raw ADC values (0–16383). |
| `baseline` | `0.0` | ADC-count DC offset to subtract before conversion (e.g. dark-current baseline). |
| `responsivity` | `1.077` | Detector responsivity in A/W (== µA/µW). |
| `transimpedance` | `20_000` | TIA feedback resistance in ohms. Use `TRANSIMPEDANCE_638` (10 kΩ) for the 638 nm channel. |

---

## Test mode (manual sessions)

Every bench test that drives a laser DAC runs in one mode: the master in
override, and the laser boards' 1-second leases on each channel the PC owns.
`docs/TEST_MODE.md` has the design. Each laser is either off, PC-controlled
(START low, leased channels) or automatic (START high, its own firmware).

- `begin_system_manual(timeout_s=1.0, optical_diagnostics=False,
  keepalive_ms=2000)` powers both boards with START low and both DACs taken
  at 0. Its last step arms the master keepalive (command `K`): if the PC then
  sends nothing for `keepalive_ms`, the master powers everything down
  (white ×4). A master without `K` refuses the session.
- `renew_system_manual()` renews every lease, which also feeds the
  keepalive. Call it at least every 0.5 s.
- `start_1550_automatic(warmup_s=10.0)` hands the 1550 back to its own
  firmware inside the session. While its START is high the master watches
  the 1550 and the probe signal, and refuses manual commands to the 1550.
- `take_over_638(dac)` turns a box locked in RUN into a session: override in
  place, the 638 START lowered, and the 638 DAC taken at `dac`. The 1550 keeps
  running on its own.
- `wait_tec_lock(target)` waits for a TEC to hold within ±0.3 °C for 1 s.
- `finish_system_manual()` ends any session with a confirmed STOP.
  `last_master_fault` then names the master fault that ended it, if any (for
  example `"probe signal lost"` or `"PC keepalive expired"`).

Each `ManualReply` carries `stream_sample`: the number of samples received
when the reply arrived. The master sends the reply after the board applied
the value, on the same endpoint as the ADC records, so it marks the change in
the sample stream. `round_trip_ns` is the host time from request to reply.
`examples/measure_manual_timing.py` measures both on the bench.

## Probe Sweep

`Controller.run_probe_sweep(...)` steps the 638 DAC from 0 to the board's cap
over the master USB connection and averages the photodetector at each step.
It opens and always closes its own test-mode session: the 1550 runs on its
own firmware at its stored setpoint, and each 638 SET must come back with the
requested value. The average at each step covers `settle_s` to `dwell_s`
after the reply's stream position.

```python
from ultracoustics import Controller

ctrl = Controller(ring_seconds=2.0)
ctrl.connect()
ctrl.begin_stream()

result = ctrl.run_probe_sweep()
# result.current          -> [0, 100, ..., 52400]  (applied 638 DAC)
# result.photodetector     -> mean ADC counts per DAC
# result.saturated         -> True if the PD saturated (curve ends there)
# result.temperature_c     -> 638 TEC temperature per point

ctrl.end_stream()
ctrl.close()
```

| Parameter | Default | Description |
|---|---|---|
| `step` | `100` | DAC increment. The top is always included. |
| `dwell_s` | `0.025` | Time at each DAC. |
| `settle_s` | `0.005` | Start of the average after the step. |
| `top` | board cap | Highest DAC (never above the cap read from channel 7). |
| `warmup_s` | `10.0` | 1550 warm-up after its TEC locks. |
| `saturation_threshold` | `0.95` | Fraction of `ADC_MAX_VALUE` that stops the sweep. |
| `cancel`, `on_progress` | `None` | Polled between points / called after each point. |

Returns a `ProbeCharacterizationResult` (`current`, `photodetector`,
`saturated`, `sample_rate`, `bin_seconds` (the dwell), `step_size`,
`max_current`, `settle_seconds`, `temperature_c`; `.as_dict()` for a
JSON-friendly view). With a 25 ms dwell a full sweep takes about 15–40 s plus
the warm-up, depending on the command round trip.

**Post-analysis helpers** (pure functions, no hardware needed — operate on a `ProbeCharacterizationResult` or a saved curve):

| Function | Description |
|---|---|
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
python examples/measure_manual_timing.py

`probe_characterization.py` runs a probe sweep and produces a CSV + plot; the
plot needs the optional `plot` extra:

```bash
pip install -e ".[plot]"
```

## Troubleshooting

### Firmware-paced 638 FP scan

For a 638 board advertising manual channels 7–9, select optical diagnostics
before entering manual override. The standard manual setup sets both lasers to
zero, so explicitly command a nonzero 1550 DAC for illumination before the
scan. The SDK checks the acknowledged 1550 readback; it transfers the 638 zero
DAC lease to optical control, runs the firmware-paced scan, and returns the 638
to zero/IDLE. The scan ramps to the board's cap (channel 7, at most 52400).

```python
ctrl.begin_system_manual(optical_diagnostics=True)
ctrl.system_manual_command(1550, MANUAL_SET, CHANNEL_LASER_DAC, illuminated_dac)
result = ctrl.capture_fp_scan(cancel=cancel_requested, on_progress=show_progress)
rows = result["rows"]  # commanded_dac and main_pd_adc_counts
```

The `MANUAL_SET` and `CHANNEL_LASER_DAC` constants are available from
`ultracoustics._internal.control`. `capture_fp_scan` accepts a 75-second
maximum deadline and keeps manual leases alive. The firmware steps the DAC
every 1 ms (1000 steps) and marks the start and end of the ramp in the
stream; the SDK places those markers on the ADC samples (to within 300 µs),
splits the samples between them into 1000 equal bins, and labels bin `i`
with `cap * i / 999`. That label is an estimate from the firmware timing,
not an analog DAC readback, and `raw_adc` keeps the full-rate samples. The
result includes capture ID, sample rate, duration, DAC cap, the marker
bounds and quality flags. A canceled or failed scan raises `FPScanError`
with any rows already binned in `.rows`.

`read_optical_cap_638`, `read_controller_timing_638`,
`set_controller_timing_638(divider, held)`, and `read_fp_scan_638` expose the
new capability, runtime hold/rate, and scan status words. Timing accepts
dividers 1, 2, 3, 5, and 10 and can be changed in automatic RUN.

### Runtime normalized PI and box profile

`read_normalized_pi_638()` returns `kp`, `ki_per_s`, `active`, `overridden`,
`slope_valid`, and protocol `version`. Call `set_normalized_pi_638(kp,
ki_per_s)` to change both gains atomically, or `restore_normalized_pi_638()`
to recover the compiled box defaults. These calls require a healthy locked
normalized PI path with a valid measured slope; a rejected request does not
switch to the legacy gain law. The host accepts finite Kp from 0 to 1 and Ki
from 0 to 1638.3 per second on current firmware; older firmware may reject
Ki above 1000/s. Wire resolution is 0.001 Kp and 0.1/s Ki. Active
overrides are volatile and survive bounded recovery/reacquisition; full
IDLE/Stop resets them. Existing `optical_pid_638()` setters are legacy manual
gain controls and select the legacy law.

The 638 manual protocol uses channel 10 for one packed gain pair:
`(round(Kp * 1000) << 14) | round(Ki_per_s * 10)`. `GET` and successful `SET`
return the complete pair in the ACK's signed 24-bit value; decode it with
`value & 0xffffff` before unpacking. Channel 11 `GET` reports contract
version 1 in bits 0–7, normalized law active in bit 8, override in bit 9,
and slope valid in bit 10. `SET 0` on channel 11 restores profile defaults.
Channel 12 is a read-only numeric box profile: `1` Belycomm and `2`
QPhotonics. `read_profile_638()` resolves it to a stable profile name; this
ID is not a firmware version.

These channels require updated master and 638 firmware. The SDK probes
channel 11 before tuning and reports unsupported firmware or a failed
capability probe explicitly. In automatic RUN the master permits these
bounded 638 requests; in manual mode, release a manually owned 638 laser DAC
before optical tuning. Run these synchronous SDK calls from a worker when
using a graphical interface.

### Local slope readiness telemetry compatibility

`OpticalLive.slope_readiness` decodes the packed quiet-gate status only when
live flag `0x20` is present; it returns `None` for older pages. The packed
status requires 638 firmware commit `c5cf633` and master firmware commit
`7cf06c31121a250aa123d7bc837b25d6ee0e0e9d` or later. Earlier masters
reject the new flag and discard those live pages. Flag `0x10` still means
PI held, and the live page length and layout are unchanged.

### Bounded PI characterization report

`run_pi_characterization(controller, report_dir, progress=None, cancel=None,
verified_master_filter=None, driver_pole_hz=None, driver_pole_range_hz=None,
driver_pole_source=None)`
uses an **existing** connected, streaming Controller with optical diagnostics
format 2 and a healthy automatic 638 lock. It does not start or stop the
system, connect a second session, command a robot, or apply PI gains. Call it
from a GUI worker that exclusively owns Controller commands. `progress`
receives a dictionary with `stage`, `message`, and `elapsed_s`; `cancel` is a
zero-argument predicate. The return value and `summary.json` have a `status`
of `complete`, `unqualified`, or `cancelled`, and a reason when appropriate.

The routine checks profile identity (Belycomm 1 or QPhotonics 2), runtime PI,
DAC cap and headroom, 10 kHz timing, configurable-trace support, fresh lock,
and up to 120 seconds of settling. It requests one 4096-row, D4, hold4,
8-DAC trace and waits at most 240 seconds for indexed replay.
During settling only, a fresh firmware `MEASURE_SLOPE` state is allowed for
at most five continuous seconds; the 15-second stability history and 20-second
minimum stable time restart on return to `LOCKED`. Other states still reject,
and the pre-capture/capture checks require `LOCKED` throughout. It keeps ADC
counter changes during the capture separate from replay changes. The report
directory contains a concise `report.md`, machine-readable `summary.json`,
and a compressed `trace.npz` of just the bounded control trace; partial replay
rows are saved as `trace-partial.npz` after failure or cancellation. No full
10 MSPS ADC stream is saved. The measured FRFs, model diagnostics when
supported, and exploratory PI candidate screen are in the JSON report. Pass
`verified_master_filter="single_sample"` only after independently verifying
the installed master filter is OFF; otherwise the report withholds gain
screening and retains measured FRFs. These candidates are local
evidence for review, not an automatic gain recommendation. The installed
master feedback filter must be verified to be `single_sample`; the SDK cannot
read that setting, so the report records the caller's assertion. The driver
pole is also installation specific: pass an independent nominal
`driver_pole_hz` in 300–5000 Hz and a nonempty `driver_pole_source` naming
the measurement, board revision/BOM, or explicit estimate. The optional
`driver_pole_range_hz=[low, high]` must contain the nominal value and stay
inside 300–5000 Hz; omitted bounds use nominal ±25%, clipped to that range.
The reported first-order fit and robust model screen are conditional on this
prior. With no pole prior, the routine still captures and reports the FRFs
and can show candidates only when both empirical halves satisfy the strict
measured-band gates. The driver and thermal poles cannot be distinguished
from this trace alone, and no pole is inferred from the box or laser brand.
If the current PI crossover is below the measured band, its measured-band
sensitivity can still serve as the baseline comparator; every proposed pair
must have a resolved crossover in both halves. Empirical phase margins are
point estimates, not confidence-bound stability margins.
A cancellation
returns promptly without stopping the shared controller; an excitation
already started in firmware may finish after the caller returns.

`review_saved_pi_characterization(report_dir, driver_pole_hz=...,
driver_pole_range_hz=..., driver_pole_source=...)` reanalyzes a complete
saved `trace.npz` and `summary.json` without hardware or modifications to the
original report. It checks the saved capture integrity and returns measured
coverage, model diagnostics, and any qualified exploratory screen. This
allows a later board measurement or schematic review to supply the driver
pole without repeating the laser excitation.

For a standalone session, `python examples/pi_characterization.py REPORT_DIR
--start-and-stop --verified-single-sample-filter` explicitly starts and then
stops the laser system with the filter assertion. Omit the filter flag to
collect only measured FRFs. Use `--driver-pole-hz`, optional
`--driver-pole-low-hz` and `--driver-pole-high-hz`, and
`--driver-pole-source` to supply an independent prior. Do not
run that program concurrently with a GUI owning the same device.

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
