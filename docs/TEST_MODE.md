# Single closed-box test mode

Status: **built and bench-tested.** Planned and approved on 2026-10-19 (all
eight decisions at the end), then implemented the same day on a `test-mode`
branch in four repos: `Photodetector-PSSIandUSBHS` (master firmware),
`ultracoustics-sdk`, `Ultracoustics-Internal_Software` (diagnostics app) and
`638-Laser-Firmware`, and committed locally (not pushed). The user reported
it working on the bench on 2026-10-20.

The separate mainboard for the 638 noise test runs a variant master build
with no 1550 and no fans: see `VISIBLE_PD_BOARD.md`.

## Goal

Run every bench test that drives a laser DAC from one box mode, over the
master USB cable only, with the master's safety checks still running:

- 1550 and 638 laser power, laser noise and laser temp (these already used
  the mode; they gained safety checks and stream-aligned samples)
- probe sweep (moved off the 638's 40 Hz self-ramp)
- optical lock step response (moved off the second USB cable into the 638)

Out of scope: lock stability stays in normal RUN with the readout GUI, and
the GUI's 1 ms FP scan stays as it is. It already runs inside this mode.

## Before this change

| Test | Who set the 638 DAC | Box mode | Cables | Master safety checks |
|---|---|---|---|---|
| 638 power, noise, temp | PC, one confirmed SET per point | override, manual session | master USB | none |
| Probe sweep | 638 firmware self-ramp, 40 Hz | override, legacy commands | master USB | none; no leases |
| Optical lock step | 638 lock loop, then PC over serial `S` / `L=` | RUN | master USB + 638 USB | RUN checks; 638 lock drop is amber only |

Override switched off every runtime check on the master:
`Boot_CheckRuntimeFaults` runs only in RUN, `Boot_CheckFanHealth` returned
early in OVERRIDE, and `Boot_CheckFastSample` only acts in A2/RUN. The master
has no watchdog. In override the only protection was each slave's 1-second
lease, which drops that laser's DAC to 0 when the PC stops renewing.

## The test mode

The test mode is the SDK's manual session: the master in OVERRIDE, and the
slaves' 1-second leases on every channel the PC owns. Inside it each laser
has one of three roles:

| Role | Rail | START line | Who drives the laser |
|---|---|---|---|
| Off | off | low | nobody |
| PC-controlled | on | low | the PC, through leased manual channels (SET confirmed by the applied value) |
| Automatic | on | high | the board's own firmware (1550 at its stored setpoint; 638 calibrates and locks) |

Two ways in, one way out:

- **Cold start** (`begin_system_manual`): confirmed STOP, then override, both
  START lines low, rails on, both DACs taken at 0, keepalive armed.
  `start_1550_automatic()` then hands the 1550 back to its own firmware.
- **Take over from lock** (`take_over_638(dac)`): boot normally to RUN, let
  the 638 lock, then switch to override in place and lower only the 638
  START. The rails stay on and the 1550 keeps running automatically.
- **Out:** `finish_system_manual()`, a confirmed STOP to IDLE (rails off),
  on every exit path. `last_master_fault` then says if the master had
  stopped the box itself.

| Test | Entry | 1550 | 638 |
|---|---|---|---|
| 1550 power / noise / temp | cold start | PC-controlled | PC-controlled at 0 |
| 638 power / noise / temp | cold start | PC-controlled at 0 | PC-controlled |
| Probe sweep | cold start | automatic | PC-controlled |
| Optical lock step | take over from lock | automatic | automatic, then PC-controlled at the lock DAC |

### Take-over sequence

Uses commands the master and 638 firmware already accepted; only the SDK
gained an entry point.

1. In IDLE, select stream format 2 (negotiable only in IDLE).
2. `S` (boot). Read the 638's live page until it reports LOCKED, then for the
   observation window (90 s by default). The median DAC over the window is
   the setpoint.
3. `O` (enter override). The master changes no GPIO; the rails and both
   START lines stay up, and the 1550 and probe-signal checks keep running.
4. `K` (arm the keepalive).
5. `T` 638 = 0 (lower the 638 START). The 638 goes to IDLE on the falling
   edge and its DAC passes through 0.
6. Wait for 638 IDLE on channel 3, then TAKE channel 2 at the median DAC.
7. Step with confirmed SETs on channel 2; renew the lease every 0.5 s.
8. Confirmed STOP.

### Lining DAC steps up with samples

The SDK keeps up to about 210 ms of USB transfers in flight, so the PC clock
and the sample stream can disagree by up to that much. Before this change the
noise sweep waited 0.1 s after the SET reply and took whatever samples
arrived next, which nothing guaranteed came after the step.

The master sends each manual reply (the 32-byte `UTMC` record) on the same
bulk IN endpoint as the ADC records, after the slave has applied the value.
libusb completes transfers on an endpoint in order, so the number of samples
received when the reply arrives marks where the change sits in the stream.
The reader now records it: `ManualReply.stream_sample`. `read_after_reply`
and `read_samples` read by that position, and the probe sweep, the noise
sweep and the optical step all use it. The expected uncertainty is about one
SPI frame plus one ADC record (roughly 2 ms); `examples/measure_manual_timing.py`
measures it.

### What the delays cost

| Step | Time |
|---|---|
| SDK sends the command | one command per reader-loop turn; a turn waits up to 50 ms for USB events, much less while the stream runs |
| Master relays it to the 638 | next NORMAL SPI frame, ~0.82 ms (1.22 kHz) |
| 638 applies it | microseconds; up to 150 ms only while a serial `B` command runs |
| 638 acknowledges it | next frame, ~0.82 ms |
| Reply reaches the PC | next bulk IN record, normally a few ms |

With stream positions, latency only stretches total time; it never moves
samples into the wrong bin. The probe sweep is 525 steps at a 25 ms dwell:
13.1 s of dwell plus 525 round trips, roughly 15–40 s, plus the 1550 warm-up.
Dwell is set by physics: the sensor heats in about 0.5 ms but has a slow tail
lasting tens of ms, which is why the 1 ms FP scan does not replace this sweep.

## Safety in the test mode

| Protection | Override before | Override now |
|---|---|---|
| Fan stall (2 s, FAULT, amber ×1) | off | on whenever the fans are enabled |
| 1550 drops ready (FAULT, blue ×3) | off | on while the 1550 rail is on and its START is high, once it has reported ready |
| Probe signal lost (FAULT, red ×3) | off | same condition, once the signal has been seen |
| Probe signal never seen (FAULT, red ×1) | off | same condition, 500 ms after the 1550 reports ready |
| PC gone (FAULT, white ×4, rails off) | none | `K` arms it; any USB command feeds it |
| Slave lease (DAC to 0 after 1 s) | on | unchanged |
| 638 temperature limits, DAC clamp at 52400 | on | unchanged |

Still unprotected, by design or out of scope:

- **A 1550 lit by the PC with its START low** (1550 power and noise, FP-scan
  illumination) gets no probe-signal check: those tests sweep the 1550 down
  to 0 on purpose.
- **A master HardFault or `Error_Handler`** leaves the rails on (existing
  behaviour; the DEV_DIAGNOSTICS build resets instead).
- **No hardware watchdog** on the master or the 638.

The keepalive is opt-in, so tools that sit in override and talk to a laser
over its own USB cable keep working (the app's `diagnostics_session`, the
SDK's `run_laser_noise_sweep` and `DiagnosticsManager`). They get the fan
check but not the keepalive.

The override probe-signal check uses the absolute 470-count threshold. Across
the last 40 probe sweeps in `Ultracoustics-Internal_Software/probe_data` the
deepest dip was 2,687 counts (32% of that run's level); the only runs below
470 were the 2026-09-10 runs at about 165 counts, with no light on the
detector.

## What was built

### Master firmware (`Photodetector-PSSIandUSBHS`)

- **Probe-ramp command removed:** `Z` / `CMD_PROBE_CHAR`,
  `SPI_SetCommandId`, `g_spi_command_id`; SPI frames always carry
  `CMD_PSSI_DATA`. Also removed from `libs/ultracoustics` and
  `tests/offline_protocol/control_trace_test.c`.
- **Override checks:**
  - `Boot_CheckFanHealth` now runs in OVERRIDE while `FAN1_EN` is high.
  - New `Boot_CheckOverrideFaults()` runs every main-loop pass. Its decision
    logic is the pure header `Boot/Core/Inc/override_monitor.h`
    (`OverrideMonitor_Step`).
  - `Boot_SignalFailureReason(need_dark)`: override has no dark baseline.
  - `Boot_CheckFastSample` publishes (but does not fault) in OVERRIDE, so the
    averaged check stays fresh across a RUN-to-override handover.
- **Keepalive:**
  - `K` (`CMD_KEEPALIVE`, wValue = timeout ms, 0 off, 200–65,535), accepted
    only in OVERRIDE.
  - Every USB command feeds it, and leaving OVERRIDE disarms it.
  - On expiry: new `BOOT_FAULT_HOST_TIMEOUT` (17), white ×4.
- **Docs:** `LED_CODES.md` (white ×4) and `FAN_HEALTH.md` (override
  behaviour).
- **Test:** new host test `tests/test_override_monitor.c`. It covers START
  gating, the acquire window, signal lost, 1550 dropped, disarm, tick wrap,
  and keepalive validity and expiry.

Sizes are the `.bin` image, which IAP writes and the 57,344 B limit applies
to. Built with Arm GNU 14.3.rel1, CMake Release, from `dd65688`:

| Build | Before | After | Change | Free |
|---|---|---|---|---|
| Release (`make build`) | 55,292 | 55,580 | +288 | **1,764** |
| DEV_DIAGNOSTICS | 56,876 | 57,172 | +296 | **172** |

The only warning is the existing linker RWX note. The new code adds 11 B of
RAM.

### SDK (`ultracoustics-sdk`)

- **Manual session:**
  - `begin_system_manual(..., keepalive_ms=2000)` arms `K` as its last step
    and refuses a master that rejects it.
  - `renew_system_manual` feeds the keepalive.
  - `stop_system_confirmed` records `last_master_fault`, and lease-renewal
    errors name it.
- **New methods:** `take_over_638(dac)`, `start_1550_automatic()`,
  `wait_tec_lock(target)`, `read_samples(start, end)` and
  `read_after_reply(reply, n, settle_s)`.
- **New fields:** `ManualReply.stream_sample` and `round_trip_ns`.
  `RuntimeMetrics.fault` gives the fault name, and `MASTER_FAULT_NAMES` has
  the table.
- **`run_probe_sweep(step=100, dwell_s=0.025, settle_s=0.005, ...)`:** the
  probe sweep lives in the SDK so the app and the scripts share it.
  `ProbeCharacterizationResult` gained `settle_seconds` and `temperature_c`.
- **Removed:** `run_probe_characterization`, `bin_ramp`, `CMD_PROBE_CHAR`
  and `DiagnosticsManager.start/stop_probe_characterization`.
- **Examples:** `examples/probe_characterization.py` now uses the new sweep.
  New `examples/measure_manual_timing.py` is the bench timing script.
- **README:** new test-mode and probe-sweep sections, and the FP-scan
  description corrected.
- **Tests:** new `test_test_mode.py` (17 tests), plus a stream-position check
  in `test_telemetry_stream.py`. The suite runs 161 tests, all passing apart
  from the 2 files that need pytest, which is not installed.

### Diagnostics app (`Ultracoustics-Internal_Software`)

- **Probe sweep:**
  - `ProbeSweepWorker` calls `run_probe_sweep`, with measured progress and a
    Cancel button.
  - The top defaults to the board's cap.
  - The CSV keeps its columns and gains dwell/settle header lines.
  - `probe_characterization_bulk.py` uses the new sweep.
- **Optical step:**
  - `OpticalStepWorker` boots the box, watches the lock on the format-2 live
    page, then `take_over_638` at the median.
  - Steps are confirmed SETs, and each capture window is placed by
    `stream_sample`.
  - The serial-port controls and the 638 cable are gone, and the console is
    now an event log.
  - From the pages: `scale_factor` is minimum / baseline from the acquisition
    page, and `k_local` and slope valid come from the live page. Fields only
    the serial `W` reply had (jitter, wind-up, slope attempts) stay empty in
    the report.
  - Each step records its measured command round trip.
  - The latency estimate now follows the master path
    (`manual_command_latency`).
- **Noise sweeps:** each block starts `settle_delay_s` after the reply's
  stream position.
- **Fault reporting:**
  - Power, noise, probe, optical and thermal failures append "master fault:
    …" when the master stopped the box.
  - `master_codes.py` knows codes 16 and 17 and the IDENTITY state.

The app compiles, and a simulated optical-step run against a fake controller
completes the whole sequence. There are no automated app tests.

### 638 firmware (`638-Laser-Firmware`)

- **Removed:** the 40 Hz probe ramp, which was the constants and flags, `Start`,
  `RequestStop`, `Stop`, `Tick`, `CheckMasterCommand`, their call sites, and
  `Optical_ProbeRampTick()` in the TIM6 callback.
- **Docs:** `THERMAL_SAFETY.md` scenario 7 removed. The README notes the
  serial `S` / `L=` limits (decision 8).
- **Size:** both presets build with no warnings.

| Preset | Before | After |
|---|---|---|
| qphotonics-release | 81,740 | 81,380 (−360) |
| belycomm-release | 81,588 | 81,228 (−360) |

## Bench checklist

Flash the master `test-mode` Release image and the matching 638 `test-mode`
image. The SDK is installed editable from its checkout, so the app picks up
the `test-mode` branch directly.

1. **Host test** (on a machine with a C compiler; this PC has none):
   `cc -std=c11 -Wall -Wextra -Werror -pedantic -IBoot/Core/Inc
   tests/test_override_monitor.c -o t && ./t`, and `make test` in
   `tests/offline_protocol`.
2. **Regression:** software checks, 1550 and 638 temp, power and noise all run
   as before and end with both rails off.
3. **Keepalive:** in a Python shell, `begin_system_manual()`, then call
   nothing for 3 s. Expect white ×4 and rails off. The slave leases drop
   the DACs at 1 s first.
4. **Fan:** start a 638 power test, unplug one fan. Expect amber ×1, rails
   off, and the app reporting "master fault: fan stall".
5. **Probe sweep:** run it on a probe with a known dip (Tom's earlier sweeps
   dipped to ~32% near DAC 21,000–24,000). Mid-way through a second run,
   unplug the probe fiber: expect red ×3 and "probe signal lost".
6. **Timing:** `python examples/measure_manual_timing.py --low A --high B`
   with A and B on a slope of the probe curve. If the largest positive edge
   offset plus ~0.5 ms exceeds 5 ms, raise `settle_s` (SDK default and
   `ProbeSweepWorker.SETTLE_S`). Note the round trips.
7. **Optical step** with nothing plugged into the 638: lock, watch, take-over,
   several steps, Return to lock, Stop. Compare the sensor bandwidth with
   earlier `optical_step_data` for the same unit.
8. **Lock stability** (readout GUI, RUN path) still works, and the GUI's FP
   scan and power-calibration dialog still work with the keepalive.

## Decisions (approved 2026-10-19)

| # | Question | Decision |
|---|---|---|
| 1 | Keepalive timeout the SDK arms | 2 s |
| 2 | Fan check whenever override has the fans on (a rig without fans faults within 2 s of the first `P`) | yes |
| 3 | Probe-signal check only while the 1550 START is high | yes |
| 4 | LED code for the keepalive fault | white ×4 |
| 5 | SDK refuses the test session on master firmware without `K` | yes |
| 6 | Probe sweep: 1550 automatic at its stored setpoint, step 100, dwell 25 ms, 0 to cap | yes |
| 7 | Remove the serial-cable optical step entirely | yes |
| 8 | Leave the 638 serial `S` / `L=` problems, note them in the 638 README | yes |
