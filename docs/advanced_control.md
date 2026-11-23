# Advanced control

Most scripts need one call to start the readout box: `ctrl.start_ready()`. This
page is for the few that need more: the start-up one stage at a time, the full
calibration of the box's control settings, the signal level, and changing
probes.

**A word on names.** The box keeps its laser in tune with the sensor using a
control loop. The rest of the documentation calls the result "ready". The API
calls it the **lock**: the box is "locked" when the loop is holding, and the
loop's settings are the "lock gains". This page uses the API's words, because
they are in the names of the calls: `check_lock`, `LockReport`, `LockError`,
`calibrate_lock`, `lock_gains`, `ctrl.locked`, `require_lock`.

## What `start_ready()` runs

`ctrl.start_ready()` is these calls, in this order:

```python
import ultracoustics as uc

def main():
    with uc.Controller() as ctrl:
        dark = ctrl.capture_dark()                 # lasers-off reference: only possible before start()
        ctrl.start()                               # lasers on. Returns before the box is ready.
        report = ctrl.check_lock(retune=True)      # waits for a settled lock, checks the gains, corrects them
        if not report.ready:
            raise SystemExit(report.message)
        ctrl.calibrate_pressure(dark)              # this START's pressure calibration
        rec = ctrl.capture(1.0)

if __name__ == "__main__":
    main()
```

`start_ready()` adds these things to them:

- It stops the box first if it is running.
- It turns the lasers off and raises `uc.StartError` on any failure, including
  a command the box did not answer. With the separate calls that is yours to
  do.
- It notices a signal too strong for the detector and says to set the signal
  level (`err.signal_too_strong`).
- It carries on, with `report.calibrated == False`, if only the pressure
  calibration fails. `ctrl.calibrate_pressure()`, with no arguments, then
  tries again on the same dark reference without restarting.

[How it works](how_it_works.md#start-up) says what each failure leaves behind,
and [what happens to the lasers](how_it_works.md#when-the-script-ends) however
a script ends.

Use the separate calls when you need one of these:

| You need | Use |
|---|---|
| To be ready without the check's 1.6 s disturbance | `ctrl.start()`, then `ctrl.wait_ready()` |
| To check the gains but never change what is stored on the box | `ctrl.check_lock()` without `retune` |
| To keep the dark and calibrate later, or again | `ctrl.capture_dark()` before `start()`, `ctrl.calibrate_pressure(dark)` whenever the surroundings are quiet |
| A progress display or a cancel button | `progress` and `cancel`, which `start_ready()` takes too |
| A faster next start | `ctrl.warm()`: standby with the lasers off and temperatures held. See below. |
| A probe with its own constants | `ctrl.calibrate_pressure(dark, probe=uc.ProbeParameters(...))`. `start_ready()` uses the standard constants. |

The order matters. `capture_dark()` needs the lasers off, so it comes before
`start()`. `calibrate_pressure()` needs a settled lock and quiet surroundings.
`check_lock()` and `calibrate_lock()` are refused while a recording or a
monitor is running, because they disturb the signal.

**After `ctrl.warm()`**, start again with `ctrl.restart()` or
`ctrl.start_ready()`. `ctrl.start()` does nothing when this script has already
started the box, and a box in standby still counts as started. `warm()` is
sent without waiting for the box's answer.

## The lock check

The lock gains depend on the probe and can change from one START to the next.

`ctrl.check_lock(retune=True)` is what `start_ready()` and the Ultracoustics
GUI run at every START. It waits for a settled lock, confirms the gains suit
this probe and this start, and corrects them if they do not. Keep the probe still and away from
noise while it runs, ideally before the monitored machine starts. It disturbs
the signal for about 1.6 s. If conditions are too noisy it reports that the
gains could not be measured, leaves them unchanged, and you can run it again at
the next START.

Without `retune` it only measures: it never changes the gains. With
`retune=True`, when the gains need it, it applies the recommended gains,
verifies them, stores them on the box and restarts it, which takes about a
minute. Gains that fail are taken back, so a run that cannot improve the lock
leaves it as it was.

```python
report = ctrl.check_lock(retune=True)
report.status, report.message      # "ok", "retuned", "inconclusive", ...
report.ready                       # a settled lock to take data on
report.ok                          # and its gains were measured to be good
```

What each `report.status` means:

| Status | Means |
|---|---|
| `ok` | The gains are good. |
| `improvable` | The gains are good, and a retune would make the lock faster. |
| `needs_retune` | The gains are not safe for this probe: retune them. |
| `retuned` | `retune=True` found better gains, verified them and stored them. |
| `calibrated` | The full calibration designed gains, verified them and kept them. |
| `not_verified` | The calibration's gains failed their checks. The old ones are back. |
| `rejected` | The calibration's measurement was not good enough. Nothing changed. |
| `inconclusive` | Too noisy to measure. The gains are unchanged. |
| `not_ready` | No settled lock to measure. Do not take data yet. |
| `cancelled` | `cancel()` stopped it. |
| `failed` | It could not run. `message` says why. |

`ctrl.wait_ready()` waits for a settled lock without disturbing the signal: the
box locked for 10 s, the last 5 s of them quiet. It does not measure the gains.
It raises `uc.LockError` if no settled lock came within `timeout_s`, 90 s by
default.

`ctrl.locked` is `True` while the box reports a lock. It says there is one, not
how good it is.

## The full calibration

`ctrl.calibrate_lock()` is more thorough than the check, and longer. It
measures the loop for about a minute, designs the gains from the measurement
and verifies them: about two minutes. It then stores them on the box, which
takes about a minute more, so about three minutes in all. Run it for a new
probe, or when the check keeps asking for a retune. Keep the probe still and
the surroundings quiet. Gains that fail their checks are never kept.

```python
report = ctrl.calibrate_lock()     # the box must be running: after start() or start_ready()
print(report.status, report.message)
ctrl.lock_gains                    # the gains running now: kp, ki_per_s, zero_hz and where they came from
ctrl.restore_lock_gains()          # drop gains that were not stored: back to the box's
```

To remove the gains stored on the box, stop it first:

```python
ctrl.stop()
ctrl.clear_stored_lock_gains()     # the factory gains from the next START
```

With `store=False` the new gains last until the next STOP. `ctrl.lock_gains`
gives a `uc.LockGains`, whose `source` is `"live"` (set this START and not
stored), `"stored"` (kept on the box) or `"default"` (the factory gains). The
gains are reported, never set by hand: choosing Kp and Ki by hand is a service
task.

`check_lock()` and `calibrate_lock()` also take `progress` (a function called
as it goes, for a display), `cancel` (a function that returns `True` to stop
early) and `report_dir` (a folder to keep the measurement in, if support asks
for it).

## The signal level

The 1550 nm laser runs at a power setting stored on the box. If it is too high
for a probe, the reflected light saturates the detector and the box cannot get
ready. The setting is chosen once per probe. These calls are the GUI's
Diagnostics > Adjust signal level. Each takes a few minutes, runs the lasers,
needs nothing recording or monitoring, and leaves the box stopped.

```python
level = ctrl.measure_signal_level()     # sweeps the laser and recommends a setting. Changes nothing.
print(level.describe())                 # read it before you store
stored = ctrl.store_signal_level(level) # stores it, and checks the box comes back at it
print(stored.verified)                  # True: the box came back at the new setting
```

- `start_ready()` tells you when to run it: its `uc.StartError` says the signal
  is too strong, with `err.signal_too_strong` set. So does the self-test:
  `report.recommends_signal_adjustment` is `True` when the detector saturated
  during the test.
- `measure_signal_level()` raises `uc.SignalLevelError` when no setting fits
  this probe: contact Ultracoustics. It returns `None` if a `cancel` function
  you passed stopped it.
- Storing changes a setting kept on the box. Do it by hand, once, because a
  person decided to, not automatically inside a measurement script. It cannot
  be cancelled.
- If `stored.verified` is `False`, the box did not come back at the level the
  sweep found: run the self-test.
- Call `ctrl.start_ready()` again afterwards.

## Changing probes

The box's settings and every calibration belong to one probe. When you fit a
different one:

1. Stop the box: `ctrl.stop()`, or end the script.
2. Unplug the 9 VDC supply before you touch the optical connectors. Connect
   the new probe as in [Getting started](getting_started.md#1-set-up-the-hardware),
   then plug the supply in again.
3. Start with `ctrl.start_ready()`, from a new script or a new `Controller`.
   If it says the signal is too strong, set [the signal level](#the-signal-level)
   for this probe and start again.
4. For a probe new to this box, run [the full calibration](#the-full-calibration),
   `ctrl.calibrate_lock()`, then `ctrl.start_ready()` once more.
5. Take a new background after that start.

Never reuse a pressure calibration from another probe. A `Controller` keeps
its last calibration when a later start's calibration fails, so after a probe
change use a new `Controller`, or check `report.calibrated` and stop if it is
`False`.

## While the box runs

- The stream starts at `connect()`, so `ctrl.buffer` shows the signal before
  START, with the lasers off.
- `ctrl.telemetry` has the lasers and temperatures while the box runs.
  `ctrl.request_telemetry()` asks a stopped box for them too.
- `fault = ctrl.stop()` says why, if the box had already stopped itself (the
  probe signal lost, say): `fault.code` and `fault.reason`, or `None`.
- `ctrl.capture(seconds, require_lock=False)` and
  `ctrl.record(path, duration_s=60, require_lock=False)` take data whether or not the box is
  locked. `ctrl.monitor(..., require_lock=False)` watches regardless.

[How it works](how_it_works.md) explains what the box does on its own and what
these calls decide. The [reference](reference.md#the-lock) lists every name.
