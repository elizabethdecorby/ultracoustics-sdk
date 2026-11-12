# Monitoring

A monitor watches every sample of the live stream for one or more triggers. It
is built to run unattended, so it says what it did: the events it caught, the
alarms it raised, the samples it lost and the time it was not watching.

```python
import time
import ultracoustics as uc

def main():
    with uc.Controller() as ctrl:
        dark = ctrl.capture_dark()
        ctrl.start()
        ctrl.check_lock(retune=True)       # before the monitor: it disturbs the signal
        ctrl.calibrate_pressure(dark)      # for limits in pressure

        healthy = ctrl.capture(5.0).spectrum()
        watch = [uc.BandPowerTrigger((20e3, 100e3), threshold_db=10, baseline=healthy, name="bearing"),
                 uc.LevelTrigger(level_pa=1.0, band_hz=(2e3, 20e3), name="impacts")]
        monitor = ctrl.monitor(watch, folder="pump_3", max_gb=50, on_alarm=print, on_status=print)
        time.sleep(3600)
        monitor.stop()
        print(monitor.summary())

if __name__ == "__main__":
    main()
```

`examples/monitor.py` is this, for a minute.

## Triggers

**`BandPowerTrigger(band_hz, threshold_db=10)`** fires when the power in a band
rises `threshold_db` above normal. What counts as normal is one of three things:

| Setting | Normal is | Use for |
|---|---|---|
| Neither of the two below (default) | Learned for `learn_seconds` (1 s) on arming, then following slow drift with time constant `track_seconds` (10 s) | Catching sudden events on a machine whose level wanders |
| `baseline=spectrum` | A `Spectrum` of the machine when healthy, held for good | Catching a machine slowly getting worse |
| `level_spl=80` | An absolute limit in dB SPL for the band. Needs a calibration. `threshold_db` is then not used. | Limits set by a specification |

A learned, tracking baseline follows a slowly growing fault too, and learns one
that is already there when it arms as normal. The other two know what normal is
in advance, so they catch both.

The band is measured with short FFTs, sized for about four bins across the
band: 0.2 ms frames for wide bands, up to 6.6 ms for narrow ones. Frames overlap
by 75 %, so an event is caught wherever it falls.

**`LevelTrigger(level_uw=1.0, edge="rising")`** fires when the optical power
crosses a level, measured from its slowly moving mean. `edge` is `"rising"`,
`"falling"` or `"either"`. `band_hz=(lo, hi)` filters the signal first, for
the trigger only. `level_pa` gives the level in pressure instead, which needs a
calibration and `band_hz`, because the sensitivity depends on frequency.

Every trigger has a `name`, used in events, alarms and the trend log, and a
hysteresis: after firing, the level must fall back below the limit less the
hysteresis before it can fire again.

## Events and alarms

- An **event** is a stretch of data around a trigger: `pre_s` before it and
  `post_s` after. The trigger then stays off for `holdoff_s`, `post_s` by
  default.
- An **alarm** is a condition that stays above its limit. It is raised once,
  when a trigger first fires, and clears after the level has stayed below the
  limit less the hysteresis for 1 s.

While an alarm is raised its trigger captures no further events, so a fault that
persists gives one alarm and one event, not one event per holdoff. Pass
`events_during_alarm=True` to capture them all.

`on_event(event)`, `on_alarm(change)` and `on_status(text)` are called from the
monitor's thread as things happen. Keep them short. `monitor.active_alarms`
names the alarms raised now.

## The lock

While the lock is lost there is nothing to watch: events then would only be the
lock moving. The monitor pauses, says so through `on_status`, and counts the
time. When the lock returns it waits 1 s for the signal to settle, then carries
on. An event fired as the lock was being lost is dropped with it. Pass
`require_lock=False` to watch regardless.

So a monitor can start before the lock settles. Run `ctrl.check_lock()` before
starting it, though: the check's 1.6 s burst is refused while a monitor runs.

## The run folder

With `folder="pump_3"`, each run gets its own subfolder, such as
`pump_3/20261109_141500/`:

| File | Holds |
|---|---|
| `trend.csv` | A row every `trend_s` (1 s): UTC, lock, mean optical power, rms, peak, crest factor, kurtosis, temperature, events so far, samples lost, time not watched, and each trigger's highest level and alarm state |
| `alarms.csv` | A row each time an alarm is raised or cleared, with its level and limit |
| `events_0001.sigmf-meta`, `-data` | The events, a new file every 1 GiB. See [saving data](saving_data.md). |
| `events_0001_events.csv` | One row per event: time, trigger, peak, dominant frequency, band level, rms, crest factor, kurtosis |
| `summary.json` | Written at the end: the same as `monitor.summary()` |

`trend.csv` is often the main output of a long run. A machine getting slowly
worse shows up there before any trigger fires.

`max_gb` limits the event files in all. Past it no more events are saved, and
they are counted in the summary's `events_not_saved`. Nothing is ever deleted,
and watching and logging go on.

## Event features

Each event carries `features`, also in the events CSV:

| Feature | Meaning |
|---|---|
| `peak_uw`, `peak_ms` | The largest excursion from the baseline, and when, from the trigger |
| `dominant_hz` | The strongest frequency after the trigger, inside the trigger's band if it has one |
| `band_db` | For a band trigger: the band's level above normal, or in dB SPL with `level_spl` |
| `rms_uw` | The rms after the trigger |
| `crest` | Peak over rms: about 4 to 5 for noise, 1.4 for a sine, higher for impacts |
| `kurtosis` | 3 for random noise, 1.5 for a sine, high for impacts and bursts |

## Did it watch everything?

```python
s = monitor.summary()
s.complete            # every second watched, nothing lost or unsaved, no error
s.samples_lost        # the computer fell behind the stream
s.not_monitored_s     # paused for the lock
s.not_watched         # (UTC start, seconds) of each pause
s.events, s.events_not_saved, s.alarms_raised, s.alarms_active
```

## Tuning limits offline

`find_events` replays the same triggers over a saved recording, exactly as a
live monitor would. Tune limits there, with no box:

```python
rec = uc.load_recording("pump_3_one_hour")
events = uc.find_events(rec, watch, pre_s=0.01, post_s=0.1)
```

`examples/find_events.py` adds three bursts to the sample recording and finds
them.

## Limits

- **Triggering runs on the computer, not in the box.** The computer must stay
  connected and keep up. A level trigger costs little. Each band trigger takes
  about 0.3 s of one core per second of data, and bands with the same FFT size
  share their FFT. Check `samples_lost` after a trial run: anything lost is
  counted, never skipped silently.
- **Limits in dB SPL read short-time peaks.** `level_spl` compares each short
  frame with the limit, and steady noise reaches about 6 dB above its band
  level in single frames. Set the limit at least 6 dB above the quiet level. A
  loud tone reads within 0.2 dB of its level.
- **Thresholds against a learned baseline.** On quiet data, 8 and 10 dB gave no
  false triggers, 6 dB about one a second. Start at 10 dB.
- **A learned baseline is blind while it learns.** It listens for
  `learn_seconds` after arming, and checks from the next 0.1 s block. It keeps
  what it learned through pauses for the lock. A healthy `baseline` or
  `level_spl` checks from the start.
- **Pressure limits assume a flat sensitivity across the band.** A warning
  says when it swings more than 1 dB, as near a resonance: then the limit is
  right only near the band's centre.
- **Events must fit in the live ring.** `pre_s + post_s` must be less than the
  Controller's `ring_seconds` (1.2 s by default).
