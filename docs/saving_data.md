# Saving data

## Ways to save

| Call | Saves | Length |
|---|---|---|
| `ctrl.capture(seconds)`, then `.save(path)` | The next stretch, every sample, with the lock checked throughout | What memory holds |
| `ctrl.snapshot(seconds)`, then `.save(path)` | What just happened, from the live ring | Up to `ring_seconds` (1.2 s by default) |
| `ctrl.record(path, duration_s=...)` | The stream, written straight to disk as it arrives | Any |
| `ctrl.monitor(..., folder=...)` | A stretch around each trigger, and a log of levels | Any |
| `rec.save(path, t0_s, t1_s)` | A recording, or the part between two times | |

Nothing is ever overwritten: saving to a name that already has a recording
raises `FileExistsError`. `path` is a name such as `"runs/test_3"`, or either
file's path.

`capture()` refuses data it cannot vouch for: it raises `CaptureError` if a
sample was lost, or the lock was not confirmed from start to finish. `record()`
keeps going and marks what was lost in the file.

## The files

Every recording is a [SigMF](https://sigmf.org) pair:

- `name.sigmf-data`: the raw ADC counts as little-endian unsigned 16-bit
  integers (SigMF datatype `ru16_le`), 14 bits used. 10 MS/s is 20 MB per
  second.
- `name.sigmf-meta`: JSON describing the data.

Any tool can read the samples: `numpy.fromfile("name.sigmf-data", "<u2")`.
`uc.load_recording(path)` reads both files, without loading the samples into
memory.

`record()` splits long recordings into 2 GiB files of about 107 s each:
`name`, `name_001`, `name_002` and so on. Opening any of them opens the whole
recording, joined in order.

## The metadata

The `global` object holds what applies to the whole recording.

| Field | Meaning |
|---|---|
| `core:sample_rate` | 10000000 |
| `core:datatype`, `core:version` | `ru16_le`, SigMF 1.2.0 |
| `core:recorder` | `Ultracoustics SDK`, or the GUI with its version |
| `core:description`, `core:hw` | What it is, and `BROADSONIC` |
| `broadsonic:kind` | How it was made: `capture`, `snapshot`, `continuous` or `events` from the SDK. The GUI adds its own. |
| `broadsonic:adc_bits` | 14 |
| `broadsonic:uw_per_count` | Optical power per ADC count, µW |
| `broadsonic:system_state` | `RUNNING` or `IDLE` when it was made. The GUI writes its own state names. |
| `broadsonic:locked` | Whether the box reported a lock when it was made |
| `broadsonic:box_serial`, `broadsonic:sdk_version` | Where it came from |
| `broadsonic:probe` | The probe's constants: dome mass and radius, temperature, any acoustic correction, name |
| `broadsonic:calibration` | The pressure calibration, with a reduced dark spectrum. See below. |
| `broadsonic:stream_drops` | The stream's loss counters when it began |
| `broadsonic:stream_drops_end` | The same counters at the end of each file of a `record()` |
| `broadsonic:triggers` | A monitor's triggers and their settings |
| `broadsonic:trimmed_from` | For a part saved from a longer recording: the source and its times |
| `broadsonic:display_dc_removed` | GUI only: whether its display removed the mean |
| `broadsonic:trigger` | GUI only: the trigger and capture window of an event file |

The SigMF `captures` list starts a new segment wherever the stream jumps,
after a gap or at each event. Each segment has:

- `core:sample_start`: its first sample in this file.
- `core:global_index`: the same sample's index in the stream since it began.
- `core:datetime`: the UTC time of that sample, from the computer's clock.

The `annotations` list marks places in the data:

- `gap`: samples lost at this point, counted in `broadsonic:lost_samples`.
- `trigger`: an event, with `broadsonic:event` (its number),
  `broadsonic:pre_trigger_samples`, `broadsonic:baseline_counts` and the
  event's features (see [monitoring](monitoring.md)).

## Gaps

Samples can be lost in two places:

- **Before they reach the computer.** The stream's counters rise. `record()`
  reports these in its result's `dropped`, and the file's
  `broadsonic:stream_drops_end`.
- **On the computer**, when the disk falls behind. `record()` writes a `gap`
  annotation and counts the samples in its result's `lost`. The next block
  starts a new segment at its true stream index.

A recording knows its gaps: `rec.gaps` lists `(sample index, samples
missing)`. Spectra never average across a gap, and `rec.spans()` gives the
unbroken stretches.

## Events

A monitor with a `folder` writes its events to `events_0001.sigmf-data` and
`-meta`, starting `events_0002` past 1 GiB. Each event is a segment of its own,
with a `trigger` annotation. `events_0001_events.csv` lists them as a table.

```python
events = uc.load_recording("runs/20261109_141500/events_0001.sigmf-meta")
for event in events.events:
    event.number, event.utc, event.features
    event.uw()            # µW about the baseline the trigger measured from
```

## Calibrations in files

`broadsonic:calibration` holds the START's pressure calibration: the fitted
dome modes, the noise-equivalent pressure, the noise terms and a 512-bin dark
spectrum. That is everything needed for SPL and the noise budget, from the
recording alone. It is about 30 KB.

`rec.calibration` gives it back. If a file holds a calibration this SDK cannot
use, `rec.calibration` is `None`, `load_recording` warns, and
`rec.calibration_problem` says why.

## Raw `.bin` files

Older captures were raw samples with no metadata. They load too, with the rate
given, but carry no calibration:

```python
rec = uc.load_recording("old_capture.bin", sample_rate=10e6)
rec.calibration = uc.load_recording("same_start.sigmf-meta").calibration
rec.save("old_capture")          # now a SigMF recording with its calibration
```

Only give a recording a calibration from the same probe at the same START.
