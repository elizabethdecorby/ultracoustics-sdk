# Analysis guide

Everything here works on a saved recording, with no box. The code runs on the
sample recording in `examples/data`: 0.1 s of a probe in a quiet room, with the
calibration taken at that START.

```python
import numpy as np
import ultracoustics as uc

rec = uc.load_recording("examples/data/sample.sigmf-meta")
cal = rec.calibration
fs = rec.sample_rate          # 10 MS/s
```

## 1. Counts and optical power

The box samples a photodetector at 10 MS/s with a 14-bit ADC. `rec.counts()`
gives the raw counts (0 to 16383). `rec.power_uw()` gives optical power in µW,
which is counts × `rec.uw_per_count` (about 0.0084 µW per count). The mean is
the light reaching the detector at the lock point. Sound is the small
fluctuation on top of it.

```python
x = rec.power_uw()                     # µW
part = rec.power_uw(0.02, 0.03)        # 10 ms of it, by time in seconds
print(f"{x.mean():.1f} µW mean, {x.std():.3f} µW rms fluctuation")
```

Recordings stay on disk until they are read, so a long one costs no memory
until you ask for a stretch of it.

## 2. Spectra

```python
spectrum = rec.spectrum(fft_size=16384)
spectrum.freqs, spectrum.psd_w2hz      # Hz, W²/Hz
spectrum.num_averages                  # FFTs averaged: 61 here
spectrum.sigma                         # one standard deviation of each bin
```

- **Bin width** is `sample_rate / fft_size`: 610 Hz at 16384 points.
- **Averaging.** Every whole Hann-windowed FFT in the range is averaged, and
  none crosses a gap in the recording. One bin scatters by
  `psd / sqrt(num_averages)`, which `sigma` gives.
- **Part of a recording.** `rec.spectrum(16384, t0_s=0.02, t1_s=0.05)`.
- **Narrow bands at fine bins.** `band=(lo, hi)` decimates to the band first,
  so `rec.spectrum(2**18, band=(30e3, 50e3))` gives 38 Hz bins quickly.
- **Peaks.** `estimator="multitaper"` gives lower scatter for the same data, at
  slightly wider peaks.

## 3. From optical power to pressure

The probe's dome moves with sound, and the box reads that motion as optical
power. The dome also moves on its own, from thermal motion, by an amount set by
physics alone. Each START, `ctrl.calibrate_pressure()` fits this thermal motion
in a quiet room and so measures the probe's sensitivity. That fit is the
**calibration**. Recordings keep it, as `rec.calibration`. A calibration file
saved by the GUI loads with `uc.load_calibration(path)`.

```python
cal.fundamental.f0_hz                  # the dome's first resonance: 2.09 MHz here
cal.sensitivity(spectrum.freqs)        # W²/Hz of optical power per Pa²/Hz of sound
cal.nep2                               # noise-equivalent pressure², Pa²/Hz
pressure = cal.to_pressure(spectrum)   # a PressureSpectrum, Pa²/Hz
```

- The sensitivity peaks at the dome's resonances. Below about 500 kHz it is
  nearly flat.
- `cal.nep2` is the thermal noise floor in pressure units, the quietest sound
  the probe can resolve in 1 Hz. On the sample it is 4e-10 Pa²/Hz, which is
  0 dB SPL in 1 Hz.
- `to_pressure` keeps the calibrated band only, up to the 4.5 MHz anti-alias
  filter. It warns if the optical power has moved by more than 10 % since the
  calibration, because the scale depends on it.
- Away from resonance the sensitivity is lower, so the same detector noise
  reads as more pressure. The noise budget (section 5) shows where that happens.

## 4. Three numbers in dB SPL

"How loud is it" has three answers. Each has its own function:

| Quantity | Unit | Function | Use for |
|---|---|---|---|
| Density | dB SPL in 1 Hz (dB SPL/√Hz) | `pressure.spl` | Noise and broadband sound. It does not depend on the FFT size. |
| Band level | dB SPL | `pressure.band_level(lo, hi)`, `pressure.octave_levels()` | The total sound in a band |
| Tone level | dB SPL | `uc.tone_level(rec, cal, f)` | One steady tone |

```python
pressure.spl                                   # density at each bin
level = pressure.band_level(20e3, 100e3)       # a BandLevel
level.spl_db, level.p_rms_pa
octaves = pressure.octave_levels(fraction=3)   # third-octave bands, IEC 61260-1
octaves.center_hz, octaves.level_db
tone = uc.tone_level(rec, cal, 40e3, bandwidth_hz=100)    # a ToneLevel
```

They are related through bandwidth. A flat density of D dB SPL/√Hz across a
band B Hz wide gives a band level of D + 10·log10(B). A pure tone has all its
power at one frequency, so its density depends on the bin width and means
nothing on its own: read it with `tone_level`, or with a band level over a few
bins around it.

A band level adds whole bins. A band narrower than two bins has
`resolved == False` and is only indicative: use a larger `fft_size`.

## 5. Removing noise

Four methods remove different things:

| Method | Removes | Works on | Limits |
|---|---|---|---|
| Measured background subtraction | Everything present in a quiet capture | Averaged spectra | Needs a quiet capture at the same optical power |
| Model subtraction | The detector's noise and the fitted thermal noise | Averaged spectra | Only above about 500 kHz |
| Band-pass or notch filter | Noise outside the signal's band, pickup lines | Waveforms | Keeps whatever is in band |
| Averaging repeated events | Anything that does not repeat, thermal noise included | Waveforms | Needs repeatable events |

**Thermal noise cannot be filtered out of a single waveform.** It moves the
dome exactly as sound does, so in pressure units it is flat at NEP² across the
band. It can only be subtracted from averaged power, reduced by narrowing the
band, or averaged down over repeated events.

### Measured background

Capture the same probe with the source off, at the same optical power, and
subtract power from power:

```python
background = rec.spectrum(16384, t0_s=0.0, t1_s=0.05)    # source off
measured = rec.spectrum(16384, t0_s=0.05, t1_s=0.1)      # source on
clean = measured.subtract(background)
real = clean.significant(sigma=5)        # bins truly above the background
cal.to_pressure(clean).band_level(140e3, 160e3)
```

Both spectra need the same `fft_size`. Bins that come out negative stay
negative: they are scatter, and clipping them would bias a band level upward.
`examples/remove_background.py` adds a known tone and reads it back.

### The noise model

The calibration's fit describes the noise as

    S(f) = dark(f) + G(f) · (shot + white + thermal(f) + broad(f))

where `dark` is the electronics with the lasers off, `shot` the photon noise of
the light, `white` any further flat noise such as laser noise, `thermal` the
dome's motion, `broad` broad features of the readout's response and `G` the
anti-alias filter.

```python
budget = cal.noise_budget(spectrum)
budget.dark, budget.shot, budget.white, budget.thermal, budget.broad   # W²/Hz
budget.limited_by_at(1e6)        # 'shot': what sets the floor at 1 MHz
budget.noise_spl                 # the whole model in dB SPL/√Hz
residual = budget.residual()     # spectrum less model, NaN below 500 kHz
```

Where `limited_by` says `thermal`, the probe is at its physical limit. Below
about 500 kHz the measured noise rises above the model, so `residual()` gives
NaN there and a measured background is the way to go.

### Filters

```python
y = uc.band_pass(rec.power_uw(), (20e3, 200e3), fs)    # 4th-order Butterworth, zero phase
y = uc.notch(y, 50e3, fs, q=30)                         # one pickup line
```

Both run forwards and backwards, so they add no delay.

### Repeated events

When the same event repeats, align the copies on their trigger and average.
Anything that does not repeat, thermal noise included, falls as 1/√N:

```python
events = uc.find_events(rec, uc.LevelTrigger(level_uw=0.3, band_hz=(20e3, 200e3)))
same = [e.uw() for e in events if e.samples.size == events[0].samples.size]
average = np.mean(same, axis=0) if same else None
```

## 6. Waveforms in pressure

```python
pa = cal.pressure_waveform(rec.power_uw(), band=(20e3, 200e3), sample_rate=fs)
scale, at_hz = cal.pa_per_uw((20e3, 200e3))     # Pa per µW, and where it was taken
```

A waveform gets one scale factor, so it holds only where the sensitivity is
flat: below about 500 kHz on the probes measured. It warns if the sensitivity
swings more than 1 dB across the band, as it does near a resonance. Then use a
narrower band, or `tone_level` for a tone.

## 7. Tones

```python
tone = uc.tone_level(rec, cal, 40e3, bandwidth_hz=100)
tone.spl_db, tone.p_rms_pa, tone.bandwidth_hz
```

`tone_level` is a lock-in at exactly one frequency. Its noise is the density
times `bandwidth_hz`, so a narrower bandwidth reads weaker tones. It also needs
a longer recording: at least 1.5 / `bandwidth_hz` seconds, so 0.15 s for the
default 10 Hz. The sample is 0.1 s, hence 100 Hz here. A tone at a frequency
between FFT bins reads correctly, with no correction for the window. The GUI's
per-bin view reads the same tone through the window's bandwidth.

## 8. Compared with a healthy machine

```python
healthy = rec.spectrum(16384, t1_s=0.05)
now = rec.spectrum(16384, t0_s=0.05)
change_db = now.difference_db(healthy)       # dB at each bin
```

A ratio needs no calibration, because the sensitivity cancels. It needs the
same probe, at about the same optical power.
