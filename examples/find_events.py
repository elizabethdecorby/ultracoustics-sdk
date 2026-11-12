"""Tune a trigger on saved data before watching a machine live. Runs with no box.

    python examples/find_events.py

find_events replays the same triggers a live monitor runs. Here three 2 ms
bursts at 50 kHz are added to the sample recording in software, and the
recording itself is the healthy baseline.
"""
from pathlib import Path

import numpy as np

import ultracoustics as uc

rec = uc.load_recording(Path(__file__).parent / "data" / "sample.sigmf-meta")
fs = rec.sample_rate
healthy = rec.spectrum()    # the machine when healthy: what the trigger compares with

x = rec.counts().astype(np.float64)
t = np.arange(int(0.002 * fs)) / fs
burst = 0.5 / rec.uw_per_count * np.sin(2 * np.pi * 50e3 * t) * np.hanning(t.size)    # 0.5 µW peak
for at_s in (0.02, 0.05, 0.08):
    start = int(at_s * fs)
    x[start:start + burst.size] += burst
test = uc.Recording("with bursts", x, fs, "recording", calibration=rec.calibration)

trigger = uc.BandPowerTrigger((20e3, 100e3), threshold_db=10, baseline=healthy, name="50 kHz band")
for event in uc.find_events(test, trigger, pre_s=0.001, post_s=0.005, holdoff_s=0.01):
    f = event.features
    print(f"{event.trigger_index / fs * 1e3:6.2f} ms: {f['trigger']}, +{f['band_db']:.1f} dB, "
          f"{f['dominant_hz'] / 1e3:.1f} kHz, crest factor {f['crest']:.1f}, kurtosis {f['kurtosis']:.1f}")
