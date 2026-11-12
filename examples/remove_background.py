"""Measure a weak sound by subtracting a measured background. Runs with no box.

    python examples/remove_background.py

The sample recording is a quiet room. Its first half serves as the background.
A 150 kHz tone at 70 dB SPL is added to the second half in software, so the
right answer is known.
"""
from pathlib import Path

import numpy as np

import ultracoustics as uc

TONE_HZ, TONE_SPL = 150e3, 70.0

rec = uc.load_recording(Path(__file__).parent / "data" / "sample.sigmf-meta")
cal, fs = rec.calibration, rec.sample_rate
half = rec.length // 2
counts = rec.counts().astype(np.float64)

# The tone, in ADC counts: pressure × sensitivity gives optical power
p_rms = 20e-6 * 10 ** (TONE_SPL / 20)
watts = p_rms * np.sqrt(cal.sensitivity(np.array([TONE_HZ]))[0])
amplitude = np.sqrt(2) * watts / (rec.uw_per_count * 1e-6)
t = np.arange(half) / fs
background = uc.Recording("background", counts[:half], fs, "recording")
measured = uc.Recording("measurement", counts[half:2 * half] + amplitude * np.cos(2 * np.pi * TONE_HZ * t), fs,
                        "recording")

# Subtract power from power, at the same bins and the same optical power
clean = measured.spectrum(16384).subtract(background.spectrum(16384))

# Which bins truly stand above the background, beyond its scatter?
found = clean.freqs[clean.significant(sigma=5)]
print(f"bins more than 5 sigma above the background: {np.round(found / 1e3, 1)} kHz")

df = clean.freqs[1]
level = cal.to_pressure(clean).band_level(TONE_HZ - 5 * df, TONE_HZ + 5 * df)
print(f"band level after subtraction: {level.spl_db:.1f} dB SPL (put in {TONE_SPL:.1f})")

# A lock-in reads one frequency exactly, and narrows the noise to its bandwidth
tone = uc.tone_level(measured, cal, TONE_HZ, bandwidth_hz=100)
print(f"lock-in reading:              {tone.spl_db:.1f} dB SPL")
