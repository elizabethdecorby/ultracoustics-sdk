"""A saved recording, from raw counts to sound pressure level. Runs with no box.

    python examples/analyze_recording.py [a .sigmf-meta file]

docs/analysis_guide.md explains each step.
"""
import sys
from pathlib import Path

import ultracoustics as uc

path = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "data" / "sample.sigmf-meta"
rec = uc.load_recording(path)
print(f"{rec.duration_s:.2f} s at {rec.sample_rate / 1e6:g} MS/s, taken {rec.start_time}")

# The average power spectrum in W²/Hz. Every whole FFT is used, so one bin
# scatters by psd / sqrt(num_averages).
spectrum = rec.spectrum(fft_size=16384)
print(f"{spectrum.freqs.size} bins of {spectrum.freqs[1]:.0f} Hz, {spectrum.num_averages} averages")

# The recording's own calibration turns optical power into pressure.
cal = rec.calibration
pressure = cal.to_pressure(spectrum)

# Three different answers to "how loud is it"
i = abs(pressure.freqs - 1e6).argmin()
print(f"\ndensity at 1 MHz:   {pressure.spl[i]:.1f} dB SPL in a 1 Hz band")
band = pressure.band_level(20e3, 100e3)
print(f"level, 20-100 kHz:  {band.spl_db:.1f} dB SPL ({band.p_rms_pa * 1e3:.1f} mPa rms)")
print("octave bands:")
octaves = pressure.octave_levels(fraction=1, lo_hz=20e3, hi_hz=4e6)
for centre, level in zip(octaves.center_hz, octaves.level_db):
    print(f"  {centre / 1e3:7.1f} kHz  {level:5.1f} dB SPL")

# In a quiet room all of this is noise: what limits detection at each frequency?
budget = cal.noise_budget(spectrum)
print()
for f in (100e3, 1e6, cal.fundamental.f0_hz):
    print(f"at {f / 1e3:6.0f} kHz the floor is set by {budget.limited_by_at(f)} noise")
