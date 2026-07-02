#!/usr/bin/env python3
"""
Probe Characterization example for the Ultracoustics Python SDK.

Runs the bulk-only probe-characterization ramp via
:meth:`ultracoustics.Controller.run_probe_characterization`, saves the
resulting LI curve to CSV, optionally applies a laser-power calibration, and
plots it (with resonance-dip analysis if a calibration file is present).

This is the *opinionated* layer: serial-number prompt, output paths, CSV
format, and plotting all live here. The acquisition and binning logic is in
the SDK. Copy this file and edit the paths / plotting to taste.

Requirements
------------
- The SDK installed (``pip install -e .``).
- For the plot: the ``plot`` extra (``pip install -e ".[plot]"``). Capture
  and CSV save work without it.

Usage
-----

    python examples/probe_characterization.py
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from ultracoustics import (
    Controller,
    load_laser_calibration,
    apply_laser_calibration,
    detect_resonance_dips,
    analyze_dips,
)

# Plotting is optional — capture + CSV save work without matplotlib.
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:  # pragma: no cover
    plt = None
    _HAS_MPL = False


# ----------------------------------------------------------------------------
# Configuration — edit these to match your setup.
# ----------------------------------------------------------------------------

# Ring buffer must hold the whole ~9 s sweep in one piece (see SDK docs).
RING_SECONDS = 12.0

# Output directories / files.
PROBE_DATA_DIR = Path("probe_data")
LASER_CALIBRATION_DIR = Path("laser_calibration")
# Update this path if you retake the calibration.
LASER_CALIBRATION_CSV = LASER_CALIBRATION_DIR / "laser_power_cal_638#004.csv"


# ----------------------------------------------------------------------------
# CSV output
# ----------------------------------------------------------------------------

def save_csv(result, probe_serial):
    """Save a ProbeCharacterizationResult to a timestamped CSV."""
    PROBE_DATA_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = PROBE_DATA_DIR / f"probe_char_{probe_serial}_{timestamp}.csv"

    with open(filename, "w") as f:
        f.write("# Probe Characterization (bulk-only)\n")
        f.write(f"# Probe Serial: {probe_serial}\n")
        f.write(f"# Date: {datetime.now().isoformat()}\n")
        f.write(f"# Max Current: {result.max_current}\n")
        f.write(f"# Step Size: {result.step_size}\n")
        f.write(f"# Bin: {result.bin_seconds*1e3:.0f} ms "
                f"@ {result.sample_rate/1e6:.0f} MSPS\n")
        f.write("#\n")
        f.write("current_setpoint,photodetector_avg\n")
        for curr, pd in zip(result.current, result.photodetector):
            f.write(f"{int(curr)},{pd:.2f}\n")

    print(f"\n✓ Data saved: {filename}")
    return filename


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------

def plot_characterization(result, probe_serial, optical_power_mw, csv_filename):
    """Plot the LI curve, calibrated curve, and resonance dips."""
    if not _HAS_MPL:
        print("⚠ matplotlib not installed — skipping plot. "
              "Install with: pip install -e \".[plot]\"")
        return None

    current = np.array(result.current)
    photodetector = np.array(result.photodetector)
    power = np.array(optical_power_mw) if optical_power_mw is not None else None

    dip_data = None
    if power is not None:
        dip_indices, _ = detect_resonance_dips(power, photodetector)
        if len(dip_indices) > 0:
            dip_data = analyze_dips(power, photodetector, dip_indices)
            print(f"\n  Detected {len(dip_indices)} resonance dips")

    plt.style.use("seaborn-v0_8-darkgrid")
    if power is not None and dip_data is not None:
        fig = plt.figure(figsize=(20, 6))
        ax1 = plt.subplot(1, 3, 1)
        ax2 = plt.subplot(1, 3, 2)
        ax3 = plt.subplot(1, 3, 3)
    elif power is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        ax3 = None
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        ax2 = None
        ax3 = None

    ax1.plot(current, photodetector, "b-", linewidth=1.5, label="Measured Data")
    ax1.set_xlabel("Laser Current DAC Value")
    ax1.set_ylabel("Photodetector Reading (ADC counts)")
    ax1.set_title("Laser Current vs Photodetector Response",
                  fontsize=12, fontweight="bold", pad=15)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, framealpha=0.9)
    stats1 = (f"Probe Serial: {probe_serial}\n"
              f"Data Points: {len(current)}\n"
              f"DAC Range: {current.min():.0f} - {current.max():.0f}\n"
              f"PD Range: {photodetector.min():.0f} - {photodetector.max():.0f}")
    ax1.text(0.02, 0.98, stats1, transform=ax1.transAxes, fontsize=9,
             verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8,
                       edgecolor="gray"))

    if power is not None:
        ax2.plot(power, photodetector, "r-", linewidth=1.5, label="Calibrated")
        ax2.set_xlabel("638nm Optical Power (mW)")
        ax2.set_ylabel("Photodetector Reading (ADC counts)")
        ax2.set_title("Optical Power vs Photodetector Response",
                      fontsize=12, fontweight="bold", pad=15)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10, framealpha=0.9)

    if ax3 is not None and dip_data is not None:
        ax3.plot(power, photodetector, "g-", linewidth=1.5, alpha=0.7,
                 label="Response Curve")
        ax3.plot(dip_data["power_mw"], dip_data["pd_counts"], "ro", markersize=8,
                 label=f"Detected Dips ({len(dip_data['power_mw'])})", zorder=5)
        for i, (pwr, pd) in enumerate(zip(dip_data["power_mw"],
                                          dip_data["pd_counts"])):
            ax3.annotate(f"#{i+1}", xy=(pwr, pd), xytext=(5, 5),
                         textcoords="offset points", fontsize=8,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow",
                                   alpha=0.7))
        ax3.set_xlabel("638nm Optical Power (mW)")
        ax3.set_ylabel("Photodetector Reading (ADC counts)")
        ax3.set_title("Resonance Dip Analysis", fontsize=12, fontweight="bold",
                      pad=15)
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=9, loc="upper left")

    fig.suptitle(f"Probe Characterization - Serial: {probe_serial}",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    png_filename = str(csv_filename).replace(".csv", ".png")
    plt.savefig(png_filename, dpi=300, bbox_inches="tight")
    print(f"✓ Plot saved: {png_filename}")
    plt.close()
    return png_filename


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("PROBE CHARACTERIZATION SYSTEM (BULK-ONLY)")
    print("=" * 60 + "\n")

    probe_serial = input("Enter Probe Serial Number: ").strip()
    if not probe_serial:
        print("Error: Probe serial number is required")
        return 1
    print(f"\nProbe Serial: {probe_serial}")

    ctrl = Controller(verbose=True, ring_seconds=RING_SECONDS)
    try:
        print("\nConnecting to Master Board (Controller)...")
        ctrl.connect()
        ctrl.begin_stream()
        print("Streaming PD data.")

        # One SDK call does override/power/trigger/ramp/binning.
        result = ctrl.run_probe_characterization()

        csv_file = save_csv(result, probe_serial)

        laser_cal = load_laser_calibration(LASER_CALIBRATION_CSV)
        if laser_cal is None:
            print(f"\n⚠ Laser calibration not found: {LASER_CALIBRATION_CSV}\n"
                  "  Plot 2/3 will not be generated.")
            optical_power_mw = None
        else:
            print(f"\n✓ Loaded laser calibration ({len(laser_cal['current'])} pts)")
            optical_power_mw = apply_laser_calibration(result, laser_cal)

        print("\nGenerating characterization plot...")
        plot_file = plot_characterization(result, probe_serial,
                                          optical_power_mw, csv_file)

        print("\n" + "=" * 60)
        print("CHARACTERIZATION COMPLETE")
        print("=" * 60)
        print(f"Probe Serial: {probe_serial}")
        print(f"Data CSV: {csv_file}")
        print(f"Plot PNG: {plot_file}")
        if result.saturated:
            print("⚠ Measurement stopped early due to saturation.")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        try:
            ctrl.end_stream()
        except Exception:
            pass
        try:
            ctrl.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
