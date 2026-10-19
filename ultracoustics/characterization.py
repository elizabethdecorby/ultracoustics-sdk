"""
Probe sweep result type and post-analysis.

This module holds the pure-computation half of the probe sweep run by
:meth:`ultracoustics.controller.Controller.run_probe_sweep`. Splitting it out
keeps the acquisition logic (which owns the USB device) in the
:class:`Controller` and the analysis logic (which only touches numpy / scipy)
here, where it can be unit-tested and reused on saved data without any
hardware connected.

Functions
---------
load_laser_calibration(path)
    Load a laser current -> optical power calibration CSV.
apply_laser_calibration(probe_data, laser_cal)
    Interpolate laser current to optical power (mW).
detect_resonance_dips(optical_power_mw, photodetector, ...)
    Find resonance dips in a characterization curve.
analyze_dips(optical_power_mw, photodetector, dip_indices)
    Summarize detected dips (depth, spacing, baseline).

Classes
-------
ProbeCharacterizationResult
    Structured return type of :meth:`Controller.run_probe_sweep`.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ProbeCharacterizationResult:
    """Outcome of a probe sweep.

    Attributes
    ----------
    current : list[int]
        638 DAC values applied, as the board confirmed them (0, 100, ...).
    photodetector : list[float]
        Mean photodetector reading (ADC counts) at each DAC, in the same
        order as ``current``.
    saturated : bool
        ``True`` if the photodetector saturated and the sweep stopped early
        (the curve ends at the saturated point).
    sample_rate : int
        ADC sample rate in Hz.
    bin_seconds : float
        Time the DAC was held at each value (the dwell).
    step_size : int
        DAC increment between points.
    max_current : int
        Highest DAC the sweep was asked to reach (the board's cap by default).
    settle_seconds : float or None
        Start of each average after the step; the average covers
        ``settle_seconds`` to ``bin_seconds`` after the board applied it.
    temperature_c : list[float] or None
        638 TEC temperature at each point, from telemetry.
    """

    current: List[int]
    photodetector: List[float]
    saturated: bool
    sample_rate: int
    bin_seconds: float
    step_size: int
    max_current: int
    settle_seconds: Optional[float] = None
    temperature_c: Optional[List[float]] = None

    def as_dict(self) -> Dict[str, Any]:
        """Return a plain-dict view (JSON-serializable)."""
        return {
            "current": list(self.current),
            "photodetector": list(self.photodetector),
            "saturated": self.saturated,
            "sample_rate": self.sample_rate,
            "bin_seconds": self.bin_seconds,
            "step_size": self.step_size,
            "max_current": self.max_current,
            "settle_seconds": self.settle_seconds,
            "temperature_c": (None if self.temperature_c is None
                              else list(self.temperature_c)),
        }


# ---------------------------------------------------------------------------
# Laser power calibration
# ---------------------------------------------------------------------------

def load_laser_calibration(calibration_csv_path) -> Optional[Dict[str, np.ndarray]]:
    """Load a laser current -> optical power calibration from CSV.

    The CSV may carry its power column in W or mW; the header is scanned for
    ``"mw"`` and the values are normalized to mW either way.

    Parameters
    ----------
    calibration_csv_path : path-like
        Path to the calibration CSV.

    Returns
    -------
    dict or None
        ``{"current": np.ndarray, "optical_power_mw": np.ndarray}``, or
        ``None`` if the file does not exist.
    """
    cal_path = Path(calibration_csv_path)
    if not cal_path.exists():
        return None

    cal_current: List[float] = []
    cal_power: List[float] = []
    is_mw = False
    with open(cal_path, "r") as f:
        for _ in range(10):
            if "mw" in f.readline().lower():
                is_mw = True
                break
    with open(cal_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            if line.startswith("current,") or line.startswith("dac,"):
                continue
            parts = line.split(",")
            if len(parts) == 2:
                cal_current.append(float(parts[0]))
                cal_power.append(float(parts[1]))

    cal_current_arr = np.array(cal_current)
    cal_power_mw = np.array(cal_power) if is_mw else np.array(cal_power) * 1000.0
    return {"current": cal_current_arr, "optical_power_mw": cal_power_mw}


def apply_laser_calibration(probe_data, laser_cal) -> np.ndarray:
    """Interpolate laser current (DAC) -> optical power (mW).

    Parameters
    ----------
    probe_data : dict or ProbeCharacterizationResult
        Characterization data carrying a ``current`` key/field (DAC values).
    laser_cal : dict
        Calibration dict from :func:`load_laser_calibration`.

    Returns
    -------
    numpy.ndarray
        Optical power in mW at each setpoint in ``probe_data``.
    """
    current = (probe_data.current
               if isinstance(probe_data, ProbeCharacterizationResult)
               else probe_data["current"])
    power_interpolator = interp1d(
        laser_cal["current"], laser_cal["optical_power_mw"],
        kind="linear", fill_value="extrapolate",
    )
    return power_interpolator(np.asarray(current))


# ---------------------------------------------------------------------------
# Resonance dip analysis
# ---------------------------------------------------------------------------

def detect_resonance_dips(
    optical_power_mw,
    photodetector,
    prominence_threshold: float = 0.15,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Find resonance dips in a characterization curve.

    A dip is a local minimum in the photodetector response. Dips are found by
    inverting the curve and calling :func:`scipy.signal.find_peaks` with a
    prominence scaled to the signal's dynamic range.

    Parameters
    ----------
    optical_power_mw : array-like
        Optical power axis (mW) — used only to scale the min-distance.
    photodetector : array-like
        Photodetector response (ADC counts), same length as ``optical_power_mw``.
    prominence_threshold : float
        Dip prominence as a fraction of the signal's peak-to-peak range.

    Returns
    -------
    (indices, props)
        ``indices`` is an int array of dip locations; ``props`` is the
        ``find_peaks`` properties dict.
    """
    photodetector = np.asarray(photodetector, dtype=np.float64)
    inverted = -photodetector
    signal_range = photodetector.max() - photodetector.min()
    absolute_prominence = prominence_threshold * signal_range
    min_distance = max(1, len(optical_power_mw) // 50)
    dip_indices, dip_props = find_peaks(
        inverted, prominence=absolute_prominence,
        distance=min_distance, width=1,
    )
    return dip_indices, dip_props


def analyze_dips(optical_power_mw, photodetector, dip_indices) -> Optional[Dict[str, Any]]:
    """Summarize detected resonance dips.

    Parameters
    ----------
    optical_power_mw : array-like
        Optical power axis (mW).
    photodetector : array-like
        Photodetector response (ADC counts).
    dip_indices : array-like
        Dip locations from :func:`detect_resonance_dips`.

    Returns
    -------
    dict or None
        ``None`` if no dips; otherwise a dict with ``indices``, ``power_mw``,
        ``pd_counts``, ``depth_pct``, ``spacing_mw``, and ``initial_value``
        (the baseline PD level taken as the median of the first 5% of points).
    """
    optical_power_mw = np.asarray(optical_power_mw)
    photodetector = np.asarray(photodetector)
    dip_indices = np.asarray(dip_indices)

    if len(dip_indices) == 0:
        return None

    baseline_idx = int(len(photodetector) * 0.05)
    initial_value = np.median(photodetector[:baseline_idx])
    dip_power = optical_power_mw[dip_indices]
    dip_pd = photodetector[dip_indices]
    depth_pct = ((initial_value - dip_pd) / initial_value) * 100
    spacing_mw = np.diff(dip_power) if len(dip_power) > 1 else np.array([])
    return {
        "indices": dip_indices, "power_mw": dip_power, "pd_counts": dip_pd,
        "depth_pct": depth_pct, "spacing_mw": spacing_mw,
        "initial_value": initial_value,
    }
