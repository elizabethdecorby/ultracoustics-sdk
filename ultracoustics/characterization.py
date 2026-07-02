"""
Probe characterization helpers – ramp binning and post-analysis.

This module holds the pure-computation half of the probe-characterization
workflow used by :meth:`ultracoustics.controller.Controller.run_probe_characterization`.
Splitting it out keeps the acquisition logic (which owns the USB device) in
the :class:`Controller` and the analysis logic (which only touches numpy /
scipy) here, where it can be unit-tested and reused on saved data without
any hardware connected.

Functions
---------
bin_ramp(ramp, sample_rate, ...)
    Time-bin a captured ramp waveform into (DAC setpoint -> mean PD) pairs.
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
    Structured return type of
    :meth:`Controller.run_probe_characterization`.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

from .config import SAMPLE_RATE, ADC_MAX_VALUE


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ProbeCharacterizationResult:
    """Outcome of a bulk-only probe-characterization sweep.

    Attributes
    ----------
    current : list[int]
        DAC setpoints swept (e.g. 0, 100, ..., 33000).
    photodetector : list[float]
        Mean photodetector reading (ADC counts) for each setpoint, in the
        same order as ``current``.
    saturated : bool
        ``True`` if the photodetector saturated mid-ramp and the sweep was
        aborted early (the curve is trimmed to end at the saturation point).
    sample_rate : int
        ADC sample rate in Hz the binning was performed at.
    bin_seconds : float
        Width of each time bin in seconds (one firmware ramp tick).
    step_size : int
        DAC increment per bin.
    max_current : int
        DAC setpoint the ramp tops out at (matches firmware PROBE_RAMP_MAX).
    """

    current: List[int]
    photodetector: List[float]
    saturated: bool
    sample_rate: int
    bin_seconds: float
    step_size: int
    max_current: int

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
        }


# ---------------------------------------------------------------------------
# Ramp binning
# ---------------------------------------------------------------------------

def bin_ramp(
    ramp,
    sample_rate: int = SAMPLE_RATE,
    bin_seconds: float = 0.025,
    step_size: int = 100,
    max_current: int = 33000,
    saturated: bool = False,
    saturation_limit: Optional[int] = None,
    start_offset_samples: int = 0,
) -> Dict[str, list]:
    """Time-bin a captured ramp waveform into setpoint -> mean-PD pairs.

    Forward-bins from the ramp origin: ``ramp`` is assumed to be exactly the
    samples captured from the moment the start command was issued (array
    index 0 == T0), so bin ``k`` covers samples
    ``[start_offset + k*bin_samples, start_offset + (k+1)*bin_samples)`` and
    is labelled setpoint ``k * step_size``.

    This is independent of the PD amplitude shape (no edge/threshold
    detection), which matters because resonance-dip probes do not produce a
    large stop transition to anchor against.

    Parameters
    ----------
    ramp : array-like
        Captured uint16 PD samples spanning [T0, end-of-capture].
    sample_rate : int
        ADC sample rate in Hz.
    bin_seconds : float
        Width of one bin in seconds (one firmware ramp tick, 25 ms at 40 Hz).
    step_size : int
        DAC increment per bin.
    max_current : int
        DAC setpoint the ramp tops out at; ``n_steps = max_current // step_size``.
    saturated : bool
        If ``True``, trailing bins at or above ``saturation_limit`` are
        trimmed so the curve ends cleanly at the saturation point.
    saturation_limit : int, optional
        ADC count treated as saturation; required when ``saturated`` is True.
    start_offset_samples : int
        Sample offset for the first bin (calibration fudge for the unknown
        40 Hz tick phase vs the host start command). Default 0 is correct
        within ~1 bin.

    Returns
    -------
    dict
        ``{"current": [...], "photodetector": [...]}`` with one entry per
        non-empty bin.

    Raises
    ------
    RuntimeError
        If the ramp is flat (ramp did not run / PD dead) or binning produces
        no points.
    """
    ramp = np.asarray(ramp)
    if float(ramp.max()) - float(ramp.min()) <= 0:
        raise RuntimeError("PD signal is flat — ramp did not run or PD is dead.")

    bin_samples = int(sample_rate * bin_seconds)   # 250_000 samples / 25 ms
    n_steps = max_current // step_size              # 330 steps (100..33000)

    setpoints: List[int] = []
    pd_vals: List[float] = []

    # Forward: k=0 -> setpoint 0, ..., k=n_steps -> setpoint max_current.
    for k in range(n_steps + 1):
        a = start_offset_samples + k * bin_samples
        b = a + bin_samples
        if b > ramp.size:
            break  # captured ramp ended early (e.g. saturation abort)
        seg = ramp[a:b]
        setpoints.append(k * step_size)
        pd_vals.append(float(np.mean(seg)))

    if not setpoints:
        raise RuntimeError("Binning produced no points — check ramp/PD signal.")

    # If we aborted for saturation, trim any trailing bins beyond the
    # saturation point so the curve ends cleanly at saturation.
    if saturated:
        if saturation_limit is None:
            saturation_limit = int(ADC_MAX_VALUE * 0.95)
        for i, pd in enumerate(pd_vals):
            if pd >= saturation_limit:
                setpoints = setpoints[:i + 1]
                pd_vals = pd_vals[:i + 1]
                break

    return {"current": setpoints, "photodetector": pd_vals}


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
