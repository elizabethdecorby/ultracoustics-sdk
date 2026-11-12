"""The probe sweep's result, and the probe's dips and first-order mode in it: the
Self-check's probe step and the lab's probe test."""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
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
# The first-order mode (the lab's probe test and the Self-check's probe step)
# ---------------------------------------------------------------------------

#: Smallest dip listed, as a fraction of the curve's median level.
DIP_MIN_DEPTH = 0.05
#: Smallest dip that can be the first-order mode, in percent. Shallower dips are
#: listed but never picked: on the noisy early probe curves every dip is a 5-14%
#: wobble, while real modes on good probes are 25-70% deep.
MODE_MIN_DEPTH_PCT = 15.0


def find_dips(dac, pd) -> List[Dict[str, Any]]:
    """Every dip at least DIP_MIN_DEPTH deep, lowest DAC first.

    As the 638 DAC rises it heats the probe through its optical modes, and the
    master's photodetector dips at each. A dip's depth is its prominence (how far
    it falls below the lower of the levels either side) as a percentage of that
    level, the depth an operator reads off the curve by hand. Checked against the
    saved bench curves and the values read off them: DAC 25,700 at 40.0% (read
    25,700 / 41%), DAC 25,500 at 39.2% (25,625 / 40%) and DAC 39,600 (40,000).

    Each is ``{'index', 'dac', 'pd_counts', 'level', 'depth_pct'}``, where ``level``
    is the surrounding level the depth is measured from.
    """
    dac, pd = np.asarray(dac, dtype=float), np.asarray(pd, dtype=float)
    level = float(np.median(pd)) if len(pd) >= 3 else 0.0
    if level <= 0:
        return []
    indices, props = find_peaks(-pd, prominence=DIP_MIN_DEPTH * level, width=1)
    dips = []
    for index, prominence in zip(indices, props['prominences']):
        around = float(pd[index]) + float(prominence)
        dips.append({'index': int(index), 'dac': int(dac[index]), 'pd_counts': float(pd[index]),
                     'level': around, 'depth_pct': 100.0 * float(prominence) / around})
    return dips


def first_order_mode(dips) -> Optional[Dict[str, Any]]:
    """The lowest-DAC dip deep enough to be an optical mode, or None."""
    return next((d for d in dips if d['depth_pct'] >= MODE_MIN_DEPTH_PCT), None)
