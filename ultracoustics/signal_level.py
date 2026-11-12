"""Choosing the 1550 nm laser's power for this probe, as the GUI's Diagnostics › Adjust signal level.

The 1550 nm laser runs at a setting stored on its board. Too high, and the light
reflected from the probe's brightest fringe saturates the detector; near a mode
hop, the laser is noisy. :func:`measure_signal_level` sweeps the laser on the
closed box with the probe in and finds its modes, the stretches between the
hops. It recommends the highest setting in the highest mode that has room, kept
clear of the hops on either side, with the brightest fringe under 90 % of the
detector's range. It changes nothing; :func:`store_signal_level` stores the
recommendation and checks the box comes back at it.

Each takes a few minutes and leaves the box stopped.
"""
from dataclasses import dataclass, field
from typing import Optional

from .readout import CLIP_FRACTION

#: (key, expected seconds) of the measurement's stages, as its progress events name
#: them: for a progress bar. Storing adds 'write' and 'verify'.
STAGES = (('stored', 25), ('sweep', 35), ('fringe', 40))


class SignalLevelError(RuntimeError):
    """No setting fits this probe and laser; the message says why."""


@dataclass(frozen=True)
class SignalLevel:
    """What :func:`measure_signal_level` found. Fractions are of the detector's full scale."""
    recommended_dac: int
    peak_fraction: float                 # the brightest fringe at the recommended setting
    current_dac: Optional[int]           # the stored setting, estimated from its level; None if off the
                                         # curve, or on more than one mode's
    current_peak_fraction: float         # the brightest fringe now (1 or more: it saturates)
    details: dict = field(default_factory=dict, compare=False, repr=False)   # the whole measurement

    @classmethod
    def from_details(cls, details):
        stored = details["stored"]
        return cls(int(details["recommended_dac"]), float(details["peak_fraction"]), stored.get("dac_estimate"),
                   float(stored["peak_fraction"]), details)

    @property
    def current_text(self):
        """The stored setting in the GUI's words: how high its signal peaks."""
        stored = self.details.get("stored") or {}
        if stored.get("level_counts", 0) >= CLIP_FRACTION * self.details.get("full_scale_counts", float("inf")):
            return "too strong for the box to measure"
        if self.current_peak_fraction >= 1.0:
            return "too strong: its peaks are cut off"
        return f"peaks at about {self.current_peak_fraction:.0%} of what the box can measure"

    def describe(self):
        """In the GUI's words: how high the signal peaks now, and at the recommended setting."""
        return (f"Now: {self.current_text}. Recommended: setting {self.recommended_dac}, peaking at "
                f"{self.peak_fraction:.0%} of what the box can measure.")


@dataclass(frozen=True)
class StoredSignalLevel:
    """What :func:`store_signal_level` stored, and whether the box came back at it."""
    dac: int
    verified: bool                       # the stored setting gives the level the sweep found there
    level_counts: float
    expected_level_counts: float
    details: dict = field(default_factory=dict, compare=False, repr=False)


def _require_free(controller, what):
    if controller._recorders or controller._monitors:
        raise RuntimeError(f"{what} stops the box: finish the recording or monitor first")
    controller._box._ensure_connected()


def measure_signal_level(controller, progress=None, cancel=None):
    """Sweep the 1550 nm laser and recommend its setting. Returns a :class:`SignalLevel`,
    or None if ``cancel()`` stopped it.

    ``progress(event)`` gets ``{'step', 'message', 'fraction'}``. SignalLevelError
    when no setting fits. The box is left stopped, its stored setting unchanged.
    """
    from .config import SAMPLE_RATE
    from ._internal.checks import laser_check
    from ._internal.checks.system_report import Cancelled

    _require_free(controller, "Measuring the signal level")
    try:
        return SignalLevel.from_details(laser_check.measure(controller._box, progress=progress, cancel=cancel,
                                                           sample_rate=SAMPLE_RATE))
    except laser_check.CheckFailed as exc:
        raise SignalLevelError(str(exc)) from exc
    except Cancelled:
        return None


def store_signal_level(controller, level, progress=None):
    """Store ``level``'s recommended setting on the 1550 nm board, then check the box
    comes back at it. Returns a :class:`StoredSignalLevel`.

    Not cancellable: the board's write must not be cut in half. RuntimeError if
    the box refused the write. The box is left stopped.
    """
    from ._internal.checks import laser_check

    _require_free(controller, "Storing the signal level")
    raw = laser_check.write_setpoint(controller._box, level.details, progress=progress)
    return StoredSignalLevel(int(raw["dac"]), bool(raw["verified"]), float(raw["level_counts"]),
                             float(raw["expected_level_counts"]), raw)
