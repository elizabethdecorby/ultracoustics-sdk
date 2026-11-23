"""Ultracoustics Python SDK.

The whole of a measurement script::

    import ultracoustics as uc

    def main():
        with uc.Controller() as ctrl:        # connects, and turns the lasers off at the end
            report = ctrl.start_ready()      # lasers on and ready: about 30 s
            rec = ctrl.capture(1.0)          # one second of signal, checked
            rec.save("a_new_name")           # SigMF files; never overwrites

    if __name__ == "__main__":               # required
        main()

Three rules:

- Keep the script's work under ``if __name__ == "__main__":``. The USB is read by
  a second process, which imports the script when it starts.
- Use the names on ``ultracoustics`` and the methods of ``Controller``. Never use
  ``ultracoustics._internal`` or a name that starts with an underscore.
- Call ``ctrl.start_ready()`` first. When anything is wrong, run
  ``ctrl.self_test()`` and keep its report.

Analysis needs no box and no USB libraries. The names that talk to the box, and
the software and firmware update checks, load the first time they are used.
"""
from importlib import import_module

__version__ = "2.0.1"

from .analysis.calibration import Calibration, CalibrationError, ProbeParameters, load_calibration
from .analysis.filters import band_pass, notch
from .analysis.noise import NoiseBudget
from .analysis.pressure import PressureSpectrum, tone_level
from .analysis.spectrum import Spectrum
from .live import LiveSpectrum
from .monitor import AlarmChange, Monitor, MonitorSummary, find_events
from .readout import counts_to_uw
from .recorder import CaptureError
from .recording import Recording, load_recording
from .selftest import SelfTestReport
from .signal_level import SignalLevel, SignalLevelError, StoredSignalLevel
from .trigger import BandPowerTrigger, LevelTrigger

_ON_DEMAND = {
    ".controller": ("BoxFault", "Controller", "StartError", "StartReport", "Status"),
    ".lock": ("LockError", "LockGains", "LockReport"),
    # The rest of ultracoustics.updates (one board at a time, the update list itself) stays in that module
    ".updates": ("FirmwareRefused", "FirmwareUpdateError", "UpdateCheckError", "UpdateReport",
                  "check_for_updates", "update_firmware"),
}
_MODULE_OF = {name: module for module, names in _ON_DEMAND.items() for name in names}

__all__ = [
    "__version__",
    "AlarmChange",
    "BandPowerTrigger",
    "Calibration",
    "CalibrationError",
    "CaptureError",
    "LevelTrigger",
    "LiveSpectrum",
    "Monitor",
    "MonitorSummary",
    "NoiseBudget",
    "PressureSpectrum",
    "ProbeParameters",
    "Recording",
    "SelfTestReport",
    "SignalLevel",
    "SignalLevelError",
    "Spectrum",
    "StoredSignalLevel",
    "band_pass",
    "counts_to_uw",
    "find_events",
    "load_calibration",
    "load_recording",
    "notch",
    "tone_level",
    *_MODULE_OF,
]


def __getattr__(name):
    if name not in _MODULE_OF:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = globals()[name] = getattr(import_module(_MODULE_OF[name], __name__), name)
    return value


def __dir__():
    return sorted(__all__)
