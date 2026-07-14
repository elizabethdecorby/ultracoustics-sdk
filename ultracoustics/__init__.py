"""Ultracoustics Python SDK."""

__version__ = "0.1.0"

# High-level tools for External Use
from .controller import Controller
from .processing import compute_psd, load_binary, adc_to_uw, compute_noise_metrics
from .nep import (
    compute_psd_w2hz,
    shot_noise_psd_w2hz,
    w_per_count,
    sho_psd,
    dual_sho_psd,
    evaluate_sho_psd,
    find_peak,
    fit_sho_log,
    nep_th,
    nep_th_squared,
    psd_to_pressure_pa2hz,
    spl_db_per_rthz,
)
from .characterization import (
    ProbeCharacterizationResult,
    bin_ramp,
    load_laser_calibration,
    apply_laser_calibration,
    detect_resonance_dips,
    analyze_dips,
)

__all__ = [
    "__version__",
    "Controller",
    "compute_psd",
    "load_binary",
    "adc_to_uw",
    "compute_noise_metrics",
    # NEP / pressure-calibration helpers
    "compute_psd_w2hz",
    "shot_noise_psd_w2hz",
    "w_per_count",
    "sho_psd",
    "dual_sho_psd",
    "evaluate_sho_psd",
    "find_peak",
    "fit_sho_log",
    "nep_th",
    "nep_th_squared",
    "psd_to_pressure_pa2hz",
    "spl_db_per_rthz",
    # Probe characterization
    "ProbeCharacterizationResult",
    "bin_ramp",
    "load_laser_calibration",
    "apply_laser_calibration",
    "detect_resonance_dips",
    "analyze_dips",
]
