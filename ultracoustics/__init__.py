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
    load_laser_calibration,
    apply_laser_calibration,
    detect_resonance_dips,
    analyze_dips,
)
from .manual_sweep import ManualSweepError, run_manual_sweep
from .pi_characterization import run_pi_characterization, review_saved_pi_characterization
from .lock_calibration import calibrate_lock_638, apply_saved_lock_gains_638

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
    "load_laser_calibration",
    "apply_laser_calibration",
    "detect_resonance_dips",
    "analyze_dips",
    "ManualSweepError",
    "run_manual_sweep",
    "run_pi_characterization",
    "review_saved_pi_characterization",
    "calibrate_lock_638",
    "apply_saved_lock_gains_638",
]
