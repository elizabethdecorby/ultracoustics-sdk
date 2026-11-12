"""The readout's signal chain: ADC counts to optical power.

ADC counts give the voltage at the ADC. The differential driver's gain is
divided out to reach the TIA output, the TIA's feedback resistor gives the
photocurrent, and the photodiode's responsivity gives the optical power.
"""

import numpy as np

from .config import ADC_MAX_VALUE

ADC_VREF = 5.0                  # V, ADC full scale

# ADA4940 differential driver. The schematic rounds this resistor ratio to 1.667.
DIFF_GAIN = 470 / 280

# TIA feedback resistors. The two readout channels have different front ends,
# so a given ADC count is twice the photocurrent on the 638 nm channel.
TRANSIMPEDANCE_1550 = 20_000    # ohm
TRANSIMPEDANCE_638 = 10_000     # ohm

# Photodiode responsivity. The 1550 nm figure starts from the datasheet's
# 0.9 A/W at 1310 nm, a quantum efficiency of 0.852 at that wavelength, which
# stays flat or rises slightly toward 1550 nm.
RESPONSIVITY_1550 = 1.077       # A/W, InGaAs
RESPONSIVITY_638 = 0.3          # A/W, silicon


def w_per_count(R_PD: float = RESPONSIVITY_1550,
                R_feedback: float = TRANSIMPEDANCE_1550,
                diff_gain: float = DIFF_GAIN,
                V_ref: float = ADC_VREF,
                adc_fullscale: int = ADC_MAX_VALUE) -> float:
    """Optical power (W) per ADC count."""
    return V_ref / adc_fullscale / (R_feedback * diff_gain * R_PD)


#: µW per ADC count on the 1550 nm channel.
ADC_TO_POWER_UW = w_per_count() * 1e6


def counts_to_uw(counts, baseline=0.0, uw_per_count=ADC_TO_POWER_UW):
    """Optical power in µW from ADC counts, as float64: ``(counts - baseline) * uw_per_count``.

    ``baseline``: counts to subtract first, such as the dark level. What
    :meth:`Recording.power_uw` gives for a recording, for counts from anywhere else,
    such as the live ``ctrl.buffer``.
    """
    return (np.asarray(counts, dtype=np.float64) - baseline) * uw_per_count

#: Time for the readout to fall to its dark level after a confirmed STOP.
DARK_SETTLE_S = 0.6
#: Above this mean the lasers are on. A dark reads about 240 counts, a lock about 4400.
DARK_MAX_COUNTS = 1000

#: A level at this fraction of the detector's full scale or more is saturated: the
#: brightest light the box can measure, not a measurement of it.
CLIP_FRACTION = 0.95
