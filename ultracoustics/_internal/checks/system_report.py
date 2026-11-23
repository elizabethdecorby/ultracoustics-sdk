"""System report: what the box can tell about itself, closed with the probe in.

The tests of the box that need no opening and no other equipment.

- the software and USB checks;
- the laser output on each board's own monitor photodiode;
- both TEC heat-load steps;
- the probe sweep (``Box.run_probe_sweep``);
- the lock, the data rate, the dark and locked levels and the spectrum, with
  the thermal-noise fit;
- the lock calibration's measurement (``calibrate_lock_638``), measured only:
  the STOP that ends the report drops the gains it applies, and nothing is
  stored.

``run_report()`` drives a connected, streaming Box from one
thread and returns JSON-safe step dicts. The caller adds ``device_step()`` once
it has read the firmware and identity (``device_info.read_device_info``), which
needs the controller closed, and ``host_section()`` about the computer.
"""
import math
import platform
import time
import traceback
from dataclasses import asdict, is_dataclass
from datetime import date, datetime

import numpy as np
from scipy.ndimage import median_filter

from ...analysis import calibration as nepcal
from ...readout import CLIP_FRACTION
# The report's format is the public selftest module's, so reports are read without this one
from ...selftest import (MEASURED, PROBE_SERIAL_MAX_CHARS, SCHEMA, STEPS, TITLES,  # noqa: F401
                        SelfTestReport, probe_serial_text, short_uid)
from . import laser_board, li_curve, system_checks, thermal_step
from .probe import MODE_MIN_DEPTH_PCT, find_dips, first_order_mode  # noqa: F401
from .device_info import BOARD_LABELS, IDENTITY_FIELDS, IDENTITY_STATUS, field_text, firmware_text
from .laser_board import manual_session as _manual_session  # noqa: F401 (laser_check's)
from .li_curve import NoPhotodiode, analyse_li, li_sweep
from .system_checks import BOOT_FAULTS, BOOT_STATES, ended_text, fault_name, state_name  # noqa: F401
from .thermal_step import analyse_thermal_step, run_thermal_step

STOP_TIMEOUT_S = system_checks.STOP_TIMEOUT_S
#: Telemetry newer than the manual session's power-up, before judging the boards.
TELEMETRY_SETTLE_S = system_checks.TELEMETRY_SETTLE_S
POWER_UP_ATTEMPTS = system_checks.POWER_UP_ATTEMPTS
POWER_UP_RETRY_GAP_S = system_checks.POWER_UP_RETRY_GAP_S

# The procedures have one implementation each: the checks
# (system_checks), the LI sweep (li_curve), the heat-load step (thermal_step) and
# the probe's dips (characterization). These are the report's choices for them.

# Laser output on the board's own monitor photodiode: DAC 0 -> top, the 638 to
# its cap and the 1550 to 25000 (it never runs near its 43253 limit).
LI_TOP_DAC = {1550: 25000, 638: None}
LI_POINTS = li_curve.POINTS

# TEC heat-load step: the laser DAC steps 0 -> DAC -> 0 and the TEC holds its
# target against the heat.
TEC_STEP_DAC = {1550: 25000, 638: 52400}
TEC_BASELINE_S = thermal_step.BASELINE_S
TEC_EDGE_S = thermal_step.EDGE_S
TEC_LOCK_HOLD_S = laser_board.LOCK_HOLD_S
TEC_SETTLE_WINDOW_S = laser_board.SETTLE_WINDOW_S
TEC_SETTLE_PP_MDEGC = laser_board.SETTLE_PP_MDEGC
TEC_LOCK_TIMEOUT_S = laser_board.LOCK_TIMEOUT_S
TEC_RECOVERY_BAND_MDEGC = thermal_step.RECOVERY_BAND_MDEGC
#: Windows before an edge (its starting level) and at the end (the residual).
PRE_EDGE_S = thermal_step.PRE_EDGE_S
TAIL_S = thermal_step.TAIL_S

# Probe sweep.
PROBE_STEP = 100
PROBE_DWELL_S = 0.025
PROBE_SETTLE_S = 0.005
PROBE_WARMUP_S = 10.0
# The sweep runs the 638 to its cap: the SDK's stop at 95% of full scale is off,
# and the report says where the detector reached that level instead
PROBE_FULL_SCALE = CLIP_FRACTION
#: What a saturated detector asks for (laser_check chooses a lower 1550 power). The
#: GUI shows it as Help › Self-check, which names no laser.
LASER_CHECK_ADVICE = ('The detector saturates at the current 1550 nm setting: Diagnostics › '
                      'Adjust signal level chooses a lower one.')

# The master gives up first: up to 60 s for the 1550, then 30 s for the 638 to lock
# (the master firmware's boot timeouts)
LOCK_TIMEOUT_S = 100.0
OBSERVE_S = 20.0
#: The GUI flags a data rate under 90% of the 10 MSPS stream.
RATE_FRACTION = 0.9
#: The recorded fit's dark term, median-smoothed over this many bins (~40 kHz)
FIT_DARK_SMOOTH_BINS = 65
STREAM_ERRORS = (('drops_seq', 'sequence gaps'), ('drops_fw', 'firmware drops'),
                 ('transfer_errors', 'transfer errors'),
                 ('transfer_timeouts', 'transfer timeouts'), ('malformed', 'malformed packets'))

# The 638 optical loop's states, as its firmware numbers them
OPTICAL_STATES = ('IDLE', 'CALIBRATING', 'ROOT FINDING', 'LOCKED', 'ERROR', 'MEASURE SLOPE')
LOCKED = 3


class Cancelled(Exception):
    """The report was cancelled."""


class NotRun(Exception):
    """The step can't run on this box; the message says why."""


def format_pressure(pa):
    """A pressure as the report shows it: µPa, mPa or Pa."""
    if pa < 1e-3:
        return f'{pa * 1e6:.2f} µPa'
    if pa < 1.0:
        return f'{pa * 1e3:.3f} mPa'
    return f'{pa:.3f} Pa'


def format_nep(nep2):
    """A noise-equivalent pressure from NEP² in Pa²/Hz, per √Hz."""
    return f'{format_pressure(float(np.sqrt(nep2)))}/√Hz'


def plain(value):
    """JSON-safe copy: dataclasses and objects as dicts, arrays as lists, NaN as None."""
    if is_dataclass(value) and not isinstance(value, type):
        return plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    if hasattr(value, '_asdict'):
        return plain(value._asdict())
    if hasattr(value, '__dict__'):
        return plain({k: v for k, v in vars(value).items() if not k.startswith('_')})
    return str(value)


def rounded(values, digits=6):
    """A float array as a list at ``digits`` significant figures (smaller JSON)."""
    return [float(f'{v:.{digits}g}') if math.isfinite(v) else None
            for v in np.asarray(values, dtype=float)]


def uw(counts):
    """ADC counts above the dark level, as optical power on the detector."""
    from ...readout import ADC_TO_POWER_UW
    return float(counts) * ADC_TO_POWER_UW


def probe_data(probe):
    """The probe's constants as the report records them: without its name."""
    return {key: value for key, value in plain(probe).items() if key != 'name'}


class Step:
    """One section of the report: table rows, the data behind them, a status."""

    def __init__(self, key, title=None):
        self.key, self.title = key, title or TITLES[key]
        self.rows = []      # [label, value, ok]; ok None: information only
        self.data = {}
        self.summary = ''
        self.status = None  # set by a step that decides it: 'skipped', 'error'...

    def row(self, label, value, ok=None):
        self.rows.append([label, str(value), ok])

    def as_dict(self, seconds=None):
        flagged = [label for label, _, ok in self.rows if ok is False]
        status = self.status or ('attention' if flagged else 'ok')
        summary = self.summary
        if status == 'attention':
            summary = ' '.join(filter(None, ['Needs a look: ' + '; '.join(flagged) + '.',
                                             summary]))
        elif not summary and status == 'ok':
            checked = sum(ok is not None for _, _, ok in self.rows)
            summary = (f'All {checked} checks passed.' if checked > 1 else
                       'Passed.' if checked else 'Recorded.')
        return dict(key=self.key, title=self.title, status=status, summary=summary,
                    rows=self.rows, data=plain(self.data),
                    seconds=None if seconds is None else round(seconds, 1))


def not_run(reason, keys=None, status='skipped'):
    """Every step (or ``keys``) as not run, with ``reason``."""
    steps = []
    for key, _title, _ in STEPS:
        if keys is None or key in keys:
            step = Step(key)
            step.status, step.summary = status, reason
            steps.append(step.as_dict())
    return steps


class _Run:
    """What the steps share: the controller, what earlier steps found, progress."""

    def __init__(self, ctrl, progress, cancel, sample_rate):
        self.ctrl = ctrl
        self._progress = progress
        self.cancelled = cancel
        self.sample_rate = sample_rate
        self.key = None
        self.telemetry = True    # board telemetry (stream format 1)
        self.optical = False     # 638 optical diagnostics (stream format 2)
        self.cap_638 = None
        self.dark = None         # nepcal.Spectrum with the lasers off
        self.locked = False
        self.pi = None           # the 638's running normalized PI, read at lock
        self.stopped = False     # the last step's confirmed STOP has run

    def say(self, message, fraction=None):
        """Progress, from anywhere: an SDK callback must never be made to raise."""
        self._progress({'step': self.key, 'message': message, 'fraction': fraction})

    def stage(self, message, fraction=None):
        self.check()
        self.say(message, fraction)

    def check(self):
        if self.cancelled():
            raise Cancelled()


def run_report(ctrl, probe=None, progress=None, cancel=None, sample_rate=10_000_000):
    """Run every on-box step in order; returns ``{'steps', 'cancelled'}``.

    ``ctrl`` is connected and streaming and nothing else commands it meanwhile.
    ``probe`` (a ProbeParameters) scales the thermal fit's NEP.
    ``progress(event)`` gets ``{'step', 'message', 'fraction'}`` while a step
    runs and ``{'step', 'done'}`` (the step's dict) after it. A step that fails
    is reported and the next one runs. The box is left stopped (IDLE).
    """
    progress = progress or (lambda event: None)
    run = _Run(ctrl, progress, cancel or (lambda: False), sample_rate)
    measures = {'checks': _checks, 'dark': _dark, 'li_1550': _li(1550), 'li_638': _li(638),
                'tec_1550': _tec(1550),
                'tec_638': _tec(638), 'probe': _probe, 'lock': lambda r, s: _lock(r, s, probe),
                'lock_loop': _lock_loop, 'stop': _stop}
    steps, cancelled = [], False
    try:
        for key, _title, _ in STEPS:
            if key not in measures:
                continue
            run.key = key
            step = Step(key)
            started = time.monotonic()
            if cancelled:
                step.status, step.summary = 'cancelled', 'Not run: the report was cancelled.'
            else:
                run.say('Starting…')
                try:
                    measures[key](run, step)
                except NotRun as exc:
                    step.status, step.summary = 'skipped', f'Not run: {exc}'
                except Exception as exc:
                    if isinstance(exc, Cancelled) or run.cancelled():
                        cancelled = True
                        step.status, step.summary = 'cancelled', 'Cancelled.'
                    else:
                        step.status, step.summary = 'error', f'Did not complete: {exc}'
                        step.data['error'] = traceback.format_exc()
            done = step.as_dict(time.monotonic() - started)
            steps.append(done)
            progress({'step': key, 'done': done})
    finally:
        if not run.stopped:
            try:
                ctrl.stop_system_confirmed(timeout_s=STOP_TIMEOUT_S)
            except Exception as exc:
                print(f'[system-report] confirmed STOP after the report failed: {exc}', flush=True)
    return {'steps': steps, 'cancelled': cancelled}


# -- Software and USB checks (system_checks) -------------------------------------

def _checks(run, step):
    result = system_checks.run_system_checks(
        run.ctrl, progress=run.say, check=run.check, read_tuning=True,
        power_up_attempts=POWER_UP_ATTEMPTS, retry_gap_s=POWER_UP_RETRY_GAP_S,
        settle_s=TELEMETRY_SETTLE_S, stop_timeout_s=STOP_TIMEOUT_S)
    for item in result.items:
        step.row(item.label, item.value, item.passed)
    step.data.update(result.values)
    step.summary = result.summary
    run.telemetry, run.optical, run.cap_638 = result.telemetry, result.optical, result.cap_638
    if result.exception is not None:
        raise result.exception


# -- Detector with the lasers off --------------------------------------------

def _capture(run, message):
    """CAPTURE_SECONDS of fresh samples and their spectrum."""
    run.stage(message)
    n = int(nepcal.CAPTURE_SECONDS * run.sample_rate)
    samples = run.ctrl.collect_fresh(n, timeout_s=5.0)
    return samples, nepcal.capture_spectrum(samples, run.sample_rate)


def _spectrum_data(spectrum):
    """The PSD on its uniform grid: bin k is at k × bin_hz."""
    return dict(bin_hz=float(spectrum.freqs[1] - spectrum.freqs[0]),
                psd_w2hz=rounded(spectrum.psd_w2hz, 4),
                mean_counts=spectrum.mean_counts, fft_size=spectrum.fft_size,
                num_averages=spectrum.num_averages)


def _dark(run, step):
    run.stage('Stopping the box…')
    run.ctrl.stop_system_confirmed(timeout_s=STOP_TIMEOUT_S)
    samples, run.dark = _capture(run, 'Recording the detector with the lasers off…')
    step.data['spectrum'] = _spectrum_data(run.dark)
    step.data['counts'] = dict(mean=float(samples.mean()), std=float(samples.std()),
                               min=int(samples.min()), max=int(samples.max()))
    step.row('Dark level', f'{samples.mean():.1f} counts')
    step.row('Dark noise', f'{samples.std():.2f} counts rms')
    step.summary = f'{samples.mean():.1f} counts.'


# -- Laser output and TEC heat-load steps (li_curve, thermal_step) -----------------

class _Laser(laser_board.ManualLaser):
    """A laser board on the report's run, waiting with the report's settings."""

    def __init__(self, run, target):
        super().__init__(run.ctrl, target, check=run.check, say=run.say)
        self.run = run

    def settle_tec(self):
        return super().settle_tec(TEC_SETTLE_PP_MDEGC, hold_s=TEC_LOCK_HOLD_S,
                                  window_s=TEC_SETTLE_WINDOW_S, timeout_s=TEC_LOCK_TIMEOUT_S)


def _li(target):
    def measure(run, step):
        if target not in run.ctrl.laser_boards:
            raise NotRun(f'this box has no {target} nm board.')
        if not run.telemetry:
            raise NotRun('it needs board telemetry, which the main board firmware lacks.')
        sweep = _li_sweep(run, target)
        dac, pd = [r[0] for r in sweep['rows']], [r[1] for r in sweep['rows']]
        result = analyse_li(dac, pd, sweep['saturated_at_dac'] is not None, sweep['pd_full_scale'])
        step.data.update(sweep=sweep, analysis=result)
        full = sweep['pd_full_scale']
        if sweep['saturated_at_dac'] is None:
            step.row('Swept', f"DAC 0 to {sweep['top']} in {len(dac)} points")
        else:
            step.row('Swept', f"stopped at DAC {sweep['saturated_at_dac']}: the photodiode "
                     'reached full scale, the most its ADC input can take', False)
        step.row('Dark level, laser off', f"{result['dark_counts']:.0f} counts")
        threshold = result['threshold_dac']
        step.row('Lasing threshold', 'no threshold found' if threshold is None else
                 f'DAC {threshold:.0f}', threshold is not None)
        # The slopes and the largest step stay in the analysis data, unshown
        share = 100.0 * result['pd_at_top'] / full if full else 0.0
        step.row('Photodiode at the top', f"{result['pd_at_top']} of {full} counts "
                 f"({share:.0f}%) at DAC {result['top_dac']}")
        if threshold is not None and sweep['saturated_at_dac'] is None:
            step.summary = (f'Lasing from DAC {threshold:.0f}; the photodiode reads {share:.0f}% '
                            f"of full scale at DAC {result['top_dac']}.")
    return measure


def _li_sweep(run, target):
    """``target``'s LI sweep on its monitor photodiode, after its TEC settles."""
    ctrl = run.ctrl
    run.stage(f'Powering both laser boards for the {target} nm laser…')
    with _manual_session(ctrl):
        laser = _Laser(run, target)
        # Reading the 638's cap also raises the SDK's 638 bound to it
        top = (int(ctrl.read_optical_cap_638()['max_dac']) if target == 638
               else LI_TOP_DAC[target])
        target_c, lock_wait_s, _spread = laser.settle_tec()
        try:
            sweep = li_sweep(laser, top, LI_POINTS, progress=run.say)
        except NoPhotodiode as exc:
            raise NotRun(str(exc))
        run.say('Stopping (confirmed shutdown)…')
    return {'target_c': target_c, 'lock_wait_s': lock_wait_s, **sweep}


def _tec(target):
    def measure(run, step):
        ctrl = run.ctrl
        if target not in ctrl.laser_boards:
            raise NotRun(f'this box has no {target} nm board.')
        if not run.telemetry:
            raise NotRun('it needs board telemetry, which the main board firmware lacks.')
        step_dac = TEC_STEP_DAC[target]
        if target == 638 and run.cap_638:
            step_dac = min(step_dac, run.cap_638)
        capture = run_thermal_step(
            ctrl, target, step_dac, baseline_s=TEC_BASELINE_S, edge_s=TEC_EDGE_S,
            settle_pp_mdegc=TEC_SETTLE_PP_MDEGC, lock_timeout_s=TEC_LOCK_TIMEOUT_S,
            hold_s=TEC_LOCK_HOLD_S, settle_window_s=TEC_SETTLE_WINDOW_S, check=run.check,
            say=run.say)
        result = analyse_thermal_step(capture, TEC_RECOVERY_BAND_MDEGC, pre_edge_s=PRE_EDGE_S,
                                      tail_s=TAIL_S)
        step.data.update(capture=capture, analysis=result)
        step.row('TEC target', f"{capture['target_c']:.2f} °C, settled in "
                 f"{capture['lock_wait_s']:.1f} s")
        noise, offset = result['baseline_noise_mdegc'], result['baseline_offset_mdegc']
        if noise is not None:
            step.row('Before the step, laser off', f'{noise:.1f} m°C rms about the target, '
                     f'offset {offset:+.1f} m°C')
        band = TEC_RECOVERY_BAND_MDEGC
        for edge, label in zip(result['edges'], (f'Laser on (DAC {step_dac})', 'Laser off again')):
            if edge['peak_mdegc'] is None:
                step.row(label, 'no samples', False)
                continue
            back = (f"back within ±{band:g} m°C in {edge['recovery_s']:.1f} s"
                    if edge['settled'] else f'not back within ±{band:g} m°C after '
                    f"{edge['window_s']:.0f} s")
            step.row(label, f"moved {edge['peak_mdegc']:+.0f} m°C; {back}",
                     edge['settled'] and abs(edge['residual_mdegc']) <= band)
        if result['passed']:
            step.summary = 'Met target.'
    return measure


# -- Probe sweep ------------------------------------------------------------------

def _probe(run, step):
    ctrl = run.ctrl
    if 1550 not in ctrl.laser_boards:
        raise NotRun('the probe sweep needs the 1550 nm board.')
    run.stage('Starting the 1550 nm laser and both TECs (about 20 s)…')

    def on_progress(point):
        run.say(f"Sweeping the 638 nm laser: DAC {point['dac']}",
                (point['index'] + 1) / point['count'])
    result = ctrl.run_probe_sweep(step=PROBE_STEP, dwell_s=PROBE_DWELL_S, settle_s=PROBE_SETTLE_S,
                                  warmup_s=PROBE_WARMUP_S, saturation_threshold=float('inf'),
                                  cancel=run.cancelled, on_progress=on_progress)
    run.check()
    dac = np.asarray(result.current, dtype=float)
    pd = np.asarray(result.photodetector, dtype=float)
    dips = find_dips(dac, pd)
    mode = first_order_mode(dips)
    dark = run.dark.mean_counts if run.dark is not None else 0.0
    from ...config import ADC_MAX_VALUE
    high = np.flatnonzero(pd >= PROBE_FULL_SCALE * ADC_MAX_VALUE)
    full_scale_dac = int(dac[high[0]]) if len(high) else None
    step.data.update(dac=rounded(dac), photodetector_counts=rounded(pd),
                     temperature_c=plain(getattr(result, 'temperature_c', None)),
                     full_scale_from_dac=full_scale_dac, dark_counts=dark, dips=dips, mode=mode)
    if not len(pd):
        step.row('Sweep', 'no points measured', False)
        return
    step.row('Swept', f'638 nm DAC 0 to {dac[-1]:.0f} in steps of {PROBE_STEP}')
    if full_scale_dac is not None:
        # The 1550 runs at its stored DAC here: that setting saturates the detector
        step.row('Detector near full scale', f'from DAC {full_scale_dac}: {len(high)} of '
                 f'{len(pd)} points at {PROBE_FULL_SCALE:.0%} of full scale or more', False)
    step.row('Reflected level, 638 nm off (unlocked)',
             f'{uw(pd[0] - dark):.1f} µW ({pd[0]:.0f} counts)')
    step.row('First-order optical mode', f"DAC {mode['dac']}, {mode['depth_pct']:.0f}% deep"
             if mode else f'no dip at least {MODE_MIN_DEPTH_PCT:g}% deep', mode is not None)
    step.row('Dips', ', '.join(f"DAC {d['dac']} ({d['depth_pct']:.0f}%)" for d in dips) or 'none')
    if mode:
        step.summary = f"First-order mode at DAC {mode['dac']}, {mode['depth_pct']:.0f}% deep."
    if full_scale_dac is not None:
        step.summary = ' '.join(filter(None, [step.summary, LASER_CHECK_ADVICE]))


def recommends_laser_check(report):
    """The report saw the detector saturate at the 1550's stored power: see
    SelfTestReport.recommends_signal_adjustment."""
    return SelfTestReport(report).recommends_signal_adjustment


# -- Lock, data rate and spectrum --------------------------------------------------

def _live_638(ctrl):
    """The newest fresh 638 live page as a dict, or None."""
    snapshot = ctrl.telemetry
    entry = getattr(snapshot, 'optical_live_638', None) if snapshot is not None else None
    if entry is None or snapshot.host_stale or entry.host_stale:
        return None
    page = entry.page
    return {'state': int(page.state), 'dac': int(page.dac), 'feedback': float(page.feedback),
            'target': float(page.target), 'sequence': int(page.sequence)}


def _optical_name(state):
    return OPTICAL_STATES[state] if 0 <= state < len(OPTICAL_STATES) else f'state {state}'


def _lock(run, step, probe):
    from ...config import ADC_MAX_VALUE
    ctrl = run.ctrl
    run.stage('Starting the box…')
    ctrl.stop_system_confirmed(timeout_s=STOP_TIMEOUT_S)
    if run.optical:
        run.optical = ctrl.try_optical_diagnostics()
    started = time.monotonic()
    ctrl.start_confirmed(timeout_s=STOP_TIMEOUT_S)
    if run.optical:
        states, live, shown = [], None, None
        while True:
            run.check()
            elapsed = time.monotonic() - started
            live = _live_638(ctrl) or live
            if live is not None:
                if not states or states[-1][1] != live['state']:
                    states.append([round(elapsed, 2), live['state']])
                if live['state'] == LOCKED:
                    break
            if elapsed > LOCK_TIMEOUT_S:
                break
            text = (f"Waiting for the lock: {elapsed:.0f} s, the 638 is "
                    f"{_optical_name(live['state']) if live else 'starting'}")
            if text != shown:
                run.say(text, min(elapsed / 60.0, 0.95))
                shown = text
            time.sleep(0.1)
        step.data['lock_states'] = [[t, _optical_name(s)] for t, s in states]
        run.locked = live is not None and live['state'] == LOCKED
        step.row('Locks after START', f'in {elapsed:.1f} s' if run.locked else
                 f'not locked after {LOCK_TIMEOUT_S:.0f} s (the 638 is '
                 f"{_optical_name(live['state']) if live else 'not reporting'}); the box's "
                 'own record of why is under Stop and record', run.locked)
    else:
        step.row('Lock state', 'not reported by this main board firmware')
        run.stage('Letting the box start…')
        _wait(run, 15.0)
    if 1550 in ctrl.laser_boards:
        _setpoint_1550(run, step)

    run.stage(f'Watching the lock and the data rate for {OBSERVE_S:g} s…')
    before, count0, t0 = dict(ctrl.stream_stats or {}), ctrl.samples_received, time.monotonic()
    pages, seen = [], None
    while (elapsed := time.monotonic() - t0) < OBSERVE_S:
        run.check()
        live = _live_638(ctrl) if run.optical else None
        if live is not None and live['sequence'] != seen:
            seen = live['sequence']
            pages.append([round(elapsed, 3), live['state'], live['dac'], live['feedback'],
                          live['target']])
        run.say(f'Watching the lock and the data rate: {OBSERVE_S - elapsed:.0f} s left',
                elapsed / OBSERVE_S * 0.6)
        time.sleep(0.05)
    count1, t1, after = ctrl.samples_received, time.monotonic(), dict(ctrl.stream_stats or {})
    rate = (count1 - count0) / (t1 - t0)
    step.data['stream'] = {'rate_sps': rate, 'before': before, 'after': after}
    step.row('Data rate', f'{rate / 1e6:.2f} MSPS', rate >= RATE_FRACTION * run.sample_rate)
    errors = [f'{after.get(k, 0) - before.get(k, 0)} {name}' for k, name in STREAM_ERRORS
              if after.get(k, 0) != before.get(k, 0)]
    step.row(f'Stream errors in {OBSERVE_S:g} s', ', '.join(errors) or 'none', not errors)
    if pages:
        step.data['lock_pages'] = {'columns': ['t_s', 'state', 'dac', 'feedback', 'target'],
                                   'rows': pages}
        states = [p[1] for p in pages]
        unlocks = sum(a == LOCKED and b != LOCKED for a, b in zip(states, states[1:]))
        held = all(s == LOCKED for s in states)
        step.row('Stays locked', f'for all {OBSERVE_S:g} s' if held else
                 f'left the lock {unlocks} times' if unlocks else
                 f'locked {states.count(LOCKED)} of {len(states)} reports', held)
        dacs = [p[2] for p in pages if p[1] == LOCKED]
        if dacs:
            step.row('Lock point', f'638 nm DAC {np.median(dacs):.0f}, moving '
                     f'{max(dacs) - min(dacs)} counts peak to peak')
    snapshot = ctrl.telemetry
    if snapshot is not None and not snapshot.host_stale:
        for target in ctrl.laser_boards:
            board = getattr(snapshot, f'board_{target}', None)
            if board is not None and board.temperature_valid:
                step.row(f'{target} nm TEC while running', f'{board.temperature_mdegc / 1000:.2f} °C')
    if run.locked and hasattr(ctrl, 'read_normalized_pi_638'):
        try:
            run.pi = dict(ctrl.read_normalized_pi_638(timeout_s=2.0))
        except Exception as exc:
            step.row('Lock gains running', f'not read: {exc}')
        else:
            step.data['normalized_pi'] = run.pi
            source = ('stored on the box' if run.pi.get('record_active') else
                      'set at runtime' if run.pi.get('overridden') else 'firmware defaults')
            step.row('Lock gains running', f"Kp {run.pi['kp']:.3f}, Ki {run.pi['ki_per_s']:g}/s "
                     f'({source})')

    samples, signal = _capture(run, 'Recording the spectrum…')
    dark = run.dark.mean_counts if run.dark is not None else 0.0
    clipped = int(np.count_nonzero((samples >= ADC_MAX_VALUE) | (samples <= 0)))
    step.data['spectrum'] = _spectrum_data(signal)
    step.row('Optical level on the detector', f'{uw(signal.mean_counts - dark):.1f} µW '
             f'({signal.mean_counts:.0f} counts)')
    step.row('Detector clipping', f'{clipped} of {samples.size} samples' if clipped else 'none',
             not clipped)
    if clipped:
        step.summary = LASER_CHECK_ADVICE
    if probe is None:
        return
    if not run.locked:
        step.row('Thermal noise fit', 'not measured: the box did not lock')
        return
    run.stage('Fitting the thermal noise peaks…', 0.8)
    try:
        cal = nepcal.calibrate(signal, run.dark, probe, run.sample_rate)
    except nepcal.CalibrationError as exc:
        step.row('Thermal noise fit', str(exc), False)
        return
    ref = cal.fundamental
    # The fitted spectrum over the fitted band, on the spectrum's grid from first_bin;
    # the measured dark term smoothed, so the curve is the model and not the dark's noise
    freqs = np.asarray(signal.freqs, dtype=float)
    band = np.flatnonzero((freqs >= nepcal.MIN_MODE_FREQ_HZ)
                          & (freqs <= nepcal.fit_max_hz(run.sample_rate)))
    terms = nepcal.noise_terms(cal, freqs[band], signal.mean_counts)
    terms['dark'] = median_filter(terms['dark'], size=FIT_DARK_SMOOTH_BINS, mode='nearest')
    model = sum(terms.values())
    step.data['fit'] = dict(
        modes=[dict(f0_hz=m.f0_hz, q=m.Q, a=m.A) for m in cal.modes],
        fundamental_hz=ref.f0_hz, fundamental_q=ref.Q, nep2_pa2hz=cal.nep2,
        floor_w2hz=cal.floor_w2hz, shot_w2hz=cal.shot_psd_w2hz,
        photocurrent_counts=cal.photocurrent_counts, filter_params=cal.filter_params,
        probe=probe_data(probe),
        model=dict(first_bin=int(band[0]), psd_w2hz=rounded(model, 4)) if len(band) else None)
    step.row('Fundamental mode', f'{ref.f0_hz / 1e6:.3f} MHz, Q {ref.Q:.1f}')
    step.row('Modes fitted', ', '.join(f'{m.f0_hz / 1e6:.2f}' for m in cal.modes) + ' MHz')
    step.row('Noise-equivalent pressure', format_nep(cal.nep2))
    step.row('White noise above dark and shot noise',
             f'{nepcal.floor_over_noise_db(cal):+.1f} dB')


def _setpoint_1550(run, step):
    """The 1550's stored setpoint: while the box runs, the main board passes a
    read-only GET of the 1550's DAC, and the 1550 answers with the DAC it runs
    at, the word in its flash (clamped to its hardware limit; 15885 if none
    was ever stored). Main board firmware without that read refuses it."""
    from ..box import SlaveCommandRejected
    from ..control import CHANNEL_LASER_DAC, MANUAL_GET
    label = 'Stored 1550 nm setpoint'
    run.stage('Reading the 1550 nm setpoint…')
    try:
        reply = run.ctrl.manual_command(1550, MANUAL_GET, CHANNEL_LASER_DAC, timeout_s=1.0)
        if reply.status != 0:
            raise SlaveCommandRejected(reply.target, reply.channel, reply.status,
                                       getattr(reply, 'opcode', None), reply)
    except Exception as exc:
        step.row(label, f'not read: {exc}')
        return
    dac = int(reply.applied_value)
    step.data['setpoint_1550_dac'] = dac
    step.row(label, f'DAC {dac}' if dac else 'not read: the 1550 was not on')


def _wait(run, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        run.check()
        time.sleep(0.1)


# -- Lock loop measurement (calibrate_lock_638, nothing stored) --------------

def _lock_loop(run, step):
    try:
        from ..lock_calibration import calibrate_lock_638
    except ImportError:
        raise NotRun('the installed software has no lock calibration.')
    if not run.locked:
        raise NotRun('it needs the box locked.')
    if not (run.pi and run.pi.get('active') and run.pi.get('slope_valid')):
        raise NotRun("the 638 nm firmware's lock does not run normalized PI.")

    def progress(event):
        if isinstance(event, dict):
            run.say(event.get('message') or event.get('stage') or '', event.get('fraction'))
    report = calibrate_lock_638(run.ctrl, None, progress=progress, cancel=run.cancelled)
    step.data['calibration'] = report
    status = report.get('status')
    if status == 'cancelled':
        raise Cancelled()
    number = lambda value, spec='.1f': '—' if value is None else format(value, spec)
    previous, gains, fit = report.get('previous_gains'), report.get('gains'), report.get('fit')
    step.row('Measurement', {'complete': 'complete', 'rejected': 'did not pass the fit check',
                             'failed': 'failed'}.get(status, status)
             + (f": {report['message']}" if report.get('message') else ''),
             status == 'complete')
    if fit:
        step.row('Lock loop', f"gain {number(fit.get('g'), '.3g')}, thermal time constant "
                 f"{number(fit.get('tau_ms'), '.2f')} ms, delay {number(fit.get('delay_us'), '.0f')} µs")
    if previous:
        step.row('Gains running', f"Kp {previous['kp']:.3f}, Ki {previous['ki_per_s']:g}/s")
    if status == 'complete' and gains:
        step.row('Gains designed for this probe',
                 f"Kp {gains['kp']:.3f}, Ki {gains['ki_per_s']:g}/s; worst-case peak |S| "
                 f"{number(report.get('predicted_worst_peak'), '.2f')} (limit "
                 f"{number(report.get('ms_limit'), '.2f')})")
        step.row('Lock residual, running → designed gains',
                 f"{number(report.get('residual_before_adc'))} → "
                 f"{number(report.get('residual_after_adc'))} ADC rms")
        relock = report.get('relock') or {}
        if relock:
            step.row('Relocks on the designed gains', f"in {number(relock.get('seconds'))} s"
                     if relock.get('relocked') else 'no', bool(relock.get('relocked')))
    step.summary = ('Measured the running lock and designed gains for this probe. Nothing was '
                    'stored on the box: stopping it returned it to its own gains.'
                    if status == 'complete' else '')


# -- Stop, and the master's record of the run --------------------------------

def _stop(run, step):
    run.stage('Stopping the box…')
    # The confirmed STOP returns the master's retained record of the run it ended
    window = run.ctrl.stop_system_confirmed(timeout_s=STOP_TIMEOUT_S)
    run.stopped = True
    step.data['runtime'] = window
    step.row('Stops on request (IDLE confirmed)', state_name(window.current_state),
             window.current_state == 0)
    step.row('The run', ended_text(window), window.fault_code == 0)
    active = getattr(window, 'active_percent', None)
    step.row('Main board CPU load', f'{active:.1f}%' if active is not None and window.flags & 8
             else 'not reported')
    step.row('ADC buffer overflows / late services',
             f'{window.pssi_overflows} / {window.pssi_service_late}', window.pssi_overflows == 0)


# -- Firmware and identity (read by device_info.read_device_info) ----------------

def box_number(report):
    """The box's number as the report shows it: see SelfTestReport.box_number."""
    return SelfTestReport(report).box_number


def device_step(info=None, error=None, seconds=None):
    """The 'device' step from read_device_info()'s result, or the error that stopped it."""
    step = Step('device')
    if info is None:
        step.status, step.summary = 'error', f'Did not complete: {error}'
        return step.as_dict(seconds)
    step.data.update(info)
    step.row('USB serial', info.get('usb_serial') or 'unknown')
    firmware = info['firmware']
    for board, label in BOARD_LABELS.items():
        fw = firmware.get(board)
        if board == 'main':
            step.row(f'{label} firmware', firmware_text(fw), fw is not None)
        elif fw is not None or board in info['identity']:
            step.row(f'{label} bootloader', firmware_text((fw or {}).get('boot')))
            step.row(f'{label} application', firmware_text((fw or {}).get('app')))
    for board, label in BOARD_LABELS.items():
        result = info['identity'].get(board)
        if result is None:
            continue
        if result.record is None:
            step.row(f'{label} identity', IDENTITY_STATUS.get(result.status_name,
                                                              result.status_name))
            continue
        identity = result.record.identity
        for name, field_label in IDENTITY_FIELDS[board]:
            value = getattr(identity, name, None)
            if value is not None:
                step.row(f'{label}: {field_label}', field_text(name, value))
    for warning in info['warnings']:   # 'What: why'
        what, _, why = warning.partition(': ')
        step.row(what, why or 'not read', False)
    step.summary = ('Firmware versions and the identity stored on each board.'
                    if not info['warnings'] else '')
    return step.as_dict(seconds)


def host_section(software, box_serial=None, probe=None, calibration=None, lock_gains=None, qt=None):
    """Rows about this computer, the software and what it has saved for the box.

    ``software``: what ran the report, such as ``'Ultracoustics GUI 2.1.0'``.
    ``qt``: the Qt versions it runs on, such as ``'Qt 6.12.0, PySide 6.12.0'``.
    ``box_serial`` is not shown: the report names the box by ``box_number()``,
    which needs the device step. ``lock_gains`` (the gains a GUI keeps for the
    box) is no longer reported: the lock step reads the gains the box runs.
    """
    from ... import __version__
    step = Step('host', 'Computer and software')
    step.row('Readout software', software)
    step.row('Ultracoustics SDK', __version__)
    step.row('Computer', f'{platform.platform()} ({platform.machine()})')
    if qt:
        step.row('Python and Qt', f'Python {platform.python_version()}, {qt}')
    else:
        step.row('Python', platform.python_version())
    if probe is not None:
        step.row('Probe profile', f'effective mass {probe.m_eff_kg * 1e12:.3g} pg, '
                 f'dome radius {probe.radius_m * 1e6:.3g} µm, '
                 f'{probe.temperature_k - 273.15:.1f} °C'
                 + (f', acoustic correction ×{probe.acoustic_correction:.3g}'
                    if probe.acoustic_correction != 1.0 else ''))
        step.data['probe'] = probe_data(probe)
    if calibration is not None:
        ref = calibration.fundamental
        step.row('Saved thermal calibration', f'{ref.f0_hz / 1e6:.3f} MHz, Q {ref.Q:.1f}, NEP '
                 f'{format_nep(calibration.nep2)} ({calibration.created_at[:16]})')
        step.data['calibration'] = dict(fundamental_hz=ref.f0_hz, fundamental_q=ref.Q,
                                        nep2_pa2hz=calibration.nep2,
                                        created_at=calibration.created_at)
    step.status, step.summary = 'info', ''
    return step.as_dict()
