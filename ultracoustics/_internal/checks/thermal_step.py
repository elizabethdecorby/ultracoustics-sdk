"""The TEC heat-load step: a laser's own drive stepped 0 -> DAC -> 0 while its TEC holds.

The lab's temperature tests and the Self-check's TEC steps both run
:func:`run_thermal_step` and :func:`analyse_thermal_step`.

The heat the laser dumps into its mount is the disturbance the TEC loop exists
to reject, so the step needs no serial cable, no scope and no widened target.

1. A whole-system manual session: both rails up with START low, both laser DACs
   at zero. On the 638 the step must be within the board's own cap, read first
   (which also raises the SDK's 638 bound to it), and the box profile that
   chose its compiled gains is recorded.
2. ``before_settle(laser)``, if given: the lab sets trial gains here.
3. Wait until the TEC holds its target and has settled (``ManualLaser.settle_tec``).
4. Record the baseline, SET the step, record an edge window, SET zero, record
   the second window. A confirmed STOP ends it on every path.

Temperature is the board's telemetry, the reading the loop regulates on (40 Hz).
Samples are de-duplicated on the board's sample tick and placed on its clock,
so their spacing is exact. The SETs happen on the host clock and are mapped
onto the board's through the median offset between the two; an edge is the
midpoint of the SET and its acknowledgement, known to within half that round
trip (``uncertainty_s``), a few milliseconds against a 25 ms sample period.

The analysis puts time zero at the laser-on edge. Per edge, from the samples
after it up to the next edge: ``peak_mdegc``, the largest excursion from the
level over the PRE_EDGE_S before it (positive warms); ``time_to_peak_s``;
``recovery_s``, until the temperature is back within ±band of the target and
stays there (None: still out when the window ended); ``residual_mdegc``, the
mean error over the last TAIL_S. And from the baseline its noise (standard
deviation) and offset from the target.
"""
import math
import statistics
import time
from collections import namedtuple
from datetime import datetime, timezone

import numpy as np

from .laser_board import (LOCK_HOLD_S, LOCK_TIMEOUT_S, POLL_S, SETTLE_PP_MDEGC, SETTLE_WINDOW_S,
                          ManualLaser, manual_session, with_master_fault)

BASELINE_S = 5.0
EDGE_S = 5.0
#: An edge recovers once back within this of the target, and stays (a placeholder band).
RECOVERY_BAND_MDEGC = 10.0
#: Windows before an edge (its starting level) and at the end (the residual).
PRE_EDGE_S = 1.0
TAIL_S = 1.0
EDGE_NAMES = ('laser_on', 'laser_off')

#: One temperature sample: board tick (ms), host time (s), °C, the DAC in force,
#: the internal photodiode or None. Saved as a row in this order.
ThermalSample = namedtuple('ThermalSample', 'sensor_tick_ms host_t_s temperature_c dac pd_raw')
COLUMNS = list(ThermalSample._fields)


def run_thermal_step(ctrl, target, step_dac, baseline_s=BASELINE_S, edge_s=EDGE_S,
                     settle_pp_mdegc=SETTLE_PP_MDEGC, lock_timeout_s=LOCK_TIMEOUT_S,
                     hold_s=LOCK_HOLD_S, settle_window_s=SETTLE_WINDOW_S, check=None, say=None,
                     on_sample=None, before_settle=None):
    """Run the heat-load step on ``target`` with a connected, streaming ``ctrl``.

    ``check()`` is called from every wait and raises to cancel; ``say(message)``
    reports each stage; ``on_sample(sample)`` gets each new :class:`ThermalSample`.
    ``before_settle(laser)`` runs once the boards are powered, before the TEC wait.

    Returns the capture as a dict: ``target``, ``step_dac``, ``target_c``,
    ``baseline_s``, ``edge_s``, ``lock_wait_s``, ``settle_pp_mdegc`` (the spread
    when the baseline began), ``started_utc``, ``pd_available``, ``profile`` (the
    638's box profile), ``clock_offset_s``, ``columns``, ``samples`` (rows) and
    ``edges``. Never returns with a rail powered.
    """
    if target not in (638, 1550):
        raise ValueError('target must be 638 or 1550')
    say = say or (lambda message: None)
    samples, edges = [], []
    capture = {'target': target, 'step_dac': int(step_dac), 'target_c': None,
               'baseline_s': float(baseline_s), 'edge_s': float(edge_s), 'lock_wait_s': None,
               'settle_pp_mdegc': None,
               'started_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
               'pd_available': False, 'profile': None, 'clock_offset_s': None,
               'columns': list(COLUMNS), 'samples': samples, 'edges': edges}
    cancelled = []

    def checked():
        try:
            if check is not None:
                check()
        except BaseException:
            cancelled.append(True)
            raise
    laser = ManualLaser(ctrl, target, check=checked, say=say)
    last = {'tick': None}

    def record(seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            laser.check()
            laser.renew()
            snapshot, b = laser.board()
            if b is not None and b.temperature_sample_tick_ms != last['tick']:
                last['tick'] = b.temperature_sample_tick_ms
                pd = (int(b.pd_raw_average) if b.pd_valid and not b.pd_stale
                      and not b.pd_backend_disabled and not b.pd_fault else None)
                capture['pd_available'] = capture['pd_available'] or pd is not None
                sample = ThermalSample(int(b.temperature_sample_tick_ms),
                                       snapshot.received_monotonic_ns / 1e9
                                       - b.temperature_age_ms / 1e3,
                                       b.temperature_mdegc / 1000.0, laser.dac, pd)
                samples.append(sample)
                if on_sample is not None:
                    on_sample(sample)
            time.sleep(POLL_S)

    def step_to(dac):
        dac_from = laser.dac
        sent = laser.set_dac(dac)
        edges.append({'dac_from': dac_from, 'dac_to': int(dac),
                      'host_t_s': (sent.sent + sent.acked) / 2.0,
                      'uncertainty_s': (sent.acked - sent.sent) / 2.0,
                      'applied_dac': int(sent.reply.applied_value), 'sensor_t_s': None})

    say(f'Powering both laser boards for the {target} nm TEC…')
    try:
        with manual_session(ctrl):
            if target == 638:
                # The board's own cap is the limit, checked before the laser moves
                cap = int(ctrl.read_optical_cap_638()['max_dac'])
                if step_dac > cap:
                    raise RuntimeError(f"Step DAC {step_dac} is above this 638 board's cap ({cap}); "
                                       'lower the step or flash firmware with a higher cap.')
                try:
                    capture['profile'] = dict(ctrl.read_profile_638())
                except Exception:
                    capture['profile'] = None   # firmware without box profiles
            if before_settle is not None:
                before_settle(laser)
            capture['target_c'], capture['lock_wait_s'], capture['settle_pp_mdegc'] = \
                laser.settle_tec(settle_pp_mdegc, hold_s=hold_s, window_s=settle_window_s,
                                 timeout_s=lock_timeout_s)
            say(f'{target} nm TEC: recording {baseline_s:g} s with the laser off…')
            record(baseline_s)
            say(f'{target} nm laser on at DAC {step_dac}: recording {edge_s:g} s…')
            step_to(step_dac)
            record(edge_s)
            say(f'{target} nm laser off: recording {edge_s:g} s…')
            step_to(0)
            record(edge_s)
            say('Stopping (confirmed shutdown)…')
    except Exception as exc:
        # If the master stopped the box itself, say why (a cancel is not its doing)
        message = with_master_fault(exc, ctrl)
        if message != str(exc) and not cancelled:
            raise RuntimeError(message) from exc
        raise
    if samples:
        capture['clock_offset_s'] = statistics.median(s.host_t_s - s.sensor_tick_ms / 1000.0
                                                      for s in samples)
        for edge in edges:
            edge['sensor_t_s'] = edge['host_t_s'] - capture['clock_offset_s']
    return capture


def capture_arrays(capture):
    """Board-clock time (s, zero at the laser-on edge), °C, DAC and photodiode (NaN
    where none) of a capture, and its edge times: ``(t, temp_c, dac, pd, edge_times)``.

    Reads samples as rows (``columns`` order, or ThermalSample's when absent) or as
    dicts (the lab's older captures); edges without ``sensor_t_s`` are placed on the
    board clock through the median offset.
    """
    rows = capture.get('samples') or []
    if not rows:
        raise ValueError('no temperature samples were recorded')
    if isinstance(rows[0], dict):
        def column(name):
            return [r.get(name) for r in rows]
    else:
        index = {name: i for i, name in enumerate(capture.get('columns') or COLUMNS)}

        def column(name):
            return [r[index[name]] for r in rows]
    tick_s = np.array(column('sensor_tick_ms'), dtype=float) / 1e3
    temp = np.array(column('temperature_c'), dtype=float)
    dac = np.array(column('dac'), dtype=float)
    pd = np.array([np.nan if v is None else v for v in column('pd_raw')], dtype=float)
    edge_t = [e.get('sensor_t_s') for e in capture.get('edges') or []]
    if any(t is None for t in edge_t):
        host = column('host_t_s')
        if None in host:
            edge_t = []
        else:
            offset = float(np.median(np.array(host, dtype=float) - tick_s))
            edge_t = [e['host_t_s'] - offset for e in capture['edges']]
    zero = edge_t[0] if edge_t else tick_s[0]
    return tick_s - zero, temp, dac, pd, [t - zero for t in edge_t]


def _finite(value):
    return None if value is None or not math.isfinite(value) else float(value)


def _rounded(values, digits=6):
    return [float(f'{v:.{digits}g}') if math.isfinite(v) else None
            for v in np.asarray(values, dtype=float)]


def analyse_thermal_step(capture, band_mdegc=RECOVERY_BAND_MDEGC, pre_edge_s=PRE_EDGE_S,
                         tail_s=TAIL_S):
    """Baseline noise and, per edge, its excursion, recovery and residual, as a JSON-safe
    dict. Time zero is the laser-on edge; ``t_s`` and ``error_mdegc`` are the trace."""
    t, temp, _dac, pd, edges = capture_arrays(capture)
    target_c = float(capture['target_c'])
    err = (temp - target_c) * 1000.0
    band = float(band_mdegc)
    baseline = t < 0 if edges else np.ones_like(t, dtype=bool)
    result = {'target_c': target_c, 'step_dac': capture.get('step_dac'), 'band_mdegc': band,
              'samples': int(len(t)),
              'sample_period_ms': _finite(round(float(np.median(np.diff(t))) * 1000.0, 3)
                                          if len(t) > 1 else None),
              'clock_offset_s': capture.get('clock_offset_s'),
              't_s': _rounded(t), 'error_mdegc': _rounded(err, 5), 'edge_t_s': list(edges),
              'baseline_noise_mdegc': _finite(float(np.std(err[baseline]))
                                              if baseline.sum() > 1 else None),
              'baseline_offset_mdegc': _finite(float(np.mean(err[baseline]))
                                               if baseline.any() else None),
              'edges': []}
    for i, at in enumerate(edges):
        last = i + 1 == len(edges)
        end = float(t[-1]) if last else edges[i + 1]
        # Up to the next edge, never a sample of it
        pre = (t >= at - pre_edge_s) & (t < at)
        post = (t > at) & ((t <= end) if last else (t < end))
        edge = {'name': EDGE_NAMES[i] if i < len(EDGE_NAMES) else f'edge_{i}', 't_s': at,
                'window_s': end - at, 'samples': int(post.sum()), 'peak_mdegc': None,
                'time_to_peak_s': None, 'recovery_s': None, 'residual_mdegc': None,
                'settled': False}
        if pre.any() and post.any():
            pre_level = float(np.mean(temp[pre]))
            t_post, err_post = t[post], err[post]
            deviation = (temp[post] - pre_level) * 1000.0
            peak = int(np.argmax(np.abs(deviation)))
            outside = np.abs(err_post) > band
            if not outside.any():
                recovery = 0.0
            elif outside[-1]:
                recovery = None   # still out of the band when the window ended
            else:
                recovery = float(t_post[np.nonzero(outside)[0][-1] + 1] - at)
            tail = t_post >= end - tail_s
            pd_post = pd[post]
            edge.update(pre_level_c=pre_level, peak_mdegc=float(deviation[peak]),
                        time_to_peak_s=float(t_post[peak] - at), recovery_s=recovery,
                        settled=recovery is not None,
                        residual_mdegc=float(np.mean(err_post[tail])),
                        pd_start=_finite(float(pd_post[0])),
                        pd_end=_finite(float(np.nanmean(pd_post[tail]))
                                       if np.isfinite(pd_post[tail]).any() else None))
        result['edges'].append(edge)
    result['passed'] = bool(len(result['edges']) == len(EDGE_NAMES) and all(
        e['settled'] and e['residual_mdegc'] is not None and abs(e['residual_mdegc']) <= band
        for e in result['edges']))
    return result
