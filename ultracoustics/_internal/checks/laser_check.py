"""Laser self-check: choose the 1550 nm laser's operating power on the box.

The 1550 runs at a DAC word in its own flash (0x08004000; 15885 when blank),
which its bootloader writes when the master sends CMD_PROGRAM_1550, as the lab's
diagnostics app does. This check picks that word on the closed box with the
probe in, so the laser stays clear of its mode hops and the detector stays
below full scale on the probe's brightest fringe:

1. The current setting: the 1550 on its own firmware (its stored DAC), the 638
   off, and the reflected level on the main detector.
2. The 1550 swept 0 -> SWEEP_TOP_DAC with the 638 off, reading the main
   detector after each confirmed SET. A mode hop shows as a jump in the AC
   noise (the lab's RIN profiles: +9 to +17 dB, the first at DAC 17900-24900
   on the lasers measured, 700-2700 DAC wide, often a second near 23000-25000
   and a third near 28600-30500). A hop starts at the first point at least
   HOP_JUMP_DB above the points before it, with the next still HOP_HOLD_DB
   above, and ends at its last point before HOP_SETTLE_POINTS in a row are back
   under HOP_HOLD_DB above the noise before it. Each later hop is found the
   same way after the one before. The sweep stops where the detector reaches
   CLIP_FRACTION of full scale.
3. The modes between the hops, each used only HOP_MARGIN_DAC clear of the hops
   on either side (and the first MIN_ABOVE_THRESHOLD_DAC above threshold). A
   hop whose noise never settles ends them. The highest mode goes first: its
   highest DAC, then the 638 swept 0 -> its cap there, and the brightest fringe
   must stay at or under TARGET_FRACTION of full scale. If it does not, the DAC
   that would put it there is found on the sweep's curve and checked the same
   way. A mode where nothing fits hands over to the mode below it.

measure() changes nothing on the box; write_setpoint() stores the chosen DAC and
checks that the 1550 comes back at it. The master reports the write only on its
LED and in its fault code (violet x1 refused, x2 timed out), so the check is
the level the stored setting gives, against the sweep's at that DAC.
"""
import time

import numpy as np

from ...signal_level import STAGES  # noqa: F401  (the measurement's stages: the public module's)
from . import system_report as sr

#: Above the report's 25000 (system_report.LI_TOP_DAC), so the sweep reaches the
#: second and third modes on the lasers measured; well under the firmware's 43253
SWEEP_TOP_DAC = 35000
SWEEP_STEP = 100
SETTLE_S = 0.005
POINT_SAMPLES = 20000          # 2 ms at 10 MSPS after each SET
CLIP_FRACTION = sr.PROBE_FULL_SCALE
TARGET_FRACTION = 0.90
HOP_MARGIN_DAC = 1000
HOP_JUMP_DB = 6.0
HOP_HOLD_DB = 3.0
HOP_HISTORY = 8                # points before a candidate hop that set its baseline
#: Quiet points in a row that end a hop (3 and 5 agree on the lab's RIN profiles; 2 cuts
#: some hops short)
HOP_SETTLE_POINTS = 5
MIN_SIGNAL_COUNTS = 50
#: Less light than this back from the probe at the sweep's top: no probe
MIN_RISE_COUNTS = 500
#: The chosen DAC must be at least this far above the lasing threshold
MIN_ABOVE_THRESHOLD_DAC = 1000
FRINGE_TRIES = 4
#: The first guess at brightest fringe over the 638-off level, before it is measured
INITIAL_FRINGE_SCALE = 2.0
STORED_WARMUP_S = 2.0
LEVEL_SAMPLES = 200000
#: The master's program sequence (power up, SPI, ACK, power down, 2 s) and
#: the diagnostics app's wait for it (PROGRAM_SETPOINT_SETTLE_S)
PROGRAM_SETTLE_S = 4.5
#: The master's BootFaultCode_t for a refused / timed-out DAC write
DAC_WRITE_FAULTS = (8, 9)
VERIFY_TOLERANCE = 0.05



class CheckFailed(RuntimeError):
    """The check could not choose a DAC; the message says why."""


def _full_scale():
    from ...config import ADC_MAX_VALUE
    return ADC_MAX_VALUE


def measure(ctrl, progress=None, cancel=None, sample_rate=10_000_000):
    """Measure the box and choose its 1550 DAC; returns a JSON-safe dict.

    ``ctrl`` is connected and streaming and nothing else commands it
    meanwhile. ``progress(event)`` gets ``{'step', 'message', 'fraction'}``.
    Raises CheckFailed when no DAC fits, sr.Cancelled when cancelled. The box
    is left stopped (IDLE) on every path.
    """
    run = sr._Run(ctrl, progress or (lambda event: None), cancel or (lambda: False),
                  sample_rate)
    if 1550 not in ctrl.laser_boards or 638 not in ctrl.laser_boards:
        raise CheckFailed('this box is missing one of its two boards.')
    full = _full_scale()
    try:
        run.key = 'stored'
        run.stage('Stopping the box…')
        ctrl.stop_system_confirmed(timeout_s=sr.STOP_TIMEOUT_S)
        dark = float(ctrl.collect_fresh(LEVEL_SAMPLES, timeout_s=5.0).mean())
        stored = _stored_level(run)

        run.key = 'sweep'
        run.stage('Powering both laser boards…')
        with sr._manual_session(ctrl):
            laser = sr._Laser(run, 1550)
            laser.settle_tec()
            sweep = _sweep_1550(run, laser, full)
            dac, mean, std = (np.array([r[i] for r in sweep['rows']], dtype=float)
                              for i in range(3))
            threshold = sr.analyse_li(dac, mean, sweep['clipped_at_dac'] is not None)['threshold_dac']
            # The GUI shows every message here to the customer (Help › Self-check): no "laser"
            if mean.max() - dark < MIN_RISE_COUNTS:
                raise CheckFailed('hardly any light comes back from the microphone: is it plugged in?')
            if threshold is None:
                raise CheckFailed('the light source shows no threshold on the detector.')
            hops = [(int(dac[first]), None if last is None else int(dac[last]))
                    for first, last in mode_hops(dac, mean, std, dark, threshold)]
            modes = mode_ranges(threshold, hops)
            # A clipped last point is out of the running, and so is every point within a
            # hop's margin
            end = len(dac) - (1 if sweep['clipped_at_dac'] is not None else 0)
            candidates = [[i for i in range(end) if m['low_dac'] <= dac[i]
                           and (m['high_dac'] is None or dac[i] <= m['high_dac'])
                           and mean[i] <= TARGET_FRACTION * full] for m in modes]
            if not any(candidates):
                raise CheckFailed(_no_room(threshold, hops, sweep))

            # The brightest fringe scales with the reflected level, so the sweep's curve
            # says where it fits once its ratio to the 638-off level is measured. Each
            # mode measures its own: the wavelength, and the probe's fringe, jump at a hop.
            run.key = 'fringe'
            tries = FRINGE_TRIES * sum(1 for c in candidates if c)
            checks, ratios, under = [], {}, {}
            for number in range(len(modes), 0, -1):   # the highest mode first
                if candidates[number - 1]:
                    under, ratios[number] = _fit_fringe(run, laser, candidates[number - 1], dac,
                                                        mean, dark, full, checks, tries)
                    if under:
                        break
            if not under:
                worst = checks[-1] if checks else None
                raise CheckFailed(
                    "the probe's brightest fringe stays above "
                    f"{TARGET_FRACTION:.0%} of the detector's full scale"
                    + (f" ({worst['peak_fraction']:.0%} at DAC {worst['dac']})" if worst else '')
                    + ' at every usable setting.')
            chosen = max(under)
            laser.set_dac(0)
            run.say('Stopping (confirmed shutdown)…')
    finally:
        try:
            ctrl.stop_system_confirmed(timeout_s=sr.STOP_TIMEOUT_S)
        except Exception as exc:
            print(f'[laser-check] confirmed STOP after the check: {exc}', flush=True)

    final = under[chosen]
    peak_scale = (final['peak_counts'] - dark) / (mean[chosen] - dark)
    # The current setting's fringe, at its own mode's ratio when another mode was measured
    stored['dac_estimate'], stored['mode'] = estimate_dac(dac, mean, stored['level_counts'], modes)
    ratio = (peak_scale if stored['mode'] in (None, number)
             else ratios.get(stored['mode'], peak_scale))
    stored['peak_fraction'] = (dark + ratio * (stored['level_counts'] - dark)) / full
    return sr.plain({
        'dark_counts': dark, 'full_scale_counts': full, 'target_fraction': TARGET_FRACTION,
        'stored': stored, 'sweep': sweep, 'threshold_dac': threshold,
        'hop_dac': hops[0][0] if hops else None,
        'limit_dac': modes[0]['high_dac'], 'hops': hops, 'modes': modes, 'mode': number,
        'fringe_checks': checks, 'recommended_dac': int(dac[chosen]),
        'expected_level_counts': float(mean[chosen]), 'peak_fraction': final['peak_fraction']})


def _fit_fringe(run, laser, candidates, dac, mean, dark, full, checks, tries):
    """One mode's fringe checks: ``candidates`` are its sweep indices, lowest first.

    Returns ({index: check} for those whose brightest fringe stayed at or under
    TARGET_FRACTION, the fringe's last estimated ratio to the 638-off level).
    Every check is appended to ``checks``.
    """
    def highest(scale):
        fits = [i for i in candidates if dark + scale * (mean[i] - dark) <= TARGET_FRACTION * full]
        return fits[-1] if fits else None
    scale, under = INITIAL_FRINGE_SCALE, {}
    # Nothing fits the first guess: the mode's dimmest point measures the real ratio, so
    # a mode above a hop, already bright, is not passed over on a guess
    index = highest(scale)
    if index is None:
        index = candidates[0]
    for _ in range(FRINGE_TRIES):
        if index is None:
            break
        check = _fringe_check(run, laser, int(dac[index]), full, len(checks), tries)
        checks.append(check)
        measured = (check['peak_counts'] - dark) / (mean[index] - dark)
        # A clipped peak only bounds the ratio from below
        scale = (measured if check['peak_fraction'] < CLIP_FRACTION
                 else max(measured, scale) * 1.5)
        if check['peak_fraction'] <= TARGET_FRACTION:
            under[index] = check
        following = highest(scale)
        if index in under and following == index:
            break   # verified, and the measured ratio allows nothing higher
        index = following
    return under, scale


def _no_room(threshold, hops, sweep):
    if hops and hops[0][0] - HOP_MARGIN_DAC < threshold + MIN_ABOVE_THRESHOLD_DAC:
        return (f'the light source hops mode at DAC {hops[0][0]}, too close to its '
                f'threshold (DAC {threshold:.0f}).')
    if sweep['clipped_at_dac'] is not None:
        return (f'the detector is near full scale from DAC {sweep["clipped_at_dac"]}, too close '
                f'to the threshold (DAC {threshold:.0f}).')
    return 'no setting is clear of the mode hops.'


def _set(laser, dac):
    """A confirmed SET; returns the reply, whose stream position marks the change."""
    from ..control import CHANNEL_LASER_DAC, MANUAL_SET
    laser.renew()
    reply = laser.ctrl.system_manual_command(laser.target, MANUAL_SET, CHANNEL_LASER_DAC, int(dac))
    if int(reply.applied_value) != int(dac):
        raise RuntimeError(f'the {laser.target} nm board applied DAC {reply.applied_value}, '
                           f'not {dac}')
    laser.dac = int(dac)
    return reply


def _after(laser, reply):
    samples = laser.ctrl.read_after_reply(reply, POINT_SAMPLES, settle_s=SETTLE_S)
    return float(samples.mean()), float(samples.std())


def _sweep_1550(run, laser, full):
    rows, clipped_at = [], None
    values = range(0, SWEEP_TOP_DAC + 1, SWEEP_STEP)
    for index, dac in enumerate(values):
        run.check()
        run.say(f'1550 nm laser at DAC {dac}', index / len(values))
        mean, std = _after(laser, _set(laser, dac))
        rows.append([dac, mean, std])
        if mean >= CLIP_FRACTION * full:
            clipped_at = dac
            break
    return {'columns': ['dac', 'mean_counts', 'std_counts'], 'rows': rows,
            'clipped_at_dac': clipped_at}


def _fringe_check(run, laser, dac_1550, full, attempt, tries=FRINGE_TRIES):
    """The 638 swept 0 -> its cap with the 1550 at ``dac_1550``: the brightest level.
    ``attempt`` of ``tries`` sets the progress."""
    ctrl = run.ctrl
    run.stage(f'1550 nm laser at DAC {dac_1550}: waiting for its TEC…', attempt / tries)
    _set(laser, dac_1550)
    laser.settle_tec()
    red = sr._Laser(run, 638)
    cap = int(ctrl.read_optical_cap_638()['max_dac'])   # also raises the SDK's 638 bound
    red.settle_tec()
    values = list(range(0, cap + 1, sr.PROBE_STEP))
    levels = []
    for index, dac in enumerate(values):
        run.check()
        run.say(f'Sweeping the 638 nm laser with the 1550 nm at DAC {dac_1550}: DAC {dac}',
                (attempt + index / len(values)) / tries)
        levels.append(_after(red, _set(red, dac))[0])
    _set(red, 0)
    peak = int(np.argmax(levels))
    return {'dac': dac_1550, 'peak_counts': levels[peak], 'peak_at_638_dac': values[peak],
            'unlocked_counts': levels[0], 'peak_fraction': levels[peak] / full}


def _stored_level(run):
    """The reflected level with the 1550 on its own firmware (its stored DAC), 638 off."""
    ctrl = run.ctrl
    run.stage('Running the 1550 nm laser at its current setting…')
    with sr._manual_session(ctrl):
        ctrl.start_1550_automatic(warmup_s=STORED_WARMUP_S, cancel=run.cancelled)
        run.check()
        samples = ctrl.collect_fresh(LEVEL_SAMPLES, timeout_s=5.0)
        run.say('Stopping (confirmed shutdown)…')
    return {'level_counts': float(samples.mean()), 'std_counts': float(samples.std())}


def first_hop(dac, mean, std, dark, threshold):
    """Index of the first mode hop on a 1550 sweep, or None.

    Above the threshold the AC noise on the detector changes slowly with the
    light; at a hop it jumps. A hop is the first point at least HOP_JUMP_DB
    above the median of the HOP_HISTORY points before it, with the next point
    still HOP_HOLD_DB above: one noisy reading is not a hop. The noise is in
    counts, not relative to the level, so light that falls (the probe's fringe
    as the laser tunes) cannot pass for one.
    """
    signal = np.asarray(mean, dtype=float) - dark
    usable = (np.asarray(dac) > threshold) & (signal > MIN_SIGNAL_COUNTS)
    noise_db = _noise_db(std)
    for i in range(HOP_HISTORY, len(signal) - 1):
        history = range(i - HOP_HISTORY, i)
        if not (usable[i] and usable[i + 1] and all(usable[j] for j in history)):
            continue
        base = float(np.median(noise_db[list(history)]))
        if noise_db[i] - base >= HOP_JUMP_DB and noise_db[i + 1] - base >= HOP_HOLD_DB:
            return i
    return None


def _noise_db(std):
    return 20 * np.log10(np.maximum(np.asarray(std, dtype=float), 1e-9))


def mode_hops(dac, mean, std, dark, threshold):
    """Every mode hop on a 1550 sweep, as (first, last) indices of its noisy points.

    Each starts as first_hop finds it, searched from just after the hop before. It
    lasts until HOP_SETTLE_POINTS points in a row are back under HOP_HOLD_DB above
    the HOP_HISTORY points before it; ``last`` is None when the noise has not
    settled by the sweep's end, and that hop is the last one.
    """
    dac, mean, std = (np.asarray(a, dtype=float) for a in (dac, mean, std))
    noise_db, hops, offset = _noise_db(std), [], 0
    while True:
        first = first_hop(dac[offset:], mean[offset:], std[offset:], dark, threshold)
        if first is None:
            return hops
        first += offset
        quiet = noise_db < float(np.median(noise_db[first - HOP_HISTORY:first])) + HOP_HOLD_DB
        last = next((j - 1 for j in range(first + 1, len(dac) - HOP_SETTLE_POINTS + 1)
                     if quiet[j:j + HOP_SETTLE_POINTS].all()), None)
        hops.append((first, last))
        if last is None:
            return hops
        offset = last + 1


def mode_ranges(threshold, hops):
    """The modes between ``hops`` ((first, last) noisy DACs), lowest first.

    Each is ``{'start_dac', 'stop_dac', 'low_dac', 'high_dac'}``: the mode runs
    from start to just before stop, and may be used from low to high,
    HOP_MARGIN_DAC clear of the hop on either side (the first mode starts at
    the threshold and is used from MIN_ABOVE_THRESHOLD_DAC above it). None for
    stop and high: up to the sweep's end. A hop that never settles ends them.
    """
    modes, start, low = [], threshold, threshold + MIN_ABOVE_THRESHOLD_DAC
    for first, last in hops:
        modes.append({'start_dac': start, 'stop_dac': first, 'low_dac': low,
                      'high_dac': first - HOP_MARGIN_DAC})
        if last is None:
            return modes
        start, low = last + 1, last + HOP_MARGIN_DAC
    modes.append({'start_dac': start, 'stop_dac': None, 'low_dac': low, 'high_dac': None})
    return modes


def dac_for_level(dac, mean, level, start, stop=None):
    """The DAC that gives ``level`` on the sweep's rising curve from ``start`` to
    just before ``stop`` (one mode; None: the sweep's end), or None outside it."""
    dac, mean = np.asarray(dac, dtype=float), np.asarray(mean, dtype=float)
    keep = (dac >= start) & (dac < (np.inf if stop is None else stop))
    x, y = dac[keep], np.maximum.accumulate(mean[keep])
    if len(x) < 2 or not y[0] <= level <= y[-1]:
        return None
    return float(np.interp(level, y, x))


def estimate_dac(dac, mean, level, modes):
    """(DAC, mode number) that gives ``level``, or (None, None) when no mode's curve
    has it or more than one does: the level alone cannot tell those modes apart."""
    found = [(estimate, number) for number, m in enumerate(modes, 1)
             for estimate in [dac_for_level(dac, mean, level, m['start_dac'], m['stop_dac'])]
             if estimate is not None]
    return found[0] if len(found) == 1 else (None, None)


def write_setpoint(ctrl, result, progress=None):
    """Store ``result['recommended_dac']`` as the 1550's DAC and check it took.

    Not cancellable: the master's program sequence must not be cut in half.
    Returns ``{'dac', 'level_counts', 'expected_level_counts', 'dac_estimate',
    'verified'}``; RuntimeError if the master records a refused or timed-out
    write. The box is left stopped.
    """
    run = sr._Run(ctrl, progress or (lambda event: None), lambda: False, 10_000_000)
    dac = int(result['recommended_dac'])
    run.key = 'write'
    run.stage('Stopping the box…')
    before = ctrl.stop_system_confirmed(timeout_s=sr.STOP_TIMEOUT_S).fault_code
    run.stage(f'Storing DAC {dac} on the 1550 nm board…')
    ctrl.program_1550_setpoint(dac, timeout_s=1.0)
    time.sleep(PROGRAM_SETTLE_S)
    after = ctrl.stop_system_confirmed(timeout_s=sr.STOP_TIMEOUT_S).fault_code
    if after in DAC_WRITE_FAULTS and after != before:
        raise RuntimeError(f'the box did not store it: {sr.fault_name(after)}.')
    run.key = 'verify'
    level = _stored_level(run)['level_counts']
    ctrl.stop_system_confirmed(timeout_s=sr.STOP_TIMEOUT_S)
    expected, dark = result['expected_level_counts'], result['dark_counts']
    rows = result['sweep']['rows']
    # On the chosen mode's curve: the box is meant to come back in that mode
    mode = (result['modes'][result['mode'] - 1] if result.get('modes')
            else {'start_dac': result['threshold_dac'], 'stop_dac': result['hop_dac']})
    estimate = dac_for_level([r[0] for r in rows], [r[1] for r in rows], level,
                             mode['start_dac'], mode['stop_dac'])
    return sr.plain({'dac': dac, 'level_counts': level, 'expected_level_counts': expected,
                     'dac_estimate': estimate,
                     'verified': abs(level - expected) <= VERIFY_TOLERANCE * (expected - dark)})


def stored_text(result):
    """The current setting, in the user's words: SignalLevel.current_text."""
    from ...signal_level import SignalLevel
    return SignalLevel.from_details(result).current_text
