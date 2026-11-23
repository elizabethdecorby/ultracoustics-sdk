"""Lock acquisition: where the dip fell against the 638's search.

The Self-check's lock step keeps every acquisition page it sees and the raw ADC
from START until LOCKED or ERROR; :func:`analyse_acquisition` reads them. The
firmware's hit and bounds are the truth; the trace only says where the dip fell.

The 638 bisects from its calibration DAC. With dwell N a step lasts (N+1)T, its
counted reads cover [-T, (N-1)T - ACQ_READ_END_MS] around the DAC write, and the
buffer up to NT after the write is never read.
"""
import math
from itertools import count

import numpy as np

from ...config import ADC_MAX_VALUE
from ...readout import CLIP_FRACTION, DARK_MAX_COUNTS

ACQ_FRAME_MS = 0.8192           # T, one buffer of the stream
ACQ_READ_END_MS = 0.013
ACQ_TOP_DAC = 52400             # the calibration DAC, both profiles
ACQ_RESOLUTION = 10             # the search stops once its bounds are this close
ACQ_HIT_FACTOR = 1.10           # a hit is a read below this times the calibration minimum
#: Dwell on attempts 1, 2, 3: flat on these 638 builds, longer on each retry from bd6cd70 on
ACQ_ATTEMPTS = 3
ACQ_FLAT_FIRMWARE = frozenset({'8135918'})
ACQ_DWELLS_FLAT = (6, 6, 6)
ACQ_DWELLS_RETRY = (6, 8, 10)
#: After a full failed search the next calibration transit comes (11 + writes (N+1)) T later:
#: 83.56 ms at N=6 (bd6cd70 and 8135918), 62.3 at N=4, 51.6 at N=3 measured. A spacing this far
#: off is an attempt that failed elsewhere, or a dwell that is not the one expected.
ACQ_ATTEMPT_FRAMES = 11
ACQ_SEARCH_WRITES = 13
ACQ_SPACING_TOLERANCE_MS = 0.5
#: The write moves by tens of µs, so a dip this near a read boundary can go either way: the
#: firmware's own hit decides, and the step reads 'on the edge of the reads'. A dip seen to
#: begin 0.09 ms past the read end was never read.
ACQ_EDGE_MS = 0.08
#: A lock whose dip could hide in an unread buffer by moving less than this is a near miss
ACQ_MARGIN_WARN_MS = 0.3
#: A lock lands within 300 DAC of where its search ended (14 real locks). On a warm dome a
#: step's dip can come after the next step's write and be credited to it: one such search
#: ended 1932 DAC low, and the flank walk reached the dip with about 100 DAC to spare
ACQ_SEARCH_OFF_DAC = 1000
# Finding the search in the raw trace. The DAC step to the calibration value crosses the dip in
# tens of µs; the plateau follows, then the first write.
ACQ_TRANSIT_FRACTION = 0.75     # of the baseline: calibration wants a dip at least 25% deep
ACQ_CLEAR_FRACTION = 0.9        # of the baseline: before a transit and on its plateau
ACQ_BEFORE_MS = (0.2, 0.1)      # before a transit
ACQ_PLATEAU_MS = (2.0, 6.0)     # after a transit
ACQ_MIN_ATTEMPT_MS = 50.0       # the shortest search; dips nearer than this are its own steps
ACQ_FIRST_WRITE_MS = (6.3, 6.9)
ACQ_FIRST_WRITE_TYPICAL_MS = 6.67
ACQ_FIRST_WRITE_COUNTS = 150
ACQ_TRACE_POINTS = 1400
ACQ_TRACE_PAD_MS = 0.5
#: Why the 638 gave an attempt up, in words, as its firmware numbers them
ACQ_REASONS = ('no failure', 'calibration read an empty minimum',
               'the calibration dip was too shallow', 'calibration timed out',
               'the search never crossed its threshold', 'the lock integrator wound out',
               'flank: bad input', 'flank: no crossing', 'flank: weak slope',
               'flank: did not converge')
ACQ_CALIBRATION_FAILS = (1, 2, 3)
ACQ_CAL_DIP = 2
ACQ_ROOT_NO_HIT = 4
ACQ_FLANK_FAILS = (6, 7, 8, 9)
ACQ_WHERE = {'read': 'read', 'unread': 'in the unread buffer', 'edge': 'on the edge of the reads'}


def _report():
    """The report module, whose helpers and wording this one shares (it imports this module)."""
    from . import system_report
    return system_report


def firmware_638(steps):
    """The 638 application's version from the 'device' step, or None if it was not read."""
    for step in steps:
        if step['key'] == 'device':
            firmware = (step.get('data') or {}).get('firmware') or {}
            return ((firmware.get('638') or {}).get('app') or {}).get('version')
    return None


def acquisition_dwells(firmware):
    """The root search's dwell on attempts 1, 2, 3 for a 638 application version."""
    return ACQ_DWELLS_FLAT if firmware in ACQ_FLAT_FIRMWARE else ACQ_DWELLS_RETRY


def no_lock_reason(result, live):
    """What the report names when START did not lock."""
    if live is None:
        return 'the 638 nm board sent no status after START'
    if result['verdict'] != 'hidden_all':
        return result['text']
    attempts = result['attempts']
    return (f'Every earlier check passed and the lock search missed the dip: {len(attempts)} '
            f"attempts, no hit, final DAC {attempts[-1]['final_dac']}. This is search timing, not "
            'the optics.')


def attempt_text(attempt):
    """One attempt in words, for the report's row."""
    parts = [] if attempt['page'] else ['No page received, so from the raw trace.']
    if attempt['baseline']:
        parts.append(f"Baseline {attempt['baseline']:.0f}, minimum {attempt['minimum']:.0f} "
                     f"({attempt['depth_pct']:.0f}% deep), threshold {attempt['threshold']:.0f}.")
    if attempt['steps'] or attempt['lower_dac'] or attempt['upper_dac']:
        parts.append(f"Best reading {attempt['best_reading']}, bounds {attempt['lower_dac']} to "
                     f"{attempt['upper_dac']}"
                     + (f", final DAC {attempt['final_dac']}." if attempt['page'] else '.'))
    if attempt['page']:
        parts.append('Hit.' if attempt['hit'] else f"No hit: {attempt['reason']}.")
    if attempt['dwell'] is not None:
        parts.append(f"Dwell {attempt['dwell']} ({attempt['dwell_source']}).")
    elif attempt['dwell_source']:
        parts.append(f"Dwell unknown ({attempt['dwell_source']}): timing not compared.")
    for step in attempt['steps']:
        if step['step'] in attempt['hidden_steps'] or step['step'] == attempt['margin_step']:
            parts.append(f"Step {step['step']} (DAC {step['dac_from']} to {step['dac_to']}): "
                         f"below threshold {step['from_ms']:+.2f} to {step['to_ms']:+.2f} ms, "
                         f"{ACQ_WHERE[step['where']]}; reads end {attempt['unread_ms'][0]:+.2f} "
                         f"and resume {attempt['unread_ms'][1]:+.2f} ms.")
    return ' '.join(parts)


def _hide_margin(start, stop, unread, period):
    """How far a dip spanning ``start``..``stop`` ms has to move to sit wholly in an unread
    buffer; negative when it does, by how far it has to move to come out. None when it is wider
    than the buffer, and cannot hide."""
    low, high = unread
    if stop - start >= high - low:
        return None
    if low <= start and stop <= high:
        return -min(start - low, high - stop)
    return min(max(low + q * period - start, stop - high - q * period) for q in (-1, 0, 1))


def _transits(x, baseline, ms):
    """Each calibration transit: the sample where the DAC's step to the calibration value takes
    the signal down through the dip, from a clear level, with the plateau after it."""
    below = np.flatnonzero(x < ACQ_TRANSIT_FRACTION * baseline)
    if not below.size:
        return []
    clear, found = ACQ_CLEAR_FRACTION * baseline, []
    for i in below[np.r_[True, np.diff(below) > ms]].tolist():
        if found and i - found[-1] < ACQ_MIN_ATTEMPT_MS * ms:
            continue
        before = x[max(0, i - int(ACQ_BEFORE_MS[0] * ms)):max(0, i - int(ACQ_BEFORE_MS[1] * ms))]
        plateau = x[i + int(ACQ_PLATEAU_MS[0] * ms):i + int(ACQ_PLATEAU_MS[1] * ms)]
        if before.size and plateau.size and before.min() > clear and plateau.min() > clear:
            found.append(i)
    return found


def _first_write(x, transit, ms):
    """The sample where the signal first answers the search's first DAC write."""
    level = np.median(x[transit + int(3 * ms):transit + int(6 * ms)])
    start = transit + int(ACQ_FIRST_WRITE_MS[0] * ms)
    near = x[start:transit + int(ACQ_FIRST_WRITE_MS[1] * ms)].astype(np.int32)
    moved = np.flatnonzero(np.abs(near - level) > ACQ_FIRST_WRITE_COUNTS)
    return start + int(moved[0]) if moved.size else transit + int(ACQ_FIRST_WRITE_TYPICAL_MS * ms)


def _search(x, write, dwell, threshold, ms, bounds=None):
    """The 638's bisection step by step over the raw trace, and the DACs it took.

    ``bounds``, the page's final (lower, upper), give the firmware's own decision on each
    step: it is the truth, the raw reads only have to agree with it (within the edge band).
    Without a page the reads decide.
    """
    frame, edge = ACQ_FRAME_MS * ms, ACQ_EDGE_MS * ms
    period = (dwell + 1) * frame
    read_end, read_resume = (dwell - 1) * frame - ACQ_READ_END_MS * ms, dwell * frame
    low, high, dac, steps = 0, ACQ_TOP_DAC, ACQ_TOP_DAC, []
    while high - low >= ACQ_RESOLUTION:
        at = write + len(steps) * period
        window = x[int(at - frame - edge):int(at + read_end + edge)]
        if len(window) < 4 * edge:
            break
        margin = int(edge)
        counted, sure = window[margin:-margin or None], window[2 * margin:-2 * margin or None]
        target = (low + high) // 2
        swept = min(dac, target), max(dac, target)
        hit = (swept[0] <= bounds[0] and bounds[1] <= swept[1] if bounds
               else bool(counted.min() < threshold))
        step = dict(step=len(steps) + 1, dac_from=dac, dac_to=target, best_read=int(counted.min()),
                    hit=hit, below=False, agrees=(bool(window.min() < threshold) if hit
                                                  else not sure.min() < threshold))
        trace = x[int(at):int(at + period)]
        below = np.flatnonzero(trace < threshold)
        if below.size:
            breaks = np.flatnonzero(np.diff(below) > max(1, ms / 100))
            if breaks.size:   # the first stretch: a later one is the next step's crossing
                below = below[:breaks[0] + 1]
            read = (below < read_end - edge) | (below > read_resume + edge)
            on_edge = ((np.abs(below - read_end) <= edge) | (np.abs(below - read_resume) <= edge))
            start, stop = float(below[0]) / ms, float(below[-1]) / ms
            bottom = int(below[0] + np.argmin(trace[below[0]:below[-1] + 1]))
            step.update(below=True, from_ms=round(start, 3), to_ms=round(stop, 3),
                        bottom_ms=round(bottom / ms, 3), bottom=int(trace[bottom]),
                        where='read' if read.any() else 'edge' if on_edge.any() else 'unread')
            step['hidden'] = step['where'] != 'read' and not hit
            step['margin_ms'] = _hide_margin(start, stop, (read_end / ms, read_resume / ms),
                                             period / ms)
        steps.append(step)
        if hit:
            low, high = swept
        elif dac == low:
            low = target
        else:
            high = target
        dac = target
    return steps, low, high


def _spacing_ms(dwell):
    """From one calibration transit to the next, after a full failed search at ``dwell``."""
    return (ACQ_ATTEMPT_FRAMES + ACQ_SEARCH_WRITES * (dwell + 1)) * ACQ_FRAME_MS


def analyse_acquisition(pages, samples=None, gaps=(), locked=False, sample_rate=10_000_000,
                        dwells=ACQ_DWELLS_RETRY):
    """How a START went: the 638's acquisition pages against the raw ADC. JSON-safe.

    ``pages`` are the acquisition pages seen, as dicts, in any order, and some will be missing.
    ``samples`` is the raw ADC across the acquisition, None when there is none; ``gaps`` are
    [first, last] sample ranges of it where the stream lost data. ``locked``: the START ended
    locked. ``dwells`` are the dwells of attempts 1, 2, 3 (acquisition_dwells); a measured
    spacing only checks them.

    The firmware's page is the truth for hit, reason and bounds; the trace says where the
    dip fell. Returns ``verdict`` (a key), ``text``, ``attention``, per ``attempts`` their
    numbers and bisection ``steps`` with the ``margin_ms`` to an unread buffer, and ``figure``,
    the most telling step as a short binned trace. Times are ms after the step's DAC write.
    """
    ms = sample_rate / 1000.0
    x = None if samples is None or not len(samples) else np.asarray(samples)
    level = None
    if x is not None:
        coarse = x[::max(1, int(ms / 100))]
        lit = coarse[coarse > DARK_MAX_COUNTS]
        level = float(np.percentile(lit, 90)) if lit.size else None
    # Later bursts are recovery, past what this START needed
    first_tick = min((page['start_tick_ms'] for page in pages), default=None)
    pages = [page for page in pages if page['start_tick_ms'] == first_tick]
    baseline = max((page['baseline'] for page in pages), default=0) or level
    transits = _transits(x, baseline, ms) if x is not None and baseline else []
    # The transits count the attempts. Event ids start at 1 when the 638 boots at START and run on
    # through any recovery; an attempt lost in calibration counts, with no search to find.
    lost = sum(page['retry_reason'] in ACQ_CALIBRATION_FAILS for page in pages)
    last = max((page['event_id'] for page in pages), default=0)
    base = last - len(transits) - lost if last > ACQ_ATTEMPTS else 0
    pages = {page['event_id'] - base: page for page in pages}
    searched = (k for k in count(1)
                if pages.get(k, {}).get('retry_reason') not in ACQ_CALIBRATION_FAILS)
    transit_of = dict(zip(searched, transits))
    attempts, marks = [], {}
    for number in sorted(set(pages) | set(transit_of)):
        attempt, mark = _attempt(x, number, pages.get(number), transit_of, dwells, ms)
        attempts.append(attempt)
        marks[number] = mark
        if attempt['dwell'] is None and attempt['dwell_source']:
            dwells = ()   # the schedule did not hold: nothing after this one can lean on it
    spans = [mark for mark in marks.values() if mark]
    lost_data = bool(spans) and any(stop >= spans[0][0] and start <= spans[-1][1]
                                    for start, stop in gaps)
    unknown = [a['attempt'] for a in attempts if a['dwell'] is None and a['dwell_source']]
    note = ''
    if lost_data:
        note = 'The stream lost data during the acquisition, so its timing is not reliable.'
        for attempt in attempts:
            attempt.update(steps=[], hidden_steps=[], read_missed_steps=[], margin_ms=None,
                           margin_step=None)
    elif unknown:
        note = (f"The dwell of attempt {', '.join(map(str, unknown))} could not be confirmed, so "
                'its timing was not compared.')
    seconds = 0.0 if x is None else len(x) / sample_rate
    key, text, attention = _acquisition_verdict(attempts, locked, level, note, seconds)
    return _report().plain(dict(verdict=key, text=text, attention=attention, locked=bool(locked),
                      timing_reliable=not note, timing_note=note, pages_seen=len(pages),
                      recorded_s=round(seconds, 3), level_counts=level, attempts=attempts,
                      figure=_acquisition_figure(x, attempts, marks, ms)))


def _attempt(x, number, page, transit_of, dwells, ms):
    """One attempt's part of the result, and its (transit, search end, first write) samples."""
    page = page or {}
    reason = page.get('retry_reason')
    named = reason is not None and 0 <= reason < len(ACQ_REASONS)
    attempt = dict(attempt=number, page=bool(page), hit=page.get('hit'), retry_reason=reason,
                   reason=ACQ_REASONS[reason] if named else None if reason is None
                   else f'reason {reason}', baseline=page.get('baseline'),
                   minimum=page.get('minimum'), depth_pct=None, threshold=None,
                   best_reading=page.get('best_feedback'), lower_dac=page.get('lower_dac'),
                   upper_dac=page.get('upper_dac'), final_dac=page.get('final_dac'),
                   iterations=page.get('iterations'), dwell=None, dwell_source=None,
                   spacing_ms=None, unread_ms=None, steps=[], hidden_steps=[],
                   read_missed_steps=[], margin_ms=None, margin_step=None)
    transit = transit_of.get(number)
    if transit is not None and not page:
        plateau = x[transit + int(ACQ_PLATEAU_MS[0] * ms):transit + int(ACQ_PLATEAU_MS[1] * ms)]
        attempt.update(baseline=int(np.median(plateau)),
                       minimum=int(x[max(0, transit - int(0.05 * ms)):
                                     transit + int(0.3 * ms)].min()))
    if attempt['baseline']:
        attempt.update(threshold=round(ACQ_HIT_FACTOR * attempt['minimum'], 1),
                       depth_pct=round(100.0 * (1 - attempt['minimum'] / attempt['baseline']), 1))
    if transit is None:
        return attempt, None
    if number > len(dwells):
        attempt['dwell_source'] = 'no schedule for this attempt'
        return attempt, None
    dwell, source = dwells[number - 1], 'firmware schedule'
    following = transit_of.get(number + 1)
    if following is not None:
        attempt['spacing_ms'] = round((following - transit) / ms, 2)
        if not page or reason == ACQ_ROOT_NO_HIT:   # others space differently
            expected = _spacing_ms(dwell)
            if abs(attempt['spacing_ms'] - expected) > ACQ_SPACING_TOLERANCE_MS:
                attempt['dwell_source'] = (f"spacing {attempt['spacing_ms']:.2f} ms, not "
                                           f'{expected:.2f}')
                return attempt, None
            source += ', spacing agrees'
    searched = page.get('hit') or reason in ACQ_FLANK_FAILS
    bounds = ((page['lower_dac'], page['upper_dac'])
              if searched or reason == ACQ_ROOT_NO_HIT else None)
    write = _first_write(x, transit, ms)
    steps, low, high = _search(x, write, dwell, attempt['threshold'], ms, bounds)
    if searched:   # no next transit: the raw reads have to fit the firmware's decisions
        if not all(step['agrees'] for step in steps):
            attempt['dwell_source'] = "the raw reads do not fit the firmware's decisions"
            return attempt, None
        source += ', search agrees'
    if not page:
        attempt.update(hit=any(step['hit'] for step in steps), lower_dac=low, upper_dac=high,
                       best_reading=min((step['best_read'] for step in steps), default=None))
    counted = [step for step in steps if step.get('margin_ms') is not None
               and (step['hit'] or step['hidden'])]
    closest = min(counted, key=lambda step: step['margin_ms'], default=None)
    period = (dwell + 1) * ACQ_FRAME_MS
    attempt.update(dwell=dwell, dwell_source=source, steps=steps,
                   hidden_steps=[step['step'] for step in steps if step.get('hidden')],
                   unread_ms=[round((dwell - 1) * ACQ_FRAME_MS - ACQ_READ_END_MS, 3),
                              round(dwell * ACQ_FRAME_MS, 3)],
                   margin_ms=None if closest is None else round(closest['margin_ms'], 3),
                   margin_step=None if closest is None else closest['step'])
    if reason == ACQ_ROOT_NO_HIT and page['best_feedback'] >= attempt['threshold']:
        attempt['read_missed_steps'] = [step['step'] for step in steps if not step['agrees']]
    return attempt, (transit, write + int(len(steps) * period * ms), write)


def _acquisition_verdict(attempts, locked, level, note, seconds):
    """(key, plain text, needs a look): what was seen, never why it happened."""
    advice = _report().LASER_CHECK_ADVICE
    full_scale = CLIP_FRACTION * ADC_MAX_VALUE
    if not attempts:
        if locked or not seconds:
            return 'unavailable', ('Not available: neither the pages nor the raw trace show the '
                                   'search.'), False
        if level is None:
            return 'no_light', ('Not locked: the detector stayed dark in the '
                                f'{seconds:.1f} s recorded.'), False
        if level >= full_scale:
            return 'saturated', (f'Not locked: the detector sits at {level:.0f} counts with no '
                                 f'dip. {advice}'), False
        return 'no_acquisition', ('Not locked: light reached the detector and no search started '
                                  f'in the {seconds:.1f} s recorded.'), False
    if not locked and all(a['retry_reason'] == ACQ_CAL_DIP for a in attempts):
        last = attempts[-1]
        if last['baseline'] >= full_scale:
            return 'saturated', (f"Not locked: the detector sits at {last['baseline']} counts "
                                 f"with no dip ({last['depth_pct']:.0f}% deep). "
                                 + advice), False
        return 'shallow', ('Not locked: the calibration dip was too shallow on '
                           f"{_attempts_text(attempts)}: {last['depth_pct']:.0f}% deep on the "
                           'last.'), False
    winner = next((a for a in attempts if a['hit']), attempts[-1]) if locked else None
    outcome = f"Locked on attempt {winner['attempt']}." if locked else 'Not locked.'
    odd = [(a['attempt'], step) for a in attempts for step in a['read_missed_steps']]
    if odd:
        return 'unexplained', (f'{outcome} Attempt {odd[0][0]}, step {odd[0][1]}: the signal went '
                               'below threshold well inside a read buffer and the attempt was '
                               'still scored a miss. This should not happen.'), True
    if locked:
        earlier = [a for a in attempts if a['attempt'] < winner['attempt']]
        text = ' '.join([outcome] + [_missed_text(a) for a in earlier])
        off = abs(winner['final_dac'] - winner['lower_dac']) if winner['page'] else 0
        if off > ACQ_SEARCH_OFF_DAC:
            return 'locked_off_search', (
                f"{text} A near miss: the search ended at DAC {winner['lower_dac']}, {off} from "
                f"the lock at DAC {winner['final_dac']}; only the final walk found the dip."), True
        if note:
            return 'locked_later' if earlier else 'timing_unreliable', f'{text} {note}', False
        margin = winner['margin_ms']
        if margin is not None and margin < ACQ_MARGIN_WARN_MS:
            step = next(s for s in winner['steps'] if s['step'] == winner['margin_step'])
            return 'locked_near_miss', (
                f"{text} A near miss: step {step['step']} went below threshold "
                f"{step['from_ms']:+.2f} to {step['to_ms']:+.2f} ms, {margin:.2f} ms from hiding "
                'in an unread buffer.'), True
        return 'locked_later' if earlier else 'locked_first', text, False
    if note and not any(a['steps'] for a in attempts):
        return 'timing_unreliable', f'{outcome} {note}', False
    if all(a['hidden_steps'] for a in attempts):
        last = next(s for s in attempts[-1]['steps'] if s.get('hidden'))
        place = ('in an unread buffer' if all(s['where'] == 'unread' for a in attempts
                                              for s in a['steps'] if s.get('hidden'))
                 else 'in or on the edge of an unread buffer')
        return 'hidden_all', (f'Not locked: on {_attempts_text(attempts)} a real dip fell {place} '
                              f"of the search (step {last['step']}, {last['from_ms']:+.2f} to "
                              f"{last['to_ms']:+.2f} ms on the last)."), False
    if all(a['steps'] for a in attempts) and not any(
            step['below'] for a in attempts for step in a['steps']):
        return 'no_dip', (f'Not locked: on {_attempts_text(attempts)} the search never went below '
                          'its threshold, read or unread.'), False
    return 'other', ' '.join([outcome] + [_missed_text(a) for a in attempts]
                             + ([note] if note else [])), False


def _attempts_text(attempts):
    return 'attempt 1' if len(attempts) == 1 else f'all {len(attempts)} attempts'


def _missed_text(attempt):
    """What became of an attempt that did not lock."""
    number = attempt['attempt']
    if attempt['hit']:
        return f'Attempt {number} hit, and no lock followed.'
    for step in attempt['steps']:
        if step.get('hidden'):
            return (f"Attempt {number} missed: on step {step['step']} the dip was "
                    f"{ACQ_WHERE[step['where']]} ({step['from_ms']:+.2f} to {step['to_ms']:+.2f} "
                    'ms).')
    return f"Attempt {number}: {attempt['reason'] or 'no page received'}."


def _acquisition_figure(x, attempts, marks, ms):
    """The most telling step as a binned trace: one read and missed, else the last dip left
    unread, else the winning search's closest call."""
    chosen = ([(a, a['read_missed_steps'][0]) for a in attempts if a['read_missed_steps']]
              + [(a, a['hidden_steps'][0]) for a in reversed(attempts) if a['hidden_steps']]
              + [(a, a['margin_step']) for a in attempts if a['margin_step']])
    if not chosen:
        return None
    attempt, number = chosen[0]
    step = attempt['steps'][number - 1]
    period = (attempt['dwell'] + 1) * ACQ_FRAME_MS
    start = marks[attempt['attempt']][2] + int(((number - 1) * period - ACQ_TRACE_PAD_MS) * ms)
    trace = x[start:start + int((period + 2 * ACQ_TRACE_PAD_MS) * ms)]
    size = max(1, math.ceil(len(trace) / ACQ_TRACE_POINTS))
    counts = trace[:len(trace) // size * size].reshape(-1, size).mean(1)
    return dict(attempt=attempt['attempt'], step=number, dac_from=step['dac_from'],
                dac_to=step['dac_to'], threshold=attempt['threshold'],
                first_ms=round(-ACQ_TRACE_PAD_MS + size / ms / 2, 4), bin_ms=round(size / ms, 5),
                counts=_report().rounded(counts, 5), unread_ms=attempt['unread_ms'])
