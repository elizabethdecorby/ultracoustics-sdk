"""The LI sweep on a laser's own monitor photodiode, and its analysis.

The Self-check's laser steps run :func:`li_sweep` and :func:`analyse_li`.

**The sweep** steps the laser DAC 0 -> top inside a manual session. Every SET is
confirmed by the board's reply, and every reading is a photodiode sample taken
after it (``ManualLaser.fresh_pd``). It stops at the first reading at full
scale: the photodiode's amplifier runs from 5 V into the 3.3 V ADC input,
which software cannot protect. The laser goes back to zero afterwards.

**The threshold** is where two lines cross: one fitted below the knee (the
spontaneous emission, not flat on these lasers) and one to the first points
above it, where the lasing curve is still straight (NEAR_THRESHOLD_FRACTION of
the rise, at least MIN_FIT_POINTS). A single line through the whole upper curve
is wrong here: the curve bends over, so that line lands thousands of DAC below
the knee.

The knee is the largest rise in slope averaged over KNEE_WINDOW segments, so a
one-point mode hop (a dip and its recovery) cancels itself out, searched in the
lower half of the rise. Two rules keep it off features above threshold:

- the knee and the line above it stop where the light first falls: only a laser
  that lases can lose light (a lasting hop, a probe's fringe), and the rise
  after a fall would otherwise score as the knee;
- the rise counts relative to the slope before it: at threshold the slope grows
  many times over, while above it a reading that repeats the one before and the
  catch-up after it only double the slope, yet can rise more in counts.

Reported: ``threshold_dac``, ``knee_dac``, the slope just above threshold and at
the top (``linearity`` is their ratio: 1 is straight, below 1 rolls over), the
largest single-point step (a mode hop shows up there), the dark level (mean
over the bottom DARK_FRACTION of the DACs) and the last reading.
"""
import numpy as np

from .laser_board import saturated

#: Points in a sweep, unless the caller chooses
POINTS = 100
DARK_FRACTION = 0.05
KNEE_SEARCH_FRACTION = 0.5
#: A fall of more than this many counts below the highest reading so far ends the knee search
FALL_COUNTS = 10
NEAR_THRESHOLD_FRACTION = 0.15
MIN_FIT_POINTS = 3
KNEE_WINDOW = 3
TOP_SLOPE_FRACTION = 0.1
#: The rows li_sweep records
COLUMNS = ['dac', 'pd_counts', 'temperature_c']


class NoPhotodiode(RuntimeError):
    """The board's firmware was built without its internal photodiode."""


def li_sweep(laser, top, points=POINTS, progress=None, on_point=None):
    """Step ``laser`` (a ManualLaser whose TEC has settled) 0 -> ``top`` in ``points``
    evenly spaced DACs and read its photodiode after each SET; back to 0 afterwards.

    ``progress(message, fraction)`` reports each point. ``on_point(point)`` gets each
    reading as a dict: ``dac``, ``dac_applied``, ``pd_counts``, ``pd_full_scale``,
    ``saturated``, ``pd_tick_ms``, ``pd_age_ms``, ``temperature_c``, ``index``, ``count``.

    Returns ``{'top', 'points', 'columns', 'rows', 'pd_full_scale', 'saturated_at_dac',
    'completed'}``; ``rows`` are ``[dac, pd_counts, temperature_c]``. Raises
    NoPhotodiode when the board's firmware has none.
    """
    if not 2 <= points <= 1000:
        raise ValueError('points must be between 2 and 1000')
    top = int(top)
    _snapshot, b = laser.board()
    if b is not None and b.pd_backend_disabled:
        raise NoPhotodiode(f'the {laser.target} nm board firmware has no internal photodiode.')
    rows, saturated_at, full = [], None, None
    try:
        for index in range(points):
            dac = round(top * index / (points - 1))
            if progress is not None:
                progress(f'{laser.target} nm laser at DAC {dac}', index / points)
            sent = laser.set_dac(dac)
            b = laser.fresh_pd(sent.acked)
            full = int(b.pd_full_scale)
            clipped = saturated(b)
            rows.append([dac, int(b.pd_raw_average), round(b.temperature_mdegc / 1000.0, 3)])
            if on_point is not None:
                on_point({'dac': dac, 'dac_applied': int(sent.reply.applied_value),
                          'pd_counts': int(b.pd_raw_average), 'pd_full_scale': full,
                          'saturated': clipped, 'pd_tick_ms': int(b.pd_sample_tick_ms),
                          'pd_age_ms': int(b.pd_age_ms),
                          'temperature_c': b.temperature_mdegc / 1000.0,
                          'index': index, 'count': points})
            if clipped:
                saturated_at = dac
                break
    except BaseException:
        # Back to zero if it can; the session's confirmed STOP is the guarantee,
        # and a second failure here must not hide the first
        try:
            laser.set_dac(0)
        except Exception:
            pass
        raise
    laser.set_dac(0)
    return {'top': top, 'points': points, 'columns': list(COLUMNS), 'rows': rows,
            'pd_full_scale': full, 'saturated_at_dac': saturated_at,
            'completed': saturated_at is None and len(rows) == points}


def analyse_li(dac, pd, saturated=False, full_scale=None):
    """Dark level, threshold, slopes and hops of an LI sweep. ``saturated``: the
    last point is (a sweep stops at its first saturated reading). ``full_scale``:
    the photodiode's, recorded as ``pd_full_scale``."""
    dac, pd = np.asarray(dac, dtype=float), np.asarray(pd, dtype=float)
    if not len(dac):
        raise ValueError('the sweep has no readings')
    top = float(dac.max())
    dark_mask = dac <= max(DARK_FRACTION * top, dac.min())
    dark = float(np.mean(pd[dark_mask]))
    n = len(dac) - 1 if saturated else len(dac)   # a clipped reading bends the curve
    result = {'points': int(len(dac)), 'top_dac': int(dac[-1]), 'pd_at_top': int(pd[-1]),
              'pd_full_scale': None if full_scale is None else int(full_scale),
              'dark_counts': dark, 'dark_noise_counts': float(np.std(pd[dark_mask])),
              'saturated': bool(saturated),
              'saturated_at_dac': int(dac[-1]) if saturated else None,
              'max_unsaturated_dac': int(dac[:n].max()) if n else None,
              'slope_counts_per_kdac': None, 'threshold_dac': None,
              'top_slope_counts_per_kdac': None, 'linearity': None,
              'worst_step_counts': None, 'worst_step_dac': None}
    x, y = dac[:n], pd[:n]
    if n < 2 * MIN_FIT_POINTS + 1:
        return result
    rise = float(y.max()) - dark
    if rise <= 0:
        return result
    seg = np.diff(y) / np.diff(x)
    w = KNEE_WINDOW
    # A slope this small is the photodiode's resolution: below it, nothing rises
    resolution = max(result['dark_noise_counts'], 1.0) / float(np.median(np.diff(x)))

    def rise_at(k):
        return float(np.mean(seg[k:k + w]) - np.mean(seg[k - w:k]))

    def score(k):
        return rise_at(k) / (abs(float(np.mean(seg[k - w:k]))) + resolution)

    fallen = np.flatnonzero(y < np.maximum.accumulate(y) - FALL_COUNTS)
    end = int(fallen[0]) if len(fallen) else n
    candidates = [k for k in range(w, min(end, len(seg) - w + 1))
                  if y[k] <= dark + KNEE_SEARCH_FRACTION * rise
                  and MIN_FIT_POINTS - 1 <= k <= n - MIN_FIT_POINTS - 1]
    if not candidates:
        return result
    knee = max(candidates, key=score)
    if rise_at(knee) <= 0:
        return result
    below_slope, below_icpt = np.polyfit(x[:knee + 1], y[:knee + 1], 1)
    ceiling = y[knee + 1] + NEAR_THRESHOLD_FRACTION * rise
    above = [j for j in range(knee + 1, end) if y[j] <= ceiling]
    if len(above) < MIN_FIT_POINTS:
        above = list(range(knee + 1, min(n, knee + 1 + MIN_FIT_POINTS)))
    if len(above) < MIN_FIT_POINTS:
        return result
    above_slope, above_icpt = np.polyfit(x[above], y[above], 1)
    if above_slope <= below_slope:
        return result
    threshold = (below_icpt - above_icpt) / (above_slope - below_slope)
    top_mask = x >= x[-1] - TOP_SLOPE_FRACTION * (x[-1] - x[0])
    top_slope = np.polyfit(x[top_mask], y[top_mask], 1)[0] if top_mask.sum() >= 2 else None
    steps = [(abs(y[i] - (y[i - 1] + y[i + 1]) / 2.0), i) for i in range(knee + 2, n - 1)]
    worst = max(steps) if steps else None
    result.update(
        threshold_dac=float(threshold), slope_counts_per_kdac=float(above_slope) * 1000.0,
        fit_intercept=float(above_icpt),
        below_slope_counts_per_kdac=float(below_slope) * 1000.0, below_intercept=float(below_icpt),
        knee_dac=int(x[knee]), fit_from_dac=int(x[above[0]]), fit_to_dac=int(x[above[-1]]),
        top_slope_counts_per_kdac=None if top_slope is None else float(top_slope) * 1000.0,
        linearity=None if top_slope is None else float(top_slope / above_slope),
        worst_step_counts=None if worst is None else float(worst[0]),
        worst_step_dac=None if worst is None else int(x[worst[1]]))
    return result
