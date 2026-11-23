"""One laser board in a whole-system manual session, as the Self-check drives it.

``manual_session(ctrl)`` powers both boards with START low and their DACs at zero,
and ends with a confirmed STOP on every path. Inside it, a :class:`ManualLaser`
keeps the leases renewed, reads the board's telemetry, sets its DAC with a
confirmed SET, waits for its TEC and takes photodiode readings no older than a
SET. Every one of the Self-check's laser steps uses it, so a laser is driven
one way everywhere.
"""
import time
from collections import namedtuple
from contextlib import contextmanager

#: The firmware's THERMAL_ERROR_LIMIT: the TEC counts as locked inside it.
LOCK_BAND_C = 0.3
#: How long the TEC must stay in the band.
LOCK_HOLD_S = 1.0
#: The window over which the TEC must also be steady, and how steady.
SETTLE_WINDOW_S = 1.0
SETTLE_PP_MDEGC = 20.0
LOCK_TIMEOUT_S = 60.0
#: Manual leases expire unless renewed; the SDK's own loops renew at ~0.6 s.
RENEW_S = 0.5
POLL_S = 0.005
#: Longest wait for a photodiode sample after a SET (it publishes every 50 ms).
FRESH_PD_S = 2.0
#: A photodiode reading this close to full scale is saturated (the SDK's manual sweep's margin):
#: its amplifier runs from 5 V into the 3.3 V ADC input, which software cannot protect.
SATURATION_MARGIN = 5

#: A confirmed SET: the board's reply, and the host times it was sent and acknowledged.
DacSet = namedtuple('DacSet', 'reply sent acked')


@contextmanager
def manual_session(ctrl, **options):
    """A whole-system manual session that ends with a confirmed STOP on every path."""
    try:
        ctrl.begin_system_manual(**options)
        yield
    except BaseException as primary:
        try:
            end_manual(ctrl)
        except Exception as cleanup:
            raise RuntimeError(f'{primary}; the shutdown was not confirmed either: '
                               f'{cleanup}') from primary
        raise
    ctrl.finish_system_manual()


def end_manual(ctrl, stop_timeout_s=4.0):
    """Confirmed STOP: the manual session's own if it opened, else a global one."""
    if ctrl.system_manual_active:
        return ctrl.finish_system_manual()
    return ctrl.stop_system_confirmed(timeout_s=stop_timeout_s)


def with_master_fault(message, ctrl):
    """``message`` with the master's fault appended, when the master stopped the box itself."""
    fault = getattr(ctrl, 'last_master_fault', None)
    if fault and 'master fault' not in str(message):
        return f'{message} (master fault: {fault})'
    return str(message)


def saturated(board):
    """The board's photodiode reading is at full scale."""
    return board.pd_raw_average >= min(4090, max(0, int(board.pd_full_scale) - SATURATION_MARGIN))


def pd_is_fresh(snapshot, board, link_flags, prior_tick, elapsed_ms):
    """A photodiode sample taken after the last SET: a new sample tick, younger than the
    time since, from a healthy link and board."""
    physical_age_ms = board.pd_age_ms + snapshot.host_age_s * 1000.0
    return (board.pd_sample_tick_ms != prior_tick and board.pd_valid and
            not board.pd_stale and not board.pd_backend_disabled and
            board.pd_fault == 0 and link_flags == 0 and
            board.temperature_valid and not board.temperature_stale and
            board.temperature_fault == 0 and
            physical_age_ms < elapsed_ms and not snapshot.host_stale)


class ManualLaser:
    """One laser board inside a manual session: its leases, its telemetry, its DAC.

    ``check()`` is called from every wait and raises to cancel; ``say(message)``
    reports progress. ``cap``, when given, refuses SETs above it here.
    """

    def __init__(self, ctrl, target, check=None, say=None, cap=None):
        self.ctrl, self.target = ctrl, target
        self.check = check or (lambda: None)
        self.say = say or (lambda message: None)
        self.cap = cap
        self.dac = 0
        self.pd_tick = None   # the last photodiode sample taken, by its tick
        self._renew = time.monotonic() + RENEW_S

    def renew(self):
        if time.monotonic() >= self._renew:
            self.ctrl.renew_system_manual()
            self._renew = time.monotonic() + RENEW_S

    def board(self):
        """``(snapshot, board)`` with a valid temperature on a healthy link, else Nones.
        A temperature sensor fault raises."""
        snapshot = self.ctrl.telemetry
        if snapshot is None or snapshot.host_stale:
            return None, None
        b = getattr(snapshot, f'board_{self.target}')
        if b is None:
            return None, None
        if b.temperature_fault:
            raise RuntimeError(f'{self.target} nm temperature sensor fault {b.temperature_fault}')
        if getattr(snapshot, f'link_flags_{self.target}') or not b.temperature_valid \
                or b.temperature_stale:
            return None, None
        return snapshot, b

    def target_c(self):
        """The TEC's target, °C, as the board reports it."""
        from ..control import CHANNEL_TEMPERATURE, MANUAL_GET
        return self.ctrl.system_manual_command(
            self.target, MANUAL_GET, CHANNEL_TEMPERATURE).applied_value / 1000.0

    def set_dac(self, dac):
        """A confirmed SET: the board must apply exactly ``dac``. Returns a :class:`DacSet`."""
        from ..control import CHANNEL_LASER_DAC, MANUAL_SET
        dac = int(dac)
        if self.cap is not None and not 0 <= dac <= self.cap:
            raise ValueError(f'DAC {dac} is outside 0..{self.cap}')
        self.renew()
        sent = time.monotonic()
        reply = self.ctrl.system_manual_command(self.target, MANUAL_SET, CHANNEL_LASER_DAC, dac)
        acked = time.monotonic()
        if int(reply.applied_value) != dac:
            raise RuntimeError(f'the {self.target} nm board applied DAC {reply.applied_value}, '
                               f'not {dac}')
        self.dac = dac
        return DacSet(reply, sent, acked)

    def settle_tec(self, settle_pp_mdegc=SETTLE_PP_MDEGC, hold_s=LOCK_HOLD_S,
                   window_s=SETTLE_WINDOW_S, timeout_s=LOCK_TIMEOUT_S, target_c=None):
        """Wait for the TEC to hold its target; returns ``(target °C, seconds, peak to peak)``.

        It holds once inside the firmware's ±LOCK_BAND_C for ``hold_s`` and, unless
        ``settle_pp_mdegc`` is None, its last ``window_s`` of samples span no more than
        that: the band alone lets a measurement start while the TEC still pulls in.
        """
        target = self.target
        if target_c is None:
            target_c = self.target_c()
        self.say(f'Waiting for the {target} nm TEC to settle at {target_c:.2f} °C…')
        started = time.monotonic()
        in_band_since, recent, last_tick, spread = None, [], None, None
        while True:
            self.check()
            self.renew()
            _snapshot, b = self.board()
            now = time.monotonic()
            if b is not None and abs(b.temperature_mdegc / 1000.0 - target_c) <= LOCK_BAND_C:
                in_band_since = in_band_since or now
                if settle_pp_mdegc is None:
                    if now - in_band_since >= hold_s:
                        return target_c, time.monotonic() - started, None
                else:
                    tick = b.temperature_sample_tick_ms
                    if tick != last_tick:
                        last_tick = tick
                        recent = [(k, v) for k, v in recent + [(tick, b.temperature_mdegc)]
                                  if tick - k <= window_s * 1000]
                    if recent and recent[-1][0] - recent[0][0] >= window_s * 1000 - 50:
                        spread = max(v for _, v in recent) - min(v for _, v in recent)
                        if now - in_band_since >= hold_s and spread <= settle_pp_mdegc:
                            return target_c, time.monotonic() - started, spread
            else:
                in_band_since, recent = None, []
            if now - started > timeout_s:
                if in_band_since is None:
                    raise TimeoutError(f'the {target} nm TEC did not reach ±{LOCK_BAND_C:g} °C of '
                                       f'{target_c:.2f} °C in {timeout_s:.0f} s')
                raise TimeoutError(
                    f'the {target} nm TEC did not settle at {target_c:.2f} °C in '
                    f'{timeout_s:.0f} s' + ('' if spread is None else
                                            f' ({spread:.0f} m°C peak to peak)'))
            time.sleep(0.02)

    def fresh_pd(self, since, timeout_s=FRESH_PD_S):
        """The board's first photodiode sample taken after host time ``since`` (the SDK's
        ``pd_is_fresh`` rule: a new sample tick, younger than the time since)."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self.check()
            self.renew()
            snapshot, b = self.board()
            elapsed_ms = int((time.monotonic() - since) * 1000)
            if b is not None and pd_is_fresh(snapshot, b, 0, self.pd_tick, elapsed_ms):
                self.pd_tick = b.pd_sample_tick_ms
                return b
            time.sleep(POLL_S)
        raise TimeoutError(f'no fresh {self.target} nm photodiode sample within '
                           f'{timeout_s:g} s of the DAC change')
