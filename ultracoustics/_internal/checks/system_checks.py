"""The software and USB checks: the paths every later test depends on, over the master's USB.

The Self-check's first step runs :func:`run_system_checks`:

1. A confirmed STOP returns the master to IDLE, with its record of the run before.
2. It advertises stream formats 1 (board telemetry) and 2 (638 optical diagnostics).
3. A whole-system manual session powers both laser boards with START low and the
   laser DACs at zero; a board that does not answer gets a second power-on.
4. A leased round trip to each board (its TEC target and laser DAC), and on the 638
   its DAC cap, optical state, box profile and controller timing. The replies carry
   the applied value: proof the board executed the command.
5. Optionally each TEC reaches its target with the lasers at zero (``settle_tec``).
6. Each board's telemetry: a fresh temperature with no fault, and its photodiode
   (one compiled out of the firmware is recorded, not failed).
7. A confirmed STOP with both rails off, on every path.
8. Optionally the 638's stored lock gains (its tuning record).

No laser emits. The 638's cap is read here, so later tests ramp to what the board says.
"""
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from .laser_board import end_manual

STOP_TIMEOUT_S = 4.0
#: Telemetry newer than the manual session's power-up, before judging the boards.
TELEMETRY_SETTLE_S = 0.5
#: A laser board that does not answer after power-on gets one more power-on: a
#: board that answers the second time is flagged, but the checks still run.
POWER_UP_ATTEMPTS = 2
POWER_UP_RETRY_GAP_S = 1.0
#: How long begin_system_manual waits for both boards: 1.25 s, then up to 3 s
POWER_UP_WAIT_S = 4
#: begin_system_manual's timeout names each board's state; these are its words
BOARD_STATES = {'link unavailable': 'did not answer',
                'snapshot missing': 'sent no status',
                'no post-power snapshot': 'sent no status after power-on',
                'temperature invalid': 'sent no valid temperature',
                'temperature stale': 'sent a stale temperature',
                'temperature too old': 'sent no fresh temperature'}

# The master's boot states and fault codes, as its firmware numbers them
# (boot_sequencer.h BootState_t and BootFaultCode_t, plain sequential enums)
BOOT_STATES = ('IDLE', 'WARM', 'A0_POWER_1550', 'A1_WAIT_1550', 'A2_WAIT_638', 'RUN',
               'FAULT', 'OVERRIDE', 'PROGRAM_1550', 'FLASH_1550', 'PROGRAM_638', 'FLASH_638',
               'VERSION_1550', 'VERSION_638', 'IAP', 'FWINFO', 'METRICS',
               'TELEMETRY_DIAGNOSTIC', 'STREAM_CONTROL', 'RUNTIME_METRICS', 'IDENTITY')
BOOT_FAULTS = (
    'none', 'optical signal never acquired at startup (red ×1)',
    'optical signal lost waiting for the 638 (red ×2)', 'optical signal lost while running (red ×3)',
    '1550 board not ready (blue ×1)', '638 board not ready (blue ×2)',
    '1550 board dropped its ready signal (blue ×3)', 'fan stalled (amber ×1)',
    'DAC write rejected (violet ×1)', 'DAC write timed out (violet ×2)',
    'firmware version query failed (violet ×3)', 'slave flash failed (violet ×4)',
    'firmware image bad header (violet ×5)', 'firmware image bad CRC (violet ×6)',
    'firmware image write failed (violet ×7)', 'illegal state (white ×1)',
    '1550 startup DAC refused (blue ×4)', 'host stopped answering (white ×4)')


def state_name(code):
    return BOOT_STATES[code] if isinstance(code, int) and 0 <= code < len(BOOT_STATES) \
        else f'state {code}'


def fault_name(code):
    return BOOT_FAULTS[code] if isinstance(code, int) and 0 <= code < len(BOOT_FAULTS) \
        else f'fault {code}'


def runtime_text(metrics):
    parts = [f'state {state_name(metrics.current_state)}',
             f'ADC overflows {metrics.pssi_overflows}', f'late {metrics.pssi_service_late}']
    active = getattr(metrics, 'active_percent', None)
    if active is not None and metrics.flags & 8:
        parts.append(f'CPU {active:.1f}%')
    return ', '.join(parts)


def ended_text(metrics):
    return (f'ended in {state_name(metrics.terminal_state)} after '
            f'{state_name(metrics.last_operational_state)}; fault: {fault_name(metrics.fault_code)}')


@dataclass
class Check:
    """One line of the checks. ``passed`` None: recorded, but it decides nothing."""
    key: str
    label: str
    passed: Optional[bool]
    value: str = ''


@dataclass
class SystemChecks:
    """What :func:`run_system_checks` found."""
    items: list = field(default_factory=list)
    #: What was read off the boards, by name
    values: dict = field(default_factory=dict)
    #: A sentence about the whole, when there is something to say
    summary: str = ''
    telemetry: bool = True       # the master streams board telemetry (format 1)
    optical: bool = False        # and 638 optical diagnostics (format 2)
    cap_638: Optional[int] = None
    #: What later tests cannot run without, and was found broken: 'link', 'telemetry',
    #: 'boards', 'tec_1550', 'tec_638' -> why, in words. The first one found stands.
    broken: dict = field(default_factory=dict)
    #: What stopped the checks early, if anything did (they still shut down)
    exception: Optional[BaseException] = None

    def add(self, key, label, passed, value=''):
        self.items.append(Check(key, label, passed, str(value)))

    @property
    def error(self):
        exc = self.exception
        return None if exc is None else f'{type(exc).__name__}: {exc}'

    @property
    def passed(self):
        """Every deciding check passed and nothing stopped them."""
        return self.exception is None and all(c.passed for c in self.items if c.passed is not None)

    def failures(self):
        return [c for c in self.items if c.passed is False]


def run_system_checks(ctrl, progress=None, check=None, read_tuning=True,
                      power_up_attempts=POWER_UP_ATTEMPTS, retry_gap_s=POWER_UP_RETRY_GAP_S,
                      settle_s=TELEMETRY_SETTLE_S, stop_timeout_s=STOP_TIMEOUT_S, settle_tec=None):
    """Run the checks on a connected, streaming ``ctrl``; returns :class:`SystemChecks`.

    ``progress(message)`` reports each stage; ``check()`` is called before each and
    raises to cancel, which propagates once the box is stopped. Any other error is
    kept in ``exception`` and the result returned. ``read_tuning`` also reads the
    638's stored lock gains after the shutdown. ``settle_tec(target)`` waits for a
    board's TEC inside the manual session and returns ``(target_c, seconds, spread)``;
    a TimeoutError or RuntimeError from it is that TEC not reaching its target.
    """
    say = progress or (lambda message: None)
    cancelled = []

    def stage(message):
        if check is not None:
            try:
                check()
            except BaseException:
                cancelled.append(True)
                raise
        say(message)

    result = SystemChecks()
    try:
        _run(ctrl, result, stage, say, power_up_attempts, retry_gap_s, settle_s, stop_timeout_s,
             settle_tec)
        if read_tuning and 638 in ctrl.laser_boards and hasattr(ctrl, 'read_tuning_638'):
            stage("Reading the 638 nm board's tuning record…")
            _tuning(ctrl, result)
    except Exception as exc:
        if cancelled:
            raise
        result.exception = exc
    return result


def _run(ctrl, result, stage, say, attempts, retry_gap_s, settle_s, stop_timeout_s, settle_tec=None):
    from ..control import CHANNEL_LASER_DAC, CHANNEL_TEMPERATURE, MANUAL_GET
    stage('Stopping the box and reading the record of its last run…')
    try:
        metrics = ctrl.stop_system_confirmed(timeout_s=stop_timeout_s)
    except Exception as exc:
        result.broken.setdefault('link', f'The box did not confirm STOP: {exc}.')
        raise
    result.values['previous_run'] = metrics
    result.add('master_idle', 'Stops on request (IDLE confirmed)', metrics.current_state == 0,
               runtime_text(metrics))
    # What ran before: context (often the very problem), not a failure now
    result.add('previous_run', 'Last run before this report', None, ended_text(metrics))
    if metrics.fault_code:
        result.summary = (f'The last run before this report ended with a fault: '
                          f'{fault_name(metrics.fault_code)}.')

    stage('Reading the stream capabilities…')
    caps = ctrl.telemetry_capabilities(timeout_s=1.0)
    result.values['stream_capabilities'] = caps
    result.add('format1', 'Board telemetry (stream format 1)', caps.supports_format1,
               'supported' if caps.supports_format1 else 'not supported')
    result.add('format2', '638 nm optical diagnostics (stream format 2)', caps.supports_format2,
               'supported' if caps.supports_format2 else 'not supported')
    result.optical = bool(caps.supports_format2)
    if not caps.supports_format1:
        result.telemetry = False
        result.summary = ('The main board firmware has no board telemetry, so the steps '
                          'that need it are not run.')
        result.broken.setdefault('telemetry', 'the main board firmware has no board telemetry')
        return

    stage('Powering both laser boards, lasers held at zero…')
    try:
        if not _power_up(ctrl, result, stage, attempts, retry_gap_s):
            return
        for target in ctrl.laser_boards:
            stage(f'Manual control round trip to the {target} nm board…')
            try:
                temp = ctrl.system_manual_command(target, MANUAL_GET, CHANNEL_TEMPERATURE)
                dac = ctrl.system_manual_command(target, MANUAL_GET, CHANNEL_LASER_DAC)
            except Exception as exc:
                result.add(f'manual_{target}', f'{target} nm board answers', False, exc)
                result.broken.setdefault('boards', f'the {target} nm board did not answer: {exc}')
                continue
            result.values[f'temperature_target_{target}_c'] = temp.applied_value / 1000.0
            result.add(f'manual_{target}', f'{target} nm board answers', dac.applied_value == 0,
                       f'TEC target {temp.applied_value / 1000.0:.2f} °C, laser DAC '
                       f'{dac.applied_value} (expected 0)')
        if 638 in ctrl.laser_boards:
            _optical_638(ctrl, result, stage)
        if settle_tec is not None and 'boards' not in result.broken:
            for target in ctrl.laser_boards:
                _settle_tec(ctrl, result, stage, target, settle_tec)
        stage('Checking the board telemetry…')
        time.sleep(settle_s)
        snapshot = ctrl.telemetry
        if snapshot is None or snapshot.host_stale:
            result.add('telemetry_fresh', 'Board telemetry arrives', False, 'none in the last 500 ms')
        else:
            for target in ctrl.laser_boards:
                _board_telemetry(result, snapshot, target)
    finally:
        say('Confirming the shutdown…')
        try:
            metrics = end_manual(ctrl, stop_timeout_s)
            result.values['runtime_after_checks'] = metrics
            result.add('shutdown', 'Confirmed STOP, both rails off', metrics.current_state == 0,
                       runtime_text(metrics))
        except Exception as exc:
            result.add('shutdown', 'Confirmed STOP, both rails off', False, exc)


def _power_up(ctrl, result, stage, attempts, retry_gap_s):
    """Open the manual session, powering the boards on again if one did not answer.

    True once both answered. Each try's time and the SDK's diagnostics go in
    ``values['power_up']``: whether a board was slow or silent.
    """
    tries = result.values['power_up'] = []
    errors = []
    for attempt in range(attempts):
        if attempt:
            stage('A laser board did not answer: powering both on again…')
            time.sleep(retry_gap_s)
        started = time.monotonic()
        try:
            ctrl.begin_system_manual(optical_diagnostics=result.optical)
        except Exception as exc:   # the SDK has stopped the box again
            errors.append(exc)
            tries.append({'seconds': time.monotonic() - started, 'error': str(exc),
                          'diagnostics': getattr(exc, 'diagnostics', None)})
            continue
        tries.append({'seconds': time.monotonic() - started, 'error': None})
        break
    label = 'Laser boards power up and report'
    if not errors:
        result.add('slaves_powered', label, True, 'yes')
        return True
    late = power_up_text(errors[0])
    if len(errors) == len(tries):
        result.add('slaves_powered', label, False, f'{late}, on {len(tries)} power-ons')
        result.broken.setdefault('boards', f'{late}, on {len(tries)} power-ons')
        return False
    result.add('slaves_powered', label, False, f'{late}; both answered when powered on again')
    result.summary = ' '.join(filter(None, [
        result.summary, 'A laser board did not answer the first time it was powered on, '
                        'and did the second time; the checks below are from then.']))
    return True


def power_up_text(exc):
    """begin_system_manual's timeout in words: which board did not come up."""
    reason = str(exc).partition('timed out: ')[2]
    states = re.findall(r'\b(638|1550) ([a-z][^;]*)', reason)
    if not states:
        return str(exc)
    late = [f'the {target} nm board '
            f'{BOARD_STATES.get(state, "reported " + state)}'
            for target, state in states if state != 'ready']
    fine = [f'the {target} nm board did' for target, state in states if state == 'ready']
    text = f'{" and ".join(late)} within {POWER_UP_WAIT_S} s of power-on'
    return text + (f' ({" and ".join(fine)})' if fine else '')


def _settle_tec(ctrl, result, stage, target, settle_tec):
    """The TEC reaches its target with the lasers at zero: everything after relies on it."""
    label = f'{target} nm TEC reaches its target'
    stage(f'Waiting for the {target} nm TEC to reach its target…')
    ctrl.renew_system_manual()   # the session is older than the laser's own renewals
    try:
        target_c, seconds, _spread = settle_tec(target)
    except (TimeoutError, RuntimeError) as exc:
        result.add(f'tec_{target}', label, False, exc)
        result.broken.setdefault(f'tec_{target}', str(exc))
    else:
        result.add(f'tec_{target}', label, True, f'{target_c:.2f} °C in {seconds:.1f} s')


def _optical_638(ctrl, result, stage):
    stage('Reading the 638 nm optical configuration…')
    try:
        result.cap_638 = int(ctrl.read_optical_cap_638()['max_dac'])
        result.values['dac_cap_638'] = result.cap_638
        result.add('cap_638', '638 nm laser DAC cap', None, f'{result.cap_638} counts')
    except Exception as exc:
        result.add('cap_638', '638 nm laser DAC cap', False, f'not read: {exc}')
    try:
        state = ctrl.read_state_638()
        result.values['optical_state_638'] = state
        result.add('optical_idle_638', '638 nm optical control idle', state['state'] == 0,
                   state['name'])
    except Exception as exc:
        result.add('optical_idle_638', '638 nm optical control idle', False, exc)
    for key, label, read in (('profile_638', '638 nm box profile', ctrl.read_profile_638),
                             ('timing_638', '638 nm controller timing',
                              ctrl.read_controller_timing_638)):
        try:
            value = read()
        except Exception as exc:
            result.add(key, label, None, f'not read: {exc}')
            continue
        result.values[label] = value
        result.add(key, label, None, ', '.join(f'{k} {v}' for k, v in dict(value).items()))


def _board_telemetry(result, snapshot, target):
    board = getattr(snapshot, f'board_{target}')
    link = getattr(snapshot, f'link_flags_{target}')
    if board is None:
        result.add(f'telemetry_{target}', f'{target} nm board telemetry',
                   False, 'missing from the telemetry record')
        return
    result.values[f'telemetry_{target}'] = board
    result.values[f'link_flags_{target}'] = link
    result.values[f'pd_available_{target}'] = not board.pd_backend_disabled
    temp_ok = (link == 0 and board.temperature_valid and not board.temperature_stale
               and board.temperature_fault == 0)
    result.add(f'temperature_{target}', f'{target} nm temperature', temp_ok,
               f'{board.temperature_mdegc / 1000.0:.2f} °C'
               + ('' if temp_ok else f' (fault {board.temperature_fault}, link 0x{link:04X})'))
    if board.pd_backend_disabled:
        # A build choice, not a fault (the 1550 needs DEV_PD_RAW_READ=ON)
        result.add(f'pd_{target}', f'{target} nm internal photodiode', None,
                   'not built into this firmware')
        return
    pd_ok = board.pd_valid and not board.pd_stale and board.pd_fault == 0
    result.add(f'pd_{target}', f'{target} nm internal photodiode', pd_ok,
               f'{board.pd_raw_average} of {board.pd_full_scale} counts, laser off')


def _tuning(ctrl, result):
    label = '638 nm lock gains stored on the box'
    try:
        tuning = dict(ctrl.read_tuning_638(timeout_s=8.0))
    except Exception as exc:
        result.add('tuning_638', label, None, f'not read: {exc}')
        return
    result.values['tuning_638'] = tuning
    if not tuning.get('stored'):
        text = 'none stored: the 638 runs its firmware defaults'
    elif tuning.get('lock_kp') is None:
        text = 'record stored without lock gains'
    else:
        text = f"Kp {tuning['lock_kp']:.3f}, Ki {tuning['lock_ki_per_s']:g}/s"
        if tuning.get('report_id'):
            text += f" (calibration {tuning['report_id']})"
    if tuning.get('rejected'):
        text += f"; the 638 ignores it: {tuning['rejected']}"
    result.add('tuning_638', label, False if tuning.get('rejected') else None, text)
