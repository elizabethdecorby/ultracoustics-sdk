"""Whole-system manual sessions. Call through one serialized command owner.

Methods are bounded synchronous operations; a GUI must run them off its paint
thread and call renew_system_manual periodically while a session is active.
"""
import math
import struct
import numpy as np
import time
from ._internal.comms import TransportUncertainError
from ._internal.control import (
    MANUAL_GET, MANUAL_TAKE, MANUAL_SET, MANUAL_RELEASE, MANUAL_RENEW,
    CHANNEL_TEMPERATURE, CHANNEL_LASER_DAC,
)
from ._internal.protocol import (
    CMD_BOOT, CMD_KEEPALIVE, CMD_OVERRIDE_ENTER, CMD_POWER, CMD_TRIGGER,
    CMD_STARTUP_1550, CMD_IDENTITY_READ, CMD_IDENTITY_WRITE, pack_command,
)
from ._internal.optical_diagnostics import OpticalTrace
from . import identity as _identity
from . import tec_tuning as _tuning
from .characterization import ProbeCharacterizationResult
from .config import ADC_MAX_VALUE, SAMPLE_RATE

CHANNEL_OPTICAL = 3
CHANNEL_KP = 4
CHANNEL_KI_NEGATIVE = 5
CHANNEL_KI_POSITIVE = 6
CHANNEL_DAC_CAP = 7
CHANNEL_OPTICAL_RUNTIME = 8
CHANNEL_FP_SCAN = 9
CHANNEL_NORMALIZED_PI = 10
CHANNEL_NORMALIZED_PI_INFO = 11
CHANNEL_PROFILE_ID = 12
CHANNEL_TRACE_CONFIG = 13
NORMALIZED_PI_VERSION = 1
NORMALIZED_PI_MAX_KI_PER_S = 0x3fff / 10.0
NORMALIZED_PI_INFO_MASKS = {1: 0x7ff, 2: 0xfffff}  # contract version -> defined info bits
NORMALIZED_PI_RECORD = 1 << 17  # info: lock gains read from the tuning record; bits 18-19 need it
NORMALIZED_PI_V2_MAX_KP = 1023 * 5 / 1000.0      # 0.005 steps
NORMALIZED_PI_V2_MAX_KI_PER_S = 0x3fff            # integer 1/s
NORMALIZED_PI_V2_MAX_ZERO_HZ = 63
FP_SCAN_STATES = ('idle', 'armed', 'qualifying', 'active', 'complete', 'aborted')
FP_SCAN_RESULTS = ('none', 'complete', 'explicit_abort', 'invalid_or_fault')
FP_SCAN_MIN_RECORDS = 500
FP_SCAN_MAX_RECORDS = 1002
GAIN_SCALE = 1_000_000
OPTICAL_STATES = ('IDLE', 'CALIBRATING', 'ROOT_FINDING', 'LOCKED', 'ERROR', 'MEASURE_SLOPE')
OPTICAL_ACTIONS = {'abort': 0, 'start': 1, 'reacquire': 2, 'retune': 3,
                   'trace': 4, 'identify': 5}
#: Master keepalive armed for every manual session (firmware command K). Any
#: command feeds it; sessions renew leases every 0.2-0.5 s.
KEEPALIVE_MS = 2000
#: Lease renewal period for the SDK's own waits.
RENEW_INTERVAL_S = 0.5
#: Stream counters that must not move while a sweep is binning samples.
STREAM_HEALTH_KEYS = ('drops_seq', 'transfer_errors', 'transfer_timeouts', 'malformed')


class SlaveCommandRejected(RuntimeError):
    def __init__(self, target, channel, status, opcode=None, reply=None):
        self.target = target
        self.channel = channel
        self.status = status
        self.opcode = opcode
        self.reply = reply
        reason = {6: 'value outside bounds or TEC not yet LOCKED',
                  7: 'channel is not owned', 8: 'channel is already owned',
                  10: 'optical transition busy'}.get(status, 'command rejected')
        super().__init__(f'Board {target}, channel {channel}: {reason} (status {status})')


def require_applied(reply):
    if reply.status != 0:
        raise SlaveCommandRejected(reply.target, reply.channel, reply.status,
                                   getattr(reply, 'opcode', None), reply)
    return reply


class FPScanError(RuntimeError):
    """Scan failure with any causally paired rows already received."""

    def __init__(self, message, rows=(), cleanup_error=None, arm_rejections=()):
        self.rows = tuple(rows)
        self.cleanup_error = cleanup_error
        self.arm_rejections = tuple(arm_rejections)
        if cleanup_error is not None:
            message = f'{message}; cleanup also failed: {cleanup_error}'
        super().__init__(message)


class SystemControlMixin:
    def _startup_illumination_request(self, action, dac=0, timeout_s=1.0):
        if self._stream is None or not self._stream.running:
            raise RuntimeError('call begin_stream() before 1550 startup control')
        result = self._stream._request_stream_control(
            pack_command(CMD_STARTUP_1550, action, dac),
            'startup_1550', timeout_s=timeout_s)
        return result['firmware_response']

    def read_startup_illumination_1550(self, timeout_s=1.0):
        """Read volatile one-use stage/application state; old firmware fails closed."""
        return self._startup_illumination_request(0, timeout_s=timeout_s)

    def stage_startup_illumination_1550(self, dac, timeout_s=1.0):
        """Stage one automatic START (1..43253); IDLE with both rails off."""
        if type(dac) is not int or not 1 <= dac <= 43253:
            raise ValueError('1550 startup DAC must be an integer from 1 to 43253')
        result = self._startup_illumination_request(1, dac, timeout_s)
        if result.state != 'staged' or result.staged_dac != dac:
            raise RuntimeError('1550 startup staging readback differs from request')
        return result

    def clear_startup_illumination_1550(self, timeout_s=1.0):
        """Clear an unused stage from IDLE with rails off."""
        result = self._startup_illumination_request(2, timeout_s=timeout_s)
        if result.state != 'clear':
            raise RuntimeError('1550 startup stage did not clear')
        return result

    def start_with_startup_illumination_1550(self, dac, timeout_s=1.0):
        """STOP, stage once, start AUTO, and require exact pre-trigger ACK."""
        self.stop_system_confirmed(timeout_s)
        self._require_system_board('Starting the box')
        remaining = .2 - (time.monotonic_ns() - self._last_manual_rail_off_ns) / 1e9
        if remaining > 0:
            time.sleep(remaining)
        try:
            self.stage_startup_illumination_1550(dac, timeout_s)
            delivery = self._send_confirmed(CMD_BOOT, timeout_s=timeout_s)
            self._running = True
            # The 1550 bootloader waits a second before application telemetry
            # can qualify. Keep the per-request timeout independent of the
            # bounded pre-trigger application wait in master firmware.
            deadline = time.monotonic() + max(5.0, timeout_s)
            while time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                status = self.read_startup_illumination_1550(
                    timeout_s=min(timeout_s, max(0.05, remaining)))
                if status.state == 'applied':
                    if status.applied_dac != dac:
                        raise RuntimeError('1550 startup applied DAC differs from request')
                    self._last_manual_rail_off_ns = None
                    return {'delivery': delivery, 'illumination': status}
                if status.state == 'clear':
                    raise RuntimeError('1550 startup cleared before exact application')
                time.sleep(0.025)
            raise TimeoutError('1550 startup application was not acknowledged')
        except Exception as exc:
            # A failed/uncertain stage must never continue into 638 acquisition.
            try:
                self.stop_system_confirmed(timeout_s)
            except Exception as cleanup_exc:
                raise RuntimeError(
                    f'{exc}; STOP verification also failed: {cleanup_exc}') from exc
            raise

    @staticmethod
    def _decode_trace_config(word):
        if type(word) is not int or word < 0 or (word & ~0x1fffff) != 0x400000:
            raise RuntimeError('Unsupported 638 trace configuration version/reserved bits')
        amplitude, hold = word & 0x7f, (word >> 7) & 255
        decimation = ((word >> 15) & 63) + 1
        if not 1 <= amplitude <= 64 or not 1 <= hold <= 255:
            raise RuntimeError('638 returned invalid trace configuration')
        return dict(version=1, amplitude_dac=amplitude, hold_updates=hold,
                    decimation=decimation, packed=word,
                    recorder_filter='four_boxcars' if decimation > 1 else 'none')

    def read_control_trace_config_638(self, timeout_s=1.0):
        """Read versioned identification settings; requires supporting firmware."""
        reply = self._normalized_pi_command_638(MANUAL_GET, CHANNEL_TRACE_CONFIG,
                                               timeout_s=timeout_s)
        if reply.status == 4:
            raise RuntimeError('638 firmware does not support configurable identification')
        return self._decode_trace_config(require_applied(reply).applied_value)

    def configure_control_trace_638(self, amplitude_dac=2, hold_updates=1,
                                    decimation=1, timeout_s=1.0):
        """Configure the next excited trace without starting it or changing PI gains.

        The slave requires healthy locked normalized control and no busy capture.
        For decimation > 1, y/u/d share four length-D boxcar filters at the control
        rate. Record timestamps are output times; nominal filter delay is 2(D-1)
        control updates. Quantization/coherence must be checked before fitting.
        """
        for value, maximum in ((amplitude_dac, 64), (hold_updates, 255), (decimation, 64)):
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError('Trace settings require integer amplitude 1..64, hold 1..255, decimation 1..64')
        self.read_control_trace_config_638(timeout_s)
        word = 0x400000 | ((decimation-1) << 15) | (hold_updates << 7) | amplitude_dac
        reply = require_applied(self._normalized_pi_command_638(
            MANUAL_SET, CHANNEL_TRACE_CONFIG, word, timeout_s))
        if reply.applied_value != word:
            raise RuntimeError('638 trace configuration acknowledgement differs from request')
        result = self.read_control_trace_config_638(timeout_s)
        if result['packed'] != word:
            raise RuntimeError('638 trace configuration readback differs from request')
        return result

    def _normalized_pi_command_638(self, opcode, channel, value=0, timeout_s=1.0):
        if self.system_manual_active:
            if (638, CHANNEL_LASER_DAC) in self._system_manual_owned:
                raise RuntimeError('Release the manually-owned 638 laser DAC before normalized PI control')
            command = self.system_manual_command
        else:
            command = self.manual_command
        return command(638, opcode, channel, value, timeout_s)

    def _normalized_pi_info_638(self, timeout_s=1.0):
        try:
            reply = self._normalized_pi_command_638(MANUAL_GET, CHANNEL_NORMALIZED_PI_INFO,
                                                    timeout_s=timeout_s)
        except (RuntimeError, TimeoutError) as exc:
            if 'manually-owned 638 laser DAC' in str(exc):
                raise
            raise RuntimeError('638 normalized PI capability probe failed; '
                               'check master/638 firmware support and connection: '
                               f'{exc}') from exc
        if reply.status == 4:
            raise RuntimeError('638 firmware does not support normalized PI runtime controls')
        require_applied(reply)
        raw = reply.applied_value
        mask = NORMALIZED_PI_INFO_MASKS.get(raw & 255) if raw >= 0 else None
        if mask is not None and not raw & NORMALIZED_PI_RECORD:
            mask &= 0x3ffff  # record active/rejected are defined only with record support
        # A record's lock gains are either the box defaults or rejected, never both.
        if mask is None or raw & ~mask or ((raw >> 18) & 3) == 3:
            raise RuntimeError(f'638 returned unsupported normalized PI contract {raw}')
        return {'version': raw & 255, 'active': bool(raw & 256),
                'overridden': bool(raw & 512), 'slope_valid': bool(raw & 1024),
                'zero_hz': (raw >> 11) & 63, 'record_supported': bool(raw & NORMALIZED_PI_RECORD),
                'record_active': bool(raw & 1 << 18), 'record_rejected': bool(raw & 1 << 19)}

    def read_normalized_pi_638(self, timeout_s=1.0):
        """Read the explicit symmetric normalized PI pair, second-integrator zero and law."""
        info = self._normalized_pi_info_638(timeout_s)
        reply = require_applied(self._normalized_pi_command_638(
            MANUAL_GET, CHANNEL_NORMALIZED_PI, timeout_s=timeout_s))
        packed = reply.applied_value & 0xffffff  # ACK is signed 24-bit.
        if info['version'] == 2:  # kp in 0.005 units, ki in integer 1/s
            return dict(info, kp=(packed >> 14) * 5 / 1000.0, ki_per_s=float(packed & 0x3fff),
                        packed=packed)
        kp_milli, ki_deci = packed >> 14, packed & 0x3fff
        if kp_milli > 1000:
            raise RuntimeError(f'638 returned out-of-range normalized PI pair {packed}')
        return dict(info, kp=kp_milli / 1000.0, ki_per_s=ki_deci / 10.0,
                    packed=packed)

    def set_normalized_pi_638(self, kp, ki_per_s, zero_hz=0, timeout_s=1.0):
        """Commit Kp, Ki/s and the second-integrator zero together; verify the applied law.

        Contract v1 firmware: 14-bit Ki field 0..1638.3/s, Kp 0..1, no second
        integrator (zero_hz must be 0). Older firmware may reject values above
        1000/s; propagate that rejection without retries. Contract v2 firmware:
        Kp 0..5.115 in 0.005 steps, Ki 0..16383 integer 1/s, zero_hz integer 0..63.
        Runtime overrides do not change the compiled profile defaults.
        """
        if (type(kp) not in (int, float) or type(ki_per_s) not in (int, float) or
                not math.isfinite(kp) or not math.isfinite(ki_per_s) or
                not 0 <= kp <= NORMALIZED_PI_V2_MAX_KP or
                not 0 <= ki_per_s <= NORMALIZED_PI_V2_MAX_KI_PER_S):
            raise ValueError('Normalized PI requires finite Kp 0..5.115 and Ki 0..16383 per second')
        if type(zero_hz) is not int or not 0 <= zero_hz <= NORMALIZED_PI_V2_MAX_ZERO_HZ:
            raise ValueError('Normalized PI zero_hz must be an integer 0..63')
        info = self._normalized_pi_info_638(timeout_s)
        if info['version'] == 1:
            if zero_hz:
                raise ValueError('638 normalized PI contract v1 has no second integrator (zero_hz must be 0)')
            if not 0 <= kp <= 1 or not 0 <= ki_per_s <= NORMALIZED_PI_MAX_KI_PER_S:
                raise ValueError('Normalized PI contract v1 requires Kp 0..1 and Ki 0..1638.3 per second')
        if not info['active'] or not info['slope_valid']:
            raise RuntimeError('638 normalized PI must be active with a valid slope before tuning')
        if info['version'] == 2:
            packed = (round(kp / 0.005) << 14) | round(ki_per_s)
            word = (1 << 30) | (zero_hz << 24) | packed
        else:
            packed = word = (round(kp * 1000) << 14) | round(ki_per_s * 10)
        reply = require_applied(self._normalized_pi_command_638(
            MANUAL_SET, CHANNEL_NORMALIZED_PI, word, timeout_s))
        if (reply.applied_value & 0xffffff) != packed:
            raise RuntimeError('638 normalized PI applied pair differs from request')
        result = self.read_normalized_pi_638(timeout_s)
        if (not result['active'] or not result['overridden'] or result['packed'] != packed or
                result['zero_hz'] != zero_hz or result['version'] != info['version']):
            raise RuntimeError('638 normalized PI verification failed after SET')
        return result

    def restore_normalized_pi_638(self, timeout_s=1.0):
        """Restore the box defaults (tuning record lock gains over the compiled profile); verify control stays active."""
        info = self._normalized_pi_info_638(timeout_s)
        if not info['active'] or not info['slope_valid']:
            raise RuntimeError('638 normalized PI must be active with a valid slope before restore')
        require_applied(self._normalized_pi_command_638(
            MANUAL_SET, CHANNEL_NORMALIZED_PI_INFO, 0, timeout_s))
        result = self.read_normalized_pi_638(timeout_s)
        # The compiled profile has no second integrator; a tuning record may.
        if (not result['active'] or result['overridden'] or
                (result['zero_hz'] and not result['record_active'])):
            raise RuntimeError('638 normalized PI verification failed after restore')
        return result

    # -- Lock PI gains in the 638 tuning record (docs/TEC_TUNING.md) --------

    def _await_mainboard_streaming(self, limit_s=1.0):
        """Return once an ADC packet arrives after this call (False after *limit_s*).

        The mainboard pauses the ADC stream for each maintenance command
        (stream control, runtime metrics, board records) and resumes it on
        its way back to IDLE, about 100 ms after a record reply. Until then
        it refuses 'n'/'N' without a reply and ignores START.
        """
        baseline = int((self.stream_stats or {}).get('packets', 0))
        deadline = time.monotonic() + limit_s
        while int((self.stream_stats or {}).get('packets', 0)) <= baseline:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.001)
        return True

    def _tuning_exchange_638(self, tuning=None, timeout_s=8.0):
        """Read ('n') or write ('N') the 638 tuning record on this connection.

        The stream owns the USB device, so the mainboard's reply comes back
        through it. Returns the TuningResult (OK or EMPTY); RuntimeError
        otherwise.
        """
        what = 'read' if tuning is None else 'write'
        if getattr(self, '_running', False):
            raise RuntimeError('638 tuning record access needs the mainboard in IDLE; STOP first')
        if self.system_manual_active:
            raise RuntimeError('Finish the manual session before 638 tuning record access '
                               '(the mainboard power-cycles the laser boards and returns to IDLE)')
        if self._stream is None or not self._stream.running:
            raise RuntimeError('call begin_stream() before 638 tuning record access')
        command = CMD_IDENTITY_READ if tuning is None else CMD_IDENTITY_WRITE
        record = b'' if tuning is None else _tuning.build_record(tuning)
        self._await_mainboard_streaming()  # out of any earlier maintenance state
        try:
            raw = self._stream._request_stream_control(
                pack_command(command, _tuning.KIND, 638) + record, 'record',
                timeout_s=timeout_s + 0.5, response_timeout_s=timeout_s)['firmware_response']
        except TransportUncertainError:
            raise
        except RuntimeError as exc:
            if 'response timed out' not in str(exc):
                raise
            raise RuntimeError(f'638 tuning record {what}: no reply; the mainboard refused the '
                               'command (busy, or firmware without board records)') from exc
        # Back in IDLE before the next command: another record exchange, or START.
        self._await_mainboard_streaming()
        try:
            result = _tuning.parse_reply(raw)
        except _tuning.TuningError as exc:  # incl. an identity reply from older mainboard firmware
            raise RuntimeError(f'638 tuning record {what}: {exc}') from exc
        if (result.op, result.target) != (_identity.OP_READ if tuning is None else _identity.OP_WRITE, 638):
            raise RuntimeError(f'638 tuning record {what}: the reply does not match the request')
        if result.status in (_identity.STATUS_OK, _identity.STATUS_EMPTY):
            return result
        reason = {_identity.STATUS_NO_RESPONSE: 'the 638 did not answer (no board, or a '
                                                'bootloader without tuning-record support)',
                  _identity.STATUS_BAD_STATE: 'the mainboard is not in IDLE or OVERRIDE; STOP first',
                  _identity.STATUS_BAD_KIND: 'the mainboard firmware does not support tuning records',
                  }.get(result.status, 'failed')
        raise RuntimeError(f'638 tuning record {what}: {reason} ({result.status_name})')

    @staticmethod
    def _tuning_readback(result):
        stored = result.record
        tuning = stored.tuning if stored is not None else _tuning.TecTuning(board=_tuning.BOARD_638)
        rejected = [f'{group} fields rejected: {reason}' for group, reason in (
            ('TEC', stored and stored.rejected), ('lock', stored and stored.lock_rejected)) if reason]
        return dict({name: getattr(tuning, name) for name in
                     _tuning.LOCK_FIELDS + _tuning.TEC_FIELDS + ('tuned_at', 'report_id')},
                    stored=stored is not None, sequence=None if stored is None else stored.seq,
                    rejected='; '.join(rejected) or None)

    def read_tuning_638(self, timeout_s=8.0):
        """Read the 638's tuning record (lock PI gains and TEC fields) on this connection.

        Needs the mainboard in IDLE (STOP first) and no manual session. The
        mainboard power-cycles the 638 into its bootloader for the read and
        pauses the ADC stream; no other command may run meanwhile. Returns
        ``stored`` (a valid record is stored), ``lock_kp``, ``lock_ki_per_s``,
        ``lock_zero_hz``, the TEC fields (``kp``, ``ki``, ``kd``,
        ``integrator_limit``, ``setpoint_filter``,
        ``conditional_integration``), ``tuned_at``, ``report_id``,
        ``sequence`` and ``rejected``: None, or why the 638 would ignore the
        record's TEC fields and/or its lock fields. None means "not set".
        RuntimeError outside IDLE/OVERRIDE, without tuning support (older
        mainboard firmware, or a 638 bootloader without the tuning opcodes)
        and on any other failure.
        """
        return self._tuning_readback(self._tuning_exchange_638(timeout_s=timeout_s))

    def _write_lock_fields_638(self, lock, report_id, timeout_s):
        tuning = _tuning.TecTuning(board=_tuning.BOARD_638, tuned_at=int(time.time()),
                                   report_id=report_id, **lock)
        _tuning.validate(tuning)  # before anything is sent
        stored = self._tuning_exchange_638(timeout_s=timeout_s)
        tuning = _tuning.keep_stored(tuning, stored.record, 'tec')
        result = self._tuning_exchange_638(tuning, timeout_s)
        sent = _tuning.parse_record(_tuning.build_record(tuning)).tuning
        if (result.status != _identity.STATUS_OK or result.record is None
                or not _tuning.same_record(result.record.tuning, sent)):
            raise RuntimeError('638 tuning record write: the read-back record differs from '
                               'the one written (VERIFY_FAIL)')
        return self._tuning_readback(result)

    def write_lock_tuning_638(self, kp, ki_per_s, zero_hz=0, report_id=None, timeout_s=15.0):
        """Store lock PI gains in the 638's tuning record; return the read-back record.

        Read-modify-write: the stored TEC fields are kept (a stored group the
        firmware rejects is dropped; it ignores it either way). The gains must
        lie on the normalized PI contract v2 wire grid (Kp 0..5.115 in 0.005
        steps, Ki an integer 0..16383 per s, zero_hz an integer 0..63), so
        channel 10 later reads back exactly these values; ValueError before
        anything is sent otherwise. *report_id* (at most 24 bytes) and the
        write time are stored for traceability. Same preconditions and errors
        as :meth:`read_tuning_638`; *timeout_s* bounds each mainboard reply.

        The 638 applies the gains as its box defaults at its next start-up:
        after START, :meth:`read_normalized_pi_638` shows ``record_active``,
        not ``overridden``, and these gains. Returns the dict of
        :meth:`read_tuning_638` for the record as read back after the write.
        """
        for name, value, step, high in (('Kp', kp, 0.005, NORMALIZED_PI_V2_MAX_KP),
                                        ('Ki', ki_per_s, 1, NORMALIZED_PI_V2_MAX_KI_PER_S)):
            if (type(value) not in (int, float) or not math.isfinite(value) or
                    not 0 <= value <= high or abs(value / step - round(value / step)) > 1e-6):
                raise ValueError('Lock PI gains need Kp 0..5.115 in 0.005 steps and Ki an '
                                 f'integer 0..16383 per second (the contract v2 wire grid); {name} {value!r}')
        if type(zero_hz) is not int or not 0 <= zero_hz <= NORMALIZED_PI_V2_MAX_ZERO_HZ:
            raise ValueError('Lock PI zero_hz must be an integer 0..63')
        return self._write_lock_fields_638(
            {'lock_kp': round(kp / 0.005) * 5 / 1000.0, 'lock_ki_per_s': float(round(ki_per_s)),
             'lock_zero_hz': zero_hz}, report_id, timeout_s)

    def clear_lock_tuning_638(self, report_id=None, timeout_s=15.0):
        """Remove the lock PI gains from the 638's tuning record; return the read-back record.

        Read-modify-write: the stored TEC fields are kept. From its next
        start-up the 638 runs its compiled profile lock gains
        (``record_active`` False). Same preconditions, errors and return value
        as :meth:`write_lock_tuning_638`.
        """
        return self._write_lock_fields_638({}, report_id, timeout_s)

    def read_profile_638(self, timeout_s=1.0):
        """Read the compiled box profile identity from the 638 board."""
        reply = require_applied(self.manual_command(
            638, MANUAL_GET, CHANNEL_PROFILE_ID, timeout_s=timeout_s))
        names = {1: '638-belycomm-v1', 2: '638-qphotonics-v1'}
        if reply.applied_value not in names:
            raise RuntimeError(f'638 returned unknown profile ID {reply.applied_value}')
        return {'id': reply.applied_value, 'name': names[reply.applied_value]}

    def read_optical_cap_638(self, timeout_s=1.0):
        """Read this board's authoritative optical DAC ceiling (channel 7)."""
        reply = require_applied(self.manual_command(
            638, MANUAL_GET, CHANNEL_DAC_CAP, timeout_s=timeout_s))
        if not 0 < reply.applied_value <= 65535:
            raise RuntimeError('638 reported an invalid optical DAC cap')
        # The board's cap is authoritative; manual laser SETs are bounded by
        # it from now on instead of the 52400 default.
        caps = getattr(self, '_dac_caps', None)
        if caps is not None:
            caps[638] = reply.applied_value
        return {'max_dac': reply.applied_value}

    def read_controller_timing_638(self, timeout_s=1.0):
        """Read held state and valid-FAST-frame control divider."""
        reply = require_applied(self.manual_command(
            638, MANUAL_GET, CHANNEL_OPTICAL_RUNTIME, timeout_s=timeout_s))
        return self._decode_controller_timing_638(reply.applied_value)

    @staticmethod
    def _decode_controller_timing_638(value):
        divider = (value >> 8) & 255
        if value & ~0xff01 or divider not in (1, 2, 3, 5, 10):
            raise RuntimeError(f'638 returned invalid controller timing word {value}')
        return {'divider': divider, 'held': bool(value & 1), 'raw': value}

    def set_controller_timing_638(self, divider, held, timeout_s=1.0):
        """Set both runtime fields atomically, then verify the applied word."""
        if type(divider) is not int or divider not in (1, 2, 3, 5, 10) or type(held) is not bool:
            raise ValueError('divider must be 1, 2, 3, 5, or 10 and held must be bool')
        if self.system_manual_active:
            self._take_optical_638(timeout_s)
        requested = (divider << 8) | int(held)
        command = self.system_manual_command if self.system_manual_active else self.manual_command
        reply = require_applied(command(638, MANUAL_SET, CHANNEL_OPTICAL_RUNTIME,
                                        requested, timeout_s))
        applied = self._decode_controller_timing_638(reply.applied_value)
        if applied['raw'] != requested:
            raise RuntimeError('638 controller timing readback differs from request')
        return applied

    def read_fp_scan_638(self, timeout_s=1.0):
        reply = require_applied(self.manual_command(
            638, MANUAL_GET, CHANNEL_FP_SCAN, timeout_s=timeout_s))
        raw = reply.applied_value
        state, result, count = raw & 15, (raw >> 4) & 15, (raw >> 8) & 0xffff
        if raw < 0 or state >= len(FP_SCAN_STATES) or result >= len(FP_SCAN_RESULTS) or count > FP_SCAN_MAX_RECORDS:
            raise RuntimeError(f'638 returned invalid FP scan status {raw}')
        return {'state': FP_SCAN_STATES[state], 'state_code': state,
                'result': FP_SCAN_RESULTS[result], 'result_code': result,
                'count': count, 'raw': raw}

    def capture_fp_scan(self, cancel=None, on_progress=None, timeout_s=5.0):
        """Capture the full live ADC sweep in RAM using master packet markers.

        The plotted DAC axis estimates the known1ms ramp staircase; full-rate
        ADC is retained in raw_adc. Old firmware fails explicitly without markers.
        """
        self._require_system_board('The FP scan')
        if not 1.0 <= timeout_s <= 75.0:
            raise ValueError('scan timeout must be between 1 and 75 seconds')
        if not self.system_manual_active:
            raise RuntimeError('Open a system manual session before FP scan')
        if not getattr(self, 'streaming', False):
            raise RuntimeError('An active stream is required for FP scan pages')
        if (1550, CHANNEL_LASER_DAC) not in self._system_manual_owned:
            raise RuntimeError('Take the 1550 manual laser DAC before FP scan illumination')
        illumination = require_applied(self.system_manual_command(
            1550, MANUAL_GET, CHANNEL_LASER_DAC)).applied_value
        if illumination <= 0:
            raise RuntimeError('Set a nonzero 1550 manual laser DAC for FP scan illumination')
        cap = self.read_optical_cap_638()['max_dac']
        if not 0 < cap <= 52400:
            raise RuntimeError('638 does not advertise the approved FP scan DAC cap')
        preflight = self.read_fp_scan_638()
        if preflight['state'] not in ('idle', 'complete', 'aborted'):
            raise RuntimeError(f"638 FP scan is already {preflight['state']}")
        if (self.stream_stats or {}).get('stream_format') != 2:
            raise RuntimeError('Enable optical diagnostics stream format 2 before FP scan')
        if self.read_state_638()['state'] != 0:
            raise RuntimeError('638 optical control must be IDLE before FP scan')

        deadline = time.monotonic() + timeout_s
        rows = []
        started = False
        acquired = (638, CHANNEL_OPTICAL) not in self._system_manual_owned
        cleanup_error = None
        arm_rejections = []
        chunks = []
        cursor = origin = self.samples_received
        initial_stats = dict(self.stream_stats)
        snapshot = self.telemetry
        initial_epoch = snapshot.stream_epoch if snapshot else None
        old_sync = getattr(snapshot, 'scan_sync', None)
        baseline_id = struct.unpack_from('<I', old_sync, 4)[0] if old_sync else None
        capture_id = None
        anchors = None
        last_renew = time.monotonic()
        last_progress = 0.0
        live_index = 0

        def collect():
            nonlocal cursor, capture_id, anchors
            end = self.samples_received
            if end - cursor > self.buffer_capacity:
                raise RuntimeError('ADC ring overrun during FP scan')
            if end > cursor:
                count = end - cursor
                index = cursor % self.buffer_capacity
                chunk = (self.buffer[index:index+count].copy() if index+count <= self.buffer_capacity else
                         np.concatenate((self.buffer[index:], self.buffer[:index+count-self.buffer_capacity])))
                if self.samples_received - cursor > self.buffer_capacity:
                    raise RuntimeError('ADC ring overwritten during FP copy')
                chunks.append(chunk)
                cursor = end
            stats = self.stream_stats
            if any(stats.get(k, 0) != initial_stats.get(k, 0) for k in
                   ('drops_seq', 'transfer_errors', 'transfer_timeouts', 'malformed')):
                raise RuntimeError('ADC discontinuity during FP scan; DAC alignment rejected')
            snap = self.telemetry
            if snap is None or snap.host_stale:
                return
            if snap.stream_epoch != initial_epoch:
                raise RuntimeError('Stream epoch changed during FP scan')
            sync = getattr(snap, 'scan_sync', None)
            if not sync:
                return
            sid = struct.unpack_from('<I', sync, 4)[0]
            if sid == baseline_id:
                return
            if capture_id is not None and sid != capture_id:
                raise RuntimeError('FP capture identity changed')
            capture_id = sid
            if sync[1] & 4:
                raise RuntimeError('Firmware could not align sweep markers to ADC packets')
            points = []
            for offset in (8, 14, 20, 26):
                seq, index = struct.unpack_from('<IH', sync, offset)
                if index >= 8192:
                    raise RuntimeError('Invalid scan marker sample index')
                delta = ((seq - snap.record_sequence + 0x80000000) & 0xffffffff) - 0x80000000
                points.append(snap.received_sample_end - 8192 + delta * 8192 + index)
            anchors = (sync[1], points)

        def scan_arm_receipt(exc):
            reply = exc.reply
            if reply is None:
                return {'target': exc.target, 'channel': exc.channel,
                        'opcode': exc.opcode, 'status': exc.status}
            return {'target': reply.target, 'transaction': reply.transaction,
                    'opcode': reply.opcode, 'channel': reply.channel,
                    'status': reply.status, 'owner': reply.owner,
                    'applied_value': reply.applied_value,
                    'raw_ack_hex': reply.raw_ack.hex()}

        try:
            if acquired:
                self._take_optical_638(1.0)
            # A status-10 ACK is an explicit firmware rejection: the scan was
            # not armed. Retry only that exact response, within the original
            # caller deadline. Capture begins before each dispatch so accepted
            # markers still bound the first sample; rejected attempt samples
            # are discarded and never accumulated during the wait.
            arm_progress_tick = 0.0
            arm_renew_tick = time.monotonic()
            while True:
                if cancel is not None and cancel():
                    raise RuntimeError('FP scan canceled while waiting for 638 readiness')
                if time.monotonic() >= deadline:
                    if arm_rejections:
                        raise TimeoutError(
                            '638 FP scan not ready (temperature or optical transition)')
                    raise TimeoutError('FP scan arm deadline expired')
                chunks.clear()
                cursor = origin = self.samples_received
                anchors = None
                capture_id = None
                live_index = 0
                snapshot = self.telemetry
                if snapshot is not None and not snapshot.host_stale:
                    if snapshot.stream_epoch != initial_epoch:
                        raise RuntimeError('Stream epoch changed before FP scan arm')
                    sync = getattr(snapshot, 'scan_sync', None)
                    baseline_id = struct.unpack_from('<I', sync, 4)[0] if sync else None
                collect()
                remaining = deadline - time.monotonic()
                if remaining < 1.5:
                    raise TimeoutError(
                        '638 FP scan not ready (temperature or optical transition)')
                try:
                    self.system_manual_command(
                        638, MANUAL_SET, CHANNEL_FP_SCAN, 1,
                        timeout_s=min(1.0, remaining))
                    break
                except SlaveCommandRejected as exc:
                    if not (exc.target == 638 and exc.channel == CHANNEL_FP_SCAN and
                            exc.opcode == MANUAL_SET and exc.status == 10):
                        raise
                    arm_rejections.append(scan_arm_receipt(exc))
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            '638 FP scan not ready (temperature or optical transition)') from exc
                    now = time.monotonic()
                    if now - arm_renew_tick >= .3:
                        for key in ((638, CHANNEL_OPTICAL),
                                    (1550, CHANNEL_LASER_DAC)):
                            if key in self._system_manual_owned:
                                remaining = deadline - time.monotonic()
                                if remaining <= 0:
                                    break
                                self.system_manual_command(
                                    key[0], MANUAL_RENEW, key[1],
                                    timeout_s=min(.25, remaining))
                        arm_renew_tick = time.monotonic()
                    if on_progress is not None and now - arm_progress_tick >= .05:
                        telemetry = self.telemetry
                        board = (getattr(telemetry, 'board_638', None)
                                 if telemetry is not None and not telemetry.host_stale else None)
                        temp_c = (None if board is None else
                                  board.temperature_mdegc / 1000.0)
                        on_progress({'state': 'Waiting for 638 readiness',
                                     'detail': 'temperature or optical transition',
                                     'temperature_c': temp_c,
                                     'acquired': 0, 'received': 0, 'expected': 1000,
                                     'estimated': True, 'new_rows': ()})
                        arm_progress_tick = now
                    # No collect() during this wait: the ADC ring is finite.
                    time.sleep(min(.025, max(0.0, deadline-time.monotonic())))
            # Capture begins before dispatch; marker alignment comes from the
            # accepted firmware scan, never from host command timing.
            started = True
            start_host = time.monotonic()
            while time.monotonic() < deadline:
                if cancel is not None and cancel():
                    raise RuntimeError('FP scan canceled')
                collect()
                now = time.monotonic()
                if now-last_renew >= .3:
                    for target, channel in sorted(self._system_manual_owned):
                        if (target, channel) != (638, CHANNEL_OPTICAL):
                            self.system_manual_command(target, MANUAL_RENEW, channel, timeout_s=.5)
                    last_renew = time.monotonic()
                if on_progress is not None and now-last_progress >= .05:
                    live_rows = []
                    if anchors:
                        live_begin = (anchors[1][0]+anchors[1][1]+132)//2
                        ready = min(1000, max(0, (cursor-live_begin)//10000))
                        for i in range(live_index, ready):
                            absolute = live_begin+i*10000
                            if self.samples_received-absolute > self.buffer_capacity:
                                raise RuntimeError('ADC ring overwritten during live plotting')
                            j = absolute % self.buffer_capacity
                            v = (self.buffer[j:j+10000].copy() if j+10000 <= self.buffer_capacity else
                                 np.concatenate((self.buffer[j:], self.buffer[:j+10000-self.buffer_capacity])))
                            live_rows.append({'index': i, 'commanded_dac': cap*i/999,
                                              'main_pd_adc_counts': float(v.mean())})
                        live_index = ready
                    on_progress({'state': 'capturing live ADC', 'acquired': min(1000, int((now-start_host)*1000)),
                                 'received': min(1000, int((now-start_host)*1000)), 'expected': 1000,
                                 'estimated': True, 'new_rows': tuple(live_rows)})
                    last_progress = now
                if anchors and anchors[0] & 2:
                    break
                if now-start_host > 2.5:
                    raise RuntimeError('No complete live ADC scan markers; update master and638 firmware')
                time.sleep(.005)
            if not anchors or not anchors[0] & 2:
                raise TimeoutError('FP scan marker timeout')
            lo, hi, endlo, endhi = anchors[1]
            # Guarded sample is32 samples behind the DMA frontier. The upper
            # bound includes10us for wire transfer/ISR/DAC launch; validate width.
            hi += 132
            endhi += 132
            if not 0 < hi-lo <= 3000 or not 0 < endhi-endlo <= 3000:
                raise RuntimeError('FP marker uncertainty exceeds300us')
            begin = (lo+hi)//2
            finish = (endlo+endhi)//2
            duration_s = (finish-begin)/10_000_000
            if not .98 <= duration_s <= 1.03:
                raise RuntimeError(f'Unexpected FP ramp duration {duration_s:.6f}s')
            # Let the terminal marker drain and638 restore NORMAL before GET.
            while time.monotonic()-start_host < 1.35:
                collect();time.sleep(.005)
            status = self.read_fp_scan_638(timeout_s=.5)
            if status['state'] != 'complete' or status['result'] != 'complete':
                raise RuntimeError(f'FP firmware scan did not complete: {status}')
            collect()
            raw = np.concatenate(chunks)
            if begin < origin or finish > cursor:
                raise RuntimeError('ADC capture does not contain the full sweep')
            raw = raw[begin-origin:finish-origin].copy()
            edges = np.linspace(0,len(raw),1001,dtype=np.int64)
            for i,(left,right) in enumerate(zip(edges[:-1],edges[1:])):
                values = raw[left:right]
                rows.append({'index': i, 'commanded_dac': cap*i/999,
                             'main_pd_adc_counts': float(values.mean()),
                             'adc_min': int(values.min()), 'adc_max': int(values.max()),
                             'time_s': float((left+right)/2/10_000_000),
                             'dac_source': 'estimated_from_firmware_timed_ramp'})
            if on_progress:
                on_progress({'state': 'complete', 'acquired': 1000, 'received': 1000,
                             'expected': 1000, 'estimated': False, 'new_rows': ()})
            return {'status': 'complete', 'rows': tuple(rows), 'raw_adc': raw,
                    'sample_rate_hz': 10_000_000, 'capture_id': capture_id,
                    'duration_s': duration_s, 'dac_cap': cap, 'point_count': len(raw),
                    'paired_count': len(rows), 'illumination_1550_dac': illumination,
                    'arm_rejections': tuple(arm_rejections),
                    'alignment_bound_us': max(hi-lo,endhi-endlo)/20,
                    'quality_flags': ('live_master_adc', 'dac_axis_estimated_1ms_staircase',
                                      'commanded_dac_not_analog_verified'),
                    'marker_sample_bounds': (lo,hi,endlo,endhi)}
        except Exception as exc:
            if started:
                try:
                    self.system_manual_command(638, MANUAL_SET, CHANNEL_FP_SCAN, 0, timeout_s=.5)
                except Exception as cleanup:
                    cleanup_error = cleanup
            raise FPScanError(str(exc), sorted(rows, key=lambda row: row['index']),
                              cleanup_error, arm_rejections) from exc
        finally:
            if acquired and (638, CHANNEL_OPTICAL) in self._system_manual_owned:
                try:
                    self.system_manual_command(638, MANUAL_RELEASE, CHANNEL_OPTICAL, timeout_s=.5)
                except Exception as cleanup:
                    try:
                        self.stop_system_confirmed(timeout_s=1.0)
                    except Exception as shutdown:
                        raise FPScanError(
                            f'optical lease release failed; global STOP unconfirmed: {shutdown}',
                            sorted(rows, key=lambda row: row['index']), cleanup) from cleanup
                    raise FPScanError('optical lease release failed; system stopped',
                                      sorted(rows, key=lambda row: row['index']), cleanup) from cleanup
    @property
    def system_manual_active(self):
        return bool(getattr(self, '_system_manual_active', False))

    @property
    def visible_pd_board(self):
        """True when the master runs the visible-PD build (638 only, no 1550
        or fans), False for the system build, None until a capability
        response has been read (docs/VISIBLE_PD_BOARD.md)."""
        capabilities = getattr(self, '_capabilities', None)
        return None if capabilities is None else capabilities.visible_pd_board

    @property
    def laser_boards(self):
        """The laser boards this master drives: ``(638,)`` or ``(638, 1550)``."""
        return (638,) if self.visible_pd_board else (638, 1550)

    def _require_system_board(self, what, query=True):
        """Refuse *what* on the visible-PD build.

        With the profile not yet known and the master IDLE, asks once. If it
        cannot be learned, *what* goes ahead and the firmware's own refusals
        still apply.
        """
        if (self.visible_pd_board is None and query
                and not getattr(self, '_running', False)
                and hasattr(self, 'telemetry_capabilities')):
            try:
                self.telemetry_capabilities()
            except Exception:
                pass
        if self.visible_pd_board:
            raise RuntimeError(f'{what} needs the system mainboard; this master '
                               f'runs the visible-PD build (638 only)')

    def _session_boards_for(self, boards):
        """The laser boards a manual session powers."""
        visible = self.visible_pd_board
        if boards is None:
            return (638,) if visible else (638, 1550)
        boards = tuple(sorted({int(board) for board in boards}))
        if not boards or not set(boards) <= {638, 1550}:
            raise ValueError('boards must be a non-empty subset of (638, 1550)')
        if visible and 1550 in boards:
            raise RuntimeError('This master runs the visible-PD build; it has no 1550')
        return boards

    def stop_system_confirmed(self, timeout_s=1.0):
        """Request global IDLE and verify firmware state; never infer it from UI.

        ``last_master_fault`` names the master fault that ended the window
        this STOP closed (a safety check in override, say), or is ``None``.
        """
        self._system_manual_active = False
        self._system_manual_owned = set()
        self._session_boards = None
        self._keepalive_ms = 0
        self._auto_1550 = False
        self._ensure_connected()
        self.begin_stream()
        delivered = self.stop_confirmed(timeout_s=timeout_s)
        self._last_manual_rail_off_ns = delivered['bulk_write_completed_monotonic_ns']
        metrics = self.runtime_metrics(timeout_s=timeout_s)
        self.last_master_fault = getattr(metrics, 'fault', None)
        if metrics.current_state != 0:
            raise RuntimeError(f'Global STOP did not reach IDLE: state {metrics.current_state}')
        return metrics

    def _stop_after(self, original, timeout_s):
        """Confirmed STOP after *original*; return the error to raise."""
        try:
            self.stop_system_confirmed(timeout_s=max(.5, timeout_s))
        except Exception as cleanup:
            return RuntimeError(f'{original}; STOP unconfirmed: {cleanup}')
        fault = getattr(self, 'last_master_fault', None)
        if fault:
            return RuntimeError(f'{original} (master fault: {fault})')
        return None

    def _arm_keepalive(self, keepalive_ms, timeout_s):
        """Arm the master's override keepalive; refuse firmware without it."""
        if not keepalive_ms:
            self._keepalive_ms = 0
            return
        if type(keepalive_ms) is not int or not 200 <= keepalive_ms <= 65535:
            raise ValueError('keepalive_ms must be 0 or 200..65535')
        try:
            self._send_confirmed(CMD_KEEPALIVE, wValue=keepalive_ms, timeout_s=timeout_s)
        except TransportUncertainError as exc:
            if getattr(exc, 'halt_cleared', False):
                raise RuntimeError('Master firmware rejected the test-mode keepalive (K); '
                                   'update the master firmware') from exc
            raise
        self._keepalive_ms = keepalive_ms

    def _renew_if_due(self):
        now = time.monotonic()
        if now >= getattr(self, '_next_renew', 0.0):
            self.renew_system_manual()
            self._next_renew = now + RENEW_INTERVAL_S

    def _dwell(self, seconds, cancel=None):
        """Wait *seconds* while renewing the session's leases."""
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if cancel is not None and cancel():
                raise RuntimeError('canceled')
            self._renew_if_due()
            time.sleep(.01)

    def start_confirmed(self, timeout_s=1.0):
        """Recover via confirmed IDLE, then submit BOOT; does not promise a lock."""
        self.stop_system_confirmed(timeout_s)
        self._require_system_board('Starting the box')
        remaining = .2 - (time.monotonic_ns() - self._last_manual_rail_off_ns) / 1e9
        if remaining > 0:
            time.sleep(remaining)
        result = self._send_confirmed(CMD_BOOT, timeout_s=timeout_s)
        self._running = True
        self._last_manual_rail_off_ns = None
        return result

    def begin_system_manual(self, timeout_s=1.0, optical_diagnostics=False,
                            keepalive_ms=KEEPALIVE_MS, boards=None):
        """Power the laser boards with START low and take their DACs at zero.

        *boards* is ``None`` (every board this master drives: both on the
        system build, the 638 alone on the visible-PD build) or an explicit
        tuple; asking a visible-PD master for the 1550 raises before
        override. The last step arms the master's keepalive: if the host
        then sends nothing for *keepalive_ms*, the master powers everything
        down. ``renew_system_manual`` feeds it. A master without the
        keepalive rejects the session.
        """
        if self.system_manual_active:
            return self.system_manual_readback(timeout_s)
        self.stop_system_confirmed(timeout_s)
        if optical_diagnostics:
            self.enable_optical_diagnostics(timeout_s)
        else:
            self.enable_telemetry(timeout_s)
        boards = self._session_boards_for(boards)
        remaining = .2 - (time.monotonic_ns() - self._last_manual_rail_off_ns) / 1e9
        if remaining > 0:
            time.sleep(remaining)
        self._system_manual_owned = set()
        try:
            self._send_confirmed(CMD_OVERRIDE_ENTER, wValue=1, timeout_s=timeout_s)
            # START low on both, whatever the build (both accept it).
            for target in (638, 1550):
                self._send_confirmed(CMD_TRIGGER, wValue=0, wIndex=target, timeout_s=timeout_s)
            power_times = {}
            for target in boards:
                ack = self._send_confirmed(CMD_POWER, wValue=1, wIndex=target, timeout_s=timeout_s)
                power_times[target] = ack['bulk_write_completed_monotonic_ns']
            initial_stats = self.stream_stats or {}
            initial_samples = initial_stats.get('total_samples')
            time.sleep(1.25)
            deadline = time.monotonic() + 3.0
            last_snapshot = None
            while time.monotonic() < deadline:
                snapshot = self.telemetry
                last_snapshot = snapshot
                if snapshot is not None and not snapshot.host_stale:
                    ready = True
                    for target in boards:
                        b = getattr(snapshot, f'board_{target}')
                        if b is None:
                            ready = False
                            continue
                        age = b.temperature_age_ms + snapshot.host_age_s * 1000
                        elapsed = (time.monotonic_ns() - power_times[target]) / 1e6
                        ready &= (getattr(snapshot, f'link_flags_{target}') == 0 and
                                  b.temperature_valid and not b.temperature_stale and
                                  b.temperature_fault == 0 and age < min(500, elapsed) and
                                  snapshot.received_monotonic_ns > power_times[target])
                    if ready:
                        break
                time.sleep(.01)
            else:
                final_stats = self.stream_stats or {}
                now_ns = time.monotonic_ns()
                host_stale = None if last_snapshot is None else last_snapshot.host_stale
                board_info = {}
                for target in boards:
                    board = (None if last_snapshot is None else
                             getattr(last_snapshot, f'board_{target}', None))
                    power_ns = power_times[target]
                    receipt_ns = (None if last_snapshot is None else
                                  last_snapshot.received_monotonic_ns)
                    board_info[target] = {
                        'link_flags': (None if last_snapshot is None else
                                       getattr(last_snapshot, f'link_flags_{target}', None)),
                        'snapshot_tick_ms': (None if board is None else board.snapshot_tick_ms),
                        'temperature_valid': (None if board is None else board.temperature_valid),
                        'temperature_stale': (None if board is None else board.temperature_stale),
                        'temperature_fault': (None if board is None else board.temperature_fault),
                        'temperature_age_ms': (None if board is None else
                            board.temperature_age_ms + last_snapshot.host_age_s * 1000),
                        'power_ack_elapsed_ms': (now_ns - power_ns) / 1e6,
                        'post_power_receipt_delay_ms': (None if receipt_ns is None else
                                                       (receipt_ns - power_ns) / 1e6),
                    }
                samples = final_stats.get('total_samples')
                diagnostics = {
                    'snapshot_missing': last_snapshot is None,
                    'snapshot_stale': host_stale,
                    'snapshot_record_sequence': (None if last_snapshot is None else
                                                 last_snapshot.record_sequence),
                    'boards': board_info,
                    'adc': {
                        'samples_total': samples,
                        'samples_since_power': (None if samples is None or initial_samples is None
                                                else samples - initial_samples),
                        'stream_fatal': final_stats.get('fatal'),
                        'stream_ready': final_stats.get('ready'),
                        'device_lost': final_stats.get('device_lost'),
                        'stream_format': final_stats.get('stream_format'),
                    },
                }
                if last_snapshot is None:
                    reason = 'ADC telemetry unavailable'
                elif host_stale:
                    reason = 'ADC telemetry stale'
                elif final_stats.get('fatal'):
                    reason = 'ADC stream fatal'
                elif final_stats.get('stream_format') not in (1, 2):
                    reason = f"ADC stream format {final_stats.get('stream_format')}"
                else:
                    board_states = []
                    for target in boards:
                        state = board_info[target]
                        board = (None if last_snapshot is None else
                                 getattr(last_snapshot, f'board_{target}', None))
                        if state['link_flags']:
                            status = 'link unavailable'
                        elif board is None:
                            status = 'snapshot missing'
                        elif not state['temperature_valid']:
                            status = 'temperature invalid'
                        elif state['temperature_stale']:
                            status = 'temperature stale'
                        elif state['temperature_fault']:
                            status = f"temperature fault {state['temperature_fault']}"
                        elif (state['temperature_age_ms'] is None or
                              state['temperature_age_ms'] >= min(500, state['power_ack_elapsed_ms'])):
                            status = 'temperature too old'
                        elif (state['post_power_receipt_delay_ms'] is None or
                              state['post_power_receipt_delay_ms'] <= 0):
                            status = 'no post-power snapshot'
                        else:
                            status = 'ready'
                        board_states.append(f'{target} {status}')
                    reason = '; '.join(board_states)
                error = TimeoutError(f'Manual readiness timed out: {reason}')
                error.diagnostics = diagnostics
                raise error
            self._system_manual_active = True
            self._session_boards = boards
            for target in boards:
                try:
                    self.system_manual_command(target, MANUAL_TAKE, CHANNEL_LASER_DAC, 0, timeout_s)
                except Exception as exc:
                    raise RuntimeError(f'Board {target} zero-DAC takeover failed: {exc}') from exc
            self._arm_keepalive(keepalive_ms, timeout_s)
            self._next_renew = time.monotonic() + RENEW_INTERVAL_S
            return self.system_manual_readback(timeout_s)
        except Exception as original:
            try:
                self.stop_system_confirmed(timeout_s)
            except Exception as cleanup:
                error = RuntimeError(f'{original}; global shutdown unconfirmed: {cleanup}')
                if hasattr(original, 'diagnostics'):
                    error.diagnostics = original.diagnostics
                raise error from original
            raise

    def take_over_638(self, dac, timeout_s=1.0, keepalive_ms=KEEPALIVE_MS,
                      idle_timeout_s=2.0):
        """Turn a locked box in RUN into a manual session holding the 638 at *dac*.

        The master enters override without changing any rail or START line,
        so the 1550 keeps running on its own firmware and the master's
        override checks keep watching it and the probe signal. Only the 638
        START drops: the 638 goes to IDLE (its DAC passes through 0) and is
        then taken at *dac*. The 1550 cannot be commanded in this session.
        ``finish_system_manual`` ends it with a confirmed STOP.
        """
        if self.system_manual_active:
            raise RuntimeError('A manual session is already open')
        if not getattr(self, 'streaming', False):
            raise RuntimeError('An active stream is required for take-over')
        self._require_system_board('Take-over of a locked box', query=False)
        cap = self.read_optical_cap_638(timeout_s)['max_dac']
        if type(dac) is not int or not 0 <= dac <= cap:
            raise ValueError(f'638 DAC must be an integer 0..{cap}')
        self._system_manual_owned = set()
        try:
            self._send_confirmed(CMD_OVERRIDE_ENTER, wValue=1, timeout_s=timeout_s)
            self._running = False
            self._system_manual_active = True
            self._session_boards = (638, 1550)
            self._auto_1550 = True
            self._arm_keepalive(keepalive_ms, timeout_s)
            self._send_confirmed(CMD_TRIGGER, wValue=0, wIndex=638, timeout_s=timeout_s)
            self._wait_optical_idle(idle_timeout_s)
            reply = require_applied(self.system_manual_command(
                638, MANUAL_TAKE, CHANNEL_LASER_DAC, dac, timeout_s))
            if reply.applied_value != dac:
                raise RuntimeError(f'638 applied DAC {reply.applied_value}, not {dac}')
            self._next_renew = time.monotonic() + RENEW_INTERVAL_S
            return reply
        except Exception as original:
            error = self._stop_after(original, timeout_s)
            if error is not None:
                raise error from original
            raise

    def start_1550_automatic(self, timeout_s=1.0, lock_timeout_s=60.0,
                             warmup_s=10.0, cancel=None):
        """Hand the 1550 back to its own firmware inside a manual session.

        The 1550 then runs at its stored setpoint with its START high, which
        arms the master's override checks on the 1550 and the probe signal.
        Returns once its TEC has locked and *warmup_s* has passed. The master
        refuses manual commands to the 1550 until the session ends.
        """
        if not self.system_manual_active:
            raise RuntimeError('Open a system manual session first')
        if getattr(self, '_auto_1550', False):
            return
        self._require_system_board('Running the 1550', query=False)
        target_c = require_applied(self.system_manual_command(
            1550, MANUAL_GET, CHANNEL_TEMPERATURE, timeout_s=timeout_s)).applied_value / 1000.0
        for channel in (CHANNEL_LASER_DAC, CHANNEL_TEMPERATURE):
            if (1550, channel) in self._system_manual_owned:
                if channel == CHANNEL_LASER_DAC:
                    self.system_manual_command(1550, MANUAL_SET, channel, 0, timeout_s)
                self.system_manual_command(1550, MANUAL_RELEASE, channel, timeout_s=timeout_s)
        self._send_confirmed(CMD_TRIGGER, wValue=1, wIndex=1550, timeout_s=timeout_s)
        self._auto_1550 = True
        self.wait_tec_lock(1550, target_c=target_c, timeout_s=lock_timeout_s, cancel=cancel)
        self._dwell(warmup_s, cancel)

    def wait_tec_lock(self, target, target_c=None, band_c=0.3, hold_s=1.0,
                      timeout_s=60.0, cancel=None):
        """Wait until *target*'s TEC holds within *band_c* of its target.

        Reads temperature from telemetry and renews the session's leases while
        it waits. Returns the target temperature in °C.
        """
        if target_c is None:
            target_c = require_applied(self.system_manual_command(
                target, MANUAL_GET, CHANNEL_TEMPERATURE)).applied_value / 1000.0
        started = time.monotonic()
        in_band_since = None
        while True:
            if cancel is not None and cancel():
                raise RuntimeError('canceled')
            self._renew_if_due()
            snapshot = self.telemetry
            now = time.monotonic()
            board = (None if snapshot is None or snapshot.host_stale
                     else getattr(snapshot, f'board_{target}'))
            if (board is not None and not getattr(snapshot, f'link_flags_{target}') and
                    board.temperature_valid and not board.temperature_stale and
                    abs(board.temperature_mdegc / 1000.0 - target_c) <= band_c):
                in_band_since = in_band_since or now
                if now - in_band_since >= hold_s:
                    return target_c
            else:
                in_band_since = None
            if now - started > timeout_s:
                raise TimeoutError(f'{target} nm TEC did not lock within ±{band_c} °C '
                                   f'of {target_c:.2f} °C in {timeout_s:.0f} s')
            time.sleep(.02)

    def _stream_health(self):
        stats = self.stream_stats or {}
        return {key: stats.get(key, 0) for key in STREAM_HEALTH_KEYS}

    def read_samples(self, start, end, timeout_s=2.0):
        """Samples [start, end) by absolute stream position (``samples_received``).

        Waits for them to arrive, then copies them out of the ring; raises if
        the ring has already overwritten any of them.
        """
        if not 0 <= start < end or end - start > self.buffer_capacity:
            raise ValueError('need 0 <= start < end and a span the ring can hold')
        return self._ring_window(int(start), int(end), timeout_s)

    def read_after_reply(self, reply, n_samples, settle_s=0.0, timeout_s=2.0):
        """*n_samples* starting *settle_s* after a manual reply's stream position.

        The reply arrives after the board applied the value, so these samples
        all follow the change, whatever the host latency was.
        """
        if getattr(reply, 'stream_sample', None) is None:
            raise RuntimeError('Manual reply has no stream position')
        start = reply.stream_sample + round(settle_s * SAMPLE_RATE)
        return self.read_samples(start, start + int(n_samples), timeout_s + settle_s)

    def _ring_window(self, start, end, timeout_s=2.0):
        """Wait for samples [start, end) and copy them out of the ring."""
        deadline = time.monotonic() + timeout_s
        while self.samples_received < end:
            if time.monotonic() > deadline:
                raise TimeoutError('ADC samples stopped arriving')
            time.sleep(.001)
        capacity = self.buffer_capacity
        if self.samples_received - start > capacity:
            raise RuntimeError('ADC ring overwritten before the window was read')
        index, count = start % capacity, end - start
        buffer = self.buffer
        chunk = (buffer[index:index + count].copy() if index + count <= capacity else
                 np.concatenate((buffer[index:], buffer[:index + count - capacity])))
        if self.samples_received - start > capacity:
            raise RuntimeError('ADC ring overwritten while the window was copied')
        return chunk

    def run_probe_sweep(self, step=100, dwell_s=0.025, settle_s=0.005, top=None,
                        warmup_s=10.0, saturation_threshold=0.95, timeout_s=1.0,
                        cancel=None, on_progress=None):
        """Step the 638 DAC from 0 to its cap and average the photodetector.

        Opens its own manual session and always ends it with a confirmed
        STOP. The 1550 runs automatically at its stored setpoint (so the
        master watches it and the probe signal), and the 638 is stepped by
        confirmed SETs: each must come back with the requested value. Each
        reply's ``stream_sample`` marks the step in the ADC stream, and the
        average covers *settle_s* to *dwell_s* after it. The sweep stops at
        the first point whose mean reaches *saturation_threshold* of full
        scale.

        *on_progress* receives ``{'index', 'count', 'dac', 'photodetector'}``
        after each point; *cancel* is polled between points.
        """
        if type(step) is not int or not 1 <= step <= 10000:
            raise ValueError('step must be an integer 1..10000')
        if not 0 <= settle_s < dwell_s <= 1.0:
            raise ValueError('need 0 <= settle_s < dwell_s <= 1 s')
        if self.system_manual_active:
            raise RuntimeError('Close the open manual session before a probe sweep')
        if not getattr(self, 'streaming', False):
            raise RuntimeError('An active stream is required for a probe sweep')
        self._require_system_board('The probe sweep')
        settle_n = round(settle_s * SAMPLE_RATE)
        dwell_n = round(dwell_s * SAMPLE_RATE)
        if dwell_n > self.buffer_capacity // 2:
            raise ValueError('dwell_s needs a larger ring_seconds')
        limit = saturation_threshold * ADC_MAX_VALUE
        current, means, temps = [], [], []
        saturated = False
        self.begin_system_manual(timeout_s)
        try:
            cap = self.read_optical_cap_638(timeout_s)['max_dac']
            top = cap if top is None else min(int(top), cap)
            values = list(range(0, top + 1, step))
            if values[-1] != top:
                values.append(top)
            target_638 = require_applied(self.system_manual_command(
                638, MANUAL_GET, CHANNEL_TEMPERATURE, timeout_s=timeout_s)).applied_value / 1000.0
            self.start_1550_automatic(timeout_s=timeout_s, warmup_s=warmup_s, cancel=cancel)
            self.wait_tec_lock(638, target_c=target_638, cancel=cancel)
            health = self._stream_health()
            for index, dac in enumerate(values):
                if cancel is not None and cancel():
                    raise RuntimeError('canceled')
                reply = require_applied(self.system_manual_command(
                    638, MANUAL_SET, CHANNEL_LASER_DAC, dac, timeout_s))
                if reply.applied_value != dac:
                    raise RuntimeError(f'638 applied DAC {reply.applied_value}, not {dac}; '
                                       'its TEC may have left lock')
                if reply.stream_sample is None:
                    raise RuntimeError('Manual reply has no stream position')
                window = self._ring_window(reply.stream_sample + settle_n,
                                           reply.stream_sample + dwell_n)
                if self._stream_health() != health:
                    raise RuntimeError('ADC discontinuity during the probe sweep')
                mean = float(window.mean())
                snapshot = self.telemetry
                board = None if snapshot is None else snapshot.board_638
                current.append(dac)
                means.append(mean)
                temps.append(float('nan') if board is None else board.temperature_mdegc / 1000.0)
                if on_progress is not None:
                    on_progress({'index': index, 'count': len(values), 'dac': dac,
                                 'photodetector': mean})
                if mean >= limit:
                    saturated = True
                    break
            self.system_manual_command(638, MANUAL_SET, CHANNEL_LASER_DAC, 0, timeout_s)
        except Exception as original:
            error = self._stop_after(original, timeout_s)
            if error is not None:
                raise error from original
            raise
        self.finish_system_manual(timeout_s)
        return ProbeCharacterizationResult(
            current=current, photodetector=means, saturated=saturated,
            sample_rate=SAMPLE_RATE, bin_seconds=dwell_s, step_size=step,
            max_current=top, settle_seconds=settle_s, temperature_c=temps)

    def system_manual_command(self, target, opcode, channel, value=0, timeout_s=1.0):
        if not self.system_manual_active:
            raise RuntimeError('Open a system manual session first')
        if target not in (638, 1550):
            raise ValueError('Unknown laser board')
        boards = getattr(self, '_session_boards', None)
        if boards is not None and target not in boards:
            raise RuntimeError(f'The {target} board is not part of this session')
        if target == 1550 and getattr(self, '_auto_1550', False):
            raise RuntimeError('The 1550 runs on its own firmware in this session')
        if channel >= CHANNEL_OPTICAL and target != 638:
            raise ValueError('Optical locking controls are available only on 638')
        owned = self._system_manual_owned
        key = (target, channel)
        if opcode in (MANUAL_SET, MANUAL_TAKE) and channel == CHANNEL_TEMPERATURE:
            if not 24000 <= value <= 26000:
                raise ValueError('TEC target must be 24–26 °C')
        if opcode == MANUAL_SET and channel in (CHANNEL_TEMPERATURE, CHANNEL_LASER_DAC) and key not in owned:
            if channel == CHANNEL_LASER_DAC and (638, CHANNEL_OPTICAL) in owned and target == 638:
                self.system_manual_command(638, MANUAL_RELEASE, CHANNEL_OPTICAL, timeout_s=timeout_s)
                self._wait_optical_idle(timeout_s)
            initial = 0 if channel == CHANNEL_LASER_DAC else require_applied(
                self.manual_command(target, MANUAL_GET, channel, timeout_s=timeout_s)).applied_value
            self.system_manual_command(target, MANUAL_TAKE, channel, initial, timeout_s)
        reply = require_applied(self.manual_command(target, opcode, channel, value, timeout_s))
        if opcode == MANUAL_TAKE:
            owned.add(key)
        elif opcode == MANUAL_RELEASE:
            owned.discard(key)
        return reply

    def renew_system_manual(self, timeout_s=.25):
        """Renew every owned lease; that also feeds the master keepalive.

        With no lease to renew, the keepalive is re-armed instead.
        """
        if not self.system_manual_active:
            return []
        try:
            replies = [self.system_manual_command(t, MANUAL_RENEW, ch, timeout_s=timeout_s)
                       for t, ch in sorted(self._system_manual_owned)]
            keepalive_ms = getattr(self, '_keepalive_ms', 0)
            if not replies and keepalive_ms:
                self._send_confirmed(CMD_KEEPALIVE, wValue=keepalive_ms, timeout_s=max(.5, timeout_s))
            return replies
        except Exception as original:
            error = self._stop_after(f'Lease renewal failed: {original}', timeout_s)
            if error is not None:
                raise error from original
            raise RuntimeError(f'Lease renewal failed; system stopped: {original}') from original

    def finish_system_manual(self, timeout_s=1.0):
        # IDLE powers both rails off regardless of stale per-channel ownership.
        return self.stop_system_confirmed(timeout_s)

    def system_manual_readback(self, timeout_s=1.0):
        result = {}
        boards = getattr(self, '_session_boards', None) or (638, 1550)
        for target in boards:
            if target == 1550 and getattr(self, '_auto_1550', False):
                continue
            result[target] = {
                'temperature': self.system_manual_command(target, MANUAL_GET, CHANNEL_TEMPERATURE, timeout_s=timeout_s),
                'laser': self.system_manual_command(target, MANUAL_GET, CHANNEL_LASER_DAC, timeout_s=timeout_s),
            }
        return result

    def _take_optical_638(self, timeout_s):
        if not self.system_manual_active:
            raise RuntimeError('Open a system manual session first')
        if (638, CHANNEL_OPTICAL) not in self._system_manual_owned:
            if (638, CHANNEL_LASER_DAC) in self._system_manual_owned:
                self.system_manual_command(638, MANUAL_SET, CHANNEL_LASER_DAC, 0, timeout_s)
                self.system_manual_command(638, MANUAL_RELEASE, CHANNEL_LASER_DAC, timeout_s=timeout_s)
            self.system_manual_command(638, MANUAL_TAKE, CHANNEL_OPTICAL, 0, timeout_s)

    def _wait_optical_idle(self, timeout_s):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.read_state_638(timeout_s)['state'] == 0:
                return
            time.sleep(.01)
        raise TimeoutError('638 did not finish aborting before DAC takeover')

    def reset_optical_pid_638(self, timeout_s=1.0):
        """Abort optical control and restore compiled defaults for this session."""
        self._take_optical_638(timeout_s)
        self.system_manual_command(638, MANUAL_RELEASE, CHANNEL_OPTICAL, timeout_s=timeout_s)
        self._wait_optical_idle(timeout_s)
        return self.optical_pid_638(timeout_s=timeout_s)

    def control_638(self, action, timeout_s=1.0):
        """Send one acknowledged optical action in automatic or manual mode.

        Automatic RUN routing is owned by the master and therefore does not
        acquire a slave lease.  An active manual session may use an existing
        optical lease, but never silently replaces a manually-owned laser DAC.
        For ``retune``, the reply confirms a queued local slope request only;
        the subsequent ABBA telemetry event reports acceptance or rejection.
        """
        if action not in OPTICAL_ACTIONS:
            raise ValueError('Unknown optical action')
        if self.system_manual_active and action == 'retune':
            raise RuntimeError('Local slope measurement requires automatic locking; '
                               'manual-session lease renewals interrupt its quiet interval')
        if self.system_manual_active:
            if (638, CHANNEL_LASER_DAC) in self._system_manual_owned:
                raise RuntimeError('Release the manually-owned 638 laser DAC before optical control')
            if (638, CHANNEL_OPTICAL) not in self._system_manual_owned:
                self.system_manual_command(638, MANUAL_TAKE, CHANNEL_OPTICAL, 0, timeout_s)
            command = self.system_manual_command
        else:
            if action not in ('reacquire', 'retune', 'trace', 'identify'):
                raise RuntimeError(
                    'Automatic RUN accepts only reacquire, retune, trace, or identify; '
                    'start and abort require a manual optical lease')
            command = self.manual_command
        return require_applied(command(638, MANUAL_SET, CHANNEL_OPTICAL,
                                       OPTICAL_ACTIONS[action], timeout_s))

    def lock_638(self, action='start', timeout_s=1.0):
        """Compatibility wrapper for the optical action API."""
        if action not in ('start', 'abort'):
            raise ValueError('Lock action must be start or abort')
        return self.control_638(action, timeout_s)

    def capture_638_action(self, action, pre_s=.5, post_s=.5, timeout_s=1.0):
        """Capture bounded ring-buffer windows around one automatic lock action.

        This reuses the controller's existing stream and USB owner. It creates
        no reader, background queue, or unbounded recording. The returned
        acknowledgement means the request was queued; the snapshot may still
        contain an older event and does not establish action completion.
        """
        if action not in ('reacquire', 'retune'):
            raise ValueError('Captured automatic action must be reacquire or retune')
        if not 0 < pre_s <= 2 or not 0 < post_s <= 2:
            raise ValueError('pre_s and post_s must each be within 0..2 seconds')
        if not getattr(self, 'streaming', False):
            raise RuntimeError('An active stream is required for action capture')
        before = self.save(pre_s)
        acknowledgement = self.control_638(action, timeout_s)
        time.sleep(post_s)
        after = self.save(post_s)
        snapshot = self.telemetry
        return {
            'action': action, 'acknowledgement': acknowledgement,
            'pre_samples': before, 'post_samples': after,
            'live': getattr(snapshot, 'optical_live_638', None),
            'acquisition': getattr(snapshot, 'optical_acquisition_638', None),
            'abba': getattr(snapshot, 'optical_abba_638', None),
        }

    def read_state_638(self, timeout_s=1.0):
        reply = self.system_manual_command(638, MANUAL_GET, CHANNEL_OPTICAL, timeout_s=timeout_s)
        value = reply.applied_value
        state = value & 255
        return {'state': state, 'name': OPTICAL_STATES[state] if state < len(OPTICAL_STATES) else 'UNKNOWN',
                'kp_override': bool(value & 256), 'ki_negative_override': bool(value & 512),
                'ki_positive_override': bool(value & 1024), 'normalized_enabled': bool(value & 2048),
                'slope_valid': bool(value & 4096), 'owner': reply.owner}

    def optical_pid_638(self, kp=None, ki_negative=None, ki_positive=None, timeout_s=1.0):
        requested = ((CHANNEL_KP, kp), (CHANNEL_KI_NEGATIVE, ki_negative), (CHANNEL_KI_POSITIVE, ki_positive))
        for _, value in requested:
            if value is not None and (not math.isfinite(value) or not 0 <= value <= .2):
                raise ValueError('Optical PI gains must be finite and between 0 and 0.2')
        if any(value is not None for _, value in requested):
            self._take_optical_638(timeout_s)
        for channel, value in requested:
            if value is not None:
                self.system_manual_command(638, MANUAL_SET, channel, round(value * GAIN_SCALE), timeout_s)
        values = [self.system_manual_command(638, MANUAL_GET, ch, timeout_s=timeout_s).applied_value / GAIN_SCALE
                  for ch, _ in requested]
        return dict(zip(('kp', 'ki_negative', 'ki_positive'), values), state=self.read_state_638(timeout_s))
