"""The commands the customer calls stand on, on the master board and the 638.

Not an API. A :class:`~ultracoustics._internal.device.Box` is built from these, and a
Controller keeps one private.
"""
from dataclasses import replace
import math
import time
from .comms import TransportUncertainError
from .control import MANUAL_GET, MANUAL_TAKE, MANUAL_SET, CHANNEL_LASER_DAC, pack_manual_request
from .protocol import (CMD_BOOT, CMD_IDLE, CMD_MANUAL_CONTROL, CMD_RUNTIME_METRICS,
                       CMD_STREAM_CAPABILITIES, CMD_STREAM_FORMAT, TARGET_1550, TARGET_638,
                       CMD_IDENTITY_READ, CMD_IDENTITY_WRITE, pack_command)
from . import identity as _identity
from . import tec_tuning as _tuning
from ..config import STREAM_HEALTH_KEYS

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
#: A probe sweep's longest dwell. Nothing renews the leases during a dwell,
#: and they last 1 s.
MAX_DWELL_S = 0.5
#: Stream counters that must not move while a sweep is binning samples.


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


class StreamCommands:
    """Commands through the reader process, which alone holds the USB device."""

    def retained_control_trace(self):
        """Read the latest bounded indexed trace without polling away older pages.

        Historical capture, not a claim that the board is still locked. Check
        epoch/capture identity, complete and conflict before using for tuning.
        """
        if self._stream is None or self._stream._telemetry_shm is None:
            return None
        from .telemetry import read_retained_trace
        return read_retained_trace(self._stream._telemetry_shm)

    def telemetry_capabilities(self, timeout_s: float = 1.0):
        """Query the IDLE master's negotiated stream capabilities.

        This does not select a new format. The laser system must be stopped,
        and the stream reader must already be running so it can receive the
        dedicated capability response on Bulk IN.
        """
        if self._running:
            raise RuntimeError("stream negotiation is IDLE-only; call stop() first")
        if self._stream is None or not self._stream.running:
            raise RuntimeError("call begin_stream() before querying capabilities")
        from .protocol import pack_command
        capabilities = self._stream.query_stream_capabilities(
            pack_command(CMD_STREAM_CAPABILITIES, 0, 0), timeout_s=timeout_s,
        )
        # Kept: which build the master runs (visible_pd_board) decides what
        # a manual session powers and which tests may run at all.
        self._capabilities = capabilities
        return capabilities

    def enable_optical_diagnostics(self, timeout_s: float = 1.0):
        """Negotiate format 2 and return its first coherent cached snapshot."""
        if self._running:
            if (self.stream_stats or {}).get('stream_format') == 2:
                return self.telemetry
            raise RuntimeError('optical diagnostics must be negotiated while IDLE')
        capabilities = self.telemetry_capabilities(timeout_s=timeout_s)
        if not capabilities.supports_format2:
            raise RuntimeError("connected master does not advertise optical diagnostics")
        from .protocol import pack_command
        result = self._stream.select_stream_format_confirmed(
            pack_command(CMD_STREAM_FORMAT, 2, 0), 2, timeout_s=timeout_s)
        expected_epoch = result["firmware_response"]["stream_epoch"]
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            snapshot = self.telemetry
            if snapshot is not None and snapshot.stream_epoch == expected_epoch:
                return snapshot
            time.sleep(.005)
        raise RuntimeError("format 2 was accepted but no coherent telemetry snapshot arrived")

    def runtime_metrics(self, timeout_s: float = 1.0):
        """Return the retained runtime window after an explicit STOP/IDLE."""
        if self._running:
            raise RuntimeError("runtime metrics require an explicit stop() first")
        if self._stream is None or not self._stream.running:
            raise RuntimeError("call begin_stream() before requesting runtime metrics")
        from .protocol import pack_command
        result = self._stream._request_stream_control(
            pack_command(CMD_RUNTIME_METRICS), "runtime", timeout_s=timeout_s)
        # Firmware sends the response before its runtime-maintenance state
        # finishes and restarts ADC.  A packet received after this baseline is
        # the observable completion barrier; returning earlier lets a following
        # override replace that state and strand acquisition stopped.
        stats = self.stream_stats or {}
        baseline_packets = int(stats.get("packets", 0))
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            stats = self.stream_stats or {}
            if int(stats.get("packets", 0)) > baseline_packets:
                return result["firmware_response"]
            if stats.get("device_lost") or stats.get("fatal"):
                raise RuntimeError("stream ended before runtime maintenance completed")
            time.sleep(0.001)
        raise TimeoutError("no ADC packet received after runtime metrics response")

    def dac_cap(self, target: int) -> int:
        """Highest laser DAC this controller will send to *target*.

        For the 638 this is the board's own ``CALIBRATION_DAC_VALUE`` once
        :meth:`read_optical_cap_638` has read it, else 52400 (the highest box
        cap; each 638 still enforces its own).
        The 1550 has no cap channel and stays at its hardware limit, 43253.
        """
        if target not in self._dac_caps:
            raise ValueError("target must be 638 or 1550")
        return self._dac_caps[target]

    def manual_command(self, target: int, opcode: int, channel: int,
                       value: int = 0, timeout_s: float = 1.0):
        """Execute one manual slave transaction and validate its full identity."""
        if target not in (TARGET_638, TARGET_1550):
            raise ValueError("target must be 638 or 1550")
        if opcode in (MANUAL_TAKE, MANUAL_SET) and channel == CHANNEL_LASER_DAC:
            cap = self.dac_cap(target)
            if not 0 <= value <= cap:
                raise ValueError(f"laser DAC must be between 0 and {cap}")
        if self._stream is None or not self._stream.running:
            raise RuntimeError("call begin_stream() before manual control")
        self._manual_transaction = (self._manual_transaction + 1) & 0xFF
        txn = self._manual_transaction
        request = pack_manual_request(opcode, channel, value, txn)
        from .protocol import pack_command
        result = self._stream._request_stream_control(
            pack_command(CMD_MANUAL_CONTROL, 0, target) + request,
            "manual", timeout_s=timeout_s)
        reply = replace(result["firmware_response"],
                        stream_sample=result.get("stream_sample"),
                        round_trip_ns=result.get("host_round_trip_bound_ns"))
        if (reply.target, reply.transaction, reply.opcode, reply.channel) != (
                target, txn, opcode, channel):
            raise RuntimeError("manual response does not match the request identity")
        reply.require_transport()
        return reply

    def begin_stream(self):
        """Start reading USB bulk data into the circular buffer without turning the laser on.

        Call this after :meth:`connect` to begin collecting data immediately.
        The laser measurement system remains off until :meth:`start` is called.
        """
        self._ensure_connected()
        if not self._streaming:
            self._stream.start()
            self._streaming = True

    def end_stream(self):
        """Stop reading USB bulk data from the device.

        This does *not* send an IDLE command — use :meth:`stop` first if the
        laser is currently active.
        """
        self._streaming = False
        if self._stream:
            self._stream.stop()

    def try_optical_diagnostics(self, timeout_s: float = 1.0) -> bool:
        """Select optical diagnostics if this master has them. Old masters may
        advertise format 1 only; any other failure is real and raises."""
        try:
            self.enable_optical_diagnostics(timeout_s=timeout_s)
        except RuntimeError as exc:
            if "does not advertise optical diagnostics" not in str(exc):
                raise
            return False
        return True

    def stop_confirmed(self, timeout_s: float = 1.0) -> dict:
        """Request IDLE and require bounded confirmation of USB delivery.

        A successful result proves only that the complete command packet was
        delivered by the host USB transport.  It is not an acknowledgement
        that firmware received or applied IDLE, and it does not measure the
        actuator or laser state.

        ``running`` is cleared only after confirmed transport delivery.  A
        failed or uncertain delivery raises and preserves the prior state.
        The full transport result is returned for diagnostic provenance.
        """
        result = self._send_confirmed(CMD_IDLE, timeout_s=timeout_s)
        if result.get("transport_status") != "delivered":
            raise RuntimeError("IDLE command transport was not confirmed delivered")
        self._running = False
        return result

    def _ensure_connected(self):
        """Raise if the USB connection is not established."""
        if not self.connected:
            raise RuntimeError("Not connected. Call connect() first.")

    def _send(self, cmd_byte, wValue=0, wIndex=0, extra=b"", timeout_ms=5000):
        """Send a command packet over USB bulk to the Master Board.

        Routes through the streaming subprocess's command queue, since that
        process owns the libusb device handle exclusively.  ``timeout_ms`` is
        accepted for API compatibility; the subprocess applies its own
        bulkWrite timeout (2 s) which is sufficient for command packets.
        """
        from .protocol import pack_command
        if self._stream is None or not self._stream.running:
            raise RuntimeError(
                "Not streaming — call begin_stream() before sending commands."
            )
        pkt = pack_command(cmd_byte, wValue, wIndex) + extra
        self._stream.send_command(pkt)

    def _send_confirmed(self, cmd_byte, wValue=0, wIndex=0, extra=b"",
                        timeout_s=0.5):
        """Internal bounded OUT delivery primitive for diagnostic epochs."""
        from .protocol import pack_command
        if self._stream is None or not self._stream.running:
            raise RuntimeError("USB stream must be running for confirmed command")
        pkt = pack_command(cmd_byte, wValue, wIndex) + extra
        return self._stream.send_command_confirmed(pkt, timeout_s=timeout_s)


class BoxCommands:
    """Lock, state and tuning-record commands. Bounded and synchronous: a GUI runs them off
    its paint thread."""

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
        names = {1: '638-profile-1', 2: '638-profile-2'}
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

    @property
    def system_manual_active(self):
        return bool(getattr(self, '_system_manual_active', False))

    @property
    def visible_pd_board(self):
        """True when the master runs the visible-PD build (638 only, no 1550
        or fans), False for the system build, None until a capability
        response has been read."""
        capabilities = getattr(self, '_capabilities', None)
        return None if capabilities is None else capabilities.visible_pd_board

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

    def stop_system_confirmed(self, timeout_s=1.0):
        """Request global IDLE and verify firmware state; never infer it from UI.

        ``last_master_fault`` names the master fault that ended the window
        this STOP closed (a safety check in override, say), or is ``None``.
        """
        self._ensure_connected()
        self.begin_stream()
        delivered = self.stop_confirmed(timeout_s=timeout_s)
        # Only now: until the STOP is delivered the session still needs closing.
        self._system_manual_active = False
        self._system_manual_owned = set()
        self._session_boards = None
        self._keepalive_ms = 0
        self._auto_1550 = False
        self._last_manual_rail_off_ns = delivered['bulk_write_completed_monotonic_ns']
        metrics = self.runtime_metrics(timeout_s=timeout_s)
        self.last_master_fault = getattr(metrics, 'fault', None)
        if metrics.current_state != 0:
            raise RuntimeError(f'Global STOP did not reach IDLE: state {metrics.current_state}')
        return metrics

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

    def _stream_health(self):
        stats = self.stream_stats or {}
        return {key: stats.get(key, 0) for key in STREAM_HEALTH_KEYS}

    def _wait_optical_idle(self, timeout_s):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.read_state_638(timeout_s)['state'] == 0:
                return
            time.sleep(.01)
        raise TimeoutError('638 did not finish aborting before DAC takeover')

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

    def read_state_638(self, timeout_s=1.0):
        reply = self.system_manual_command(638, MANUAL_GET, CHANNEL_OPTICAL, timeout_s=timeout_s)
        value = reply.applied_value
        state = value & 255
        return {'state': state, 'name': OPTICAL_STATES[state] if state < len(OPTICAL_STATES) else 'UNKNOWN',
                'kp_override': bool(value & 256), 'ki_negative_override': bool(value & 512),
                'ki_positive_override': bool(value & 1024), 'normalized_enabled': bool(value & 2048),
                'slope_valid': bool(value & 4096), 'owner': reply.owner}
