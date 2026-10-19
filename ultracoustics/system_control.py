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
)
from ._internal.optical_diagnostics import OpticalTrace
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


def require_applied(reply):
    if reply.status != 0:
        reason = {6: 'value outside bounds or TEC not yet LOCKED', 7: 'channel is not owned', 8: 'channel is already owned', 10: 'optical transition busy'}.get(reply.status, 'command rejected')
        raise RuntimeError(f'Board {reply.target}, channel {reply.channel}: {reason} (status {reply.status})')
    return reply


class FPScanError(RuntimeError):
    """Scan failure with any causally paired rows already received."""

    def __init__(self, message, rows=(), cleanup_error=None):
        self.rows = tuple(rows)
        self.cleanup_error = cleanup_error
        if cleanup_error is not None:
            message = f'{message}; cleanup also failed: {cleanup_error}'
        super().__init__(message)


class SystemControlMixin:
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
        if raw < 0 or raw & ~0x7ff or (raw & 255) != NORMALIZED_PI_VERSION:
            raise RuntimeError(f'638 returned unsupported normalized PI contract {raw}')
        return {'version': raw & 255, 'active': bool(raw & 256),
                'overridden': bool(raw & 512), 'slope_valid': bool(raw & 1024)}

    def read_normalized_pi_638(self, timeout_s=1.0):
        """Read the explicit symmetric normalized PI pair and active law."""
        info = self._normalized_pi_info_638(timeout_s)
        reply = require_applied(self._normalized_pi_command_638(
            MANUAL_GET, CHANNEL_NORMALIZED_PI, timeout_s=timeout_s))
        packed = reply.applied_value & 0xffffff  # ACK is signed 24-bit.
        kp_milli, ki_deci = packed >> 14, packed & 0x3fff
        if kp_milli > 1000:
            raise RuntimeError(f'638 returned out-of-range normalized PI pair {packed}')
        return dict(info, kp=kp_milli / 1000.0, ki_per_s=ki_deci / 10.0,
                    packed=packed)

    def set_normalized_pi_638(self, kp, ki_per_s, timeout_s=1.0):
        """Commit Kp and Ki/s together; verify the applied pair and law.

        The existing 14-bit Ki field permits 0..1638.3/s. Older firmware
        may reject values above 1000/s; propagate that rejection without retries.
        Runtime overrides do not change the compiled profile defaults.
        """
        if (type(kp) not in (int, float) or type(ki_per_s) not in (int, float) or
                not math.isfinite(kp) or not math.isfinite(ki_per_s) or
                not 0 <= kp <= 1 or not 0 <= ki_per_s <= NORMALIZED_PI_MAX_KI_PER_S):
            raise ValueError('Normalized PI requires finite Kp 0..1 and Ki 0..1638.3 per second')
        info = self._normalized_pi_info_638(timeout_s)
        if not info['active'] or not info['slope_valid']:
            raise RuntimeError('638 normalized PI must be active with a valid slope before tuning')
        packed = (round(kp * 1000) << 14) | round(ki_per_s * 10)
        reply = require_applied(self._normalized_pi_command_638(
            MANUAL_SET, CHANNEL_NORMALIZED_PI, packed, timeout_s))
        if (reply.applied_value & 0xffffff) != packed:
            raise RuntimeError('638 normalized PI applied pair differs from request')
        result = self.read_normalized_pi_638(timeout_s)
        if not result['active'] or not result['overridden'] or result['packed'] != packed:
            raise RuntimeError('638 normalized PI verification failed after SET')
        return result

    def restore_normalized_pi_638(self, timeout_s=1.0):
        """Restore profile defaults and verify normalized control remains active."""
        info = self._normalized_pi_info_638(timeout_s)
        if not info['active'] or not info['slope_valid']:
            raise RuntimeError('638 normalized PI must be active with a valid slope before restore')
        require_applied(self._normalized_pi_command_638(
            MANUAL_SET, CHANNEL_NORMALIZED_PI_INFO, 0, timeout_s))
        result = self.read_normalized_pi_638(timeout_s)
        if not result['active'] or result['overridden']:
            raise RuntimeError('638 normalized PI verification failed after restore')
        return result

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
        # it from now on instead of the historical 44000.
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

        try:
            if acquired:
                self._take_optical_638(1.0)
            # Capture begins before dispatch; alignment comes from hardware markers,
            # never from the host command clock or command acknowledgement.
            collect()
            self.system_manual_command(638, MANUAL_SET, CHANNEL_FP_SCAN, 1)
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
            raise FPScanError(str(exc), sorted(rows, key=lambda row: row['index']), cleanup_error) from exc
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

    def stop_system_confirmed(self, timeout_s=1.0):
        """Request global IDLE and verify firmware state; never infer it from UI.

        ``last_master_fault`` names the master fault that ended the window
        this STOP closed (a safety check in override, say), or is ``None``.
        """
        self._system_manual_active = False
        self._system_manual_owned = set()
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
        remaining = .2 - (time.monotonic_ns() - self._last_manual_rail_off_ns) / 1e9
        if remaining > 0:
            time.sleep(remaining)
        result = self._send_confirmed(CMD_BOOT, timeout_s=timeout_s)
        self._running = True
        self._last_manual_rail_off_ns = None
        return result

    def begin_system_manual(self, timeout_s=1.0, optical_diagnostics=False,
                            keepalive_ms=KEEPALIVE_MS):
        """Power both boards with START low and take both laser DACs at zero.

        The last step arms the master's keepalive: if the host then sends
        nothing for *keepalive_ms*, the master powers everything down.
        ``renew_system_manual`` feeds it. A master without the keepalive
        rejects the session.
        """
        if self.system_manual_active:
            return self.system_manual_readback(timeout_s)
        self.stop_system_confirmed(timeout_s)
        if optical_diagnostics:
            self.enable_optical_diagnostics(timeout_s)
        else:
            self.enable_telemetry(timeout_s)
        remaining = .2 - (time.monotonic_ns() - self._last_manual_rail_off_ns) / 1e9
        if remaining > 0:
            time.sleep(remaining)
        self._system_manual_owned = set()
        try:
            self._send_confirmed(CMD_OVERRIDE_ENTER, wValue=1, timeout_s=timeout_s)
            for target in (638, 1550):
                self._send_confirmed(CMD_TRIGGER, wValue=0, wIndex=target, timeout_s=timeout_s)
            power_times = {}
            for target in (638, 1550):
                ack = self._send_confirmed(CMD_POWER, wValue=1, wIndex=target, timeout_s=timeout_s)
                power_times[target] = ack['bulk_write_completed_monotonic_ns']
            time.sleep(1.25)
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                snapshot = self.telemetry
                if snapshot is not None and not snapshot.host_stale:
                    ready = True
                    for target in (638, 1550):
                        b = getattr(snapshot, f'board_{target}')
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
                raise TimeoutError('Both boards did not produce fresh post-power temperatures')
            self._system_manual_active = True
            for target in (638, 1550):
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
                raise RuntimeError(f'{original}; global shutdown unconfirmed: {cleanup}') from original
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
        cap = self.read_optical_cap_638(timeout_s)['max_dac']
        if type(dac) is not int or not 0 <= dac <= cap:
            raise ValueError(f'638 DAC must be an integer 0..{cap}')
        self._system_manual_owned = set()
        try:
            self._send_confirmed(CMD_OVERRIDE_ENTER, wValue=1, timeout_s=timeout_s)
            self._running = False
            self._system_manual_active = True
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
        for target in ((638,) if getattr(self, '_auto_1550', False) else (638, 1550)):
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
