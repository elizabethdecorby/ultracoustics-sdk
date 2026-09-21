"""Whole-system manual sessions. Call through one serialized command owner.

Methods are bounded synchronous operations; a GUI must run them off its paint
thread and call renew_system_manual periodically while a session is active.
"""
import math
import time
from ._internal.control import (
    MANUAL_GET, MANUAL_TAKE, MANUAL_SET, MANUAL_RELEASE, MANUAL_RENEW,
    CHANNEL_TEMPERATURE, CHANNEL_LASER_DAC,
)
from ._internal.protocol import CMD_BOOT, CMD_OVERRIDE_ENTER, CMD_POWER, CMD_TRIGGER
from ._internal.optical_diagnostics import OpticalTrace

CHANNEL_OPTICAL = 3
CHANNEL_KP = 4
CHANNEL_KI_NEGATIVE = 5
CHANNEL_KI_POSITIVE = 6
CHANNEL_DAC_CAP = 7
CHANNEL_OPTICAL_RUNTIME = 8
CHANNEL_FP_SCAN = 9
FP_SCAN_STATES = ('idle', 'armed', 'qualifying', 'active', 'complete', 'aborted')
FP_SCAN_RESULTS = ('none', 'complete', 'explicit_abort', 'invalid_or_fault')
FP_SCAN_MIN_RECORDS = 500
FP_SCAN_MAX_RECORDS = 1002
GAIN_SCALE = 1_000_000
OPTICAL_STATES = ('IDLE', 'CALIBRATING', 'ROOT_FINDING', 'LOCKED', 'ERROR', 'MEASURE_SLOPE')
OPTICAL_ACTIONS = {'abort': 0, 'start': 1, 'reacquire': 2, 'retune': 3,
                   'trace': 4, 'identify': 5}


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
    def read_optical_cap_638(self, timeout_s=1.0):
        """Read this board's authoritative optical DAC ceiling (channel 7)."""
        reply = require_applied(self.manual_command(
            638, MANUAL_GET, CHANNEL_DAC_CAP, timeout_s=timeout_s))
        if not 0 < reply.applied_value <= 65535:
            raise RuntimeError('638 reported an invalid optical DAC cap')
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

    def capture_fp_scan(self, cancel=None, on_progress=None, timeout_s=20.0):
        """Run a firmware-paced scan and return causally paired DAC/ADC rows.

        Requires an active whole-system manual session with the 1550 DAC
        already commanded above zero. Each trace record's feedback precedes
        its own DAC write, so row *i* pairs feedback[i] with commanded DAC[i-1].
        The first trace record has no prior scan command and is excluded.
        This routine never infers analog DAC delivery from the SPI command.
        """
        if not 1.0 <= timeout_s <= 20.0:
            raise ValueError('scan timeout must be between 1 and 20 seconds')
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
        if not 0 < cap <= 50000:
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
        last_renew = time.monotonic()
        last_status = 0.0
        baseline_page = getattr(getattr(self.telemetry, 'optical_trace_638', None), 'page', None)
        baseline_id = baseline_page.capture_id if isinstance(baseline_page, OpticalTrace) else None
        pages = {}
        next_pair = 1
        capture_id = None
        total = None
        clock_hz = None
        start_tick_ms = None
        def ingest(snapshot):
            nonlocal capture_id, total, clock_hz, start_tick_ms, next_pair
            new_rows = []
            if snapshot is None or snapshot.host_stale or snapshot.link_638_stale:
                return new_rows
            cached = snapshot.optical_trace_638
            if (cached is None or cached.host_stale or
                    cached.received_monotonic_ns <= scan_start_ns):
                return new_rows
            page = cached.page
            if (not isinstance(page, OpticalTrace) or not page.flags & 8 or
                    not page.flags & 1 or page.total_samples > FP_SCAN_MAX_RECORDS or
                    (baseline_id is not None and page.capture_id == baseline_id)):
                return new_rows
            if capture_id is None:
                capture_id, total = page.capture_id, page.total_samples
                clock_hz, start_tick_ms = page.clock_hz, page.start_tick_ms
            if (page.capture_id != capture_id or page.total_samples != total or
                    page.clock_hz != clock_hz or page.start_tick_ms != start_tick_ms):
                return new_rows
            for offset, sample in enumerate(page.samples):
                index = page.start_index + offset
                if index in pages and pages[index] != sample:
                    raise RuntimeError(f'conflicting FP trace record {index}')
                pages[index] = sample
            while next_pair in pages and next_pair - 1 in pages:
                previous, current = pages[next_pair - 1], pages[next_pair]
                row = {'index': next_pair, 'commanded_dac': previous.actual_dac,
                       'main_pd_adc_counts': current.feedback,
                       'sample_cycles': current.cycles,
                       'preceding_command_cycles': previous.cycles,
                       'command_age_s': ((current.cycles - previous.cycles) & 0xffffffff) / clock_hz,
                       'dac_source': 'SPI5_command_not_analog_readback'}
                rows.append(row)
                new_rows.append(row)
                next_pair += 1
            return new_rows
        try:
            if acquired:
                # The standard system-manual setup owns the 638 DAC at zero.
                # Hand it to optical control while preserving 1550 illumination.
                self._take_optical_638(1.0)
            scan_start_ns = time.monotonic_ns()
            self.system_manual_command(638, MANUAL_SET, CHANNEL_FP_SCAN, 1)
            started = True
            # The 638 firmware refreshes its optical lease internally during
            # FAST scan. A manual GET/RENEW would pause FAST and abort it.
            scan_start = time.monotonic()
            last_active_progress = scan_start
            while time.monotonic() - scan_start < 1.35:
                if cancel is not None and cancel():
                    raise RuntimeError('FP scan canceled')
                if time.monotonic() >= deadline:
                    raise TimeoutError('FP scan exceeded deadline')
                if time.monotonic() - last_renew >= .4:
                    for target, channel in sorted(self._system_manual_owned):
                        if (target, channel) != (638, CHANNEL_OPTICAL):
                            self.system_manual_command(target, MANUAL_RENEW, channel, timeout_s=.5)
                    last_renew = time.monotonic()
                new_rows = ingest(self.telemetry)
                if on_progress is not None and (new_rows or time.monotonic() - last_active_progress >= .05):
                    on_progress({'state': 'active', 'acquired': min(1000,
                                 int((time.monotonic() - scan_start) * 1000)),
                                 'received': len(pages), 'expected': 1000,
                                 'estimated': True, 'new_rows': tuple(new_rows)})
                    last_active_progress = time.monotonic()
                time.sleep(.003)
            status = None
            last_progress = None
            while time.monotonic() < deadline:
                if cancel is not None and cancel():
                    raise RuntimeError('FP scan canceled')
                now = time.monotonic()
                if now - last_renew >= .4:
                    self.system_manual_command(638, MANUAL_RENEW, CHANNEL_OPTICAL, timeout_s=.5)
                    for target, channel in sorted(self._system_manual_owned):
                        if (target, channel) != (638, CHANNEL_OPTICAL):
                            self.system_manual_command(target, MANUAL_RENEW, channel, timeout_s=.5)
                    last_renew = time.monotonic()
                new_rows = ingest(self.telemetry)
                if status is None or now - last_status >= .5:
                    status = self.read_fp_scan_638(timeout_s=.5)
                    last_status = time.monotonic()
                    if status['state'] == 'aborted':
                        raise RuntimeError(f"638 aborted FP scan: {status['result']}")
                    if status['state'] == 'complete' and not FP_SCAN_MIN_RECORDS <= status['count'] <= FP_SCAN_MAX_RECORDS:
                        raise RuntimeError(f"638 completed unexpected {status['count']} FP scan records")
                progress = (status['count'] if status else 0, len(pages))
                if on_progress is not None and progress != last_progress:
                    on_progress({'state': status['state'] if status else 'unknown',
                                 'acquired': progress[0], 'received': progress[1],
                                 'expected': status['count'] if status['state'] == 'complete' else 1000,
                                 'estimated': False,
                                 'new_rows': tuple(new_rows)})
                    last_progress = progress
                if (total is not None and FP_SCAN_MIN_RECORDS <= total <= FP_SCAN_MAX_RECORDS and
                        len(pages) == total and status['state'] == 'complete' and status['count'] == total):
                    if len(rows) != total - 1:
                        raise RuntimeError('FP trace records are not contiguous')
                    commands = [pages[index].actual_dac for index in range(total)]
                    if (commands[0] != 0 or commands[-2] != cap or commands[-1] != 0 or
                            any(right < left or right > cap for left, right in
                                zip(commands[:-2], commands[1:-1]))):
                        raise RuntimeError('FP trace DAC sequence violates 0-to-cap-to-zero scan')
                    duration_s = ((pages[total - 1].cycles - pages[0].cycles) & 0xffffffff) / clock_hz
                    if not .85 <= duration_s <= 1.25:
                        raise RuntimeError(f'FP trace duration {duration_s:.3f}s is not the expected scan')
                    quality = ['commanded_dac_not_analog_verified', 'first_feedback_unpaired']
                    if total < 900:
                        quality.append('sparse_scan_records')
                    if any(row['command_age_s'] > .0025 for row in rows):
                        quality.append('sample_gap_over_2p5ms')
                    return {'status': 'complete',
                            'quality_flags': tuple(quality),
                            'rows': tuple(rows),
                            'capture_id': capture_id, 'clock_hz': clock_hz,
                            'start_tick_ms': start_tick_ms, 'point_count': total,
                            'paired_count': len(rows), 'duration_s': duration_s,
                            'dac_cap': cap,
                            'illumination_1550_dac': illumination,
                            'first_unpaired_feedback': pages[0].feedback}
                time.sleep(.003)
            raise TimeoutError(f'FP scan timed out with {len(pages)} trace records')
        except Exception as exc:
            if started:
                try:
                    self.system_manual_command(638, MANUAL_SET, CHANNEL_FP_SCAN, 0, timeout_s=.5)
                except Exception as cleanup:
                    cleanup_error = cleanup
            raise FPScanError(str(exc), rows, cleanup_error) from exc
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
                            rows, cleanup) from cleanup
                    raise FPScanError('optical lease release failed; system stopped',
                                      rows, cleanup) from cleanup
    @property
    def system_manual_active(self):
        return bool(getattr(self, '_system_manual_active', False))

    def stop_system_confirmed(self, timeout_s=1.0):
        """Request global IDLE and verify firmware state; never infer it from UI."""
        self._system_manual_active = False
        self._system_manual_owned = set()
        self._ensure_connected()
        self.begin_stream()
        delivered = self.stop_confirmed(timeout_s=timeout_s)
        self._last_manual_rail_off_ns = delivered['bulk_write_completed_monotonic_ns']
        metrics = self.runtime_metrics(timeout_s=timeout_s)
        if metrics.current_state != 0:
            raise RuntimeError(f'Global STOP did not reach IDLE: state {metrics.current_state}')
        return metrics

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

    def begin_system_manual(self, timeout_s=1.0, optical_diagnostics=False):
        """Power both boards with START low and take both laser DACs at zero."""
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
            return self.system_manual_readback(timeout_s)
        except Exception as original:
            try:
                self.stop_system_confirmed(timeout_s)
            except Exception as cleanup:
                raise RuntimeError(f'{original}; global shutdown unconfirmed: {cleanup}') from original
            raise

    def system_manual_command(self, target, opcode, channel, value=0, timeout_s=1.0):
        if not self.system_manual_active:
            raise RuntimeError('Open a system manual session first')
        if target not in (638, 1550):
            raise ValueError('Unknown laser board')
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
        if not self.system_manual_active:
            return []
        try:
            return [self.system_manual_command(t, MANUAL_RENEW, ch, timeout_s=timeout_s)
                    for t, ch in sorted(self._system_manual_owned)]
        except Exception as original:
            try:
                self.stop_system_confirmed(timeout_s=max(.5, timeout_s))
            except Exception as cleanup:
                raise RuntimeError(f'Lease renewal failed: {original}; STOP unconfirmed: {cleanup}') from original
            raise RuntimeError(f'Lease renewal failed; system stopped: {original}') from original

    def finish_system_manual(self, timeout_s=1.0):
        # IDLE powers both rails off regardless of stale per-channel ownership.
        return self.stop_system_confirmed(timeout_s)

    def system_manual_readback(self, timeout_s=1.0):
        result = {}
        for target in (638, 1550):
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
        """
        if action not in OPTICAL_ACTIONS:
            raise ValueError('Unknown optical action')
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
        no reader, background queue, or unbounded recording.
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
