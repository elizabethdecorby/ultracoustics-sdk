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

CHANNEL_OPTICAL = 3
CHANNEL_KP = 4
CHANNEL_KI_NEGATIVE = 5
CHANNEL_KI_POSITIVE = 6
GAIN_SCALE = 1_000_000
OPTICAL_STATES = ('IDLE', 'CALIBRATING', 'ROOT_FINDING', 'LOCKED', 'ERROR', 'MEASURE_SLOPE')
OPTICAL_ACTIONS = {'abort': 0, 'start': 1, 'reacquire': 2, 'retune': 3,
                   'trace': 4, 'identify': 5}


def require_applied(reply):
    if reply.status != 0:
        reason = {6: 'value outside bounds or TEC not yet LOCKED', 7: 'channel is not owned', 8: 'channel is already owned', 10: 'optical transition busy'}.get(reply.status, 'command rejected')
        raise RuntimeError(f'Board {reply.target}, channel {reply.channel}: {reason} (status {reply.status})')
    return reply


class SystemControlMixin:
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

    def begin_system_manual(self, timeout_s=1.0):
        """Power both boards with START low and take both laser DACs at zero."""
        if self.system_manual_active:
            return self.system_manual_readback(timeout_s)
        self.stop_system_confirmed(timeout_s)
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
