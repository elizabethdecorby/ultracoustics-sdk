"""The manual session the box checks run: every board up with its lasers held at zero,
one laser at a time under host control, raw reads from the ring, and the probe sweep.

Private: the self-test, the signal level and a firmware update use it through a
Controller's :class:`~ultracoustics._internal.device.Box`. Methods are bounded
synchronous operations: a GUI runs
them off its paint thread and calls renew_system_manual periodically while a session
is active.
"""
import time

import numpy as np

from ..config import ADC_MAX_VALUE, RING_GUARD_SAMPLES, SAMPLE_RATE
from .box import BoxCommands, CHANNEL_OPTICAL, KEEPALIVE_MS, MAX_DWELL_S, RENEW_INTERVAL_S, require_applied
from .checks.probe import ProbeCharacterizationResult
from .comms import TransportUncertainError
from .control import (CHANNEL_LASER_DAC, CHANNEL_TEMPERATURE, MANUAL_GET, MANUAL_RELEASE, MANUAL_RENEW,
                      MANUAL_SET, MANUAL_TAKE)
from .protocol import CMD_KEEPALIVE, CMD_OVERRIDE_ENTER, CMD_POWER, CMD_PROGRAM_1550, CMD_STREAM_FORMAT, CMD_TRIGGER


class SessionCommands(BoxCommands):
    """The manual session and raw reads, on top of the box's commands."""

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

    @property
    def laser_boards(self):
        """The laser boards this master drives: ``(638,)`` or ``(638, 1550)``."""
        return (638,) if self.visible_pd_board else (638, 1550)

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

    def program_1550_setpoint(self, dac, timeout_s=1.0):
        """Ask the master to store *dac* on the 1550 board. The box must be stopped.

        This confirms delivery only. The master reports a refused or timed-out
        write as the fault the next confirmed STOP reads.
        """
        return self._send_confirmed(CMD_PROGRAM_1550, wValue=int(dac), timeout_s=timeout_s)

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
        limit = capacity - RING_GUARD_SAMPLES
        if self.samples_received - start > limit:
            raise RuntimeError('ADC ring overwritten before the window was read')
        index, count = start % capacity, end - start
        buffer = self.buffer
        chunk = (buffer[index:index + count].copy() if index + count <= capacity else
                 np.concatenate((buffer[index:], buffer[:index + count - capacity])))
        if self.samples_received - start > limit:
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
        if not 0 <= settle_s < dwell_s <= MAX_DWELL_S:
            raise ValueError(f'need 0 <= settle_s < dwell_s <= {MAX_DWELL_S:g} s')
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

    def enable_telemetry(self, timeout_s: float = 1.0):
        """Explicitly negotiate stream format 1 and return its first snapshot.

        Legacy framing remains the default. A successful OUT transfer alone
        is insufficient: this method requires a validated capability response,
        a matching format acknowledgement, and a valid format-1 data record.
        It never interprets command delivery as a board setpoint or actuator
        acknowledgement.
        """
        capabilities = self.telemetry_capabilities(timeout_s=timeout_s)
        if not capabilities.supports_format1:
            raise RuntimeError("connected master does not advertise stream format 1")
        from .._internal.protocol import pack_command
        result = self._stream.select_stream_format_confirmed(
            pack_command(CMD_STREAM_FORMAT, 1, 0), 1, timeout_s=timeout_s,
        )
        expected_epoch = result["firmware_response"]["stream_epoch"]
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            snapshot = self.telemetry
            if snapshot is not None and snapshot.stream_epoch == expected_epoch:
                return snapshot
            time.sleep(0.005)
        raise RuntimeError("format 1 was accepted but no matching valid telemetry record arrived")

    def collect_fresh(self, n_samples: int, timeout_s: float = 5.0) -> np.ndarray:
        """Collect *n_samples* generated AFTER this call, discarding stale data.

        Records the cumulative sample counter, waits until at least
        *n_samples* new samples have arrived in the ring, then snapshots
        that fresh window. Use this (rather than :meth:`save`) when you
        need data produced *after* a setpoint change — e.g. after stepping
        a laser DAC, so the block does not include samples from the
        previous setpoint.

        Parameters
        ----------
        n_samples : int
            Number of fresh samples to collect. Must be ≤ the ring buffer
            capacity (see ``ring_seconds`` at construction).
        timeout_s : float
            Maximum wait for the samples to arrive.

        Returns
        -------
        numpy.ndarray
            uint16 array of exactly *n_samples* samples.

        Raises
        ------
        RuntimeError
            If not streaming, or *n_samples* exceeds the ring capacity.
        TimeoutError
            If fewer than *n_samples* arrive within *timeout_s*.
        """
        self._ensure_connected()
        if not self._streaming:
            raise RuntimeError(
                "Not streaming — call begin_stream() before collecting samples."
            )
        if n_samples > self._buf_len - RING_GUARD_SAMPLES:
            raise RuntimeError(
                f"Requested {n_samples} fresh samples but ring holds only "
                f"{self._buf_len}. Reconstruct with a larger ring_seconds."
            )

        t0 = self.samples_received
        deadline = time.time() + timeout_s
        while self.samples_received - t0 < n_samples:
            if time.time() > deadline:
                got = self.samples_received - t0
                raise TimeoutError(
                    f"Only got {got}/{n_samples} fresh samples before timeout."
                )
            time.sleep(0.002)

        return self._ring_window(t0, t0 + n_samples)
