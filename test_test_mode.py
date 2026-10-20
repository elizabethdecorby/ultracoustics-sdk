"""Single closed-box test mode: keepalive, take-over, fault names, probe sweep."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ultracoustics import Controller
from ultracoustics._internal.comms import TransportUncertainError
from ultracoustics._internal.control import (
    CHANNEL_LASER_DAC, CHANNEL_TEMPERATURE, MANUAL_GET, MANUAL_RELEASE,
    MANUAL_RENEW, MANUAL_SET, MANUAL_TAKE, RuntimeMetrics,
    parse_manual_response,
)
from ultracoustics._internal.protocol import (
    CMD_KEEPALIVE, CMD_OVERRIDE_ENTER, CMD_TRIGGER,
)
from ultracoustics.system_control import SystemControlMixin

from test_control_api import manual_response


def reply(target, channel, value, status=0, stream_sample=None):
    return SimpleNamespace(target=target, channel=channel, status=status,
                           owner=1, applied_value=value,
                           stream_sample=stream_sample)


class Fake(SystemControlMixin):
    """Session-level fake: records sends and manual commands."""

    def __init__(self):
        self._system_manual_active = False
        self._system_manual_owned = set()
        self._running = True
        self.streaming = True
        self.calls = []
        self.values = {(638, 7): 52400, (638, 3): 0,
                       (638, 1): 25000, (1550, 1): 25000}
        self.reject_keepalive = False
        self.metrics = SimpleNamespace(current_state=0, fault=None)

    def _ensure_connected(self):
        pass

    def begin_stream(self):
        pass

    def stop_confirmed(self, timeout_s):
        self.calls.append(('stop',))
        self._running = False
        return {'bulk_write_completed_monotonic_ns': 0}

    def runtime_metrics(self, timeout_s):
        return self.metrics

    def _send_confirmed(self, cmd, wValue=0, wIndex=0, timeout_s=.5):
        if cmd == CMD_KEEPALIVE and self.reject_keepalive:
            raise TransportUncertainError('pipe', halt_cleared=True)
        self.calls.append(('send', cmd, wValue, wIndex))
        return {'transport_status': 'delivered'}

    def manual_command(self, target, opcode, channel, value=0, timeout_s=1):
        self.calls.append((target, opcode, channel, value))
        if opcode in (MANUAL_SET, MANUAL_TAKE):
            self.values[target, channel] = value
        return reply(target, channel, self.values.get((target, channel), 0))


class KeepaliveTests(unittest.TestCase):
    def test_renew_feeds_keepalive_when_nothing_is_owned(self):
        f = Fake()
        f._system_manual_active = True
        f._keepalive_ms = 2000
        self.assertEqual(f.renew_system_manual(), [])
        self.assertEqual(f.calls, [('send', CMD_KEEPALIVE, 2000, 0)])

    def test_renew_with_leases_does_not_resend_keepalive(self):
        f = Fake()
        f._system_manual_active = True
        f._system_manual_owned = {(638, CHANNEL_LASER_DAC)}
        f._keepalive_ms = 2000
        f.renew_system_manual()
        self.assertEqual(f.calls, [(638, MANUAL_RENEW, CHANNEL_LASER_DAC, 0)])

    def test_arm_rejects_bad_values_and_old_firmware(self):
        f = Fake()
        for bad in (1, 199, 65536, 2.0):
            with self.assertRaises(ValueError):
                f._arm_keepalive(bad, 1)
        f.reject_keepalive = True
        with self.assertRaisesRegex(RuntimeError, 'rejected the test-mode keepalive'):
            f._arm_keepalive(2000, 1)
        f._arm_keepalive(0, 1)
        self.assertEqual(f._keepalive_ms, 0)

    def test_renew_failure_names_the_master_fault(self):
        f = Fake()
        f._system_manual_active = True
        f._system_manual_owned = {(638, CHANNEL_LASER_DAC)}
        f.metrics = SimpleNamespace(current_state=0, fault='fan stall')

        def fail(*args, **kwargs):
            raise TimeoutError('ack')

        f.manual_command = fail
        with self.assertRaisesRegex(RuntimeError, r'master fault: fan stall'):
            f.renew_system_manual()
        self.assertEqual(f.last_master_fault, 'fan stall')
        self.assertFalse(f.system_manual_active)


class TakeOverTests(unittest.TestCase):
    def test_sequence_keeps_1550_running_and_takes_638_at_dac(self):
        f = Fake()
        result = f.take_over_638(31234)
        self.assertEqual(result.applied_value, 31234)
        sends = [c for c in f.calls if c[0] == 'send']
        self.assertEqual(sends, [
            ('send', CMD_OVERRIDE_ENTER, 1, 0),
            ('send', CMD_KEEPALIVE, 2000, 0),
            ('send', CMD_TRIGGER, 0, 638),
        ])
        self.assertIn((638, MANUAL_TAKE, CHANNEL_LASER_DAC, 31234), f.calls)
        self.assertNotIn(('stop',), f.calls)
        self.assertEqual(f._system_manual_owned, {(638, CHANNEL_LASER_DAC)})
        self.assertFalse(f._running)
        with self.assertRaisesRegex(RuntimeError, 'own firmware'):
            f.system_manual_command(1550, MANUAL_GET, CHANNEL_TEMPERATURE)
        self.assertEqual(list(f.system_manual_readback()), [638])

    def test_dac_outside_cap_is_refused_before_override(self):
        f = Fake()
        with self.assertRaises(ValueError):
            f.take_over_638(52401)
        self.assertFalse(any(c[0] == 'send' for c in f.calls))

    def test_failure_stops_the_box(self):
        f = Fake()
        f.reject_keepalive = True
        with self.assertRaisesRegex(RuntimeError, 'keepalive'):
            f.take_over_638(1000)
        self.assertIn(('stop',), f.calls)
        self.assertFalse(f.system_manual_active)

    def test_wrong_applied_value_stops_the_box(self):
        class Clamping(Fake):
            def manual_command(self, target, opcode, channel, value=0, timeout_s=1):
                result = super().manual_command(target, opcode, channel, value, timeout_s)
                if opcode == MANUAL_TAKE and channel == CHANNEL_LASER_DAC:
                    return reply(target, channel, 0)
                return result

        f = Clamping()
        with self.assertRaisesRegex(RuntimeError, 'applied DAC 0, not 1000'):
            f.take_over_638(1000)
        self.assertIn(('stop',), f.calls)


class FaultNameTests(unittest.TestCase):
    def metrics(self, terminal_state, fault_code):
        values = [0] * 19
        values[16], values[17] = terminal_state, fault_code
        return RuntimeMetrics(*values)

    def test_fault_named_only_when_window_ended_in_fault(self):
        self.assertEqual(self.metrics(6, 17).fault, 'PC keepalive expired')
        self.assertEqual(self.metrics(6, 7).fault, 'fan stall')
        self.assertEqual(self.metrics(6, 99).fault, 'fault 99')
        self.assertIsNone(self.metrics(7, 17).fault)
        self.assertIsNone(self.metrics(6, 0).fault)


class StreamPositionTests(unittest.TestCase):
    def test_manual_command_carries_stream_sample_and_round_trip(self):
        class Stream:
            running = True

            def _request_stream_control(self, payload, kind, timeout_s=1.0):
                return {'firmware_response': parse_manual_response(manual_response(txn=1)),
                        'stream_sample': 123456, 'host_round_trip_bound_ns': 2_500_000}

        ctrl = Controller()
        ctrl._stream = Stream()
        result = ctrl.manual_command(638, MANUAL_SET, CHANNEL_LASER_DAC, 33000)
        self.assertEqual(result.stream_sample, 123456)
        self.assertEqual(result.round_trip_ns, 2_500_000)
        self.assertEqual(result.applied_value, 33000)


class SweepFake(Fake):
    """Ring-backed fake: each SET writes its own block of samples."""

    SETTLE = 50_000   # 5 ms at 10 MSPS
    DWELL = 250_000   # 25 ms

    def __init__(self, level, applied=None):
        super().__init__()
        self._running = False
        self.values[638, 7] = 1000
        self.level = level
        self.applied = applied or (lambda dac: dac)
        self.buffer_capacity = 4_000_000
        self.buffer = np.zeros(self.buffer_capacity, dtype=np.uint16)
        self.samples_received = 0
        self.stream_stats = {'drops_seq': 0}
        self.telemetry = SimpleNamespace(board_638=SimpleNamespace(temperature_mdegc=25012))
        self.started_1550 = False

    def begin_system_manual(self, timeout_s=1.0, optical_diagnostics=False,
                            keepalive_ms=2000):
        self._system_manual_active = True
        self._system_manual_owned = {(638, CHANNEL_LASER_DAC), (1550, CHANNEL_LASER_DAC)}

    def start_1550_automatic(self, **kwargs):
        self.started_1550 = True
        self._system_manual_owned.discard((1550, CHANNEL_LASER_DAC))
        self._auto_1550 = True

    def wait_tec_lock(self, target, target_c=None, **kwargs):
        return target_c

    def manual_command(self, target, opcode, channel, value=0, timeout_s=1):
        if not (opcode == MANUAL_SET and channel == CHANNEL_LASER_DAC):
            return super().manual_command(target, opcode, channel, value, timeout_s)
        self.calls.append((target, opcode, channel, value))
        start = self.samples_received
        block = np.full(self.DWELL + 1000, self.level(value), dtype=np.uint16)
        block[:self.SETTLE] = 1  # the step transient the average must skip
        index = start % self.buffer_capacity
        self.buffer[index:index + len(block)] = block
        self.samples_received += len(block)
        return reply(target, channel, self.applied(value), stream_sample=start)


class ProbeSweepTests(unittest.TestCase):
    def test_sweep_averages_after_settle_and_ends_with_stop(self):
        f = SweepFake(lambda dac: 5000 + dac)
        progress = []
        result = f.run_probe_sweep(step=100, on_progress=progress.append)
        self.assertTrue(f.started_1550)
        self.assertEqual(result.current, list(range(0, 1001, 100)))
        self.assertEqual(result.photodetector, [5000.0 + d for d in range(0, 1001, 100)])
        self.assertFalse(result.saturated)
        self.assertEqual(result.max_current, 1000)
        self.assertEqual(result.temperature_c[0], 25.012)
        self.assertEqual(len(progress), 11)
        self.assertEqual(f.calls[-1], ('stop',))
        self.assertIn((638, MANUAL_SET, CHANNEL_LASER_DAC, 0), f.calls[-3:])

    def test_top_that_is_not_a_step_multiple_is_included(self):
        f = SweepFake(lambda dac: 5000)
        result = f.run_probe_sweep(step=300)
        self.assertEqual(result.current, [0, 300, 600, 900, 1000])

    def test_saturation_stops_early(self):
        f = SweepFake(lambda dac: 16000 if dac >= 500 else 5000)
        result = f.run_probe_sweep(step=100)
        self.assertTrue(result.saturated)
        self.assertEqual(result.current[-1], 500)

    def test_board_not_applying_the_step_stops_the_box(self):
        f = SweepFake(lambda dac: 5000, applied=lambda dac: 0)
        with self.assertRaisesRegex(RuntimeError, 'applied DAC 0, not 100'):
            f.run_probe_sweep(step=100)
        self.assertEqual(f.calls[-1], ('stop',))

    def test_stream_loss_rejects_the_sweep(self):
        class Lossy(SweepFake):
            def manual_command(self, target, opcode, channel, value=0, timeout_s=1):
                result = super().manual_command(target, opcode, channel, value, timeout_s)
                if value == 300:
                    self.stream_stats = {'drops_seq': 1}
                return result

        f = Lossy(lambda dac: 5000)
        with self.assertRaisesRegex(RuntimeError, 'discontinuity'):
            f.run_probe_sweep(step=100)
        self.assertEqual(f.calls[-1], ('stop',))

    def test_read_after_reply_starts_after_the_settle(self):
        f = SweepFake(lambda dac: 7000)
        f._system_manual_active = True
        f._system_manual_owned = {(638, CHANNEL_LASER_DAC)}
        step = f.system_manual_command(638, MANUAL_SET, CHANNEL_LASER_DAC, 400)
        block = f.read_after_reply(step, 1000, settle_s=0.005)
        self.assertEqual(len(block), 1000)
        self.assertTrue((block == 7000).all())
        self.assertEqual(f.read_after_reply(step, 10)[0], 1)  # the transient
        with self.assertRaisesRegex(RuntimeError, 'no stream position'):
            f.read_after_reply(SimpleNamespace(stream_sample=None), 10)
        with self.assertRaises(ValueError):
            f.read_samples(10, 10)

    def test_argument_checks_happen_before_any_command(self):
        f = SweepFake(lambda dac: 5000)
        for kwargs in ({'step': 0}, {'settle_s': .03}, {'dwell_s': 2.0}):
            with self.assertRaises(ValueError):
                f.run_probe_sweep(**kwargs)
        self.assertEqual(f.calls, [])


# -- Visible-PD build (docs/VISIBLE_PD_BOARD.md) ------------------------------

def capabilities(flags):
    from ultracoustics._internal.telemetry import StreamCapabilities
    return StreamCapabilities(flags, 16392, 16532, 503, 1, 1)


class SessionFake(Fake):
    """Runs the real begin_system_manual: ready telemetry, acks with times."""

    def __init__(self, flags):
        super().__init__()
        self._running = False
        self.flags = flags
        board = SimpleNamespace(temperature_age_ms=1, temperature_valid=True,
                                temperature_stale=False, temperature_fault=0)
        self.telemetry = SimpleNamespace(
            host_stale=False, host_age_s=0.0, received_monotonic_ns=2 ** 62,
            link_flags_638=0, link_flags_1550=0,
            board_638=board, board_1550=board)

    def telemetry_capabilities(self, timeout_s=1.0):
        self._capabilities = capabilities(self.flags)
        return self._capabilities

    def enable_telemetry(self, timeout_s=1.0):
        self.telemetry_capabilities(timeout_s)
        return self.telemetry

    def _send_confirmed(self, cmd, wValue=0, wIndex=0, timeout_s=.5):
        result = super()._send_confirmed(cmd, wValue, wIndex, timeout_s)
        import time
        result['bulk_write_completed_monotonic_ns'] = time.monotonic_ns() - 10 ** 10
        return result


def sends(fake):
    return [c[1:] for c in fake.calls if c[0] == 'send']


class VisiblePDSessionTests(unittest.TestCase):
    def setUp(self):
        patcher = patch('ultracoustics.system_control.time.sleep')
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_capability_bit_three(self):
        self.assertTrue(capabilities(15).visible_pd_board)
        self.assertFalse(capabilities(7).visible_pd_board)
        f = Fake()
        self.assertIsNone(f.visible_pd_board)
        self.assertEqual(f.laser_boards, (638, 1550))
        f._capabilities = capabilities(15)
        self.assertEqual(f.laser_boards, (638,))

    def test_visible_board_session_powers_and_takes_only_the_638(self):
        from ultracoustics._internal.protocol import CMD_POWER
        f = SessionFake(15)
        readback = f.begin_system_manual()
        self.assertEqual(list(readback), [638])
        self.assertEqual(f._session_boards, (638,))
        powered = [s for s in sends(f) if s[0] == CMD_POWER]
        self.assertEqual(powered, [(CMD_POWER, 1, 638)])
        triggers = [s for s in sends(f) if s[0] == CMD_TRIGGER]
        self.assertEqual(triggers, [(CMD_TRIGGER, 0, 638), (CMD_TRIGGER, 0, 1550)])
        self.assertIn((638, MANUAL_TAKE, CHANNEL_LASER_DAC, 0), f.calls)
        self.assertNotIn((1550, MANUAL_TAKE, CHANNEL_LASER_DAC, 0), f.calls)
        with self.assertRaisesRegex(RuntimeError, 'not part of this session'):
            f.system_manual_command(1550, MANUAL_GET, CHANNEL_TEMPERATURE)
        f.stop_system_confirmed()
        self.assertIsNone(f._session_boards)

    def test_system_board_session_is_unchanged(self):
        from ultracoustics._internal.protocol import CMD_POWER
        f = SessionFake(7)
        self.assertEqual(sorted(f.begin_system_manual()), [638, 1550])
        powered = [s for s in sends(f) if s[0] == CMD_POWER]
        self.assertEqual(powered, [(CMD_POWER, 1, 638), (CMD_POWER, 1, 1550)])
        self.assertIn((1550, MANUAL_TAKE, CHANNEL_LASER_DAC, 0), f.calls)

    def test_explicit_638_only_on_a_system_board(self):
        from ultracoustics._internal.protocol import CMD_POWER
        f = SessionFake(7)
        self.assertEqual(list(f.begin_system_manual(boards=(638,))), [638])
        powered = [s for s in sends(f) if s[0] == CMD_POWER]
        self.assertEqual(powered, [(CMD_POWER, 1, 638)])

    def test_asking_a_visible_board_for_the_1550_raises_before_override(self):
        f = SessionFake(15)
        with self.assertRaisesRegex(RuntimeError, 'has no 1550'):
            f.begin_system_manual(boards=(638, 1550))
        self.assertNotIn((CMD_OVERRIDE_ENTER, 1, 0), sends(f))

    def test_system_tests_are_refused_on_the_visible_board(self):
        from ultracoustics._internal.protocol import CMD_BOOT
        f = SessionFake(15)
        with self.assertRaisesRegex(RuntimeError, 'visible-PD build'):
            f.start_confirmed()
        self.assertNotIn(CMD_BOOT, [s[0] for s in sends(f)])

        f = SessionFake(15)
        f._capabilities = capabilities(15)
        with self.assertRaisesRegex(RuntimeError, 'visible-PD build'):
            f.take_over_638(1000)
        self.assertEqual(sends(f), [])

        f = SessionFake(15)
        f.streaming = True
        with self.assertRaisesRegex(RuntimeError, 'visible-PD build'):
            f.run_probe_sweep(step=100)
        self.assertEqual(sends(f), [])

        f = SessionFake(15)
        f._capabilities = capabilities(15)
        f._system_manual_active = True
        with self.assertRaisesRegex(RuntimeError, 'visible-PD build'):
            f.start_1550_automatic()
        self.assertEqual(sends(f), [])


class _Sent(Exception):
    pass


class _Conn:
    def __init__(self, reply=None):
        self.reply = reply
        self.sent = []

    def send_command(self, cmd, **kwargs):
        from ultracoustics._internal.protocol import CMD_IAP
        self.sent.append(cmd)
        if cmd == CMD_IAP:
            raise _Sent()

    def receive(self, size, timeout_ms=0):
        return self.reply


def master_image(target_id):
    import struct
    return b'UCBT' + struct.pack('<HHII', 1, target_id, 0, 0) + bytes(4096)


class VisiblePDFirmwareInfoTests(unittest.TestCase):
    def test_build_profile_is_read(self):
        import struct
        from ultracoustics._internal.maintenance import DiagnosticsManager
        for profile, name in ((0, 'system'), (1, 'visible_pd'), (7, 'unknown(7)')):
            raw = b'3fb7c27'.ljust(32, b'\0') + struct.pack('<IHH', 0, 1, profile)
            manager = DiagnosticsManager(_Conn(raw))
            info = manager._read_firmware_info(40, 500)
            self.assertEqual((info['build_profile'], info['profile']), (profile, name))

    def test_flash_master_checks_the_running_build(self):
        from ultracoustics._internal.maintenance import Programmer
        cases = [
            ('system', 0x0001, False, _Sent),
            ('visible_pd', 0x0002, False, _Sent),
            ('system', 0x0002, False, ValueError),
            (None, 0x0002, False, ValueError),
            ('visible_pd', 0x0001, False, ValueError),
            ('visible_pd', 0x0001, True, _Sent),
            (None, 0x0001, False, _Sent),
        ]
        for running, target, allow, expected in cases:
            programmer = Programmer(_Conn())
            with patch.object(Programmer, '_running_profile', return_value=running):
                with self.assertRaises(expected, msg=(running, target, allow)):
                    programmer.flash_master(master_image(target),
                                            allow_profile_change=allow)

    def test_unknown_target_is_refused(self):
        from ultracoustics._internal.maintenance import Programmer
        with self.assertRaisesRegex(ValueError, 'target_id=0x0003'):
            Programmer._validate_master_magic(master_image(0x0003))

    def test_system_info_skips_the_1550_on_a_visible_board(self):
        from ultracoustics._internal.maintenance import DiagnosticsManager
        manager = DiagnosticsManager(_Conn())
        asked = []
        manager.query_firmware_info = lambda: {'profile': 'visible_pd'}
        manager.query_slave_version = lambda target: asked.append(('fw', target)) or {}
        manager.read_identity = lambda target: asked.append(('id', target)) or None
        info = manager.read_system_info()
        self.assertNotIn(('fw', 1550), asked)
        self.assertNotIn(('id', 1550), asked)
        self.assertIn(('fw', 638), asked)
        self.assertEqual(set(info['skipped']), {'fw_1550', 'identity_1550'})
        self.assertEqual(info['errors'], {})


if __name__ == '__main__':
    unittest.main()
