"""Offline one-use startup wire and ordering checks."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ultracoustics._internal.startup_illumination import (
    StartupIlluminationProtocolError, parse_startup_illumination)
from ultracoustics.system_control import SystemControlMixin


def packet(state, staged=0, applied=0, boot=0):
    raw = bytearray(32)
    raw[:6] = b'UTAI\x01\x20'
    raw[6] = state
    raw[8:10] = staged.to_bytes(2, 'little')
    raw[10:12] = applied.to_bytes(2, 'little')
    raw[12] = boot
    return bytes(raw)


class FakeSystem(SystemControlMixin):
    def __init__(self):
        self.events = []
        self._running = False
        self._last_manual_rail_off_ns = 0
        self._stream = SimpleNamespace(running=True, _request_stream_control=self.request)
        self.stage_reply = None
        self.cleanup_fail = False
        self.old_firmware = False

    def stop_system_confirmed(self, timeout_s):
        self.events.append('stop')
        if self.cleanup_fail and len([e for e in self.events if e == 'stop']) > 1:
            raise RuntimeError('rail readback unavailable')
        self._last_manual_rail_off_ns = 0

    def stop_confirmed(self, timeout_s):
        self.events.append('stop_after_failure')

    def _send_confirmed(self, command, timeout_s=1.0):
        self.events.append('boot')
        return {'transport_status': 'delivered'}

    def request(self, payload, kind, timeout_s=1.0):
        self.events.append(('request', payload[1], payload[3] | payload[4] << 8))
        action, dac = payload[1], payload[3] | payload[4] << 8
        if self.old_firmware:
            raise TimeoutError('no UTAI response')
        if action == 1:
            return {'firmware_response': self.stage_reply or
                    parse_startup_illumination(packet(1, dac))}
        return {'firmware_response': parse_startup_illumination(packet(3, applied=7500, boot=2))}


class StartupTests(unittest.TestCase):
    def test_parser_rejects_old_or_inconsistent_firmware(self):
        for raw in (b'\0' * 32, packet(1), packet(3, staged=7500, applied=7500),
                    packet(3, applied=50000)):
            with self.assertRaises(StartupIlluminationProtocolError):
                parse_startup_illumination(raw)

    def test_stop_stage_then_boot_and_exact_application(self):
        fake = FakeSystem()
        with patch('ultracoustics.system_control.time.sleep'):
            result = fake.start_with_startup_illumination_1550(7500)
        self.assertEqual(fake.events[:3], ['stop', ('request', 1, 7500), 'boot'])
        self.assertEqual(result['illumination'].applied_dac, 7500)

    def test_out_of_range_never_boots(self):
        fake = FakeSystem()
        with patch('ultracoustics.system_control.time.sleep'):
            with self.assertRaises(ValueError):
                fake.start_with_startup_illumination_1550(43254)
        self.assertNotIn('boot', fake.events)
        self.assertEqual(fake.events.count('stop'), 2)

    def test_staging_mismatch_clears_with_verified_stop(self):
        fake = FakeSystem()
        fake.stage_reply = parse_startup_illumination(packet(1, 6500))
        with patch('ultracoustics.system_control.time.sleep'):
            with self.assertRaisesRegex(RuntimeError, 'staging readback differs'):
                fake.start_with_startup_illumination_1550(7500)
        self.assertNotIn('boot', fake.events)
        self.assertEqual(fake.events.count('stop'), 2)

    def test_cleanup_failure_is_reported(self):
        fake = FakeSystem()
        fake.cleanup_fail = True
        with patch('ultracoustics.system_control.time.sleep'):
            with self.assertRaisesRegex(RuntimeError, 'STOP verification also failed'):
                fake.start_with_startup_illumination_1550(43254)

    def test_old_master_never_receives_boot(self):
        fake = FakeSystem()
        fake.old_firmware = True
        with patch('ultracoustics.system_control.time.sleep'):
            with self.assertRaisesRegex(TimeoutError, 'no UTAI response'):
                fake.start_with_startup_illumination_1550(7500)
        self.assertNotIn('boot', fake.events)
        self.assertEqual(fake.events.count('stop'), 2)

    def test_visible_pd_master_is_refused_after_stop(self):
        fake = FakeSystem()
        fake._capabilities = SimpleNamespace(visible_pd_board=True)
        with patch('ultracoustics.system_control.time.sleep'):
            with self.assertRaisesRegex(RuntimeError, 'visible-PD'):
                fake.start_with_startup_illumination_1550(7500)
        self.assertEqual(fake.events, ['stop'])

    def test_default_confirmed_start_sends_no_startup_command(self):
        fake = FakeSystem()
        with patch('ultracoustics.system_control.time.sleep'):
            fake.start_confirmed()
        self.assertEqual(fake.events, ['stop', 'boot'])


if __name__ == '__main__':
    unittest.main()
