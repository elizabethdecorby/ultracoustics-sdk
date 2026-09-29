"""Strict one-shot 1550 automatic startup readback."""

from dataclasses import dataclass


class StartupIlluminationProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class StartupIllumination:
    state: str
    staged_dac: int
    applied_dac: int
    boot_state: int


def parse_startup_illumination(raw: bytes) -> StartupIllumination:
    states = ('clear', 'staged', 'applying', 'applied')
    if (len(raw) != 32 or raw[:4] != b'UTAI' or raw[4:6] != b'\x01\x20'
            or raw[6] >= len(states) or raw[7] or any(raw[13:])):
        raise StartupIlluminationProtocolError('invalid UTAI response')
    staged = int.from_bytes(raw[8:10], 'little')
    applied = int.from_bytes(raw[10:12], 'little')
    state = states[raw[6]]
    if ((state == 'clear' and (staged or applied)) or
            (state in ('staged', 'applying') and (not 1 <= staged <= 43253 or applied)) or
            (state == 'applied' and (staged or not 1 <= applied <= 43253))):
        raise StartupIlluminationProtocolError('inconsistent UTAI state')
    return StartupIllumination(state, staged, applied, raw[12])
