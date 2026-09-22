"""Strict bounded parsers for runtime and leased manual control."""

from dataclasses import dataclass
import struct
from typing import Optional

MANUAL_GET = 0x41
MANUAL_TAKE = 0x42
MANUAL_SET = 0x43
MANUAL_RELEASE = 0x44
MANUAL_RENEW = 0x45
CHANNEL_TEMPERATURE = 1
CHANNEL_LASER_DAC = 2


class ControlProtocolError(ValueError):
    pass


class ManualTransportError(RuntimeError):
    def __init__(self, status: int):
        names = {1: "SPI timeout", 2: "context canceled"}
        super().__init__(f"manual transport failed: {names.get(status, status)}")
        self.status = status


def crc8_poly31(data: bytes) -> int:
    crc = 0
    for value in data:
        crc ^= value
        for _ in range(8):
            crc = ((crc << 1) ^ 0x31) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def pack_manual_request(opcode: int, channel: int, value: int,
                        transaction: int) -> bytes:
    if opcode not in range(MANUAL_GET, MANUAL_RENEW + 1) or channel not in range(1, 13):
        raise ValueError("invalid manual opcode or channel")
    if not -(1 << 31) <= value < (1 << 31):
        raise ValueError("manual value is outside int32")
    if not 0 <= transaction <= 0xFF:
        raise ValueError("manual transaction is outside uint8")
    body = bytes((0xAA, opcode, channel, 0)) + struct.pack(">iB", value, transaction)
    return body + bytes((crc8_poly31(body),))


@dataclass(frozen=True)
class ManualReply:
    target: int
    transport_status: int
    transaction: int
    opcode: int
    channel: int
    status: Optional[int]
    owner: Optional[int]
    applied_value: Optional[int]
    raw_ack: bytes

    def require_transport(self):
        if self.transport_status:
            raise ManualTransportError(self.transport_status)
        return self


def parse_manual_response(raw: bytes) -> ManualReply:
    if len(raw) != 32:
        raise ControlProtocolError("UTMC response must be exactly 32 bytes")
    magic, schema, size, target = struct.unpack_from("<4sHHH", raw)
    transport, reserved, txn, opcode, channel, reserved2 = raw[10:16]
    if (magic != b"UTMC" or schema != 1 or size != 32 or
            target not in (638, 1550) or reserved or reserved2 or any(raw[26:])):
        raise ControlProtocolError("invalid UTMC header, target, or reserved bytes")
    ack = bytes(raw[16:26])
    if transport:
        if transport not in (1, 2) or any(ack):
            raise ControlProtocolError("invalid failed-transport UTMC response")
        return ManualReply(target, transport, txn, opcode, channel,
                           None, None, None, ack)
    if ack[:3] != b"\xBB\xBB\xA5" or ack[3] != txn:
        raise ControlProtocolError("manual ACK identity mismatch")
    if crc8_poly31(ack[:9]) != ack[9]:
        raise ControlProtocolError("manual ACK CRC mismatch")
    return ManualReply(target, 0, txn, opcode, channel, ack[4], ack[5],
                       int.from_bytes(ack[6:9], "big", signed=True), ack)


@dataclass(frozen=True)
class RuntimeMetrics:
    flags: int
    snapshot_ms: int
    window_start_ms: int
    window_end_ms: int
    window_age_ms: int
    window_duration_ms: int
    idle_residency_ms: int
    non_sleep_cycles: int
    idle_entries: int
    pssi_completed: int
    pssi_overflows: int
    pssi_service_late: int
    active_basis_points: int
    idle_basis_points: int
    current_state: int
    last_operational_state: int
    terminal_state: int
    fault_code: int
    pending_max_depth: int

    @property
    def active_cycles_valid(self) -> bool:
        return bool(self.flags & (1 << 3))

    @property
    def active_percent(self):
        return self.active_basis_points / 100.0 if self.active_cycles_valid else None

    @property
    def idle_percent(self):
        return self.idle_basis_points / 100.0 if self.active_cycles_valid else None


def parse_runtime_metrics(raw: bytes) -> RuntimeMetrics:
    if len(raw) != 72 or raw[:4] != b"URTM":
        raise ControlProtocolError("URTM response identity or length mismatch")
    schema, size = struct.unpack_from("<HH", raw, 4)
    if schema != 1 or size != 72:
        raise ControlProtocolError("unsupported URTM schema")
    return RuntimeMetrics(*struct.unpack_from("<7IQ4IHHBBBBI", raw, 8))
