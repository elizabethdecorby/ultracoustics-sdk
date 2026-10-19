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
    if opcode not in range(MANUAL_GET, MANUAL_RENEW + 1) or channel not in range(1, 14):
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
    #: Samples the reader had received when this reply arrived. The master
    #: sends the reply after the slave applied the value, on the same bulk
    #: endpoint as the ADC records, so this marks the change in the stream.
    stream_sample: Optional[int] = None
    #: Host bound from queueing the request to receiving the reply.
    round_trip_ns: Optional[int] = None

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


#: Master ``BootState_t`` value for FAULT.
BOOT_STATE_FAULT = 6

#: Master fault codes, ``BootFaultCode_t`` in the master's boot_sequencer.h.
MASTER_FAULT_NAMES = {
    1: "probe signal never seen",
    2: "probe signal lost while waiting for the 638",
    3: "probe signal lost",
    4: "1550 never reported ready",
    5: "638 never locked",
    6: "1550 dropped ready",
    7: "fan stall",
    8: "slave refused the DAC write",
    9: "slave DAC programming timed out",
    10: "slave version query failed",
    11: "slave flash relay failed",
    12: "IAP image has a bad header",
    13: "IAP image CRC mismatch",
    14: "IAP flash write failed",
    15: "boot state machine reached an undefined state",
    16: "1550 startup DAC not confirmed",
    17: "PC keepalive expired",
}


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

    @property
    def fault(self) -> Optional[str]:
        """Why the last window ended in FAULT, or ``None``."""
        if self.terminal_state != BOOT_STATE_FAULT or not self.fault_code:
            return None
        return MASTER_FAULT_NAMES.get(self.fault_code, f"fault {self.fault_code}")


def parse_runtime_metrics(raw: bytes) -> RuntimeMetrics:
    if len(raw) != 72 or raw[:4] != b"URTM":
        raise ControlProtocolError("URTM response identity or length mismatch")
    schema, size = struct.unpack_from("<HH", raw, 4)
    if schema != 1 or size != 72:
        raise ControlProtocolError("unsupported URTM schema")
    return RuntimeMetrics(*struct.unpack_from("<7IQ4IHHBBBBI", raw, 8))
