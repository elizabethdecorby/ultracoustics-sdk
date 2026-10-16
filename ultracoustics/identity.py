"""
Board identity records: hardware details programmed into each board.

The record layout, TLV tags and mainboard reply format are specified in
docs/BOARD_IDENTITY.md. This module is the host-side codec; it has no I/O.
"""

import struct
from dataclasses import dataclass, field, fields
from typing import Dict, Optional

RECORD_SIZE = 128
HEADER_SIZE = 16
PAYLOAD_MAX = RECORD_SIZE - HEADER_SIZE
MAGIC = b"UCID"
LAYOUT_VERSION = 1

REPLY_SIZE = 140
REPLY_MAGIC = b"UTID"
REPLY_SCHEMA = 1

BOARD_MAIN = 1
BOARD_1550 = 2
BOARD_638 = 3

# USB wIndex targets for the 'n' / 'N' commands.
TARGET_MAIN = 0
TARGET_1550 = 1550
TARGET_638 = 638

BOARD_FOR_TARGET = {TARGET_MAIN: BOARD_MAIN, TARGET_1550: BOARD_1550, TARGET_638: BOARD_638}

OP_READ = 0
OP_WRITE = 1

STATUS_OK = 0
STATUS_EMPTY = 1
STATUS_NO_RESPONSE = 2
STATUS_LINK_CRC = 3
STATUS_WRITE_FAIL = 4
STATUS_VERIFY_FAIL = 5
STATUS_BAD_RECORD = 6
STATUS_BAD_STATE = 7

STATUS_NAMES = {
    STATUS_OK: "OK",
    STATUS_EMPTY: "EMPTY",
    STATUS_NO_RESPONSE: "NO_RESPONSE",
    STATUS_LINK_CRC: "LINK_CRC",
    STATUS_WRITE_FAIL: "WRITE_FAIL",
    STATUS_VERIFY_FAIL: "VERIFY_FAIL",
    STATUS_BAD_RECORD: "BAD_RECORD",
    STATUS_BAD_STATE: "BAD_STATE",
}

_STR, _U32 = "str", "u32"
# Pot resistances: u16 counts of 1/scale ohm. Records written before this
# change used float32 (4 bytes); decode still accepts that length.
_OHM_X10, _OHM = "ohm_x10", "ohm"
_OHM_SCALE = {_OHM_X10: 10, _OHM: 1}
_ALL = (BOARD_MAIN, BOARD_1550, BOARD_638)
_LASERS = (BOARD_1550, BOARD_638)

# attribute -> (tag, kind, max string length, boards allowed)
_FIELDS = {
    "serial":         (0x01, _STR, 24, _ALL),
    "pcb_rev":        (0x02, _STR, 12, _ALL),
    "pd_pcb_rev":     (0x03, _STR, 12, (BOARD_MAIN,)),
    "circ_sn":        (0x04, _STR, 24, (BOARD_MAIN,)),
    "driver_pot_ohm": (0x10, _OHM_X10, 0, _LASERS),  # 0.1 ohm, <= 6553.5
    "lpf_pot_ohm":    (0x11, _OHM, 0, _LASERS),      # 1 ohm, <= 65535
    "pd_pot_ohm":     (0x12, _OHM, 0, _LASERS),      # 1 ohm, <= 65535
    "programmed_at":  (0x20, _U32, 0, _ALL),
    "report_id":      (0x21, _STR, 24, _ALL),
}
_BY_TAG = {spec[0]: (name, spec) for name, spec in _FIELDS.items()}
# Limits are chosen so a record with every field at maximum length fits.


class IdentityError(ValueError):
    """A record or reply that cannot be encoded or decoded."""


def crc32_mpeg2(data: bytes) -> int:
    """CRC-32/MPEG-2 (poly 0x04C11DB7, init 0xFFFFFFFF, no reflect/xorout)."""
    crc = 0xFFFFFFFF
    for b in data:
        crc ^= b << 24
        for _ in range(8):
            crc = ((crc << 1) ^ 0x04C11DB7) if crc & 0x80000000 else (crc << 1)
            crc &= 0xFFFFFFFF
    return crc


@dataclass
class BoardIdentity:
    """Hardware details for one board. None means "not recorded"."""

    board: int
    serial: Optional[str] = None
    pcb_rev: Optional[str] = None
    pd_pcb_rev: Optional[str] = None
    circ_sn: Optional[str] = None
    driver_pot_ohm: Optional[float] = None
    lpf_pot_ohm: Optional[float] = None
    pd_pot_ohm: Optional[float] = None
    programmed_at: Optional[int] = None
    report_id: Optional[str] = None
    unknown: Dict[int, bytes] = field(default_factory=dict)

    def details(self) -> dict:
        """Recorded hardware fields only (no provenance), for comparisons."""
        skip = {"board", "programmed_at", "report_id", "unknown"}
        return {f.name: getattr(self, f.name) for f in fields(self)
                if f.name not in skip and getattr(self, f.name) is not None}


@dataclass
class StoredRecord:
    """A record as read back from a board."""

    board: int
    seq: int
    identity: BoardIdentity
    raw: bytes


@dataclass
class IdentityResult:
    """One mainboard 'n'/'N' reply."""

    op: int
    target: int
    status: int
    slots_free: int
    record: Optional[StoredRecord]
    raw: bytes

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    @property
    def status_name(self) -> str:
        return STATUS_NAMES.get(self.status, f"UNKNOWN({self.status})")


def encode_payload(identity: BoardIdentity) -> bytes:
    if identity.board not in _ALL:
        raise IdentityError(f"unknown board type {identity.board}")
    out = bytearray()
    for name, (tag, kind, max_len, boards) in _FIELDS.items():
        value = getattr(identity, name)
        if value is None:
            continue
        if identity.board not in boards:
            raise IdentityError(f"{name} is not stored on board type {identity.board}")
        if kind == _STR:
            data = str(value).encode("utf-8")
            if len(data) > max_len:
                raise IdentityError(f"{name} is {len(data)} bytes; max {max_len}")
        elif kind in _OHM_SCALE:
            scale = _OHM_SCALE[kind]
            counts = round(float(value) * scale)
            if not 0 <= counts <= 0xFFFF:
                raise IdentityError(
                    f"{name} = {value} ohm is outside 0-{0xFFFF / scale:g} ohm")
            data = struct.pack("<H", counts)
        else:
            data = struct.pack("<I", int(value))
        out += bytes((tag, len(data))) + data
    for tag, data in sorted(identity.unknown.items()):
        out += bytes((tag, len(data))) + data
    if len(out) > PAYLOAD_MAX:
        raise IdentityError(f"payload is {len(out)} bytes; max {PAYLOAD_MAX}")
    return bytes(out)


def decode_payload(board: int, payload: bytes) -> BoardIdentity:
    identity = BoardIdentity(board=board)
    i = 0
    while i < len(payload):
        if i + 2 > len(payload):
            raise IdentityError("truncated TLV header")
        tag, length = payload[i], payload[i + 1]
        value = payload[i + 2:i + 2 + length]
        if len(value) != length:
            raise IdentityError(f"truncated TLV 0x{tag:02X}")
        i += 2 + length
        known = _BY_TAG.get(tag)
        if known is None:
            identity.unknown[tag] = bytes(value)
            continue
        name, (_tag, kind, _max, _boards) = known
        if kind == _STR:
            setattr(identity, name, value.decode("utf-8", errors="replace"))
        elif kind in _OHM_SCALE and length == 2:
            setattr(identity, name, struct.unpack("<H", value)[0] / _OHM_SCALE[kind])
        elif kind in _OHM_SCALE and length == 4:  # float32 from before u16
            setattr(identity, name, struct.unpack("<f", value)[0])
        elif kind == _U32 and length == 4:
            setattr(identity, name, struct.unpack("<I", value)[0])
        else:
            identity.unknown[tag] = bytes(value)
    return identity


def _record_crc(header12: bytes, payload: bytes) -> int:
    return crc32_mpeg2(header12 + payload)


def build_record(identity: BoardIdentity, seq: int = 0) -> bytes:
    """Build a 128-byte record. Firmware re-stamps board, seq and CRC on store."""
    payload = encode_payload(identity)
    header12 = MAGIC + struct.pack("<BBHI", LAYOUT_VERSION, identity.board, len(payload), seq)
    crc = _record_crc(header12, payload)
    record = header12 + struct.pack("<I", crc) + payload
    return record + b"\xFF" * (RECORD_SIZE - len(record))


def parse_record(raw: bytes) -> Optional[StoredRecord]:
    """Decode a stored record. Returns None for an erased (all 0xFF) record."""
    if len(raw) != RECORD_SIZE:
        raise IdentityError(f"record is {len(raw)} bytes; expected {RECORD_SIZE}")
    raw = bytes(raw)
    if raw == b"\xFF" * RECORD_SIZE:
        return None
    if raw[:4] != MAGIC:
        raise IdentityError("bad record magic")
    version, board, length, seq, crc = struct.unpack_from("<BBHII", raw, 4)
    if version != LAYOUT_VERSION:
        raise IdentityError(f"unsupported layout version {version}")
    if length > PAYLOAD_MAX:
        raise IdentityError(f"payload length {length} > {PAYLOAD_MAX}")
    payload = raw[HEADER_SIZE:HEADER_SIZE + length]
    if _record_crc(raw[:12], payload) != crc:
        raise IdentityError("record CRC mismatch")
    return StoredRecord(board=board, seq=seq, identity=decode_payload(board, payload), raw=raw)


def parse_reply(raw: bytes) -> IdentityResult:
    """Decode the mainboard's 140-byte 'UTID' reply to 'n' or 'N'."""
    raw = bytes(raw)
    if len(raw) != REPLY_SIZE or raw[:4] != REPLY_MAGIC:
        raise IdentityError("not a UTID identity reply")
    schema, op, target, status, slots_free = struct.unpack_from("<BBHBB", raw, 4)
    if schema != REPLY_SCHEMA:
        raise IdentityError(f"unsupported reply schema {schema}")
    record = None
    if status == STATUS_OK:
        record = parse_record(raw[12:12 + RECORD_SIZE])
    elif status == STATUS_VERIFY_FAIL:
        try:  # best effort: show what the board actually holds
            record = parse_record(raw[12:12 + RECORD_SIZE])
        except IdentityError:
            record = None
    return IdentityResult(op=op, target=target, status=status,
                          slots_free=slots_free, record=record, raw=raw)
