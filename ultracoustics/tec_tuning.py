"""
TEC tuning records: per-laser thermal PID gains stored on each laser board.

The record layout, firmware bounds and mainboard commands are specified in
docs/TEC_TUNING.md. Records share the identity record header and the 'n' /
'N' commands (record kind 1). On the 638 the record also carries the lock PI
gains (docs/TEC_TUNING.md, "Lock PI fields (638)"). This module is the
host-side codec; it has no I/O.
"""

import math
import struct
from dataclasses import dataclass, fields, replace
from typing import Optional

from . import identity as _identity
from .identity import (BOARD_1550, BOARD_638, HEADER_SIZE, PAYLOAD_MAX,
                       RECORD_SIZE, STATUS_NAMES, STATUS_OK, STATUS_VERIFY_FAIL,
                       TARGET_1550, TARGET_638, crc32_mpeg2)

MAGIC = b"UCTN"
LAYOUT_VERSION = 1
KIND = _identity.KIND_TEC_TUNING
PAYLOAD_SIZE = 48  # layout version 1 writers
LOCK_PAYLOAD_SIZE = 60  # writers that carry lock PI fields

BOARD_FOR_TARGET = {TARGET_1550: BOARD_1550, TARGET_638: BOARD_638}

# Firmware acceptance ranges: name -> (low, low inclusive, high inclusive).
BOUNDS = {
    "kp": (0.0, True, 1.0),
    "ki": (0.001, True, 2.0),
    "kd": (0.0, True, 0.5),
    "integrator_limit": (0.0, False, 20.0),
}
# Both boards clamp the normalized PID output to +/-0.3.
PID_OUTPUT_MAX = 0.3

_FLOATS = ("kp", "ki", "kd", "integrator_limit")  # payload offsets 0, 4, 8, 12
_FLAGS = ("setpoint_filter", "conditional_integration")  # offsets 16, 17
_TUNED_AT, _REPORT_ID = 20, 24
REPORT_ID_MAX = 24
_UNSET32 = b"\xFF" * 4
# The 1550 firmware has neither pull-in feature (docs/TEC_TUNING.md).
_FLAG_BOARDS = (BOARD_638,)
TEC_FIELDS = _FLOATS + _FLAGS

# Lock PI fields, 638 only: the normalized PI gains of normalized PI contract
# v2 (Kp, Ki in 1/s, second-integrator zero in Hz). Validated as a group of
# their own, independently of the TEC fields. Accepted ranges, inclusive.
LOCK_BOUNDS = {
    "lock_kp": (0.0, 5.115),
    "lock_ki_per_s": (0.0, 16383.0),
    "lock_zero_hz": (0, 63),
}
_LOCK_FLOATS = ("lock_kp", "lock_ki_per_s")  # payload offsets 48, 52
_LOCK_ZERO = 56  # u8, 0xFF not set; 57-59 reserved 0xFF
LOCK_FIELDS = _LOCK_FLOATS + ("lock_zero_hz",)
# A read-modify-write waits this long between the read reply and the write:
# the mainboard resumes streaming and returns to IDLE 100 ms after a reply.
RMW_GAP_S = 0.25


class TuningError(ValueError):
    """A tuning record or reply that cannot be encoded or decoded."""


@dataclass
class TecTuning:
    """TEC PID settings for one laser board, plus the lock PI gains on the
    638. None means "compiled default"."""

    board: int
    kp: Optional[float] = None
    ki: Optional[float] = None
    kd: Optional[float] = None
    integrator_limit: Optional[float] = None
    setpoint_filter: Optional[bool] = None
    conditional_integration: Optional[bool] = None
    tuned_at: Optional[int] = None
    report_id: Optional[str] = None
    lock_kp: Optional[float] = None
    lock_ki_per_s: Optional[float] = None
    lock_zero_hz: Optional[int] = None

    def settings(self) -> dict:
        """TEC gains and flags that override a compiled default (no
        provenance, no lock fields)."""
        return {name: getattr(self, name) for name in _FLOATS + _FLAGS
                if getattr(self, name) is not None}

    def effective_integrator_limit(self, compiled_default: float) -> float:
        """The integrator limit the firmware runs with for this record."""
        if self.integrator_limit is not None:
            return self.integrator_limit
        if self.ki is not None:
            return PID_OUTPUT_MAX / _f32(self.ki)
        return compiled_default


@dataclass
class StoredTuning:
    """A tuning record as read back from a board.

    ``rejected`` is None if the laser firmware would apply this record, or
    the reason it keeps its compiled defaults instead. ``lock_rejected`` is
    the same for the lock PI fields, which the 638 checks as a group of their
    own: None if it would use them (or none are set), else the reason it
    keeps its profile lock gains.
    """

    board: int
    seq: int
    tuning: TecTuning
    raw: bytes
    rejected: Optional[str] = None
    lock_rejected: Optional[str] = None


@dataclass
class TuningResult:
    """One mainboard 'n'/'N' reply for record kind 1."""

    op: int
    target: int
    status: int
    slots_free: int
    record: Optional[StoredTuning]
    raw: bytes

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    @property
    def status_name(self) -> str:
        return STATUS_NAMES.get(self.status, f"UNKNOWN({self.status})")


def _f32(value: float) -> float:
    """The float32 the firmware will see."""
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _short_f32(value: float) -> float:
    """The shortest decimal with this float32: a lock gain written as 0.6
    reads back as 0.6, not 0.6000000238."""
    for digits in range(1, 10):  # 9 significant digits always round-trip
        candidate = float(f"{value:.{digits}g}")
        if _f32(candidate) == value:
            return candidate
    return value


def _lock_problem(tuning: TecTuning) -> Optional[str]:
    """Why the 638 would ignore this record's lock PI fields, or None."""
    if all(getattr(tuning, name) is None for name in LOCK_FIELDS):
        return None
    if tuning.board != BOARD_638:
        return f"lock PI fields are 638 only (board type {tuning.board})"
    for name in _LOCK_FLOATS:
        value = getattr(tuning, name)
        if value is None:
            continue
        low, high = LOCK_BOUNDS[name]
        try:
            v = _f32(value)  # compare as the firmware does
        except (OverflowError, TypeError, ValueError):
            return f"{name} = {value!r} is not a float32 value"
        if not math.isfinite(v) or not _f32(low) <= v <= _f32(high):
            return f"{name} = {value} is outside [{low:g}, {high:g}]"
    zero = tuning.lock_zero_hz
    low, high = LOCK_BOUNDS["lock_zero_hz"]
    if zero is not None and (isinstance(zero, bool) or not isinstance(zero, int)
                             or not low <= zero <= high):
        return f"lock_zero_hz = {zero!r} is outside {low}..{high}"
    return None


def validate(tuning: TecTuning) -> None:
    """Raise TuningError unless the laser firmware would accept this record."""
    if tuning.board not in BOARD_FOR_TARGET.values():
        raise TuningError(f"board type {tuning.board} has no TEC tuning record")
    for name in _FLOATS:
        value = getattr(tuning, name)
        if value is None:
            continue
        low, low_inclusive, high = BOUNDS[name]
        try:
            v, lo, hi = _f32(value), _f32(low), _f32(high)  # compare as the firmware does
        except (OverflowError, TypeError, ValueError):
            raise TuningError(f"{name} = {value!r} is not a float32 value") from None
        if not math.isfinite(v) or v > hi or v < lo or (v == lo and not low_inclusive):
            bracket = "[" if low_inclusive else "("
            raise TuningError(f"{name} = {value} is outside {bracket}{low:g}, {high:g}]")
    for name in _FLAGS:
        value = getattr(tuning, name)
        if value is not None and not isinstance(value, bool):
            raise TuningError(f"{name} must be True, False or None")
        if value and tuning.board not in _FLAG_BOARDS:
            raise TuningError(f"{name} is not supported on board type {tuning.board}")
    if tuning.report_id is not None and len(tuning.report_id.encode("utf-8")) > REPORT_ID_MAX:
        raise TuningError(f"report_id is longer than {REPORT_ID_MAX} bytes")
    if tuning.tuned_at is not None and not 0 <= int(tuning.tuned_at) < 0xFFFFFFFF:
        raise TuningError("tuned_at is not a u32 timestamp")
    problem = _lock_problem(tuning)
    if problem is not None:
        raise TuningError(problem)


def encode_payload(tuning: TecTuning) -> bytes:
    validate(tuning)
    lock = any(getattr(tuning, name) is not None for name in LOCK_FIELDS)
    out = bytearray(b"\xFF" * (LOCK_PAYLOAD_SIZE if lock else PAYLOAD_SIZE))
    for i, name in enumerate(_FLOATS):
        value = getattr(tuning, name)
        if value is not None:
            out[4 * i:4 * i + 4] = struct.pack("<f", float(value))
    for i, name in enumerate(_FLAGS):
        value = getattr(tuning, name)
        if value is not None:
            out[16 + i] = 1 if value else 0
    if tuning.tuned_at is not None:
        out[_TUNED_AT:_TUNED_AT + 4] = struct.pack("<I", int(tuning.tuned_at))
    if tuning.report_id is not None:
        data = tuning.report_id.encode("utf-8")
        out[_REPORT_ID:_REPORT_ID + len(data)] = data
    for i, name in enumerate(_LOCK_FLOATS):
        value = getattr(tuning, name)
        if value is not None:
            out[48 + 4 * i:52 + 4 * i] = struct.pack("<f", float(value))
    if tuning.lock_zero_hz is not None:
        out[_LOCK_ZERO] = tuning.lock_zero_hz
    return bytes(out)


def _decode(board: int, payload: bytes):
    """(tuning, problem): fields past the payload end are unset, as in the
    firmware. A flag byte other than 0, 1 or 0xFF is left unset and reported."""
    tuning = TecTuning(board=board)
    problem = None
    for i, name in enumerate(_FLOATS):
        raw = payload[4 * i:4 * i + 4]
        if len(raw) == 4 and raw != _UNSET32:
            value = struct.unpack("<f", raw)[0]
            if not math.isnan(value):
                setattr(tuning, name, value)
    for i, name in enumerate(_FLAGS):
        if len(payload) > 16 + i and payload[16 + i] != 0xFF:
            if payload[16 + i] in (0, 1):
                setattr(tuning, name, payload[16 + i] == 1)
            elif problem is None:
                problem = f"{name} byte is 0x{payload[16 + i]:02X}"
    raw = payload[_TUNED_AT:_TUNED_AT + 4]
    if len(raw) == 4 and raw != _UNSET32:
        tuning.tuned_at = struct.unpack("<I", raw)[0]
    raw = payload[_REPORT_ID:_REPORT_ID + REPORT_ID_MAX]
    if len(raw) == REPORT_ID_MAX and raw.rstrip(b"\xFF"):
        tuning.report_id = raw.rstrip(b"\xFF").decode("utf-8", errors="replace")
    for i, name in enumerate(_LOCK_FLOATS):
        raw = payload[48 + 4 * i:52 + 4 * i]
        if len(raw) == 4 and raw != _UNSET32:
            value = struct.unpack("<f", raw)[0]
            if not math.isnan(value):
                setattr(tuning, name, _short_f32(value))
    if len(payload) > _LOCK_ZERO and payload[_LOCK_ZERO] != 0xFF:
        tuning.lock_zero_hz = payload[_LOCK_ZERO]
    return tuning, problem


def decode_payload(board: int, payload: bytes) -> TecTuning:
    """Decode a payload as the firmware reads it: fields past its end are unset."""
    return _decode(board, payload)[0]


def build_record(tuning: TecTuning, seq: int = 0) -> bytes:
    """Build a 128-byte record. Firmware re-stamps board, seq and CRC on store."""
    payload = encode_payload(tuning)
    header12 = MAGIC + struct.pack("<BBHI", LAYOUT_VERSION, tuning.board, len(payload), seq)
    crc = crc32_mpeg2(header12 + payload)
    record = header12 + struct.pack("<I", crc) + payload
    return record + b"\xFF" * (RECORD_SIZE - len(record))


def parse_record(raw: bytes) -> Optional[StoredTuning]:
    """Decode a stored record. Returns None for an erased (all 0xFF) record."""
    if len(raw) != RECORD_SIZE:
        raise TuningError(f"record is {len(raw)} bytes; expected {RECORD_SIZE}")
    raw = bytes(raw)
    if raw == b"\xFF" * RECORD_SIZE:
        return None
    if raw[:4] != MAGIC:
        raise TuningError("bad record magic")
    version, board, length, seq, crc = struct.unpack_from("<BBHII", raw, 4)
    if version != LAYOUT_VERSION:
        raise TuningError(f"unsupported layout version {version}")
    if length > PAYLOAD_MAX:
        raise TuningError(f"payload length {length} > {PAYLOAD_MAX}")
    payload = raw[HEADER_SIZE:HEADER_SIZE + length]
    if crc32_mpeg2(raw[:12] + payload) != crc:
        raise TuningError("record CRC mismatch")
    tuning, rejected = _decode(board, payload)
    if rejected is None:
        try:  # the TEC fields; the lock fields are checked as their own group
            validate(replace(tuning, **{name: None for name in LOCK_FIELDS}))
        except TuningError as exc:
            rejected = str(exc)
    return StoredTuning(board=board, seq=seq, tuning=tuning, raw=raw, rejected=rejected,
                        lock_rejected=_lock_problem(tuning))


def parse_reply(raw: bytes) -> TuningResult:
    """Decode the mainboard's 140-byte 'UTID' reply to 'n' or 'N' with kind 1."""
    raw = bytes(raw)
    op, target, status, slots_free, kind = _identity.reply_header(raw)
    if kind != KIND:
        raise TuningError("the mainboard answered with an identity reply; its "
                          "firmware does not support TEC tuning records")
    record = None
    if status == STATUS_OK:
        record = parse_record(raw[12:12 + RECORD_SIZE])
    elif status == STATUS_VERIFY_FAIL:
        try:  # best effort: show what the board actually holds
            record = parse_record(raw[12:12 + RECORD_SIZE])
        except TuningError:
            record = None
    expected = BOARD_FOR_TARGET.get(target)
    if record is not None and record.rejected is None and record.board != expected:
        record.rejected = f"board type {record.board} on target {target}"
    if (record is not None and record.lock_rejected is None and record.board != expected
            and any(getattr(record.tuning, name) is not None for name in LOCK_FIELDS)):
        record.lock_rejected = f"board type {record.board} on target {target}"
    return TuningResult(op=op, target=target, status=status,
                        slots_free=slots_free, record=record, raw=raw)


def same_record(a: TecTuning, b: TecTuning) -> bool:
    """True if two tunings encode to the same payload (float32 compare)."""
    return all(_payload_field(a, f.name) == _payload_field(b, f.name)
               for f in fields(TecTuning) if f.name != "board")


def _payload_field(tuning: TecTuning, name: str):
    value = getattr(tuning, name)
    return _f32(value) if name in _FLOATS + _LOCK_FLOATS and value is not None else value


def keep_stored(tuning: TecTuning, stored: Optional[StoredTuning], group: str) -> TecTuning:
    """Read-modify-write: *tuning* with its *group* fields taken from *stored*.

    *group* is ``"lock"`` (a TEC write keeps the stored lock PI fields) or
    ``"tec"`` (a lock write keeps the stored TEC fields). *stored* is the
    board's current record, or None if it has none. A stored group the
    firmware rejects is not kept: the firmware ignores it either way, so the
    board keeps running what it runs now.
    """
    names = {"tec": TEC_FIELDS, "lock": LOCK_FIELDS}[group]
    reason = None if stored is None else (
        stored.rejected if group == "tec" else stored.lock_rejected)
    source = stored.tuning if stored is not None and reason is None else None
    return replace(tuning, **{name: getattr(source, name, None) for name in names})
