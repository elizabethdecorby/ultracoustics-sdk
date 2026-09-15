"""Bounded parser and shared sidecar for negotiated stream telemetry."""

from __future__ import annotations

from dataclasses import astuple, dataclass, replace
import struct
import time
import zlib
from multiprocessing.shared_memory import SharedMemory
from typing import Optional

import numpy as np
from .optical_diagnostics import (CachedOpticalPage, OpticalABBA,
                                  OpticalAcquisition, OpticalDiagnosticError,
                                  OpticalLive, parse_page)


LEGACY_RECORD_BYTES = 16_392
FORMAT1_RECORD_BYTES = 16_532
FORMAT1_TRAILER_BYTES = 140
FORMAT1_MAX_TRAILER_BYTES = 503
FORMAT1_MAGIC = b"UTL1"
FORMAT1_VERSION = 1
FORMAT1_HEADER_BYTES = 16
TLV_638 = 1
TLV_1550 = 2
TLV_OPTICAL_LIVE = 3
TLV_OPTICAL_ACQUISITION = 4
TLV_OPTICAL_ABBA = 5
TLV_INTEGRITY = 0x7FFF
OPTIONAL_TLV_BIT = 0x8000
CANONICAL_SNAPSHOT_BYTES = 52
STALE_AFTER_NS = 500_000_000
TELEMETRY_VALID = 1 << 0
TELEMETRY_FAULT = 1 << 1
TELEMETRY_STALE = 1 << 2
TELEMETRY_UNCALIBRATED = 1 << 3
TELEMETRY_BACKEND_DISABLED = 1 << 4
LINK_STALE = 1 << 0
LINK_OVERFLOW = 1 << 1
LINK_DECODE_ERROR = 1 << 2
LINK_IDENTITY_ERROR = 1 << 3
LINK_UNAVAILABLE = 1 << 15

_CANONICAL = struct.Struct("<HHIIiIIHHHHIIHhHbBHH")
assert _CANONICAL.size == CANONICAL_SNAPSHOT_BYTES


class TelemetryFormatError(ValueError):
    """A negotiated record violates the required bounded wire format."""


@dataclass(frozen=True)
class StreamCapabilities:
    capability_flags: int
    legacy_record_bytes: int
    format1_record_bytes: int
    max_trailer_bytes: int
    current_format: int
    stream_epoch: int

    @property
    def supports_format1(self) -> bool:
        return bool(self.capability_flags & 1)

    @property
    def supports_format2(self) -> bool:
        return bool(self.capability_flags & 4)


def parse_capabilities(raw) -> StreamCapabilities:
    if len(raw) != 24:
        raise TelemetryFormatError("capability response must be exactly 24 bytes")
    magic, schema, size, flags, legacy, format1, max_trailer, current, reserved, epoch = (
        struct.unpack("<4sHHIHHHBBI", raw)
    )
    if magic != b"UTCP" or schema != 1 or size != 24 or reserved != 0:
        raise TelemetryFormatError("invalid capability response header")
    if legacy != LEGACY_RECORD_BYTES or format1 != FORMAT1_RECORD_BYTES:
        raise TelemetryFormatError("firmware record lengths do not match SDK")
    if max_trailer != FORMAT1_MAX_TRAILER_BYTES or current not in (0, 1, 2):
        raise TelemetryFormatError("unsupported capability response values")
    return StreamCapabilities(flags, legacy, format1, max_trailer, current, epoch)


def parse_format_ack(raw, requested_format: int) -> int:
    if len(raw) != 16:
        raise TelemetryFormatError("format acknowledgement must be exactly 16 bytes")
    magic, schema, size, accepted, reserved, epoch = struct.unpack("<4sHHB3sI", raw)
    if (magic != b"UTFM" or schema != 1 or size != 16 or reserved != b"\0\0\0"
            or accepted != requested_format):
        raise TelemetryFormatError("invalid or mismatched format acknowledgement")
    return epoch


@dataclass(frozen=True)
class BoardTelemetry:
    schema: int
    board: int
    snapshot_sequence: int
    snapshot_tick_ms: int
    temperature_mdegc: int
    temperature_sample_tick_ms: int
    temperature_age_ms: int
    temperature_flags: int
    temperature_fault: int
    pd_raw_average: int
    pd_full_scale: int
    pd_sample_tick_ms: int
    pd_age_ms: int
    pd_reference_mv: int
    pd_offset_counts: int
    pd_calibration_version: int
    pd_polarity: int
    reserved: int
    pd_flags: int
    pd_fault: int

    @classmethod
    def from_bytes(cls, value: bytes) -> "BoardTelemetry":
        if len(value) != CANONICAL_SNAPSHOT_BYTES:
            raise TelemetryFormatError("canonical snapshot must be exactly 52 bytes")
        return cls(*_CANONICAL.unpack(value))

    @property
    def temperature_valid(self) -> bool:
        return bool(self.temperature_flags & TELEMETRY_VALID)

    @property
    def temperature_stale(self) -> bool:
        return bool(self.temperature_flags & TELEMETRY_STALE)

    @property
    def pd_valid(self) -> bool:
        return bool(self.pd_flags & TELEMETRY_VALID)

    @property
    def pd_stale(self) -> bool:
        return bool(self.pd_flags & TELEMETRY_STALE)

    @property
    def pd_uncalibrated(self) -> bool:
        return bool(self.pd_flags & TELEMETRY_UNCALIBRATED)

    @property
    def pd_backend_disabled(self) -> bool:
        return bool(self.pd_flags & TELEMETRY_BACKEND_DISABLED)


@dataclass(frozen=True)
class TelemetrySnapshot:
    stream_epoch: int
    record_sequence: int
    received_monotonic_ns: int
    board_638: Optional[BoardTelemetry]
    board_1550: Optional[BoardTelemetry]
    link_flags_638: int = 0
    link_flags_1550: int = 0
    wire_format: int = 1
    optical_live_638: Optional[CachedOpticalPage] = None
    optical_acquisition_638: Optional[CachedOpticalPage] = None
    optical_abba_638: Optional[CachedOpticalPage] = None
    optical_page_raw: Optional[bytes] = None

    @property
    def host_age_s(self) -> float:
        return max(0, time.monotonic_ns() - self.received_monotonic_ns) / 1e9

    @property
    def host_stale(self) -> bool:
        return time.monotonic_ns() - self.received_monotonic_ns > STALE_AFTER_NS

    @property
    def link_638_stale(self) -> bool:
        return bool(self.link_flags_638 & (LINK_STALE | LINK_UNAVAILABLE))

    @property
    def link_1550_stale(self) -> bool:
        return bool(self.link_flags_1550 & (LINK_STALE | LINK_UNAVAILABLE))


@dataclass(frozen=True)
class ParsedRecord:
    sequence: int
    drops_fw: int
    samples: np.ndarray
    telemetry: Optional[TelemetrySnapshot]


def _parse_board_tlv(payload: bytes, expected_board: int) -> tuple[BoardTelemetry, int]:
    if len(payload) != 54:
        raise TelemetryFormatError("required board TLV length must be 54")
    board = BoardTelemetry.from_bytes(payload[:52])
    if board.schema != 1 or board.board != expected_board or board.reserved != 0:
        raise TelemetryFormatError("board snapshot schema, identity, or reserved byte invalid")
    link_flags, = struct.unpack_from("<H", payload, 52)
    return board, link_flags


def parse_format1_trailer(trailer, *, include_optical=False):
    """Parse a current or bounded future format-1 trailer.

    Unknown optional TLVs are skipped. Unknown required TLVs fail. Current
    stream negotiation still requires the exact 140-byte trailer; this helper
    makes the forward-compatibility rule executable without weakening current
    record framing.
    """
    trailer = memoryview(trailer)
    if len(trailer) < FORMAT1_HEADER_BYTES or len(trailer) > FORMAT1_MAX_TRAILER_BYTES:
        raise TelemetryFormatError("format-1 trailer outside bounded length")
    magic, version, header_len, trailer_len, epoch, flags, tlv_count = struct.unpack_from(
        "<4sBBHIHH", trailer, 0
    )
    if magic != FORMAT1_MAGIC or version not in (FORMAT1_VERSION, 2):
        raise TelemetryFormatError("bad format-1 magic or version")
    if header_len != FORMAT1_HEADER_BYTES or trailer_len != len(trailer):
        raise TelemetryFormatError("bad format-1 declared length")
    if flags != 0 or tlv_count < 3 or tlv_count > 121:
        raise TelemetryFormatError("unsupported format-1 flags or TLV count")

    offset = header_len
    board_638 = None
    board_1550 = None
    link_flags_638 = 0
    link_flags_1550 = 0
    for index in range(tlv_count):
        if offset + 4 > trailer_len:
            raise TelemetryFormatError("truncated TLV header")
        tlv_type, length = struct.unpack_from("<HH", trailer, offset)
        start = offset
        offset += 4
        end = offset + length
        if end > trailer_len:
            raise TelemetryFormatError("TLV exceeds declared trailer")
        payload = bytes(trailer[offset:end])
        offset = end
        if tlv_type == TLV_INTEGRITY:
            if index != tlv_count - 1 or length != 4 or offset != trailer_len:
                raise TelemetryFormatError("integrity TLV must be final and exact")
            expected_crc, = struct.unpack("<I", payload)
            actual_crc = zlib.crc32(trailer[:start]) & 0xFFFFFFFF
            if expected_crc != actual_crc:
                raise TelemetryFormatError("format-1 trailer CRC mismatch")
        elif tlv_type == TLV_638:
            if board_638 is not None:
                raise TelemetryFormatError("duplicate 638 TLV")
            board_638, link_flags_638 = _parse_board_tlv(payload, 638)
        elif tlv_type == TLV_1550:
            if board_1550 is not None:
                raise TelemetryFormatError("duplicate 1550 TLV")
            board_1550, link_flags_1550 = _parse_board_tlv(payload, 1550)
        elif version == 2 and tlv_type in (TLV_OPTICAL_LIVE, TLV_OPTICAL_ACQUISITION, TLV_OPTICAL_ABBA):
            if len(payload) != 54:
                raise TelemetryFormatError('optical diagnostic TLV must be 54 bytes in format 2')
            try:
                page = parse_page(payload[:52])
            except OpticalDiagnosticError as exc:
                raise TelemetryFormatError(str(exc)) from exc
            expected = {TLV_OPTICAL_LIVE: OpticalLive,
                        TLV_OPTICAL_ACQUISITION: OpticalAcquisition,
                        TLV_OPTICAL_ABBA: OpticalABBA}[tlv_type]
            if not isinstance(page, expected):
                raise TelemetryFormatError('optical TLV type does not match page type')
            optical_raw = payload[:52]
            link_flags_638, = struct.unpack_from('<H', payload, 52)
        elif not (tlv_type & OPTIONAL_TLV_BIT):
            raise TelemetryFormatError(f"unknown required TLV {tlv_type:#x}")
    if offset != trailer_len or board_1550 is None or (board_638 is None and 'optical_raw' not in locals()):
        raise TelemetryFormatError("missing required board telemetry")
    result = (epoch, board_638, board_1550, link_flags_638, link_flags_1550)
    return result + (locals().get('optical_raw'), version) if include_optical else result


def parse_record(raw, expected_format: int, *, received_monotonic_ns: Optional[int] = None) -> ParsedRecord:
    """Parse one exact application record for the explicitly selected format.

    Format zero accepts only the legacy 16,392-byte record. Format one accepts
    only the negotiated 16,532-byte record and never exposes trailer bytes as
    ADC samples.
    """
    view = memoryview(raw)
    required = LEGACY_RECORD_BYTES if expected_format == 0 else FORMAT1_RECORD_BYTES
    if expected_format not in (0, 1, 2):
        raise TelemetryFormatError(f"unsupported selected stream format {expected_format}")
    if len(view) != required:
        raise TelemetryFormatError(
            f"format {expected_format} requires exactly {required} bytes, got {len(view)}"
        )
    sequence, drops_fw = struct.unpack_from("<II", view, 0)
    samples = np.frombuffer(view, dtype="<u2", offset=8, count=8192)
    if expected_format == 0:
        return ParsedRecord(sequence, drops_fw, samples, None)

    trailer = view[LEGACY_RECORD_BYTES:]
    epoch, board_638, board_1550, link638, link1550, optical_raw, trailer_version = parse_format1_trailer(
        trailer, include_optical=True)
    if trailer_version != expected_format:
        raise TelemetryFormatError(
            f'format {expected_format} requires trailer version {expected_format}')
    received = time.monotonic_ns() if received_monotonic_ns is None else received_monotonic_ns
    telemetry = TelemetrySnapshot(epoch, sequence, received, board_638,
                                  board_1550, link638, link1550,
                                  wire_format=expected_format,
                                  optical_page_raw=optical_raw)
    return ParsedRecord(sequence, drops_fw, samples, telemetry)


# Seqlock sidecar: generation, epoch/record, receive time, two canonical blobs,
# records published, parse errors, epoch changes. One subprocess writer.
SIDECAR_BYTES = 384
_OPTICAL_SLOT_OFFSETS = (168, 240, 312)


def initialise_sidecar(shm: SharedMemory) -> None:
    shm.buf[:SIDECAR_BYTES] = b"\x00" * SIDECAR_BYTES


class TelemetrySidecarWriter:
    def __init__(self, shm: SharedMemory):
        self._buf = shm.buf
        self._generation = 0
        self._last_epoch: Optional[int] = None
        self._boards = {}

    def _begin(self) -> None:
        self._generation += 2
        struct.pack_into("<Q", self._buf, 0, self._generation - 1)

    def _end(self) -> None:
        struct.pack_into("<Q", self._buf, 0, self._generation)

    def select_format(self, stream_format: int) -> None:
        self._begin()
        struct.pack_into("<II", self._buf, 152, stream_format, 0)
        self._end()

    def reset_to_legacy(self) -> None:
        self._begin()
        records, errors, changes = struct.unpack_from("<QQQ", self._buf, 128)
        struct.pack_into("<QQQ", self._buf, 128, records, errors, changes + 1)
        struct.pack_into("<II", self._buf, 152, 0, 0)
        self._last_epoch = None
        self._boards.clear()
        self._end()

    def publish(self, snapshot: TelemetrySnapshot) -> None:
        self._begin()
        if self._last_epoch is not None and self._last_epoch != snapshot.stream_epoch:
            self._boards.clear()
            for offset in _OPTICAL_SLOT_OFFSETS:
                self._buf[offset:offset + 72] = b'\0' * 72
        if snapshot.board_638 is not None:
            self._boards[638] = (snapshot.board_638, snapshot.link_flags_638)
        if snapshot.board_1550 is not None:
            self._boards[1550] = (snapshot.board_1550, snapshot.link_flags_1550)
        struct.pack_into("<IIQ", self._buf, 8, snapshot.stream_epoch,
                         snapshot.record_sequence, snapshot.received_monotonic_ns)
        if 638 in self._boards:
            self._buf[24:76] = _CANONICAL.pack(*astuple(self._boards[638][0]))
        if 1550 in self._boards:
            self._buf[76:128] = _CANONICAL.pack(*astuple(self._boards[1550][0]))
        if snapshot.optical_page_raw is not None:
            page = parse_page(snapshot.optical_page_raw)
            slot = (0 if isinstance(page, OpticalLive) else
                    1 if isinstance(page, OpticalAcquisition) else 2)
            offset = _OPTICAL_SLOT_OFFSETS[slot]
            struct.pack_into('<IIIQ', self._buf, offset, 1, snapshot.stream_epoch,
                             snapshot.record_sequence, snapshot.received_monotonic_ns)
            self._buf[offset + 20:offset + 72] = snapshot.optical_page_raw
        records, errors, changes = struct.unpack_from("<QQQ", self._buf, 128)
        if self._last_epoch is not None and self._last_epoch != snapshot.stream_epoch:
            changes += 1
        self._last_epoch = snapshot.stream_epoch
        struct.pack_into("<QQQ", self._buf, 128, records + 1, errors, changes)
        valid = int(638 in self._boards and 1550 in self._boards)
        struct.pack_into("<II", self._buf, 152, snapshot.wire_format, valid)
        struct.pack_into("<HH", self._buf, 160,
                         self._boards.get(638, (None, 0))[1],
                         self._boards.get(1550, (None, 0))[1])
        self._end()

    def record_error(self) -> None:
        self._begin()
        records, errors, changes = struct.unpack_from("<QQQ", self._buf, 128)
        struct.pack_into("<QQQ", self._buf, 128, records, errors + 1, changes)
        self._end()


def read_sidecar(shm: SharedMemory) -> tuple[Optional[TelemetrySnapshot], dict]:
    for _ in range(8):
        before, = struct.unpack_from("<Q", shm.buf, 0)
        if before & 1:
            continue
        data = bytes(shm.buf[:SIDECAR_BYTES])
        after, = struct.unpack_from("<Q", shm.buf, 0)
        if before == after and not (after & 1):
            break
    else:
        return None, {"coherent": False}
    records, errors, changes = struct.unpack_from("<QQQ", data, 128)
    active_format, valid = struct.unpack_from("<II", data, 152)
    stats = {"coherent": True, "records": records, "parse_errors": errors,
             "epoch_changes": changes, "active_format": active_format}
    if before == 0 or records == 0 or not valid:
        return None, stats
    epoch, sequence, received = struct.unpack_from("<IIQ", data, 8)
    link638, link1550 = struct.unpack_from("<HH", data, 160)
    optical = []
    for offset in _OPTICAL_SLOT_OFFSETS:
        slot_valid, slot_epoch, slot_sequence, slot_received = struct.unpack_from('<IIIQ', data, offset)
        if slot_valid and slot_epoch == epoch:
            optical.append(CachedOpticalPage(parse_page(data[offset + 20:offset + 72]),
                                             slot_epoch, slot_sequence, slot_received))
        else:
            optical.append(None)
    snapshot = TelemetrySnapshot(
        epoch, sequence, received,
        BoardTelemetry.from_bytes(data[24:76]),
        BoardTelemetry.from_bytes(data[76:128]),
        link638, link1550, active_format, *optical,
    )
    return snapshot, stats
