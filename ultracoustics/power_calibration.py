"""Power-meter calibration records and bounded interpolation.

The raw acquisition rows are the source record.  A curve is usable only after
meter readings have been entered and validated; this module never fabricates
or extrapolates optical power.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Optional


FORMAT = "ultracoustics-power-calibration-v1"
REQUIRED_POINTS = 15


class PowerCalibrationError(ValueError):
    """The observations cannot define an unambiguous calibration curve."""


@dataclass
class CalibrationPoint:
    dac_requested: int
    dac_applied: int
    pd_raw_counts: int
    measured_mw: Optional[float] = None
    saturated: bool = False
    sample_tick_ms: Optional[int] = None
    captured_at_utc: Optional[str] = None
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_raw(cls, row, measured_mw=None):
        """Copy a sweep/sample row without discarding any source fields."""
        data = dict(row)
        return cls(
            dac_requested=int(data["dac_requested"]),
            dac_applied=int(data["dac_applied"]),
            pd_raw_counts=int(data["pd_raw_counts"]),
            measured_mw=measured_mw,
            saturated=bool(data.get("saturated", False)),
            sample_tick_ms=data.get("sample_tick_ms"),
            captured_at_utc=data.get("captured_at_utc"),
            raw=data,
        )

    def to_dict(self):
        return {
            "dac_requested": self.dac_requested,
            "dac_applied": self.dac_applied,
            "pd_raw_counts": self.pd_raw_counts,
            "measured_mw": self.measured_mw,
            "saturated": self.saturated,
            "sample_tick_ms": self.sample_tick_ms,
            "captured_at_utc": self.captured_at_utc,
            "raw": self.raw,
        }


@dataclass
class PowerCalibration:
    board_id: int
    board_serial: str
    points: list[CalibrationPoint]
    hardware_metadata: dict = field(default_factory=dict)
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if self.board_id not in (638, 1550):
            raise PowerCalibrationError("board_id must be 638 or 1550")
        if not isinstance(self.board_serial, str) or not self.board_serial.strip():
            raise PowerCalibrationError("board_serial is required")
        self.points = [p if isinstance(p, CalibrationPoint)
                       else CalibrationPoint(**p) for p in self.points]
        self.hardware_metadata = dict(self.hardware_metadata)

    @property
    def complete(self):
        try:
            self._curve()
        except PowerCalibrationError:
            return False
        return True

    def _curve(self):
        if len(self.points) != REQUIRED_POINTS:
            raise PowerCalibrationError(
                f"exactly {REQUIRED_POINTS} calibration points are required")
        curve = []
        for index, point in enumerate(self.points, 1):
            if point.saturated:
                raise PowerCalibrationError(f"point {index} is saturated")
            if point.measured_mw is None:
                raise PowerCalibrationError(f"point {index} has no meter reading")
            value = point.measured_mw
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(value) or value < 0):
                raise PowerCalibrationError(
                    f"point {index} measured_mw must be finite and nonnegative")
            if point.pd_raw_counts < 0:
                raise PowerCalibrationError(f"point {index} has negative PD counts")
            curve.append((point.pd_raw_counts, float(value)))
        curve.sort()
        collapsed = []
        for pd, mw in curve:
            if collapsed and collapsed[-1][0] == pd:
                if collapsed[-1][1] != mw:
                    raise PowerCalibrationError(
                        f"PD count {pd} has ambiguous meter readings")
                continue
            collapsed.append((pd, mw))
        for (pd0, mw0), (pd1, mw1) in zip(collapsed, collapsed[1:]):
            if pd0 == pd1:
                raise AssertionError("duplicate PD counts were not collapsed")
            if mw1 <= mw0:
                raise PowerCalibrationError(
                    "measured power must increase strictly with PD counts")
        if len(collapsed) < 2:
            raise PowerCalibrationError("at least two unique PD counts are required")
        return collapsed

    @property
    def supported_pd_range(self):
        curve = self._curve()
        return curve[0][0], curve[-1][0]

    def power_mw(self, pd_raw_counts):
        """Linearly interpolate inside the observed PD range only."""
        if (isinstance(pd_raw_counts, bool) or
                not isinstance(pd_raw_counts, (int, float)) or
                not math.isfinite(pd_raw_counts)):
            raise PowerCalibrationError("pd_raw_counts must be finite")
        curve = self._curve()
        if pd_raw_counts < curve[0][0] or pd_raw_counts > curve[-1][0]:
            raise PowerCalibrationError("PD counts are outside the calibrated range")
        for pd, mw in curve:
            if pd_raw_counts == pd:
                return mw
        for (pd0, mw0), (pd1, mw1) in zip(curve, curve[1:]):
            if pd0 < pd_raw_counts < pd1:
                fraction = (pd_raw_counts - pd0) / (pd1 - pd0)
                return mw0 + fraction * (mw1 - mw0)
        raise PowerCalibrationError("PD counts are outside the calibrated range")

    def to_dict(self):
        return {
            "format": FORMAT,
            "board": {"id": self.board_id, "serial": self.board_serial},
            "hardware_metadata": self.hardware_metadata,
            "created_at_utc": self.created_at_utc,
            "complete": self.complete,
            "points": [point.to_dict() for point in self.points],
        }

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n",
                              encoding="utf-8")

    @classmethod
    def from_dict(cls, data, expected_board_id=None, expected_serial=None,
                  expected_hardware_metadata=None):
        if data.get("format") != FORMAT:
            raise PowerCalibrationError("unsupported calibration format")
        board = data.get("board", {})
        if expected_board_id is not None and board.get("id") != expected_board_id:
            raise PowerCalibrationError("calibration belongs to a different board id")
        if expected_serial is not None and board.get("serial") != expected_serial:
            raise PowerCalibrationError("calibration belongs to a different board serial")
        metadata = data.get("hardware_metadata", {})
        for key, expected in (expected_hardware_metadata or {}).items():
            if metadata.get(key) != expected:
                raise PowerCalibrationError(
                    f"calibration hardware metadata mismatch for {key}")
        return cls(board_id=board.get("id"), board_serial=board.get("serial"),
                   hardware_metadata=metadata,
                   created_at_utc=data.get("created_at_utc", ""),
                   points=[CalibrationPoint(**point)
                           for point in data.get("points", [])])

    @classmethod
    def load(cls, path, expected_board_id=None, expected_serial=None,
             expected_hardware_metadata=None):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data, expected_board_id, expected_serial,
                             expected_hardware_metadata)
