"""The self-test: what the box can tell about itself, closed with the probe in.

It is the GUI's Diagnostics > Self-check. In about six minutes it reads every
board's firmware and identity, checks the software and USB link, the detector
with the lasers off, both lasers' output and temperature control, the probe,
the lock, the data rate and the spectrum, and measures the lock loop. It changes
nothing that is stored, and leaves the box stopped.

Run it first when something seems wrong, and send its report to Ultracoustics.
"""
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import time
import zlib

#: The report's format. /2 records ``probe_serial`` and no probe name; the box's
#: number comes from the device step (:attr:`SelfTestReport.box_number`).
SCHEMA = "broadsonic-system-report/2"
#: The longest probe serial number a report takes
PROBE_SERIAL_MAX_CHARS = 24
#: (key, title, expected seconds) of each step, in the order they run: the
#: seconds weight a progress bar. 'device' runs first because the 638's
#: discarded wake-up query latches the master's violet x3 blink code, and the
#: START in 'lock' clears it.
STEPS = (
    ("device", "Firmware and identity", 35),
    ("checks", "Software and USB checks", 20),
    ("dark", "Detector with the lasers off", 2),
    ("li_1550", "1550 nm laser output (internal photodiode)", 20),
    ("li_638", "638 nm laser output (internal photodiode)", 20),
    ("tec_1550", "1550 nm TEC under laser load", 25),
    ("tec_638", "638 nm TEC under laser load", 25),
    ("probe", "Probe sweep", 40),
    ("lock", "Lock, data rate and spectrum", 50),
    ("lock_loop", "Lock loop measurement", 100),
    ("stop", "Stop and record", 3),
)
TITLES = {key: title for key, title, _ in STEPS}
#: The steps measured on the box, after the firmware read: all but 'device'
MEASURED = tuple(key for key, _title, _ in STEPS if key != "device")
#: A step needs a look when its status is one of these.
PROBLEM_STATUSES = ("attention", "error")
#: After the firmware read the box gets this long to come back on the USB, tried every RECONNECT_POLL_S.
RECONNECT_S = 30.0
RECONNECT_POLL_S = 1.0


@dataclass(frozen=True)
class SelfTestReport:
    """The self-test's result. ``data`` is the whole report, JSON-safe, as the GUI saves it."""
    data: dict

    @property
    def steps(self):
        """Each step as a dict: ``title``, ``status``, ``summary``, ``rows`` and ``data``."""
        return self.data.get("steps") or []

    @property
    def cancelled(self):
        """Whether ``cancel()`` stopped the self-test before it finished."""
        return bool(self.data.get("cancelled"))

    @property
    def problems(self):
        """(title, summary) of every step that needs a look or could not complete."""
        return [(step["title"], step["summary"]) for step in self.steps if step["status"] in PROBLEM_STATUSES]

    @property
    def ok(self):
        """Every step ran, and none needs a look."""
        return not self.cancelled and not self.problems

    @property
    def recommends_signal_adjustment(self):
        """The detector saturated at the stored 1550 nm power, in the probe sweep or while locked:
        ``ctrl.measure_signal_level()`` chooses a lower one."""
        for step in self.steps:
            if step["key"] == "probe" and (step.get("data") or {}).get("full_scale_from_dac") is not None:
                return True
            if step["key"] == "lock" and any(label == "Detector clipping" and ok is False
                                             for label, _value, ok in step["rows"]):
                return True
        return False

    @property
    def box_number(self):
        """The box's number as the report shows it: the serial in its main board's identity
        record, else the short form of its USB serial; None with no box. Reads any schema."""
        device = next((s for s in self.steps if s.get("key") == "device"), None)
        data = (device or {}).get("data") or {}
        record = ((data.get("identity") or {}).get("main") or {}).get("record") or {}
        serial = str((record.get("identity") or {}).get("serial") or "").strip()
        return serial or short_uid(self.data.get("box_serial") or data.get("usb_serial"))

    def text(self):
        """The report as plain text: each step's outcome and every reading, flagged ones marked."""
        probe = self.data.get("probe_serial")
        lines = [f"Self-test of box {self.box_number or 'unknown'}" + (f", probe {probe}" if probe else "")
                 + f", {self.data.get('created', '')}"]
        for section in [self.data.get("host")] + list(self.steps):
            if not section:
                continue
            lines += ["", f"{section['title']}: {section['status']}", f"  {section['summary']}".rstrip()]
            for label, value, ok in section["rows"]:
                mark = "  !! " if ok is False else "     "
                lines.append(f"{mark}{label}: {value}")
        return "\n".join(lines)

    def save(self, path):
        """Write the report as JSON at ``path``; FileExistsError rather than overwrite. Returns the path."""
        path = Path(path)
        with open(path, "x", encoding="utf-8") as file:
            json.dump(self.data, file, indent=2)
        return path


def probe_serial_text(value):
    """A probe serial number as the report records it: spaces trimmed, None when blank.
    ValueError past PROBE_SERIAL_MAX_CHARS."""
    text = " ".join(str(value or "").split())
    if len(text) > PROBE_SERIAL_MAX_CHARS:
        raise ValueError(f"the probe serial number is longer than {PROBE_SERIAL_MAX_CHARS} characters")
    return text or None


def short_uid(usb_serial):
    """The short form of a box's 24-character USB serial: its CRC-32 in eight hex
    digits, which names a box whose main board has no serial of its own. None for none."""
    uid = (usb_serial or "").strip().strip(".")
    if uid in ("", "N/A"):
        return None
    return f"{zlib.crc32(uid.upper().encode('ascii', 'replace')):08X}"


def run_self_test(controller, probe=None, progress=None, cancel=None, software=None, *, calibration=None,
                  qt=None, saved_gains=None, probe_serial=None):
    """Run the self-test on ``controller``'s box. Returns a :class:`SelfTestReport`.

    This is the GUI's Self-check too, step for step. Firmware and identity come
    first, over the maintenance link, which needs the USB to itself: the controller
    closes, with a confirmed STOP, and connects again (the box gets RECONNECT_S to
    come back). ``probe`` (ProbeParameters) scales the thermal fit, that of the
    calibration by default. ``progress(event)`` gets ``{'step', 'message',
    'fraction'}`` while a step runs and ``{'step', 'done'}`` after it; ``cancel()``
    returning True stops it after the current step. The box is left stopped.
    ``probe_serial`` is the probe's serial number, recorded as given (at most
    24 characters; ValueError beyond).

    For the report's computer section: ``software`` names what ran it (this SDK by
    default), ``calibration`` is the one in use (the controller's by default), ``qt``
    the Qt versions of a GUI, ``saved_gains`` the lock gains its computer keeps for
    the box. ``controller`` None gives the report of a computer with no box.
    """
    from . import __version__
    from .config import SAMPLE_RATE
    from ._internal.checks import system_report as sr
    from ._internal.checks.device_info import read_device_info

    say = progress or (lambda event: None)
    cancelled = cancel or (lambda: False)
    probe_serial = probe_serial_text(probe_serial)
    if controller is not None:
        if controller._recorders or controller._monitors:
            raise RuntimeError("the self-test stops the box: finish the recording or monitor first")
        controller._box._ensure_connected()
        if calibration is None:
            calibration = getattr(controller, "calibration", None)
    probe = probe if probe is not None else (calibration.probe if calibration is not None else None)
    serial = None if controller is None else (str(controller.device_serial or "").strip() or None)
    report = {"schema": SCHEMA, "created": datetime.now().astimezone().isoformat(timespec="seconds"),
              "box_serial": serial, "probe_serial": probe_serial, "cancelled": False, "steps": [],
              "host": sr.host_section(software or f"Ultracoustics SDK {__version__}", serial, probe, calibration,
                                      saved_gains, qt=qt)}
    if controller is None:
        report["steps"] = sr.not_run("Not run: the box was not connected.")
        return SelfTestReport(report)

    started = time.monotonic()
    say({"step": "device", "message": "Handing the USB to the maintenance link…"})
    try:
        try:
            controller.stop(timeout_s=sr.STOP_TIMEOUT_S)   # identity reads need IDLE
        finally:
            controller.close()
        info, error = read_device_info(progress=lambda message: say({"step": "device", "message": message})), None
    except Exception as exc:
        info, error = None, exc
    device = sr.device_step(info, error, time.monotonic() - started)
    report["steps"].append(device)
    say({"step": "device", "done": device})

    if not _reconnect(controller, cancelled):
        report["steps"] += sr.not_run("Not run: the box did not reconnect after the firmware read.",
                                      keys=MEASURED, status="error")
        return SelfTestReport(report)
    if cancelled():
        report["cancelled"] = True
        report["steps"] += sr.not_run("Not run: the self-test was cancelled.", keys=MEASURED, status="cancelled")
        return SelfTestReport(report)
    measured = sr.run_report(controller._box, probe=probe, progress=say, cancel=cancelled,
                             sample_rate=SAMPLE_RATE)
    report["steps"] += measured["steps"]
    report["cancelled"] = measured["cancelled"]
    return SelfTestReport(report)


def _reconnect(controller, cancelled):
    """Connect and stream again once the maintenance link has let go: True once streaming.
    A cancelled self-test tries once, and leaves the rest to the caller."""
    deadline = time.monotonic() + RECONNECT_S
    while True:
        try:
            controller.connect()   # the stream starts with it
            return True
        except Exception:
            left = deadline - time.monotonic()
            if cancelled() or left <= 0:
                return False
            time.sleep(min(RECONNECT_POLL_S, left))
