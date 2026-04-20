"""
SDK maintenance utilities: firmware flashing, diagnostics, override control,
and calibration placeholders.

Classes:
    Programmer
        Firmware flashing interface for all three boards in the system.
        Sends .bin firmware over USB Bulk to the Master board, which either
        programs itself (IAP) or relays pages to the 1550 nm / 638 nm slave
        boards over SPI.  Provides flash_master(), flash_1550(), flash_638().

    DiagnosticsManager
        Firmware version queries and override-mode control for the Master
        Board and its slave laser boards.  Provides query_firmware_info(),
        query_slave_version(), enter_override(), exit_override(),
        set_power(), set_trigger(), program_1550().

    LaserSerialController
        Serial link to a slave laser board for direct current control.
        Provides connect(), set_current(), disable(), close().

    Calibrator
        Placeholder for future photodetector / laser calibration logic
        (dark-current baseline, responsivity sweep, wavelength correction).
        All methods currently raise NotImplementedError.

Helper functions:
    _crc32_mpeg2(data)
        Computes a CRC-32 using the MPEG-2 polynomial (0x04C11DB7),
        matching the CRC implementation on the slave bootloaders.
"""

import struct
import time
import zlib
from datetime import datetime, timezone, timedelta

from .comms import USBBulkConnection, SerialConnection
from .protocol import (
    CMD_FLASH_1550, CMD_FLASH_638, CMD_IAP,
    CMD_OVERRIDE_ENTER, CMD_POWER, CMD_TRIGGER, CMD_PROGRAM_1550,
    CMD_BOOT_VERSION, CMD_BOOT_VERSION_638, CMD_FWINFO,
    TARGET_1550, TARGET_638,
)


# ---------------------------------------------------------------------------
# CRC-32 (Ethernet / MPEG-2 polynomial – matches slave bootloaders)
# ---------------------------------------------------------------------------

def _crc32_mpeg2(data: bytes) -> int:
    """Compute CRC-32 using the Ethernet/MPEG-2 polynomial (0x04C11DB7)."""
    crc = 0xFFFFFFFF
    for b in data:
        crc ^= b << 24
        for _ in range(8):
            if crc & 0x80000000:
                crc = ((crc << 1) ^ 0x04C11DB7) & 0xFFFFFFFF
            else:
                crc = (crc << 1) & 0xFFFFFFFF
    return crc


# ---------------------------------------------------------------------------
# Programmer – handles Board 1 (Master), Board 2 (1550), Board 3 (638)
# ---------------------------------------------------------------------------

class Programmer:
    """
    Firmware programming interface for all three boards.

    Requires an open :class:`comms.USBBulkConnection` to the Master Board.
    All operations go through USB Bulk – the Master relays pages to slave
    boards over SPI.

    Board mapping:
        * **Board 1 – Master** (STM32H7RS, internal flash, IAP over USB)
        * **Board 2 – 1550 nm slave** (relayed via SPI3, ``CMD_FLASH_1550``)
        * **Board 3 – 638 nm slave** (relayed via SPI2, ``CMD_FLASH_638``)
    """

    SLAVE_PAGE_SIZE = 256
    SLAVE_MAX_FW = 128 * 1024 - 8  # 131 064 bytes

    MASTER_PAGE_SIZE = 512
    MASTER_MAX_FW = 65536  # 64 KB

    def __init__(self, connection: USBBulkConnection, verbose=False):
        self._conn = connection
        self.verbose = verbose

    # -- Board 2: 1550 nm Slave -----------------------------------------------

    def flash_1550(self, firmware: bytes, progress_callback=None):
        """Flash firmware to Board 2 (1550 nm) via SPI3 relay.

        Args:
            firmware: Raw .bin contents.
            progress_callback: Optional ``callback(bytes_sent, total_bytes)``.
        """
        self._flash_slave(
            firmware, CMD_FLASH_1550, "1550nm", progress_callback
        )

    # -- Board 3: 638 nm Slave ------------------------------------------------

    def flash_638(self, firmware: bytes, progress_callback=None):
        """Flash firmware to Board 3 (638 nm) via SPI2 relay."""
        self._flash_slave(
            firmware, CMD_FLASH_638, "638nm", progress_callback
        )

    # -- Board 1: Master (IAP) ------------------------------------------------

    def flash_master(self, firmware: bytes, progress_callback=None):
        """Flash firmware to Board 1 (Master) via In-Application Programming.

        The device will reset after a successful commit; the USB connection
        will be lost.

        Args:
            firmware: Raw .bin contents (≤ 64 KB).
            progress_callback: Optional ``callback(page_num, total_pages)``.
        """
        fw = bytearray(firmware)
        fw_size = len(fw)
        if fw_size == 0 or fw_size > self.MASTER_MAX_FW:
            raise ValueError(
                f"Firmware size {fw_size} out of range (1–{self.MASTER_MAX_FW})"
            )

        # Pad to 16-byte alignment (quad-word flash programming unit)
        pad = (16 - (fw_size % 16)) % 16
        if pad:
            fw += b"\xFF" * pad
        fw_size_padded = len(fw)

        # CRC-32 (ISO 3309 / zlib)
        crc = zlib.crc32(bytes(fw)) & 0xFFFFFFFF
        total_pages = (fw_size_padded + self.MASTER_PAGE_SIZE - 1) // self.MASTER_PAGE_SIZE

        if self.verbose:
            print(f"IAP: {fw_size} bytes (padded {fw_size_padded}), "
                  f"CRC=0x{crc:08X}, {total_pages} pages")

        # Phase 0: IDLE (ensure clean state)
        try:
            from .protocol import CMD_IDLE
            self._conn.send_command(CMD_IDLE)
            time.sleep(0.3)
        except Exception:
            pass

        # Phase 1: BEGIN
        payload = struct.pack(">II", fw_size_padded, crc)
        self._conn.send_command(CMD_IAP, wValue=1, extra=payload, timeout_ms=5000)
        if self.verbose:
            print("  BEGIN sent")

        # Phase 2: PAGES
        for page_num in range(total_pages):
            offset = page_num * self.MASTER_PAGE_SIZE
            chunk = fw[offset : offset + self.MASTER_PAGE_SIZE]
            self._conn.send_command(CMD_IAP, wValue=2, wIndex=page_num,
                                    extra=bytes(chunk), timeout_ms=5000)
            if progress_callback:
                progress_callback(page_num, total_pages)

        # Phase 3: COMMIT
        try:
            self._conn.send_command(CMD_IAP, wValue=3, timeout_ms=10000)
        except Exception:
            pass  # device resets – USB disconnect is expected

        if self.verbose:
            print("  COMMIT sent – device is resetting")

    # -- Internal: slave flash ------------------------------------------------

    def _flash_slave(self, firmware, cmd, label, progress_callback):
        fw_size = len(firmware)
        if fw_size == 0 or fw_size > self.SLAVE_MAX_FW:
            raise ValueError(
                f"Firmware size {fw_size} out of range (1–{self.SLAVE_MAX_FW})"
            )

        crc = _crc32_mpeg2(firmware)
        total_pages = (fw_size + self.SLAVE_PAGE_SIZE - 1) // self.SLAVE_PAGE_SIZE

        if self.verbose:
            print(f"Flash {label}: {fw_size} bytes, {total_pages} pages, "
                  f"CRC=0x{crc:08X}")

        # BEGIN (wValue=1, data = 4-byte size big-endian)
        begin_data = struct.pack(">I", fw_size)
        self._conn.send_command(cmd, wValue=1, extra=begin_data, timeout_ms=35000)
        if self.verbose:
            print("  BEGIN sent")

        # PAGES (wValue=2)
        for page_num in range(total_pages):
            offset = page_num * self.SLAVE_PAGE_SIZE
            end = min(offset + self.SLAVE_PAGE_SIZE, fw_size)
            page = firmware[offset:end]
            if len(page) < self.SLAVE_PAGE_SIZE:
                page = page + b"\xFF" * (self.SLAVE_PAGE_SIZE - len(page))
            self._conn.send_command(cmd, wValue=2, wIndex=page_num,
                                    extra=page, timeout_ms=30000)
            if progress_callback:
                progress_callback(min(end, fw_size), fw_size)
            time.sleep(0.005)

        # DONE (wValue=3, data = 4-byte CRC big-endian)
        done_data = struct.pack(">I", crc)
        self._conn.send_command(cmd, wValue=3, extra=done_data, timeout_ms=15000)

        if self.verbose:
            print(f"  DONE sent (CRC=0x{crc:08X})")
            print(f"  {label} flash complete: {fw_size} bytes, {total_pages} pages")


# ---------------------------------------------------------------------------
# Calibrator – placeholder for future calibration workflows
# ---------------------------------------------------------------------------

class Calibrator:
    """
    Placeholder for photodetector / laser calibration logic.

    Future methods might include:

    * ``calibrate_dark()``        – measure dark-current baseline
    * ``calibrate_responsivity()`` – known-power sweep and polynomial fit
    * ``calibrate_wavelength()``  – wavelength-dependent correction table
    * ``save_calibration()``      – persist calibration to file
    * ``load_calibration()``      – restore calibration from file
    """

    def __init__(self, controller=None):
        self._ctrl = controller

    def calibrate_dark(self):
        """Measure dark-current baseline (not yet implemented)."""
        raise NotImplementedError("Dark calibration is not yet implemented.")

    def calibrate_responsivity(self):
        """Sweep known optical powers to derive responsivity curve."""
        raise NotImplementedError("Responsivity calibration is not yet implemented.")


# ---------------------------------------------------------------------------
# DiagnosticsManager – firmware queries, override mode, manual rail control
# ---------------------------------------------------------------------------

class DiagnosticsManager:
    """Firmware version queries and override-mode control.

    Requires an open :class:`USBBulkConnection` to the Master Board.
    Use this for diagnostic tasks such as reading firmware versions from
    the master or slave boards, and for manually controlling power rails
    and trigger lines in override mode.

    Usage::

        from ultracoustics._internal.comms import USBBulkConnection
        from ultracoustics._internal.maintenance import DiagnosticsManager

        conn = USBBulkConnection()
        diag = DiagnosticsManager(conn, verbose=True)
        print(diag.query_firmware_info())
        diag.close()
    """

    def __init__(self, connection: USBBulkConnection, verbose=False):
        """Initialise the diagnostics manager.

        Parameters
        ----------
        connection : USBBulkConnection
            An already-open USB bulk connection to the Master Board.
        verbose : bool, optional
            Print diagnostic messages. Defaults to ``False``.
        """
        self._conn = connection
        self.verbose = verbose

    # -- Firmware info queries ------------------------------------------------

    def query_firmware_info(self, timeout_ms=5000):
        """Query the Master Board firmware_info_t struct (40 bytes).

        Returns a dict with *version*, *build_date*, and *hardware_rev*.
        """
        self._conn.send_command(CMD_FWINFO)
        time.sleep(0.5)
        return self._read_firmware_info(40, timeout_ms)

    def query_slave_version(self, target=TARGET_1550, timeout_ms=5000):
        """Query a slave board's bootloader + app firmware info (80 bytes).

        Returns a dict with boot and app version sub-dicts.
        """
        cmd = CMD_BOOT_VERSION if target == TARGET_1550 else CMD_BOOT_VERSION_638
        self._conn.send_command(cmd)
        time.sleep(0.5)
        return self._read_slave_version(timeout_ms)

    # -- Override mode --------------------------------------------------------

    def enter_override(self):
        """Enter override mode, allowing manual rail and trigger control.

        In override mode the firmware stops its automatic sequencing and
        lets the host control VREGEN rails and IRQ triggers individually
        via :meth:`set_power` and :meth:`set_trigger`.
        """
        self._conn.send_command(CMD_OVERRIDE_ENTER, wValue=1)

    def exit_override(self):
        """Exit override mode and return to normal firmware sequencing."""
        self._conn.send_command(CMD_OVERRIDE_ENTER, wValue=0)

    def set_power(self, target, on: bool):
        """Enable or disable the VREGEN power rail for a laser target.

        Parameters
        ----------
        target : int
            ``TARGET_1550`` or ``TARGET_638``.
        on : bool
            ``True`` to enable, ``False`` to disable.
        """
        self._conn.send_command(CMD_POWER, wValue=int(on), wIndex=target)

    def set_trigger(self, target, on: bool):
        """Assert or de-assert the IRQ trigger line for a laser target.

        Parameters
        ----------
        target : int
            ``TARGET_1550`` or ``TARGET_638``.
        on : bool
            ``True`` to assert, ``False`` to de-assert.
        """
        self._conn.send_command(CMD_TRIGGER, wValue=int(on), wIndex=target)

    def program_1550(self, dac_val: int):
        """Write a raw DAC code to the 1550 nm slave laser board.

        Parameters
        ----------
        dac_val : int
            16-bit DAC value (0–65535) controlling the laser drive level.
        """
        self._conn.send_command(CMD_PROGRAM_1550, wValue=dac_val)

    # -- Internals ------------------------------------------------------------

    def _read_firmware_info(self, expected_len, timeout_ms):
        """Read and parse a ``firmware_info_t`` response from the Master Board.

        The struct is 40 bytes: 32-byte version string + uint32 build
        timestamp + uint16 hardware revision + uint16 padding.

        Returns
        -------
        dict
            Keys: ``version`` (str), ``build_date`` (datetime or None),
            ``hardware_rev`` (int), ``raw`` (bytes).

        Raises
        ------
        TimeoutError
            If no valid response arrives within *timeout_ms*.
        """
        MST = timezone(timedelta(hours=-7), "MST")
        deadline = time.time() + timeout_ms / 1000.0
        while time.time() < deadline:
            remaining = max(100, int((deadline - time.time()) * 1000))
            chunk = self._conn.receive(512, timeout_ms=min(remaining, 500))
            if chunk is None or len(chunk) != expected_len:
                continue
            version_str = chunk[:32].rstrip(b"\x00\xff").decode("ascii", errors="replace")
            build_date, hw_rev, _ = struct.unpack_from("<IHH", chunk, 32)
            build_dt = datetime.fromtimestamp(build_date, tz=MST) if build_date else None
            return {
                "version": version_str,
                "build_date": build_dt,
                "hardware_rev": hw_rev,
                "raw": chunk,
            }
        raise TimeoutError("No firmware info response received.")

    def _read_slave_version(self, timeout_ms):
        """Read and parse the 80-byte bootloader + app version response.

        The response contains two consecutive ``firmware_info_t`` structs
        (40 bytes each): the first for the bootloader, the second for the
        application firmware.

        Returns
        -------
        dict
            Keys: ``boot`` (dict), ``app`` (dict), ``raw`` (bytes).
            Each sub-dict has ``version``, ``build_date``, ``hardware_rev``.

        Raises
        ------
        TimeoutError
            If no valid 80-byte response arrives within *timeout_ms*.
        """
        MST = timezone(timedelta(hours=-7), "MST")
        deadline = time.time() + timeout_ms / 1000.0
        while time.time() < deadline:
            remaining = max(100, int((deadline - time.time()) * 1000))
            chunk = self._conn.receive(512, timeout_ms=min(remaining, 500))
            if chunk is None or len(chunk) != 80:
                continue

            boot_ver = chunk[:32].rstrip(b"\x00\xff").decode("ascii", errors="replace")
            boot_date, boot_hw, _ = struct.unpack_from("<IHH", chunk, 32)
            app_ver = chunk[40:72].rstrip(b"\x00\xff").decode("ascii", errors="replace")
            app_date, app_hw, _ = struct.unpack_from("<IHH", chunk, 72)
            boot_dt = datetime.fromtimestamp(boot_date, tz=MST) if boot_date else None
            app_dt = datetime.fromtimestamp(app_date, tz=MST) if app_date else None

            return {
                "boot": {"version": boot_ver, "build_date": boot_dt, "hardware_rev": boot_hw},
                "app": {"version": app_ver, "build_date": app_dt, "hardware_rev": app_hw},
                "raw": chunk,
            }
        raise TimeoutError("No version response received.")


# ---------------------------------------------------------------------------
# LaserSerialController – serial link for direct laser current control
# ---------------------------------------------------------------------------

class LaserSerialController:
    """Serial interface to a slave laser board for current control.

    Usage::

        from ultracoustics._internal.maintenance import LaserSerialController

        laser = LaserSerialController(port="COM3", verbose=True)
        laser.set_current(2048)
        laser.disable()
        laser.close()
    """

    def __init__(self, port=None, verbose=False):
        """Open the serial connection to a slave laser board.

        Parameters
        ----------
        port : str or None, optional
            Serial port name (e.g. ``'COM3'`` or ``'/dev/ttyUSB0'``).
            If ``None``, :class:`SerialConnection` will attempt
            auto-detection.
        verbose : bool, optional
            Print diagnostic messages. Defaults to ``False``.
        """
        self._serial = SerialConnection(port=port, verbose=verbose)
        self.verbose = verbose

    def set_current(self, dac_value: int):
        """Set the laser driver current.

        Parameters
        ----------
        dac_value : int
            Raw DAC value forwarded over the serial ``l=<value>`` command.
        """
        self._serial.send(f"l={int(dac_value)}")

    def disable(self):
        """Disable the laser driver by setting current to 0."""
        self._serial.send("l=0")

    def close(self):
        """Close the serial connection."""
        if self._serial:
            self._serial.close()
            self._serial = None
