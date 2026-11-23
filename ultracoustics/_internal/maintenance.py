"""
The maintenance link: what a firmware update and the self-test's device read need.

Classes:
    Programmer
        Firmware flashing interface for all three boards in the system.
        Sends .bin firmware over USB Bulk to the Master board, which either
        programs itself (IAP) or relays pages to the 1550 nm / 638 nm slave
        boards over SPI.  Provides flash_master(), flash_1550(), flash_638().

    DiagnosticsManager
        Firmware version queries and override-mode control for the Master
        Board and its slave laser boards.  Provides query_firmware_info(),
        query_slave_version(), read_identity(), enter_override(),
        exit_override(), set_power(), set_trigger().

Nothing here writes a board's stored records or sets a laser's current.
"""

import struct
import time
import zlib
from datetime import datetime, timezone

from .comms import USBBulkConnection
from .protocol import (
    CMD_FLASH_1550, CMD_FLASH_638, CMD_IAP,
    CMD_OVERRIDE_ENTER, CMD_POWER, CMD_TRIGGER,
    CMD_BOOT_VERSION, CMD_BOOT_VERSION_638, CMD_FWINFO,
    CMD_IDENTITY_READ, BULK_OUT_EP,
    TARGET_1550, TARGET_638, TARGET_MAIN,
)
from . import identity as _identity


# A diagnostic reply can follow a 16,532-byte format-2 stream record on the
# same bulk endpoint.  Read a whole USB transfer so an old record is discarded
# as one unit instead of consuming it in 512-byte fragments.
DIAGNOSTIC_READ_BYTES = 64 * 1024


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

    Every ``flash_*`` call takes the *release* that names its image, and sends
    nothing unless Ultracoustics approved that release and it lists exactly these
    bytes (:func:`ultracoustics.updates.require_approved`).
    """

    SLAVE_PAGE_SIZE = 256
    SLAVE_MAX_FW = 128 * 1024 - 8  # 131 064 bytes

    MASTER_PAGE_SIZE = 512
    # With the identity store on the external SPI NOR, master IAP erases sectors
    # 0-7 and takes 64 KB. A master on the older 56 KB layout (identity in
    # sector 7) refuses a larger image at IAP BEGIN, before anything is erased.
    MASTER_MAX_FW = 64 * 1024

    # ---- Master IAP magic header (must match the firmware)
    MASTER_MAGIC_BYTES         = b"UCBT"   # FW_MAGIC_BYTES
    MASTER_MAGIC_HEADER_VER    = 1         # FW_MAGIC_HEADER_VERSION
    #: Master builds by IAP target id. The visible-PD build runs the separate
    #: 638 noise-test mainboard.
    MASTER_TARGET_IDS          = {0x0001: "system", 0x0002: "visible_pd"}
    MASTER_MAGIC_SCAN_BYTES    = 4096      # FW_MAGIC_SCAN_BYTES
    MASTER_MAGIC_STRUCT_SIZE   = 16        # sizeof(fw_magic_t)

    def __init__(self, connection: USBBulkConnection, verbose=False):
        self._conn = connection
        self.verbose = verbose

    # -- Approval -------------------------------------------------------------

    def _require_approved(self, board, firmware, release):
        """Raise :class:`ultracoustics.updates.FirmwareRefused`, before anything is
        sent, unless *release* is approved and lists *firmware* by SHA-256."""
        from ..updates import require_approved    # here: updates imports this module
        require_approved(board, bytes(firmware), release)

    # -- Magic header pre-flight --------------------------------------------

    @classmethod
    def validate_master_image(cls, firmware: bytes):
        """Verify a candidate master firmware contains the UCBT magic header.

        Mirrors the check in the firmware.
        Scans the first ``MASTER_MAGIC_SCAN_BYTES`` of the image on a
        4-byte alignment for the ``UCBT`` tag, then checks header version
        and target id.

        Returns
        -------
        str
            The image's build: ``"system"`` or ``"visible_pd"``.

        Raises
        ------
        ValueError
            If no valid magic header is present, or the header indicates
            a different target / unsupported schema version.
        """
        scan_limit = min(len(firmware), cls.MASTER_MAGIC_SCAN_BYTES)
        if scan_limit < cls.MASTER_MAGIC_STRUCT_SIZE:
            raise ValueError(
                "Firmware too small to contain a UCBT magic header — "
                "this does not look like a Boot firmware image."
            )

        for off in range(0, scan_limit - cls.MASTER_MAGIC_STRUCT_SIZE + 1, 4):
            if firmware[off:off + 4] != cls.MASTER_MAGIC_BYTES:
                continue
            header_ver, target_id, image_size, _reserved = struct.unpack_from(
                "<HHII", firmware, off + 4
            )
            if header_ver != cls.MASTER_MAGIC_HEADER_VER:
                raise ValueError(
                    f"Firmware magic header_version={header_ver} is not "
                    f"supported by this SDK (expected "
                    f"{cls.MASTER_MAGIC_HEADER_VER})."
                )
            if target_id not in cls.MASTER_TARGET_IDS:
                raise ValueError(
                    f"Firmware target_id=0x{target_id:04X} does not match "
                    f"this device (expected 0x0001 = master Boot for "
                    f"STM32H7RS, or 0x0002 = its visible-PD build)."
                )
            if image_size != 0 and image_size != len(firmware):
                raise ValueError(
                    f"Firmware magic image_size={image_size} disagrees with "
                    f"file size {len(firmware)}."
                )
            return cls.MASTER_TARGET_IDS[target_id]

        raise ValueError(
            "No 'UCBT' magic header found in the first "
            f"{cls.MASTER_MAGIC_SCAN_BYTES} bytes — this file is not a "
            "Boot firmware image built for this device. Refusing to flash."
        )

    # -- Board 2: 1550 nm Slave -----------------------------------------------

    def flash_1550(self, firmware: bytes, progress_callback=None, *, release=None):
        """Flash firmware to Board 2 (1550 nm) via SPI3 relay.

        Args:
            firmware: Raw .bin contents.
            progress_callback: Optional ``callback(bytes_sent, total_bytes)``.
            release: The approved release that lists this image; refused without it.
        """
        self._require_approved("1550", firmware, release)
        self._flash_slave(
            firmware, CMD_FLASH_1550, "1550nm", progress_callback
        )

    # -- Board 3: 638 nm Slave ------------------------------------------------

    def flash_638(self, firmware: bytes, progress_callback=None, *, release=None):
        """Flash firmware to Board 3 (638 nm) via SPI2 relay. *release* is the
        approved release that lists this image; refused without it."""
        self._require_approved("638", firmware, release)
        self._flash_slave(
            firmware, CMD_FLASH_638, "638nm", progress_callback
        )

    # -- Board 1: Master (IAP) ------------------------------------------------

    def _running_profile(self):
        """The connected master's build (``"system"``/``"visible_pd"``), or
        ``None`` if it does not answer."""
        try:
            return DiagnosticsManager(self._conn).query_firmware_info().get("profile")
        except Exception:
            return None

    def flash_master(self, firmware: bytes, progress_callback=None,
                     allow_profile_change=False, *, release=None):
        """Flash firmware to Board 1 (Master) via In-Application Programming.

        The device will reset after a successful commit; the USB connection
        will be lost.

        A visible-PD image goes only onto a board already running the
        visible-PD build (a system board's firmware refuses it too). A system
        image goes onto a visible-PD board only with *allow_profile_change*.

        Args:
            firmware: Raw .bin contents (≤ 64 KB).
            progress_callback: Optional ``callback(page_num, total_pages)``.
            allow_profile_change: Allow a system image onto a visible-PD board.
            release: The approved release that lists this image; refused without it.
        """
        self._require_approved("main", firmware, release)
        fw = bytearray(firmware)
        fw_size = len(fw)
        if fw_size == 0 or fw_size > self.MASTER_MAX_FW:
            raise ValueError(
                f"Firmware size {fw_size} out of range (1–{self.MASTER_MAX_FW})"
            )

        # Reject obviously wrong files (text files, slave images, images for
        # the wrong product, etc.) BEFORE we touch the device. The firmware
        # makes the same check.
        image_profile = self.validate_master_image(bytes(fw))
        running = self._running_profile()
        if image_profile == "visible_pd" and running != "visible_pd":
            raise ValueError(
                "This is the visible-PD master build and the connected board "
                + ("runs the system build" if running == "system"
                   else "did not report its build")
                + ". Install it on the visible-PD board over SWD first.")
        if (image_profile == "system" and running == "visible_pd"
                and not allow_profile_change):
            raise ValueError(
                "The connected board runs the visible-PD build. Pass "
                "allow_profile_change=True to put the system build on it.")

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

        # Phase 3: COMMIT.  Do not swallow a transport exception: a reset may
        # be expected after application, but without an application ACK the
        # caller must treat a failed/uncertain transfer as unknown state and
        # explicitly verify device identity before deciding what to do next.
        self._conn.send_command(CMD_IAP, wValue=3, timeout_ms=10000)

        if self.verbose:
            print("  COMMIT sent – device is resetting")

    # -- Internal: slave flash ------------------------------------------------

    def _flash_slave(self, firmware, cmd, label, progress_callback):
        fw_size = len(firmware)
        if fw_size == 0 or fw_size > self.SLAVE_MAX_FW:
            raise ValueError(
                f"Firmware size {fw_size} out of range (1–{self.SLAVE_MAX_FW})"
            )

        crc = _identity.crc32_mpeg2(firmware)
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
        return self._read_firmware_info(40, timeout_ms)

    def query_slave_version(self, target=TARGET_1550, timeout_ms=5000):
        """Query a slave board's bootloader + app firmware info (80 bytes).

        Returns a dict with boot and app version sub-dicts.
        """
        cmd = CMD_BOOT_VERSION if target == TARGET_1550 else CMD_BOOT_VERSION_638
        self._conn.send_command(cmd)
        return self._read_slave_version(timeout_ms)

    # -- Board identity ------------------------------------------------------

    def read_identity(self, target=TARGET_MAIN, timeout_ms=8000):
        """Read the identity record stored on one board.

        Parameters
        ----------
        target : int
            ``TARGET_MAIN`` (0), ``TARGET_1550`` or ``TARGET_638``. Slave
            reads power-cycle that board into its bootloader, like
            :meth:`query_slave_version`.

        Returns
        -------
        ultracoustics.identity.IdentityResult
            ``status`` is ``STATUS_OK`` with ``record`` set, or ``EMPTY``,
            ``NO_RESPONSE`` (no board / bootloader without identity support),
            etc. A missing reply raises ``TimeoutError``.
        """
        _check_identity_target(target)
        self._conn.send_command(CMD_IDENTITY_READ, wIndex=target)
        return self._read_identity_reply(_identity.OP_READ, target, timeout_ms)

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

        Example
        -------
        Manually cycle the 638 nm rail on then off while in override mode::

            from ultracoustics._internal.comms import USBBulkConnection
            from ultracoustics._internal.maintenance import DiagnosticsManager
            from ultracoustics._internal.protocol import TARGET_638

            conn = USBBulkConnection()
            diag = DiagnosticsManager(conn, verbose=True)

            diag.enter_override()              # take manual control of the rails
            diag.set_power(TARGET_638, True)   # 638 nm VREGEN ON
            time.sleep(1.0)                    # let the rail settle / observe
            diag.set_power(TARGET_638, False)  # 638 nm VREGEN OFF
            diag.exit_override()               # return to normal sequencing
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
        deadline = time.monotonic() + timeout_ms / 1000.0
        while time.monotonic() < deadline:
            remaining = max(1, int((deadline - time.monotonic()) * 1000))
            chunk = self._conn.receive(DIAGNOSTIC_READ_BYTES,
                                       timeout_ms=min(remaining, 500))
            if chunk is None or len(chunk) != expected_len:
                continue
            version_str = chunk[:32].rstrip(b"\x00\xff").decode("ascii", errors="replace")
            build_date, hw_rev, profile = struct.unpack_from("<IHH", chunk, 32)
            build_dt = datetime.fromtimestamp(build_date, tz=timezone.utc) if build_date else None
            return {
                "version": version_str,
                "build_date": build_dt,
                "hardware_rev": hw_rev,
                # Was padding (always 0), so older firmware reads as system.
                "build_profile": profile,
                "profile": {0: "system", 1: "visible_pd"}.get(
                    profile, f"unknown({profile})"),
            }
        raise TimeoutError("No firmware info response received.")

    def _read_identity_reply(self, op, target, timeout_ms):
        """Wait for the identity reply matching *op* and *target*."""
        return _identity.parse_reply(self._read_utid_reply(op, target, timeout_ms))

    def _read_utid_reply(self, op, target, timeout_ms):
        """Wait for the 140-byte UTID reply matching *op* and *target*.

        Returns the raw reply; identity and tuning replies share the format.
        """
        deadline = time.time() + timeout_ms / 1000.0
        while time.time() < deadline:
            remaining = max(100, int((deadline - time.time()) * 1000))
            chunk = self._conn.receive(512, timeout_ms=min(remaining, 500))
            if (chunk is None or len(chunk) != _identity.REPLY_SIZE
                    or bytes(chunk[:4]) != _identity.REPLY_MAGIC):
                continue
            reply_op, reply_target, _status, _free, _kind = _identity.reply_header(chunk)
            if reply_op == op and reply_target == target:
                return bytes(chunk)
        # The master answers every identity command it accepts, so silence
        # means it rejected this one (busy finishing another maintenance
        # command, or firmware without identity support). A rejection stalls
        # BULK OUT; clear it so the next command, on this or a later
        # connection, is not refused with PIPE.
        dev = getattr(self._conn, "dev", None)
        if dev is not None:
            try:
                dev.clear_halt(BULK_OUT_EP)
            except Exception:
                pass
        raise TimeoutError(
            "No identity response received; the master was busy or its "
            "firmware does not support board identity.")

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
        deadline = time.monotonic() + timeout_ms / 1000.0
        while time.monotonic() < deadline:
            remaining = max(1, int((deadline - time.monotonic()) * 1000))
            chunk = self._conn.receive(DIAGNOSTIC_READ_BYTES,
                                       timeout_ms=min(remaining, 500))
            if chunk is None or len(chunk) != 80:
                continue

            boot_ver = chunk[:32].rstrip(b"\x00\xff").decode("ascii", errors="replace")
            boot_date, boot_hw, _ = struct.unpack_from("<IHH", chunk, 32)
            app_ver = chunk[40:72].rstrip(b"\x00\xff").decode("ascii", errors="replace")
            app_date, app_hw, _ = struct.unpack_from("<IHH", chunk, 72)
            boot_dt = datetime.fromtimestamp(boot_date, tz=timezone.utc) if boot_date else None
            app_dt = datetime.fromtimestamp(app_date, tz=timezone.utc) if app_date else None

            return {
                "boot": {"version": boot_ver, "build_date": boot_dt, "hardware_rev": boot_hw},
                "app": {"version": app_ver, "build_date": app_dt, "hardware_rev": app_hw},
                }
        raise TimeoutError("No version response received.")


def _check_identity_target(target):
    if target not in _identity.BOARD_FOR_TARGET:
        raise ValueError(f"unknown identity target {target!r}; "
                         "use TARGET_MAIN, TARGET_1550 or TARGET_638")


# ---------------------------------------------------------------------------
# LaserSerialController – serial link for direct laser current control
# ---------------------------------------------------------------------------
