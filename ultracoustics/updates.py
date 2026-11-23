"""Check for newer software and firmware, and install approved firmware.

What is current is listed in one small file, ``updates.json`` on
ultracoustics.ca (docs/UPDATES.md). Nothing here runs by itself and the box
never needs a network: a check is one plain download that sends only the SDK's
name and version, and raises :class:`UpdateCheckError` when offline, and :func:`update_firmware`
can work from a folder of files carried over by hand.

Images are public, so a file proves nothing by where it came from. A release
goes on a box only with a signature from a key in :data:`FIRMWARE_KEYS`, and that
key never leaves Ultracoustics.

Firmware versions are git hashes, which have no order, so each board's list in
the file is newest first. A box on the first entry is current, a box on a later
entry can update, and anything else (a development build) is left alone::

    from ultracoustics import check_for_updates
    print(check_for_updates().describe())
"""
import base64
import hashlib
import http.client
import json
import re
import ssl
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from . import __version__
from ._internal.comms import USBBulkConnection
from ._internal.device import Box
from ._internal.maintenance import DiagnosticsManager, Programmer
from ._internal.protocol import TARGET_1550, TARGET_638

MANIFEST_URL = "https://ultracoustics.ca/updates.json"
MANIFEST_SCHEMA = 1
MAX_MANIFEST_BYTES = 1 << 20
MAX_IMAGE_BYTES = 128 * 1024

#: The keys that may approve firmware: Ed25519 public keys, raw and base64. An
#: old key stays here for a while after it is replaced, for releases it approved.
FIRMWARE_KEYS = ()

BOARD_NAMES = {"main": "Main board", "1550": "1550 nm laser board",
               "638": "638 nm laser board"}

# BoardStatus.state
CURRENT = "current"      # on the newest published release
UPDATE = "update"        # on an older published release
UNLISTED = "unlisted"    # not a published release, e.g. a development build
UNREAD = "unread"        # the board's version could not be read
STATE_TEXT = {CURRENT: "up to date", UPDATE: "update available",
              UNLISTED: "not a published release", UNREAD: "not read"}

# The firmware read's timing, tuned on hardware: the master needs a gap after
# every slave query, and the 638 ignores the first query after it powers up.
DELAY_BEFORE_1550_S = 3.0
RETRY_1550_ATTEMPTS = 3
RETRY_GAP_S = 4.0
PRIMING_638_TIMEOUT_MS = 2000
MIN_DELAY_BEFORE_638_S = 3.0
READY_638_DEADLINE_S = 30.0
POLL_638_TIMEOUT_MS = 1500
POLL_638_GAP_S = 1.0

#: A whole-box update: the laser boards through the master's relay, then the
#: master, which restarts. The fleet alignment ran in this order.
UPDATE_ORDER = ("1550", "638", "main")

#: How long a master has to come back on USB after its update.
MASTER_RESET_DEADLINE_S = 20.0
#: After a controller lets go of the USB, and after every firmware read: a slave
#: query keeps the master busy for 2 s.
USB_SETTLE_S = 3.0

NOT_FINISHED = ("Power-cycle the box before trying again. If it still fails, "
                "contact Ultracoustics.")


class UpdateCheckError(Exception):
    """The update list could not be fetched or understood."""


class FirmwareUpdateError(Exception):
    """A firmware update did not finish, or could not be confirmed."""


class FirmwareRefused(FirmwareUpdateError):
    """An update was refused before anything was written: the box is unchanged."""


# -- The update list ----------------------------------------------------------

def _tls_context():
    """The system's certificates where possible: a packaged app has no others."""
    try:
        import truststore
    except ImportError:
        return ssl.create_default_context()
    return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


def _download(url, timeout_s, limit, client="ultracoustics-sdk"):
    # The site's firewall refuses Python's default User-Agent.
    request = urllib.request.Request(url, headers={"User-Agent": f"{client}/{__version__}"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s, context=_tls_context()) as reply:
            return reply.read(limit + 1)
    except (OSError, http.client.HTTPException) as exc:
        raise UpdateCheckError(f"Could not reach {url}: {exc}") from exc


def fetch_manifest(url=MANIFEST_URL, timeout_s=5.0, client="ultracoustics-sdk"):
    """Download and check the update list; raises :class:`UpdateCheckError`."""
    return parse_manifest(_download(url, timeout_s, MAX_MANIFEST_BYTES, client))


def parse_manifest(raw):
    """The update list as a dict. A list with anything wrong is refused whole."""
    if len(raw) > MAX_MANIFEST_BYTES:
        raise UpdateCheckError("The update list is too large.")
    try:
        manifest = json.loads(raw)
    except ValueError as exc:
        raise UpdateCheckError(f"The update list is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise UpdateCheckError("The update list has an unknown format.")
    firmware = manifest.get("firmware", {})
    if not isinstance(firmware, dict) or set(firmware) - set(BOARD_NAMES):
        raise UpdateCheckError("The update list names an unknown board.")
    for board, releases in firmware.items():
        if not isinstance(releases, list) or not all(map(_valid_release, releases)):
            raise UpdateCheckError(f"The update list's {board} firmware entry is malformed.")
    return manifest


def _valid_release(release):
    if not isinstance(release, dict) or not isinstance(release.get("version"), str):
        return False
    images = release.get("images", {})
    return (isinstance(images, dict) and isinstance(release.get("from", []), list)
            and all(isinstance(image, dict)
                    and re.fullmatch(r"[0-9a-f]{64}", str(image.get("sha256"))) is not None
                    for image in images.values()))


def release_statement(board, release):
    """The bytes an approval signs: what decides whether and where a release goes on.
    The download address is outside it, so a file can move."""
    images = {build: {"sha256": image["sha256"], "dac_cap": image.get("dac_cap")}
              for build, image in release.get("images", {}).items()}
    return json.dumps({"board": board, "version": release["version"],
                       "from": release.get("from", []), "images": images},
                      sort_keys=True, separators=(",", ":")).encode("ascii")


def verify_release(board, release):
    """Raise :class:`FirmwareUpdateError` unless a trusted key approved this release."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    refused = FirmwareRefused(
        f"{BOARD_NAMES[board]} {release['version']} is not approved by Ultracoustics.")
    try:
        signature = base64.b64decode(release["signature"], validate=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise refused from exc
    for key in FIRMWARE_KEYS:
        try:
            Ed25519PublicKey.from_public_bytes(base64.b64decode(key)).verify(
                signature, release_statement(board, release))
            return
        except InvalidSignature:
            continue
    raise refused


def require_approved(board, data, release):
    """Raise :class:`FirmwareRefused` unless *release* is approved for *board* and
    lists exactly these bytes. The last check before an image is written: the
    code that sends it calls this itself, so no caller can leave it out."""
    if board not in BOARD_NAMES:
        raise ValueError(f"unknown board {board!r}; use one of {tuple(BOARD_NAMES)}")
    if not _valid_release(release):
        raise FirmwareRefused(
            f"This {BOARD_NAMES[board]} image comes with no approved release. Firmware goes on "
            "a box through update_firmware() or install_firmware().")
    verify_release(board, release)
    digest = hashlib.sha256(data).hexdigest()
    if all(image["sha256"] != digest for image in release.get("images", {}).values()):
        raise FirmwareRefused(
            f"{BOARD_NAMES[board]} {release['version']} does not list this image (SHA-256 {digest}).")


def download_image(image, timeout_s=30.0):
    """The bytes of one listed image, checked against its SHA-256."""
    url = str(image.get("url", ""))
    if not url.startswith("https://"):
        raise FirmwareRefused("This image has no download address.")
    data = _download(url, timeout_s, MAX_IMAGE_BYTES)
    if hashlib.sha256(data).hexdigest() != image["sha256"]:
        raise FirmwareRefused(f"The download from {url} is damaged.")
    return data


def newer_version(latest, running):
    """True when dotted version *latest* is above *running*."""
    def parts(version):
        match = re.match(r"\d+(\.\d+)*", str(version))
        if match is None:
            return None
        numbers = [int(part) for part in match.group().split(".")]
        return numbers + [0] * (3 - len(numbers))
    latest, running = parts(latest), parts(running)
    return latest is not None and running is not None and latest > running


# -- Reading the box ----------------------------------------------------------

def read_firmware_versions(diag=None, progress=None, sleep=time.sleep, clock=time.monotonic):
    """Read every board's firmware version; takes about 10 s.

    Needs the USB device to itself: close any :class:`Controller` first, or
    pass the *diag* of a maintenance link you hold.

    Returns ``(firmware, warnings)``. ``firmware`` maps ``"main"`` to the
    master's firmware info and ``"1550"`` / ``"638"`` to ``{"boot", "app"}``;
    a query that failed leaves its board out and says why in ``warnings``.
    """
    if diag is None:
        connection = USBBulkConnection(verbose=False)
        try:
            return read_firmware_versions(DiagnosticsManager(connection, verbose=False),
                                          progress, sleep, clock)
        finally:
            connection.close()
    say = progress or (lambda message: None)
    firmware, warnings = {}, []
    say("Reading the main board firmware…")
    try:
        firmware["main"] = diag.query_firmware_info()
    except Exception as exc:
        warnings.append(f"Main board firmware: {exc}")
    # The visible-PD build has no 1550 and refuses its queries.
    if (firmware.get("main") or {}).get("profile") != "visible_pd":
        sleep(DELAY_BEFORE_1550_S)
        for attempt in range(1, RETRY_1550_ATTEMPTS + 1):
            say(f"Reading the 1550 nm board firmware (attempt {attempt})…")
            try:
                firmware["1550"] = diag.query_slave_version(TARGET_1550)
                break
            except Exception as exc:
                if attempt == RETRY_1550_ATTEMPTS:
                    warnings.append(f"1550 nm board firmware, {attempt} attempts: {exc}")
                else:
                    sleep(RETRY_GAP_S)

    say("Waking the 638 nm board (its first reply is discarded)…")
    try:
        diag.query_slave_version(TARGET_638, timeout_ms=PRIMING_638_TIMEOUT_MS)
    except Exception:
        pass
    sleep(MIN_DELAY_BEFORE_638_S)
    deadline = clock() + READY_638_DEADLINE_S
    attempt = 0
    while True:
        attempt += 1
        say(f"Reading the 638 nm board firmware (attempt {attempt})…")
        try:
            firmware["638"] = diag.query_slave_version(
                TARGET_638, timeout_ms=POLL_638_TIMEOUT_MS)
            break
        except Exception as exc:
            if clock() >= deadline:
                warnings.append(f"638 nm board firmware, {attempt} attempts: {exc}")
                break
            sleep(POLL_638_GAP_S)
    return firmware, warnings


def _application(board, firmware):
    """The running application's info for one board, or an empty dict."""
    info = firmware.get(board) or {}
    return info if board == "main" else (info.get("app") or {})


# -- Comparing ----------------------------------------------------------------

@dataclass(frozen=True)
class BoardStatus:
    """One board against the update list."""
    board: str
    running: Optional[str]
    latest: Optional[str]
    state: str
    #: There is an image, and this release installs directly over what runs now.
    installable: bool = False

    @property
    def name(self):
        return BOARD_NAMES[self.board]

    @property
    def update_available(self):
        return self.state == UPDATE

    def describe(self):
        return STATE_TEXT[self.state]


def firmware_summary(statuses):
    """One state for the whole box: what its user needs to know."""
    states = {status.state for status in statuses} or {UNREAD}
    return next(state for state in (UPDATE, UNREAD, UNLISTED, CURRENT) if state in states)


def compare_firmware(firmware, manifest):
    """Each board against the update list, as a tuple of :class:`BoardStatus`.

    Only the running application is compared. The bootloaders do not take
    updates over USB, and a visible-PD master is not a published product.
    """
    return tuple(_board_status(board, _application(board, firmware),
                               manifest.get("firmware", {}).get(board, []))
                 for board in BOARD_NAMES)


def _board_status(board, running, releases):
    version = running.get("version")
    latest = releases[0] if releases else {}
    newest = latest.get("version")
    if not version:
        return BoardStatus(board, None, newest, UNREAD)
    if running.get("profile", "system") != "system" or not releases:
        return BoardStatus(board, version, newest, UNLISTED)
    if version == newest:
        return BoardStatus(board, version, newest, CURRENT)
    older = any(version == release["version"] for release in releases[1:])
    installable = (older and bool(latest.get("images"))
                   and version in latest.get("from", []))
    return BoardStatus(board, version, newest, UPDATE if older else UNLISTED, installable)


@dataclass(frozen=True)
class UpdateReport:
    """This SDK and the connected box against the update list."""
    sdk_running: str
    sdk_latest: Optional[str]
    boards: Tuple[BoardStatus, ...]
    warnings: Tuple[str, ...] = ()

    @property
    def sdk_update(self):
        """Whether a newer SDK than this one is published."""
        return newer_version(self.sdk_latest, self.sdk_running)

    @property
    def firmware_updates(self):
        """The boards with a firmware update available, one :class:`BoardStatus` each."""
        return tuple(status for status in self.boards if status.update_available)

    def describe(self):
        """The report as a few lines of text, to print."""
        sdk = f"update available: {self.sdk_latest}" if self.sdk_update else "up to date"
        lines = [f"Ultracoustics SDK {self.sdk_running}: {sdk}",
                 f"Firmware: {STATE_TEXT[firmware_summary(self.boards)]}"]
        return "\n".join(lines + list(self.warnings))


def check_for_updates(url=MANIFEST_URL, timeout_s=5.0, firmware=None):
    """Compare this SDK and the connected box with the update list.

    Reads the box unless *firmware* (from :func:`read_firmware_versions`) is
    given, so close any :class:`Controller` first; with no box the firmware
    stays unread and ``warnings`` says why. Raises :class:`UpdateCheckError`
    when the list cannot be fetched.
    """
    manifest = fetch_manifest(url, timeout_s)
    warnings = []
    if firmware is None:
        try:
            firmware, warnings = read_firmware_versions()
        except Exception as exc:
            firmware, warnings = {}, [f"Firmware not read: {exc}"]
    return UpdateReport(__version__, manifest.get("sdk", {}).get("version"),
                        compare_firmware(firmware, manifest), tuple(warnings))


# -- Installing ---------------------------------------------------------------

def update_firmware(manifest=None, folder=None, progress=None,
                    sleep=time.sleep, clock=time.monotonic):
    """Bring every board with a published update to its newest release.

    Close any :class:`Controller` first. Everything is checked and downloaded
    before the first write, so a refusal known in advance leaves the box as it
    was. Boards then go on one at a time through :func:`install_firmware`, whose
    warning about power applies; a failure stops there, and running this again
    carries on from what is left.

    With no network, pass *folder*: the unzipped ``ultracoustics-firmware.zip``
    from the website, carried over from another computer. Its signatures are
    checked the same way.

    Returns one :func:`install_firmware` result per board, ``[]`` if nothing
    needed updating. Raises :class:`FirmwareUpdateError` if refused or not
    confirmed, :class:`FirmwareRefused` (a subclass) if nothing was written, and
    :class:`UpdateCheckError` if the list or an image cannot be had.
    """
    say = progress or (lambda message: None)
    if folder is not None:
        manifest = parse_manifest((Path(folder) / "updates.json").read_bytes())
    elif manifest is None:
        say("Fetching the update list…")
        manifest = fetch_manifest()

    say("Stopping the box…")
    laser = _stop_box(read_laser=True)[1]
    sleep(USB_SETTLE_S)
    firmware, warnings = read_firmware_versions(progress=say, sleep=sleep, clock=clock)
    statuses = {status.board: status for status in compare_firmware(firmware, manifest)}
    boards = [board for board in UPDATE_ORDER if statuses[board].update_available]
    images = {}
    for board in boards:
        release = manifest["firmware"][board][0]
        build = {"main": (firmware.get("main") or {}).get("profile"), "1550": "default",
                 "638": laser[0]}[board]
        image = release.get("images", {}).get(build)
        if image is None:
            raise FirmwareRefused(
                f"No {BOARD_NAMES[board]} image for this box ({build}). Contact Ultracoustics.")
        verify_release(board, release)
        _check_box(board, release, build, image, firmware, warnings, laser)
        say(f"Getting the {BOARD_NAMES[board]} firmware…")
        images[board] = (_read_image(folder, image) if folder is not None
                         else download_image(image))
    results = []
    for board in boards:
        sleep(USB_SETTLE_S)
        say(f"Updating the {BOARD_NAMES[board]}…")
        try:
            results.append(install_firmware(board, images[board], manifest, say, sleep, clock))
        except FirmwareRefused as exc:
            if results:     # an earlier board has changed
                raise FirmwareUpdateError(str(exc)) from exc
            raise
    return results


def _read_image(folder, image):
    """One image from a folder, by the file name in its download address."""
    path = Path(folder) / str(image.get("url", "")).rsplit("/", 1)[-1]
    if not path.is_file():
        raise FirmwareRefused(f"{path.name} is not in {folder}.")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != image["sha256"]:
        raise FirmwareRefused(f"{path.name} is damaged.")
    return data


def install_firmware(board, image, manifest=None, progress=None,
                     sleep=time.sleep, clock=time.monotonic):
    """Install one approved firmware image on one board of the connected box.

    *image* is the file's path or its bytes. It goes on only if the update list
    names it, by SHA-256, in a release signed by a key in :data:`FIRMWARE_KEYS`,
    so a wrong, damaged or unapproved file is refused before the box is touched
    (:class:`FirmwareRefused`).
    The box must run a version that release lists under ``from``. A 638 must also
    run this image's profile and DAC cap, so an update never changes the laser
    current limit.

    *manifest* is the update list; it is fetched if not given.

    It stops the box, turns the laser rails off around the transfer, reads every
    board's version back, then starts the laser boards with their lasers held at
    zero and reads the 638's profile and cap back, which is what shows the new
    firmware runs. Returns ``{"serial", "board", "from", "to", "firmware"}``, the
    last being every board as read back. Raises :class:`FirmwareUpdateError` if
    refused or not confirmed.

    **Keep the box powered and plugged in until this returns.** A transfer that
    fails leaves the old firmware running, but in the last step the board
    rewrites itself: a power cut then leaves it needing service, and a laser
    board may even report the new version. There is no recovery over USB.
    """
    if board not in BOARD_NAMES:
        raise ValueError(f"unknown board {board!r}; use one of {tuple(BOARD_NAMES)}")
    say = progress or (lambda message: None)
    data = image if isinstance(image, bytes) else Path(image).read_bytes()
    if manifest is None:
        manifest = fetch_manifest()
    release, build, listed = _find_image(manifest.get("firmware", {}).get(board, []), board, data)
    verify_release(board, release)
    target = release["version"]
    if target.encode("ascii") + b"\0" not in data:
        raise FirmwareRefused(f"The image does not carry its version {target}.")

    say("Stopping the box…")
    serial, laser = _stop_box(read_laser=True)
    sleep(USB_SETTLE_S)
    before, warnings = read_firmware_versions(progress=say, sleep=sleep, clock=clock)
    _check_box(board, release, build, listed, before, warnings, laser)
    sleep(USB_SETTLE_S)
    try:
        problem = _transfer(board, data, serial, say, sleep, clock, release)
        say("Confirming the new firmware…")
        after, warnings = read_firmware_versions(progress=say, sleep=sleep, clock=clock)
        sleep(USB_SETTLE_S)
        laser_after = _stop_box(read_laser=True)[1]
    except FirmwareUpdateError:
        raise
    except Exception as exc:
        raise FirmwareUpdateError(f"Not confirmed: {exc}. {NOT_FINISHED}") from exc
    finally:
        try:
            _stop_box()
        except Exception as exc:
            say(f"Could not confirm STOP after the update: {exc}")

    versions = {name: _application(name, after).get("version") for name in BOARD_NAMES}
    expected = {name: _application(name, before).get("version") for name in BOARD_NAMES}
    expected[board] = target
    if versions != expected or laser_after != laser:
        detail = f" The transfer reported: {problem}." if problem else ""
        raise FirmwareUpdateError(
            f"Not confirmed: expected {expected} and 638 {laser}, read {versions} and 638 "
            f"{laser_after}. {'; '.join(warnings)}{detail} {NOT_FINISHED}")
    return {"serial": serial, "board": board, "from": _application(board, before)["version"],
            "to": target, "firmware": after}


def _transfer(board, data, serial, say, sleep, clock, release):
    """Write the image. Returns any uncertain master error."""
    if board == "main":
        problem = _write_image(board, data, serial, say, release)
        say("Waiting for the box to restart…")
        _wait_for_box(serial, sleep, clock)
        sleep(1.0)
        return problem
    say("Turning the laser rails off…")
    _rails_off(serial)
    sleep(1.0)
    problem = _write_image(board, data, serial, say, release)
    sleep(2.0)
    _rails_off(serial)
    sleep(USB_SETTLE_S)
    return problem


def _find_image(releases, board, data):
    digest = hashlib.sha256(data).hexdigest()
    for release in releases:
        for profile, image in release.get("images", {}).items():
            if image["sha256"] == digest:
                return release, profile, image
    raise FirmwareRefused(
        f"This file is not a published {BOARD_NAMES[board]} image (SHA-256 {digest}).")


def _check_box(board, release, build, image, before, warnings, laser):
    """Refuse before anything is written. Every board must have been read, so
    the read-back afterwards can show the others did not change."""
    unread = [BOARD_NAMES[name] for name in BOARD_NAMES if not _application(name, before)]
    if unread:
        raise FirmwareRefused(f"Could not read {', '.join(unread)}. {'; '.join(warnings)}")
    running = _application(board, before)
    target = release["version"]
    if running["version"] == target:
        raise FirmwareRefused(f"{BOARD_NAMES[board]} is already on {target}.")
    if running["version"] not in release.get("from", []):
        raise FirmwareRefused(
            f"{BOARD_NAMES[board]} runs {running['version']}, which this image does not "
            "update directly. Contact Ultracoustics.")
    if board == "main" and running.get("profile") != build:
        raise FirmwareRefused(
            f"This image is the {build} build; the board runs {running.get('profile')}.")
    if board == "638":
        box_profile, cap = laser
        if box_profile != build or cap != image.get("dac_cap"):
            raise FirmwareRefused(
                f"This image is {build} with DAC cap {image.get('dac_cap')}; the box runs "
                f"{box_profile} with cap {cap}. An update never changes the laser current "
                "limit, so it is refused. Contact Ultracoustics.")


def _stop_box(read_laser=False):
    """Confirm STOP. Returns the USB serial and, on request, the 638's profile and
    cap, read in a manual session: the laser boards start, with their lasers at zero."""
    box = Box()
    box.connect()
    try:
        serial = box.device_serial
        box.stop_system_confirmed(timeout_s=4)
        if not read_laser:
            return serial, None
        capabilities = box.telemetry_capabilities()
        box.begin_system_manual(optical_diagnostics=capabilities.supports_format2)
        try:
            return serial, (box.read_profile_638()["name"],
                            box.read_optical_cap_638()["max_dac"])
        finally:
            box.finish_system_manual()
    finally:
        box.close()


def _open_box(serial):
    connection = USBBulkConnection(verbose=False)
    if connection.dev.serial_number != serial:
        connection.close()
        raise FirmwareUpdateError("A different box is connected.")
    return connection


def _rails_off(serial):
    connection = _open_box(serial)
    try:
        diag = DiagnosticsManager(connection, verbose=False)
        diag.enter_override()
        for wavelength in (638, 1550):
            diag.set_trigger(wavelength, False)
            diag.set_power(wavelength, False)
        diag.exit_override()
    finally:
        connection.close()


def _write_image(board, data, serial, say, release):
    """Send the image, with the approved *release* that lists it: the programmer
    checks both again before its first byte. For the master returns any transport
    error instead of raising: its reset can cut the last reply, so only the
    read-back can tell."""
    connection = _open_box(serial)
    try:
        programmer = Programmer(connection, verbose=False)
        if board == "main":
            def pages(page, total):
                say(f"Writing the main board: {100 * (page + 1) // total}%")
            try:
                programmer.flash_master(data, progress_callback=pages, release=release)
            except (ValueError, FirmwareRefused):
                raise
            except Exception as exc:
                return str(exc)
            return None

        def sent(written, total):
            say(f"Writing the {BOARD_NAMES[board]}: {100 * written // total}%")
        flash = programmer.flash_638 if board == "638" else programmer.flash_1550
        flash(data, progress_callback=sent, release=release)
        return None
    finally:
        connection.close()


def _wait_for_box(serial, sleep, clock):
    deadline = clock() + MASTER_RESET_DEADLINE_S
    while clock() < deadline:
        try:
            _open_box(serial).close()
            return
        except Exception:
            sleep(0.5)
    raise FirmwareUpdateError("The box did not come back on USB after the update.")
