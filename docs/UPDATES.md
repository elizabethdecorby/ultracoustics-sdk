# Software and firmware updates

`ultracoustics.updates` compares this SDK and the connected readout box with
one small file on the website, `https://ultracoustics.ca/updates.json`, and
installs approved firmware. The Ultracoustics GUI runs the same code. Using the
box never needs a network. A check is one plain download: it sends the SDK's
name and version as its User-Agent and nothing about you or the box, and it
raises `UpdateCheckError` when offline.

**This release carries no approval key yet**, so `update_firmware()` refuses
every release with `FirmwareRefused`, before anything is written to the box.
`check_for_updates()` works.

```python
from ultracoustics import check_for_updates, update_firmware

def main():
    report = check_for_updates()        # close any Controller first
    print(report.describe())
    if report.firmware_updates:
        update_firmware(progress=print)

if __name__ == "__main__":              # required: update_firmware() starts the USB reader process
    main()
```

Both calls open the box themselves, over USB, with no `Controller`: close any
`Controller`, the GUI and any other script first. `update_firmware()` stops the
box. With no box connected, `check_for_updates()` still compares the SDK, and
its report says the firmware was not read.

Those two, `UpdateReport` and the errors `UpdateCheckError`, `FirmwareRefused`
and `FirmwareUpdateError` are also on `ultracoustics` itself. The steps they
are built from are in this module only, for a program that shows its own
progress, as the GUI does:

```python
from ultracoustics.updates import (BoardStatus, compare_firmware, fetch_manifest,
                                   install_firmware, read_firmware_versions)

manifest = fetch_manifest()                       # the file below
firmware, warnings = read_firmware_versions()     # what each board runs: close any Controller first
for board in compare_firmware(firmware, manifest):    # one BoardStatus per board
    print(board.board, board.state)
```

## The file

```json
{
  "schema": 1,
  "gui": {
    "macos":   {"version": "2.0.0"},
    "windows": {"version": "2.0.0"}
  },
  "sdk": {"version": "2.0.0"},
  "firmware": {
    "638": [
      {
        "version": "8135918",
        "from": ["a608f8c"],
        "images": {
          "638-profile-1": {
            "url": "https://.../ultracoustics-638-8135918-638-profile-1.bin",
            "sha256": "<64 hex>", "dac_cap": 52400
          }
        },
        "signature": "<base64>"
      },
      {"version": "a608f8c"}
    ]
  }
}
```

- `firmware` has the boards `main`, `1550` and `638`, each a list, newest first.
  `version` is the string the board reports, the bare commit hash.
- A board on the first entry is current and one on a later entry can update.
  Any other version, such as a development build, is left alone.
- `from` lists the versions a release installs over. Anything else is told to
  contact Ultracoustics.
- `images` is keyed by the build: `system` on the main board, `default` on the
  1550, and the name `read_profile_638` reports on the 638 (with `dac_cap`).
- `signature` is the approval: Ed25519 over `release_statement()`, which covers the
  board, version, `from`, and each image's build, SHA-256 and `dac_cap`. The download
  address is outside it, so moving a file needs no new approval.

## Installing

This release's `FIRMWARE_KEYS` is empty: it carries no approval key yet, so
`update_firmware()` and `install_firmware()` refuse every release. What follows
describes how they work once a release carries a key.

`update_firmware()` brings every board with an update to its newest release, the
laser boards first. It checks and downloads everything before the first write.
`ultracoustics.updates.install_firmware(board, image)` does one board.

An image goes on only if a release signed by a key in `FIRMWARE_KEYS` names it by
SHA-256, and the box runs a version under that release's `from`. The code that
writes the image checks the signature again itself, just before its first byte. A 638 must also
run the image's profile and DAC cap, so an update never changes the laser current
limit. Each board's update stops the box, turns the laser rails off around the
transfer and reads all three versions back. Those come from flash, so it then
starts the laser boards, lasers at zero, and reads the 638's profile and cap back.

**Keep the box powered and plugged in until it returns.** A transfer that fails
leaves the old firmware running. In the last step the board rewrites itself, and
a power cut then leaves it needing service, with no recovery over USB. A laser
board in that state can still report the new version.

On a computer with no network, download `ultracoustics-firmware.zip` from the
website on another computer, unzip it and carry the folder over:

```python
update_firmware(folder="firmware-update", progress=print)
```

A release goes on a box only with a signature from Ultracoustics.
