# Visible-PD mainboard build (638 noise test)

Status: **implemented, not bench-tested.** Planned 2026-10-19, built
2026-10-20 with every recommended decision except 5: the visible-PD board
has no LED, so there is no rose idle LED. The work is uncommitted, on a
`visible-pd` branch from `test-mode` in `Photodetector-PSSIandUSBHS` and
`ultracoustics-sdk`, and on `diagnostics-next` in
`Ultracoustics-Internal_Software`. The images were built from that tree:
the visible-PD build is 54,564 B, and the normal Release build is
byte-identical to `3fb7c27` apart from its version suffix and build date.
The SDK tests pass. The bench checks in section 10 are what is left.

## Summary

The 638 noise test runs on a second STM32H7RS mainboard with a visible Si
photodiode front end. That board has the 638 laser board on its FFC and
nothing else: no 1550 and no fans. The test-mode master firmware cannot run
there. It faults on the fans within 2 s, and its start sequence waits for a
1550. The SDK's manual session also powers and waits for both laser boards.

The plan:

- **Firmware:** one master source tree with a `VISIBLE_PD_BOARD` CMake option
  (preset `VisiblePD`). It turns off the fan check, never powers or starts a
  1550, refuses the boot sequence and the 1550 commands, and keeps every 638
  safety check. With the option OFF, the normal image is byte-identical.
- **Identity:** three marks, each one constant: the IAP target id, a
  firmware-info profile field and a stream capability bit. Unchanged
  system-build firmware already refuses the visible image over IAP.
- **SDK:** `begin_system_manual(boards=None)` opens a 638-only session when
  the master reports the visible-PD build. System tests refuse to run on
  that board. The test-mode protocol does not change.
- **App:** the 638 noise page requires the visible-PD build, every other test
  requires the system build, and the hardware scan shows which build a board
  runs.

| Release image (`.bin`, 57,344 B cap) | Size | Free |
|---|---|---|
| Normal, `3fb7c27` as committed | 55,580 B | 1,764 B |
| Normal, with this plan's patch (option OFF) | 55,580 B, byte-identical | 1,764 B |
| Visible-PD (`VisiblePD` preset) | **54,564 B** | **2,780 B** |

## The board and the test

- The user moves the 638's FFC from the system mainboard to the visible-PD
  mainboard. Both mainboards are powered off whenever the FFC moves (the
  user confirmed this). The system mainboard is then powered back up, into
  WARM (no USB), to run the box fans while the 638 is on the visible-PD
  board.
- Only the visible-PD board's USB goes to the PC. The SDK's
  `usb.core.find` opens the first 0483:5740 device, so only one mainboard
  may be connected at a time. This plan does not change that.
- The noise sweep is a manual session: override, 638 rail on with START low,
  and the 638 DAC stepped by confirmed SETs under a 1 s lease and the 2 s
  keepalive.
- Photodiode conversion is host-only. In
  `diagnostics_app/models/test_types.py` the 638 uses 0.3 A/W and 10 kΩ.
  The firmware needs no PD constants.

## 1. Selecting the build

- **CMake option** `VISIBLE_PD_BOARD` (default OFF) in `Boot/CMakeLists.txt`.
  It adds the compile definition `VISIBLE_PD_BOARD`, the same way
  `DEV_DIAGNOSTICS` does.
- **Preset** `VisiblePD` in `Boot/CMakePresets.json`. It is Release plus
  `VISIBLE_PD_BOARD=ON` and builds into `Boot/build/VisiblePD`, so it never
  shares a build tree with `build/Release`. The output file names stay the
  same.
- **Makefile:** none needed. `make build BUILD_TYPE=VisiblePD` and
  `make flash BUILD_TYPE=VisiblePD` already work, because the wrapper passes
  `BUILD_TYPE` as the preset name. Two aliases, `make visible` and
  `make flash-visible`, are optional.
- **Build manifest:** `scripts/build_manifest.py` records
  `"visible_pd_board": true/false`, read from the CMake cache like
  `fast_adc_feedback_fir`.
- **Not measured:** `VISIBLE_PD_BOARD` together with `DEV_DIAGNOSTICS`.

## 2. How the build identifies itself

| Mark | Where | System build | Visible-PD build | Read by |
|---|---|---|---|---|
| IAP target id | `fw_magic.target_id` | `0x0001` | `0x0002` (`FW_TARGET_MASTER_VISIBLE_PD`) | master IAP (`iap_validate_magic`) and SDK `flash_master` pre-flight |
| Build profile | `firmware_info_t` bytes 38–39, the old `_reserved` (always 0), renamed `build_profile` | `0` | `1` (`FW_PROFILE_VISIBLE_PD`) | `i` reply: app hardware scan, `flash_master` |
| Capability bit 3 | UTCP `capability_flags` (`c` reply) | `7` | `15` (`USB_STREAM_CAP_VISIBLE_PD_BOARD = 8`) | SDK session logic: every `begin_system_manual` already reads it |

The prototype confirmed each mark in the built images: target `02 00` at
0x2B6, profile `01 00` at 0x2E6, and `movs r3, #15` in
`USBStreamFormat_PrepareCapabilities`.

Older SDKs ignore all three. `parse_capabilities` masks the bits it knows,
and `_read_firmware_info` discards the last two bytes.

### Keeping it off a system mainboard

- **IAP (USB):** blocked twice, both by code that already exists.
  - A system board's `iap_validate_magic` accepts only `0x0001`. It refuses
    the visible image at commit (status 4, LED violet ×5) and flashes
    nothing.
  - The SDK's `_validate_master_magic` refuses `0x0002` before it touches the
    device.
- **IAP onto a visible-PD board:** the visible build accepts `0x0002` and,
  under decision 2, `0x0001` too. A visible-PD board can then go back to
  the system build over USB, but a system board never takes the visible
  image.
- **SWD (ST-Link):** nothing can stop it in firmware. Mitigations:
  - the separate preset and build folder;
  - the SDK and app read the marks on every connection and refuse system
    tests (the board has no LED to show which build it runs).
- **If it lands on a system box anyway**, the failure is mostly safe:
  - the 1550 is never powered;
  - BOOT and WARM are refused;
  - only PC-supervised 638 override sessions run;
  - the fans are still driven (decision 4) but not checked.

## 3. Firmware changes (visible-PD build only)

All changes are `#ifdef VISIBLE_PD_BOARD`. The full prototype diff is in the
appendix.

| Function / file | Change |
|---|---|
| `Boot_CheckFanHealth` (`boot_sequencer.c`) | returns at once: no fan-stall fault in any state |
| `Boot_Hardware_SetPower` | forces `enable_1550 = false`. Every caller (WARM, A0, slave programmer, identity relay) leaves VREGEN1550 low. Fans stay coupled to the 638 rail |
| `Boot_Manual_Set1550Power`, `Boot_Manual_Set1550Trigger` | force `on = false`: VREGEN1550 and IRQ_1550 can only be written low |
| `Boot_TriggerStart` | no-op (covers the button's short press and USB `S`) |
| `Boot_TriggerWarm` | no-op (decision 3; covers the medium press and USB `W`) |
| `USB_Dispatcher_ProcessRequest` (`usb_command_dispatcher.c`) | the cases for `S`, `W`, `k`* (1550 startup DAC), `D`, `F`, `V` are compiled out, so they reach `default` and are refused. `P`/`T` 1550 **ON** and `n`/`N` for target 1550 return 1. `P`/`T` 1550 OFF are still accepted, as no-ops |
| `iap_validate_magic` (`iap_receiver.c`) | accepts `0x0002` and `0x0001` (decision 2) |
| `firmware_info.c`, `firmware_info.h` | `target_id` and `build_profile` from the build |
| `usb_stream_format.c`, `.h` | capability flags 7 or 15 |

\* `CMD_STARTUP_1550`, the one-shot automatic startup DAC.

A refused command returns 1. The SDK sees that as a halted request
(`TransportUncertainError(halt_cleared=True)`) and gets an error straight
away, instead of waiting on a board that is not there.

### States

| State | Visible-PD build |
|---|---|
| IDLE | unchanged: rails off. A long press still forces IDLE |
| OVERRIDE | unchanged for the 638: `P`/`T` 638, manual channels, `K`. The 1550 rail and START stay low |
| FAULT | unchanged. It is reached through the keepalive (white ×4), IAP and 638 programming faults |
| WARM | unreachable (decision 3) |
| A0_POWER_1550, A2_WAIT_638, RUN | unreachable: `S` refused, short press does nothing. The code stays linked |
| PROGRAM_1550, FLASH_1550, VERSION_1550 | unreachable: `D`/`F`/`V` refused |
| PROGRAM_638, FLASH_638, VERSION_638, IAP, FWINFO, METRICS, TELEMETRY_DIAGNOSTIC, STREAM_CONTROL, RUNTIME_METRICS, IDENTITY | unchanged; identity target 1550 is refused |

### Checks

| Check | Visible-PD build |
|---|---|
| Fan stall (amber ×1) | **off** |
| 1550 never ready / dropped (blue ×1, ×3) in A0, A2, RUN | unreachable, since there is no boot sequence |
| 1550 startup DAC (blue ×4) | unreachable (`k` refused) |
| Override 1550 drop and probe signal (blue ×3, red ×3, red ×1) | inert. `Boot_CheckOverrideFaults` is unchanged, but its 1550 and probe checks arm only while VREGEN1550 and IRQ_1550 are both high, which this build never drives |
| RUN signal loss, 638 drop warning | unreachable (RUN only) |
| Host keepalive `K` (white ×4, rails off) | **unchanged** |

### What keeps protecting the 638 (all unchanged)

- The keepalive: the SDK arms 2 s. On expiry the master faults and drops the
  rails.
- `MasterManual_Allowed(638)`: manual commands are accepted only in OVERRIDE,
  with the 638 rail on and its START low.
- The 638's own 1 s lease (DAC to 0), its temperature limits and its DAC
  clamp at 52,400.
- Confirmed STOP to IDLE on every SDK exit path (rails off).
- `MasterSPI2_NormalizeIf638Off` and the 638 telemetry/stream paths.

No fault code, LED fault code or USB record changes. The box-fan check moves
entirely to the system mainboard (see "Remaining risks").

## 4. What does not change

- **Test-mode protocol:**
  - `K` keepalive, `O` override, and `P`/`T` for the 638;
  - `j` manual control (TAKE/SET/RENEW/RELEASE with leases) and the 32-byte
    `UTMC` reply with its stream position;
  - stream formats 0/1/2 and all telemetry records. The 1550 TLV is still
    sent, marked unavailable, as with the 1550 unplugged;
  - `r` runtime metrics and the fault table (codes 0–17).
- **Board identity:** sector 7 at `0x0800E000`, the 128-byte record, 64
  slots, and `n`/`N` for targets 0 and 638 behave as in BOARD_IDENTITY.md.
  The visible board keeps its own main-board record. Only target 1550 is
  refused.
- **Image cap:**
  - `stm32h7r3xx_flash.ld` is untouched (FLASH = 56 KB, ID_STORE = sector
    7), and IAP still erases sectors 0–6 only.
  - The visible ELF's flash image ends at `0x0800D524`, so an SWD write (`STM32_Programmer_CLI -w … .elf`) also
    leaves sector 7 alone.
- **Normal build:** byte-identical to `3fb7c27` apart from the 4-byte
  `build_date`. Two rebuilds of the patched tree with the option OFF
  differed from the unpatched build only at 0x2E0–0x2E1.

## 5. SDK changes (`ultracoustics-sdk`)

### 638-only manual session

`begin_system_manual(timeout_s=1.0, optical_diagnostics=False,
keepalive_ms=KEEPALIVE_MS, boards=None)`:

- **`boards=None`:** after `enable_telemetry` or
  `enable_optical_diagnostics`, both of which already read the UTCP
  capabilities in IDLE, the session uses `(638,)` if bit 3 is set and
  `(638, 1550)` otherwise.
- **Explicit `boards`:** `(638,)` is allowed on either board. `(638, 1550)`
  on a visible-PD board raises before override: "this master runs the
  visible-PD build; it has no 1550".
- **Sequence:** `O`, then `T`=0 for **both** targets (accepted by both
  builds), then `P`=1, the fresh-telemetry wait and the zero-DAC TAKE for the
  session's boards only, then `K`. The timeout message names the boards it
  waited for.
- **State:** `self._session_boards` (cleared by `stop_system_confirmed`).
  - `system_manual_command` refuses a target outside it.
  - `system_manual_readback` iterates it, still skipping a 1550 running
    automatically.
  - `take_over_638` sets it to `(638, 1550)` with `_auto_1550` as today.
- Leases, renewals, `K`, `finish_system_manual` and every wire format stay
  as they are.

### Knowing the board

- `Controller.telemetry_capabilities()` caches the last capabilities
  response.
- New read-only properties:
  - `Controller.visible_pd_board`: `True`/`False`, or `None` until a
    capability response has been read;
  - `Controller.laser_boards`: `(638,)` or `(638, 1550)`.
- `_require_system_board(what)` raises "*what* needs the system mainboard;
  this master runs the visible-PD build". If the profile is unknown and the
  master is IDLE, it queries `c` once first.

### Refusing system tests on the visible-PD board

| Entry point | Refusal |
|---|---|
| `start_confirmed` | after its confirmed STOP, before `S`. The firmware also refuses `S` |
| `start` (plain `S`) | from the cached profile only; otherwise the firmware's refusal surfaces |
| `begin_manual(1550)` | after its IDLE check, before override |
| `take_over_638`, `start_1550_automatic`, `run_probe_sweep`, `capture_fp_scan` | up front, before any command (`run_probe_sweep` before it opens a session) |
| `DiagnosticsManager.read_system_info` | skips `fw_1550` and `identity_1550` when `i` reports profile 1, and records "no 1550 on the visible-PD board" instead of a timeout |

### Firmware info and flashing

- `_read_firmware_info` returns `build_profile` (int) and `profile`
  (`"system"`, `"visible_pd"` or `"unknown(n)"`). Old firmware reports 0, so
  it shows as system.
- `MASTER_TARGET_ID` becomes `MASTER_TARGET_IDS = {0x0001: "system",
  0x0002: "visible_pd"}`.
- `flash_master(firmware, progress_callback=None,
  allow_profile_change=False)` first reads the running board's profile. It
  refuses:
  - a visible image on a system board, always (the firmware would refuse it
    anyway);
  - a system image on a visible-PD board, unless
    `allow_profile_change=True`.

### Tests (`test_test_mode.py`, using the existing `Fake`)

- With capabilities 15: `boards=None` sends no `P` 1550, waits only for 638
  telemetry, takes only the 638 DAC, and readback has no 1550.
- With capabilities 7: unchanged sequence. Explicit `(638,)` powers only the
  638.
- `(638, 1550)` with capabilities 15 raises before `O`.
- `start_confirmed`, `take_over_638`, `start_1550_automatic`,
  `run_probe_sweep` and `begin_manual(1550)` raise with capabilities 15, and
  send nothing after the STOP.
- `system_manual_command(1550, …)` in a 638-only session raises.
- `parse_capabilities` keeps bit 3. `_read_firmware_info` parses profile 0
  and 1.
- `flash_master` target selection for all four combinations.

### Docs

- README test-mode section: the `boards` argument and the visible-PD board.
- TEST_MODE.md: one line pointing here.

### An unchanged SDK against the visible-PD build

It fails safely:

- `begin_system_manual` is refused at `P` 1550 ON and ends with its
  confirmed STOP.
- `start` and `start_confirmed` are refused at `S`.
- The scan reports 1550 query errors.
- Capability and firmware-info parsing are unaffected.

## 6. Diagnostics app changes (`Ultracoustics-Internal_Software`, `diagnostics-next`)

The `diagnostics-next` work, including the untracked `board_swap.py`, is
uncommitted in the main `Ultracoustics-Internal_Software` checkout. The
earlier worktree (`…\worktrees\Ultracoustics-Internal_Software-next`) no
longer exists. Paths below are relative to `diagnostics_app/`.

- **`views/runners/noise_runner.py`**
  - The 638 sweep starts only if the page's TO_VISIBLE re-detect scan
    reported `master_profile == "visible_pd"`. Otherwise the run button
    stays disabled with "Re-detect: this mainboard runs the system build".
    `validate()` gets the same check.
  - The 1550 sweep requires `"system"`.
- **`hardware/workers.py`, `NoiseSweepWorker`**
  - Opens `controller_session(..., board="visible_pd")` for 638 and
    `"system"` for 1550 (below). That checks the board actually connected,
    not only the last scan.
  - The result's `board_info` records the profile.
- **`hardware/laser_power.py`, `LaserPowerSession`:** `open()` keeps calling
  `begin_system_manual()`, which now opens a 638-only session on the
  visible-PD board. The stage text becomes "Powering the 638 board (laser
  held at zero)..." when `ctrl.laser_boards == (638,)`.
- **`hardware/device_manager.py`, `controller_session(...,
  board="system")`**
  - After `begin_stream()`, it compares `controller.visible_pd_board` with
    *board* (querying `c` if the master is IDLE). It raises
    `HardwareUnavailable` on a mismatch, so every other test page refuses the
    visible-PD board without per-page code.
  - If the master is not IDLE and the profile is unknown, it proceeds; the
    SDK refusals still apply.
- **`hardware/discovery.py`**
  - Stores `info.master_profile` from `query_firmware_info()`.
  - When it is `"visible_pd"`, it skips the 1550 version retries and the 1550
    identity read, which would otherwise burn the whole retry budget on
    refused commands.
  - `fmt_firmware_info` appends "visible-PD build".
  - `HardwareInfo` gains the field, and the report shows it next to the
    master UID.
- **`views/runners/board_swap.py`**
  - The TO_VISIBLE steps follow the confirmed procedure:
    1. unplug the system mainboard's USB;
    2. power off **both** mainboards;
    3. move the FFC;
    4. power the system mainboard back up (it starts in IDLE, white) and
       put it in WARM (blue);
    5. power the visible-PD board and plug in its USB;
    6. press Re-detect.

    Today the steps instead keep the system mainboard powered in IDLE
    during the move. TO_SYSTEM already powers off both boards.
  - The TO_VISIBLE note becomes "The visible-PD mainboard must run the
    visible-PD master build (VISIBLE_PD_BOARD.md)".
  - `_on_finished` shows ⚠ when the detected profile is wrong for the
    direction: TO_VISIBLE expects visible_pd, TO_SYSTEM expects system.
- **`hardware/flashing.py`:** shows the running board's profile and the
  image's target before an IAP flash. It passes `allow_profile_change` only
  from an explicit confirmation.

## 7. Sizes

| | `.bin` | Free of 57,344 | RAM / DTCM |
|---|---|---|---|
| Normal Release, `3fb7c27` | 55,580 B | 1,764 B | 67,912 / 8,712 B |
| Normal Release, patched (option OFF) | 55,580 B | 1,764 B | same |
| Visible-PD | 54,564 B | 2,780 B | same |

Built with Arm GNU 14.3.rel1 (arm-14.174), CMake 4.0.3 and Ninja, Release
(`-Os`), from `git archive 3fb7c27`. The only warning is the existing
linker RWX note.

Where the visible build is smaller (per-symbol, 1,022 B total; the image
shrinks by 1,016 B after alignment):

| Symbol | Δ bytes |
|---|---|
| `USB_Dispatcher_ProcessRequest` | −180 |
| `Startup1550_USBRequest` | −160 |
| `Boot_CheckFanHealth` | −102 |
| `SlaveProg_FlashDone`, `SlaveProg_FlashBegin`, `SlaveProg_FlashPage` (1550) | −96, −80, −64 |
| `Boot_TriggerStart`, `Boot_TriggerWarm` | −88, −52 |
| `MasterStartup1550_StageAllowed`, `SlaveProg_TriggerDAC`, `Startup1550_Consume`, `Boot_PollButton`, `SlaveProg_TriggerVersionCheck`, `BULK_ResetStreamSequence`, others | −52, −48, −28, −24, −24, −20, −8 |
| `IAP_Commit` (two accepted targets) | +4 |

A first prototype that rejected the commands with an up-front `if` in the
dispatcher came out at 55,660 B, 80 B **larger** than normal. Leaving the
cases out of the `switch` lets the linker drop the 1550 entry points.

## 8. Flashing the visible-PD build and checking a board

**First install: SWD (ST-Link).** The visible-PD board runs a system build
today, and that build refuses the `0x0002` image over IAP. On this PC CMake
finds `STM32_Programmer_CLI` (`…\Documents\stmstuff\bin`), so `make flash`
uses it (`-c port=SWD -w <elf> -v -rst`). With the ST-Link on the
visible-PD board only:

```bash
cd "/c/Users/Elizabeth D/Documents/GitHub/Photodetector-PSSIandUSBHS/Boot" && make flash BUILD_TYPE=VisiblePD
```

This configures and builds `build/VisiblePD` and writes the ELF. It never
touches `build/Release`. Do not use a full-chip erase: that also wipes
sector 7 and the board's identity record.

**Later updates:** IAP over USB once the SDK change is in
(`flash_master` or the app's flashing page), or SWD as above.

**Back to the system build:** IAP with `allow_profile_change=True` under
decision 2, otherwise SWD with `make flash`.

**What a board runs:**

The visible-PD board has no LED, so the build is read over USB:

1. **Any SDK, even today's:** with the board in IDLE and alone on USB,

   ```python
   from ultracoustics import Controller
   c = Controller(); c.connect(); c.begin_stream()
   print(c.telemetry_capabilities().capability_flags)  # 7 system, 15 visible-PD
   c.close()
   ```

2. **After the SDK change:**
   `DiagnosticsManager(conn).query_firmware_info()["profile"]`, or the app's
   hardware scan line, e.g. "FW master 3fb7c27 hw1 built 2026-10-19,
   visible-PD build".
3. **The build itself:** `build/VisiblePD/…build-manifest.json` has
   `"visible_pd_board": true`.

## 9. Remaining risks

- **Box fans during the 638 test.** Their only check is on the system
  mainboard in WARM. If a box fan stalls, that board faults and powers down
  everything it drives, **including the fans**. The visible-PD board does not
  know and keeps the 638 running. The 638's own temperature limits are then
  the only protection, for as long as the sweep lasts. Decision 7.
- **The visible build cannot tell it is in a system box.** The tach inputs
  are `GPIO_NOPULL` and float on a board without fans, so tach activity is
  no reliable sign. This plan does not try.
- **The fan-enable outputs are still driven** with the 638 rail
  (decision 4). This assumes they only switch the empty fan connectors on
  the visible-PD board; check the schematic.
- **Unchanged by design:** a HardFault or `Error_Handler` leaves the rails
  on (as in TEST_MODE.md), there is no hardware watchdog, and only one
  mainboard may be on the USB.

## 10. Implementation order

1. **Master firmware:** apply the appendix patch. Then:
   - build both presets;
   - check the sizes above, and that `Release` is byte-identical apart from
     `build_date`;
   - syntax-check with `arm-none-eabi-gcc -fsyntax-only`;
   - update `FAN_HEALTH.md` (no fan check in the visible build).
2. **SDK:** the session, refusals, profile and flashing changes, with tests.
   Run them with the Internal_Software venv
   (`-m unittest discover -p "test_*.py"`).
3. **App:** the changes in section 6.
4. **Bench, visible-PD board:**
   - flash by SWD; capabilities read 15 and `i` reports profile 1;
   - `S`, `W`, `k`, `D`, `F`, `V` and `P` 1550 ON are refused, and the short
     and medium presses do nothing;
   - a 638 noise sweep runs end to end and ends in IDLE;
   - after `begin_system_manual()`, sending nothing for 3 s ends the
     session: a confirmed STOP then reads `last_master_fault` "PC keepalive
     expired", and the 638 rail is off;
   - `n` for targets 0 and 638 works, and 1550 is refused;
   - IAP of the visible image onto a system board is refused, first by the
     SDK and, if forced, by the master (IAP status 4, nothing written).
5. **Bench, system board:** the TEST_MODE.md checklist still passes on the
   unchanged image.

## Decisions

| # | Question | Recommendation |
|---|---|---|
| 1 | Identity marks: IAP target id `0x0002`, firmware-info `build_profile`, capability bit 3 | all three; each is one constant and old tools ignore them |
| 2 | Visible build's IAP also accepts system images | yes: a visible-PD board can return to system duty over USB; system boards still refuse the visible image |
| 3 | WARM on the visible-PD board | off: the 638 is powered only inside a PC-supervised override session (WARM there would power the 638 with no keepalive and no fan check) |
| 4 | Fan-enable outputs on the visible build | keep driving them with the 638 rail: harmless with no fans, and cooling if the image is ever on a system box |
| 5 | Rose idle LED on the visible build (+8 B) | **dropped:** the visible-PD board has no LED |
| 6 | SDK `boards=None` auto-detects from the capability bit; the app requires `visible_pd` for the 638 noise test and `system` for everything else | yes |
| 7 | Box-fan stall during the 638 test is covered only by the 638's own temperature limits | accept, or have the app show a "watch the system board's LED" step |
| 8 | Where to implement | after the test-mode bench test: master and SDK on a `visible-pd` branch from `test-mode` (or on `test-mode`), the app on `diagnostics-next` |

## Appendix: prototype patch (against `3fb7c27`)

This is what was built and measured, apart from whitespace in the
capability macro. The `led.h` hunk was not applied (decision 5 dropped).

```diff
--- a/Boot/CMakeLists.txt
+++ b/Boot/CMakeLists.txt
@@ -39,6 +39,8 @@
 option(DEV_DIAGNOSTICS "Enable persistent stream-control diagnostics" OFF)
+option(VISIBLE_PD_BOARD
+    "Master for the visible-PD test mainboard: 638 only, no 1550, no fans" OFF)
@@ -121,6 +123,7 @@
     $<$<BOOL:${DEV_DIAGNOSTICS}>:DEV_DIAGNOSTICS>
+    $<$<BOOL:${VISIBLE_PD_BOARD}>:VISIBLE_PD_BOARD>
 )
--- a/Boot/CMakePresets.json
+++ b/Boot/CMakePresets.json
         {
             "name": "Release", ...
+        },
+        {
+            "name": "VisiblePD",
+            "inherits": "default",
+            "cacheVariables": {
+                "CMAKE_BUILD_TYPE": "Release",
+                "VISIBLE_PD_BOARD": "ON"
+            }
         }
     ],
     "buildPresets": [ ...
+        {
+            "name": "VisiblePD",
+            "configurePreset": "VisiblePD"
+        }
--- a/Boot/scripts/build_manifest.py
+++ b/Boot/scripts/build_manifest.py
-              "FAST_ADC_FEEDBACK_MEAN_SAMPLES", "FAST_ADC_FEEDBACK_FIR"}
+              "FAST_ADC_FEEDBACK_MEAN_SAMPLES", "FAST_ADC_FEEDBACK_FIR",
+              "VISIBLE_PD_BOARD"}
@@
         "fast_adc_feedback_fir": cache.get("FAST_ADC_FEEDBACK_FIR", "OFF") == "ON",
+        "visible_pd_board": cache.get("VISIBLE_PD_BOARD", "OFF") == "ON",
--- a/Boot/Core/Inc/firmware_info.h
+++ b/Boot/Core/Inc/firmware_info.h
     uint16_t hardware_rev;     /* PCB revision */
-    uint16_t _reserved;        /* Alignment padding */
+    uint16_t build_profile;    /* FW_PROFILE_*; was padding, always 0 */
 } firmware_info_t;
 
+/* Build profiles (ultracoustics-sdk docs/VISIBLE_PD_BOARD.md) */
+#define FW_PROFILE_SYSTEM      0U  /* system mainboard: 1550 + 638 + fans */
+#define FW_PROFILE_VISIBLE_PD  1U  /* visible-PD test mainboard: 638 only */
@@
-#define FW_TARGET_MASTER_BOOT_H7RS  0x0001U    /* this project, this build */
+#define FW_TARGET_MASTER_BOOT_H7RS  0x0001U    /* this project, system build */
+#define FW_TARGET_MASTER_VISIBLE_PD 0x0002U    /* this project, visible-PD build */
+
+#ifdef VISIBLE_PD_BOARD
+#define FW_TARGET_THIS_BUILD   FW_TARGET_MASTER_VISIBLE_PD
+#define FW_PROFILE_THIS_BUILD  FW_PROFILE_VISIBLE_PD
+#else
+#define FW_TARGET_THIS_BUILD   FW_TARGET_MASTER_BOOT_H7RS
+#define FW_PROFILE_THIS_BUILD  FW_PROFILE_SYSTEM
+#endif
--- a/Boot/Core/Src/firmware_info.c
+++ b/Boot/Core/Src/firmware_info.c
-    ._reserved    = 0,
+    .build_profile = FW_PROFILE_THIS_BUILD,
@@
-    .target_id      = FW_TARGET_MASTER_BOOT_H7RS,
+    .target_id      = FW_TARGET_THIS_BUILD,
--- a/Boot/Core/Src/iap_receiver.c
+++ b/Boot/Core/Src/iap_receiver.c
+#ifdef VISIBLE_PD_BOARD
+        /* A visible-PD board may also go back to the system build; a system
+         * board accepts only system images, so it never gets this one. */
+        if (m->target_id != FW_TARGET_MASTER_VISIBLE_PD &&
+            m->target_id != FW_TARGET_MASTER_BOOT_H7RS)
+#else
         if (m->target_id != FW_TARGET_MASTER_BOOT_H7RS)
+#endif
             return 3U;
--- a/Boot/Core/Inc/usb_stream_format.h
+++ b/Boot/Core/Inc/usb_stream_format.h
 #define USB_STREAM_COPY_SLOT_COUNT 7U
+/* UTCP capability bits: 0 format 1, 1 integrity, 2 optical format 2, and
+ * 3 visible-PD build (638 only: no 1550, no fans, no boot sequence). */
+#define USB_STREAM_CAPS_SYSTEM 7U
+#define USB_STREAM_CAP_VISIBLE_PD_BOARD 8U
+#ifdef VISIBLE_PD_BOARD
+#define USB_STREAM_CAPS_THIS_BUILD (USB_STREAM_CAPS_SYSTEM | USB_STREAM_CAP_VISIBLE_PD_BOARD)
+#else
+#define USB_STREAM_CAPS_THIS_BUILD USB_STREAM_CAPS_SYSTEM
+#endif
--- a/Boot/Core/Src/usb_stream_format.c
+++ b/Boot/Core/Src/usb_stream_format.c
-  capabilities.capability_flags = 7U; /* sensor, integrity, optical format 2 */
+  capabilities.capability_flags = USB_STREAM_CAPS_THIS_BUILD;
--- a/Boot/Core/Inc/led.h
+++ b/Boot/Core/Inc/led.h
+#ifdef VISIBLE_PD_BOARD
+#define LED_STATE_IDLE          255, 0,   128   /**< Rose   - powered down, visible-PD build */
+#else
 #define LED_STATE_IDLE          255, 255, 255   /**< White  - powered down */
+#endif
--- a/Boot/Core/Src/boot_sequencer.c
+++ b/Boot/Core/Src/boot_sequencer.c
 void Boot_Hardware_SetPower(bool enable_1550, bool enable_638) {
+#ifdef VISIBLE_PD_BOARD
+  enable_1550 = false; /* no 1550 on the visible-PD board */
+#endif
@@
 void Boot_TriggerStart(void) {
+#ifdef VISIBLE_PD_BOARD
+  return; /* no boot sequence without a 1550 (button and USB) */
+#endif
@@
 void Boot_TriggerWarm(void) {
+#ifdef VISIBLE_PD_BOARD
+  return; /* the 638 is powered only in a PC-supervised override */
+#endif
@@
 void Boot_CheckFanHealth(void) {
+#ifdef VISIBLE_PD_BOARD
+  return; /* no fans or tachs on the visible-PD board */
+#endif
@@
 void Boot_Manual_Set1550Power(bool on) {
+#ifdef VISIBLE_PD_BOARD
+  on = false;
+#endif
@@
 void Boot_Manual_Set1550Trigger(bool on) {
+#ifdef VISIBLE_PD_BOARD
+  on = false;
+#endif
--- a/USB_DEVICE_BULK/App/usb_command_dispatcher.c
+++ b/USB_DEVICE_BULK/App/usb_command_dispatcher.c
   switch (bRequest) {
+#ifndef VISIBLE_PD_BOARD
+  /* The visible-PD build has no 1550, fans or boot sequence: the commands in
+   * #ifndef VISIBLE_PD_BOARD blocks reach default and are refused. */
   case CMD_STARTUP_1550:
     return len == 0U && Startup1550_USBRequest(wValue, wIndex) ? 0U : 1U;
+#endif
@@
+#ifndef VISIBLE_PD_BOARD
   case CMD_BOOT:
     Boot_TriggerStart();
     break;
 
   case CMD_WARM:
     Boot_TriggerWarm();
     break;
+#endif
@@ case CMD_POWER: / case CMD_TRIGGER: (same in both)
     if (wIndex == TARGET_1550) {
+#ifdef VISIBLE_PD_BOARD
+      if (wValue == VAL_ON)
+        return 1U;
+#endif
@@
+#ifndef VISIBLE_PD_BOARD
   case CMD_PROGRAM_1550: ... break;
+#endif
@@
+#ifndef VISIBLE_PD_BOARD
   case CMD_FLASH_1550: ... return 1; // Invalid sub-command parameters
+#endif
@@
   case CMD_IDENTITY_READ:
   case CMD_IDENTITY_WRITE:
     /* Always answered with a 140-byte "UTID" reply (identity_store.h) */
+#ifdef VISIBLE_PD_BOARD
+    if (wIndex == TARGET_1550)
+      return 1U;
+#endif
@@
+#ifndef VISIBLE_PD_BOARD
   case CMD_VERSION:
     /* Query 1550nm firmware versions via SPI */
     SlaveProg_TriggerVersionCheck();
     break;
+#endif
```
