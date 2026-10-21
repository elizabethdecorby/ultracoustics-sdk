# TEC tuning records

Each laser board can store a **tuning record**: TEC PID gains for the laser
fitted to that board. The 1550 and 638 firmware is built once per laser type.
At start-up the firmware applies the board's record over its compiled
defaults, so one image serves every unit.

The gains in use come from three layers, each overriding the one below:

1. Compiled defaults: `1550/App/Inc/app_config.h`, and the box profile
   (`638-Laser-Firmware/cmake/profiles/*.cmake`) for the 638.
2. The stored tuning record, for each field it sets.
3. Live changes over the laser board's own USB console (`TP=`, `TI=`, `TD=`,
   `pid`). These are lost at the next reset.

This file is the single source of truth for the record layout and the wire
protocol. The records share the identity records' header, storage scheme and
mainboard commands ([BOARD_IDENTITY.md](BOARD_IDENTITY.md)); only the
differences are repeated here. `1550_boot` stores the records, the `1550` and
`638-Laser-Firmware` apps read them, and `Photodetector-PSSIandUSBHS` relays
them. `ultracoustics/tec_tuning.py` implements the host side.

## Record (128 bytes, little-endian)

The 16-byte header is the identity header with its own magic:

| Offset | Size | Field | Set by |
|---|---|---|---|
| 0 | 4 | magic `"UCTN"` | host |
| 4 | 1 | layout version, `1` | host |
| 5 | 1 | board type: `2` 1550, `3` 638 | mainboard firmware (overwrites the host value) |
| 6 | 2 | payload length, 0–112 | host |
| 8 | 4 | sequence number | bootloader: previous + 1, first record = 1 |
| 12 | 4 | CRC-32/MPEG-2 over bytes 0–11 plus the first *payload length* payload bytes | bootloader |
| 16 | 112 | payload; unused bytes are `0xFF` | host |

### Payload (fixed layout)

| Payload offset | Size | Field | Type and unit | Not set |
|---|---|---|---|---|
| 0 | 4 | `kp` | float32, normalized output per °C | NaN |
| 4 | 4 | `ki` | float32, normalized output per °C·s | NaN |
| 8 | 4 | `kd` | float32, normalized output per °C/s | NaN |
| 12 | 4 | `integrator_limit` | float32, °C·s, applied as ±limit | NaN |
| 16 | 1 | `setpoint_filter` | u8: `0` off, `1` on | `0xFF` |
| 17 | 1 | `conditional_integration` | u8: `0` off, `1` on | `0xFF` |
| 18 | 2 | reserved | `0xFF 0xFF` | |
| 20 | 4 | `tuned_at`: unix seconds | u32, host only | `0xFFFFFFFF` |
| 24 | 24 | `report_id`: diagnostics session | UTF-8, padded with `0xFF`, host only | all `0xFF` |

Layout version 1 writers use a 48-byte payload. A field is set only if it
lies entirely inside the payload length and holds a value other than its
"not set" encoding. Erased flash (`0xFFFFFFFF`) is a NaN, so an unwritten
float reads as "not set". Readers ignore payload bytes past offset 48; later
fields are appended there without a layout version change (the 638 lock PI
fields, [below](#lock-pi-fields-638)).

A record that sets no gain or flag (payload length 0, or every field "not
set") returns the board to its compiled defaults. Slots are append-only, so
this is how the host clears a board.

`tuned_at` and `report_id` are for traceability. The firmware does not read
them.

## Validation and use (laser app firmware)

At start-up, `Thermal_Init` finds the newest valid record before the TEC is
enabled: correct magic, layout version 1, payload length ≤ 112 and a matching
CRC. It then rejects the whole record and runs **all** compiled defaults
(never a mix) if any of the following is true:

- byte 5 is not the app's own board type (`2` for the 1550 app, `3` for the
  638 app);
- a set float is infinite or outside the bounds below;
- a flag byte is not `0`, `1` or `0xFF`;
- the 1550 app sees a flag set to `1`. The 1550 firmware has neither
  pull-in feature, so it accepts only `0` or "not set" there.

| Field | Accepted range |
|---|---|
| `kp` | 0 ≤ kp ≤ 1.0 |
| `ki` | 0.001 ≤ ki ≤ 2.0 |
| `kd` | 0 ≤ kd ≤ 0.5 |
| `integrator_limit` | 0 < limit ≤ 20 |

The bounds are sanity limits, about ten times the gains in use today (Kp
0.07–0.09, Ki 0.081–0.22). They catch a corrupted value or a unit mistake;
loop stability remains the tuner's job. The record cannot change any safety
limit: the ±0.3 PID output clamp, the TEC DAC clamp, the temperature limits
and the loop rate stay compiled.

Fields the record does not set keep their compiled defaults, with one
exception. If the record sets `ki` but not `integrator_limit`, the limit
becomes `THERMAL_PID_OUTPUT_MAX / ki` (0.3 / ki on both boards), so the
integral term alone can just reach the output clamp. The compiled limit
belongs to the compiled Ki, so keeping it with a new Ki would be wrong. The
derived limit is not held to the 20 °C·s bound: a small Ki gives a large
limit (up to 300 at Ki = 0.001), but the integral term still tops out at 0.3.

The record is read once per power-up. Storing a record power-cycles the board
(see below), so new gains take effect at the next start.

Each laser board's USB console shows the gains in use with `TG`, followed by
`src=`, where start-up got them: `defaults` (no record, or one that sets
nothing), `record` or `rejected`. Live console changes alter the gains but
not `src`.

A read through the host API also reports why the firmware would reject a
stored record (`StoredTuning.rejected`), using the same rules.

## Storage on the laser boards

Flash sectors 8 and 9 of the 1 MB STM32F429 (`0x08080000` and `0x080A0000`,
128 KB each) hold 1024 slots per sector, used ping-pong as in identity sectors
2–3.

- When the active sector is full, the new record is written to the other
  sector before the old one is erased.
- Erasing a 128 KB sector takes about 1 s (2 s at most), so a migration can
  take up to about 4 s. A normal store takes milliseconds.
- App updates (`F`/`G`, staging sector 7 to app sector 5, or SWD images up to
  128 KB) and DAC programming (`D`, sector 1) never touch sectors 8–9. A
  full-chip erase does.
- The addresses assume the default single-bank option bytes (`DB1M` = 0),
  as do the other sector numbers in `1550_boot`.

## Mainboard USB commands

Tuning records use the identity commands `n` (read) and `N` (write, then read
back), with **`wValue` as the record kind**: `0` identity, `1` TEC tuning.
`wIndex` is the target, as for identity. Kind 1 applies to `1550` and `638`
only. The mainboard accepts the commands only in IDLE or OVERRIDE.

The reply is the same 140-byte `"UTID"` reply (schema 1). Byte 10, reserved
in the original layout, carries the record kind:

| Offset | Size | Field |
|---|---|---|
| 10 | 1 | record kind: `0` identity, `1` TEC tuning |
| 11 | 1 | reserved, `0` |

Mainboard firmware older than this ignores `wValue` and answers with an
identity reply, so byte 10 is `0`. The host must check byte 10 and treat a
kind mismatch as "mainboard firmware without tuning support".

Kind 1 adds one status code:

| Status | Name | Meaning |
|---|---|---|
| 8 | `BAD_KIND` | `wValue` is not a known record kind, or kind 1 was sent to target `0` |

For a write, the mainboard checks the magic `"UCTN"`, the layout version and
the payload length before it starts. A record that fails gets `BAD_RECORD`.

## Slave bootloader opcodes (mainboard ↔ `1550_boot` over SPI)

These work exactly like `0x15` / `0x16`, in the same 1 s boot window and with
the same handshake. They read and write the tuning store instead of the
identity store.

| Opcode | Request | Reply |
|---|---|---|
| `0x17 GET_TUNING` | `[0x17,0,0,0,0,crc8]` | 131 B `[0xA1][record 128][free slots u8][crc8]` |
| `0x18 PUT_TUNING` | `[0x18,0,0,0,0,crc8]`; poll for ACK `0xA1` ("ready"); then send 129 B `[record 128][crc8]` | poll for ACK `0xA1` (stored and verified) or NACK `0xA2` |

- **Free slots** saturates at 255, because a tuning sector has 1024 slots.
- **Reply delay:** scanning 2048 slots takes longer than the identity store,
  so the master waits 60 ms (not 20 ms) after `GET_TUNING` before it clocks
  out the reply.
- **Final ACK:** the master polls for up to 5 s, which covers a ping-pong
  migration.
- **Board type:** the bootloader keeps the board type the mainboard stamped.
- **Old bootloaders** ignore both opcodes, and the mainboard reports
  `NO_RESPONSE`.

## Host API

`ultracoustics.tec_tuning` builds and parses records and applies the bounds
above before anything is sent. `DiagnosticsManager` has three methods:

- `read_tec_tuning(target)`
- `write_tec_tuning(target, tuning)`
- `clear_tec_tuning(target)`, which writes a record that sets no TEC field

On the 638 both writes are read-modify-write and keep the stored lock PI
fields ([below](#writes-are-read-modify-write)).

A read shows what is stored. Whether the app accepted it shows on the board's
USB console.

## Lock PI fields (638)

The 638 record also carries the optical lock's PI gains: the normalized PI
gains of contract v2 (Kp, Ki in 1/s and the optional second-integrator zero)
that the 638 app uses as its box defaults, the gains that STOP, restore and
power-up return to. The 1550 has no lock fields.

| Payload offset | Size | Field | Type and unit | Not set | Accepted |
|---|---|---|---|---|---|
| 48 | 4 | `lock_kp` | float32, normalized PI Kp | NaN | 0 ≤ kp ≤ 5.115 |
| 52 | 4 | `lock_ki_per_s` | float32, normalized PI Ki, 1/s | NaN | 0 ≤ ki ≤ 16383 |
| 56 | 1 | `lock_zero_hz` | u8, second-integrator zero, Hz | `0xFF` | 0 ≤ z ≤ 63 |
| 57 | 3 | reserved | `0xFF 0xFF 0xFF` | | |

- Writers that carry lock fields use payload length 60. A record without
  them keeps the 48-byte layout version 1 payload. The TEC readers of both
  apps ignore bytes past offset 48.
- The SDK writes the values on the contract v2 wire grid (Kp in 0.005 steps,
  Ki in whole 1/s), so the 638's normalized PI readback (channel 10) equals
  what was stored. It reads a float back as the shortest decimal with the
  same float32 (0.6, not 0.6000000238).

### Validation and use (638 app)

The 638 app reads the lock fields once at start-up, before control starts,
from the newest valid record. It checks them as a group of their own,
independently of the TEC fields:

- the record must be valid and byte 5 must be `3` (638);
- if any set lock field is infinite or outside the range above, all lock
  fields are ignored and the compiled profile's lock gains stay (rejected);
- fields the record does not set keep the compiled profile value.

The box defaults are the record's lock fields over the compiled profile.
Runtime overrides (normalized PI channel 10 SET) work as before and end at
STOP. The 638 app never writes flash.

Normalized PI info (manual channel 11 GET, contract v2) reports where the
box defaults came from, in addition to bits 0–16:

| Bit | Meaning | `read_normalized_pi_638()` |
|---|---|---|
| 17 | lock gains are read from the tuning record (always 1 on this firmware) | `record_supported` |
| 18 | the box defaults' lock gains came from a record | `record_active` |
| 19 | a record's lock fields were rejected (profile gains in use) | `record_rejected` |

Bits 18 and 19 are defined only with bit 17 and are never both set. Channel
11 SET `0` restores the box defaults; any other value is `BAD_VALUE`.

### Writes are read-modify-write

The TEC fields and the lock fields share one record, and every write stores
a whole record. The host therefore reads the stored record first, then
writes one group new and the other as stored:

- a TEC write (`DiagnosticsManager.write_tec_tuning`, `clear_tec_tuning`)
  on the 638 keeps the stored lock fields;
- a lock write (`Controller.write_lock_tuning_638`, `clear_lock_tuning_638`)
  keeps the stored TEC fields.

A stored group that the firmware rejects is not kept: the firmware ignores
it either way, so the board keeps running what it runs now. If the read does
not return `OK` or `EMPTY`, nothing is written. `tuned_at` and `report_id`
describe the new write.

The mainboard stays in its record state for about 100 ms after each reply
and refuses another `n` / `N` without a reply until it is back in IDLE (it
also ignores START then). `DiagnosticsManager` waits 0.25 s between the read
and the write. A `Controller` sees the mainboard leave a maintenance state
when its ADC stream resumes (the stream pauses for every one), so it waits
for an ADC packet before and after each exchange.

### Host API on a running Controller

A GUI holds the USB device through its `Controller` stream, so the
`Controller` sends `n` / `N` itself and receives the reply through the
stream; no second USB owner is needed. If the mainboard does not answer
(it rejected the command), the stream clears the BULK OUT stall.

- `read_tuning_638(timeout_s=8.0)` returns a dict: `stored` (a valid record
  is stored), `lock_kp`, `lock_ki_per_s`, `lock_zero_hz`, the TEC fields
  (`kp`, `ki`, `kd`, `integrator_limit`, `setpoint_filter`,
  `conditional_integration`), `tuned_at`, `report_id`, `sequence` and
  `rejected`: None, or why the 638 would ignore the record's TEC fields
  and/or its lock fields. None means "not set".
- `write_lock_tuning_638(kp, ki_per_s, zero_hz=0, report_id=None,
  timeout_s=15.0)` stores lock gains and returns the same dict for the record
  as read back after the write. Values off the wire grid raise `ValueError`
  before anything is sent.
- `clear_lock_tuning_638(report_id=None, timeout_s=15.0)` stores the record
  with no lock field and returns the same dict.

The mainboard must be in IDLE (STOP first; it refuses outside IDLE and
OVERRIDE), and the `Controller` must not hold a manual session: each
exchange power-cycles the laser boards and leaves the mainboard in IDLE.
The ADC stream pauses for about a second per exchange, and no other command
may run meanwhile. The methods raise `RuntimeError` outside IDLE, when the
mainboard firmware predates tuning records (an identity reply, or no reply),
when the 638 bootloader lacks the tuning opcodes (`NO_RESPONSE`), and on any
other failure, including a read-back that differs from the record written.

Saving gains: STOP, `write_lock_tuning_638` (IDLE), START, then once LOCKED
`read_normalized_pi_638()` shows `record_active`, not `overridden`, and the
gains written.

## Not covered yet

- **Telemetry:** the tuning source is not in the board telemetry snapshot.
  The snapshot's reserved byte must stay `0` for current SDK parsers, so
  carrying it needs a telemetry schema change.
- **Profile settings:** the other 638 profile settings (DAC cap, ABBA settle)
  are still per-build.
