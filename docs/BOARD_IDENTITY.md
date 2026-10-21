# Board identity records

Each board stores one **identity record**: the hardware details recorded at
production diagnostics, such as serial numbers, PCB revisions and pot
resistances. The host reads and writes every record through the mainboard USB
link, the same way it reads firmware versions (`i`, `V`, `v`).

This file is the single source of truth for the record layout and the wire
protocol. The firmware in `Photodetector-PSSIandUSBHS` (mainboard) and
`1550_boot` (both laser boards) follows it. `ultracoustics/identity.py`
implements the host side.

## Record (128 bytes, little-endian)

| Offset | Size | Field | Set by |
|---|---|---|---|
| 0 | 4 | magic `"UCID"` | host |
| 4 | 1 | layout version, `1` | host |
| 5 | 1 | board type: `1` main, `2` 1550, `3` 638 | mainboard firmware (overwrites the host value) |
| 6 | 2 | payload length, 0–112 | host |
| 8 | 4 | sequence number | storing firmware: previous + 1, first record = 1 |
| 12 | 4 | CRC-32 over bytes 0–11 plus the first *payload length* payload bytes | storing firmware, after it has stamped the board type and sequence |
| 16 | 112 | payload (TLV); unused bytes are `0xFF` | host |

The CRC is CRC-32/MPEG-2: polynomial `0x04C11DB7`, init `0xFFFFFFFF`, not
reflected, no final XOR. It is the same CRC the slave bootloader uses for
firmware images.

The firmware treats the payload as opaque and checks only the magic, the
layout version, the length and the CRC. New fields therefore need only a host
change.

### Payload TLVs

The payload is a sequence of `[tag u8][len u8][value]` entries. Readers keep
unknown tags and otherwise ignore them. Strings are UTF-8 with no terminator.
Pot resistances are u16 counts: tenths of an ohm for the driver pot (up to
6553.5 Ω), whole ohms for the LPF and PD pots (up to 65 535 Ω). Readers also
accept a 4-byte float32 value for these tags, the format of the earliest
records.

| Tag | Name | Type | Boards |
|---|---|---|---|
| `0x01` | `serial`: main board S/N or laser S/N | str ≤ 24 | all |
| `0x02` | `pcb_rev`: CPU PCB on the main board, laser board PCB on the slaves | str ≤ 12 | all |
| `0x03` | `pd_pcb_rev`: photodetector PCB | str ≤ 12 | main |
| `0x04` | `circ_sn`: circulator S/N | str ≤ 24 | main |
| `0x10` | `driver_pot_ohm` | u16, 0.1 Ω | 1550, 638 |
| `0x11` | `lpf_pot_ohm` | u16, 1 Ω | 1550, 638 |
| `0x12` | `pd_pot_ohm` | u16, 1 Ω | 1550, 638 |
| `0x20` | `programmed_at`: unix seconds | u32 | all |
| `0x21` | `report_id`: diagnostics session (timestamp + short UID) | str ≤ 24 | all |

A field that was never recorded is simply absent. The length limits guarantee
that a record with every field at its maximum length still fits in 112 bytes.

## Storage on each board

Each board keeps its records in append-only slots. A read returns the valid
record with the highest sequence number. A write programs the next erased slot
and never touches the previous record, so a power loss mid-write leaves the
old record readable.

- **Mainboard:** internal flash sector 7 (`0x0800E000`, 8 KB), 64 slots.
  - When all slots are full, the sector is erased and the new record goes
    into slot 0. A power loss between that erase and the write leaves the
    board empty.
  - Master IAP erases only sectors 0–6, so firmware images are limited to
    56 KB.
- **Laser boards:** sectors 2 and 3 (`0x08008000` and `0x0800C000`, 16 KB
  each), 128 slots per sector, used ping-pong.
  - When the active sector is full, the new record is written to the other
    sector before the old one is erased.
  - App updates (`F`/`G` or SWD) and DAC programming (`D`, sector 1) never
    touch sectors 2–3.

## Mainboard USB commands

The commands use the standard 5-byte frame `[cmd][wValue u16][wIndex u16]`,
with `wIndex` as the target: `0` mainboard, `1550`, `638`. `wValue` is the
record kind: `0` for identity records. Kind `1` carries TEC tuning records on
the same commands ([TEC_TUNING.md](TEC_TUNING.md)). Both commands are
accepted only in IDLE or OVERRIDE.

| Cmd | Meaning | Payload |
|---|---|---|
| `n` | read identity | none |
| `N` | write identity, then read it back | 128-byte record (the firmware assigns the sequence number and CRC) |

The mainboard **always** answers both commands with a 140-byte reply:

| Offset | Size | Field |
|---|---|---|
| 0 | 4 | `"UTID"` |
| 4 | 1 | reply schema, `1` |
| 5 | 1 | op: `0` read, `1` write |
| 6 | 2 | target (`0`, `1550` or `638`) |
| 8 | 1 | status |
| 9 | 1 | free slots left |
| 10 | 1 | record kind: `0` identity, `1` TEC tuning (was reserved; older firmware sends `0`) |
| 11 | 1 | reserved |
| 12 | 128 | record as read back (all `0xFF` if empty) |

| Status | Name | Meaning |
|---|---|---|
| 0 | `OK` | the record is valid; after a write, the readback matches what was written |
| 1 | `EMPTY` | no valid record stored |
| 2 | `NO_RESPONSE` | the slave did not answer: no board, or a bootloader older than the identity opcodes |
| 3 | `LINK_CRC` | the slave reply failed its CRC-8 |
| 4 | `WRITE_FAIL` | the flash erase or program failed |
| 5 | `VERIFY_FAIL` | the readback after a write does not match |
| 6 | `BAD_RECORD` | the host record failed header or length checks |
| 7 | `BAD_STATE` | the mainboard was not in IDLE or OVERRIDE |
| 8 | `BAD_KIND` | `wValue` is not a known record kind, or the kind is not stored on that target |

## Slave bootloader opcodes (mainboard ↔ `1550_boot` over SPI)

These are sent in the bootloader's 1 s window after power-up, in the same
format as `0x14 GET_VERSION`, with CRC-8 polynomial 0x31 and init 0. Both
restart the boot window.

| Opcode | Request | Reply |
|---|---|---|
| `0x15 GET_IDENTITY` | `[0x15,0,0,0,0,crc8]` | 131 B `[0xA1][record 128][free slots u8][crc8]` |
| `0x16 PUT_IDENTITY` | `[0x16,0,0,0,0,crc8]`; poll for ACK `0xA1` ("ready"); then send 129 B `[record 128][crc8]` | poll for ACK `0xA1` (stored and verified) or NACK `0xA2` |

`PUT_IDENTITY` uses the same handshake as `FLASH_BEGIN` / `FLASH_PAGE`. After
each ACK or NACK, the slave loads `0xFF` (idle) so a later poll can't read a
stale ACK. A normal store takes milliseconds. A ping-pong migration erases up
to two 16 KB sectors, so the master should poll for the final ACK for at
least 3 s.

The bootloader assigns the sequence number and computes the record CRC. It
keeps the board type the mainboard stamped, because the bootloader itself
cannot tell which laser board it runs on. Bootloaders that predate these
opcodes ignore them, and the mainboard reports `NO_RESPONSE`.
