"""
Ultracoustics USB Command Protocol Definition
Shared constants and packing logic for firmware commands.
"""

import struct

# USB Endpoint Configuration
BULK_OUT_EP = 0x01
BULK_IN_EP = 0x81

# Command IDs (Must match firmware usb_command_dispatcher.h)
CMD_BOOT = ord('S')  # Start Boot Sequence (turn on lasers, so measurements can be taken during firmware update
CMD_IDLE = ord('I')  # Enter Idle State (turn off lasers, stop measurements)
CMD_WARM = ord('W')  # Enter Warm State (turn off lasers, but keep warm, no measurements can be taken in this state)
CMD_OVERRIDE_ENTER = ord('O')  # Enter/Exit Override Mode
CMD_POWER = ord('P')  # Turn on power to slave boards (toggles VREGEN)
CMD_TRIGGER = ord('T')  # Tell slave to begin startup sequence (laser enable)
CMD_PROGRAM_1550 = ord('D') # Program 1550nm laser power DAC (Val=DAC)
CMD_FLASH_1550 = ord('F')   # Flash 1550nm firmware (Val=SubCMD)
CMD_FLASH_638 = ord('G')    # Flash 638nm firmware (Val=SubCMD)
CMD_IAP = ord('U')          # In-Application Programming (self-update master code)
CMD_BOOT_VERSION = ord('V')      # Query 1550nm bootloader firmware version
CMD_BOOT_VERSION_638 = ord('v')  # Query 638nm bootloader firmware version
CMD_FWINFO = ord('i')       # Query master board firmware version

# Target IDs for Manual Control
TARGET_1550 = 1550
TARGET_638 = 638

def pack_command(cmd_byte, wValue=0, wIndex=0):
    """
    Pack command into 5-byte structure for Bulk OUT.
    Format: [CMD, VAL_L, VAL_H, IDX_L, IDX_H]
    """
    # Struct format: '<BHH' (Little-endian: UChar, UShort, UShort)
    return struct.pack('<BHH', cmd_byte, wValue, wIndex)
