"""
Configuration constants for the Ultracoustics SDK.
"""

# USB Device Configuration
VENDOR_ID = 0x2E9D  # Ultracoustics Technologies Ltd.
PRODUCT_ID = 0x000A  # BROADSONIC

# ADC and Sampling Configuration
SAMPLE_RATE = 10_000_000  # 10 MHz
ADC_BITS = 14  # 14-bit ADC
ADC_MAX_VALUE = (1 << ADC_BITS) - 1  # 16383
