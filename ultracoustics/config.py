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

# One USB packet: the reader may be part-way through writing this many of the
# oldest samples in the ring before its counters say so.
RING_GUARD_SAMPLES = 8192

# The stream's own loss counters: any rise means samples went missing.
STREAM_HEALTH_KEYS = ('drops_seq', 'transfer_errors', 'transfer_timeouts', 'malformed')
