# Ultracoustics SDK

Centralized Python SDK for Ultracoustics BROADSONIC System

## Structure

```
ultracoustics-sdk/
├── pyproject.toml                 # Package metadata & dependencies
├── README.md
└── ultracoustics/                 # Importable package
    ├── __init__.py                # Package exports
    ├── config.py                  # Device constants (VID/PID, sample rate, etc.)
    ├── controller.py              # Controller class: connect / start measurement / stop measurement / save data
    ├── processing.py              # compute_psd, adc_to_uw, load_binary
    └── _internal/                 # Internal modules (not part of public API)
        ├── __init__.py
        ├── comms.py               # USB Bulk & USB-Serial communication (USBBulkConnection, USBStream, SerialConnection)
        ├── maintenance.py         # Firmware flashing, diagnostics, override control & calibration (Programmer, DiagnosticsManager, LaserSerialController, Calibrator)
        └── protocol.py            # USB command definitions & packing
```

## Installation

Follow these steps to install the SDK in EDITABLE Mode. This allows you to use the SDK across your entire computer while still being able to modify the source code if needed.

1. Create and Activate a Virtual Environment

A virtual environment ensures that the SDK’s dependencies—NumPy, PyUSB, and PySerial—do not interfere with other Python projects on your system.

Create the environment:

Bash
python -m venv venv

Activate the environment:

Windows: .\venv\Scripts\activate
macOS / Linux: source venv/bin/activate

2. Install the SDK (Perform Inside venv)
Crucial: This step must be performed after you have activated your virtual environment. Navigate to the SDK folder "ultracoustics-sdk" (containing pyproject.toml) and run:

Bash
pip install -e "C:\path\to\folder\...\ultracoustics-sdk"

Note on Editable Mode: Using the -e flag means the SDK source code remains in the folder where you downloaded it. Instead of copying files into the virtual environment, Python creates a "path link" (shortcut). This allows you to edit the SDK source code and have those changes take effect immediately without needing to reinstall the package.

Note: This install automatically pulls in the necessary dependancies: NumPy, PyUSB, and PySerial.

3. Hardware Connection
The Ultracoustics Master Board supports WCID (Plug-and-Play).

Plug the Master Board into a USB port.

Windows will automatically recognize the device and assign the correct WinUSB driver.

No manual driver installation or Zadig selection is required.

See Install Troubleshooting section for manual driver instructions. 

4. Quick Start Example
Verify your installation by running this script:

python
import time
from ultracoustics import Controller

# Initialize and connect
ctrl = Controller(verbose=True)
ctrl.connect() 

# Start high-speed measurement
ctrl.start()

# Wait for buffer to fill (10 MHz sampling)
time.sleep(1.1)

# Save 1 second of 16-bit ADC data (~20 MB)
ctrl.save(duration_s=1.0, path="capture.bin")

# Shutdown
ctrl.stop()
ctrl.close()


## Install Troubleshooting 

"ModuleNotFoundError": Ensure you ran the pip install -e . command in the folder containing pyproject.toml.

"Access Denied" (Linux): You may need to create a udev rule or run your script with sudo to access USB devices.

"Device not found": Double-check that the Master Board is powered on and the USB cable is data-capable. Also check the device driver was correctly assigned.  

    If automatic driver setup failed (Windows only): Windows requires a specific driver to allow Python to talk to the BROADSONIC USB Hardware.

        Download and run Zadig (zadig.akeo.ie).
        Plug in the Master Board via USB.
        In Zadig, go to Options -> List All Devices.
        Select BROADSONIC (or Device ID 2E9D : 000A) from the dropdown.
        Select WinUSB as the driver and click Replace Driver.

## Quick Start

```python
from ultracoustics import Controller, compute_psd, Programmer

# --- Connect, stream, and save 1 second of data ---
ctrl = Controller(verbose=True)
ctrl.connect()
ctrl.start()
# ... wait for buffer to fill ...
import time; time.sleep(1.5)
data = ctrl.save(duration_s=1.0, path="capture.bin")
ctrl.stop()
ctrl.close()

# --- Offline FFT analysis ---
from ultracoustics import load_binary, compute_psd
samples = load_binary("capture.bin")
freq_hz, psd_db = compute_psd(samples, fft_size=8192, num_averages=20)

# --- Flash firmware (VIA Master Board USB, flash Board 2 – 1550 nm slave) ---
from ultracoustics import USBBulkConnection, Programmer
conn = USBBulkConnection(verbose=True)
prog = Programmer(conn, verbose=True)
prog.flash_1550(open("1550.bin", "rb").read())
conn.close()
```

## Device Identification

| Device | VID | PID | Connection |
|--------|-----|-----|-----------|
| Master Board (Photodetector) | 0x2E9D | 0x000A | USB Bulk HS |
| 1550nm Slave (Laser) | 0x0483 | 0x5740 | USB-Serial |
| 638nm Slave (Laser) | 0x0483 | 0x5740 | USB-Serial |

*1550 and 638 serial connections should only be made during troubleshooting with manufacturer
