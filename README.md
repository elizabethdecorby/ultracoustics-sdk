# Ultracoustics SDK

Python SDK for the Ultracoustics BROADSONIC hardware platform.

## Project Structure

```text
ultracoustics-sdk/
├── pyproject.toml
├── README.md
├── examples/
│   ├── basic_capture.py
│   └── offline_psd.py
└── ultracoustics/
    ├── __init__.py
    ├── config.py
    ├── controller.py
    ├── processing.py
    └── _internal/
        ├── __init__.py
        ├── comms.py
        ├── maintenance.py
        ├── protocol.py
        └── bin/
```

## At a glance

- The public API is exposed through the `ultracoustics` package.
- Modules under `_internal` are implementation details and are included for guided diagnostics.

## Installation

Install in editable mode so local source edits are available immediately.

1. Create and activate a virtual environment.

Windows PowerShell:

```powershell
py -3.13 -m venv venv_name
.\venv_name\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python3 -m venv venv _name
source venv_name/bin/activate
```

2. Install the SDK from the repository root (the folder that contains pyproject.toml).

```bash
pip install -e .
```

This installs NumPy, PyUSB, and PySerial automatically.

## Hardware Connection

The Master Board supports WCID plug-and-play on Windows.

1. Plug the board into a USB data port.
2. Confirm Windows assigns WinUSB driver automatically (see "BROADSONIC" in Device Manager)
3. If automatic driver assignment fails, use the Zadig steps in Troubleshooting.

## Quick Start

```python
import time
from ultracoustics import Controller, load_binary, compute_psd

ctrl = Controller(verbose=True)
ctrl.connect()

# Start USB data ingestion first (laser remains off)
ctrl.begin_stream()

# Enter measurement mode (laser on, probe turns red, pink indicater)
ctrl.start()

# Wait for system lock and buffer fill
time.sleep(3.1)

# Save 1 second of raw uint16 ADC data
data = ctrl.save(duration_s=1.0, path="capture.bin")

ctrl.stop()        # System to IDLE state (laser off, white indicator)
ctrl.end_stream()  # Stop USB reader thread
ctrl.close()

# Optional offline PSD analysis
samples = load_binary("capture.bin")
freq_hz, psd_db = compute_psd(samples, fft_size=8192, num_averages=20)
```

## Examples

Run examples from the repository root after installation:

```bash
python examples/basic_capture.py
python examples/offline_psd.py
```

## Troubleshooting

ModuleNotFoundError:

- Ensure the virtual environment is activated.
- Ensure installation was run from the repo root with `pip install -e .`.

Access denied on Linux:

- Configure a udev rule for USB access, or run with elevated privileges.

Device not found:

- Verify the board is powered and the USB cable supports data.
- Confirm the driver assignment in Device Manager.

If automatic Windows driver setup fails:

1. Download and run Zadig: https://zadig.akeo.ie
2. Plug in the Master Board.
3. In Zadig, open Options -> List All Devices.
4. Select BROADSONIC (VID:PID 2E9D:000A).
5. Select WinUSB and click Replace Driver.

