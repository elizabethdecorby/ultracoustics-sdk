# Third-party notices

## libusb

The Windows package includes `ultracoustics/_internal/bin/libusb-1.0.dll`,
libusb 1.0.28, from <https://libusb.info>.

libusb is free software under the GNU Lesser General Public License, version
2.1 or later. Its text is in `ultracoustics/_internal/bin/LGPL-2.1.txt`. The
source code is at <https://github.com/libusb/libusb>, and that of this
release under the tag `v1.0.28`.

The SDK loads the DLL at run time and does not change it. You may replace it
with another build of libusb 1.0 of your own.

## Python packages

The SDK installs these from PyPI. Each keeps its own licence.

| Package | Licence |
|---|---|
| NumPy | BSD 3-Clause, with parts under 0BSD, MIT, Zlib and CC0 |
| SciPy | BSD 3-Clause |
| PyUSB | BSD 3-Clause |
| libusb1 (python-libusb1) | LGPL 2.1 or later |
| cryptography | Apache 2.0 or BSD 3-Clause, with OpenSSL under Apache 2.0 |
| truststore (Python 3.10 and later) | MIT |
