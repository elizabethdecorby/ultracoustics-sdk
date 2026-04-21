"""Basic 1-second capture example for Ultracoustics Python SDK."""

import time

from ultracoustics import Controller


def main() -> None:
    ctrl = Controller(verbose=True)
    try:
        ctrl.connect()
        ctrl.begin_stream()
        ctrl.start()

        # Wait for device lock and enough samples for a 1-second snapshot.
        time.sleep(3.1)

        data = ctrl.save(duration_s=1.0, path="capture.bin")
        print(f"Captured {len(data)} samples to capture.bin")

        ctrl.stop()
        ctrl.end_stream()
    finally:
        ctrl.close()


if __name__ == "__main__":
    main()
