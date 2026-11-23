"""Capture a calibrated recording from the box. Needs a box connected.

    python examples/capture.py

Writes capture_<date>_<time>.sigmf-meta/-data and long_run_<date>_<time>.sigmf-meta/-data
in the current folder. Nothing is ever overwritten, so each run gets names of its own.
"""
import time

import ultracoustics as uc


def main():
    with uc.Controller() as ctrl:              # close() always runs, with a confirmed stop
        # Required. Lasers on, ready and calibrated, as START in the GUI: about 30 s.
        # Keep the probe still and the surroundings quiet until it returns.
        report = ctrl.start_ready()
        print(report.message)

        stamp = time.strftime("%Y%m%d_%H%M%S")
        rec = ctrl.capture(1.0)                # the next second, checked for lost samples
        print("saved", rec.save(f"capture_{stamp}"))

        # Straight to disk, any length. It gives back the recorder, which says how it went.
        recorder = ctrl.record(f"long_run_{stamp}", duration_s=10)
        print(f"recorded {recorder.samples / rec.sample_rate:.1f} s, {recorder.lost} samples lost, "
              f"{recorder.not_ready_s:.1f} s not ready")


if __name__ == "__main__":     # needed: the USB reader runs in a process of its own, which imports this file
    main()
