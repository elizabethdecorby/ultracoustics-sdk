"""Capture a calibrated recording from the box. Needs a box connected.

    python examples/capture.py

Writes capture.sigmf-meta/-data and long_run.sigmf-meta/-data in the current
folder. It refuses to overwrite them, so move or rename them to run it again.
"""
import ultracoustics as uc


def main():
    with uc.Controller() as ctrl:              # close() always runs, with a confirmed stop
        dark = ctrl.capture_dark()             # the lasers-off reference, before START
        ctrl.start()

        # Every START: confirms the lock suits this probe, and retunes it if not.
        # Keep the probe still and the room quiet for it.
        report = ctrl.check_lock(retune=True)
        print(f"lock: {report.status}. {report.message}")
        if not report.ready:
            raise SystemExit("no settled lock: check the probe and the room, then run this again")

        cal = ctrl.calibrate_pressure(dark)    # this START's pressure calibration
        print(f"calibrated on the {cal.fundamental.f0_hz / 1e6:.2f} MHz mode")

        rec = ctrl.capture(1.0)                # the next second, checked for lost samples and lock
        print("saved", rec.save("capture"))

        job = ctrl.record("long_run", duration_s=10)    # straight to disk: any length
        print(f"recorded {job.samples / rec.sample_rate:.1f} s, {job.lost} samples lost")


if __name__ == "__main__":     # needed: the USB reader runs in a process of its own, which imports this file
    main()
