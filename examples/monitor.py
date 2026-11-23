"""Watch a machine for faults for a minute. Needs a box connected.

    python examples/monitor.py

Take the healthy baseline while the machine runs normally. The monitor then
raises an alarm when the band rises 10 dB above it, saves an event around each
onset, and logs levels every second. Each run gets its own folder under
monitor_runs/. docs/monitoring.md explains the files.
"""
import time

import ultracoustics as uc


def show(change):
    """Called on the monitor's thread when an alarm is raised or cleared."""
    print(f"{change.utc:%H:%M:%S} {change.trigger}: {change.state} (level {change.level:.1f}, limit {change.limit:.1f})")


def main():
    with uc.Controller() as ctrl:
        report = ctrl.start_ready()            # lasers on, ready and calibrated: about 30 s
        if not report.calibrated:              # the limits below are in pressure
            raise SystemExit(report.message)

        healthy = ctrl.capture(5.0).spectrum()    # the machine when it is healthy
        watch = [uc.BandPowerTrigger((20e3, 100e3), threshold_db=10, baseline=healthy, name="bearing band"),
                 uc.LevelTrigger(level_pa=1.0, band_hz=(2e3, 20e3), name="impacts")]

        monitor = ctrl.monitor(watch, folder="monitor_runs", on_alarm=show, on_status=print)
        try:
            time.sleep(60)
        finally:
            monitor.stop()
        summary = monitor.summary()
        print(f"{summary.events} events, {summary.alarms_raised} alarms, "
              f"{summary.not_monitored_s:.1f} s not watched, {summary.samples_lost} samples lost")
        print("files in", summary.folder)


if __name__ == "__main__":     # needed: the USB reader runs in a process of its own, which imports this file
    main()
