#!/usr/bin/env python3
"""Explicit standalone 638 characterization; never changes PI gains.

Usage: python examples/pi_characterization.py REPORT_DIR --start-and-stop
       [--verified-single-sample-filter]

The flag explicitly authorizes this standalone program to start and stop the
laser system. Applications with an existing RUN Controller should call
``ultracoustics.run_pi_characterization`` directly instead.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultracoustics import Controller, run_pi_characterization


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_dir")
    parser.add_argument("--start-and-stop", action="store_true", required=True,
                        help="explicitly start and later stop the laser system")
    parser.add_argument("--verified-single-sample-filter", action="store_true",
                        help="assert the installed master feedback filter was independently verified OFF")
    parser.add_argument("--driver-pole-hz", type=float,
                        help="independent nominal driver pole estimate in Hz (300–5000)")
    parser.add_argument("--driver-pole-low-hz", type=float,
                        help="lower bound for driver-pole sensitivity range")
    parser.add_argument("--driver-pole-high-hz", type=float,
                        help="upper bound for driver-pole sensitivity range")
    parser.add_argument("--driver-pole-source",
                        help="measurement, board revision/BOM, or explicit estimate provenance")
    args = parser.parse_args()
    if (args.driver_pole_low_hz is None) != (args.driver_pole_high_hz is None):
        parser.error("supply both driver-pole range bounds")
    if args.driver_pole_hz is not None and not args.driver_pole_source:
        parser.error("driver-pole source is required with a pole prior")
    if args.driver_pole_hz is None and (args.driver_pole_low_hz is not None or args.driver_pole_source):
        parser.error("driver-pole prior is required with bounds or source")
    controller = Controller(verbose=False, ring_seconds=1.5)
    started = False
    try:
        controller.connect()
        controller.begin_stream()
        controller.enable_optical_diagnostics()
        started = True
        controller.start_confirmed()
        report = run_pi_characterization(
            controller, args.report_dir,
            progress=lambda event: print(f"{event['stage']}: {event['message']}", flush=True),
            verified_master_filter="single_sample" if args.verified_single_sample_filter else None,
            driver_pole_hz=args.driver_pole_hz,
            driver_pole_range_hz=([args.driver_pole_low_hz, args.driver_pole_high_hz]
                                  if args.driver_pole_low_hz is not None else None),
            driver_pole_source=args.driver_pole_source,
        )
        print(f"{report['status']}: {report.get('reason') or 'see report.md'}")
        return 0 if report["status"] == "complete" else 2
    finally:
        try:
            if started:
                controller.stop_system_confirmed()
        finally:
            try:
                controller.end_stream()
            finally:
                controller.close()


if __name__ == "__main__":
    raise SystemExit(main())
