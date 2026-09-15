#!/usr/bin/env python3
"""CLI for the shared manual-sweep SDK policy. No writes occur on import."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from ultracoustics import Controller
from ultracoustics.manual_sweep import ManualSweepError, run_manual_sweep

FIELDS = ("target", "dac_requested", "dac_applied", "pd_raw_counts",
          "pd_full_scale_counts", "saturated", "sample_tick_ms", "captured_at_utc",
          "pd_age_ms", "pd_flags", "pd_fault", "link_flags",
          "temperature_target_c", "temperature_measured_c")


def write_results(output, target, rows, error=None, cleanup_error=None):
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = output / f"manual_{target}_{stamp}.csv"
    json_path = csv_path.with_suffix(".json")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps({
        "units": {"pd_raw_counts": "ADC counts", "temperature": "degC"},
        "error": None if error is None else str(error),
        "cleanup_error": None if cleanup_error is None else str(cleanup_error),
        "rows": rows,
    }, indent=2) + "\n")
    return csv_path, json_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, required=True, choices=(638, 1550))
    parser.add_argument("--points", type=int, default=100, choices=range(2, 1001))
    parser.add_argument("--max-dac", type=int)
    parser.add_argument("--output", type=Path, default=Path("manual_sweeps"))
    args = parser.parse_args()
    ctrl = Controller()
    rows = []
    error = None
    cleanup_error = None
    try:
        ctrl.connect()
        rows = run_manual_sweep(ctrl, args.target, args.points,
                                max_dac=args.max_dac)
    except ManualSweepError as exc:
        rows = exc.rows
        error = exc.primary_error
        cleanup_error = exc.cleanup_error
    except Exception as exc:
        error = exc
    finally:
        ctrl.close()
    csv_path, json_path = write_results(
        args.output, args.target, rows, error, cleanup_error)
    print(csv_path)
    print(json_path)
    if error is not None:
        raise error


if __name__ == "__main__":
    main()
