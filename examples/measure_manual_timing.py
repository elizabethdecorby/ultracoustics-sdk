#!/usr/bin/env python3
"""
Measure manual-command timing and where replies land in the ADC stream.

Bench check for the single test mode (docs/TEST_MODE.md, phase 2). It opens a
manual session with the 1550 on its own firmware and the probe connected,
toggles the 638 DAC between two values, and for each SET reports:

- the host round trip, from queueing the request to receiving the reply;
- the offset from the reply's stream position (``ManualReply.stream_sample``)
  to the step edge found in the photodetector data. Negative means the edge
  came before the reply, which is what the protocol promises.

The probe sweep's ``settle_s`` must exceed the largest positive offset plus
the probe's own response; the round trips set how long a sweep takes.

Pick two DACs where the probe readback has a clear slope (look at a probe
sweep first), so each step makes a visible edge.

Usage
-----

    python examples/measure_manual_timing.py --low 20000 --high 22000 --steps 40
"""

import argparse
import csv
import statistics
import sys
import time

import numpy as np

from ultracoustics import Controller
from ultracoustics._internal.control import CHANNEL_LASER_DAC, MANUAL_SET

SAMPLE_RATE = 10_000_000
BIN = 100          # 10 us bins for the edge search
PRE = 200_000      # 20 ms of history before the reply
POST = 100_000     # 10 ms after it
BASELINE = 100_000  # first 10 ms of the window is the baseline


def ring_window(ctrl, start, end, timeout_s=2.0):
    """Copy samples [start, end) from the controller's ring."""
    deadline = time.monotonic() + timeout_s
    while ctrl.samples_received < end:
        if time.monotonic() > deadline:
            raise TimeoutError("ADC samples stopped arriving")
        time.sleep(0.001)
    capacity = ctrl.buffer_capacity
    if ctrl.samples_received - start > capacity:
        raise RuntimeError("ring overwritten; increase ring_seconds")
    index, count = start % capacity, end - start
    buf = ctrl.buffer
    if index + count <= capacity:
        return buf[index:index + count].astype(np.float64)
    return np.concatenate((buf[index:], buf[:index + count - capacity])).astype(np.float64)


def edge_offset_us(window, min_shift):
    """Offset of the first bin leaving the baseline, relative to the reply."""
    bins = window[: len(window) // BIN * BIN].reshape(-1, BIN).mean(axis=1)
    base = bins[: BASELINE // BIN]
    threshold = max(6.0 * float(base.std()), float(min_shift))
    moved = np.flatnonzero(np.abs(bins - base.mean()) > threshold)
    moved = moved[moved >= BASELINE // BIN]
    if moved.size == 0:
        return None
    return (int(moved[0]) * BIN - PRE) / SAMPLE_RATE * 1e6


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--low", type=int, default=20000)
    parser.add_argument("--high", type=int, default=22000)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--hold", type=float, default=0.1,
                        help="seconds between steps (default 0.1)")
    parser.add_argument("--warmup", type=float, default=10.0,
                        help="1550 warm-up after its TEC locks (default 10 s)")
    parser.add_argument("--min-shift", type=float, default=20.0,
                        help="smallest ADC shift counted as an edge (default 20)")
    parser.add_argument("--csv", help="write one row per step to this file")
    args = parser.parse_args()

    ctrl = Controller(ring_seconds=2.0)
    rows = []
    try:
        ctrl.connect()
        ctrl.begin_stream()
        ctrl.begin_system_manual()
        print("Starting the 1550 on its own firmware...")
        ctrl.start_1550_automatic(warmup_s=args.warmup)
        ctrl.wait_tec_lock(638)
        ctrl.system_manual_command(638, MANUAL_SET, CHANNEL_LASER_DAC, args.low)
        time.sleep(0.5)

        for step in range(args.steps):
            dac = args.high if step % 2 == 0 else args.low
            reply = ctrl.system_manual_command(638, MANUAL_SET, CHANNEL_LASER_DAC, dac)
            window = ring_window(ctrl, reply.stream_sample - PRE, reply.stream_sample + POST)
            offset = edge_offset_us(window, args.min_shift)
            round_trip_ms = reply.round_trip_ns / 1e6
            rows.append({"step": step, "dac": dac, "round_trip_ms": round_trip_ms,
                         "edge_offset_us": offset})
            shown = "no edge" if offset is None else f"{offset:+8.1f} us"
            print(f"{step:3d}  DAC {dac:5d}  round trip {round_trip_ms:6.2f} ms  edge {shown}")
            time.sleep(args.hold)

        ctrl.system_manual_command(638, MANUAL_SET, CHANNEL_LASER_DAC, 0)
    finally:
        try:
            ctrl.finish_system_manual()
        except Exception as exc:
            print(f"STOP failed: {exc}")
        ctrl.close()

    trips = [r["round_trip_ms"] for r in rows]
    offsets = [r["edge_offset_us"] for r in rows if r["edge_offset_us"] is not None]
    print("\nRound trip (ms): median {:.2f}, 95th {:.2f}, max {:.2f}".format(
        statistics.median(trips), percentile(trips, .95), max(trips)))
    if offsets:
        print("Edge offset (us): min {:+.1f}, median {:+.1f}, max {:+.1f} "
              "({} of {} steps)".format(min(offsets), statistics.median(offsets),
                                        max(offsets), len(offsets), len(rows)))
        print("settle_s must exceed {:.1f} us plus the probe's own response."
              .format(max(0.0, max(offsets))))
    else:
        print("No edges found: choose --low/--high on a slope of the probe curve.")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
