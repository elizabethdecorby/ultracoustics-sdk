"""
High-level controller for the Ultracoustics BROADSONIC System.

This module provides the :class:`Controller` class, which is the primary
user-facing interface for the Ultracoustics SDK. It manages:

- **System state management** (BOOT / IDLE / WARM modes)
- **USB Bulk connection** to the Master Board for streaming measurement data
- **Data acquisition** with read access to a rolling circular buffer and snapshot-based saving

Internally, :meth:`Controller.connect` performs a quick probe via
:class:`comms.USBBulkConnection`, then constructs a
:class:`comms.USBStream`.  The stream spawns a dedicated reader
*subprocess* (see :mod:`ultracoustics._internal.stream_proc`) which owns
the libusb-1.0 device handle exclusively and pumps async USB transfers
directly into a **shared-memory ring buffer**.  The parent process reads
samples lock-free and zero-copy via the ``buffer`` / ``buffer_head`` /
``samples_received`` properties.  Outbound commands (BOOT/IDLE/WARM) are
routed to the subprocess through a multiprocessing queue, since only the
subprocess holds the device handle.

For diagnostics, override mode, firmware queries, laser serial control, and
firmware flashing, see :mod:`ultracoustics._internal.maintenance`.

Typical usage::

    from ultracoustics.controller import Controller

    ctrl = Controller(verbose=True)
    ctrl.connect()             # USB bulk to master board
    ctrl.begin_stream()        # start reading USB data into buffer (laser still off)
    ctrl.start()               # send BOOT command (turn on laser measurement system)
    import time; time.sleep(2) # wait for system to lock, stabilize, and for buffer to fill
    data = ctrl.save(1.0)      # snapshot 1 second of samples
    ctrl.stop()                # send IDLE (turn off laser measurement system, stream keeps running)
    ctrl.end_stream()          # stop reading USB data
    ctrl.close()               # release all resources
"""

import time
from pathlib import Path
from typing import Optional, Union

import numpy as np

from ._internal.comms import USBBulkConnection, USBStream
from ._internal.protocol import (
    CMD_BOOT, CMD_IDLE, CMD_WARM,
    CMD_OVERRIDE_ENTER, CMD_POWER, CMD_TRIGGER, CMD_PROBE_CHAR,
    TARGET_1550, TARGET_638,
)
from .config import SAMPLE_RATE, ADC_MAX_VALUE
from .characterization import bin_ramp, ProbeCharacterizationResult


class Controller:
    """
    Top-level interface to the Ultracoustics system.

    Usage::

    ctrl = Controller(verbose=True)
        ctrl.connect()             # USB bulk to master board
        ctrl.begin_stream()        # start reading USB data into buffer (laser still off)
        ctrl.start()               # send BOOT command (turn on laser measurement system)
        import time; time.sleep(2) # wait for system to lock, stabilize, and for buffer to fill
        data = ctrl.save(1.0)      # snapshot 1 second of samples
        ctrl.stop()                # send IDLE (turn off laser, stream keeps running)
        ctrl.end_stream()          # stop reading USB data
        ctrl.close()               # release all resources

        Real-time buffer access
        -----------------------
        For GUI or live-analysis consumers that need zero-copy access to
        the streaming data at high refresh rates (e.g. 20 Hz), the
        following read-only properties expose the internal circular buffer
        without allocating a copy on every call (unlike :meth:`save`)::

            ctrl.buffer           # np.ndarray – the live uint16 sample ring
            ctrl.buffer_head      # int        – current write-head index
            ctrl.buffer_capacity  # int        – ring length in samples
            ctrl.samples_received # int        – cumulative sample count
            ctrl.streaming        # bool       – True while reader subprocess is running
            ctrl.running          # bool       – True between start() and stop()
            ctrl.stream_stats     # dict       – packet-level diagnostics

        The buffer is written by the reader *subprocess* into shared
        memory; readers in the parent process should treat it as a
        lock-free snapshot (read ``buffer_head`` first, then slice the
        array).  Counter reads are atomic 64-bit aligned loads on
        x86-64.
    """

    def __init__(self, verbose=False, ring_seconds: float = 1.2):
        """Initialise the controller (no hardware interaction yet).

        Parameters
        ----------
        verbose : bool, optional
            If ``True``, print diagnostic messages during USB I/O
            and data capture operations. Defaults to ``False``.
        ring_seconds : float, optional
            Capacity of the shared-memory ring buffer, expressed in
            seconds of samples at the configured sample rate. Defaults
            to ``1.2`` (the live-display window the GUI expects). Increase
            this for long unattended captures that must fit in the ring
            in one piece (e.g. an ~8 s probe-characterization ramp);
            remember each second is ``SAMPLE_RATE`` uint16 samples
            (~20 MB/s at 10 MSPS).

        Attributes
        ----------
        _stream : USBStream or None
            Wrapper around the reader subprocess and its shared-memory
            ring buffer (set by :meth:`connect`).
        _running : bool
            ``True`` while measurement is active (between :meth:`start`
            and :meth:`stop`).
        _streaming : bool
            ``True`` while the reader subprocess is alive (between
            :meth:`begin_stream` and :meth:`end_stream`).
        _connected : bool
            ``True`` once :meth:`connect` has succeeded.
        _buf_len : int
            Capacity of the shared-memory ring buffer in samples
            (default ~1.2 s at the configured sample rate).
        """
        self.verbose = verbose
        self._stream: Optional[USBStream] = None
        self._running = False
        self._streaming = False
        self._connected = False

        # Ring capacity in samples, sized from ring_seconds. Lives in shared
        # memory once the stream subprocess is spawned (allocated by USBStream)
        # so the parent and reader share it zero-copy.
        self._buf_len = int(SAMPLE_RATE * ring_seconds)

    # -- Connection lifecycle -------------------------------------------------

    def connect(self):
        """Probe for the Master Board and prepare a streaming session.

        Performs a quick :class:`USBBulkConnection` open/close to fail fast
        if the device is missing, then constructs a :class:`USBStream`.  The
        stream's reader subprocess is *not* spawned here — that happens in
        :meth:`begin_stream` — which avoids holding the USB device claim
        between :meth:`connect` and :meth:`begin_stream`.

        Raises
        ------
        RuntimeError
            If the Master Board is not found or the USB claim fails.
        """
        probe = USBBulkConnection(verbose=self.verbose)
        probe.close()  # release the device so the reader subprocess can claim it
        self._stream = USBStream(
            ring_capacity_samples=self._buf_len,
            verbose=self.verbose,
        )
        self._connected = True

    @property
    def connected(self) -> bool:
        """``True`` once :meth:`connect` has succeeded and resources are live."""
        return self._connected and self._stream is not None

    # -- Live buffer access (read-only) ---------------------------------------

    @property
    def buffer(self) -> np.ndarray:
        """Read-only view of the live circular sample buffer (uint16).

        Backed by shared memory written by the reader subprocess.
        """
        if self._stream is None:
            # Pre-connect: return an empty placeholder so callers don't
            # crash on attribute access.
            return np.zeros(0, dtype=np.uint16)
        return self._stream.buffer

    @property
    def buffer_head(self) -> int:
        """Current write-head index in the circular buffer."""
        if self._stream is None:
            return 0
        return self._stream.ring_head

    @property
    def buffer_capacity(self) -> int:
        """Total capacity of the circular buffer in samples."""
        return self._buf_len

    @property
    def samples_received(self) -> int:
        """Cumulative number of samples received since streaming started."""
        if self._stream is None:
            return 0
        return self._stream.ring_total

    @property
    def running(self) -> bool:
        """``True`` while the laser measurement system is active (between :meth:`start` and :meth:`stop`)."""
        return self._running

    @property
    def streaming(self) -> bool:
        """``True`` while USB data is being read into the buffer (between :meth:`begin_stream` and :meth:`end_stream`)."""
        return self._streaming

    @property
    def stream_stats(self) -> Optional[dict]:
        """Packet-level stream diagnostics when supported by firmware.

        Returns a dict with ``packets``, ``drops_host``, ``malformed``, and
        ``last_seq`` while streaming; returns ``None`` when not connected.
        """
        if self._stream is None:
            return None
        return self._stream.get_stream_stats()

    # -- State management -----------------------------------------------------

    def begin_stream(self):
        """Start reading USB bulk data into the circular buffer without turning the laser on.

        Call this after :meth:`connect` to begin collecting data immediately.
        The laser measurement system remains off until :meth:`start` is called.
        """
        self._ensure_connected()
        if not self._streaming:
            self._stream.start()
            self._streaming = True

    def end_stream(self):
        """Stop reading USB bulk data from the device.

        This does *not* send an IDLE command — use :meth:`stop` first if the
        laser is currently active.
        """
        self._streaming = False
        if self._stream:
            self._stream.stop()

    def start(self):
        """Send BOOT command to turn the laser measurement system on.

        If :meth:`begin_stream` has not been called yet, the USB data stream
        is started automatically so that samples begin arriving in the buffer.
        """
        self._ensure_connected()
        if not self._streaming:
            self.begin_stream()
        self._send(CMD_BOOT)
        self._running = True

    def stop(self):
        """Send IDLE command to turn off the laser measurement system.

        The USB data stream continues running so the buffer stays live and
        the GUI can keep displaying data. Call :meth:`end_stream` (or
        :meth:`close`) to fully halt data collection.
        """
        self._running = False
        try:
            self._send(CMD_IDLE)
        except Exception:
            pass

    def warm(self):
        """Enter the WARM / standby state. Lasers off but system still powered, allowing faster startup than BOOT."""
        self._ensure_connected()
        self._send(CMD_WARM)

    def close(self):
        """Release all hardware resources."""
        if self._running:
            self.stop()
        self.end_stream()
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        self._connected = False

    # -- Data capture ---------------------------------------------------------

    def save(self, duration_s: float = 1.0, path: Optional[Union[str, Path]] = None) -> np.ndarray:
        """Capture *duration_s* seconds of data and optionally write to disk.

        If *path* is given the raw uint16 samples are written as a binary
        file.  The captured numpy array is always returned.
        """
        samples_needed = int(SAMPLE_RATE * duration_s)
        total_samples = self.samples_received
        buf_idx = self.buffer_head

        if total_samples < samples_needed:
            raise RuntimeError(
                f"Buffer contains only {total_samples} samples "
                f"({samples_needed} needed for {duration_s}s)."
            )

        buf = self.buffer
        # Extract from circular buffer (shared memory — we copy on the way
        # out so callers can hold the result independently of the reader).
        if buf_idx >= samples_needed:
            data = buf[buf_idx - samples_needed : buf_idx].copy()
        else:
            data = np.concatenate([
                buf[self._buf_len - (samples_needed - buf_idx) :],
                buf[:buf_idx],
            ])

        if path is not None:
            data.tofile(str(path))
            if self.verbose:
                mb = len(data) * 2 / (1024 * 1024)
                print(f"Saved {len(data):,} samples ({mb:.2f} MB) to {path}")

        return data

    # -- Probe characterization -----------------------------------------------

    def run_probe_characterization(
        self,
        *,
        max_current: int = 33000,
        step_size: int = 100,
        bin_seconds: float = 0.025,
        start_offset_s: float = 0.0,
        laser_warmup_s: float = 10.0,
        boot_settle_s: float = 10.0,
        capture_guard_s: float = 0.7,
        saturation_threshold: float = 0.95,
        verbose: Optional[bool] = None,
    ) -> ProbeCharacterizationResult:
        """Run the bulk-only probe-characterization ramp and return the curve.

        Drives the 638 probe-characterization self-ramp over the **master USB
        bulk connection only** (no USB-serial connection to the 638 required).
        The host sends one bulk start command; the master stamps the SPI
        command_id; the 638 edge-detects it and self-ramps its DAC
        ``0 -> max_current`` at 40 Hz; the master streams photodetector data
        continuously; this method time-bins the stream into
        ``max_current // step_size + 1`` windows of ``bin_seconds`` each.

        Override / power / trigger sequencing is handled internally and is
        **not** part of the public API — callers only invoke this method.

        .. note::

            The capture must fit in the ring buffer in one piece. The
            :class:`Controller` must therefore be constructed with
            ``ring_seconds`` large enough for the whole sweep, e.g.::

                ctrl = Controller(ring_seconds=12)  # >= ramp + guard + margin

            This method raises a clear :class:`RuntimeError` if the ring is
            too small, quoting the required ``ring_seconds``.

        Parameters
        ----------
        max_current : int
            DAC setpoint the ramp tops out at. Must match the 638 firmware
            ``PROBE_RAMP_MAX`` (default 33000 = ``CALIBRATION_DAC_VALUE``).
        step_size : int
            DAC increment per ``bin_seconds`` tick (default 100). Must match
            firmware; the host cannot change the firmware step.
        bin_seconds : float
            One ramp tick in seconds (25 ms at the firmware 40 Hz rate).
        start_offset_s : float
            Forward-bin offset (seconds) between recording T0 and the first
            40 Hz tick taking effect. Default 0.0 is correct within ~1 bin;
            tune only if bin 0/1 clearly mix setpoints on a given setup.
        laser_warmup_s : float
            Time for the 1550 turn-on transient to settle before the 638 ramp
            begins, so the low-DAC start of the curve is flat.
        boot_settle_s : float
            638 boot / rail settle time after power-on.
        capture_guard_s : float
            Extra capture beyond the ramp duration for startup + stop latency.
        saturation_threshold : float
            Fraction of ``ADC_MAX_VALUE`` treated as photodetector saturation;
            if exceeded the sweep aborts early and the curve is trimmed.
        verbose : bool, optional
            Override ``Controller.verbose`` for this call only. Defaults to
            the controller's setting.

        Returns
        -------
        ProbeCharacterizationResult
            Binned ``current`` / ``photodetector`` curve plus the
            ``saturated`` flag and the binning parameters used.

        Raises
        ------
        RuntimeError
            If the ring buffer is too small, no samples are captured, the PD
            signal is flat, or binning produces no points.
        """
        if verbose is None:
            verbose = self.verbose

        bin_samples = int(SAMPLE_RATE * bin_seconds)   # 250_000 samples / 25 ms
        n_steps = max_current // step_size             # 330 steps (100..max)
        ramp_seconds = (n_steps + 1) * bin_seconds     # 331 ticks @ 25 ms
        capture_seconds = ramp_seconds + capture_guard_s
        # Ring must hold the whole capture + post-stop settle in one piece.
        ring_seconds_needed = capture_seconds + 1.5

        saturation_limit = int(ADC_MAX_VALUE * saturation_threshold)

        if verbose:
            print("\n" + "=" * 60)
            print("PROBE CHARACTERIZATION (BULK-ONLY)")
            print("=" * 60)
            print(f"Current range: 0 to {max_current} (step {step_size})")
            print(f"Setpoints: {n_steps + 1} | bin: {bin_seconds*1e3:.0f} ms "
                  f"({bin_samples:,} samples @ {SAMPLE_RATE/1e6:.0f} MSPS)")
            print(f"Capture: ~{capture_seconds:.2f} s "
                  f"(ring >= {ring_seconds_needed:.1f} s)")
            print("=" * 60 + "\n")

        # Ring-capacity guard: save() snapshots the last N samples, which
        # equals [T0, now] only if the ring never wrapped past T0.
        if self._buf_len < int(SAMPLE_RATE * ring_seconds_needed):
            raise RuntimeError(
                f"Ring buffer too small for this sweep: need ring_seconds >= "
                f"{ring_seconds_needed:.1f} ({int(SAMPLE_RATE * ring_seconds_needed):,} "
                f"samples), but this Controller was built with "
                f"{self._buf_len / SAMPLE_RATE:.1f} s. Reconstruct with "
                f"Controller(ring_seconds={ring_seconds_needed:.1f})."
            )

        self._ensure_connected()
        if not self._streaming:
            self.begin_stream()

        saturated = False
        ramp = None
        try:
            # -- 1. Override + power rails ------------------------------------
            if verbose:
                print("Entering override and powering rails...")
            self._send(CMD_OVERRIDE_ENTER, wValue=1)
            time.sleep(0.1)
            # 1550 nm first: power ON, trigger ON, then let it settle so its
            # turn-on transient is done before readout begins. The 638 is
            # powered after the warmup so its boot settle is unaffected.
            self._send(CMD_POWER, wValue=1, wIndex=TARGET_1550)
            self._send(CMD_TRIGGER, wValue=1, wIndex=TARGET_1550)
            if verbose:
                print(f"Waiting {laser_warmup_s:.1f} s for 1550 to settle...")
            time.sleep(laser_warmup_s)
            # 638 nm: power ON, trigger OFF -> 638 stays IDLE (DAC free for ramp)
            self._send(CMD_POWER, wValue=1, wIndex=TARGET_638)
            self._send(CMD_TRIGGER, wValue=0, wIndex=TARGET_638)
            if verbose:
                print(f"Waiting {boot_settle_s:.1f} s for 638 to boot...")
            time.sleep(boot_settle_s)

            # -- 2. Start the ramp; record sample-count origin ----------------
            t0_sample = self.samples_received
            if verbose:
                print("Starting 638 self-ramp (CMD_PROBE_CHAR start)...")
            self._send(CMD_PROBE_CHAR, wValue=1)

            # -- 3. Capture, monitoring for saturation ------------------------
            t_start = time.monotonic()
            t_end = t_start + capture_seconds
            while time.monotonic() < t_end:
                time.sleep(0.1)
                # Recent ~20 ms mean as a saturation trip (save() handles wrap).
                try:
                    recent = self.save(0.02)
                except Exception:
                    continue
                if recent.size and float(np.mean(recent)) >= saturation_limit:
                    saturated = True
                    if verbose:
                        print(f"\n⚠ Saturation detected "
                              f"({float(np.mean(recent)):.0f} counts) — "
                              f"aborting ramp early.")
                    break

            if verbose:
                print(f"Capture window: {time.monotonic() - t_start:.2f} s.")

            # -- 4. Stop the ramp (restore command_id; the 638 self-terminates
            # at max_current anyway). Post-stop sleep lets the stream settle
            # before the snapshot; binning forward-bins from T0, not the stop.
            if verbose:
                print("Stopping ramp (CMD_PROBE_CHAR stop)...")
            self._send(CMD_PROBE_CHAR, wValue=0)
            time.sleep(0.3)

            # -- 5. Extract [T0, now] from the ring --------------------------
            # save(N) returns the last N samples = [samples_received - N,
            # samples_received] = [t0_sample, now] when N = now - t0_sample.
            n_ramp = self.samples_received - t0_sample
            if n_ramp <= 0:
                raise RuntimeError("No samples captured during the ramp.")
            ramp = self.save(n_ramp / SAMPLE_RATE)
            if verbose:
                print(f"Extracted {ramp.size:,} ramp samples "
                      f"(~{ramp.size / SAMPLE_RATE:.2f} s).\n")

        except Exception:
            # Ensure the ramp is stopped even if capture/setup failed.
            try:
                self._send(CMD_PROBE_CHAR, wValue=0)
            except Exception:
                pass
            raise
        finally:
            # Always exit override (streaming is ended/closed by the caller).
            try:
                self._send(CMD_OVERRIDE_ENTER, wValue=0)
            except Exception:
                pass

        if ramp is None:
            raise RuntimeError("No ramp data captured.")

        if verbose:
            print(f"Binning {ramp.size:,} ramp samples into "
                  f"{n_steps + 1} windows...")
        binned = bin_ramp(
            ramp,
            sample_rate=SAMPLE_RATE,
            bin_seconds=bin_seconds,
            step_size=step_size,
            max_current=max_current,
            saturated=saturated,
            saturation_limit=saturation_limit,
            start_offset_samples=int(SAMPLE_RATE * start_offset_s),
        )

        return ProbeCharacterizationResult(
            current=binned["current"],
            photodetector=binned["photodetector"],
            saturated=saturated,
            sample_rate=SAMPLE_RATE,
            bin_seconds=bin_seconds,
            step_size=step_size,
            max_current=max_current,
        )

    # -- Internals ------------------------------------------------------------

    def _ensure_connected(self):
        """Raise if the USB connection is not established."""
        if not self.connected:
            raise RuntimeError("Not connected. Call connect() first.")

    def _send(self, cmd_byte, wValue=0, wIndex=0, extra=b"", timeout_ms=5000):
        """Send a command packet over USB bulk to the Master Board.

        Routes through the streaming subprocess's command queue, since that
        process owns the libusb device handle exclusively.  ``timeout_ms`` is
        accepted for API compatibility; the subprocess applies its own
        bulkWrite timeout (2 s) which is sufficient for command packets.
        """
        from ._internal.protocol import pack_command
        if self._stream is None or not self._stream.running:
            raise RuntimeError(
                "Not streaming — call begin_stream() before sending commands."
            )
        pkt = pack_command(cmd_byte, wValue, wIndex) + extra
        self._stream.send_command(pkt)
