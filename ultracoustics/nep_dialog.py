"""
Optional PyQt5 / matplotlib GUI helper for NEP peak selection.

The interactive dialog lets a user drag selection windows across one or
two thermomechanical peaks, mark spurious bands to ignore, and click to
pin the lower limit of the SHO fit window.  It then drives
:func:`ultracoustics.nep.fit_sho_log` and returns the populated analysis
dictionary.

This module is *opt-in*: PyQt5 and matplotlib are imported lazily so a
headless caller can ``import ultracoustics`` without pulling in the GUI
stack.  If either dependency is missing, :data:`InteractiveFitDialog`
is ``None`` and :data:`AVAILABLE` is ``False``.

Typical use::

    from ultracoustics.nep_dialog import InteractiveFitDialog, AVAILABLE
    if not AVAILABLE:
        raise RuntimeError("PyQt5 / matplotlib not installed")
    dlg = InteractiveFitDialog(freqs, psd_clean, psd_raw, psd_dark,
                               shot_psd, analysis_dict)
    if dlg.exec_() == QtWidgets.QDialog.Accepted:
        analysis = dlg.analysis    # peak_freq, fwhm, ignore_regions, …
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - GUI optional
    from PyQt5 import QtWidgets
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvasQTAgg as _FigureCanvas,
        NavigationToolbar2QT as _NavigationToolbar,
    )
    from matplotlib.figure import Figure
    from matplotlib.widgets import SpanSelector
    AVAILABLE = True
except Exception:  # noqa: BLE001 — any failure means GUI unavailable
    AVAILABLE = False
    InteractiveFitDialog = None  # type: ignore[assignment]
else:

    class InteractiveFitDialog(QtWidgets.QDialog):  # type: ignore[misc]
        """Drag/click dialog that produces a peak-analysis dict.

        After ``exec_() == Accepted`` the ``self.analysis`` attribute
        carries the keys consumed by :func:`ultracoustics.nep.fit_sho_log`:

        ``peak_freq``, ``peak_psd``, ``half_max``, ``f_left``, ``f_right``,
        ``fwhm``, ``fw10m``, ``span_lo``, ``span_hi``, ``fit_lo_hz``,
        ``ignore_regions`` and the ``peak2_*`` analogues for the second
        peak when present.
        """

        def __init__(self, freqs, psd_clean, psd_raw, psd_dark,
                     shot_psd_w2_hz, analysis_dict, parent=None):
            super().__init__(parent)
            self.setWindowTitle(
                "Drag windows across peaks, right-click to ignore, click for limit."
            )
            self.resize(1000, 600)
            self.setStyleSheet("QDialog { background: #1E1E1E; }")

            self.freqs = np.asarray(freqs)
            self.psd_clean = np.asarray(psd_clean)
            self.psd_raw = np.asarray(psd_raw)
            self.psd_dark = (np.asarray(psd_dark)
                             if psd_dark is not None else None)
            self.shot_psd_w2_hz = float(shot_psd_w2_hz or 0.0)
            self.analysis = dict(analysis_dict)

            self.fit_lo_hz = None
            self.ignore_regions: list[tuple[float, float]] = []
            self.ignore_patches: list = []
            self.v_limit = None
            self.peaks: list[dict] = []  # up to 2

            layout = QtWidgets.QVBoxLayout(self)
            self.fig = Figure(figsize=(8, 5), dpi=100, facecolor="#121212")
            self.canvas = _FigureCanvas(self.fig)
            self.toolbar = _NavigationToolbar(self.canvas, self)
            layout.addWidget(self.toolbar)

            ctrl = QtWidgets.QHBoxLayout()
            inst = QtWidgets.QLabel(
                "<b>Controls:</b> Left-Drag = Select Peak  |  "
                "Right-Drag = Ignore Region  |  Click = Set Fit Limit"
            )
            inst.setStyleSheet("color:#B0BEC5;")
            ctrl.addWidget(inst)
            self.btn_clear = QtWidgets.QPushButton("Clear Custom Limits")
            self.btn_clear.setStyleSheet("background:#455A64; color:white;")
            self.btn_clear.clicked.connect(self._on_clear_masks)
            ctrl.addStretch()
            ctrl.addWidget(self.btn_clear)
            layout.addLayout(ctrl)
            layout.addWidget(self.canvas, stretch=1)

            self.lbl_warning = QtWidgets.QLabel("Select at least one peak.")
            self.lbl_warning.setStyleSheet(
                "color:#FFA726; font-weight:bold;"
            )
            btns = QtWidgets.QHBoxLayout()
            btns.addWidget(self.lbl_warning)
            btns.addStretch()
            self.btn_cancel = QtWidgets.QPushButton("✕ Cancel")
            self.btn_cancel.setStyleSheet(
                "background:#D32F2F; color:white; font-weight:bold;"
                " padding:8px 20px; border-radius:4px;"
            )
            self.btn_cancel.clicked.connect(self.reject)
            btns.addWidget(self.btn_cancel)
            self.btn_accept = QtWidgets.QPushButton("✓ Accept")
            self.btn_accept.setStyleSheet(
                "background:#388E3C; color:white; font-weight:bold;"
                " padding:8px 20px; border-radius:4px;"
            )
            self.btn_accept.clicked.connect(self._on_accept)
            self.btn_accept.setEnabled(False)
            btns.addWidget(self.btn_accept)
            layout.addLayout(btns)

            self.ax = self.fig.add_subplot(111, facecolor="#1E1E1E")
            self._setup_plot()

            self.span_peak = SpanSelector(
                self.ax, self._on_select_peak, 'horizontal',
                useblit=False, button=1,
                props=dict(alpha=0.3, facecolor="#81C784"),
                interactive=False,
            )
            self.span_ignore = SpanSelector(
                self.ax, self._on_select_ignore, 'horizontal',
                useblit=False, button=3,
                props=dict(alpha=0.3, facecolor="#E57373"),
                interactive=False,
            )
            self.fig.canvas.mpl_connect('button_press_event', self._on_press)
            self.fig.canvas.mpl_connect('button_release_event', self._on_release)
            self._press_x = None

        # ----- mouse plumbing ----------------------------------------
        def _on_press(self, event):
            if event.inaxes == self.ax:
                self._press_x = event.xdata

        def _on_release(self, event):
            if event.inaxes == self.ax and self._press_x is not None:
                if event.button == 1:
                    span_range = self.ax.get_xlim()[1] - self.ax.get_xlim()[0]
                    dx = abs(event.xdata - self._press_x)
                    if dx < span_range * 0.005:
                        self.fit_lo_hz = event.xdata * 1e6
                        if self.v_limit:
                            self.v_limit.remove()
                        self.v_limit = self.ax.axvline(
                            event.xdata, color="#4DD0E1", lw=2, zorder=8,
                        )
                        self.canvas.draw_idle()
                self._press_x = None

        def _on_clear_masks(self):
            self.fit_lo_hz = None
            self.ignore_regions = []
            if self.v_limit:
                self.v_limit.remove()
                self.v_limit = None
            for p in self.ignore_patches:
                p.remove()
            self.ignore_patches = []
            self.peaks.clear()
            for art in (self.p_peak, self.p_left, self.p_right,
                        self.p_peak2, self.p_left2, self.p_right2):
                art.set_data([], [])
            self.btn_accept.setEnabled(False)
            self.lbl_warning.setText("Select at least one peak.")
            self.lbl_warning.setStyleSheet(
                "color:#FFA726; font-weight:bold;"
            )
            self.canvas.draw_idle()

        # ----- plot setup --------------------------------------------
        def _setup_plot(self):
            f_mhz = self.freqs / 1e6
            self.ax.plot(f_mhz, np.maximum(self.psd_raw, 1e-35),
                         color="#90CAF9", lw=0.6, alpha=0.3,
                         label="Raw PSD")
            if self.psd_dark is not None:
                self.ax.plot(f_mhz, np.maximum(self.psd_dark, 1e-35),
                             "--", color="#FFB74D", lw=0.6, alpha=0.4,
                             label="Dark PSD")
            self.ax.plot(f_mhz, np.maximum(self.psd_clean, 1e-35),
                         color="#4FC3F7", lw=1.2, alpha=0.9,
                         label="Clean PSD")
            if self.shot_psd_w2_hz > 0:
                self.ax.axhline(self.shot_psd_w2_hz, color="#E0E0E0",
                                ls=":", lw=1.0, alpha=0.5,
                                label="Shot Noise")

            self.ax.set_yscale("log")
            self.ax.set_title(
                "Drag windows across peaks, right-click to ignore, "
                "click for limit.", color="white", fontweight="bold")
            self.ax.set_xlabel("Frequency (MHz)", color="#B0BEC5")
            self.ax.set_ylabel("PSD (W²/Hz)", color="#B0BEC5")
            self.ax.grid(True, alpha=0.15, color="#546E7A", which="both")
            self.ax.tick_params(colors="#B0BEC5")
            self.ax.legend(loc="upper right", framealpha=0.6,
                           edgecolor="#455A64", facecolor="#1E1E1E",
                           fontsize=8)
            for s in self.ax.spines.values():
                s.set_color("#455A64")
            self.ax.set_xlim(0, f_mhz[-1])

            p_safe = np.maximum(self.psd_clean, 1e-35)
            valid = p_safe[p_safe > 1e-35]
            ymin = (max(1e-35, np.percentile(valid, 5) * 0.5)
                    if valid.size else 1e-35)
            self.ax.set_ylim(bottom=ymin)

            self.p_peak,  = self.ax.plot([], [], "v", color="#EF5350",
                                         ms=12, zorder=10,
                                         markeredgecolor="white")
            self.p_left,  = self.ax.plot([], [], "s", color="#81C784",
                                         ms=10, zorder=10,
                                         markeredgecolor="white")
            self.p_right, = self.ax.plot([], [], "s", color="#81C784",
                                         ms=10, zorder=10,
                                         markeredgecolor="white")
            self.p_peak2, = self.ax.plot([], [], "v", color="#BA68C8",
                                         ms=12, zorder=10,
                                         markeredgecolor="white")
            self.p_left2, = self.ax.plot([], [], "s", color="#BA68C8",
                                         ms=10, zorder=10,
                                         markeredgecolor="white")
            self.p_right2,= self.ax.plot([], [], "s", color="#BA68C8",
                                         ms=10, zorder=10,
                                         markeredgecolor="white")

        # ----- selection callbacks -----------------------------------
        def _on_select_ignore(self, xmin, xmax):
            if abs(xmax - xmin) > 0.01:
                self.ignore_regions.append((xmin * 1e6, xmax * 1e6))
                p = self.ax.axvspan(xmin, xmax, color="#E57373",
                                    alpha=0.3, zorder=7)
                self.ignore_patches.append(p)
            self.span_ignore.extents = (0, 0)
            self.canvas.draw_idle()

        def _on_select_peak(self, xmin, xmax):
            f_mhz = self.freqs / 1e6
            idx_min = int(np.searchsorted(f_mhz, xmin))
            idx_max = int(np.searchsorted(f_mhz, xmax))
            if idx_min >= idx_max - 2:
                self.lbl_warning.setText("Window too narrow.")
                self.lbl_warning.setStyleSheet(
                    "color:#EF5350; font-weight:bold;")
                return

            span_psd = self.psd_clean[idx_min:idx_max]
            if span_psd.size == 0:
                return

            local = int(np.argmax(span_psd))
            peak_i = idx_min + local

            # parabolic interpolation for sub-bin peak
            if 0 < local < span_psd.size - 1:
                a = span_psd[local - 1]
                b = span_psd[local]
                c = span_psd[local + 1]
                denom = a - 2.0 * b + c
                if abs(denom) > 1e-12:
                    p = 0.5 * (a - c) / denom
                    df = (self.freqs[1] - self.freqs[0]
                          if self.freqs.size > 1 else 0.0)
                    peak_freq = self.freqs[peak_i] + p * df
                    peak_psd = b - 0.25 * (a - c) * p
                else:
                    peak_freq = self.freqs[peak_i]
                    peak_psd = self.psd_clean[peak_i]
            else:
                peak_freq = self.freqs[peak_i]
                peak_psd = self.psd_clean[peak_i]

            half = peak_psd / 2.0
            tenth = peak_psd * 0.10

            # half-max walk (constrained to drag window)
            f_left = f_right = None
            for i in range(max(peak_i, idx_min), idx_min - 1, -1):
                if self.psd_clean[i] <= half:
                    d = self.psd_clean[i + 1] - self.psd_clean[i]
                    frac = (half - self.psd_clean[i]) / d if d != 0 else 0.0
                    f_left = self.freqs[i] + frac * (
                        self.freqs[i + 1] - self.freqs[i])
                    break
            for i in range(min(peak_i, idx_max - 1), idx_max):
                if self.psd_clean[i] <= half:
                    d = self.psd_clean[i] - self.psd_clean[i - 1]
                    frac = (half - self.psd_clean[i - 1]) / d if d != 0 else 0.0
                    f_right = self.freqs[i - 1] + frac * (
                        self.freqs[i] - self.freqs[i - 1])
                    break
            if f_left is None or f_right is None:
                self.lbl_warning.setText(
                    "Widen the window to include both half-max crossings.")
                self.lbl_warning.setStyleSheet(
                    "color:#EF5350; font-weight:bold;")
                return

            # tenth-max walk for FW10M
            f_l10 = f_r10 = None
            sz_min = max(0, idx_min - (idx_max - idx_min))
            for i in range(max(peak_i, sz_min), sz_min - 1, -1):
                if self.psd_clean[i] <= tenth:
                    d = self.psd_clean[i + 1] - self.psd_clean[i]
                    frac = (tenth - self.psd_clean[i]) / d if d != 0 else 0.0
                    f_l10 = self.freqs[i] + frac * (
                        self.freqs[i + 1] - self.freqs[i])
                    break
            sz_max = min(self.psd_clean.size, idx_max + (idx_max - idx_min))
            for i in range(min(peak_i, sz_max - 1), sz_max):
                if self.psd_clean[i] <= tenth:
                    d = self.psd_clean[i] - self.psd_clean[i - 1]
                    frac = (tenth - self.psd_clean[i - 1]) / d if d != 0 else 0.0
                    f_r10 = self.freqs[i - 1] + frac * (
                        self.freqs[i] - self.freqs[i - 1])
                    break
            fw10m = ((f_r10 - f_l10)
                     if (f_l10 is not None and f_r10 is not None) else None)

            peak_data = dict(peak_freq=float(peak_freq),
                             peak_psd=float(peak_psd),
                             half_max=float(half),
                             f_left=float(f_left),
                             f_right=float(f_right),
                             fwhm=float(f_right - f_left),
                             fw10m=fw10m,
                             span_lo=float(xmin * 1e6),
                             span_hi=float(xmax * 1e6))

            if len(self.peaks) == 2:
                d1 = abs(self.peaks[0]["peak_freq"] - peak_data["peak_freq"])
                d2 = abs(self.peaks[1]["peak_freq"] - peak_data["peak_freq"])
                if d1 < d2:
                    self.peaks[0] = peak_data
                else:
                    self.peaks[1] = peak_data
            else:
                self.peaks.append(peak_data)
            self.peaks.sort(key=lambda p: p["peak_freq"])

            self.btn_accept.setEnabled(True)
            self.lbl_warning.setText(
                f"Valid logic: {len(self.peaks)} peak(s) found.")
            self.lbl_warning.setStyleSheet(
                "color:#81C784; font-weight:bold;")

            p1 = self.peaks[0]
            self.p_peak.set_data([p1["peak_freq"] / 1e6], [p1["peak_psd"]])
            self.p_left.set_data([p1["f_left"] / 1e6], [p1["half_max"]])
            self.p_right.set_data([p1["f_right"] / 1e6], [p1["half_max"]])

            if len(self.peaks) > 1:
                p2 = self.peaks[1]
                self.p_peak2.set_data([p2["peak_freq"] / 1e6],
                                      [p2["peak_psd"]])
                self.p_left2.set_data([p2["f_left"] / 1e6],
                                      [p2["half_max"]])
                self.p_right2.set_data([p2["f_right"] / 1e6],
                                       [p2["half_max"]])
            else:
                self.p_peak2.set_data([], [])
                self.p_left2.set_data([], [])
                self.p_right2.set_data([], [])

            self.span_peak.extents = (0, 0)
            self.canvas.draw_idle()

        def _on_accept(self):
            self.analysis["fit_lo_hz"] = self.fit_lo_hz
            self.analysis["ignore_regions"] = list(self.ignore_regions)
            if self.peaks:
                self.analysis.update(self.peaks[0])
            if len(self.peaks) > 1:
                for k, v in self.peaks[1].items():
                    self.analysis[f"peak2_{k}"] = v
            self.accept()


__all__ = ["AVAILABLE", "InteractiveFitDialog"]
