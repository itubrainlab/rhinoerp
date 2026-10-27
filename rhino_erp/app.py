"""Rhino ERP: a desktop app for offline ERP analysis of BDF recordings.

Workflow (left panel, top to bottom):
  1. Open a .bdf file (MNE).
  2. Select channels, set their types and assign electrode labels.
  3. Pick the trigger channel, then tick and name the events to analyse.
  4. Filter (g.Pype pipeline).
  5. Epoch, average and inspect the ERPs.
"""

from __future__ import annotations

import sys
import traceback
from contextlib import contextmanager
from pathlib import Path

import matplotlib
import mne
import numpy as np
import pyqtgraph as pg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QIcon
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gpype.frontend import theme as gp_theme

from . import processing as proc
from .script_export import AnalysisConfig, Component, build_script

COLORS = [
    "#4e9cf5", "#f5a54e", "#5fd38d", "#f5576c", "#b07cf5",
    "#f5e04e", "#4ef5e6", "#f54ec8", "#a3a3a3", "#8cf54e",
]

#: Tab indices
TAB_SIGNALS, TAB_EPOCHS, TAB_STATS, TAB_MONTAGE = range(4)

CUSTOM = "Custom"


def _spin(lo, hi, value, step=1.0, decimals=2, suffix="", special=None):
    box = QDoubleSpinBox()
    box.setRange(lo, hi)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    box.setValue(value)
    if suffix:
        box.setSuffix(f" {suffix}")
    if special:
        box.setSpecialValueText(special)  # shown at the minimum
    return box


class InputError(ValueError):
    """A problem with the user's settings: shown in a dialog, not logged."""


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, height=3.0):
        self.figure = Figure(figsize=(5, height), layout="constrained")
        super().__init__(self.figure)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rhino ERP")
        self.resize(1400, 900)

        self.raw_file: mne.io.BaseRaw | None = None  # as read from disk
        self.path: str | None = None
        self.raw_orig: mne.io.BaseRaw | None = None  # selected channels, positions
        self.raw_pre: mne.io.BaseRaw | None = None  # filtered
        self.events: np.ndarray | None = None
        self.epochs: mne.Epochs | None = None
        self.evokeds: dict[str, mne.Evoked] = {}
        self.stats: proc.ComparisonResult | None = None
        self.latency: float = 0.3
        self._stats_requested = False  # rerun statistics when the ERPs change
        self._electrode_labels = [""] + proc.electrode_labels()

        splitter = QSplitter()
        splitter.addWidget(self._build_controls())
        splitter.addWidget(self._build_views())
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([460, 940])
        self.setCentralWidget(splitter)
        self.statusBar().showMessage("Open a BDF file to start.")
        self._update_enabled()

    # ------------------------------------------------------------ layout

    def _build_controls(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 1. File
        box = QGroupBox("1. Recording")
        lay = QVBoxLayout(box)
        self.btn_open = QPushButton("Open BDF…")
        self.btn_open.clicked.connect(self.open_file)
        self.lbl_file = QLabel("No file loaded")
        self.lbl_file.setWordWrap(True)
        lay.addWidget(self.btn_open)
        lay.addWidget(self.lbl_file)
        layout.addWidget(box)

        # 2. Channels and electrodes
        box = QGroupBox("2. Channels and electrodes")
        lay = QVBoxLayout(box)
        self.tbl_chans = QTableWidget(0, 3)
        self.tbl_chans.setHorizontalHeaderLabels(["Channel", "Type", "Electrode"])
        self.tbl_chans.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tbl_chans.verticalHeader().setVisible(False)
        self.tbl_chans.setMinimumHeight(240)
        self.tbl_chans.setToolTip(
            "Tick the channels to visualise and analyse. Electrode assigns a 10-05 "
            "position (type to search); leave it empty for no position.")
        lay.addWidget(self.tbl_chans)
        row = QHBoxLayout()
        for label, state in (("All", True), ("None", False)):
            btn = QPushButton(label)
            btn.clicked.connect(lambda _=False, s=state: self._check_all_channels(s))
            row.addWidget(btn)
        btn = QPushButton("Unicorn defaults")
        btn.setToolTip("First 8 channels as EEG: Fz, C3, Cz, C4, Pz, PO7, Oz, PO8")
        btn.clicked.connect(self._reset_channel_table)
        row.addWidget(btn)
        lay.addLayout(row)
        self.btn_chans = QPushButton("Apply channel selection")
        self.btn_chans.clicked.connect(self.apply_channels)
        lay.addWidget(self.btn_chans)
        self.lbl_chans = QLabel("")
        self.lbl_chans.setWordWrap(True)
        lay.addWidget(self.lbl_chans)
        layout.addWidget(box)

        # 3. Reference and filtering
        box = QGroupBox("3. Reference (optional) and filtering (g.Pype)")
        lay = QVBoxLayout(box)
        form = QFormLayout()
        self.cmb_ref = QComboBox()
        self.cmb_ref.setToolTip("Keep the recording reference, use the average of all "
                                "selected EEG channels, or one EEG channel.")
        self.cmb_ref2 = QComboBox()
        self.cmb_ref2.setToolTip("Optional second reference channel; the mean of both "
                                 "is used (e.g. linked mastoids).")
        self.cmb_ref.currentTextChanged.connect(
            lambda text: self.cmb_ref2.setEnabled(
                text not in ("", proc.REF_NONE, proc.REF_AVERAGE)))
        form.addRow("Reference", self.cmb_ref)
        form.addRow("… averaged with", self.cmb_ref2)
        self.spn_hp = _spin(0, 10, 0.1, 0.05, 2, "Hz", "off")
        self.spn_lp = _spin(0, 200, 30, 1, 1, "Hz", "off")
        self.cmb_notch = QComboBox()
        self.cmb_notch.addItems(["off", "50 Hz", "60 Hz"])
        self.cmb_notch.setCurrentIndex(1)
        self.spn_order = QSpinBox()
        self.spn_order.setRange(1, 8)
        self.spn_order.setValue(4)
        form.addRow("High-pass", self.spn_hp)
        form.addRow("Low-pass", self.spn_lp)
        form.addRow("Line noise", self.cmb_notch)
        form.addRow("Order", self.spn_order)
        lay.addLayout(form)
        self.btn_filter = QPushButton("Apply reference + filter")
        self.btn_filter.clicked.connect(self.preprocess)
        lay.addWidget(self.btn_filter)
        layout.addWidget(box)

        # 4. Trigger and events
        box = QGroupBox("4. Trigger and events")
        lay = QVBoxLayout(box)
        form = QFormLayout()
        self.cmb_trigger = QComboBox()
        self.cmb_trigger.setToolTip("Channel carrying the trigger codes, or the "
                                    "file's annotations.")
        self.spn_mask = QSpinBox()
        self.spn_mask.setRange(0, 0xFFFFFF)
        self.spn_mask.setDisplayIntegerBase(16)
        self.spn_mask.setPrefix("0x")
        self.spn_mask.setValue(0xFFFF)
        self.spn_mask.setToolTip("Bit mask applied to the trigger codes "
                                 "(BioSemi uses the lower 16 bits; 0 = no mask).")
        form.addRow("Trigger channel", self.cmb_trigger)
        form.addRow("Trigger mask", self.spn_mask)
        # EEG delay: slider for quick tuning, spin box for exact values.
        self.sld_delay = QSlider(Qt.Horizontal)
        self.sld_delay.setRange(-500, 500)
        self.sld_delay.setTickInterval(100)
        self.sld_delay.setTickPosition(QSlider.TicksBelow)
        self.spn_delay = QSpinBox()
        self.spn_delay.setRange(-500, 500)
        self.spn_delay.setSuffix(" ms")
        tip = ("Recording delay of the EEG relative to the triggers. Positive: the EEG "
               "lags (events are moved later). Applied to epochs and event markers; "
               "ERPs update automatically.")
        self.sld_delay.setToolTip(tip)
        self.spn_delay.setToolTip(tip)
        self.sld_delay.valueChanged.connect(self.spn_delay.setValue)
        self.spn_delay.valueChanged.connect(self.sld_delay.setValue)
        self._delay_timer = QTimer(self)
        self._delay_timer.setSingleShot(True)
        self._delay_timer.setInterval(300)
        self._delay_timer.timeout.connect(self._delay_changed)
        self.spn_delay.valueChanged.connect(lambda _: self._delay_timer.start())
        drow = QHBoxLayout()
        drow.addWidget(self.sld_delay, 1)
        drow.addWidget(self.spn_delay)
        form.addRow("EEG delay", drow)
        lay.addLayout(form)
        self.btn_events = QPushButton("Find events")
        self.btn_events.clicked.connect(self.find_events)
        lay.addWidget(self.btn_events)
        self.tbl_events = QTableWidget(0, 3)
        self.tbl_events.setHorizontalHeaderLabels(["Code", "Count", "Condition name"])
        self.tbl_events.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.tbl_events.verticalHeader().setVisible(False)
        self.tbl_events.setMinimumHeight(140)
        self.tbl_events.setToolTip("Tick the events to analyse; edit names to "
                                   "label conditions. Events sharing a name are pooled.")
        lay.addWidget(self.tbl_events)
        layout.addWidget(box)

        # 5. ERPs
        box = QGroupBox("5. Epochs and ERPs")
        lay = QVBoxLayout(box)
        form = QFormLayout()
        self.spn_tmin = _spin(-2, 0, -0.2, 0.05, 3, "s")
        self.spn_tmax = _spin(0.05, 5, 0.8, 0.05, 3, "s")
        self.chk_baseline = QCheckBox("tmin … 0 s")
        self.chk_baseline.setChecked(True)
        self.spn_reject = _spin(0, 1000, 100, 10, 0, "µV", "off")
        form.addRow("tmin", self.spn_tmin)
        form.addRow("tmax", self.spn_tmax)
        form.addRow("Baseline", self.chk_baseline)
        form.addRow("Reject p-p >", self.spn_reject)
        lay.addLayout(form)
        self.btn_erp = QPushButton("Compute ERPs")
        self.btn_erp.clicked.connect(self.compute_erps)
        self.btn_save = QPushButton("Save ERPs (.fif)…")
        self.btn_save.clicked.connect(self.save_erps)
        lay.addWidget(self.btn_erp)
        lay.addWidget(self.btn_save)
        self.btn_script = QPushButton("Export as MNE script…")
        self.btn_script.setToolTip("Write a standalone Python/MNE script that reproduces "
                                   "the analysis configured here.")
        self.btn_script.clicked.connect(self.export_script)
        lay.addWidget(self.btn_script)
        self.lbl_counts = QLabel("")
        self.lbl_counts.setWordWrap(True)
        lay.addWidget(self.lbl_counts)
        layout.addWidget(box)
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(440)
        return scroll

    def _build_views(self) -> QWidget:
        self.tabs = QTabWidget()

        # Signals tab
        w = QWidget()
        lay = QVBoxLayout(w)
        bar = QHBoxLayout()
        self.cmb_stage = QComboBox()
        self.cmb_stage.addItems(["Raw", "Filtered"])
        self.cmb_stage.currentIndexChanged.connect(self._plot_signals)
        self.spn_window = _spin(1, 60, 10, 1, 0, "s")
        self.spn_window.valueChanged.connect(self._plot_signals)
        self.spn_scale = _spin(5, 2000, 100, 10, 0, "µV/div")
        self.spn_scale.valueChanged.connect(self._plot_signals)
        for label, wid in (("Stage", self.cmb_stage), ("Window", self.spn_window),
                           ("Scale", self.spn_scale)):
            bar.addWidget(QLabel(label))
            bar.addWidget(wid)
        bar.addStretch(1)
        lay.addLayout(bar)
        self.plt_signals = pg.PlotWidget()
        self.plt_signals.setLabel("bottom", "Time", "s")
        self.plt_signals.showGrid(x=True, alpha=0.2)
        self.plt_signals.setMouseEnabled(x=False, y=False)
        lay.addWidget(self.plt_signals, 1)
        self.sld_time = QSlider(Qt.Horizontal)
        self.sld_time.valueChanged.connect(self._plot_signals)
        lay.addWidget(self.sld_time)
        self.tabs.addTab(w, "Signals")

        # ERP tab
        w = QWidget()
        lay = QVBoxLayout(w)
        bar = QHBoxLayout()
        self.cmb_view = QComboBox()
        self.cmb_view.addItems(["Single channel", "Mean of channels", "GFP", "Butterfly"])
        self.cmb_view.currentIndexChanged.connect(self._plot_erps)
        self.lst_chan = QListWidget()
        self.lst_chan.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.lst_chan.itemSelectionChanged.connect(self._plot_erps)
        self.chk_invert = QCheckBox("Negative up")
        self.chk_invert.toggled.connect(self._plot_erps)
        bar.addWidget(QLabel("View"))
        bar.addWidget(self.cmb_view)
        bar.addWidget(self.chk_invert)
        bar.addStretch(1)
        bar.addWidget(QLabel("Click the plot to set the topomap latency"))
        lay.addLayout(bar)
        body = QSplitter()
        self.lst_chan.setMaximumWidth(110)
        body.addWidget(self.lst_chan)
        right = QSplitter(Qt.Vertical)
        self.plt_erp = pg.PlotWidget()
        self.plt_erp.setLabel("bottom", "Time", "s")
        self.plt_erp.setLabel("left", "Amplitude", "µV")
        self.plt_erp.showGrid(x=True, y=True, alpha=0.2)
        self.plt_erp.addLegend(offset=(10, 10))
        self.plt_erp.scene().sigMouseClicked.connect(self._erp_clicked)
        right.addWidget(self.plt_erp)
        self.cnv_topo = MplCanvas(2.6)
        right.addWidget(self.cnv_topo)
        right.setSizes([520, 260])
        body.addWidget(right)
        body.setStretchFactor(1, 1)
        lay.addWidget(body, 1)
        self.tabs.addTab(w, "Epochs")

        self.tabs.addTab(self._build_stats_tab(), "Statistics")

        # Montage tab
        self.cnv_montage = MplCanvas(5)
        self.tabs.addTab(self.cnv_montage, "Montage")
        return self.tabs

    def _build_stats_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        # Component definition and comparison
        cfg = QGridLayout()
        self.cmb_comp = QComboBox()
        self.cmb_comp.addItems(list(proc.COMPONENT_PRESETS) + [CUSTOM])
        self.cmb_comp.setToolTip("Preset windows and sites are conventional starting "
                                 "points; adjust them to your paradigm.")
        self.edt_comp = QLineEdit()
        self.edt_comp.setPlaceholderText("component name")
        self.spn_win_lo = QSpinBox()
        self.spn_win_hi = QSpinBox()
        for spn in (self.spn_win_lo, self.spn_win_hi):
            spn.setRange(-2000, 5000)
            spn.setSingleStep(10)
            spn.setSuffix(" ms")
        self.cmb_polarity = QComboBox()
        self.cmb_polarity.addItems(["negative", "positive"])
        self.cmb_cond_a = QComboBox()
        self.cmb_cond_b = QComboBox()
        self.spn_perm = QSpinBox()
        self.spn_perm.setRange(500, 20000)
        self.spn_perm.setSingleStep(500)
        self.spn_perm.setValue(5000)
        self.spn_perm.setToolTip("Label shuffles for the permutation test of the window "
                                 "mean amplitude.")
        self.btn_stats = QPushButton("Run statistics")
        self.btn_stats.clicked.connect(lambda: self.run_stats())
        self.btn_export = QPushButton("Export CSV…")
        self.btn_export.clicked.connect(self.export_stats)

        def pair(*widgets):
            box = QHBoxLayout()
            for wid in widgets:
                box.addWidget(wid) if isinstance(wid, QWidget) else box.addWidget(QLabel(wid))
            return box

        cfg.addWidget(QLabel("Component"), 0, 0)
        cfg.addLayout(pair(self.cmb_comp, self.edt_comp), 0, 1)
        cfg.addWidget(QLabel("Window"), 0, 2)
        cfg.addLayout(pair(self.spn_win_lo, "to", self.spn_win_hi), 0, 3)
        cfg.addWidget(QLabel("Polarity"), 0, 4)
        cfg.addWidget(self.cmb_polarity, 0, 5)
        cfg.addWidget(QLabel("Compare"), 1, 0)
        cfg.addLayout(pair(self.cmb_cond_a, "vs", self.cmb_cond_b), 1, 1)
        cfg.addWidget(QLabel("Permutations"), 1, 2)
        cfg.addWidget(self.spn_perm, 1, 3)
        cfg.addLayout(pair(self.btn_stats, self.btn_export), 1, 4, 1, 2)
        cfg.setColumnStretch(1, 1)
        lay.addLayout(cfg)

        self.cmb_comp.currentTextChanged.connect(self._apply_preset)
        for spn in (self.spn_win_lo, self.spn_win_hi):
            spn.valueChanged.connect(self._component_edited)
        self.cmb_polarity.currentIndexChanged.connect(self._component_edited)

        body = QSplitter()
        self.lst_stat_chan = QListWidget()
        self.lst_stat_chan.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.lst_stat_chan.setToolTip("Channels averaged into the component measure.")
        self.lst_stat_chan.setMaximumWidth(110)
        self.lst_stat_chan.itemSelectionChanged.connect(self._component_edited)
        body.addWidget(self.lst_stat_chan)
        right = QSplitter(Qt.Vertical)
        self.plt_stats = pg.PlotWidget()
        self.plt_stats.setLabel("bottom", "Time", "s")
        self.plt_stats.setLabel("left", "Amplitude", "µV")
        self.plt_stats.showGrid(x=True, y=True, alpha=0.2)
        self.plt_stats.addLegend(offset=(10, 10))
        right.addWidget(self.plt_stats)
        bottom = QWidget()
        blay = QVBoxLayout(bottom)
        blay.setContentsMargins(0, 0, 0, 0)
        self.tbl_stats = QTableWidget(0, 4)
        self.tbl_stats.setHorizontalHeaderLabels(["Measure", "A", "B", "A − B / test"])
        header = self.tbl_stats.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setStretchLastSection(True)
        self.tbl_stats.verticalHeader().setVisible(False)
        self.tbl_stats.setEditTriggers(QAbstractItemView.NoEditTriggers)
        blay.addWidget(self.tbl_stats)
        self.lbl_stats = QLabel("Compute ERPs, choose a component and two conditions, "
                                "then run statistics.")
        self.lbl_stats.setWordWrap(True)
        blay.addWidget(self.lbl_stats)
        right.addWidget(bottom)
        right.setSizes([440, 300])
        body.addWidget(right)
        body.setStretchFactor(1, 1)
        lay.addWidget(body, 1)
        self._apply_preset(self.cmb_comp.currentText())
        return w

    def _apply_preset(self, name: str):
        preset = proc.COMPONENT_PRESETS.get(name)
        if preset is None:  # Custom: keep the current settings
            return
        for wid in (self.spn_win_lo, self.spn_win_hi, self.cmb_polarity,
                    self.lst_stat_chan):
            wid.blockSignals(True)
        self.edt_comp.setText(preset.name)
        self.spn_win_lo.setValue(round(preset.tmin * 1000))
        self.spn_win_hi.setValue(round(preset.tmax * 1000))
        self.cmb_polarity.setCurrentIndex(0 if preset.polarity < 0 else 1)
        self._select_stat_channels(preset.channels)
        for wid in (self.spn_win_lo, self.spn_win_hi, self.cmb_polarity,
                    self.lst_stat_chan):
            wid.blockSignals(False)

    def _select_stat_channels(self, wanted):
        """Select the wanted channels that exist; keep the selection if none do."""
        names = [self.lst_stat_chan.item(i).text() for i in range(self.lst_stat_chan.count())]
        if not set(wanted) & set(names):
            return
        for i, name in enumerate(names):
            self.lst_stat_chan.item(i).setSelected(name in wanted)

    def _component_edited(self, *_):
        """A hand edit turns the preset into a custom component."""
        if self.cmb_comp.currentText() != CUSTOM:
            self.cmb_comp.blockSignals(True)
            self.cmb_comp.setCurrentText(CUSTOM)
            self.cmb_comp.blockSignals(False)

    def _populate_stats_inputs(self):
        """Offer the computed conditions and EEG channels, keeping choices."""
        conds = list(self.evokeds)
        for cmb, default in ((self.cmb_cond_a, 0), (self.cmb_cond_b, 1)):
            keep = cmb.currentText()
            cmb.clear()
            cmb.addItems(conds)
            if keep in conds:
                cmb.setCurrentText(keep)
            elif conds:
                cmb.setCurrentIndex(min(default, len(conds) - 1))
        eeg = [self.epochs.ch_names[i] for i in mne.pick_types(self.epochs.info, eeg=True)]
        keep = {i.text() for i in self.lst_stat_chan.selectedItems()}
        self.lst_stat_chan.blockSignals(True)
        self.lst_stat_chan.clear()
        self.lst_stat_chan.addItems(eeg)
        preset = proc.COMPONENT_PRESETS.get(self.cmb_comp.currentText())
        wanted = keep & set(eeg) or (set(preset.channels) if preset else set())
        for i in range(self.lst_stat_chan.count()):
            item = self.lst_stat_chan.item(i)
            item.setSelected(item.text() in wanted)
        self.lst_stat_chan.blockSignals(False)

    def _component(self) -> proc.Component:
        channels = tuple(i.text() for i in self.lst_stat_chan.selectedItems())
        if not channels:
            raise InputError("Select at least one channel for the component.")
        return proc.Component(
            name=self.edt_comp.text().strip() or "component",
            tmin=self.spn_win_lo.value() / 1000,
            tmax=self.spn_win_hi.value() / 1000,
            channels=channels,
            polarity=-1 if self.cmb_polarity.currentText() == "negative" else 1,
        )

    def run_stats(self, quiet: bool = False):
        if not self.evokeds:
            if not quiet:
                QMessageBox.information(self, "No ERPs", "Compute ERPs in step 5 first.")
            return
        with self._busy("Running ERP statistics…"):
            a, b = self.cmb_cond_a.currentText(), self.cmb_cond_b.currentText()
            if a == b:
                raise InputError("Choose two different conditions to compare.")
            comp = self._component()
            if comp.tmin >= comp.tmax:
                raise InputError("The window must start before it ends.")
            self.stats = proc.compare_conditions(self.epochs, a, b, comp,
                                                 n_perm=self.spn_perm.value())
            self._stats_requested = True
            if not quiet:
                self.tabs.setCurrentIndex(TAB_STATS)
            self.statusBar().showMessage(
                f"{comp.name}: {a} − {b} = {self.stats.diff:+.2f} µV, "
                f"p = {self.stats.p_welch:.3g} (Welch), {self.stats.p_perm:.3g} (perm.)")
        self._show_stats()

    def _show_stats(self):
        res = self.stats
        plot = self.plt_stats
        plot.clear()
        self.tbl_stats.setRowCount(0)
        if res is None:
            return
        comp = res.component
        names = list(self.evokeds)
        # Colour follows the condition, as on the ERP tab.
        for s, erp, se in ((res.a, res.erp_a, res.se_a), (res.b, res.erp_b, res.se_b)):
            color = QColor(COLORS[names.index(s.name) % len(COLORS)] if s.name in names
                           else COLORS[0])
            band = QColor(color)
            band.setAlpha(60)
            # Bare curves: a PlotDataItem without a pen has no path to fill between.
            hi = pg.PlotCurveItem(res.times, erp + se, pen=pg.mkPen(None))
            lo = pg.PlotCurveItem(res.times, erp - se, pen=pg.mkPen(None))
            plot.addItem(pg.FillBetweenItem(hi, lo, brush=band))
            plot.plot(res.times, erp, pen=pg.mkPen(color, width=2),
                      name=f"{s.name} (n={s.n})")
            if not s.peak_at_edge:
                plot.plot([s.peak_latency], [s.peak_amp], pen=None, symbol="o",
                          symbolSize=8, symbolBrush=color, symbolPen=pg.mkPen("#1a1a19"))
        region = pg.LinearRegionItem((comp.tmin, comp.tmax), movable=False,
                                     brush=QColor(255, 255, 255, 25))
        region.setZValue(-10)
        plot.addItem(region)
        plot.addItem(pg.InfiniteLine(0, pen=pg.mkPen("#888", style=Qt.DashLine)))
        plot.addItem(pg.InfiniteLine(0, angle=0, pen=pg.mkPen("#666")))
        plot.setTitle(f"{comp.name} at {'+'.join(comp.channels)}, "
                      f"{comp.tmin*1000:.0f}–{comp.tmax*1000:.0f} ms (shaded)")

        def fmt_lat(s):
            return f"{s.peak_latency*1000:.0f} ms" + (" (edge — no clear peak)"
                                                       if s.peak_at_edge else "")

        lat_test = ("not testable (no clear peak in both)" if np.isnan(res.latency_p)
                    else f"{res.latency_diff*1000:+.0f} ms, jackknife p = {res.latency_p:.3g}")
        rows = [
            ("Condition", res.a.name, res.b.name, ""),
            ("Trials", str(res.a.n), str(res.b.n), ""),
            ("Mean amplitude (window)", f"{res.a.mean_amp:+.2f} ± {res.a.sem:.2f} µV",
             f"{res.b.mean_amp:+.2f} ± {res.b.sem:.2f} µV",
             f"{res.diff:+.2f} µV; t({res.df:.1f}) = {res.t:.2f}, p = {res.p_welch:.3g}"),
            ("Permutation test (mean amp.)", "", "",
             f"p = {res.p_perm:.3g} ({res.n_perm} permutations)"),
            ("Effect size", "", "", f"Hedges' g = {res.hedges_g:+.2f}"),
            ("Peak amplitude (average ERP)", f"{res.a.peak_amp:+.2f} µV",
             f"{res.b.peak_amp:+.2f} µV", ""),
            ("Peak latency (average ERP)", fmt_lat(res.a), fmt_lat(res.b), lat_test),
        ]
        for row in rows:
            r = self.tbl_stats.rowCount()
            self.tbl_stats.insertRow(r)
            for c, text in enumerate(row):
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                self.tbl_stats.setItem(r, c, item)
        self.tbl_stats.resizeColumnsToContents()
        self.lbl_stats.setText(
            "The window test is only valid if the window was fixed before looking "
            "at the data; trials are treated as independent observations "
            "(single-subject statistics).")

    def export_stats(self):
        if self.stats is None:
            QMessageBox.information(self, "No statistics", "Run statistics first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export statistics",
                                              f"{self.stats.component.name}_stats.csv",
                                              "CSV (*.csv)")
        if not path:
            return
        with self._busy("Exporting…"):
            import csv
            res = self.stats
            summary_path = path[:-4] + "_summary.csv" if path.endswith(".csv") else \
                path + "_summary.csv"
            trials = proc.comparison_table(res)
            with open(path, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(trials[0]))
                writer.writeheader()
                writer.writerows(trials)
            with open(summary_path, "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["measure", "value"])
                c = res.component
                for key, value in [
                    ("component", c.name), ("window_ms", f"{c.tmin*1000:.0f}-{c.tmax*1000:.0f}"),
                    ("channels", "+".join(c.channels)),
                    ("polarity", "negative" if c.polarity < 0 else "positive"),
                    ("delay_ms", self.spn_delay.value()),
                    ("condition_a", res.a.name), ("condition_b", res.b.name),
                    ("n_a", res.a.n), ("n_b", res.b.n),
                    ("mean_amp_a_uV", res.a.mean_amp), ("mean_amp_b_uV", res.b.mean_amp),
                    ("diff_uV", res.diff), ("welch_t", res.t), ("welch_df", res.df),
                    ("welch_p", res.p_welch), ("perm_p", res.p_perm),
                    ("n_permutations", res.n_perm), ("hedges_g", res.hedges_g),
                    ("peak_amp_a_uV", res.a.peak_amp), ("peak_amp_b_uV", res.b.peak_amp),
                    ("peak_lat_a_ms", res.a.peak_latency * 1000),
                    ("peak_lat_b_ms", res.b.peak_latency * 1000),
                    ("peak_a_at_edge", res.a.peak_at_edge),
                    ("peak_b_at_edge", res.b.peak_at_edge),
                    ("latency_diff_ms", res.latency_diff * 1000),
                    ("latency_jackknife_p", res.latency_p),
                ]:
                    writer.writerow([key, value])
            self.statusBar().showMessage(f"Saved {path} and {summary_path}")

    # ------------------------------------------------------------ helpers

    @contextmanager
    def _busy(self, message: str):
        self.statusBar().showMessage(message)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            yield
        except Exception as exc:  # surface every processing error to the user
            if not isinstance(exc, InputError):
                traceback.print_exc()
            QMessageBox.critical(self, "Error", f"{message}\n\n{exc}")
            self.statusBar().showMessage(f"Failed: {exc}")
        finally:
            QApplication.restoreOverrideCursor()
        self._update_enabled()

    def _update_enabled(self):
        loaded = self.raw_file is not None
        self.btn_chans.setEnabled(loaded)
        self.btn_events.setEnabled(loaded)
        self.btn_filter.setEnabled(self.raw_orig is not None)
        self.btn_erp.setEnabled(self.raw_orig is not None and self.events is not None)
        self.btn_save.setEnabled(bool(self.evokeds))
        self.btn_script.setEnabled(loaded)
        self.btn_stats.setEnabled(len(self.evokeds) >= 2)
        self.btn_export.setEnabled(self.stats is not None)
        if self.stats is None and self.tbl_stats.rowCount():
            self._show_stats()  # clear results that no longer match the data

    def _current_raw(self) -> mne.io.BaseRaw | None:
        return self.raw_pre if self.raw_pre is not None else self.raw_orig

    def _invalidate(self, from_stage: str):
        """Drop results that depend on a stage that just changed."""
        if from_stage == "pre":
            self.raw_pre = None
        self.epochs = None
        self.evokeds = {}
        self.stats = None

    # ------------------------------------------------------------ channel table

    def _channel_rows(self):
        """(name, ticked, type, electrode) for every row of the channel table."""
        for row in range(self.tbl_chans.rowCount()):
            item = self.tbl_chans.item(row, 0)
            yield (item.text(), item.checkState() == Qt.Checked,
                   self.tbl_chans.cellWidget(row, 1).currentText(),
                   self.tbl_chans.cellWidget(row, 2).currentText().strip())

    def _check_all_channels(self, state: bool):
        for row in range(self.tbl_chans.rowCount()):
            self.tbl_chans.item(row, 0).setCheckState(Qt.Checked if state else Qt.Unchecked)

    def _reset_channel_table(self):
        if self.raw_file is None:
            return
        self.tbl_chans.setRowCount(0)
        for name, selected, kind, electrode in proc.default_channel_setup(self.raw_file):
            row = self.tbl_chans.rowCount()
            self.tbl_chans.insertRow(row)
            item = QTableWidgetItem(name)
            item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            item.setCheckState(Qt.Checked if selected else Qt.Unchecked)
            self.tbl_chans.setItem(row, 0, item)
            cmb = QComboBox()
            cmb.addItems(proc.CHANNEL_TYPES)
            cmb.setCurrentText(kind)
            self.tbl_chans.setCellWidget(row, 1, cmb)
            ele = QComboBox()
            ele.setEditable(True)  # type a label, or one outside the 10-05 list
            ele.setInsertPolicy(QComboBox.NoInsert)
            ele.addItems(self._electrode_labels)
            ele.setCurrentText(electrode)
            ele.setMinimumWidth(80)
            self.tbl_chans.setCellWidget(row, 2, ele)
        self.tbl_chans.resizeRowsToContents()

    # ------------------------------------------------------------ actions

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open recording", "", "BioSemi BDF (*.bdf);;EDF (*.edf);;All files (*)"
        )
        if path:
            self.load(path)

    def load(self, path: str):
        with self._busy(f"Loading {path}…"):
            raw = proc.load_bdf(path)
            self.raw_file = raw
            self.path = path
            self.raw_orig = None
            self._stats_requested = False
            self.events = None
            self._invalidate("pre")
            sf = raw.info["sfreq"]
            self.lbl_file.setText(
                f"<b>{path.split('/')[-1]}</b><br>{len(raw.ch_names)} channels, "
                f"{sf:g} Hz, {raw.times[-1]:.1f} s"
            )
            self.spn_lp.setMaximum(sf / 2 - 1)
            self.cmb_trigger.clear()
            self.cmb_trigger.addItems([proc.ANNOTATIONS] + raw.ch_names)
            self.cmb_trigger.setCurrentText(proc.guess_trigger_channel(raw))
            self.tbl_events.setRowCount(0)
            self._reset_channel_table()
            self.sld_time.setRange(0, max(0, int(raw.times[-1]) - 1))
            self.sld_time.setValue(0)
            self.cmb_stage.setCurrentIndex(0)
        if self.raw_file is not None:
            self.apply_channels()
            self.find_events()
            self.statusBar().showMessage(
                "Loaded. Check channels and trigger, select events, then filter.")

    def apply_channels(self):
        if self.raw_file is None:
            return
        with self._busy("Applying channel selection…"):
            rows = list(self._channel_rows())
            trigger = self.cmb_trigger.currentText()
            include = [name for name, ticked, _, _ in rows if ticked and name != trigger]
            if not any(name in include and kind == "eeg" for name, _, kind, _ in rows):
                raise InputError("Select at least one EEG channel.")
            raw, missing = proc.apply_channel_setup(
                self.raw_file,
                include,
                {name: kind for name, _, kind, _ in rows},
                {name: electrode for name, _, _, electrode in rows},
            )
            self.raw_orig = raw
            self._invalidate("pre")
            n_eeg = len(mne.pick_types(raw.info, eeg=True))
            n_pos = len(proc.eeg_with_positions(raw.info))
            text = (f"{len(include)} channels selected ({n_eeg} EEG); "
                    f"{n_pos}/{n_eeg} EEG with positions.")
            if missing:
                shown = ", ".join(missing[:8]) + (" …" if len(missing) > 8 else "")
                text += f"<br><span style='color:#f5a54e'>No position: {shown}</span>"
            self.lbl_chans.setText(text)
            self._populate_reference(raw)
            self.cmb_stage.setCurrentIndex(0)
            self.statusBar().showMessage("Channel selection applied.")
        self._plot_montage()
        self._plot_signals()

    def _populate_reference(self, raw: mne.io.BaseRaw):
        """Offer the selected EEG channels as references, keeping the choice."""
        eeg = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True)]
        keep = self.cmb_ref.currentText(), self.cmb_ref2.currentText()
        self.cmb_ref.clear()
        self.cmb_ref.addItems([proc.REF_NONE, proc.REF_AVERAGE] + eeg)
        self.cmb_ref2.clear()
        self.cmb_ref2.addItems(["—"] + eeg)
        for cmb, text in zip((self.cmb_ref, self.cmb_ref2), keep):
            if cmb.findText(text) >= 0:
                cmb.setCurrentText(text)

    def _reference_choice(self) -> str | list[str]:
        ref = self.cmb_ref.currentText()
        if ref in (proc.REF_NONE, proc.REF_AVERAGE):
            return ref
        second = self.cmb_ref2.currentText()
        return [ref] if second in ("—", ref) else [ref, second]

    def find_events(self):
        if self.raw_file is None:
            return
        with self._busy("Finding events…"):
            events, event_id = proc.find_events(
                self.raw_file, self.cmb_trigger.currentText(), self.spn_mask.value())
            self.events = events
            self._invalidate("erp")
            counts = proc.event_counts(events) if len(events) else {}
            names = {v: k for k, v in event_id.items()}
            self.tbl_events.setRowCount(0)
            for code, n in sorted(counts.items()):
                row = self.tbl_events.rowCount()
                self.tbl_events.insertRow(row)
                item = QTableWidgetItem(str(code))
                item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                # One-off codes are usually start/end markers, not conditions.
                item.setCheckState(Qt.Checked if n >= 2 else Qt.Unchecked)
                item.setData(Qt.UserRole, int(code))
                self.tbl_events.setItem(row, 0, item)
                cnt = QTableWidgetItem(str(n))
                cnt.setFlags(Qt.ItemIsEnabled)
                self.tbl_events.setItem(row, 1, cnt)
                label = names.get(code, str(code))
                self.tbl_events.setItem(row, 2, QTableWidgetItem(
                    label if not label.isdigit() else f"event {label}"))
            self.statusBar().showMessage(
                f"{len(events)} events, {len(counts)} codes on "
                f"{self.cmb_trigger.currentText()}.")
        self._plot_signals()

    def _delayed_events(self) -> np.ndarray:
        """Events moved by the EEG delay set in step 4."""
        return proc.shift_events(self.events, self.spn_delay.value(),
                                 self.raw_file.info["sfreq"])

    def _delay_changed(self):
        self._plot_signals()
        if self.evokeds:  # keep the ERPs in step with the slider
            self.compute_erps(quiet=True)
        else:
            self.statusBar().showMessage(f"EEG delay {self.spn_delay.value()} ms.")

    def _event_groups(self) -> dict[str, list[int]]:
        """Condition name -> trigger codes for the ticked rows."""
        groups: dict[str, list[int]] = {}
        for row in range(self.tbl_events.rowCount()):
            item = self.tbl_events.item(row, 0)
            if item.checkState() == Qt.Checked:
                name = self.tbl_events.item(row, 2).text().strip() or item.text()
                groups.setdefault(name, []).append(item.data(Qt.UserRole))
        return groups

    def _selected_event_id(self) -> tuple[np.ndarray, dict[str, int]]:
        """Events and event_id for the ticked rows; names may pool codes."""
        return proc.group_events(self._delayed_events(), self._event_groups())

    def preprocess(self):
        if self.raw_orig is None:
            return
        with self._busy("Referencing and filtering with g.Pype…"):
            ref = self._reference_choice()
            notch = {"off": None, "50 Hz": 50.0, "60 Hz": 60.0}[self.cmb_notch.currentText()]
            settings = proc.FilterSettings(
                highpass=self.spn_hp.value() or None,
                lowpass=self.spn_lp.value() or None,
                notch=notch,
                order=self.spn_order.value(),
            )
            self._invalidate("pre")
            referenced = proc.apply_reference(self.raw_orig, ref)
            self.raw_pre = proc.apply_filter(referenced, settings)
            self.cmb_stage.setCurrentIndex(1)
            ref_text = ref if isinstance(ref, str) else " + ".join(ref)
            self.statusBar().showMessage(
                f"Reference: {ref_text}; filtered ({settings.describe()}).")
        self._plot_signals()

    def compute_erps(self, _checked=False, quiet=False):
        raw = self._current_raw()
        if raw is None or self.events is None:
            return
        if raw is self.raw_orig and not quiet:
            QMessageBox.information(self, "Unfiltered data",
                                    "ERPs will be computed on unfiltered data. "
                                    "Apply step 3 first for filtered ERPs.")
        rerun_stats = False
        with self._busy("Epoching and averaging…"):
            events, event_id = self._selected_event_id()
            if not event_id:
                raise InputError("Tick at least one event in step 4.")
            settings = proc.EpochSettings(
                tmin=self.spn_tmin.value(),
                tmax=self.spn_tmax.value(),
                baseline=(None, 0.0) if self.chk_baseline.isChecked() else None,
                reject_uv=self.spn_reject.value() or None,
            )
            self.epochs, self.evokeds = proc.compute_erps(raw, events, event_id, settings)
            total = {k: int((events[:, 2] == v).sum()) for k, v in event_id.items()}
            kept = {k: len(self.epochs[k]) if k in self.epochs.event_id else 0
                    for k in event_id}
            text = "<br>".join(f"{k}: {kept[k]}/{total[k]} epochs kept" for k in event_id)
            causes = proc.rejection_causes(self.epochs)
            if causes:
                worst = ", ".join(f"{ch} ({n})" for ch, n in list(causes.items())[:4])
                text += (f"<br><span style='color:#f5a54e'>Rejections caused by: {worst}"
                         "</span>")
            self.lbl_counts.setText(text)
            if not self.evokeds:
                raise InputError(
                    "Every epoch was rejected. Deselect noisy channels in step 2 or "
                    "raise/disable the p-p rejection threshold.")
            chans = self.epochs.ch_names
            previous = {i.text() for i in self.lst_chan.selectedItems()}
            self.lst_chan.blockSignals(True)
            self.lst_chan.clear()
            self.lst_chan.addItems(chans)
            sel = previous & set(chans) or {next((c for c in chans if c.upper() in
                                                  ("CZ", "PZ")), chans[0])}
            for i in range(self.lst_chan.count()):
                self.lst_chan.item(i).setSelected(self.lst_chan.item(i).text() in sel)
            self.lst_chan.blockSignals(False)
            if not quiet:
                self.tabs.setCurrentIndex(TAB_EPOCHS)
            self.statusBar().showMessage(
                f"{len(self.epochs)} epochs, {len(self.evokeds)} conditions.")
            rerun_stats = self._stats_requested
            self._populate_stats_inputs()
        self._plot_erps()
        if self.evokeds and rerun_stats:  # keep statistics in step (e.g. delay slider)
            self.run_stats(quiet=True)

    def _analysis_config(self) -> AnalysisConfig:
        """Everything configured in the panel, for the script export."""
        rows = list(self._channel_rows())
        trigger = self.cmb_trigger.currentText()
        include = [(name, kind, ele) for name, ticked, kind, ele in rows
                   if ticked and name != trigger]
        if not include:
            raise InputError("Select at least one channel in step 2.")
        groups = self._event_groups()
        if not groups:
            raise InputError("Tick at least one event in step 4.")
        raw = self.raw_file

        annotations = trigger == proc.ANNOTATIONS
        if annotations:
            # The script selects annotations by description, not by MNE's codes.
            _, ids = proc.find_events(raw, proc.ANNOTATIONS)
            description = {code: desc for desc, code in ids.items()}
            conditions = {name: [description[c] for c in codes]
                          for name, codes in groups.items()}
            mask = None
        else:
            conditions = groups
            mask = self.spn_mask.value() or None
            if mask is not None:  # only write a mask that changes the codes
                masked = proc.find_events(raw, trigger, mask)[0]
                unmasked = proc.find_events(raw, trigger, 0)[0]
                mask = None if np.array_equal(masked, unmasked) else mask

        ref = self._reference_choice()
        notch = {"off": None, "50 Hz": 50.0, "60 Hz": 60.0}[self.cmb_notch.currentText()]
        component = None
        a, b = self.cmb_cond_a.currentText(), self.cmb_cond_b.currentText()
        stat_channels = [i.text() for i in self.lst_stat_chan.selectedItems()]
        if a and b and a != b and stat_channels:
            component = Component(
                name=self.edt_comp.text().strip() or "component",
                channels=stat_channels,
                tmin=self.spn_win_lo.value() / 1000,
                tmax=self.spn_win_hi.value() / 1000,
                condition_a=a,
                condition_b=b,
                polarity=-1 if self.cmb_polarity.currentText() == "negative" else 1,
                n_permutations=self.spn_perm.value(),
            )

        labels = proc.electrode_labels()
        eeg = [ele or name for name, kind, ele in include if kind == "eeg"]
        positioned = [ch for ch in eeg if ch in labels]
        return AnalysisConfig(
            recording=self.path,
            channels={name: ele or name for name, _, ele in include},
            channel_types={name: kind for name, kind, _ in include if kind != "eeg"},
            reference="none" if ref == proc.REF_NONE else ref,
            highpass=self.spn_hp.value() or None,
            lowpass=self.spn_lp.value() or None,
            notch=notch,
            filter_order=self.spn_order.value(),
            trigger_channel=None if annotations else trigger,
            trigger_is_stim=not annotations and trigger in proc.stim_channels(raw),
            trigger_scale=1.0 if annotations else proc.trigger_scale(raw, trigger),
            trigger_mask=mask,
            delay_ms=self.spn_delay.value(),
            conditions=conditions,
            tmin=self.spn_tmin.value(),
            tmax=self.spn_tmax.value(),
            baseline=self.chk_baseline.isChecked(),
            reject_uv=self.spn_reject.value() or None,
            component=component,
            topomap_latency=round(self.latency, 3),
            # Topomaps need positions for every EEG channel the script plots.
            topomap_ok=len(positioned) >= 3 and len(positioned) == len(eeg),
        )

    def export_script(self):
        if self.raw_file is None:
            return
        try:
            cfg = self._analysis_config()
        except InputError as exc:
            QMessageBox.critical(self, "Cannot export", str(exc))
            return
        stem = Path(self.path).stem
        path, _ = QFileDialog.getSaveFileName(self, "Export MNE script",
                                              f"{stem}_erp_analysis.py", "Python (*.py)")
        if path:
            with self._busy("Writing MNE script…"):
                Path(path).write_text(build_script(cfg))
                self.statusBar().showMessage(f"Saved {path}")

    def save_erps(self):
        if not self.evokeds:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save ERPs", "erps-ave.fif",
                                              "MNE evoked (*-ave.fif)")
        if path:
            with self._busy("Saving…"):
                for name, ev in self.evokeds.items():
                    ev.comment = name
                mne.write_evokeds(path, list(self.evokeds.values()), overwrite=True,
                                  verbose="error")
                self.statusBar().showMessage(f"Saved {path}")

    # ------------------------------------------------------------ plots

    def _plot_signals(self):
        plot = self.plt_signals
        plot.clear()
        raw = self.raw_pre if self.cmb_stage.currentIndex() == 1 else self.raw_orig
        if raw is None:
            raw = self._current_raw()
        if raw is None:
            return
        picks = mne.pick_types(raw.info, eeg=True, eog=True, emg=True, ecg=True,
                               misc=True)
        sf = raw.info["sfreq"]
        t0 = float(self.sld_time.value())
        dur = self.spn_window.value()
        a, b = int(t0 * sf), min(raw.n_times, int((t0 + dur) * sf))
        if b <= a or not len(picks):
            return
        data = raw.get_data(picks=picks, start=a, stop=b) * 1e6
        if raw is self.raw_orig:
            data = data - data.mean(axis=1, keepdims=True)  # hide DC offsets
        step = max(1, (b - a) // 4000)
        times = raw.times[a:b:step]
        data = data[:, ::step]
        scale = self.spn_scale.value()
        ticks = []
        for k, (idx, trace) in enumerate(zip(picks, data)):
            name = raw.ch_names[idx]
            color = "#9ec5ff" if mne.channel_type(raw.info, idx) == "eeg" else "#f5a54e"
            plot.plot(times, trace / scale - k, pen=pg.mkPen(color, width=1))
            ticks.append((-k, name))
        plot.getAxis("left").setTicks([ticks[:: max(1, len(ticks) // 40)]])
        if self.events is not None:
            events = self._delayed_events()
            ev = events[(events[:, 0] - raw.first_samp >= a) &
                        (events[:, 0] - raw.first_samp < b)]
            for s, _, code in ev:
                t = (s - raw.first_samp) / sf
                plot.addItem(pg.InfiniteLine(t, pen=pg.mkPen("#5fd38d", style=Qt.DashLine)))
                label = pg.TextItem(str(code), color="#5fd38d", anchor=(0, 1))
                label.setPos(t, 1.4)
                plot.addItem(label)
        plot.setXRange(t0, t0 + dur, padding=0)
        plot.setYRange(-len(picks), 1.5, padding=0)

    def _plot_erps(self):
        plot = self.plt_erp
        plot.clear()
        plot.getPlotItem().invertY(self.chk_invert.isChecked())
        if not self.evokeds:
            return
        view = self.cmb_view.currentText()
        selected = [i.text() for i in self.lst_chan.selectedItems()]
        for c, (name, ev) in enumerate(self.evokeds.items()):
            color = QColor(COLORS[c % len(COLORS)])
            data = ev.data * 1e6
            if view == "Butterfly":
                eeg = mne.pick_types(ev.info, eeg=True)
                color.setAlpha(90)
                for i, trace in enumerate(data[eeg]):
                    plot.plot(ev.times, trace, pen=pg.mkPen(color),
                              name=f"{name} (n={ev.nave})" if i == 0 else None)
                continue
            if view == "GFP":
                eeg = mne.pick_types(ev.info, eeg=True)
                trace = data[eeg].std(axis=0)
                label = f"{name} GFP (n={ev.nave})"
            else:
                chans = selected or [ev.ch_names[0]]
                if view == "Single channel":
                    chans = chans[:1]
                idx = [ev.ch_names.index(ch) for ch in chans]
                trace = data[idx].mean(axis=0)
                where = chans[0] if len(chans) == 1 else f"mean of {len(chans)}"
                label = f"{name} @ {where} (n={ev.nave})"
            plot.plot(ev.times, trace, pen=pg.mkPen(color, width=2), name=label)
        plot.addItem(pg.InfiniteLine(0, pen=pg.mkPen("#888", style=Qt.DashLine)))
        plot.addItem(pg.InfiniteLine(0, angle=0, pen=pg.mkPen("#666")))
        plot.addItem(pg.InfiniteLine(self.latency, pen=pg.mkPen("#f5e04e", width=1)))
        self._plot_topo()

    def _erp_clicked(self, event):
        if not self.evokeds:
            return
        vb = self.plt_erp.getPlotItem().vb
        if not vb.sceneBoundingRect().contains(event.scenePos()):
            return
        self.latency = float(vb.mapSceneToView(event.scenePos()).x())
        self._plot_erps()

    def _no_positions(self, fig, canvas):
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "Fewer than 3 EEG channels with electrode positions —\n"
                "assign electrodes in step 2 to see topographies",
                ha="center", va="center")
        ax.axis("off")
        canvas.draw_idle()

    def _plot_topo(self):
        fig = self.cnv_topo.figure
        fig.clear()
        evs = list(self.evokeds.items())
        if not evs:
            return
        ev0 = evs[0][1]
        t = float(np.clip(self.latency, ev0.times[0], ev0.times[-1]))
        positioned = proc.eeg_with_positions(ev0.info)
        if len(positioned) < 3:
            self._no_positions(fig, self.cnv_topo)
            return
        picked = [(n, e.copy().pick(positioned)) for n, e in evs]
        idx = ev0.time_as_index(t)[0]
        lim = max(np.abs(e.data[:, idx]).max() for _, e in picked) * 1e6 or 1
        for i, (name, ev) in enumerate(picked):
            ax = fig.add_subplot(1, len(picked), i + 1)
            im, _ = mne.viz.plot_topomap(ev.data[:, idx] * 1e6, ev.info, axes=ax,
                                         show=False, vlim=(-lim, lim), contours=4)
            ax.set_title(f"{name}\n{1000 * t:.0f} ms", fontsize=9)
        fig.colorbar(im, ax=fig.axes, shrink=0.8, label="µV")
        self.cnv_topo.draw_idle()

    def _plot_montage(self):
        fig = self.cnv_montage.figure
        fig.clear()
        raw = self.raw_orig
        if raw is None:
            return
        positioned = proc.eeg_with_positions(raw.info)
        if len(positioned) == 0:
            self._no_positions(fig, self.cnv_montage)
            return
        ax = fig.add_subplot(111)
        info = mne.pick_info(raw.info, positioned)
        mne.viz.plot_sensors(info, kind="topomap", show_names=True, axes=ax,
                             show=False, pointsize=40, linewidth=0)
        ax.set_title(f"{len(positioned)} EEG channels with positions", fontsize=10)
        self.cnv_montage.draw_idle()


ICON = Path(__file__).parent / "resources" / "rhinoerp.svg"


def main(argv=None):
    argv = sys.argv if argv is None else argv
    if "--version" in argv[1:]:
        from . import __version__
        print(f"Rhino ERP {__version__}")
        return 0
    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName("Rhino ERP")
    app.setWindowIcon(QIcon(str(ICON)))
    gp_theme.apply(app, gp_theme.DARK)
    pg.setConfigOptions(antialias=True,
                        background=app.palette().window().color(),
                        foreground=app.palette().text().color())
    matplotlib.rcParams.update({"figure.facecolor": "#2b2b2b", "text.color": "#ddd",
                                "axes.labelcolor": "#ddd", "xtick.color": "#ddd",
                                "ytick.color": "#ddd", "axes.facecolor": "#2b2b2b"})
    win = MainWindow()
    win.show()
    if len(argv) > 1:
        win.load(argv[1])
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
