"""The whole workflow through the window, offscreen, and the script export."""

import runpy

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from rhino_erp import __version__, app as A


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, recording, monkeypatch):
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a: errors.append(a[-1]))
    monkeypatch.setattr(QMessageBox, "information", lambda *a: None)
    w = A.MainWindow()
    w.load(str(recording[0]))
    yield w
    w.close()
    assert not errors, errors


def test_version(capsys):
    assert A.main(["rhino-erp", "--version"]) == 0
    assert __version__ in capsys.readouterr().out


def test_load_defaults(window):
    assert window.cmb_trigger.currentText() == "TRIG"
    assert window.raw_orig.ch_names == ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
    ticked = [window.tbl_events.item(r, 0).text() for r in range(window.tbl_events.rowCount())
              if window.tbl_events.item(r, 0).checkState() == Qt.Checked]
    assert ticked == ["20", "30"]


def test_time_range(window, recording):
    # Prefilled with the first and last events (the first is at 2 s).
    sf = window.raw_full.info["sfreq"]
    assert window.spn_start.value() == pytest.approx(2.0)
    assert window.raw_file.times[-1] < window.raw_full.times[-1]
    assert sum(window.tbl_events.item(r, 1).text() != "" for r in
               range(window.tbl_events.rowCount())) == 2
    total = sum(int(window.tbl_events.item(r, 1).text())
                for r in range(window.tbl_events.rowCount()))
    assert total == len(recording[1])
    # A shorter range keeps fewer events; the processing follows it.
    window.tbl_events.item(0, 2).setText("standard")
    window.spn_start.setValue(50)
    window.spn_end.setValue(100)
    window.apply_range()
    assert window.raw_orig.times[-1] == pytest.approx(50, abs=1 / sf)
    total = sum(int(window.tbl_events.item(r, 1).text())
                for r in range(window.tbl_events.rowCount()))
    assert 0 < total < len(recording[1])
    assert window.tbl_events.item(0, 2).text() == "standard"  # names are kept
    window.preprocess()
    assert window.raw_pre.times[-1] == pytest.approx(50, abs=1 / sf)
    window._range_full()
    window.apply_range()
    assert window.raw_file is window.raw_full


def test_workflow_and_exported_script(window, tmp_path, monkeypatch):
    window.cmb_ref.setCurrentText("average")
    window.preprocess()
    window.fit_ica()
    # Components labelled as artifacts start unticked, the rest ticked.
    labels = window._component_labels()
    blink = labels.index("Eye blink")
    removed = [i for i, label in enumerate(labels) if label in A.proc.ARTIFACT_LABELS]
    assert blink in removed
    assert window._selected_components() == [i for i in range(len(labels))
                                             if i not in removed]
    # Relabelling ticks or unticks the component to match.
    other = next(i for i in range(len(labels)) if i != blink)
    window.tbl_ica.cellWidget(other, 1).setCurrentText("Eye blink")
    assert other not in window._selected_components()
    window.tbl_ica.cellWidget(other, 1).setCurrentText("EEG")
    assert other in window._selected_components()
    removed = [i for i in removed if i != other]
    window.apply_ica()
    assert window._current_raw() is window.raw_clean
    assert window.raw_clean.get_data(picks="Fz").std() < 0.5 * window.raw_pre.get_data(picks="Fz").std()
    window.spn_reject.setValue(0)
    window.spn_delay.setValue(8)
    window.compute_erps()
    assert set(window.evokeds) == {"event 20", "event 30"}

    window.cmb_comp.setCurrentText("N170")
    window.cmb_cond_a.setCurrentText("event 30")
    window.cmb_cond_b.setCurrentText("event 20")
    window.spn_perm.setValue(1000)
    window.run_stats()
    stats = window.stats
    assert stats.p_welch < 0.001 and stats.diff < 0

    script = tmp_path / "analysis.py"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(script), "")))
    window.export_script()
    assert (tmp_path / "analysis-ica.fif").exists()
    text = script.read_text()
    assert "raw.crop(*TIME_RANGE)" in text  # the first-to-last-event range
    assert f"REMOVED_COMPONENTS = {removed}" in text and "Eye blink" in text
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    ns = runpy.run_path(str(script))
    plt.close("all")
    # MNE's filter differs from the app's g.Pype filter only slightly.
    assert ns["n_a"] == stats.a.n and ns["n_b"] == stats.b.n
    assert np.isclose(ns["difference"], stats.diff, atol=0.05)
    assert np.isclose(ns["welch"].statistic, stats.t, rtol=0.01)
    assert ns["peak_a"][1] == pytest.approx(stats.a.peak_latency, abs=0.005)
