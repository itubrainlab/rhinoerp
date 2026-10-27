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


def test_workflow_and_exported_script(window, tmp_path, monkeypatch):
    window.cmb_ref.setCurrentText("average")
    window.preprocess()
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
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    ns = runpy.run_path(str(script))
    plt.close("all")
    # MNE's filter differs from the app's g.Pype filter only slightly.
    assert ns["n_a"] == stats.a.n and ns["n_b"] == stats.b.n
    assert np.isclose(ns["difference"], stats.diff, atol=0.05)
    assert np.isclose(ns["welch"].statistic, stats.t, atol=0.05)
    assert ns["peak_a"][1] == pytest.approx(stats.a.peak_latency, abs=0.005)
