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
    window.preprocess()
    # As many components as the rank of the filtered (unreferenced) EEG.
    assert window.spn_ncomp.value() == window.spn_ncomp.maximum() == 8
    window.fit_ica()
    assert window.ica.n_components == 8
    # Fitted from the first to the last trigger, on all the data otherwise.
    assert "fitted on 2.0–" in window.lbl_ica.text()
    assert window.ica.sources.shape[1] == window.raw_pre.n_times
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
    # The reference comes after ICA, when the epochs are cut.
    window.cmb_ref.setCurrentText("average")
    assert window._current_raw() is window.raw_clean
    assert window.raw_clean.get_data(picks="Fz").std() < 0.5 * window.raw_pre.get_data(picks="Fz").std()
    window.spn_reject.setValue(0)
    window.spn_delay.setValue(8)
    window.compute_erps()
    assert set(window.evokeds) == {"event 20", "event 30"}
    assert np.abs(window.raw_ref.get_data(picks="eeg").sum(axis=0)).max() < 1e-9
    # The ICA tab shows a row of charts for every component.
    assert len(window._ica_rows) == window.ica.n_components

    window.cmb_comp.setCurrentText("N170")
    window.cmb_cond_a.setCurrentText("event 30")
    window.cmb_cond_b.setCurrentText("event 20")
    window.run_stats()
    stats = window.stats
    assert stats.p < 0.001 and stats.diff < 0

    script = tmp_path / "analysis.py"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(script), "")))
    window.export_script()
    assert (tmp_path / "analysis-ica.fif").exists()
    text = script.read_text()
    assert text.index("ica.apply(") < text.index("raw.set_eeg_reference(")
    assert f"REMOVED_COMPONENTS = {removed}" in text and "Eye blink" in text
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    ns = runpy.run_path(str(script))
    plt.close("all")
    # The script repeats the app's MNE processing exactly.
    assert ns["n_a"] == stats.a.n and ns["n_b"] == stats.b.n
    assert np.isclose(ns["difference"], stats.diff)
    assert np.isclose(ns["welch"].statistic, stats.t)
    assert np.isclose(ns["welch"].pvalue, stats.p)
    assert ns["peak_a"] == pytest.approx((stats.a.peak_amp, stats.a.peak_latency))
