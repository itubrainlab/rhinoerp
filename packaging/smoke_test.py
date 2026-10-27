"""Check a built package without a display.

    <package python> -I smoke_test.py <library folder> <expected version>

Run with the package's own Python against the library folder its first
launch set up, so it checks what users get: the bundled Python, the
libraries installed from PyPI (g.Pype, MNE, Qt) and the bundled wheel. It
filters a synthetic recording through g.Pype, epochs it with MNE, runs
the statistics, opens the main window and builds an MNE script.
"""
import os
import sys

sys.path.insert(0, sys.argv[1])
EXPECTED = sys.argv[2]
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["MPLBACKEND"] = "Agg"

import gpype  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402
import rhino_erp  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from rhino_erp import processing as P  # noqa: E402
from rhino_erp.app import MainWindow  # noqa: E402
from rhino_erp.script_export import AnalysisConfig, build_script  # noqa: E402

print(f"Rhino ERP {rhino_erp.__version__}, g.Pype {gpype.__version__}, "
      f"MNE {mne.__version__}, Python {sys.version.split()[0]} on {sys.platform}",
      flush=True)
assert rhino_erp.__version__ == EXPECTED, rhino_erp.__version__

# A recording with an N170-like effect for code 30 at PO7/PO8.
sfreq, n = 250, 250 * 120
rng = np.random.default_rng(0)
data = rng.normal(0, 6e-6, (9, n))
events = []
for i, onset in enumerate(range(500, n - 500, 300)):
    code = 30 if i % 2 else 20
    data[8, onset:onset + 5] = code
    events.append(code)
    if code == 30:
        t = np.arange(150) / sfreq
        data[[5, 7], onset:onset + 150] -= 8e-6 * np.exp(-((t - 0.17) ** 2) / 8e-4)
info = mne.create_info(list(P.UNICORN_LABELS) + ["TRIG"], sfreq, ["eeg"] * 8 + ["stim"])
raw = mne.io.RawArray(data, info, verbose="error")
raw.set_montage("standard_1020", on_missing="ignore", verbose="error")

found, _ = P.find_events(raw, "TRIG")
assert found[:, 2].tolist() == events
filtered = P.apply_filter(raw.copy().pick("eeg"), P.FilterSettings(0.1, 30, 50, 4))
print("g.Pype filter: ok", flush=True)
epochs, evokeds = P.compute_erps(filtered, found, {"20": 20, "30": 30},
                                 P.EpochSettings(-0.2, 0.6, (None, 0.0), None))
result = P.compare_conditions(epochs, "30", "20", P.COMPONENT_PRESETS["N170"],
                              n_perm=500)
assert result.diff < 0 and result.p_welch < 0.01, (result.diff, result.p_welch)
print(f"statistics: ok (N170 {result.diff:+.1f} µV, p = {result.p_welch:.2g})",
      flush=True)

app = QApplication([])
window = MainWindow()
window.show()
app.processEvents()
window.close()
print("main window: ok", flush=True)

script = build_script(AnalysisConfig(
    recording="recording.bdf", channels={"EEG 1": "Fz"}, channel_types={},
    reference="none", highpass=0.1, lowpass=30.0, notch=50.0, filter_order=4,
    trigger_channel="TRIG", trigger_is_stim=False, trigger_scale=1.0,
    trigger_mask=None, delay_ms=0, conditions={"a": [20]}, tmin=-0.2, tmax=0.8,
    baseline=True, reject_uv=None, component=None, topomap_latency=0.3,
    topomap_ok=False))
compile(script, "exported.py", "exec")
print("script export: ok")
print("PASS")
