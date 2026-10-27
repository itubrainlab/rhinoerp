"""A synthetic Unicorn Hybrid Black recording shared by the tests.

Eight EEG channels (EEG 1-8 at the Unicorn positions Fz, C3, Cz, C4, Pz,
PO7, Oz, PO8), accelerometer channels typed as EEG by the reader, and a
TRIG channel with codes 20 and 30. Code 30 carries an N170-like
negativity at PO7/PO8 that code 20 lacks, so a real effect exists.
"""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

SFREQ = 250
LABELS = ["EEG 1", "EEG 2", "EEG 3", "EEG 4", "EEG 5", "EEG 6", "EEG 7", "EEG 8",
          "ACC X", "ACC Y", "TRIG"]
PO7, PO8 = 5, 7


def write_recording(path, seconds=200, seed=0):
    import pyedflib

    rng = np.random.default_rng(seed)
    n = seconds * SFREQ
    t = np.arange(n) / SFREQ
    eeg = rng.normal(0, 6, (8, n)) + 20 * np.sin(2 * np.pi * 50 * t)  # µV
    eeg += 117_000 + rng.uniform(-500, 500, (8, 1))  # the Unicorn's DC offset
    trig = np.zeros(n)
    erp_t = np.arange(int(0.6 * SFREQ)) / SFREQ
    n170 = -8 * np.exp(-((erp_t - 0.17) ** 2) / (2 * 0.02 ** 2))
    onset, codes = 2 * SFREQ, []
    while onset < n - 2 * SFREQ:
        code = 30 if len(codes) % 2 else 20
        codes.append(code)
        trig[onset:onset + 5] = code
        if code == 30:
            eeg[[PO7, PO8], onset:onset + len(erp_t)] += n170
        onset += int(SFREQ * rng.uniform(1.0, 1.4))
    acc = rng.normal(0, 0.01, (2, n))
    signals = list(eeg) + list(acc) + [trig]
    writer = pyedflib.EdfWriter(str(path), len(LABELS), file_type=pyedflib.FILETYPE_BDFPLUS)
    headers = []
    for label in LABELS:
        is_eeg = label.startswith("EEG")
        headers.append({
            "label": label, "dimension": "uV" if is_eeg else "n/a",
            "sample_frequency": SFREQ,
            "physical_min": -262144 if is_eeg else -1000,
            "physical_max": 262143 if is_eeg else 1000,
            "digital_min": -8388608, "digital_max": 8388607,
        })
    writer.setSignalHeaders(headers)
    writer.writeSamples(signals)
    writer.close()
    return codes


@pytest.fixture(scope="session")
def recording(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "unicorn.bdf"
    codes = write_recording(path)
    return path, codes
