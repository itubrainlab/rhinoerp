"""Write a synthetic BioSemi-style BDF with ERPs, blinks and line noise.

    python scripts/make_demo_bdf.py demo.bdf

32 EEG channels (biosemi32 layout), EXG1 as VEOG and a Status channel.
Code 1 = standard, code 2 = oddball (adds a P3 over parietal sites).
"""

import sys

import mne
import numpy as np
import pyedflib


def main(path: str, seconds: int = 300, sfreq: int = 512, seed: int = 0):
    rng = np.random.default_rng(seed)
    montage = mne.channels.make_standard_montage("biosemi32")
    names = montage.ch_names
    pos = np.array([montage.get_positions()["ch_pos"][n] for n in names])
    n, t = seconds * sfreq, np.arange(seconds * sfreq) / sfreq

    eeg = rng.normal(0, 8e-6, (len(names), n))
    eeg += 4e-6 * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 6, (len(names), 1)))
    eeg += 20e-6 * np.sin(2 * np.pi * 50 * t)
    eeg += np.cumsum(rng.normal(0, 2e-7, (len(names), n)), axis=1)  # drift
    eeg += rng.uniform(-20e-3, 20e-3, (len(names), 1))  # BioSemi DC offsets

    status = np.zeros(n)
    erp_t = np.arange(int(0.8 * sfreq)) / sfreq
    n1 = -5e-6 * np.exp(-((erp_t - 0.1) ** 2) / (2 * 0.02**2))
    p3 = 10e-6 * np.exp(-((erp_t - 0.35) ** 2) / (2 * 0.06**2))
    central = np.exp(-np.sum((pos - pos[names.index("Cz")]) ** 2, 1) / 0.004)
    parietal = np.exp(-np.sum((pos - pos[names.index("Pz")]) ** 2, 1) / 0.004)
    onset = sfreq * 2
    while onset < n - sfreq * 2:
        code = 2 if rng.random() < 0.2 else 1
        status[onset : onset + 10] = code
        seg = slice(onset, onset + len(erp_t))
        eeg[:, seg] += np.outer(central, n1)
        if code == 2:
            eeg[:, seg] += np.outer(parietal, p3)
        onset += int(sfreq * rng.uniform(0.9, 1.3))

    # Blinks: large frontal positivity decaying towards the back.
    frontal = np.exp(-np.sum((pos - pos[names.index("Fp1")]) ** 2, 1) / 0.01)
    frontal += np.exp(-np.sum((pos - pos[names.index("Fp2")]) ** 2, 1) / 0.01)
    veog = rng.normal(0, 5e-6, n) + np.cumsum(rng.normal(0, 1e-7, n))
    bt = np.arange(int(0.4 * sfreq)) / sfreq
    shape = 150e-6 * np.sin(np.pi * bt / 0.4) ** 2
    for b in rng.uniform(1, seconds - 1, seconds // 4):
        s = int(b * sfreq)
        eeg[:, s : s + len(bt)] += np.outer(frontal / frontal.max() * 0.8, shape)
        veog[s : s + len(bt)] += shape

    signals = list(eeg) + [veog, status]
    labels = names + ["EXG1", "Status"]
    writer = pyedflib.EdfWriter(path, len(labels), file_type=pyedflib.FILETYPE_BDFPLUS)
    headers = []
    for label in labels:
        is_stat = label == "Status"
        headers.append({
            "label": label,
            "dimension": "Boolean" if is_stat else "uV",
            "sample_frequency": sfreq,
            "physical_min": -8388608 if is_stat else -262144,
            "physical_max": 8388607 if is_stat else 262143,
            "digital_min": -8388608,
            "digital_max": 8388607,
        })
    writer.setSignalHeaders(headers)
    writer.writeSamples([s if lab == "Status" else s * 1e6
                         for s, lab in zip(signals, labels)])
    writer.close()
    print(f"wrote {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "demo.bdf")
