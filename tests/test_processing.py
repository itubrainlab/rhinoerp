import mne
import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt, tf2sos

from rhino_erp import processing as P


@pytest.fixture(scope="module")
def raw(recording):
    return P.load_bdf(str(recording[0]))


@pytest.fixture(scope="module")
def prepared(raw):
    rows = P.default_channel_setup(raw)
    include = [n for n, selected, _, _ in rows if selected]
    channels, _ = P.apply_channel_setup(
        raw, include, {n: k for n, _, k, _ in rows}, {n: e for n, _, _, e in rows})
    return P.apply_filter(channels, P.FilterSettings(0.1, 30, 50, 4))


def test_default_channels_are_the_unicorn_eight(raw):
    rows = P.default_channel_setup(raw)
    assert [r[3] for r in rows[:8]] == list(P.UNICORN_LABELS)
    assert all(r[1] and r[2] == "eeg" for r in rows[:8])
    assert all(not r[1] and r[2] == "misc" for r in rows[8:])


def test_trigger_channel_events(raw, recording):
    assert P.guess_trigger_channel(raw) == "TRIG"
    events, event_id = P.find_events(raw, "TRIG")
    assert events[:, 2].tolist() == recording[1]
    assert event_id == {"20": 20, "30": 30}
    # Same events as MNE's own reader once the channel is typed as stim.
    stim = raw.copy()
    stim.set_channel_types({"TRIG": "stim"}, verbose="error")
    stim.apply_function(lambda x: np.round(x), picks="TRIG")  # as the exported script does
    expected = mne.find_events(stim, "TRIG", shortest_event=1, consecutive=True,
                               verbose="error")
    assert np.array_equal(events, expected)


def test_gpype_filter_is_zero_phase_butterworth(raw):
    x = raw.get_data(picks=["EEG 1", "EEG 6"])
    sf = raw.info["sfreq"]
    out = P.gpype_filter(x, sf, ["a", "b"], P.FilterSettings(0.1, 30, 50, 4))
    expected = (x - x.mean(axis=1, keepdims=True)).astype(np.float32)
    for order, band, kind in ((4, [0.1, 30], "bandpass"), (2, [48, 52], "bandstop")):
        b, a = butter(order, np.array(band) / (sf / 2), kind)
        expected = sosfiltfilt(tf2sos(b, a), expected, axis=1).astype(np.float32)
    assert np.array_equal(out, expected)


def test_shift_and_group_events():
    events = np.array([[100, 0, 21], [200, 0, 31], [300, 0, 20]])
    shifted = P.shift_events(events, 40, 250)
    assert shifted[:, 0].tolist() == [110, 210, 310]
    grouped, event_id = P.group_events(shifted, {"pooled": [21, 31], "other": [20]})
    assert event_id["other"] == 20
    assert (grouped[:, 2] == event_id["pooled"]).sum() == 2


def test_reference_average_sums_to_zero(prepared):
    referenced = P.apply_reference(prepared, P.REF_AVERAGE)
    assert np.abs(referenced.get_data(picks="eeg").sum(axis=0)).max() < 1e-12


def test_statistics_find_the_simulated_n170(raw, prepared):
    events, _ = P.find_events(raw, "TRIG")
    epochs, evokeds = P.compute_erps(prepared, events, {"20": 20, "30": 30},
                                     P.EpochSettings(-0.2, 0.6, (None, 0.0), None))
    assert set(evokeds) == {"20", "30"}
    result = P.compare_conditions(epochs, "30", "20", P.COMPONENT_PRESETS["N170"],
                                  n_perm=1000)
    assert result.component.channels == ("PO7", "PO8")
    assert result.diff < -4 and result.p_welch < 0.001 and result.p_perm < 0.01
    assert abs(result.a.peak_latency - 0.17) < 0.012 and not result.a.peak_at_edge
    # Same seed, same permutation p.
    again = P.compare_conditions(epochs, "30", "20", P.COMPONENT_PRESETS["N170"],
                                 n_perm=1000)
    assert again.p_perm == result.p_perm


def test_ica_component_count(prepared):
    rank = P.ica_rank(prepared)
    assert P.fit_ica(prepared, 3).n_components == 3
    assert P.fit_ica(prepared, 99).n_components == rank  # capped at the rank


def test_ica_labels_and_removes_the_blinks(prepared):
    result = P.fit_ica(prepared)
    assert result.n_components == 8
    assert np.isclose(result.explained.sum(), 1, atol=0.1)
    assert set(result.labels) <= set(P.COMPONENT_LABELS)
    blinks = [i for i, label in enumerate(result.labels) if label == "Eye blink"]
    assert len(blinks) == 1
    fz = prepared.get_data(picks="Fz")
    cleaned = P.remove_components(prepared, result.ica, blinks)
    # Blinks dominate Fz; removing the component removes most of its variance.
    assert cleaned.get_data(picks="Fz").std() < 0.3 * fz.std()
    freqs, times, power = P.component_spectrogram(result.sources[blinks[0]], result.sfreq)
    assert power.shape == (len(freqs), len(times)) and freqs[-1] <= 40


def test_suggested_labels():
    assert P.suggest_label(1.0, 0.3, 10, 0.05, 1, 0.1, 8, 300, 1.5) == "Eye blink"
    assert P.suggest_label(1.0, 0.3, 10, 0.05, 1, 0.002, 8, 300, 1.5) == "EEG"  # too small
    assert P.suggest_label(0.7, 0.3, 10, 0.05, 1, 0.1, 8, 300, 1.5) == "EEG"  # peak not frontal
    assert P.suggest_label(1.0, 0.3, 10, 0.05, 1, 0.1, 8, 40, 1.5) == "EEG"  # too brief
    assert P.suggest_label(0.2, 0.3, 1, 0.6, 1, 0.1, 8, np.nan, 1.5) == "Muscle (EMG)"
    assert P.suggest_label(0.2, 0.3, 1, 0.1, 20, 0.1, 8, np.nan, 1.5) == "Line noise"
    assert P.suggest_label(0.0, 0.8, 50, 0.0, 1, 0.1, 8, 300, 1.5) == "Channel noise"
    # Spiky but spread out: brief pops are channel noise, the rest "Other".
    assert P.suggest_label(0.2, 0.3, 240, 0.0, 1, 0.8, 8, 90, 1.5) == "Channel noise"
    assert P.suggest_label(0.5, 0.3, 120, 0.0, 1, 0.01, 8, 480, 1.5) == "Other"
    # Calm most of the time, but rare large events carry the variance.
    assert P.suggest_label(0.2, 0.3, 1, 0.0, 1, 0.8, 8, 90, 600) == "Channel noise"
    assert P.suggest_label(0.2, 0.3, 1, 0.0, 1, 0.8, 8, 2600, 600) == "Other"
    assert P.suggest_label(0.3, 0.3, 1, 0.1, 1, 0.1, 8, np.nan, 1.5) == "EEG"
