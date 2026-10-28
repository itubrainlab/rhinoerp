import mne
import numpy as np
import pytest

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
    # A recording that starts on an event keeps it.
    cropped = P.crop(raw, events[0, 0] / raw.info["sfreq"], raw.times[-1])
    assert P.find_events(cropped, "TRIG")[0][0, 0] == events[0, 0]


def test_filter_is_mne_butterworth(raw):
    out = P.apply_filter(raw, P.FilterSettings(0.1, 30, 50, 4))
    expected = raw.copy()
    expected.filter(0.1, 30, method="iir", iir_params=dict(ftype="butter", order=4),
                    verbose="error")
    expected.filter(52, 48, method="iir", iir_params=dict(ftype="butter", order=2),
                    verbose="error")
    assert np.array_equal(out.get_data(), expected.get_data())


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
    result = P.compare_conditions(epochs, "30", "20", P.COMPONENT_PRESETS["N170"])
    assert result.component.channels == ("PO7", "PO8")
    assert result.diff < -4 and result.t < -5 and result.p < 0.001 and result.df > 0
    assert abs(result.a.peak_latency - 0.17) < 0.012


def test_ica_fit_segment(raw, prepared):
    events, _ = P.find_events(raw, "TRIG")
    segment = P.event_time_range(prepared, events)
    result = P.fit_ica(prepared, segment=segment)
    assert result.sources.shape == (result.n_components, prepared.n_times)


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
    assert result.frontal_channel == "Fz" and abs(result.blink_score[blinks[0]]) > 0.8
