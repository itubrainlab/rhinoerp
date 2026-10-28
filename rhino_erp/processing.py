"""Offline ERP processing: loading, channels, events, filtering, ICA, ERPs.

Everything here is Qt-free so it can be scripted or tested on its own.
All processing is done with MNE functions, so the exported MNE script
can repeat it exactly; the one exception is the Welch t-test (SciPy).
"""

from __future__ import annotations

from dataclasses import dataclass

import mne
import numpy as np

# ---------------------------------------------------------------------------
# Loading


def load_bdf(path: str) -> mne.io.BaseRaw:
    """Read a BioSemi BDF (or EDF) file fully into memory."""
    if path.lower().endswith(".bdf"):
        return mne.io.read_raw_bdf(path, preload=True, verbose="error")
    return mne.io.read_raw_edf(path, preload=True, verbose="error")


# ---------------------------------------------------------------------------
# Channels and electrode positions

#: Channel types the user can assign in the channel table.
CHANNEL_TYPES = ("eeg", "eog", "emg", "ecg", "misc")

#: g.tec Unicorn Hybrid Black: fixed electrode positions of EEG 1-8.
UNICORN_LABELS = ("Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8")

#: How many leading channels are preselected as EEG.
N_DEFAULT_EEG = len(UNICORN_LABELS)


def _position_montage() -> mne.channels.DigMontage:
    """Standard 10-05 positions that electrode labels are looked up in."""
    # MNE 1.13 deprecated standard_1005 in favour of colin27_1005.
    builtin = mne.channels.get_builtin_montages()
    name = "colin27_1005" if "colin27_1005" in builtin else "standard_1005"
    return mne.channels.make_standard_montage(name)


def electrode_labels() -> list[str]:
    """10-05 electrode names offered in the channel table."""
    return list(_position_montage().ch_names)


def data_channels(raw: mne.io.BaseRaw) -> list[str]:
    """Every channel except stim channels, in file order."""
    stims = set(stim_channels(raw))
    return [c for c in raw.ch_names if c not in stims]


def default_channel_setup(raw: mne.io.BaseRaw) -> list[tuple[str, bool, str, str]]:
    """``(name, selected, type, electrode)`` for every data channel.

    The first eight channels are selected as EEG and carry the Unicorn
    Hybrid Black electrode labels; the rest start deselected.
    """
    rows = []
    for i, name in enumerate(data_channels(raw)):
        if i < N_DEFAULT_EEG:
            rows.append((name, True, "eeg", UNICORN_LABELS[i]))
        else:
            # Readers type every BDF channel as EEG, including a Unicorn's
            # accelerometer, gyroscope and counter channels.
            rows.append((name, False, "misc", ""))
    return rows


def apply_channel_setup(
    raw: mne.io.BaseRaw,
    include: list[str],
    types: dict[str, str],
    electrodes: dict[str, str],
) -> tuple[mne.io.BaseRaw, list[str]]:
    """Keep ``include``, set channel types and assign electrode positions.

    A channel given an electrode label is renamed to it, and gets that
    label's 10-05 position.

    Returns:
        The new raw and the EEG channels left without a position.
    """
    out = raw.copy().pick(include)
    out.set_channel_types({c: types[c] for c in include}, verbose="error")
    mapping = {c: electrodes[c].strip() for c in include
               if electrodes.get(c, "").strip() and electrodes[c].strip() != c}
    labels = [mapping.get(c, c) for c in include]
    duplicates = sorted({x for x in labels if labels.count(x) > 1})
    if duplicates:
        raise ValueError(f"Electrode labels used more than once: {duplicates}")
    out.rename_channels(mapping)
    montage = _position_montage()
    out.set_montage(montage, on_missing="ignore", verbose="error")
    eeg = [out.ch_names[i] for i in mne.pick_types(out.info, eeg=True)]
    return out, [c for c in eeg if c not in montage.ch_names]


def eeg_with_positions(info: mne.Info) -> np.ndarray:
    """Indices of EEG channels that have a usable electrode position."""
    picks = mne.pick_types(info, eeg=True)
    return np.array([i for i in picks
                     if np.all(np.isfinite(info["chs"][i]["loc"][:3]))
                     and np.any(info["chs"][i]["loc"][:3])], dtype=int)


# ---------------------------------------------------------------------------
# Trigger channel and events

ANNOTATIONS = "(annotations)"

_UNIT_SCALE = {"V": 1.0, "mV": 1e3, "uV": 1e6, "µV": 1e6, "nV": 1e9}


def stim_channels(raw: mne.io.BaseRaw) -> list[str]:
    return [raw.ch_names[i] for i in mne.pick_types(raw.info, stim=True)]


def guess_trigger_channel(raw: mne.io.BaseRaw) -> str:
    """A stim channel, else one named like a trigger, else annotations."""
    stims = stim_channels(raw)
    if stims:
        return stims[0]
    for name in raw.ch_names:
        if any(k in name.lower() for k in ("trig", "status", "marker", "event", "stim")):
            return name
    return ANNOTATIONS


def trigger_scale(raw: mne.io.BaseRaw, channel: str) -> float:
    """Factor that undoes the reader's unit scaling of a trigger channel.

    Readers convert a channel stored in e.g. µV to volts; trigger codes
    need the stored numbers back. Stim channels are not scaled.
    """
    if channel in stim_channels(raw):
        return 1.0
    unit = getattr(raw, "_orig_units", {}).get(channel, "V")
    return _UNIT_SCALE.get(unit, 1.0)


def find_events(
    raw: mne.io.BaseRaw, trigger_channel: str, mask: int | None = 0xFFFF
) -> tuple[np.ndarray, dict[str, int]]:
    """Return MNE-style events and a name->code map.

    Any channel can carry the triggers: a copy of it is typed as a stim
    channel, its values converted back to the units stored in the file and
    rounded to whole codes (``mne.find_events`` would truncate 19.99 to 19),
    then ``mne.find_events`` takes every change to a non-zero code as an
    event, including one at the very first sample. With ``ANNOTATIONS``
    the file's annotations are used instead.
    """
    if trigger_channel == ANNOTATIONS:
        events, event_id = mne.events_from_annotations(raw, verbose="error")
        return events, {k: int(v) for k, v in event_id.items()}
    scale = trigger_scale(raw, trigger_channel)
    trig = raw.copy().pick([trigger_channel])
    trig.set_channel_types({trigger_channel: "stim"}, verbose="error")
    trig.apply_function(lambda x: np.round(x * scale), picks=[trigger_channel])
    events = mne.find_events(
        trig, stim_channel=trigger_channel, shortest_event=1, consecutive=True,
        initial_event=True, mask=mask or None, mask_type="and", verbose="error")
    found = sorted(set(events[:, 2].tolist()))
    return events, {str(c): int(c) for c in found}


def shift_events(events: np.ndarray, delay_ms: float, sfreq: float) -> np.ndarray:
    """Compensate an EEG recording delay by moving the events.

    A positive delay means the EEG lags the triggers (e.g. wireless
    transmission), so each event is moved that much later; a negative one
    moves events earlier. Epochs that no longer fit are dropped by MNE.
    """
    return mne.event.shift_time_events(events, None, delay_ms / 1000.0, sfreq)


def group_events(
    events: np.ndarray, groups: dict[str, list[int]]
) -> tuple[np.ndarray, dict[str, int]]:
    """Events and an MNE event_id for condition name -> trigger codes.

    A condition pooling several codes gets one new code of its own.
    """
    event_id = {}
    for name, codes in groups.items():
        if len(codes) == 1:
            event_id[name] = int(codes[0])
        else:
            new = int(events[:, 2].max()) + 1000 + len(event_id)
            events = mne.merge_events(events, codes, new, replace_events=False)
            event_id[name] = new
    return events, event_id


def event_time_range(raw: mne.io.BaseRaw, events: np.ndarray) -> tuple[float, float]:
    """Times (s from the start of ``raw``) of its first and last events.

    Rounded outwards to the millisecond so that cropping to the range
    keeps both events. The whole recording if there are no events.
    """
    if not len(events):
        return 0.0, float(raw.times[-1])
    samples = events[:, 0] - raw.first_samp
    sf = raw.info["sfreq"]
    start = np.floor(1000 * samples.min() / sf) / 1000
    end = np.ceil(1000 * samples.max() / sf) / 1000
    return float(start), float(min(end, raw.times[-1]))


def crop(raw: mne.io.BaseRaw, start: float, end: float) -> mne.io.BaseRaw:
    """The part of ``raw`` from ``start`` to ``end`` (s from its start, both kept)."""
    if start <= 0 and end >= raw.times[-1] - 1e-3:  # the whole recording (to the ms)
        return raw
    if end <= start:
        raise ValueError(f"The range must end after it starts ({start:g}-{end:g} s).")
    return raw.copy().crop(tmin=max(0.0, start), tmax=min(end, raw.times[-1]))


def event_counts(events: np.ndarray) -> dict[int, int]:
    codes, counts = np.unique(events[:, 2], return_counts=True)
    return dict(zip(codes.tolist(), counts.tolist()))


# ---------------------------------------------------------------------------
# Filtering


@dataclass
class FilterSettings:
    highpass: float | None = 0.1
    lowpass: float | None = 30.0
    notch: float | None = 50.0
    order: int = 4

    def describe(self) -> str:
        parts = []
        if self.highpass:
            parts.append(f"HP {self.highpass:g} Hz")
        if self.lowpass:
            parts.append(f"LP {self.lowpass:g} Hz")
        if self.notch:
            parts.append(f"notch {self.notch:g} Hz")
        return ", ".join(parts) or "no filter"


REF_NONE = "none (as recorded)"
REF_AVERAGE = "average"


def apply_reference(raw: mne.io.BaseRaw, ref: str | list[str]) -> mne.io.BaseRaw:
    """Return a re-referenced copy of ``raw``'s EEG channels.

    ``ref`` is ``REF_NONE``, ``REF_AVERAGE`` or a list of EEG channel names
    whose mean becomes the reference (e.g. linked mastoids).
    """
    out = raw.copy()
    if ref == REF_NONE or not ref:
        return out
    ref_channels = "average" if ref == REF_AVERAGE else list(ref)
    out.set_eeg_reference(ref_channels, projection=False, verbose="error")
    return out


def apply_filter(raw: mne.io.BaseRaw, settings: FilterSettings) -> mne.io.BaseRaw:
    """Return a filtered copy of ``raw``; stim channels are left untouched.

    MNE zero-phase IIR Butterworth filters: a high-pass and/or low-pass of
    the given order, and a band-stop of order 2 at the line frequency ±2 Hz.
    """
    out = raw.copy()
    hp, lp = settings.highpass, settings.lowpass
    nyq = out.info["sfreq"] / 2
    if lp and lp >= nyq:
        raise ValueError(f"Low-pass {lp} Hz must be below Nyquist ({nyq} Hz).")
    if hp and lp and hp >= lp:
        raise ValueError("High-pass cutoff must be below low-pass cutoff.")
    if hp or lp:
        out.filter(hp, lp, method="iir",
                   iir_params=dict(ftype="butter", order=settings.order), verbose="error")
    if settings.notch and settings.notch < nyq - 2:
        f = settings.notch
        # A band-stop is a filter whose low edge lies above its high edge.
        out.filter(f + 2, f - 2, method="iir", iir_params=dict(ftype="butter", order=2),
                   verbose="error")
    return out


# ---------------------------------------------------------------------------
# Independent component analysis (EEG channels only, no EOG)


#: ICA is fitted on a copy high-passed at this frequency: slow drifts
#: otherwise dominate the decomposition. The fit is then applied to the
#: data as filtered in step 3.
ICA_HIGHPASS = 1.0

#: Labels a component can be given. The first is the default guess.
COMPONENT_LABELS = ("EEG", "Eye blink", "Muscle (EMG)", "Line noise",
                    "Channel noise", "Other")
#: Labels of components that are not kept by default.
ARTIFACT_LABELS = ("Eye blink", "Muscle (EMG)", "Line noise", "Channel noise")


#: Blink detection: absolute correlation of a component with the most
#: anterior EEG electrode (1-10 Hz), above which it is labelled a blink.
BLINK_CORRELATION = 0.8
#: Muscle detection: MNE's muscle score above which it is labelled muscle.
MUSCLE_SCORE = 0.5


@dataclass
class ICAResult:
    ica: mne.preprocessing.ICA
    sources: np.ndarray  # (n_components, n_times) component time courses
    sfreq: float
    explained: np.ndarray  # fraction of EEG variance per component (MNE)
    frontal_channel: str | None  # the blink proxy; None without positions
    blink_score: np.ndarray  # correlation with the frontal channel (nan: none)
    muscle_score: np.ndarray  # MNE's muscle score (nan: not computable)
    labels: list[str]  # suggested label per component

    @property
    def n_components(self) -> int:
        return len(self.explained)


def frontal_channel(info: mne.Info) -> str | None:
    """The most anterior EEG electrode with a position, the blink proxy."""
    picks = eeg_with_positions(info)
    if not len(picks):
        return None
    y = [info["chs"][i]["loc"][1] for i in picks]
    return info["ch_names"][picks[int(np.argmax(y))]]


def ica_rank(raw: mne.io.BaseRaw) -> int:
    """How many independent components the EEG channels of ``raw`` can hold.

    The number of EEG channels, less one after an average reference and in
    general the rank of the data, estimated from the data: a reference
    applied to the data (not as a projector) removes a dimension the
    metadata does not record. The relative tolerance ignores the float
    rounding left in that missing dimension.
    """
    eeg = raw.copy().pick("eeg")
    return int(mne.compute_rank(eeg, rank=None, tol=1e-4, tol_kind="relative",
                                verbose="error")["eeg"])


def fit_ica(raw: mne.io.BaseRaw, n_components: int | None = None,
            segment: tuple[float, float] | None = None, seed: int = 0) -> ICAResult:
    """Extended-Infomax ICA of the EEG channels, with suggested labels.

    The fit, the explained variance and the labels use only ``segment``
    (start and end in s from the start of ``raw``, e.g. from the first to
    the last trigger); the component time courses cover the whole of
    ``raw``. ``n_components`` defaults to the number of EEG channels and is
    capped at the rank of the data (see ``ica_rank``). Labels come from MNE's
    detectors on the EEG alone: ``find_bads_eog`` with the most anterior EEG
    electrode standing in for an EOG channel ("Eye blink"), and
    ``find_bads_muscle`` ("Muscle (EMG)"); everything else is "EEG". A
    guess to check, not a classification.
    """
    whole = raw.copy().pick("eeg")
    eeg = crop(whole, *segment) if segment else whole
    rank = ica_rank(eeg)
    n_components = min(n_components or len(eeg.ch_names), rank)
    if n_components < 2:
        raise ValueError("ICA needs at least 2 components (and EEG channels).")
    training = eeg.copy().filter(ICA_HIGHPASS, None, verbose="error")
    ica = mne.preprocessing.ICA(
        n_components=n_components, method="infomax", fit_params=dict(extended=True),
        random_state=seed, max_iter="auto", verbose="error")
    ica.fit(training, verbose="error")
    n = ica.n_components_
    sources = ica.get_sources(whole).get_data()
    explained = np.array([
        ica.get_explained_variance_ratio(eeg, components=[i], ch_type="eeg")["eeg"]
        for i in range(n)])

    labels = ["EEG"] * n
    blink_score = np.full(n, np.nan)
    front = frontal_channel(eeg.info)
    if front is not None:
        blinks, blink_score = ica.find_bads_eog(
            eeg, ch_name=front, measure="correlation", threshold=BLINK_CORRELATION,
            verbose="error")
        blink_score = np.asarray(blink_score, dtype=float)
    else:
        blinks = []
    try:
        muscles, muscle_score = ica.find_bads_muscle(eeg, threshold=MUSCLE_SCORE,
                                                     verbose="error")
        muscle_score = np.asarray(muscle_score, dtype=float)
    except Exception:  # e.g. no electrode positions
        muscles, muscle_score = [], np.full(n, np.nan)
    for i in muscles:
        labels[i] = "Muscle (EMG)"
    for i in blinks:
        labels[i] = "Eye blink"
    return ICAResult(ica=ica, sources=sources, sfreq=raw.info["sfreq"],
                     explained=explained, frontal_channel=front,
                     blink_score=blink_score, muscle_score=muscle_score, labels=labels)


def remove_components(raw: mne.io.BaseRaw, ica: mne.preprocessing.ICA,
                      exclude: list[int]) -> mne.io.BaseRaw:
    """A copy of ``raw`` reconstructed from all components but ``exclude``."""
    return ica.apply(raw.copy(), exclude=list(exclude), verbose="error")


@dataclass
class ComponentProperties:
    """What MNE's ``ICA.plot_properties`` shows for a continuous recording."""

    segments: np.ndarray  # (n_segments, n_components, n_times): 2-s segments
    times: np.ndarray  # s within a segment
    freqs: np.ndarray
    psd_db: np.ndarray  # (n_segments, n_components, n_freqs), dB re 1 AU²/Hz
    lowpass: float | None  # the recording's low-pass edge, if below Nyquist


def component_properties(ica: mne.preprocessing.ICA,
                         raw: mne.io.BaseRaw) -> ComponentProperties:
    """The component time courses cut into consecutive 2-s segments, and
    their spectra, as ``ICA.plot_properties`` computes them: multitaper
    PSD up to 1.25 times the low-pass edge (or Nyquist)."""
    sources = ica.get_sources(raw)
    segments = mne.make_fixed_length_epochs(sources, duration=2.0, preload=True,
                                            verbose="error")
    nyquist = raw.info["sfreq"] / 2
    lowpass = raw.info["lowpass"]
    spectrum = segments.compute_psd(picks="all", fmax=min(lowpass * 1.25, nyquist),
                                    verbose="error")
    psds, freqs = spectrum.get_data(return_freqs=True, picks="all", exclude=[])
    return ComponentProperties(
        segments=segments.get_data(picks="all"), times=segments.times, freqs=freqs,
        psd_db=10 * np.log10(psds), lowpass=lowpass if lowpass < nyquist else None)


# ---------------------------------------------------------------------------
# Epochs and ERPs


@dataclass
class EpochSettings:
    tmin: float = -0.2
    tmax: float = 0.8
    baseline: tuple[float | None, float | None] | None = (None, 0.0)
    reject_uv: float | None = 100.0


def compute_erps(
    raw: mne.io.BaseRaw,
    events: np.ndarray,
    event_id: dict[str, int],
    settings: EpochSettings,
) -> tuple[mne.Epochs, dict[str, mne.Evoked]]:
    reject = None
    if settings.reject_uv:
        reject = {"eeg": settings.reject_uv * 1e-6}
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=settings.tmin,
        tmax=settings.tmax,
        baseline=settings.baseline,
        reject=reject,
        picks=["eeg", "eog"],
        preload=True,
        on_missing="ignore",
        verbose="error",
    )
    evokeds = {
        name: epochs[name].average(picks="all")
        for name in event_id
        if name in epochs.event_id and len(epochs[name])
    }
    return epochs, evokeds


def rejection_causes(epochs: mne.Epochs) -> dict[str, int]:
    """How many epochs each channel caused to be rejected, worst first."""
    counts: dict[str, int] = {}
    for reasons in epochs.drop_log:
        for reason in reasons:
            if reason in epochs.ch_names:
                counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


# ---------------------------------------------------------------------------
# ERP statistics


@dataclass
class Component:
    """An ERP component to measure: a time window over a channel cluster."""

    name: str
    tmin: float  # s
    tmax: float  # s
    channels: tuple[str, ...]
    polarity: int  # -1: negative-going (peak = minimum), +1: positive-going


#: Common components with conventional windows and sites. Starting points
#: only -- windows depend on paradigm and population.
COMPONENT_PRESETS = {
    "P1": Component("P1", 0.08, 0.13, ("O1", "O2", "Oz", "PO7", "PO8"), +1),
    "N1": Component("N1", 0.08, 0.15, ("Fz", "FCz", "Cz"), -1),
    "N170": Component("N170", 0.13, 0.20, ("PO7", "PO8", "P7", "P8"), -1),
    "P2": Component("P2", 0.15, 0.25, ("Fz", "FCz", "Cz"), +1),
    "N2": Component("N2", 0.20, 0.35, ("Fz", "FCz", "Cz"), -1),
    "MMN": Component("MMN", 0.10, 0.25, ("Fz", "FCz", "Cz"), -1),
    "P300": Component("P300", 0.25, 0.50, ("Pz", "CPz", "Cz"), +1),
    "N400": Component("N400", 0.30, 0.50, ("Cz", "CPz", "Pz"), -1),
    "LPP": Component("LPP", 0.40, 0.80, ("Pz", "CPz", "Cz"), +1),
}


def cluster_trials(epochs: mne.Epochs, condition: str, channels) -> np.ndarray:
    """Single trials of ``condition``, averaged over ``channels``, in µV.

    Returns:
        ``(n_trials, n_times)``.
    """
    picks = [epochs.ch_names.index(c) for c in channels]
    roi = mne.channels.combine_channels(epochs[condition], {"roi": picks},
                                        method="mean", verbose="error")
    return roi.get_data()[:, 0] * 1e6


@dataclass
class ConditionSummary:
    name: str
    n: int
    mean_amp: float  # µV, mean over the window, averaged over trials
    sem: float
    peak_amp: float  # µV, on the condition's average ERP
    peak_latency: float  # s
    trial_amps: np.ndarray  # per-trial window means, µV


@dataclass
class ComparisonResult:
    component: Component
    a: ConditionSummary
    b: ConditionSummary
    diff: float  # mean_amp a - b
    t: float  # Welch t-test of the window mean amplitudes
    df: float
    p: float  # two-sided
    latency_diff: float  # s, a - b, from the average ERPs
    times: np.ndarray
    erp_a: np.ndarray  # (n_times,) µV
    erp_b: np.ndarray
    se_a: np.ndarray
    se_b: np.ndarray


def _peak(epochs: mne.Epochs, condition: str, channels, comp: Component):
    """(amplitude µV, latency s) of the peak of the condition's average ERP."""
    picks = [epochs.ch_names.index(c) for c in channels]
    roi = mne.channels.combine_channels(epochs[condition].average(), {"roi": picks},
                                        method="mean", verbose="error")
    _, latency, amplitude = roi.get_peak(
        tmin=comp.tmin, tmax=comp.tmax, mode="neg" if comp.polarity < 0 else "pos",
        return_amplitude=True, strict=False)
    return float(amplitude) * 1e6, float(latency)


def compare_conditions(
    epochs: mne.Epochs,
    a: str,
    b: str,
    comp: Component,
) -> ComparisonResult:
    """Compare two conditions on ``comp`` across trials.

    The window mean amplitude of each trial is compared with a two-sided
    Welch t-test (SciPy: MNE has no Welch test with a p-value). Peak
    amplitude and latency come from each condition's average ERP
    (``Evoked.get_peak``).
    """
    from scipy import stats

    channels = [c for c in comp.channels if c in epochs.ch_names]
    if not channels:
        raise ValueError(f"None of {list(comp.channels)} are among the epoch channels.")
    if comp.tmin >= comp.tmax:
        raise ValueError("The component window must start before it ends.")
    times = epochs.times
    win = (times >= comp.tmin) & (times <= comp.tmax)
    if not win.any():
        raise ValueError("The component window lies outside the epoch.")
    xa, xb = cluster_trials(epochs, a, channels), cluster_trials(epochs, b, channels)
    if len(xa) < 2 or len(xb) < 2:
        raise ValueError("Each condition needs at least 2 trials.")

    def summary(name, x):
        amps = x[:, win].mean(1)
        pk, lat = _peak(epochs, name, channels, comp)
        return ConditionSummary(name, len(x), float(amps.mean()),
                                float(amps.std(ddof=1) / np.sqrt(len(x))), pk, lat, amps)

    sa, sb = summary(a, xa), summary(b, xb)
    welch = stats.ttest_ind(sa.trial_amps, sb.trial_amps, equal_var=False)

    return ComparisonResult(
        component=Component(comp.name, comp.tmin, comp.tmax, tuple(channels), comp.polarity),
        a=sa, b=sb, diff=sa.mean_amp - sb.mean_amp, t=float(welch.statistic),
        df=float(welch.df), p=float(welch.pvalue),
        latency_diff=sa.peak_latency - sb.peak_latency,
        times=times, erp_a=xa.mean(0), erp_b=xb.mean(0),
        se_a=xa.std(0, ddof=1) / np.sqrt(len(xa)), se_b=xb.std(0, ddof=1) / np.sqrt(len(xb)),
    )


def comparison_table(res: ComparisonResult) -> list[dict]:
    """Rows for export: one per trial with its window mean amplitude."""
    rows = []
    for s in (res.a, res.b):
        for i, amp in enumerate(s.trial_amps):
            rows.append({"component": res.component.name, "condition": s.name,
                         "trial": i + 1, "mean_amp_uV": float(amp),
                         "window_ms": f"{res.component.tmin*1000:.0f}-{res.component.tmax*1000:.0f}",
                         "channels": "+".join(res.component.channels)})
    return rows
