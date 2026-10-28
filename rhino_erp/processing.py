"""Offline ERP processing: loading, channels, events, filtering, ERPs.

Everything here is Qt-free so it can be scripted or tested on its own.
MNE handles file I/O, epoching and averaging; the filtering runs through
a g.Pype batch pipeline (zero-phase Butterworth sections).
"""

from __future__ import annotations

from dataclasses import dataclass

import mne
import numpy as np

import gpype as gp
from gpype.backend.sources.base.recording_reader import (
    Recording,
    RecordingReader,
)

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

    Any channel can carry the triggers: its values are converted back to
    the units stored in the file, rounded to integer codes, masked, and
    every change to a non-zero code is an event onset. With
    ``ANNOTATIONS`` the file's annotations are used instead.
    """
    if trigger_channel == ANNOTATIONS:
        events, event_id = mne.events_from_annotations(raw, verbose="error")
        return events, {k: int(v) for k, v in event_id.items()}
    values = raw.get_data(picks=[trigger_channel])[0] * trigger_scale(raw, trigger_channel)
    codes = np.rint(values).astype(np.int64)
    if mask:
        codes &= mask
    previous = np.concatenate([[0], codes[:-1]])
    onsets = np.flatnonzero((codes != previous) & (codes != 0))
    events = np.column_stack(
        [onsets + raw.first_samp, previous[onsets], codes[onsets]]
    ).astype(int)
    found = sorted(set(events[:, 2].tolist()))
    return events, {str(c): int(c) for c in found}


def shift_events(events: np.ndarray, delay_ms: float, sfreq: float) -> np.ndarray:
    """Compensate an EEG recording delay by moving the events.

    A positive delay means the EEG lags the triggers (e.g. wireless
    transmission), so each event is moved that much later; a negative one
    moves events earlier. Shifting the events is equivalent to shifting
    the EEG the other way, and needs no padding at the recording edges --
    epochs that no longer fit are simply dropped by MNE.
    """
    shifted = events.copy()
    shifted[:, 0] += int(round(delay_ms * sfreq / 1000.0))
    return shifted


def group_events(
    events: np.ndarray, groups: dict[str, list[int]]
) -> tuple[np.ndarray, dict[str, int]]:
    """Events and an MNE event_id for condition name -> trigger codes.

    A condition pooling several codes gets one new code of its own.
    """
    events = events.copy()
    event_id = {}
    for name, codes in groups.items():
        if len(codes) == 1:
            event_id[name] = int(codes[0])
        else:
            new = int(events[:, 2].max()) + 1000 + len(event_id)
            events = mne.merge_events(events, codes, new, replace_events=False)
            event_id[name] = new
    return events, event_id


def event_counts(events: np.ndarray) -> dict[int, int]:
    codes, counts = np.unique(events[:, 2], return_counts=True)
    return dict(zip(codes.tolist(), counts.tolist()))


# ---------------------------------------------------------------------------
# Filtering with g.Pype


class _ArrayReader(RecordingReader):
    """g.Pype source that emits an in-memory array as one batch block."""

    _pending: Recording | None = None

    def __init__(self, data: np.ndarray, sfreq: float, labels, **kwargs):
        # _load() is called from the base constructor, so stage the data
        # on the class first.
        _ArrayReader._pending = Recording(
            samples=np.ascontiguousarray(data, dtype=np.float32),
            labels=list(labels),
            sampling_rate=float(sfreq),
        )
        super().__init__(file_name="<memory>", mode="batch", **kwargs)

    def _load(self, file_name: str) -> Recording:
        recording, _ArrayReader._pending = _ArrayReader._pending, None
        return recording


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


def gpype_filter(
    data: np.ndarray, sfreq: float, labels, settings: FilterSettings
) -> np.ndarray:
    """Zero-phase filter ``(channel, time)`` data with a g.Pype pipeline."""
    nodes = []
    hp, lp, order = settings.highpass, settings.lowpass, settings.order
    nyq = sfreq / 2
    if lp and lp >= nyq:
        raise ValueError(f"Low-pass {lp} Hz must be below Nyquist ({nyq} Hz).")
    if hp and lp:
        if hp >= lp:
            raise ValueError("High-pass cutoff must be below low-pass cutoff.")
        nodes.append(gp.Bandpass(f_lo=hp, f_hi=lp, order=order, phase="zero"))
    elif hp:
        nodes.append(gp.Highpass(f_c=hp, order=order, phase="zero"))
    elif lp:
        nodes.append(gp.Lowpass(f_c=lp, order=order, phase="zero"))
    if settings.notch and settings.notch < nyq - 2:
        f = settings.notch
        nodes.append(gp.Bandstop(f_lo=f - 2, f_hi=f + 2, order=2, phase="zero"))
    if not nodes:
        return data.copy()

    # Remove DC first: BioSemi data carries large offsets, which would
    # otherwise ring at the edges of the high-pass.
    offset = data.mean(axis=1, keepdims=True)
    pipeline = gp.Pipeline()
    previous = _ArrayReader((data - offset).T, sfreq, labels)
    for node in nodes:
        pipeline.connect(previous, node)
        previous = node
    pipeline.connect(previous, gp.Collector(name="out"))
    result = pipeline.run()
    if isinstance(result, dict):
        result = result["out"]
    return np.asarray(result.data, dtype=np.float64).T


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
    """Return a filtered copy of ``raw``; stim channels are left untouched."""
    out = raw.copy()
    picks = mne.pick_types(out.info, eeg=True, eog=True, emg=True, ecg=True,
                           misc=True)
    if len(picks):
        labels = [out.ch_names[i] for i in picks]
        out._data[picks] = gpype_filter(
            out._data[picks], out.info["sfreq"], labels, settings
        )
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


@dataclass
class ICAResult:
    ica: mne.preprocessing.ICA
    sources: np.ndarray  # (n_components, n_times) component time courses
    sfreq: float
    explained: np.ndarray  # fraction of EEG variance per component
    frontality: np.ndarray  # weight at the most anterior electrodes, 0-1
    focality: np.ndarray  # largest channel weight over the sum of weights
    kurtosis: np.ndarray  # median excess kurtosis over 10-s windows
    high_freq: np.ndarray  # share of 1-40 Hz power above 20 Hz
    line_ratio: np.ndarray  # power at 50/60 Hz over its neighbourhood
    transient_ms: np.ndarray  # median duration of the large deflections (nan: none)
    outlier_ratio: np.ndarray  # variance over robust (MAD-based) variance
    labels: list[str]  # suggested label per component

    @property
    def n_components(self) -> int:
        return len(self.explained)


#: Deflections of a blink component last this long (ms).
BLINK_MS = (150.0, 500.0)
#: Excess kurtosis above which a time course is too spiky to be brain activity.
SPIKY_KURTOSIS = 20.0
#: Kurtosis is measured in windows of this length (s) and the median taken,
#: so a few large transients (e.g. at the start of a recording) do not make
#: an otherwise clean component look spiky.
KURTOSIS_WINDOW = 10.0
#: Variance over robust variance above which rare, large events dominate a
#: component (1-3 for EEG).
OUTLIER_RATIO = 20.0


def suggest_label(frontality, focality, kurtosis, high_freq, line_ratio,
                  explained, n_channels, transient_ms, outlier_ratio) -> str:
    """A first guess at what a component is, from the EEG alone.

    Eye blinks peak at the most anterior electrode and are large, slow
    deflections of 150-500 ms carrying at least 1% of the variance; muscle
    activity is broadband and strong above 20 Hz; line noise is a narrow
    peak at 50 or 60 Hz. A component that sits on a single electrode is
    taken as channel noise. So is one that is spiky throughout, or whose
    variance comes from rare, large events, when those events are shorter
    than 100 ms (electrode pops); otherwise it is "Other". The rest is
    taken to be brain activity. A guess to check, not a
    classification.
    """
    lo, hi = BLINK_MS
    if (np.isfinite(frontality) and frontality >= 0.9 and kurtosis > 5
            and high_freq < 0.3 and explained >= 0.01
            and np.isfinite(transient_ms) and lo <= transient_ms <= hi):
        return "Eye blink"
    if line_ratio > 5:
        return "Line noise"
    if high_freq > 0.4:
        return "Muscle (EMG)"
    if n_channels >= 4 and focality > 0.45:
        return "Channel noise"
    if kurtosis > SPIKY_KURTOSIS or outlier_ratio > OUTLIER_RATIO:
        if np.isfinite(transient_ms) and transient_ms < 100:
            return "Channel noise"
        return "Other"
    return "EEG"


def _transient_ms(source: np.ndarray, sfreq: float) -> float:
    """Median duration of a time course's large deflections, in ms.

    A deflection is a run beyond 4 robust SDs that reaches 8 somewhere;
    nan if there is none.
    """
    mad = 1.4826 * np.median(np.abs(source - np.median(source)))
    if mad == 0:
        return np.nan
    z = np.abs(source - np.median(source)) / mad
    edges = np.flatnonzero(np.diff(np.r_[0, (z > 4).astype(int), 0]))
    widths = [b - a for a, b in zip(edges[::2], edges[1::2]) if z[a:b].max() > 8]
    return 1000 * np.median(widths) / sfreq if widths else np.nan


def fit_ica(raw: mne.io.BaseRaw, seed: int = 0) -> ICAResult:
    """Extended-Infomax ICA of the EEG channels.

    The number of components is the rank of the EEG data (one fewer than
    the channels after an average reference).
    """
    from scipy import stats
    from scipy.signal import welch

    eeg = raw.copy().pick("eeg")
    # Estimated from the data: an average reference applied to the data
    # (not as a projector) removes a dimension the metadata does not record.
    # The relative tolerance ignores the float32 rounding of the g.Pype
    # filter, which leaves ~1e-7 of noise in that missing dimension.
    rank = mne.compute_rank(eeg, rank=None, tol=1e-4, tol_kind="relative",
                            verbose="error")["eeg"]
    if rank < 2:
        raise ValueError("ICA needs at least 2 EEG channels.")
    training = eeg.copy().filter(ICA_HIGHPASS, None, verbose="error")
    ica = mne.preprocessing.ICA(
        n_components=rank, method="infomax", fit_params=dict(extended=True),
        random_state=seed, max_iter="auto", verbose="error")
    ica.fit(training, verbose="error")
    sources = ica.get_sources(eeg).get_data()
    patterns = ica.get_components()  # (n_channels, n_components)
    n = ica.n_components_
    sfreq = raw.info["sfreq"]

    # Share of the EEG variance each component accounts for: the variance
    # of the EEG reconstructed from that component alone, over the total.
    total = eeg.get_data().var(axis=1).sum()
    explained = np.array([
        ica.apply(eeg.copy(), include=[i], exclude=[], verbose="error")
        .get_data().var(axis=1).sum() for i in range(n)]) / total

    # Scalp map: weight at the most anterior electrodes (top quarter of the
    # front-back range) relative to the largest weight, 0-1 (1: the map
    # peaks at the front), and how much
    # of the map sits on a single electrode.
    weights = np.abs(patterns)
    focality = weights.max(axis=0) / weights.sum(axis=0)
    locs = np.array([ch["loc"][:3] for ch in eeg.info["chs"]])
    frontality = np.full(n, np.nan)
    if np.isfinite(locs).all() and np.ptp(locs[:, 1]) > 0:
        y = locs[:, 1]
        front = y >= y.max() - 0.25 * np.ptp(y)
        frontality = weights[front].mean(axis=0) / weights.max(axis=0)

    # Spectrum of each time course.
    freqs, psd = welch(sources, fs=sfreq, nperseg=min(sources.shape[1], int(2 * sfreq)))
    band = (freqs >= 1) & (freqs <= 40)
    high_freq = psd[:, band & (freqs > 20)].sum(axis=1) / psd[:, band].sum(axis=1)
    line_ratio = np.zeros(n)
    for f0 in (50.0, 60.0):
        if f0 + 4 < sfreq / 2:
            peak = psd[:, np.abs(freqs - f0) <= 1].mean(axis=1)
            around = psd[:, (np.abs(freqs - f0) > 2) & (np.abs(freqs - f0) <= 6)].mean(axis=1)
            line_ratio = np.maximum(line_ratio, peak / around)

    win = int(KURTOSIS_WINDOW * sfreq)
    if sources.shape[1] >= 2 * win:
        n_win = sources.shape[1] // win
        windows = sources[:, :n_win * win].reshape(n, n_win, win)
        kurtosis = np.median(stats.kurtosis(windows, axis=2), axis=1)
    else:
        kurtosis = stats.kurtosis(sources, axis=1)
    transient_ms = np.array([_transient_ms(s, sfreq) for s in sources])
    robust_sd = 1.4826 * np.median(
        np.abs(sources - np.median(sources, axis=1, keepdims=True)), axis=1)
    outlier_ratio = sources.var(axis=1) / np.maximum(robust_sd, 1e-30) ** 2
    labels = [suggest_label(frontality[i], focality[i], kurtosis[i], high_freq[i],
                            line_ratio[i], explained[i], len(eeg.ch_names),
                            transient_ms[i], outlier_ratio[i])
              for i in range(n)]
    return ICAResult(ica=ica, sources=sources, sfreq=sfreq, explained=explained,
                     frontality=frontality, focality=focality, kurtosis=kurtosis,
                     high_freq=high_freq, line_ratio=line_ratio,
                     transient_ms=transient_ms, outlier_ratio=outlier_ratio,
                     labels=labels)


def remove_components(raw: mne.io.BaseRaw, ica: mne.preprocessing.ICA,
                      exclude: list[int]) -> mne.io.BaseRaw:
    """A copy of ``raw`` reconstructed from all components but ``exclude``."""
    return ica.apply(raw.copy(), exclude=list(exclude), verbose="error")


def component_spectrogram(source: np.ndarray, sfreq: float, fmax: float = 40.0):
    """Time-frequency power of a component: (freqs, times, power in dB)."""
    from scipy.signal import spectrogram

    nperseg = int(2 * sfreq)
    freqs, times, power = spectrogram(source, fs=sfreq, nperseg=nperseg,
                                      noverlap=nperseg // 2, scaling="density")
    keep = freqs <= fmax
    return freqs[keep], times, 10 * np.log10(power[keep] + 1e-30)


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
    return epochs[condition].get_data(picks=list(channels)).mean(axis=1) * 1e6


@dataclass
class ConditionSummary:
    name: str
    n: int
    mean_amp: float  # µV, mean over the window, averaged over trials
    sem: float
    peak_amp: float  # µV, on the condition's average ERP
    peak_latency: float  # s
    peak_at_edge: bool  # no clear peak inside the window
    trial_amps: np.ndarray  # per-trial window means, µV


@dataclass
class ComparisonResult:
    component: Component
    a: ConditionSummary
    b: ConditionSummary
    diff: float  # mean_amp a - b
    t: float
    df: float
    p_welch: float
    p_perm: float
    n_perm: int
    hedges_g: float
    latency_diff: float  # s, a - b, from the average ERPs
    latency_p: float  # jackknife test of the latency difference; nan if untestable
    times: np.ndarray
    erp_a: np.ndarray  # (n_times,) µV
    erp_b: np.ndarray
    se_a: np.ndarray
    se_b: np.ndarray


def _peak(erp: np.ndarray, times: np.ndarray, comp: Component) -> tuple[float, float]:
    win = (times >= comp.tmin) & (times <= comp.tmax)
    seg = erp[win] * comp.polarity
    i = int(np.argmax(seg))
    return float(erp[win][i]), float(times[win][i])


def _at_edge(latency: float, comp: Component, times: np.ndarray) -> bool:
    """A 'peak' on the window boundary is a slope, not a peak."""
    step = times[1] - times[0]
    return latency - comp.tmin < step / 2 or comp.tmax - latency < step / 2


def _welch(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    from scipy import stats

    res = stats.ttest_ind(x, y, equal_var=False)
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    df = (vx + vy) ** 2 / (vx**2 / (len(x) - 1) + vy**2 / (len(y) - 1))
    return float(res.statistic), float(df), float(res.pvalue)


def _jackknife_latency(xa, xb, times, comp) -> tuple[float, float]:
    """Peak-latency difference and its jackknife p (Miller et al., 1998).

    Single-trial peak latencies are too noisy; the jackknife measures the
    peak on leave-one-out averages and corrects the t statistic for the
    reduced variance of those averages.
    """
    from scipy import stats

    def loo(x):
        total = x.sum(0)
        return np.array([_peak((total - xi) / (len(x) - 1), times, comp)[1] for xi in x])

    la, lb = loo(xa), loo(xb)
    na, nb = len(xa), len(xb)
    d = _peak(xa.mean(0), times, comp)[1] - _peak(xb.mean(0), times, comp)[1]
    se = np.sqrt((na - 1) / na * ((la - la.mean()) ** 2).sum()
                 + (nb - 1) / nb * ((lb - lb.mean()) ** 2).sum())
    if se == 0:
        return d, float("nan")
    t = d / se
    return d, float(2 * stats.t.sf(abs(t), na + nb - 2))


def compare_conditions(
    epochs: mne.Epochs,
    a: str,
    b: str,
    comp: Component,
    n_perm: int = 5000,
    seed: int = 0,
) -> ComparisonResult:
    """Compare two conditions on ``comp`` across trials.

    Window mean amplitude: Welch t-test, two-sided label-permutation test
    and Hedges' g. Peak amplitude and latency come from the average ERPs;
    the latency difference is tested with the jackknife.
    """
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
        pk, lat = _peak(x.mean(0), times, comp)
        return ConditionSummary(name, len(x), float(amps.mean()),
                                float(amps.std(ddof=1) / np.sqrt(len(x))), pk, lat,
                                _at_edge(lat, comp, times), amps)

    sa, sb = summary(a, xa), summary(b, xb)
    t, df, p = _welch(sa.trial_amps, sb.trial_amps)

    rng = np.random.default_rng(seed)
    pooled = np.concatenate([sa.trial_amps, sb.trial_amps])
    observed = sa.mean_amp - sb.mean_amp
    perms = np.array([rng.permutation(len(pooled)) for _ in range(n_perm)])
    shuffled = pooled[perms]
    null = shuffled[:, : sa.n].mean(1) - shuffled[:, sa.n:].mean(1)
    p_perm = (np.sum(np.abs(null) >= abs(observed)) + 1) / (n_perm + 1)

    sd = np.sqrt(((sa.n - 1) * sa.trial_amps.var(ddof=1)
                  + (sb.n - 1) * sb.trial_amps.var(ddof=1)) / (sa.n + sb.n - 2))
    g = observed / sd * (1 - 3 / (4 * (sa.n + sb.n) - 9)) if sd else 0.0

    lat_d, lat_p = _jackknife_latency(xa, xb, times, comp)
    if sa.peak_at_edge or sb.peak_at_edge:
        lat_p = float("nan")  # latency is undefined without a peak in both

    return ComparisonResult(
        component=Component(comp.name, comp.tmin, comp.tmax, tuple(channels), comp.polarity),
        a=sa, b=sb, diff=observed, t=t, df=df, p_welch=p, p_perm=float(p_perm),
        n_perm=n_perm, hedges_g=float(g), latency_diff=lat_d, latency_p=lat_p,
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
