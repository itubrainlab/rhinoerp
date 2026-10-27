"""Export the analysis configured in the app as a short, readable MNE script.

Only the steps the configuration needs are written (no notch-filter line
when the notch is off, no unit conversion for a trigger channel that does
not need one), so the script reads top to bottom like a methods section.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class Component:
    name: str
    channels: list[str]  # electrode names
    tmin: float  # s
    tmax: float
    condition_a: str
    condition_b: str
    polarity: int  # -1: negative-going peak, +1: positive-going
    n_permutations: int


@dataclass
class AnalysisConfig:
    recording: str
    channels: dict[str, str]  # recording name -> electrode name (or itself)
    channel_types: dict[str, str]  # only channels that are not EEG
    reference: str | list[str]  # "none", "average" or electrode names
    highpass: float | None
    lowpass: float | None
    notch: float | None
    filter_order: int
    trigger_channel: str | None  # None = use the file's annotations
    trigger_is_stim: bool  # already typed as a stim channel by the reader
    trigger_scale: float  # undo the reader's unit scaling (1 = none needed)
    trigger_mask: int | None  # only when it changes the codes
    delay_ms: float
    conditions: dict[str, list]  # name -> trigger codes (or annotation descriptions)
    tmin: float
    tmax: float
    baseline: bool
    reject_uv: float | None
    component: Component | None  # None: topographic maps only
    topomap_latency: float  # used when there is no component
    topomap_ok: bool  # at least 3 EEG channels with positions


def _py(value) -> str:
    """A literal that reads naturally in the script (short floats, no 0x for 0)."""
    if isinstance(value, float):
        return f"{value:g}"
    return repr(value)


def _dict(d: dict, indent: str = "    ") -> str:
    if not d:
        return "{}"
    body = "".join(f"{indent}{k!r}: {v!r},\n" for k, v in d.items())
    return "{\n" + body + "}"


def build_script(cfg: AnalysisConfig) -> str:
    comp = cfg.component
    stem = Path(cfg.recording).name
    out: list[str] = []
    add = out.append

    # ---------------------------------------------------------------- header
    steps = [
        "1. Load the recording and find the trigger events.",
        "2. Keep the selected channels, name them after their electrodes and add",
        "   standard 10-05 electrode positions.",
        "3. Re-reference (optional) and filter.",
        "4. Cut epochs around the events and average them into ERPs.",
    ]
    if comp:
        steps += [
            f"5. Topographic maps of the {comp.name} ({_ms(comp.tmin)}-{_ms(comp.tmax)} ms).",
            f"6. Statistics: {comp.condition_a} vs {comp.condition_b} on the {comp.name}.",
        ]
    elif cfg.topomap_ok:
        steps.append(f"5. Topographic maps at {_ms(cfg.topomap_latency)} ms.")
    add(f'"""ERP analysis of {stem}.\n')
    add(f"Exported from Rhino ERP on {datetime.now():%Y-%m-%d %H:%M}. Steps:\n")
    add("\n".join(steps) + "\n")
    add("Run:  python this_script.py   (requires mne, numpy, scipy, matplotlib)")
    add('"""\n')
    add("import matplotlib.pyplot as plt")
    add("import mne")
    add("import numpy as np")
    if comp:
        add("from scipy import stats")
    add("")

    # ---------------------------------------------------------------- settings
    add("# --- Settings (edit these) " + "-" * 50)
    add(f"RECORDING = {cfg.recording!r}\n")
    add("# Channels to keep: name in the recording -> electrode name (10-05 system)")
    add(f"CHANNELS = {_dict(cfg.channels)}")
    if cfg.channel_types:
        add("# Channels that are not EEG")
        add(f"CHANNEL_TYPES = {_dict(cfg.channel_types)}")
    add("")
    if cfg.trigger_channel:
        add(f"TRIGGER_CHANNEL = {cfg.trigger_channel!r}")
    add("# Delay of the EEG relative to the triggers (e.g. wireless transmission).")
    add("# Events are moved this much later; 0 = no correction.")
    add(f"EEG_DELAY_MS = {_py(cfg.delay_ms)}\n")
    kind = "trigger code(s)" if cfg.trigger_channel else "annotation description(s)"
    add(f"# Conditions: name -> {kind}. Several values are pooled into one condition.")
    add(f"CONDITIONS = {_dict(cfg.conditions)}\n")
    if cfg.reference != "none":
        add('# Reference: "average" or a list of electrodes (e.g. linked mastoids)')
        add(f"REFERENCE = {cfg.reference!r}")
    if cfg.highpass or cfg.lowpass:
        add("# Zero-phase Butterworth filter (Hz); None = no high-/low-pass")
        add(f"HIGHPASS, LOWPASS = {_py(cfg.highpass)}, {_py(cfg.lowpass)}")
    if cfg.notch:
        add(f"LINE_NOISE = {_py(cfg.notch)}  # Hz, removed with a +-2 Hz band-stop")
    add("")
    add("# Epochs (s) and artifact rejection")
    add(f"TMIN, TMAX = {_py(cfg.tmin)}, {_py(cfg.tmax)}")
    if cfg.reject_uv:
        add(f"REJECT_UV = {_py(cfg.reject_uv)}  # drop epochs exceeding this peak-to-peak "
            "amplitude")
    if comp:
        add("")
        add("# ERP component to analyse, and the two conditions to compare")
        add(f"COMPONENT = {comp.name!r}")
        add(f"COMPONENT_CHANNELS = {comp.channels!r}  # averaged")
        add(f"WINDOW = ({_py(comp.tmin)}, {_py(comp.tmax)})  # s")
        add(f"POLARITY = {comp.polarity}  # -1: the peak is negative-going, +1: positive")
        add(f"CONDITION_A, CONDITION_B = {comp.condition_a!r}, {comp.condition_b!r}")
        add(f"N_PERMUTATIONS = {comp.n_permutations}  # for the permutation test")
    elif cfg.topomap_ok:
        add(f"TOPOMAP_TIME = {_py(cfg.topomap_latency)}  # s")
    add("")

    # ---------------------------------------------------------------- 1. load
    add(_section("1. Load the recording and find the events"))
    reader = "read_raw_bdf" if cfg.recording.lower().endswith(".bdf") else "read_raw_edf"
    add(f"raw = mne.io.{reader}(RECORDING, preload=True)")
    if cfg.trigger_channel:
        if not cfg.trigger_is_stim:
            add("# The reader types every channel as EEG; mark the trigger channel as a")
            add("# stim channel so MNE reads its values as event codes.")
            add('raw.set_channel_types({TRIGGER_CHANNEL: "stim"})')
        # mne.find_events truncates, so a stored 19.99999 would read as 19.
        if cfg.trigger_scale != 1:
            add("# Whole trigger codes: undo the reader's unit scaling, then round.")
            add(f"raw.apply_function(lambda x: np.round(x * {cfg.trigger_scale:g}), "
                "picks=TRIGGER_CHANNEL)")
        else:
            add("# Round to whole trigger codes (find_events would truncate 19.99 to 19).")
            add("raw.apply_function(lambda x: np.round(x), picks=TRIGGER_CHANNEL)")
        mask = (f", mask={hex(cfg.trigger_mask)}, mask_type=\"and\""
                if cfg.trigger_mask else "")
        add("events = mne.find_events(raw, stim_channel=TRIGGER_CHANNEL, "
            f"shortest_event=1, consecutive=True{mask})")
        add("# Tag each code with its condition: epochs['name'] then selects all of them.")
        add("event_id = {f\"{name}/{code}\": code")
        add("            for name, codes in CONDITIONS.items() for code in codes}")
    else:
        add("events, codes = mne.events_from_annotations(raw)")
        add("# Tag each description with its condition: epochs['name'] then selects all.")
        add("event_id = {f\"{name}/{description}\": codes[description]")
        add("            for name, descriptions in CONDITIONS.items()")
        add("            for description in descriptions}")
    add('events[:, 0] += round(EEG_DELAY_MS * raw.info["sfreq"] / 1000)\n')

    # ---------------------------------------------------------------- 2. channels
    add(_section("2. Channels and electrode positions"))
    add("raw.pick(list(CHANNELS))")
    if cfg.channel_types:
        add("raw.set_channel_types(CHANNEL_TYPES)")
    add("raw.rename_channels(CHANNELS)")
    add('raw.set_montage("colin27_1005", on_missing="ignore")  # standard_1005 before '
        "MNE 1.13\n")

    # ---------------------------------------------------------------- 3. filter
    add(_section("3. Reference and filters"))
    if cfg.reference != "none":
        add("raw.set_eeg_reference(REFERENCE)")
    if cfg.highpass or cfg.lowpass or cfg.notch:
        add(f'butterworth = dict(ftype="butter", order={cfg.filter_order})')
    if cfg.highpass or cfg.lowpass:
        add('raw.filter(HIGHPASS, LOWPASS, method="iir", iir_params=butterworth)')
    if cfg.notch:
        add("# A band-stop is a filter whose low edge lies above its high edge.")
        add('raw.filter(LINE_NOISE + 2, LINE_NOISE - 2, method="iir",')
        add('           iir_params=dict(ftype="butter", order=2))')
    add("")

    # ---------------------------------------------------------------- 4. epochs
    add(_section("4. Epochs and ERPs"))
    add("epochs = mne.Epochs(")
    add("    raw, events, event_id, tmin=TMIN, tmax=TMAX,")
    if cfg.baseline:
        add("    baseline=(None, 0),  # subtract the pre-stimulus mean")
    else:
        add("    baseline=None,")
    if cfg.reject_uv:
        add('    reject=dict(eeg=REJECT_UV * 1e-6),  # MNE works in volts')
    if cfg.channel_types:
        add('    picks=["eeg", "eog"],')
    add('    preload=True, on_missing="ignore",')
    add(")")
    add("print(epochs)")
    add("evokeds = {name: epochs[name].average() for name in CONDITIONS}\n")

    # ---------------------------------------------------------------- 5. topomaps
    if comp and cfg.topomap_ok:
        add(_section(f"5. Topographic maps of the component"))
        add("# Mean amplitude over the component window, for each condition and for")
        add("# their difference, on one colour scale.")
        add("difference = mne.combine_evoked([evokeds[CONDITION_A], evokeds[CONDITION_B]],"
            " weights=[1, -1])")
        add("maps = {CONDITION_A: evokeds[CONDITION_A], CONDITION_B: evokeds[CONDITION_B],")
        add('        f"{CONDITION_A} - {CONDITION_B}": difference}')
        add("window = (difference.times >= WINDOW[0]) & (difference.times <= WINDOW[1])")
        add("amplitudes = {title: evoked.copy().pick(\"eeg\").data[:, window].mean(axis=1)"
            " * 1e6")
        add("              for title, evoked in maps.items()}  # µV")
        add("limit = max(np.abs(a).max() for a in amplitudes.values())")
        add("fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), layout=\"constrained\")")
        add("for ax, (title, amplitude) in zip(axes, amplitudes.items()):")
        add("    image, _ = mne.viz.plot_topomap(amplitude, maps[title].copy().pick(\"eeg\")"
            ".info, axes=ax,")
        add('                                    vlim=(-limit, limit), cmap="RdBu_r", '
            "show=False)")
        add("    ax.set_title(title)")
        add('fig.colorbar(image, ax=axes, shrink=0.7, label="µV")')
        add("fig.suptitle(f\"{COMPONENT}, {WINDOW[0] * 1000:.0f}-{WINDOW[1] * 1000:.0f}"
            " ms\")\n")
    elif not comp and cfg.topomap_ok:
        add(_section("5. Topographic maps"))
        add("for name, evoked in evokeds.items():")
        add('    evoked.plot_topomap(times=TOPOMAP_TIME, time_format=f"{name} %0.3f s",'
            " show=False)\n")

    # ---------------------------------------------------------------- 6. stats
    if comp:
        add(_section("6. Statistics: condition A vs condition B"))
        add("# Single trials, averaged over the component channels, in µV.")
        add("# Trials are the observations: these are single-subject statistics.")
        add("a = epochs[CONDITION_A].get_data(picks=COMPONENT_CHANNELS).mean(axis=1) * 1e6")
        add("b = epochs[CONDITION_B].get_data(picks=COMPONENT_CHANNELS).mean(axis=1) * 1e6")
        add("n_a, n_b = len(a), len(b)")
        add("times = epochs.times")
        add("window = (times >= WINDOW[0]) & (times <= WINDOW[1])\n")
        add("# Mean amplitude in the window, per trial. These tests are only valid if the")
        add("# window was chosen before looking at the data.")
        add("amp_a, amp_b = a[:, window].mean(axis=1), b[:, window].mean(axis=1)")
        add("difference = amp_a.mean() - amp_b.mean()\n")
        add("# (a) Welch's t-test (no equal-variance assumption).")
        add("welch = stats.ttest_ind(amp_a, amp_b, equal_var=False)\n")
        add("# (b) Permutation test: shuffle the condition labels and count how often a")
        add("#     difference at least as large arises by chance (two-sided).")
        add("rng = np.random.default_rng(0)")
        add("pooled = np.concatenate([amp_a, amp_b])")
        add("shuffled = pooled[[rng.permutation(len(pooled)) for _ in range(N_PERMUTATIONS)]]")
        add("null = shuffled[:, :n_a].mean(axis=1) - shuffled[:, n_a:].mean(axis=1)")
        add("p_permutation = (np.sum(np.abs(null) >= abs(difference)) + 1) / "
            "(N_PERMUTATIONS + 1)\n")
        add("# (c) Effect size: Hedges' g (Cohen's d with a small-sample correction).")
        add("pooled_sd = np.sqrt(((n_a - 1) * amp_a.var(ddof=1) + (n_b - 1) * "
            "amp_b.var(ddof=1))")
        add("                    / (n_a + n_b - 2))")
        add("hedges_g = difference / pooled_sd * (1 - 3 / (4 * (n_a + n_b) - 9))\n")
        add("# (d) Peak amplitude and latency on each condition's average ERP: the most")
        add("#     negative (or positive) point in the window. A 'peak' on the window edge")
        add("#     is just a slope, so there is no clear peak.")
        add("def peak(erp):")
        add("    i = np.argmax(erp[window] * POLARITY)")
        add("    latency = times[window][i]")
        add("    on_edge = i in (0, window.sum() - 1)")
        add("    return erp[window][i], latency, on_edge\n")
        add("peak_a, peak_b = peak(a.mean(axis=0)), peak(b.mean(axis=0))\n")
        add("# (e) Latency difference, tested with the jackknife (Miller et al., 1998):")
        add("#     single-trial peaks are too noisy, so the peak is measured on")
        add("#     leave-one-out averages and the t statistic corrected for their")
        add("#     reduced variance. Only meaningful when both conditions have a peak.")
        add("def leave_one_out_latencies(x):")
        add("    return np.array([peak((x.sum(axis=0) - trial) / (len(x) - 1))[1] for trial in x])\n")
        add("latency_difference = peak_a[1] - peak_b[1]")
        add("p_latency = np.nan")
        add("if not (peak_a[2] or peak_b[2]):")
        add("    jack_a, jack_b = leave_one_out_latencies(a), leave_one_out_latencies(b)")
        add("    se = np.sqrt((n_a - 1) / n_a * np.sum((jack_a - jack_a.mean()) ** 2)")
        add("                 + (n_b - 1) / n_b * np.sum((jack_b - jack_b.mean()) ** 2))")
        add("    if se > 0:")
        add("        p_latency = 2 * stats.t.sf(abs(latency_difference / se), n_a + n_b - 2)\n")
        add("def describe_peak(amplitude, latency, on_edge):")
        add('    edge = " (window edge: no clear peak)" if on_edge else ""')
        add('    return f"{amplitude:+.2f} µV at {latency * 1000:.0f} ms{edge}"\n')
        add('print(f"\\n{COMPONENT} at {\'+\'.join(COMPONENT_CHANNELS)}, "')
        add('      f"{WINDOW[0] * 1000:.0f}-{WINDOW[1] * 1000:.0f} ms")')
        add('print(f"  {CONDITION_A}: n = {n_a}, mean {amp_a.mean():+.2f} ± "')
        add('      f"{amp_a.std(ddof=1) / np.sqrt(n_a):.2f} µV, peak {describe_peak(*peak_a)}")')
        add('print(f"  {CONDITION_B}: n = {n_b}, mean {amp_b.mean():+.2f} ± "')
        add('      f"{amp_b.std(ddof=1) / np.sqrt(n_b):.2f} µV, peak {describe_peak(*peak_b)}")')
        add('print(f"  difference {difference:+.2f} µV: Welch t({welch.df:.1f}) = "')
        add('      f"{welch.statistic:.2f}, p = {welch.pvalue:.4f}; permutation p = "')
        add('      f"{p_permutation:.4f} ({N_PERMUTATIONS}); Hedges\' g = {hedges_g:+.2f}")')
        add("if np.isnan(p_latency):")
        add('    print("  latency difference: not testable (no clear peak in both)")')
        add("else:")
        add('    print(f"  latency difference {latency_difference * 1000:+.0f} ms, "')
        add('          f"jackknife p = {p_latency:.4f}")\n')
        add("# Graph: mean ± standard error of both conditions, the component window")
        add("# (grey) and each condition's peak (dot, when there is a clear one).")
        add("fig, ax = plt.subplots(figsize=(9, 4.5), layout=\"constrained\")")
        add('for x, (amp, lat, edge), name, color in ((a, peak_a, CONDITION_A, "#eb6834"),')
        add('                                          (b, peak_b, CONDITION_B, "#2a78d6")):')
        add("    mean, se = x.mean(axis=0), x.std(axis=0, ddof=1) / np.sqrt(len(x))")
        add('    ax.plot(times * 1000, mean, color=color, lw=2, label=f"{name} (n={len(x)})")')
        add("    ax.fill_between(times * 1000, mean - se, mean + se, color=color, alpha=0.2, "
            "lw=0)")
        add("    if not edge:")
        add('        ax.plot(lat * 1000, amp, "o", color=color, markeredgecolor="black")')
        add('ax.axvspan(WINDOW[0] * 1000, WINDOW[1] * 1000, color="0.85", zorder=0,')
        add('           label=f"{COMPONENT} window")')
        add('ax.axhline(0, color="0.5", lw=0.8)')
        add('ax.axvline(0, color="0.5", lw=0.8, ls="--")')
        add('ax.set(xlabel="Time (ms)", ylabel="Amplitude (µV)",')
        add("       title=f\"{COMPONENT} at {', '.join(COMPONENT_CHANNELS)}: \"")
        add('             f"t = {welch.statistic:.2f}, p = {welch.pvalue:.3f}, "')
        add('             f"permutation p = {p_permutation:.3f}")')
        add('ax.legend(frameon=False, loc="upper left")\n')

    add("plt.show()")
    return "\n".join(out) + "\n"


def _ms(seconds: float) -> str:
    return f"{seconds * 1000:.0f}"


def _section(title: str) -> str:
    return f"# --- {title} " + "-" * max(4, 72 - len(title))
