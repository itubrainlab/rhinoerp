# Rhino ERP

<img src="docs/icon.png" alt="Rhino ERP icon" width="128" align="right">

Rhino ERP is a desktop app for offline **ERP analysis** of EEG recordings (BDF/EDF),
made for recordings from [Rhino Recorder](https://github.com/itubrainlab/rhinorecorder)
and the g.tec **Unicorn Hybrid Black**, on Windows, macOS and Linux. Load a recording,
pick channels, filter, pick trigger and conditions, remove ICA components, re-reference,
compute ERPs, test a component between two conditions, and export the whole analysis as
a short MNE script.

- **MNE** does all the processing: loading, events, reference, filtering, ICA and its
  component labels, epochs, ERPs and peaks, so the exported MNE script repeats the
  app's analysis exactly. The one exception is the Welch t-test, from SciPy.
- **g.Pype** provides the Qt theme.

![Rhino ERP](docs/screenshot.png)

## Download

Packages for Windows, macOS (Apple Silicon) and Linux are on the
[releases page](https://github.com/itubrainlab/rhinoerp/releases).

**First launch needs an internet connection.** The packages contain their own Python
and Rhino ERP, but not g.Pype: the g.tec Non-Commercial License does not allow g.Pype
to be redistributed in a bundle. On first launch the app downloads g.Pype, MNE and the
other libraries from PyPI (several hundred MB, a few minutes), at the versions the
release was tested with. Later launches start directly.

- **Windows**: unpack the zip and double-click `install.cmd`, then open Rhino ERP from
  the Start menu.
- **macOS**: open the DMG and drag **Rhino ERP** to Applications. The app is not
  notarized by Apple, so the first time, right-click it and choose **Open**.
- **Linux** (x86_64, glibc 2.34+): unpack the tarball and run `./install.sh`.

## Run from source

```bash
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -e ".[test]"
.venv/bin/python -m rhino_erp path/to/recording.bdf
.venv/bin/python -m pytest -q
```

A synthetic test recording (oddball paradigm with blinks and 50 Hz noise):

```bash
.venv/bin/python scripts/make_demo_bdf.py demo.bdf
```

`packaging/` builds the three packages (`packaging/macos/build_app.sh`,
`packaging/linux/build_linux.sh`, `packaging/windows/build_windows.sh`); the
[workflow](.github/workflows/build.yml) builds, installs and checks them, and attaches
them to the release when a `v*` tag is pushed.

## Workflow

1. **Recording** – open a `.bdf` (an `.edf` also works).
2. **Channels and electrodes** – the first 8 channels are preselected as EEG and
   assigned the Unicorn Hybrid Black electrodes (Fz, C3, Cz, C4, Pz, PO7, Oz, PO8);
   all other channels start deselected as `misc`. Tick channels to include them in
   visualisation and analysis, set their type, and pick (or type) an electrode label
   per channel. A labelled channel is renamed to its electrode and gets the standard
   10-05 position, shown in the *Montage* tab. *Unicorn defaults* restores the
   preselection.
3. **Filtering** – high-pass, low-pass and line-noise notch (MNE zero-phase IIR
   Butterworth filters).
4. **Trigger and events** – choose the trigger channel (any channel, e.g. `TRIG` or
   BioSemi `Status`, or the file's annotations) and the bit mask, then *Find events*.
   Tick the codes to analyse and name them; codes sharing a name are pooled.
   **EEG delay** (slider or ms box, ±500 ms) compensates a recording delay between
   triggers and EEG: a positive value means the EEG lags, so events are moved later.
   It applies to the epochs and the event markers, and computed ERPs update live.
5. **Independent components (ICA, optional)** – *Fit ICA* decomposes the filtered EEG
   (extended Infomax on the EEG channels only, high-passed at 1 Hz for the fit; no EOG
   channel is used), fitted on the data from the first to the last trigger and then
   applied to the whole recording. *Components* sets how many to fit: by default, and
   at most, the rank of the filtered EEG (normally the number of EEG channels). The
   *ICA* tab shows, for every component, the panels of MNE's `ICA.plot_properties`
   computed on the filtered recording and drawn as charts in the app: scalp map; image
   of consecutive 2-s segments of the whole recording (first at the bottom, colours
   saturating at the 99th percentile) over their average with its 95% confidence
   interval; multitaper spectrum (mean ± SD over segments); and the variance of each
   segment. Its table lists the
   components (named as in MNE: ICA000, ICA001, …) with their explained
   variance and a suggested label from MNE's detectors on the EEG alone: *Eye blink*
   when it correlates with the most anterior EEG electrode (`find_bads_eog`,
   |r| ≥ 0.8), *Muscle (EMG)* from `find_bads_muscle`, otherwise *EEG*. Line noise,
   Channel noise and Other can be set by hand. Components labelled Eye blink, Muscle
   (EMG), Line noise or Channel noise start unticked, the rest ticked; changing a label
   ticks or unticks the component to match. Adjust the ticks and click *Reconstruct
   from selected*. Later steps then use the reconstructed signal (also shown as *ICA
   cleaned* in the *Signals* tab); *Undo* goes back.
6. **Reference (optional)** – keep the recording reference, or use the average of the
   selected EEG channels, one EEG channel, or the mean of two such as linked mastoids.
   Applied after ICA, to the data the epochs are cut from (*Referenced* in the
   *Signals* tab); computed ERPs update when it changes.
7. **Epochs and ERPs** – set the window, baseline and peak-to-peak rejection, then
   compute. The panel reports which channels caused rejections. The *Epochs* tab shows
   a single channel, the mean of selected channels, GFP or a butterfly plot; click the
   plot to set the topomap latency. ERPs can be saved as an MNE `-ave.fif` file.
   **Export as MNE script…** writes a short, commented Python script (MNE, NumPy,
   SciPy, matplotlib) that repeats the configured analysis with the same MNE calls. It starts
   with an editable settings block, then runs numbered steps: load and find events,
   channels and positions, filters, ICA reconstruction (the fitted ICA is saved next
   to the script as `<script>-ica.fif`, with the removed components and their
   labels), reference, epochs, and - when a component is set up in the *Statistics*
   tab - topographic maps of the component window (each condition and their
   difference) plus the same statistics as the app (Welch t-test, peaks)
   and a graph of both conditions with the window and peaks. Only the steps the
   configuration needs are written.
8. **Statistics** (tab) – define the ERP component to test: pick a preset (P1, N1,
   N170, P2, N2, MMN, P300, N400, LPP) or edit the name, time window, polarity and
   channel cluster freely (any edit makes it *Custom*). Choose two conditions and run:
   - **window mean amplitude**: two-sided Welch t-test with its p-value (SciPy's
     `ttest_ind`, as MNE has no Welch test with a p-value);
   - **peak amplitude and latency** on the average ERPs (`Evoked.get_peak`) and their
     difference.

   Statistics rerun automatically when the ERPs change (e.g. moving the EEG delay
   slider). *Export CSV…* writes per-trial window amplitudes plus a summary file.
   Trials are the observations, so these are single-subject statistics; the window
   test is only valid if the window was chosen before looking at the data.

## Licence

Rhino ERP is MIT licensed ([LICENSE](LICENSE)). g.Pype, which provides its Qt theme
and which the packages install from PyPI on first launch, is under the g.tec
Non-Commercial License: free for private and educational use with a g.tec amplifier;
commercial use needs a licence from g.tec.
