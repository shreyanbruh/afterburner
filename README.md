<p align="center">
  <img src="assets/logo.png" alt="afterburner" />
</p>

<p align="center"><b>Turn audio files into graphs.</b></p>

<p align="center">
  <a href="https://github.com/shreyanbruh/afterburner/releases"><img alt="Release" src="https://img.shields.io/github/v/release/shreyanbruh/afterburner?include_prereleases&label=release" /></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-blue" />
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green" /></a>
</p>

<p align="center">
  <a href="#installation">Installation</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#graph-reference">Graphs</a> ·
  <a href="#plot-options">Plot options</a> ·
  <a href="#recipes">Recipes</a> ·
  <a href="CHANGELOG.md">Changelog</a>
</p>

<!-- NAV: add a link to any new top-level section in the row above. -->

**afterburner** is a Python library that turns an audio file into graphs. Load a track, compute the data behind a graph, and plot it, in three lines. It covers nine graph types, from standard mel spectrograms to Chladni plate patterns and 3D terrain maps, and every plot shares the same options for colour, size, grid, resolution and export.

```python
import afterburner as ab

audio = ab.audio.load("song.mp3")
S = ab.datagen.mel(audio)
ab.plots.plot_mel(S, audio, cmap="inferno", save_path="mel.png")
```

## Contents

- [Features](#features)
- [Installation](#installation)
- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Loading audio](#loading-audio)
- [Graph reference](#graph-reference)
  - [Mel spectrogram](#mel-spectrogram)
  - [Chromagram](#chromagram)
  - [Phase spectrogram](#phase-spectrogram)
  - [Reassigned spectrogram](#reassigned-spectrogram)
  - [Radial spectrogram](#radial-spectrogram)
  - [Self-similarity matrix](#self-similarity-matrix)
  - [3D terrain](#3d-terrain)
  - [Waterfall](#waterfall)
  - [Chladni pattern](#chladni-pattern)
- [Plot options](#plot-options)
- [Recipes](#recipes)
- [Errors and validation](#errors-and-validation)
- [Troubleshooting](#troubleshooting)
- [Project structure](#project-structure)
- [Versioning and releases](#versioning-and-releases)
- [Development](#development)
- [Dependencies and licenses](#dependencies-and-licenses)
- [License](#license)

<!-- CONTENTS: add a line here for any new top-level section, in the order it appears below. -->

## Features

- **Nine graph types** in one library: mel spectrogram, chromagram, phase spectrogram, reassigned spectrogram, radial spectrogram, self-similarity matrix, 3D terrain, waterfall and Chladni pattern.
- **Data and plotting are separate.** Every graph has a data function in `ab.datagen` and a plot function in `ab.plots`, so you can compute once, plot many times, or work with the raw numbers yourself.
- **One set of options everywhere.** Colormap, figure size, grid styling, resolution, saving and showing work the same way in every plot function.
- **Picture-only export.** `image_only=True` saves just the image, with no axes, labels or margins, at an exact pixel size. This is useful for wallpapers, posters and artwork.
- **Plain matplotlib underneath.** Every plot function returns the matplotlib `Figure` and `Axes`, so you can keep customising the result.
- **Clear errors.** Inputs are checked up front, and an invalid value raises a `ValueError` that says what is wrong.
- **Any audio librosa can read**, including WAV, FLAC, OGG and MP3, loaded as mono.

<!-- FEATURES: add a bullet here for any new capability. -->

## Installation

### Requirements

- **Python 3.10 or newer**
- **Git**, because afterburner is installed straight from GitHub (pip uses Git to download it)

pip installs the dependencies (NumPy, SciPy, librosa and Matplotlib) automatically. See [Dependencies and licenses](#dependencies-and-licenses) for the versions it was tested with.

### Install from GitHub

Install the latest version:

```bash
pip install git+https://github.com/shreyanbruh/afterburner.git
```

Install a specific release by adding its tag after an `@`. The tags are listed on the [Releases page](https://github.com/shreyanbruh/afterburner/releases):

```bash
pip install git+https://github.com/shreyanbruh/afterburner.git@v0.1.0
```

The project is called `afterburner-cli` when installed (this is the name `pip` shows) and `afterburner` when imported:

```python
import afterburner as ab
```

<!-- INSTALL: add extra install options (optional extras, other install methods) as new subsections here. -->

### Use a virtual environment (recommended)

A virtual environment keeps afterburner and its dependencies separate from your other Python projects.

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install git+https://github.com/shreyanbruh/afterburner.git
```

If PowerShell refuses to run the activation script, allow scripts for your user once with `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then try again.

**macOS and Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install git+https://github.com/shreyanbruh/afterburner.git
```

Each time you open a new terminal, activate the environment again before using afterburner.

### Verify the install

```bash
pip show afterburner-cli
python -c "import afterburner as ab; print(ab.audio.Audio)"
```

The first command prints the installed version. The second should print the `Audio` class without any error.

### Upgrade and uninstall

```bash
pip install --upgrade git+https://github.com/shreyanbruh/afterburner.git
pip uninstall afterburner-cli
```

If pip says the requirement is already satisfied when you expected an update, add `--force-reinstall` to the first command.

### Install from a clone (for development)

```bash
git clone https://github.com/shreyanbruh/afterburner.git
cd afterburner
pip install -e .
```

The `-e` flag installs in editable mode, so changes to the source take effect without reinstalling. See [Development](#development).

## Quick start

Every graph follows the same three steps: **load**, **compute**, **plot**.

```python
import afterburner as ab

# 1. Load an audio file (WAV, FLAC, OGG, MP3, ...)
audio = ab.audio.load("song.mp3")

# 2. Compute the data for a graph
S = ab.datagen.mel(audio)

# 3. Plot it. A window opens; close it to continue.
ab.plots.plot_mel(S, audio)
```

Save the picture to a file instead of (or as well as) showing it:

```python
# Show the plot and save it
ab.plots.plot_mel(S, audio, cmap="inferno", save_path="mel.png")

# Save it without opening a window
ab.plots.plot_mel(S, audio, cmap="inferno", save_path="mel.png", show=False)
```

Try a different graph by swapping the data and plot functions:

```python
data = ab.datagen.chladni(audio)
ab.plots.plot_chladni(data, cmap="inferno", save_path="chladni.png", show=False)
```

All nine graphs are listed in [How it works](#how-it-works) and described in the [Graph reference](#graph-reference).

## How it works

afterburner has three layers, and each one is a sub-module of the package.

| Layer | What it does | Main entry points |
|---|---|---|
| `ab.audio` | Loads a file into an immutable `Audio` object holding the samples and the sample rate. | `ab.audio.load` |
| `ab.datagen` | Turns an `Audio` (or a spectrogram made from one) into the numbers behind a graph. | `mel`, `chromagram`, `phase_spectrogram`, `reassigned_spectrogram`, `radial_spectrogram`, `self_similarity`, `terrain_spectrogram`, `waterfall_spectrogram`, `chladni` |
| `ab.plots` | Draws that data with matplotlib and optionally saves it. | `plot_mel`, `plot_chroma`, `plot_phase`, `plot_reassigned`, `plot_radial`, `plot_self_similarity`, `plot_terrain`, `plot_waterfall`, `plot_chladni` |

Keeping data and plotting apart means you can compute a spectrogram once and draw it several ways, or take the numbers and use them for something else entirely.

Each graph pairs a data function with a plot function:

| Graph | Data function | Takes | Returns | Plot function |
|---|---|---|---|---|
| [Mel spectrogram](#mel-spectrogram) | `mel` | `audio` | dB array `S` | `plot_mel(S, audio)` |
| [Chromagram](#chromagram) | `chromagram` | `audio` | `ChromaData` | `plot_chroma(data)` |
| [Phase spectrogram](#phase-spectrogram) | `phase_spectrogram` | `audio` | `PhaseData` | `plot_phase(data)` |
| [Reassigned spectrogram](#reassigned-spectrogram) | `reassigned_spectrogram` | `audio` | `ReassignedData` | `plot_reassigned(data)` |
| [Radial spectrogram](#radial-spectrogram) | `radial_spectrogram` | `S` and `audio` | `RadialData` | `plot_radial(data)` |
| [Self-similarity matrix](#self-similarity-matrix) | `self_similarity` | `S` | 2-D array `ssm` | `plot_self_similarity(ssm, audio)` |
| [3D terrain](#3d-terrain) | `terrain_spectrogram` | `S` | 2-D array `Z` | `plot_terrain(Z, audio)` |
| [Waterfall](#waterfall) | `waterfall_spectrogram` | `S` | `WaterfallData` | `plot_waterfall(data, audio)` |
| [Chladni pattern](#chladni-pattern) | `chladni` | `audio` | `ChladniData` | `plot_chladni(data)` |

Four of the graphs (radial, self-similarity, terrain and waterfall) are built from the mel spectrogram `S`, so compute it once with `ab.datagen.mel` and reuse it. The other five are computed straight from the audio.

Results that bundle several arrays (`ChromaData`, `PhaseData`, `RadialData`, `ReassignedData`, `WaterfallData`, `ChladniData`) are named tuples. You can read their fields by name or unpack them:

```python
data = ab.datagen.chromagram(audio)

print(data.chroma.shape)             # (12, n_frames)
chroma, times, labels = data         # unpacking works too
```

In every plot function the data arguments come first and every option after them is **keyword-only**, so write `cmap="inferno"` rather than passing it by position.

## Loading audio

`ab.audio.load(path, sr=None, offset=0.0, duration=None)` reads a file and returns an `Audio` object. The audio is always converted to mono.

| Parameter | Default | Description |
|---|---|---|
| `path` | required | Location of the audio file, as a string or `pathlib.Path`. |
| `sr` | `None` | Sample rate in Hz to resample to. `None` keeps the file's own rate. |
| `offset` | `0.0` | Where to start reading, in seconds. Must be less than the file's length. |
| `duration` | `None` | How many seconds to read from `offset`. `None` reads to the end. |

```python
import afterburner as ab

audio = ab.audio.load("song.mp3")                            # the whole file
intro = ab.audio.load("song.mp3", duration=30)               # the first 30 seconds
chorus = ab.audio.load("song.mp3", offset=62.5, duration=20) # 20 seconds starting at 1:02.5
light = ab.audio.load("song.mp3", sr=22050)                  # resampled to 22,050 Hz
```

Lowering `sr` (22,050 Hz is a common choice) makes everything downstream faster and smaller, at the cost of removing frequencies above half that rate.

An `Audio` object holds the clip and some details about it:

```python
audio = ab.audio.load("song.mp3", offset=10, duration=15)

print(audio.audio_array.shape)   # the samples, a 1-D float32 array
print(audio.sampling_rate)       # samples per second (Hz)
print(audio.duration)            # length of this clip in seconds
print(audio.n_samples)           # number of samples in the clip
print(audio.offset)              # 10, where the clip starts in the original file
print(audio.file_path)           # the path exactly as you passed it
```

`Audio` objects are immutable. To use a different section or sample rate, call `ab.audio.load` again. The class itself is also available as `ab.Audio`, and `ab.audio.load` is the same function as `ab.Audio.load`.

Loading fails with a `ValueError` if `offset` is at or beyond the end of the file, or if fewer than 2,048 samples are read (about 0.09 seconds at 22,050 Hz). Unreadable files raise librosa's own error unchanged.

## Graph reference

Each section below says what the graph shows, then lists the parameters of its data function and its plot function with their defaults, followed by examples. The options shared by every plot function (colormap, grid, saving and so on) are in [Plot options](#plot-options).

The examples assume this setup:

```python
import afterburner as ab

audio = ab.audio.load("song.mp3")
S = ab.datagen.mel(audio)
```

### Mel spectrogram

How energy is spread across frequencies over time. Frequency is on the mel scale, which spaces pitches the way we hear them, and loudness is in decibels. It is the standard starting point, and the input for the radial, self-similarity, terrain and waterfall graphs.

**Data function:** `ab.datagen.mel`

| Parameter | Default | Description |
|---|---|---|
| `input_audio` | required | The `Audio` to analyse. |
| `n_fft` | `2048` | Window length in samples for each frame. Larger values give finer frequency detail but blurrier timing. |
| `hop_length` | `None` | Samples between frames. `None` means `n_fft // 4`. Smaller values give more frames. |
| `fmin` | `0` | Lowest frequency covered, in Hz. |
| `fmax` | `None` | Highest frequency covered, in Hz. `None` means half the sample rate. |
| `amin` | `1e-10` | Smallest power value allowed before converting to dB. Rarely needs changing. |
| `top_db` | `80.0` | Values more than this many dB below the loudest are raised to that floor. `None` turns it off. |

**Returns:** a 2-D array in dB with 128 mel bands and one column per frame. The loudest value is 0 and everything else is negative.

**Plot function:** `ab.plots.plot_mel`

| Parameter | Default | Description |
|---|---|---|
| `S` | required | The array returned by `mel`. |
| `input_audio` | required | The `Audio` that `S` came from. |
| `fmin` | `0` | The `fmin` you gave `mel`, so the frequency axis is labelled correctly. |
| `fmax` | `None` | The `fmax` you gave `mel`. `None` means half the sample rate. |
| `figsize` | `(10, 5)` | Width and height in inches. |

```python
# Defaults
ab.plots.plot_mel(S, audio)

# Zoom in on 100 Hz to 8 kHz, and pass the same range to the plot
S_zoom = ab.datagen.mel(audio, fmin=100, fmax=8000)
ab.plots.plot_mel(S_zoom, audio, fmin=100, fmax=8000, cmap="magma")

# Finer frequency detail, and keep only the top 60 dB
S_fine = ab.datagen.mel(audio, n_fft=4096, hop_length=1024, top_db=60)
ab.plots.plot_mel(S_fine, audio, cmap="viridis", figsize=(14, 5))
```

### Chromagram

How much energy sits in each of the 12 pitch classes (C, C#, D, ... B) over time, ignoring which octave a note is in. It is useful for seeing chords and key changes.

**Data function:** `ab.datagen.chromagram`

| Parameter | Default | Description |
|---|---|---|
| `input_audio` | required | The `Audio` to analyse. Very short clips give unreliable results. |
| `hop_length` | `512` | Samples between frames. |
| `threshold` | `0.01` | Chroma values below this level are set to 0 before each frame is scaled, so silence stays empty. Use a smaller value for very quiet recordings. `0` turns it off. |
| `gamma` | `1.0` | Exponent applied to the scaled values. Above 1 makes only the strongest notes stand out. Below 1 lifts weaker notes. |

**Returns:** a `ChromaData` named tuple `(chroma, times, labels)`. `chroma` has shape `(12, n_frames)` with values from 0 to 1.

**Plot function:** `ab.plots.plot_chroma`

| Parameter | Default | Description |
|---|---|---|
| `data` | required | The `ChromaData` returned by `chromagram`. |
| `figsize` | `(10, 5)` | Width and height in inches. |

```python
data = ab.datagen.chromagram(audio)
ab.plots.plot_chroma(data)

# Make only the strongest notes stand out
sharp = ab.datagen.chromagram(audio, gamma=2.0)
ab.plots.plot_chroma(sharp, cmap="viridis", figsize=(12, 4))

# Coarser time steps for a long track
coarse = ab.datagen.chromagram(audio, hop_length=2048)
ab.plots.plot_chroma(coarse, cmap="magma")
```

### Phase spectrogram

A spectrogram where each point is coloured by its **phase deviation** and made brighter the louder it is. Phase deviation measures how far a tone sits from the centre of its frequency bin, so steady tones appear as smooth bands of colour. Because phase wraps around, a cyclic colormap such as `"twilight"` or `"hsv"` suits it well.

**Data function:** `ab.datagen.phase_spectrogram`

| Parameter | Default | Description |
|---|---|---|
| `input_audio` | required | The `Audio` to analyse. |
| `n_fft` | `2048` | Window size in samples. Larger values separate close frequencies better but blur fast changes. |
| `hop_length` | `None` | Samples between frames. `None` means `n_fft // 4`. Smaller values give a smoother picture. |
| `top_db` | `80.0` | How far below the loudest value levels may go, in dB. `None` turns it off. |

**Returns:** a `PhaseData` named tuple `(level_db, phase_deviation, frequencies, times)`. The first frame has no earlier frame to compare with, so every array starts at the second frame.

**Plot function:** `ab.plots.plot_phase`

| Parameter | Default | Description |
|---|---|---|
| `data` | required | The `PhaseData` returned by `phase_spectrogram`. |
| `figsize` | `(10, 5)` | Width and height in inches. |
| `fmax` | `None` | Highest frequency to show, in Hz. `None` shows everything up to half the sample rate. |
| `dynamic_range` | `60.0` | How far below the loudest point, in dB, a point can be and still be visible. Anything quieter is black. |

```python
data = ab.datagen.phase_spectrogram(audio)
ab.plots.plot_phase(data, cmap="twilight")

# Show only the lower frequencies, and only the louder parts
ab.plots.plot_phase(data, cmap="hsv", fmax=5000, dynamic_range=40)

# A smoother, more detailed analysis
detailed = ab.datagen.phase_spectrogram(audio, n_fft=4096, hop_length=256)
ab.plots.plot_phase(detailed, cmap="twilight", fmax=4000)
```

### Reassigned spectrogram

A sharper spectrogram. Instead of placing energy at the centre of each analysis window, it moves each bit to where it really sits in time and frequency, so tones appear as thin lines. Frequency is on a log scale.

**Data function:** `ab.datagen.reassigned_spectrogram`

| Parameter | Default | Description |
|---|---|---|
| `input_audio` | required | The `Audio` to analyse. |
| `n_fft` | `2048` | Window size in samples. |
| `hop_length` | `None` | Samples between frames. `None` means `n_fft // 8`. Smaller values give denser lines. |
| `fmin` | `40` | Lowest frequency on the grid, in Hz. Must be greater than 0. |
| `fmax` | `None` | Highest frequency on the grid, in Hz. `None` means half the sample rate. |
| `n_freq_bins` | `300` | Number of log-spaced frequency bins. |
| `n_time_bins` | `None` | Number of time bins. `None` means one per analysis frame. |
| `sigma` | `0.8` | Gaussian blur in grid cells, which joins up dotted lines. `0` turns it off. |
| `top_db` | `80.0` | How far below the loudest value levels may go, in dB. `None` turns it off. |

**Returns:** a `ReassignedData` named tuple `(level_db, frequencies, times)`. `frequencies` and `times` hold bin edges.

**Plot function:** `ab.plots.plot_reassigned`

| Parameter | Default | Description |
|---|---|---|
| `data` | required | The `ReassignedData` returned by `reassigned_spectrogram`. |
| `figsize` | `(10, 5)` | Width and height in inches. |
| `dynamic_range` | `60.0` | How far below the loudest point, in dB, the colour scale reaches. `None` uses the whole range of the data. |

```python
data = ab.datagen.reassigned_spectrogram(audio)
ab.plots.plot_reassigned(data)

# Sharper lines: bigger window, more frames, less blur
sharp = ab.datagen.reassigned_spectrogram(audio, n_fft=4096, hop_length=256, sigma=0.4)
ab.plots.plot_reassigned(sharp, cmap="magma", dynamic_range=50)

# Focus on the range of most musical notes
mid = ab.datagen.reassigned_spectrogram(audio, fmin=100, fmax=4000, n_freq_bins=400)
ab.plots.plot_reassigned(mid, cmap="inferno")
```

### Radial spectrogram

A spectrogram wrapped into a circle. Time runs **outward** from the centre, and frequency runs **clockwise** from the top.

**Data function:** `ab.datagen.radial_spectrogram`

| Parameter | Default | Description |
|---|---|---|
| `S` | required | The mel spectrogram from `ab.datagen.mel`, or any 2-D spectrogram with frequency bands as rows. |
| `input_audio` | required | The `Audio` that `S` came from. Its length sets the radius, in seconds. |
| `max_cols` | `None` | The most rings (time columns) to draw. `None` keeps every frame. Otherwise frames are averaged in groups, which draws faster and avoids hairline seams between very thin rings. It must not exceed the number of frames in `S`. |

**Returns:** a `RadialData` named tuple `(theta, r, values)`.

**Plot function:** `ab.plots.plot_radial`

| Parameter | Default | Description |
|---|---|---|
| `data` | required | The `RadialData` returned by `radial_spectrogram`. |
| `figsize` | `(7, 7)` | Width and height in inches. |

```python
data = ab.datagen.radial_spectrogram(S, audio)
ab.plots.plot_radial(data)

# Average down to at most 300 rings for a faster, cleaner plot
smooth = ab.datagen.radial_spectrogram(S, audio, max_cols=300)
ab.plots.plot_radial(smooth, cmap="magma", grid=True, grid_color="white", grid_alpha=0.3)
```

### Self-similarity matrix

Compares every moment of the track with every other moment. Both axes are time. Bright stripes away from the diagonal mark sections that repeat, such as a chorus coming back.

**Data function:** `ab.datagen.self_similarity`

| Parameter | Default | Description |
|---|---|---|
| `S` | required | A spectrogram in dB, such as the output of `ab.datagen.mel`. Raw power values give meaningless results. |
| `n_mfcc` | `20` | How many MFCC coefficients describe each frame. Fewer compare only the broad shape of the sound, and more include finer detail. It cannot exceed the number of rows in `S`. |

**Returns:** a square 2-D array of shape `(n_frames, n_frames)`. Entry `[i, j]` is the cosine similarity of frames `i` and `j`, close to 1 for the same sound.

**Plot function:** `ab.plots.plot_self_similarity`

| Parameter | Default | Description |
|---|---|---|
| `ssm` | required | The matrix returned by `self_similarity`. |
| `input_audio` | required | The `Audio` that the matrix came from. Its length sets the time axes. |
| `figsize` | `(7, 7)` | Width and height in inches. |

```python
ssm = ab.datagen.self_similarity(S)
ab.plots.plot_self_similarity(ssm, audio)

# More detail in each frame's description
detailed = ab.datagen.self_similarity(S, n_mfcc=40)
ab.plots.plot_self_similarity(detailed, audio, cmap="viridis")
```

The matrix has `n_frames` squared entries, so it grows quickly with the length of the track. For a full-length song, use a larger hop length when computing the mel spectrogram, or analyse a section:

```python
S_long = ab.datagen.mel(audio, hop_length=2048)
ssm = ab.datagen.self_similarity(S_long)
ab.plots.plot_self_similarity(ssm, audio)
```

### 3D terrain

A spectrogram as a 3D landscape. Time and frequency run across the ground and loudness is the height.

**Data function:** `ab.datagen.terrain_spectrogram`

| Parameter | Default | Description |
|---|---|---|
| `S` | required | A 2-D spectrogram, such as the output of `ab.datagen.mel`. |
| `max_cols` | `None` | The most time columns the result may have. `None` keeps every frame. Otherwise frames are averaged in groups. It must not exceed the number of frames in `S`. |
| `sigma` | `(1.6, 0.8)` | Smoothing as `(across frequency bands, across time columns)`, in grid cells. `0` turns smoothing off along that axis. |

**Returns:** a 2-D height map of shape `(n_bands, n_cols)` with values between 0 and 1.

**Plot function:** `ab.plots.plot_terrain`

| Parameter | Default | Description |
|---|---|---|
| `Z` | required | The height map returned by `terrain_spectrogram`. |
| `input_audio` | required | The `Audio` that `Z` came from. Its length sets the time axis. |
| `figsize` | `(9, 6)` | Width and height in inches. |
| `elevation` | `35` | How high the camera is above the terrain, in degrees, from -90 to 90. |
| `azimuth` | `-60` | How far the camera is turned around the terrain, in degrees. |

```python
Z = ab.datagen.terrain_spectrogram(S)
ab.plots.plot_terrain(Z, audio)

# A different viewpoint
ab.plots.plot_terrain(Z, audio, cmap="magma", elevation=60, azimuth=-30)

# Fewer columns and smoother hills
smooth = ab.datagen.terrain_spectrogram(S, max_cols=300, sigma=(2.5, 1.5))
ab.plots.plot_terrain(smooth, audio, cmap="terrain", elevation=25, azimuth=-120)
```

### Waterfall

The spectrum at evenly spaced moments, stacked as overlapping lines with the earliest at the back and the latest at the front. Nearer lines hide the ones behind them, and a colour bar shows time.

**Data function:** `ab.datagen.waterfall_spectrogram`

| Parameter | Default | Description |
|---|---|---|
| `S` | required | A 2-D spectrogram, such as the output of `ab.datagen.mel`. |
| `n_lines` | `70` | How many lines (moments in time) to draw. They are evenly spaced and always include the first and last frame. About 30 to 150 looks best. It must not exceed the number of frames in `S`. |
| `gamma` | `1.3` | Exponent applied to the heights. Above 1 makes only strong peaks stand out. Below 1 lifts quiet values. |
| `sigma` | `1.2` | Blur across frequency bands within each line, in bands. `0` turns it off. |

**Returns:** a `WaterfallData` named tuple `(heights, frame_indices)`.

**Plot function:** `ab.plots.plot_waterfall`

| Parameter | Default | Description |
|---|---|---|
| `data` | required | The `WaterfallData` returned by `waterfall_spectrogram`. |
| `input_audio` | required | The `Audio` that the spectrogram came from. Its length sets the time labels. |
| `figsize` | `(8, 7)` | Width and height in inches. |
| `height_scale` | `4.0` | How tall the peaks are, in line spacings. At 4 the tallest peak reaches over the four lines behind it. |
| `line_width` | `0.8` | Thickness of the lines, in points. |

```python
data = ab.datagen.waterfall_spectrogram(S)
ab.plots.plot_waterfall(data, audio)

# Fewer, taller lines
bold = ab.datagen.waterfall_spectrogram(S, n_lines=40, gamma=1.6)
ab.plots.plot_waterfall(bold, audio, cmap="magma", height_scale=6, line_width=1.2)

# Many fine lines
fine = ab.datagen.waterfall_spectrogram(S, n_lines=120, sigma=0.8)
ab.plots.plot_waterfall(fine, audio, cmap="viridis", height_scale=3, line_width=0.5)
```

### Chladni pattern

The pattern that sand forms on a vibrating metal plate. The loudest pitches in the clip are added together as standing waves, and the **nodal lines**, where the plate stays still, are drawn as bright lines. Different songs give different patterns.

**Data function:** `ab.datagen.chladni`

| Parameter | Default | Description |
|---|---|---|
| `input_audio` | required | The `Audio` to analyse. |
| `n_peaks` | `6` | The most pitches to add together. Fewer give a simpler pattern and more give a busier one. |
| `fmin` | `70` | Lowest frequency to take pitches from, in Hz. |
| `fmax` | `2500` | Highest frequency to take pitches from, in Hz. It must not exceed half the sample rate. |
| `n_fft` | `2048` | Window size in samples. Larger values separate pitches that are close together. |
| `resolution` | `800` | Width and height of the field in grid points. Larger values give finer detail and use more memory. |

**Returns:** a `ChladniData` named tuple `(field, frequencies, weights)`. `frequencies` and `weights` list the pitches that were used, loudest first.

**Plot function:** `ab.plots.plot_chladni`

| Parameter | Default | Description |
|---|---|---|
| `data` | required | The `ChladniData` returned by `chladni`. |
| `figsize` | `(7, 7)` | Width and height in inches. Use a square size to keep the plate undistorted. |
| `line_width` | `0.12` | How thick the nodal lines are, as a fraction of the field's spread. Smaller gives thinner, sharper lines and larger gives wide, soft ones. |
| `circular` | `True` | Cut the square into a round plate, leaving everything outside the circle empty. |

```python
data = ab.datagen.chladni(audio)
ab.plots.plot_chladni(data)

# Print the pitches that made the pattern
print(data.frequencies)    # Hz, loudest first
print(data.weights)        # 1.0 for the loudest

# A busier, thinner-lined pattern
busy = ab.datagen.chladni(audio, n_peaks=12, fmax=4000)
ab.plots.plot_chladni(busy, cmap="inferno", line_width=0.05)

# A square plate with a grid
ab.plots.plot_chladni(data, circular=False, grid=True, grid_color="white", grid_alpha=0.3)
```

## Plot options

These options work the same way in all nine plot functions.

**Shared plot options**

| Parameter | Default | Description |
|---|---|---|
| `cmap` | `"binary"` | The colormap. Any matplotlib colormap name (`"inferno"`, `"viridis_r"`, ...) or a matplotlib `Colormap` object. The default draws strong values in black on white. |
| `figsize` | per graph | Width and height of the figure in inches. Both must be greater than 0. The defaults are listed below. |
| `grid` | `False` | Whether to draw grid lines over the plot. |
| `grid_color` | `"gray"` | Colour of the grid lines, any matplotlib colour. |
| `grid_linestyle` | `"--"` | Style of the grid lines: `"-"`, `"--"`, `"-."`, `":"`, `"solid"`, `"dashed"`, `"dashdot"` or `"dotted"`. |
| `grid_linewidth` | `0.6` | Width of the grid lines in points. Must be greater than 0. |
| `grid_alpha` | `0.6` | How opaque the grid lines are, from 0 (invisible) to 1 (solid). |
| `image_only` | `False` | If `True`, the saved file contains only the picture, with no axes, labels, colour bar, grid or margins. Needs `save_path`. |
| `save_path` | `None` | Where to save the picture. The file type comes from the extension. The folder must already exist. `None` saves nothing. |
| `dpi` | `200` | Resolution of the saved file in dots per inch. Must be greater than 0. |
| `show` | `True` | Whether to open the plot in a window. |

The grid options only have an effect when `grid=True`.

**Default `figsize` by plot**

| Plot function | Default `figsize` |
|---|---|
| `plot_mel` | `(10, 5)` |
| `plot_chroma` | `(10, 5)` |
| `plot_phase` | `(10, 5)` |
| `plot_reassigned` | `(10, 5)` |
| `plot_radial` | `(7, 7)` |
| `plot_self_similarity` | `(7, 7)` |
| `plot_terrain` | `(9, 6)` |
| `plot_waterfall` | `(8, 7)` |
| `plot_chladni` | `(7, 7)` |

### Colormaps

`cmap` takes any [matplotlib colormap](https://matplotlib.org/stable/users/explain/colors/colormaps.html) name. Adding `_r` to a name reverses it. The default `"binary"` gives a clean black-on-white print.

```python
ab.plots.plot_mel(S, audio)                       # "binary": strong values are dark
ab.plots.plot_mel(S, audio, cmap="inferno")       # dark background, warm highlights
ab.plots.plot_mel(S, audio, cmap="viridis")       # perceptually even
ab.plots.plot_mel(S, audio, cmap="magma_r")       # magma, reversed
```

You can also pass your own colormap object:

```python
from matplotlib.colors import LinearSegmentedColormap

ember = LinearSegmentedColormap.from_list("ember", ["#000000", "#ff3d00", "#ffd180"])
ab.plots.plot_mel(S, audio, cmap=ember)
```

### Grid

```python
# Faint white dotted grid on a dark colormap
ab.plots.plot_mel(S, audio, cmap="inferno",
                  grid=True, grid_color="white", grid_linestyle=":",
                  grid_linewidth=0.5, grid_alpha=0.3)
```

### Size and resolution

`figsize` is in inches and `dpi` is dots per inch, so a picture-only export is exactly `figsize × dpi` pixels. For example, `figsize=(10, 5)` at `dpi=200` is 2000 × 1000 pixels. A full-figure export is cropped to its content, so it comes out somewhat smaller than that.

```python
ab.plots.plot_mel(S, audio, figsize=(16, 6), dpi=300, save_path="wide.png", show=False)
```

### Saving

The extension of `save_path` decides the file type, so `.png`, `.jpg`, `.pdf`, `.svg`, `.webp` and the other formats matplotlib supports all work. The folder must already exist, and the path can be a string or a `pathlib.Path`.

```python
ab.plots.plot_mel(S, audio, save_path="mel.png")       # raster image
ab.plots.plot_mel(S, audio, save_path="mel.pdf")       # vector, good for print
ab.plots.plot_mel(S, audio, save_path="mel.svg")       # vector, good for the web
```

By default the whole figure is saved: axes, labels, title and colour bar.

### Picture only

`image_only=True` saves just the picture, edge to edge, with no axes, labels, colour bar, grid or margins. It needs `save_path`. The figure shown on screen is not affected.

```python
ab.plots.plot_chladni(data, cmap="inferno", figsize=(8, 8), dpi=300,
                      save_path="chladni_art.png", image_only=True, show=False)
```

### Showing and returning

With `show=True` (the default) the plot opens in a window and your script waits until you close it. With `show=False` nothing is displayed and the figure is closed after saving. Either way, the function returns the matplotlib `Figure` and `Axes`:

```python
fig, ax = ab.plots.plot_mel(S, audio, show=False)
print(type(fig).__name__, type(ax).__name__)
```

See [Customise the returned figure](#customise-the-returned-figure) for what to do with them.

## Recipes

Short, copy-and-paste examples for common jobs. They assume `import afterburner as ab` and an `audio` loaded as in the [Quick start](#quick-start).

### Compute once, plot many

The mel spectrogram feeds four graphs, so compute it once:

```python
S = ab.datagen.mel(audio)

ab.plots.plot_mel(S, audio, save_path="mel.png", show=False)

radial = ab.datagen.radial_spectrogram(S, audio, max_cols=300)
ab.plots.plot_radial(radial, save_path="radial.png", show=False)

ssm = ab.datagen.self_similarity(S)
ab.plots.plot_self_similarity(ssm, audio, save_path="similarity.png", show=False)

Z = ab.datagen.terrain_spectrogram(S)
ab.plots.plot_terrain(Z, audio, save_path="terrain.png", show=False)

waterfall = ab.datagen.waterfall_spectrogram(S)
ab.plots.plot_waterfall(waterfall, audio, save_path="waterfall.png", show=False)
```

### Try several colormaps

Compute the data once and loop over colormaps:

```python
data = ab.datagen.chladni(audio)

for name in ["binary", "inferno", "magma", "viridis", "twilight"]:
    ab.plots.plot_chladni(data, cmap=name, save_path=f"chladni_{name}.png", show=False)
```

### Analyse one part of a song

Load only the section you care about. The time axes then show that section:

```python
verse = ab.audio.load("song.mp3", offset=30, duration=20)
S = ab.datagen.mel(verse)
ab.plots.plot_mel(S, verse, cmap="magma")
```

### Graph every file in a folder

```python
from pathlib import Path

out = Path("graphs")
out.mkdir(exist_ok=True)

for path in Path("music").glob("*.mp3"):
    track = ab.audio.load(path, sr=22050)
    S = ab.datagen.mel(track)
    ab.plots.plot_mel(S, track, cmap="magma",
                      save_path=out / f"{path.stem}_mel.png",
                      show=False)
```

### Make poster art

Combine `image_only` with a large size and a high `dpi` for a wallpaper or print:

```python
data = ab.datagen.chladni(audio, n_peaks=10, resolution=1600)
ab.plots.plot_chladni(data, cmap="inferno", line_width=0.06,
                      figsize=(12, 12), dpi=300,
                      save_path="poster.png", image_only=True, show=False)
```

That saves a 3600 × 3600 pixel image.

### Customise the returned figure

Every plot function returns the `Figure` and `Axes`, so you can change anything matplotlib can. Use `show=False` so the figure is handed back straight away instead of after a window closes:

```python
fig, ax = ab.plots.plot_mel(S, audio, cmap="magma", show=False)

ax.set_title("Verse 1, mel spectrogram")
fig.savefig("custom.png", dpi=300, bbox_inches="tight")
```

### Look at the numbers

The data functions return plain NumPy arrays and named tuples, so you can use them for other work:

```python
S = ab.datagen.mel(audio)
print(S.shape)                  # (128, n_frames)
print(S.max(), S.min())         # 0.0 and no lower than -80.0 by default

chroma = ab.datagen.chromagram(audio)
strongest = chroma.chroma.mean(axis=1).argmax()
print(chroma.labels[strongest])  # the most prominent pitch class
```

<!-- RECIPES: add new recipes above this line. -->

<!-- SECTIONS: add new top-level sections here, then add them to the Contents list at the top. -->

## Errors and validation

The public functions check their inputs before doing any work. An invalid value raises a `ValueError` with a message that says what is wrong and, where it helps, what the limits are.

```python
try:
    S = ab.datagen.mel(audio, n_fft=1)
except ValueError as error:
    print(error)    # n_fft must be greater than or equal to 2
```

```python
try:
    ab.plots.plot_mel(S, audio, cmap="not_a_colormap")
except ValueError as error:
    print(error)    # cmap 'not_a_colormap' is not a matplotlib colormap
```

Typical checks include:

- Numeric options must be in a sensible range, such as `n_fft` of at least 2, `fmin < fmax <= sampling_rate / 2` and `hop_length <= n_fft`.
- Plot options must be valid, such as a real colormap name, a real colour for `grid_color`, `grid_alpha` between 0 and 1, and a `save_path` whose folder exists and whose extension is a supported image type.
- `image_only=True` needs a `save_path`.
- A silent or flat input is rejected where the result would be meaningless.

The one exception is a file that cannot be read at all. In that case `ab.audio.load` lets librosa's own error through unchanged.

## Troubleshooting

**`ModuleNotFoundError: No module named 'afterburner'`**
Python is not the one you installed into. Activate your virtual environment, then check which Python you are running with `python -c "import sys; print(sys.executable)"`. Reinstall with `python -m pip install git+https://github.com/shreyanbruh/afterburner.git` to be sure pip and Python match.

**`pip` says it cannot find `git`**
Install [Git](https://git-scm.com/downloads), restart your terminal, and try again.

**A file will not load**
Convert it to WAV or FLAC, or install [FFmpeg](https://ffmpeg.org/download.html) so that librosa can decode more formats.

**No window appears (for example on a server or over SSH)**
Use `show=False` together with `save_path` to write the picture to a file.

**A plot is slow or the file is huge**
Long tracks produce a lot of frames. Use `max_cols` in `radial_spectrogram` and `terrain_spectrogram`, a larger `hop_length` in `mel`, a lower `sr` in `ab.audio.load`, or load only a section with `offset` and `duration`.

**`self_similarity` uses a lot of memory**
It builds an `n_frames × n_frames` matrix. Compute the mel spectrogram with a larger `hop_length` (such as 2048), or analyse a section of the track, as shown in the [self-similarity section](#self-similarity-matrix).

**`ValueError: ... cannot be larger than ...`**
Options such as `n_fft`, `max_cols` and `n_lines` have upper limits that depend on the length of your clip. Very short clips can hit them. The message states the limit.

## Project structure

```text
afterburner/
├── assets/
│   └── logo.png               # the logo at the top of this README
├── src/
│   └── afterburner/
│       ├── __init__.py        # exposes audio, datagen, plots and Audio
│       ├── audio.py           # the Audio class and load()
│       ├── datagen/           # data functions, one file per graph
│       │   ├── __init__.py
│       │   ├── mel.py
│       │   ├── chroma.py
│       │   ├── phase.py
│       │   ├── reassigned.py
│       │   ├── radial.py
│       │   ├── self_similarity.py
│       │   ├── terrain.py
│       │   ├── waterfall.py
│       │   └── chladni.py
│       └── plots/             # plot functions, one file per graph
│           ├── __init__.py
│           ├── plot_mel.py
│           ├── plot_chroma.py
│           ├── plot_phase.py
│           ├── plot_reassigned.py
│           ├── plot_radial.py
│           ├── plot_self_similarity.py
│           ├── plot_terrain.py
│           ├── plot_waterfall.py
│           └── plot_chladni.py
├── CHANGELOG.md               # what changed in each version
├── LICENSE                    # MIT license
├── pyproject.toml             # package metadata and dependencies
└── README.md
```

<!-- STRUCTURE: add any new folder to the tree above, in the right place. -->

| Part | Role |
|---|---|
| `audio.py` | Defines the immutable `Audio` class and the `load` function, and checks that a clip has enough samples to analyse. |
| `datagen/` | One module per graph. Each holds the data function, its input checks and, where a graph returns several arrays, the named tuple that carries them. |
| `plots/` | One module per graph. Each holds the plot function, which checks its options, draws with matplotlib, and saves or shows the result. |
| `__init__.py` files | Re-export the public names, which is what makes `ab.datagen.mel` and `ab.plots.plot_mel` work. |
| `pyproject.toml` | The package name, version, Python requirement, dependencies and license. |

## Versioning and releases

The current version is **0.1.0**. afterburner follows [Semantic Versioning](https://semver.org/):

| Change | Version bump | Example |
|---|---|---|
| Bug fix, no change to how the library is used | Patch | 0.1.0 → 0.1.1 |
| New feature, or a change to existing behaviour | Minor | 0.1.0 → 0.2.0 |
| Stable, committed API | Major | 0.x → 1.0.0 |

While the version is below 1.0.0, the API may still change between minor versions, so pin the version you depend on:

```bash
pip install git+https://github.com/shreyanbruh/afterburner.git@v0.1.0
```

Every release is tagged on the [Releases page](https://github.com/shreyanbruh/afterburner/releases), and every change is listed in the [changelog](CHANGELOG.md). To see which version you have installed, run `pip show afterburner-cli`.

## Development

Clone the repository and install it in editable mode with the development tools:

```bash
git clone https://github.com/shreyanbruh/afterburner.git
cd afterburner
python -m venv .venv
```

Activate the environment (`.venv\Scripts\Activate.ps1` on Windows PowerShell, `source .venv/bin/activate` on macOS and Linux), then:

```bash
pip install -e ".[dev]"
```

The `dev` extra installs `pytest`. Editable mode means your changes to the files in `src/afterburner/` take effect immediately.

### Conventions

Every graph follows the same pattern, which keeps the library predictable:

- **One data function and one plot function per graph**, in `datagen/<name>.py` and `plots/plot_<name>.py`.
- **Plot options are keyword-only** and use the shared set: `cmap`, `figsize`, the `grid_*` options, `image_only`, `save_path`, `dpi` and `show`.
- **Plot functions return `(fig, ax)`.**
- **Inputs are validated** up front, and bad values raise a `ValueError` with a plain message.
- **Docstrings list `Args`, `Returns` and `Raises`.**

To add a new graph, add the two modules, export them from `datagen/__init__.py` and `plots/__init__.py`, add a section to this README, and add a line under `Unreleased` in the changelog.

### Contributing

Bug reports and ideas are welcome on the [issue tracker](https://github.com/shreyanbruh/afterburner/issues). For a code change, create a branch, make your change, and open a pull request. Please say what you changed and why, and add a changelog entry.

## Dependencies and licenses

afterburner's dependencies are not pinned, so pip installs the latest compatible versions. The versions below are the ones it was last tested with.

| Dependency | Used for | Tested version | License |
|---|---|---|---|
| [Python](https://www.python.org/) | Runtime | 3.13 (3.10 or newer required) | PSF License |
| [librosa](https://librosa.org/) | Loading audio and spectrogram analysis | 1.0.0 | ISC |
| [NumPy](https://numpy.org/) | Arrays and numerical work | 2.5.3 | BSD 3-Clause |
| [SciPy](https://scipy.org/) | Smoothing and signal processing | 1.18.1 | BSD 3-Clause |
| [Matplotlib](https://matplotlib.org/) | Drawing and saving the plots | 3.11.2 | Matplotlib License (PSF-based) |

Each dependency keeps its own license. afterburner's code does not bundle any of them.

## License

afterburner is released under the **MIT License**. Copyright (c) 2026 Shreyan Banerjee. See [LICENSE](LICENSE) for the full text.
