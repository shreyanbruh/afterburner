import numpy as np
import librosa
from scipy.signal import find_peaks
from typing import NamedTuple
from afterburner.audio import Audio


class ChladniData(NamedTuple):
    """The Chladni field for a clip, with the pitches it was built from.

    It is a tuple, so it can be unpacked (``field, frequencies, weights =
    data``), and its fields can also be read by name.

    Attributes:
        field (np.ndarray): The standing-wave field, a 2-D float array of shape
            ``(resolution, resolution)`` with values between -1 and 1 (the
            largest absolute value is exactly 1). ``field[i, j]`` is the value
            at ``y = i / (resolution - 1)`` and ``x = j / (resolution - 1)``,
            so row 0 is y = 0; plot it with ``origin="lower"`` to put y = 0 at
            the bottom. The nodal lines, where sand would collect on a real
            plate, are where the field crosses 0. The field is antisymmetric
            about the diagonal: ``field[i, j] == -field[j, i]``.
        frequencies (np.ndarray): The frequencies in Hz of the spectral peaks
            that were used, a 1-D float array of shape ``(n_used,)``, loudest
            first. ``n_used`` is at most ``n_peaks``.
        weights (np.ndarray): The relative strength of each peak, a 1-D float
            array of shape ``(n_used,)``. The loudest peak is 1 and the others
            are between 0 and 1.
    """
    field: np.ndarray
    frequencies: np.ndarray
    weights: np.ndarray


def chladni(input_audio: Audio,
            n_peaks: int = 6,
            fmin: float = 70,
            fmax: float = 2500,
            n_fft: int = 2048,
            resolution: int = 800):
    """Build a Chladni-style standing-wave field from the loudest pitches in a clip.

    Args:
        input_audio (Audio): The clip to analyse. Its ``audio_array`` and
            ``sampling_rate`` are used as the signal and its sample rate.
        n_peaks (int): The maximum number of pitches to add together. Defaults
            to 6. Must be at least 1. Fewer pitches give a simpler pattern and
            more give a busier one. If the clip has fewer distinct peaks in
            the frequency range than this, all of them are used.
        fmin (float): The lowest frequency in Hz that peaks are taken from.
            Defaults to 70. Must be at least 0 and less than ``fmax``.
        fmax (float): The highest frequency in Hz that peaks are taken from.
            Defaults to 2500. Must be no greater than half the sampling rate.
        n_fft (int): The size of the analysis window in samples. Defaults to
            2048. Bigger values separate pitches that are close together; two
            pitches closer than ``sampling_rate / n_fft`` Hz are merged into
            one peak. Must be a whole number of at least 2 and no greater than
            the number of samples in the clip.
        resolution (int): The width and height of the field in grid points.
            Defaults to 800. Must be a whole number of at least 2. Larger
            values give finer detail and use more memory.

    Returns:
        ChladniData: A named tuple ``(field, frequencies, weights)``. ``field``
        has shape ``(resolution, resolution)`` with values between -1 and 1,
        and ``frequencies`` and ``weights`` list the peaks that were used,
        loudest first.

    Raises:
        ValueError: If ``n_peaks`` is not a whole number.
        ValueError: If ``n_peaks`` is less than 1.
        ValueError: If ``fmin`` is not a number.
        ValueError: If ``fmax`` is not a number.
        ValueError: If ``fmin`` is negative.
        ValueError: If ``fmin`` is not less than ``fmax``.
        ValueError: If ``fmax`` is greater than half the sampling rate.
        ValueError: If ``n_fft`` is not a whole number.
        ValueError: If ``n_fft`` is less than 2.
        ValueError: If ``n_fft`` is greater than the number of samples in the
            clip.
        ValueError: If ``resolution`` is not a whole number.
        ValueError: If ``resolution`` is less than 2.
        ValueError: If no frequency bins fall between ``fmin`` and ``fmax``.
        ValueError: If the clip has no energy between ``fmin`` and ``fmax``.
        ValueError: If no spectral peaks are found between ``fmin`` and
            ``fmax``.

    """

    sampling_rate = input_audio.sampling_rate
    nyquist = sampling_rate / 2

    if isinstance(n_peaks, bool) or not isinstance(n_peaks, int):
        raise ValueError("n_peaks must be a whole number")

    if (n_peaks < 1):
        raise ValueError("n_peaks must be greater than or equal to 1")

    if isinstance(fmin, bool) or not isinstance(fmin, (int, float)):
        raise ValueError("fmin must be a number")

    if isinstance(fmax, bool) or not isinstance(fmax, (int, float)):
        raise ValueError("fmax must be a number")

    if (fmin < 0):
        raise ValueError("fmin cannot be negative")

    if (fmin >= fmax):
        raise ValueError("fmin must be less than fmax")

    if (fmax > nyquist):
        raise ValueError(f"fmax cannot exceed {nyquist} Hz (half the sampling rate)")

    if isinstance(n_fft, bool) or not isinstance(n_fft, int):
        raise ValueError("n_fft must be a whole number")

    if (n_fft < 2):
        raise ValueError("n_fft must be greater than or equal to 2")

    if (n_fft > input_audio.n_samples):
        raise ValueError(f"n_fft cannot exceed {input_audio.n_samples} samples (the length of the clip)")

    if isinstance(resolution, bool) or not isinstance(resolution, int):
        raise ValueError("resolution must be a whole number")

    if (resolution < 2):
        raise ValueError("resolution must be greater than or equal to 2")

    # Average spectrum over time: one strength value per frequency bin
    spectrum = np.abs(librosa.stft(input_audio.audio_array, n_fft=n_fft)).mean(axis=1)
    bin_freqs = librosa.fft_frequencies(sr=sampling_rate, n_fft=n_fft)

    # Keep only the bins between fmin and fmax
    in_band = (bin_freqs >= fmin) & (bin_freqs <= fmax)
    band_freqs = bin_freqs[in_band]
    band_spectrum = spectrum[in_band]

    if band_spectrum.size == 0:
        raise ValueError("No frequency bins fall between fmin and fmax; widen the range or increase n_fft")

    if band_spectrum.max() == 0:
        raise ValueError("The clip has no energy between fmin and fmax")

    # Find the peaks, then keep the n_peaks strongest, loudest first
    peak_prominence = 0.03      # a peak must stand out by 3% of the band's strongest value
    peak_idx, _ = find_peaks(band_spectrum, prominence=band_spectrum.max() * peak_prominence)

    if peak_idx.size == 0:
        raise ValueError("No spectral peaks found between fmin and fmax")

    peak_freqs = band_freqs[peak_idx]
    peak_amps = band_spectrum[peak_idx]

    order = np.argsort(peak_amps)[::-1][:n_peaks]
    frequencies = peak_freqs[order]
    weights = peak_amps[order] / peak_amps[order[0]]

    # Build the field: one standing-wave pattern per peak, added together
    xs = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(xs, xs)
    field = np.zeros_like(X)

    for frequency, weight in zip(frequencies, weights):
        n = 2 + int(frequency) % 7              # pattern complexity along x
        m = 3 + (int(frequency) // 7) % 7       # pattern complexity along y
        if n == m:
            m += 1                              # n == m would cancel to zero
        field += weight * (np.cos(n * np.pi * X) * np.cos(m * np.pi * Y)
                           - np.cos(m * np.pi * X) * np.cos(n * np.pi * Y))

    field = field / np.max(np.abs(field))

    return ChladniData(field=field, frequencies=frequencies, weights=weights)
