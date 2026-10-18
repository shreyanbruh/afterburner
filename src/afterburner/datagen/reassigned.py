import numpy as np
import librosa
from scipy.ndimage import gaussian_filter
from typing import NamedTuple
from afterburner.audio import Audio


class ReassignedData(NamedTuple):
    """The reassigned spectrogram, with the bin edges to draw it on.

    It is a tuple, so it can be unpacked (``level_db, frequencies, times =
    data``), and its fields can also be read by name.

    Attributes:
        level_db (np.ndarray): The strength at each point of the grid in dB, a
            2-D float array of shape ``(n_freq_bins, n_time_bins)``. Rows are
            frequency bins from lowest to highest and columns are time bins.
            The loudest value is 0 and everything else is negative, no lower
            than ``-top_db`` when ``top_db`` is set.
        frequencies (np.ndarray): The edges of the frequency bins in Hz, a 1-D
            float array of shape ``(n_freq_bins + 1,)``. The edges are
            log-spaced, from ``fmin`` to ``fmax``.
        times (np.ndarray): The edges of the time bins in seconds, a 1-D float
            array of shape ``(n_time_bins + 1,)``, evenly spaced from 0 to the
            length of the clip.
    """
    level_db: np.ndarray
    frequencies: np.ndarray
    times: np.ndarray


def reassigned_spectrogram(input_audio: Audio,
                           n_fft: int = 2048,
                           hop_length: int | None = None,
                           fmin: float = 40,
                           fmax: float | None = None,
                           n_freq_bins: int = 300,
                           n_time_bins: int | None = None,
                           sigma: float = 0.8,
                           top_db: float | None = 80.0):
    """Compute a reassigned spectrogram: a spectrogram with its energy moved to where it really is.

    Args:
        input_audio (Audio): The clip to analyse. Its ``audio_array`` and
            ``sampling_rate`` are used as the signal and its sample rate.
        n_fft (int): The size of the analysis window in samples. Defaults to
            2048. Must be a whole number of at least 2 and no greater than the
            number of samples in the clip.
        hop_length (int | None): Number of samples between the starts of
            consecutive frames. Defaults to ``None``, which means ``n_fft // 8``
            (at least 1). Must be a whole number of at least 1 and no greater
            than ``n_fft``. Smaller values give more frames, so more points
            are moved and the lines look denser.
        fmin (float): The lowest frequency in Hz on the grid. Defaults to 40.
            Must be greater than 0 (the grid is logarithmic) and less than
            ``fmax``.
        fmax (float | None): The highest frequency in Hz on the grid. Defaults
            to ``None``, which means the Nyquist frequency,
            ``sampling_rate / 2``. Must not exceed that value.
        n_freq_bins (int): The number of log-spaced frequency bins. Defaults to
            300. Must be a whole number of at least 1.
        n_time_bins (int | None): The number of evenly spaced time bins.
            Defaults to ``None``, which means one bin per analysis frame. Must
            be a whole number of at least 1. Energy is placed using its
            reassigned time, which is more precise than the frame spacing, so
            more bins than frames is allowed, but they get sparse.
        sigma (float): Standard deviation of the Gaussian blur applied to the
            grid, in grid cells (not Hz or seconds). Defaults to 0.8, which
            joins up dotted lines. Must be at least 0; 0 turns the blur off.
        top_db (float | None): How far below the loudest value, in dB, the
            levels are allowed to go. Anything lower is raised to that floor.
            Defaults to 80. Must be at least 0. ``None`` turns the floor off,
            in which case empty cells sit at a very low value.

    Returns:
        ReassignedData: A named tuple ``(level_db, frequencies, times)``.
        ``level_db`` has shape ``(n_freq_bins, n_time_bins)``, ``frequencies``
        has shape ``(n_freq_bins + 1,)`` and ``times`` has shape
        ``(n_time_bins + 1,)``. The last two are bin edges, which is what
        ``plt.pcolormesh`` wants.

    Raises:
        ValueError: If ``n_fft`` is not a whole number.
        ValueError: If ``n_fft`` is less than 2.
        ValueError: If ``n_fft`` is greater than the number of samples in the
            clip.
        ValueError: If ``hop_length`` is not a whole number.
        ValueError: If ``hop_length`` is less than 1.
        ValueError: If ``hop_length`` is greater than ``n_fft``.
        ValueError: If ``fmin`` is not a number.
        ValueError: If ``fmin`` is not greater than 0.
        ValueError: If ``fmax`` is not a number or ``None``.
        ValueError: If ``fmin`` is not less than ``fmax``.
        ValueError: If ``fmax`` is greater than half the sampling rate.
        ValueError: If ``n_freq_bins`` is not a whole number.
        ValueError: If ``n_freq_bins`` is less than 1.
        ValueError: If ``n_time_bins`` is not a whole number or ``None``.
        ValueError: If ``n_time_bins`` is less than 1.
        ValueError: If ``sigma`` is not a number.
        ValueError: If ``sigma`` is negative.
        ValueError: If ``top_db`` is not a number or ``None``.
        ValueError: If ``top_db`` is negative.
        ValueError: If no usable energy is found between ``fmin`` and ``fmax``
            (the clip is silent or very quiet).
    """

    sampling_rate = input_audio.sampling_rate
    nyquist = sampling_rate / 2

    if isinstance(n_fft, bool) or not isinstance(n_fft, int):
        raise ValueError("n_fft must be a whole number")

    if (n_fft < 2):
        raise ValueError("n_fft must be greater than or equal to 2")

    if (n_fft > input_audio.n_samples):
        raise ValueError(f"n_fft cannot exceed {input_audio.n_samples} samples (the length of the clip)")

    if hop_length is None:
        hop_length = max(1, n_fft // 8)

    if isinstance(hop_length, bool) or not isinstance(hop_length, int):
        raise ValueError("hop_length must be a whole number")

    if (hop_length < 1):
        raise ValueError("hop_length must be greater than or equal to 1")

    if (hop_length > n_fft):
        raise ValueError(f"hop_length cannot exceed n_fft ({n_fft})")

    if isinstance(fmin, bool) or not isinstance(fmin, (int, float)):
        raise ValueError("fmin must be a number")

    if (fmin <= 0):
        raise ValueError("fmin must be greater than 0 (the frequency grid is logarithmic)")

    if fmax is not None:
        if isinstance(fmax, bool) or not isinstance(fmax, (int, float)):
            raise ValueError("fmax must be a number or None")
    else:
        fmax = nyquist

    if (fmin >= fmax):
        raise ValueError("fmin must be less than fmax")

    if (fmax > nyquist):
        raise ValueError(f"fmax cannot exceed {nyquist} Hz (half the sampling rate)")

    if isinstance(n_freq_bins, bool) or not isinstance(n_freq_bins, int):
        raise ValueError("n_freq_bins must be a whole number")

    if (n_freq_bins < 1):
        raise ValueError("n_freq_bins must be greater than or equal to 1")

    if n_time_bins is not None:
        if isinstance(n_time_bins, bool) or not isinstance(n_time_bins, int):
            raise ValueError("n_time_bins must be a whole number or None")

        if (n_time_bins < 1):
            raise ValueError("n_time_bins must be greater than or equal to 1")

    if isinstance(sigma, bool) or not isinstance(sigma, (int, float)):
        raise ValueError("sigma must be a number")

    if (sigma < 0):
        raise ValueError("sigma cannot be negative")

    if top_db is not None:
        if isinstance(top_db, bool) or not isinstance(top_db, (int, float)):
            raise ValueError("top_db must be a number or None")

        if (top_db < 0):
            raise ValueError("top_db cannot be negative")

    cell_freqs, cell_times, cell_mags = librosa.reassigned_spectrogram(y=input_audio.audio_array,
                                                                       sr=sampling_rate,
                                                                       n_fft=n_fft,
                                                                       hop_length=hop_length)

    usable = np.isfinite(cell_freqs) & np.isfinite(cell_times) & (cell_mags > 0)

    if n_time_bins is None:
        n_time_bins = cell_mags.shape[1]

    duration = input_audio.n_samples / sampling_rate
    time_edges = np.linspace(0, duration, n_time_bins + 1)
    frequency_edges = np.geomspace(fmin, fmax, n_freq_bins + 1)

    grid, _, _ = np.histogram2d(cell_times[usable],
                                cell_freqs[usable],
                                bins=(time_edges, frequency_edges),
                                weights=cell_mags[usable])
    grid = grid.T                   # rows = frequency, columns = time

    if grid.max() == 0:
        raise ValueError("No usable energy found between fmin and fmax; the clip may be silent or very quiet")

    grid = gaussian_filter(grid, sigma=sigma)

    level_db = librosa.amplitude_to_db(grid, ref=np.max, amin=1e-10, top_db=top_db)

    return ReassignedData(level_db=level_db,
                          frequencies=frequency_edges,
                          times=time_edges)
