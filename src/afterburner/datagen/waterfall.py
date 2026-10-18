import numpy as np
from scipy.ndimage import gaussian_filter
from typing import NamedTuple


class WaterfallData(NamedTuple):
    """The heights and source frames for drawing a waterfall plot.

    Attributes:
        heights (np.ndarray): The height map, a 2-D float array of shape
            ``(n_bands, n_lines)`` with values between 0 and 1. Each column is
            one line: the spectrum at one moment, from the lowest frequency band
            (row 0) to the highest. Columns are in time order, earliest first.
        frame_indices (np.ndarray): The time frames that were picked, an
            integer array of shape ``(n_lines,)`` in increasing order. Entry
            ``j`` is the column of the input spectrogram that line ``j`` was
            taken from. To convert to seconds, use
            ``frame_indices * hop_length / sampling_rate`` with the hop length
            and sample rate the spectrogram was computed with.
    """
    heights: np.ndarray
    frame_indices: np.ndarray

def waterfall_spectrogram(S: np.ndarray,
                          n_lines: int = 70,
                          gamma: float = 1.3,
                          sigma: float = 1.2):
    
    """Turn a spectrogram into the line heights for a waterfall plot.

    Args:
        S (np.ndarray): A 2-D spectrogram of shape ``(n_bands, n_frames)``, for
            example the dB array returned by ``mel_spectrogram``. Rows are
            frequency bands and columns are time frames. It must not be flat
            (all values equal). The normalisation uses the whole of ``S``, so
            the height of a given frame does not change with ``n_lines``.
        n_lines (int): The number of lines (time frames) to pick. Defaults to
            70. Must be at least 1 and no greater than ``n_frames``. The frames
            are evenly spaced and always include the first and last frame;
            nothing is averaged, so each line is a real spectrum from one
            moment. Roughly 30 to 150 lines gives the best look; large values
            are slow to draw and the lines blur together.
        gamma (float): Exponent applied to the 0 to 1 heights. Above 1 pushes
            quiet values down so only strong peaks stand out; below 1 lifts
            quiet values. Defaults to 1.3. Must be greater than 0.
        sigma (float): Standard deviation of the Gaussian blur applied across
            frequency bands to each line, in bands (not Hz). It is a single
            number, not a tuple, because lines are not blurred across time.
            Defaults to 1.2. Must be at least 0; 0 turns the blur off.

    Returns:
        WaterfallData: A named tuple ``(heights, frame_indices)``. ``heights``
        has shape ``(n_bands, n_lines)`` with values between 0 and 1, and
        ``frame_indices`` has shape ``(n_lines,)``.

    Raises:
        ValueError: If ``S`` is not a 2-D array.
        ValueError: If ``S`` is flat (its maximum equals its minimum).
        ValueError: If ``n_lines`` is less than 1.
        ValueError: If ``n_lines`` is greater than the number of frames in
            ``S``.
        ValueError: If ``gamma`` is not greater than 0.
        ValueError: If ``sigma`` is not a single int or float.
        ValueError: If ``sigma`` is negative.
    """

    if S.ndim != 2:
        raise ValueError(f"S must be a 2-D array, got {S.ndim} dimension(s)")

    value_range = S.max() - S.min()

    if value_range == 0:
        raise ValueError("Spectrogram is flat; nothing to plot")

    n_frames = S.shape[1]

    if (n_lines < 1):
        raise ValueError("Number of lines must be greater than or equal to 1")

    if (n_lines > n_frames):
        raise ValueError(f"Number of lines cannot exceed {n_frames} frames")

    if (gamma <= 0):
        raise ValueError("gamma must be greater than 0")

    if not isinstance(sigma, (int, float)):
        raise ValueError("sigma must be a single number, not a tuple")

    if (sigma < 0):
        raise ValueError("sigma cannot be negative")

    Z = (S - S.min()) / value_range     # Normalise the whole spectrogram first

    frame_indices = np.linspace(0, n_frames - 1, n_lines).astype(int)
    Z = Z[:, frame_indices]             # Pick the frames: one column per line

    Z = Z ** gamma

    Z = gaussian_filter(Z, sigma=(sigma, 0))   # blur across bands only

    return WaterfallData(heights=Z, frame_indices=frame_indices)
