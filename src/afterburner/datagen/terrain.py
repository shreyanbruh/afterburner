import numpy as np
import math
from scipy.ndimage import gaussian_filter

def terrain_spectrogram(S: np.ndarray,
                        max_cols: int | None = None,
                        sigma: tuple[float, float] = (1.6, 0.8)):
    """Turn a spectrogram into a smoothed height map for a 3D terrain plot.

    Args:
        S (np.ndarray): A 2-D spectrogram of shape ``(n_bands, n_frames)``, for
            example the dB array returned by ``mel_spectrogram``. Rows are
            frequency bands and columns are time frames. It must not be flat
            (all values equal). Only its relative values matter, so the dB
            range it was computed with does not change the result.
        max_cols (int | None): The most time columns the result may have.
            ``None`` (default) means no averaging, so every frame is kept.
            Otherwise it must be at least 1 and no greater than ``n_frames``.
            Frames are averaged in groups of ``ceil(n_frames / max_cols)``, and
            any frames at the end that do not fill a whole group are dropped.
        sigma (tuple[float, float]): Standard deviation of the Gaussian blur,
            as ``(band_sigma, time_sigma)``. Index 0 blurs across frequency
            bands and index 1 blurs across time columns. Both are measured in
            grid cells (not Hz or seconds), so the same value blurs more real
            time when more frames are averaged per column. Each must be at
            least 0; 0 turns the blur off along that axis. Defaults to
            ``(1.6, 0.8)``.

    Returns:
        np.ndarray: A 2-D height map of shape ``(n_bands, n_cols)``, where
        ``n_cols`` is ``n_frames // frames_per_column`` and is at most
        ``max_cols``. Values lie between 0 and 1, but because of the averaging
        and blurring the minimum and maximum usually do not reach exactly 0 and
        1.

    Raises:
        ValueError: If ``S`` is not a 2-D array.
        ValueError: If ``S`` is flat (its maximum equals its minimum).
        ValueError: If ``max_cols`` is less than 1.
        ValueError: If ``max_cols`` is greater than the number of frames in
            ``S``.
        ValueError: If ``sigma`` is a single int or float instead of a pair.
        ValueError: If ``sigma`` does not contain exactly two values.
        ValueError: If either value in ``sigma`` is negative.
    """

    if S.ndim != 2:
        raise ValueError(f"S must be a 2-D array, got {S.ndim} dimension(s)")

    value_range = S.max() - S.min()

    if value_range == 0:
        raise ValueError("Spectrogram is flat; nothing to plot")

    if(max_cols is None):
        max_cols = S.shape[1]

    if (max_cols < 1):
        raise ValueError("Max columns must be greater than or equal to 1")

    if (max_cols > S.shape[1]):
        raise ValueError(f"Max columns cannot exceed {S.shape[1]} columns")

    if isinstance(sigma, (int, float)):
        raise ValueError("sigma must be a tuple of two values (band_sigma, time_sigma), not a single number")

    if len(sigma) != 2:
        raise ValueError("sigma must contain exactly two values: (band_sigma, time_sigma)")

    if sigma[0] < 0 or sigma[1] < 0:
        raise ValueError("Sigma values cannot be negative")


    Z = (S - S.min()) / value_range # Normalise to a height map


    n_frames = Z.shape[1]
    frames_per_column = max(1, math.ceil(n_frames / max_cols))

    cols = (n_frames // frames_per_column) * frames_per_column
    Z = Z[:, :cols].reshape(Z.shape[0], -1, frames_per_column).mean(2)

    Z = gaussian_filter(Z, sigma=sigma)   # smooth: sigma = (band_sigma, time_sigma)

    return Z