import math
import numpy as np
from afterburner.audio import Audio
from typing import NamedTuple


class RadialData(NamedTuple):
    """The mesh needed to draw a radial spectrogram with ``pcolormesh``.

    Attributes:
        theta (np.ndarray): Angle edges in radians, a 1-D float array of shape
            ``(n_bands + 1,)`` running from 0 to 2π. Each consecutive pair of
            edges bounds the wedge for one frequency band, from the lowest band
            to the highest.
        r (np.ndarray): Radius edges, a 1-D float array of shape
            ``(n_cols + 1,)`` running outward from 0. Each consecutive pair
            bounds the ring for one time column, earliest column innermost. The
            units are seconds.
        values (np.ndarray): The values to colour the cells with, shape
            ``(n_cols, n_bands)``. This is the input spectrogram transposed,
            with groups of neighbouring time frames averaged if ``max_cols``
            was given. ``n_cols`` is the number of time columns (rings); it
            equals the number of frames when no averaging is done.
    """
    theta: np.ndarray
    r: np.ndarray
    values: np.ndarray


def radial_spectrogram(S: np.ndarray,
                       input_audio: Audio,
                       max_cols: int | None = None):
    """Prepare a spectrogram for drawing as a radial (polar) plot.

    Args:
        S (np.ndarray): A 2-D spectrogram of shape ``(n_bands, n_frames)``, for
            example the dB array returned by ``mel_spectrogram``. Rows are
            frequency bands and columns are time frames.
        input_audio (Audio): The clip that ``S`` was computed from. Its
            ``duration`` attribute (length in seconds) sets the radius edges, so
            the radius is measured in seconds. This is not checked against
            ``S``: pass the same clip that was given to ``mel_spectrogram``.
        max_cols (int | None): The most time columns (rings) the result may
            have. ``None`` (default) means no averaging, so every frame is
            kept. Otherwise it must be at least 1 and no greater than
            ``n_frames``. Frames are averaged in groups of
            ``ceil(n_frames / max_cols)``, and any frames at the end that do not
            fill a whole group are dropped, so the radius then ends slightly
            before the end of the clip. Fewer rings make the plot faster to
            draw and avoid fine seams between rings that are thinner than one
            pixel.

    Returns:
        RadialData: A named tuple ``(theta, r, values)`` with shapes
        ``(n_bands + 1,)``, ``(n_cols + 1,)`` and ``(n_cols, n_bands)``, where
        ``n_cols`` is the number of time columns after averaging (equal to
        ``n_frames`` when ``max_cols`` is ``None``) and is at most ``max_cols``.

    Raises:
        ValueError: If ``S`` is not a 2-D array.
        ValueError: If ``max_cols`` is less than 1.
        ValueError: If ``max_cols`` is greater than the number of frames in
            ``S``.
    """

    if S.ndim != 2:
        raise ValueError(f"S must be a 2-D array, got {S.ndim} dimension(s)")

    duration = input_audio.duration

    n_bands, n_frames = S.shape

    if(max_cols is None):
        max_cols = n_frames

    if (max_cols < 1):
        raise ValueError("Max columns must be greater than or equal to 1")

    if (max_cols > n_frames):
        raise ValueError(f"Max columns cannot exceed {n_frames} columns")

    frames_per_column = max(1, math.ceil(n_frames / max_cols))

    kept_frames = (n_frames // frames_per_column) * frames_per_column
    values = S[:, :kept_frames].reshape(n_bands, -1, frames_per_column).mean(2)

    n_cols = values.shape[1]

    r_end = duration * kept_frames / n_frames

    theta = np.linspace(0, 2 * np.pi, n_bands + 1)
    r = np.linspace(0, r_end, n_cols + 1)

    return RadialData(theta=theta, r=r, values=values.T)