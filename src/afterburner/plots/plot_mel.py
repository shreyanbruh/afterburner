import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Colormap, is_color_like
from matplotlib.figure import Figure
from afterburner.audio import Audio


def plot_mel(S: np.ndarray,
             input_audio: Audio,
             fmin: float = 0,
             fmax: float | None = None,
             *,
             cmap: str | Colormap = "binary",
             figsize: tuple[float, float] = (10, 5),
             grid: bool = False,
             grid_color: str = "gray",
             grid_linestyle: str = "--",
             grid_linewidth: float = 0.6,
             grid_alpha: float = 0.6,
             image_only: bool = False,
             save_path: str | os.PathLike | None = None,
             dpi: float = 200,
             show: bool = True):
    """Plot a mel spectrogram, and optionally save it to a file.

    Args:
        S (np.ndarray): The mel spectrogram in dB, a 2-D array of shape
            ``(n_mels, n_frames)``, for example the array returned by ``mel``.
            Rows are mel bands from lowest to highest frequency and columns are
            time frames.
        input_audio (Audio): The clip ``S`` was computed from. Its length and
            sampling rate set the time axis and the frequency range.
        fmin (float): The ``fmin`` that ``S`` was computed with, so the
            frequency axis is labelled correctly. Defaults to 0. Must be at
            least 0 and less than ``fmax``.
        fmax (float | None): The ``fmax`` that ``S`` was computed with.
            ``None`` (default) means the Nyquist frequency,
            ``input_audio.sampling_rate / 2``. Must not exceed that value.
        cmap (str | Colormap): The colormap. Any matplotlib colormap name
            (``"inferno"``, ``"viridis_r"``...) or a matplotlib ``Colormap``
            object. Defaults to ``"binary"``.
        figsize (tuple[float, float]): The width and height of the figure in
            inches. Defaults to ``(10, 5)``. Both must be greater than 0.
        grid (bool): Whether to draw grid lines over the plot. Defaults to
            ``False``.
        grid_color (str): The colour of the grid lines, any matplotlib colour.
            Defaults to ``"gray"``. Only used when ``grid`` is ``True``.
        grid_linestyle (str): The style of the grid lines: ``"-"``, ``"--"``,
            ``"-."``, ``":"``, ``"solid"``, ``"dashed"``, ``"dashdot"`` or
            ``"dotted"``. Defaults to ``"--"``. Only used when ``grid`` is
            ``True``.
        grid_linewidth (float): The width of the grid lines in points. Defaults
            to 0.6. Must be greater than 0. Only used when ``grid`` is
            ``True``.
        grid_alpha (float): How opaque the grid lines are, from 0 (invisible)
            to 1 (solid). Defaults to 0.6. Only used when ``grid`` is
            ``True``.
        image_only (bool): If ``True``, the saved file contains only the
            spectrogram picture: no axes, labels, colour bar, grid or margins.
            Requires ``save_path``. Defaults to ``False``, which saves the
            whole figure. The figure shown on screen is not affected.
        save_path (str | os.PathLike | None): Where to save the picture. The
            file type comes from the extension (``.png``, ``.jpg``, ``.pdf``,
            ``.svg``...). The folder must already exist. Defaults to ``None``,
            which saves nothing.
        dpi (float): The resolution of the saved file in dots per inch. The
            saved size in pixels is ``figsize`` times ``dpi``. Defaults to
            200. Must be greater than 0.
        show (bool): Whether to open the plot in a window. Defaults to
            ``True``. If ``False`` the figure is closed after any saving, so
            nothing is displayed, but it is still returned.

    Returns:
        tuple[Figure, Axes]: The matplotlib figure and axes of the full plot,
        so you can keep customising it (for example ``ax.set_title(...)``).

    Raises:
        ValueError: If ``S`` is not a 2-D array.
        ValueError: If ``fmin`` is not a number.
        ValueError: If ``fmax`` is not a number or ``None``.
        ValueError: If ``0 <= fmin < fmax <= sampling_rate / 2`` does not hold,
            after ``fmax`` has been resolved.
        ValueError: If ``cmap`` is a name that is not a matplotlib colormap.
        ValueError: If ``cmap`` is neither a name nor a ``Colormap``.
        ValueError: If ``figsize`` is not a pair of numbers greater than 0.
        ValueError: If ``grid`` is not ``True`` or ``False``.
        ValueError: If ``grid_color`` is not a valid matplotlib colour.
        ValueError: If ``grid_linestyle`` is not one of the styles listed above.
        ValueError: If ``grid_linewidth`` is not a number greater than 0.
        ValueError: If ``grid_alpha`` is not a number from 0 to 1.
        ValueError: If ``image_only`` is not ``True`` or ``False``.
        ValueError: If ``image_only`` is ``True`` and ``save_path`` is ``None``.
        ValueError: If ``save_path`` is not a string or path.
        ValueError: If ``save_path`` has no file extension.
        ValueError: If the extension of ``save_path`` is not a supported image
            type.
        ValueError: If the folder in ``save_path`` does not exist.
        ValueError: If ``dpi`` is not a number greater than 0.
        ValueError: If ``show`` is not ``True`` or ``False``.

    """

    if S.ndim != 2:
        raise ValueError(f"S must be a 2-D array, got {S.ndim} dimension(s)")

    sampling_rate = input_audio.sampling_rate

    if isinstance(fmin, bool) or not isinstance(fmin, (int, float)):
        raise ValueError("fmin must be a number")

    if fmax is not None:
        if isinstance(fmax, bool) or not isinstance(fmax, (int, float)):
            raise ValueError("fmax must be a number or None")

    if fmax is None:
        fmax = sampling_rate / 2

    if not(0 <= fmin < fmax <= sampling_rate / 2):
        raise ValueError("Require 0 <= fmin < fmax <= sampling_rate/2")

    if isinstance(cmap, str):
        if cmap not in colormaps:
            raise ValueError(f"cmap '{cmap}' is not a matplotlib colormap")
    elif not isinstance(cmap, Colormap):
        raise ValueError("cmap must be a colormap name or a matplotlib Colormap")

    if (not isinstance(figsize, (tuple, list)) or len(figsize) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0 for v in figsize)):
        raise ValueError("figsize must be a pair of numbers greater than 0, like (10, 5)")

    if not isinstance(grid, bool):
        raise ValueError("grid must be True or False")

    if not is_color_like(grid_color):
        raise ValueError(f"grid_color '{grid_color}' is not a valid matplotlib colour")

    if grid_linestyle not in ("-", "--", "-.", ":", "solid", "dashed", "dashdot", "dotted"):
        raise ValueError("grid_linestyle must be one of '-', '--', '-.', ':', 'solid', 'dashed', 'dashdot', 'dotted'")

    if isinstance(grid_linewidth, bool) or not isinstance(grid_linewidth, (int, float)) or grid_linewidth <= 0:
        raise ValueError("grid_linewidth must be a number greater than 0")

    if isinstance(grid_alpha, bool) or not isinstance(grid_alpha, (int, float)) or not (0 <= grid_alpha <= 1):
        raise ValueError("grid_alpha must be a number from 0 to 1")

    if not isinstance(image_only, bool):
        raise ValueError("image_only must be True or False")

    if image_only and save_path is None:
        raise ValueError("image_only needs a save_path to save the image to")

    if save_path is not None:
        if not isinstance(save_path, (str, os.PathLike)):
            raise ValueError("save_path must be a string or path")

        save_path = os.fspath(save_path)
        extension = os.path.splitext(save_path)[1].lstrip(".").lower()
        supported = Figure().canvas.get_supported_filetypes()

        if extension == "":
            raise ValueError("save_path needs a file extension, like .png")

        if extension not in supported:
            raise ValueError(f"save_path extension '.{extension}' is not supported; use one of {sorted(supported)}")

        folder = os.path.dirname(save_path)

        if folder and not os.path.isdir(folder):
            raise ValueError(f"The folder '{folder}' does not exist")

    if isinstance(dpi, bool) or not isinstance(dpi, (int, float)) or dpi <= 0:
        raise ValueError("dpi must be a number greater than 0")

    if not isinstance(show, bool):
        raise ValueError("show must be True or False")

    mel_low, mel_high = librosa.hz_to_mel([fmin, fmax])
    extent = [0, input_audio.duration, mel_low, mel_high]

    candidates = np.array([0, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
    tick_hz = candidates[(candidates >= fmin) & (candidates <= fmax)]

    if tick_hz.size < 2:
        tick_hz = np.round(np.linspace(fmin, fmax, 5))

    # The full figure: axes, labels, colour bar and optional grid
    fig, ax = plt.subplots(figsize=figsize)
    img = ax.imshow(S, origin="lower", aspect="auto", cmap=cmap, extent=extent)
    ax.set_yticks(librosa.hz_to_mel(tick_hz))
    ax.set_yticklabels([f"{hz:g}" for hz in tick_hz])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz, mel scale)")
    ax.set_title("Mel spectrogram")
    fig.colorbar(img, ax=ax, label="dB")

    if grid:
        ax.grid(True,
                color=grid_color,
                linestyle=grid_linestyle,
                linewidth=grid_linewidth,
                alpha=grid_alpha)

    if save_path is not None:
        if image_only:
            bare = plt.figure(figsize=figsize)
            bare_ax = bare.add_axes([0, 0, 1, 1])
            bare_ax.imshow(S, origin="lower", aspect="auto", cmap=cmap, extent=extent)
            bare_ax.set_axis_off()
            bare.savefig(save_path, dpi=dpi)
            plt.close(bare)
        else:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax
