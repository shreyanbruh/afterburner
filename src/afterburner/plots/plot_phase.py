import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, Normalize, is_color_like
from matplotlib.figure import Figure
from afterburner.datagen.phase import PhaseData


def plot_phase(data: PhaseData,
               *,
               cmap: str | Colormap = "binary",
               figsize: tuple[float, float] = (10, 5),
               grid: bool = False,
               grid_color: str = "gray",
               grid_linestyle: str = "--",
               grid_linewidth: float = 0.6,
               grid_alpha: float = 0.6,
               fmax: float | None = None,
               dynamic_range: float = 60.0,
               image_only: bool = False,
               save_path: str | os.PathLike | None = None,
               dpi: float = 200,
               show: bool = True):
    """Plot a phase-coloured spectrogram, and optionally save it to a file.

    Args:
        data (PhaseData): The result of ``phase_spectrogram``. Its
            ``phase_deviation`` sets the colour of each point and its
            ``level_db`` sets how bright it is.
        cmap (str | Colormap): The colormap, used for the phase. Any
            matplotlib colormap name (``"inferno"``, ``"viridis_r"``...) or a
            matplotlib ``Colormap`` object. Defaults to ``"binary"``. A cyclic
            colormap such as ``"twilight"`` or ``"hsv"`` shows the phase best.
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
        fmax (float | None): The highest frequency in Hz to show. Defaults to
            ``None``, which shows every frequency up to the top of the data
            (half the sampling rate). Must be greater than 0 and no greater than
            the highest frequency in ``data.frequencies``.
        dynamic_range (float): How far below the loudest point, in dB, a point
            can be and still be visible. Louder points are brighter, and
            anything this much quieter or more is black. Defaults to 60. Must be
            greater than 0.
        image_only (bool): If ``True``, the saved file contains only the
            picture: no axes, labels, colour bar, grid or margins. Requires
            ``save_path``. Defaults to ``False``, which saves the whole figure.
            The figure shown on screen is not affected.
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
        ValueError: If ``data`` is not a ``PhaseData``.
        ValueError: If ``data.level_db`` is not a 2-D array.
        ValueError: If ``data.phase_deviation`` does not have the same shape as
            ``data.level_db``.
        ValueError: If ``data.level_db`` has fewer than 2 rows or 2 columns.
        ValueError: If ``data.frequencies`` does not have one entry per row.
        ValueError: If ``data.times`` does not have one entry per column.
        ValueError: If ``cmap`` is a name that is not a matplotlib colormap.
        ValueError: If ``cmap`` is neither a name nor a ``Colormap``.
        ValueError: If ``figsize`` is not a pair of numbers greater than 0.
        ValueError: If ``grid`` is not ``True`` or ``False``.
        ValueError: If ``grid_color`` is not a valid matplotlib colour.
        ValueError: If ``grid_linestyle`` is not one of the styles listed above.
        ValueError: If ``grid_linewidth`` is not a number greater than 0.
        ValueError: If ``grid_alpha`` is not a number from 0 to 1.
        ValueError: If ``fmax`` is not a number or ``None``.
        ValueError: If ``fmax`` is not greater than 0.
        ValueError: If ``fmax`` is greater than the highest frequency in
            ``data.frequencies``.
        ValueError: If ``dynamic_range`` is not a number greater than 0.
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

    if not isinstance(data, PhaseData):
        raise ValueError("data must be a PhaseData, the result of phase_spectrogram()")

    level_db = data.level_db
    frequencies = data.frequencies
    times = data.times

    if level_db.ndim != 2:
        raise ValueError(f"data.level_db must be a 2-D array, got {level_db.ndim} dimension(s)")

    if data.phase_deviation.shape != level_db.shape:
        raise ValueError(f"data.phase_deviation must have the same shape as data.level_db, got {data.phase_deviation.shape} and {level_db.shape}")

    if level_db.shape[0] < 2 or level_db.shape[1] < 2:
        raise ValueError("data.level_db must have at least 2 rows and 2 columns to plot")

    if len(frequencies) != level_db.shape[0]:
        raise ValueError(f"data.frequencies must have one entry per row, got {len(frequencies)} frequencies and {level_db.shape[0]} row(s)")

    if len(times) != level_db.shape[1]:
        raise ValueError(f"data.times must have one entry per column, got {len(times)} time(s) and {level_db.shape[1]} column(s)")

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

    if fmax is not None:
        if isinstance(fmax, bool) or not isinstance(fmax, (int, float)):
            raise ValueError("fmax must be a number or None")

        if fmax <= 0:
            raise ValueError("fmax must be greater than 0")

        if fmax > frequencies[-1]:
            raise ValueError(f"fmax cannot exceed {frequencies[-1]} Hz (the top of the data)")

    if isinstance(dynamic_range, bool) or not isinstance(dynamic_range, (int, float)) or dynamic_range <= 0:
        raise ValueError("dynamic_range must be a number greater than 0")

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

    colour_map = colormaps[cmap] if isinstance(cmap, str) else cmap

    limit = np.pi * times[0] * frequencies[1]
    position = np.clip((data.phase_deviation / limit + 1) / 2, 0, 1)
    brightness = np.clip(1 + level_db / dynamic_range, 0, 1).astype(np.float32)
    picture = (colour_map(position, bytes=True)[..., :3] * brightness[..., None]).astype(np.uint8)

    extent = [times[0], times[-1], 0, frequencies[-1]]
    top = frequencies[-1] if fmax is None else fmax

    # The full figure: axes, labels, colour bar and optional grid
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(picture, origin="lower", aspect="auto", extent=extent)
    ax.set_ylim(0, top)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Phase-coloured spectrogram")
    fig.colorbar(ScalarMappable(norm=Normalize(-limit, limit), cmap=colour_map),
                 ax=ax, label="Phase deviation (rad)")

    if grid:
        ax.grid(True,
                color=grid_color,
                linestyle=grid_linestyle,
                linewidth=grid_linewidth,
                alpha=grid_alpha)

    if save_path is not None:
        if image_only:
            # A second figure with a single axes that fills the whole canvas
            bare = plt.figure(figsize=figsize)
            bare_ax = bare.add_axes([0, 0, 1, 1])
            bare_ax.imshow(picture, origin="lower", aspect="auto", extent=extent)
            bare_ax.set_ylim(0, top)
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
