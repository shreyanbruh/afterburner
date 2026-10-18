import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Colormap, is_color_like, to_rgba
from matplotlib.figure import Figure
from afterburner.audio import Audio


def plot_terrain(Z: np.ndarray,
                 input_audio: Audio,
                 *,
                 cmap: str | Colormap = "binary",
                 figsize: tuple[float, float] = (9, 6),
                 grid: bool = False,
                 grid_color: str = "gray",
                 grid_linestyle: str = "--",
                 grid_linewidth: float = 0.6,
                 grid_alpha: float = 0.6,
                 elevation: float = 35,
                 azimuth: float = -60,
                 image_only: bool = False,
                 save_path: str | os.PathLike | None = None,
                 dpi: float = 200,
                 show: bool = True):
    """Plot a spectrogram as a 3D terrain, and optionally save it to a file.

    Args:
        Z (np.ndarray): The height map, a 2-D array of shape
            ``(n_bands, n_cols)``, for example the array returned by
            ``terrain_spectrogram``. Rows are frequency bands from lowest to
            highest and columns are time.
        input_audio (Audio): The clip ``Z`` was computed from. Its length sets
            the time axis.
        cmap (str | Colormap): The colormap, used to colour the surface by
            height. Any matplotlib colormap name (``"inferno"``,
            ``"viridis_r"``...) or a matplotlib ``Colormap`` object. Defaults to
            ``"binary"``.
        figsize (tuple[float, float]): The width and height of the figure in
            inches. Defaults to ``(9, 6)``. Both must be greater than 0.
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
        elevation (float): How high the camera is above the terrain, in degrees.
            Defaults to 35. Must be from -90 to 90.
        azimuth (float): How far the camera is turned around the terrain, in
            degrees. Defaults to -60. Any number is allowed.
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
        ValueError: If ``Z`` is not a NumPy array.
        ValueError: If ``Z`` is not a 2-D array.
        ValueError: If ``Z`` has fewer than 2 rows or 2 columns.
        ValueError: If ``cmap`` is a name that is not a matplotlib colormap.
        ValueError: If ``cmap`` is neither a name nor a ``Colormap``.
        ValueError: If ``figsize`` is not a pair of numbers greater than 0.
        ValueError: If ``grid`` is not ``True`` or ``False``.
        ValueError: If ``grid_color`` is not a valid matplotlib colour.
        ValueError: If ``grid_linestyle`` is not one of the styles listed above.
        ValueError: If ``grid_linewidth`` is not a number greater than 0.
        ValueError: If ``grid_alpha`` is not a number from 0 to 1.
        ValueError: If ``elevation`` is not a number from -90 to 90.
        ValueError: If ``azimuth`` is not a number.
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

    if not isinstance(Z, np.ndarray):
        raise ValueError("Z must be a NumPy array")

    if Z.ndim != 2:
        raise ValueError(f"Z must be a 2-D array, got {Z.ndim} dimension(s)")

    if Z.shape[0] < 2 or Z.shape[1] < 2:
        raise ValueError("Z must have at least 2 rows and 2 columns to draw a surface")

    if isinstance(cmap, str):
        if cmap not in colormaps:
            raise ValueError(f"cmap '{cmap}' is not a matplotlib colormap")
    elif not isinstance(cmap, Colormap):
        raise ValueError("cmap must be a colormap name or a matplotlib Colormap")

    if (not isinstance(figsize, (tuple, list)) or len(figsize) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0 for v in figsize)):
        raise ValueError("figsize must be a pair of numbers greater than 0, like (9, 6)")

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

    if isinstance(elevation, bool) or not isinstance(elevation, (int, float)) or not (-90 <= elevation <= 90):
        raise ValueError("elevation must be a number from -90 to 90")

    if isinstance(azimuth, bool) or not isinstance(azimuth, (int, float)):
        raise ValueError("azimuth must be a number")

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

    n_bands, n_cols = Z.shape
    X, Y = np.meshgrid(np.linspace(0, input_audio.duration, n_cols), np.arange(n_bands))
    surface_options = dict(cmap=cmap, vmin=Z.min(), vmax=Z.max(),
                           rcount=min(n_bands, 200), ccount=min(n_cols, 400),
                           linewidth=0, antialiased=False)

    # The full figure: axes, labels, colour bar and optional grid
    grid_style = {"grid.color": to_rgba(grid_color, grid_alpha),
                  "grid.linestyle": grid_linestyle,
                  "grid.linewidth": grid_linewidth}

    with plt.rc_context(grid_style):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection="3d")

    surface = ax.plot_surface(X, Y, Z, **surface_options)
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_box_aspect((2, 1, 0.6))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency band")
    ax.set_zlabel("Relative level", labelpad=8)
    ax.set_title("Spectrogram terrain")
    fig.colorbar(surface, ax=ax, shrink=0.6, pad=0.18, label="Relative level")

    ax.grid(grid)

    if save_path is not None:
        if image_only:
            # A second figure with a single axes that fills the whole canvas
            bare = plt.figure(figsize=figsize)
            bare_ax = bare.add_axes([0, 0, 1, 1], projection="3d")
            bare_ax.plot_surface(X, Y, Z, **surface_options)
            bare_ax.view_init(elev=elevation, azim=azimuth)
            bare_ax.set_box_aspect((2, 1, 0.6), zoom=1.3)
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
