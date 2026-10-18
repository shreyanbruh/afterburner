import numpy as np
import librosa
from typing import NamedTuple
from afterburner.audio import Audio


class PhaseData(NamedTuple):
    """The level and phase information for a phase-coloured spectrogram.

    It is a tuple, so it can be unpacked, and its fields can also be read by
    name. The first frame of the analysis has no earlier frame to compare with,
    so every array here starts at the second frame; ``n_frames`` below means
    the number of columns that remain.

    Attributes:
        level_db (np.ndarray): The strength of each frequency at each moment in
            dB, a 2-D float array of shape ``(n_bins, n_frames)``. The loudest
            value is 0 and everything else is negative, no lower than
            ``-top_db`` when ``top_db`` is set. Rows are frequency bins from
            lowest to highest and columns are time frames.
        phase_deviation (np.ndarray): How far each bin's phase moved between
            one frame and the next, compared with how far it would move for a
            tone exactly at the bin's centre frequency. A 2-D float array of
            shape ``(n_bins, n_frames)`` in radians, from -pi to pi. A steady
            tone gives a steady value along time, and the value says how far
            the tone sits from the bin's centre frequency. Where ``level_db``
            is very low the phase is mostly noise and means little.
        frequencies (np.ndarray): The centre frequency of each row in Hz, a 1-D
            float array of shape ``(n_bins,)``, with ``n_bins = 1 + n_fft // 2``.
        times (np.ndarray): The time of each column in seconds, a 1-D float
            array of shape ``(n_frames,)``.
    """
    level_db: np.ndarray
    phase_deviation: np.ndarray
    frequencies: np.ndarray
    times: np.ndarray


def phase_spectrogram(input_audio: Audio,
                      n_fft: int = 2048,
                      hop_length: int | None = None,
                      top_db: float | None = 80.0):
    """Compute the level and phase data for a phase-coloured spectrogram.

    Args:
        input_audio (Audio): The clip to analyse. Its ``audio_array`` and
            ``sampling_rate`` are used as the signal and its sample rate.
        n_fft (int): The size of the analysis window in samples. Defaults to
            2048. Must be a whole number of at least 2 and no greater than the
            number of samples in the clip. Bigger values separate close
            frequencies better but blur fast changes in time.
        hop_length (int | None): Number of samples between the starts of
            consecutive frames. Defaults to ``None``, which means ``n_fft // 4``
            (at least 1). Must be a whole number of at least 1 and no greater
            than ``n_fft``. Smaller values give more frames and a smoother
            picture.
        top_db (float | None): How far below the loudest value, in dB, the
            levels are allowed to go. Anything lower is raised to that floor.
            Defaults to 80. Must be at least 0. ``None`` turns the floor off.

    Returns:
        PhaseData: A named tuple ``(level_db, phase_deviation, frequencies,
        times)``. ``level_db`` and ``phase_deviation`` have shape
        ``(n_bins, n_frames)`` with ``n_bins = 1 + n_fft // 2`` and
        ``n_frames = n_samples // hop_length``. ``frequencies`` has shape
        ``(n_bins,)`` and ``times`` has shape ``(n_frames,)``.

    Raises:
        ValueError: If ``n_fft`` is not a whole number.
        ValueError: If ``n_fft`` is less than 2.
        ValueError: If ``n_fft`` is greater than the number of samples in the
            clip.
        ValueError: If ``hop_length`` is not a whole number.
        ValueError: If ``hop_length`` is less than 1.
        ValueError: If ``hop_length`` is greater than ``n_fft``.
        ValueError: If ``top_db`` is not a number or ``None``.
        ValueError: If ``top_db`` is negative.
        ValueError: If the clip is silent.
    """

    if isinstance(n_fft, bool) or not isinstance(n_fft, int):
        raise ValueError("n_fft must be a whole number")

    if (n_fft < 2):
        raise ValueError("n_fft must be greater than or equal to 2")

    if (n_fft > input_audio.n_samples):
        raise ValueError(f"n_fft cannot exceed {input_audio.n_samples} samples (the length of the clip)")

    if hop_length is None:
        hop_length = max(1, n_fft // 4)

    if isinstance(hop_length, bool) or not isinstance(hop_length, int):
        raise ValueError("hop_length must be a whole number")

    if (hop_length < 1):
        raise ValueError("hop_length must be greater than or equal to 1")

    if (hop_length > n_fft):
        raise ValueError(f"hop_length cannot exceed n_fft ({n_fft})")

    if top_db is not None:
        if isinstance(top_db, bool) or not isinstance(top_db, (int, float)):
            raise ValueError("top_db must be a number or None")

        if (top_db < 0):
            raise ValueError("top_db cannot be negative")

    sampling_rate = input_audio.sampling_rate

    D = librosa.stft(input_audio.audio_array, n_fft=n_fft, hop_length=hop_length)

    magnitude = np.abs(D[:, 1:])

    if magnitude.max() == 0:
        raise ValueError("The clip is silent")

    level_db = librosa.amplitude_to_db(magnitude, ref=np.max, top_db=top_db)

    advance = np.angle(D[:, 1:] * np.conj(D[:, :-1]))

    expected = 2 * np.pi * hop_length * np.arange(D.shape[0])[:, None] / n_fft

    # The difference, wrapped into -pi to pi
    phase_deviation = np.angle(np.exp(1j * (advance - expected)))

    frequencies = librosa.fft_frequencies(sr=sampling_rate, n_fft=n_fft)
    times = np.arange(1, D.shape[1]) * hop_length / sampling_rate

    return PhaseData(level_db=level_db,
                     phase_deviation=phase_deviation,
                     frequencies=frequencies,
                     times=times)
