import numbers
import numpy as np
import librosa
from typing import NamedTuple
from afterburner.audio import Audio

PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


class ChromaData(NamedTuple):
    """The chromagram, with its time axis and row labels.

    It is a tuple, so it can be unpacked (``chroma, times, labels = data``), and
    its fields can also be read by name.

    Attributes:
        chroma (np.ndarray): The chromagram, a 2-D float32 array of shape
            ``(12, n_frames)`` with values between 0 and 1. Rows are the 12
            pitch classes in the order of ``labels`` (C first, B last) and
            columns are time frames. Each frame is scaled so that its largest
            value is 1, except frames where everything fell below the
            threshold, which are all 0.
        times (np.ndarray): The time of each frame in seconds, a 1-D float
            array of shape ``(n_frames,)``.
        labels (tuple[str, ...]): The 12 pitch-class names, one per row of
            ``chroma``: ``("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#",
            "A", "A#", "B")``.
    """
    chroma: np.ndarray
    times: np.ndarray
    labels: tuple[str, ...]


def chromagram(input_audio: Audio,
               hop_length: int = 512,
               threshold: float = 0.01,
               gamma: float = 1.0):
    """Compute a chromagram: the energy in each of the 12 pitch classes over time.

    Args:
        input_audio (Audio): The clip to analyse. Its ``audio_array`` and
            ``sampling_rate`` are used as the signal and its sample rate. The
            lowest notes need analysis windows over a second long, so very
            short clips give unreliable results and librosa may warn.
        hop_length (int): Number of samples between consecutive frames.
            Defaults to 512. Must be a positive whole number.
        threshold (float): Absolute level below which a chroma value is set to
            0 before each frame is scaled to a maximum of 1. Defaults to 0.01.
            This keeps silent frames empty instead of letting scaling turn
            near-silence into full-strength noise. It is an absolute level, so a
            very quiet recording may need a smaller value. Must be at least 0;
            0 turns the threshold off.
        gamma (float): Exponent applied to the 0 to 1 values after scaling.
            Defaults to 1.0, which leaves them unchanged. Above 1 pushes weaker
            notes down so only the strongest stand out; below 1 lifts weaker
            notes. Must be a finite number greater than 0.

    Returns:
        ChromaData: A named tuple ``(chroma, times, labels)``. ``chroma`` has
        shape ``(12, n_frames)`` with ``n_frames = 1 + n_samples // hop_length``,
        ``times`` has shape ``(n_frames,)``, and ``labels`` holds the 12 note
        names.

    Raises:
        ValueError: If ``hop_length`` is not a whole number.
        ValueError: If ``hop_length`` is less than 1.
        ValueError: If ``threshold`` is not a number.
        ValueError: If ``threshold`` is negative.
        ValueError: If ``gamma`` is not a number.
        ValueError: If ``gamma`` is not finite and greater than 0.
    """

    if isinstance(hop_length, bool) or not isinstance(hop_length, numbers.Integral):
        raise ValueError("hop_length must be a whole number")

    if (hop_length < 1):
        raise ValueError("hop_length must be greater than or equal to 1")

    if isinstance(threshold, bool) or not isinstance(threshold, numbers.Real):
        raise ValueError("threshold must be a number")

    if (threshold < 0):
        raise ValueError("threshold cannot be negative")

    if isinstance(gamma, bool) or not isinstance(gamma, numbers.Real):
        raise ValueError("gamma must be a number")

    if (not np.isfinite(gamma)) or (gamma <= 0):
        raise ValueError("gamma must be a finite number greater than 0")

    chroma = librosa.feature.chroma_cqt(y=input_audio.audio_array,
                                        sr=input_audio.sampling_rate,
                                        hop_length=hop_length,
                                        threshold=threshold)

    chroma = chroma ** gamma

    times = librosa.frames_to_time(np.arange(chroma.shape[1]),
                                   sr=input_audio.sampling_rate,
                                   hop_length=hop_length)

    return ChromaData(chroma=chroma, times=times, labels=PITCH_CLASSES)
