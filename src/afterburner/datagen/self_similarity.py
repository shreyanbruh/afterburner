import numpy as np
import librosa


def self_similarity(S: np.ndarray, n_mfcc: int = 20):
    """Compare every moment of a clip with every other moment.

    Args:
        S (np.ndarray): A 2-D spectrogram of shape ``(n_bands, n_frames)`` in
            dB, for example the array returned by ``mel_spectrogram``. Rows are
            frequency bands and columns are time frames. It must be in dB (log
            power), not raw power, or the result is meaningless. It must not
            be flat (all values equal).
        n_mfcc (int): The number of MFCC coefficients used to describe each
            frame. Defaults to 20. Coefficient 0, which is mostly overall
            loudness, is included, so loud and quiet versions of the same
            sound are not treated as identical. Must be a whole number of at
            least 1 and no greater than ``n_bands``. Fewer coefficients compare
            only the broad shape of the spectrum; more include finer detail.

    Returns:
        np.ndarray: A 2-D float array of shape ``(n_frames, n_frames)``. Entry
        ``[i, j]`` is the cosine similarity of frame ``i`` and frame ``j``:
        close to 1 for the same sound, close to 0 for unrelated sounds, and
        close to -1 for opposite ones. The array is symmetric and its diagonal
        is 1. Both axes are time in the same order as the columns of ``S``; to
        convert a frame number to seconds, use
        ``frame * hop_length / sampling_rate`` with the hop length and sample
        rate the spectrogram was computed with. The array has ``n_frames``
        squared entries, so long clips with small hop lengths give very large
        arrays.

    Raises:
        ValueError: If ``S`` is not a 2-D array.
        ValueError: If ``S`` is flat (its maximum equals its minimum).
        ValueError: If ``n_mfcc`` is not a whole number.
        ValueError: If ``n_mfcc`` is less than 1.
        ValueError: If ``n_mfcc`` is greater than ``n_bands``.
        ValueError: If a frame is identical to the average frame, which
            happens when ``S`` has only one frame, so it cannot be scaled to
            length 1.
    """

    if S.ndim != 2:
        raise ValueError(f"S must be a 2-D array, got {S.ndim} dimension(s)")

    if S.max() == S.min():
        raise ValueError("Spectrogram is flat; nothing to compare")

    if isinstance(n_mfcc, bool) or not isinstance(n_mfcc, int):
        raise ValueError("n_mfcc must be a whole number")

    if (n_mfcc < 1):
        raise ValueError("n_mfcc must be greater than or equal to 1")


    if (n_mfcc > S.shape[0]):
        raise ValueError(f"n_mfcc cannot exceed {S.shape[0]}")

    M = librosa.feature.mfcc(S=S, n_mfcc=n_mfcc)

    # Centre each coefficient over time
    M = M - M.mean(axis=1, keepdims=True)

    # Scale each frame (column) to length 1
    norms = np.linalg.norm(M, axis=0)

    if np.any(norms == 0):
        raise ValueError("A frame is identical to the average frame, so it cannot be scaled to length 1")

    M = M / norms

    return M.T @ M