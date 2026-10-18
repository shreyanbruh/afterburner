import numpy as np
import librosa
from afterburner.audio import Audio

def mel(input_audio: Audio,
                    n_fft: int = 2048,
                    hop_length: int | None = None,
                    fmin: float = 0,
                    fmax: float | None = None,
                    amin: float = 1e-10,
                    top_db: float | None = 80.0):
    """Compute a mel spectrogram of an ``Audio`` clip, in decibels.

    Args:
        input_audio (Audio): The clip to analyse. Its ``audio_array`` and
            ``sampling_rate`` are used as the signal and its sample rate.
        n_fft (int): Window length in samples for each FFT frame. Larger values
            give finer frequency detail but blurrier timing. Defaults to 2048.
            Must not be larger than ``input_audio.n_samples``.
        hop_length (int | None): Number of samples between the starts of
            consecutive frames. ``None`` (default) lets librosa choose
            ``n_fft // 4``. Must not be greater than ``n_fft``.
        fmin (float): Lowest frequency (Hz) covered by the mel bands. Defaults
            to 0. Must be at least 0 and less than ``fmax``.
        fmax (float | None): Highest frequency (Hz) covered by the mel bands.
            ``None`` (default) uses the Nyquist frequency,
            ``input_audio.sampling_rate / 2``. Must not exceed that value.
        amin (float): Absolute floor on the mel power values, applied before
            the log so that zeros never reach it. Defaults to 1e-10. Must be
            greater than 0 and less than the largest power value in the
            spectrogram, otherwise the image would be flat.
        top_db (float | None): Relative floor in dB. Any value more than
            ``top_db`` below the loudest value is raised to that floor, which
            sets how much quiet detail is kept. Defaults to 80.0. ``None``
            turns this clipping off.

    Returns:
        np.ndarray: The mel spectrogram in dB, a 2-D float32 array of shape
        ``(n_mels, n_frames)``. ``n_mels`` is librosa's default of 128 (it is
        not a parameter of this function) and ``n_frames`` is
        ``1 + n_samples // hop_length``. Rows are mel bands from lowest to
        highest frequency; columns are time frames. Values are relative to the
        loudest cell, so the maximum is 0.0 and, unless ``top_db`` is ``None``,
        no value is below ``-top_db``.

    Raises:
        ValueError: If ``n_fft`` is not a whole number.
        ValueError: If ``n_fft`` is less than 2.
        ValueError: If ``hop_length`` is not a whole number.
        ValueError: If ``hop_length`` is less than 1.
        ValueError: If ``hop_length`` is greater than ``n_fft``.
        ValueError: If ``n_fft`` is larger than ``input_audio.n_samples``.
        ValueError: If ``fmin`` is not a number.
        ValueError: If ``fmax`` is not a number or ``None``.
        ValueError: If ``0 <= fmin < fmax <= sampling_rate / 2`` does not hold,
            after ``fmax`` has been resolved.
        ValueError: If ``amin`` is not a number.
        ValueError: If ``amin`` is not greater than 0.
        ValueError: If ``top_db`` is not a number or ``None``.
        ValueError: If ``top_db`` is negative.
        ValueError: If ``amin`` is not less than the largest power value in the
            spectrogram (this check runs after the mel spectrogram is
            computed).
    """

    if isinstance(n_fft, bool) or not isinstance(n_fft, int):
        raise ValueError("n_fft must be a whole number")

    if (n_fft < 2):
        raise ValueError("n_fft must be greater than or equal to 2")

    if hop_length is not None:
        if isinstance(hop_length, bool) or not isinstance(hop_length, int):
            raise ValueError("hop_length must be a whole number")

        if (hop_length < 1):
            raise ValueError("hop_length must be greater than or equal to 1")

    if((hop_length is not None) and hop_length > n_fft):
        raise ValueError("Hop length cannot be greater than n_fft")

    if(n_fft > input_audio.n_samples):
        raise ValueError("n_fft cannot be larger than audio file sample count")

    if isinstance(fmin, bool) or not isinstance(fmin, (int, float)):
        raise ValueError("fmin must be a number")

    if fmax is not None:
        if isinstance(fmax, bool) or not isinstance(fmax, (int, float)):
            raise ValueError("fmax must be a number or None")

    if fmax is None:
        fmax = input_audio.sampling_rate / 2

    if not(0 <= fmin < fmax <= (input_audio.sampling_rate) / 2):
        raise ValueError("Require 0 <= fmin < fmax <= sampling_rate/2")

    if isinstance(amin, bool) or not isinstance(amin, (int, float)):
        raise ValueError("amin must be a number")

    if(amin <= 0):
        raise ValueError("amin cannot be less than 0")

    if top_db is not None:
        if isinstance(top_db, bool) or not isinstance(top_db, (int, float)):
            raise ValueError("top_db must be a number or None")

        if (top_db < 0):
            raise ValueError("top_db cannot be negative")

    spectrogram = librosa.feature.melspectrogram( y=input_audio.audio_array,
                                    sr=input_audio.sampling_rate,
                                    n_fft=n_fft,
                                    hop_length=hop_length,
                                    fmin=fmin,
                                    fmax=fmax )

    if (amin >= np.max(spectrogram)):
        raise ValueError(f"amin cannot be greater than the max value of spectrogram {np.max(spectrogram)}")

    return librosa.power_to_db(S=spectrogram,
                               ref = np.max(spectrogram),
                               amin=amin,
                               top_db=top_db)