from dataclasses import dataclass
import librosa
import numpy as np
from typing import ClassVar


@dataclass (frozen = True, eq = False)
class Audio:
    """An immutable, mono audio clip loaded from a file.

    Create instances with ``Audio.load(path)``. Because the class is frozen,
    attributes cannot be reassigned after creation; to use a different file,
    offset, or sample rate, call ``Audio.load`` again to get a new instance.

    Attributes:
        MIN_SAMPLES (int): Class-level constant (2048), shared by all instances
            and not a constructor argument. The minimum number of samples, i.e.
            ``len(audio_array)``, that ``load`` will accept. At 22,050 Hz this
            is about 0.09 seconds of audio.
        audio_array (np.ndarray): The waveform as a 1-D array of shape
            ``(n_samples,)``. Mono, dtype float32 (librosa's default), with
            values roughly in the range -1.0 to 1.0.
        file_path (str | os.PathLike): The path exactly as it was passed to
            ``load``, so it keeps whichever type was given. It is stored for
            later use (e.g. naming outputs) and is not re-read.
        sampling_rate (int): Samples per second (Hz) of ``audio_array``. This is
            the file's own rate, or the rate requested via ``sr`` in ``load``.
        offset (float): Seconds into the original file where this clip starts.
            ``load`` defaults it to 0.0.
        duration (float): Read-only property. Length of this clip in seconds,
            computed from ``audio_array`` and ``sampling_rate``.
        n_samples (int): Read-only property. Number of samples in
            ``audio_array``.
    """
    MIN_SAMPLES: ClassVar[int] = 2048
    audio_array: np.ndarray
    file_path: str
    sampling_rate: int
    offset: float

    def __post_init__(self):
        """Validate the instance right after the generated ``__init__`` runs.

        Called automatically by the dataclass machinery; you never call it
        yourself. It runs for every construction, including direct calls like
        ``Audio(y, path, sr, 0.0)`` that skip ``load``.

        Raises:
            ValueError: If ``audio_array`` contains no samples.
        """
        if(len(self.audio_array) == 0):
            raise ValueError("No audio file was loaded")

    @property
    def duration(self):
        """Length of the loaded clip in seconds (float).

        Computed from ``audio_array`` and ``sampling_rate``, so it is the length
        of this clip, not of the whole original file. For example, a clip
        loaded with ``duration=5`` gives about 5.0. Read-only; access it as
        ``audio.duration``, without parentheses.
        """
        return librosa.get_duration(y = self.audio_array, sr = self.sampling_rate)

    @property
    def n_samples(self):
        """Number of samples in the loaded clip (int).

        The length of ``audio_array``: a plain count, not seconds. Divide by
        ``sampling_rate`` to get seconds, or use ``duration`` directly.
        Read-only; access it as ``audio.n_samples``, without parentheses.
        """
        return len(self.audio_array)

    @classmethod
    def load(cls,
             path,
             sr=None,
             offset = 0.0,
             duration = None):
        """Read an audio file from disk and return it as an ``Audio`` instance.

        This is the intended way to create an ``Audio``. The audio is always
        converted to mono. Errors raised by librosa itself (for example an
        unreadable or unsupported file) are not caught and pass through as is.

        Args:
            path (str | os.PathLike): Location of the audio file (wav, flac,
                mp3, ...). Stored unchanged as ``file_path`` on the result.
            sr (int | None): Target sampling rate in Hz. ``None`` (default)
                keeps the file's own rate; an int resamples to that rate. The
                value itself is not validated; only the number of samples read
                is checked (see Raises).
            offset (float): Start time in seconds. Defaults to 0.0 (the
                beginning). Must be less than the file's total duration.
            duration (float | None): Number of seconds to read from ``offset``.
                ``None`` (default) reads to the end of the file. A value that
                runs past the end of the file is allowed and simply returns the
                audio that remains. This is the length you ask for; the length
                you actually got is the ``duration`` property of the result.

        Returns:
            Audio: A new instance holding the mono float32 samples, the
            resulting sampling rate (int, Hz), the source path, and the offset.

        Raises:
            ValueError: If ``offset`` is greater than or equal to the file's
                duration.
            ValueError: If fewer than ``MIN_SAMPLES`` samples were read, which
                can happen with a very short ``duration``, an ``offset`` close
                to the end of the file, or a very short file.
        """
        file_duration = librosa.get_duration(path = path)
        if(offset >= file_duration):
            raise ValueError("Offset is larger than file duration")


        y, sr = librosa.load(path,
                        sr = sr,
                        mono = True,
                        offset = offset,
                        duration = duration)

        if((sr is not None) and (len(y) < cls.MIN_SAMPLES)):
            raise ValueError(f"Sample count is too low. Please use an audio file with atleast {cls.MIN_SAMPLES} samples")

        return cls(audio_array = y,
                   file_path = path,
                   sampling_rate = sr,
                   offset = offset)
        
        
load = Audio.load