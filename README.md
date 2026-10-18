# afterburner

Turn audio files into graphs.

## Install

```
pip install git+https://github.com/<your-username>/afterburner.git
```

## Quickstart

```python
import afterburner as ab

audio = ab.audio.load("song.mp3")
S = ab.datagen.mel(audio)
ab.plots.plot_mel(S, audio)
```

## License

MIT