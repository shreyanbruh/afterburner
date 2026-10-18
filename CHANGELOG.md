# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-10-17

First public release.

### Added
- `afterburner.audio`: load an audio file with `ab.audio.load(path)`.
- `afterburner.datagen`: data generators for nine graph types: mel
  spectrogram, chromagram, phase spectrogram, radial spectrogram, reassigned
  spectrogram, self-similarity matrix, terrain, waterfall and Chladni pattern.
- `afterburner.plots`: a plot function for each graph type (`plot_mel`,
  `plot_chroma`, `plot_phase`, `plot_radial`, `plot_reassigned`,
  `plot_self_similarity`, `plot_terrain`, `plot_waterfall`, `plot_chladni`).
- Shared plot options: colormap, figure size, grid styling, `image_only`
  export, `save_path`, `dpi` and `show`.
- Input validation with clear `ValueError` messages on every public function.
- Packaging with a `src` layout, MIT license, and optional `tui` and `dev`
  extras.

[Unreleased]: https://github.com/shreyanbruh/afterburner/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/shreyanbruh/afterburner/releases/tag/v0.1.0