# Changelog

Keeps a running log of changes to the project codebase.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## Planned / Unreleased

- Improve performance by removing unnecessary calculations.
- Remove unnecessary tensorboard logs.
- Add precision and recall calculations.
- Use underscores to declare private variables and methods.

## 0.6.1

### Added

- A `utils` sub-package under `layers/`.
- The `AccumulatorData` object, a `namedtuple` for storing the data used to update the `Accumulator`.
- The `ConfusionMatrix.performance_metrics()` method, for calculating accuracy, precision, recall and f1_score.

### Changed

- Moved `Accumulator`, `ConfusionMatrix` and `Metrics` into separate submodules in `layers/utils/`.
- `Accumulator.update()` now takes an `AccumulatorData` object as the `data` parameter, not a `list`.

### Fixed

- Some pylint warnings.

## 0.6.0

### Added

- MIT License.
- Proper setup.py, to allow the project to be installed locally as a pip project.

### Changed

- Folder structure.
- Old setup.py module moved to cmd_arg_parser.py
- Code style-related stuff.

### Removed

- All methods from setup.py that didn't have to do with reading command-line arguments.

## 0.5.1 - 2018-05-21

### Added

- This changelog file, for keeping track of changes.

### Changed

- Split performance layer tests into two: one for the accumulator and one for the confusion matrix.
- Indicated some private methods with underscores in Accumulator object.

### Fixed

- Fixed out-of-date accumulator tests.
- Fixed out-of-date batchmaker tests.
- Fixed running average calculation error in accumulator - initial average is now the average of the first batch, not 0.

## Format

### Added

New stuff!

### Changed

Old stuff that isn't quite the same anymore.

### Deprecated

Old stuff that will soon be buried.

### Removed

RIP.

### Fixed

Messes that were cleaned up.

### Security

Changes that affect security.