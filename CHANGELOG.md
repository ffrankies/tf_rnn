# Changelog

Keeps a running log of changes to the project codebase.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project plans to adhere to [Semantic Versioning](http://semver.org/spec/v2.0.0.html) once it reaches version 1.0.0.

## Planned

### For 0.6.3

- Add support for an "observer" that shows what the network is predicting during validation (and testing).

### For 0.6.4

- Add support for having separate vocabularies for input and output features

### For later

- Non-custom variables in `SettingsNamespace`s should be pre-defined so that IDEs could apply auto-completion to them
- Fix tensorflow deprecation warning for calling softmax with dim instead of axis
  ```
  calling softmax (from tensorflow.python.ops.nn_ops) with dim is deprecated and will be removed in a future version.
  Instructions for updating:
  dim is deprecated, use axis instead
  ```
- Fix the loss-starts-at-0 problem in accumulator.
- Add custom errors, and use them where necessary.
- Properly set requirements.txt (currently there are version discrepancies)
- Update type hints in documentation (and in function parameters) to the correct standard
- Add type hints to initialized variables, where needed
- Turn Logger into a static class, to prevent overhead from re-instantiation of the class
- Turn Settings into a static class, for same reason as above

### Continuously Working On...

- Improve performance by removing unnecessary calculations.
- Use underscores to declare private variables and methods.

## Unreleased

### Added

- Added an Observer static class for comparing predictions and labels during training.
- Added extensible settings subclasses:
  - GeneralSettings
  - LoggingSettings
  - RNNSettings
  - TrainingSettings
  - DatasetSettings

### Changed

### Removed

### Deprecated

### Fixed

### Security

## 0.6.2

### Added

- `constants` now has information about where data partitions are stored
- The predictions made on the test partition are now being saved to a CSV file

### Changed

- `shuffle_seed` has been moved from `Settings.rnn` to `Settings.data`
- `END` tokens now indexed for padding
- `trainer.get_feed_dict()` parameter changed from `dataset` to `partition` for clarity
- `DatasetBase` class now expects data to already be shuffled
- `DatasetBase` class now expects data sequences to be split into training, validation and testing partitions
- `batchmaker` module now makes batches using a `multiprocessing.Pool` to speed things up
- Batches are now stored in a `Batch` namedtuple defined in `batchmaker`
- The `DataPartition` class now saves the partition to a file upon creation
- The `DataPartition` class is now an iterable - items returned are batches
- The `num_features` and `shuffle_seed` parameters have been moved from `settings.rnn` to `settings.data`
- Renamed `row_lengths` and `get_row_lengths` to `sequence_lengths` and `get_sequence_lengths` for clarity

### Removed

- The `shuffle` method from the `DatasetBase` class
- The 'CrossValidationDataset' class
- `trace` statements used by the `batchmaker` functions being called within the `multiprocessing.Pool`

## 0.6.1

### Added

- A `utils` sub-package under `layers/`.
- The `AccumulatorData` object, a `namedtuple` for storing the data used to update the `Accumulator`.
- The `ConfusionMatrix.performance_metrics()` method, for calculating accuracy, precision, recall and f1_score.
- The `TimestepAccuracies` class for storing cumulative timestep accuracy information within the `Accumulator`.
- The `Accumulator.best_accuracy()` and `Accumulator.is_best_accuracy()` methods.
- The `trainer` module will now save the actual sequences and predictions generated while testing the model.
- The `plotter` now plots f1_scores.
- The `Accumulator` now has the `get_timestep_accuracies()` method for retrieving timestep accuracies.

### Changed

- Moved `Accumulator`, `ConfusionMatrix` and `Metrics` into separate submodules in `layers/utils/`.
- `Accumulator.update()` now takes an `AccumulatorData` object as the `data` parameter, not a `list`.
- Renamed `plot` to `axes` in `plotter` to avoid name collisions.
- The `Accumulator` now uses the `TimestepAccuracies` class to store timestep accuracy info.
- The `Accumulator`, `ConfusionMatrix` and `MetaInfo` classes no longer accept a logger as an argument.

### Removed

- Tensorboard no longer logs average accuracy and test partition results, since those are available elsewhere.
- The `Accumulator.best_accuracy` and `Accumulator.is_best_accuracy` instance variables.

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