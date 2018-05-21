# Changelog

Keeps a running log of changes to the project codebase.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## Planned / Unreleased

- Separate performance_layer.py into multiple python modules.
- Use underscores to declare private variables and methods.

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