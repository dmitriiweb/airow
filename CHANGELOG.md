# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-06-19

- Added automatic backend selection for pandas and Polars DataFrames
- Added a public `DataFrameBackend` interface for custom implementations
- Added `pandas`, `polars`, and `all` optional dependency extras
- Fixed pandas result writes for DataFrames with non-default indexes
- Added backend-specific examples and expanded integration test coverage
- **Breaking:** Installing `airow` no longer installs pandas; use
  `airow[pandas]` to retain pandas support

## [0.1.0] - 2025-09-13

- Support for string-based model specification in addition to Model objects
- Retry functionality with configurable retry count (default: 3)
- Example usage in README showing string model specifications
