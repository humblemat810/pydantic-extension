# Changelog

## 0.0.8 - 2026-10-21

- Cache resolved type hints and `Annotated` metadata to reduce repeated mode-slicing work.
- Invalidate resolved hints when class or base-class annotations are replaced or mutated.
- Preserve support for mutable markers and unhashable `Annotated` metadata.
