# Architecture Decisions (pydantic-extension)

- ADR-001: Mode slicing as first-class concept
- ADR-002: Field annotation via `typing.Annotated` + Mode metadata
- ADR-003: Default-deny policy & conflict resolution
- ADR-004: OpenAPI/schema generation for mode-specific models
- ADR-005: Runtime dump/filter semantics (`model_dump(field_mode=...)`)
- ADR-006: Dynamic Model Slicing via __class_getitem__