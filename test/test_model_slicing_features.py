import json
import warnings
from typing import Annotated, ClassVar, Literal, Optional, Union

import pytest
from pydantic import BaseModel, Field

from pydantic_extension.model_slicing.mixin import (
    BackendField,
    DtoField,
    ExcludeMode,
    FrontendField,
    LLMField,
    ModeSlicingMixin,
    NotMode,
    promote_to_base,
    use_mode,
)


class StableModeModel(ModeSlicingMixin, BaseModel):
    stack_mode_detection: ClassVar[bool] = False


class Crew(StableModeModel):
    default_include_modes: ClassVar[set[str]] = {"dto", "frontend"}
    default_exclude_modes: ClassVar[set[str]] = {"llm"}
    include_unmarked_for_modes: ClassVar[set[str]] = {"dto", "backend"}

    note: Annotated[str, DtoField()]
    codename: Annotated[str, FrontendField()]
    secret: Annotated[str, BackendField(), ExcludeMode("llm")]
    vector: Annotated[int, FrontendField(), LLMField()]
    misc: str = "loose"


class RegistryModel(StableModeModel):
    pass


AdminField = RegistryModel.register_mode("admin")


class Staff(RegistryModel):
    name: Annotated[str, DtoField(), FrontendField()]
    clearance: RegistryModel.build_annotated(int, "admin", "backend") = 0
    admin_code: AdminField[str]
    note: str = "ops"


class NestedChild(StableModeModel):
    visible: Annotated[str, DtoField()]
    internal: Annotated[str, BackendField()]


class NestedParent(StableModeModel):
    default_include_modes: ClassVar[set[str]] = {"dto"}

    child: Annotated[NestedChild, DtoField()]
    children: Annotated[list[NestedChild], DtoField()] = Field(default_factory=list)


class Span(BaseModel):
    start: int
    end: int


class Grounding(StableModeModel):
    default_include_modes: ClassVar[set[str]] = {"dto"}
    spans: Annotated[list[Span], DtoField()] = Field(default_factory=list)


class Node(StableModeModel):
    default_include_modes: ClassVar[set[str]] = {"dto"}
    mentions: Annotated[list[Grounding], DtoField()] = Field(default_factory=list)


class Envelope(StableModeModel):
    default_include_modes: ClassVar[set[str]] = {"dto"}
    points: Annotated[list[Span], DtoField()] = Field(default_factory=list)
    trail: Annotated[tuple[Span, ...], DtoField()] = Field(default_factory=tuple)
    lookup: Annotated[dict[str, Span], DtoField()] = Field(default_factory=dict)


class ReferenceSession(StableModeModel):
    collection_page_url: Annotated[str, DtoField()]
    doc_id: Annotated[str, DtoField()]
    verification: Annotated[Optional[str], BackendField()] = None


class GraphEntityBase(StableModeModel):
    label: Annotated[str, DtoField()]
    type: Annotated[Literal["entity", "relationship"], DtoField()]
    summary: Annotated[str, DtoField()]
    domain_id: Annotated[Optional[str], BackendField()] = None
    references: Annotated[list[ReferenceSession], DtoField()]


class StructuredOutput(StableModeModel):
    public: Annotated[str, LLMField()]
    secret: Annotated[str, BackendField()]


class PlainNoDefaults(StableModeModel):
    note: Annotated[str, DtoField()]
    secret: Annotated[str, BackendField()]
    misc: str = "plain"


class NodeChild(StableModeModel):
    default_include_modes: ClassVar[set[str]] = {"dto"}
    value: Annotated[int, DtoField()] = 1


class NodeParent(StableModeModel):
    default_include_modes: ClassVar[set[str]] = {"dto"}
    child: Annotated[NodeChild, DtoField()] = Field(default_factory=NodeChild)
    label: Annotated[str, DtoField()] = "parent"


def props(model_cls) -> set[str]:
    if isinstance(model_cls, dict):
        return set(model_cls.get("properties", {}).keys())
    return set(model_cls.model_json_schema().get("properties", {}).keys())


def make_crew() -> Crew:
    return Crew(note="brief", codename="Vega", secret="vault", vector=9, misc="loose")


def test_register_mode_and_generic_marker_syntax_are_preserved():
    staff = Staff(name="Neo", clearance=7, admin_code="T1@RA810")

    assert "admin" in Staff.modes()
    assert staff.model_dump(field_mode="admin") == {"clearance": 7, "admin_code": "T1@RA810"}
    assert props(Staff[AdminField]) == {"clearance", "admin_code"}


def test_build_annotated_rejects_unknown_mode_and_unsupported_specs():
    with pytest.raises(ValueError, match="Unknown mode"):
        RegistryModel.build_annotated(str, "ghost")

    with pytest.raises(TypeError, match="Unsupported marker spec"):
        RegistryModel.build_annotated(str, object())


def test_default_slice_slice_for_and_wrapper_helpers_match_current_schema_behavior():
    assert props(Crew.default_slice()) == {"note", "codename", "misc"}
    assert props(Crew.slice_for("backend")) == {"secret", "misc"}
    assert props(Crew.slice_for("*", exclude_modes="llm")) == {"note", "codename", "misc"}
    assert props(Crew.json_schema_for("backend")) == {"secret", "misc"}
    assert Crew.openapi_response_model("backend") is Crew.slice_for("backend")


def test_slice_cache_and_marker_class_slicing_work():
    assert Crew["dto"] is Crew["dto"]
    assert props(Crew[FrontendField]) == {"codename", "vector"}


def test_unknown_slice_tokens_raise_errors():
    with pytest.raises(TypeError, match="Unknown mode"):
        Crew["ghost"]

    with pytest.raises(TypeError, match="Unsupported slice token"):
        Crew[123]

    with pytest.raises(TypeError, match="Unknown mode"):
        Crew[NotMode("ghost")]


def test_runtime_dump_modes_and_unmarked_behavior():
    crew = make_crew()

    assert crew.model_dump() == {"note": "brief", "codename": "Vega", "misc": "loose"}
    assert crew.model_dump(field_mode="frontend") == {"codename": "Vega", "vector": 9}
    assert crew.model_dump(field_mode="backend") == {"secret": "vault", "misc": "loose"}
    assert crew.model_dump(field_mode="llm") == {"vector": 9}


def test_use_mode_context_controls_runtime_dump():
    crew = make_crew()

    with use_mode("backend"):
        dumped = crew.model_dump()

    assert dumped == {"secret": "vault", "misc": "loose"}


def test_field_mode_exclude_without_include_uses_default_slice_then_subtracts():
    crew = make_crew()

    assert crew.model_dump(field_mode_exclude="llm") == {"note": "brief", "codename": "Vega", "misc": "loose"}


def test_field_mode_exclude_outside_active_default_modes_is_ignored():
    crew = make_crew()

    assert crew.model_dump(field_mode_exclude="backend") == {"note": "brief", "codename": "Vega", "misc": "loose"}


def test_field_exclude_alias_works_with_explicit_include():
    crew = make_crew()

    assert crew.model_dump(field_mode="*", field_exclude="llm") == {
        "note": "brief",
        "codename": "Vega",
        "misc": "loose",
    }


def test_field_mode_exclude_without_active_modes_is_a_no_op():
    plain = PlainNoDefaults(note="brief", secret="vault", misc="plain")

    assert plain.model_dump(field_mode_exclude="llm") == {
        "note": "brief",
        "secret": "vault",
        "misc": "plain",
    }


def test_node_mode_drops_nested_relationship_fields():
    parent = NodeParent()

    assert parent.model_dump(field_mode="dto", node_mode=True) == {"label": "parent"}


def test_deep_sliced_nested_children_validate_without_backend_fields():
    dto_parent = NestedParent["dto"].model_validate(
        {"child": {"visible": "a"}, "children": [{"visible": "b"}]}
    )

    assert dto_parent.model_dump() == {
        "child": {"visible": "a"},
        "children": [{"visible": "b"}],
    }


def test_nested_serialization_does_not_emit_unexpected_value_warning_for_nested_models():
    node = Node(mentions=[Grounding(spans=[Span(start=1, end=3)])])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dumped = node.model_dump()

    assert dumped == {"mentions": [{"spans": [{"start": 1, "end": 3}]}]}
    assert all("PydanticSerializationUnexpectedValue" not in str(w.message) for w in caught)


def test_plain_basemodel_containers_serialize_without_unexpected_value_warning():
    envelope = Envelope(
        points=[Span(start=2, end=4)],
        trail=(Span(start=5, end=8),),
        lookup={"lead": Span(start=9, end=10)},
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dumped = envelope.model_dump()

    assert dumped == {
        "points": [{"start": 2, "end": 4}],
        "trail": ({"start": 5, "end": 8},),
        "lookup": {"lead": {"start": 9, "end": 10}},
    }
    assert all("PydanticSerializationUnexpectedValue" not in str(w.message) for w in caught)


def test_model_dump_json_matches_json_mode_dump():
    crew = make_crew()

    assert json.loads(crew.model_dump_json(field_mode="backend")) == crew.model_dump(
        dump_format="json", field_mode="backend"
    )


def test_model_json_schema_prefers_llm_context_and_preserves_base_title():
    with use_mode("llm"):
        schema = StructuredOutput.model_json_schema()

    assert schema["title"] == "StructuredOutput"
    assert set(schema["properties"].keys()) == {"public"}


def test_conflict_policy_warns_on_class_definition():
    with pytest.warns(UserWarning, match="default include/exclude overlap"):
        class WarnOverlap(StableModeModel):
            default_include_modes: ClassVar[set[str]] = {"dto", "llm"}
            default_exclude_modes: ClassVar[set[str]] = {"llm"}
            default_conflict_policy: ClassVar[str] = "warn"

            a: Annotated[int, DtoField()] = 1
            b: Annotated[int, LLMField()] = 2


def test_conflict_policy_errors_on_class_definition():
    with pytest.raises(ValueError, match="default include/exclude overlap"):
        class ErrorOverlap(StableModeModel):
            default_include_modes: ClassVar[set[str]] = {"dto"}
            default_exclude_modes: ClassVar[set[str]] = {"dto"}
            default_conflict_policy: ClassVar[str] = "error"

            a: Annotated[int, DtoField()] = 1


def test_promote_to_base_restores_base_model_from_slice_and_fills_missing_values():
    dto_instance = GraphEntityBase["dto"].model_validate(
        {
            "label": "Apple",
            "type": "entity",
            "summary": "Tech company",
            "references": [{"collection_page_url": "http://coll", "doc_id": "doc-123"}],
        }
    )

    promoted = promote_to_base(
        dto_instance,
        defaults={"domain_id": "finance"},
        path_defaults={"references[0].verification": "filled from context"},
    )

    assert type(promoted) is GraphEntityBase
    assert promoted.domain_id == "finance"
    assert promoted.references[0].verification == "filled from context"


def test_promote_to_base_can_overwrite_existing_nested_values_when_requested():
    entity = GraphEntityBase(
        label="Apple",
        type="entity",
        summary="Tech company",
        references=[
            ReferenceSession(
                collection_page_url="http://coll",
                doc_id="doc-123",
                verification="original",
            )
        ],
    )

    promoted = promote_to_base(
        entity,
        path_defaults={"references[0].verification": "override"},
        only_if_missing=False,
    )

    assert promoted.references[0].verification == "override"


def test_llm_mode_redirects_validation_to_llm_slice():
    with use_mode("llm"):
        validated = StructuredOutput.model_validate({"public": "hello"})

    assert type(validated) is StructuredOutput["llm"]
    assert validated.model_dump() == {"public": "hello"}


def test_explicit_context_blocks_llm_redirect():
    with use_mode("llm"):
        validated = StructuredOutput.model_validate(
            {"public": "hello", "secret": "vault"},
            context={StructuredOutput._ctx_mode_key(): "dto"},
        )

    assert type(validated) is StructuredOutput
    assert validated.model_dump(field_mode="*") == {"public": "hello", "secret": "vault"}
