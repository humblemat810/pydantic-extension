from __future__ import annotations

import copy
import json
import inspect
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from typing import (
    Any, Annotated, ClassVar, Iterable, Literal, Optional, TypeAlias, TypeVar,
    Union, get_args, get_origin, get_type_hints, Callable
)

from pydantic import BaseModel as PydBaseModel
from pydantic import Field, ConfigDict, create_model
"""
PydBaseModel: mode-based field slicing for Pydantic/SQLModel with per-class dynamic modes.

Key features
------------
- Mark fields with modes using typing.Annotated metadata (e.g., DtoField(), FrontendField(), BackendField(), LLMField()).
- Create dynamic "slice" classes via MyModel["dto"] / MyModel["dto","frontend"] / MyModel["*","-llm"] / MyModel[NotMode("llm")].
- Dump instances with the same logic: obj.model_dump(field_mode="dto", field_exclude="llm", node_mode=True).
- Per-class **dynamic mode registry**:
    * Register new modes at runtime: MyModel.register_mode("admin")
    * Build dynamic multi-marker annotations: MyModel.build_annotated(str, "dto", "admin", SomeMarkerClass(), AnotherMarker())
- Configurable defaults per subclass:
    * default_include_modes: modes to include when caller passes NONE.
    * default_exclude_modes: modes to exclude when caller passes NONE.
    * include_unmarked_for_modes: modes that also include UNMARKED fields (e.g., {"dto","frontend","backend"}; "llm" usually omitted).
- Conflict policy:
    * default_conflict_policy in {"ignore","warn","error"}; detects overlap of default include/exclude.
    * Warnings/errors at subclass creation time AND when defaults are actually used at runtime/slicing.
- Pydantic v2, v3-friendly:
    * Only uses class-level .model_fields (no deprecated instance access).
    * v3 fallback if create_model rejects reusing FieldInfo directly.

Notes
-----
- Always hand FastAPI or LangChain the **slice class** (e.g., MyModel["dto"]) to get the correct OpenAPI/JSON schema.
- Your runtime dumps do NOT affect schemas; schemas are class-based.
"""
"""
Mode-based field slicing for Pydantic/SQLModel.

This module provides:
  - ModeSlicingMixin: a mixin that adds mode slicing to ANY Pydantic model.

Key features preserved from earlier versions:
  * Per-class dynamic modes (register_mode / build_annotated)
  * Marker classes and convenient type aliases (DtoType, FrontendType, BackendType, LLMType)
  * Context-aware dumps via `use_mode("llm")` + optional stack sniffing for LangChain
  * Context-aware model_json_schema() that prefers 'llm' slice on LC structured-output stack
    and normalizes schema "title" back to base class name (tool/function naming stability)
  * Back-compat: `field_exclude=` still supported (alias of `field_mode_exclude=`)
  * node_mode: drops relation-like fields;
"""


# ----------------------------
# Markers & aliases
# ----------------------------

class BaseMarker: ...
class ExcludeMode(BaseMarker):
    def __init__(self, *modes: str):
        self.modes = {m.lower() for m in modes}

class DtoField(BaseMarker): ...
class BackendField(BaseMarker): ...
class FrontendField(BaseMarker): ...
class LLMField(BaseMarker): ...

T = TypeVar("T")
DtoType: TypeAlias = Annotated[T, DtoField()]
BackendType: TypeAlias = Annotated[T, BackendField()]
FrontendType: TypeAlias = Annotated[T, FrontendField()]
LLMType: TypeAlias = Annotated[T, LLMField()]


# ----------------------------
# Context + stack detection
# ----------------------------

_CURRENT_MODE: ContextVar[str | None] = ContextVar("_current_mode", default=None)

@contextmanager
def use_mode(mode: str | None):
    """Context manager to set default field_mode within a block."""
    token = _CURRENT_MODE.set(mode)
    try:
        yield
    finally:
        _CURRENT_MODE.reset(token)

def _infer_mode_from_stack(
    *,
    module_hints: tuple[str, ...] = ("langchain", "langgraph"),
    module_contains_any: tuple[str, ...] = ("output_parsers", "tools", "structured"),
    function_hints: tuple[str, ...] = (
        "with_structured_output",
        "as_structured_output",
        "parse_result",
        "parse",
        "bind_tools",
        "tool",
    ),
    max_depth: int = 40,
) -> str | None:
    """Heuristic: if we appear to be inside LC structured-output/tool-calling, treat as 'llm'."""
    try:
        stack = inspect.stack(context=0)
    except Exception:
        return None

    for fi in stack[:max_depth]:
        mod = inspect.getmodule(fi.frame)
        mod_name = getattr(mod, "__name__", "") or ""
        fn_name = fi.function or ""
        if not any(h in mod_name for h in module_hints):
            continue
        if not any(s in mod_name for s in module_contains_any) and not any(h in fn_name for h in function_hints):
            continue
        return "llm"
    return None


# ----------------------------
# Helpers
# ----------------------------

def _extract_all_metadata(hint: Any) -> tuple:
    """
    Recursively collect Annotated metadata and walk Optional/Union.
    Handles nested Annotated and Optional[Annotated[...]].
    """
    out: list[Any] = []
    origin = get_origin(hint)

    if origin is Annotated:
        base, *meta = get_args(hint)
        out.extend(meta)
        out.extend(_extract_all_metadata(base))
        return tuple(out)

    if origin in (Union, getattr(__import__("typing"), "UnionType", None)):
        for arg in get_args(hint):
            if arg is type(None):  # noqa: E721
                continue
            out.extend(_extract_all_metadata(arg))
        return tuple(out)

    meta = getattr(hint, "__metadata__", None)
    if meta:
        out.extend(meta)
    return tuple(out)


class NotMode:
    """Exclude token for slice specs: MyModel['dto', NotMode('llm')]."""
    def __init__(self, *modes: str):
        self.modes = {m.lower() for m in modes}


def _default_overlap_msg(cls: type, overlap: set[str]) -> str:
    return (
        f"{cls.__name__}: default include/exclude overlap on modes {sorted(overlap)} — "
        "exclusion wins if defaults are applied. Pass field_mode/field_mode_exclude to override."
    )


# ----------------------------
# Mixin (works with any Pydantic model)
# ----------------------------

class ModeSlicingMixin:
    """
    Drop-in mixin that adds mode-based slicing to any Pydantic model.

    Use:
        class User(ModeSlicingMixin, pydantic.BaseModel): ...
        class Row(ModeSlicingMixin, SQLModel): ...  # mixin first in MRO!

    Notes:
      - Slices are created as plain Pydantic models via create_model(__base__=BaseModel).
      - node_mode uses a predicate so it works without SQLModel;
        overrides it to treat pydantic instances as nodes.
    """

    # Pydantic config (merged by Pydantic with the final base)
    model_config = ConfigDict(populate_by_name=True, from_attributes=True)

    # Defaults / policies
    default_include_modes: ClassVar[set[str]] = set()
    default_exclude_modes: ClassVar[set[str]] = set()
    include_unmarked_for_modes: ClassVar[set[str]] = {"dto", "frontend", "backend"}
    default_conflict_policy: ClassVar[Literal["ignore", "warn", "error"]] = "warn"

    # Stack/context behavior toggles
    stack_mode_detection: ClassVar[bool] = True
    stack_mode_module_hints: ClassVar[tuple[str, ...]] = ("langchain", "langgraph")
    stack_mode_module_contains_any: ClassVar[tuple[str, ...]] = ("output_parsers", "tools", "structured")
    stack_mode_function_hints: ClassVar[tuple[str, ...]] = (
        "with_structured_output", "as_structured_output", "parse_result", "parse", "bind_tools", "tool"
    )
    stack_mode_max_depth: ClassVar[int] = 40
    prefer_llm_schema_on_stack: ClassVar[bool] = True

    # Relationship detection for node_mode; override in shim/class if needed
    node_model_predicate: ClassVar[Callable] = staticmethod(lambda v: isinstance(v, PydBaseModel))

    # Per-class registry and caches
    _mode_markers: ClassVar[dict[str, type[BaseMarker]]] = {}
    _slice_cache: ClassVar[dict] = {}
    _conflict_warned_classes: ClassVar[set[type]] = set()

    # ----- subclass init: register default modes & check defaults overlap -----
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        parent = getattr(cls, "_mode_markers", None)
        cls._mode_markers = dict(parent) if isinstance(parent, dict) else {}
        if not cls._mode_markers:
            cls.register_mode("dto", DtoField)
            cls.register_mode("backend", BackendField)
            cls.register_mode("frontend", FrontendField)
            cls.register_mode("llm", LLMField)

        inc = set(getattr(cls, "default_include_modes", set()))
        exc = set(getattr(cls, "default_exclude_modes", set()))
        overlap = inc & exc
        if overlap:
            pol = getattr(cls, "default_conflict_policy", "warn")
            msg = _default_overlap_msg(cls, overlap)
            if pol == "error":
                raise ValueError(msg)
            if pol == "warn" and cls not in cls._conflict_warned_classes:
                warnings.warn(msg, stacklevel=2)
                cls._conflict_warned_classes.add(cls)

    # ----- Registry API -----
    @classmethod
    def register_mode(cls, name: str, marker_cls: type[BaseMarker] | None = None) -> type[BaseMarker]:
        key = name.lower()
        if marker_cls is None:
            marker_cls = type(f"{name.capitalize()}Field", (BaseMarker,), {})
        cls._mode_markers[key] = marker_cls
        return marker_cls

    @classmethod
    def modes(cls) -> set[str]:
        return set(cls._mode_markers.keys())

    @classmethod
    def build_annotated(cls, base_type: Any, *modes_or_markers: Any):
        """Build Annotated[base_type, markers...] using this class's mode registry."""
        metadata: list[Any] = []
        for spec in modes_or_markers:
            if isinstance(spec, str):
                mk = cls._mode_markers.get(spec.lower())
                if not mk:
                    raise ValueError(f"Unknown mode '{spec}' for {cls.__name__}")
                metadata.append(mk())
            elif isinstance(spec, type) and issubclass(spec, BaseMarker):
                metadata.append(spec())
            elif isinstance(spec, BaseMarker):
                metadata.append(spec)
            else:
                raise TypeError(f"Unsupported marker spec: {spec!r}")
        return Annotated[base_type, *metadata]

    # ----- Helpers tied to this class's registry -----
    @classmethod
    def _field_modes_from_hint(cls, hint: Any) -> set[str]:
        meta = _extract_all_metadata(hint)
        return {name for name, mk in cls._mode_markers.items() if any(isinstance(m, mk) for m in meta)}

    @classmethod
    def _has_exclude_for_modes(cls, hint: Any, active_modes: set[str]) -> bool:
        return any(isinstance(m, ExcludeMode) and (m.modes & active_modes) for m in _extract_all_metadata(hint))

    # ----- Default slice & schema helpers -----
    @classmethod
    def default_slice(cls):
        inc, exc = set(cls.default_include_modes), set(cls.default_exclude_modes)
        if not inc and not exc:
            return cls
        specs: list[Any] = sorted(inc)
        if exc:
            specs.append(NotMode(*sorted(exc)))
        return cls.__class_getitem__(tuple(specs))

    @classmethod
    def slice_for(
        cls,
        modes: str | Iterable[str] | None = None,
        *,
        exclude_modes: str | Iterable[str] | None = None,
    ):
        if modes is None and exclude_modes is None:
            return cls.default_slice()
        specs: list[Any] = []
        if modes is not None:
            if isinstance(modes, str):
                specs.extend(modes.split("+") if modes != "*" else ["*"])
            else:
                specs.extend(list(modes))
        if exclude_modes:
            specs.append(NotMode(*(exclude_modes if isinstance(exclude_modes, (list, set, tuple)) else [exclude_modes])))
        return cls if not specs else cls.__class_getitem__(tuple(specs))

    @classmethod
    def json_schema_for(
        cls,
        modes: str | Iterable[str] | None = None,
        *,
        exclude_modes: str | Iterable[str] | None = None,
        **schema_kwargs,
    ) -> dict:
        return cls.slice_for(modes, exclude_modes=exclude_modes).model_json_schema(**schema_kwargs)

    @classmethod
    def openapi_response_model(cls, modes: str | Iterable[str] | None = None, *, exclude_modes: str | Iterable[str] | None = None):
        return cls.slice_for(modes, exclude_modes=exclude_modes)

    # ----- Class slicing to dynamic Pydantic model -----
    @classmethod
    def __class_getitem__(cls, item: Any):
        modes_tuple = (item,) if not isinstance(item, tuple) else item
        include_modes: set[str] = set()
        exclude_modes: set[str] = set()

        for spec in modes_tuple:
            if isinstance(spec, str):
                tok = spec.strip().lower()
                if tok == "*":
                    include_modes.update(cls._mode_markers.keys())
                elif tok.startswith(("-", "!", "~")):
                    key = tok.lstrip("-!~")
                    if key not in cls._mode_markers:
                        raise TypeError(f"Unknown mode '{key}' for {cls.__name__}")
                    exclude_modes.add(key)
                else:
                    if tok not in cls._mode_markers:
                        raise TypeError(f"Unknown mode '{tok}' for {cls.__name__}")
                    include_modes.add(tok)
            elif isinstance(spec, NotMode):
                for key in spec.modes:
                    if key not in cls._mode_markers:
                        raise TypeError(f"Unknown mode '{key}' for {cls.__name__}")
                    exclude_modes.add(key)
            elif isinstance(spec, BaseMarker) or (isinstance(spec, type) and issubclass(spec, BaseMarker)):
                target = spec if isinstance(spec, type) else type(spec)
                for name, mk in cls._mode_markers.items():
                    if mk is target:
                        include_modes.add(name)
                        break
                else:
                    raise TypeError(f"Marker {target.__name__} not registered for {cls.__name__}")
            else:
                raise TypeError(f"Unsupported slice token: {spec!r}")

        applied_defaults = False
        if not include_modes and not exclude_modes:
            include_modes, exclude_modes = set(cls.default_include_modes), set(cls.default_exclude_modes)
            applied_defaults = True

        if applied_defaults:
            overlap = include_modes & exclude_modes
            if overlap:
                pol = getattr(cls, "default_conflict_policy", "warn")
                msg = _default_overlap_msg(cls, overlap)
                if pol == "error":
                    raise ValueError(msg)
                if pol == "warn" and cls not in cls._conflict_warned_classes:
                    warnings.warn(msg, stacklevel=2)
                    cls._conflict_warned_classes.add(cls)

        if not include_modes and not exclude_modes:
            return cls

        cache_key = (cls, tuple(sorted(include_modes)), tuple(sorted(exclude_modes)))
        if cache_key in cls._slice_cache:
            return cls._slice_cache[cache_key]

        hints = get_type_hints(cls, include_extras=True)
        fields_for_dynamic: dict[str, tuple[Any, Any]] = {}

        for fname, hint in hints.items():
            original = cls.model_fields.get(fname)  # class-level access (v3-safe)
            if not original:
                continue

            f_modes = cls._field_modes_from_hint(hint)
            pos = bool(f_modes & include_modes)
            unmarked_ok = (not f_modes) and bool(include_modes & cls.include_unmarked_for_modes)
            # NOTE: keep "exclusion wins" semantics (matches earlier behavior & your tests)
            excluded_by_global = bool(f_modes & exclude_modes)
            excluded_by_field_marker = cls._has_exclude_for_modes(hint, include_modes if include_modes else exclude_modes)

            if (pos or unmarked_ok) and not excluded_by_global and not excluded_by_field_marker:
                info_copy = copy.deepcopy(original)
                # v3 fallback if FieldInfo can't be reused directly
                try:
                    fields_for_dynamic[fname] = (info_copy.annotation, info_copy)
                except TypeError:
                    fi = info_copy
                    new_field = Field(
                        default=fi.default if fi.default is not Ellipsis else ...,
                        description=getattr(fi, "description", None),
                        alias=getattr(fi, "alias", None),
                        json_schema_extra=getattr(fi, "json_schema_extra", None),
                    )
                    fields_for_dynamic[fname] = (fi.annotation, new_field)

        inc_suffix = "".join(m.capitalize() for m in sorted(include_modes)) or "None"
        exc_suffix = "Not" + "".join(m.capitalize() for m in sorted(exclude_modes)) if exclude_modes else ""
        model_name = f"{cls.__name__}{inc_suffix}{exc_suffix}Slice"

        Dynamic = (
            create_model(model_name, __base__=PydBaseModel, **fields_for_dynamic)
            if fields_for_dynamic
            else create_model(f"{model_name}Empty")
        )
        cfg = cls.model_config.copy()
        cfg["from_attributes"] = True
        Dynamic.model_config = ConfigDict(**cfg)
        Dynamic.__module__ = cls.__module__

        cls._slice_cache[cache_key] = Dynamic
        return Dynamic

    # ----- Context-aware schema (LLM bias in LC structured-output stacks) -----
    @classmethod
    def _schema_modes_from_context(cls) -> str | None:
        # 1) ContextVar
        try:
            mode = _CURRENT_MODE.get()
            if mode:
                return mode
        except Exception:
            pass
        # 2) Stack sniff (if enabled)
        if getattr(cls, "prefer_llm_schema_on_stack", True) and getattr(cls, "stack_mode_detection", True):
            try:
                guessed = _infer_mode_from_stack(
                    module_hints=getattr(cls, "stack_mode_module_hints", ("langchain", "langgraph")),
                    module_contains_any=getattr(cls, "stack_mode_module_contains_any", ("output_parsers", "tools", "structured")),
                    function_hints=getattr(cls, "stack_mode_function_hints", ("with_structured_output", "as_structured_output", "parse_result", "parse", "bind_tools", "tool")),
                    max_depth=getattr(cls, "stack_mode_max_depth", 40),
                )
                if guessed:
                    return "llm"
            except Exception:
                pass
        return None

    @classmethod
    def model_json_schema(cls, *args, **kwargs) -> dict:
        """
        If called on the base class (not an explicit slice), we may prefer 'llm' schema
        in LC structured-output stacks. Normalize 'title' back to base class name so
        tool/function mappers find the provided class.
        """
        modes = cls._schema_modes_from_context()
        if modes:
            Slice = cls.__class_getitem__(modes if isinstance(modes, str) else tuple(modes))
            schema = Slice.model_json_schema(*args, **kwargs)
            schema["title"] = cls.__name__
            if "$id" in schema:
                schema["$id"] = cls.__name__
            return schema
        return super().model_json_schema(*args, **kwargs)

    # ----- Runtime dumping -----
    def model_dump(
        self,
        *,
        field_mode: Iterable[str] | str | None = None,
        field_mode_exclude: Iterable[str] | str | None = None,
        dump_format: Literal["python", "json"] = "python",
        node_mode: bool = False,
        context: dict | None = None,   # <-- NEW: accept inbound context
        **kwargs,
    ) -> dict[str, Any]:
        # Back-compat alias
        if field_mode_exclude is None and "field_exclude" in kwargs:
            field_mode_exclude = kwargs.pop("field_exclude")

        if dump_format not in {"python", "json"}:
            raise ValueError("dump_format must be 'python' or 'json'")

        # ---- (A) Resolve mode: explicit > ContextVar > inbound context > stack ----
        if field_mode is None:
            # ContextVar
            try:
                ctx_mode = _CURRENT_MODE.get()
            except Exception:
                ctx_mode = None

            if ctx_mode:
                field_mode = ctx_mode
            else:
                # inbound context from a parent dump
                if isinstance(context, dict):
                    token = context.get("hasql_field_mode")
                    if isinstance(token, str) and token:
                        field_mode = token
                # last resort: stack sniff
                if field_mode is None and getattr(self.__class__, "stack_mode_detection", True):
                    guessed = _infer_mode_from_stack(
                        module_hints=getattr(self.__class__, "stack_mode_module_hints", ("langchain","langgraph")),
                        module_contains_any=getattr(self.__class__, "stack_mode_module_contains_any", ("output_parsers","tools","structured")),
                        function_hints=getattr(self.__class__, "stack_mode_function_hints", ("with_structured_output","as_structured_output","parse_result","parse","bind_tools","tool")),
                        max_depth=getattr(self.__class__, "stack_mode_max_depth", 40),
                    )
                    if guessed:
                        field_mode = guessed

        # If exclude modes not explicitly provided, read from inbound context
        if field_mode_exclude is None and isinstance(context, dict):
            exc_token = context.get("hasql_field_mode_exclude")
            if isinstance(exc_token, str) and exc_token:
                field_mode_exclude = exc_token.split(",")

        # ---- (B) Build inc/exc sets (explicit or defaults) ----
        inc: set[str] = set()
        exc: set[str] = set()

        if field_mode:
            if isinstance(field_mode, str):
                tok = field_mode.strip().lower()
                inc = set(self.__class__._mode_markers.keys()) if tok == "*" else set(tok.split("+"))
            else:
                inc = {m.lower() for m in field_mode}

        if field_mode_exclude:
            if isinstance(field_mode_exclude, str):
                exc = {field_mode_exclude.lower()}
            else:
                exc = {m.lower() for m in field_mode_exclude}

        applied_defaults = False
        if not inc and not exc:
            inc = set(getattr(self.__class__, "default_include_modes", set()))
            exc = set(getattr(self.__class__, "default_exclude_modes", set()))
            applied_defaults = True

        if applied_defaults:
            overlap = inc & exc
            if overlap:
                pol = getattr(self.__class__, "default_conflict_policy", "warn")
                msg = _default_overlap_msg(self.__class__, overlap)
                if pol == "error":
                    raise ValueError(msg)
                if pol == "warn" and self.__class__ not in self.__class__._conflict_warned_classes:
                    warnings.warn(msg, stacklevel=2)
                    self.__class__._conflict_warned_classes.add(self.__class__)

        # ---- (C) Compute include/exclude-by-name from modes (your existing logic) ----
        include = kwargs.pop("include", None)
        exclude = kwargs.pop("exclude", None)
        current_include = None if include is None else (set(include) if not isinstance(include, dict) else include)
        current_exclude = None if exclude is None else (set(exclude) if not isinstance(exclude, dict) else exclude)

        if inc or exc:
            hints = get_type_hints(self.__class__, include_extras=True)
            wanted: set[str] = set()
            for fname, hint in hints.items():
                f_modes = self.__class__._field_modes_from_hint(hint)
                pos = bool(f_modes & inc) if inc else False
                unmarked_ok = (not f_modes) and bool(inc & self.__class__.include_unmarked_for_modes)
                excluded_by_global = bool(f_modes & exc)  # (keep your “exclusion wins” semantics)
                excluded_by_field = self.__class__._has_exclude_for_modes(hint, inc if inc else exc)
                if (pos or unmarked_ok) and not excluded_by_global and not excluded_by_field:
                    wanted.add(fname)

            if current_include is None:
                current_include = wanted
            elif isinstance(current_include, set):
                current_include.intersection_update(wanted)
            else:
                current_include = {k: v for k, v in current_include.items() if k in wanted}

        # ---- (D) node_mode: drop nested models (unchanged) ----
        node_drops = set()
        if node_mode:
            fields_to_check = (
                current_include if isinstance(current_include, set)
                else current_include.keys() if isinstance(current_include, dict)
                else self.__class__.model_fields.keys()
            )
            pred = getattr(self.__class__, "node_model_predicate", lambda v: isinstance(v, PydBaseModel))
            for fname in fields_to_check:
                val = getattr(self, fname, None)
                if pred(val) or (isinstance(val, list) and any(pred(it) for it in val)):
                    node_drops.add(fname)

        if current_exclude is None:
            current_exclude = node_drops
        elif isinstance(current_exclude, set):
            current_exclude.update(node_drops)
        else:
            for f in node_drops:
                current_exclude[f] = True

        # ---- (E) NEW: propagate effective modes to children via `context` ----
        out_ctx = dict(context) if isinstance(context, dict) else {}
        # mode include token
        if inc:
            out_ctx["hasql_field_mode"] = "+".join(sorted(inc))
        elif isinstance(field_mode, str) and field_mode:  # e.g., "*" or from stack
            out_ctx["hasql_field_mode"] = field_mode
        # mode exclude token
        if exc:
            out_ctx["hasql_field_mode_exclude"] = ",".join(sorted(exc))

        return super().model_dump(
            mode=dump_format,
            include=current_include,
            exclude=current_exclude,
            context=out_ctx,          # <-- CHILDREN SEE THE SAME MODES
            **kwargs,
        )

    def model_dump_json(self, **kwargs) -> str:
        data = self.model_dump(dump_format="json", **kwargs)
        return json.dumps(data, ensure_ascii=False, indent=kwargs.get("indent", 2))


# ----------------------------
# DEMOS (non-pytest smoke tests)
# ----------------------------
if __name__ == "__main__":
    from typing import Optional, Union, Annotated

    print("== Demo: base modes & defaults ==")
    class MixinModel(ModeSlicingMixin, PydBaseModel):
        pass
    class Alien(MixinModel):
        # Defaults when no explicit modes are passed
        default_include_modes = {"dto", "frontend"}
        default_exclude_modes = {"llm"}
        include_unmarked_for_modes = {"dto", "frontend", "backend"}  # llm sees no unmarked by default

        # Use marker classes directly in annotations
        secret: Annotated[str, BackendField(), ExcludeMode("llm")] = Field(..., description="secret-key")
        weapon: Annotated[int, DtoField(), FrontendField(), LLMField()] = Field(description="weapon code")
        message: Annotated[str, DtoField()] = Field(description="external message")
        misc: str = Field(description="unmarked general field")

    x = Alien(secret="S3CR3T", weapon=7, message="hi", misc="note")

    print("Default dump (no modes) -> dto+frontend minus llm:")
    print(x.model_dump())  # {'message': 'hi', 'misc': 'note'}

    print("LLM dump explicitly requested (defaults ignored):")
    print(x.model_dump(field_mode="llm"))  # {'weapon': 7}

    print("Backend dump includes unmarked:")
    print(x.model_dump(field_mode="backend"))  # {'secret': 'S3CR3T', 'misc': 'note'}

    print("DTO dump (includes unmarked per class rule):")
    print(x.model_dump(field_mode="dto"))  # {'weapon': 7, 'message': 'hi', 'misc': 'note'}

    # ---- Schema checks for slices ----
    AlienLLM = Alien["llm"]
    print("Alien['llm'] schema properties:", list(AlienLLM.model_json_schema().get("properties", {}).keys()))
    # ['weapon']
    class Alien2(MixinModel):
        # Defaults when no explicit modes are passed
        default_include_modes = {"llm"}
        include_unmarked_for_modes = {"dto", "frontend", "backend", "llm"}  # llm sees no unmarked by default

        # Use marker classes directly in annotations
        secret: Annotated[str, BackendField(), ExcludeMode("llm")] = Field(..., description="secret-key")
        weapon: Annotated[int, DtoField(), FrontendField(), LLMField()] = Field(description="weapon code")
        message: Annotated[str, DtoField(), LLMField()] = Field(description="external message")
        misc: str = Field(description="unmarked general field")
    
    x2 = Alien2(secret="S3CR3T", weapon=7, message="hi", misc="note")
    
    print("default include all but excluded by llm ==")
    print(x2.model_dump(field_mode="llm"))
    
    print("\n== Demo: per-class dynamic mode registration and dynamic Annotated ==")

    class WithAdmin(MixinModel):
        pass

    # Register a new mode on this class branch and capture the marker class
    AdminField = WithAdmin.register_mode("admin")

    class Staff(WithAdmin):
        # Compose markers statically
        name: Annotated[str, DtoField(), FrontendField()] = Field()
        # Or dynamically build Annotated from mode names
        clearance: WithAdmin.build_annotated(int, "admin", "backend") = Field(default=0)
        note: str = Field(default="unmarked")

    s = Staff(name="Neo", clearance=5, note="ops")

    print("Staff modes:", Staff.modes())  # {'dto','backend','frontend','llm','admin'}

    print("Staff dto dump (includes unmarked by default):")
    print(s.model_dump(field_mode="dto"))  # {'name': 'Neo', 'note': 'unmarked'}

    print("Staff admin slice class properties:")
    print(list(Staff["admin"].model_json_schema().get("properties", {}).keys()))  # ['clearance']

    print("Staff all-but-admin slice:")
    print(list(Staff["*", "-admin"].model_json_schema().get("properties", {}).keys()))
    # everything except admin-marked

    print("\n== Demo: field-level ExcludeMode ==")

    class Secretive(MixinModel):
        default_include_modes = {"dto", "frontend"}
        secret: Annotated[str, BackendField(), ExcludeMode("llm")] = Field(...)
        title: Annotated[str, DtoField()] = Field()
        default_secret: str = Field("dEf@u1+", description = 'default secret')

    sec = Secretive(secret="K", title="T")
    print("Secretive llm dump (secret excluded by ExcludeMode):")
    print(sec.model_dump(field_mode="llm"))  # {}

    print("\n== Demo: defaults conflict warning ==")

    class Overlap(MixinModel):
        default_include_modes = {"dto", "llm"}
        default_exclude_modes = {"llm"}  # conflict -> warn/error once
        a: Annotated[int, DtoField()] = Field(1)
        b: Annotated[int, LLMField()] = Field(2)

    ov = Overlap()
    print("Overlap default dump (llm excluded by default overlap):")
    print(ov.model_dump())  # {'a': 1}

    print("\n== Demo: node_mode (drops SQLModel-valued relationships) ==")

    class NodeChild(MixinModel):
        default_include_modes = {"dto"}
        value: Annotated[int, DtoField()] = Field(1)

    class NodeParent(MixinModel):
        default_include_modes = {"dto"}
        child: Annotated[NodeChild, DtoField()] = Field(default_factory=NodeChild)
        label: Annotated[str, DtoField()] = Field("parent")

    parent = NodeParent()
    print("NodeParent dto dump (with node_mode=True drops relationship 'child'):")
    print(parent.model_dump(field_mode="dto", node_mode=True))  # {'label': 'parent'}

    # ------------------------------------------------------------------
    # Extra checks from your second main (kept & adjusted to avoid errors)
    # ------------------------------------------------------------------

    print("\n--- Alien: default + exclude-only (field_exclude='llm') behaves as default-minus-llm ---")
    print(x.model_dump(field_mode_exclude="llm"))
    # If your Model implements: when inc is empty and exc is set -> include defaults (or all) minus exc

    print("\n--- Alien: backend + dto + frontend with field_exclude='llm' (all-minus-llm) ---")
    print(x.model_dump(field_mode=["backend", "dto", "frontend"], field_mode_exclude="llm"))

    print("\n--- Alien: class default slice (dto+frontend, not llm) ---")
    AlienDefault = Alien.default_slice()
    print(AlienDefault.model_json_schema()["properties"].keys())

    print("\n--- Spy (exclude-mode) extended demo ---")

    class Spy(MixinModel):
        default_include_modes = {"backend"}
        include_unmarked_for_modes = {"backend"}

        codename: Annotated[str, FrontendField()] = Field("X")
        notes: Annotated[str, BackendField(), ExcludeMode("dto", "llm")] = Field("internal")
        public: Annotated[str, DtoField()] = Field("ok")
    class Infiltrated(MixinModel):
        spies : list[Spy]
        
    s2 = Spy()
    
    print("--- Spy: dto (notes excluded by ExcludeMode) ---")
    print(s2.model_dump(field_mode="dto"))  # {'public': 'ok'}

    print("--- Spy: backend (notes included; codename excluded) ---")
    print(s2.model_dump(field_mode="backend"))  # {'notes': 'internal'}

    print("\n--- Mixed: Optional/Union nesting demo ---")
    hq = Infiltrated(spies = [s2])
    print("--- nested model test")
    print(hq.model_dump(field_mode="dto"))
    
    with use_mode("dto") as um:
        print(hq.model_dump())
        pass
    
    class Mixed(MixinModel):
        default_include_modes = {"dto"}
        include_unmarked_for_modes = {"dto"}

        # Nested Annotated inside Optional
        maybe_msg: Annotated[Optional[str], DtoField()] = Field(None)
        # Union with multiple markers (dto or frontend)
        either: Annotated[Union[int, str], DtoField(), FrontendField()] = Field(0)
        # Unmarked (shows up for dto since include_unmarked_for_modes contains dto)
        plain: int = Field(5)

    m = Mixed()
    print("--- Mixed: defaults (dto) includes nested annotations and unmarked ---")
    print(m.model_dump())  # {'maybe_msg': None, 'either': 0, 'plain': 5}

    print("--- Mixed: frontend slice includes 'either' but not unmarked 'plain' (per rule) ---")
    print(m.model_dump(field_mode="frontend"))  # {'either': 0}

    print("\n--- Nesting Annotated aliases demo ---")

    class Nesting_dto_backend(MixinModel):
        # Here we rely on recursive metadata extraction to see Frontend + Backend + Dto
        name: FrontendType[BackendType[DtoType[str]]] = Field("n")

    # Instantiate slices; for LLM (empty slice), don't pass name to avoid validation error
    print(
        Nesting_dto_backend[FrontendField](name="f"),   # ok
        Nesting_dto_backend[BackendField](name="b"),    # ok
        Nesting_dto_backend[LLMField](),                # should be empty (no fields)
    )

    print("\n--- Conflict_default class (warn at class creation) ---")

    class Conflict_default(MixinModel):
        default_include_modes = {"dto"}
        default_exclude_modes = {"dto"}
        name: DtoType[str] = Field("n")
        ui: FrontendType[str] = Field("u")
        sys: BackendType[str] = Field("b")
        vec: LLMType[int] = Field(1)

    # ------------------ Exclude-only class slice ------------------
    class Both(MixinModel):
        name: DtoType[str] = Field("n")
        ui: FrontendType[str] = Field("u")
        sys: BackendType[str] = Field("b")
        vec: LLMType[int] = Field(1)

    print("\n--- Both: class slice with only '-llm' (all but llm) ---")
    AllButLLM = Both["dto", 'backend',"-llm"]
    print(AllButLLM.model_json_schema()["properties"].keys())  # dict_keys(['name','ui','sys'])

    b = Both()
    print("\n--- Both: runtime dump with field_exclude={'llm'} (all but llm) ---")
    print(b.model_dump(field_mode_exclude={"llm"}))

    # ------------------ Defaults conflict warnings (runtime) ------------------
    class Conflicting(MixinModel):
        default_include_modes = {"dto", "llm"}
        default_exclude_modes = {"llm"}  # overlap on 'llm'
        default_conflict_policy = "warn"  # set to 'error' to raise

        a: DtoType[int] = Field(1)
        z: LLMType[int] = Field(9)

    c = Conflicting()
    print("\n--- Conflicting: defaults applied (will warn once; excludes win for llm) ---")
    print(c.model_dump())

    print("\nAll demos complete.")