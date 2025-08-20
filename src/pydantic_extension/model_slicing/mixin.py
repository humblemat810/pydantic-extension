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

from pydantic import BaseModel as PydBaseModel, ValidationInfo, model_validator
from pydantic import Field, ConfigDict, create_model
from pydantic import field_serializer, SerializationInfo


"""
Mode-based field slicing for Pydantic v2+ (and SQLModel via a shim or your own base).

What it does
------------
- Mark fields with modes using typing.Annotated metadata (DtoField(), FrontendField(), BackendField(), LLMField()).
- Create dynamic slice classes:  MyModel["dto"] • MyModel["dto","frontend"] • MyModel["*","-llm"] • MyModel[NotMode("llm")].
- Dump instances with the same logic: obj.model_dump(field_mode="dto", field_mode_exclude="llm", node_mode=True).
- Per-class dynamic mode registry:
    * MyModel.register_mode("admin") → returns a new marker class AdminField
    * MyModel.build_annotated(str, "dto", "admin") → Annotated[str, DtoField(), AdminField()]
- Configurable defaults per subclass:
    * default_include_modes, default_exclude_modes
    * include_unmarked_for_modes (defaults to {"dto","frontend","backend"}; "llm" usually omitted)
- Conflict policy: default_conflict_policy in {"ignore","warn","error"} warns/errors on default include/exclude overlap.
- Pydantic v2/v3-friendly: only uses class-level .model_fields and create_model; no instance-level deprecated access.
- Nested dumps: the resolved mode is propagated to children via Pydantic’s `context` dict.

Notes
-----
- For OpenAPI / LangChain: pass a SLICE class (e.g., MyModel["dto"]) when you need a frozen schema.
- If you call model_json_schema() on the base class from a structured-output LC stack, we’ll prefer the 'llm'
  slice shape but normalize the schema "title" back to the base class name so tool/function mapping remains stable.
"""
# ----------------------------
# Context keys (generic + overridable)
# ----------------------------
CONTEXT_MODE_KEY = "modeslice:mode"
CONTEXT_EXCLUDE_KEY = "modeslice:exclude"

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
# ContextVar + stack detection
# ----------------------------

_CURRENT_MODE: ContextVar[str | None] = ContextVar("_modeslice_current_mode", default=None)

_VALIDATION_REDIRECT_GUARD: ContextVar[bool] = ContextVar(
    "modeslice_validation_redirect_guard", default=False
)

_MS_CONTEXT: ContextVar[dict] = ContextVar("_modeslice_context", default={})

@contextmanager
def use_mode(mode: str | None):
    """Context manager to set default field_mode within a block."""
    token = _CURRENT_MODE.set(mode)
    try:
        yield
    finally:
        _CURRENT_MODE.reset(token)
from typing import ForwardRef, List, Set, Tuple, Dict

def _deep_slice_annotation(
    ann: Any,
    include_modes: set[str],
    exclude_modes: set[str],
) -> Any:
    """Return an annotation where any ModeSlicingMixin subclass inside is replaced
    by its corresponding slice, recursively through Optional/Union/Annotated and containers.
    """
    origin = get_origin(ann)

    # Deal with Annotated[T, ...] -> slice T, keep metadata
    if origin is Annotated:
        base, *meta = get_args(ann)
        sliced = _deep_slice_annotation(base, include_modes, exclude_modes)
        return Annotated[sliced, *meta]

    # Optional/Union recursion
    if origin is not None and origin in (Union, getattr(__import__("typing"), "UnionType", None)):
        return Union[tuple(_deep_slice_annotation(a, include_modes, exclude_modes) for a in get_args(ann))]  # type: ignore

    # Containers
    if origin in (list, List, set, Set, tuple, Tuple):
        args = tuple(_deep_slice_annotation(a, include_modes, exclude_modes) for a in get_args(ann))
        return origin[args]  # type: ignore
    if origin in (dict, Dict):
        k, v = get_args(ann)
        return dict[_deep_slice_annotation(k, include_modes, exclude_modes),  # type: ignore
                    _deep_slice_annotation(v, include_modes, exclude_modes)]

    # ForwardRef: keep as-is (or resolve if you want)
    if isinstance(ann, ForwardRef):
        return ann

    # Bare class: if it’s a ModeSlicingMixin subclass, slice it
    try:
        if isinstance(ann, type) and issubclass(ann, ModeSlicingMixin):
            # build the same include/exclude spec this parent is using
            if include_modes or exclude_modes:
                spec: list[Any] = sorted(include_modes)
                if exclude_modes:
                    spec.append(NotMode(*sorted(exclude_modes)))
                return ann.__class_getitem__(tuple(spec))
            else:
                return ann  # nothing to do
    except TypeError:
        pass

    return ann
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
def print_pydantic_like_schema(model_cls):
    print(f"class {model_cls.__name__}(BaseModel):")
    for name, field in model_cls.model_fields.items():
        ann = field.annotation.__name__ if hasattr(field.annotation, "__name__") else field.annotation
        default = "..." if field.is_required() else repr(field.default)
        print(f"    {name}: {ann} = {default}")
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
      - node_mode uses a predicate so it works without SQLModel; default treats BaseModel as a node.
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

    # Relationship detection for node_mode; override per-class if needed
    node_model_predicate: ClassVar[Callable] = staticmethod(lambda v: isinstance(v, PydBaseModel))

    # Context key names (generic & overridable)
    context_mode_key: ClassVar[str] = CONTEXT_MODE_KEY
    context_exclude_key: ClassVar[str] = CONTEXT_EXCLUDE_KEY

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

    @classmethod
    def _ctx_mode_key(cls) -> str:
        return getattr(cls, "context_mode_key", CONTEXT_MODE_KEY)

    @classmethod
    def _ctx_exclude_key(cls) -> str:
        return getattr(cls, "context_exclude_key", CONTEXT_EXCLUDE_KEY)

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
            original = cls.model_fields.get(fname)  # class-level access
            if not original:
                continue

            f_modes = cls._field_modes_from_hint(hint)
            pos = bool(f_modes & include_modes)
            unmarked_ok = (not f_modes) and bool(include_modes & cls.include_unmarked_for_modes)
            # exclusion wins
            excluded_by_global = bool(f_modes & exclude_modes)
            excluded_by_field_marker = cls._has_exclude_for_modes(hint, include_modes if include_modes else exclude_modes)

            if (pos or unmarked_ok) and not excluded_by_global and not excluded_by_field_marker:
                info_copy = copy.deepcopy(original)
                sliced_ann = _deep_slice_annotation(info_copy.annotation, include_modes, exclude_modes)
                try:
                    info_copy.annotation = sliced_ann
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
        Dynamic.__module__ = cls.__module__
        Dynamic.__modeslice_base__ = cls                      # <— add
        Dynamic.__modeslice_modes__ = (include_modes, exclude_modes)  # <— add (optional but handy)
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

    # ----- Runtime dumping (propagates mode to children via `context`) -----
    def model_dump(
        self,
        *,
        field_mode: Iterable[str] | str | None = None,
        field_mode_exclude: Iterable[str] | str | None = None,
        dump_format: Literal["python", "json"] = "python",
        node_mode: bool = False,
        context: dict | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        # Back-compat alias
        if field_mode_exclude is None and "field_exclude" in kwargs:
            field_mode_exclude = kwargs.pop("field_exclude")

        if dump_format not in {"python", "json"}:
            raise ValueError("dump_format must be 'python' or 'json'")

        # ---- (A) Resolve mode: explicit > ContextVar > inbound context > stack ----
        if field_mode is None:
            try:
                ctx_mode = _CURRENT_MODE.get()
            except Exception:
                ctx_mode = None

            if ctx_mode:
                field_mode = ctx_mode
            else:
                if isinstance(context, dict):
                    token = context.get(self.__class__._ctx_mode_key())
                    if isinstance(token, str) and token:
                        field_mode = token
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
            exc_token = context.get(self.__class__._ctx_exclude_key())
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

        # ---- (C) Compute include/exclude-by-name from modes ----
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
                excluded_by_global = bool(f_modes & exc)  # exclusion wins
                excluded_by_field = self.__class__._has_exclude_for_modes(hint, inc if inc else exc)
                if (pos or unmarked_ok) and not excluded_by_global and not excluded_by_field:
                    wanted.add(fname)

            if current_include is None:
                current_include = wanted
            elif isinstance(current_include, set):
                current_include.intersection_update(wanted)
            else:
                current_include = {k: v for k, v in current_include.items() if k in wanted}

        # ---- (D) node_mode: drop nested models ----
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

        # ---- (E) Propagate effective modes to children via `context` ----
        out_ctx = dict(context) if isinstance(context, dict) else {}
        # include token
        if inc:
            out_ctx[self.__class__._ctx_mode_key()] = "+".join(sorted(inc))
        elif isinstance(field_mode, str) and field_mode:
            out_ctx[self.__class__._ctx_mode_key()] = field_mode
        # exclude token
        if exc:
            out_ctx[self.__class__._ctx_exclude_key()] = ",".join(sorted(exc))

        return super().model_dump(
            mode=dump_format,
            include=current_include,
            exclude=current_exclude,
            context=out_ctx,
            **kwargs,
        )

    def model_dump_json(self, **kwargs) -> str:
        data = self.model_dump(dump_format="json", **kwargs)
        return json.dumps(data, ensure_ascii=False, indent=kwargs.get("indent", 2))

    # ------------------------------------------------------------------
    # WILDCARD FIELD SERIALIZER
    # Ensures nested ModeSlicingMixin values (and containers of them)
    # are serialized with the SAME effective mode as the parent.
    # ------------------------------------------------------------------
    @field_serializer("*", mode="wrap")
    def _modeslice_all_fields(self, value, handler, info: SerializationInfo):
        # 1) Figure out the mode/exclude to propagate to children
        ctx_in = info.context or {}
        mode_token = ctx_in.get(self.__class__._ctx_mode_key()) or None

        if mode_token is None:
            # Try ContextVar
            try:
                cv = _CURRENT_MODE.get()
            except Exception:
                cv = None
            if cv:
                mode_token = cv
            else:
                # Default to this class's defaults (include wins for token)
                inc = getattr(self.__class__, "default_include_modes", set())
                if inc:
                    mode_token = "+".join(sorted(inc))

        excl_token = ctx_in.get(self.__class__._ctx_exclude_key()) or None
        if excl_token is None:
            exc = getattr(self.__class__, "default_exclude_modes", set())
            if exc:
                excl_token = ",".join(sorted(exc))

        # 2) Build outbound context (passed to children)
        ctx_out = dict(ctx_in)
        if mode_token:
            ctx_out[self.__class__._ctx_mode_key()] = mode_token
        if excl_token:
            ctx_out[self.__class__._ctx_exclude_key()] = excl_token

        # 3) Match Pydantic's target ("json" vs "python")
        dump_format = "json" if getattr(info, "mode", None) == "json" else "python"

        # 4) Helpers to serialize nested values
        def dump_child(obj):
            # Only force custom dump for *our* mixin subclasses
            if isinstance(obj, ModeSlicingMixin):
                return obj.model_dump(dump_format=dump_format, context=ctx_out)
            # Let Pydantic handle other BaseModels (keeps their custom serializers)
            if isinstance(obj, PydBaseModel):
                return handler(obj)
            return handler(obj)

        def transform_container(obj):
            # Lists / tuples: replace mixin items, then let handler finish (e.g. datetimes)
            if isinstance(obj, list):
                changed = False
                tmp = []
                for it in obj:
                    if isinstance(it, ModeSlicingMixin):
                        tmp.append(it.model_dump(dump_format=dump_format, context=ctx_out))
                        changed = True
                    else:
                        tmp.append(it)
                return handler(tmp) if changed else handler(obj)
            if isinstance(obj, tuple):
                changed = False
                tmp = []
                for it in obj:
                    if isinstance(it, ModeSlicingMixin):
                        tmp.append(it.model_dump(dump_format=dump_format, context=ctx_out))
                        changed = True
                    else:
                        tmp.append(it)
                return handler(tuple(tmp)) if changed else handler(obj)
            if isinstance(obj, dict):
                changed = False
                tmp = {}
                for k, v in obj.items():
                    if isinstance(v, ModeSlicingMixin):
                        tmp[k] = v.model_dump(dump_format=dump_format, context=ctx_out)
                        changed = True
                    else:
                        tmp[k] = v
                return handler(tmp) if changed else handler(obj)
            return None  # not a container

        # 5) Apply logic: single model, container, or other
        if isinstance(value, ModeSlicingMixin):
            return dump_child(value)

        maybe = transform_container(value)
        if maybe is not None:
            return maybe

        # Scalars / all other cases → default behavior
        return handler(value)
    prefer_llm_validation_on_stack: ClassVar[bool] = True  # turn off if you don’t want this behavior

    @classmethod
    def _should_redirect_to_llm_slice_for_validation(cls, context: dict | None) -> bool:
        # If context explicitly sets a mode, don't redirect
        if isinstance(context, dict):
            tok = context.get(cls._ctx_mode_key())
            if isinstance(tok, str) and tok:
                return False

        # If a ContextVar mode is set, respect it (no redirect)
        try:
            cv = _CURRENT_MODE.get()
        except Exception:
            cv = None
        if cv == "llm":
            return True
        elif cv:
            return False
        # cv is None here
        if not getattr(cls, "prefer_llm_validation_on_stack", True):
            return False

        # Otherwise sniff the stack: inside LC structured output?
        guessed = _infer_mode_from_stack(
            module_hints=getattr(cls, "stack_mode_module_hints", ("langchain","langgraph")),
            module_contains_any=getattr(cls, "stack_mode_module_contains_any", ("output_parsers","tools","structured")),
            function_hints=getattr(cls, "stack_mode_function_hints", ("with_structured_output","as_structured_output","parse_result","parse","bind_tools","tool")),
            max_depth=getattr(cls, "stack_mode_max_depth", 40),
        )
        return guessed == "llm"

    @model_validator(mode="wrap")
    @classmethod
    def _redirect_ctor_to_llm_slice(cls, input_value, handler, info: ValidationInfo):
        # prevent infinite recursion
        if _VALIDATION_REDIRECT_GUARD.get():
            return handler(input_value)
        cv_sniff_env_result = cls._should_redirect_to_llm_slice_for_validation(info.context)
        if cv_sniff_env_result:
            Slice = cls.__class_getitem__("llm")
            try:
                _VALIDATION_REDIRECT_GUARD.set(True)
                # Validate against the LLM slice and RETURN THE SLICE INSTANCE
                # This means callers receive `Slice` objects, not `cls`.
                return Slice.model_validate(input_value, context=info.context)
            finally:
                _VALIDATION_REDIRECT_GUARD.set(False)

        # normal path → construct `cls`
        return handler(input_value)
    
    
#==================================
# 
# Sliced to Base implementation
# 
#====================    
# ---------- helpers to detect slice/base ----------
def _is_modeslice_model_type(t: Any) -> bool:
    return isinstance(t, type) and issubclass(t, PydBaseModel) and hasattr(t, "__modeslice_base__")

def _modeslice_base_of(t: Any) -> Any:
    return getattr(t, "__modeslice_base__", None)

def _is_pydantic_model_type(t: Any) -> bool:
    return isinstance(t, type) and issubclass(t, PydBaseModel)

# ---------- resolve the intended target model type from an annotation ----------
def _resolve_target_model_type(field_type: Any) -> Any:
    origin = get_origin(field_type)
    if origin is Union or origin is getattr(__import__("typing"), "UnionType", None):
        for arg in get_args(field_type):
            base = _resolve_target_model_type(arg)
            if _is_pydantic_model_type(base):
                return base
        return field_type
    if origin is Annotated:
        base, *_ = get_args(field_type)
        return _resolve_target_model_type(base)
    if _is_modeslice_model_type(field_type):
        return _modeslice_base_of(field_type)
    return field_type

def _coerce_value_to_target(value: Any, target_type: Any) -> Any:
    origin = get_origin(target_type)
    args = get_args(target_type)

    # Strip Annotated
    if origin is Annotated:
        target_type = args[0]
        origin = get_origin(target_type)
        args = get_args(target_type)

    # Containers
    if origin in (list, tuple, set):
        inner = args[0] if args else Any
        seq = [] if value is None else list(value)
        casted = [_coerce_value_to_target(v, inner) for v in seq]
        if origin is list:
            return casted
        if origin is tuple:
            return tuple(casted)
        if origin is set:
            return set(casted)

    if origin is dict:
        kt, vt = args if len(args) == 2 else (Any, Any)
        if value is None:
            return None
        return {
            _coerce_value_to_target(k, kt): _coerce_value_to_target(v, vt)
            for k, v in value.items()
        }

    # Union – best effort
    if origin in (Union, getattr(__import__("typing"), "UnionType", None)):
        for arm in args:
            try:
                return _coerce_value_to_target(value, arm)
            except Exception:
                continue
        return value

    # If the target type is a modeslice class, resolve to its base
    if _is_modeslice_model_type(target_type):
        target_type = _modeslice_base_of(target_type)

    # If value is a Pydantic model, convert to plain dict (do NOT validate to the target here)
    if _is_pydantic_model_type(target_type):
        if isinstance(value, PydBaseModel):
            return value.model_dump()
        # If it's already a dict-like for that model, keep as-is; validation happens at the top level
        return value

    # Scalars / everything else
    return value

# ---------- JSONPath-lite setters: a.b.c, arr[0], props["k"] ----------
def _tokenize_path(path: str):
    tokens, buf, i = [], "", 0
    in_brackets = False
    in_quotes = False
    quote_char = ""
    while i < len(path):
        ch = path[i]
        if in_brackets:
            if in_quotes:
                if ch == quote_char:
                    in_quotes = False
                else:
                    buf += ch
            else:
                if ch in ("'", '"'):
                    in_quotes = True; quote_char = ch
                elif ch == ']':
                    tokens.append(("index", buf))
                    buf = ""; in_brackets = False
                else:
                    buf += ch
        else:
            if ch == '.':
                if buf:
                    tokens.append(("attr", buf)); buf = ""
            elif ch == '[':
                if buf:
                    tokens.append(("attr", buf)); buf = ""
                in_brackets = True
            else:
                buf += ch
        i += 1
    if buf:
        tokens.append(("attr", buf))
    return tokens

def _ensure_container(parent: Any, key: Any):
    if isinstance(key, int):
        # ensure list long enough
        while len(parent) <= key:
            parent.append({})
    else:
        if key not in parent:
            parent[key] = {}

def _get_or_create_root(data: dict) -> dict:
    return data

def set_by_path(data: dict, path: str, value: Any, *, overwrite: bool, value_fn_ok: bool = True):
    tokens = _tokenize_path(path)
    cur = data
    parent_stack = []  # [(parent, key_or_index)]

    def _ensure_dict_here():
        nonlocal cur
        if isinstance(cur, PydBaseModel):
            cur = cur.model_dump()
            p, k = parent_stack[-1] if parent_stack else (None, None)
            if p is not None:
                p[k] = cur
        elif not isinstance(cur, (dict, list)):
            # replace unknown with dict
            replacement = {}
            p, k = parent_stack[-1] if parent_stack else (None, None)
            if p is not None:
                p[k] = replacement
            cur = replacement

    i = 0
    while i < len(tokens):
        kind, tok = tokens[i]
        last = (i == len(tokens) - 1)

        if kind == "attr":
            key = tok
            _ensure_dict_here()
            if isinstance(cur, list):
                # Cannot use attribute on a list; this indicates the previous
                # step should have navigated into an item. Treat as error-safe dict.
                # Convert the list slot we just came from into dict if needed.
                pass
            if last:
                if not overwrite and key in cur and cur[key] is not None:
                    return
                cur[key] = value(cur.get(key)) if (value_fn_ok and callable(value)) else value
            else:
                if key not in cur or not isinstance(cur[key], (dict, list, PydBaseModel)):
                    cur[key] = {}
                parent_stack.append((cur, key))
                cur = cur[key]

        elif kind == "index":
            # Try integer index; if fail, treat as dict key string
            try:
                idx = int(tok)
                # If current node is a model, turn into dict first
                _ensure_dict_here()
                if isinstance(cur, dict):
                    # We must have arrived here via an attribute just before; turn that slot into a list
                    # parent is the last pushed (cur was already updated)
                    # Make sure current dict is actually the list holder; create list if needed
                    # Find parent and key to replace dict with list (if just created)
                    # If we can't find parent, just create a list in place
                    p, k = parent_stack[-1] if parent_stack else (None, None)
                    if p is not None and isinstance(p, dict) and p.get(k) is cur:
                        if not isinstance(p[k], list):
                            p[k] = []
                        cur = p[k]
                    else:
                        # If we didn't just come from an attr, ensure cur is a list
                        if not isinstance(cur, list):
                            # Replace this node with a list only if we can
                            # Otherwise, create a temporary list and discard (best-effort)
                            tmp = []
                            cur = tmp

                # Now ensure it's a list
                if not isinstance(cur, list):
                    # Create a new list if it's not
                    new_list = []
                    # attach to parent if possible
                    if parent_stack:
                        p, k = parent_stack[-1]
                        p[k] = new_list
                    cur = new_list

                # Ensure length
                while len(cur) <= idx:
                    cur.append({})

                if last:
                    if not overwrite and cur[idx] not in (None, {}, []):
                        return
                    # If the target slot is a model, convert to dict before writing
                    if isinstance(cur[idx], PydBaseModel):
                        cur[idx] = cur[idx].model_dump()
                    cur[idx] = value(cur[idx]) if (value_fn_ok and callable(value)) else value
                else:
                    # Descend into the element; ensure it’s dict-like
                    if isinstance(cur[idx], PydBaseModel):
                        cur[idx] = cur[idx].model_dump()
                    if not isinstance(cur[idx], (dict, list)):
                        cur[idx] = {}
                    parent_stack.append((cur, idx))
                    cur = cur[idx]
            except ValueError:
                # Non-integer inside brackets -> treat as dict key (quoted key)
                key = tok.strip().strip('"').strip("'")
                _ensure_dict_here()
                if last:
                    if not overwrite and key in cur:
                        return
                    cur[key] = value(cur.get(key)) if (value_fn_ok and callable(value)) else value
                else:
                    if key not in cur or not isinstance(cur[key], (dict, list, PydBaseModel)):
                        cur[key] = {}
                    parent_stack.append((cur, key))
                    cur = cur[key]
        i += 1

    
    
def promote_to_base(
    instance: PydBaseModel,
    *,
    defaults: dict[str, Any] | None = None,
    path_defaults: dict[str, Any] | None = None,
    only_if_missing: bool = True,
) -> PydBaseModel:
    """
    Convert a sliced instance (or base) to the *base class* instance.

    - Recursively converts nested slice models to base model types.
    - Fills missing fields from `defaults` (top-level) and `path_defaults` (by path).
    - Pydantic then fills remaining defaults or raises if required fields are still missing.

    `defaults`: { "field": value | callable(current_value)->new_value }
    `path_defaults`: { "a.b[0].c": value | callable(current_value)->new_value }
    `only_if_missing`: if False, overrides existing values too.
    """
    src_cls = instance.__class__
    base_cls = getattr(src_cls, "__modeslice_base__", src_cls)

    # Build a dict targeting base types
    data: dict[str, Any] = {}
    for name, field in base_cls.model_fields.items():
        target_type = _resolve_target_model_type(field.annotation)
        if hasattr(instance, name):
            raw = getattr(instance, name)
            data[name] = _coerce_value_to_target(raw, target_type)
        # else: leave absent, so Pydantic can apply defaults or error

    # Apply top-level defaults
    if defaults:
        for k, v in defaults.items():
            if only_if_missing and k in data and data[k] is not None:
                continue
            data[k] = v(data.get(k)) if callable(v) else v

    # Apply path defaults (nested)
    if path_defaults:
        for path, v in path_defaults.items():
            set_by_path(data, path, v, overwrite=not only_if_missing)

    # Validate into the base class (fills field defaults, runs validators)
    return base_cls.model_validate(data)
# ----------------------------
# DEMOS (non-pytest smoke tests)
# ----------------------------
if __name__ == "__main__":
    from typing import Optional, Union
    
    class MixinModel(ModeSlicingMixin, PydBaseModel):
        pass
    class xdemo (MixinModel):
        # Defaults when no explicit modes are passed
        default_include_modes = {"dto", "frontend"}
        default_exclude_modes = {"llm"}
        include_unmarked_for_modes = {"dto", "frontend", "backend"}  # llm sees no unmarked by default

        secret: Annotated[str, BackendField(), ExcludeMode("llm")] = Field(..., description="secret-key")
        format_instruction : LLMType[str] = Field("Hi AI", description="secret-key")
    print(xdemo(secret = '1234').model_dump(field_mode = "llm"))
    print("== Demo: base modes & defaults ==")


    class Spy(MixinModel):
        default_include_modes = {"backend"}
        include_unmarked_for_modes = {"backend"}

        codename: Annotated[str, FrontendField()] = Field("X")
        notes: Annotated[str, BackendField(), ExcludeMode("dto", "llm")] = Field("internal")
        public: Annotated[str, DtoField()] = Field("ok")
        def model_dump(self, *arg, **kwarg):
            return super().model_dump(*arg, **kwarg)
            

    class Infiltrated(MixinModel):
        spies: list[Spy]

    s2 = Spy()
    hq = Infiltrated(spies=[s2])
    print("--- nested model test (backend keeps unmarked list) ---")
    print(hq.model_dump(field_mode="backend"))  # {'spies': [{'notes': 'internal'}]}

    class Alien(MixinModel):
        # Defaults when no explicit modes are passed
        default_include_modes = {"dto", "frontend"}
        default_exclude_modes = {"llm"}
        include_unmarked_for_modes = {"dto", "frontend", "backend"}  # llm sees no unmarked by default

        secret: Annotated[str, BackendField(), ExcludeMode("llm")] = Field(..., description="secret-key")
        weapon: Annotated[int, DtoField(), FrontendField(), LLMField()] = Field(description="weapon code")
        message: Annotated[str, DtoField()] = Field(description="external message")
        misc: str = Field(description="unmarked general field")
        

    x = Alien(secret="S3CR3T", weapon=7, message="hi", misc="note")

    print("Default dump (no modes) -> dto+frontend minus llm:")
    print(x.model_dump())  # {'weapon': 7, 'message': 'hi', 'misc': 'note'}

    print("LLM dump explicitly requested (defaults ignored):")
    print(x.model_dump(field_mode="llm"))  # {'weapon': 7}

    print("Backend dump includes unmarked:")
    print(x.model_dump(field_mode="backend"))  # {'secret': 'S3CR3T', 'misc': 'note'}

    print("DTO dump (includes unmarked per class rule):")
    print(x.model_dump(field_mode="dto"))  # {'weapon': 7, 'message': 'hi', 'misc': 'note'}

    AlienLLM = Alien["llm"]
    print("Alien['llm'] schema properties:", list(AlienLLM.model_json_schema().get("properties", {}).keys()))

    class Alien2(MixinModel):
        default_include_modes = {"llm"}
        include_unmarked_for_modes = {"dto", "frontend", "backend", "llm"}

        secret: Annotated[str, BackendField(), ExcludeMode("llm")] = Field(..., description="secret-key")
        weapon: Annotated[int, DtoField(), FrontendField(), LLMField()] = Field(description="weapon code")
        message: Annotated[str, DtoField(), LLMField()] = Field(description="external message")
        misc: str = Field(description="unmarked general field")

    x2 = Alien2(secret="S3CR3T", weapon=7, message="hi", misc="note")
    print("default include llm (unmarked allowed by class):")
    print(x2.model_dump(field_mode="llm"))

    print("\n== Demo: per-class dynamic mode registration and dynamic Annotated ==")

    class WithAdmin(MixinModel):
        pass

    AdminField = WithAdmin.register_mode("admin")

    class Staff(WithAdmin):
        name: Annotated[str, DtoField(), FrontendField()] = Field()
        clearance: WithAdmin.build_annotated(int, "admin", "backend") = Field(default=0)
        note: str = Field(default="unmarked")

    s = Staff(name="Neo", clearance=5, note="ops")

    print("Staff modes:", Staff.modes())
    print("Staff dto dump (includes unmarked by default):")
    print(s.model_dump(field_mode="dto"))
    print("Staff admin slice class properties:")
    print(list(Staff["admin"].model_json_schema().get("properties", {}).keys()))
    print("Staff all-but-admin slice:")
    print(list(Staff["*", "-admin"].model_json_schema().get("properties", {}).keys()))

    print("\n== Demo: field-level ExcludeMode ==")

    class Secretive(MixinModel):
        default_include_modes = {"dto", "frontend"}
        secret: Annotated[str, BackendField(), ExcludeMode("llm")] = Field(...)
        title: Annotated[str, DtoField()] = Field()
        default_secret: str = Field("dEf@u1+", description='default secret')

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

    print("\n== Demo: node_mode (drops nested models) ==")

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

    print("\n--- Alien: default + exclude-only (field_mode_exclude='llm') behaves as default-minus-llm ---")
    print(x.model_dump(field_mode_exclude="llm"))

    print("\n--- Alien: backend + dto + frontend with field_mode_exclude='llm' (all-minus-llm) ---")
    print(x.model_dump(field_mode=["backend", "dto", "frontend"], field_mode_exclude="llm"))

    print("\n--- Alien: class default slice (dto+frontend, not llm) ---")
    AlienDefault = Alien.default_slice()
    print(AlienDefault.model_json_schema()["properties"].keys())

    print("\n--- Spy (exclude-mode) extended demo ---")
    print("--- Spy: dto (notes excluded by ExcludeMode) ---")
    print(s2.model_dump(field_mode="dto"))  # {'public': 'ok'}
    print("--- Spy: backend (notes included; codename excluded) ---")
    print(s2.model_dump(field_mode="backend"))  # {'notes': 'internal'}

    print("\n--- Mixed: Optional/Union nesting demo ---")
    print("--- nested model test (dto) ---")
    print(hq.model_dump(field_mode="dto"))
    with use_mode("dto"):
        print(hq.model_dump())

    class Mixed(MixinModel):
        default_include_modes = {"dto"}
        include_unmarked_for_modes = {"dto"}

        maybe_msg: Annotated[Optional[str], DtoField()] = Field(None)
        either: Annotated[Union[int, str], DtoField(), FrontendField()] = Field(0)
        plain: int = Field(5)

    m = Mixed()
    print("--- Mixed: defaults (dto) includes nested annotations and unmarked ---")
    print(m.model_dump())
    print("--- Mixed: frontend slice includes 'either' but not unmarked 'plain' ---")
    print(m.model_dump(field_mode="frontend"))

    print("\n--- Nesting Annotated aliases demo ---")
    class Nesting_dto_backend(MixinModel):
        name: FrontendType[BackendType[DtoType[str]]] = Field("n")

    print(
        Nesting_dto_backend[FrontendField](name="f"),
        Nesting_dto_backend[BackendField](name="b"),
        Nesting_dto_backend[LLMField](),  # empty slice
    )

    print("\n--- Conflict_default class (warn at class creation) ---")
    class Conflict_default(MixinModel):
        default_include_modes = {"dto"}
        default_exclude_modes = {"dto"}
        name: DtoType[str] = Field("n")
        ui: FrontendType[str] = Field("u")
        sys: BackendType[str] = Field("b")
        vec: LLMType[int] = Field(1)

    class Both(MixinModel):
        name: DtoType[str] = Field("n")
        ui: FrontendType[str] = Field("u")
        sys: BackendType[str] = Field("b")
        vec: LLMType[int] = Field(1)

    print("\n--- Both: class slice with '-llm' ---")
    AllButLLM = Both["dto", "backend", "-llm"]
    print(AllButLLM.model_json_schema()["properties"].keys())

    b = Both()
    print("\n--- Both: runtime dump with field_mode_exclude={'llm'} (all but llm) ---")
    print(b.model_dump(field_mode_exclude={"llm"}))

    class Conflicting(MixinModel):
        default_include_modes = {"dto", "llm"}
        default_exclude_modes = {"llm"}
        default_conflict_policy = "warn"
        a: DtoType[int] = Field(1)
        z: LLMType[int] = Field(9)

    c = Conflicting()
    print("\n--- Conflicting: defaults applied (warn once; excludes win for llm) ---")
    print(c.model_dump())

    print("\nAll demos complete.")
