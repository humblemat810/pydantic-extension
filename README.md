# Pydantic Model Slicing

A handy Pydantic extension for advanced, mode-based field slicing, designed for seamless integration with FastAPI, LangChain, and other modern Python frameworks.

This library allows you to define different "views" or "slices" of your Pydantic models for various use cases like DTOs, frontend payloads, backend-only fields, or LLM-specific contexts, using simple and declarative annotations.

## Key Features

- **Declarative Field Modes**: Mark model fields with modes like `dto`, `frontend`, `llm`, etc., using `typing.Annotated`.
- **Dynamic Model Slicing**: Generate specialized Pydantic models on-the-fly for specific modes (e.g., `MyModel["dto"]`), ensuring correct OpenAPI/JSON schemas in frameworks like FastAPI.
- **Mode-Aware Data Dumping**: Serialize model instances to dictionaries or JSON, including only the fields relevant to a specified mode (e.g., `instance.model_dump(field_mode="llm")`).
- **Dynamic & Extensible**: Register custom modes at runtime to fit your application's unique needs.
- **Configurable Defaults**: Define class-level defaults for including/excluding modes and handling unmarked fields.
- **Context-Aware Slicing**: Automatically infers modes from the call stack, providing seamless integration with frameworks like LangChain for structured outputs.
- **Broad Compatibility**: Works as a simple mixin for any `pydantic.BaseModel` or `sqlmodel.SQLModel`.

## Installation

```bash
pip install pydantic-model-slicing
```
*(Note: This assumes the package will be published with this name. For now, you can include the source in your project.)*

## Quick Start

Define your Pydantic model by inheriting from `ModeSlicingMixin` and annotate fields with the desired modes.

```python
from typing import Annotated
from pydantic import Field, BaseModel
from model_slicing.mixin import ModeSlicingMixin, DtoField, BackendField, LLMField

class User(ModeSlicingMixin, BaseModel):
    # This field is available in 'dto' and 'llm' modes
    username: Annotated[str, DtoField(), LLMField()]

    # This field is only for internal backend use
    hashed_password: Annotated[str, BackendField()]

    # An unmarked field, included in default modes like 'dto'
    email: str

# --- Create an instance ---
user = User(username="ada", hashed_password="abc...", email="ada@example.com")

# --- Runtime Data Dumping ---

# 1. Dump for a DTO payload
# -> {'username': 'ada', 'email': 'ada@example.com'}
dto_data = user.model_dump(field_mode="dto")
print(dto_data)

# 2. Dump for an LLM context
# -> {'username': 'ada'}
llm_data = user.model_dump(field_mode="llm")
print(llm_data)


# --- Schema Generation for FastAPI ---

from fastapi import FastAPI

app = FastAPI()

# Use the sliced model to generate the correct OpenAPI schema
UserDTO = User["dto"]

@app.post("/users/", response_model=UserDTO)
async def create_user(user: UserDTO):
    return user
```

## Detailed Features

### 1. Declaring Modes

Use `typing.Annotated` to associate one or more modes with a field. The library provides built-in markers:
- `DtoField`
- `FrontendField`
- `BackendField`
- `LLMField`

You can also exclude a field from a specific mode using `ExcludeMode`.

```python
from model_slicing.mixin import ExcludeMode

class Task(ModeSlicingMixin, BaseModel):
    title: Annotated[str, DtoField(), FrontendField()]
    
    # Available in the backend, but specifically excluded from LLM mode
    internal_id: Annotated[str, BackendField(), ExcludeMode("llm")]
```

### 2. Creating Sliced Models (for Schemas)

To generate a Pydantic model with a subset of fields for schema purposes (e.g., FastAPI, documentation), use dictionary-style access on the class:

```python
# A model containing only fields marked with 'dto'
DTOModel = User["dto"]

# A model containing fields from 'dto' OR 'frontend'
APIModel = User["dto", "frontend"]

# A model with all fields EXCEPT those marked 'llm'
SafeModel = User["*", "-llm"]

# A model with 'backend' fields, excluding any also marked 'dto'
InternalModel = User["backend", NotMode("dto")]
```

### 3. Dumping Sliced Data (Runtime)

To serialize an *instance* of your model, use the `model_dump` method with mode arguments:

```python
user_instance = User(...)

# Include fields from 'dto' and 'frontend' modes
api_payload = user_instance.model_dump(field_mode=["dto", "frontend"])

# Include all fields except those in 'backend' mode
public_data = user_instance.model_dump(field_mode_exclude="backend")
```

### 4. Dynamic Mode Registration

You can register new, custom modes on your models to suit your domain.

```python
class Project(ModeSlicingMixin, BaseModel):
    pass

# Register a new 'admin' mode
AdminField = Project.register_mode("admin")

class Project(Project): # Re-declare to use the new mode
    name: str
    budget: Annotated[float, AdminField()]

# Now you can slice and dump using "admin"
AdminProject = Project["admin"]
project_instance = Project(name="Apollo", budget=1000.0)
admin_data = project_instance.model_dump(field_mode="admin") # -> {'budget': 1000.0}
```

### 5. Configuration and Defaults

You can control the default behavior of slicing and dumping by setting class variables on your model:

- `default_include_modes`: A `set` of modes to include when no modes are specified.
- `default_exclude_modes`: A `set` of modes to exclude when no modes are specified.
- `include_unmarked_for_modes`: A `set` of modes that should also include any fields that have no mode markers.
- `default_conflict_policy`: What to do if a mode is in both include and exclude defaults (`"ignore"`, `"warn"`, or `"error"`).

```python
class Document(ModeSlicingMixin, BaseModel):
    # By default, dump 'dto' fields and exclude 'llm' fields
    default_include_modes = {"dto"}
    default_exclude_modes = {"llm"}
    
    # The 'dto' mode will also get any unmarked fields
    include_unmarked_for_modes = {"dto"}

    title: Annotated[str, DtoField()]
    content: str  # Unmarked, so it's part of 'dto'
    embedding: Annotated[list[float], LLMField()]

doc = Document(title="T", content="C", embedding=[...])

# This will apply the defaults: include 'dto', exclude 'llm'
# -> {'title': 'T', 'content': 'C'}
default_dump = doc.model_dump()
```

## License

This project is licensed under the MIT License.
