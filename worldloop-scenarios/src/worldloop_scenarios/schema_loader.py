"""JSON Schema loader and validator (S-02).

Loads the bundled ``spec_v0.schema.json`` and validates parsed YAML/JSON
dicts against it. This is the FIRST validation layer (syntactic); the
semantic validator (S-03) runs AFTER this passes.

Design rules:
- The schema is bundled with the package (``schemas/spec_v0.schema.json``);
  users do NOT need to manage schema files.
- Validation errors are raised as :class:`SchemaValidationError` with the
  full jsonschema error list for debugging.
- LLM-generated specs MUST pass this layer; there is no bypass.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Any, Mapping

import jsonschema

__all__ = [
    "SchemaValidationError",
    "load_spec_v0_schema",
    "validate_against_schema",
]


class SchemaValidationError(ValueError):
    """Raised when a spec dict fails JSON Schema validation."""

    def __init__(self, errors: list[jsonschema.ValidationError]):
        self.errors = errors
        messages = [f"  - {e.message} (at {list(e.absolute_path)})" for e in errors]
        super().__init__(
            f"Spec failed JSON Schema validation with {len(errors)} error(s):\n"
            + "\n".join(messages)
        )


def load_spec_v0_schema() -> dict[str, Any]:
    """Load the bundled spec_v0 JSON Schema.

    The schema is shipped as package data under
    ``worldloop_scenarios/schemas/spec_v0.schema.json``.
    """
    schema_path = resources.files("worldloop_scenarios.schemas").joinpath(
        "spec_v0.schema.json"
    )
    return json.loads(schema_path.read_text(encoding="utf-8"))


def validate_against_schema(spec_dict: Mapping[str, Any]) -> None:
    """Validate a spec dict against the spec_v0 schema.

    Raises :class:`SchemaValidationError` on failure.
    """
    schema = load_spec_v0_schema()
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(spec_dict), key=lambda e: list(e.absolute_path))
    if errors:
        raise SchemaValidationError(errors)
