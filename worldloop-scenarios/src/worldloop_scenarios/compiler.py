"""Spec compiler — ScenarioSpec → ScenarioPackage (S-04).

This is the integration layer that turns a validated :class:`ScenarioSpec`
into a runnable :class:`ScenarioPackage`. The compile pipeline is:

    raw spec (dict / YAML / JSON)
        │
        ▼
    ScenarioSpec.from_dict           (S-01)
        │
        ▼
    JSON Schema validation          (S-02, syntactic)
        │  failure → SchemaValidationError
        ▼
    Static semantic validation      (S-03, 11 checks)
        │  failure → SemanticValidationError
        ▼
    World factory closure           (S-04)
        │  wraps ParameterizedWorld(spec) + reset(seed)
        ▼
    ScenarioPackage                 (frozen, ready to run)

Design rules (per main plan §13.4):
- The compiler is the ONLY public entry point that produces a
  :class:`ScenarioPackage`. Direct construction of
  :class:`ParameterizedWorld` is allowed for testing but not
  recommended for kernel consumers.
- The compiler MUST run schema + semantic validation even if the
  caller claims the spec is already validated — defense in depth.
- The world factory is a pure closure: calling it twice with the
  same seed produces equivalent worlds (same reset state).
- The compiler does NOT run the world — that's the kernel's job.
  It only verifies the spec is sound and produces a factory.
- LLM-generated specs MUST pass through the same compile pipeline;
  there is NO bypass (per main plan §13.6).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from worldloop_scenarios.parameterized_world import ParameterizedWorld
from worldloop_scenarios.scenario_package import (
    COMPILE_SCHEMA_VERSION,
    ScenarioPackage,
)
from worldloop_scenarios.schema_loader import (
    SchemaValidationError,
    validate_against_schema,
)
from worldloop_scenarios.spec import ScenarioSpec
from worldloop_scenarios.validator import (
    ValidationResult,
    validate_semantics,
)

__all__ = [
    "SemanticValidationError",
    "CompileError",
    "compile_spec",
    "compile_dict",
    "compile_file",
    "COMPILE_SCHEMA_VERSION",
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CompileError(RuntimeError):
    """Base class for compiler errors."""


class SemanticValidationError(CompileError):
    """Raised when a spec fails static semantic validation (S-03).

    The spec passed JSON Schema validation (S-02) but one or more
    semantic checks found error-level issues. The full
    :class:`ValidationResult` is attached for debugging.
    """

    def __init__(self, result: ValidationResult) -> None:
        self.result = result
        errors = result.errors
        messages = [
            f"  - [{e.check_id}] {e.check_name}: {e.message}"
            + (f" (at {e.location})" if e.location else "")
            for e in errors
        ]
        super().__init__(
            f"Spec failed semantic validation with {len(errors)} error(s):\n"
            + "\n".join(messages)
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compile_spec(spec: ScenarioSpec) -> ScenarioPackage:
    """Compile a validated :class:`ScenarioSpec` into a :class:`ScenarioPackage`.

    The spec MUST already be a :class:`ScenarioSpec` instance. To compile
    a raw dict / YAML / JSON, use :func:`compile_dict` or :func:`compile_file`.

    The compiler:
    1. Re-runs JSON Schema validation (defense in depth).
    2. Runs static semantic validation (11 checks).
    3. Builds a world factory closure that produces fresh
       :class:`ParameterizedWorld` instances.
    4. Returns a frozen :class:`ScenarioPackage`.

    Args:
        spec: The :class:`ScenarioSpec` to compile.

    Returns:
        A :class:`ScenarioPackage` ready to drive via
        :class:`worldloop_kernel.WorldProtocol`.

    Raises:
        SchemaValidationError: If the spec fails JSON Schema validation.
        SemanticValidationError: If the spec fails semantic validation.
    """
    # Step 1: schema validation (re-run; defense in depth).
    validate_against_schema(spec.to_dict())

    # Step 2: semantic validation.
    sem_result = validate_semantics(spec)
    if not sem_result.is_valid:
        raise SemanticValidationError(sem_result)

    # Step 3: build world factory closure.
    # The factory is a pure callable: world_factory(seed) -> ParameterizedWorld.
    # Each call produces a fresh instance with reset(seed) already applied.
    captured_spec = spec  # closure capture (defensive against caller mutation)

    def world_factory(seed: int) -> ParameterizedWorld:
        world = ParameterizedWorld(captured_spec)
        world.reset(seed)
        return world

    # Step 4: build the package.
    params_hash = spec.world_parameters_hash()
    return ScenarioPackage(
        spec=spec,
        world_factory=world_factory,
        world_parameters_hash=params_hash,
        compile_schema_version=COMPILE_SCHEMA_VERSION,
        compiler_version="0.1.0",
        provenance={
            "compiled_at": datetime.now(timezone.utc).isoformat(),
            "schema_validated": True,
            "semantic_validated": True,
            "n_semantic_errors": len(sem_result.errors),
            "n_semantic_warnings": len(sem_result.warnings),
            "compiler": "worldloop_scenarios.compiler:0.1.0",
        },
    )


def compile_dict(spec_dict: Mapping[str, Any]) -> ScenarioPackage:
    """Compile a raw spec dict (parsed YAML / JSON) into a :class:`ScenarioPackage`.

    This is a convenience wrapper that:
    1. Runs JSON Schema validation on the raw dict FIRST (so missing
       required fields raise :class:`SchemaValidationError`, not KeyError).
    2. Parses the dict into a :class:`ScenarioSpec` via
       :meth:`ScenarioSpec.from_dict`.
    3. Delegates to :func:`compile_spec` (which re-runs schema validation
       + semantic validation — defense in depth).

    Args:
        spec_dict: The parsed spec dict.

    Returns:
        A :class:`ScenarioPackage`.

    Raises:
        SchemaValidationError: If the dict fails JSON Schema validation.
        SemanticValidationError: If the spec fails semantic validation.
    """
    # Validate the raw dict BEFORE parsing — this catches missing required
    # fields with a clean SchemaValidationError rather than a KeyError
    # from ScenarioSpec.from_dict.
    validate_against_schema(spec_dict)
    spec = ScenarioSpec.from_dict(spec_dict)
    return compile_spec(spec)


def compile_file(path: str | Path) -> ScenarioPackage:
    """Compile a YAML or JSON spec file into a :class:`ScenarioPackage`.

    The file format is detected by extension:
    - ``.yaml`` / ``.yml`` → parsed as YAML
    - ``.json`` → parsed as JSON

    Args:
        path: Path to the spec file.

    Returns:
        A :class:`ScenarioPackage`.

    Raises:
        FileNotFoundError: If the file does not exist.
        CompileError: If the file extension is unsupported.
        SchemaValidationError: If the file fails JSON Schema validation.
        SemanticValidationError: If the file fails semantic validation.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Spec file not found: {p}")
    suffix = p.suffix.lower()
    text = p.read_text(encoding="utf-8")
    if suffix in (".yaml", ".yml"):
        data = yaml.safe_load(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise CompileError(
            f"Unsupported spec file extension: {suffix!r} "
            f"(expected .yaml, .yml, or .json)"
        )
    if not isinstance(data, Mapping):
        raise CompileError(
            f"Spec file must parse to a mapping; got {type(data).__name__}"
        )
    return compile_dict(data)
