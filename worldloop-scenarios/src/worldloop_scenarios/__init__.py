"""WorldLoop v2 Phase F (M3): ScenarioSpec v0 — schema-driven world definition.

This package provides:
- :class:`ScenarioSpec` — the v0 spec dataclass (mirrors the YAML 11-section structure).
- :mod:`worldloop_scenarios.schema_loader` — JSON Schema loading and validation.
- :mod:`worldloop_scenarios.validator` — static semantic validator (11 checks).
- :mod:`worldloop_scenarios.compiler` — spec → ScenarioPackage compiler.
- :mod:`worldloop_scenarios.scenario_package` — the compiled output container.
- :mod:`worldloop_scenarios.parameterized_world` — ParameterizedWorld implementing
  ``worldloop_kernel.WorldProtocol``.

Design rules (per main plan §13 and ADR §3):
- The spec is the ONLY way to define a world in M3. No Python codegen.
- The spec is validated by JSON Schema (syntactic) + semantic validator (semantic).
- LLM-generated specs MUST pass the same validation pipeline; no bypass.
- ``ParameterizedWorld`` implements ``worldloop_kernel.WorldProtocol``; the kernel
  treats it as a black box that exposes the 7-method protocol.
"""

from worldloop_scenarios.spec import (
    ScenarioSpec,
    ScenarioMeta,
    TimeSpec,
    SpaceSpec,
    FieldsSpec,
    EntitiesSpec,
    RelationsSpec,
    RegistriesSpec,
    ActionsSpec,
    ExogenousSpec,
    TerminationSpec,
    DataSpec,
)
from worldloop_scenarios.scenario_package import (
    ScenarioPackage,
    COMPILE_SCHEMA_VERSION,
)
from worldloop_scenarios.compiler import (
    compile_spec,
    compile_dict,
    compile_file,
    CompileError,
    SemanticValidationError,
)

__all__ = [
    # Spec (S-01)
    "ScenarioSpec",
    "ScenarioMeta",
    "TimeSpec",
    "SpaceSpec",
    "FieldsSpec",
    "EntitiesSpec",
    "RelationsSpec",
    "RegistriesSpec",
    "ActionsSpec",
    "ExogenousSpec",
    "TerminationSpec",
    "DataSpec",
    # Package + compiler (S-04)
    "ScenarioPackage",
    "COMPILE_SCHEMA_VERSION",
    "compile_spec",
    "compile_dict",
    "compile_file",
    "CompileError",
    "SemanticValidationError",
]

__version__ = "0.1.3"
