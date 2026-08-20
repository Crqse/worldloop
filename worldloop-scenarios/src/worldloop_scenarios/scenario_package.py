"""ScenarioPackage — the output of the spec compiler (S-04).

A :class:`ScenarioPackage` bundles everything the kernel needs to run a
scenario defined by a :class:`ScenarioSpec`:

- the validated spec (single source of truth)
- a world factory (callable that produces a fresh :class:`ParameterizedWorld`)
- the world parameters hash (for reproducibility checks)
- compile-time metadata (when/where compiled, compiler version)

The package is immutable. The kernel consumes it by calling
``world_factory(seed)`` to get a fresh world instance, then drives it
via :class:`WorldProtocol`.

Design rules (per main plan §13.4):
- The package is the END of the compile pipeline. After compile, no
  further validation is needed — the world is ready to run.
- The world factory MUST be deterministic: calling it twice with the
  same seed produces worlds that produce identical transition records.
- The package does NOT hold a world instance; it holds a factory. This
  allows the kernel to create fresh worlds for replay / branching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from worldloop_scenarios.spec import ScenarioSpec
    from worldloop_scenarios.parameterized_world import ParameterizedWorld

__all__ = [
    "ScenarioPackage",
    "COMPILE_SCHEMA_VERSION",
]

#: Compiler version. Bumped when the compile pipeline output shape
#: changes in a backward-incompatible way.
COMPILE_SCHEMA_VERSION = "0.1.0"

#: Type alias for the world factory callable.
WorldFactory = Callable[[int], "ParameterizedWorld"]


@dataclass(frozen=True)
class ScenarioPackage:
    """The compiled output of a :class:`ScenarioSpec`.

    Attributes
    ----------
    spec:
        The validated :class:`ScenarioSpec` that produced this package.
    world_factory:
        Callable ``world_factory(seed: int) -> ParameterizedWorld``.
        Produces a fresh world instance each call. The world is ready
        to drive via :class:`WorldProtocol`.
    world_parameters_hash:
        Stable hash of the spec's structure + dynamics fields (from
        :meth:`ScenarioSpec.world_parameters_hash`). Used to verify
        that two packages with the same hash produce identical worlds.
    compile_schema_version:
        Compiler version (see :data:`COMPILE_SCHEMA_VERSION`).
    compiler_version:
        Implementation version of the compiler that produced this package.
    provenance:
        Free-form metadata (e.g., ``{"compiled_at": "...", "schema_validated": True}``).
    """

    spec: "ScenarioSpec"
    world_factory: WorldFactory
    world_parameters_hash: str
    compile_schema_version: str = COMPILE_SCHEMA_VERSION
    compiler_version: str = "0.1.0"
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.world_parameters_hash:
            raise ValueError("world_parameters_hash must be non-empty")
        if not self.compile_schema_version:
            raise ValueError("compile_schema_version must be non-empty")
        if not callable(self.world_factory):
            raise ValueError("world_factory must be callable")
