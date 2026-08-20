"""ScenarioSpec v0 — the world definition dataclass (S-01).

Mirrors the YAML 11-section structure defined in main plan §13.2:
    scenario / time / space / fields / entities / relations / registries
    / actions / exogenous / termination / data

Design rules:
- The spec is the SINGLE source of truth for a scenario. The compiler turns
  it into a :class:`ParameterizedWorld`; no Python codegen.
- All dataclasses are frozen (immutable). Mutation produces a new spec via
  ``dataclasses.replace``.
- ``to_dict`` / ``from_dict`` round-trip with the YAML representation.
- ``world_parameters_hash`` is computed from the structure + dynamics fields
  only (per main plan §13.3); it is stable across runs and unaffected by
  policy / experiment / data / claim fields.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Mapping


__all__ = [
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
]


# ---------------------------------------------------------------------------
# Section dataclasses (mirror YAML 11 sections)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioMeta:
    """Section 1: scenario metadata."""

    scenario_id: str
    scenario_version: str
    description: str = ""
    author: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class TimeSpec:
    """Section 2: time model."""

    dt: float = 1.0
    max_ticks: int = 100
    #: Whether the world is deterministic given the same seed + actions.
    deterministic: bool = True


@dataclass(frozen=True)
class SpaceSpec:
    """Section 3: space topology.

    Three supported types (per main plan §13.2):
    - ``"discrete"``: integer grid with shape ``shape`` (e.g., (10, 10)).
    - ``"continuous"``: continuous box with bounds ``bounds`` (lower, upper).
    - ``"graph"``: nodes + edges; no spatial embedding.
    """

    type: str  # "discrete" | "continuous" | "graph"
    shape: tuple[int, ...] = field(default_factory=tuple)
    bounds: tuple[tuple[float, float], ...] = field(default_factory=tuple)
    node_ids: tuple[str, ...] = field(default_factory=tuple)
    edges: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    periodic: bool = False


@dataclass(frozen=True)
class FieldsSpec:
    """Section 4: WST-compatible field channels.

    Each channel has a name, shape, dtype, and unit. The world owns the
    channel semantics; the kernel stores / hashes / diffs them.
    """

    channels: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    #: Whether this scenario populates the ``fields`` slot on StateView.
    enabled: bool = False


@dataclass(frozen=True)
class EntitiesSpec:
    """Section 5: entity table schema.

    Defines the column schema for the :class:`EntityTable`. The actual
    entity rows are populated at ``reset`` time.
    """

    columns: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    initial_count: int = 0
    #: Template for spawning new entities at reset (e.g., energy range, position dist).
    spawn_template: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RelationsSpec:
    """Section 6: relation graph schema (WorldGraph exchange format)."""

    edge_types: tuple[str, ...] = field(default_factory=tuple)
    directed: bool = True
    enabled: bool = False


@dataclass(frozen=True)
class RegistriesSpec:
    """Section 7: registry schema (object / concept / tool / artifact).

    ``initial_entries`` declares registry entries populated at ``reset``
    time (M5 v0.2). Each entry is a mapping with at least ``entry_id``,
    ``registry_type``, ``state``; optional ``owner_id``, ``metadata``,
    ``node``. This enables scenarios where COLLECT/DELIVER/REPAIR operate
    on pre-existing stockpiles and facilities without runtime spawn.
    """

    registry_types: tuple[str, ...] = field(default_factory=tuple)
    enabled: bool = False
    initial_entries: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ActionsSpec:
    """Section 8: action definitions.

    Each action has:
    - ``action_type``: stable string (e.g., "move", "forage", "give").
    - ``params_schema``: mapping of param name → type spec.
    - ``effects``: tuple of effect descriptors (what this action writes).
    - ``cost``: energy / resource cost (signed; negative = gain).
    - ``preconditions``: optional list of condition descriptors.
    """

    actions: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    #: Whether the action space is closed (only listed action_types accepted).
    is_closed: bool = True


@dataclass(frozen=True)
class ExogenousSpec:
    """Section 9: exogenous inputs (events applied BEFORE actions each tick)."""

    events: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    seed: int | None = None


@dataclass(frozen=True)
class TerminationSpec:
    """Section 10: termination conditions.

    At least one stop condition is REQUIRED (per main plan §13.5 static check).
    ``max_ticks`` from :class:`TimeSpec` is always an implicit stop.
    """

    stop_conditions: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    #: Whether to truncate (vs. terminate) when max_ticks is reached.
    truncate_at_max: bool = True


@dataclass(frozen=True)
class DataSpec:
    """Section 11: data production hooks (M4 — most fields are placeholder).

    M3 only defines the hook points; actual data production logic is M4.
    """

    policy_hooks: tuple[str, ...] = field(default_factory=tuple)
    coverage_hooks: tuple[str, ...] = field(default_factory=tuple)
    counterfactual_hooks: tuple[str, ...] = field(default_factory=tuple)
    export_hooks: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# ScenarioSpec — top-level container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioSpec:
    """Top-level ScenarioSpec v0 — the SINGLE source of truth for a scenario.

    A spec is accepted ONLY after:
    1. JSON Schema validation (syntactic, S-02).
    2. Static semantic validation (11 checks, S-03).
    3. Successful compilation to a :class:`ScenarioPackage` (S-04).

    The spec is immutable; mutation produces a new spec via
    :func:`dataclasses.replace`. The ``world_parameters_hash`` is stable
    across runs and depends only on structure + dynamics fields.
    """

    scenario: ScenarioMeta
    time: TimeSpec
    space: SpaceSpec
    entities: EntitiesSpec
    actions: ActionsSpec
    termination: TerminationSpec
    fields: FieldsSpec = field(default_factory=FieldsSpec)
    relations: RelationsSpec = field(default_factory=RelationsSpec)
    registries: RegistriesSpec = field(default_factory=RegistriesSpec)
    exogenous: ExogenousSpec = field(default_factory=ExogenousSpec)
    data: DataSpec = field(default_factory=DataSpec)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict suitable for YAML / JSON dumping.

        Tuples are converted to lists for JSON compatibility.
        """

        def _normalize(obj: Any) -> Any:
            if isinstance(obj, tuple):
                return [_normalize(x) for x in obj]
            if isinstance(obj, Mapping):
                return {k: _normalize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_normalize(x) for x in obj]
            return obj

        return {
            "scenario": _normalize(asdict(self.scenario)),
            "time": _normalize(asdict(self.time)),
            "space": _normalize(asdict(self.space)),
            "fields": _normalize(asdict(self.fields)),
            "entities": _normalize(asdict(self.entities)),
            "relations": _normalize(asdict(self.relations)),
            "registries": _normalize(asdict(self.registries)),
            "actions": _normalize(asdict(self.actions)),
            "exogenous": _normalize(asdict(self.exogenous)),
            "termination": _normalize(asdict(self.termination)),
            "data": _normalize(asdict(self.data)),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScenarioSpec":
        """Deserialize from a dict (e.g., parsed YAML).

        Lists are converted back to tuples where the dataclass expects tuples.
        """
        return cls(
            scenario=ScenarioMeta(
                scenario_id=data["scenario"]["scenario_id"],
                scenario_version=data["scenario"]["scenario_version"],
                description=data["scenario"].get("description", ""),
                author=data["scenario"].get("author", ""),
                tags=tuple(data["scenario"].get("tags", [])),
            ),
            time=TimeSpec(
                dt=data["time"].get("dt", 1.0),
                max_ticks=data["time"].get("max_ticks", 100),
                deterministic=data["time"].get("deterministic", True),
            ),
            space=SpaceSpec(
                type=data["space"]["type"],
                shape=tuple(data["space"].get("shape", [])),
                bounds=tuple(
                    tuple(b) for b in data["space"].get("bounds", [])
                ),
                node_ids=tuple(data["space"].get("node_ids", [])),
                edges=tuple(
                    tuple(e) for e in data["space"].get("edges", [])
                ),
                periodic=data["space"].get("periodic", False),
            ),
            fields=FieldsSpec(
                channels=tuple(data.get("fields", {}).get("channels", [])),
                enabled=data.get("fields", {}).get("enabled", False),
            ),
            entities=EntitiesSpec(
                columns=tuple(data["entities"].get("columns", [])),
                initial_count=data["entities"].get("initial_count", 0),
                spawn_template=data["entities"].get("spawn_template", {}),
            ),
            relations=RelationsSpec(
                edge_types=tuple(
                    data.get("relations", {}).get("edge_types", [])
                ),
                directed=data.get("relations", {}).get("directed", True),
                enabled=data.get("relations", {}).get("enabled", False),
            ),
            registries=RegistriesSpec(
                registry_types=tuple(
                    data.get("registries", {}).get("registry_types", [])
                ),
                enabled=data.get("registries", {}).get("enabled", False),
                initial_entries=tuple(
                    data.get("registries", {}).get("initial_entries", [])
                ),
            ),
            actions=ActionsSpec(
                actions=tuple(data["actions"].get("actions", [])),
                is_closed=data["actions"].get("is_closed", True),
            ),
            exogenous=ExogenousSpec(
                events=tuple(data.get("exogenous", {}).get("events", [])),
                seed=data.get("exogenous", {}).get("seed"),
            ),
            termination=TerminationSpec(
                stop_conditions=tuple(
                    data["termination"].get("stop_conditions", [])
                ),
                truncate_at_max=data["termination"].get(
                    "truncate_at_max", True
                ),
            ),
            data=DataSpec(
                policy_hooks=tuple(
                    data.get("data", {}).get("policy_hooks", [])
                ),
                coverage_hooks=tuple(
                    data.get("data", {}).get("coverage_hooks", [])
                ),
                counterfactual_hooks=tuple(
                    data.get("data", {}).get("counterfactual_hooks", [])
                ),
                export_hooks=tuple(
                    data.get("data", {}).get("export_hooks", [])
                ),
            ),
        )

    # ------------------------------------------------------------------
    # Hashing — world_parameters_hash (per main plan §13.3)
    # ------------------------------------------------------------------

    def world_parameters_hash(self) -> str:
        """Stable hash of structure + dynamics fields only.

        Excludes: scenario metadata (author, tags), data hooks (M4 policy).
        Includes: time, space, fields, entities, relations, registries,
        actions, exogenous, termination.

        The hash is SHA-256 of the canonical JSON dump. Two specs with the
        same ``world_parameters_hash`` produce worlds with the same
        structure and dynamics (policy and experiment fields may differ).
        """
        payload = {
            "time": asdict(self.time),
            "space": asdict(self.space),
            "fields": asdict(self.fields),
            "entities": asdict(self.entities),
            "relations": asdict(self.relations),
            "registries": asdict(self.registries),
            "actions": asdict(self.actions),
            "exogenous": asdict(self.exogenous),
            "termination": asdict(self.termination),
        }
        canonical = json.dumps(payload, sort_keys=True, default=_json_default)
        return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _json_default(obj: Any) -> Any:
    """JSON default handler for non-standard types."""
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
