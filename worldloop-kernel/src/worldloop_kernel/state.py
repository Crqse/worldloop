"""State view and component types (K-04).

Defines the minimal set of state types the kernel natively recognizes.
The kernel does NOT implement WST dynamics, WorldGraph updates, or
Registry projection — those remain in each world implementation
(Native five-layer, external adapter, learned simulator).

Design rules (per ADR §3 and main plan §4.2 / §4.4):
- ``entities`` is the only mandatory slot — a world without entities is
  not a world (enforced by :class:`CapabilityProfile`).
- ``fields`` / ``relations`` / ``registries`` / ``population`` / ``events``
  are optional and MUST be reflected in ``capabilities`` and ``missing_mask``.
- :class:`StateView` is the externally observable, exchangeable state.
  It is NOT sufficient to restore a world — restoration requires
  :class:`Checkpoint` from :mod:`worldloop_kernel.transition`.
- ``missing_mask`` MUST NOT be ``True`` for a slot the world declares
  ``capabilities.<slot>=False``. Capability says "world has it";
  missing_mask says "world has it but this record lacks it".
- All state types are immutable (``frozen=True``). Mutation produces a
  new :class:`StateView` via ``apply_delta`` (K-05).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from worldloop_kernel.capability import CapabilityProfile

__all__ = [
    "StateMeta",
    "FieldState",
    "EntityTable",
    "RelationGraph",
    "RegistrySnapshot",
    "PopulationState",
    "EventContext",
    "StateView",
    "StateError",
]

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class StateError(ValueError):
    """Raised when a state component is internally inconsistent."""


# ---------------------------------------------------------------------------
# State meta
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StateMeta:
    """Per-tick metadata for a state observation.

    All fields are immutable strings/ints so :class:`StateMeta` is hashable
    and safe to embed in canonical encoding (K-05).
    """

    scenario_id: str
    run_id: str
    tick: int
    config_hash: str
    #: Reference to the RNG state at this tick (e.g., ``"sha256:..."`` or
    #: an opaque token the world can resolve). ``None`` if the world does
    #: not expose RNG state (e.g., a stateless external adapter).
    rng_state_ref: str | None


# ---------------------------------------------------------------------------
# Fields — WST-compatible slot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldState:
    """WST-compatible field state.

    The kernel does NOT interpret the field contents; it only stores,
    hashes, and diffs them. The world owns the channel semantics.

    Attributes
    ----------
    schema_id:
        Stable identifier for the field schema (channels, shapes, units).
        Two FieldState objects with the same ``schema_id`` are comparable;
        differing schemas are not.
    channels:
        Mapping from channel name to a compact representation. The kernel
        treats values as opaque; the world MAY use ``bytes``, ``tuple``,
        ``Mapping``, or any picklable / JSON-serializable structure.
    units:
        Per-channel unit label (e.g., ``"energy"``, ``"position_xy"``).
        Helps consumers interpret the channel without parsing the value.
    """

    schema_id: str
    channels: Mapping[str, Any]
    units: Mapping[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Entities — minimum required world skeleton
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntityTable:
    """Minimum-required world skeleton.

    Each row is an entity (agent, resource, threat, structure, ...).
    The kernel treats columns as opaque typed values; the world owns
    the column schema and lifecycle.

    Attributes
    ----------
    schema_id:
        Stable identifier for the entity column schema. Two EntityTable
        objects with the same ``schema_id`` are comparable.
    ids:
        Tuple of entity IDs in a stable order. Order matters for hashing.
    columns:
        Mapping from column name to a tuple of values aligned with ``ids``.
        Each column tuple MUST have the same length as ``ids``.
    """

    schema_id: str
    ids: tuple[str | int, ...]
    columns: Mapping[str, tuple[Any, ...]]

    def __post_init__(self) -> None:
        n = len(self.ids)
        for col_name, col_values in self.columns.items():
            if len(col_values) != n:
                raise StateError(
                    f"EntityTable column {col_name!r} has {len(col_values)} "
                    f"values but ids has {n}; they must align."
                )


# ---------------------------------------------------------------------------
# Relations — WorldGraph exchange format
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RelationEdge:
    """A single directed edge in the relation graph."""

    src: str | int
    dst: str | int
    edge_type: str
    weight: float = 1.0
    #: Tick at which the edge was created; ``None`` if the world does not
    #: track edge lifecycles.
    born_at_tick: int | None = None


@dataclass(frozen=True)
class RelationGraph:
    """WorldGraph exchange format (sparse nodes / edges).

    The kernel does NOT interpret edge types or weights; it only stores,
    hashes, and diffs them. The world owns the WorldGraph update logic.

    Attributes
    ----------
    schema_id:
        Stable identifier for the relation schema.
    node_ids:
        Tuple of node IDs in stable order. May differ from entity IDs if
        the world models non-entity nodes (e.g., abstract locations).
    edges:
        Tuple of :class:`RelationEdge` in stable order (sorted by
        (src, dst, edge_type) by convention).
    """

    schema_id: str
    node_ids: tuple[str | int, ...]
    edges: tuple[RelationEdge, ...]


# ---------------------------------------------------------------------------
# Registries — object / concept / tool / artifact stable identities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegistryEntry:
    """A single registry entry (object, concept, tool, artifact, ...)."""

    entry_id: str
    registry_type: str  # e.g., "object", "concept", "tool", "artifact"
    state: str  # e.g., "active", "consumed", "destroyed"
    #: Optional reference to the owning entity (``None`` if unowned).
    owner_id: str | int | None = None
    #: Free-form metadata the world attaches to the entry.
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RegistrySnapshot:
    """Snapshot of all registries at a single tick.

    Attributes
    ----------
    schema_id:
        Stable identifier for the registry schema.
    entries:
        Tuple of :class:`RegistryEntry` in stable order (sorted by
        (registry_type, entry_id) by convention).
    """

    schema_id: str
    entries: tuple[RegistryEntry, ...]


# ---------------------------------------------------------------------------
# Population — birth / death / lineage (optional)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BirthRecord:
    """A single birth event in the population slot."""

    parent_ids: tuple[str | int, ...]
    child_id: str | int
    tick: int


@dataclass(frozen=True)
class DeathRecord:
    """A single death event in the population slot."""

    agent_id: str | int
    tick: int
    #: Stable cause code (e.g., ``"starvation"``, ``"hazard"``,
    #: ``"old_age"``). The kernel does NOT enumerate causes; the world
    #: owns the taxonomy.
    cause: str


@dataclass(frozen=True)
class PopulationState:
    """Birth / death / lineage state for the current tick.

    Only populated when ``capabilities.population=True``.

    Attributes
    ----------
    alive_ids:
        Tuple of currently alive agent IDs in stable order.
    births_this_tick:
        Tuple of births that occurred in the current tick.
    deaths_this_tick:
        Tuple of deaths that occurred in the current tick.
    cumulative_births:
        Running total of births since ``reset``.
    cumulative_deaths:
        Running total of deaths since ``reset``.
    """

    alive_ids: tuple[str | int, ...]
    births_this_tick: tuple[BirthRecord, ...] = field(default_factory=tuple)
    deaths_this_tick: tuple[DeathRecord, ...] = field(default_factory=tuple)
    cumulative_births: int = 0
    cumulative_deaths: int = 0


# ---------------------------------------------------------------------------
# Events — tick-scoped event context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EventRecord:
    """A single tick-scoped event.

    Events are observations the world chooses to surface (e.g.,
    "resource_depleted", "agent_collided", "threat_spawned"). The kernel
    does NOT interpret event kinds; it stores, hashes, and diffs them.
    """

    kind: str
    tick: int
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EventContext:
    """Tick-scoped event context.

    Only populated when ``capabilities.events=True``.
    """

    events: tuple[EventRecord, ...]


# ---------------------------------------------------------------------------
# StateView — the externally observable, exchangeable state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StateView:
    """The externally observable, exchangeable state.

    Per main plan §4.4, :class:`StateView` is NOT sufficient to restore
    a world. Restoration requires :class:`Checkpoint` from
    :mod:`worldloop_kernel.transition`.

    Attributes
    ----------
    meta:
        Per-tick metadata (scenario_id, run_id, tick, config_hash,
        rng_state_ref). Always present.
    fields:
        WST-compatible field state. ``None`` if ``capabilities.fields=False``.
    entities:
        Minimum-required world skeleton. Always present (possibly empty).
    relations:
        WorldGraph exchange format. ``None`` if ``capabilities.relations=False``.
    registries:
        Object / concept / tool / artifact snapshot. ``None`` if
        ``capabilities.registries=False``.
    population:
        Birth / death / lineage state. ``None`` if
        ``capabilities.population=False``.
    events:
        Tick-scoped event context. ``None`` if
        ``capabilities.events=False``.
    capabilities:
        The :class:`CapabilityProfile` declared by the world. Copied into
        the state view so consumers can branch on capabilities without
        holding a world reference.
    missing_mask:
        Per-slot mask: ``True`` means the world has the capability but
        this particular record lacks the value. MUST NOT be ``True`` for
        a slot the world declares ``capabilities.<slot>=False``.
    """

    meta: StateMeta
    entities: EntityTable
    capabilities: "CapabilityProfile"
    missing_mask: Mapping[str, bool] = field(default_factory=dict)
    fields: FieldState | None = None
    relations: RelationGraph | None = None
    registries: RegistrySnapshot | None = None
    population: PopulationState | None = None
    events: EventContext | None = None

    def __post_init__(self) -> None:
        # Rule: missing_mask keys must be a subset of declared capabilities.
        cap_flags = self.capabilities.slot_flags()
        unknown_keys = set(self.missing_mask) - set(cap_flags)
        if unknown_keys:
            raise StateError(
                f"missing_mask has keys not in CAPABILITY_SLOTS: {unknown_keys}"
            )
        # Rule: missing_mask MUST NOT be True for a slot the world declares
        # capabilities.<slot>=False.
        for slot, missing in self.missing_mask.items():
            if missing and not cap_flags[slot]:
                raise StateError(
                    f"missing_mask[{slot!r}]=True but capabilities.{slot}=False; "
                    "missing_mask may only be True for slots the world has."
                )
        # Per-slot consistency: capability / missing_mask / value triple.
        for slot in cap_flags:
            value = getattr(self, slot)
            missing = self.missing_mask.get(slot, False)
            has_cap = cap_flags[slot]
            if not has_cap:
                # World does not have this capability. Value MUST be None.
                if value is not None:
                    raise StateError(
                        f"capabilities.{slot}=False but the slot value is not "
                        "None; a world without this capability must not "
                        "provide a value."
                    )
            elif missing:
                # World has the capability but this record lacks the value.
                if value is not None:
                    raise StateError(
                        f"missing_mask[{slot!r}]=True but the slot value is "
                        "not None; a missing slot must be None."
                    )
            else:
                # World has the capability and the record should have the value.
                if value is None:
                    raise StateError(
                        f"capabilities.{slot}=True and "
                        f"missing_mask[{slot!r}]=False but the slot value is "
                        "None; either populate the slot or set "
                        f"missing_mask[{slot!r}]=True."
                    )
