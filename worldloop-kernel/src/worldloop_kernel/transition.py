"""Transition and checkpoint types (K-04, K-07).

Defines the transfer record (one tick's worth of state transition) and
the checkpoint (full restorable world state).

Design rules (per ADR §3 and main plan §4.4):
- :class:`TransitionRecord` is the externally publishable, exchangeable
  unit. :class:`Checkpoint` is the internally restorable unit. They
  MUST be separate types — declaring a :class:`StateView` "sufficient
  for restore" is forbidden unless ``capability.exact_restore=False``
  AND the world publishes a ``CheckpointCodec`` (K-07).
- ``opaque_payload`` keeps the kernel free of world-specific internal
  state. The kernel records and hashes the bytes; the world owns
  encoding / decoding.
- ``schema_version`` is the kernel protocol version (independent from
  ``producer_version``, which is the world implementation version).

K-04 lands:
- :class:`StateDelta` and per-slot change sub-types
- :class:`TransitionRecord`
- :class:`Checkpoint` (data shape only; codec lands in K-07)

K-07 will add:
- :class:`CheckpointCodec` Protocol
- :func:`encode_checkpoint` / :func:`decode_checkpoint`
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from worldloop_kernel.action import (
        ActionProposal,
        ExecutedAction,
        ActionReceipt,
        ExogenousInput,
    )
    from worldloop_kernel.capability import CapabilityProfile
    from worldloop_kernel.state import (
        StateMeta,
        StateView,
        RelationEdge,
        RegistryEntry,
    )

__all__ = [
    "FieldChange",
    "EntityChange",
    "EntityChanges",
    "RelationChange",
    "RelationChanges",
    "RegistryChange",
    "RegistryChanges",
    "PopulationChange",
    "PopulationChanges",
    "EventRecord",
    "StateDelta",
    "TransitionRecord",
    "Checkpoint",
    "TransitionError",
    "PROTOCOL_SCHEMA_VERSION",
]

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TransitionError(ValueError):
    """Raised when a transition or checkpoint type is internally inconsistent."""


#: Kernel protocol schema version. Independent from any world
#: implementation version. Bumped when the on-disk shape of
#: :class:`TransitionRecord` or :class:`Checkpoint` changes in a
#: backward-incompatible way.
PROTOCOL_SCHEMA_VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# Per-slot change sub-types (used by StateDelta)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldChange:
    """Change to a single field channel between two ticks.

    Attributes
    ----------
    channel:
        Channel name that changed.
    before:
        Opaque representation of the channel value before the tick.
    after:
        Opaque representation of the channel value after the tick.
    """

    channel: str
    before: Any
    after: Any


@dataclass(frozen=True)
class EntityChange:
    """A single entity-level change (add / remove / update).

    Attributes
    ----------
    kind:
        One of ``"add"``, ``"remove"``, ``"update"``.
    entity_id:
        ID of the affected entity.
    column:
        Column that changed (only for ``kind="update"``). ``None`` for
        add / remove.
    before:
        Column value before the change (``None`` for add).
    after:
        Column value after the change (``None`` for remove).
    """

    kind: str  # "add" | "remove" | "update"
    entity_id: str | int
    column: str | None = None
    before: Any = None
    after: Any = None

    def __post_init__(self) -> None:
        if self.kind not in ("add", "remove", "update"):
            raise TransitionError(
                f"EntityChange.kind must be 'add' / 'remove' / 'update', "
                f"got {self.kind!r}"
            )
        if self.kind == "update" and not self.column:
            raise TransitionError(
                "EntityChange.kind='update' requires a non-empty column"
            )


@dataclass(frozen=True)
class EntityChanges:
    """Bundle of entity-level changes between two ticks.

    Attributes
    ----------
    schema_id:
        Schema ID of the :class:`EntityTable` this diff was computed
        against. MUST match the schema of the before/after states.
    changes:
        Tuple of :class:`EntityChange` in stable order.
    ids_after:
        K-05 round-trip extension: the after-state ``ids`` tuple when
        it differs from before. ``None`` means "no id order change"
        (apply_delta keeps before's id order). This is needed because
        ``EntityTable.ids`` order is part of the canonical hash, and
        the world may reorder ids between ticks in ways that cannot be
        derived from add/remove/update changes alone.
    """

    schema_id: str
    changes: tuple[EntityChange, ...] = field(default_factory=tuple)
    ids_after: tuple[str | int, ...] | None = None


@dataclass(frozen=True)
class RelationChange:
    """A single relation-level change (add / remove / update weight).

    Attributes
    ----------
    kind:
        One of ``"add"``, ``"remove"``, ``"update_weight"``.
    src:
        Source node ID.
    dst:
        Destination node ID.
    edge_type:
        Edge type.
    before_weight:
        Edge weight before the change (``None`` for add).
    after_weight:
        Edge weight after the change (``None`` for remove).
    """

    kind: str  # "add" | "remove" | "update_weight"
    src: str | int
    dst: str | int
    edge_type: str
    before_weight: float | None = None
    after_weight: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("add", "remove", "update_weight"):
            raise TransitionError(
                f"RelationChange.kind must be 'add' / 'remove' / "
                f"'update_weight', got {self.kind!r}"
            )


@dataclass(frozen=True)
class RelationChanges:
    """Bundle of relation-level changes between two ticks.

    K-05 extension: ``node_ids_after`` captures the after-state
    ``node_ids`` when it differs. ``edges_after`` captures the full
    after-state edges tuple when needed for exact round-trip (edge
    order is world-defined and may not follow the sort convention).
    Both default to ``None`` (no change).
    """

    schema_id: str
    changes: tuple[RelationChange, ...] = field(default_factory=tuple)
    node_ids_after: tuple[str | int, ...] | None = None
    edges_after: tuple["RelationEdge", ...] | None = None


@dataclass(frozen=True)
class RegistryChange:
    """A single registry-level change (add / state-change / remove).

    Attributes
    ----------
    kind:
        One of ``"add"``, ``"remove"``, ``"state_change"``.
    entry_id:
        ID of the affected registry entry.
    registry_type:
        Registry type (``"object"``, ``"concept"``, ...).
    before_state:
        State before the change (``None`` for add).
    after_state:
        State after the change (``None`` for remove).
    """

    kind: str  # "add" | "remove" | "state_change"
    entry_id: str
    registry_type: str
    before_state: str | None = None
    after_state: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("add", "remove", "state_change"):
            raise TransitionError(
                f"RegistryChange.kind must be 'add' / 'remove' / "
                f"'state_change', got {self.kind!r}"
            )


@dataclass(frozen=True)
class RegistryChanges:
    """Bundle of registry-level changes between two ticks.

    K-05 extension: ``entries_after`` captures the after-state
    ``entries`` tuple when it differs, for exact round-trip (entry
    order is world-defined). ``None`` means "no change".
    """

    schema_id: str
    changes: tuple[RegistryChange, ...] = field(default_factory=tuple)
    entries_after: tuple["RegistryEntry", ...] | None = None


@dataclass(frozen=True)
class PopulationChange:
    """A single population-level change (birth / death).

    The kernel records births and deaths as separate change records;
    lineage updates are surfaced as birth records with parent IDs.
    """

    kind: str  # "birth" | "death"
    agent_id: str | int
    tick: int
    #: For births: tuple of parent IDs. ``None`` for deaths.
    parent_ids: tuple[str | int, ...] | None = None
    #: For deaths: cause code. ``None`` for births.
    cause: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("birth", "death"):
            raise TransitionError(
                f"PopulationChange.kind must be 'birth' / 'death', "
                f"got {self.kind!r}"
            )
        if self.kind == "birth" and self.parent_ids is None:
            raise TransitionError(
                "PopulationChange.kind='birth' requires parent_ids"
            )
        if self.kind == "death" and not self.cause:
            raise TransitionError(
                "PopulationChange.kind='death' requires a non-empty cause"
            )


@dataclass(frozen=True)
class PopulationChanges:
    """Bundle of population-level changes between two ticks.

    K-05 extension: ``alive_ids_after`` captures the after-state
    ``alive_ids`` tuple when it differs from before. This is needed
    for round-trip correctness because ``alive_ids`` order is
    world-defined and cannot be derived from births/deaths alone.
    ``None`` means "no change" (apply_delta keeps before's alive_ids).
    """

    changes: tuple[PopulationChange, ...] = field(default_factory=tuple)
    alive_ids_after: tuple[str | int, ...] | None = None


# ---------------------------------------------------------------------------
# EventRecord — surfaced in StateDelta.event_log
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EventRecord:
    """A single tick-scoped event recorded in the transition.

    Note: this mirrors :class:`worldloop_kernel.state.EventRecord` but
    is duplicated here to keep the transition module self-contained
    for serialization. The two types are structurally compatible.

    K-05 will provide canonical encoding rules that ensure both
    representations hash to the same digest.
    """

    kind: str
    tick: int
    payload: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# StateDelta — output of diff_state(before, after) (K-05)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StateDelta:
    """Compact per-slot changes between two :class:`StateView` objects.

    Computed by :func:`worldloop_kernel.diff_apply.diff_state` (K-05).
    A :class:`StateDelta` is itself immutable and hashable, suitable
    for embedding in :class:`TransitionRecord`.

    Per-slot rules:
    - A slot the world does NOT have (``capabilities.<slot>=False``) has
      ``None`` for its changes bundle.
    - A slot the world has but no changes occurred has an EMPTY changes
      bundle (e.g., ``EntityChanges(changes=())``), NOT ``None``.

    K-05 extension (round-trip correctness):
    - ``meta_after`` captures the after-state ``StateMeta`` when it
      differs from before (e.g., ``meta.tick`` advances every tick).
      :func:`apply_delta` uses ``meta_after`` if present, else keeps
      ``before.meta``. Without this field, the round-trip invariant
      ``hash_state(apply_delta(before, diff_state(before, after))) ==
      hash_state(after)`` could not hold when ``meta.tick`` differs.
    - ``missing_mask_after`` captures the after-state ``missing_mask``
      when it differs from before (rare — happens only when a slot
      transitions between missing and present mid-run).
    - ``capabilities`` are NOT captured in the delta: they are declared
      by the world implementation and MUST be identical between
      ``before`` and ``after``. :func:`diff_state` raises if they differ.
    """

    field_changes: tuple[FieldChange, ...] | None = None
    entity_changes: EntityChanges | None = None
    relation_changes: RelationChanges | None = None
    registry_changes: RegistryChanges | None = None
    population_changes: PopulationChanges | None = None
    event_log: tuple[EventRecord, ...] | None = None
    # K-05 round-trip extension: meta and missing_mask are part of
    # StateView's hash, so the delta must capture their after-state when
    # they differ. None means "no change" (apply_delta keeps before's value).
    meta_after: "StateMeta | None" = None
    missing_mask_after: Mapping[str, bool] | None = None


# ---------------------------------------------------------------------------
# TransitionRecord — one complete transition S_t -> S_{t+1}
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransitionRecord:
    """One complete transition S_t -> S_{t+1}.

    This is the externally publishable, exchangeable unit. The kernel
    records one :class:`TransitionRecord` per tick. Consumers (datasets,
    evaluators, replay engines) read this type; they do NOT need to
    hold a world reference.

    Attributes
    ----------
    schema_version:
        Kernel protocol schema version (see :data:`PROTOCOL_SCHEMA_VERSION`).
    producer_id:
        Stable identifier of the world implementation that produced this
        record (e.g., ``"worldloop-native-v1"``, ``"pettingzoo-simple-spread"``).
    producer_version:
        Version of the world implementation that produced this record.
    tick:
        Tick at which the transition occurs (the ``t`` in ``S_t -> S_{t+1}``).
    state_before_hash:
        Hash of the state BEFORE the transition (computed via
        :func:`worldloop_kernel.canonical.hash_state` in K-05).
    candidate_actions:
        Mapping from agent ID to the :class:`ActionProposal` the policy
        produced. May be empty if no agents acted this tick.
    executed_actions:
        Mapping from agent ID to the :class:`ExecutedAction` the world
        executed. May be empty if no agents acted.
    exogenous_input:
        Tick-scoped exogenous input applied BEFORE actions, or ``None``.
    receipts:
        Mapping from agent ID to the :class:`ActionReceipt` the world
        returned. Keys MUST match ``executed_actions``.
    state_delta:
        Compact per-slot changes between ``state_before`` and ``state_after``.
    state_after_hash:
        Hash of the state AFTER the transition.
    capability_profile:
        The :class:`CapabilityProfile` declared by the world. Copied
        into the record so consumers can branch on capabilities without
        holding a world reference.
    provenance:
        Free-form provenance mapping (e.g., ``{"git_commit": "...",
        "config_hash": "..."}``). The kernel does NOT interpret the
        contents.
    """

    schema_version: str
    producer_id: str
    producer_version: str
    tick: int
    state_before_hash: str
    candidate_actions: Mapping[str | int, "ActionProposal"]
    executed_actions: Mapping[str | int, "ExecutedAction"]
    exogenous_input: "ExogenousInput | None"
    receipts: Mapping[str | int, "ActionReceipt"]
    state_delta: StateDelta
    state_after_hash: str
    capability_profile: "CapabilityProfile"
    provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != PROTOCOL_SCHEMA_VERSION:
            raise TransitionError(
                f"schema_version {self.schema_version!r} does not match "
                f"PROTOCOL_SCHEMA_VERSION {PROTOCOL_SCHEMA_VERSION!r}"
            )
        if self.tick < 0:
            raise TransitionError(f"tick must be >= 0, got {self.tick}")
        if not self.producer_id:
            raise TransitionError("producer_id must be a non-empty string")
        if not self.producer_version:
            raise TransitionError("producer_version must be a non-empty string")
        if not self.state_before_hash:
            raise TransitionError("state_before_hash must be non-empty")
        if not self.state_after_hash:
            raise TransitionError("state_after_hash must be non-empty")
        # Rule: executed_actions and receipts MUST have matching keys.
        exec_keys = set(self.executed_actions.keys())
        receipt_keys = set(self.receipts.keys())
        if exec_keys != receipt_keys:
            raise TransitionError(
                f"executed_actions keys {exec_keys} do not match receipts "
                f"keys {receipt_keys}; they must be identical."
            )


# ---------------------------------------------------------------------------
# Checkpoint — full restorable world state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Checkpoint:
    """Full restorable world state. May include hidden variables, internal
    caches, RNG state, scheduler state — anything the world needs to
    resume exactly.

    Per main plan §4.4, :class:`StateView` alone is NOT enough for
    restoration. :class:`Checkpoint` carries an ``opaque_payload`` the
    world owns; the kernel records and hashes the bytes but does NOT
    interpret them.

    Attributes
    ----------
    schema_version:
        Kernel protocol schema version (see :data:`PROTOCOL_SCHEMA_VERSION`).
    world_id:
        Stable identifier of the world implementation that produced this
        checkpoint (same as :attr:`TransitionRecord.producer_id`).
    world_version:
        Version of the world implementation (same as
        :attr:`TransitionRecord.producer_version`).
    tick:
        Tick at which the checkpoint was taken.
    state_view:
        The :class:`StateView` at this tick. Provided so consumers can
        inspect the observable state without decoding the
        ``opaque_payload``.
    opaque_payload:
        World-specific bytes the world needs to resume exactly. The
        kernel does NOT interpret them; the world owns encoding /
        decoding (via a ``CheckpointCodec`` in K-07).
    payload_codec:
        Codec identifier (e.g., ``"pickle+v1"``, ``"json+v1"``,
        ``"cbor+v1"``). The kernel uses this to dispatch to the right
        decoder; it does NOT implement the codecs itself.
    capability_profile:
        The :class:`CapabilityProfile` declared by the world. Copied
        into the checkpoint so consumers can branch on capabilities
        without decoding the payload.
    rng_bundle:
        Optional mapping from RNG stream name to a stringified state
        (e.g., ``{"main": "MT19937:...", "spawn": "PCG64:..."}``). The
        kernel does NOT interpret the contents; the world owns the
        format. ``None`` if the world does not expose RNG state.
    checksum:
        Stable checksum of the ``opaque_payload`` (and ideally the
        ``state_view`` + ``rng_bundle``). The kernel recomputes this on
        restore and rejects mismatches.
    """

    schema_version: str
    world_id: str
    world_version: str
    tick: int
    state_view: "StateView"
    opaque_payload: bytes
    payload_codec: str
    capability_profile: "CapabilityProfile"
    rng_bundle: Mapping[str, str] | None = None
    checksum: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != PROTOCOL_SCHEMA_VERSION:
            raise TransitionError(
                f"schema_version {self.schema_version!r} does not match "
                f"PROTOCOL_SCHEMA_VERSION {PROTOCOL_SCHEMA_VERSION!r}"
            )
        if self.tick < 0:
            raise TransitionError(f"tick must be >= 0, got {self.tick}")
        if not self.world_id:
            raise TransitionError("world_id must be a non-empty string")
        if not self.world_version:
            raise TransitionError("world_version must be a non-empty string")
        if not self.payload_codec:
            raise TransitionError("payload_codec must be a non-empty string")
        if not isinstance(self.opaque_payload, (bytes, bytearray)):
            raise TransitionError(
                f"opaque_payload must be bytes, got {type(self.opaque_payload)}"
            )
        # Rule: if capability.exact_restore=True, the checkpoint MUST
        # carry a non-empty checksum so the kernel can verify restoration.
        if self.capability_profile.exact_restore and not self.checksum:
            raise TransitionError(
                "capability_profile.exact_restore=True REQUIRES a non-empty "
                "checksum; the kernel verifies restoration by recomputing "
                "the checksum."
            )
