"""Diff and apply (K-05).

Provides ``diff_state(before, after) -> StateDelta`` and
``apply_delta(before, delta) -> StateView``. The pair MUST satisfy the
round-trip invariant:

    hash_state(apply_delta(before, diff_state(before, after))) == hash_state(after)

Design rules (per main plan §4.6):
- Diff is computed per-slot; only changed top-level slots appear in the
  delta with non-empty changes.
- ``apply_delta`` MUST NOT mutate ``before``; :class:`StateView` is frozen.
- Slots that are ``None`` in ``after`` but present in ``before`` are
  encoded as explicit deletions in the delta (not skipped).
- Diff of ``entities`` (:class:`EntityTable`) uses primary-key set
  operations to express insert / update / delete compactly.
- Capabilities MUST be identical between ``before`` and ``after``;
  ``diff_state`` raises :class:`DiffApplyError` if they differ.

K-05 round-trip extensions to K-04 types:
- :class:`StateDelta` adds ``meta_after`` and ``missing_mask_after``
  because ``meta.tick`` and ``missing_mask`` are part of the canonical
  hash and may change between ticks.
- :class:`EntityChanges` adds ``ids_after`` to preserve exact id order.
- :class:`RelationChanges` adds ``node_ids_after`` and ``edges_after``.
- :class:`RegistryChanges` adds ``entries_after``.
- :class:`PopulationChanges` adds ``alive_ids_after``.
- :class:`StateDelta.event_log`` is now ``tuple | None`` (None = no
  change, () = after has no events, (...) = after has these events).

Without these extensions, the round-trip invariant could not hold when
the world reorders entities, changes tick metadata, or toggles
``missing_mask`` between ticks. The extensions are backward-compatible:
all new fields default to ``None`` (treated as "no change").
"""

from __future__ import annotations

from dataclasses import replace as dc_replace
from typing import Any, Mapping

from worldloop_kernel.canonical import canonical_encode
from worldloop_kernel.state import (
    StateView,
    StateMeta,
    FieldState,
    EntityTable,
    RelationGraph,
    RelationEdge,
    RegistrySnapshot,
    RegistryEntry,
    PopulationState,
    BirthRecord,
    DeathRecord,
    EventContext,
    EventRecord as StateEventRecord,
)
from worldloop_kernel.transition import (
    StateDelta,
    FieldChange,
    EntityChange,
    EntityChanges,
    RelationChange,
    RelationChanges,
    RegistryChange,
    RegistryChanges,
    PopulationChange,
    PopulationChanges,
    EventRecord as TransitionEventRecord,
    TransitionError,
)

__all__ = [
    "DiffApplyError",
    "diff_state",
    "apply_delta",
]


class DiffApplyError(ValueError):
    """Raised when diff or apply cannot proceed."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canonical_equal(a: Any, b: Any) -> bool:
    """Return True iff ``a`` and ``b`` canonical-encode to the same bytes."""
    return canonical_encode(a) == canonical_encode(b)


def _slot_diff_or_none(before_slot: Any, after_slot: Any) -> bool:
    """True iff the two slot values differ by canonical encoding."""
    return not _canonical_equal(before_slot, after_slot)


# ---------------------------------------------------------------------------
# Fields diff / apply
# ---------------------------------------------------------------------------


def _diff_fields(
    before: FieldState | None,
    after: FieldState | None,
) -> tuple[FieldChange, ...] | None:
    """Diff two FieldState slots.

    Returns:
        - ``None`` if both are ``None`` (slot missing in both).
        - Empty tuple ``()`` if both present and equal.
        - Non-empty tuple of FieldChange if they differ.
    """
    if before is None and after is None:
        return None
    if before is None and after is not None:
        # Slot was added — emit add for every channel.
        return tuple(
            FieldChange(channel=ch, before=None, after=val)
            for ch, val in after.channels.items()
        )
    if before is not None and after is None:
        # Slot was removed — emit remove for every channel.
        return tuple(
            FieldChange(channel=ch, before=val, after=None)
            for ch, val in before.channels.items()
        )
    # Both present.
    if not _slot_diff_or_none(before, after):
        return ()
    # Schema mismatch is a hard error (world changed schema mid-run).
    if before.schema_id != after.schema_id:
        raise DiffApplyError(
            f"FieldState schema_id changed: {before.schema_id!r} -> "
            f"{after.schema_id!r}; schema must be stable within a run."
        )
    changes: list[FieldChange] = []
    before_ch = before.channels
    after_ch = after.channels
    for ch in before_ch:
        if ch not in after_ch:
            changes.append(FieldChange(channel=ch, before=before_ch[ch], after=None))
        elif not _canonical_equal(before_ch[ch], after_ch[ch]):
            changes.append(
                FieldChange(channel=ch, before=before_ch[ch], after=after_ch[ch])
            )
    for ch in after_ch:
        if ch not in before_ch:
            changes.append(FieldChange(channel=ch, before=None, after=after_ch[ch]))
    return tuple(changes)


def _apply_fields(
    before: FieldState | None,
    changes: tuple[FieldChange, ...] | None,
) -> FieldState | None:
    """Apply field changes to produce the after-state."""
    if changes is None:
        return before
    if not changes:
        # Empty tuple = no changes.
        return before
    if before is None:
        # All changes are adds.
        channels: dict[str, Any] = {}
        for ch in changes:
            channels[ch.channel] = ch.after
        return FieldState(schema_id="", channels=channels)
    # Apply changes to a copy of before.channels.
    new_channels = dict(before.channels)
    for ch in changes:
        if ch.after is None:
            new_channels.pop(ch.channel, None)
        else:
            new_channels[ch.channel] = ch.after
    return FieldState(
        schema_id=before.schema_id,
        channels=new_channels,
        units=before.units,
    )


# ---------------------------------------------------------------------------
# Entities diff / apply
# ---------------------------------------------------------------------------


def _entity_row(
    table: EntityTable, entity_id: str | int
) -> dict[str, Any]:
    """Extract a single entity's row as a {column: value} mapping."""
    idx = table.ids.index(entity_id)
    return {col: values[idx] for col, values in table.columns.items()}


def _diff_entities(
    before: EntityTable | None,
    after: EntityTable | None,
) -> EntityChanges | None:
    """Diff two EntityTable slots.

    Note: ``entities`` is mandatory (CapabilityProfile requires it), so
    both ``before`` and ``after`` should always be non-None. We handle
    the None cases defensively.
    """
    if before is None and after is None:
        return None
    if before is None:
        # All entities are adds.
        changes = tuple(
            EntityChange(
                kind="add",
                entity_id=eid,
                after=_entity_row(after, eid),
            )
            for eid in after.ids
        )
        return EntityChanges(
            schema_id=after.schema_id,
            changes=changes,
            ids_after=after.ids,
        )
    if after is None:
        # All entities are removes.
        changes = tuple(
            EntityChange(
                kind="remove",
                entity_id=eid,
                before=_entity_row(before, eid),
            )
            for eid in before.ids
        )
        return EntityChanges(
            schema_id=before.schema_id,
            changes=changes,
            ids_after=(),
        )
    # Both present.
    if before.schema_id != after.schema_id:
        raise DiffApplyError(
            f"EntityTable schema_id changed: {before.schema_id!r} -> "
            f"{after.schema_id!r}; schema must be stable within a run."
        )
    if not _slot_diff_or_none(before, after):
        # No changes — return empty bundle with no ids_after.
        return EntityChanges(schema_id=before.schema_id, changes=())
    before_ids = set(before.ids)
    after_ids = set(after.ids)
    removed = before_ids - after_ids
    added = after_ids - before_ids
    common = before_ids & after_ids
    changes: list[EntityChange] = []
    for eid in removed:
        changes.append(
            EntityChange(
                kind="remove",
                entity_id=eid,
                before=_entity_row(before, eid),
            )
        )
    for eid in added:
        changes.append(
            EntityChange(
                kind="add",
                entity_id=eid,
                after=_entity_row(after, eid),
            )
        )
    for eid in common:
        before_row = _entity_row(before, eid)
        after_row = _entity_row(after, eid)
        for col in before.columns:
            if col in after.columns and not _canonical_equal(
                before_row[col], after_row[col]
            ):
                changes.append(
                    EntityChange(
                        kind="update",
                        entity_id=eid,
                        column=col,
                        before=before_row[col],
                        after=after_row[col],
                    )
                )
    # Sort for determinism: (kind_order, entity_id, column).
    kind_order = {"remove": 0, "update": 1, "add": 2}
    changes.sort(
        key=lambda c: (kind_order[c.kind], str(c.entity_id), c.column or "")
    )
    # Capture ids_after if id order changed.
    ids_after = after.ids if before.ids != after.ids else None
    return EntityChanges(
        schema_id=before.schema_id,
        changes=tuple(changes),
        ids_after=ids_after,
    )


def _apply_entities(
    before: EntityTable | None,
    changes: EntityChanges | None,
) -> EntityTable | None:
    """Apply entity changes to produce the after-state."""
    if changes is None:
        return before
    if before is None:
        # All changes are adds; reconstruct from scratch.
        ids: list[str | int] = []
        columns: dict[str, list[Any]] = {}
        for ch in changes.changes:
            if ch.kind != "add":
                raise DiffApplyError(
                    f"Cannot apply {ch.kind!r} change to None EntityTable; "
                    "first change must be 'add'."
                )
            ids.append(ch.entity_id)
            for col, val in ch.after.items():
                columns.setdefault(col, []).append(val)
        return EntityTable(
            schema_id=changes.schema_id,
            ids=tuple(ids),
            columns={col: tuple(vals) for col, vals in columns.items()},
        )
    if not changes.changes and changes.ids_after is None:
        # No changes at all.
        return before
    # Build a mutable id -> row mapping from before.
    rows: dict[str | int, dict[str, Any]] = {}
    for eid in before.ids:
        rows[eid] = _entity_row(before, eid)
    # Apply changes in sorted order (removes, updates, adds).
    for ch in changes.changes:
        if ch.kind == "remove":
            rows.pop(ch.entity_id, None)
        elif ch.kind == "update":
            if ch.entity_id not in rows:
                raise DiffApplyError(
                    f"Cannot update missing entity {ch.entity_id!r}"
                )
            rows[ch.entity_id][ch.column] = ch.after
        elif ch.kind == "add":
            if not isinstance(ch.after, Mapping):
                raise DiffApplyError(
                    f"EntityChange 'add' for {ch.entity_id!r} requires "
                    "after to be a Mapping of column -> value."
                )
            rows[ch.entity_id] = dict(ch.after)
    # Determine final id order.
    if changes.ids_after is not None:
        final_ids = list(changes.ids_after)
        # Verify all ids_after are in rows (defensive).
        for eid in final_ids:
            if eid not in rows:
                raise DiffApplyError(
                    f"ids_after references missing entity {eid!r}"
                )
    else:
        final_ids = list(before.ids)
    # Build columns aligned with final_ids, using before's column order
    # plus any new columns introduced by adds.
    all_columns: list[str] = list(before.columns.keys())
    for ch in changes.changes:
        if ch.kind == "add" and isinstance(ch.after, Mapping):
            for col in ch.after:
                if col not in all_columns:
                    all_columns.append(col)
    new_columns: dict[str, tuple[Any, ...]] = {}
    for col in all_columns:
        new_columns[col] = tuple(rows[eid].get(col) for eid in final_ids)
    return EntityTable(
        schema_id=changes.schema_id,
        ids=tuple(final_ids),
        columns=new_columns,
    )


# ---------------------------------------------------------------------------
# Relations diff / apply
# ---------------------------------------------------------------------------


def _diff_relations(
    before: RelationGraph | None,
    after: RelationGraph | None,
) -> RelationChanges | None:
    """Diff two RelationGraph slots."""
    if before is None and after is None:
        return None
    if before is None:
        return RelationChanges(
            schema_id=after.schema_id,
            changes=tuple(),
            node_ids_after=after.node_ids,
            edges_after=after.edges,
        )
    if after is None:
        return RelationChanges(
            schema_id=before.schema_id,
            changes=tuple(),
            node_ids_after=(),
            edges_after=(),
        )
    if before.schema_id != after.schema_id:
        raise DiffApplyError(
            f"RelationGraph schema_id changed: {before.schema_id!r} -> "
            f"{after.schema_id!r}; schema must be stable within a run."
        )
    if not _slot_diff_or_none(before, after):
        return RelationChanges(schema_id=before.schema_id, changes=())
    # For round-trip safety, capture full after-state. The per-edge
    # changes are computed for readability but apply_delta uses edges_after.
    before_edges = {(e.src, e.dst, e.edge_type): e for e in before.edges}
    after_edges = {(e.src, e.dst, e.edge_type): e for e in after.edges}
    changes: list[RelationChange] = []
    for key, e in before_edges.items():
        if key not in after_edges:
            changes.append(
                RelationChange(
                    kind="remove",
                    src=e.src,
                    dst=e.dst,
                    edge_type=e.edge_type,
                    before_weight=e.weight,
                )
            )
        elif not _canonical_equal(e.weight, after_edges[key].weight):
            changes.append(
                RelationChange(
                    kind="update_weight",
                    src=e.src,
                    dst=e.dst,
                    edge_type=e.edge_type,
                    before_weight=e.weight,
                    after_weight=after_edges[key].weight,
                )
            )
    for key, e in after_edges.items():
        if key not in before_edges:
            changes.append(
                RelationChange(
                    kind="add",
                    src=e.src,
                    dst=e.dst,
                    edge_type=e.edge_type,
                    after_weight=e.weight,
                )
            )
    changes.sort(key=lambda c: (c.kind, str(c.src), str(c.dst), c.edge_type))
    node_ids_after = after.node_ids if before.node_ids != after.node_ids else None
    edges_after = after.edges if not _canonical_equal(before.edges, after.edges) else None
    return RelationChanges(
        schema_id=before.schema_id,
        changes=tuple(changes),
        node_ids_after=node_ids_after,
        edges_after=edges_after,
    )


def _apply_relations(
    before: RelationGraph | None,
    changes: RelationChanges | None,
) -> RelationGraph | None:
    """Apply relation changes to produce the after-state."""
    if changes is None:
        return before
    if before is None:
        # Use edges_after directly.
        edges = changes.edges_after if changes.edges_after is not None else ()
        node_ids = changes.node_ids_after if changes.node_ids_after is not None else ()
        return RelationGraph(
            schema_id=changes.schema_id,
            node_ids=node_ids,
            edges=edges,
        )
    if not changes.changes and changes.node_ids_after is None and changes.edges_after is None:
        return before
    # If edges_after is provided, use it for exact round-trip.
    if changes.edges_after is not None:
        edges = changes.edges_after
    else:
        # Reconstruct from before + changes.
        edge_map = {(e.src, e.dst, e.edge_type): e for e in before.edges}
        for ch in changes.changes:
            key = (ch.src, ch.dst, ch.edge_type)
            if ch.kind == "remove":
                edge_map.pop(key, None)
            elif ch.kind == "add":
                edge_map[key] = RelationEdge(
                    src=ch.src,
                    dst=ch.dst,
                    edge_type=ch.edge_type,
                    weight=ch.after_weight if ch.after_weight is not None else 1.0,
                )
            elif ch.kind == "update_weight":
                if key not in edge_map:
                    raise DiffApplyError(
                        f"Cannot update weight of missing edge {key!r}"
                    )
                old = edge_map[key]
                edge_map[key] = RelationEdge(
                    src=old.src,
                    dst=old.dst,
                    edge_type=old.edge_type,
                    weight=ch.after_weight if ch.after_weight is not None else old.weight,
                    born_at_tick=old.born_at_tick,
                )
        edges = tuple(edge_map.values())
    node_ids = (
        changes.node_ids_after
        if changes.node_ids_after is not None
        else before.node_ids
    )
    return RelationGraph(
        schema_id=changes.schema_id,
        node_ids=node_ids,
        edges=edges,
    )


# ---------------------------------------------------------------------------
# Registries diff / apply
# ---------------------------------------------------------------------------


def _diff_registries(
    before: RegistrySnapshot | None,
    after: RegistrySnapshot | None,
) -> RegistryChanges | None:
    """Diff two RegistrySnapshot slots."""
    if before is None and after is None:
        return None
    if before is None:
        return RegistryChanges(
            schema_id=after.schema_id,
            changes=tuple(),
            entries_after=after.entries,
        )
    if after is None:
        return RegistryChanges(
            schema_id=before.schema_id,
            changes=tuple(),
            entries_after=(),
        )
    if before.schema_id != after.schema_id:
        raise DiffApplyError(
            f"RegistrySnapshot schema_id changed: {before.schema_id!r} -> "
            f"{after.schema_id!r}; schema must be stable within a run."
        )
    if not _slot_diff_or_none(before, after):
        return RegistryChanges(schema_id=before.schema_id, changes=())
    before_entries = {(e.registry_type, e.entry_id): e for e in before.entries}
    after_entries = {(e.registry_type, e.entry_id): e for e in after.entries}
    changes: list[RegistryChange] = []
    for key, e in before_entries.items():
        if key not in after_entries:
            changes.append(
                RegistryChange(
                    kind="remove",
                    entry_id=e.entry_id,
                    registry_type=e.registry_type,
                    before_state=e.state,
                )
            )
        elif e.state != after_entries[key].state:
            changes.append(
                RegistryChange(
                    kind="state_change",
                    entry_id=e.entry_id,
                    registry_type=e.registry_type,
                    before_state=e.state,
                    after_state=after_entries[key].state,
                )
            )
    for key, e in after_entries.items():
        if key not in before_entries:
            changes.append(
                RegistryChange(
                    kind="add",
                    entry_id=e.entry_id,
                    registry_type=e.registry_type,
                    after_state=e.state,
                )
            )
    changes.sort(key=lambda c: (c.kind, c.registry_type, c.entry_id))
    entries_after = (
        after.entries
        if not _canonical_equal(before.entries, after.entries)
        else None
    )
    return RegistryChanges(
        schema_id=before.schema_id,
        changes=tuple(changes),
        entries_after=entries_after,
    )


def _apply_registries(
    before: RegistrySnapshot | None,
    changes: RegistryChanges | None,
) -> RegistrySnapshot | None:
    """Apply registry changes to produce the after-state."""
    if changes is None:
        return before
    if before is None:
        entries = changes.entries_after if changes.entries_after is not None else ()
        return RegistrySnapshot(
            schema_id=changes.schema_id,
            entries=entries,
        )
    if not changes.changes and changes.entries_after is None:
        return before
    if changes.entries_after is not None:
        entries = changes.entries_after
    else:
        entry_map = {(e.registry_type, e.entry_id): e for e in before.entries}
        for ch in changes.changes:
            key = (ch.registry_type, ch.entry_id)
            if ch.kind == "remove":
                entry_map.pop(key, None)
            elif ch.kind == "add":
                entry_map[key] = RegistryEntry(
                    entry_id=ch.entry_id,
                    registry_type=ch.registry_type,
                    state=ch.after_state or "",
                )
            elif ch.kind == "state_change":
                if key not in entry_map:
                    raise DiffApplyError(
                        f"Cannot change state of missing entry {key!r}"
                    )
                old = entry_map[key]
                entry_map[key] = RegistryEntry(
                    entry_id=old.entry_id,
                    registry_type=old.registry_type,
                    state=ch.after_state or "",
                    owner_id=old.owner_id,
                    metadata=old.metadata,
                )
        entries = tuple(entry_map.values())
    return RegistrySnapshot(
        schema_id=changes.schema_id,
        entries=entries,
    )


# ---------------------------------------------------------------------------
# Population diff / apply
# ---------------------------------------------------------------------------


def _diff_population(
    before: PopulationState | None,
    after: PopulationState | None,
) -> PopulationChanges | None:
    """Diff two PopulationState slots."""
    if before is None and after is None:
        return None
    if before is None:
        # Full add — capture all births.
        changes = tuple(
            PopulationChange(
                kind="birth",
                agent_id=b.child_id,
                tick=b.tick,
                parent_ids=b.parent_ids,
            )
            for b in after.births_this_tick
        )
        return PopulationChanges(
            changes=changes,
            alive_ids_after=after.alive_ids,
        )
    if after is None:
        # Full remove — capture all deaths.
        changes = tuple(
            PopulationChange(
                kind="death",
                agent_id=d.agent_id,
                tick=d.tick,
                cause=d.cause,
            )
            for d in before.deaths_this_tick
        )
        return PopulationChanges(
            changes=changes,
            alive_ids_after=(),
        )
    if not _slot_diff_or_none(before, after):
        return PopulationChanges(changes=())
    # Build change list from births and deaths.
    changes: list[PopulationChange] = []
    for b in after.births_this_tick:
        changes.append(
            PopulationChange(
                kind="birth",
                agent_id=b.child_id,
                tick=b.tick,
                parent_ids=b.parent_ids,
            )
        )
    for d in after.deaths_this_tick:
        changes.append(
            PopulationChange(
                kind="death",
                agent_id=d.agent_id,
                tick=d.tick,
                cause=d.cause,
            )
        )
    changes.sort(key=lambda c: (c.kind, str(c.agent_id), c.tick))
    alive_ids_after = (
        after.alive_ids if before.alive_ids != after.alive_ids else None
    )
    return PopulationChanges(
        changes=tuple(changes),
        alive_ids_after=alive_ids_after,
    )


def _apply_population(
    before: PopulationState | None,
    changes: PopulationChanges | None,
) -> PopulationState | None:
    """Apply population changes to produce the after-state."""
    if changes is None:
        return before
    if before is None:
        # Reconstruct from changes (all births).
        births = tuple(
            BirthRecord(
                parent_ids=c.parent_ids or (),
                child_id=c.agent_id,
                tick=c.tick,
            )
            for c in changes.changes
            if c.kind == "birth"
        )
        deaths = tuple(
            DeathRecord(
                agent_id=c.agent_id,
                tick=c.tick,
                cause=c.cause or "",
            )
            for c in changes.changes
            if c.kind == "death"
        )
        alive_ids = changes.alive_ids_after if changes.alive_ids_after is not None else ()
        return PopulationState(
            alive_ids=alive_ids,
            births_this_tick=births,
            deaths_this_tick=deaths,
            cumulative_births=len(births),
            cumulative_deaths=len(deaths),
        )
    if not changes.changes and changes.alive_ids_after is None:
        return before
    births = tuple(
        BirthRecord(
            parent_ids=c.parent_ids or (),
            child_id=c.agent_id,
            tick=c.tick,
        )
        for c in changes.changes
        if c.kind == "birth"
    )
    deaths = tuple(
        DeathRecord(
            agent_id=c.agent_id,
            tick=c.tick,
            cause=c.cause or "",
        )
        for c in changes.changes
        if c.kind == "death"
    )
    if changes.alive_ids_after is not None:
        alive_ids = changes.alive_ids_after
    else:
        # Derive from before: remove dead, add born.
        dead_ids = {d.agent_id for d in deaths}
        born_ids = [b.child_id for b in births]
        alive_ids = tuple(
            eid for eid in before.alive_ids if eid not in dead_ids
        ) + tuple(born_ids)
    return PopulationState(
        alive_ids=alive_ids,
        births_this_tick=births,
        deaths_this_tick=deaths,
        cumulative_births=before.cumulative_births + len(births),
        cumulative_deaths=before.cumulative_deaths + len(deaths),
    )


# ---------------------------------------------------------------------------
# Events diff / apply
# ---------------------------------------------------------------------------


def _diff_events(
    before: EventContext | None,
    after: EventContext | None,
) -> tuple[TransitionEventRecord, ...] | None:
    """Diff two EventContext slots.

    Events are tick-scoped, so we capture the full after-state. Returns:
        - ``None`` if both None (no change).
        - ``()`` if both present and equal, OR if after is None (slot removed).
        - Non-empty tuple if after has events (captured from after.events).
    """
    if before is None and after is None:
        return None
    if before is not None and after is None:
        # Slot was present, now missing — emit empty tuple to signal "clear".
        return ()
    if before is None and after is not None:
        # Slot was missing, now present — capture all events.
        return tuple(
            TransitionEventRecord(kind=e.kind, tick=e.tick, payload=e.payload)
            for e in after.events
        )
    # Both present.
    if not _slot_diff_or_none(before, after):
        return None  # No change.
    return tuple(
        TransitionEventRecord(kind=e.kind, tick=e.tick, payload=e.payload)
        for e in after.events
    )


def _apply_events(
    before: EventContext | None,
    event_log: tuple[TransitionEventRecord, ...] | None,
) -> EventContext | None:
    """Apply event changes to produce the after-state."""
    if event_log is None:
        return before
    # event_log is () or non-empty — both replace before.events entirely.
    return EventContext(
        events=tuple(
            StateEventRecord(kind=e.kind, tick=e.tick, payload=e.payload)
            for e in event_log
        )
    )


# ---------------------------------------------------------------------------
# Public API: diff_state and apply_delta
# ---------------------------------------------------------------------------


def diff_state(before: StateView, after: StateView) -> StateDelta:
    """Compute the :class:`StateDelta` between two :class:`StateView` objects.

    The delta captures every observable difference between ``before`` and
    ``after``. Combined with ``before``, it is sufficient to reconstruct
    ``after`` via :func:`apply_delta` such that:

        hash_state(apply_delta(before, diff_state(before, after))) == hash_state(after)

    Args:
        before: The state before the transition.
        after: The state after the transition.

    Returns:
        A :class:`StateDelta` capturing all changes.

    Raises:
        DiffApplyError: If ``before.capabilities`` and ``after.capabilities``
            differ (capabilities are declared by the world implementation
            and MUST be stable within a run).
    """
    # Capabilities MUST be identical.
    if not _canonical_equal(before.capabilities, after.capabilities):
        raise DiffApplyError(
            "CapabilityProfile changed between before and after; capabilities "
            "are declared by the world implementation and MUST be stable "
            "within a run."
        )
    cap_flags = before.capabilities.slot_flags()

    # Compute per-slot diffs.
    field_changes = (
        _diff_fields(before.fields, after.fields)
        if cap_flags["fields"]
        else None
    )
    entity_changes = _diff_entities(before.entities, after.entities)
    relation_changes = (
        _diff_relations(before.relations, after.relations)
        if cap_flags["relations"]
        else None
    )
    registry_changes = (
        _diff_registries(before.registries, after.registries)
        if cap_flags["registries"]
        else None
    )
    population_changes = (
        _diff_population(before.population, after.population)
        if cap_flags["population"]
        else None
    )
    event_log = (
        _diff_events(before.events, after.events)
        if cap_flags["events"]
        else None
    )

    # Meta and missing_mask are part of the canonical hash.
    meta_after = after.meta if not _canonical_equal(before.meta, after.meta) else None
    missing_mask_after = (
        after.missing_mask
        if not _canonical_equal(before.missing_mask, after.missing_mask)
        else None
    )

    return StateDelta(
        field_changes=field_changes,
        entity_changes=entity_changes,
        relation_changes=relation_changes,
        registry_changes=registry_changes,
        population_changes=population_changes,
        event_log=event_log,
        meta_after=meta_after,
        missing_mask_after=missing_mask_after,
    )


def apply_delta(before: StateView, delta: StateDelta) -> StateView:
    """Apply a :class:`StateDelta` to a :class:`StateView` to produce the after-state.

    The returned :class:`StateView` is observationally identical to the
    original ``after`` passed to :func:`diff_state`:

        hash_state(apply_delta(before, diff_state(before, after))) == hash_state(after)

    ``apply_delta`` does NOT mutate ``before``; :class:`StateView` is frozen.

    Args:
        before: The state before the transition.
        delta: The changes computed by :func:`diff_state`.

    Returns:
        A new :class:`StateView` representing the after-state.
    """
    cap_flags = before.capabilities.slot_flags()

    # Apply per-slot changes.
    new_fields = (
        _apply_fields(before.fields, delta.field_changes)
        if cap_flags["fields"]
        else None
    )
    new_entities = _apply_entities(before.entities, delta.entity_changes)
    new_relations = (
        _apply_relations(before.relations, delta.relation_changes)
        if cap_flags["relations"]
        else None
    )
    new_registries = (
        _apply_registries(before.registries, delta.registry_changes)
        if cap_flags["registries"]
        else None
    )
    new_population = (
        _apply_population(before.population, delta.population_changes)
        if cap_flags["population"]
        else None
    )
    new_events = (
        _apply_events(before.events, delta.event_log)
        if cap_flags["events"]
        else None
    )

    # Meta and missing_mask.
    new_meta = delta.meta_after if delta.meta_after is not None else before.meta
    new_missing_mask = (
        delta.missing_mask_after
        if delta.missing_mask_after is not None
        else before.missing_mask
    )

    return StateView(
        meta=new_meta,
        entities=new_entities,
        capabilities=before.capabilities,
        missing_mask=new_missing_mask,
        fields=new_fields,
        relations=new_relations,
        registries=new_registries,
        population=new_population,
        events=new_events,
    )
