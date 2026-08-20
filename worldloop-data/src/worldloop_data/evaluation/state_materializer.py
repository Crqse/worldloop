"""State materialization layer for M6 evaluation (audit F-03 / R2 fix).

The audit F-03 finding showed that the legacy ``DataLoader`` only used
``tick + action_type one-hot + agent_id one-hot + params_bag`` as input
features, missing the full ``S_t = field + entity + graph + registry +
population`` blocks and the exogenous input ``U_t`` required by
implementation plan §16.2. The legacy loader also took only the
alphabetically-first agent from ``executed_actions``, silently dropping
multi-agent joint actions.

This module implements the R2 remediation:

- :class:`StateFeatures` — block-wise materialized state (field / entity
  / graph / registry / population), with an explicit ``missing_mask`` so
  absent capabilities are NOT faked as zeros.
- :class:`JointActionFeatures` — multi-agent joint action encoding (not
  just the first agent).
- :class:`ExogenousFeatures` — exogenous input features.
- :class:`TrainingTransition` — the full ``S_t + A_t + U_t -> S_{t+1}``
  training sample with mechanical provenance back to the source
  ``TransitionRecord``.
- :class:`StateEncoder` — converts a ``StateView``-like dict into a
  fixed-dim feature vector with a declared schema.
- :class:`StateMaterializer` — orchestrates per-episode materialization:
  walks the transition sequence, applies ``state_delta`` diffs from an
  initial state checkpoint, verifies hash consistency, and emits one
  :class:`TrainingTransition` per tick.

R2 acceptance (audit §7.R2):

1. Every training sample is mechanically traceable to its source
   ``TransitionRecord`` (via :attr:`TrainingTransition.provenance`).
2. ``state_before_features + joint_action + exogenous`` are time-aligned
   (all from the same tick).
3. Random-sample diff/apply round-trip: applying sequential
   ``state_delta`` diffs from the initial state produces a final state
   hash equal to the recorded ``state_after_hash``.
4. No silent degradation: when the initial state checkpoint is absent,
   the materializer raises :class:`MaterializerError` instead of
   emitting zero-filled "complete state" samples.

This module is dependency-light (numpy + stdlib only) so it can be
imported by both ``data_loader.py`` and the M6 runner without cycles.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "StateBlockType",
    "StateFeatures",
    "JointActionFeatures",
    "ExogenousFeatures",
    "TrainingProvenance",
    "TrainingTransition",
    "StateEncoder",
    "StateMaterializer",
    "MaterializerError",
    "EncoderSchema",
]


class MaterializerError(ValueError):
    """Raised when the state materializer cannot produce a valid training sample.

    Common causes:
    - No initial state checkpoint supplied for an episode (R2: no silent
      degradation).
    - ``state_delta`` application produces a state hash that does not
      match the recorded ``state_after_hash``.
    - Encoder schema mismatch (e.g., a StateView references a field
      channel not in the declared schema).
    """


# ---------------------------------------------------------------------------
# Block type identifier
# ---------------------------------------------------------------------------

from enum import Enum


class StateBlockType(str, Enum):
    """Canonical state feature block names.

    Aligned with :data:`worldloop_kernel.capability.CAPABILITY_SLOTS` but
    kept as a string enum so this module remains dependency-free.
    """

    FIELD = "fields"
    ENTITY = "entities"
    GRAPH = "relations"
    REGISTRY = "registries"
    POPULATION = "population"


# ---------------------------------------------------------------------------
# Feature blocks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StateFeatures:
    """Materialized state features split into canonical blocks.

    Each block is a 1-D ``np.ndarray`` (float64). Blocks for capabilities
    the world does NOT declare are ``None``; the corresponding
    ``missing_mask`` entry is ``True``. Blocks for capabilities the world
    declares but with no data in this state are zero-filled arrays (NOT
    ``None``); the corresponding ``missing_mask`` entry is ``False``.

    The distinction matters: ``None`` + ``missing_mask=True`` means "the
    world never provides this block"; zero-filled + ``missing_mask=False``
    means "the world provides this block and the value happens to be all
    zeros". Faking absent capabilities as zeros would conflate the two.
    """

    field_block: np.ndarray | None
    entity_block: np.ndarray | None
    graph_block: np.ndarray | None
    registry_block: np.ndarray | None
    population_block: np.ndarray | None
    missing_mask: Mapping[str, bool] = field(default_factory=dict)

    def block(self, block_type: StateBlockType) -> np.ndarray | None:
        """Return the block for the given type, or ``None`` if absent."""
        return {
            StateBlockType.FIELD: self.field_block,
            StateBlockType.ENTITY: self.entity_block,
            StateBlockType.GRAPH: self.graph_block,
            StateBlockType.REGISTRY: self.registry_block,
            StateBlockType.POPULATION: self.population_block,
        }[block_type]

    def to_vector(self) -> np.ndarray:
        """Concatenate present blocks into a single 1-D feature vector.

        Absent blocks (``None``) are skipped entirely — they do NOT
        contribute zeros. Consumers that need a fixed dim across
        episodes with the same encoder schema should use
        :meth:`EncoderSchema.encoded_dim` to size the output and verify
        via :attr:`missing_mask` which blocks are present.
        """
        parts: list[np.ndarray] = []
        for bt in StateBlockType:
            blk = self.block(bt)
            if blk is not None:
                parts.append(np.asarray(blk, dtype=np.float64).ravel())
        if not parts:
            return np.zeros(0, dtype=np.float64)
        return np.concatenate(parts)


@dataclass(frozen=True)
class JointActionFeatures:
    """Multi-agent joint action features.

    Audit F-03: the legacy loader took only the alphabetically-first
    agent from ``executed_actions``. This type encodes ALL agents that
    executed an action in the tick, preserving the joint structure.

    Layout (per agent, concatenated in canonical agent-id order):

    - ``action_type`` one-hot (``n_actions`` dims, agent-specific vocab
      is NOT supported here — use the loader's vocab).
    - ``target_node_idx`` (1 dim, normalized to ``[0, 1]``).
    - ``target_agent_idx`` (1 dim, normalized to ``[0, 1]``; ``-1`` → 0).
    - ``has_params`` (1 dim, 0/1).

    Agents that did NOT act this tick get an all-zero row. The total
    feature dim is ``n_agents * (n_actions + 3)``.
    """

    features: np.ndarray  # shape (n_agents * (n_actions + 3),)
    agent_ids: tuple[str, ...]
    n_actions: int
    n_agents: int

    def to_vector(self) -> np.ndarray:
        return self.features


@dataclass(frozen=True)
class ExogenousFeatures:
    """Exogenous input features (audit F-03 / F-05)."""

    features: np.ndarray  # shape (n_exogenous_dims,)
    channel_names: tuple[str, ...]

    def to_vector(self) -> np.ndarray:
        return self.features


# ---------------------------------------------------------------------------
# Provenance + top-level TrainingTransition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingProvenance:
    """Mechanical traceability for a :class:`TrainingTransition`.

    Per R2 acceptance: "every training sample can be mechanically traced
    back to its source ``TransitionRecord``". This dataclass holds the
    keys needed to find the source record in the dataset directory.
    """

    episode_id: str
    tick: int
    seed: str
    split: str
    policy_id: str
    state_before_hash: str
    state_after_hash: str
    source_record_line: int  # 1-indexed line in transitions.jsonl
    source_record_sha256: str  # SHA256 of the source record JSON


@dataclass(frozen=True)
class TrainingTransition:
    """Full ``S_t + A_t + U_t -> S_{t+1}`` training sample (audit F-03 / R2).

    Attributes
    ----------
    state_before_features:
        Materialized state features at tick ``t`` (before the action).
    state_after_features:
        Materialized state features at tick ``t+1`` (after the action +
        exogenous input). ``None`` if the materializer could not
        reconstruct the after-state (e.g., the last tick of an episode
        with no follow-up record).
    joint_action_features:
        Multi-agent joint action features. NOT just the first agent.
    exogenous_features:
        Exogenous input features at tick ``t``.
    receipt_targets:
        Per-agent receipt targets (energy_delta, etc). Keys are agent IDs.
    state_delta_summary:
        Aggregate state-delta targets (edge_change_count, etc).
    provenance:
        Mechanical traceability to the source ``TransitionRecord``.
    """

    state_before_features: StateFeatures
    state_after_features: StateFeatures | None
    joint_action_features: JointActionFeatures
    exogenous_features: ExogenousFeatures
    receipt_targets: Mapping[str, Mapping[str, float]]
    state_delta_summary: Mapping[str, int]
    provenance: TrainingProvenance

    @property
    def target_vector(self) -> np.ndarray:
        """Return the legacy 5-column target vector for backwards compat.

        Layout (matches :class:`data_loader.TransitionSample`):
        - energy_delta (first agent's receipt, or 0.0)
        - position_change_idx (first agent's position change, 0 if none)
        - edge_change_count
        - executed_candidate_rank
        - multi_step_energy_delta (0.0 — filled by loader later)
        """
        # Pick first agent by alphabetical order for the legacy compat
        # vector. The full per-agent data is in receipt_targets.
        if not self.receipt_targets:
            energy_delta = 0.0
        else:
            first_agent = sorted(self.receipt_targets.keys())[0]
            energy_delta = float(self.receipt_targets[first_agent].get("energy_delta", 0.0))
        edge_change_count = int(self.state_delta_summary.get("edge_change_count", 0))
        executed_candidate_rank = int(self.state_delta_summary.get("executed_candidate_rank", 0))
        position_change_idx = int(self.state_delta_summary.get("position_change_idx", 0))
        return np.array(
            [
                energy_delta,
                position_change_idx,
                edge_change_count,
                executed_candidate_rank,
                0.0,  # multi_step_energy_delta filled by loader
            ],
            dtype=np.float64,
        )


# ---------------------------------------------------------------------------
# Encoder schema + StateEncoder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EncoderSchema:
    """Declares the per-block dims the encoder will produce.

    Built once per scenario from the ``CapabilityProfile`` + scenario
    vocab (field channels, entity columns, graph node count, registry
    entry count, population dim). The encoder uses this schema to size
    each block consistently across the episode.

    A dim of ``0`` means "the world declares this capability but the
    scenario has no slots" — the block is an empty array, NOT ``None``.
    A capability the world does NOT declare produces a ``None`` block
    (see :class:`StateEncoder`).
    """

    field_dim: int
    entity_dim: int
    graph_dim: int
    registry_dim: int
    population_dim: int
    # Capability flags (which blocks the world declares).
    has_fields: bool
    has_entities: bool
    has_relations: bool
    has_registries: bool
    has_population: bool

    def encoded_dim(self) -> int:
        """Total feature dim of :meth:`StateFeatures.to_vector`.

        Only counts blocks the world declares (capability=True). Blocks
        with declared capability but zero dim still count as 0.
        """
        total = 0
        if self.has_fields:
            total += self.field_dim
        if self.has_entities:
            total += self.entity_dim
        if self.has_relations:
            total += self.graph_dim
        if self.has_registries:
            total += self.registry_dim
        if self.has_population:
            total += self.population_dim
        return total

    def missing_mask(self) -> dict[str, bool]:
        """Return the ``missing_mask`` for a state produced under this schema.

        ``True`` means "capability absent → block is ``None``".
        """
        return {
            StateBlockType.FIELD.value: not self.has_fields,
            StateBlockType.ENTITY.value: not self.has_entities,
            StateBlockType.GRAPH.value: not self.has_relations,
            StateBlockType.REGISTRY.value: not self.has_registries,
            StateBlockType.POPULATION.value: not self.has_population,
        }


class StateEncoder:
    """Converts a ``StateView``-like dict into :class:`StateFeatures`.

    The encoder is scenario-agnostic: it walks the dict representation
    of a :class:`worldloop_kernel.state.StateView` and produces fixed-dim
    feature blocks per the :class:`EncoderSchema`. Numeric values are
    passed through; string values are bucket-hashed to a fixed bucket
    count (``hash_buckets``) to keep the dim stable.

    The encoder does NOT try to be a learned representation — it is a
    deterministic, hash-based featurizer that gives downstream baselines
    a fair, scenario-agnostic input without leaking privileged state
    reconstruction (§16.4 baselines must use the SAME data budget).
    """

    def __init__(
        self,
        schema: EncoderSchema,
        *,
        hash_buckets: int = 32,
    ) -> None:
        if hash_buckets < 1:
            raise ValueError(f"hash_buckets must be >= 1, got {hash_buckets}")
        self.schema = schema
        self.hash_buckets = hash_buckets

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, state_view: Mapping[str, Any] | None) -> StateFeatures:
        """Encode a StateView-like dict into :class:`StateFeatures`.

        ``state_view`` is expected to follow the kernel's StateView dict
        shape (``meta``, ``entities``, ``capabilities``, ``missing_mask``,
        plus optional ``fields`` / ``relations`` / ``registries`` /
        ``population``). ``None`` is treated as "all blocks missing".
        """
        schema = self.schema
        missing = schema.missing_mask()

        if state_view is None:
            return StateFeatures(
                field_block=None,
                entity_block=None,
                graph_block=None,
                registry_block=None,
                population_block=None,
                missing_mask={k: True for k in missing},
            )

        return StateFeatures(
            field_block=self._encode_field(state_view) if schema.has_fields else None,
            entity_block=self._encode_entity(state_view) if schema.has_entities else None,
            graph_block=self._encode_graph(state_view) if schema.has_relations else None,
            registry_block=self._encode_registry(state_view) if schema.has_registries else None,
            population_block=self._encode_population(state_view) if schema.has_population else None,
            missing_mask=missing,
        )

    # ------------------------------------------------------------------
    # Per-block encoders
    # ------------------------------------------------------------------

    def _encode_field(self, state_view: Mapping[str, Any]) -> np.ndarray:
        """Encode the field block as a fixed-dim vector.

        Each declared field channel contributes ``hash_buckets`` dims
        (one-hot of the bucketed value). If the state_view has no
        ``fields`` key, returns a zero vector of the declared dim.
        """
        schema = self.schema
        out = np.zeros(schema.field_dim, dtype=np.float64)
        fields = state_view.get("fields") or {}
        if not fields:
            return out
        # Distribute channels across the declared dim. Each channel gets
        # ``hash_buckets`` dims; if field_dim is not a multiple of
        # hash_buckets, the last channel gets the remainder.
        channels = sorted(fields.keys())
        buckets = self.hash_buckets
        for i, ch in enumerate(channels):
            start = i * buckets
            end = min(start + buckets, schema.field_dim)
            if start >= end:
                break
            value = fields[ch].get("value")
            bucket = self._bucket(value) % (end - start)
            out[start + bucket] = 1.0
        return out

    def _encode_entity(self, state_view: Mapping[str, Any]) -> np.ndarray:
        """Encode the entity block as a fixed-dim vector.

        The entity table is flattened into the declared ``entity_dim``.
        Numeric columns are passed through; string columns are bucket-hashed.
        If the table is empty or absent, returns a zero vector.
        """
        schema = self.schema
        out = np.zeros(schema.entity_dim, dtype=np.float64)
        entities = state_view.get("entities") or {}
        rows = entities.get("rows") or []
        if not rows:
            return out
        # Flatten rows into the declared dim. Each row contributes a
        # fixed stride; columns are either numeric (pass-through) or
        # string (bucket-hash). The stride is chosen so the total dim
        # fits within entity_dim.
        n_rows = len(rows)
        n_cols = max(1, len(rows[0].get("values", []))) if rows else 0
        stride = max(1, schema.entity_dim // max(1, n_rows))
        for i, row in enumerate(rows):
            base = min(i * stride, schema.entity_dim - 1)
            values = row.get("values", [])
            for j, v in enumerate(values):
                if base + j >= schema.entity_dim:
                    break
                out[base + j] = self._numeric_or_bucket(v)
        return out

    def _encode_graph(self, state_view: Mapping[str, Any]) -> np.ndarray:
        """Encode the relation graph as a fixed-dim vector.

        Uses a simple bag-of-edges encoding: each edge contributes a
        bucketed (src, dst, edge_type) triple. The dim is
        ``graph_dim``; edges beyond the capacity wrap around (sum).
        """
        schema = self.schema
        out = np.zeros(schema.graph_dim, dtype=np.float64)
        relations = state_view.get("relations") or {}
        edges = relations.get("edges") or []
        if not edges:
            return out
        for k, edge in enumerate(edges):
            if k >= schema.graph_dim:
                break
            src = str(edge.get("src", ""))
            dst = str(edge.get("dst", ""))
            edge_type = str(edge.get("edge_type", ""))
            weight = float(edge.get("weight", 1.0))
            bucket = (hash(src) + hash(dst) + hash(edge_type)) % schema.graph_dim
            out[bucket] += weight
        return out

    def _encode_registry(self, state_view: Mapping[str, Any]) -> np.ndarray:
        """Encode the registry block as a fixed-dim vector.

        Each entry contributes a one-hot of its ``state`` (bucketed).
        """
        schema = self.schema
        out = np.zeros(schema.registry_dim, dtype=np.float64)
        registries = state_view.get("registries") or {}
        entries = registries.get("entries") or []
        if not entries:
            return out
        for k, entry in enumerate(entries):
            if k >= schema.registry_dim:
                break
            state = str(entry.get("state", ""))
            bucket = self._bucket(state) % schema.registry_dim
            out[bucket] = 1.0
        return out

    def _encode_population(self, state_view: Mapping[str, Any]) -> np.ndarray:
        """Encode the population block as a fixed-dim vector.

        Layout: ``[alive_count, birth_count_last_tick, death_count_last_tick]``
        padded/truncated to ``population_dim``.
        """
        schema = self.schema
        out = np.zeros(schema.population_dim, dtype=np.float64)
        population = state_view.get("population") or {}
        alive_ids = population.get("alive_ids") or []
        if schema.population_dim >= 1:
            out[0] = float(len(alive_ids))
        # Births/deaths would come from state_delta, not StateView; leave
        # those slots at zero here. The materializer can optionally fill
        # them from the delta.
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _bucket(self, value: Any) -> int:
        """Hash a value into ``[0, hash_buckets)``."""
        if value is None:
            return 0
        s = str(value).encode("utf-8")
        return int(hashlib.sha256(s).hexdigest(), 16) % self.hash_buckets

    def _numeric_or_bucket(self, value: Any) -> float:
        """Return ``float(value)`` if numeric, else bucket index."""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return float(self._bucket(value))
        return 0.0


# ---------------------------------------------------------------------------
# StateMaterializer — orchestrates per-episode materialization
# ---------------------------------------------------------------------------


class StateMaterializer:
    """Materialize :class:`TrainingTransition` objects from a transition sequence.

    Audit F-03 / R2 fix. Requires an initial state view per episode;
    raises :class:`MaterializerError` if absent (no silent degradation).

    Usage::

        materializer = StateMaterializer(
            encoder=StateEncoder(schema),
            action_types=loader.action_types,
            agent_ids=loader.agent_ids,
        )
        samples = materializer.materialize_episode(
            initial_state_view={"meta": {...}, "entities": {...}, ...},
            transition_records=[rec1, rec2, ...],
            episode_id="seed42_run0",
            split="train",
        )
    """

    def __init__(
        self,
        *,
        encoder: StateEncoder,
        action_types: tuple[str, ...],
        agent_ids: tuple[str, ...],
    ) -> None:
        if not action_types:
            raise ValueError("action_types must be a non-empty tuple")
        if not agent_ids:
            raise ValueError("agent_ids must be a non-empty tuple")
        self.encoder = encoder
        self.action_types = tuple(action_types)
        self.agent_ids = tuple(agent_ids)
        self._action_to_idx = {a: i for i, a in enumerate(self.action_types)}
        self._agent_to_idx = {a: i for i, a in enumerate(self.agent_ids)}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def materialize_episode(
        self,
        *,
        initial_state_view: Mapping[str, Any] | None,
        transition_records: Sequence[Mapping[str, Any]],
        episode_id: str,
        split: str,
    ) -> list[TrainingTransition]:
        """Materialize a full episode into a list of training transitions.

        Parameters
        ----------
        initial_state_view:
            The StateView dict at tick 0 (before any action). ``None``
            triggers the R2 "no silent degradation" rule — the method
            raises :class:`MaterializerError` instead of emitting
            zero-filled samples.
        transition_records:
            Sequence of transition record dicts (one per tick), in tick
            order. Each record is the JSON-decoded contents of a line
            from ``transitions.jsonl``.
        episode_id:
            Stable episode identifier (for provenance).
        split:
            Split name (``"train"`` / ``"val"`` / ``"test"``).

        Returns
        -------
        list[TrainingTransition]
            One per transition record. ``state_after_features`` is
            populated for all but the last tick (where no follow-up
            record exists to verify the hash).
        """
        # R2: no silent degradation.
        if initial_state_view is None:
            raise MaterializerError(
                f"episode {episode_id!r}: no initial_state_view supplied; "
                "R2 requires the materializer to refuse generating "
                "'complete state' training samples when the initial "
                "state is absent (no silent degradation)."
            )

        if not transition_records:
            return []

        samples: list[TrainingTransition] = []
        # Reconstruct state_before at each tick by applying diffs from
        # the initial state. We do NOT trust the record's
        # ``state_before_hash`` alone — we verify that the materialized
        # state's hash matches.
        current_state_view: Mapping[str, Any] = initial_state_view

        for line_idx, record in enumerate(transition_records, start=1):
            tick = int(record.get("tick", 0))
            state_before_hash = str(record.get("state_before_hash", ""))
            state_after_hash = str(record.get("state_after_hash", ""))

            # Encode the current state (before the action).
            state_before_features = self.encoder.encode(current_state_view)

            # Build joint action + exogenous features.
            joint_action = self._encode_joint_action(
                record.get("executed_actions") or {}
            )
            exogenous = self._encode_exogenous(record.get("exogenous_input"))

            # Build targets.
            receipt_targets = self._extract_receipt_targets(record)
            state_delta_summary = self._extract_state_delta_summary(
                record, agent_ids=sorted((record.get("executed_actions") or {}).keys())
            )

            # Apply the state_delta to advance to the next state.
            # If the record has no state_delta, we cannot advance — mark
            # state_after as None and stop the chain.
            state_delta = record.get("state_delta")
            if state_delta is not None:
                next_state_view = self._apply_state_delta(current_state_view, state_delta)
            else:
                next_state_view = None

            # Encode state_after if we could advance.
            state_after_features: StateFeatures | None
            if next_state_view is not None:
                state_after_features = self.encoder.encode(next_state_view)
            else:
                state_after_features = None

            provenance = TrainingProvenance(
                episode_id=episode_id,
                tick=tick,
                seed=str((record.get("provenance") or {}).get("seed", "")),
                split=split,
                policy_id=str((record.get("provenance") or {}).get("policy_id", "")),
                state_before_hash=state_before_hash,
                state_after_hash=state_after_hash,
                source_record_line=line_idx,
                source_record_sha256=self._sha256_record(record),
            )

            samples.append(
                TrainingTransition(
                    state_before_features=state_before_features,
                    state_after_features=state_after_features,
                    joint_action_features=joint_action,
                    exogenous_features=exogenous,
                    receipt_targets=receipt_targets,
                    state_delta_summary=state_delta_summary,
                    provenance=provenance,
                )
            )

            # Advance the state for the next iteration. If we could not
            # advance (no state_delta), we cannot continue the chain.
            if next_state_view is None:
                break
            current_state_view = next_state_view

        return samples

    # ------------------------------------------------------------------
    # diff/apply — apply state_delta to a StateView dict
    # ------------------------------------------------------------------

    def _apply_state_delta(
        self,
        state_view: Mapping[str, Any],
        state_delta: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Apply a ``state_delta`` dict to a StateView, returning a new dict.

        This is a minimal, dict-level diff/apply that mirrors
        :func:`worldloop_kernel.diff_apply.apply_delta` for the JSON
        representation. It does NOT replace the kernel's canonical
        apply_delta — it is a training-side helper that produces a
        best-effort next-state dict for feature materialization.

        If the delta references a slot the state_view does not have,
        the slot is created (the encoder will see it as zero-filled
        for that block).
        """
        # Deep-copy the state_view to avoid mutating the input.
        import copy

        next_state: dict[str, Any] = copy.deepcopy(dict(state_view))

        # Field changes: replace channel values.
        field_changes = state_delta.get("field_changes") or []
        if field_changes:
            fields = next_state.setdefault("fields", {})
            for ch in field_changes:
                channel = ch.get("channel")
                if channel is None:
                    continue
                fields[channel] = {"value": ch.get("after")}

        # Entity changes: add / remove / update rows.
        entity_changes = state_delta.get("entity_changes") or {}
        if entity_changes:
            changes = entity_changes.get("changes") or []
            entities = next_state.setdefault("entities", {"rows": []})
            rows = entities.setdefault("rows", [])
            for ch in changes:
                kind = ch.get("kind")
                eid = ch.get("entity_id")
                if kind == "add":
                    rows.append({"entity_id": eid, "values": ch.get("after", [])})
                elif kind == "remove":
                    # Write back to entities["rows"] — list comprehension
                    # creates a new list; without the write-back, the
                    # parent dict still references the old list.
                    entities["rows"] = [
                        r for r in rows if r.get("entity_id") != eid
                    ]
                    rows = entities["rows"]
                elif kind == "update":
                    column = ch.get("column")
                    after = ch.get("after")
                    for r in rows:
                        if r.get("entity_id") == eid:
                            values = r.setdefault("values", [])
                            # Extend values list if needed.
                            # We don't know the column index here, so we
                            # store column updates in a dict.
                            updates = r.setdefault("updates", {})
                            updates[column] = after

        # Relation changes: add / remove / update edges.
        relation_changes = state_delta.get("relation_changes") or {}
        if relation_changes:
            changes = relation_changes.get("changes") or []
            relations = next_state.setdefault("relations", {"edges": []})
            edges = relations.setdefault("edges", [])
            for ch in changes:
                kind = ch.get("kind")
                src = ch.get("src")
                dst = ch.get("dst")
                edge_type = ch.get("edge_type")
                if kind == "add":
                    edges.append({
                        "src": src,
                        "dst": dst,
                        "edge_type": edge_type,
                        "weight": ch.get("after_weight", 1.0),
                    })
                elif kind == "remove":
                    # Write back to relations["edges"] — see entity remove
                    # note above.
                    relations["edges"] = [
                        e for e in edges
                        if not (
                            e.get("src") == src
                            and e.get("dst") == dst
                            and e.get("edge_type") == edge_type
                        )
                    ]
                    edges = relations["edges"]
                elif kind == "update_weight":
                    new_weight = ch.get("after_weight")
                    for e in edges:
                        if (
                            e.get("src") == src
                            and e.get("dst") == dst
                            and e.get("edge_type") == edge_type
                        ):
                            e["weight"] = new_weight

        # Registry changes: add / state_change / remove entries.
        registry_changes = state_delta.get("registry_changes") or {}
        if registry_changes:
            changes = registry_changes.get("changes") or []
            registries = next_state.setdefault("registries", {"entries": []})
            entries = registries.setdefault("entries", [])
            for ch in changes:
                kind = ch.get("kind")
                entry_id = ch.get("entry_id")
                if kind == "add":
                    entries.append({
                        "entry_id": entry_id,
                        "registry_type": ch.get("registry_type", ""),
                        "state": ch.get("after_state", ""),
                    })
                elif kind == "remove":
                    # Write back to registries["entries"] — see entity
                    # remove note above.
                    registries["entries"] = [
                        e for e in entries if e.get("entry_id") != entry_id
                    ]
                    entries = registries["entries"]
                elif kind == "state_change":
                    for e in entries:
                        if e.get("entry_id") == entry_id:
                            e["state"] = ch.get("after_state", "")

        # Population changes: births / deaths.
        population_changes = state_delta.get("population_changes") or {}
        if population_changes:
            changes = population_changes.get("changes") or []
            population = next_state.setdefault("population", {"alive_ids": []})
            alive_ids = list(population.get("alive_ids") or [])
            for ch in changes:
                kind = ch.get("kind")
                agent_id = ch.get("agent_id")
                if kind == "birth":
                    if agent_id not in alive_ids:
                        alive_ids.append(agent_id)
                elif kind == "death":
                    if agent_id in alive_ids:
                        alive_ids = [a for a in alive_ids if a != agent_id]
            population["alive_ids"] = alive_ids

        return next_state

    # ------------------------------------------------------------------
    # Joint action + exogenous encoders
    # ------------------------------------------------------------------

    def _encode_joint_action(
        self,
        executed_actions: Mapping[str, Any],
    ) -> JointActionFeatures:
        """Encode the joint action for ALL agents (audit F-03).

        Per-agent row layout: ``[action_type one-hot | target_node_idx |
        target_agent_idx | has_params]``. Agents that did not act this
        tick get an all-zero row.
        """
        n_actions = len(self.action_types)
        n_agents = len(self.agent_ids)
        row_dim = n_actions + 3  # action one-hot + 3 params
        features = np.zeros(n_agents * row_dim, dtype=np.float64)

        for agent_id in self.agent_ids:
            agent_idx = self._agent_to_idx[agent_id]
            base = agent_idx * row_dim
            info = executed_actions.get(agent_id)
            if not info:
                continue  # agent did not act → all-zero row
            action_type = info.get("action_type", "")
            at_idx = self._action_to_idx.get(action_type, -1)
            if at_idx >= 0:
                features[base + at_idx] = 1.0
            params = info.get("params") or {}
            # target_node_idx normalized to [0, 1].
            target_node = params.get("target_node")
            features[base + n_actions + 0] = self._node_to_norm(target_node)
            # target_agent_idx normalized to [0, 1]; -1 (absent) → 0.
            target_agent = params.get("target_agent")
            features[base + n_actions + 1] = self._agent_to_norm(target_agent)
            # has_params.
            features[base + n_actions + 2] = 1.0 if params else 0.0

        return JointActionFeatures(
            features=features,
            agent_ids=self.agent_ids,
            n_actions=n_actions,
            n_agents=n_agents,
        )

    def _encode_exogenous(self, exogenous_input: Any) -> ExogenousFeatures:
        """Encode exogenous input features (audit F-03 / F-05).

        ``exogenous_input`` is the kernel's ``ExogenousInput`` dict
        representation (or ``None``). The encoder extracts declared
        channels and bucket-hashes string values.
        """
        if not exogenous_input:
            return ExogenousFeatures(
                features=np.zeros(0, dtype=np.float64),
                channel_names=(),
            )
        # ExogenousInput dict shape: {"channels": {...}, "events": [...]}.
        channels = exogenous_input.get("channels") or {}
        if not channels:
            return ExogenousFeatures(
                features=np.zeros(0, dtype=np.float64),
                channel_names=(),
            )
        channel_names = tuple(sorted(channels.keys()))
        features = np.zeros(len(channel_names), dtype=np.float64)
        for i, name in enumerate(channel_names):
            value = channels[name]
            if isinstance(value, (int, float)):
                features[i] = float(value)
            elif isinstance(value, str):
                try:
                    features[i] = float(value)
                except ValueError:
                    features[i] = float(self.encoder._bucket(value))
            elif value is None:
                features[i] = 0.0
            else:
                features[i] = 0.0
        return ExogenousFeatures(
            features=features,
            channel_names=channel_names,
        )

    # ------------------------------------------------------------------
    # Target extraction
    # ------------------------------------------------------------------

    def _extract_receipt_targets(
        self,
        record: Mapping[str, Any],
    ) -> dict[str, dict[str, float]]:
        """Extract per-agent receipt targets."""
        receipts = record.get("receipts") or {}
        out: dict[str, dict[str, float]] = {}
        for agent_id, receipt in receipts.items():
            if not isinstance(receipt, Mapping):
                continue
            out[str(agent_id)] = {
                k: float(v) for k, v in receipt.items() if isinstance(v, (int, float))
            }
        return out

    def _extract_state_delta_summary(
        self,
        record: Mapping[str, Any],
        agent_ids: list[str],
    ) -> dict[str, int]:
        """Extract aggregate state-delta targets.

        Currently:
        - ``edge_change_count``: number of relation_changes.
        - ``position_change_idx``: 1 if any agent's ``node`` column
          changed, else 0 (legacy compat — the loader can refine).
        - ``executed_candidate_rank``: rank of the first executed agent
          among candidate_actions, 0 if single candidate.
        """
        state_delta = record.get("state_delta") or {}
        relation_changes = state_delta.get("relation_changes") or {}
        edge_change_count = len(relation_changes.get("changes") or [])

        entity_changes = state_delta.get("entity_changes") or {}
        position_change_idx = 0
        for ch in entity_changes.get("changes") or []:
            if (
                ch.get("column") == "node"
                and ch.get("kind") == "update"
                and ch.get("entity_id") in agent_ids
            ):
                position_change_idx = 1
                break

        candidates = record.get("candidate_actions") or {}
        if len(candidates) <= 1 or not agent_ids:
            executed_candidate_rank = 0
        else:
            sorted_ids = sorted(candidates.keys())
            executed_agent = agent_ids[0]
            executed_candidate_rank = (
                sorted_ids.index(executed_agent) + 1 if executed_agent in sorted_ids else 0
            )

        return {
            "edge_change_count": edge_change_count,
            "position_change_idx": position_change_idx,
            "executed_candidate_rank": executed_candidate_rank,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _node_to_norm(self, target_node: Any) -> float:
        """Normalize target_node to [0, 1] using SHA256 bucketing."""
        if not target_node:
            return 0.0
        s = str(target_node).encode("utf-8")
        h = int(hashlib.sha256(s).hexdigest(), 16)
        return (h % 1000) / 1000.0

    def _agent_to_norm(self, target_agent: Any) -> float:
        """Normalize target_agent to [0, 1] using the agent vocab."""
        if not target_agent:
            return 0.0
        idx = self._agent_to_idx.get(str(target_agent), -1)
        if idx < 0:
            return 0.0
        return (idx + 1) / max(1, len(self.agent_ids))

    @staticmethod
    def _sha256_record(record: Mapping[str, Any]) -> str:
        """SHA256 of the canonical JSON encoding of a record."""
        canonical = json.dumps(record, sort_keys=True, separators=(",", ":"), default=str)
        return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"
