"""ParameterizedWorld — spec-driven WorldProtocol implementation (S-05/S-06/S-07).

A single class that implements ``worldloop_kernel.WorldProtocol`` by
interpreting a :class:`ScenarioSpec` at runtime. No codegen — effects
are evaluated by a small interpreter that dispatches on ``target``
(entity / field / registry / relation).

Design rules (per main plan §13.4 and ADR §3):
- The world is constructed by :func:`worldloop_scenarios.compiler.compile_spec`
  via a factory (``world_factory(seed) -> ParameterizedWorld``).
- ``reset(seed)`` initializes entities / fields / registries / relations
  per the spec. All randomness flows from ``seed`` (deterministic given
  the same seed + action sequence).
- ``step()`` applies action effects in declared order, then post-step
  rules (energy floor at 0, death check if ``no_alive`` termination is
  set).
- ``checkpoint()`` pickles the full internal state; ``restore()`` decodes
  it back. The world is ``exact_restore=True`` and
  ``executable_deterministic_replay=True``.

Supported space types (S-05/S-06/S-07 in one class):
- ``"discrete"``: integer grid with ``shape`` (e.g., (10, 10)). Entity
  positions are integer (x, y).
- ``"continuous"``: continuous box with ``bounds``. Entity positions
  are float (x, y). Fields (resource density, hazard, ...) are 1D/2D
  arrays.
- ``"graph"``: nodes + edges. Entity positions are node IDs (string).
  Relations and registries are typically enabled.

Supported effect targets:
- ``entity``: modify a column on the acting agent (op: add/sub/set/clear/mul).
- ``field``: modify a field channel uniformly across all cells
  (op: add/sub/set). Spatial drainage is M4+ — v0.1 drains uniformly.
- ``registry``: modify a registry entry (op: set owner / set state /
  remove).
- ``relation``: add / remove an edge.

Supported preconditions:
- ``edge_exists``: for graph worlds; checks ``space.edges`` (or the
  runtime relation graph if relations are enabled).
- ``registry_unowned``: checks the referenced registry entry has no owner.
- ``energy_above``: checks the acting agent's energy is >= threshold.

Value reference syntax (resolved by :meth:`_resolve_value`):
- ``$param.<name>``: action parameter (e.g., ``$param.dx``).
- ``$entity.<column>``: acting agent's column value (e.g.,
  ``$entity.energy``, ``$entity.node``).
- ``$entity.id``: the acting agent's ID string.
- ``$<name>``: bare reference, treated as a param.
- Anything else: literal value.

Out of scope for M3 v0.1:
- Stochastic exogenous events beyond simple rate-based spawn (M4).
- Custom termination predicates (``kind="custom"``).
- Per-action reward shaping summed into a scalar (M4 / policy concern).
- Spatial drainage on field channels (v0.1 drains uniformly).
- Counterfactual ``legal_actions(state=...)`` queries.
"""

from __future__ import annotations

import pickle
import random
from typing import Any, Mapping

from worldloop_kernel.action import (
    ActionProposal,
    ActionReceipt,
    ExecutedAction,
    ExogenousInput,
    OUTCOME_ILLEGAL_ACTION,
    OUTCOME_OK,
    OUTCOME_UNRECOGNIZED_INTENT,
)
from worldloop_kernel.canonical import hash_state
from worldloop_kernel.capability import CapabilityProfile
from worldloop_kernel.diff_apply import diff_state
from worldloop_kernel.observation import (
    OBSERVATION_SCHEMA_VERSION,
    AgentObservationView,
    FocalAgentAttributes,
    OmissionPolicy,
    PreviousActionSummary,
    VisibleEntity,
    VisibleEvent,
    VisibleRelation,
)
from worldloop_kernel.protocol import ActionSpace, LegalAction
from worldloop_kernel.replay import compute_checkpoint_checksum
from worldloop_kernel.state import (
    BirthRecord,
    DeathRecord,
    EntityTable,
    EventContext,
    EventRecord,
    FieldState,
    PopulationState,
    RegistryEntry,
    RegistrySnapshot,
    RelationEdge,
    RelationGraph,
    StateMeta,
    StateView,
)
from worldloop_kernel.transition import (
    Checkpoint,
    PROTOCOL_SCHEMA_VERSION,
    TransitionRecord,
)

from worldloop_scenarios.spec import ScenarioSpec

__all__ = [
    "ParameterizedWorld",
    "PRODUCER_ID_PREFIX",
    "PRODUCER_VERSION",
    "PAYLOAD_CODEC",
]


#: Producer ID prefix (combined with scenario_id at construction).
PRODUCER_ID_PREFIX = "worldloop-scenarios"
#: Implementation version of ParameterizedWorld.
PRODUCER_VERSION = "0.1.0"
#: Checkpoint payload codec identifier.
PAYLOAD_CODEC = "pickle+v1"

#: Default visibility policy — entity columns visible to ALL agents
#: about each other. Columns NOT in this set (e.g., ``energy``) are
#: ``self_visible`` only — they appear in the focal agent's
#: :attr:`FocalAgentAttributes.self_visible_attributes` but are OMITTED
#: from other agents' :class:`VisibleEntity` projections.
#:
#: Rationale (per Phase 1 / Beta correction §5.3): position / alive
#: status / graph-node membership are public environmental facts any
#: agent can observe; energy is private internal state only the focal
#: agent can introspect. Future scenarios MAY override this via a
#: spec-level ``visibility`` section (out of scope for v0.1).
_DEFAULT_PUBLIC_ENTITY_COLUMNS: frozenset[str] = frozenset(
    {"x", "y", "node", "alive"}
)


class ParameterizedWorld:
    """Spec-driven ``WorldProtocol`` implementation.

    A single instance drives any of the three space types (discrete /
    continuous / graph) by interpreting the :class:`ScenarioSpec` at
    runtime. The world is constructed via the factory returned by
    :func:`worldloop_scenarios.compiler.compile_spec`; direct construction
    is allowed but the caller must ensure the spec has already passed
    schema + semantic validation.
    """

    def __init__(self, spec: ScenarioSpec) -> None:
        self._spec = spec
        self._cap = self._infer_capability(spec)
        scenario_id = spec.scenario.scenario_id
        scenario_version = spec.scenario.scenario_version
        self._producer_id = (
            f"{PRODUCER_ID_PREFIX}-{scenario_id}-v{scenario_version}"
        )
        # Mutable internal state — populated by reset().
        self._tick: int = 0
        self._seed: int = 0
        self._rng: random.Random | None = None
        # Dedicated RNG for exogenous event generation. Decoupled from
        # ``self._rng`` so that ``generate_exogenous(tick)`` does NOT
        # advance the world's primary RNG. This is critical for Q1
        # Traceability: ``state_before_hash[t+1]`` must equal
        # ``state_after_hash[t]``, which requires the world state
        # (including RNG state) to be unchanged between ``step(t)`` and
        # ``step(t+1)`` except for the action/exogenous applied inside
        # ``step``. Before this fix, ``generate_exogenous`` consumed
        # ``self._rng`` between ticks, breaking the hash chain.
        self._exogenous_rng: random.Random | None = None
        # Entity storage: dict[column_name -> list[values]] aligned with _entity_ids.
        self._entity_ids: list[str] = []
        self._entity_columns: dict[str, list[Any]] = {}
        # Field storage: dict[channel_name -> scalar | list | nested-list].
        self._field_channels: dict[str, Any] = {}
        # Registry storage: list of dicts with keys
        # (entry_id, registry_type, state, owner_id, metadata).
        self._registry_entries: list[dict[str, Any]] = []
        # Relation storage: list of (src, dst, edge_type, weight) tuples.
        self._relation_edges: list[tuple[str, str, str, float]] = []
        self._relation_node_ids: list[str] = []
        # Population tracking.
        self._alive_ids: list[str] = []
        self._cumulative_births: int = 0
        self._cumulative_deaths: int = 0
        self._births_this_tick: list[BirthRecord] = []
        self._deaths_this_tick: list[DeathRecord] = []
        # Events surfaced this tick.
        self._events_this_tick: list[EventRecord] = []

    # ------------------------------------------------------------------
    # Capability inference
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_capability(spec: ScenarioSpec) -> CapabilityProfile:
        """Infer the :class:`CapabilityProfile` from a :class:`ScenarioSpec`.

        Rules:
        - ``fields`` = ``spec.fields.enabled``.
        - ``entities`` = True (mandatory per main plan §4.2).
        - ``relations`` = ``spec.relations.enabled``.
        - ``registries`` = ``spec.registries.enabled``.
        - ``population`` = True iff any termination has ``kind=no_alive``,
          any action effect targets ``population``, or any exogenous event
          has kind in {birth, death}.
        - ``events`` = True iff any exogenous events exist.
        - ``exact_restore`` = True (we pickle the full world).
        - ``executable_deterministic_replay`` = True (deterministic given
          seed + actions).
        - ``authority`` = ``"rule"``; ``ground_truth`` = True.
        - ``transition_mode`` = ``"deterministic"`` iff ``spec.time.deterministic``.
        """
        fields_cap = bool(spec.fields.enabled)
        relations_cap = bool(spec.relations.enabled)
        registries_cap = bool(spec.registries.enabled)
        has_no_alive = any(
            cond.get("kind") == "no_alive"
            for cond in spec.termination.stop_conditions
        )
        has_population_target = any(
            effect.get("target") == "population"
            for action in spec.actions.actions
            for effect in action.get("effects", ())
        )
        has_pop_event = any(
            event.get("kind") in ("birth", "death")
            for event in spec.exogenous.events
        )
        population_cap = (
            has_no_alive or has_population_target or has_pop_event
        )
        events_cap = bool(spec.exogenous.events)
        deterministic = bool(spec.time.deterministic)
        return CapabilityProfile(
            fields=fields_cap,
            entities=True,
            relations=relations_cap,
            registries=registries_cap,
            population=population_cap,
            events=events_cap,
            exact_restore=True,
            executable_deterministic_replay=True,
            authority="rule",
            ground_truth=True,
            transition_mode="deterministic" if deterministic else "stochastic",
        )

    # ------------------------------------------------------------------
    # WorldProtocol property
    # ------------------------------------------------------------------

    @property
    def capabilities(self) -> CapabilityProfile:
        return self._cap

    # ------------------------------------------------------------------
    # WorldProtocol: reset / observe
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int,
        parameters: Mapping[str, Any] | None = None,
    ) -> StateView:
        """Reset the world to its initial state for the given seed."""
        self._seed = int(seed)
        self._rng = random.Random(self._seed)
        # Derive a deterministic but independent RNG for exogenous events.
        # XOR with a constant ensures the exogenous stream differs from
        # the primary stream while remaining reproducible.
        self._exogenous_rng = random.Random(self._seed ^ 0x5EED_E05E)
        self._tick = 0
        # Reset all mutable state.
        self._entity_ids = []
        self._entity_columns = {
            c["name"]: [] for c in self._spec.entities.columns
        }
        self._field_channels = {}
        self._registry_entries = []
        self._relation_edges = []
        self._relation_node_ids = list(self._spec.space.node_ids)
        self._alive_ids = []
        self._cumulative_births = 0
        self._cumulative_deaths = 0
        self._births_this_tick = []
        self._deaths_this_tick = []
        self._events_this_tick = []
        # Spawn slots in declared order.
        self._spawn_entities()
        if self._cap.fields:
            self._spawn_fields()
        if self._cap.registries:
            self._spawn_registries()
        if self._cap.relations:
            self._spawn_relations()
        # Initial alive set is every spawned entity.
        self._alive_ids = list(self._entity_ids)
        return self._build_state_view()

    def observe(self) -> StateView:
        return self._build_state_view()

    # ------------------------------------------------------------------
    # ObservationProjector: observe_agent (Phase 1 / Beta correction §5.3)
    # ------------------------------------------------------------------

    def observe_agent(
        self,
        agent_id: str | int,
        *,
        state: StateView | None = None,
    ) -> AgentObservationView:
        """Project the per-agent observation for ``agent_id``.

        Implements :class:`worldloop_kernel.observation.ObservationProjector`.
        The projection applies the default visibility policy:

        - Focal agent sees ALL its own columns (public + private).
        - Other agents see only :data:`_DEFAULT_PUBLIC_ENTITY_COLUMNS`
          (position / node / alive) — energy is hidden.
        - Field channels, relation edges, registry entries, events and
          population counts are PUBLIC (visible to all agents).
        - RNG state, internal caches, and other kernel-internal fields
          are NEVER projected (no field exists for them on
          :class:`AgentObservationView`).

        Counterfactual projection (``state is not None``) is NOT
        supported in v0.1 — it raises ``NotImplementedError`` (matches
        :meth:`legal_actions` behavior).

        Parameters
        ----------
        agent_id:
            ID of the focal agent. MUST be a currently-spawned entity
            (call :meth:`reset` first).
        state:
            Optional counterfactual state. NOT supported in v0.1.

        Returns
        -------
        AgentObservationView
            The authorized observation for ``agent_id``. The
            ``omission_policy`` field lists every capability slot the
            world declares as ``False`` so downstream consumers (prompt
            builder, leakage checker) can verify no hidden field was
            accidentally fabricated.
        """
        if state is not None:
            raise NotImplementedError(
                "ParameterizedWorld.observe_agent(state=...) is not "
                "supported in v0.1 (counterfactual projection is M4+)."
            )
        if agent_id not in self._entity_ids:
            raise KeyError(
                f"observe_agent: agent_id {agent_id!r} not in spawned "
                f"entities {self._entity_ids!r}; call reset(seed) first."
            )
        focal_idx = self._entity_ids.index(agent_id)

        # --- Focal agent attributes ------------------------------------
        public_attrs: dict[str, Any] = {}
        self_visible_attrs: dict[str, Any] = {}
        for col_name, values in self._entity_columns.items():
            if focal_idx >= len(values):
                continue
            value = values[focal_idx]
            if col_name in _DEFAULT_PUBLIC_ENTITY_COLUMNS:
                public_attrs[col_name] = value
            else:
                # Private column (e.g., energy) — focal-only.
                self_visible_attrs[col_name] = value
        focal_agent = FocalAgentAttributes(
            agent_id=agent_id,
            public_attributes=public_attrs,
            self_visible_attributes=self_visible_attrs,
        )

        # --- Visible entities (other agents, public columns only) -----
        visible_entities: list[VisibleEntity] = []
        for i, other_id in enumerate(self._entity_ids):
            if i == focal_idx:
                continue  # focal agent is in focal_agent, not visible_entities
            other_public: dict[str, Any] = {}
            for col_name in _DEFAULT_PUBLIC_ENTITY_COLUMNS:
                values = self._entity_columns.get(col_name)
                if values is None or i >= len(values):
                    continue
                other_public[col_name] = values[i]
            visible_entities.append(
                VisibleEntity(entity_id=other_id, columns=other_public)
            )

        # --- Visible fields (public environmental state) --------------
        visible_fields: dict[str, Any] = {}
        if self._cap.fields:
            for ch in self._spec.fields.channels:
                ch_name = ch.get("name", "")
                if ch_name in self._field_channels:
                    visible_fields[ch_name] = _freeze(
                        self._field_channels[ch_name]
                    )

        # --- Visible relations (public graph topology) ---------------
        visible_relations: tuple[VisibleRelation, ...] = ()
        if self._cap.relations:
            visible_relations = tuple(
                VisibleRelation(src=s, dst=d, edge_type=et, weight=w)
                for s, d, et, w in sorted(self._relation_edges)
            )

        # --- Visible events (public events this tick) ----------------
        visible_events: tuple[VisibleEvent, ...] = ()
        if self._cap.events:
            visible_events = tuple(
                VisibleEvent(
                    kind=ev.kind,
                    tick=ev.tick,
                    payload=dict(ev.payload),
                )
                for ev in self._events_this_tick
            )

        # --- Legal actions (from existing WorldProtocol method) ------
        action_space = self.legal_actions(agent_id)
        legal_actions = action_space.legal_actions

        # --- Omission policy (which capabilities are False) -----------
        unsupported: list[str] = []
        for slot in ("fields", "relations", "registries", "population", "events"):
            if not getattr(self._cap, slot):
                unsupported.append(slot)
        omission = OmissionPolicy(
            omitted_slots=tuple(unsupported),
            reason="capability_unavailable",
            unsupported_capabilities=tuple(unsupported),
        )

        return AgentObservationView(
            schema_version=OBSERVATION_SCHEMA_VERSION,
            scenario_id=self._spec.scenario.scenario_id,
            scenario_version=self._spec.scenario.scenario_version,
            tick=self._tick,
            focal_agent=focal_agent,
            previous_action=PreviousActionSummary(),  # v0.1: not tracked per-agent
            visible_fields=visible_fields,
            visible_entities=tuple(visible_entities),
            visible_relations=visible_relations,
            visible_events=visible_events,
            legal_actions=legal_actions,
            omission_policy=omission,
        )

    # ------------------------------------------------------------------
    # Spawn helpers
    # ------------------------------------------------------------------

    def _spawn_entities(self) -> None:
        """Initialize entities per ``spec.entities``."""
        columns = self._spec.entities.columns
        n = self._spec.entities.initial_count
        spawn_tpl = self._spec.entities.spawn_template
        space_type = self._spec.space.type
        for i in range(n):
            entity_id = f"e{i}"
            self._entity_ids.append(entity_id)
            for col in columns:
                col_name = col["name"]
                if col_name == "energy":
                    energy_range = list(spawn_tpl.get("energy_range", [0.0, 10.0]))
                    if len(energy_range) != 2:
                        energy_range = [0.0, 10.0]
                    lo, hi = float(energy_range[0]), float(energy_range[1])
                    if hi < lo:
                        lo, hi = hi, lo
                    self._entity_columns["energy"].append(self._rng.uniform(lo, hi))
                elif col_name in ("x", "y"):
                    self._spawn_position_column(col_name, space_type)
                elif col_name == "node":
                    if space_type == "graph" and self._spec.space.node_ids:
                        node = self._rng.choice(self._spec.space.node_ids)
                        self._entity_columns["node"].append(node)
                    else:
                        self._entity_columns["node"].append("")
                elif col_name == "alive":
                    self._entity_columns["alive"].append(True)
                else:
                    # Unknown column — default to 0.
                    self._entity_columns[col_name].append(0)

    def _spawn_position_column(self, col_name: str, space_type: str) -> None:
        """Spawn a position column (x or y) based on space type."""
        dim_idx = 0 if col_name == "x" else 1
        if space_type == "discrete":
            shape = self._spec.space.shape
            if dim_idx < len(shape):
                upper = int(shape[dim_idx])
                if upper <= 0:
                    self._entity_columns[col_name].append(0)
                else:
                    self._entity_columns[col_name].append(
                        self._rng.randint(0, upper - 1)
                    )
            else:
                self._entity_columns[col_name].append(0)
        elif space_type == "continuous":
            bounds = self._spec.space.bounds
            if dim_idx < len(bounds):
                lo, hi = float(bounds[dim_idx][0]), float(bounds[dim_idx][1])
                self._entity_columns[col_name].append(self._rng.uniform(lo, hi))
            else:
                self._entity_columns[col_name].append(0.0)
        else:
            # graph or unknown — x/y not applicable.
            self._entity_columns[col_name].append(0)

    def _spawn_fields(self) -> None:
        """Initialize field channels to zeros."""
        for channel in self._spec.fields.channels:
            ch_name = channel["name"]
            ch_shape = list(channel.get("shape", []))
            if not ch_shape:
                self._field_channels[ch_name] = 0.0
            elif len(ch_shape) == 1:
                self._field_channels[ch_name] = [0.0] * ch_shape[0]
            elif len(ch_shape) == 2:
                rows, cols = ch_shape
                self._field_channels[ch_name] = [
                    [0.0] * cols for _ in range(rows)
                ]
            else:
                # Higher-dim not supported in v0.1; flatten to 1D.
                total = 1
                for d in ch_shape:
                    total *= d
                self._field_channels[ch_name] = [0.0] * total

    def _spawn_registries(self) -> None:
        """Initialize registry entries from ``spec.registries.initial_entries``.

        M5 v0.2: each entry mapping is copied into the internal registry
        storage with keys (entry_id, registry_type, state, owner_id,
        metadata). Unknown keys are ignored. Entries referencing undeclared
        ``registry_types`` are still loaded (fail-closed at effect time).
        """
        for entry in self._spec.registries.initial_entries:
            self._registry_entries.append(
                {
                    "entry_id": str(entry.get("entry_id", "")),
                    "registry_type": str(entry.get("registry_type", "")),
                    "state": str(entry.get("state", "available")),
                    "owner_id": entry.get("owner_id"),
                    "metadata": dict(entry.get("metadata", {})),
                }
            )

    def _spawn_relations(self) -> None:
        """Initialize relation edges from ``space.edges`` (graph worlds)."""
        for src, dst in self._spec.space.edges:
            self._relation_edges.append((src, dst, "default", 1.0))

    # ------------------------------------------------------------------
    # WorldProtocol: legal_actions / validate_action
    # ------------------------------------------------------------------

    def legal_actions(
        self,
        agent_id: str | int,
        state: StateView | None = None,
    ) -> ActionSpace:
        if state is not None:
            raise NotImplementedError(
                "ParameterizedWorld v0.1 does not support counterfactual "
                "legal_actions(state=...)"
            )
        legal: list[LegalAction] = []
        for action_def in self._spec.actions.actions:
            # v0.1.1: generate default params for actions with
            # params_schema so that preconditions and effects referencing
            # ``$param.<name>`` can resolve. Actions with empty
            # params_schema (e.g., forage / rest) are unaffected.
            params = self._generate_default_params(action_def, agent_id)
            if self._check_preconditions(action_def, agent_id, params):
                legal.append(
                    LegalAction(
                        action_type=action_def["action_type"],
                        params=dict(params),
                        description=action_def.get("description", ""),
                    )
                )
        return ActionSpace(
            agent_id=agent_id,
            legal_actions=tuple(legal),
            is_closed=self._spec.actions.is_closed,
        )

    def _generate_default_params(
        self,
        action_def: Mapping[str, Any],
        agent_id: str | int,
    ) -> dict[str, Any]:
        """Generate default params for an action with ``params_schema``.

        v0.1 returned ``params={}`` for every action, which left
        ``$param.<name>`` references in effects/preconditions unresolved
        (resolving to ``None``). This caused ``TypeError`` on ``add`` /
        ``sub`` ops and made preconditions like ``edge_exists`` always
        fail (target node resolved to the string ``"None"``).

        v0.1.1 generates sensible defaults based on field name and
        dtype so that demo scenarios with parameterised actions
        (e.g., ``emergency_resource.yaml``) can run end-to-end without
        requiring the policy to fill in params manually.

        Default strategy (by field name, then dtype):
        - ``entry_id``: first available (non-destroyed) registry entry
          whose ``registry_type`` matches the action's effect/precondition.
        - ``target_node``: first neighbour of the agent's current node
          in the relation graph.
        - ``target_agent``: first other alive agent's id.
        - ``amount``: ``1.0``.
        - Other ``str``: empty string.
        - Other ``float``: ``1.0``.
        - Other ``int``: ``1``.

        Actions with empty ``params_schema`` return ``{}`` (unchanged
        from v0.1), so existing scenarios (e.g., ``discrete_grid``)
        are not affected.
        """
        schema = action_def.get("params_schema") or {}
        if not schema:
            return {}
        params: dict[str, Any] = {}
        # Infer the registry_type this action targets (if any) by
        # scanning effects and preconditions for ``registry_type`` or
        # by matching effect target == "registry".
        inferred_registry_type = self._infer_action_registry_type(action_def)
        for field_name, field_spec in schema.items():
            dtype = field_spec.get("dtype", "str") if isinstance(field_spec, dict) else "str"
            if field_name == "entry_id":
                params[field_name] = self._first_registry_entry_id(
                    inferred_registry_type
                )
            elif field_name == "target_node":
                params[field_name] = self._first_neighbour_node(agent_id)
            elif field_name == "target_agent":
                params[field_name] = self._first_other_agent(agent_id)
            elif field_name == "amount":
                params[field_name] = 1.0
            elif dtype == "float":
                params[field_name] = 1.0
            elif dtype == "int":
                params[field_name] = 1
            else:
                params[field_name] = ""
        return params

    def _infer_action_registry_type(
        self, action_def: Mapping[str, Any]
    ) -> str | None:
        """Infer the registry_type an action operates on, if any."""
        for effect in action_def.get("effects", ()):
            if effect.get("target") == "registry":
                # The effect's field is the state field, not the type;
                # we scan initial_entries for a match below.
                pass
        for precond in action_def.get("preconditions", ()):
            if precond.get("kind") == "registry_unowned":
                rt = precond.get("registry_type")
                if isinstance(rt, str):
                    return rt
        # Fallback: if any effect targets registry, return None (caller
        # will pick the first available entry regardless of type).
        for effect in action_def.get("effects", ()):
            if effect.get("target") == "registry":
                return None
        return None

    def _first_registry_entry_id(
        self, registry_type: str | None
    ) -> str:
        """Return the first non-destroyed registry entry id."""
        for entry in self._registry_entries:
            if entry.get("state") == "destroyed":
                continue
            if registry_type is None or entry.get("registry_type") == registry_type:
                return str(entry.get("entry_id", ""))
        # Fallback: any entry at all.
        for entry in self._registry_entries:
            return str(entry.get("entry_id", ""))
        return ""

    def _first_neighbour_node(self, agent_id: str | int) -> str:
        """Return the first neighbour of the agent's current node."""
        current = self._get_entity_column(agent_id, "node")
        if current is None:
            return ""
        current = str(current)
        for src, dst, _, _ in self._relation_edges:
            if src == current:
                return dst
            if dst == current:
                return src
        return current  # no neighbour found; return current as no-op

    def _first_other_agent(self, agent_id: str | int) -> str:
        """Return the first other alive agent's id."""
        for i, eid in enumerate(self._entity_ids):
            if eid == agent_id:
                continue
            alive_col = self._entity_columns.get("alive")
            if alive_col and i < len(alive_col) and not alive_col[i]:
                continue
            return str(eid)
        # Fallback: any other agent.
        for eid in self._entity_ids:
            if eid != agent_id:
                return str(eid)
        return ""

    def validate_action(
        self,
        proposal: ActionProposal,
    ) -> tuple[ExecutedAction, ActionReceipt]:
        action_def = self._find_action_def(proposal.action_type)
        executed = ExecutedAction(
            agent_id=proposal.agent_id,
            action_type=proposal.action_type,
            params=proposal.params,
            executed_at_tick=proposal.proposed_at_tick,
            proposal_hash=hash_state(proposal),
        )
        if action_def is None:
            if self._spec.actions.is_closed:
                receipt = ActionReceipt(
                    executed_action_hash=hash_state(executed),
                    outcome_code=OUTCOME_UNRECOGNIZED_INTENT,
                    success=False,
                    energy_delta=0.0,
                )
            else:
                receipt = ActionReceipt(
                    executed_action_hash=hash_state(executed),
                    outcome_code=OUTCOME_OK,
                    success=True,
                    energy_delta=0.0,
                )
            return executed, receipt
        if not self._check_preconditions(
            action_def, proposal.agent_id, dict(proposal.params)
        ):
            receipt = ActionReceipt(
                executed_action_hash=hash_state(executed),
                outcome_code=OUTCOME_ILLEGAL_ACTION,
                success=False,
                energy_delta=0.0,
            )
            return executed, receipt
        cost = float(action_def.get("cost", 0.0))
        receipt = ActionReceipt(
            executed_action_hash=hash_state(executed),
            outcome_code=OUTCOME_OK,
            success=True,
            energy_delta=-cost,
        )
        return executed, receipt

    def _find_action_def(self, action_type: str) -> Mapping[str, Any] | None:
        for ad in self._spec.actions.actions:
            if ad["action_type"] == action_type:
                return ad
        return None

    # ------------------------------------------------------------------
    # Preconditions
    # ------------------------------------------------------------------

    def _check_preconditions(
        self,
        action_def: Mapping[str, Any],
        agent_id: str | int,
        params: Mapping[str, Any],
    ) -> bool:
        """Return True iff all preconditions hold for ``agent_id`` + ``params``."""
        for precond in action_def.get("preconditions", ()):
            kind = precond.get("kind")
            if kind == "edge_exists":
                from_node = self._resolve_value(
                    precond.get("from", ""), agent_id, params
                )
                to_node = self._resolve_value(
                    precond.get("to", ""), agent_id, params
                )
                if not self._has_edge(str(from_node), str(to_node)):
                    return False
            elif kind == "registry_unowned":
                rt = self._resolve_value(
                    precond.get("registry_type", ""), agent_id, params
                )
                eid = self._resolve_value(
                    precond.get("entry_id", ""), agent_id, params
                )
                if not self._is_registry_unowned(str(rt), str(eid)):
                    return False
            elif kind == "energy_above":
                threshold = float(precond.get("value", 0.0))
                energy = self._get_entity_column(agent_id, "energy")
                if energy is None or energy < threshold:
                    return False
            # Unknown preconditions pass (warn-level; not enforced).
        return True

    def _has_edge(self, from_node: str, to_node: str) -> bool:
        """Check if an edge exists between two nodes (directed or undirected)."""
        for src, dst, _, _ in self._relation_edges:
            if src == from_node and dst == to_node:
                return True
            if not self._spec.relations.directed and src == to_node and dst == from_node:
                return True
        # Also check space.edges (static graph topology).
        for src, dst in self._spec.space.edges:
            if src == from_node and dst == to_node:
                return True
            if not self._spec.relations.directed and src == to_node and dst == from_node:
                return True
        return False

    def _is_registry_unowned(self, registry_type: str, entry_id: str) -> bool:
        """Return True iff the referenced registry entry has no owner.

        Returns False if the entry is not found (fail-closed).
        """
        for entry in self._registry_entries:
            if (
                entry["entry_id"] == entry_id
                and entry["registry_type"] == registry_type
            ):
                return entry.get("owner_id") is None
        return False

    # ------------------------------------------------------------------
    # WorldProtocol: step
    # ------------------------------------------------------------------

    def step(
        self,
        action: ExecutedAction,
        exogenous: ExogenousInput | None = None,
    ) -> TransitionRecord:
        before = self._build_state_view()
        # Clear per-tick accumulators.
        self._births_this_tick = []
        self._deaths_this_tick = []
        self._events_this_tick = []
        # Apply exogenous events BEFORE the action (caller-supplied).
        if exogenous is not None:
            self._apply_exogenous(exogenous)
        # Apply action effects in declared order.
        action_def = self._find_action_def(action.action_type)
        if action_def is not None:
            for effect in action_def.get("effects", ()):
                self._apply_effect(effect, action.agent_id, dict(action.params))
        # Post-step: energy floor at 0.
        if "energy" in self._entity_columns:
            energies = self._entity_columns["energy"]
            for i in range(len(energies)):
                if energies[i] < 0:
                    energies[i] = 0.0
        # Post-step: death check (only when population capability is on
        # AND the alive column exists).
        if self._cap.population and "alive" in self._entity_columns:
            self._check_deaths()
        # Advance tick.
        self._tick += 1
        after = self._build_state_view()
        # Build receipt (success path — step trusts that the action was
        # already validated; replay injects frozen ExecutedActions that
        # should have been validated at original execution time).
        cost = 0.0
        if action_def is not None:
            cost = float(action_def.get("cost", 0.0))
        receipt = ActionReceipt(
            executed_action_hash=hash_state(action),
            outcome_code=OUTCOME_OK,
            success=True,
            energy_delta=-cost,
        )
        return TransitionRecord(
            schema_version=PROTOCOL_SCHEMA_VERSION,
            producer_id=self._producer_id,
            producer_version=PRODUCER_VERSION,
            tick=before.meta.tick,
            state_before_hash=hash_state(before),
            candidate_actions={},
            executed_actions={action.agent_id: action},
            exogenous_input=exogenous,
            receipts={action.agent_id: receipt},
            state_delta=diff_state(before, after),
            state_after_hash=hash_state(after),
            capability_profile=self._cap,
            provenance={"seed": str(self._seed)},
        )

    def _check_deaths(self) -> None:
        """Mark entities with energy<=0 as dead (if ``alive`` column exists)."""
        alive_col = self._entity_columns.get("alive", [])
        energy_col = self._entity_columns.get("energy", [])
        for i, eid in enumerate(self._entity_ids):
            if eid not in self._alive_ids:
                continue
            if i >= len(alive_col) or not alive_col[i]:
                continue
            energy = energy_col[i] if i < len(energy_col) else 0.0
            if energy <= 0:
                alive_col[i] = False
                if eid in self._alive_ids:
                    self._alive_ids.remove(eid)
                self._deaths_this_tick.append(
                    DeathRecord(
                        agent_id=eid, tick=self._tick, cause="starvation"
                    )
                )
                self._cumulative_deaths += 1

    # ------------------------------------------------------------------
    # Effect interpreter
    # ------------------------------------------------------------------

    def _apply_exogenous(self, exogenous: ExogenousInput) -> None:
        """Apply a tick-scoped exogenous input.

        v0.2 supports:
        - ``kind="resource_spawn"`` with ``rate`` (float): uniformly add
          ``rate`` to every cell of every field channel.
        - ``kind="hazard_escalation"`` with ``rate`` (float) and optional
          ``field`` (str, default ``"hazard_level"``): adds ``rate`` to
          the named field channel. Used by M5 emergency scenario to
          counterbalance REPAIR's hazard reduction.
        Other kinds are recorded as events but not applied (M4 territory).
        """
        kind = exogenous.kind
        payload = exogenous.payload
        if kind == "resource_spawn" and self._field_channels:
            rate = float(payload.get("rate", 0.0))
            if rate != 0.0:
                for ch_name in list(self._field_channels.keys()):
                    self._add_to_field_channel(ch_name, rate)
            self._events_this_tick.append(
                EventRecord(kind=kind, tick=self._tick, payload=dict(payload))
            )
        elif kind == "hazard_escalation":
            rate = float(payload.get("rate", 0.0))
            target_field = str(payload.get("field", "hazard_level"))
            if rate != 0.0 and target_field in self._field_channels:
                self._add_to_field_channel(target_field, rate)
            self._events_this_tick.append(
                EventRecord(kind=kind, tick=self._tick, payload=dict(payload))
            )
        else:
            self._events_this_tick.append(
                EventRecord(kind=kind, tick=self._tick, payload=dict(payload))
            )

    def generate_exogenous(
        self,
        tick: int,
        *,
        apply: bool = False,
    ) -> ExogenousInput | None:
        """Generate an exogenous event for ``tick`` per spec rates.

        Iterates ``spec.exogenous.events`` and triggers each event whose
        ``rate`` exceeds a uniform random draw. Returns the first
        triggered :class:`ExogenousInput` (for record-keeping), or
        ``None`` if no event fired.

        Uses ``self._exogenous_rng`` (an independent stream derived from
        the seed) instead of ``self._rng``. This is critical for Q1
        Traceability: the world's primary RNG state must remain
        unchanged between ``step(t)`` and ``step(t+1)`` so that
        ``state_after_hash[t] == state_before_hash[t+1]``. Before this
        fix, ``generate_exogenous`` consumed ``self._rng`` between
        ticks, breaking the hash chain (23/147 links invalid).

        Parameters
        ----------
        tick:
            Tick at which the exogenous event would be applied.
        apply:
            If ``True``, the triggered event is also applied to world
            state via :meth:`_apply_exogenous`. If ``False`` (default),
            the event is only generated and returned — the caller (e.g.,
            the rollout orchestrator) is responsible for passing it to
            ``step``.
        """
        if self._exogenous_rng is None:
            return None
        first_triggered: ExogenousInput | None = None
        for event in self._spec.exogenous.events:
            rate = float(event.get("rate", 0.0))
            if rate <= 0.0:
                continue
            if self._exogenous_rng.random() < rate:
                kind = str(event.get("kind", ""))
                if not kind:
                    continue
                payload = {k: v for k, v in event.items() if k != "kind"}
                exo = ExogenousInput(
                    tick=tick, kind=kind, payload=payload
                )
                if apply:
                    self._apply_exogenous(exo)
                if first_triggered is None:
                    first_triggered = exo
        return first_triggered

    def _add_to_field_channel(self, ch_name: str, delta: float) -> None:
        """Add ``delta`` uniformly to every cell of field channel ``ch_name``."""
        ch_val = self._field_channels[ch_name]
        if isinstance(ch_val, (int, float)):
            self._field_channels[ch_name] = ch_val + delta
        elif isinstance(ch_val, list):
            for i, row in enumerate(ch_val):
                if isinstance(row, list):
                    for j, v in enumerate(row):
                        ch_val[i][j] = v + delta
                else:
                    ch_val[i] = row + delta

    def _apply_effect(
        self,
        effect: Mapping[str, Any],
        agent_id: str | int,
        params: Mapping[str, Any],
    ) -> None:
        """Apply a single effect descriptor to the world."""
        target = effect.get("target")
        field_name = effect.get("field", "")
        op = effect.get("op", "")
        value = self._resolve_value(effect.get("value"), agent_id, params)
        if target == "entity":
            self._apply_entity_effect(agent_id, field_name, op, value)
        elif target == "field":
            self._apply_field_effect(field_name, op, value)
        elif target == "registry":
            self._apply_registry_effect(field_name, op, value, params)
        elif target == "relation":
            self._apply_relation_effect(field_name, op, params, agent_id, value)
        # Unknown targets: silently ignored (validator already flagged them).

    def _apply_entity_effect(
        self,
        agent_id: str | int,
        col_name: str,
        op: str,
        value: Any,
    ) -> None:
        if agent_id not in self._entity_ids:
            return
        if col_name not in self._entity_columns:
            return
        idx = self._entity_ids.index(agent_id)
        current = self._entity_columns[col_name][idx]
        if op == "add":
            self._entity_columns[col_name][idx] = current + value
        elif op == "sub":
            self._entity_columns[col_name][idx] = current - value
        elif op == "set":
            self._entity_columns[col_name][idx] = value
        elif op == "clear":
            self._entity_columns[col_name][idx] = None
        elif op == "mul":
            self._entity_columns[col_name][idx] = current * value
        # Unknown ops: silently ignored.

    def _apply_field_effect(
        self,
        ch_name: str,
        op: str,
        value: Any,
    ) -> None:
        if ch_name not in self._field_channels:
            return
        if op == "add":
            self._add_to_field_channel(ch_name, float(value))
        elif op == "sub":
            self._add_to_field_channel(ch_name, -float(value))
        elif op == "set":
            # Uniform set: replace every cell with ``value``.
            ch_val = self._field_channels[ch_name]
            if isinstance(ch_val, (int, float)):
                self._field_channels[ch_name] = float(value)
            elif isinstance(ch_val, list):
                for i, row in enumerate(ch_val):
                    if isinstance(row, list):
                        for j in range(len(row)):
                            ch_val[i][j] = float(value)
                    else:
                        ch_val[i] = float(value)

    def _apply_registry_effect(
        self,
        field_name: str,
        op: str,
        value: Any,
        params: Mapping[str, Any],
    ) -> None:
        entry_id = str(params.get("entry_id", ""))
        registry_type = str(params.get("registry_type", ""))
        for entry in self._registry_entries:
            # Match by entry_id; if registry_type is also provided in
            # params, require both to match. When registry_type is empty
            # (the common case — action params_schema only declares
            # entry_id), match on entry_id alone, which is globally
            # unique per spec.registries.initial_entries.
            if entry["entry_id"] != entry_id:
                continue
            if registry_type and entry["registry_type"] != registry_type:
                continue
            if op == "set":
                if field_name == "owner":
                    entry["owner_id"] = value
                elif field_name == "state":
                    entry["state"] = str(value)
            elif op == "remove":
                entry["state"] = "destroyed"
                entry["owner_id"] = None
            break

    def _apply_relation_effect(
        self,
        edge_type: str,
        op: str,
        params: Mapping[str, Any],
        agent_id: str | int,
        value: Any,
    ) -> None:
        """Apply a relation effect (add/remove edge).

        Resolves ``src`` and ``dst`` endpoints with this priority:
        1. Explicit ``params["src"]`` / ``params["dst"]`` (literal strings).
        2. Acting agent ID for ``src`` and the resolved ``value`` for ``dst``
           (enables ``value: "$target_agent"`` to specify the destination).

        This supports the M5 SHARE / COMMUNICATE action contracts where
        the relation edge is ``($entity.id, $target_agent, communication)``.
        """
        if "src" in params:
            src = str(params["src"])
        else:
            src = str(agent_id)
        if "dst" in params:
            dst = str(params["dst"])
        elif value is not None:
            dst = str(value)
        else:
            return  # no destination, cannot add edge
        et = edge_type or "default"
        if op == "add":
            self._relation_edges.append((src, dst, et, 1.0))
        elif op == "remove":
            self._relation_edges = [
                e
                for e in self._relation_edges
                if not (e[0] == src and e[1] == dst and e[2] == et)
            ]

    # ------------------------------------------------------------------
    # Value resolution
    # ------------------------------------------------------------------

    def _resolve_value(
        self,
        value: Any,
        agent_id: str | int,
        params: Mapping[str, Any],
    ) -> Any:
        """Resolve a value reference (literal or ``$``-prefixed variable)."""
        if not isinstance(value, str):
            return value
        if value.startswith("$param."):
            return params.get(value[len("$param.") :])
        if value.startswith("$entity."):
            attr = value[len("$entity.") :]
            if attr == "id":
                return agent_id
            return self._get_entity_column(agent_id, attr)
        if value.startswith("$"):
            return params.get(value[1:])
        return value

    def _get_entity_column(
        self, entity_id: str | int, column_name: str
    ) -> Any:
        if entity_id not in self._entity_ids:
            return None
        if column_name not in self._entity_columns:
            return None
        idx = self._entity_ids.index(entity_id)
        return self._entity_columns[column_name][idx]

    # ------------------------------------------------------------------
    # StateView construction
    # ------------------------------------------------------------------

    def _build_state_view(self) -> StateView:
        entity_table = EntityTable(
            schema_id=f"{self._producer_id}:entities:v1",
            ids=tuple(self._entity_ids),
            columns={
                col: tuple(values)
                for col, values in self._entity_columns.items()
            },
        )
        field_state: FieldState | None = None
        if self._cap.fields:
            field_state = FieldState(
                schema_id=f"{self._producer_id}:fields:v1",
                channels={
                    ch.get("name", ""): _freeze(self._field_channels.get(ch.get("name", "")))
                    for ch in self._spec.fields.channels
                },
                units={
                    ch.get("name", ""): ch.get("unit", "")
                    for ch in self._spec.fields.channels
                },
            )
        relation_graph: RelationGraph | None = None
        if self._cap.relations:
            edges_tuple = tuple(
                RelationEdge(src=s, dst=d, edge_type=et, weight=w)
                for s, d, et, w in sorted(self._relation_edges)
            )
            relation_graph = RelationGraph(
                schema_id=f"{self._producer_id}:relations:v1",
                node_ids=tuple(self._relation_node_ids),
                edges=edges_tuple,
            )
        registry_snapshot: RegistrySnapshot | None = None
        if self._cap.registries:
            entries_tuple = tuple(
                RegistryEntry(
                    entry_id=e["entry_id"],
                    registry_type=e["registry_type"],
                    state=e["state"],
                    owner_id=e.get("owner_id"),
                    metadata=dict(e.get("metadata", {})),
                )
                for e in sorted(
                    self._registry_entries,
                    key=lambda x: (x["registry_type"], x["entry_id"]),
                )
            )
            registry_snapshot = RegistrySnapshot(
                schema_id=f"{self._producer_id}:registries:v1",
                entries=entries_tuple,
            )
        population_state: PopulationState | None = None
        if self._cap.population:
            population_state = PopulationState(
                alive_ids=tuple(self._alive_ids),
                births_this_tick=tuple(self._births_this_tick),
                deaths_this_tick=tuple(self._deaths_this_tick),
                cumulative_births=self._cumulative_births,
                cumulative_deaths=self._cumulative_deaths,
            )
        event_context: EventContext | None = None
        if self._cap.events:
            event_context = EventContext(events=tuple(self._events_this_tick))
        rng_state_ref: str | None = None
        if self._rng is not None:
            state = self._rng.getstate()
            # state[1] is a tuple of ints; take first 4 as a fingerprint.
            if isinstance(state, tuple) and len(state) >= 2:
                head = state[1][:4] if isinstance(state[1], tuple) else ()
                rng_state_ref = f"MT19937:head={head}"
            else:
                rng_state_ref = "MT19937:unknown"
        meta = StateMeta(
            scenario_id=self._spec.scenario.scenario_id,
            run_id=f"seed-{self._seed}",
            tick=self._tick,
            config_hash=self._spec.world_parameters_hash(),
            rng_state_ref=rng_state_ref,
        )
        # missing_mask: all False for slots the world has (every present
        # slot has its value); absent slots are not in the mask.
        missing_mask = {
            slot: False
            for slot in ("fields", "entities", "relations", "registries",
                         "population", "events")
            if getattr(self._cap, slot)
        }
        return StateView(
            meta=meta,
            entities=entity_table,
            capabilities=self._cap,
            missing_mask=missing_mask,
            fields=field_state,
            relations=relation_graph,
            registries=registry_snapshot,
            population=population_state,
            events=event_context,
        )

    # ------------------------------------------------------------------
    # WorldProtocol: checkpoint / restore
    # ------------------------------------------------------------------

    def checkpoint(self) -> Checkpoint:
        state_view = self._build_state_view()
        rng_state = self._rng.getstate() if self._rng is not None else None
        exo_rng_state = (
            self._exogenous_rng.getstate()
            if self._exogenous_rng is not None
            else None
        )
        payload_dict = {
            "tick": self._tick,
            "seed": self._seed,
            "rng_state": rng_state,
            "exogenous_rng_state": exo_rng_state,
            "entity_ids": self._entity_ids,
            "entity_columns": self._entity_columns,
            "field_channels": self._field_channels,
            "registry_entries": self._registry_entries,
            "relation_edges": self._relation_edges,
            "relation_node_ids": self._relation_node_ids,
            "alive_ids": self._alive_ids,
            "cumulative_births": self._cumulative_births,
            "cumulative_deaths": self._cumulative_deaths,
            "producer_id": self._producer_id,
            # Per-tick accumulators. These MUST be saved/restored so
            # that ``state_after_hash[t]`` matches ``state_before_hash[t+1]``
            # when a checkpoint/restore cycle occurs between two ticks
            # (e.g., when the counterfactual brancher is enabled). Without
            # this, the brancher's ``world.restore(parent_saved)`` resets
            # these lists to ``[]``, causing the next tick's
            # ``state_before_hash`` to differ from the previous tick's
            # ``state_after_hash`` (Q1 Traceability failure).
            "events_this_tick": list(self._events_this_tick),
            "births_this_tick": list(self._births_this_tick),
            "deaths_this_tick": list(self._deaths_this_tick),
        }
        payload = pickle.dumps(payload_dict)
        checkpoint = Checkpoint(
            schema_version=PROTOCOL_SCHEMA_VERSION,
            world_id=self._producer_id,
            world_version=PRODUCER_VERSION,
            tick=self._tick,
            state_view=state_view,
            opaque_payload=payload,
            payload_codec=PAYLOAD_CODEC,
            capability_profile=self._cap,
            rng_bundle={"main": _rng_to_str(rng_state)} if rng_state is not None else None,
            # checksum is filled after construction (it depends on the
            # fields above, all of which are now finalized).
            checksum="pending",
        )
        checksum = compute_checkpoint_checksum(checkpoint)
        # Checkpoint is frozen; bypass via object.__setattr__ to set the
        # final checksum computed over the now-final payload.
        object.__setattr__(checkpoint, "checksum", checksum)
        return checkpoint

    def restore(self, checkpoint: Checkpoint) -> None:
        if checkpoint.payload_codec != PAYLOAD_CODEC:
            raise ValueError(
                f"Cannot restore checkpoint with codec "
                f"{checkpoint.payload_codec!r}; expected {PAYLOAD_CODEC!r}"
            )
        payload_dict = pickle.loads(checkpoint.opaque_payload)
        self._tick = int(payload_dict["tick"])
        self._seed = int(payload_dict["seed"])
        rng_state = payload_dict["rng_state"]
        self._rng = random.Random()
        if rng_state is not None:
            self._rng.setstate(rng_state)
        # Restore the exogenous RNG. Backwards-compatible: if the
        # checkpoint was created before the exogenous RNG existed,
        # derive a fresh one from the seed (the exogenous stream will
        # restart from the beginning, which is acceptable for legacy
        # checkpoints).
        exo_rng_state = payload_dict.get("exogenous_rng_state")
        self._exogenous_rng = random.Random()
        if exo_rng_state is not None:
            self._exogenous_rng.setstate(exo_rng_state)
        else:
            self._exogenous_rng = random.Random(self._seed ^ 0x5EED_E05E)
        self._entity_ids = list(payload_dict["entity_ids"])
        # Rebuild entity_columns as dict[str, list[Any]] (lists are mutable).
        self._entity_columns = {
            col: list(values)
            for col, values in payload_dict["entity_columns"].items()
        }
        self._field_channels = dict(payload_dict["field_channels"])
        self._registry_entries = list(payload_dict["registry_entries"])
        self._relation_edges = list(payload_dict["relation_edges"])
        self._relation_node_ids = list(payload_dict["relation_node_ids"])
        self._alive_ids = list(payload_dict["alive_ids"])
        self._cumulative_births = int(payload_dict["cumulative_births"])
        self._cumulative_deaths = int(payload_dict["cumulative_deaths"])
        self._producer_id = payload_dict.get(
            "producer_id", self._producer_id
        )
        # Restore per-tick accumulators. Backwards-compatible: legacy
        # checkpoints created before this field existed default to empty
        # lists (which matches the previous behavior, so old checkpoints
        # still restore without error — they just don't carry forward
        # the previous tick's events).
        self._events_this_tick = list(payload_dict.get("events_this_tick", []))
        self._births_this_tick = list(payload_dict.get("births_this_tick", []))
        self._deaths_this_tick = list(payload_dict.get("deaths_this_tick", []))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _freeze(value: Any) -> Any:
    """Recursively convert lists to tuples for canonical encoding.

    ``canonical_encode`` does not handle ``list``; we freeze any list
    encountered in a field channel value into a tuple (recursively for
    nested lists) so :class:`FieldState.channels` is canonical-friendly.
    """
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, dict):
        return {k: _freeze(v) for k, v in value.items()}
    return value


def _rng_to_str(rng_state: Any) -> str:
    """Stringify a ``random.Random.getstate()`` tuple for the rng_bundle.

    The state is a tuple ``(version, internaltuple, gauss_next)``. We
    produce a stable, human-readable string fingerprint.
    """
    if not isinstance(rng_state, tuple) or len(rng_state) < 2:
        return "unknown"
    version = rng_state[0]
    internal = rng_state[1]
    if isinstance(internal, tuple) and internal:
        head = internal[:8]
        return f"v{version}:head={head}"
    return f"v{version}:empty"
