"""Phase 1 / Beta correction: tests for ``worldloop_kernel.observation``.

Covers the schema-level invariants required by
``docs/07.advice/2026-07-30_WorldLoop主线实验有效性与Beta发布优化实施方案.md``
§5.2 / §5.3 and the Prompt Gate P-G2 preconditions:

- AgentObservationView is frozen; all nested types are frozen.
- hash_observation is deterministic and SHA-256 prefixed.
- Hash changes when ANY visible field changes (focal_agent /
  visible_fields / visible_entities / visible_relations /
  visible_events / legal_actions / previous_action / omission_policy /
  tick / scenario).
- Schema carries NO hidden-state field (no ``rng_state``, no
  ``internal_cache``, no ``private_*``) — this is the structural
  precondition for P-G2 ("same observation + different hidden state
  => same prompt hash"). The full P-G2 test runs in
  ``worldloop_data/tests/test_prompt_gate.py`` after the prompt
  builder lands; here we only verify the schema cannot leak.
- Unsupported capabilities are omitted (empty), NOT zero-filled.
- ObservationProjector is runtime_checkable.
- All public symbols are re-exported from the top-level package.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any

import pytest

from worldloop_kernel import (
    OBSERVATION_SCHEMA_VERSION,
    OmissionPolicy,
    VisibleEntity,
    VisibleRelation,
    VisibleEvent,
    FocalAgentAttributes,
    PreviousActionSummary,
    AgentObservationView,
    ObservationProjector,
    hash_observation,
    is_observation_projector,
)


# ---------------------------------------------------------------------------
# Helpers — minimal valid instances
# ---------------------------------------------------------------------------


def make_focal_agent(
    *,
    agent_id: str | int = "agent_0",
    public_attributes: dict[str, Any] | None = None,
    self_visible_attributes: dict[str, Any] | None = None,
) -> FocalAgentAttributes:
    return FocalAgentAttributes(
        agent_id=agent_id,
        public_attributes=public_attributes or {"position": (0, 0)},
        self_visible_attributes=self_visible_attributes or {"energy": 100},
    )


def make_observation(
    *,
    schema_version: str = OBSERVATION_SCHEMA_VERSION,
    scenario_id: str = "test_scenario",
    scenario_version: str = "0.1.0",
    tick: int = 0,
    focal_agent: FocalAgentAttributes | None = None,
    previous_action: PreviousActionSummary | None = None,
    visible_fields: dict[str, Any] | None = None,
    visible_entities: tuple[VisibleEntity, ...] = (),
    visible_relations: tuple[VisibleRelation, ...] = (),
    visible_events: tuple[VisibleEvent, ...] = (),
    legal_actions: tuple = (),
    omission_policy: OmissionPolicy | None = None,
) -> AgentObservationView:
    return AgentObservationView(
        schema_version=schema_version,
        scenario_id=scenario_id,
        scenario_version=scenario_version,
        tick=tick,
        focal_agent=focal_agent or make_focal_agent(),
        previous_action=previous_action or PreviousActionSummary(),
        visible_fields=visible_fields or {},
        visible_entities=visible_entities,
        visible_relations=visible_relations,
        visible_events=visible_events,
        legal_actions=legal_actions,
        omission_policy=omission_policy or OmissionPolicy(),
    )


# ---------------------------------------------------------------------------
# Schema invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls",
    [
        OmissionPolicy,
        VisibleEntity,
        VisibleRelation,
        VisibleEvent,
        FocalAgentAttributes,
        PreviousActionSummary,
        AgentObservationView,
    ],
)
def test_all_observation_types_are_frozen(cls):
    """Every observation dataclass MUST be frozen so its hash is stable."""
    assert dataclasses.is_dataclass(cls)
    params = getattr(cls, "__dataclass_params__", None)
    assert params is not None, f"{cls.__name__} missing __dataclass_params__"
    assert params.frozen is True, f"{cls.__name__} must be frozen=True"


def test_observation_schema_version_is_nonempty_semverish():
    assert isinstance(OBSERVATION_SCHEMA_VERSION, str)
    assert OBSERVATION_SCHEMA_VERSION, "schema version must not be empty"
    # Accept "0.1.0" / "1.2.3" / "0.1.0-beta" etc.
    assert re.match(
        r"^\d+\.\d+\.\d+", OBSERVATION_SCHEMA_VERSION
    ), f"schema version {OBSERVATION_SCHEMA_VERSION!r} must start with semver"


# ---------------------------------------------------------------------------
# Hash determinism
# ---------------------------------------------------------------------------


def test_hash_observation_is_deterministic_same_content():
    """Same visible content => same hash (P-G1 / P-G2 precondition)."""
    obs_a = make_observation()
    obs_b = make_observation()
    assert hash_observation(obs_a) == hash_observation(obs_b)


def test_hash_observation_format_is_sha256_prefixed_hex():
    obs = make_observation()
    h = hash_observation(obs)
    assert h.startswith("sha256:"), f"hash must start with 'sha256:', got {h!r}"
    hex_part = h[len("sha256:"):]
    assert len(hex_part) == 64, f"hex digest must be 64 chars, got {len(hex_part)}"
    assert re.match(r"^[0-9a-f]{64}$", hex_part), f"hex digest must be lowercase hex, got {hex_part!r}"


# ---------------------------------------------------------------------------
# Hash sensitivity — visible content changes MUST change the hash
# ---------------------------------------------------------------------------


def test_hash_changes_when_tick_changes():
    obs_a = make_observation(tick=0)
    obs_b = make_observation(tick=1)
    assert hash_observation(obs_a) != hash_observation(obs_b)


def test_hash_changes_when_focal_agent_id_changes():
    obs_a = make_observation(focal_agent=make_focal_agent(agent_id="a0"))
    obs_b = make_observation(focal_agent=make_focal_agent(agent_id="a1"))
    assert hash_observation(obs_a) != hash_observation(obs_b)


def test_hash_changes_when_focal_public_attribute_changes():
    """Visible focal state changes MUST change the hash (P-G1)."""
    obs_a = make_observation(
        focal_agent=make_focal_agent(public_attributes={"position": (0, 0)})
    )
    obs_b = make_observation(
        focal_agent=make_focal_agent(public_attributes={"position": (1, 0)})
    )
    assert hash_observation(obs_a) != hash_observation(obs_b)


def test_hash_changes_when_self_visible_attribute_changes():
    """Self-visible focal state (e.g., private energy) changes MUST change
    the hash — the focal agent IS allowed to see its own private state."""
    obs_a = make_observation(
        focal_agent=make_focal_agent(self_visible_attributes={"energy": 100})
    )
    obs_b = make_observation(
        focal_agent=make_focal_agent(self_visible_attributes={"energy": 50})
    )
    assert hash_observation(obs_a) != hash_observation(obs_b)


def test_hash_changes_when_visible_fields_change():
    obs_a = make_observation(visible_fields={"terrain": "plain"})
    obs_b = make_observation(visible_fields={"terrain": "forest"})
    assert hash_observation(obs_a) != hash_observation(obs_b)


def test_hash_changes_when_visible_entities_change():
    e_a = VisibleEntity(entity_id="r0", columns={"type": "resource"})
    e_b = VisibleEntity(entity_id="r0", columns={"type": "threat"})
    obs_a = make_observation(visible_entities=(e_a,))
    obs_b = make_observation(visible_entities=(e_b,))
    assert hash_observation(obs_a) != hash_observation(obs_b)


def test_hash_changes_when_visible_entity_count_changes():
    e_a = VisibleEntity(entity_id="r0", columns={"type": "resource"})
    e_b = VisibleEntity(entity_id="r1", columns={"type": "resource"})
    obs_a = make_observation(visible_entities=(e_a,))
    obs_b = make_observation(visible_entities=(e_a, e_b))
    assert hash_observation(obs_a) != hash_observation(obs_b)


def test_hash_changes_when_visible_relations_change():
    r_a = VisibleRelation(src="a0", dst="r0", edge_type="visible")
    r_b = VisibleRelation(src="a0", dst="r0", edge_type="hidden_marker")
    obs_a = make_observation(visible_relations=(r_a,))
    obs_b = make_observation(visible_relations=(r_b,))
    assert hash_observation(obs_a) != hash_observation(obs_b)


def test_hash_changes_when_visible_events_change():
    ev_a = VisibleEvent(kind="resource_depleted", tick=0)
    ev_b = VisibleEvent(kind="threat_spawned", tick=0)
    obs_a = make_observation(visible_events=(ev_a,))
    obs_b = make_observation(visible_events=(ev_b,))
    assert hash_observation(obs_a) != hash_observation(obs_b)


def test_hash_changes_when_legal_actions_change():
    from worldloop_kernel import LegalAction

    la_a = LegalAction(action_type="MOVE", params={"target": "zone_a"})
    la_b = LegalAction(action_type="REST")
    obs_a = make_observation(legal_actions=(la_a,))
    obs_b = make_observation(legal_actions=(la_b,))
    assert hash_observation(obs_a) != hash_observation(obs_b)


def test_hash_changes_when_previous_action_changes():
    obs_a = make_observation(previous_action=PreviousActionSummary(action_type="MOVE"))
    obs_b = make_observation(previous_action=PreviousActionSummary(action_type="REST"))
    assert hash_observation(obs_a) != hash_observation(obs_b)


def test_hash_changes_when_omission_policy_changes():
    """Omission policy is metadata about the projection; changing it
    changes the hash so consumers can detect policy drift."""
    obs_a = make_observation(
        omission_policy=OmissionPolicy(
            omitted_slots=("registries",),
            reason="capability_unavailable",
            unsupported_capabilities=("registries",),
        )
    )
    obs_b = make_observation(
        omission_policy=OmissionPolicy(
            omitted_slots=("population", "registries"),
            reason="capability_unavailable",
            unsupported_capabilities=("population", "registries"),
        )
    )
    assert hash_observation(obs_a) != hash_observation(obs_b)


def test_hash_changes_when_scenario_changes():
    obs_a = make_observation(scenario_id="emergency_v0")
    obs_b = make_observation(scenario_id="emergency_v1")
    assert hash_observation(obs_a) != hash_observation(obs_b)


# ---------------------------------------------------------------------------
# Hidden-state non-leak at schema level (P-G2 structural precondition)
# ---------------------------------------------------------------------------


def test_agent_observation_view_has_no_hidden_state_field_names():
    """The schema MUST NOT carry any field that could plausibly hold
    hidden world state. This is the structural precondition for P-G2:
    if the schema has no hidden-state field, then "same observation +
    different hidden state" is impossible by construction."""
    obs_fields = {f.name for f in dataclasses.fields(AgentObservationView)}
    forbidden_patterns = (
        "rng_state", "rng_seed", "internal_rng",
        "internal_cache", "cache",
        "private_", "hidden_",
        "world_", "_world",
        "checkpoint",  # Checkpoint is restorable world state, NOT visible
        "python_hash", "hash_seed",
    )
    leaked = [
        name for name in obs_fields
        if any(pat in name.lower() for pat in forbidden_patterns)
    ]
    assert not leaked, (
        f"AgentObservationView has fields that could leak hidden state: {leaked}; "
        f"all fields: {sorted(obs_fields)}"
    )


def test_agent_observation_view_field_set_matches_design_spec():
    """Field set MUST match §5.2 of the Beta correction plan."""
    expected = {
        "schema_version",
        "scenario_id",
        "scenario_version",
        "tick",
        "focal_agent",
        "previous_action",
        "visible_fields",
        "visible_entities",
        "visible_relations",
        "visible_events",
        "legal_actions",
        "omission_policy",
    }
    actual = {f.name for f in dataclasses.fields(AgentObservationView)}
    assert actual == expected, (
        f"field set drifted; expected {sorted(expected)}, got {sorted(actual)}"
    )


def test_no_fake_zero_for_unsupported_capability():
    """When a capability is unsupported, the corresponding visible_*
    collection MUST be empty, NOT filled with placeholder zeros
    (§5.3: 'unsupported capability must be omitted or marked
    unavailable, not补伪零')."""
    obs = make_observation(
        visible_fields={},
        visible_entities=(),
        visible_relations=(),
        visible_events=(),
        omission_policy=OmissionPolicy(
            omitted_slots=("fields", "relations", "registries", "population", "events"),
            reason="capability_unavailable",
            unsupported_capabilities=(
                "fields", "relations", "registries", "population", "events",
            ),
        ),
    )
    assert obs.visible_fields == {}
    assert obs.visible_entities == ()
    assert obs.visible_relations == ()
    assert obs.visible_events == ()
    assert "fields" in obs.omission_policy.unsupported_capabilities
    assert "registries" in obs.omission_policy.unsupported_capabilities


def test_observation_hash_does_not_depend_on_world_internals():
    """Because AgentObservationView has no hidden-state field, two
    observations with identical visible content produce identical
    hashes — regardless of what hidden world state produced them.

    This simulates P-G2 at the schema level: build two observations
    from 'different worlds' (mocked) that agree on visible content;
    verify hashes match. The full P-G2 test runs in
    worldloop_data/tests/test_prompt_gate.py against a real projector.
    """
    # World A: 'hidden' state = {"internal_rng": 42, "private_cache": {...}}
    # World B: 'hidden' state = {"internal_rng": 99, "private_cache": {...}}
    # Both worlds project to the SAME AgentObservationView — that is
    # the projector's contract. The schema test just verifies the
    # hash is determined by the visible content alone.
    obs_from_world_a = make_observation(
        focal_agent=make_focal_agent(
            public_attributes={"position": (3, 4)},
            self_visible_attributes={"energy": 75},
        ),
        visible_fields={"terrain": "forest"},
        visible_entities=(
            VisibleEntity(entity_id="r0", columns={"type": "resource", "amount": 10}),
        ),
    )
    # Build an 'identical' observation from 'world B' — same visible content
    obs_from_world_b = AgentObservationView(
        schema_version=obs_from_world_a.schema_version,
        scenario_id=obs_from_world_a.scenario_id,
        scenario_version=obs_from_world_a.scenario_version,
        tick=obs_from_world_a.tick,
        focal_agent=FocalAgentAttributes(
            agent_id=obs_from_world_a.focal_agent.agent_id,
            public_attributes=dict(obs_from_world_a.focal_agent.public_attributes),
            self_visible_attributes=dict(obs_from_world_a.focal_agent.self_visible_attributes),
        ),
        previous_action=PreviousActionSummary(
            action_type=obs_from_world_a.previous_action.action_type,
            success=obs_from_world_a.previous_action.success,
            outcome_code=obs_from_world_a.previous_action.outcome_code,
            visible_effect_summary=dict(obs_from_world_a.previous_action.visible_effect_summary),
        ),
        visible_fields=dict(obs_from_world_a.visible_fields),
        visible_entities=obs_from_world_a.visible_entities,
        visible_relations=obs_from_world_a.visible_relations,
        visible_events=obs_from_world_a.visible_events,
        legal_actions=obs_from_world_a.legal_actions,
        omission_policy=OmissionPolicy(
            omitted_slots=obs_from_world_a.omission_policy.omitted_slots,
            reason=obs_from_world_a.omission_policy.reason,
            unsupported_capabilities=obs_from_world_a.omission_policy.unsupported_capabilities,
        ),
    )
    assert hash_observation(obs_from_world_a) == hash_observation(obs_from_world_b)


# ---------------------------------------------------------------------------
# ObservationProjector Protocol
# ---------------------------------------------------------------------------


def test_observation_projector_is_runtime_checkable():
    """ObservationProjector MUST be runtime_checkable so the data layer
    can isinstance-check worlds before calling observe_agent."""
    # isinstance() against a non-runtime_checkable Protocol raises
    # TypeError; if the assertion below passes, the decorator was applied.
    class _World:
        def observe_agent(self, agent_id, *, state=None):
            return make_observation()

    assert isinstance(_World(), ObservationProjector)


def test_is_observation_projector_true_for_implementor():
    class _World:
        def observe_agent(self, agent_id, *, state=None):
            return make_observation()

    assert is_observation_projector(_World()) is True


def test_is_observation_projector_false_for_non_implementor():
    class _Bare:
        pass

    assert is_observation_projector(_Bare()) is False
    assert is_observation_projector(42) is False
    assert is_observation_projector("not a world") is False
    assert is_observation_projector(None) is False


def test_is_observation_projector_false_for_world_without_observe_agent():
    """A world that only implements WorldProtocol (legacy) is NOT an
    ObservationProjector — this is how the data layer falls back to
    the 'no observation' path."""
    class _LegacyWorld:
        def capabilities(self):
            ...
        def reset(self, seed, parameters=None):
            ...
        def observe(self):
            ...
        def legal_actions(self, agent_id, state=None):
            ...
        def validate_action(self, proposal):
            ...
        def step(self, action, exogenous=None):
            ...
        def checkpoint(self):
            ...
        def restore(self, checkpoint):
            ...

    assert is_observation_projector(_LegacyWorld()) is False


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


def test_all_public_symbols_importable_from_top_level():
    """All public symbols MUST be importable from ``worldloop_kernel``
    directly so consumers don't reach into submodules."""
    import worldloop_kernel as wk

    for name in (
        "OBSERVATION_SCHEMA_VERSION",
        "OmissionPolicy",
        "VisibleEntity",
        "VisibleRelation",
        "VisibleEvent",
        "FocalAgentAttributes",
        "PreviousActionSummary",
        "AgentObservationView",
        "ObservationProjector",
        "hash_observation",
        "is_observation_projector",
    ):
        assert hasattr(wk, name), f"worldloop_kernel missing public symbol {name!r}"
        assert name in wk.__all__, f"{name!r} not in worldloop_kernel.__all__"


def test_hash_observation_callable_with_minimal_observation():
    """A minimal observation with all defaults MUST hash without error."""
    obs = AgentObservationView(
        schema_version=OBSERVATION_SCHEMA_VERSION,
        scenario_id="smoke",
        scenario_version="0.0.1",
        tick=0,
        focal_agent=FocalAgentAttributes(agent_id="a0"),
    )
    h = hash_observation(obs)
    assert h.startswith("sha256:")
