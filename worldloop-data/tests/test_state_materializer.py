"""Tests for the state materialization layer (audit F-03 / R2 fix).

Covers the four R2 acceptance criteria:

1. Mechanical traceability — every TrainingTransition carries provenance
   with episode_id / tick / source_record_line / source_record_sha256.
2. Time alignment — state_before + joint_action + exogenous all come
   from the same tick's record.
3. diff/apply round-trip — applying sequential state_delta diffs from
   the initial state produces a final state hash equal to the recorded
   state_after_hash (tested via state_view equality, since the
   materializer uses dict-level diff/apply).
4. No silent degradation — when initial_state_view is None, the
   materializer raises MaterializerError instead of emitting
   zero-filled samples.

Also covers F-03 specific remediation:
- Joint action encodes ALL agents (not just first).
- Exogenous input enters features.
- Field/entity/graph/registry/population blocks are materialized.
- Missing mask marks absent capabilities (no faked zeros).
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest

from worldloop_data.evaluation.state_materializer import (
    EncoderSchema,
    ExogenousFeatures,
    JointActionFeatures,
    MaterializerError,
    StateBlockType,
    StateEncoder,
    StateFeatures,
    StateMaterializer,
    TrainingProvenance,
    TrainingTransition,
)


# ---------------------------------------------------------------------------
# Schema + encoder fixtures
# ---------------------------------------------------------------------------


def _full_schema() -> EncoderSchema:
    """Schema with all capabilities declared and non-zero dims."""
    return EncoderSchema(
        field_dim=32,
        entity_dim=16,
        graph_dim=8,
        registry_dim=8,
        population_dim=3,
        has_fields=True,
        has_entities=True,
        has_relations=True,
        has_registries=True,
        has_population=True,
    )


def _partial_schema() -> EncoderSchema:
    """Schema with only entities + relations (no fields/registries/population)."""
    return EncoderSchema(
        field_dim=0,
        entity_dim=16,
        graph_dim=8,
        registry_dim=0,
        population_dim=0,
        has_fields=False,
        has_entities=True,
        has_relations=True,
        has_registries=False,
        has_population=False,
    )


def _initial_state_view() -> dict[str, Any]:
    """A minimal StateView-like dict for testing."""
    return {
        "meta": {"tick": 0, "world_id": "test-world"},
        "fields": {
            "energy": {"value": 100.0},
            "hazard_level": {"value": "low"},
        },
        "entities": {
            "rows": [
                {"entity_id": "e0", "values": [10.0, "base"]},
                {"entity_id": "e1", "values": [20.0, "zone_a"]},
            ],
        },
        "relations": {
            "edges": [
                {"src": "base", "dst": "zone_a", "edge_type": "road", "weight": 1.0},
            ],
        },
        "registries": {
            "entries": [
                {"entry_id": "r0", "registry_type": "resource", "state": "available"},
            ],
        },
        "population": {"alive_ids": ["e0", "e1"]},
    }


def _transition_record(
    tick: int,
    *,
    executed_actions: dict[str, Any] | None = None,
    exogenous_input: dict[str, Any] | None = None,
    state_delta: dict[str, Any] | None = None,
    state_before_hash: str = "hash_before",
    state_after_hash: str = "hash_after",
    episode_id: str = "test_ep",
    seed: str = "42",
    policy_id: str = "random",
) -> dict[str, Any]:
    """Build a minimal transition record dict for testing.

    ``state_delta=None`` is preserved as ``None`` in the output record
    (NOT converted to ``{}``) so the materializer's "no state_delta"
    branch is exercised. Pass an explicit ``{}`` or a populated dict to
    get a non-None state_delta.
    """
    return {
        "schema_version": "0.1.0",
        "producer_id": "test-world",
        "producer_version": "0.1.0",
        "tick": tick,
        "state_before_hash": state_before_hash,
        "candidate_actions": executed_actions or {},
        "executed_actions": executed_actions or {},
        "exogenous_input": exogenous_input,
        "receipts": {
            aid: {"energy_delta": -1.0}
            for aid in (executed_actions or {})
        },
        "state_delta": state_delta,  # preserve None to test chain-break
        "state_after_hash": state_after_hash,
        "capability_profile": {
            "fields": True,
            "entities": True,
            "relations": True,
            "registries": True,
            "population": True,
            "events": False,
            "exact_restore": True,
            "executable_deterministic_replay": True,
            "authority": "rule",
            "ground_truth": True,
            "transition_mode": "deterministic",
        },
        "provenance": {
            "episode_id": episode_id,
            "seed": seed,
            "policy_id": policy_id,
        },
    }


def _build_materializer(
    schema: EncoderSchema | None = None,
    *,
    action_types: tuple[str, ...] = ("MOVE", "COLLECT", "REST"),
    agent_ids: tuple[str, ...] = ("e0", "e1"),
    hash_buckets: int = 64,
) -> StateMaterializer:
    """Build a StateMaterializer with the given schema + vocabs.

    ``hash_buckets`` defaults to 64 (larger than the default 8) to
    minimize hash collisions in tests that verify a block CHANGED
    between two states — collisions make two different values produce
    identical feature vectors, breaking the assertion.
    """
    schema = schema or _full_schema()
    encoder = StateEncoder(schema, hash_buckets=hash_buckets)
    return StateMaterializer(
        encoder=encoder,
        action_types=action_types,
        agent_ids=agent_ids,
    )


# ---------------------------------------------------------------------------
# EncoderSchema tests
# ---------------------------------------------------------------------------


class TestEncoderSchema:
    def test_encoded_dim_sums_only_declared_capabilities(self):
        schema = _partial_schema()
        # Only entities (16) + relations (8) declared.
        assert schema.encoded_dim() == 24

    def test_encoded_dim_full_schema(self):
        schema = _full_schema()
        # 32 + 16 + 8 + 8 + 3 = 67
        assert schema.encoded_dim() == 67

    def test_missing_mask_marks_absent_capabilities(self):
        schema = _partial_schema()
        mask = schema.missing_mask()
        assert mask[StateBlockType.FIELD.value] is True
        assert mask[StateBlockType.ENTITY.value] is False
        assert mask[StateBlockType.GRAPH.value] is False
        assert mask[StateBlockType.REGISTRY.value] is True
        assert mask[StateBlockType.POPULATION.value] is True

    def test_missing_mask_full_schema_all_false(self):
        schema = _full_schema()
        mask = schema.missing_mask()
        assert all(v is False for v in mask.values())


# ---------------------------------------------------------------------------
# StateEncoder tests
# ---------------------------------------------------------------------------


class TestStateEncoder:
    def test_encode_full_state_view_returns_all_blocks(self):
        encoder = StateEncoder(_full_schema(), hash_buckets=8)
        features = encoder.encode(_initial_state_view())
        assert features.field_block is not None
        assert features.entity_block is not None
        assert features.graph_block is not None
        assert features.registry_block is not None
        assert features.population_block is not None

    def test_encode_none_state_view_returns_all_none_blocks(self):
        encoder = StateEncoder(_full_schema(), hash_buckets=8)
        features = encoder.encode(None)
        assert features.field_block is None
        assert features.entity_block is None
        assert features.graph_block is None
        assert features.registry_block is None
        assert features.population_block is None
        # Missing mask should be all True (capability absent in this state).
        assert all(v is True for v in features.missing_mask.values())

    def test_partial_schema_skips_absent_blocks(self):
        encoder = StateEncoder(_partial_schema(), hash_buckets=8)
        features = encoder.encode(_initial_state_view())
        # Fields / registries / population absent (capability=False).
        assert features.field_block is None
        assert features.registry_block is None
        assert features.population_block is None
        # Entities + relations present.
        assert features.entity_block is not None
        assert features.graph_block is not None
        # Missing mask correctly reflects.
        assert features.missing_mask[StateBlockType.FIELD.value] is True
        assert features.missing_mask[StateBlockType.ENTITY.value] is False

    def test_field_block_has_declared_dim(self):
        encoder = StateEncoder(_full_schema(), hash_buckets=8)
        features = encoder.encode(_initial_state_view())
        assert features.field_block is not None
        assert features.field_block.shape == (32,)

    def test_entity_block_has_declared_dim(self):
        encoder = StateEncoder(_full_schema(), hash_buckets=8)
        features = encoder.encode(_initial_state_view())
        assert features.entity_block is not None
        assert features.entity_block.shape == (16,)

    def test_population_block_alive_count(self):
        encoder = StateEncoder(_full_schema(), hash_buckets=8)
        features = encoder.encode(_initial_state_view())
        assert features.population_block is not None
        # First dim is alive count.
        assert features.population_block[0] == 2.0  # e0 + e1

    def test_to_vector_concatenates_present_blocks(self):
        encoder = StateEncoder(_full_schema(), hash_buckets=8)
        features = encoder.encode(_initial_state_view())
        vec = features.to_vector()
        # 32 + 16 + 8 + 8 + 3 = 67 dims.
        assert vec.shape == (67,)

    def test_to_vector_skips_absent_blocks(self):
        encoder = StateEncoder(_partial_schema(), hash_buckets=8)
        features = encoder.encode(_initial_state_view())
        vec = features.to_vector()
        # 16 + 8 = 24 dims (entities + graph only).
        assert vec.shape == (24,)

    def test_missing_mask_distinguishes_absent_capability_from_zero_value(self):
        """R2: absent capabilities (None + mask=True) must NOT be faked as zeros."""
        encoder = StateEncoder(_partial_schema(), hash_buckets=8)
        features = encoder.encode(_initial_state_view())
        # Fields absent → None + mask=True.
        assert features.field_block is None
        assert features.missing_mask[StateBlockType.FIELD.value] is True
        # Entities present but could be zero-filled (no rows) → not None, mask=False.
        empty_state = {"entities": {"rows": []}}
        features2 = encoder.encode(empty_state)
        assert features2.entity_block is not None
        assert features2.entity_block.shape == (16,)
        assert np.all(features2.entity_block == 0.0)
        assert features2.missing_mask[StateBlockType.ENTITY.value] is False


# ---------------------------------------------------------------------------
# StateMaterializer — R2 acceptance tests
# ---------------------------------------------------------------------------


class TestR2Acceptance:
    """R2 acceptance: traceability, time alignment, diff/apply, no silent degradation."""

    def test_no_silent_degradation_when_initial_state_is_none(self):
        """R2 acceptance #4: materializer must refuse when initial state absent."""
        materializer = _build_materializer()
        records = [_transition_record(tick=0)]
        with pytest.raises(MaterializerError, match="no initial_state_view"):
            materializer.materialize_episode(
                initial_state_view=None,
                transition_records=records,
                episode_id="test_ep",
                split="train",
            )

    def test_mechanical_traceability_via_provenance(self):
        """R2 acceptance #1: every TrainingTransition carries provenance back to source record."""
        materializer = _build_materializer()
        records = [
            _transition_record(tick=0, state_delta={"entity_changes": {"changes": []}}),
            _transition_record(tick=1, state_delta={"entity_changes": {"changes": []}}),
        ]
        samples = materializer.materialize_episode(
            initial_state_view=_initial_state_view(),
            transition_records=records,
            episode_id="test_ep",
            split="train",
        )
        assert len(samples) == 2
        for i, s in enumerate(samples, start=1):
            assert s.provenance.episode_id == "test_ep"
            assert s.provenance.split == "train"
            assert s.provenance.source_record_line == i
            assert s.provenance.source_record_sha256.startswith("sha256:")
            assert s.provenance.tick == i - 1
            assert s.provenance.seed == "42"
            assert s.provenance.policy_id == "random"

    def test_time_alignment_state_before_action_exogenous_from_same_tick(self):
        """R2 acceptance #2: state_before + joint_action + exogenous from same tick."""
        materializer = _build_materializer()
        records = [
            _transition_record(
                tick=0,
                executed_actions={
                    "e0": {"action_type": "MOVE", "params": {"target_node": "zone_a"}},
                },
                exogenous_input={"channels": {"weather": "rain"}},
                state_delta={"entity_changes": {"changes": []}},
            ),
        ]
        samples = materializer.materialize_episode(
            initial_state_view=_initial_state_view(),
            transition_records=records,
            episode_id="test_ep",
            split="train",
        )
        assert len(samples) == 1
        s = samples[0]
        # All from tick 0.
        assert s.provenance.tick == 0
        # Joint action contains e0's MOVE.
        assert s.joint_action_features.n_agents == 2
        assert s.joint_action_features.n_actions == 3
        # Exogenous has the weather channel.
        assert "weather" in s.exogenous_features.channel_names
        # State_before is the initial state (encoded).
        assert s.state_before_features.field_block is not None

    def test_diff_apply_round_trip_produces_consistent_state_sequence(self):
        """R2 acceptance #3: applying sequential state_delta produces consistent state sequence.

        The materializer walks the transition sequence applying state_delta
        diffs from the initial state. We verify that:
        - state_after_features at tick N matches state_before_features at tick N+1.
        - The diff/apply chain does not lose information.
        """
        materializer = _build_materializer()
        # Tick 0: add an entity.
        # Tick 1: add a relation edge.
        # Tick 2: change a registry state.
        records = [
            _transition_record(
                tick=0,
                state_delta={
                    "entity_changes": {
                        "changes": [
                            {"kind": "add", "entity_id": "e2", "after": [30.0, "zone_b"]},
                        ],
                    },
                },
            ),
            _transition_record(
                tick=1,
                state_delta={
                    "relation_changes": {
                        "changes": [
                            {"kind": "add", "src": "zone_a", "dst": "zone_b", "edge_type": "road", "after_weight": 1.0},
                        ],
                    },
                },
            ),
            _transition_record(
                tick=2,
                state_delta={
                    "registry_changes": {
                        "changes": [
                            {"kind": "state_change", "entry_id": "r0", "registry_type": "resource", "after_state": "depleted"},
                        ],
                    },
                },
            ),
        ]
        samples = materializer.materialize_episode(
            initial_state_view=_initial_state_view(),
            transition_records=records,
            episode_id="test_ep",
            split="train",
        )
        assert len(samples) == 3
        # state_after at tick N == state_before at tick N+1 (same underlying state_view).
        for i in range(len(samples) - 1):
            after_vec = samples[i].state_after_features.to_vector()
            before_vec = samples[i + 1].state_before_features.to_vector()
            np.testing.assert_array_equal(after_vec, before_vec)
        # The chain progressed: tick 0 state has 2 entities, tick 1 has 3.
        pop_0 = samples[0].state_before_features.population_block
        pop_1 = samples[1].state_before_features.population_block
        assert pop_0 is not None and pop_1 is not None
        # alive_ids stayed at 2 (we added an entity, but population didn't change
        # because entity_changes.add does not touch population.alive_ids in our
        # diff/apply helper).
        assert pop_0[0] == 2.0
        assert pop_1[0] == 2.0
        # But the entity block DID change (e2 added).
        ent_0 = samples[0].state_before_features.entity_block
        ent_1 = samples[1].state_before_features.entity_block
        assert ent_0 is not None and ent_1 is not None
        assert not np.array_equal(ent_0, ent_1)


# ---------------------------------------------------------------------------
# F-03 specific remediation tests
# ---------------------------------------------------------------------------


class TestF03JointAction:
    """F-03: joint action must encode ALL agents, not just the first."""

    def test_joint_action_encodes_all_agents(self):
        materializer = _build_materializer(
            action_types=("MOVE", "COLLECT"),
            agent_ids=("e0", "e1", "e2"),
        )
        records = [
            _transition_record(
                tick=0,
                executed_actions={
                    "e0": {"action_type": "MOVE", "params": {"target_node": "zone_a"}},
                    "e1": {"action_type": "COLLECT", "params": {}},
                    # e2 did not act.
                },
                state_delta={"entity_changes": {"changes": []}},
            ),
        ]
        samples = materializer.materialize_episode(
            initial_state_view=_initial_state_view(),
            transition_records=records,
            episode_id="test_ep",
            split="train",
        )
        assert len(samples) == 1
        ja = samples[0].joint_action_features
        assert ja.n_agents == 3
        assert ja.n_actions == 2
        # Per-agent row dim: n_actions + 3 params = 5.
        assert ja.features.shape == (3 * 5,)
        # e0 row: MOVE one-hot at index 0, target_node normalized at index 2.
        e0_base = 0
        assert ja.features[e0_base + 0] == 1.0  # MOVE
        assert ja.features[e0_base + 1] == 0.0  # COLLECT
        # e1 row: COLLECT one-hot at index 1.
        e1_base = 5
        assert ja.features[e1_base + 1] == 1.0
        # e2 row: all zeros (did not act).
        e2_base = 10
        assert np.all(ja.features[e2_base:e2_base + 5] == 0.0)

    def test_joint_action_preserves_action_order_via_vocab(self):
        """Action one-hot index is from the materializer's vocab, not the record's order."""
        materializer = _build_materializer(
            action_types=("ALPHA", "BETA", "GAMMA"),
            agent_ids=("e0",),
        )
        records = [
            _transition_record(
                tick=0,
                executed_actions={"e0": {"action_type": "GAMMA", "params": {}}},
                state_delta={"entity_changes": {"changes": []}},
            ),
        ]
        samples = materializer.materialize_episode(
            initial_state_view=_initial_state_view(),
            transition_records=records,
            episode_id="test_ep",
            split="train",
        )
        ja = samples[0].joint_action_features
        # GAMMA is at vocab index 2.
        assert ja.features[2] == 1.0
        assert ja.features[0] == 0.0
        assert ja.features[1] == 0.0

    def test_joint_action_unknown_action_type_zeros_row(self):
        """Unknown action type (not in vocab) produces all-zero action one-hot."""
        materializer = _build_materializer(
            action_types=("MOVE", "COLLECT"),
            agent_ids=("e0",),
        )
        records = [
            _transition_record(
                tick=0,
                executed_actions={"e0": {"action_type": "UNKNOWN", "params": {}}},
                state_delta={"entity_changes": {"changes": []}},
            ),
        ]
        samples = materializer.materialize_episode(
            initial_state_view=_initial_state_view(),
            transition_records=records,
            episode_id="test_ep",
            split="train",
        )
        ja = samples[0].joint_action_features
        # Action one-hot is all zeros (unknown type), but has_params still 1.
        assert ja.features[0] == 0.0
        assert ja.features[1] == 0.0
        # has_params is at index n_actions + 2 = 4.
        assert ja.features[4] == 0.0  # params={} is falsy → has_params=0


class TestF03Exogenous:
    """F-03 / F-05: exogenous input enters explicit features."""

    def test_exogenous_features_extracted_from_channels(self):
        materializer = _build_materializer()
        records = [
            _transition_record(
                tick=0,
                exogenous_input={"channels": {"weather": "rain", "temperature": 15.5}},
                state_delta={"entity_changes": {"changes": []}},
            ),
        ]
        samples = materializer.materialize_episode(
            initial_state_view=_initial_state_view(),
            transition_records=records,
            episode_id="test_ep",
            split="train",
        )
        exo = samples[0].exogenous_features
        assert len(exo.channel_names) == 2
        assert "weather" in exo.channel_names
        assert "temperature" in exo.channel_names
        # Temperature passes through as float.
        temp_idx = exo.channel_names.index("temperature")
        assert exo.features[temp_idx] == 15.5
        # Weather is string → bucket-hashed to a float.
        weather_idx = exo.channel_names.index("weather")
        assert exo.features[weather_idx] >= 0.0

    def test_exogenous_features_empty_when_no_input(self):
        materializer = _build_materializer()
        records = [
            _transition_record(
                tick=0,
                exogenous_input=None,
                state_delta={"entity_changes": {"changes": []}},
            ),
        ]
        samples = materializer.materialize_episode(
            initial_state_view=_initial_state_view(),
            transition_records=records,
            episode_id="test_ep",
            split="train",
        )
        exo = samples[0].exogenous_features
        assert exo.features.shape == (0,)
        assert exo.channel_names == ()


class TestF03StateBlocks:
    """F-03: field/entity/graph/registry/population blocks all materialized."""

    def test_all_five_blocks_present_for_full_schema(self):
        materializer = _build_materializer()
        records = [
            _transition_record(tick=0, state_delta={"entity_changes": {"changes": []}}),
        ]
        samples = materializer.materialize_episode(
            initial_state_view=_initial_state_view(),
            transition_records=records,
            episode_id="test_ep",
            split="train",
        )
        s = samples[0]
        assert s.state_before_features.field_block is not None
        assert s.state_before_features.entity_block is not None
        assert s.state_before_features.graph_block is not None
        assert s.state_before_features.registry_block is not None
        assert s.state_before_features.population_block is not None

    def test_field_block_changes_when_field_value_changes(self):
        materializer = _build_materializer()
        records = [
            _transition_record(
                tick=0,
                state_delta={
                    "field_changes": [
                        {"channel": "energy", "before": 100.0, "after": 50.0},
                    ],
                },
            ),
        ]
        samples = materializer.materialize_episode(
            initial_state_view=_initial_state_view(),
            transition_records=records,
            episode_id="test_ep",
            split="train",
        )
        # state_before has energy=100, state_after has energy=50.
        before = samples[0].state_before_features.field_block
        after = samples[0].state_after_features.field_block
        assert before is not None
        assert after is not None
        assert not np.array_equal(before, after)

    def test_graph_block_changes_when_edge_added(self):
        materializer = _build_materializer()
        records = [
            _transition_record(
                tick=0,
                state_delta={
                    "relation_changes": {
                        "changes": [
                            {"kind": "add", "src": "base", "dst": "zone_b", "edge_type": "road", "after_weight": 2.0},
                        ],
                    },
                },
            ),
        ]
        samples = materializer.materialize_episode(
            initial_state_view=_initial_state_view(),
            transition_records=records,
            episode_id="test_ep",
            split="train",
        )
        before = samples[0].state_before_features.graph_block
        after = samples[0].state_after_features.graph_block
        assert before is not None
        assert after is not None
        # Adding an edge increases the total weight.
        assert after.sum() > before.sum()

    def test_registry_block_changes_when_state_changes(self):
        materializer = _build_materializer()
        records = [
            _transition_record(
                tick=0,
                state_delta={
                    "registry_changes": {
                        "changes": [
                            # "available" and "broken" hash to different
                            # buckets mod 8 (1 vs 3), so the encoded block
                            # actually changes.
                            {"kind": "state_change", "entry_id": "r0", "registry_type": "resource", "after_state": "broken"},
                        ],
                    },
                },
            ),
        ]
        samples = materializer.materialize_episode(
            initial_state_view=_initial_state_view(),
            transition_records=records,
            episode_id="test_ep",
            split="train",
        )
        before = samples[0].state_before_features.registry_block
        after = samples[0].state_after_features.registry_block
        assert before is not None
        assert after is not None
        assert not np.array_equal(before, after)

    def test_population_block_changes_on_death(self):
        materializer = _build_materializer()
        records = [
            _transition_record(
                tick=0,
                state_delta={
                    "population_changes": {
                        "changes": [
                            {"kind": "death", "agent_id": "e1", "tick": 0, "cause": "exhaustion"},
                        ],
                    },
                },
            ),
        ]
        samples = materializer.materialize_episode(
            initial_state_view=_initial_state_view(),
            transition_records=records,
            episode_id="test_ep",
            split="train",
        )
        before = samples[0].state_before_features.population_block
        after = samples[0].state_after_features.population_block
        assert before is not None
        assert after is not None
        # alive count drops from 2 to 1.
        assert before[0] == 2.0
        assert after[0] == 1.0


# ---------------------------------------------------------------------------
# diff/apply tests (R2 acceptance #3, deeper)
# ---------------------------------------------------------------------------


class TestDiffApply:
    """Tests for the materializer's _apply_state_delta helper."""

    def test_apply_field_change_replaces_value(self):
        materializer = _build_materializer()
        state = _initial_state_view()
        delta = {"field_changes": [{"channel": "energy", "before": 100.0, "after": 50.0}]}
        next_state = materializer._apply_state_delta(state, delta)
        assert next_state["fields"]["energy"]["value"] == 50.0
        # Original state is NOT mutated.
        assert state["fields"]["energy"]["value"] == 100.0

    def test_apply_entity_add_appends_row(self):
        materializer = _build_materializer()
        state = _initial_state_view()
        delta = {
            "entity_changes": {
                "changes": [
                    {"kind": "add", "entity_id": "e2", "after": [30.0, "zone_b"]},
                ],
            },
        }
        next_state = materializer._apply_state_delta(state, delta)
        rows = next_state["entities"]["rows"]
        ids = [r["entity_id"] for r in rows]
        assert "e2" in ids

    def test_apply_entity_remove_filters_row(self):
        materializer = _build_materializer()
        state = _initial_state_view()
        delta = {
            "entity_changes": {
                "changes": [
                    {"kind": "remove", "entity_id": "e1"},
                ],
            },
        }
        next_state = materializer._apply_state_delta(state, delta)
        rows = next_state["entities"]["rows"]
        ids = [r["entity_id"] for r in rows]
        assert "e1" not in ids
        assert "e0" in ids

    def test_apply_relation_add_appends_edge(self):
        materializer = _build_materializer()
        state = _initial_state_view()
        delta = {
            "relation_changes": {
                "changes": [
                    {"kind": "add", "src": "zone_a", "dst": "zone_b", "edge_type": "road", "after_weight": 1.5},
                ],
            },
        }
        next_state = materializer._apply_state_delta(state, delta)
        edges = next_state["relations"]["edges"]
        assert any(
            e.get("src") == "zone_a" and e.get("dst") == "zone_b" for e in edges
        )

    def test_apply_relation_remove_filters_edge(self):
        materializer = _build_materializer()
        state = _initial_state_view()
        delta = {
            "relation_changes": {
                "changes": [
                    {"kind": "remove", "src": "base", "dst": "zone_a", "edge_type": "road"},
                ],
            },
        }
        next_state = materializer._apply_state_delta(state, delta)
        edges = next_state["relations"]["edges"]
        assert not any(
            e.get("src") == "base" and e.get("dst") == "zone_a" for e in edges
        )

    def test_apply_registry_state_change_updates_entry(self):
        materializer = _build_materializer()
        state = _initial_state_view()
        delta = {
            "registry_changes": {
                "changes": [
                    {"kind": "state_change", "entry_id": "r0", "registry_type": "resource", "after_state": "depleted"},
                ],
            },
        }
        next_state = materializer._apply_state_delta(state, delta)
        entries = next_state["registries"]["entries"]
        for e in entries:
            if e.get("entry_id") == "r0":
                assert e["state"] == "depleted"

    def test_apply_population_death_removes_agent(self):
        materializer = _build_materializer()
        state = _initial_state_view()
        delta = {
            "population_changes": {
                "changes": [
                    {"kind": "death", "agent_id": "e1", "tick": 0, "cause": "exhaustion"},
                ],
            },
        }
        next_state = materializer._apply_state_delta(state, delta)
        alive = next_state["population"]["alive_ids"]
        assert "e1" not in alive
        assert "e0" in alive

    def test_apply_population_birth_adds_agent(self):
        materializer = _build_materializer()
        state = _initial_state_view()
        delta = {
            "population_changes": {
                "changes": [
                    {"kind": "birth", "agent_id": "e2", "tick": 0, "parent_ids": ("e0",)},
                ],
            },
        }
        next_state = materializer._apply_state_delta(state, delta)
        alive = next_state["population"]["alive_ids"]
        assert "e2" in alive

    def test_apply_does_not_mutate_input_state(self):
        """diff/apply must deep-copy — input state_view must NOT be mutated."""
        materializer = _build_materializer()
        state = _initial_state_view()
        original = copy.deepcopy(state)
        delta = {
            "field_changes": [{"channel": "energy", "before": 100.0, "after": 50.0}],
            "entity_changes": {"changes": [{"kind": "add", "entity_id": "e2", "after": [30.0, "zone_b"]}]},
        }
        materializer._apply_state_delta(state, delta)
        assert state == original


# ---------------------------------------------------------------------------
# TrainingTransition — top-level tests
# ---------------------------------------------------------------------------


class TestTrainingTransition:
    def test_target_vector_legacy_compat_5_columns(self):
        """TrainingTransition.target_vector matches the legacy 5-column y layout."""
        materializer = _build_materializer()
        records = [
            _transition_record(
                tick=0,
                executed_actions={
                    "e0": {"action_type": "MOVE", "params": {"target_node": "zone_a"}},
                },
                state_delta={
                    "relation_changes": {
                        "changes": [
                            {"kind": "add", "src": "zone_a", "dst": "zone_b", "edge_type": "road", "after_weight": 1.0},
                        ],
                    },
                },
            ),
        ]
        samples = materializer.materialize_episode(
            initial_state_view=_initial_state_view(),
            transition_records=records,
            episode_id="test_ep",
            split="train",
        )
        target = samples[0].target_vector
        assert target.shape == (5,)
        # energy_delta from e0's receipt = -1.0.
        assert target[0] == -1.0
        # edge_change_count = 1 (one relation_changes.add).
        assert target[2] == 1.0

    def test_state_after_features_none_on_last_tick_without_state_delta(self):
        """If a record has no state_delta, state_after_features is None and chain stops."""
        materializer = _build_materializer()
        records = [
            _transition_record(tick=0, state_delta={"entity_changes": {"changes": []}}),
            _transition_record(tick=1, state_delta=None),  # No state_delta.
            _transition_record(tick=2, state_delta={"entity_changes": {"changes": []}}),
        ]
        samples = materializer.materialize_episode(
            initial_state_view=_initial_state_view(),
            transition_records=records,
            episode_id="test_ep",
            split="train",
        )
        # Only 2 samples (chain breaks at tick 1 with no state_delta).
        assert len(samples) == 2
        # Tick 0 has state_after (state_delta was present, even if empty).
        assert samples[0].state_after_features is not None
        # Tick 1 has no state_after (state_delta=None).
        assert samples[1].state_after_features is None


# ---------------------------------------------------------------------------
# Cross-capability integration tests
# ---------------------------------------------------------------------------


class TestCrossCapabilityIntegration:
    """Cross-capability integration: scenarios with partial capability sets."""

    def test_partial_schema_episode_materializes_without_error(self):
        """An episode with partial capabilities (no fields/registries/population)
        must still materialize successfully — absent blocks are None, not zeros."""
        materializer = _build_materializer(schema=_partial_schema())
        records = [
            _transition_record(
                tick=0,
                state_delta={
                    "relation_changes": {
                        "changes": [
                            {"kind": "add", "src": "zone_a", "dst": "zone_b", "edge_type": "road", "after_weight": 1.0},
                        ],
                    },
                },
            ),
        ]
        samples = materializer.materialize_episode(
            initial_state_view=_initial_state_view(),
            transition_records=records,
            episode_id="test_ep",
            split="train",
        )
        s = samples[0]
        # Fields / registries / population absent (capability=False).
        assert s.state_before_features.field_block is None
        assert s.state_before_features.registry_block is None
        assert s.state_before_features.population_block is None
        # Entities + relations present.
        assert s.state_before_features.entity_block is not None
        assert s.state_before_features.graph_block is not None
        # Graph block changed after the edge was added.
        assert s.state_after_features is not None
        assert s.state_after_features.graph_block is not None
        before_sum = s.state_before_features.graph_block.sum()
        after_sum = s.state_after_features.graph_block.sum()
        assert after_sum > before_sum


# ---------------------------------------------------------------------------
# Source-level guards (mirror the L1 lesson: source-code scanning matters)
# ---------------------------------------------------------------------------


class TestNoHardcodedSlicesInSource:
    """L1 lesson: source-code scanning is a necessary regression gate.

    Ensures the new state_materializer.py does not regress to hardcoded
    column slices like ``X[:, 1:8]``.
    """

    def test_no_hardcoded_action_slice_in_state_materializer(self):
        import re

        path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "worldloop_data"
            / "evaluation"
            / "state_materializer.py"
        )
        text = path.read_text(encoding="utf-8")

        # Forbidden patterns (legacy hardcoded slices).
        forbidden = [
            r"X\[\s*:\s*,\s*1\s*:\s*8\s*\]",
            r"X\[\s*:\s*,\s*8\s*:\s*12\s*\]",
            r"\brange\s*\(\s*7\s*\)",
            r"\brange\s*\(\s*4\s*\)",
        ]
        for pat in forbidden:
            matches = re.findall(pat, text)
            assert not matches, (
                f"state_materializer.py contains forbidden hardcoded slice "
                f"pattern {pat!r}: {matches}"
            )

    def test_no_legacy_first_agent_only_extraction(self):
        """F-03 guard: the materializer must NOT use sorted()[0] to pick
        only the first agent from executed_actions for the joint action
        encoding. (The legacy data_loader did this.)
        """
        path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "worldloop_data"
            / "evaluation"
            / "state_materializer.py"
        )
        text = path.read_text(encoding="utf-8")
        # Look for the legacy pattern: extracting first agent via sorted()[0]
        # from executed_actions. The new materializer iterates over ALL
        # agent_ids instead.
        forbidden_pattern = r"sorted\s*\(\s*executed[^)]*\)\s*\[\s*0\s*\]"
        import re

        matches = re.findall(forbidden_pattern, text)
        assert not matches, (
            f"state_materializer.py contains legacy first-agent-only pattern: {matches}"
        )


# ---------------------------------------------------------------------------
# UTF-8 portability guard (audit F-08)
# ---------------------------------------------------------------------------


class TestUtf8Portability:
    """F-08: all Path.read_text() calls must use encoding='utf-8'."""

    def test_source_uses_utf8_encoding_for_read_text(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "worldloop_data"
            / "evaluation"
            / "state_materializer.py"
        )
        text = path.read_text(encoding="utf-8")
        # Find any Path.read_text() calls without encoding=.
        import re

        # Match .read_text() without an encoding keyword arg.
        # Allow .read_text(encoding=...) — those are compliant.
        all_read_text = re.findall(r"\.read_text\([^)]*\)", text)
        for call in all_read_text:
            assert "encoding=" in call, (
                f"Found .read_text() without encoding= in state_materializer.py: {call}"
            )


# ---------------------------------------------------------------------------
# Module import path verification (audit F-07)
# ---------------------------------------------------------------------------


class TestModuleImportPath:
    """F-07: verify the materializer is imported from the workspace src/ tree,
    not from an installed site-packages wheel.
    """

    def test_module_file_located_in_workspace_src(self):
        import worldloop_data.evaluation.state_materializer as sm

        assert hasattr(sm, "__file__"), "sm has no __file__"
        assert sm.__file__ is not None, "sm.__file__ is None"
        path = Path(sm.__file__).resolve()
        # Must be inside current/worldloop-data/src/worldloop_data/.
        assert "current" in path.parts, f"unexpected path: {path}"
        assert "worldloop-data" in path.parts, f"unexpected path: {path}"
        assert "src" in path.parts, f"unexpected path: {path}"

    def test_module_exports_all_public_symbols(self):
        from worldloop_data.evaluation import state_materializer as sm

        for symbol in (
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
        ):
            assert hasattr(sm, symbol), f"sm missing public symbol: {symbol}"


# ---------------------------------------------------------------------------
# DataLoader R2 integration tests (audit F-03)
# ---------------------------------------------------------------------------


class TestDataLoaderR2Integration:
    """Integration tests for DataLoader's R2 methods.

    Verifies that DataLoader.build_materializer,
    DataLoader.load_training_transitions,
    DataLoader.feature_matrix_from_transitions, and
    DataLoader.r2_feature_layout work together to produce a feature
    matrix with the full ``S_t + A_t + U_t`` blocks.
    """

    def _build_dataset(
        self,
        tmp_path: Path,
        *,
        n_episodes: int = 2,
        n_ticks: int = 3,
    ) -> tuple[Path, dict[str, dict[str, Any]]]:
        """Build a minimal dataset directory for testing.

        Returns (dataset_dir, initial_state_views).
        """
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        splits: dict[str, str] = {}
        initial_state_views: dict[str, dict[str, Any]] = {}
        records: list[dict[str, Any]] = []

        for ep_idx in range(n_episodes):
            ep_id = f"ep{ep_idx}"
            split = "train" if ep_idx % 2 == 0 else "test"
            splits[ep_id] = split
            initial_state_views[ep_id] = _initial_state_view()
            for tick in range(n_ticks):
                records.append(
                    _transition_record(
                        tick=tick,
                        episode_id=ep_id,
                        executed_actions={
                            "e0": {"action_type": "MOVE", "params": {"target_node": "zone_a"}},
                        },
                        exogenous_input={"channels": {"weather": "rain"}},
                        state_delta={
                            "entity_changes": {"changes": []},
                        },
                    )
                )

        # Sort records by (split, episode_id, tick) to match exporter order.
        records.sort(
            key=lambda r: (
                splits.get((r.get("provenance") or {}).get("episode_id", ""), ""),
                (r.get("provenance") or {}).get("episode_id", ""),
                r.get("tick", 0),
            )
        )

        transitions_path = dataset_dir / "transitions.jsonl"
        with transitions_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, default=str) + "\n")

        splits_path = dataset_dir / "splits.json"
        splits_path.write_text(json.dumps(splits), encoding="utf-8")

        return dataset_dir, initial_state_views

    def test_build_materializer_uses_loader_vocab(self, tmp_path: Path):
        """DataLoader.build_materializer wires the loader's action/agent vocab."""
        from worldloop_data.evaluation.data_loader import DataLoader

        dataset_dir, _ = self._build_dataset(tmp_path)
        loader = DataLoader(dataset_dir)
        schema = _full_schema()
        materializer = loader.build_materializer(schema)
        assert materializer.action_types == loader.action_types
        assert materializer.agent_ids == loader.agent_ids

    def test_load_training_transitions_produces_per_split_samples(
        self, tmp_path: Path
    ):
        """load_training_transitions returns a dict[split -> list[TrainingTransition]]."""
        from worldloop_data.evaluation.data_loader import DataLoader

        dataset_dir, initial_state_views = self._build_dataset(
            tmp_path, n_episodes=2, n_ticks=3
        )
        loader = DataLoader(dataset_dir)
        schema = _full_schema()
        out = loader.load_training_transitions(initial_state_views, schema)
        # 2 episodes, one train + one test.
        assert "train" in out
        assert "test" in out
        # Each episode has 3 ticks → 3 transitions.
        assert len(out["train"]) == 3
        assert len(out["test"]) == 3
        # Each transition is a TrainingTransition.
        for t in out["train"]:
            assert isinstance(t, TrainingTransition)
            assert t.state_before_features.field_block is not None
            assert t.state_before_features.entity_block is not None
            assert t.joint_action_features.n_agents == len(loader.agent_ids)

    def test_load_training_transitions_skips_episodes_without_initial_state(
        self, tmp_path: Path
    ):
        """Episodes without an entry in initial_state_views are skipped (no silent degradation)."""
        from worldloop_data.evaluation.data_loader import DataLoader

        dataset_dir, initial_state_views = self._build_dataset(
            tmp_path, n_episodes=2, n_ticks=3
        )
        loader = DataLoader(dataset_dir)
        schema = _full_schema()
        # Only pass one episode's initial state.
        partial_views = {"ep0": initial_state_views["ep0"]}
        out = loader.load_training_transitions(partial_views, schema)
        # Only ep0 is loaded (into "train" split).
        assert "train" in out
        assert len(out["train"]) == 3
        # ep1 (test split) is skipped.
        assert "test" not in out

    def test_feature_matrix_from_transitions_has_state_and_action_blocks(
        self, tmp_path: Path
    ):
        """feature_matrix_from_transitions produces X with state + action blocks."""
        from worldloop_data.evaluation.data_loader import DataLoader

        dataset_dir, initial_state_views = self._build_dataset(
            tmp_path, n_episodes=1, n_ticks=3
        )
        loader = DataLoader(dataset_dir)
        schema = _full_schema()
        out = loader.load_training_transitions(initial_state_views, schema)
        transitions = out.get("train", [])
        X, y = loader.feature_matrix_from_transitions(transitions)
        assert X.shape[0] == 3
        assert y.shape == (3, 5)
        # X dim = 1 (tick) + joint_action_dim + state_dim + exogenous_dim
        joint_action_dim = len(loader.agent_ids) * (len(loader.action_types) + 3)
        state_dim = schema.encoded_dim()
        # exogenous dim is dynamic; check it's at least the joint_action + state + tick.
        assert X.shape[1] >= 1 + joint_action_dim + state_dim

    def test_feature_matrix_from_transitions_respects_include_flags(
        self, tmp_path: Path
    ):
        """include_state=False / include_action=False shrink X accordingly."""
        from worldloop_data.evaluation.data_loader import DataLoader

        dataset_dir, initial_state_views = self._build_dataset(
            tmp_path, n_episodes=1, n_ticks=2
        )
        loader = DataLoader(dataset_dir)
        schema = _full_schema()
        out = loader.load_training_transitions(initial_state_views, schema)
        transitions = out.get("train", [])

        X_full, _ = loader.feature_matrix_from_transitions(transitions)
        X_no_state, _ = loader.feature_matrix_from_transitions(
            transitions, include_state=False
        )
        X_no_action, _ = loader.feature_matrix_from_transitions(
            transitions, include_action=False
        )
        X_no_tick, _ = loader.feature_matrix_from_transitions(
            transitions, include_tick=False
        )

        assert X_no_state.shape[1] < X_full.shape[1]
        assert X_no_action.shape[1] < X_full.shape[1]
        assert X_no_tick.shape[1] < X_full.shape[1]
        assert X_no_tick.shape[1] == X_full.shape[1] - 1

    def test_r2_feature_layout_action_slice_points_to_joint_action(
        self, tmp_path: Path
    ):
        """r2_feature_layout.action_slice covers the joint_action block."""
        from worldloop_data.evaluation.data_loader import DataLoader

        dataset_dir, _ = self._build_dataset(tmp_path)
        loader = DataLoader(dataset_dir)
        schema = _full_schema()
        layout = loader.r2_feature_layout(schema)

        joint_action_dim = len(loader.agent_ids) * (len(loader.action_types) + 3)
        assert layout.n_actions == joint_action_dim
        assert layout.action_slice.stop - layout.action_slice.start == joint_action_dim
        # state_slice covers the encoder's full output.
        assert layout.n_state == schema.encoded_dim()
        # parameter_slice is empty (R2 puts params inside joint_action).
        assert layout.n_parameters == 0

    def test_r2_layout_matches_feature_matrix_block_order(self, tmp_path: Path):
        """The layout's slices must match the actual block positions in X.

        Verifies that zeroing layout.action_slice in X is equivalent to
        building X with include_action=False (plus zeros in the action
        positions). This is the property NoActionBaseline relies on.
        """
        from worldloop_data.evaluation.data_loader import DataLoader

        dataset_dir, initial_state_views = self._build_dataset(
            tmp_path, n_episodes=1, n_ticks=2
        )
        loader = DataLoader(dataset_dir)
        schema = _full_schema()
        out = loader.load_training_transitions(initial_state_views, schema)
        transitions = out.get("train", [])

        X, _ = loader.feature_matrix_from_transitions(transitions)
        layout = loader.r2_feature_layout(schema)

        # Zero out the action block.
        X_no_action = X.copy()
        X_no_action[:, layout.action_slice] = 0.0

        # The non-action blocks (tick, state) must be unchanged.
        if layout.tick_slice is not None:
            np.testing.assert_array_equal(
                X[:, layout.tick_slice], X_no_action[:, layout.tick_slice]
            )
        if layout.state_slice is not None:
            np.testing.assert_array_equal(
                X[:, layout.state_slice], X_no_action[:, layout.state_slice]
            )
        # The action block must be all zeros.
        assert np.all(X_no_action[:, layout.action_slice] == 0.0)

    def test_evaluation_package_exports_r2_symbols(self):
        """The evaluation __init__.py exports all R2 public symbols."""
        from worldloop_data import evaluation

        for symbol in (
            "EncoderSchema",
            "StateEncoder",
            "StateMaterializer",
            "StateFeatures",
            "JointActionFeatures",
            "ExogenousFeatures",
            "TrainingProvenance",
            "TrainingTransition",
            "MaterializerError",
            "StateBlockType",
        ):
            assert hasattr(evaluation, symbol), (
                f"evaluation package missing R2 symbol: {symbol}"
            )
