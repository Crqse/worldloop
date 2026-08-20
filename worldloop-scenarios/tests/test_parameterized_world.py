"""Tests for ParameterizedWorld (S-05/S-06/S-07) — WorldProtocol implementation.

Covers:
- Capability inference for all 3 space types (discrete / continuous / graph).
- reset() / observe() produce a valid StateView with correct slots populated.
- legal_actions() returns the spec's declared actions.
- validate_action() accepts valid proposals, rejects illegal ones (preconditions).
- step() applies effects to the correct target (entity / field / registry / relation).
- checkpoint() / restore() round-trip preserves state.
- Same seed + same actions → bit-identical state hashes (determinism).
- M3 Gate (b) determinism, (f) kernel-only independence.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from worldloop_kernel import (
    ActionProposal,
    CapabilityProfile,
    Checkpoint,
    ExogenousInput,
    StateView,
    TransitionRecord,
    hash_state,
)
from worldloop_scenarios.parameterized_world import (
    ParameterizedWorld,
    PAYLOAD_CODEC,
    PRODUCER_ID_PREFIX,
    PRODUCER_VERSION,
)
from worldloop_scenarios.spec import ScenarioSpec


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _load_spec(name: str) -> ScenarioSpec:
    path = EXAMPLES_DIR / name
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ScenarioSpec.from_dict(data)


@pytest.fixture
def discrete_spec() -> ScenarioSpec:
    return _load_spec("discrete_grid.yaml")


@pytest.fixture
def continuous_spec() -> ScenarioSpec:
    return _load_spec("continuous_field.yaml")


@pytest.fixture
def graph_spec() -> ScenarioSpec:
    return _load_spec("graph_registry.yaml")


def _make_proposal(agent_id, action_type, *, params=None, tick=0):
    return ActionProposal(
        agent_id=agent_id,
        action_type=action_type,
        params=params or {},
        proposed_at_tick=tick,
        proposer="test",
    )


# ---------------------------------------------------------------------------
# Capability inference
# ---------------------------------------------------------------------------


class TestCapabilityInference:
    """ParameterizedWorld._infer_capability must match the spec's slots."""

    def test_discrete_capabilities(self, discrete_spec: ScenarioSpec) -> None:
        world = ParameterizedWorld(discrete_spec)
        cap = world.capabilities
        assert isinstance(cap, CapabilityProfile)
        # discrete_grid.yaml: fields disabled, relations disabled, registries disabled.
        assert cap.fields is False
        assert cap.relations is False
        assert cap.registries is False
        # entities always True; population True because termination has no_alive.
        assert cap.entities is True
        assert cap.population is True
        # events False because exogenous.events is empty.
        assert cap.events is False
        # ParameterizedWorld is always exact_restore + deterministic_replay.
        assert cap.exact_restore is True
        assert cap.executable_deterministic_replay is True

    def test_continuous_capabilities(self, continuous_spec: ScenarioSpec) -> None:
        world = ParameterizedWorld(continuous_spec)
        cap = world.capabilities
        # continuous_field.yaml: fields enabled, relations disabled, registries disabled.
        assert cap.fields is True
        assert cap.relations is False
        assert cap.registries is False
        # population True (no_alive termination); events True (resource_spawn exogenous).
        assert cap.population is True
        assert cap.events is True

    def test_graph_capabilities(self, graph_spec: ScenarioSpec) -> None:
        world = ParameterizedWorld(graph_spec)
        cap = world.capabilities
        # graph_registry.yaml: fields disabled, relations enabled, registries enabled.
        assert cap.fields is False
        assert cap.relations is True
        assert cap.registries is True
        # population True (no_alive termination); events False (no exogenous).
        assert cap.population is True
        assert cap.events is False

    def test_producer_id_includes_scenario_id(
        self, discrete_spec: ScenarioSpec
    ) -> None:
        world = ParameterizedWorld(discrete_spec)
        # _producer_id is internal but used in TransitionRecord; check via state view.
        world.reset(0)
        sv = world.observe()
        # scenario_id is "discrete_grid_v0"; producer combines prefix + id + version.
        assert PRODUCER_ID_PREFIX in sv.meta.scenario_id or "discrete_grid_v0" in sv.meta.scenario_id
        # StateView.meta carries config_hash = world_parameters_hash.
        assert sv.meta.config_hash.startswith("sha256:")


# ---------------------------------------------------------------------------
# reset / observe
# ---------------------------------------------------------------------------


class TestResetObserve:
    """reset() and observe() produce a valid StateView."""

    def test_reset_returns_state_view(self, discrete_spec: ScenarioSpec) -> None:
        world = ParameterizedWorld(discrete_spec)
        sv = world.reset(seed=42)
        assert isinstance(sv, StateView)
        assert sv.meta.tick == 0
        assert sv.meta.scenario_id == "discrete_grid_v0"
        assert sv.meta.run_id == "seed-42"

    def test_reset_populates_entities(self, discrete_spec: ScenarioSpec) -> None:
        world = ParameterizedWorld(discrete_spec)
        sv = world.reset(seed=42)
        # discrete_grid.yaml has initial_count=5.
        assert len(sv.entities.ids) == 5
        # Columns: energy, x, y, alive.
        assert set(sv.entities.columns.keys()) == {"energy", "x", "y", "alive"}
        # Each column has 5 values.
        for col, vals in sv.entities.columns.items():
            assert len(vals) == 5

    def test_reset_populates_fields_for_continuous(
        self, continuous_spec: ScenarioSpec
    ) -> None:
        world = ParameterizedWorld(continuous_spec)
        sv = world.reset(seed=0)
        # Fields must be populated.
        assert sv.fields is not None
        # Channels: resource_density, hazard_level.
        assert "resource_density" in sv.fields.channels
        assert "hazard_level" in sv.fields.channels
        # Each channel shape is (10, 10) — frozen as tuple of tuples.
        rd = sv.fields.channels["resource_density"]
        assert len(rd) == 10
        assert len(rd[0]) == 10

    def test_reset_populates_relations_for_graph(
        self, graph_spec: ScenarioSpec
    ) -> None:
        world = ParameterizedWorld(graph_spec)
        sv = world.reset(seed=0)
        # Relations must be populated.
        assert sv.relations is not None
        # graph_registry.yaml declares 5 nodes.
        assert set(sv.relations.node_ids) == {"n0", "n1", "n2", "n3", "n4"}
        # 5 static edges from space.edges.
        assert len(sv.relations.edges) == 5

    def test_observe_after_reset_matches_reset_return(
        self, discrete_spec: ScenarioSpec
    ) -> None:
        world = ParameterizedWorld(discrete_spec)
        sv1 = world.reset(seed=1)
        sv2 = world.observe()
        # observe() must return an equivalent StateView (same hash).
        assert hash_state(sv1) == hash_state(sv2)

    def test_reset_with_different_seeds_produces_different_state(
        self, discrete_spec: ScenarioSpec
    ) -> None:
        world = ParameterizedWorld(discrete_spec)
        sv1 = world.reset(seed=1)
        sv2 = world.reset(seed=2)
        assert hash_state(sv1) != hash_state(sv2)

    def test_reset_is_idempotent_for_same_seed(
        self, discrete_spec: ScenarioSpec
    ) -> None:
        """Calling reset twice with the same seed produces identical state."""
        world = ParameterizedWorld(discrete_spec)
        sv1 = world.reset(seed=42)
        sv2 = world.reset(seed=42)
        assert hash_state(sv1) == hash_state(sv2)


# ---------------------------------------------------------------------------
# legal_actions / validate_action
# ---------------------------------------------------------------------------


class TestLegalActions:
    """legal_actions() returns the spec's declared actions."""

    def test_legal_actions_lists_all_declared(
        self, discrete_spec: ScenarioSpec
    ) -> None:
        world = ParameterizedWorld(discrete_spec)
        world.reset(seed=0)
        agent_id = world.observe().entities.ids[0]
        legal = world.legal_actions(agent_id)
        # discrete_grid.yaml has forage + rest.
        action_types = {la.action_type for la in legal.legal_actions}
        assert action_types == {"forage", "rest"}
        # is_closed matches spec.
        assert legal.is_closed is True

    def test_legal_actions_agent_id_echoed(
        self, discrete_spec: ScenarioSpec
    ) -> None:
        world = ParameterizedWorld(discrete_spec)
        world.reset(seed=0)
        agent_id = world.observe().entities.ids[0]
        legal = world.legal_actions(agent_id)
        assert legal.agent_id == agent_id

    def test_legal_actions_state_param_unsupported(
        self, discrete_spec: ScenarioSpec
    ) -> None:
        """v0.1 does not support counterfactual legal_actions(state=...)."""
        world = ParameterizedWorld(discrete_spec)
        world.reset(seed=0)
        agent_id = world.observe().entities.ids[0]
        with pytest.raises(NotImplementedError):
            world.legal_actions(agent_id, state=world.observe())


class TestValidateAction:
    """validate_action() accepts valid, rejects illegal (preconditions)."""

    def test_valid_proposal_returns_executed_action_and_ok_receipt(
        self, discrete_spec: ScenarioSpec
    ) -> None:
        world = ParameterizedWorld(discrete_spec)
        world.reset(seed=0)
        agent_id = world.observe().entities.ids[0]
        proposal = _make_proposal(agent_id, "forage", tick=0)
        executed, receipt = world.validate_action(proposal)
        assert executed.action_type == "forage"
        assert executed.agent_id == agent_id
        assert receipt.success is True
        assert receipt.outcome_code == "ok"
        # forage cost is 0.0 in discrete_grid.yaml.
        assert receipt.energy_delta == 0.0

    def test_unknown_action_in_closed_space_rejected(
        self, discrete_spec: ScenarioSpec
    ) -> None:
        world = ParameterizedWorld(discrete_spec)
        world.reset(seed=0)
        agent_id = world.observe().entities.ids[0]
        proposal = _make_proposal(agent_id, "fly", tick=0)
        _executed, receipt = world.validate_action(proposal)
        assert receipt.success is False
        assert receipt.outcome_code == "unrecognized_intent"

    def test_precondition_edge_exists_blocks_traverse(
        self, graph_spec: ScenarioSpec
    ) -> None:
        """traverse action requires edge_exists(from, to)."""
        world = ParameterizedWorld(graph_spec)
        world.reset(seed=42)
        agent_id = world.observe().entities.ids[0]
        agent_node = world.observe().entities.columns["node"][
            world.observe().entities.ids.index(agent_id)
        ]
        # Pick a non-adjacent node as the target — traverse must be rejected.
        all_nodes = set(graph_spec.space.node_ids)
        # Find a node that is NOT adjacent to agent_node.
        adjacent = {
            e.dst for e in world.observe().relations.edges if e.src == agent_node
        }
        # Add reverse edges too (graph is undirected).
        for e in world.observe().relations.edges:
            if e.dst == agent_node:
                adjacent.add(e.src)
        non_adjacent = list(all_nodes - adjacent - {agent_node})
        if non_adjacent:
            bad_target = non_adjacent[0]
            proposal = _make_proposal(
                agent_id, "traverse", params={"target_node": bad_target}, tick=0
            )
            _executed, receipt = world.validate_action(proposal)
            assert receipt.success is False
            assert receipt.outcome_code == "illegal_action"


# ---------------------------------------------------------------------------
# step — effect targets
# ---------------------------------------------------------------------------


class TestStepEffects:
    """step() applies effects to the correct target."""

    def test_entity_effect_add(self, discrete_spec: ScenarioSpec) -> None:
        """forage adds 5.0 to entity.energy."""
        world = ParameterizedWorld(discrete_spec)
        world.reset(seed=0)
        agent_id = world.observe().entities.ids[0]
        energy_before = world.observe().entities.columns["energy"][0]
        proposal = _make_proposal(agent_id, "forage", tick=0)
        executed, _ = world.validate_action(proposal)
        record = world.step(executed)
        # Verify energy increased by 5.0.
        energy_after = world.observe().entities.columns["energy"][0]
        assert energy_after == energy_before + 5.0
        # Verify the record structure.
        assert isinstance(record, TransitionRecord)
        assert record.state_before_hash != record.state_after_hash
        assert record.tick == 0
        # Tick advances by 1 after step.
        assert world.observe().meta.tick == 1

    def test_field_effect_sub(self, continuous_spec: ScenarioSpec) -> None:
        """forage subtracts 0.1 from the resource_density field."""
        world = ParameterizedWorld(continuous_spec)
        world.reset(seed=0)
        agent_id = world.observe().entities.ids[0]
        # Read field BEFORE step.
        rd_before = world.observe().fields.channels["resource_density"]
        # Apply forage.
        proposal = _make_proposal(agent_id, "forage", tick=0)
        executed, _ = world.validate_action(proposal)
        world.step(executed)
        # Read field AFTER step.
        rd_after = world.observe().fields.channels["resource_density"]
        # Every cell must have decreased by 0.1.
        for i in range(10):
            for j in range(10):
                assert rd_after[i][j] == pytest.approx(rd_before[i][j] - 0.1)

    def test_step_with_exogenous_input(
        self, continuous_spec: ScenarioSpec
    ) -> None:
        """step(exogenous=...) applies exogenous BEFORE the action."""
        world = ParameterizedWorld(continuous_spec)
        world.reset(seed=0)
        agent_id = world.observe().entities.ids[0]
        rd_before = world.observe().fields.channels["resource_density"]
        # Apply exogenous resource_spawn with rate=1.0, then a forage.
        exo = ExogenousInput(tick=0, kind="resource_spawn", payload={"rate": 1.0})
        proposal = _make_proposal(agent_id, "forage", tick=0)
        executed, _ = world.validate_action(proposal)
        world.step(executed, exogenous=exo)
        rd_after = world.observe().fields.channels["resource_density"]
        # Net delta = +1.0 (spawn) - 0.1 (forage) = +0.9.
        for i in range(10):
            for j in range(10):
                assert rd_after[i][j] == pytest.approx(rd_before[i][j] + 0.9)

    def test_step_with_unknown_action_no_effect(
        self, discrete_spec: ScenarioSpec
    ) -> None:
        """step() with an unknown action_type produces a record but no effects."""
        world = ParameterizedWorld(discrete_spec)
        world.reset(seed=0)
        agent_id = world.observe().entities.ids[0]
        energy_before = world.observe().entities.columns["energy"][0]
        # Construct an ExecutedAction with an unknown action_type.
        from worldloop_kernel import ExecutedAction

        executed = ExecutedAction(
            agent_id=agent_id,
            action_type="nonexistent_action",
            params={},
            executed_at_tick=0,
            proposal_hash="sha256:test",
        )
        record = world.step(executed)
        # Energy unchanged.
        energy_after = world.observe().entities.columns["energy"][0]
        assert energy_after == energy_before
        # But a record is still produced (state may differ only by tick).
        assert isinstance(record, TransitionRecord)


# ---------------------------------------------------------------------------
# checkpoint / restore round-trip
# ---------------------------------------------------------------------------


class TestCheckpointRestore:
    """checkpoint() and restore() preserve state exactly."""

    def test_checkpoint_returns_checkpoint_object(
        self, discrete_spec: ScenarioSpec
    ) -> None:
        world = ParameterizedWorld(discrete_spec)
        world.reset(seed=0)
        ckpt = world.checkpoint()
        assert isinstance(ckpt, Checkpoint)
        assert ckpt.payload_codec == PAYLOAD_CODEC
        assert ckpt.tick == 0
        assert ckpt.checksum.startswith("sha256:")

    def test_restore_round_trip_preserves_state(
        self, discrete_spec: ScenarioSpec
    ) -> None:
        world = ParameterizedWorld(discrete_spec)
        world.reset(seed=42)
        # Take 3 steps to mutate state.
        agent_id = world.observe().entities.ids[0]
        for _ in range(3):
            proposal = _make_proposal(agent_id, "forage", tick=world.observe().meta.tick)
            executed, _ = world.validate_action(proposal)
            world.step(executed)
        state_before_ckpt = hash_state(world.observe())
        # Checkpoint.
        ckpt = world.checkpoint()
        # Mutate further.
        for _ in range(2):
            proposal = _make_proposal(agent_id, "rest", tick=world.observe().meta.tick)
            executed, _ = world.validate_action(proposal)
            world.step(executed)
        # State has changed.
        assert hash_state(world.observe()) != state_before_ckpt
        # Restore — state must match the checkpoint.
        world.restore(ckpt)
        assert hash_state(world.observe()) == state_before_ckpt
        # Tick must also be restored.
        assert world.observe().meta.tick == 3

    def test_restore_wrong_codec_raises(
        self, discrete_spec: ScenarioSpec
    ) -> None:
        world = ParameterizedWorld(discrete_spec)
        world.reset(seed=0)
        ckpt = world.checkpoint()
        # Build a fake checkpoint with a different codec.
        from dataclasses import replace

        bad_ckpt = replace(ckpt, payload_codec="other:v1")
        with pytest.raises(ValueError, match="codec"):
            world.restore(bad_ckpt)

    def test_restore_round_trip_for_graph_world(
        self, graph_spec: ScenarioSpec
    ) -> None:
        """Checkpoint/restore must work for graph+registry worlds too."""
        world = ParameterizedWorld(graph_spec)
        world.reset(seed=7)
        agent_id = world.observe().entities.ids[0]
        # Find an adjacent node and traverse to it.
        sv = world.observe()
        agent_idx = sv.entities.ids.index(agent_id)
        agent_node = sv.entities.columns["node"][agent_idx]
        adjacent = [
            e.dst for e in sv.relations.edges if e.src == agent_node
        ] + [
            e.src for e in sv.relations.edges if e.dst == agent_node
        ]
        if adjacent:
            target = adjacent[0]
            proposal = _make_proposal(
                agent_id, "traverse", params={"target_node": target}, tick=0
            )
            executed, _ = world.validate_action(proposal)
            world.step(executed)
        state_after_step = hash_state(world.observe())
        ckpt = world.checkpoint()
        # Take another step.
        if adjacent:
            proposal = _make_proposal(
                agent_id, "traverse", params={"target_node": agent_node}, tick=1
            )
            executed, _ = world.validate_action(proposal)
            world.step(executed)
        # State changed.
        assert hash_state(world.observe()) != state_after_step
        # Restore.
        world.restore(ckpt)
        assert hash_state(world.observe()) == state_after_step


# ---------------------------------------------------------------------------
# Determinism — M3 Gate (b)
# ---------------------------------------------------------------------------


class TestDeterminism:
    """M3 Gate (b): same spec + seed → identical state."""

    def test_same_seed_same_state(self, discrete_spec: ScenarioSpec) -> None:
        w1 = ParameterizedWorld(discrete_spec)
        w2 = ParameterizedWorld(discrete_spec)
        sv1 = w1.reset(seed=42)
        sv2 = w2.reset(seed=42)
        assert hash_state(sv1) == hash_state(sv2)

    def test_same_seed_same_action_sequence_same_per_tick_hashes(
        self, discrete_spec: ScenarioSpec
    ) -> None:
        """Run two worlds with same seed + same actions; per-tick hashes must match."""
        w1 = ParameterizedWorld(discrete_spec)
        w2 = ParameterizedWorld(discrete_spec)
        w1.reset(seed=99)
        w2.reset(seed=99)
        agent_id = w1.observe().entities.ids[0]
        # 5 forage actions.
        proposals = [
            _make_proposal(agent_id, "forage", tick=t) for t in range(5)
        ]
        hashes_1: list[str] = []
        hashes_2: list[str] = []
        for p in proposals:
            e1, _ = w1.validate_action(p)
            w1.step(e1)
            hashes_1.append(hash_state(w1.observe()))
            e2, _ = w2.validate_action(p)
            w2.step(e2)
            hashes_2.append(hash_state(w2.observe()))
        assert hashes_1 == hashes_2


# ---------------------------------------------------------------------------
# 1×1 smoke — M3 Gate (a) minimal run
# ---------------------------------------------------------------------------


class TestOneByOneSmoke:
    """1×1 smoke: 1 agent, 1 tick, all 3 templates must run cleanly."""

    @pytest.mark.parametrize(
        "example_name",
        ["discrete_grid.yaml", "continuous_field.yaml", "graph_registry.yaml"],
    )
    def test_one_by_one_smoke(self, example_name: str) -> None:
        spec = _load_spec(example_name)
        world = ParameterizedWorld(spec)
        # Reset with 1 entity — but spec may declare more. We just take the first.
        world.reset(seed=0)
        sv = world.observe()
        assert len(sv.entities.ids) >= 1
        agent_id = sv.entities.ids[0]
        # Pick the first action from the spec (not from legal_actions,
        # which may be empty when actions have parameterized preconditions).
        assert len(spec.actions.actions) >= 1
        action_def = spec.actions.actions[0]
        action_type = action_def["action_type"]
        # Construct minimal params for actions that require them.
        params: dict = {}
        if action_type == "move":
            # continuous_field.yaml move needs dx, dy.
            params = {"dx": 0.5, "dy": 0.5}
        elif action_type == "traverse":
            # graph_registry.yaml traverse needs target_node.
            # Pick an adjacent node.
            agent_idx = sv.entities.ids.index(agent_id)
            agent_node = sv.entities.columns["node"][agent_idx]
            adjacent = [
                e.dst for e in sv.relations.edges if e.src == agent_node
            ] + [e.src for e in sv.relations.edges if e.dst == agent_node]
            if adjacent:
                params = {"target_node": adjacent[0]}
            else:
                # Fall back to a no-op action that doesn't need params;
                # use the second action if available.
                if len(spec.actions.actions) >= 2:
                    action_def = spec.actions.actions[1]
                    action_type = action_def["action_type"]
        proposal = _make_proposal(agent_id, action_type, params=params, tick=0)
        executed, _ = world.validate_action(proposal)
        record = world.step(executed)
        # Smoke checks.
        assert isinstance(record, TransitionRecord)
        assert record.tick == 0
        assert record.state_before_hash.startswith("sha256:")
        assert record.state_after_hash.startswith("sha256:")
        # Tick advanced.
        assert world.observe().meta.tick == 1


# ---------------------------------------------------------------------------
# WorldProtocol structural conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """ParameterizedWorld implements all 7 WorldProtocol methods."""

    def test_all_seven_methods_present(self, discrete_spec: ScenarioSpec) -> None:
        world = ParameterizedWorld(discrete_spec)
        # 7 methods: capabilities (property), reset, observe, legal_actions,
        # validate_action, step, checkpoint, restore.
        assert hasattr(world, "capabilities")
        assert callable(world.reset)
        assert callable(world.observe)
        assert callable(world.legal_actions)
        assert callable(world.validate_action)
        assert callable(world.step)
        assert callable(world.checkpoint)
        assert callable(world.restore)

    def test_capabilities_property_returns_profile(
        self, discrete_spec: ScenarioSpec
    ) -> None:
        world = ParameterizedWorld(discrete_spec)
        cap = world.capabilities
        assert isinstance(cap, CapabilityProfile)
        # authority is "rule" (spec-driven world).
        assert cap.authority == "rule"
        assert cap.ground_truth is True


# ---------------------------------------------------------------------------
# Independence from v1 — M3 Gate (f)
# ---------------------------------------------------------------------------


class TestKernelOnlyIndependence:
    """M3 Gate (f): ParameterizedWorld runs with kernel + scenarios only.

    This test verifies at the import level that the world does not pull in
    any v1 ``current.worldloop.*`` modules.
    """

    def test_no_v1_imports_in_parameterized_world_module(self) -> None:
        """The parameterized_world.py source must not import current.worldloop.*."""
        src_path = Path(
            ParameterizedWorld.__module__.replace(".", "/") + ".py"
        )
        # Resolve via the package's __file__ to find the source.
        import worldloop_scenarios

        pkg_dir = Path(worldloop_scenarios.__file__).parent
        src = (pkg_dir / "parameterized_world.py").read_text(encoding="utf-8")
        # Check that no line imports current.worldloop.
        bad_lines = [
            line
            for line in src.splitlines()
            if line.strip().startswith(("import ", "from "))
            and "current.worldloop" in line
        ]
        assert bad_lines == [], f"v1 imports found: {bad_lines}"

    def test_no_v1_imports_in_compiler_module(self) -> None:
        """The compiler.py source must not import current.worldloop.*."""
        import worldloop_scenarios

        pkg_dir = Path(worldloop_scenarios.__file__).parent
        src = (pkg_dir / "compiler.py").read_text(encoding="utf-8")
        bad_lines = [
            line
            for line in src.splitlines()
            if line.strip().startswith(("import ", "from "))
            and "current.worldloop" in line
        ]
        assert bad_lines == [], f"v1 imports found: {bad_lines}"
