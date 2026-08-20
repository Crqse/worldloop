"""Tests for the spec compiler (S-04).

Covers:
- :func:`compile_spec` with a :class:`ScenarioSpec` instance.
- :func:`compile_dict` with a raw dict.
- :func:`compile_file` with YAML / JSON files, error cases.
- Schema + semantic validation failures raise the right exceptions.
- The world factory produces fresh, equivalent, deterministic worlds.
- The compiled :class:`ScenarioPackage` has stable hash and provenance.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from worldloop_kernel import (
    ActionProposal,
    StateView,
    TransitionRecorder,
    hash_state,
)
from worldloop_scenarios import (
    COMPILE_SCHEMA_VERSION,
    CompileError,
    ScenarioPackage,
    ScenarioSpec,
    SemanticValidationError,
    compile_dict,
    compile_file,
    compile_spec,
)
from worldloop_scenarios.parameterized_world import ParameterizedWorld
from worldloop_scenarios.schema_loader import SchemaValidationError


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _load_example(name: str) -> ScenarioSpec:
    """Load an example YAML as a ScenarioSpec (no compilation)."""
    path = EXAMPLES_DIR / name
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ScenarioSpec.from_dict(data)


def _load_example_dict(name: str) -> dict:
    """Load an example YAML as a raw dict."""
    path = EXAMPLES_DIR / name
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture
def discrete_spec() -> ScenarioSpec:
    return _load_example("discrete_grid.yaml")


@pytest.fixture
def continuous_spec() -> ScenarioSpec:
    return _load_example("continuous_field.yaml")


@pytest.fixture
def graph_spec() -> ScenarioSpec:
    return _load_example("graph_registry.yaml")


# ---------------------------------------------------------------------------
# compile_spec — success path
# ---------------------------------------------------------------------------


class TestCompileSpecSuccess:
    """compile_spec with valid ScenarioSpec instances."""

    def test_returns_scenario_package(self, discrete_spec: ScenarioSpec) -> None:
        pkg = compile_spec(discrete_spec)
        assert isinstance(pkg, ScenarioPackage)

    def test_package_holds_the_same_spec(self, discrete_spec: ScenarioSpec) -> None:
        pkg = compile_spec(discrete_spec)
        assert pkg.spec is discrete_spec

    def test_world_factory_is_callable(self, discrete_spec: ScenarioSpec) -> None:
        pkg = compile_spec(discrete_spec)
        assert callable(pkg.world_factory)

    def test_world_factory_produces_parameterized_world(
        self, discrete_spec: ScenarioSpec
    ) -> None:
        pkg = compile_spec(discrete_spec)
        world = pkg.world_factory(42)
        assert isinstance(world, ParameterizedWorld)

    def test_world_factory_applies_reset(self, discrete_spec: ScenarioSpec) -> None:
        """world_factory(seed) must call reset(seed) on the world."""
        pkg = compile_spec(discrete_spec)
        world = pkg.world_factory(42)
        # reset() populated entity_ids (5 entities per discrete_grid.yaml).
        assert len(world._entity_ids) == 5
        # observe() returns a non-empty StateView.
        sv = world.observe()
        assert isinstance(sv, StateView)
        assert sv.meta.tick == 0
        assert sv.meta.scenario_id == "discrete_grid_v0"
        assert sv.meta.run_id == "seed-42"

    def test_world_factory_produces_fresh_instances(
        self, discrete_spec: ScenarioSpec
    ) -> None:
        """Each call to world_factory produces a brand-new world instance."""
        pkg = compile_spec(discrete_spec)
        w1 = pkg.world_factory(1)
        w2 = pkg.world_factory(1)
        assert w1 is not w2
        # Both should have identical state after reset with the same seed.
        assert hash_state(w1.observe()) == hash_state(w2.observe())

    def test_world_factory_different_seeds_produce_different_state(
        self, discrete_spec: ScenarioSpec
    ) -> None:
        """Different seeds produce different initial states."""
        pkg = compile_spec(discrete_spec)
        w1 = pkg.world_factory(1)
        w2 = pkg.world_factory(2)
        assert hash_state(w1.observe()) != hash_state(w2.observe())

    def test_world_parameters_hash_matches_spec(
        self, discrete_spec: ScenarioSpec
    ) -> None:
        pkg = compile_spec(discrete_spec)
        assert pkg.world_parameters_hash == discrete_spec.world_parameters_hash()
        assert pkg.world_parameters_hash.startswith("sha256:")

    def test_compile_schema_version(self, discrete_spec: ScenarioSpec) -> None:
        pkg = compile_spec(discrete_spec)
        assert pkg.compile_schema_version == COMPILE_SCHEMA_VERSION

    def test_provenance_populated(self, discrete_spec: ScenarioSpec) -> None:
        pkg = compile_spec(discrete_spec)
        assert pkg.provenance["schema_validated"] is True
        assert pkg.provenance["semantic_validated"] is True
        assert "compiled_at" in pkg.provenance
        assert "n_semantic_errors" in pkg.provenance
        assert "n_semantic_warnings" in pkg.provenance

    def test_compile_spec_idempotent(self, discrete_spec: ScenarioSpec) -> None:
        """Compiling the same spec twice produces packages with the same hash."""
        pkg1 = compile_spec(discrete_spec)
        pkg2 = compile_spec(discrete_spec)
        assert pkg1.world_parameters_hash == pkg2.world_parameters_hash


# ---------------------------------------------------------------------------
# compile_spec — all three example templates
# ---------------------------------------------------------------------------


class TestCompileAllTemplates:
    """compile_spec must succeed for all three M3 templates + M5 demo."""

    @pytest.mark.parametrize(
        "example_name",
        [
            "discrete_grid.yaml",
            "continuous_field.yaml",
            "graph_registry.yaml",
            "emergency_resource.yaml",
        ],
    )
    def test_compile_example(self, example_name: str) -> None:
        spec = _load_example(example_name)
        pkg = compile_spec(spec)
        assert isinstance(pkg, ScenarioPackage)
        # The world factory must produce a runnable world.
        world = pkg.world_factory(42)
        sv = world.observe()
        assert sv.meta.scenario_id == spec.scenario.scenario_id


class TestEmergencyResourceSpec:
    """M5 §15 demo scenario — D-01 spec + D-02 action contracts.

    Verifies the emergency_resource.yaml spec declares the six core
    action contracts (MOVE/COLLECT/DELIVER/SHARE/REPAIR/COMMUNICATE,
    REST optional) and exercises all three effect surfaces
    (field/graph/registry) required by M5 Gate §15.5 (c).
    """

    @pytest.fixture
    def emergency_spec(self) -> ScenarioSpec:
        return _load_example("emergency_resource.yaml")

    def test_six_core_actions_declared(self, emergency_spec: ScenarioSpec) -> None:
        """D-02: all six core action contracts are present."""
        action_types = {a["action_type"] for a in emergency_spec.actions.actions}
        required = {"MOVE", "COLLECT", "DELIVER", "SHARE", "REPAIR", "COMMUNICATE"}
        assert required.issubset(action_types), (
            f"Missing core actions: {required - action_types}"
        )

    def test_rest_optional_present(self, emergency_spec: ScenarioSpec) -> None:
        """REST is optional but should be present for energy regeneration."""
        action_types = {a["action_type"] for a in emergency_spec.actions.actions}
        assert "REST" in action_types

    def test_field_effect_surface(self, emergency_spec: ScenarioSpec) -> None:
        """M5 Gate §15.5 (c): at least one action writes a field channel."""
        field_targets = {
            e.get("field")
            for a in emergency_spec.actions.actions
            for e in a.get("effects", ())
            if e.get("target") == "field"
        }
        assert field_targets, "No field effect declared (field delta would be zero)"

    def test_graph_effect_surface(self, emergency_spec: ScenarioSpec) -> None:
        """M5 Gate §15.5 (c): at least one action adds/removes a relation edge."""
        relation_targets = {
            e.get("field")
            for a in emergency_spec.actions.actions
            for e in a.get("effects", ())
            if e.get("target") == "relation"
        }
        assert relation_targets, (
            "No relation effect declared (graph delta would be zero)"
        )

    def test_registry_effect_surface(self, emergency_spec: ScenarioSpec) -> None:
        """M5 Gate §15.5 (c): at least one action mutates a registry entry."""
        registry_effects = [
            e
            for a in emergency_spec.actions.actions
            for e in a.get("effects", ())
            if e.get("target") == "registry"
        ]
        assert registry_effects, (
            "No registry effect declared (registry delta would be zero)"
        )

    def test_exogenous_event_present(self, emergency_spec: ScenarioSpec) -> None:
        """M5 Gate §15.5 (e): at least one exogenous event declared."""
        assert len(emergency_spec.exogenous.events) > 0
        assert emergency_spec.exogenous.seed is not None

    def test_all_capabilities_enabled(self, emergency_spec: ScenarioSpec) -> None:
        """The M5 demo should exercise all six capability slots."""
        pkg = compile_spec(emergency_spec)
        world = pkg.world_factory(42)
        cap = world.observe().capabilities
        assert cap.fields and cap.entities and cap.relations
        assert cap.registries and cap.population and cap.events

    def test_move_action_contract(self, emergency_spec: ScenarioSpec) -> None:
        """D-02 MOVE contract: target_node param + edge_exists precondition."""
        move = next(
            a for a in emergency_spec.actions.actions if a["action_type"] == "MOVE"
        )
        assert "target_node" in move["params_schema"]
        assert any(
            p.get("kind") == "edge_exists" for p in move.get("preconditions", ())
        )

    def test_collect_deliver_repair_have_energy_precondition(
        self, emergency_spec: ScenarioSpec
    ) -> None:
        """D-02: COLLECT/DELIVER/REPAIR require energy_above (failure mode)."""
        for action_type in ("COLLECT", "DELIVER", "REPAIR"):
            action = next(
                a
                for a in emergency_spec.actions.actions
                if a["action_type"] == action_type
            )
            assert any(
                p.get("kind") == "energy_above"
                for p in action.get("preconditions", ())
            ), f"{action_type} missing energy_above precondition"


# ---------------------------------------------------------------------------
# M5 §15.5 (c) effect matrix smoke — D-03 field/graph/registry deltas
# ---------------------------------------------------------------------------


class TestEmergencyEffectMatrix:
    """M5 Gate §15.5 (c): field/graph/registry deltas all non-zero.

    D-03: exercise the effect matrix by running each of the six core
    action contracts plus the hazard_escalation exogenous event, and
    verify each effect surface produces a measurable state change.
    """

    @pytest.fixture
    def world(self) -> ParameterizedWorld:
        """Compile emergency_resource.yaml and reset with seed 42."""
        spec = _load_example("emergency_resource.yaml")
        pkg = compile_spec(spec)
        w = pkg.world_factory(42)
        w.reset(42)
        return w

    def _first_agent(self, world: ParameterizedWorld) -> str:
        return world.observe().entities.ids[0]

    def _second_agent(self, world: ParameterizedWorld) -> str:
        return world.observe().entities.ids[1]

    def _agent_node(self, world: ParameterizedWorld, agent_id: str) -> str:
        sv = world.observe()
        idx = sv.entities.ids.index(agent_id)
        return str(sv.entities.columns["node"][idx])

    def _adjacent_node(self, world: ParameterizedWorld, current: str) -> str:
        """Pick any node adjacent to ``current`` via space.edges."""
        for src, dst in world._spec.space.edges:
            if src == current:
                return dst
            if dst == current and not world._spec.relations.directed:
                return src
        # Fallback: pick a different node.
        for nid in world._spec.space.node_ids:
            if nid != current:
                return nid
        return current

    def _registry_state(
        self, world: ParameterizedWorld, entry_id: str
    ) -> str | None:
        for e in world._registry_entries:
            if e["entry_id"] == entry_id:
                return str(e["state"])
        return None

    def _field_value(self, world: ParameterizedWorld, channel: str) -> Any:
        return world._field_channels.get(channel)

    def _edge_count(self, world: ParameterizedWorld, edge_type: str) -> int:
        return sum(1 for e in world._relation_edges if e[2] == edge_type)

    def _step_action(
        self,
        world: ParameterizedWorld,
        agent_id: str,
        action_type: str,
        params: dict | None = None,
    ) -> None:
        """Validate + step a single action."""
        proposal = ActionProposal(
            agent_id=agent_id,
            action_type=action_type,
            params=params or {},
            proposed_at_tick=world.observe().meta.tick,
            proposer="test",
        )
        executed, _receipt = world.validate_action(proposal)
        world.step(executed)

    # -- D-03 (a): field delta via REPAIR (hazard_level sub) ------------

    def test_repair_produces_field_delta(self, world: ParameterizedWorld) -> None:
        """REPAIR subtracts 0.5 from hazard_level → non-zero field delta."""
        before = self._field_value(world, "hazard_level")
        agent_id = self._first_agent(world)
        self._step_action(
            world,
            agent_id,
            "REPAIR",
            {"entry_id": "fac_clinic_zone_b"},
        )
        after = self._field_value(world, "hazard_level")
        assert before == 0.0
        assert after == -0.5, f"hazard_level expected -0.5, got {after}"

    # -- D-03 (b): graph delta via SHARE / COMMUNICATE -------------------

    def test_share_produces_graph_delta(self, world: ParameterizedWorld) -> None:
        """SHARE adds a communication edge → non-zero graph delta."""
        before = self._edge_count(world, "communication")
        agent_id = self._first_agent(world)
        target = self._second_agent(world)
        self._step_action(
            world,
            agent_id,
            "SHARE",
            {"target_agent": target, "amount": 1.0},
        )
        after = self._edge_count(world, "communication")
        assert after == before + 1, (
            f"communication edges expected {before + 1}, got {after}"
        )

    def test_communicate_produces_graph_delta(
        self, world: ParameterizedWorld
    ) -> None:
        """COMMUNICATE adds a communication edge → non-zero graph delta."""
        before = self._edge_count(world, "communication")
        agent_id = self._first_agent(world)
        target = self._second_agent(world)
        self._step_action(
            world,
            agent_id,
            "COMMUNICATE",
            {"target_agent": target},
        )
        after = self._edge_count(world, "communication")
        assert after == before + 1

    # -- D-03 (c): registry delta via COLLECT / DELIVER / REPAIR ---------

    def test_collect_produces_registry_delta(
        self, world: ParameterizedWorld
    ) -> None:
        """COLLECT marks resource state → 'depleted' (non-zero registry delta)."""
        before = self._registry_state(world, "res_water_base")
        agent_id = self._first_agent(world)
        self._step_action(
            world,
            agent_id,
            "COLLECT",
            {"entry_id": "res_water_base", "amount": 2.0},
        )
        after = self._registry_state(world, "res_water_base")
        assert before == "available"
        assert after == "depleted", (
            f"res_water_base state expected 'depleted', got {after!r}"
        )

    def test_deliver_produces_registry_delta(
        self, world: ParameterizedWorld
    ) -> None:
        """DELIVER marks facility state → 'stocked' (non-zero registry delta)."""
        before = self._registry_state(world, "fac_clinic_zone_b")
        agent_id = self._first_agent(world)
        self._step_action(
            world,
            agent_id,
            "DELIVER",
            {"entry_id": "fac_clinic_zone_b", "amount": 1.0},
        )
        after = self._registry_state(world, "fac_clinic_zone_b")
        assert before == "damaged"
        assert after == "stocked"

    def test_repair_produces_registry_delta(
        self, world: ParameterizedWorld
    ) -> None:
        """REPAIR marks facility state → 'operational' (registry delta)."""
        before = self._registry_state(world, "fac_power_zone_d")
        agent_id = self._first_agent(world)
        self._step_action(
            world,
            agent_id,
            "REPAIR",
            {"entry_id": "fac_power_zone_d"},
        )
        after = self._registry_state(world, "fac_power_zone_d")
        assert before == "damaged"
        assert after == "operational"

    # -- D-03 (d): entity field delta via MOVE / REST -------------------

    def test_move_produces_entity_delta(self, world: ParameterizedWorld) -> None:
        """MOVE changes entity.node → non-zero entity field delta."""
        agent_id = self._first_agent(world)
        before = self._agent_node(world, agent_id)
        target = self._adjacent_node(world, before)
        self._step_action(
            world,
            agent_id,
            "MOVE",
            {"target_node": target},
        )
        after = self._agent_node(world, agent_id)
        assert before != target
        assert after == target, (
            f"entity.node expected {target!r}, got {after!r}"
        )

    # -- D-03 (e): exogenous event produces field delta ------------------

    def test_hazard_escalation_produces_field_delta(
        self, world: ParameterizedWorld
    ) -> None:
        """M5 Gate §15.5 (e): hazard_escalation exogenous event measurably
        increases hazard_level → non-zero field delta from exogenous input."""
        from worldloop_kernel import ExogenousInput

        before = self._field_value(world, "hazard_level")
        agent_id = self._first_agent(world)
        # REST is a no-op on hazard_level but advances the tick with an
        # exogenous hazard_escalation event applied before the action.
        exo = ExogenousInput(
            tick=world.observe().meta.tick,
            kind="hazard_escalation",
            payload={"rate": 0.3},
        )
        proposal = ActionProposal(
            agent_id=agent_id,
            action_type="REST",
            params={},
            proposed_at_tick=world.observe().meta.tick,
            proposer="test",
        )
        executed, _ = world.validate_action(proposal)
        world.step(executed, exogenous=exo)
        after = self._field_value(world, "hazard_level")
        assert before == 0.0
        assert after == 0.3, (
            f"hazard_level expected 0.3 after escalation, got {after}"
        )

    # -- D-03 aggregate: all three surfaces non-zero across matrix -------

    def test_all_three_effect_surfaces_exercised(
        self, world: ParameterizedWorld
    ) -> None:
        """M5 Gate §15.5 (c) aggregate: field + graph + registry deltas
        are all non-zero across the action matrix."""
        agent_id = self._first_agent(world)
        target = self._second_agent(world)

        # Snapshot before-state for all three surfaces.
        field_before = self._field_value(world, "hazard_level")
        graph_before = self._edge_count(world, "communication")
        reg_before = self._registry_state(world, "res_food_zone_a")

        # Apply one action per surface.
        # 1. Field surface: REPAIR reduces hazard_level.
        self._step_action(
            world, agent_id, "REPAIR", {"entry_id": "fac_clinic_zone_b"}
        )
        # 2. Graph surface: COMMUNICATE adds edge.
        self._step_action(
            world, agent_id, "COMMUNICATE", {"target_agent": target}
        )
        # 3. Registry surface: COLLECT depletes resource.
        self._step_action(
            world, agent_id, "COLLECT",
            {"entry_id": "res_food_zone_a", "amount": 1.0},
        )

        # Snapshot after-state.
        field_after = self._field_value(world, "hazard_level")
        graph_after = self._edge_count(world, "communication")
        reg_after = self._registry_state(world, "res_food_zone_a")

        # All three surfaces must have non-zero deltas.
        assert field_after != field_before, "field delta is zero"
        assert graph_after != graph_before, "graph delta is zero"
        assert reg_after != reg_before, "registry delta is zero"

    # -- Initial entries loaded ------------------------------------------

    def test_initial_entries_loaded(self, world: ParameterizedWorld) -> None:
        """M5 v0.2: registries.initial_entries populates 5 entries at reset."""
        assert len(world._registry_entries) == 5
        entry_ids = {e["entry_id"] for e in world._registry_entries}
        assert entry_ids == {
            "res_water_base",
            "res_food_zone_a",
            "res_med_zone_c",
            "fac_clinic_zone_b",
            "fac_power_zone_d",
        }


# ---------------------------------------------------------------------------
# compile_dict — success path
# ---------------------------------------------------------------------------


class TestCompileDict:
    """compile_dict with raw dicts."""

    def test_compile_dict_returns_package(self) -> None:
        data = _load_example_dict("discrete_grid.yaml")
        pkg = compile_dict(data)
        assert isinstance(pkg, ScenarioPackage)
        assert pkg.spec.scenario.scenario_id == "discrete_grid_v0"

    def test_compile_dict_factory_runnable(self) -> None:
        data = _load_example_dict("graph_registry.yaml")
        pkg = compile_dict(data)
        world = pkg.world_factory(7)
        sv = world.observe()
        assert sv.meta.scenario_id == "graph_registry_v0"
        assert sv.capabilities.relations is True
        assert sv.capabilities.registries is True


# ---------------------------------------------------------------------------
# compile_file — success path
# ---------------------------------------------------------------------------


class TestCompileFile:
    """compile_file with YAML / JSON files."""

    @pytest.mark.parametrize(
        "example_name",
        [
            "discrete_grid.yaml",
            "continuous_field.yaml",
            "graph_registry.yaml",
            "emergency_resource.yaml",
        ],
    )
    def test_compile_yaml_file(self, example_name: str) -> None:
        path = EXAMPLES_DIR / example_name
        pkg = compile_file(path)
        assert isinstance(pkg, ScenarioPackage)
        world = pkg.world_factory(0)
        assert world.observe().meta.scenario_id == pkg.spec.scenario.scenario_id

    def test_compile_json_file(self, tmp_path: Path) -> None:
        """compile_file must accept JSON files."""
        data = _load_example_dict("discrete_grid.yaml")
        json_path = tmp_path / "spec.json"
        json_path.write_text(json.dumps(data), encoding="utf-8")
        pkg = compile_file(json_path)
        assert pkg.spec.scenario.scenario_id == "discrete_grid_v0"

    def test_compile_yml_extension(self, tmp_path: Path) -> None:
        """.yml extension is also accepted."""
        data = _load_example_dict("discrete_grid.yaml")
        yml_path = tmp_path / "spec.yml"
        yml_path.write_text(yaml.safe_dump(data), encoding="utf-8")
        pkg = compile_file(yml_path)
        assert pkg.spec.scenario.scenario_id == "discrete_grid_v0"


# ---------------------------------------------------------------------------
# compile_file — error path
# ---------------------------------------------------------------------------


class TestCompileFileErrors:
    """compile_file error cases."""

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            compile_file(EXAMPLES_DIR / "nonexistent.yaml")

    def test_unsupported_extension(self, tmp_path: Path) -> None:
        bad = tmp_path / "spec.txt"
        bad.write_text("hello", encoding="utf-8")
        with pytest.raises(CompileError, match="Unsupported"):
            compile_file(bad)

    def test_top_level_not_mapping(self, tmp_path: Path) -> None:
        bad = tmp_path / "spec.yaml"
        bad.write_text("- just\n- a\n- list", encoding="utf-8")
        with pytest.raises(CompileError, match="mapping"):
            compile_file(bad)


# ---------------------------------------------------------------------------
# compile_spec — schema validation failure
# ---------------------------------------------------------------------------


class TestCompileSchemaFailure:
    """compile_spec / compile_dict must reject schema-invalid specs."""

    def test_compile_dict_missing_required_field(self) -> None:
        """A spec missing the ``space`` section fails schema validation."""
        data = _load_example_dict("discrete_grid.yaml")
        del data["space"]
        with pytest.raises(SchemaValidationError):
            compile_dict(data)

    def test_compile_dict_invalid_space_type(self) -> None:
        data = _load_example_dict("discrete_grid.yaml")
        data["space"]["type"] = "hyperbolic"  # not in enum
        with pytest.raises(SchemaValidationError):
            compile_dict(data)


# ---------------------------------------------------------------------------
# compile_spec — semantic validation failure
# ---------------------------------------------------------------------------


class TestCompileSemanticFailure:
    """compile_spec / compile_dict must reject semantically invalid specs."""

    def test_compile_dict_undeclared_column(self) -> None:
        """An action effect referencing an undeclared column fails S-03 check 1."""
        data = _load_example_dict("discrete_grid.yaml")
        # Inject an effect that references a non-existent column "stamina".
        data["actions"]["actions"][0]["effects"].append(
            {"target": "entity", "field": "stamina", "op": "add", "value": 1.0}
        )
        with pytest.raises(SemanticValidationError) as exc_info:
            compile_dict(data)
        # The error should reference check 1 (action_field_references_exist).
        errs = exc_info.value.result.errors
        assert any(e.check_id == "1" for e in errs)

    def test_semantic_error_carries_validation_result(self) -> None:
        """SemanticValidationError exposes the full ValidationResult for debugging."""
        data = _load_example_dict("discrete_grid.yaml")
        data["actions"]["actions"][0]["effects"].append(
            {"target": "entity", "field": "nonexistent", "op": "add", "value": 1.0}
        )
        with pytest.raises(SemanticValidationError) as exc_info:
            compile_dict(data)
        assert exc_info.value.result is not None
        assert not exc_info.value.result.is_valid
        assert len(exc_info.value.result.errors) >= 1


# ---------------------------------------------------------------------------
# End-to-end: compile → run → record transitions
# ---------------------------------------------------------------------------


class TestCompileEndToEnd:
    """End-to-end: compile → reset → step → checkpoint → restore → record."""

    def test_compile_and_record_transitions(
        self, discrete_spec: ScenarioSpec, tmp_path: Path
    ) -> None:
        """ACCEPTANCE §5: spec → ScenarioPackage → kernel end-to-end.

        Produce at least 5 transition records via TransitionRecorder.
        """
        pkg = compile_spec(discrete_spec)
        world = pkg.world_factory(seed=42)
        recorder = TransitionRecorder(
            output_dir=tmp_path / "records",
            world_id="discrete_grid_v0",
            producer_version="0.1.0",
            validate=False,
        )
        # Record 5 ticks of forage action.
        agent_id = world.observe().entities.ids[0]
        produced: list = []
        for _ in range(5):
            proposal = ActionProposal(
                agent_id=agent_id,
                action_type="forage",
                params={},
                proposed_at_tick=world.observe().meta.tick,
                proposer="test",
            )
            executed, _receipt = world.validate_action(proposal)
            record = world.step(executed)
            produced.append(record)
            recorder.append(record)
        recorder.close()
        # Every produced record must have non-empty before / after hashes.
        assert len(produced) == 5
        for r in produced:
            assert r.state_before_hash.startswith("sha256:")
            assert r.state_after_hash.startswith("sha256:")
            assert r.state_before_hash != r.state_after_hash
        # The recorder manifest must reflect 5 records, 0 quarantine.
        manifest = recorder.manifest()
        assert manifest.record_count == 5
        assert manifest.quarantine_count == 0

    def test_compile_and_checkpoint_restore_round_trip(
        self, continuous_spec: ScenarioSpec
    ) -> None:
        """Compile → reset → step → checkpoint → restore → state matches."""
        pkg = compile_spec(continuous_spec)
        world = pkg.world_factory(seed=123)
        # Take 3 steps to mutate state.
        agent_id = world.observe().entities.ids[0]
        for _ in range(3):
            proposal = ActionProposal(
                agent_id=agent_id,
                action_type="forage",
                params={},
                proposed_at_tick=world.observe().meta.tick,
                proposer="test",
            )
            executed, _ = world.validate_action(proposal)
            world.step(executed)
        # Checkpoint after 3 steps.
        ckpt = world.checkpoint()
        state_after_ckpt = hash_state(world.observe())
        # Take 2 more steps (these will be discarded).
        for _ in range(2):
            proposal = ActionProposal(
                agent_id=agent_id,
                action_type="forage",
                params={},
                proposed_at_tick=world.observe().meta.tick,
                proposer="test",
            )
            executed, _ = world.validate_action(proposal)
            world.step(executed)
        # State must have changed.
        assert hash_state(world.observe()) != state_after_ckpt
        # Restore — state must match the checkpoint.
        world.restore(ckpt)
        assert hash_state(world.observe()) == state_after_ckpt

    def test_compile_deterministic_replay(
        self, discrete_spec: ScenarioSpec
    ) -> None:
        """Same spec + same seed + same actions → same per-tick state hashes."""
        pkg = compile_spec(discrete_spec)
        # Run 1.
        w1 = pkg.world_factory(seed=99)
        agent_id = w1.observe().entities.ids[0]
        actions_1: list = []
        per_tick_hashes_1: list[str] = []
        for _ in range(5):
            proposal = ActionProposal(
                agent_id=agent_id,
                action_type="forage",
                params={},
                proposed_at_tick=w1.observe().meta.tick,
                proposer="test",
            )
            executed, _ = w1.validate_action(proposal)
            actions_1.append(executed)
            w1.step(executed)
            per_tick_hashes_1.append(hash_state(w1.observe()))
        # Run 2: fresh world, same seed, replay the same executed actions.
        w2 = pkg.world_factory(seed=99)
        per_tick_hashes_2: list[str] = []
        for executed in actions_1:
            w2.step(executed)
            per_tick_hashes_2.append(hash_state(w2.observe()))
        # Per-tick state hashes must match — exact_restore + deterministic replay.
        assert per_tick_hashes_1 == per_tick_hashes_2
