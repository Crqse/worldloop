"""K-04 behavior tests for the minimal ten public types.

Covers the K-04-relevant subset of §10.3 M0 tests:
- capability / missing_mask consistency
- candidate / executed action separation
- receipt required fields + outcome_code/success pairing
- transition schema (executed/receipts key matching, schema_version)
- checkpoint exact_restore / checksum rule
- WorldProtocol is runtime_checkable
- frozen dataclass mutation is rejected
- re-exports from the top-level package work
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping

import pytest

# ---------------------------------------------------------------------------
# Helpers — minimal valid instances of each type
# ---------------------------------------------------------------------------


def make_capability(
    *,
    fields: bool = False,
    entities: bool = True,
    relations: bool = False,
    registries: bool = False,
    population: bool = False,
    events: bool = False,
    exact_restore: bool = True,
    executable_deterministic_replay: bool = True,
    authority: str = "rule",
    ground_truth: bool = True,
    transition_mode: str = "deterministic",
) -> "CapabilityProfile":
    from worldloop_kernel import CapabilityProfile

    return CapabilityProfile(
        fields=fields,
        entities=entities,
        relations=relations,
        registries=registries,
        population=population,
        events=events,
        exact_restore=exact_restore,
        executable_deterministic_replay=executable_deterministic_replay,
        authority=authority,  # type: ignore[arg-type]
        ground_truth=ground_truth,
        transition_mode=transition_mode,  # type: ignore[arg-type]
    )


def make_state_meta(*, tick: int = 0) -> "StateMeta":
    from worldloop_kernel import StateMeta

    return StateMeta(
        scenario_id="test-scenario",
        run_id="test-run",
        tick=tick,
        config_hash="sha256:deadbeef",
        rng_state_ref="MT19937:abc",
    )


def make_entity_table(*, ids: tuple = (), columns: Mapping[str, tuple] | None = None) -> "EntityTable":
    from worldloop_kernel import EntityTable

    return EntityTable(
        schema_id="entity-v1",
        ids=ids,
        columns=columns or {},
    )


def make_state_view(
    *,
    capabilities: "CapabilityProfile | None" = None,
    entities: "EntityTable | None" = None,
    missing_mask: Mapping[str, bool] | None = None,
    fields: Any = None,
    relations: Any = None,
    registries: Any = None,
    population: Any = None,
    events: Any = None,
    tick: int = 0,
) -> "StateView":
    from worldloop_kernel import StateView

    cap = capabilities or make_capability()
    return StateView(
        meta=make_state_meta(tick=tick),
        entities=entities or make_entity_table(),
        capabilities=cap,
        missing_mask=missing_mask or {},
        fields=fields,
        relations=relations,
        registries=registries,
        population=population,
        events=events,
    )


def make_proposal(*, agent_id: str | int = "a1", tick: int = 0) -> "ActionProposal":
    from worldloop_kernel import ActionProposal

    return ActionProposal(
        agent_id=agent_id,
        action_type="FORAGE",
        params={"target_id": 7},
        proposed_at_tick=tick,
        proposer="reflex",
    )


def make_executed(*, agent_id: str | int = "a1", tick: int = 0) -> "ExecutedAction":
    from worldloop_kernel import ExecutedAction

    return ExecutedAction(
        agent_id=agent_id,
        action_type="FORAGE",
        params={"target_id": 7},
        executed_at_tick=tick,
        proposal_hash="sha256:proposal",
    )


def make_receipt(
    *,
    executed_action_hash: str = "sha256:executed",
    outcome_code: str = "ok",
    success: bool = True,
    energy_delta: float = 1.5,
) -> "ActionReceipt":
    from worldloop_kernel import ActionReceipt

    return ActionReceipt(
        executed_action_hash=executed_action_hash,
        outcome_code=outcome_code,
        success=success,
        energy_delta=energy_delta,
    )


# ---------------------------------------------------------------------------
# 1. All ten types are frozen dataclasses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "type_name, factory",
    [
        ("CapabilityProfile", make_capability),
        ("StateMeta", make_state_meta),
        ("EntityTable", make_entity_table),
        ("StateView", make_state_view),
        ("ActionProposal", make_proposal),
        ("ExecutedAction", make_executed),
        ("ExogenousInput", lambda: __import__(
            "worldloop_kernel", fromlist=["ExogenousInput"]
        ).ExogenousInput(tick=0, kind="pulse", payload={})),
        ("ActionReceipt", make_receipt),
        ("StateDelta", lambda: __import__(
            "worldloop_kernel", fromlist=["StateDelta"]
        ).StateDelta()),
        ("TransitionRecord", None),  # filled in test body
        ("Checkpoint", None),  # filled in test body
    ],
)
def test_type_is_frozen_dataclass(type_name: str, factory):
    """Every public K-04 type MUST be a frozen dataclass."""
    import worldloop_kernel as wk

    typ = getattr(wk, type_name)
    assert dataclasses.is_dataclass(typ), f"{type_name} is not a dataclass"
    # frozen=True is stored in the __dataclass_params__ attribute.
    params = typ.__dataclass_params__
    assert params.frozen, f"{type_name} is not frozen"


def test_frozen_mutation_rejected():
    """Frozen dataclasses MUST raise FrozenInstanceError on mutation."""
    cap = make_capability()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cap.fields = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. CapabilityProfile validators
# ---------------------------------------------------------------------------


class TestCapabilityProfile:
    def test_minimal_valid(self):
        cap = make_capability()
        assert cap.entities is True
        assert cap.authority == "rule"
        assert cap.ground_truth is True

    def test_learned_authority_requires_no_ground_truth(self):
        from worldloop_kernel import CapabilityError

        with pytest.raises(CapabilityError, match="learned.*ground_truth"):
            make_capability(authority="learned", ground_truth=True)

    def test_learned_authority_with_no_ground_truth_ok(self):
        cap = make_capability(authority="learned", ground_truth=False)
        assert cap.authority == "learned"
        assert cap.ground_truth is False

    def test_exact_restore_requires_deterministic_replay(self):
        from worldloop_kernel import CapabilityError

        with pytest.raises(CapabilityError, match="exact_restore.*replay"):
            make_capability(
                exact_restore=True,
                executable_deterministic_replay=False,
            )

    def test_entities_false_rejected(self):
        """Per main plan §4.2: a world without entities is not a world."""
        from worldloop_kernel import CapabilityError

        with pytest.raises(CapabilityError, match="entities=False"):
            make_capability(entities=False)

    def test_slot_flags_returns_all_six_slots(self):
        from worldloop_kernel import CAPABILITY_SLOTS

        cap = make_capability(fields=True, entities=True, relations=True)
        flags = cap.slot_flags()
        assert set(flags) == set(CAPABILITY_SLOTS)
        assert flags["fields"] is True
        assert flags["entities"] is True
        assert flags["relations"] is True
        assert flags["registries"] is False
        assert flags["population"] is False
        assert flags["events"] is False


# ---------------------------------------------------------------------------
# 3. StateView capability / missing_mask consistency
# ---------------------------------------------------------------------------


class TestStateViewConsistency:
    def test_minimal_valid(self):
        sv = make_state_view()
        assert sv.entities is not None
        assert sv.fields is None  # capability.fields=False

    def test_missing_mask_unknown_key_rejected(self):
        from worldloop_kernel import StateError

        with pytest.raises(StateError, match="not in CAPABILITY_SLOTS"):
            make_state_view(missing_mask={"nonexistent_slot": True})

    def test_missing_mask_true_for_disabled_capability_rejected(self):
        """missing_mask may only be True for slots the world has."""
        from worldloop_kernel import StateError

        # cap.fields=False but missing_mask["fields"]=True → error
        with pytest.raises(StateError, match="capabilities.fields=False"):
            make_state_view(missing_mask={"fields": True})

    def test_disabled_capability_with_value_rejected(self):
        """A slot the world does NOT have must be None."""
        from worldloop_kernel import StateError
        from worldloop_kernel import FieldState

        # cap.fields=False but sv.fields=<FieldState> → error
        with pytest.raises(StateError, match="capabilities.fields=False"):
            make_state_view(
                fields=FieldState(schema_id="f", channels={}, units={}),
            )

    def test_enabled_capability_with_none_and_not_missing_rejected(self):
        """A slot the world has, marked not missing, MUST have a value."""
        from worldloop_kernel import StateError

        cap = make_capability(fields=True)
        with pytest.raises(StateError, match="slot value is None"):
            make_state_view(capabilities=cap, fields=None, missing_mask={})

    def test_enabled_capability_with_value_and_missing_rejected(self):
        """A slot marked missing MUST be None."""
        from worldloop_kernel import StateError
        from worldloop_kernel import FieldState

        cap = make_capability(fields=True)
        with pytest.raises(StateError, match="missing.*not None"):
            make_state_view(
                capabilities=cap,
                fields=FieldState(schema_id="f", channels={}, units={}),
                missing_mask={"fields": True},
            )

    def test_enabled_capability_with_missing_and_none_ok(self):
        """A slot the world has, marked missing, MAY be None."""
        from worldloop_kernel import FieldState  # noqa: F401

        cap = make_capability(fields=True)
        sv = make_state_view(
            capabilities=cap,
            fields=None,
            missing_mask={"fields": True},
        )
        assert sv.fields is None
        assert sv.missing_mask["fields"] is True

    def test_enabled_capability_with_value_ok(self):
        """A slot the world has, marked not missing, MAY have a value."""
        from worldloop_kernel import FieldState

        cap = make_capability(fields=True)
        sv = make_state_view(
            capabilities=cap,
            fields=FieldState(schema_id="f", channels={"e": (1.0, 2.0)}),
            missing_mask={},
        )
        assert sv.fields is not None
        assert sv.fields.channels["e"] == (1.0, 2.0)


# ---------------------------------------------------------------------------
# 4. EntityTable column alignment
# ---------------------------------------------------------------------------


class TestEntityTable:
    def test_aligned_columns_ok(self):
        from worldloop_kernel import EntityTable

        et = EntityTable(
            schema_id="e",
            ids=("a", "b"),
            columns={"energy": (1.0, 2.0), "x": (0, 1)},
        )
        assert len(et.ids) == 2
        assert et.columns["energy"] == (1.0, 2.0)

    def test_misaligned_columns_rejected(self):
        from worldloop_kernel import EntityTable, StateError

        with pytest.raises(StateError, match="must align"):
            EntityTable(
                schema_id="e",
                ids=("a", "b"),
                columns={"energy": (1.0,)},  # only 1 value for 2 ids
            )


# ---------------------------------------------------------------------------
# 5. Action types: candidate / executed separation + outcome codes
# ---------------------------------------------------------------------------


class TestActionTypes:
    def test_proposal_and_executed_are_distinct_types(self):
        """ADR §3 / main plan §4.6: candidate and executed are separate."""
        from worldloop_kernel import ActionProposal, ExecutedAction

        proposal = make_proposal()
        executed = make_executed()
        assert not isinstance(proposal, ExecutedAction)
        assert not isinstance(executed, ActionProposal)
        assert type(proposal) is not type(executed)

    def test_proposal_empty_action_type_rejected(self):
        from worldloop_kernel import ActionError

        with pytest.raises(ActionError, match="action_type"):
            make_proposal().__class__(
                agent_id="a1",
                action_type="",
                params={},
                proposed_at_tick=0,
                proposer="reflex",
            )

    def test_proposal_negative_tick_rejected(self):
        from worldloop_kernel import ActionError, ActionProposal

        with pytest.raises(ActionError, match=">= 0"):
            ActionProposal(
                agent_id="a1",
                action_type="FORAGE",
                params={},
                proposed_at_tick=-1,
                proposer="reflex",
            )

    def test_executed_empty_proposal_hash_rejected(self):
        from worldloop_kernel import ActionError, ExecutedAction

        with pytest.raises(ActionError, match="proposal_hash"):
            ExecutedAction(
                agent_id="a1",
                action_type="FORAGE",
                params={},
                executed_at_tick=0,
                proposal_hash="",
            )

    def test_receipt_success_true_with_non_ok_code_rejected(self):
        """success=True MUST pair with outcome_code='ok'."""
        from worldloop_kernel import ActionError

        with pytest.raises(ActionError, match="success=True"):
            make_receipt().__class__(
                executed_action_hash="h",
                outcome_code="disabled_by_ablation",
                success=True,
                energy_delta=0.0,
            )

    def test_receipt_success_false_with_ok_code_rejected(self):
        """success=False MUST NOT pair with outcome_code='ok'."""
        from worldloop_kernel import ActionError

        with pytest.raises(ActionError, match="success=False"):
            make_receipt().__class__(
                executed_action_hash="h",
                outcome_code="ok",
                success=False,
                energy_delta=0.0,
            )

    def test_receipt_empty_outcome_code_rejected(self):
        from worldloop_kernel import ActionError

        with pytest.raises(ActionError, match="outcome_code"):
            make_receipt().__class__(
                executed_action_hash="h",
                outcome_code="",
                success=False,
                energy_delta=0.0,
            )

    def test_kernel_outcome_codes_non_empty_strings(self):
        from worldloop_kernel import KERNEL_OUTCOME_CODES

        assert len(KERNEL_OUTCOME_CODES) >= 4
        for code in KERNEL_OUTCOME_CODES:
            assert isinstance(code, str)
            assert code

    def test_outcome_ok_constant(self):
        from worldloop_kernel import OUTCOME_OK, KERNEL_OUTCOME_CODES

        assert OUTCOME_OK == "ok"
        assert OUTCOME_OK in KERNEL_OUTCOME_CODES


# ---------------------------------------------------------------------------
# 6. ExogenousInput basic validation
# ---------------------------------------------------------------------------


class TestExogenousInput:
    def test_valid(self):
        from worldloop_kernel import ExogenousInput

        ei = ExogenousInput(tick=5, kind="resource_pulse", payload={"amount": 10})
        assert ei.tick == 5
        assert ei.kind == "resource_pulse"

    def test_empty_kind_rejected(self):
        from worldloop_kernel import ActionError, ExogenousInput

        with pytest.raises(ActionError, match="kind"):
            ExogenousInput(tick=0, kind="", payload={})


# ---------------------------------------------------------------------------
# 7. TransitionRecord schema and key matching
# ---------------------------------------------------------------------------


class TestTransitionRecord:
    def _make_valid(self):
        from worldloop_kernel import (
            StateDelta,
            TransitionRecord,
            PROTOCOL_SCHEMA_VERSION,
        )

        return TransitionRecord(
            schema_version=PROTOCOL_SCHEMA_VERSION,
            producer_id="worldloop-native-v1",
            producer_version="0.1.0",
            tick=0,
            state_before_hash="sha256:before",
            candidate_actions={},
            executed_actions={},
            exogenous_input=None,
            receipts={},
            state_delta=StateDelta(),
            state_after_hash="sha256:after",
            capability_profile=make_capability(),
        )

    def test_minimal_valid(self):
        rec = self._make_valid()
        assert rec.tick == 0
        assert rec.producer_id == "worldloop-native-v1"

    def test_wrong_schema_version_rejected(self):
        from worldloop_kernel import TransitionError, TransitionRecord

        with pytest.raises(TransitionError, match="schema_version"):
            TransitionRecord(
                schema_version="99.0.0",  # wrong
                producer_id="p",
                producer_version="v",
                tick=0,
                state_before_hash="h1",
                candidate_actions={},
                executed_actions={},
                exogenous_input=None,
                receipts={},
                state_delta=__import__(
                    "worldloop_kernel", fromlist=["StateDelta"]
                ).StateDelta(),
                state_after_hash="h2",
                capability_profile=make_capability(),
            )

    def test_executed_receipts_key_mismatch_rejected(self):
        from worldloop_kernel import (
            TransitionError,
            TransitionRecord,
            StateDelta,
            PROTOCOL_SCHEMA_VERSION,
        )

        executed = make_executed(agent_id="a1")
        receipt = make_receipt()
        with pytest.raises(TransitionError, match="keys"):
            TransitionRecord(
                schema_version=PROTOCOL_SCHEMA_VERSION,
                producer_id="p",
                producer_version="v",
                tick=0,
                state_before_hash="h1",
                candidate_actions={},
                executed_actions={"a1": executed},
                exogenous_input=None,
                receipts={"a2": receipt},  # different key
                state_delta=StateDelta(),
                state_after_hash="h2",
                capability_profile=make_capability(),
            )

    def test_executed_receipts_key_match_ok(self):
        from worldloop_kernel import (
            TransitionRecord,
            StateDelta,
            PROTOCOL_SCHEMA_VERSION,
        )

        executed = make_executed(agent_id="a1")
        receipt = make_receipt()
        rec = TransitionRecord(
            schema_version=PROTOCOL_SCHEMA_VERSION,
            producer_id="p",
            producer_version="v",
            tick=0,
            state_before_hash="h1",
            candidate_actions={},
            executed_actions={"a1": executed},
            exogenous_input=None,
            receipts={"a1": receipt},
            state_delta=StateDelta(),
            state_after_hash="h2",
            capability_profile=make_capability(),
        )
        assert "a1" in rec.executed_actions
        assert "a1" in rec.receipts

    def test_negative_tick_rejected(self):
        from worldloop_kernel import (
            TransitionError,
            TransitionRecord,
            StateDelta,
            PROTOCOL_SCHEMA_VERSION,
        )

        with pytest.raises(TransitionError, match=">= 0"):
            TransitionRecord(
                schema_version=PROTOCOL_SCHEMA_VERSION,
                producer_id="p",
                producer_version="v",
                tick=-1,
                state_before_hash="h1",
                candidate_actions={},
                executed_actions={},
                exogenous_input=None,
                receipts={},
                state_delta=StateDelta(),
                state_after_hash="h2",
                capability_profile=make_capability(),
            )


# ---------------------------------------------------------------------------
# 8. Checkpoint exact_restore / checksum rule
# ---------------------------------------------------------------------------


class TestCheckpoint:
    def _make_valid(self, *, exact_restore: bool = True, checksum: str = "sha256:opaque"):
        from worldloop_kernel import Checkpoint, PROTOCOL_SCHEMA_VERSION

        cap = make_capability(exact_restore=exact_restore)
        return Checkpoint(
            schema_version=PROTOCOL_SCHEMA_VERSION,
            world_id="worldloop-native-v1",
            world_version="0.1.0",
            tick=0,
            state_view=make_state_view(capabilities=cap),
            opaque_payload=b"\x80\x04\x95...pickle...",
            payload_codec="pickle+v1",
            capability_profile=cap,
            rng_bundle={"main": "MT19937:abc"},
            checksum=checksum,
        )

    def test_minimal_valid(self):
        cp = self._make_valid()
        assert cp.tick == 0
        assert cp.payload_codec == "pickle+v1"

    def test_exact_restore_requires_checksum(self):
        from worldloop_kernel import TransitionError

        with pytest.raises(TransitionError, match="exact_restore.*checksum"):
            self._make_valid(exact_restore=True, checksum="")

    def test_non_exact_restore_allows_empty_checksum(self):
        cp = self._make_valid(exact_restore=False, checksum="")
        assert cp.checksum == ""

    def test_opaque_payload_must_be_bytes(self):
        from worldloop_kernel import (
            Checkpoint,
            TransitionError,
            PROTOCOL_SCHEMA_VERSION,
        )

        cap = make_capability(exact_restore=False)
        with pytest.raises(TransitionError, match="opaque_payload"):
            Checkpoint(
                schema_version=PROTOCOL_SCHEMA_VERSION,
                world_id="w",
                world_version="v",
                tick=0,
                state_view=make_state_view(capabilities=cap),
                opaque_payload="not-bytes",  # type: ignore[arg-type]
                payload_codec="pickle+v1",
                capability_profile=cap,
            )


# ---------------------------------------------------------------------------
# 9. WorldProtocol is a runtime_checkable Protocol
# ---------------------------------------------------------------------------


class TestWorldProtocol:
    def test_is_protocol(self):
        from typing import Protocol

        from worldloop_kernel import WorldProtocol

        # Protocols have a ``_is_protocol`` attribute in CPython.
        assert hasattr(WorldProtocol, "_is_protocol")
        # And they are subclasses of typing.Protocol (via _SpecialForm).
        # We test this indirectly: a non-Protocol class would NOT have
        # ``_is_protocol`` set to True.
        assert WorldProtocol._is_protocol is True

    def test_runtime_checkable(self):
        """WorldProtocol must be @runtime_checkable so the kernel can
        isinstance-check world implementations."""
        from worldloop_kernel import WorldProtocol

        # @runtime_checkable Protocols support isinstance.
        class StubWorld:
            @property
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

        assert isinstance(StubWorld(), WorldProtocol)

    def test_stub_without_methods_fails_runtime_check(self):
        from worldloop_kernel import WorldProtocol

        class Incomplete:
            pass

        assert not isinstance(Incomplete(), WorldProtocol)


# ---------------------------------------------------------------------------
# 10. PROTOCOL_SCHEMA_VERSION consistency
# ---------------------------------------------------------------------------


def test_protocol_schema_version_matches():
    """TransitionRecord/Checkpoint default schema_version is the same
    constant re-exported from the package."""
    import worldloop_kernel as wk

    assert wk.PROTOCOL_SCHEMA_VERSION == "0.1.0"
    # The constant must appear in both modules that reference it.
    from worldloop_kernel.transition import PROTOCOL_SCHEMA_VERSION as t_version

    assert wk.PROTOCOL_SCHEMA_VERSION is t_version


# ---------------------------------------------------------------------------
# 11. Top-level re-exports work
# ---------------------------------------------------------------------------


def test_top_level_reexports():
    """All ten public types MUST be importable from the top-level package."""
    import worldloop_kernel as wk

    # The minimal ten public types per main plan §4.5.
    ten_types = [
        "CapabilityProfile",
        "StateView",
        "ActionProposal",
        "ExecutedAction",
        "ExogenousInput",
        "ActionReceipt",
        "StateDelta",
        "TransitionRecord",
        "Checkpoint",
        "WorldProtocol",
    ]
    for name in ten_types:
        assert hasattr(wk, name), f"{name} not re-exported from worldloop_kernel"
        obj = getattr(wk, name)
        assert dataclasses.is_dataclass(obj) or name == "WorldProtocol", (
            f"{name} is not a dataclass (WorldProtocol is a Protocol, allowed)"
        )


# ---------------------------------------------------------------------------
# 12. Per-slot change sub-types (used by StateDelta) — basic construction
# ---------------------------------------------------------------------------


class TestStateDeltaSubTypes:
    def test_entity_change_update_requires_column(self):
        from worldloop_kernel import EntityChange, TransitionError

        with pytest.raises(TransitionError, match="update.*column"):
            EntityChange(kind="update", entity_id="a1", column="")

    def test_entity_change_invalid_kind_rejected(self):
        from worldloop_kernel import EntityChange, TransitionError

        with pytest.raises(TransitionError, match="kind"):
            EntityChange(kind="teleport", entity_id="a1")

    def test_population_change_birth_requires_parent_ids(self):
        from worldloop_kernel import PopulationChange, TransitionError

        with pytest.raises(TransitionError, match="birth.*parent_ids"):
            PopulationChange(
                kind="birth",
                agent_id="c1",
                tick=0,
                parent_ids=None,
            )

    def test_population_change_death_requires_cause(self):
        from worldloop_kernel import PopulationChange, TransitionError

        with pytest.raises(TransitionError, match="death.*cause"):
            PopulationChange(
                kind="death",
                agent_id="a1",
                tick=0,
                cause="",
            )

    def test_state_delta_default_empty(self):
        from worldloop_kernel import StateDelta

        delta = StateDelta()
        # K-05: all slots default to None (capability-not-declared or
        # no change). event_log is now None = "no change" (was () in
        # K-04, but K-05 made it consistent with other slots so that
        # None = no change, () = after has no events, (...) = after
        # has these events).
        assert delta.field_changes is None
        assert delta.entity_changes is None
        assert delta.relation_changes is None
        assert delta.registry_changes is None
        assert delta.population_changes is None
        assert delta.event_log is None
        assert delta.meta_after is None
        assert delta.missing_mask_after is None
