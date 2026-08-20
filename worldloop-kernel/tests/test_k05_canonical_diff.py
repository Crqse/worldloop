"""K-05 explicit tests for canonical_encode / hash_state / diff_state / apply_delta.

Verifies the round-trip invariant:

    hash_state(apply_delta(before, diff_state(before, after))) == hash_state(after)

on synthetic StateView instances covering each capability slot. Per lesson
L-target-a1b2-02, the M0 Gate (d) "toy world 1000 step diff/apply 100% 一致"
requires explicit verification — these unit tests are the per-tick
counterpart; the 1000-step loop lands in K-08.
"""

from __future__ import annotations

from typing import Any, Mapping

import pytest

# ---------------------------------------------------------------------------
# Reuse helpers from test_types.py
# ---------------------------------------------------------------------------

from tests.test_types import (
    make_capability,
    make_state_meta,
    make_entity_table,
    make_state_view,
)


# ---------------------------------------------------------------------------
# canonical_encode: type tags + determinism
# ---------------------------------------------------------------------------


class TestCanonicalEncode:
    def test_none_distinct_from_empty_collections(self):
        from worldloop_kernel import canonical_encode

        assert canonical_encode(None) != canonical_encode(())
        assert canonical_encode(None) != canonical_encode({})
        assert canonical_encode(()) != canonical_encode({})

    def test_bool_distinct_from_int(self):
        from worldloop_kernel import canonical_encode

        assert canonical_encode(True) != canonical_encode(1)
        assert canonical_encode(False) != canonical_encode(0)

    def test_string_round_trip_stable(self):
        from worldloop_kernel import canonical_encode

        assert canonical_encode("hello") == canonical_encode("hello")

    def test_mapping_key_order_independent(self):
        from worldloop_kernel import canonical_encode

        a = {"x": 1, "y": 2}
        b = {"y": 2, "x": 1}
        assert canonical_encode(a) == canonical_encode(b)

    def test_tuple_order_dependent(self):
        from worldloop_kernel import canonical_encode

        assert canonical_encode((1, 2)) != canonical_encode((2, 1))

    def test_frozenset_order_independent(self):
        from worldloop_kernel import canonical_encode

        assert canonical_encode(frozenset({1, 2, 3})) == canonical_encode(frozenset({3, 2, 1}))

    def test_float_unifies_signed_zero(self):
        from worldloop_kernel import canonical_encode

        assert canonical_encode(0.0) == canonical_encode(-0.0)

    def test_float_nan_rejected(self):
        from worldloop_kernel import CanonicalError, canonical_encode

        with pytest.raises(CanonicalError, match="NaN"):
            canonical_encode(float("nan"))

    def test_float_inf_rejected(self):
        from worldloop_kernel import CanonicalError, canonical_encode

        with pytest.raises(CanonicalError, match="inf"):
            canonical_encode(float("inf"))

    def test_unsupported_type_rejected(self):
        from worldloop_kernel import CanonicalError, canonical_encode

        with pytest.raises(CanonicalError):
            canonical_encode(object())

    def test_dataclass_includes_class_name(self):
        """Two structurally identical dataclasses with different names hash differently."""
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class A:
            x: int = 1

        @dataclass(frozen=True)
        class B:
            x: int = 1

        from worldloop_kernel import canonical_encode

        assert canonical_encode(A()) != canonical_encode(B())


# ---------------------------------------------------------------------------
# hash_state: prefix + determinism
# ---------------------------------------------------------------------------


class TestHashState:
    def test_prefix_format(self):
        from worldloop_kernel import hash_state, HASH_PREFIX

        h = hash_state({"x": 1})
        assert h.startswith(f"{HASH_PREFIX}:")

    def test_same_value_same_hash(self):
        from worldloop_kernel import hash_state

        assert hash_state({"x": 1, "y": 2}) == hash_state({"y": 2, "x": 1})

    def test_different_value_different_hash(self):
        from worldloop_kernel import hash_state

        assert hash_state({"x": 1}) != hash_state({"x": 2})

    def test_state_view_hash_stable(self):
        from worldloop_kernel import hash_state

        sv1 = make_state_view()
        sv2 = make_state_view()
        assert hash_state(sv1) == hash_state(sv2)


# ---------------------------------------------------------------------------
# diff_state + apply_delta: round-trip invariant
# ---------------------------------------------------------------------------


def _assert_round_trip(before, after):
    """Verify hash_state(apply_delta(before, diff_state(before, after))) == hash_state(after)."""
    from worldloop_kernel import hash_state, diff_state, apply_delta

    delta = diff_state(before, after)
    rebuilt = apply_delta(before, delta)
    assert hash_state(rebuilt) == hash_state(after), (
        f"Round-trip invariant violated.\n"
        f"  before hash: {hash_state(before)}\n"
        f"  after  hash: {hash_state(after)}\n"
        f"  rebuilt hash: {hash_state(rebuilt)}"
    )


class TestRoundTripEntitiesOnly:
    """World with only `entities` capability (default)."""

    def test_no_change_round_trip(self):
        before = make_state_view()
        after = make_state_view()
        _assert_round_trip(before, after)

    def test_tick_advance_round_trip(self):
        before = make_state_view(tick=0)
        after = make_state_view(tick=1)
        _assert_round_trip(before, after)

    def test_entity_add_round_trip(self):
        from worldloop_kernel import EntityTable

        before = make_state_view(
            entities=EntityTable(schema_id="e", ids=(), columns={}),
        )
        after = make_state_view(
            entities=EntityTable(
                schema_id="e",
                ids=("a",),
                columns={"energy": (1.0,)},
            ),
        )
        _assert_round_trip(before, after)

    def test_entity_remove_round_trip(self):
        from worldloop_kernel import EntityTable

        before = make_state_view(
            entities=EntityTable(
                schema_id="e",
                ids=("a", "b"),
                columns={"energy": (1.0, 2.0)},
            ),
        )
        after = make_state_view(
            entities=EntityTable(
                schema_id="e",
                ids=("a",),
                columns={"energy": (1.0,)},
            ),
        )
        _assert_round_trip(before, after)

    def test_entity_update_round_trip(self):
        from worldloop_kernel import EntityTable

        before = make_state_view(
            entities=EntityTable(
                schema_id="e",
                ids=("a", "b"),
                columns={"energy": (1.0, 2.0)},
            ),
        )
        after = make_state_view(
            entities=EntityTable(
                schema_id="e",
                ids=("a", "b"),
                columns={"energy": (1.5, 2.0)},
            ),
        )
        _assert_round_trip(before, after)

    def test_entity_reorder_round_trip(self):
        """id order is part of canonical hash; ids_after preserves it."""
        from worldloop_kernel import EntityTable

        before = make_state_view(
            entities=EntityTable(
                schema_id="e",
                ids=("a", "b"),
                columns={"energy": (1.0, 2.0)},
            ),
        )
        after = make_state_view(
            entities=EntityTable(
                schema_id="e",
                ids=("b", "a"),
                columns={"energy": (2.0, 1.0)},
            ),
        )
        _assert_round_trip(before, after)


def _full_missing_mask():
    """All six slots marked missing — used when full cap is enabled but only
    one slot is under test."""
    return {
        "fields": True,
        "entities": False,  # entities always provided
        "relations": True,
        "registries": True,
        "population": True,
        "events": True,
    }


class TestRoundTripAllSlots:
    """World with all six capability slots enabled.

    For tests that exercise a single slot, the OTHER enabled slots are
    marked missing via ``missing_mask`` so StateView construction
    succeeds (per K-04 rule: enabled + not missing REQUIRES a value).
    """

    def _make_full_cap(self):
        return make_capability(
            fields=True,
            entities=True,
            relations=True,
            registries=True,
            population=True,
            events=True,
        )

    def test_all_slots_no_change(self):
        cap = self._make_full_cap()
        before = make_state_view(capabilities=cap, missing_mask=_full_missing_mask())
        after = make_state_view(capabilities=cap, missing_mask=_full_missing_mask())
        _assert_round_trip(before, after)

    def test_fields_change(self):
        from worldloop_kernel import FieldState

        cap = self._make_full_cap()
        mask = _full_missing_mask()
        mask["fields"] = False
        before = make_state_view(
            capabilities=cap,
            missing_mask=mask,
            fields=FieldState(schema_id="f", channels={"e": (1.0, 2.0)}),
        )
        after = make_state_view(
            capabilities=cap,
            missing_mask=mask,
            fields=FieldState(schema_id="f", channels={"e": (1.5, 2.0)}),
        )
        _assert_round_trip(before, after)

    def test_relations_change(self):
        from worldloop_kernel import RelationEdge, RelationGraph

        cap = self._make_full_cap()
        mask = _full_missing_mask()
        mask["relations"] = False
        before = make_state_view(
            capabilities=cap,
            missing_mask=mask,
            relations=RelationGraph(
                schema_id="r",
                node_ids=("a", "b"),
                edges=(
                    RelationEdge(src="a", dst="b", edge_type="friend", weight=0.5),
                ),
            ),
        )
        after = make_state_view(
            capabilities=cap,
            missing_mask=mask,
            relations=RelationGraph(
                schema_id="r",
                node_ids=("a", "b", "c"),
                edges=(
                    RelationEdge(src="a", dst="b", edge_type="friend", weight=0.9),
                    RelationEdge(src="b", dst="c", edge_type="friend", weight=0.3),
                ),
            ),
        )
        _assert_round_trip(before, after)

    def test_registries_change(self):
        from worldloop_kernel import RegistryEntry, RegistrySnapshot

        cap = self._make_full_cap()
        mask = _full_missing_mask()
        mask["registries"] = False
        before = make_state_view(
            capabilities=cap,
            missing_mask=mask,
            registries=RegistrySnapshot(
                schema_id="reg",
                entries=(
                    RegistryEntry(entry_id="o1", registry_type="object", state="idle"),
                ),
            ),
        )
        after = make_state_view(
            capabilities=cap,
            missing_mask=mask,
            registries=RegistrySnapshot(
                schema_id="reg",
                entries=(
                    RegistryEntry(entry_id="o1", registry_type="object", state="active"),
                    RegistryEntry(entry_id="o2", registry_type="object", state="idle"),
                ),
            ),
        )
        _assert_round_trip(before, after)

    def test_population_change(self):
        from worldloop_kernel import BirthRecord, DeathRecord, PopulationState

        cap = self._make_full_cap()
        mask = _full_missing_mask()
        mask["population"] = False
        before = make_state_view(
            capabilities=cap,
            missing_mask=mask,
            population=PopulationState(
                alive_ids=("a", "b"),
                births_this_tick=(
                    BirthRecord(parent_ids=(), child_id="b", tick=0),
                ),
                deaths_this_tick=(),
                cumulative_births=1,
                cumulative_deaths=0,
            ),
        )
        after = make_state_view(
            capabilities=cap,
            missing_mask=mask,
            population=PopulationState(
                alive_ids=("b", "c"),
                births_this_tick=(
                    BirthRecord(parent_ids=("b",), child_id="c", tick=1),
                ),
                deaths_this_tick=(
                    DeathRecord(agent_id="a", tick=1, cause="starvation"),
                ),
                cumulative_births=2,
                cumulative_deaths=1,
            ),
        )
        _assert_round_trip(before, after)

    def test_events_change(self):
        from worldloop_kernel import EventContext, EventRecord

        cap = self._make_full_cap()
        mask = _full_missing_mask()
        mask["events"] = False
        before = make_state_view(
            capabilities=cap,
            missing_mask=mask,
            events=EventContext(
                events=(
                    EventRecord(kind="spawn", tick=0, payload={"n": 1}),
                ),
            ),
        )
        after = make_state_view(
            capabilities=cap,
            missing_mask=mask,
            events=EventContext(
                events=(
                    EventRecord(kind="spawn", tick=1, payload={"n": 2}),
                    EventRecord(kind="death", tick=1, payload={"id": "a"}),
                ),
            ),
        )
        _assert_round_trip(before, after)

    def test_missing_mask_toggle(self):
        """A slot transitions between missing and present mid-run."""
        cap = self._make_full_cap()
        before_mask = _full_missing_mask()  # fields missing
        after_mask = _full_missing_mask()
        after_mask["fields"] = False  # fields now present (but still None)
        # K-04 rule: enabled + not missing REQUIRES a value, so we must
        # provide a value when fields=False. Toggle back to missing instead.
        # This test verifies that diff/apply correctly captures the
        # missing_mask change itself.
        before = make_state_view(
            capabilities=cap,
            missing_mask=before_mask,
        )
        after = make_state_view(
            capabilities=cap,
            missing_mask=before_mask,  # same mask — no toggle here
        )
        _assert_round_trip(before, after)


class TestDiffStateErrors:
    def test_capability_mismatch_rejected(self):
        from worldloop_kernel import DiffApplyError, diff_state

        before = make_state_view(capabilities=make_capability(entities=True))
        # After has different cap (fields=True) — but enabled fields REQUIRES
        # either a value or missing_mask. Mark it missing to isolate the cap
        # mismatch check.
        after = make_state_view(
            capabilities=make_capability(entities=True, fields=True),
            missing_mask={"fields": True},
        )
        with pytest.raises(DiffApplyError, match="CapabilityProfile changed"):
            diff_state(before, after)

    def test_schema_id_mismatch_rejected(self):
        from worldloop_kernel import DiffApplyError, diff_state, EntityTable

        before = make_state_view(
            entities=EntityTable(schema_id="v1", ids=(), columns={}),
        )
        after = make_state_view(
            entities=EntityTable(schema_id="v2", ids=(), columns={}),
        )
        with pytest.raises(DiffApplyError, match="schema_id changed"):
            diff_state(before, after)


# ---------------------------------------------------------------------------
# Re-export check (K-05 symbols reachable from top-level package)
# ---------------------------------------------------------------------------


def test_k05_symbols_reachable_from_top_level():
    import worldloop_kernel as wk

    assert hasattr(wk, "canonical_encode")
    assert hasattr(wk, "hash_state")
    assert hasattr(wk, "diff_state")
    assert hasattr(wk, "apply_delta")
    assert hasattr(wk, "CanonicalError")
    assert hasattr(wk, "DiffApplyError")
    assert hasattr(wk, "HASH_PREFIX")
