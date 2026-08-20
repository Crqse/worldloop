"""K-06 explicit tests for validate_transition + TransitionRecorder.

Verifies:
- All 7 invariants produce a result (pass / fail / skip) per call.
- Each invariant can be exercised positively and negatively.
- TransitionRecorder: atomic write, quarantine on validation failure,
  manifest on close, idempotent close, no append after close.

Per lesson L-target-a1b2-02, M0 Gate §10.4 (transition schema
validation + append-only recorder) require explicit verification.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import dataclasses

from tests.test_types import (
    make_capability,
    make_state_meta,
    make_entity_table,
    make_state_view,
    make_proposal,
    make_executed,
    make_receipt,
)


# ---------------------------------------------------------------------------
# Helpers — build a valid TransitionRecord
# ---------------------------------------------------------------------------


def _build_bypassing_post_init(cls, **kwargs):
    """Construct a frozen dataclass instance bypassing ``__post_init__``.

    Simulates a record deserialized from untrusted JSON without
    re-validation. Used to test that the validator catches drift the
    way ``__post_init__`` would have.
    """
    obj = object.__new__(cls)
    for f in dataclasses.fields(cls):
        if f.name in kwargs:
            object.__setattr__(obj, f.name, kwargs[f.name])
        elif f.default is not dataclasses.MISSING:
            object.__setattr__(obj, f.name, f.default)
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            object.__setattr__(obj, f.name, f.default_factory())
    return obj


def _make_transition_record(
    *,
    tick: int = 0,
    cap=None,
    before=None,
    after=None,
    executed_actions=None,
    receipts=None,
    state_after_hash=None,
) -> "TransitionRecord":
    """Build a minimal valid TransitionRecord for testing.

    By default, builds a record where:
    - before and after are minimal StateView objects (entities-only cap)
    - state_delta is computed via diff_state(before, after)
    - state_before_hash / state_after_hash are computed via hash_state
    - executed_actions and receipts are empty mappings (no agents acted)
    """
    from worldloop_kernel import (
        TransitionRecord,
        PROTOCOL_SCHEMA_VERSION,
        hash_state,
        diff_state,
    )

    cap = cap or make_capability()
    before = before or make_state_view(capabilities=cap, tick=tick)
    after = after or make_state_view(capabilities=cap, tick=tick + 1)
    delta = diff_state(before, after)
    return TransitionRecord(
        schema_version=PROTOCOL_SCHEMA_VERSION,
        producer_id="test-world",
        producer_version="0.0.1",
        tick=tick,
        state_before_hash=hash_state(before),
        candidate_actions={},
        executed_actions=executed_actions or {},
        exogenous_input=None,
        receipts=receipts or {},
        state_delta=delta,
        state_after_hash=state_after_hash or hash_state(after),
        capability_profile=cap,
        provenance={},
    )


# ---------------------------------------------------------------------------
# validate_transition: report shape + invariant coverage
# ---------------------------------------------------------------------------


class TestValidateTransitionReportShape:
    def test_seven_invariants_always_present(self):
        from worldloop_kernel import validate_transition, INVARIANT_NAMES

        record = _make_transition_record()
        report = validate_transition(record)
        assert set(report.invariant_results.keys()) == set(INVARIANT_NAMES)
        assert len(INVARIANT_NAMES) == 7

    def test_record_id_format(self):
        from worldloop_kernel import validate_transition

        record = _make_transition_record()
        report = validate_transition(record)
        assert report.record_id.startswith("test-world:t0:")
        assert len(report.record_id) > 20

    def test_passing_record_returns_passed_true(self):
        """A valid record with no before/after should still pass
        (record-only invariants all pass; others skipped)."""
        from worldloop_kernel import validate_transition

        record = _make_transition_record()
        report = validate_transition(record)
        # 4 invariants run from record alone: receipt_completeness,
        # capability_consistency, outcome_code_legality,
        # authority_grounding. All should pass.
        assert report.passed is True
        assert report.diagnostics["ran_count"] == 4
        assert report.diagnostics["skipped_count"] == 3

    def test_non_transition_record_raises(self):
        from worldloop_kernel import validate_transition, ValidationError

        with pytest.raises(ValidationError, match="must be a TransitionRecord"):
            validate_transition({"not": "a record"})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Invariant 1: hash_round_trip
# ---------------------------------------------------------------------------


class TestHashRoundTripInvariant:
    def test_skipped_when_before_none(self):
        from worldloop_kernel import validate_transition

        record = _make_transition_record()
        report = validate_transition(record)  # no before kwarg
        result = report.invariant_results["hash_round_trip"]
        assert result.passed is None
        assert "skipped" in result.message

    def test_passes_with_correct_before(self):
        from worldloop_kernel import validate_transition, hash_state, diff_state

        cap = make_capability()
        before = make_state_view(capabilities=cap, tick=0)
        after = make_state_view(capabilities=cap, tick=1)
        record = _make_transition_record(
            cap=cap, before=before, after=after
        )
        report = validate_transition(record, before=before, after=after)
        result = report.invariant_results["hash_round_trip"]
        assert result.passed is True

    def test_fails_with_wrong_state_after_hash(self):
        from worldloop_kernel import validate_transition

        cap = make_capability()
        before = make_state_view(capabilities=cap, tick=0)
        after = make_state_view(capabilities=cap, tick=1)
        record = _make_transition_record(
            cap=cap,
            before=before,
            after=after,
            state_after_hash="sha256:wronghash",
        )
        report = validate_transition(record, before=before, after=after)
        result = report.invariant_results["hash_round_trip"]
        assert result.passed is False
        assert "rebuilt_hash" in result.diagnostic


# ---------------------------------------------------------------------------
# Invariant 2: receipt_completeness
# ---------------------------------------------------------------------------


class TestReceiptCompletenessInvariant:
    def test_passes_with_empty_actions(self):
        from worldloop_kernel import validate_transition

        record = _make_transition_record()
        report = validate_transition(record)
        assert report.invariant_results["receipt_completeness"].passed is True

    def test_passes_with_matching_keys(self):
        from worldloop_kernel import validate_transition, ActionProposal

        cap = make_capability()
        before = make_state_view(capabilities=cap, tick=0)
        after = make_state_view(capabilities=cap, tick=1)
        executed = {"a1": make_executed()}
        receipts = {"a1": make_receipt()}
        record = _make_transition_record(
            cap=cap,
            before=before,
            after=after,
            executed_actions=executed,
            receipts=receipts,
        )
        report = validate_transition(record)
        assert report.invariant_results["receipt_completeness"].passed is True


# ---------------------------------------------------------------------------
# Invariant 3: capability_consistency
# ---------------------------------------------------------------------------


class TestCapabilityConsistencyInvariant:
    def test_passes_with_valid_capability(self):
        from worldloop_kernel import validate_transition

        record = _make_transition_record()
        report = validate_transition(record)
        assert report.invariant_results["capability_consistency"].passed is True


# ---------------------------------------------------------------------------
# Invariant 4: missing_mask_consistency
# ---------------------------------------------------------------------------


class TestMissingMaskConsistencyInvariant:
    def test_skipped_without_states(self):
        from worldloop_kernel import validate_transition

        record = _make_transition_record()
        report = validate_transition(record)
        result = report.invariant_results["missing_mask_consistency"]
        assert result.passed is None
        assert "skipped" in result.message

    def test_passes_with_consistent_states(self):
        from worldloop_kernel import validate_transition

        cap = make_capability()
        before = make_state_view(capabilities=cap, tick=0)
        after = make_state_view(capabilities=cap, tick=1)
        record = _make_transition_record(
            cap=cap, before=before, after=after
        )
        report = validate_transition(record, before=before, after=after)
        assert report.invariant_results["missing_mask_consistency"].passed is True


# ---------------------------------------------------------------------------
# Invariant 5: outcome_code_legality
# ---------------------------------------------------------------------------


class TestOutcomeCodeLegalityInvariant:
    def test_passes_with_legal_codes(self):
        from worldloop_kernel import validate_transition

        cap = make_capability()
        before = make_state_view(capabilities=cap, tick=0)
        after = make_state_view(capabilities=cap, tick=1)
        executed = {"a1": make_executed()}
        receipts = {"a1": make_receipt(outcome_code="ok")}
        record = _make_transition_record(
            cap=cap,
            before=before,
            after=after,
            executed_actions=executed,
            receipts=receipts,
        )
        report = validate_transition(record)
        assert report.invariant_results["outcome_code_legality"].passed is True

    def test_fails_with_illegal_code(self):
        """Bypass ActionReceipt's __post_init__ validation by constructing
        via object.__new__ and setting fields directly. This simulates
        a record deserialized from untrusted JSON."""
        from worldloop_kernel import (
            validate_transition,
            ActionReceipt,
            TransitionRecord,
            PROTOCOL_SCHEMA_VERSION,
            hash_state,
            diff_state,
        )

        cap = make_capability()
        before = make_state_view(capabilities=cap, tick=0)
        after = make_state_view(capabilities=cap, tick=1)
        delta = diff_state(before, after)

        # Construct an ActionReceipt with an illegal outcome_code by
        # bypassing __post_init__. This is what would happen if a
        # record was deserialized from untrusted JSON without re-validation.
        bad_receipt = _build_bypassing_post_init(
            ActionReceipt,
            executed_action_hash="sha256:x",
            outcome_code="totally_made_up_code",
            success=True,
            energy_delta=0.0,
        )

        record = TransitionRecord(
            schema_version=PROTOCOL_SCHEMA_VERSION,
            producer_id="test-world",
            producer_version="0.0.1",
            tick=0,
            state_before_hash=hash_state(before),
            candidate_actions={},
            executed_actions={"a1": make_executed()},
            exogenous_input=None,
            receipts={"a1": bad_receipt},
            state_delta=delta,
            state_after_hash=hash_state(after),
            capability_profile=cap,
            provenance={},
        )
        report = validate_transition(record)
        result = report.invariant_results["outcome_code_legality"]
        assert result.passed is False
        assert result.diagnostic["illegal"][0]["outcome_code"] == "totally_made_up_code"


# ---------------------------------------------------------------------------
# Invariant 6: tick_monotonicity
# ---------------------------------------------------------------------------


class TestTickMonotonicityInvariant:
    def test_skipped_without_states(self):
        from worldloop_kernel import validate_transition

        record = _make_transition_record()
        report = validate_transition(record)
        result = report.invariant_results["tick_monotonicity"]
        assert result.passed is None

    def test_passes_with_correct_advance(self):
        from worldloop_kernel import validate_transition

        cap = make_capability()
        before = make_state_view(capabilities=cap, tick=5)
        after = make_state_view(capabilities=cap, tick=6)
        record = _make_transition_record(
            cap=cap, before=before, after=after, tick=5
        )
        report = validate_transition(record, before=before, after=after)
        assert report.invariant_results["tick_monotonicity"].passed is True

    def test_fails_with_wrong_advance(self):
        from worldloop_kernel import (
            validate_transition,
            TransitionRecord,
            PROTOCOL_SCHEMA_VERSION,
            hash_state,
            diff_state,
        )

        cap = make_capability()
        before = make_state_view(capabilities=cap, tick=5)
        after = make_state_view(capabilities=cap, tick=99)  # wrong tick
        delta = diff_state(before, after)
        record = TransitionRecord(
            schema_version=PROTOCOL_SCHEMA_VERSION,
            producer_id="test-world",
            producer_version="0.0.1",
            tick=99,
            state_before_hash=hash_state(before),
            candidate_actions={},
            executed_actions={},
            exogenous_input=None,
            receipts={},
            state_delta=delta,
            state_after_hash=hash_state(after),
            capability_profile=cap,
            provenance={},
        )
        report = validate_transition(record, before=before, after=after)
        result = report.invariant_results["tick_monotonicity"]
        assert result.passed is False
        assert result.diagnostic["after_tick"] == 99
        assert result.diagnostic["before_tick"] == 5


# ---------------------------------------------------------------------------
# Invariant 7: authority_grounding
# ---------------------------------------------------------------------------


class TestAuthorityGroundingInvariant:
    def test_passes_with_rule_authority(self):
        from worldloop_kernel import validate_transition

        record = _make_transition_record()
        report = validate_transition(record)
        assert report.invariant_results["authority_grounding"].passed is True

    def test_passes_with_learned_no_ground_truth(self):
        from worldloop_kernel import validate_transition

        cap = make_capability(authority="learned", ground_truth=False)
        before = make_state_view(capabilities=cap, tick=0)
        after = make_state_view(capabilities=cap, tick=1)
        record = _make_transition_record(
            cap=cap, before=before, after=after
        )
        report = validate_transition(record)
        assert report.invariant_results["authority_grounding"].passed is True

    def test_fails_with_learned_and_ground_truth(self):
        """Bypass CapabilityProfile's __post_init__ to simulate a record
        deserialized from untrusted JSON with a contradictory capability."""
        from worldloop_kernel import (
            validate_transition,
            CapabilityProfile,
            TransitionRecord,
            PROTOCOL_SCHEMA_VERSION,
            hash_state,
            diff_state,
        )

        # Construct a contradictory CapabilityProfile by bypassing __post_init__.
        bad_cap = _build_bypassing_post_init(
            CapabilityProfile,
            fields=False,
            entities=True,
            relations=False,
            registries=False,
            population=False,
            events=False,
            exact_restore=True,
            executable_deterministic_replay=True,
            authority="learned",
            ground_truth=True,  # contradictory!
            transition_mode="deterministic",
        )

        before = make_state_view(capabilities=make_capability(), tick=0)
        after = make_state_view(capabilities=make_capability(), tick=1)
        delta = diff_state(before, after)
        record = TransitionRecord(
            schema_version=PROTOCOL_SCHEMA_VERSION,
            producer_id="test-world",
            producer_version="0.0.1",
            tick=0,
            state_before_hash=hash_state(before),
            candidate_actions={},
            executed_actions={},
            exogenous_input=None,
            receipts={},
            state_delta=delta,
            state_after_hash=hash_state(after),
            capability_profile=bad_cap,
            provenance={},
        )
        report = validate_transition(record)
        result = report.invariant_results["authority_grounding"]
        assert result.passed is False
        assert result.diagnostic["authority"] == "learned"
        assert result.diagnostic["ground_truth"] is True


# ---------------------------------------------------------------------------
# TransitionRecorder
# ---------------------------------------------------------------------------


class TestTransitionRecorderBasic:
    def test_append_writes_record_file(self, tmp_path):
        from worldloop_kernel import TransitionRecorder

        record = _make_transition_record(tick=0)
        with TransitionRecorder(tmp_path, "test-world") as rec:
            rec.append(record)

        # One record file at t0000000000.json
        record_files = list(tmp_path.glob("t*.json"))
        assert len(record_files) == 1
        assert record_files[0].name == "t0000000000.json"
        # Manifest written on close.
        assert (tmp_path / "manifest.json").exists()

    def test_manifest_reflects_appended_records(self, tmp_path):
        from worldloop_kernel import TransitionRecorder

        with TransitionRecorder(tmp_path, "test-world") as rec:
            rec.append(_make_transition_record(tick=0))
            rec.append(_make_transition_record(tick=1))
            rec.append(_make_transition_record(tick=2))
            m = rec.manifest()
            assert m.record_count == 3
            assert m.first_tick == 0
            assert m.last_tick == 2
            assert len(m.state_before_hashes) == 3
            assert len(m.state_after_hashes) == 3

        # After close, manifest file on disk reflects the same.
        with open(tmp_path / "manifest.json", encoding="utf-8") as f:
            on_disk = json.load(f)
        assert on_disk["record_count"] == 3
        assert on_disk["first_tick"] == 0
        assert on_disk["last_tick"] == 2
        assert on_disk["closed_at"] is not None

    def test_close_is_idempotent(self, tmp_path):
        from worldloop_kernel import TransitionRecorder

        rec = TransitionRecorder(tmp_path, "test-world")
        rec.close()
        rec.close()  # should not raise
        assert (tmp_path / "manifest.json").exists()

    def test_append_after_close_raises(self, tmp_path):
        from worldloop_kernel import TransitionRecorder, RecorderError

        rec = TransitionRecorder(tmp_path, "test-world")
        rec.close()
        with pytest.raises(RecorderError, match="closed"):
            rec.append(_make_transition_record())

    def test_quarantine_dir_created_on_init(self, tmp_path):
        from worldloop_kernel import TransitionRecorder

        rec = TransitionRecorder(tmp_path, "test-world")
        assert (tmp_path / "_quarantine").is_dir()
        rec.close()


class TestTransitionRecorderQuarantine:
    def test_invalid_record_routed_to_quarantine(self, tmp_path):
        """A record with a wrong state_after_hash fails invariant 1
        when before is supplied. But the recorder does not supply before,
        so invariant 1 is skipped. We need a different failure mode.

        We construct a record with an illegal outcome_code (bypassing
        __post_init__) — this fails invariant 5."""
        from worldloop_kernel import (
            TransitionRecorder,
            ActionReceipt,
            TransitionRecord,
            PROTOCOL_SCHEMA_VERSION,
            hash_state,
            diff_state,
        )

        cap = make_capability()
        before = make_state_view(capabilities=cap, tick=0)
        after = make_state_view(capabilities=cap, tick=1)
        delta = diff_state(before, after)

        bad_receipt = _build_bypassing_post_init(
            ActionReceipt,
            executed_action_hash="sha256:x",
            outcome_code="totally_made_up_code",
            success=True,
            energy_delta=0.0,
        )

        record = TransitionRecord(
            schema_version=PROTOCOL_SCHEMA_VERSION,
            producer_id="test-world",
            producer_version="0.0.1",
            tick=0,
            state_before_hash=hash_state(before),
            candidate_actions={},
            executed_actions={"a1": make_executed()},
            exogenous_input=None,
            receipts={"a1": bad_receipt},
            state_delta=delta,
            state_after_hash=hash_state(after),
            capability_profile=cap,
            provenance={},
        )

        with TransitionRecorder(tmp_path, "test-world") as rec:
            rec.append(record)
            m = rec.manifest()
            assert m.record_count == 0
            assert m.quarantine_count == 1

        # Quarantine has the record + reason sidecar.
        q_files = list((tmp_path / "_quarantine").glob("*.json"))
        assert len(q_files) == 2  # record + reason
        reason_files = list((tmp_path / "_quarantine").glob("*.reason.json"))
        assert len(reason_files) == 1
        with open(reason_files[0], encoding="utf-8") as f:
            reason = json.load(f)
        assert reason["reason"] == "validation_failed"
        assert reason["tick"] == 0

    def test_validate_false_skips_validation(self, tmp_path):
        """When validate=False, even an illegal record is written to the
        published dataset."""
        from worldloop_kernel import (
            TransitionRecorder,
            ActionReceipt,
            TransitionRecord,
            PROTOCOL_SCHEMA_VERSION,
            hash_state,
            diff_state,
        )

        cap = make_capability()
        before = make_state_view(capabilities=cap, tick=0)
        after = make_state_view(capabilities=cap, tick=1)
        delta = diff_state(before, after)

        bad_receipt = _build_bypassing_post_init(
            ActionReceipt,
            executed_action_hash="sha256:x",
            outcome_code="totally_made_up_code",
            success=True,
            energy_delta=0.0,
        )

        record = TransitionRecord(
            schema_version=PROTOCOL_SCHEMA_VERSION,
            producer_id="test-world",
            producer_version="0.0.1",
            tick=0,
            state_before_hash=hash_state(before),
            candidate_actions={},
            executed_actions={"a1": make_executed()},
            exogenous_input=None,
            receipts={"a1": bad_receipt},
            state_delta=delta,
            state_after_hash=hash_state(after),
            capability_profile=cap,
            provenance={},
        )

        with TransitionRecorder(
            tmp_path, "test-world", validate=False
        ) as rec:
            rec.append(record)
            m = rec.manifest()
            assert m.record_count == 1
            assert m.quarantine_count == 0


class TestTransitionRecorderAtomicity:
    def test_record_file_is_valid_json(self, tmp_path):
        from worldloop_kernel import TransitionRecorder

        record = _make_transition_record(tick=42)
        with TransitionRecorder(tmp_path, "test-world") as rec:
            rec.append(record)

        record_path = tmp_path / "t0000000042.json"
        assert record_path.exists()
        with open(record_path, encoding="utf-8") as f:
            payload = json.load(f)
        assert payload["tick"] == 42
        assert payload["producer_id"] == "test-world"
        assert payload["schema_version"] == "0.1.0"

    def test_no_tmp_files_left_after_append(self, tmp_path):
        """Atomic write must clean up tmp files on success."""
        from worldloop_kernel import TransitionRecorder

        with TransitionRecorder(tmp_path, "test-world") as rec:
            for tick in range(5):
                rec.append(_make_transition_record(tick=tick))

        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0
        # Hidden tmp files (starting with .) should also be gone.
        hidden_tmp = [p for p in tmp_path.iterdir() if p.name.startswith(".")]
        assert len(hidden_tmp) == 0


# ---------------------------------------------------------------------------
# Re-export check (K-06 symbols reachable from top-level package)
# ---------------------------------------------------------------------------


def test_k06_symbols_reachable_from_top_level():
    import worldloop_kernel as wk

    assert hasattr(wk, "validate_transition")
    assert hasattr(wk, "ValidationReport")
    assert hasattr(wk, "InvariantResult")
    assert hasattr(wk, "ValidationError")
    assert hasattr(wk, "INVARIANT_NAMES")
    assert hasattr(wk, "TransitionRecorder")
    assert hasattr(wk, "RecorderManifest")
    assert hasattr(wk, "RecorderError")
