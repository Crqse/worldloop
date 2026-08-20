"""Transition validation (K-06).

Validates a :class:`TransitionRecord` against 7 invariants. Failed
validations go to quarantine and MUST NOT enter the published dataset.

7 invariants (per main plan §4.6 and v1 ``core/runtime/
transition_validation.py`` reference):

1. **Hash round-trip**: ``hash_state(apply_delta(before, delta)) ==
   state_after_hash``. Requires ``before`` to be supplied.
2. **Receipt completeness**: every executed_action has a receipt; every
   candidate either became an executed_action or has an explicit
   rejection receipt. Verifiable from the record alone.
3. **Capability consistency**: ``capability_profile`` matches what the
   StateView actually contains. Verifiable from the record alone
   (capability vs. slot presence is a static check).
4. **Missing-mask consistency**: ``missing_mask`` is True only for
   slots where capability is True but the value is None. Requires
   ``before`` / ``after`` to be supplied.
5. **Outcome-code legality**: receipts use only the stable outcome-code
   enum. Verifiable from the record alone.
6. **Tick monotonicity**: ``state_after.tick == record.tick`` and (when
   ``before`` is supplied) ``record.tick == before.tick + 1`` (or 0
   for reset).
7. **Authority grounding**: if ``capability_profile.authority ==
   "learned"``, then ``capability_profile.ground_truth == False``.
   Verifiable from the record alone (already enforced at
   :class:`CapabilityProfile` construction; this invariant is the
   belt-and-suspenders check at the record level).

Provenance: extracted from ``current/worldloop/core/runtime/
transition_validation.py`` 7-invariant validator (v1.0.0 tag); the v1
invariants are treated as a reference. K-06 re-implements a minimal
subset for v2 protocol and may add / drop invariants with explicit ADR.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from worldloop_kernel.action import KERNEL_OUTCOME_CODES
from worldloop_kernel.canonical import hash_state
from worldloop_kernel.diff_apply import apply_delta
from worldloop_kernel.transition import TransitionRecord, TransitionError
from worldloop_kernel.state import StateView

__all__ = [
    "InvariantResult",
    "ValidationReport",
    "ValidationError",
    "validate_transition",
    "INVARIANT_NAMES",
]


class ValidationError(ValueError):
    """Raised when validation cannot proceed (e.g., malformed record)."""


#: Canonical names of the 7 invariants, in stable order. Used as keys
#: in ``ValidationReport.invariant_results`` so consumers can branch
#: without parsing human-readable strings.
INVARIANT_NAMES: tuple[str, ...] = (
    "hash_round_trip",
    "receipt_completeness",
    "capability_consistency",
    "missing_mask_consistency",
    "outcome_code_legality",
    "tick_monotonicity",
    "authority_grounding",
)


@dataclass(frozen=True)
class InvariantResult:
    """Result of checking one invariant.

    Attributes
    ----------
    name:
        Canonical invariant name (see :data:`INVARIANT_NAMES`).
    passed:
        True iff the invariant held. False if it was violated.
        ``None`` if the invariant was skipped (e.g., required context
        not supplied). Consumers should treat ``None`` as "not run"
        and not as pass or fail.
    message:
        Human-readable description of the outcome. Empty string if
        ``passed`` is True.
    diagnostic:
        Optional structured diagnostic (e.g., offending key, expected
        vs. actual hash). ``None`` if not applicable.
    """

    name: str
    passed: bool | None
    message: str = ""
    diagnostic: Any = None


@dataclass(frozen=True)
class ValidationReport:
    """Aggregate report for one :class:`TransitionRecord`.

    Attributes
    ----------
    record_id:
        Stable identifier for the record. Currently derived from
        ``producer_id + tick + state_before_hash``. Consumers should
        treat this as opaque.
    passed:
        True iff every invariant that ran (``passed is not None``)
        returned True. Invariants that were skipped (``passed is None``)
        do not affect this field.
    invariant_results:
        Mapping from invariant name to :class:`InvariantResult`.
        Always contains all 7 names; skipped invariants have
        ``passed=None``.
    diagnostics:
        Free-form mapping for additional context (e.g., which invariants
        were skipped and why).
    """

    record_id: str
    passed: bool
    invariant_results: Mapping[str, InvariantResult]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Individual invariant checks
# ---------------------------------------------------------------------------


def _check_hash_round_trip(
    record: TransitionRecord,
    before: StateView | None,
) -> InvariantResult:
    """Invariant 1: hash_state(apply_delta(before, delta)) == state_after_hash."""
    name = "hash_round_trip"
    if before is None:
        return InvariantResult(
            name=name,
            passed=None,
            message="skipped: `before` state not supplied",
            diagnostic={"required": "before"},
        )
    try:
        rebuilt = apply_delta(before, record.state_delta)
        rebuilt_hash = hash_state(rebuilt)
    except Exception as exc:  # noqa: BLE001 — invariant check must not crash
        return InvariantResult(
            name=name,
            passed=False,
            message=f"apply_delta raised: {exc!r}",
            diagnostic={"exception_type": type(exc).__name__},
        )
    if rebuilt_hash != record.state_after_hash:
        return InvariantResult(
            name=name,
            passed=False,
            message=(
                "hash_state(apply_delta(before, delta)) != state_after_hash"
            ),
            diagnostic={
                "rebuilt_hash": rebuilt_hash,
                "record_state_after_hash": record.state_after_hash,
            },
        )
    return InvariantResult(name=name, passed=True)


def _check_receipt_completeness(record: TransitionRecord) -> InvariantResult:
    """Invariant 2: every executed_action has a receipt; every candidate
    either became an executed_action or has an explicit rejection receipt.

    The kernel's TransitionRecord schema does NOT carry rejection receipts
    for non-executed candidates (only executed agents have receipts).
    Therefore this invariant reduces to: ``executed_actions.keys() ==
    receipts.keys()``. The candidate-vs-executed gap is a world-side
    concern and is NOT enforced here.

    Note: this invariant is also enforced at TransitionRecord construction
    (``__post_init__``). This check is the belt-and-suspenders runtime
    counterpart.
    """
    name = "receipt_completeness"
    exec_keys = set(record.executed_actions.keys())
    receipt_keys = set(record.receipts.keys())
    if exec_keys != receipt_keys:
        return InvariantResult(
            name=name,
            passed=False,
            message="executed_actions keys != receipts keys",
            diagnostic={
                "executed_keys": sorted(str(k) for k in exec_keys),
                "receipt_keys": sorted(str(k) for k in receipt_keys),
                "missing_receipts": sorted(
                    str(k) for k in (exec_keys - receipt_keys)
                ),
                "extra_receipts": sorted(
                    str(k) for k in (receipt_keys - exec_keys)
                ),
            },
        )
    return InvariantResult(name=name, passed=True)


def _check_capability_consistency(record: TransitionRecord) -> InvariantResult:
    """Invariant 3: capability_profile matches what the StateView contains.

    Since the record does NOT carry the full StateView (only hashes and
    delta), this invariant at the record level can only verify that the
    capability_profile is internally consistent (already enforced at
    construction). The full state-vs-capability check happens at
    StateView construction (K-04 ``__post_init__``) and is not duplicated
    here.

    This invariant therefore always passes at the record level, but it
    is kept in the list to:
    - document the invariant,
    - provide a hook for future record-level capability drift checks,
    - preserve the 7-invariant count for downstream tooling.
    """
    name = "capability_consistency"
    cap = record.capability_profile
    # Re-run the CapabilityProfile post_init rules defensively. They
    # should never fail at this point (already validated at construction),
    # but if a record was deserialized from untrusted JSON, this catches
    # drift.
    try:
        cap.__post_init__()  # type: ignore[misc]
    except Exception as exc:  # noqa: BLE001
        return InvariantResult(
            name=name,
            passed=False,
            message=f"capability_profile failed re-validation: {exc!r}",
            diagnostic={"exception_type": type(exc).__name__},
        )
    return InvariantResult(name=name, passed=True)


def _check_missing_mask_consistency(
    record: TransitionRecord,
    before: StateView | None,
    after: StateView | None,
) -> InvariantResult:
    """Invariant 4: missing_mask True only for slots where capability=True
    but value is None.

    StateView construction already enforces this (K-04 ``__post_init__``).
    This invariant at the record level re-validates the supplied states
    if any; otherwise it is skipped.
    """
    name = "missing_mask_consistency"
    if before is None and after is None:
        return InvariantResult(
            name=name,
            passed=None,
            message="skipped: no state views supplied",
            diagnostic={"required": "before or after"},
        )
    # StateView construction already enforces missing_mask rules. If a
    # state view object exists and was constructed via the public API,
    # the invariant holds by construction. Re-validating is wasteful.
    # We only flag if a state view was supplied and is somehow
    # inconsistent (which would indicate the StateView was bypassed).
    for label, sv in (("before", before), ("after", after)):
        if sv is None:
            continue
        cap = sv.capabilities
        flags = cap.slot_flags()
        for slot_name, has_cap in flags.items():
            slot_value = getattr(sv, slot_name)
            missing = sv.missing_mask.get(slot_name, False)
            if has_cap and not missing and slot_value is None:
                return InvariantResult(
                    name=name,
                    passed=False,
                    message=(
                        f"{label} state: capability.{slot_name}=True and "
                        f"missing_mask[{slot_name!r}]=False but slot is None"
                    ),
                    diagnostic={"state": label, "slot": slot_name},
                )
            if has_cap and missing and slot_value is not None:
                return InvariantResult(
                    name=name,
                    passed=False,
                    message=(
                        f"{label} state: capability.{slot_name}=True and "
                        f"missing_mask[{slot_name!r}]=True but slot is not None"
                    ),
                    diagnostic={"state": label, "slot": slot_name},
                )
    return InvariantResult(name=name, passed=True)


def _check_outcome_code_legality(record: TransitionRecord) -> InvariantResult:
    """Invariant 5: receipts use only the stable outcome-code enum."""
    name = "outcome_code_legality"
    illegal: list[dict[str, Any]] = []
    for agent_id, receipt in record.receipts.items():
        if receipt.outcome_code not in KERNEL_OUTCOME_CODES:
            illegal.append(
                {"agent_id": str(agent_id), "outcome_code": receipt.outcome_code}
            )
    if illegal:
        return InvariantResult(
            name=name,
            passed=False,
            message=f"{len(illegal)} receipt(s) use illegal outcome_code",
            diagnostic={"illegal": illegal, "legal_codes": list(KERNEL_OUTCOME_CODES)},
        )
    return InvariantResult(name=name, passed=True)


def _check_tick_monotonicity(
    record: TransitionRecord,
    before: StateView | None,
    after: StateView | None,
) -> InvariantResult:
    """Invariant 6: tick advances by exactly 1 between before and after.

    Per :class:`TransitionRecord` docstring, ``record.tick`` is the ``t``
    in ``S_t -> S_{t+1}`` — i.e., the tick at which the transition
    occurs, equal to ``before.meta.tick``. After the transition,
    ``after.meta.tick == record.tick + 1`` (or 0 for a reset transition
    where ``before is None``).

    Checks (in order):
    1. If both ``before`` and ``after`` are None → skipped.
    2. If ``before`` supplied: ``before.meta.tick == record.tick``.
    3. If ``after`` supplied:
       - If ``before`` also supplied: ``after.meta.tick == before.tick + 1``.
       - If ``before`` is None (reset): ``after.meta.tick == record.tick``
         (record.tick for a reset is typically 0).
    """
    name = "tick_monotonicity"
    if before is None and after is None:
        return InvariantResult(
            name=name,
            passed=None,
            message="skipped: no state views supplied",
            diagnostic={"required": "before or after"},
        )

    after_tick = after.meta.tick if after is not None else None
    before_tick = before.meta.tick if before is not None else None

    # Check 2: before.tick == record.tick (when before supplied)
    if before is not None and before.meta.tick != record.tick:
        return InvariantResult(
            name=name,
            passed=False,
            message=(
                f"before.meta.tick ({before.meta.tick}) != record.tick "
                f"({record.tick}); record.tick must equal before.tick per "
                f"the S_t -> S_{{t+1}} convention"
            ),
            diagnostic={
                "before_tick": before_tick,
                "after_tick": after_tick,
                "record_tick": record.tick,
            },
        )

    # Check 3: after.tick must advance by 1 over record.tick
    # (or before.tick, which equals record.tick at this point).
    if after is not None:
        if before is not None:
            # Normal transition: after.tick == before.tick + 1
            expected_after = before.meta.tick + 1
            if after.meta.tick != expected_after:
                return InvariantResult(
                    name=name,
                    passed=False,
                    message=(
                        f"after.meta.tick ({after.meta.tick}) != before.tick+1 "
                        f"({expected_after})"
                    ),
                    diagnostic={
                        "before_tick": before_tick,
                        "after_tick": after_tick,
                        "expected_after_tick": expected_after,
                        "record_tick": record.tick,
                    },
                )
        else:
            # Reset transition (before=None): after.tick == record.tick.
            # record.tick for a reset is typically 0; we do not enforce
            # that specifically, only the equality.
            if after.meta.tick != record.tick:
                return InvariantResult(
                    name=name,
                    passed=False,
                    message=(
                        f"after.meta.tick ({after.meta.tick}) != record.tick "
                        f"({record.tick}) for a reset transition (before=None)"
                    ),
                    diagnostic={
                        "after_tick": after_tick,
                        "record_tick": record.tick,
                    },
                )

    return InvariantResult(name=name, passed=True)


def _check_authority_grounding(record: TransitionRecord) -> InvariantResult:
    """Invariant 7: learned authority requires ground_truth=False.

    Already enforced at CapabilityProfile construction. Re-checked here
    as belt-and-suspenders for records deserialized from untrusted JSON.
    """
    name = "authority_grounding"
    cap = record.capability_profile
    if cap.authority == "learned" and cap.ground_truth:
        return InvariantResult(
            name=name,
            passed=False,
            message=(
                "capability_profile.authority='learned' but ground_truth=True; "
                "learned authority MUST NOT claim ground truth"
            ),
            diagnostic={
                "authority": cap.authority,
                "ground_truth": cap.ground_truth,
            },
        )
    return InvariantResult(name=name, passed=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_transition(
    record: TransitionRecord,
    *,
    before: StateView | None = None,
    after: StateView | None = None,
) -> ValidationReport:
    """Validate a :class:`TransitionRecord` against the 7 kernel invariants.

    Args:
        record: The transition record to validate.
        before: Optional :class:`StateView` representing the state
            BEFORE the transition. Enables invariants 1, 4, 6.
        after: Optional :class:`StateView` representing the state
            AFTER the transition. Enables invariants 4, 6.

    Returns:
        A :class:`ValidationReport` with one :class:`InvariantResult`
        per invariant (7 total). Skipped invariants have ``passed=None``
        and do not affect the overall ``ValidationReport.passed`` flag.

    Raises:
        ValidationError: If the record is so malformed that validation
            cannot proceed (e.g., missing required fields). Note that
            most malformed records are rejected at construction time
            (``TransitionRecord.__post_init__``); this is a defensive
            counterpart for records deserialized from untrusted JSON.
    """
    # Defensive: confirm the record is a TransitionRecord. If a raw dict
    # was passed, the caller should deserialize first.
    if not isinstance(record, TransitionRecord):
        raise ValidationError(
            f"record must be a TransitionRecord, got {type(record).__name__}"
        )

    # Build a stable record_id for downstream deduplication / lookup.
    record_id = (
        f"{record.producer_id}:t{record.tick}:"
        f"{record.state_before_hash[:16]}"
    )

    # Run all 7 invariants. Order matters for the report — consumers
    # may short-circuit on the first failure.
    results: dict[str, InvariantResult] = {
        "hash_round_trip": _check_hash_round_trip(record, before),
        "receipt_completeness": _check_receipt_completeness(record),
        "capability_consistency": _check_capability_consistency(record),
        "missing_mask_consistency": _check_missing_mask_consistency(
            record, before, after
        ),
        "outcome_code_legality": _check_outcome_code_legality(record),
        "tick_monotonicity": _check_tick_monotonicity(record, before, after),
        "authority_grounding": _check_authority_grounding(record),
    }

    # Overall pass = every run invariant passed. Skipped (None) does
    # not affect the result.
    ran = [r for r in results.values() if r.passed is not None]
    overall_passed = all(r.passed for r in ran) if ran else False

    skipped = [name for name, r in results.items() if r.passed is None]
    diagnostics: dict[str, Any] = {}
    if skipped:
        diagnostics["skipped"] = skipped
    diagnostics["ran_count"] = len(ran)
    diagnostics["skipped_count"] = len(skipped)

    return ValidationReport(
        record_id=record_id,
        passed=overall_passed,
        invariant_results=results,
        diagnostics=diagnostics,
    )
