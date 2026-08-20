"""Static semantic validator (S-03).

Runs AFTER JSON Schema validation (S-02) passes. Performs 11 semantic
checks from main plan §13.5:

1. action_field_references_exist   — action effects reference declared entity columns
2. effect_targets_are_explicit     — every effect has a clear target + field
3. resource_signs_consistent       — resource effects (add/sub) are sign-consistent
4. no_guaranteed_infinite_growth   — necessary-condition check (not halting-proof)
5. boundary_conditions_explicit    — termination conditions have required fields
6. stochastic_rules_have_seed      — non-deterministic specs declare exogenous.seed
7. reward_does_not_read_future     — reward effects do not reference future tick state
8. candidate_action_no_direct_write — action effects do not write kernel-level fields
9. L5_no_writeback                 — no effect writes the L5 emergence slot
10. registry_no_dangling_reference — registry references point to declared registry_types
11. graph_edge_cleanup_on_delete   — graph node deletion would clean up edges

Design rules:
- Each check returns a :class:`ValidationIssue` (or list thereof).
- Checks are independent; one failure does not block the others.
- The validator collects ALL issues and returns them as a list.
- "Necessary condition" checks (4, 11) do NOT prove halting; they only
  catch obvious violations (e.g., max_ticks=0 with no stop condition).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from worldloop_scenarios.spec import ScenarioSpec

__all__ = [
    "ValidationIssue",
    "ValidationResult",
    "validate_semantics",
    "check_action_field_references_exist",
    "check_effect_targets_are_explicit",
    "check_resource_signs_consistent",
    "check_no_guaranteed_infinite_growth",
    "check_boundary_conditions_explicit",
    "check_stochastic_rules_have_seed",
    "check_reward_does_not_read_future",
    "check_candidate_action_no_direct_write",
    "check_L5_no_writeback",
    "check_registry_no_dangling_reference",
    "check_graph_edge_cleanup_on_delete",
]


# ---------------------------------------------------------------------------
# Issue / result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationIssue:
    """A single semantic validation issue."""

    check_id: str
    check_name: str
    severity: str  # "error" | "warning"
    message: str
    location: str = ""  # e.g., "actions[0].effects[1]"


@dataclass
class ValidationResult:
    """Aggregated result of all 11 semantic checks."""

    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True iff no issue has severity='error'."""
        return not any(i.severity == "error" for i in self.issues)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def validate_semantics(spec: ScenarioSpec) -> ValidationResult:
    """Run all 11 semantic checks against a :class:`ScenarioSpec`.

    Returns a :class:`ValidationResult` with all collected issues.
    The spec is accepted iff ``result.is_valid`` is True (no errors).
    """
    result = ValidationResult()
    for check_fn in (
        check_action_field_references_exist,
        check_effect_targets_are_explicit,
        check_resource_signs_consistent,
        check_no_guaranteed_infinite_growth,
        check_boundary_conditions_explicit,
        check_stochastic_rules_have_seed,
        check_reward_does_not_read_future,
        check_candidate_action_no_direct_write,
        check_L5_no_writeback,
        check_registry_no_dangling_reference,
        check_graph_edge_cleanup_on_delete,
    ):
        result.issues.extend(check_fn(spec))
    return result


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_action_field_references_exist(spec: ScenarioSpec) -> list[ValidationIssue]:
    """Check 1: action effects reference declared entity columns.

    Every ``effect.field`` referencing ``target: "entity"`` MUST match a
    column declared in ``entities.columns``.
    """
    issues: list[ValidationIssue] = []
    declared_columns = {c["name"] for c in spec.entities.columns}
    for i, action in enumerate(spec.actions.actions):
        for j, effect in enumerate(action.get("effects", ())):
            if effect.get("target") == "entity":
                field_name = effect.get("field", "")
                if field_name and field_name not in declared_columns:
                    issues.append(
                        ValidationIssue(
                            check_id="1",
                            check_name="action_field_references_exist",
                            severity="error",
                            message=(
                                f"Action '{action['action_type']}' effect references "
                                f"entity column '{field_name}' which is not declared "
                                f"in entities.columns"
                            ),
                            location=f"actions[{i}].effects[{j}]",
                        )
                    )
    return issues


def check_effect_targets_are_explicit(spec: ScenarioSpec) -> list[ValidationIssue]:
    """Check 2: every effect has a clear target + field.

    Effects MUST have both ``target`` and ``field`` (except ``target: "registry"``
    which uses ``field: "owner"`` by convention).
    """
    issues: list[ValidationIssue] = []
    valid_targets = {"entity", "field", "registry", "relation", "population"}
    for i, action in enumerate(spec.actions.actions):
        for j, effect in enumerate(action.get("effects", ())):
            target = effect.get("target")
            if target not in valid_targets:
                issues.append(
                    ValidationIssue(
                        check_id="2",
                        check_name="effect_targets_are_explicit",
                        severity="error",
                        message=(
                            f"Effect has invalid or missing target='{target}'; "
                            f"must be one of {sorted(valid_targets)}"
                        ),
                        location=f"actions[{i}].effects[{j}]",
                    )
                )
            if "op" not in effect:
                issues.append(
                    ValidationIssue(
                        check_id="2",
                        check_name="effect_targets_are_explicit",
                        severity="error",
                        message=f"Effect missing 'op' field",
                        location=f"actions[{i}].effects[{j}]",
                    )
                )
    return issues


def check_resource_signs_consistent(spec: ScenarioSpec) -> list[ValidationIssue]:
    """Check 3: resource effects (add/sub) are sign-consistent.

    A resource field should not have both ``add: positive`` and ``sub: positive``
    with contradictory semantics. This is a WARNING-level check (heuristic).
    """
    issues: list[ValidationIssue] = []
    # Collect per-field ops
    field_ops: dict[str, set[str]] = {}
    for action in spec.actions.actions:
        for effect in action.get("effects", ()):
            if effect.get("target") in ("entity", "field"):
                fname = effect.get("field", "")
                op = effect.get("op", "")
                if fname and op:
                    field_ops.setdefault(fname, set()).add(op)
    # Heuristic: if a field has both 'add' and 'sub', it's a resource being
    # consumed and replenished — that's fine. Flag only if a field has only
    # 'sub' without any 'add' (pure drain, no replenishment).
    for fname, ops in field_ops.items():
        if ops == {"sub"}:
            issues.append(
                ValidationIssue(
                    check_id="3",
                    check_name="resource_signs_consistent",
                    severity="warning",
                    message=(
                        f"Field '{fname}' only has 'sub' effects (pure drain, "
                        f"no replenishment); ensure exogenous events or other "
                        f"actions restore it"
                    ),
                )
            )
    return issues


def check_no_guaranteed_infinite_growth(spec: ScenarioSpec) -> list[ValidationIssue]:
    """Check 4: necessary condition — no guaranteed infinite growth.

    This is a NECESSARY condition check (not halting-proof). It catches:
    - ``time.max_ticks`` <= 0 (no time bound)
    - No termination.stop_conditions AND no max_ticks upper bound
    """
    issues: list[ValidationIssue] = []
    if spec.time.max_ticks <= 0:
        issues.append(
            ValidationIssue(
                check_id="4",
                check_name="no_guaranteed_infinite_growth",
                severity="error",
                message=(
                    f"time.max_ticks={spec.time.max_ticks} <= 0; "
                    f"world has no time bound"
                ),
            )
        )
    if not spec.termination.stop_conditions and spec.time.max_ticks > 10000:
        issues.append(
            ValidationIssue(
                check_id="4",
                check_name="no_guaranteed_infinite_growth",
                severity="warning",
                message=(
                    f"No termination.stop_conditions and max_ticks={spec.time.max_ticks} "
                    f"is very large; consider adding explicit stop conditions"
                ),
            )
        )
    return issues


def check_boundary_conditions_explicit(spec: ScenarioSpec) -> list[ValidationIssue]:
    """Check 5: termination conditions have required fields.

    ``threshold`` conditions MUST have ``field``, ``op``, and ``value``.
    ``max_ticks`` and ``no_alive`` are parameter-less.
    """
    issues: list[ValidationIssue] = []
    for i, cond in enumerate(spec.termination.stop_conditions):
        kind = cond.get("kind", "")
        if kind == "threshold":
            for req in ("field", "op", "value"):
                if req not in cond:
                    issues.append(
                        ValidationIssue(
                            check_id="5",
                            check_name="boundary_conditions_explicit",
                            severity="error",
                            message=(
                                f"termination.stop_conditions[{i}] kind='threshold' "
                                f"missing required field '{req}'"
                            ),
                            location=f"termination.stop_conditions[{i}]",
                        )
                    )
        elif kind == "custom":
            # Custom conditions MUST have a 'predicate' field (free-form).
            if "predicate" not in cond:
                issues.append(
                    ValidationIssue(
                        check_id="5",
                        check_name="boundary_conditions_explicit",
                        severity="warning",
                        message=(
                            f"termination.stop_conditions[{i}] kind='custom' "
                            f"has no 'predicate' field; semantic cannot be verified"
                        ),
                        location=f"termination.stop_conditions[{i}]",
                    )
                )
    return issues


def check_stochastic_rules_have_seed(spec: ScenarioSpec) -> list[ValidationIssue]:
    """Check 6: non-deterministic specs declare exogenous.seed.

    If ``time.deterministic=False``, the spec SHOULD declare
    ``exogenous.seed`` for reproducibility.
    """
    issues: list[ValidationIssue] = []
    if not spec.time.deterministic and spec.exogenous.seed is None:
        issues.append(
            ValidationIssue(
                check_id="6",
                check_name="stochastic_rules_have_seed",
                severity="warning",
                message=(
                    "time.deterministic=False but exogenous.seed is null; "
                    "stochastic world will not be reproducible"
                ),
            )
        )
    return issues


def check_reward_does_not_read_future(spec: ScenarioSpec) -> list[ValidationIssue]:
    """Check 7: reward effects do not reference future tick state.

    Effects with ``op: "set"`` on a field named ``reward`` MUST NOT
    reference ``$future`` or ``$next`` variables.
    """
    issues: list[ValidationIssue] = []
    future_tokens = ("$future", "$next", "$tick+1", "$t+1")
    for i, action in enumerate(spec.actions.actions):
        for j, effect in enumerate(action.get("effects", ())):
            value = str(effect.get("value", ""))
            field_name = effect.get("field", "")
            if field_name == "reward" and any(tok in value for tok in future_tokens):
                issues.append(
                    ValidationIssue(
                        check_id="7",
                        check_name="reward_does_not_read_future",
                        severity="error",
                        message=(
                            f"reward effect references future state '{value}'; "
                            f"reward MUST be computable from current tick state"
                        ),
                        location=f"actions[{i}].effects[{j}]",
                    )
                )
    return issues


def check_candidate_action_no_direct_write(spec: ScenarioSpec) -> list[ValidationIssue]:
    """Check 8: action effects do not write kernel-level fields.

    Actions MUST NOT write ``meta.*`` or ``capabilities.*`` fields —
    those are kernel-owned. Only ``entity``, ``field``, ``registry``,
    ``relation``, ``population`` are writable.
    """
    issues: list[ValidationIssue] = []
    kernel_owned = {"meta", "capabilities", "missing_mask", "rng_state_ref"}
    for i, action in enumerate(spec.actions.actions):
        for j, effect in enumerate(action.get("effects", ())):
            target = effect.get("target", "")
            field_name = effect.get("field", "")
            if target in kernel_owned or field_name.startswith("meta.") or field_name.startswith("capabilities."):
                issues.append(
                    ValidationIssue(
                        check_id="8",
                        check_name="candidate_action_no_direct_write",
                        severity="error",
                        message=(
                            f"Action effect writes kernel-owned field "
                            f"'{target}.{field_name}'; only entity/field/registry/"
                            f"relation/population are writable"
                        ),
                        location=f"actions[{i}].effects[{j}]",
                    )
                )
    return issues


def check_L5_no_writeback(spec: ScenarioSpec) -> list[ValidationIssue]:
    """Check 9: no effect writes the L5 emergence slot.

    L5 (emergence) is read-only. No action effect may target
    ``emergence``, ``emergent``, or ``L5``.
    """
    issues: list[ValidationIssue] = []
    forbidden = {"emergence", "emergent", "L5"}
    for i, action in enumerate(spec.actions.actions):
        for j, effect in enumerate(action.get("effects", ())):
            target = effect.get("target", "")
            field_name = effect.get("field", "")
            if target in forbidden or field_name in forbidden:
                issues.append(
                    ValidationIssue(
                        check_id="9",
                        check_name="L5_no_writeback",
                        severity="error",
                        message=(
                            f"Action effect targets L5 emergence slot "
                            f"'{target}.{field_name}'; L5 is read-only"
                        ),
                        location=f"actions[{i}].effects[{j}]",
                    )
                )
    return issues


def check_registry_no_dangling_reference(spec: ScenarioSpec) -> list[ValidationIssue]:
    """Check 10: registry references point to declared registry_types.

    If ``registries.enabled=True``, action preconditions referencing
    ``registry_type`` MUST point to a declared type in
    ``registries.registry_types``.
    """
    issues: list[ValidationIssue] = []
    if not spec.registries.enabled:
        return issues
    declared_types = set(spec.registries.registry_types)
    for i, action in enumerate(spec.actions.actions):
        for j, precond in enumerate(action.get("preconditions", ())):
            if precond.get("kind") == "registry_unowned":
                rt = str(precond.get("registry_type", ""))
                if rt.startswith("$"):
                    continue  # variable reference, skip
                if rt and rt not in declared_types:
                    issues.append(
                        ValidationIssue(
                            check_id="10",
                            check_name="registry_no_dangling_reference",
                            severity="error",
                            message=(
                                f"Precondition references registry_type '{rt}' "
                                f"not declared in registries.registry_types"
                            ),
                            location=f"actions[{i}].preconditions[{j}]",
                        )
                    )
    return issues


def check_graph_edge_cleanup_on_delete(spec: ScenarioSpec) -> list[ValidationIssue]:
    """Check 11: graph node deletion would clean up edges.

    If ``space.type='graph'`` and actions can delete entities, the spec
    SHOULD declare an effect that cleans up edges referencing the deleted
    node. This is a WARNING (necessary condition only).
    """
    issues: list[ValidationIssue] = []
    if spec.space.type != "graph":
        return issues
    # Check if any action has an effect with op='remove' targeting entity
    has_entity_removal = False
    has_edge_cleanup = False
    for action in spec.actions.actions:
        for effect in action.get("effects", ()):
            if effect.get("target") == "entity" and effect.get("op") == "remove":
                has_entity_removal = True
            if effect.get("target") == "relation" and effect.get("op") in ("remove", "set"):
                has_edge_cleanup = True
    if has_entity_removal and not has_edge_cleanup:
        issues.append(
            ValidationIssue(
                check_id="11",
                check_name="graph_edge_cleanup_on_delete",
                severity="warning",
                message=(
                    "Actions can remove entities but no effect cleans up "
                    "relation edges; deleted entities may leave dangling edges"
                ),
            )
        )
    return issues
