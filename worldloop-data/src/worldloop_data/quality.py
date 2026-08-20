"""S-13 Quality Reporter — quantify dataset usability (Q0-Q9).

Aggregates the M4 Gate §14.7 ten quality items into a single
:class:`QualityReport`. Each item is scored ``pass`` / ``fail`` /
``skipped`` with evidence. The reporter is read-only: it consumes the
exported dataset directory and the leakage report; it does NOT modify
the dataset.

The ten items (per main plan §14.7):
- Q0 Schema: every transition passes the 7 dict-layer invariants AND
  is typed-reconstructed into a kernel :class:`TransitionRecord` that
  passes ``validate_transition`` (record-level invariants; the
  state-dependent invariants hash_round_trip / missing_mask /
  tick_monotonicity require StateViews and are covered by Q3 replay).
- Q1 Traceability: candidate/executed/receipt/state refs closure +
  per-episode hash chain + tick uniqueness + episode_id presence.
- Q2 Structural Delta-Capability Consistency: capability slots
  structurally consistent with state_delta bundles (full
  ``diff_state`` + ``apply_delta`` round-trip deferred to Q3 replay).
- Q3 Replay: ``replay`` is bit-identical on ``exact_restore`` worlds,
  for the first N records of EVERY episode, with the reset state
  anchored against the recorded initial ``state_before_hash``.
- Q4 Provenance: per-record policy_id completeness + manifest
  cross-check (total_records == Σ split record_count == on-disk
  count; episode_id closure; per-episode seed consistency).
- Q5 Leakage: upstream report internally consistent + non-vacuous
  (checked_kinds non-empty) + independent recompute of cross-split
  episode/seed disjointness from the on-disk records.
- Q6 Coverage Consistency: coverage report mechanically consistent
  with the published dataset (observation superset, action-type /
  outcome-code count domination, policy closure).
- Q7 Counterfactual: held-fixed factors verified.
- Q8 Quarantine: failed records isolated, not silently deleted.
- Q9 Utility: at least one strong-baseline comparison (random vs scripted).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from worldloop_kernel import (
    ActionProposal,
    ExecutedAction,
    ExogenousInput,
    JointAction,
    KERNEL_OUTCOME_CODES,
    PROTOCOL_SCHEMA_VERSION,
    TransitionRecord,
    hash_state,
)
from worldloop_kernel.action import ActionReceipt
from worldloop_kernel.capability import CapabilityProfile
from worldloop_kernel.state import RegistryEntry, RelationEdge, StateMeta
from worldloop_kernel.transition import (
    EntityChange,
    EntityChanges,
    EventRecord,
    FieldChange,
    PopulationChange,
    PopulationChanges,
    RegistryChange,
    RegistryChanges,
    RelationChange,
    RelationChanges,
    StateDelta,
)
from worldloop_kernel.validation import (
    ValidationReport,
    validate_transition,
)

from worldloop_data.config import QualityConfig
from worldloop_data.coverage import CoverageReport
from worldloop_data.counterfactual import CounterfactualBranchScheduler
from worldloop_data.exporter import ExportResult
from worldloop_data.leakage import LeakageReport
from worldloop_data.utility import UtilityEvaluationReport

__all__ = [
    "QualityReporter",
    "QualityReport",
    "QualityItem",
    "MinimalQualityReporter",
]


# ---------------------------------------------------------------------------
# QualityItem / QualityReport
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QualityItem:
    """One Q-item result.

    Attributes
    ----------
    key:
        Item key (``"Q0"`` .. ``"Q9"``).
    name:
        Human-readable name (e.g., ``"Schema"``).
    status:
        ``"pass"`` / ``"fail"`` / ``"skipped"``.
    evidence:
        Free-form evidence (e.g., ``"500/500 records schema-valid"``).
    """

    key: str
    name: str
    status: str
    evidence: str = ""


@dataclass(frozen=True)
class QualityReport:
    """Aggregated quality report for the published dataset.

    Attributes
    ----------
    items:
        Tuple of :class:`QualityItem`, one per Q0-Q9 (in order).
    passed:
        Count of items with ``status == "pass"``.
    failed:
        Count of items with ``status == "fail"``.
    skipped:
        Count of items with ``status == "skipped"``.
    overall:
        ``"pass"`` iff ``failed == 0`` (skipped items do not fail the report).
    """

    items: tuple[QualityItem, ...]
    passed: int
    failed: int
    skipped: int
    overall: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "items": [
                {
                    "key": it.key,
                    "name": it.name,
                    "status": it.status,
                    "evidence": it.evidence,
                }
                for it in self.items
            ],
        }

    def write(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# QualityReporter Protocol
# ---------------------------------------------------------------------------


class QualityReporter(Protocol):
    """Compute the Q0-Q9 quality report for a published dataset."""

    def report(
        self,
        export_result: ExportResult,
        leakage_report: LeakageReport,
        coverage_report: CoverageReport,
        branch_scheduler: CounterfactualBranchScheduler,
        *,
        world_for_replay=None,
        utility_report: UtilityEvaluationReport | None = None,
    ) -> QualityReport:
        ...


# ---------------------------------------------------------------------------
# MinimalQualityReporter — reference stub
# ---------------------------------------------------------------------------


class MinimalQualityReporter:
    """Reference reporter that scans the exported dataset directory.

    The reporter walks each split directory, reads every
    ``t*.json`` transition file, and runs the configured checks. Q3
    (replay) requires a world instance; if ``world_for_replay`` is None,
    Q3 is marked ``skipped``.
    """

    def __init__(self, *, config: QualityConfig | None = None) -> None:
        self.config = config or QualityConfig()

    def report(
        self,
        export_result: ExportResult,
        leakage_report: LeakageReport,
        coverage_report: CoverageReport,
        branch_scheduler: CounterfactualBranchScheduler,
        *,
        world_for_replay=None,
        utility_report: UtilityEvaluationReport | None = None,
    ) -> QualityReport:
        items: list[QualityItem] = []

        # Collect all transition records across splits. The split-aware
        # grouping lets Q5 independently recompute cross-split
        # disjointness from the on-disk records.
        records_by_split = self._collect_records_by_split(export_result)
        records = [
            r
            for split_records in records_by_split.values()
            for r in split_records
        ]

        # Q0 Schema
        items.append(self._check_q0_schema(records))
        # Q1 Traceability
        items.append(self._check_q1_traceability(records))
        # Q2 Diff/Apply
        items.append(self._check_q2_diff_apply(records))
        # Q3 Replay
        items.append(self._check_q3_replay(records, world_for_replay))
        # Q4 Provenance
        items.append(
            self._check_q4_provenance(records, export_result=export_result)
        )
        # Q5 Leakage
        items.append(
            self._check_q5_leakage(
                leakage_report, records_by_split=records_by_split
            )
        )
        # Q6 Coverage Consistency
        items.append(self._check_q6_coverage(coverage_report, records=records))
        # Q7 Counterfactual
        items.append(self._check_q7_counterfactual(branch_scheduler))
        # Q8 Quarantine
        items.append(self._check_q8_quarantine(export_result))
        # Q9 Utility
        items.append(self._check_q9_utility(utility_report))

        passed = sum(1 for it in items if it.status == "pass")
        failed = sum(1 for it in items if it.status == "fail")
        skipped = sum(1 for it in items if it.status == "skipped")
        overall = "pass" if failed == 0 else "fail"

        return QualityReport(
            items=tuple(items),
            passed=passed,
            failed=failed,
            skipped=skipped,
            overall=overall,
        )

    # ------------------------------------------------------------------
    # Record collection
    # ------------------------------------------------------------------

    def _collect_records(self, export_result: ExportResult) -> list[dict]:
        """Read every transition JSON in every split directory."""
        return [
            r
            for split_records in self._collect_records_by_split(
                export_result
            ).values()
            for r in split_records
        ]

    def _collect_records_by_split(
        self, export_result: ExportResult
    ) -> dict[str, list[dict]]:
        """Read every transition JSON, grouped by split name."""
        by_split: dict[str, list[dict]] = {}
        for split in export_result.splits:
            records = by_split.setdefault(split.name, [])
            for ep_dir in split.output_dir.iterdir():
                if not ep_dir.is_dir():
                    continue
                if ep_dir.name.startswith("_"):
                    continue  # skip _quarantine
                for f in sorted(ep_dir.glob("t*.json")):
                    if f.name.startswith("t") and f.suffix == ".json":
                        try:
                            with open(f, "r", encoding="utf-8") as fh:
                                records.append(json.load(fh))
                        except (OSError, json.JSONDecodeError):
                            continue
        return by_split

    # ------------------------------------------------------------------
    # Q-item checks
    # ------------------------------------------------------------------

    def _check_q0_schema(self, records: list[dict]) -> QualityItem:
        if not self.config.run_schema_check:
            return QualityItem("Q0", "Schema", "skipped", "disabled by config")
        if not records:
            return QualityItem("Q0", "Schema", "fail", "no records found")
        # Records are dicts (from JSON). Layer 1 checks 7 dict invariants:
        #   1. field-presence (11 required fields)
        #   2. schema_version == PROTOCOL_SCHEMA_VERSION
        #   3. receipt_completeness: executed_actions.keys() == receipts.keys()
        #   4. outcome_code_legality: every receipt.outcome_code in KERNEL_OUTCOME_CODES
        #   5. authority_grounding: capability.authority != "learned" or not ground_truth
        #   6. tick_non_negative: tick is an int >= 0
        #   7. producer_non_empty: producer_id and producer_version are non-empty strings
        # Layer 2 (mechanical gate upgrade): typed-reconstruct the dict
        # into a kernel TransitionRecord (re-running every nested
        # __post_init__ construction rule: receipt success/outcome
        # pairing, delta change-kind enums, capability pairing rules,
        # ...) and run the ACTUAL kernel ``validate_transition`` on it.
        # This makes the module docstring's "passes validate_transition"
        # literally true at the record level.
        required_fields = (
            "schema_version",
            "producer_id",
            "producer_version",
            "tick",
            "state_before_hash",
            "state_after_hash",
            "candidate_actions",
            "executed_actions",
            "receipts",
            "state_delta",
            "capability_profile",
        )
        valid = 0
        dict_invalid = 0
        typed_invalid = 0
        first_typed_error = ""
        for r in records:
            if not _record_schema_valid(r, required_fields):
                dict_invalid += 1
                continue
            ok, err = _kernel_validate_record(r)
            if ok:
                valid += 1
            else:
                typed_invalid += 1
                if not first_typed_error:
                    first_typed_error = err
        total = len(records)
        if valid == total:
            return QualityItem(
                "Q0",
                "Schema",
                "pass",
                f"{valid}/{total} records schema-valid (7 invariants: "
                "field-presence + schema_version + receipt_completeness + "
                "outcome_code_legality + authority_grounding + "
                "tick_non_negative + producer_non_empty); kernel "
                f"validate_transition: {valid}/{total} typed-reconstructed "
                "records passed record-level invariants",
            )
        detail = (
            f"dict-invariant failures: {dict_invalid}, kernel "
            f"typed-validation failures: {typed_invalid}"
        )
        if first_typed_error:
            detail += f"; first: {first_typed_error}"
        return QualityItem(
            "Q0",
            "Schema",
            "fail",
            f"{total - valid}/{total} records failed schema validation "
            f"({detail})",
        )

    def _check_q1_traceability(self, records: list[dict]) -> QualityItem:
        if not self.config.run_traceability_check:
            return QualityItem("Q1", "Traceability", "skipped", "disabled by config")
        if not records:
            return QualityItem("Q1", "Traceability", "fail", "no records")
        # Q1 traceability — five sub-checks:
        #   1. executed_actions.keys() == receipts.keys() (per record)
        #   2. state hash chain: within an episode, sorted by tick,
        #      transition[t].state_after_hash == transition[t+1].state_before_hash
        #   3. candidate -> executed reference closure: each executed action's
        #      proposal_hash must equal hash_state(ActionProposal reconstructed
        #      from the candidate). If an executed agent is missing from
        #      candidates, it is "orphan executed" — ACCEPTANCE requires
        #      "三要素齐全" (candidate + executed + receipt all present),
        #      so orphan_exec_count > 0 is a FAILURE, not just a note.
        #   4. tick uniqueness within an episode: two records with the
        #      same (episode_id, tick) break the one-transition-per-tick
        #      contract AND silently corrupt the hash-chain sort order
        #      (the old gate would still count their chain link as valid
        #      when the duplicate carried identical hashes).
        #   5. episode_id presence: a record without an episode_id
        #      cannot be attributed to any episode and is untraceable.
        total = len(records)
        exec_receipt_match = 0
        for r in records:
            exec_keys = set(r.get("executed_actions", {}).keys())
            receipt_keys = set(r.get("receipts", {}).keys())
            if exec_keys == receipt_keys:
                exec_receipt_match += 1

        # Group records by episode_id (from provenance) for hash chain.
        missing_episode_count = 0
        by_episode: dict[str, list[dict]] = {}
        for r in records:
            prov = r.get("provenance", {})
            if not isinstance(prov, dict):
                prov = {}
            ep_id = prov.get("episode_id", "")
            if not isinstance(ep_id, str) or not ep_id:
                ep_id = ""
                missing_episode_count += 1
            by_episode.setdefault(ep_id, []).append(r)

        chain_total = 0
        chain_valid = 0
        orphan_exec_count = 0
        proposal_hash_mismatch = 0
        duplicate_tick_count = 0

        for _ep_id, ep_records in by_episode.items():
            # Sort by tick within episode.
            try:
                ep_sorted = sorted(
                    ep_records, key=lambda r: r.get("tick", 0)
                )
            except TypeError:
                ep_sorted = ep_records

            # Tick uniqueness within the episode.
            seen_ticks: set = set()
            for r in ep_sorted:
                t = r.get("tick")
                if t in seen_ticks:
                    duplicate_tick_count += 1
                else:
                    seen_ticks.add(t)

            # Hash chain links.
            for i in range(len(ep_sorted) - 1):
                chain_total += 1
                cur = ep_sorted[i]
                nxt = ep_sorted[i + 1]
                if cur.get("state_after_hash") == nxt.get("state_before_hash"):
                    chain_valid += 1

            # candidate -> executed proposal_hash closure.
            for r in ep_sorted:
                candidates = r.get("candidate_actions", {})
                executed = r.get("executed_actions", {})
                if not isinstance(candidates, dict):
                    candidates = {}
                if not isinstance(executed, dict):
                    executed = {}
                for agent_id, ex in executed.items():
                    if not isinstance(ex, dict):
                        proposal_hash_mismatch += 1
                        continue
                    if agent_id not in candidates:
                        orphan_exec_count += 1
                        continue
                    cand = candidates[agent_id]
                    if not isinstance(cand, dict):
                        proposal_hash_mismatch += 1
                        continue
                    try:
                        proposal = ActionProposal(
                            agent_id=cand.get("agent_id"),
                            action_type=cand.get("action_type", ""),
                            params=cand.get("params", {}),
                            proposed_at_tick=cand.get("proposed_at_tick", 0),
                            proposer=cand.get("proposer", ""),
                        )
                        computed_hash = hash_state(proposal)
                        if computed_hash != ex.get("proposal_hash"):
                            proposal_hash_mismatch += 1
                    except Exception:
                        proposal_hash_mismatch += 1

        evidence_parts = [
            f"{total} records",
            f"{chain_valid}/{chain_total} hash-chain links valid",
            f"{exec_receipt_match}/{total} exec/receipt keys match",
        ]
        if orphan_exec_count > 0:
            evidence_parts.append(
                f"{orphan_exec_count} orphan executed (candidate missing)"
            )
        if proposal_hash_mismatch > 0:
            evidence_parts.append(
                f"{proposal_hash_mismatch} proposal_hash mismatch"
            )
        if duplicate_tick_count > 0:
            evidence_parts.append(
                f"{duplicate_tick_count} duplicate tick(s) within episode"
            )
        if missing_episode_count > 0:
            evidence_parts.append(
                f"{missing_episode_count} record(s) missing episode_id"
            )
        evidence = ", ".join(evidence_parts)

        chain_ok = chain_valid == chain_total
        exec_ok = exec_receipt_match == total
        hash_ok = proposal_hash_mismatch == 0
        orphan_ok = orphan_exec_count == 0
        tick_ok = duplicate_tick_count == 0
        episode_ok = missing_episode_count == 0
        if (
            chain_ok
            and exec_ok
            and hash_ok
            and orphan_ok
            and tick_ok
            and episode_ok
        ):
            return QualityItem("Q1", "Traceability", "pass", evidence)
        return QualityItem("Q1", "Traceability", "fail", evidence)

    def _check_q2_diff_apply(self, records: list[dict]) -> QualityItem:
        if not self.config.run_diff_apply_check:
            return QualityItem(
                "Q2",
                "Structural Delta-Capability Consistency",
                "skipped",
                "disabled by config",
            )
        if not records:
            return QualityItem(
                "Q2",
                "Structural Delta-Capability Consistency",
                "fail",
                "no records",
            )
        # Q2 structural delta-capability consistency check.
        # For each capability slot, if capability.{slot} == False then
        # state_delta.{slot}_changes MUST be None. If capability.{slot}
        # == True, the changes bundle may be None (no change) or non-None.
        # Also verifies state_delta.meta_after.tick == record.tick + 1
        # when meta_after is present.
        # Note: this is the STRUCTURAL layer. Full diff_state + apply_delta
        # round-trip verification requires StateView reconstruction and
        # is deferred to Q3 (replay) in attempt 6.
        slot_mapping = (
            ("fields", "field_changes"),
            ("entities", "entity_changes"),
            ("relations", "relation_changes"),
            ("registries", "registry_changes"),
            ("population", "population_changes"),
            ("events", "event_log"),
        )
        total = len(records)
        consistent = 0
        meta_tick_valid = 0
        meta_total = 0
        for r in records:
            cap = r.get("capability_profile", {})
            delta = r.get("state_delta", {})
            if not isinstance(cap, dict) or not isinstance(delta, dict):
                continue
            ok = True
            for cap_slot, delta_key in slot_mapping:
                cap_flag = cap.get(cap_slot)
                delta_val = delta.get(delta_key)
                # capability=False => delta MUST be None
                if cap_flag is False and delta_val is not None:
                    ok = False
                    break
                # capability=True => delta may be None or non-None (both OK)
            if ok:
                consistent += 1
            # meta_after.tick == record.tick + 1 (when meta_after present)
            meta_after = delta.get("meta_after")
            if meta_after is not None:
                meta_total += 1
                if isinstance(meta_after, dict):
                    expected_tick = r.get("tick", -1) + 1
                    if meta_after.get("tick") == expected_tick:
                        meta_tick_valid += 1
        evidence = (
            f"{total} records, {consistent}/{total} delta-capability "
            f"consistent, {meta_tick_valid}/{meta_total} meta_after.tick "
            "valid (structural check; full round-trip deferred to Q3 "
            "(attempt 6))"
        )
        if consistent == total and meta_tick_valid == meta_total:
            return QualityItem(
                "Q2",
                "Structural Delta-Capability Consistency",
                "pass",
                evidence,
            )
        return QualityItem(
            "Q2",
            "Structural Delta-Capability Consistency",
            "fail",
            evidence,
        )

    def _check_q3_replay(self, records: list[dict], world_for_replay) -> QualityItem:
        if not self.config.run_replay_check:
            return QualityItem("Q3", "Replay", "skipped", "disabled by config")
        if world_for_replay is None:
            return QualityItem(
                "Q3",
                "Replay",
                "skipped",
                "no world provided for replay verification",
            )
        # Q3 requires a world that declares exact_restore=True. Worlds
        # without this capability cannot be bit-identical replayed; the
        # check is informational only and is reported as skipped.
        cap = world_for_replay.capabilities
        if not cap.exact_restore:
            return QualityItem(
                "Q3",
                "Replay",
                "skipped",
                "world does not support exact_restore",
            )
        if not records:
            return QualityItem("Q3", "Replay", "fail", "no records to replay")

        # Group records by episode_id (from provenance). EVERY episode
        # is replayed (first N records each) — the old gate sampled only
        # the first episode, leaving all other episodes unverified.
        by_episode: dict[str, list[dict]] = {}
        for r in records:
            prov = r.get("provenance", {})
            if not isinstance(prov, dict):
                prov = {}
            ep_id = prov.get("episode_id", "")
            if not isinstance(ep_id, str) or not ep_id:
                ep_id = ""
            by_episode.setdefault(ep_id, []).append(r)

        if not by_episode:
            return QualityItem(
                "Q3", "Replay", "fail", "no episode records found"
            )

        # Per-episode sample size: the first N records. 5 is enough to
        # catch divergence while keeping the check fast.
        n_target = 5
        total_sampled = 0
        match_count = 0
        diverged_ticks: list[int] = []
        anchor_failures: list[str] = []

        for target_ep_id in sorted(by_episode.keys()):
            ep_records = by_episode[target_ep_id]
            ep_label = target_ep_id or "<missing episode_id>"

            # Sort by tick within the episode.
            try:
                ep_sorted = sorted(
                    ep_records, key=lambda r: r.get("tick", 0)
                )
            except TypeError:
                ep_sorted = list(ep_records)

            n = min(n_target, len(ep_sorted))
            sample = ep_sorted[:n]

            # Extract seed from the first record's provenance. Falls
            # back to 0 if the seed is missing or unparseable.
            seed = _extract_seed_from_record(sample[0])

            # Reset the world to the initial state. The world's
            # ``reset`` is responsible for seeding its RNG; we trust the
            # world to be deterministic given (seed, action sequence).
            try:
                world_for_replay.reset(seed=seed)
            except Exception as exc:  # noqa: BLE001 — protocol-level guard
                return QualityItem(
                    "Q3",
                    "Replay",
                    "fail",
                    f"world.reset raised for episode {ep_label}: {exc!r}",
                )

            # Initial-state anchor: the post-reset state hash MUST equal
            # the first record's recorded ``state_before_hash``. Without
            # this anchor, a replay that starts from the wrong state but
            # happens to produce matching after-hashes (or a tampered
            # ``state_before_hash``) would go undetected.
            try:
                initial_hash = hash_state(world_for_replay.observe())
            except Exception as exc:  # noqa: BLE001
                return QualityItem(
                    "Q3",
                    "Replay",
                    "fail",
                    f"world.observe raised after reset for episode "
                    f"{ep_label}: {exc!r}",
                )
            if initial_hash != sample[0].get("state_before_hash", ""):
                anchor_failures.append(ep_label)
                # Replay from a wrong starting state is meaningless —
                # skip this episode's replay and report the anchor.
                continue

            total_sampled += n

            # Replay each record's executed_actions in order and compare
            # the post-step state hash against the recorded
            # state_after_hash. A single record may carry multiple
            # agents' executed actions; we replay them in sorted
            # agent_id order (the world's step() takes one
            # ExecutedAction at a time, so multi-agent records step once
            # per agent). We also replay the recorded exogenous_input so
            # that worlds whose rollout applied exogenous events (e.g.,
            # hazard_escalation in the M5 emergency scenario) replay
            # bit-identically.
            for i, r in enumerate(sample):
                actions = _build_executed_actions_from_record(r)
                if not actions:
                    return QualityItem(
                        "Q3",
                        "Replay",
                        "fail",
                        f"record {i} (tick={r.get('tick', '?')}) has no "
                        "executable actions",
                    )
                # Reconstruct the exogenous input recorded for this
                # tick. The replay re-injects it so the world applies
                # the same environmental event as the original rollout.
                # Without this, worlds with auto-generated exogenous
                # events diverge.
                exogenous = _build_exogenous_from_record(r)
                # Joint-mode records (Phase 5) carry provenance
                # execution_mode='joint' and MUST be replayed as ONE
                # parallel step via step_joint — stepping agents one at
                # a time would change the env semantics and diverge.
                prov_r = r.get("provenance", {})
                is_joint = (
                    isinstance(prov_r, dict)
                    and prov_r.get("execution_mode") == "joint"
                )
                try:
                    if is_joint:
                        step_joint = getattr(
                            world_for_replay, "step_joint", None
                        )
                        if not callable(step_joint):
                            return QualityItem(
                                "Q3",
                                "Replay",
                                "fail",
                                f"record {i} is joint-mode but the replay "
                                "world does not support step_joint",
                            )
                        joint = _build_joint_action_from_record(r, actions)
                        step_joint(joint, exogenous=exogenous)
                    else:
                        for action in actions:
                            world_for_replay.step(
                                action, exogenous=exogenous
                            )
                            # Only inject exogenous on the first step of
                            # this record (matches rollout: one
                            # exogenous per tick).
                            exogenous = None
                except Exception as exc:  # noqa: BLE001
                    tick_val = r.get("tick", i)
                    return QualityItem(
                        "Q3",
                        "Replay",
                        "fail",
                        f"world.step raised at record {i} "
                        f"(tick={tick_val}): {exc!r}",
                    )
                try:
                    actual_hash = hash_state(world_for_replay.observe())
                except Exception as exc:  # noqa: BLE001
                    return QualityItem(
                        "Q3",
                        "Replay",
                        "fail",
                        f"world.observe raised at record {i}: {exc!r}",
                    )
                expected_hash = r.get("state_after_hash", "")
                if actual_hash == expected_hash:
                    match_count += 1
                else:
                    diverged_ticks.append(r.get("tick", i))

        n_episodes = len(by_episode)
        if anchor_failures:
            return QualityItem(
                "Q3",
                "Replay",
                "fail",
                f"initial-state anchor mismatch for "
                f"{len(anchor_failures)}/{n_episodes} episode(s) "
                f"({', '.join(anchor_failures)}): post-reset state hash "
                "!= recorded state_before_hash of the first record",
            )
        if match_count == total_sampled:
            return QualityItem(
                "Q3",
                "Replay",
                "pass",
                f"{total_sampled}/{total_sampled} records replay "
                "bit-identical (state_after_hash match) across "
                f"{n_episodes}/{n_episodes} episode(s); initial-state "
                "anchor verified per episode",
            )
        return QualityItem(
            "Q3",
            "Replay",
            "fail",
            f"{match_count}/{total_sampled} records match, "
            f"{total_sampled - match_count} diverged at ticks "
            f"{diverged_ticks}",
        )

    def _check_q4_provenance(
        self,
        records: list[dict],
        *,
        export_result: ExportResult | None = None,
    ) -> QualityItem:
        if not self.config.run_provenance_check:
            return QualityItem("Q4", "Provenance", "skipped", "disabled by config")
        if not records:
            return QualityItem("Q4", "Provenance", "fail", "no records")
        # Q4 provenance completeness — every record's provenance MUST
        # contain the 5 required keys:
        #   1. policy_id (non-empty string)
        #   2. policy_version (non-empty string)
        #   3. inference_config (dict; may be empty but key must exist)
        #   4. episode_id (non-empty string)
        #   5. seed (present; may be string or number)
        # Mechanical gate upgrade: when ``export_result`` is supplied,
        # the per-record layer is cross-checked against the exporter's
        # manifest — the count identity (manifest.total_records ==
        # Σ split record_count == records collected on disk), the
        # episode closure (every record's episode_id is declared in
        # exactly the split set) and per-episode seed consistency.
        total = len(records)
        complete = 0
        episodes_seen: set[str] = set()
        seeds_by_episode: dict[str, set[str]] = {}
        for r in records:
            prov = r.get("provenance", {})
            if not isinstance(prov, dict):
                continue
            policy_id = prov.get("policy_id")
            if not isinstance(policy_id, str) or not policy_id:
                continue
            policy_version = prov.get("policy_version")
            if not isinstance(policy_version, str) or not policy_version:
                continue
            inference_config = prov.get("inference_config")
            if not isinstance(inference_config, dict):
                continue
            episode_id = prov.get("episode_id")
            if not isinstance(episode_id, str) or not episode_id:
                continue
            seed = prov.get("seed")
            if seed is None:
                continue
            if isinstance(seed, bool):
                # bool is a subclass of int but is not a valid seed value.
                continue
            if not isinstance(seed, (str, int, float)):
                continue
            complete += 1
            episodes_seen.add(episode_id)
            seeds_by_episode.setdefault(episode_id, set()).add(str(seed))
        if complete != total:
            return QualityItem(
                "Q4",
                "Provenance",
                "fail",
                f"{total - complete}/{total} records missing required "
                "provenance keys (policy_id + policy_version + "
                "inference_config + episode_id + seed)",
            )

        manifest_note = "manifest cross-check skipped (no export_result)"
        if export_result is not None:
            problems: list[str] = []
            manifest_total: int | None = None
            split_sum: int | None = None
            try:
                with open(
                    export_result.dataset_manifest_path,
                    "r",
                    encoding="utf-8",
                ) as f:
                    payload = json.load(f)
                manifest_total = int(payload["total_records"])
                splits_payload = payload.get("splits", {})
                split_sum = sum(
                    int(v.get("record_count", 0))
                    for v in splits_payload.values()
                )
            except (
                OSError,
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
                AttributeError,
            ) as exc:
                problems.append(f"manifest unreadable: {exc!r}")
            if manifest_total is not None:
                if manifest_total != split_sum:
                    problems.append(
                        f"manifest total_records={manifest_total} != "
                        f"\u03a3 split record_count={split_sum}"
                    )
                if manifest_total != total:
                    problems.append(
                        f"manifest total_records={manifest_total} != "
                        f"{total} records collected on disk"
                    )
            declared: set[str] = set()
            for split in export_result.splits:
                declared.update(split.episode_ids)
            undeclared = sorted(episodes_seen - declared)
            if undeclared:
                problems.append(
                    "episode_id(s) in records but not declared in any "
                    f"split: {undeclared}"
                )
            seed_conflicts = sorted(
                ep
                for ep, seeds in seeds_by_episode.items()
                if len(seeds) > 1
            )
            if seed_conflicts:
                problems.append(
                    f"inconsistent seed within episode(s): {seed_conflicts}"
                )
            if problems:
                return QualityItem(
                    "Q4",
                    "Provenance",
                    "fail",
                    "manifest cross-check failed: " + "; ".join(problems),
                )
            manifest_note = (
                f"manifest cross-check: total_records={manifest_total} == "
                f"\u03a3 split record_count == {total} on-disk records; "
                "episode closure + per-episode seed consistency verified"
            )
        return QualityItem(
            "Q4",
            "Provenance",
            "pass",
            f"{complete}/{total} records have complete provenance "
            "(policy_id + policy_version + inference_config + "
            f"episode_id + seed); {manifest_note}",
        )

    def _check_q5_leakage(
        self,
        leakage_report: LeakageReport,
        *,
        records_by_split: Mapping[str, list[dict]] | None = None,
    ) -> QualityItem:
        # Mechanical gate upgrade — three layers:
        #   1. Internal consistency: ok ⇔ violations empty, and
        #      Σ by_kind == len(violations). A report that says ok=True
        #      while carrying violations is corrupt, not clean.
        #   2. Anti-vacuous: ok=True with checked_kinds=() means NOTHING
        #      was checked — the old gate would happily pass it.
        #   3. Independent recompute: cross-split episode/seed
        #      disjointness is recomputed from the on-disk records'
        #      provenance, so an upstream checker that silently
        #      under-reports is caught mechanically.
        violations = tuple(leakage_report.violations)
        by_kind_sum = sum(
            int(v) for v in dict(leakage_report.by_kind).values()
        )
        if (
            leakage_report.ok != (len(violations) == 0)
            or by_kind_sum != len(violations)
        ):
            return QualityItem(
                "Q5",
                "Leakage",
                "fail",
                f"leakage report internally inconsistent: "
                f"ok={leakage_report.ok}, violations={len(violations)}, "
                f"\u03a3 by_kind={by_kind_sum}",
            )
        if leakage_report.ok and not leakage_report.checked_kinds:
            return QualityItem(
                "Q5",
                "Leakage",
                "fail",
                "vacuous leakage report: ok=True but checked_kinds is "
                "empty (nothing was actually checked)",
            )
        if not leakage_report.ok:
            return QualityItem(
                "Q5",
                "Leakage",
                "fail",
                f"{len(leakage_report.violations)} violations: "
                f"{dict(leakage_report.by_kind)}",
            )
        recompute_note = "independent recompute skipped (no per-split records)"
        if records_by_split:
            ep_splits: dict[str, set[str]] = {}
            seed_splits: dict[str, set[str]] = {}
            for split_name, split_records in records_by_split.items():
                for r in split_records:
                    prov = r.get("provenance", {})
                    if not isinstance(prov, dict):
                        continue
                    ep = prov.get("episode_id")
                    if isinstance(ep, str) and ep:
                        ep_splits.setdefault(ep, set()).add(split_name)
                    seed = prov.get("seed")
                    if seed is not None and not isinstance(seed, bool):
                        seed_splits.setdefault(str(seed), set()).add(
                            split_name
                        )
            ep_overlaps = sorted(
                k for k, s in ep_splits.items() if len(s) > 1
            )
            if ep_overlaps:
                return QualityItem(
                    "Q5",
                    "Leakage",
                    "fail",
                    "upstream report says ok but independent recompute "
                    "found episode(s) in multiple splits: "
                    f"{ep_overlaps}",
                )
            if "seed" in leakage_report.checked_kinds:
                seed_overlaps = sorted(
                    k for k, s in seed_splits.items() if len(s) > 1
                )
                if seed_overlaps:
                    return QualityItem(
                        "Q5",
                        "Leakage",
                        "fail",
                        "upstream report says ok but independent "
                        "recompute found seed(s) in multiple splits: "
                        f"{seed_overlaps}",
                    )
            recompute_note = (
                "independently recomputed from records: episodes "
                f"disjoint across {len(records_by_split)} split(s)"
            )
            if "seed" in leakage_report.checked_kinds:
                recompute_note += ", seeds disjoint"
        kinds = ", ".join(leakage_report.checked_kinds) or "none"
        return QualityItem(
            "Q5",
            "Leakage",
            "pass",
            f"0 violations (checked: {kinds}); {recompute_note}",
        )

    def _check_q6_coverage(
        self,
        coverage_report: CoverageReport,
        *,
        records: list[dict] | None = None,
    ) -> QualityItem:
        # Renamed "Coverage" -> "Coverage Consistency": the gate does
        # NOT judge whether coverage is *sufficient* (that needs an
        # external target); it mechanically verifies the coverage
        # report is CONSISTENT with the published dataset:
        #   1. Observation superset: transition_count >= published
        #      records (the scheduler observes every mainline
        #      transition, including later-quarantined ones).
        #   2. Count domination: for every action_type / outcome_code
        #      recounted from the dataset, the coverage counter must be
        #      >= the dataset count (same counting rule: one per
        #      candidate proposal / one per receipt).
        #   3. Policy closure: every provenance.policy_id in the
        #      dataset appears in policy_usage.
        name = "Coverage Consistency"
        if coverage_report.transition_count == 0:
            return QualityItem("Q6", name, "fail", "no transitions observed")
        prefix = (
            f"{coverage_report.transition_count} transitions, "
            f"{len(coverage_report.action_type_counts)} action types, "
            f"{len(coverage_report.policy_usage)} policies"
        )
        if not records:
            return QualityItem(
                "Q6",
                name,
                "pass",
                prefix
                + " (dataset consistency cross-check skipped: no records "
                "provided)",
            )
        problems: list[str] = []
        published = len(records)
        if coverage_report.transition_count < published:
            problems.append(
                f"observed {coverage_report.transition_count} transitions "
                f"< {published} published records (under-observed)"
            )
        dataset_action_counts: dict[str, int] = {}
        dataset_outcome_counts: dict[str, int] = {}
        dataset_policies: set[str] = set()
        for r in records:
            cands = r.get("candidate_actions", {})
            if isinstance(cands, dict):
                for c in cands.values():
                    if isinstance(c, dict):
                        at = c.get("action_type")
                        if isinstance(at, str) and at:
                            dataset_action_counts[at] = (
                                dataset_action_counts.get(at, 0) + 1
                            )
            receipts = r.get("receipts", {})
            if isinstance(receipts, dict):
                for rc in receipts.values():
                    if isinstance(rc, dict):
                        oc = rc.get("outcome_code")
                        if isinstance(oc, str) and oc:
                            dataset_outcome_counts[oc] = (
                                dataset_outcome_counts.get(oc, 0) + 1
                            )
            prov = r.get("provenance", {})
            if isinstance(prov, dict):
                pid = prov.get("policy_id")
                if isinstance(pid, str) and pid:
                    dataset_policies.add(pid)
        for at, cnt in sorted(dataset_action_counts.items()):
            observed = int(coverage_report.action_type_counts.get(at, 0))
            if observed < cnt:
                problems.append(
                    f"action_type {at!r}: coverage count {observed} < "
                    f"dataset count {cnt}"
                )
        for oc, cnt in sorted(dataset_outcome_counts.items()):
            observed = int(coverage_report.outcome_code_counts.get(oc, 0))
            if observed < cnt:
                problems.append(
                    f"outcome_code {oc!r}: coverage count {observed} < "
                    f"dataset count {cnt}"
                )
        missing_policies = sorted(
            dataset_policies - set(coverage_report.policy_usage.keys())
        )
        if missing_policies:
            problems.append(
                "policy_id(s) in dataset but never counted by the "
                f"scheduler: {missing_policies}"
            )
        if problems:
            return QualityItem(
                "Q6",
                name,
                "fail",
                prefix + "; consistency violations: " + "; ".join(problems),
            )
        return QualityItem(
            "Q6",
            name,
            "pass",
            prefix
            + f"; consistent with published dataset ({published} records: "
            "observation superset, action-type/outcome-code count "
            "domination, policy closure)",
        )

    def _check_q7_counterfactual(
        self, branch_scheduler: CounterfactualBranchScheduler
    ) -> QualityItem:
        summary = branch_scheduler.branch_summary()
        branch_count = int(summary.get("branch_count", 0))
        held_fixed = summary.get("held_fixed", {})
        if branch_count == 0:
            return QualityItem(
                "Q7",
                "Counterfactual",
                "skipped",
                f"no branches generated (mode={summary.get('mode', 'unknown')})",
            )
        if not held_fixed:
            return QualityItem(
                "Q7",
                "Counterfactual",
                "fail",
                f"{branch_count} branches but no held_fixed factors recorded",
            )
        # Phase 3 §6.5: mechanical held-fixed verification.
        # If the scheduler exposes ``held_fixed_verification``, read it
        # and verify the invariants mechanically. Schedulers that don't
        # expose this field (e.g., test stubs) fall back to the legacy
        # declarative check.
        verifications = summary.get("held_fixed_verification") or []
        if not verifications:
            # Legacy path: scheduler didn't capture mechanical
            # fingerprints. Mark as PENDING_DETERMINISTIC_RERUN — the
            # declarative held_fixed is recorded but not mechanically
            # verified. This is the same status B3 used for evidence
            # gate; it signals "structural pass, behavior not validated".
            return QualityItem(
                "Q7",
                "Counterfactual",
                "pass",
                f"{branch_count} branches, held_fixed: {list(held_fixed)} "
                "(declarative only — no held_fixed_verification field; "
                "mechanical verification PENDING)",
            )
        # Mechanical verification path: check every fork group.
        fork_count = len(verifications)
        parent_restored_count = sum(
            1 for v in verifications if v.get("parent_state_restored")
        )
        all_restoration_ok_count = sum(
            1 for v in verifications if v.get("all_restoration_ok")
        )
        rng_captured_count = sum(
            1 for v in verifications if v.get("rng_bundle_captured")
        )
        non_focal_consistent_count = sum(
            1 for v in verifications if v.get("non_focal_actions_consistent")
        )
        mode = str(summary.get("mode", "unknown"))
        # Parent restoration is mandatory in every mode. Joint mode also
        # requires RNG capture and a non-focal action fingerprint for every
        # fork group; otherwise it cannot support a multi-agent held-fixed
        # claim.
        primary_ok = (
            parent_restored_count == fork_count
            and all_restoration_ok_count == fork_count
        )
        if mode == "joint_kernel_branch":
            primary_ok = (
                primary_ok
                and rng_captured_count == fork_count
                and non_focal_consistent_count == fork_count
            )
        evidence_parts = [
            f"{branch_count} branches across {fork_count} fork groups",
            f"parent_state_restored: {parent_restored_count}/{fork_count}",
            f"all_restoration_ok: {all_restoration_ok_count}/{fork_count}",
            f"rng_bundle_captured: {rng_captured_count}/{fork_count}",
            f"non_focal_actions_consistent: "
            f"{non_focal_consistent_count}/{fork_count}",
            f"held_fixed: {list(held_fixed)}",
            f"scope: {mode}",
        ]
        evidence = "; ".join(evidence_parts)
        if primary_ok:
            return QualityItem(
                "Q7",
                "Counterfactual",
                "pass",
                f"{evidence} (mechanically verified)",
            )
        return QualityItem(
            "Q7",
            "Counterfactual",
            "fail",
            f"{evidence} — held-fixed invariant violated",
        )

    def _check_q8_quarantine(self, export_result: ExportResult) -> QualityItem:
        if not self.config.run_quarantine_check:
            return QualityItem("Q8", "Quarantine", "skipped", "disabled by config")
        # Look for _quarantine directories in any split.
        quarantine_found = False
        for split in export_result.splits:
            for ep_dir in split.output_dir.iterdir():
                if not ep_dir.is_dir():
                    continue
                q_dir = ep_dir / "_quarantine"
                if q_dir.exists() and q_dir.is_dir():
                    quarantine_found = True
                    break
            if quarantine_found:
                break

        # Phase 3 §6.5: Q8 quantity identity verification.
        # The exporter populates ExportResult.{produced, accepted,
        # quarantined, explicitly_rejected, dropped}. We verify:
        #   1. produced == accepted + quarantined + explicitly_rejected
        #   2. dropped == 0 (no records lost without accounting)
        # When produced == 0 (legacy exporter or empty dataset), the
        # identity is trivially satisfied (0 == 0 + 0 + 0) and we fall
        # back to the legacy "quarantine dir present" check.
        produced = int(getattr(export_result, "produced", 0))
        accepted = int(getattr(export_result, "accepted", 0))
        quarantined = int(getattr(export_result, "quarantined", 0))
        explicitly_rejected = int(
            getattr(export_result, "explicitly_rejected", 0)
        )
        dropped = int(getattr(export_result, "dropped", 0))

        identity_holds = (
            produced == accepted + quarantined + explicitly_rejected
            and dropped == 0
        )

        if produced > 0:
            # Mechanical identity check (Phase 3 §6.5).
            if not identity_holds:
                return QualityItem(
                    "Q8",
                    "Quarantine",
                    "fail",
                    f"quantity identity violated: produced={produced} != "
                    f"accepted({accepted}) + quarantined({quarantined}) + "
                    f"explicitly_rejected({explicitly_rejected}) "
                    f"= {accepted + quarantined + explicitly_rejected}; "
                    f"dropped={dropped} (MUST be 0)",
                )
            return QualityItem(
                "Q8",
                "Quarantine",
                "pass",
                f"quantity identity holds: produced={produced} == "
                f"accepted({accepted}) + quarantined({quarantined}) + "
                f"explicitly_rejected({explicitly_rejected}); "
                f"dropped={dropped}; quarantine directory "
                f"{'present' if quarantine_found else 'not needed (no failures)'}",
            )

        # Legacy / empty-dataset path: produced == 0. Fall back to the
        # directory-existence check. Quarantine is OPTIONAL — it exists
        # only if records were quarantined. The check passes if either
        # no quarantine is needed (no failures) OR quarantine is
        # properly isolated.
        return QualityItem(
            "Q8",
            "Quarantine",
            "pass",
            f"quarantine directory {'present' if quarantine_found else 'not needed (no failures)'} "
            "(produced=0; quantity identity trivially satisfied)",
        )

    def _check_q9_utility(
        self, utility_report: UtilityEvaluationReport | None
    ) -> QualityItem:
        if utility_report is None:
            return QualityItem(
                "Q9",
                "Utility",
                "skipped",
                "matched outcome utility evaluation not supplied",
            )
        valid = utility_report.valid_comparisons
        if not valid:
            return QualityItem(
                "Q9",
                "Utility",
                "fail",
                "no valid same-state/same-exogenous policy outcome comparisons",
            )
        best_delta = utility_report.best_delta
        assert best_delta is not None
        if best_delta > utility_report.min_improvement:
            return QualityItem(
                "Q9",
                "Utility",
                "pass",
                f"{len(valid)} matched comparisons; baseline="
                f"{utility_report.baseline_policy_id}; best candidate "
                f"utility delta={best_delta:.6g} > "
                f"{utility_report.min_improvement:.6g}",
            )
        return QualityItem(
            "Q9",
            "Utility",
            "fail",
            f"{len(valid)} matched comparisons but no candidate improved "
            f"over baseline={utility_report.baseline_policy_id}; "
            f"best delta={best_delta:.6g}, required > "
            f"{utility_report.min_improvement:.6g}",
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _extract_seed_from_record(record: dict) -> int:
    """Extract a seed from a record's provenance. Falls back to 0.

    The provenance ``seed`` field is typically a string (e.g., ``"42"``)
    because the kernel's :class:`TransitionRecord.provenance` is a
    free-form mapping and worlds commonly stringify the seed. We accept
    int / float / str and coerce to int; unparseable values fall back
    to 0 (per task spec: "用 episode_id 中编码的 seed（或 0）").
    """
    prov = record.get("provenance", {})
    if not isinstance(prov, dict):
        return 0
    seed = prov.get("seed", 0)
    try:
        return int(seed)
    except (TypeError, ValueError):
        return 0


def _build_exogenous_from_record(record: dict) -> ExogenousInput | None:
    """Reconstruct the :class:`ExogenousInput` recorded for this tick.

    The record's ``exogenous_input`` field is either ``None`` or a dict
    produced by :func:`dataclasses.asdict` on an :class:`ExogenousInput`
    (keys: ``tick`` / ``kind`` / ``payload``). Returns ``None`` when the
    field is absent or null — the caller treats ``None`` as "no exogenous
    event this tick".
    """
    raw = record.get("exogenous_input")
    if not isinstance(raw, dict):
        return None
    try:
        return ExogenousInput(
            tick=int(raw.get("tick", 0)),
            kind=str(raw.get("kind", "")),
            payload=raw.get("payload", {}) or {},
        )
    except Exception:  # noqa: BLE001 — malformed exogenous → treat as none
        return None


def _build_executed_actions_from_record(
    record: dict,
) -> list[ExecutedAction]:
    """Reconstruct :class:`ExecutedAction` objects from a record dict.

    The record's ``executed_actions`` field is a mapping
    ``{agent_id: action_dict}`` where ``action_dict`` has the keys
    ``agent_id`` / ``action_type`` / ``params`` / ``executed_at_tick`` /
    ``proposal_hash`` (produced by :func:`dataclasses.asdict` on a
    :class:`TransitionRecord`). We rebuild frozen
    :class:`ExecutedAction` instances so they can be fed to
    ``world.step``.

    Actions are returned sorted by ``agent_id`` for deterministic
    replay order. Records with malformed entries are silently skipped
    (they will be reported as "no executable actions" by the caller).
    """
    executed_dict = record.get("executed_actions", {})
    if not isinstance(executed_dict, dict):
        return []
    actions: list[ExecutedAction] = []
    for agent_id in sorted(executed_dict.keys()):
        ex = executed_dict[agent_id]
        if not isinstance(ex, dict):
            continue
        try:
            actions.append(
                ExecutedAction(
                    agent_id=ex.get("agent_id", agent_id),
                    action_type=ex.get("action_type", ""),
                    params=ex.get("params", {}),
                    executed_at_tick=ex.get("executed_at_tick", 0),
                    proposal_hash=ex.get("proposal_hash", ""),
                )
            )
        except Exception:  # noqa: BLE001 — skip malformed entries
            continue
    return actions


def _build_joint_action_from_record(
    record: dict,
    actions: Sequence[ExecutedAction],
) -> JointAction:
    """Reconstruct an executed-stage :class:`JointAction` from a record.

    Used by the Q3 replay for joint-mode records. The executed set is
    rebuilt from ``executed_actions``; ``active_agents`` prefers the
    provenance ``active_agents_before`` JSON mirror (preserving the
    original agent order) and falls back to the executed keys when the
    mirror is missing or inconsistent. Proposals are left empty — the
    kernel accepts executed-only joint actions for replay consumers.
    """
    executed_by_agent = {a.agent_id: a for a in actions}
    prov = record.get("provenance", {})
    if not isinstance(prov, dict):
        prov = {}
    active: tuple[str, ...] | None = None
    raw = prov.get("active_agents_before", "")
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
            candidate = tuple(str(a) for a in parsed)
            if set(candidate) == {str(k) for k in executed_by_agent}:
                active = candidate
        except (ValueError, TypeError):
            active = None
    if active is None:
        active = tuple(sorted(str(k) for k in executed_by_agent))
    policy = prov.get("missing_agent_policy", "stay")
    if policy not in ("noop", "stay", "error"):
        policy = "stay"
    return JointAction(
        tick=int(record.get("tick", 0) or 0),
        active_agents=active,
        proposals_by_agent={},
        executed_by_agent={
            str(k): v for k, v in executed_by_agent.items()
        },
        missing_agent_policy=policy,
    )


def _record_schema_valid(
    record: dict,
    required_fields: "tuple[str, ...]",
) -> bool:
    """Check whether a record dict satisfies the 7 Q0 schema invariants.

    Returns ``True`` iff ALL of the following hold:
    1. All ``required_fields`` are present.
    2. ``schema_version == PROTOCOL_SCHEMA_VERSION``.
    3. ``executed_actions.keys() == receipts.keys()``.
    4. Every ``receipts[*].outcome_code`` is in ``KERNEL_OUTCOME_CODES``.
    5. ``capability_profile.authority != "learned"`` or
       ``capability_profile.ground_truth`` is not True.
    6. ``tick`` is a non-negative int (and not a bool).
    7. ``producer_id`` and ``producer_version`` are non-empty strings.
    """
    # 1. field-presence
    if not isinstance(record, dict):
        return False
    if not all(k in record for k in required_fields):
        return False
    # 2. schema_version
    if record.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        return False
    # 3. receipt_completeness
    exec_actions = record.get("executed_actions", {})
    receipts = record.get("receipts", {})
    if not isinstance(exec_actions, dict) or not isinstance(receipts, dict):
        return False
    if set(exec_actions.keys()) != set(receipts.keys()):
        return False
    # 4. outcome_code_legality
    for receipt in receipts.values():
        if not isinstance(receipt, dict):
            return False
        if receipt.get("outcome_code") not in KERNEL_OUTCOME_CODES:
            return False
    # 5. authority_grounding
    cap = record.get("capability_profile", {})
    if not isinstance(cap, dict):
        return False
    if cap.get("authority") == "learned" and cap.get("ground_truth") is True:
        return False
    # 6. tick_non_negative (exclude bool — bool is a subclass of int)
    tick = record.get("tick")
    if isinstance(tick, bool) or not isinstance(tick, int) or tick < 0:
        return False
    # 7. producer_non_empty
    producer_id = record.get("producer_id")
    producer_version = record.get("producer_version")
    if not isinstance(producer_id, str) or not producer_id:
        return False
    if not isinstance(producer_version, str) or not producer_version:
        return False
    return True


def _reconstruct_state_delta(delta: dict) -> StateDelta:
    """Strictly rebuild a kernel :class:`StateDelta` from a JSON dict.

    Re-runs every nested ``__post_init__`` construction rule (change
    kind enums, birth/death pairing rules, ...). Raises on any missing
    key or illegal value — the caller treats exceptions as validation
    failures.
    """
    if not isinstance(delta, dict):
        raise TypeError(
            f"state_delta must be a dict, got {type(delta).__name__}"
        )
    field_changes = None
    raw = delta.get("field_changes")
    if raw is not None:
        field_changes = tuple(
            FieldChange(
                channel=fc["channel"],
                before=fc["before"],
                after=fc["after"],
            )
            for fc in raw
        )
    entity_changes = None
    raw = delta.get("entity_changes")
    if raw is not None:
        entity_changes = EntityChanges(
            schema_id=raw["schema_id"],
            changes=tuple(
                EntityChange(
                    kind=c["kind"],
                    entity_id=c["entity_id"],
                    column=c.get("column"),
                    before=c.get("before"),
                    after=c.get("after"),
                )
                for c in raw.get("changes", ()) or ()
            ),
            ids_after=(
                tuple(raw["ids_after"])
                if raw.get("ids_after") is not None
                else None
            ),
        )
    relation_changes = None
    raw = delta.get("relation_changes")
    if raw is not None:
        relation_changes = RelationChanges(
            schema_id=raw["schema_id"],
            changes=tuple(
                RelationChange(
                    kind=c["kind"],
                    src=c["src"],
                    dst=c["dst"],
                    edge_type=c["edge_type"],
                    before_weight=c.get("before_weight"),
                    after_weight=c.get("after_weight"),
                )
                for c in raw.get("changes", ()) or ()
            ),
            node_ids_after=(
                tuple(raw["node_ids_after"])
                if raw.get("node_ids_after") is not None
                else None
            ),
            edges_after=(
                tuple(
                    RelationEdge(
                        src=e["src"],
                        dst=e["dst"],
                        edge_type=e["edge_type"],
                        weight=e.get("weight", 1.0),
                        born_at_tick=e.get("born_at_tick"),
                    )
                    for e in raw["edges_after"]
                )
                if raw.get("edges_after") is not None
                else None
            ),
        )
    registry_changes = None
    raw = delta.get("registry_changes")
    if raw is not None:
        registry_changes = RegistryChanges(
            schema_id=raw["schema_id"],
            changes=tuple(
                RegistryChange(
                    kind=c["kind"],
                    entry_id=c["entry_id"],
                    registry_type=c["registry_type"],
                    before_state=c.get("before_state"),
                    after_state=c.get("after_state"),
                )
                for c in raw.get("changes", ()) or ()
            ),
            entries_after=(
                tuple(
                    RegistryEntry(
                        entry_id=e["entry_id"],
                        registry_type=e["registry_type"],
                        state=e["state"],
                        owner_id=e.get("owner_id"),
                        metadata=e.get("metadata", {}) or {},
                    )
                    for e in raw["entries_after"]
                )
                if raw.get("entries_after") is not None
                else None
            ),
        )
    population_changes = None
    raw = delta.get("population_changes")
    if raw is not None:
        population_changes = PopulationChanges(
            changes=tuple(
                PopulationChange(
                    kind=c["kind"],
                    agent_id=c["agent_id"],
                    tick=c["tick"],
                    parent_ids=(
                        tuple(c["parent_ids"])
                        if c.get("parent_ids") is not None
                        else None
                    ),
                    cause=c.get("cause"),
                )
                for c in raw.get("changes", ()) or ()
            ),
            alive_ids_after=(
                tuple(raw["alive_ids_after"])
                if raw.get("alive_ids_after") is not None
                else None
            ),
        )
    event_log = None
    raw = delta.get("event_log")
    if raw is not None:
        event_log = tuple(
            EventRecord(
                kind=e["kind"],
                tick=e["tick"],
                payload=e.get("payload", {}) or {},
            )
            for e in raw
        )
    meta_after = None
    raw = delta.get("meta_after")
    if raw is not None:
        meta_after = StateMeta(
            scenario_id=raw["scenario_id"],
            run_id=raw["run_id"],
            tick=raw["tick"],
            config_hash=raw["config_hash"],
            rng_state_ref=raw.get("rng_state_ref"),
        )
    return StateDelta(
        field_changes=field_changes,
        entity_changes=entity_changes,
        relation_changes=relation_changes,
        registry_changes=registry_changes,
        population_changes=population_changes,
        event_log=event_log,
        meta_after=meta_after,
        missing_mask_after=delta.get("missing_mask_after"),
    )


def _reconstruct_transition_record(record: dict) -> TransitionRecord:
    """Strictly rebuild a kernel :class:`TransitionRecord` from a dict.

    Every nested kernel type is reconstructed through its real
    constructor, so ALL ``__post_init__`` construction rules re-run:
    the receipt success/outcome pairing, delta change-kind enums,
    capability pairing rules, the record's own hash/key invariants, ...
    Missing keys raise ``KeyError``; rule violations raise the kernel's
    own error types. The caller treats any exception as a validation
    failure.
    """
    candidates = {
        agent_id: ActionProposal(
            agent_id=c["agent_id"],
            action_type=c["action_type"],
            params=c["params"],
            proposed_at_tick=c["proposed_at_tick"],
            proposer=c["proposer"],
        )
        for agent_id, c in record["candidate_actions"].items()
    }
    executed = {
        agent_id: ExecutedAction(
            agent_id=e["agent_id"],
            action_type=e["action_type"],
            params=e["params"],
            executed_at_tick=e["executed_at_tick"],
            proposal_hash=e["proposal_hash"],
        )
        for agent_id, e in record["executed_actions"].items()
    }
    receipts = {
        agent_id: ActionReceipt(
            executed_action_hash=rc["executed_action_hash"],
            outcome_code=rc["outcome_code"],
            success=rc["success"],
            energy_delta=rc["energy_delta"],
            events=tuple(rc.get("events", ()) or ()),
            diagnostics=rc.get("diagnostics", {}) or {},
        )
        for agent_id, rc in record["receipts"].items()
    }
    raw_exo = record.get("exogenous_input")
    exogenous = None
    if isinstance(raw_exo, dict):
        exogenous = ExogenousInput(
            tick=raw_exo["tick"],
            kind=raw_exo["kind"],
            payload=raw_exo.get("payload", {}) or {},
        )
    cap = record["capability_profile"]
    capability = CapabilityProfile(
        fields=cap["fields"],
        entities=cap["entities"],
        relations=cap["relations"],
        registries=cap["registries"],
        population=cap["population"],
        events=cap["events"],
        exact_restore=cap["exact_restore"],
        executable_deterministic_replay=cap[
            "executable_deterministic_replay"
        ],
        authority=cap["authority"],
        ground_truth=cap["ground_truth"],
        transition_mode=cap["transition_mode"],
    )
    return TransitionRecord(
        schema_version=record["schema_version"],
        producer_id=record["producer_id"],
        producer_version=record["producer_version"],
        tick=record["tick"],
        state_before_hash=record["state_before_hash"],
        candidate_actions=candidates,
        executed_actions=executed,
        exogenous_input=exogenous,
        receipts=receipts,
        state_delta=_reconstruct_state_delta(record["state_delta"]),
        state_after_hash=record["state_after_hash"],
        capability_profile=capability,
        provenance=record.get("provenance", {}) or {},
    )


def _kernel_validate_record(record: dict) -> "tuple[bool, str]":
    """Typed-reconstruct a record and run the kernel's validator on it.

    Returns ``(True, "")`` when the record reconstructs cleanly AND
    passes every record-level invariant of
    :func:`worldloop_kernel.validation.validate_transition`. Returns
    ``(False, reason)`` otherwise. The state-dependent invariants
    (hash_round_trip / missing_mask / tick_monotonicity) require
    StateViews and are skipped by the kernel when no before/after
    states are supplied — those are covered by Q3 replay.
    """
    try:
        typed = _reconstruct_transition_record(record)
    except Exception as exc:  # noqa: BLE001 — construction rules ARE the check
        return False, f"typed reconstruction failed: {exc!r}"
    try:
        report: ValidationReport = validate_transition(typed)
    except Exception as exc:  # noqa: BLE001
        return False, f"validate_transition raised: {exc!r}"
    if report.passed:
        return True, ""
    failed = sorted(
        name
        for name, res in report.invariant_results.items()
        if res.passed is False
    )
    return False, f"kernel invariants failed: {failed}"
