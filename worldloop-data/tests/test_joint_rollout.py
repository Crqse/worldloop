"""Phase 5 joint action mode tests for the data pipeline (§12/§13.5).

Optional external-dependency tests (skipped when worldloop_adapters /
pettingzoo / mpe2 are missing — same rule as
``test_external_pipeline_optional.py``).

Evidence mapping to the Phase 5 gates:
- E-G1: run_joint_rollout records executed actions + receipts for ALL
  active agents on every tick, in BOTH pilot envs.
- E-G3: no proposals for vanished agents — the episode stops once the
  env's active set is empty (max_cycles truncation).
- E-G5: joint branching leaves the parent with zero pollution
  (mechanical held-fixed verification: parent_state_restored=True).
- E-G6: same seed + same policy → identical transition digests.
- Q1/Q3: joint records pass traceability and replay bit-identically
  via step_joint.
"""
from __future__ import annotations

import dataclasses
import json

import pytest

pytest.importorskip(
    "worldloop_adapters",
    reason="worldloop-adapters not installed (optional extra 'external')",
)
pytest.importorskip("pettingzoo", reason="pettingzoo not installed")
pytest.importorskip("mpe2", reason="mpe2 not installed")

from worldloop_adapters.scenario_package import (  # noqa: E402
    SimpleSpreadConfig,
    SimpleTagConfig,
    make_simple_spread_package,
    make_simple_tag_package,
)

from worldloop_data.config import CounterfactualConfig, RolloutConfig  # noqa: E402
from worldloop_data.counterfactual import (  # noqa: E402
    JointKernelBranchScheduler,
)
from worldloop_data.policy import PolicyPool, RandomPolicy  # noqa: E402
from worldloop_data.quality import MinimalQualityReporter  # noqa: E402
from worldloop_data.rollout import run_joint_rollout  # noqa: E402

SEED = 123


def _make_package(name: str):
    if name == "spread":
        return make_simple_spread_package(
            SimpleSpreadConfig(n_agents=3, max_cycles=25)
        )
    return make_simple_tag_package(
        SimpleTagConfig(
            num_good=1, num_adversaries=2, num_obstacles=1, max_cycles=25
        )
    )


def _run(
    name: str,
    tmp_path,
    *,
    seed: int = SEED,
    num_ticks: int = 6,
    branch_every: int = 0,
    subdir: str = "ep",
):
    pkg = _make_package(name)
    world = pkg.world_factory(seed)
    scheduler = None
    if branch_every > 0:
        scheduler = JointKernelBranchScheduler(
            config=CounterfactualConfig(
                branch_every_ticks=branch_every,
                branches_per_checkpoint=2,
            )
        )
    result = run_joint_rollout(
        world=world,
        seed=seed,
        episode_id=f"seed{seed}_joint_{name}",
        output_dir=tmp_path / subdir,
        policy_pool=PolicyPool([RandomPolicy()], episode_seed=seed),
        branch_scheduler=scheduler,
        config=RolloutConfig(num_ticks=num_ticks, record=True),
    )
    return pkg, world, scheduler, result


def _load_records(output_dir):
    records = []
    for path in sorted(output_dir.glob("t*.json")):
        with open(path, encoding="utf-8") as f:
            records.append(json.load(f))
    return records


# ---------------------------------------------------------------------------
# E-G1 — joint mode records all active agents (both envs)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["spread", "tag"])
class TestJointRolloutAllAgents:
    def test_every_tick_records_all_active_agents(self, name, tmp_path):
        _, _, _, result = _run(name, tmp_path)
        assert result.tick_count == 6
        assert result.transition_count == 6
        records = _load_records(result.output_dir)
        assert len(records) == 6
        for r in records:
            active = json.loads(r["provenance"]["active_agents_before"])
            assert set(r["executed_actions"].keys()) == set(active)
            assert set(r["receipts"].keys()) == set(active)
            assert set(r["candidate_actions"].keys()) == set(active)
            assert r["provenance"]["execution_mode"] == "joint"

    def test_per_agent_reward_flags_mirrored(self, name, tmp_path):
        # E-G2 (data side): provenance JSON mirrors reconcile with the
        # per-agent receipt diagnostics in the recorded dataset.
        _, _, _, result = _run(name, tmp_path)
        for r in _load_records(result.output_dir):
            rewards = json.loads(r["provenance"]["rewards_by_agent"])
            terms = json.loads(r["provenance"]["terminations_by_agent"])
            truncs = json.loads(r["provenance"]["truncations_by_agent"])
            for agent, receipt in r["receipts"].items():
                diag = receipt["diagnostics"]
                assert rewards[agent] == pytest.approx(diag["reward"])
                assert terms[agent] == diag["info"]["termination"]
                assert truncs[agent] == diag["info"]["truncation"]


# ---------------------------------------------------------------------------
# E-G3 — vanished agents get no proposals; episode stops cleanly
# ---------------------------------------------------------------------------


class TestAgentVanishStopsProposals:
    def test_rollout_stops_after_truncation(self, tmp_path):
        # max_cycles=25 truncates every agent at the 25th env step and
        # empties the env's active set; asking for more ticks must stop
        # the loop there — no ghost steps, no proposals for vanished
        # agents.
        _, world, _, result = _run("spread", tmp_path, num_ticks=40)
        assert result.tick_count == 25
        assert result.transition_count == 25
        assert world.active_agents() == []
        records = _load_records(result.output_dir)
        assert len(records) == 25
        last = records[-1]
        truncs = json.loads(last["provenance"]["truncations_by_agent"])
        assert all(truncs.values())
        assert json.loads(last["provenance"]["active_agents_after"]) == []
        # Every record covers the full active set of its tick.
        for r in records:
            active = json.loads(r["provenance"]["active_agents_before"])
            assert active
            assert set(r["executed_actions"].keys()) == set(active)


# ---------------------------------------------------------------------------
# Joint counterfactual — held-fixed (E-G5) + focal variation
# ---------------------------------------------------------------------------


class TestJointBranching:
    def test_branches_fire_and_parent_unpolluted(self, tmp_path):
        _, _, scheduler, result = _run(
            "spread", tmp_path, num_ticks=6, branch_every=2
        )
        assert result.branch_count > 0
        assert result.branch_fail_closed_count == 0
        summary = scheduler.branch_summary()
        assert summary["mode"] == "joint_kernel_branch"
        assert summary["branch_count"] == result.branch_count
        # E-G5: every fork group mechanically verified parent-restored.
        verifications = summary["held_fixed_verification"]
        assert verifications
        for v in verifications:
            assert v["parent_state_restored"] is True
            assert v["all_restoration_ok"] is True
            # Multi-agent joint: non-focal actions were fingerprinted.
            assert v["non_focal_actions_hash"].startswith("sha256:")

    def test_focal_varies_non_focal_held_fixed(self, tmp_path):
        pkg = _make_package("spread")
        world = pkg.world_factory(SEED)
        world.reset(seed=SEED)
        from worldloop_kernel import ActionProposal, JointAction

        active = tuple(world.active_agents())
        proposals = {
            a: ActionProposal(
                agent_id=a,
                action_type="move",
                params={"discrete_action": 1},
                proposed_at_tick=2,
                proposer="test",
            )
            for a in active
        }
        baseline, _ = world.validate_joint_action(
            JointAction(
                tick=2,
                active_agents=active,
                proposals_by_agent=proposals,
                missing_agent_policy="stay",
            )
        )
        scheduler = JointKernelBranchScheduler(
            config=CounterfactualConfig(
                branch_every_ticks=2, branches_per_checkpoint=3
            )
        )
        specs = scheduler.schedule_joint_branches(
            checkpoint=world.checkpoint(),
            baseline_joint=baseline,
            world=world,
            tick=2,
        )
        assert len(specs) == 3
        focal = baseline.active_agents[0]
        for spec in specs:
            alt = spec.alternative_joint
            assert spec.focal_agent_id == focal
            # Focal action differs from baseline.
            assert (
                alt.executed_by_agent[focal].params
                != baseline.executed_by_agent[focal].params
            )
            # Non-focal actions are byte-identical to the baseline.
            for agent in baseline.active_agents:
                if agent == focal:
                    continue
                assert (
                    alt.executed_by_agent[agent]
                    == baseline.executed_by_agent[agent]
                )

    def test_branch_final_hashes_diverge_from_baseline(self, tmp_path):
        # Executing branches must produce state hashes that differ from
        # the parent's baseline step (real counterfactual variation).
        pkg = _make_package("spread")
        world = pkg.world_factory(SEED)
        world.reset(seed=SEED)
        from worldloop_kernel import ActionProposal, JointAction, hash_state

        active = tuple(world.active_agents())
        proposals = {
            a: ActionProposal(
                agent_id=a,
                action_type="move",
                params={"discrete_action": 1},
                proposed_at_tick=2,
                proposer="test",
            )
            for a in active
        }
        baseline, _ = world.validate_joint_action(
            JointAction(
                tick=2,
                active_agents=active,
                proposals_by_agent=proposals,
                missing_agent_policy="stay",
            )
        )
        scheduler = JointKernelBranchScheduler(
            config=CounterfactualConfig(
                branch_every_ticks=2, branches_per_checkpoint=2
            )
        )
        checkpoint = world.checkpoint()
        specs = scheduler.schedule_joint_branches(
            checkpoint=checkpoint,
            baseline_joint=baseline,
            world=world,
            tick=2,
        )
        parent_hash_before = hash_state(world.observe())
        results = scheduler.execute_joint_branches(
            world=world, checkpoint=checkpoint, specs=specs
        )
        assert len(results) == 2
        for r in results:
            assert r.error is None
            assert r.restoration_ok
            assert r.final_state_hash is not None
        # Parent untouched after branching (E-G5).
        assert hash_state(world.observe()) == parent_hash_before
        # Now run the baseline joint on the parent: branch hashes must
        # differ from the baseline outcome (the focal action changed).
        baseline_record = world.step_joint(baseline)
        for r in results:
            assert r.final_state_hash != baseline_record.state_after_hash


# ---------------------------------------------------------------------------
# E-G6 — same seed reruns produce identical digests
# ---------------------------------------------------------------------------


class TestDeterministicRerun:
    @pytest.mark.parametrize("name", ["spread", "tag"])
    def test_same_seed_same_transition_hashes(self, name, tmp_path):
        _, _, _, r1 = _run(name, tmp_path, subdir="run_a")
        _, _, _, r2 = _run(name, tmp_path, subdir="run_b")
        recs1 = _load_records(r1.output_dir)
        recs2 = _load_records(r2.output_dir)
        assert [r["state_after_hash"] for r in recs1] == [
            r["state_after_hash"] for r in recs2
        ]
        assert [r["executed_actions"] for r in recs1] == [
            r["executed_actions"] for r in recs2
        ]


# ---------------------------------------------------------------------------
# Q1 traceability + Q3 replay on joint records
# ---------------------------------------------------------------------------


class TestJointQualityChecks:
    def test_q7_requires_and_accepts_non_focal_action_fingerprints(
        self, tmp_path
    ):
        _, _, scheduler, _ = _run(
            "spread", tmp_path, num_ticks=6, branch_every=2
        )
        reporter = MinimalQualityReporter()
        item = reporter._check_q7_counterfactual(scheduler)
        assert item.status == "pass", item.evidence
        assert "scope: joint_kernel_branch" in item.evidence
        assert "non_focal_actions_consistent: 2/2" in item.evidence

    def test_q1_traceability_passes_on_joint_records(self, tmp_path):
        _, _, _, result = _run("spread", tmp_path)
        records = _load_records(result.output_dir)
        reporter = MinimalQualityReporter()
        item = reporter._check_q1_traceability(records)
        assert item.status == "pass", item.evidence

    @pytest.mark.parametrize("name", ["spread", "tag"])
    def test_q3_replay_joint_records_bit_identical(self, name, tmp_path):
        pkg, _, _, result = _run(name, tmp_path)
        records = _load_records(result.output_dir)
        # Fresh world for replay (exact_restore=True per allowlist).
        replay_world = pkg.world_factory(SEED)
        reporter = MinimalQualityReporter()
        item = reporter._check_q3_replay(records, replay_world)
        assert item.status == "pass", item.evidence
