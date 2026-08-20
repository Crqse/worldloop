"""Property tests for S-08 Policy Pool (per goal attempt 2).

Verifies the six policy classes required by main plan §14.2:
- :class:`RandomPolicy`
- :class:`ScriptedPolicy`
- :class:`FrozenReplayPolicy`
- :class:`LLMPolicyStub`
- :class:`AdversarialPolicy`
- :class:`PlannerPolicyStub`

Also verifies the :class:`PolicyPool` composition layer and the
``policy_version`` / ``inference_config`` provenance fields required
by §14.2 ("数据集必须记录 policy_id、policy_version 和推理配置").
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from worldloop_kernel import (
    ActionProposal,
    ActionSpace,
    ExecutedAction,
    LegalAction,
    ToyWorld,
)

from worldloop_data.policy import (
    AdversarialPolicy,
    FrozenReplayPolicy,
    LLMPolicyStub,
    PlannerPolicyStub,
    Policy,
    PolicyContext,
    PolicyPool,
    RandomPolicy,
    ScriptedPolicy,
)


# ---------------------------------------------------------------------------
# Helpers — build minimal PolicyContext without a real world
# ---------------------------------------------------------------------------


def _make_ctx(
    legal_actions: list[LegalAction],
    *,
    agent_id: str | int = "a1",
    tick: int = 0,
    seed: int = 42,
    is_closed: bool = True,
    world=None,
    state=None,
) -> PolicyContext:
    """Build a :class:`PolicyContext` with a synthetic action space."""
    return PolicyContext(
        world=world,
        agent_id=agent_id,
        state=state,
        action_space=ActionSpace(
            agent_id=agent_id,
            legal_actions=tuple(legal_actions),
            is_closed=is_closed,
        ),
        tick=tick,
        rng=random.Random(seed),
    )


def _la(action_type: str, **params) -> LegalAction:
    """Build a :class:`LegalAction` with the given params."""
    return LegalAction(action_type=action_type, params=params)


# ---------------------------------------------------------------------------
# Protocol conformance — every reference policy exposes the three slots
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "policy_factory",
    [
        lambda: RandomPolicy(),
        lambda: ScriptedPolicy(),
        lambda: ScriptedPolicy(preferred_action_type="move"),
        lambda: FrozenReplayPolicy(actions=()),
        lambda: LLMPolicyStub(),
        lambda: AdversarialPolicy(),
        lambda: PlannerPolicyStub(),
        lambda: PlannerPolicyStub(evaluation_fn=lambda la, s: 0.0),
    ],
    ids=[
        "random",
        "scripted_first",
        "scripted_preferred",
        "frozen_replay_empty",
        "llm_stub",
        "adversarial",
        "planner_no_eval",
        "planner_with_eval",
    ],
)
def test_policy_protocol_attributes(policy_factory):
    """Every reference policy exposes ``policy_id``, ``policy_version``,
    and ``inference_config`` (per §14.2 provenance requirement)."""
    p = policy_factory()
    assert isinstance(p.policy_id, str) and p.policy_id
    assert isinstance(p.policy_version, str) and p.policy_version
    # inference_config MUST be a mapping (Mapping[str, Any]).
    ic = p.inference_config
    assert hasattr(ic, "__getitem__"), f"inference_config not a mapping: {type(ic)}"
    # MappingProxyType is not JSON-serializable directly, but dict(ic) is.
    assert isinstance(dict(ic), dict)


# ---------------------------------------------------------------------------
# RandomPolicy
# ---------------------------------------------------------------------------


def test_random_policy_picks_legal_action():
    """RandomPolicy returns one of the legal actions (closed space)."""
    legal = [_la("move", direction=1), _la("noop")]
    ctx = _make_ctx(legal, seed=0)
    p = RandomPolicy()
    proposal = p.propose(ctx)
    assert proposal is not None
    assert proposal.agent_id == "a1"
    assert proposal.action_type in {"move", "noop"}
    assert proposal.proposer == "random"
    assert proposal.proposed_at_tick == 0


def test_random_policy_deterministic_with_same_rng():
    """Same RNG seed → same proposal sequence (Q3 replay support)."""
    legal = [_la(f"a{i}") for i in range(5)]
    p = RandomPolicy()
    seq1 = [p.propose(_make_ctx(legal, seed=7)).action_type for _ in range(10)]
    seq2 = [p.propose(_make_ctx(legal, seed=7)).action_type for _ in range(10)]
    assert seq1 == seq2


def test_random_policy_empty_closed_returns_none():
    """Empty closed action space → None (skip this tick)."""
    ctx = _make_ctx([], is_closed=True)
    p = RandomPolicy()
    assert p.propose(ctx) is None


def test_random_policy_empty_open_returns_noop():
    """Empty open action space → ``noop`` proposal (world decides)."""
    ctx = _make_ctx([], is_closed=False)
    p = RandomPolicy()
    proposal = p.propose(ctx)
    assert proposal is not None
    assert proposal.action_type == "noop"


# ---------------------------------------------------------------------------
# ScriptedPolicy
# ---------------------------------------------------------------------------


def test_scripted_policy_preferred_hits():
    """ScriptedPolicy picks the preferred action type when available."""
    legal = [_la("move", direction=1), _la("noop")]
    ctx = _make_ctx(legal)
    p = ScriptedPolicy(preferred_action_type="noop")
    proposal = p.propose(ctx)
    assert proposal is not None
    assert proposal.action_type == "noop"
    assert proposal.proposer == "scripted"
    assert p.policy_id == "scripted:noop"
    assert p.inference_config["preferred_action_type"] == "noop"


def test_scripted_policy_preferred_missing_falls_back_to_first():
    """If preferred is not in legal_actions, fall back to first."""
    legal = [_la("move", direction=1), _la("noop")]
    ctx = _make_ctx(legal)
    p = ScriptedPolicy(preferred_action_type="forage")
    proposal = p.propose(ctx)
    assert proposal is not None
    assert proposal.action_type == "move"  # first legal


def test_scripted_policy_first_when_no_preferred():
    """No preferred → first legal action."""
    legal = [_la("move"), _la("noop")]
    ctx = _make_ctx(legal)
    p = ScriptedPolicy()
    proposal = p.propose(ctx)
    assert proposal is not None
    assert proposal.action_type == "move"
    assert p.policy_id == "scripted:first"


def test_scripted_policy_empty_returns_none():
    """Empty action space → None."""
    ctx = _make_ctx([])
    p = ScriptedPolicy()
    assert p.propose(ctx) is None


# ---------------------------------------------------------------------------
# FrozenReplayPolicy
# ---------------------------------------------------------------------------


def test_frozen_replay_returns_recorded_action():
    """FrozenReplayPolicy replays the recorded action at (tick, agent)."""
    executed = ExecutedAction(
        agent_id="a1",
        action_type="move",
        params={"direction": 1},
        executed_at_tick=3,
        proposal_hash="sha256:test",
    )
    p = FrozenReplayPolicy(actions=[executed])
    ctx = _make_ctx([_la("move")], agent_id="a1", tick=3)
    proposal = p.propose(ctx)
    assert proposal is not None
    assert proposal.action_type == "move"
    assert proposal.params == {"direction": 1}
    assert proposal.proposer == "frozen"
    assert p.inference_config["n_replay_actions"] == 1


def test_frozen_replay_returns_none_when_no_match():
    """No recorded action for (tick, agent) → None."""
    p = FrozenReplayPolicy(actions=())
    ctx = _make_ctx([_la("move")], tick=0)
    assert p.propose(ctx) is None


# ---------------------------------------------------------------------------
# LLMPolicyStub
# ---------------------------------------------------------------------------


def test_llm_stub_is_runnable_and_picks_first_legal():
    """LLMPolicyStub.propose is runnable (M5 Gate §15.5 (h)).

    Real LLM integration is out of scope, but the stub MUST be runnable
    end-to-end so Gate §15.5 (h) "LLM policy 可运行" passes. The stub
    picks the first legal action and labels the proposal ``llm_stub``.
    """
    ctx = _make_ctx([_la("move"), _la("rest")])
    p = LLMPolicyStub()
    proposal = p.propose(ctx)
    assert proposal is not None
    assert proposal.action_type == "move"
    assert proposal.proposer == "llm_stub"


def test_llm_stub_returns_none_when_no_legal_actions():
    """LLMPolicyStub returns None on empty action space (skip tick)."""
    ctx = _make_ctx([])
    p = LLMPolicyStub()
    assert p.propose(ctx) is None


def test_llm_stub_inference_config_has_reserved_slots():
    """inference_config exposes reserved LLM slots (all None in stub)."""
    p = LLMPolicyStub()
    assert "model" in p.inference_config
    assert "temperature" in p.inference_config
    assert "max_tokens" in p.inference_config
    assert p.inference_config["model"] is None


# ---------------------------------------------------------------------------
# AdversarialPolicy
# ---------------------------------------------------------------------------


def test_adversarial_picks_rarest_type():
    """When one type is rarer than others, adversarial picks it."""
    legal = [
        _la("move", direction=1),
        _la("move", direction=-1),
        _la("noop"),
    ]
    ctx = _make_ctx(legal)
    p = AdversarialPolicy()
    proposal = p.propose(ctx)
    assert proposal is not None
    # "noop" appears once (rarest), "move" appears twice.
    assert proposal.action_type == "noop"
    assert proposal.proposer == "adversarial"


def test_adversarial_all_unique_picks_last():
    """When all types are unique, adversarial picks the LAST legal action
    (intentional contrast with ScriptedPolicy:first)."""
    legal = [_la("move"), _la("noop"), _la("forage")]
    ctx = _make_ctx(legal)
    p = AdversarialPolicy()
    proposal = p.propose(ctx)
    assert proposal is not None
    assert proposal.action_type == "forage"  # last


def test_adversarial_empty_returns_none():
    """Empty action space → None."""
    ctx = _make_ctx([])
    p = AdversarialPolicy()
    assert p.propose(ctx) is None


# ---------------------------------------------------------------------------
# PlannerPolicyStub
# ---------------------------------------------------------------------------


def test_planner_with_evaluation_fn_picks_highest_score():
    """Injected evaluation_fn determines the chosen action."""
    legal = [_la("move", direction=1), _la("noop"), _la("forage", amount=5)]
    ctx = _make_ctx(legal)

    def score(la: LegalAction, state) -> float:
        if la.action_type == "forage":
            return 10.0
        return 1.0

    p = PlannerPolicyStub(evaluation_fn=score)
    proposal = p.propose(ctx)
    assert proposal is not None
    assert proposal.action_type == "forage"
    assert proposal.proposer == "planner"
    assert p.inference_config["has_evaluation_fn"] is True


def test_planner_fallback_picks_largest_numeric_param():
    """Without evaluation_fn, picks the action with the largest numeric param."""
    legal = [
        _la("move", direction=1),
        _la("forage", amount=99),
        _la("forage", amount=3),
    ]
    ctx = _make_ctx(legal)
    p = PlannerPolicyStub()
    proposal = p.propose(ctx)
    assert proposal is not None
    # "forage" with amount=99 has the largest numeric value.
    assert proposal.action_type == "forage"
    assert proposal.params["amount"] == 99


def test_planner_fallback_no_numeric_picks_first():
    """Without numeric params, planner degrades to first legal action."""
    legal = [_la("move"), _la("noop")]
    ctx = _make_ctx(legal)
    p = PlannerPolicyStub()
    proposal = p.propose(ctx)
    assert proposal is not None
    assert proposal.action_type == "move"  # first


def test_planner_empty_returns_none():
    """Empty action space → None."""
    ctx = _make_ctx([])
    p = PlannerPolicyStub()
    assert p.propose(ctx) is None


# ---------------------------------------------------------------------------
# PolicyPool
# ---------------------------------------------------------------------------


def test_policy_pool_empty_raises():
    """PolicyPool requires at least one Policy."""
    with pytest.raises(ValueError):
        PolicyPool([])


def test_policy_pool_get_by_id():
    """Lookup by policy_id returns the registered policy."""
    p1 = RandomPolicy()
    p2 = ScriptedPolicy()
    pool = PolicyPool([p1, p2])
    assert pool.get_by_id("random") is p1
    assert pool.get_by_id("scripted:first") is p2


def test_policy_pool_unknown_id_raises():
    """Unknown policy_id raises KeyError."""
    pool = PolicyPool([RandomPolicy()])
    with pytest.raises(KeyError):
        pool.get_by_id("nonexistent")


def test_policy_pool_default_policy_is_first():
    """Default policy is the first registered."""
    p1 = ScriptedPolicy(preferred_action_type="noop")
    p2 = RandomPolicy()
    pool = PolicyPool([p1, p2])
    assert pool.default_policy() is p1
    assert pool.policy_id == "scripted:noop"


def test_policy_pool_policy_version_delegates_to_default():
    """pool.policy_version mirrors the default member's version."""
    p1 = RandomPolicy()
    pool = PolicyPool([p1])
    assert pool.policy_version == p1.policy_version == "0.1.0"


def test_policy_pool_inference_config_delegates_to_default():
    """pool.inference_config mirrors the default member's config."""
    p1 = ScriptedPolicy(preferred_action_type="move")
    pool = PolicyPool([p1])
    assert dict(pool.inference_config) == dict(p1.inference_config)


def test_policy_pool_rng_for_is_per_policy():
    """Each registered policy gets its own RNG instance."""
    pool = PolicyPool([RandomPolicy(), ScriptedPolicy()])
    r1 = pool.rng_for("random")
    r2 = pool.rng_for("scripted:first")
    assert r1 is not r2


def test_policy_pool_rng_for_unknown_raises():
    """rng_for with unknown policy_id raises KeyError."""
    pool = PolicyPool([RandomPolicy()])
    with pytest.raises(KeyError):
        pool.rng_for("nonexistent")


# ---------------------------------------------------------------------------
# Provenance augmentation — rollout records policy_version + inference_config
# ---------------------------------------------------------------------------


def test_rollout_provenance_records_policy_version(tmp_path):
    """run_rollout augments each transition's provenance with
    ``policy_id``, ``policy_version``, ``inference_config``, and
    ``episode_id`` (per main plan §14.2)."""
    from worldloop_data.config import RolloutConfig
    from worldloop_data.coverage import UniformCoverageScheduler
    from worldloop_data.rollout import run_rollout

    world = ToyWorld()
    pool = PolicyPool([RandomPolicy()])
    cov = UniformCoverageScheduler()

    result = run_rollout(
        world=world,
        seed=42,
        episode_id="test_provenance",
        output_dir=tmp_path / "provenance",
        policy_pool=pool,
        coverage=cov,
        config=RolloutConfig(num_ticks=3, record=True),
    )

    assert result.tick_count == 3
    assert result.manifest is not None
    assert result.manifest.record_count == 3

    # Read every transition JSON and verify provenance fields.
    # Recorder writes ``t{tick:010d}.json`` directly to output_dir
    # (no ``transitions/`` subdirectory in the on-disk layout).
    transitions_dir = tmp_path / "provenance"
    json_files = sorted(transitions_dir.glob("t*.json"))
    assert len(json_files) == 3, (
        f"expected 3 transition files, found {len(json_files)}: "
        f"{[p.name for p in json_files]}"
    )

    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            rec = json.load(f)
        prov = rec.get("provenance", {})
        assert "policy_id" in prov, f"missing policy_id in {jf.name}"
        assert "policy_version" in prov, f"missing policy_version in {jf.name}"
        assert "inference_config" in prov, f"missing inference_config in {jf.name}"
        assert "episode_id" in prov, f"missing episode_id in {jf.name}"
        assert prov["policy_id"] == "random"
        assert prov["policy_version"] == "0.1.0"
        assert prov["episode_id"] == "test_provenance"
        assert isinstance(prov["inference_config"], dict)


# ---------------------------------------------------------------------------
# Q9 utility contrast — adversarial vs scripted produce different trajectories
# ---------------------------------------------------------------------------


def test_q9_utility_adversarial_vs_scripted_contrast(tmp_path):
    """Q9 utility smoke: adversarial and scripted policies produce
    visibly different action sequences on a multi-action world.

    This does NOT require a positive result (per goal OUT_OF_SCOPE §6
    and Q9 spec "不要求必须正结果") — it only verifies that the two
    policies produce DIFFERENT action_type sequences, which is the
    minimum signal needed for a baseline contrast report.
    """
    from worldloop_data.config import RolloutConfig
    from worldloop_data.coverage import UniformCoverageScheduler
    from worldloop_data.rollout import run_rollout

    def _run(policy) -> list[str]:
        # Sanitize episode_id: replace characters illegal on Windows
        # (``:`` is reserved). Use ``_`` instead.
        safe_id = policy.policy_id.replace(":", "_")
        world = ToyWorld()
        pool = PolicyPool([policy])
        result = run_rollout(
            world=world,
            seed=42,
            episode_id=f"q9_{safe_id}",
            output_dir=tmp_path / f"q9_{safe_id}",
            policy_pool=pool,
            coverage=UniformCoverageScheduler(),
            config=RolloutConfig(num_ticks=5, record=True),
        )
        actions = []
        # Recorder writes ``t{tick:010d}.json`` directly to output_dir.
        # Transition JSON has ``executed_actions: dict[agent_id, dict]``
        # at the top level (not a single ``executed`` field).
        out_dir = tmp_path / f"q9_{safe_id}"
        for jf in sorted(out_dir.glob("t*.json")):
            with open(jf, "r", encoding="utf-8") as f:
                rec = json.load(f)
            # ToyWorld uses agent_id "a1"; take its executed action.
            executed = rec["executed_actions"]["a1"]
            actions.append(executed["action_type"])
        return actions

    # ToyWorld has only two action types: "move" and "noop". Both are
    # unique (count == 1 each), so AdversarialPolicy picks the LAST
    # ("noop"), while ScriptedPolicy picks the FIRST ("move").
    # Sequences should differ.
    scripted_actions = _run(ScriptedPolicy())
    adversarial_actions = _run(AdversarialPolicy())

    assert len(scripted_actions) == 5
    assert len(adversarial_actions) == 5
    assert scripted_actions != adversarial_actions, (
        f"Q9 contrast failed: scripted={scripted_actions} "
        f"adversarial={adversarial_actions} — policies must produce "
        "different action sequences for baseline contrast."
    )
