"""S-08 Policy Pool — behavior diversity for trajectory production.

Provides the :class:`Policy` :class:`typing.Protocol` and a small set of
reference policies. The pool is intentionally open: users register
``Policy`` instances by passing them into :class:`PolicyPool`; the pool
does NOT enumerate a closed set of "allowed" policies (per project rule
"反对硬编码约束限制功能").

Reference policies (M4 stub):
- :class:`RandomPolicy` — uniform random over ``ActionSpace.legal_actions``.
- :class:`ScriptedPolicy` — fixed script (default: first legal action).
- :class:`FrozenReplayPolicy` — replay a pre-recorded action sequence.
- :class:`LLMPolicyStub` — placeholder; large-scale LLM data production
  is out of scope for M4 (per goal OUT_OF_SCOPE §4).

Design rules (per main plan §14.2 and ADR §3):
- A policy NEVER calls ``world.step`` directly. It returns an
  :class:`ActionProposal`; the world's ``validate_action`` and ``step``
  own execution. This enforces the project rule "LLM 只能生成候选行动，
  不能直接改世界、判定涌现、决定死亡或出生" — extended here to all
  policies, not just LLM.
- A policy MAY decline to propose by returning ``None`` (e.g., no legal
  actions, or the policy has nothing to add this tick). The rollout
  orchestrator treats ``None`` as "skip this agent this tick".
- ``policy_id`` is a stable identifier recorded into the transition's
  provenance for Q4 (provenance completeness) and Q9 (utility: per-policy
  baseline comparison).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence, TYPE_CHECKING

from worldloop_kernel import (
    ActionProposal,
    ActionSpace,
    ExecutedAction,
    LegalAction,
    StateView,
    WorldProtocol,
)
from worldloop_kernel.canonical import hash_state

from worldloop_data.config import PolicyPoolConfig
from worldloop_data.rng_seeds import (
    PROTOCOL_HASH_DEFAULT,
    derive_per_episode_seed,
)

if TYPE_CHECKING:
    pass

__all__ = [
    "Policy",
    "PolicyContext",
    "PolicyPool",
    "RandomPolicy",
    "ScriptedPolicy",
    "FrozenReplayPolicy",
    "LLMPolicyStub",
    "AdversarialPolicy",
    "PlannerPolicyStub",
]


# ---------------------------------------------------------------------------
# PolicyContext — everything a policy needs to make a proposal
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyContext:
    """Per-call context handed to :meth:`Policy.propose`.

    Attributes
    ----------
    world:
        The world being driven. The policy MAY call ``world.observe()``
        or ``world.legal_actions(agent_id)`` if it needs fresh data, but
        it MUST NOT call ``world.step`` or ``world.validate_action``.
    agent_id:
        ID of the agent whose action is being proposed.
    state:
        Current :class:`StateView` (already observed by the rollout
        orchestrator). Provided as a convenience; policies that need
        finer-grained or fresher state MAY call ``world.observe()`` again.
    action_space:
        Legal action space for ``agent_id`` at this tick.
    tick:
        Current world tick.
    rng:
        Per-policy RNG instance. Policies that need stochasticity SHOULD
        use this RNG (not ``random`` module global state) so seeds are
        reproducible.
    """

    world: WorldProtocol
    agent_id: str | int
    state: StateView
    action_space: ActionSpace
    tick: int
    rng: random.Random


# ---------------------------------------------------------------------------
# Policy Protocol
# ---------------------------------------------------------------------------


class Policy(Protocol):
    """Behavior policy — produces :class:`ActionProposal` instances.

    Implementations may be reflex (random, scripted), learned (model-based,
    RL-trained), or LLM-driven. The pool does not discriminate by source.

    Attributes
    ----------
    policy_id:
        Stable identifier recorded into transition provenance. Two
        distinct policy instances MUST have distinct ``policy_id`` values
        if their behavior differs (otherwise Q9 utility comparison is
        meaningless).
    policy_version:
        Version string of the policy implementation. Recorded into
        transition provenance for Q4 (provenance completeness) so
        different versions of the same ``policy_id`` are distinguishable
        in the dataset. Defaults to ``"0.1.0"`` for all M4 stubs.
    inference_config:
        Read-only mapping describing the policy's inference-time
        configuration (e.g., ``{"temperature": 0.0}`` for an LLM policy,
        ``{"seed_offset": 0}`` for a deterministic policy). Recorded
        into transition provenance for Q4. Defaults to an empty mapping
        for policies that have no inference-time knobs.
    """

    policy_id: str
    policy_version: str
    inference_config: Mapping[str, Any]

    def propose(self, ctx: PolicyContext) -> ActionProposal | None:
        """Produce one :class:`ActionProposal` for ``ctx.agent_id``.

        Returns ``None`` to decline proposing (the agent is skipped this
        tick). Otherwise returns a fully-formed proposal; the rollout
        orchestrator passes it to ``world.validate_action``.
        """
        ...


# ---------------------------------------------------------------------------
# Reference policies
# ---------------------------------------------------------------------------


class RandomPolicy:
    """Uniform random over :attr:`ActionSpace.legal_actions`.

    If the action space is closed and has legal actions, picks one
    uniformly at random. If the action space is empty, returns ``None``
    (skip). If the action space is open (``is_closed=False``), returns a
    minimal ``"noop"`` proposal — the world decides whether to accept.

    Attributes
    ----------
    policy_id:
        Always ``"random"``.
    policy_version:
        Always ``"0.1.0"``.
    inference_config:
        Empty mapping — random policy has no inference-time knobs.
    """

    policy_id: str = "random"
    policy_version: str = "0.1.0"
    inference_config: Mapping[str, Any] = MappingProxyType({})

    def propose(self, ctx: PolicyContext) -> ActionProposal | None:
        if not ctx.action_space.legal_actions:
            if ctx.action_space.is_closed:
                return None
            # Open action space — emit a noop and let the world decide.
            return ActionProposal(
                agent_id=ctx.agent_id,
                action_type="noop",
                params={},
                proposed_at_tick=ctx.tick,
                proposer="random",
            )
        choice = ctx.rng.choice(ctx.action_space.legal_actions)
        return ActionProposal(
            agent_id=ctx.agent_id,
            action_type=choice.action_type,
            params=dict(choice.params),
            proposed_at_tick=ctx.tick,
            proposer="random",
        )


class ScriptedPolicy:
    """Fixed script — picks the first legal action, or a named action.

    If ``preferred_action_type`` is set and appears in
    :attr:`ActionSpace.legal_actions`, that one is picked. Otherwise the
    first legal action is picked. If no legal actions, returns ``None``.

    Attributes
    ----------
    policy_id:
        ``"scripted:<preferred>"`` if ``preferred_action_type`` is set,
        else ``"scripted:first"``.
    policy_version:
        Always ``"0.1.0"``.
    inference_config:
        Read-only mapping with key ``"preferred_action_type"`` recording
        the chosen action type (or ``None``).
    """

    policy_version: str = "0.1.0"

    def __init__(self, *, preferred_action_type: str | None = None) -> None:
        self._preferred = preferred_action_type
        self.policy_id = (
            f"scripted:{preferred_action_type}"
            if preferred_action_type
            else "scripted:first"
        )
        self.inference_config: Mapping[str, Any] = MappingProxyType(
            {"preferred_action_type": preferred_action_type}
        )

    def propose(self, ctx: PolicyContext) -> ActionProposal | None:
        if not ctx.action_space.legal_actions:
            return None
        if self._preferred:
            for la in ctx.action_space.legal_actions:
                if la.action_type == self._preferred:
                    return ActionProposal(
                        agent_id=ctx.agent_id,
                        action_type=la.action_type,
                        params=dict(la.params),
                        proposed_at_tick=ctx.tick,
                        proposer="scripted",
                    )
        # Fallback: first legal action.
        la = ctx.action_space.legal_actions[0]
        return ActionProposal(
            agent_id=ctx.agent_id,
            action_type=la.action_type,
            params=dict(la.params),
            proposed_at_tick=ctx.tick,
            proposer="scripted",
        )


class FrozenReplayPolicy:
    """Replay a pre-recorded sequence of :class:`ExecutedAction`.

    Used for Q3 (replay consistency) and Q9 (frozen-replay baseline
    comparison). The policy returns proposals that reconstruct the
    recorded actions; the world re-validates and re-executes them.

    Attributes
    ----------
    policy_id:
        Always ``"frozen_replay"``.
    policy_version:
        Always ``"0.1.0"``.
    inference_config:
        Read-only mapping with key ``"n_replay_actions"`` recording the
        length of the replay buffer at construction time.
    """

    policy_id: str = "frozen_replay"
    policy_version: str = "0.1.0"

    def __init__(self, actions: Sequence[ExecutedAction]) -> None:
        # Index by (tick, agent_id) for fast lookup.
        self._actions: dict[tuple[int, str | int], ExecutedAction] = {}
        for a in actions:
            self._actions[(a.executed_at_tick, a.agent_id)] = a
        self.inference_config: Mapping[str, Any] = MappingProxyType(
            {"n_replay_actions": len(self._actions)}
        )

    def propose(self, ctx: PolicyContext) -> ActionProposal | None:
        key = (ctx.tick, ctx.agent_id)
        executed = self._actions.get(key)
        if executed is None:
            return None
        return ActionProposal(
            agent_id=ctx.agent_id,
            action_type=executed.action_type,
            params=dict(executed.params),
            proposed_at_tick=ctx.tick,
            proposer="frozen",
        )


class LLMPolicyStub:
    """Runnable mock of an LLM-driven policy.

    M5 Gate §15.5 (h) requires "a pure-rule policy AND an LLM policy
    both runnable". Real LLM integration is out of scope (per goal
    OUT_OF_SCOPE §4 — no online RLVR, no real LLM client). This stub
    is **runnable**: it produces a deterministic mock proposal by
    picking the first legal action, simulating "LLM chose this action
    after reasoning". The ``inference_config`` records the reserved
    LLM slots (model/temperature/max_tokens) so downstream provenance
    tracks that this is a stub, not a real LLM call.

    Design rules:
    - NEVER raise ``NotImplementedError`` — the stub must be runnable
      end-to-end so Gate §15.5 (h) "LLM policy 可运行" passes.
    - The proposal is labelled ``proposer="llm_stub"`` so Q4 provenance
      distinguishes stub proposals from real LLM calls in future phases.
    - When no legal action exists, returns ``None`` (skip tick) — same
      contract as :class:`RandomPolicy`.

    Attributes
    ----------
    policy_id:
        Always ``"llm_stub"``.
    policy_version:
        Always ``"0.1.0"``.
    inference_config:
        Read-only mapping recording the reserved LLM config slots
        (``model``, ``temperature``, ``max_tokens``) — all unset, marking
        this as a stub. A real LLM policy would populate these.
    """

    policy_id: str = "llm_stub"
    policy_version: str = "0.1.0"
    inference_config: Mapping[str, Any] = MappingProxyType(
        {"model": None, "temperature": None, "max_tokens": None}
    )

    def propose(self, ctx: PolicyContext) -> ActionProposal | None:
        legal = ctx.action_space.legal_actions
        if not legal:
            return None
        # Mock "LLM reasoning": pick the first legal action. A real LLM
        # would score / rank candidates via a model call; the stub just
        # takes the first one so the pipeline can run end-to-end.
        la = legal[0]
        return ActionProposal(
            agent_id=ctx.agent_id,
            action_type=la.action_type,
            params=dict(la.params),
            proposed_at_tick=ctx.tick,
            proposer="llm_stub",
        )


# ---------------------------------------------------------------------------
# AdversarialPolicy — least-common action selection
# ---------------------------------------------------------------------------


class AdversarialPolicy:
    """Adversarial policy — picks the least-common ``action_type``.

    For each tick, counts how many times each ``action_type`` appears in
    :attr:`ActionSpace.legal_actions` and picks the first variant of the
    rarest type. Ties are broken by first occurrence. If every legal
    action has a distinct type (all counts == 1), picks the LAST legal
    action — intentional contrast with :class:`ScriptedPolicy` (which
    picks first) so the two produce visibly different trajectories for
    Q9 (utility baseline comparison).

    Used for Q9 utility comparison: demonstrates that adversarial action
    selection produces measurably different trajectories from random /
    scripted baselines, supporting the "strong baseline contrast"
    requirement (negative results allowed — only the contrast must
    exist).

    Attributes
    ----------
    policy_id:
        Always ``"adversarial"``.
    policy_version:
        Always ``"0.1.0"``.
    inference_config:
        Empty mapping — adversarial policy has no inference-time knobs.
    """

    policy_id: str = "adversarial"
    policy_version: str = "0.1.0"
    inference_config: Mapping[str, Any] = MappingProxyType({})

    def propose(self, ctx: PolicyContext) -> ActionProposal | None:
        legal = ctx.action_space.legal_actions
        if not legal:
            return None
        # Count occurrences of each action_type.
        type_counts: dict[str, int] = {}
        for la in legal:
            type_counts[la.action_type] = type_counts.get(la.action_type, 0) + 1
        min_count = min(type_counts.values())
        # If all types are unique, pick the LAST legal action (contrast
        # with ScriptedPolicy:first).
        if min_count == 1 and len(type_counts) == len(legal):
            la = legal[-1]
        else:
            # Pick the first variant of the rarest type.
            la = next(l for l in legal if l.action_type in type_counts and type_counts[l.action_type] == min_count)
        return ActionProposal(
            agent_id=ctx.agent_id,
            action_type=la.action_type,
            params=dict(la.params),
            proposed_at_tick=ctx.tick,
            proposer="adversarial",
        )


# ---------------------------------------------------------------------------
# PlannerPolicyStub — heuristic planner / world-model placeholder
# ---------------------------------------------------------------------------


class PlannerPolicyStub:
    """Planner / world-model policy stub.

    A heuristic planner that scores each legal action via an injected
    ``evaluation_fn`` and picks the highest-scoring one. If no evaluation
    function is provided, falls back to picking the legal action with
    the largest numeric value in its ``params`` (a simple "greedy on
    magnitude" heuristic). If no legal action has numeric params, picks
    the first legal action (degrades to :class:`ScriptedPolicy` first).

    M4 does NOT exercise real model-based planning (per goal
    OUT_OF_SCOPE §5 — no LLM training / fine-tuning / online RLVR). This
    stub provides the interface so downstream code can reference
    ``PlannerPolicyStub`` as a marker. Real planner integration (e.g.,
    a learned world model) is deferred to M6+.

    Attributes
    ----------
    policy_id:
        Always ``"planner_stub"``.
    policy_version:
        Always ``"0.1.0"``.
    inference_config:
        Read-only mapping recording whether an ``evaluation_fn`` was
        injected (``{"has_evaluation_fn": bool}``).
    """

    policy_id: str = "planner_stub"
    policy_version: str = "0.1.0"

    def __init__(
        self,
        *,
        evaluation_fn: Callable[[LegalAction, StateView], float] | None = None,
    ) -> None:
        self._evaluation_fn = evaluation_fn
        self.inference_config: Mapping[str, Any] = MappingProxyType(
            {"has_evaluation_fn": evaluation_fn is not None}
        )

    def propose(self, ctx: PolicyContext) -> ActionProposal | None:
        legal = ctx.action_space.legal_actions
        if not legal:
            return None

        if self._evaluation_fn is not None:
            best = max(legal, key=lambda la: self._evaluation_fn(la, ctx.state))
        else:
            # Fallback heuristic: pick the legal action whose params
            # contain the largest numeric value. Booleans are excluded
            # because bool is a subclass of int in Python.
            def _score(la: LegalAction) -> float:
                nums = [
                    v
                    for v in la.params.values()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                ]
                return max(nums) if nums else 0.0

            best = max(legal, key=_score)

        return ActionProposal(
            agent_id=ctx.agent_id,
            action_type=best.action_type,
            params=dict(best.params),
            proposed_at_tick=ctx.tick,
            proposer="planner",
        )


# ---------------------------------------------------------------------------
# PolicyPool — composition layer
# ---------------------------------------------------------------------------


class PolicyPool:
    """Composes multiple :class:`Policy` instances.

    The pool itself implements the :class:`Policy` Protocol by delegating
    to a chosen member policy. The choice is made by an external
    :class:`~worldloop_data.coverage.CoverageScheduler` via
    :meth:`select_policy`; the pool exposes the registered policies and
    a default fallback.

    Attributes
    ----------
    policies:
        Tuple of registered policies, in registration order.
    config:
        The :class:`PolicyPoolConfig` governing default proposer label
        and base RNG seed.
    """

    def __init__(
        self,
        policies: Sequence[Policy],
        *,
        config: PolicyPoolConfig | None = None,
        protocol_hash: str = PROTOCOL_HASH_DEFAULT,
        episode_seed: int | None = None,
    ) -> None:
        if not policies:
            raise ValueError("PolicyPool requires at least one Policy")
        self._policies: tuple[Policy, ...] = tuple(policies)
        self._policy_ids: tuple[str, ...] = tuple(p.policy_id for p in self._policies)
        self.config = config or PolicyPoolConfig()
        self._protocol_hash = str(protocol_hash)
        self._episode_seed: int | None = episode_seed
        self._rng = random.Random(self.config.seed)
        # Per-policy RNG instances.
        #
        # Phase 4 §8.4: when ``episode_seed`` is provided, per-policy RNG
        # is derived via ``derive_per_episode_seed(protocol_hash,
        # episode_seed, f"policy:{policy_id}")`` — this guarantees
        # per-episode isolation (different episodes get independent
        # streams) and cross-process stability (SHA-256 canonical,
        # not Python ``hash()``). Callers SHOULD call
        # :meth:`begin_episode` at the start of each episode to
        # re-derive.
        #
        # When ``episode_seed`` is ``None`` (legacy callers), we keep
        # the pre-Phase-4 behavior of ``random.Random(config.seed + i)``
        # — deterministic within a process but NOT isolated per
        # episode. New code should pass ``episode_seed``.
        self._policy_rngs: dict[str, random.Random] = self._derive_policy_rngs()
        # Default fallback policy: first registered.
        self._default_idx = 0

    def _derive_policy_rngs(self) -> dict[str, random.Random]:
        """Derive per-policy RNG instances.

        - If ``self._episode_seed`` is set: use
          :func:`derive_per_episode_seed` (Phase 4 §8.4 per-episode
          isolation + cross-process stability).
        - Else: fall back to legacy ``random.Random(config.seed + i)``
          (deterministic in-process, but shared across episodes).
        """
        if self._episode_seed is None:
            # Legacy path (pre-Phase-4).
            return {
                pid: random.Random(self.config.seed + i)
                for i, pid in enumerate(self._policy_ids)
            }
        return {
            pid: random.Random(
                derive_per_episode_seed(
                    protocol_hash=self._protocol_hash,
                    episode_seed=self._episode_seed,
                    scope_id=f"policy:{pid}",
                )
            )
            for pid in self._policy_ids
        }

    def begin_episode(self, episode_seed: int) -> None:
        """Re-derive per-policy RNG for a new episode (Phase 4 §8.4).

        Per-episode isolation: each episode gets fresh RNG streams
        derived from ``(protocol_hash, episode_seed, policy_id)``,
        forbidding the legacy pattern of sharing a single mutable RNG
        across episodes.

        Args:
            episode_seed: The episode's RNG seed (typically the
                outer-loop seed for this episode).
        """
        self._episode_seed = int(episode_seed)
        self._policy_rngs = self._derive_policy_rngs()

    @property
    def policies(self) -> tuple[Policy, ...]:
        return self._policies

    @property
    def policy_ids(self) -> tuple[str, ...]:
        return self._policy_ids

    def rng_for(self, policy_id: str) -> random.Random:
        """Return the per-policy :class:`random.Random` instance."""
        if policy_id not in self._policy_rngs:
            raise KeyError(
                f"unknown policy_id {policy_id!r}; "
                f"registered: {self._policy_ids}"
            )
        return self._policy_rngs[policy_id]

    def get_by_id(self, policy_id: str) -> Policy:
        """Look up a registered policy by ``policy_id``."""
        for p in self._policies:
            if p.policy_id == policy_id:
                return p
        raise KeyError(
            f"unknown policy_id {policy_id!r}; "
            f"registered: {self._policy_ids}"
        )

    def select(self, policy_id: str) -> Policy:
        """Select a member policy by ``policy_id``.

        Used by :class:`~worldloop_data.coverage.CoverageScheduler` to
        drive per-tick policy choice.
        """
        return self.get_by_id(policy_id)

    def default_policy(self) -> Policy:
        """Return the default (first registered) policy."""
        return self._policies[self._default_idx]

    # The pool itself is a Policy: it delegates to the default member.
    @property
    def policy_id(self) -> str:
        return self.default_policy().policy_id

    @property
    def policy_version(self) -> str:
        # Delegate to the default member so the pool satisfies the
        # Policy Protocol. Per-member versions are accessible via
        # ``pool.get_by_id(pid).policy_version``.
        return self.default_policy().policy_version

    @property
    def inference_config(self) -> Mapping[str, Any]:
        # Delegate to the default member. The pool itself adds no
        # inference-time knobs beyond what the default policy has.
        return self.default_policy().inference_config

    def propose(self, ctx: PolicyContext) -> ActionProposal | None:
        # Inject the per-policy RNG into the context.
        pid = self.default_policy().policy_id
        rng = self._policy_rngs[pid]
        new_ctx = PolicyContext(
            world=ctx.world,
            agent_id=ctx.agent_id,
            state=ctx.state,
            action_space=ctx.action_space,
            tick=ctx.tick,
            rng=rng,
        )
        return self.default_policy().propose(new_ctx)
