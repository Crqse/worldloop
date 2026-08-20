"""PettingZooParallelAdapter: PettingZoo Parallel → kernel WorldProtocol (A-01).

The :class:`PettingZooParallelAdapter` wraps a PettingZoo Parallel API
environment as a kernel :class:`WorldProtocol`. The adapter is a thin
type-mapping layer: it does NOT write business rules, does NOT call
LLMs, and does NOT mutate the env outside of the documented
``reset`` / ``step`` / ``restore`` methods.

Architecture (per A-01 spec and main plan §12.2):
- ``reset(seed)`` calls ``env.reset(seed=seed)`` then ``observe``.
- ``observe`` builds a :class:`StateView` via :func:`build_state_view`
  (read-only, idempotent).
- ``legal_actions`` returns the env's discrete action space as a tuple
  of :class:`LegalAction` entries. The action space is ``is_closed=True``
  (PettingZoo envs reject out-of-range actions).
- ``validate_action`` converts the proposal to a PettingZoo discrete
  action int via :func:`proposal_to_pettingzoo_action`, builds an
  :class:`ExecutedAction` (with ``proposal_hash`` from :func:`hash_state`),
  and returns a placeholder receipt. The actual outcome is surfaced in
  ``step``.
- ``step`` looks up the cached discrete action by ``executed_action_hash``,
  builds a per-agent action dict, calls ``env.step(actions_dict)``, reads
  the actual outcome (reward / termination / truncation), builds a
  :class:`TransitionRecord` with ``state_before_hash`` /
  ``state_after_hash`` / ``state_delta=diff_state(before, after)``.
- ``checkpoint`` / ``restore`` delegate to
  :func:`export_checkpoint` / :func:`restore_checkpoint`.

A-01 simplifications (documented for M2 Gate (i) downgrade):
- The adapter assumes PettingZoo Parallel API (not AEC). AEC envs should
  be wrapped via ``pettingzoo.utils.conversions.aec_to_parallel`` first.
- The adapter assumes discrete action spaces (MPE 5-action). Continuous
  Box action spaces are OUT_OF_SCOPE for Attempt 1; they will raise
  ``NotImplementedError`` in ``legal_actions``.
- ``step`` advances the env by ONE Parallel step (all agents act jointly).
  This differs from the Native adapter's per-agent step; the kernel
  caller MUST batch per-agent proposals into a single ``step`` call via
  ``step_batch`` (added in Attempt 2) or call ``step`` with a representative
  action and let the adapter fill in defaults for other agents (Attempt 1
  smoke only — not for production use).
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import replace as _dc_replace
from typing import Any, Mapping

from worldloop_kernel.action import (
    ActionProposal,
    ActionReceipt,
    ExecutedAction,
    ExogenousInput,
    OUTCOME_ILLEGAL_ACTION,
    OUTCOME_OK,
    OUTCOME_UNKNOWN_FAILURE,
)
from worldloop_kernel.canonical import hash_state
from worldloop_kernel.capability import CapabilityProfile
from worldloop_kernel.diff_apply import diff_state
from worldloop_kernel.joint import (
    JointAction,
    JointActionError,
    JointReceipt,
)
from worldloop_kernel.observation import (
    OBSERVATION_SCHEMA_VERSION,
    AgentObservationView,
    FocalAgentAttributes,
    OmissionPolicy,
    PreviousActionSummary,
    VisibleEntity,
)
from worldloop_kernel.protocol import ActionSpace, LegalAction
from worldloop_kernel.state import StateView
from worldloop_kernel.transition import (
    Checkpoint,
    PROTOCOL_SCHEMA_VERSION,
    TransitionRecord,
)

from .action_mapper import (
    PETTINGZOO_DEFAULT_ACTION_TYPE,
    PETTINGZOO_LEGAL_DISCRETE_ACTIONS,
    build_executed_action,
    build_receipt,
    proposal_to_pettingzoo_action,
    reject_proposal,
)
from .capability import (
    PETTINGZOO_WORLD_ID,
    PETTINGZOO_WORLD_VERSION,
    make_pettingzoo_mpe_capability,
)
from .checkpoint_mapper import export_checkpoint, restore_checkpoint
from .state_mapper import build_state_view

logger = logging.getLogger(__name__)

__all__ = [
    "PettingZooParallelAdapter",
    "make_simple_spread_env",
    "make_simple_tag_env",
]


# ---------------------------------------------------------------------------
# PettingZooParallelAdapter
# ---------------------------------------------------------------------------


class PettingZooParallelAdapter:
    """Adapter wrapping a PettingZoo Parallel env as kernel :class:`WorldProtocol`.

    The adapter is constructed with a PettingZoo Parallel env instance
    (which the caller owns and may have pre-configured). The adapter
    does NOT take ownership of the env's lifecycle; the caller is
    responsible for constructing and closing the env.

    Construction is non-stateful: the adapter caches the
    :class:`CapabilityProfile` and a dict of pending actions (between
    ``validate_action`` and ``step``) but does NOT cache env state.
    """

    def __init__(
        self,
        env: Any,
        *,
        env_id: str = "pettingzoo-parallel",
        capability: CapabilityProfile | None = None,
        action_type: str = PETTINGZOO_DEFAULT_ACTION_TYPE,
        legal_discrete_actions: tuple[int, ...] = PETTINGZOO_LEGAL_DISCRETE_ACTIONS,
        run_id: str = "pettingzoo-run",
        config_hash: str = "pettingzoo-default",
    ) -> None:
        """Construct the adapter.

        Parameters
        ----------
        env:
            The PettingZoo Parallel env handle. MUST support
            ``reset(seed=...)`` / ``step(actions_dict)`` / ``observe()``.
        env_id:
            Stable identifier for this env instance.
        capability:
            Optional override for the capability profile. Defaults to
            :func:`make_pettingzoo_mpe_capability`.
        action_type:
            The action_type string used in :class:`ActionProposal` /
            :class:`ExecutedAction`. Defaults to ``"move"``.
        legal_discrete_actions:
            Tuple of legal discrete action ints. Defaults to MPE 5-action.
        run_id:
            Stable run identifier for this trajectory.
        config_hash:
            Stable config hash for this scenario.
        """
        self._env = env
        self._env_id = env_id
        self._cap = capability or make_pettingzoo_mpe_capability()
        self._action_type = action_type
        self._legal_actions = legal_discrete_actions
        self._run_id = run_id
        self._config_hash = config_hash

        # Cached state between validate_action and step.
        self._pending_actions: dict[str, int] = {}  # executed_hash → discrete_action
        self._pending_proposals: dict[str, ActionProposal] = {}
        # Cached joint actions between validate_joint_action and step_joint
        # (joint_hash → per-agent discrete actions dict).
        self._pending_joint: dict[str, dict[str, int]] = {}

        # Cached last obs/infos (updated by reset/step).
        self._last_obs: dict[str, Any] = {}
        self._last_infos: dict[str, Any] = {}

        # Whether reset has been called.
        self._initialized = False

    # ------------------------------------------------------------------
    # WorldProtocol: capabilities
    # ------------------------------------------------------------------

    @property
    def capabilities(self) -> CapabilityProfile:
        """Static capability declaration for this adapter."""
        return self._cap

    # ------------------------------------------------------------------
    # WorldProtocol: reset / observe
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int,
        parameters: Mapping[str, Any] | None = None,
    ) -> StateView:
        """Reset the env and return the initial :class:`StateView`."""
        # PettingZoo Parallel API: reset returns (obs, infos).
        result = self._env.reset(seed=seed)
        if isinstance(result, tuple) and len(result) == 2:
            self._last_obs, self._last_infos = result
        else:
            # Some envs return only obs.
            self._last_obs = result
            self._last_infos = {}
        self._initialized = True
        # Clear pending actions on reset.
        self._pending_actions.clear()
        self._pending_proposals.clear()
        self._pending_joint.clear()
        return self.observe()

    def observe(self) -> StateView:
        """Return the current :class:`StateView` without advancing the env."""
        if not self._initialized:
            raise RuntimeError(
                "PettingZooParallelAdapter.observe() called before reset(); "
                "call reset(seed) first."
            )
        return build_state_view(
            self._env,
            self._last_obs,
            self._last_infos,
            self._cap,
            run_id=self._run_id,
            config_hash=self._config_hash,
        )

    # ------------------------------------------------------------------
    # ObservationProjector: observe_agent (Phase 1 / Beta correction §5.3)
    # ------------------------------------------------------------------

    def observe_agent(
        self,
        agent_id: str | int,
        *,
        state: StateView | None = None,
    ) -> AgentObservationView:
        """Project the per-agent observation for ``agent_id``.

        Implements :class:`worldloop_kernel.observation.ObservationProjector`.
        The projection applies the default PettingZoo MPE visibility policy:

        - Focal agent sees its own position (``obs[:2]``), velocity
          (``obs[2:4]``) and the FULL local observation vector
          (``obs``) — these are ``self_visible`` because the focal
          agent's sensorimotor stream is private to it.
        - Other agents see ONLY the focal agent's POSITION (x, y) —
          velocity and the full observation vector are private.
        - Landmarks are PUBLIC (position visible to all agents) — MPE
          landmarks are environmental features, not private state.
        - Other capability slots (fields / relations / registries /
          population / events) are OMITTED because the adapter declares
          them as ``False`` in :func:`make_pettingzoo_mpe_capability`.

        Counterfactual projection (``state is not None``) is NOT
        supported — it raises ``NotImplementedError`` (matches
        :meth:`legal_actions` behavior).

        Parameters
        ----------
        agent_id:
            ID of the focal agent. MUST be present in ``_last_obs``
            (call :meth:`reset` first).
        state:
            Optional counterfactual state. NOT supported.

        Returns
        -------
        AgentObservationView
            The authorized observation for ``agent_id``. The
            ``omission_policy`` field lists every capability slot the
            adapter declares as ``False``.
        """
        if state is not None:
            raise NotImplementedError(
                "PettingZooParallelAdapter.observe_agent(state=...) is not "
                "supported (counterfactual projection is M4+)."
            )
        if not self._initialized:
            raise RuntimeError(
                "PettingZooParallelAdapter.observe_agent() called before "
                "reset(); call reset(seed) first."
            )
        agent_id_str = str(agent_id)
        if agent_id_str not in self._last_obs:
            raise KeyError(
                f"observe_agent: agent_id {agent_id!r} not in last_obs "
                f"agents {list(self._last_obs.keys())!r}; call reset(seed) "
                f"first."
            )
        focal_obs = self._last_obs[agent_id_str]
        # MPE obs layout: [agent_x, agent_y, agent_vx, agent_vy, ...]
        try:
            focal_pos = tuple(float(x) for x in focal_obs[:2])
        except (TypeError, IndexError, ValueError):
            focal_pos = (0.0, 0.0)
        focal_velocity: tuple[float, ...] = ()
        try:
            if len(focal_obs) >= 4:
                focal_velocity = tuple(float(x) for x in focal_obs[2:4])
        except (TypeError, ValueError):
            focal_velocity = ()
        # Full observation vector — self_visible (private sensorimotor).
        try:
            focal_full_obs = tuple(float(x) for x in focal_obs)
        except (TypeError, ValueError):
            focal_full_obs = ()

        focal_agent = FocalAgentAttributes(
            agent_id=agent_id_str,
            public_attributes={"position": focal_pos},
            self_visible_attributes={
                "velocity": focal_velocity,
                "observation": focal_full_obs,
            },
        )

        # --- Visible entities: other agents (position only) + landmarks ---
        visible_entities: list[VisibleEntity] = []
        for other_id, other_obs in self._last_obs.items():
            if str(other_id) == agent_id_str:
                continue
            try:
                other_pos = tuple(float(x) for x in other_obs[:2])
            except (TypeError, IndexError, ValueError):
                other_pos = (0.0, 0.0)
            visible_entities.append(
                VisibleEntity(
                    entity_id=str(other_id),
                    columns={"position": other_pos, "kind": "agent"},
                )
            )
        # Landmarks — public environmental features.
        unwrapped = getattr(self._env, "unwrapped", self._env)
        world = getattr(unwrapped, "world", None)
        landmarks = (
            getattr(world, "landmarks", None)
            or getattr(unwrapped, "landmarks", None)
            or []
        )
        for i, lm in enumerate(landmarks):
            try:
                pos_arr = lm.state.p_pos
                lm_pos = tuple(float(x) for x in pos_arr)
            except (AttributeError, TypeError, ValueError):
                lm_pos = (0.0, 0.0)
            visible_entities.append(
                VisibleEntity(
                    entity_id=f"landmark_{i}",
                    columns={"position": lm_pos, "kind": "landmark"},
                )
            )

        # --- Legal actions (from existing WorldProtocol method) -------
        action_space = self.legal_actions(agent_id_str)
        legal_actions = action_space.legal_actions

        # --- Omission policy: list every False capability -------------
        unsupported: list[str] = []
        for slot in ("fields", "relations", "registries", "population", "events"):
            if not getattr(self._cap, slot):
                unsupported.append(slot)
        omission = OmissionPolicy(
            omitted_slots=tuple(unsupported),
            reason="capability_unavailable",
            unsupported_capabilities=tuple(unsupported),
        )

        return AgentObservationView(
            schema_version=OBSERVATION_SCHEMA_VERSION,
            scenario_id=self._env_id,
            scenario_version=PETTINGZOO_WORLD_VERSION,
            # Read env step counter WITHOUT advancing it (matches
            # state_mapper._get_env_step behavior; inlined to keep the
            # projector self-contained).
            tick=_read_env_step(self._env),
            focal_agent=focal_agent,
            previous_action=PreviousActionSummary(),
            visible_fields={},
            visible_entities=tuple(visible_entities),
            visible_relations=(),
            visible_events=(),
            legal_actions=legal_actions,
            omission_policy=omission,
        )

    # ------------------------------------------------------------------
    # WorldProtocol: legal_actions
    # ------------------------------------------------------------------

    def legal_actions(
        self,
        agent_id: str | int,
        state: StateView | None = None,
    ) -> ActionSpace:
        """Return the legal action space for ``agent_id``.

        Returns a closed action space (``is_closed=True``) containing
        the env's discrete legal actions. PettingZoo envs reject
        out-of-range actions; the adapter mirrors this by marking the
        space closed.
        """
        if state is not None:
            raise NotImplementedError(
                "PettingZooParallelAdapter.legal_actions(state=...) is not "
                "supported yet (counterfactual queries are M2 follow-up)."
            )
        legal = tuple(
            LegalAction(
                action_type=self._action_type,
                params={"discrete_action": i},
                description=f"discrete action {i}",
            )
            for i in self._legal_actions
        )
        return ActionSpace(
            agent_id=agent_id,
            legal_actions=legal,
            is_closed=True,
        )

    # ------------------------------------------------------------------
    # WorldProtocol: validate_action
    # ------------------------------------------------------------------

    def validate_action(
        self,
        proposal: ActionProposal,
    ) -> tuple[ExecutedAction, ActionReceipt]:
        """Validate a candidate :class:`ActionProposal`.

        Returns an :class:`ExecutedAction` + placeholder :class:`ActionReceipt`.
        The actual outcome (reward / termination) is surfaced in ``step``.

        If the proposal's ``discrete_action`` is outside the legal action
        space, returns a rejection receipt with ``success=False`` and
        ``outcome_code=OUTCOME_ILLEGAL_ACTION``.
        """
        discrete = proposal_to_pettingzoo_action(proposal, self._legal_actions)
        if discrete is None:
            return reject_proposal(proposal, OUTCOME_ILLEGAL_ACTION)

        executed = build_executed_action(proposal)
        # Cache the discrete action for step().
        executed_hash = hash_state(executed)
        self._pending_actions[executed_hash] = discrete
        self._pending_proposals[executed_hash] = proposal

        # Placeholder receipt; actual outcome filled in step().
        receipt = build_receipt(executed, reward=0.0, success=True, outcome_code=OUTCOME_OK)
        return executed, receipt

    # ------------------------------------------------------------------
    # WorldProtocol: step
    # ------------------------------------------------------------------

    def step(
        self,
        action: ExecutedAction,
        exogenous: ExogenousInput | None = None,
    ) -> TransitionRecord:
        """Apply an :class:`ExecutedAction` to the env and return the
        :class:`TransitionRecord`.

        For Attempt 1 smoke: the adapter uses the single executed action
        for ``agent_id`` and fills in STAY (action 0) for all other
        agents. Attempt 2 will add ``step_batch`` for proper joint action
        handling.
        """
        if not self._initialized:
            raise RuntimeError(
                "PettingZooParallelAdapter.step() called before reset(); "
                "call reset(seed) first."
            )

        # Snapshot state_before.
        state_before = self.observe()
        state_before_hash = hash_state(state_before)
        tick_before = int(state_before.meta.tick)

        # Look up cached discrete action.
        executed_hash = hash_state(action)
        discrete = self._pending_actions.pop(executed_hash, None)
        proposal = self._pending_proposals.pop(executed_hash, None)
        if discrete is None:
            # The action was not validated (or was already consumed).
            # Treat as illegal.
            state_after = self.observe()
            state_after_hash = hash_state(state_after)
            state_delta = diff_state(state_before, state_after)
            receipt = build_receipt(
                action,
                reward=0.0,
                success=False,
                outcome_code=OUTCOME_ILLEGAL_ACTION,
                info={"reason": "action not validated or already consumed"},
            )
            return TransitionRecord(
                schema_version=PROTOCOL_SCHEMA_VERSION,
                producer_id=PETTINGZOO_WORLD_ID,
                producer_version=PETTINGZOO_WORLD_VERSION,
                tick=tick_before,
                state_before_hash=state_before_hash,
                candidate_actions={action.agent_id: proposal} if proposal else {},
                executed_actions={action.agent_id: action},
                exogenous_input=exogenous,
                receipts={action.agent_id: receipt},
                state_delta=state_delta,
                state_after_hash=state_after_hash,
                capability_profile=self._cap,
                provenance={
                    "env_id": self._env_id,
                    "joint_step": "false",
                    "execution_mode": "sequential_focal_stay",
                },
            )

        # Build joint action dict: the validated agent gets `discrete`;
        # all other agents get 0 (STAY). This is Attempt 1 smoke behavior;
        # Attempt 2's step_batch will properly handle multi-agent actions.
        actions_dict: dict[str, int] = {}
        for agent_id in self._last_obs.keys():
            if str(agent_id) == str(action.agent_id):
                actions_dict[str(agent_id)] = discrete
            else:
                actions_dict[str(agent_id)] = 0  # STAY

        # Execute the env step.
        try:
            step_result = self._env.step(actions_dict)
            if len(step_result) == 5:
                obs, rewards, terminations, truncations, infos = step_result
            else:
                raise RuntimeError(
                    f"PettingZoo step returned {len(step_result)} values; "
                    f"expected 5 (Parallel API)."
                )
        except Exception as exc:
            logger.exception("PettingZoo env.step raised: %s", exc)
            state_after = self.observe()
            state_after_hash = hash_state(state_after)
            state_delta = diff_state(state_before, state_after)
            receipt = build_receipt(
                action,
                reward=0.0,
                success=False,
                outcome_code=OUTCOME_UNKNOWN_FAILURE,
                info={"exception": str(exc)},
            )
            return TransitionRecord(
                schema_version=PROTOCOL_SCHEMA_VERSION,
                producer_id=PETTINGZOO_WORLD_ID,
                producer_version=PETTINGZOO_WORLD_VERSION,
                tick=tick_before,
                state_before_hash=state_before_hash,
                candidate_actions={action.agent_id: proposal} if proposal else {},
                executed_actions={action.agent_id: action},
                exogenous_input=exogenous,
                receipts={action.agent_id: receipt},
                state_delta=state_delta,
                state_after_hash=state_after_hash,
                capability_profile=self._cap,
                provenance={
                    "env_id": self._env_id,
                    "error": str(exc),
                    "execution_mode": "sequential_focal_stay",
                },
            )

        # Update cached obs/infos.
        self._last_obs = obs
        self._last_infos = infos

        # Snapshot state_after.
        state_after = self.observe()
        state_after_hash = hash_state(state_after)
        state_delta = diff_state(state_before, state_after)

        # Build actual receipt with reward + outcome.
        agent_id_str = str(action.agent_id)
        reward = float(rewards.get(agent_id_str, 0.0))
        info_for_agent = dict(infos.get(agent_id_str, {}))
        receipt = build_receipt(
            action,
            reward=reward,
            success=True,
            outcome_code=OUTCOME_OK,
            info={
                **info_for_agent,
                "termination": bool(terminations.get(agent_id_str, False)),
                "truncation": bool(truncations.get(agent_id_str, False)),
            },
        )

        return TransitionRecord(
            schema_version=PROTOCOL_SCHEMA_VERSION,
            producer_id=PETTINGZOO_WORLD_ID,
            producer_version=PETTINGZOO_WORLD_VERSION,
            tick=tick_before,
            state_before_hash=state_before_hash,
            candidate_actions={action.agent_id: proposal} if proposal else {},
            executed_actions={action.agent_id: action},
            exogenous_input=exogenous,
            receipts={action.agent_id: receipt},
            state_delta=state_delta,
            state_after_hash=state_after_hash,
            capability_profile=self._cap,
            provenance={
                "env_id": self._env_id,
                "joint_step": "false",
                "execution_mode": "sequential_focal_stay",
            },
        )

    # ------------------------------------------------------------------
    # JointActionWorld: validate_joint_action / step_joint (Phase 5)
    # ------------------------------------------------------------------

    def _current_active_agents(self) -> list[str]:
        """Agents currently active in the env (order preserved).

        Reads ``env.agents`` (Parallel API keeps it in sync after
        reset/step); falls back to the unwrapped env. When the env
        EXPOSES an agents list and it is empty, the episode is over
        (all agents terminated/truncated) — return ``[]`` and do NOT
        resurrect agents from stale cached observations (E-G3). The
        ``_last_obs`` fallback only applies when neither the wrapper
        nor the unwrapped env exposes an ``agents`` attribute at all.
        """
        agents = getattr(self._env, "agents", None)
        if agents:
            return [str(a) for a in agents]
        unwrapped = getattr(self._env, "unwrapped", self._env)
        inner = getattr(unwrapped, "agents", None)
        if inner:
            return [str(a) for a in inner]
        if agents is not None or inner is not None:
            return []
        return [str(a) for a in self._last_obs.keys()]

    def active_agents(self) -> list[str]:
        """Public view of the currently active agents (order preserved).

        The joint-mode rollout orchestrator uses this to build the
        per-tick :class:`JointAction.active_agents` tuple. Returns an
        empty list once every agent has terminated/truncated (the
        orchestrator stops proposing — E-G3).
        """
        if not self._initialized:
            raise RuntimeError(
                "PettingZooParallelAdapter.active_agents() called before "
                "reset(); call reset(seed) first."
            )
        return self._current_active_agents()

    def validate_joint_action(
        self,
        joint: JointAction,
    ) -> tuple[JointAction, JointReceipt]:
        """Validate a proposal-stage :class:`JointAction`.

        Per-agent mapping mirrors :meth:`validate_action`:

        - A legal proposal maps to its discrete action.
        - An illegal proposal (out-of-range ``discrete_action``) is
          SUBSTITUTED with STAY (0) and surfaced as a ``success=False``
          receipt with ``OUTCOME_ILLEGAL_ACTION`` — never an exception.
        - A missing proposal is resolved per ``missing_agent_policy``
          (``noop``/``stay`` both map to MPE discrete 0; the synthesized
          proposal's ``proposer`` records the policy). The ``error``
          policy is enforced by :class:`JointAction` itself at
          construction time.

        Returns the executed-stage joint action (executed set covering
        every active agent) plus the validation-stage
        :class:`JointReceipt`. The per-agent discrete actions are cached
        under the executed joint action's hash for :meth:`step_joint`.

        The returned joint action's ``proposals_by_agent`` is the FULL
        candidate set: synthesized (missing-agent) and substituted
        (illegal) proposals are merged in, so the eventual
        ``TransitionRecord.candidate_actions`` covers every executed
        agent and each executed action's ``proposal_hash`` closes the
        loop against its candidate (Q1 orphan/hash checks).
        """
        if not self._initialized:
            raise RuntimeError(
                "PettingZooParallelAdapter.validate_joint_action() called "
                "before reset(); call reset(seed) first."
            )
        env_active = set(self._current_active_agents())
        joint_active = {str(a) for a in joint.active_agents}
        if joint_active != env_active:
            raise JointActionError(
                f"joint.active_agents {sorted(joint_active)!r} do not match "
                f"the env's active agents {sorted(env_active)!r}"
            )

        executed_map: dict[str | int, ExecutedAction] = {}
        receipts_map: dict[str | int, ActionReceipt] = {}
        proposals_map: dict[str | int, ActionProposal] = {}
        actions_dict: dict[str, int] = {}
        for agent_id in joint.active_agents:
            proposal = joint.proposals_by_agent.get(agent_id)
            if proposal is None:
                # Missing agent — synthesize per policy. MPE discrete 0
                # is the env's no-op/stay action, so both policies map
                # to 0; the proposer string records which policy fired.
                proposal = ActionProposal(
                    agent_id=agent_id,
                    action_type=self._action_type,
                    params={"discrete_action": 0},
                    proposed_at_tick=joint.tick,
                    proposer=f"missing_agent_{joint.missing_agent_policy}",
                )
                executed = build_executed_action(proposal)
                receipt = build_receipt(
                    executed,
                    reward=0.0,
                    success=True,
                    outcome_code=OUTCOME_OK,
                    info={"synthesized": joint.missing_agent_policy},
                )
                discrete = 0
                candidate = proposal
            else:
                discrete_or_none = proposal_to_pettingzoo_action(
                    proposal, self._legal_actions
                )
                if discrete_or_none is None:
                    # Illegal proposal — substitute STAY, reject in the
                    # receipt (fail-visible, per kernel rule the world
                    # MAY substitute; the receipt is authoritative).
                    substituted = ActionProposal(
                        agent_id=agent_id,
                        action_type=self._action_type,
                        params={"discrete_action": 0},
                        proposed_at_tick=joint.tick,
                        proposer="illegal_substituted_stay",
                    )
                    executed = build_executed_action(substituted)
                    receipt = build_receipt(
                        executed,
                        reward=0.0,
                        success=False,
                        outcome_code=OUTCOME_ILLEGAL_ACTION,
                        info={
                            "rejected": True,
                            "reason": "illegal_discrete_action",
                            "proposed_params": dict(proposal.params),
                        },
                    )
                    discrete = 0
                    # The substituted proposal becomes the candidate so
                    # the executed action's proposal_hash closes the
                    # loop; the original params live in the receipt.
                    candidate = substituted
                else:
                    executed = build_executed_action(proposal)
                    receipt = build_receipt(
                        executed,
                        reward=0.0,
                        success=True,
                        outcome_code=OUTCOME_OK,
                    )
                    discrete = discrete_or_none
                    candidate = proposal
            executed_map[agent_id] = executed
            receipts_map[agent_id] = receipt
            proposals_map[agent_id] = candidate
            actions_dict[str(agent_id)] = discrete

        executed_joint = _dc_replace(
            joint,
            proposals_by_agent=proposals_map,
            executed_by_agent=executed_map,
        )
        joint_hash = hash_state(executed_joint)
        self._pending_joint[joint_hash] = actions_dict
        return executed_joint, JointReceipt(
            tick=joint.tick, receipts_by_agent=receipts_map
        )

    def step_joint(
        self,
        joint: JointAction,
        *,
        exogenous: ExogenousInput | None = None,
    ) -> TransitionRecord:
        """Execute an executed-stage joint action as ONE parallel env step.

        All active agents' discrete actions are submitted to
        ``env.step(actions_dict)`` simultaneously — no focal+STAY
        filling. The returned :class:`TransitionRecord` carries executed
        actions AND receipts for EVERY active agent; per-agent
        reward/termination/truncation live in each receipt's
        ``diagnostics`` and are mirrored (JSON-encoded) in provenance
        for record-level reconciliation.

        Replay path: if the joint action was not validated (cache miss
        — e.g., reconstructed from a dataset record), the discrete
        actions are re-derived from each executed action's
        ``params["discrete_action"]``. Out-of-range values substitute
        STAY(0) and surface as ``success=False`` receipts.
        """
        if not self._initialized:
            raise RuntimeError(
                "PettingZooParallelAdapter.step_joint() called before "
                "reset(); call reset(seed) first."
            )
        if not joint.is_executed_stage:
            raise JointActionError(
                "step_joint requires an executed-stage JointAction; call "
                "validate_joint_action first."
            )

        # Snapshot state_before.
        state_before = self.observe()
        state_before_hash = hash_state(state_before)
        tick_before = int(state_before.meta.tick)
        active_before = self._current_active_agents()

        # Look up cached discrete actions; on cache miss (replay path)
        # re-derive them from the executed actions' params.
        joint_hash = hash_state(joint)
        actions_dict = self._pending_joint.pop(joint_hash, None)
        illegal_agents: set[str] = set()
        if actions_dict is None:
            actions_dict = {}
            for agent_id, executed in joint.executed_by_agent.items():
                discrete = executed.params.get("discrete_action")
                try:
                    discrete_int = int(discrete)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    discrete_int = None
                if discrete_int is None or discrete_int not in self._legal_actions:
                    illegal_agents.add(str(agent_id))
                    discrete_int = 0  # substitute STAY, receipt records failure
                actions_dict[str(agent_id)] = discrete_int

        # Execute the SINGLE parallel env step for all agents.
        try:
            step_result = self._env.step(actions_dict)
            if len(step_result) != 5:
                raise RuntimeError(
                    f"PettingZoo step returned {len(step_result)} values; "
                    f"expected 5 (Parallel API)."
                )
            obs, rewards, terminations, truncations, infos = step_result
        except Exception as exc:
            logger.exception("PettingZoo env.step (joint) raised: %s", exc)
            state_after = self.observe()
            receipts_err: dict[str | int, ActionReceipt] = {
                agent_id: build_receipt(
                    executed,
                    reward=0.0,
                    success=False,
                    outcome_code=OUTCOME_UNKNOWN_FAILURE,
                    info={"exception": str(exc)},
                )
                for agent_id, executed in joint.executed_by_agent.items()
            }
            return TransitionRecord(
                schema_version=PROTOCOL_SCHEMA_VERSION,
                producer_id=PETTINGZOO_WORLD_ID,
                producer_version=PETTINGZOO_WORLD_VERSION,
                tick=tick_before,
                state_before_hash=state_before_hash,
                candidate_actions=dict(joint.proposals_by_agent),
                executed_actions=dict(joint.executed_by_agent),
                exogenous_input=exogenous,
                receipts=receipts_err,
                state_delta=diff_state(state_before, state_after),
                state_after_hash=hash_state(state_after),
                capability_profile=self._cap,
                provenance={
                    "env_id": self._env_id,
                    "joint_step": "true",
                    "execution_mode": "joint",
                    "error": str(exc),
                },
            )

        # Update cached obs/infos, snapshot state_after.
        self._last_obs = obs
        self._last_infos = infos
        state_after = self.observe()
        state_after_hash = hash_state(state_after)
        state_delta = diff_state(state_before, state_after)
        active_after = self._current_active_agents()

        # Per-agent reconciliation: reward / termination / truncation
        # flow into each agent's receipt diagnostics.
        receipts_map: dict[str | int, ActionReceipt] = {}
        rewards_by_agent: dict[str, float] = {}
        terminations_by_agent: dict[str, bool] = {}
        truncations_by_agent: dict[str, bool] = {}
        infos_digest_by_agent: dict[str, str] = {}
        for agent_id, executed in joint.executed_by_agent.items():
            key = str(agent_id)
            reward = float(rewards.get(key, 0.0))
            termination = bool(terminations.get(key, False))
            truncation = bool(truncations.get(key, False))
            info_for_agent = dict(infos.get(key, {}))
            rewards_by_agent[key] = reward
            terminations_by_agent[key] = termination
            truncations_by_agent[key] = truncation
            infos_digest_by_agent[key] = _digest_info(info_for_agent)
            rejected = key in illegal_agents
            receipts_map[agent_id] = build_receipt(
                executed,
                reward=reward,
                success=not rejected,
                outcome_code=(
                    OUTCOME_ILLEGAL_ACTION if rejected else OUTCOME_OK
                ),
                info={
                    **info_for_agent,
                    "termination": termination,
                    "truncation": truncation,
                    **({"rejected": True} if rejected else {}),
                },
            )

        provenance: dict[str, str] = {
            "env_id": self._env_id,
            "joint_step": "true",
            "execution_mode": "joint",
            "missing_agent_policy": joint.missing_agent_policy,
            "active_agents_before": json.dumps(active_before),
            "active_agents_after": json.dumps(active_after),
            "environment_actions_by_agent": json.dumps(
                actions_dict, sort_keys=True
            ),
            "rewards_by_agent": json.dumps(rewards_by_agent, sort_keys=True),
            "terminations_by_agent": json.dumps(
                terminations_by_agent, sort_keys=True
            ),
            "truncations_by_agent": json.dumps(
                truncations_by_agent, sort_keys=True
            ),
            "infos_digest_by_agent": json.dumps(
                infos_digest_by_agent, sort_keys=True
            ),
        }

        return TransitionRecord(
            schema_version=PROTOCOL_SCHEMA_VERSION,
            producer_id=PETTINGZOO_WORLD_ID,
            producer_version=PETTINGZOO_WORLD_VERSION,
            tick=tick_before,
            state_before_hash=state_before_hash,
            candidate_actions=dict(joint.proposals_by_agent),
            executed_actions=dict(joint.executed_by_agent),
            exogenous_input=exogenous,
            receipts=receipts_map,
            state_delta=state_delta,
            state_after_hash=state_after_hash,
            capability_profile=self._cap,
            provenance=provenance,
        )

    # ------------------------------------------------------------------
    # WorldProtocol: checkpoint / restore
    # ------------------------------------------------------------------

    def checkpoint(self) -> Checkpoint:
        """Snapshot the env as a kernel :class:`Checkpoint`."""
        if not self._initialized:
            raise RuntimeError(
                "PettingZooParallelAdapter.checkpoint() called before reset(); "
                "call reset(seed) first."
            )
        return export_checkpoint(self._env, self.observe())

    def restore(self, checkpoint: Checkpoint) -> None:
        """Restore the env from a kernel :class:`Checkpoint`."""
        restore_checkpoint(self._env, checkpoint)
        # After restore, re-observe to refresh cached obs/infos.
        # We need to rebuild obs/infos from the restored env state.
        # PettingZoo Parallel envs don't have a direct "observe all" method;
        # we use the env's last_step obs if available, or rebuild from
        # unwrapped state. For Attempt 1 smoke, we use the checkpoint's
        # state_view entities as the obs source.
        sv = checkpoint.state_view
        # Rebuild _last_obs from state_view entities (agents only).
        obs: dict[str, Any] = {}
        if sv.entities is not None:
            attrs_col = sv.entities.columns.get("attributes", ()) or ()
            for i, eid in enumerate(sv.entities.ids):
                if str(eid).startswith("landmark"):
                    continue
                attrs = attrs_col[i] if i < len(attrs_col) else {}
                obs[str(eid)] = attrs.get("observation", ())
        self._last_obs = obs
        self._last_infos = {}
        self._initialized = True


# ---------------------------------------------------------------------------
# Helpers (module-private)
# ---------------------------------------------------------------------------


def _digest_info(info: Mapping[str, Any]) -> str:
    """Stable SHA-256 digest of a per-agent info dict (JSON-canonicalized)."""
    canonical = json.dumps(dict(info), sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_env_step(env: Any) -> int:
    """Read the env's step counter WITHOUT advancing the RNG.

    Mirrors :func:`worldloop_adapters.pettingzoo.state_mapper._get_env_step`
    but inlined here so the projector does not depend on a private helper
    in another module. Returns ``0`` if the env does not expose a step
    counter.
    """
    try:
        return int(env.unwrapped.steps)
    except (AttributeError, TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Env factory: Simple Spread (A-02 will add conformance tests)
# ---------------------------------------------------------------------------


def make_simple_spread_env(
    n_agents: int = 2,
    n_landmarks: int = 2,
    max_cycles: int = 25,
    local_ratio: float = 0.5,
):
    """Create a PettingZoo Parallel Simple Spread env.

    This is a convenience factory for tests and demos. The caller owns
    the env lifecycle.

    PettingZoo >= 1.24 moved MPE envs to the standalone ``mpe2`` package;
    the import path is ``from mpe2 import simple_spread_v3``.
    """
    from mpe2 import simple_spread_v3

    # simple_spread_v3.parallel_env returns a Parallel env directly.
    return simple_spread_v3.parallel_env(
        N=n_agents,
        local_ratio=local_ratio,
        max_cycles=max_cycles,
    )


def make_simple_tag_env(
    num_good: int = 1,
    num_adversaries: int = 2,
    num_obstacles: int = 1,
    max_cycles: int = 25,
):
    """Create a PettingZoo Parallel Simple Tag env (Phase 5 second env).

    Simple Tag (predator-prey): ``num_adversaries`` slow adversaries
    chase ``num_good`` fast good agents around ``num_obstacles``
    obstacles. All agents use the MPE discrete 5-action space
    (STAY/LEFT/RIGHT/DOWN/UP), so the existing action mapper applies
    unchanged. The caller owns the env lifecycle.
    """
    from mpe2 import simple_tag_v3

    return simple_tag_v3.parallel_env(
        num_good=num_good,
        num_adversaries=num_adversaries,
        num_obstacles=num_obstacles,
        max_cycles=max_cycles,
        continuous_actions=False,
    )
