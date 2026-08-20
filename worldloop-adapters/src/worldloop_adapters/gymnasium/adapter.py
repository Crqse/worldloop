"""GymnasiumAdapter: Gymnasium single-agent env → kernel WorldProtocol (A-05).

Wraps a Gymnasium single-agent env as a kernel :class:`WorldProtocol`.
The adapter is a thin type-mapping layer: it does NOT write business
rules, does NOT call LLMs, and does NOT mutate the env outside of the
documented ``reset`` / ``step`` / ``restore`` methods.

Architecture (per A-05 spec and main plan §12.2):
- ``reset(seed)`` calls ``env.reset(seed=seed)`` then ``observe``.
- ``observe`` builds a :class:`StateView` via :func:`build_state_view`
  (single entity: the agent itself).
- ``legal_actions`` returns the env's discrete action space as a tuple
  of :class:`LegalAction` entries (``is_closed=True``).
- ``validate_action`` converts the proposal to a discrete action int,
  builds an :class:`ExecutedAction`, caches it for ``step``.
- ``step`` looks up the cached action, calls ``env.step(action)``, reads
  the outcome (reward / terminated / truncated), builds a
  :class:`TransitionRecord`.
- ``checkpoint`` / ``restore`` delegate to the checkpoint mapper.

Single-agent convention:
- The agent_id is fixed to ``"agent_0"`` (configurable at construction).
- All TransitionRecord dicts are keyed by this single agent_id.
"""
from __future__ import annotations

import logging
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
from worldloop_kernel.protocol import ActionSpace, LegalAction
from worldloop_kernel.state import StateView
from worldloop_kernel.transition import (
    Checkpoint,
    PROTOCOL_SCHEMA_VERSION,
    TransitionRecord,
)

from .action_mapper import (
    GYMNASIUM_DEFAULT_ACTION_TYPE,
    build_executed_action,
    build_receipt,
    proposal_to_gymnasium_action,
    reject_proposal,
)
from .capability import (
    GYMNASIUM_WORLD_ID,
    GYMNASIUM_WORLD_VERSION,
    make_gymnasium_discrete_capability,
)
from .checkpoint_mapper import export_checkpoint, restore_checkpoint
from .state_mapper import build_state_view

logger = logging.getLogger(__name__)

__all__ = ["GymnasiumAdapter", "make_cartpole_env"]


# ---------------------------------------------------------------------------
# GymnasiumAdapter
# ---------------------------------------------------------------------------


class GymnasiumAdapter:
    """Adapter wrapping a Gymnasium single-agent env as kernel :class:`WorldProtocol`.

    The adapter is constructed with a Gymnasium env instance. The adapter
    does NOT take ownership of the env's lifecycle.
    """

    def __init__(
        self,
        env: Any,
        *,
        env_id: str = "gymnasium-env",
        capability: CapabilityProfile | None = None,
        action_type: str = GYMNASIUM_DEFAULT_ACTION_TYPE,
        agent_id: str = "agent_0",
        run_id: str = "gymnasium-run",
        config_hash: str = "gymnasium-default",
    ) -> None:
        """Construct the adapter.

        Parameters
        ----------
        env:
            The Gymnasium env handle. MUST support
            ``reset(seed=...)`` / ``step(action)`` / ``action_space``.
        env_id:
            Stable identifier for this env instance.
        capability:
            Optional override for the capability profile.
        action_type:
            The action_type string used in ActionProposal/ExecutedAction.
        agent_id:
            The single agent's id (Gymnasium is single-agent).
        run_id:
            Stable run identifier for this trajectory.
        config_hash:
            Stable config hash for this scenario.
        """
        self._env = env
        self._env_id = env_id
        self._cap = capability or make_gymnasium_discrete_capability()
        self._action_type = action_type
        self._agent_id = agent_id
        self._run_id = run_id
        self._config_hash = config_hash

        # Determine legal discrete actions from env.action_space.
        self._legal_actions: tuple[int, ...] = self._infer_legal_actions(env)

        # Cached state between validate_action and step.
        self._pending_actions: dict[str, int] = {}
        self._pending_proposals: dict[str, ActionProposal] = {}

        # Cached last obs/info.
        self._last_obs: Any = None
        self._last_info: dict[str, Any] = {}

        self._initialized = False

    @staticmethod
    def _infer_legal_actions(env: Any) -> tuple[int, ...]:
        """Infer the tuple of legal discrete actions from env.action_space.

        Supports ``gymnasium.spaces.Discrete``. Continuous (``Box``)
        action spaces raise ``NotImplementedError`` (A-05 scope: discrete only).
        """
        try:
            from gymnasium.spaces import Discrete
        except ImportError:
            # Gymnasium not installed; assume discrete 0..n-1.
            return (0, 1)
        aspace = getattr(env, "action_space", None)
        if isinstance(aspace, Discrete):
            return tuple(range(int(aspace.n)))
        raise NotImplementedError(
            f"GymnasiumAdapter currently supports only Discrete action spaces; "
            f"got {type(aspace).__name__}. Continuous (Box) actions are A-05 follow-up."
        )

    # ------------------------------------------------------------------
    # WorldProtocol: capabilities
    # ------------------------------------------------------------------

    @property
    def capabilities(self) -> CapabilityProfile:
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
        result = self._env.reset(seed=seed)
        if isinstance(result, tuple) and len(result) == 2:
            self._last_obs, self._last_info = result
        else:
            self._last_obs = result
            self._last_info = {}
        self._initialized = True
        self._pending_actions.clear()
        self._pending_proposals.clear()
        return self.observe()

    def observe(self) -> StateView:
        """Return the current :class:`StateView` without advancing the env."""
        if not self._initialized:
            raise RuntimeError(
                "GymnasiumAdapter.observe() called before reset(); "
                "call reset(seed) first."
            )
        return build_state_view(
            self._env,
            self._last_obs,
            self._last_info,
            self._cap,
            agent_id=self._agent_id,
            run_id=self._run_id,
            config_hash=self._config_hash,
        )

    # ------------------------------------------------------------------
    # WorldProtocol: legal_actions
    # ------------------------------------------------------------------

    def legal_actions(
        self,
        agent_id: str | int,
        state: StateView | None = None,
    ) -> ActionSpace:
        """Return the legal action space for ``agent_id``."""
        if state is not None:
            raise NotImplementedError(
                "GymnasiumAdapter.legal_actions(state=...) is not supported yet."
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
        """Validate a candidate :class:`ActionProposal`."""
        discrete = proposal_to_gymnasium_action(proposal, self._legal_actions)
        if discrete is None:
            return reject_proposal(proposal, OUTCOME_ILLEGAL_ACTION)

        executed = build_executed_action(proposal)
        executed_hash = hash_state(executed)
        self._pending_actions[executed_hash] = discrete
        self._pending_proposals[executed_hash] = proposal

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
        :class:`TransitionRecord`."""
        if not self._initialized:
            raise RuntimeError(
                "GymnasiumAdapter.step() called before reset(); "
                "call reset(seed) first."
            )

        state_before = self.observe()
        state_before_hash = hash_state(state_before)
        tick_before = int(state_before.meta.tick)

        executed_hash = hash_state(action)
        discrete = self._pending_actions.pop(executed_hash, None)
        proposal = self._pending_proposals.pop(executed_hash, None)
        if discrete is None:
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
                producer_id=GYMNASIUM_WORLD_ID,
                producer_version=GYMNASIUM_WORLD_VERSION,
                tick=tick_before,
                state_before_hash=state_before_hash,
                candidate_actions={action.agent_id: proposal} if proposal else {},
                executed_actions={action.agent_id: action},
                exogenous_input=exogenous,
                receipts={action.agent_id: receipt},
                state_delta=state_delta,
                state_after_hash=state_after_hash,
                capability_profile=self._cap,
                provenance={"env_id": self._env_id},
            )

        try:
            step_result = self._env.step(discrete)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
            elif len(step_result) == 4:
                # Older Gym API (obs, reward, done, info) — not expected
                # with Gymnasium >= 0.26, but handle defensively.
                obs, reward, terminated, info = step_result
                truncated = False
            else:
                raise RuntimeError(
                    f"Gymnasium step returned {len(step_result)} values; "
                    f"expected 5 (Gymnasium API)."
                )
        except Exception as exc:
            logger.exception("Gymnasium env.step raised: %s", exc)
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
                producer_id=GYMNASIUM_WORLD_ID,
                producer_version=GYMNASIUM_WORLD_VERSION,
                tick=tick_before,
                state_before_hash=state_before_hash,
                candidate_actions={action.agent_id: proposal} if proposal else {},
                executed_actions={action.agent_id: action},
                exogenous_input=exogenous,
                receipts={action.agent_id: receipt},
                state_delta=state_delta,
                state_after_hash=state_after_hash,
                capability_profile=self._cap,
                provenance={"env_id": self._env_id, "error": str(exc)},
            )

        self._last_obs = obs
        self._last_info = info if isinstance(info, dict) else {}

        state_after = self.observe()
        state_after_hash = hash_state(state_after)
        state_delta = diff_state(state_before, state_after)

        receipt = build_receipt(
            action,
            reward=float(reward),
            success=True,
            outcome_code=OUTCOME_OK,
            info={
                **(info if isinstance(info, dict) else {}),
                "termination": bool(terminated),
                "truncation": bool(truncated),
            },
        )

        return TransitionRecord(
            schema_version=PROTOCOL_SCHEMA_VERSION,
            producer_id=GYMNASIUM_WORLD_ID,
            producer_version=GYMNASIUM_WORLD_VERSION,
            tick=tick_before,
            state_before_hash=state_before_hash,
            candidate_actions={action.agent_id: proposal} if proposal else {},
            executed_actions={action.agent_id: action},
            exogenous_input=exogenous,
            receipts={action.agent_id: receipt},
            state_delta=state_delta,
            state_after_hash=state_after_hash,
            capability_profile=self._cap,
            provenance={"env_id": self._env_id},
        )

    # ------------------------------------------------------------------
    # WorldProtocol: checkpoint / restore
    # ------------------------------------------------------------------

    def checkpoint(self) -> Checkpoint:
        if not self._initialized:
            raise RuntimeError(
                "GymnasiumAdapter.checkpoint() called before reset(); "
                "call reset(seed) first."
            )
        return export_checkpoint(self._env, self.observe())

    def restore(self, checkpoint: Checkpoint) -> None:
        restore_checkpoint(self._env, checkpoint)
        # After restore, re-observe to refresh cached obs/infos.
        # Gymnasium envs don't expose a direct "observe" without stepping;
        # we rebuild _last_obs from the checkpoint's state_view entities.
        sv = checkpoint.state_view
        if sv.entities is not None:
            attrs_col = sv.entities.columns.get("attributes", ()) or ()
            if attrs_col:
                attrs = attrs_col[0]
                self._last_obs = attrs.get("observation", ())
        self._last_info = {}
        self._initialized = True


# ---------------------------------------------------------------------------
# Env factory: CartPole (A-05 conformance)
# ---------------------------------------------------------------------------


def make_cartpole_env():
    """Create a Gymnasium CartPole-v1 env.

    This is a convenience factory for tests and demos.
    """
    import gymnasium as gym

    return gym.make("CartPole-v1")
