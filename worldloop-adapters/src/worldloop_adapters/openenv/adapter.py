"""OpenEnvWorldAdapter: OpenEnv client → kernel WorldProtocol (A-06).

Wraps any client satisfying the OpenEnv reset/step/state protocol as a
kernel :class:`WorldProtocol`. The adapter uses duck typing — it does
NOT import the ``openenv`` PyPI package. Any client with the following
methods works:

- ``reset(seed: int) -> dict`` (returns initial state dict)
- ``step(action: int) -> tuple[dict, float, bool, bool, dict]``
  (returns (state, reward, terminated, truncated, info))
- ``state() -> dict`` (returns current state without advancing)
- ``action_space() -> tuple[int, ...]`` (returns legal actions)

For real OpenEnv integration, install ``openenv`` and pass its client
to this adapter. For tests, use :class:`InProcessOpenEnvClient`.
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
from worldloop_kernel.state import (
    EntityTable,
    StateMeta,
    StateView,
)
from worldloop_kernel.transition import (
    Checkpoint,
    PROTOCOL_SCHEMA_VERSION,
    TransitionRecord,
)

from .capability import (
    OPENENV_ENTITY_SCHEMA_ID,
    OPENENV_PAYLOAD_CODEC,
    OPENENV_SCENARIO_ID,
    OPENENV_WORLD_ID,
    OPENENV_WORLD_VERSION,
    make_openenv_capability,
)

logger = logging.getLogger(__name__)

__all__ = ["OpenEnvWorldAdapter", "InProcessOpenEnvClient"]


# ---------------------------------------------------------------------------
# OpenEnvWorldAdapter
# ---------------------------------------------------------------------------


class OpenEnvWorldAdapter:
    """Adapter wrapping an OpenEnv-style client as kernel :class:`WorldProtocol`.

    The adapter is constructed with a client object (any object with
    ``reset``/``step``/``state``/``action_space`` methods). The adapter
    does NOT take ownership of the client's lifecycle.
    """

    def __init__(
        self,
        client: Any,
        *,
        env_id: str = "openenv-env",
        capability: CapabilityProfile | None = None,
        action_type: str = "step",
        agent_id: str = "agent_0",
        run_id: str = "openenv-run",
        config_hash: str = "openenv-default",
    ) -> None:
        self._client = client
        self._env_id = env_id
        self._cap = capability or make_openenv_capability()
        self._action_type = action_type
        self._agent_id = agent_id
        self._run_id = run_id
        self._config_hash = config_hash

        # Infer legal actions from client.action_space().
        self._legal_actions: tuple[int, ...] = tuple(client.action_space())

        self._pending_actions: dict[str, int] = {}
        self._pending_proposals: dict[str, ActionProposal] = {}

        self._last_state: dict[str, Any] = {}
        self._last_info: dict[str, Any] = {}
        self._tick: int = 0
        self._initialized = False

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
        result = self._client.reset(seed=seed)
        if isinstance(result, tuple) and len(result) == 2:
            self._last_state, self._last_info = result
        else:
            self._last_state = result if isinstance(result, dict) else {"observation": result}
            self._last_info = {}
        self._tick = 0
        self._initialized = True
        self._pending_actions.clear()
        self._pending_proposals.clear()
        return self.observe()

    def observe(self) -> StateView:
        if not self._initialized:
            raise RuntimeError(
                "OpenEnvWorldAdapter.observe() called before reset(); "
                "call reset(seed) first."
            )
        # Read fresh state without advancing.
        try:
            fresh = self._client.state()
            if isinstance(fresh, dict):
                self._last_state = fresh
        except Exception:
            pass  # fall back to cached _last_state
        return self._build_state_view()

    def _build_state_view(self) -> StateView:
        meta = StateMeta(
            scenario_id=OPENENV_SCENARIO_ID,
            run_id=self._run_id,
            tick=self._tick,
            config_hash=self._config_hash,
            rng_state_ref=f"step:{self._tick}",
        )

        # Extract observation from state dict.
        obs = self._last_state.get("observation", self._last_state)
        try:
            obs_tuple = tuple(float(x) for x in obs)
        except (TypeError, ValueError):
            obs_tuple = ()

        position = obs_tuple[:2] if len(obs_tuple) >= 2 else (0.0, 0.0)
        velocity = obs_tuple[2:4] if len(obs_tuple) >= 4 else ()

        attributes: dict[str, Any] = {"observation": obs_tuple}
        if velocity:
            attributes["velocity"] = velocity

        entities = EntityTable(
            schema_id=OPENENV_ENTITY_SCHEMA_ID,
            ids=(self._agent_id,),
            columns={
                "position": (position,),
                "kind": ("agent",),
                "attributes": (attributes,),
            },
        )

        return StateView(
            meta=meta,
            entities=entities,
            capabilities=self._cap,
            missing_mask={},
            fields=None,
            relations=None,
            registries=None,
            population=None,
            events=None,
        )

    # ------------------------------------------------------------------
    # WorldProtocol: legal_actions
    # ------------------------------------------------------------------

    def legal_actions(
        self,
        agent_id: str | int,
        state: StateView | None = None,
    ) -> ActionSpace:
        if state is not None:
            raise NotImplementedError(
                "OpenEnvWorldAdapter.legal_actions(state=...) is not supported yet."
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
        discrete = self._proposal_to_action(proposal)
        if discrete is None:
            return self._reject_proposal(proposal, OUTCOME_ILLEGAL_ACTION)

        executed = self._build_executed_action(proposal)
        executed_hash = hash_state(executed)
        self._pending_actions[executed_hash] = discrete
        self._pending_proposals[executed_hash] = proposal

        receipt = self._build_receipt(executed, reward=0.0, success=True, outcome_code=OUTCOME_OK)
        return executed, receipt

    def _proposal_to_action(self, proposal: ActionProposal) -> int | None:
        discrete = proposal.params.get("discrete_action")
        if discrete is None:
            return None
        try:
            discrete_int = int(discrete)
        except (TypeError, ValueError):
            return None
        if discrete_int not in self._legal_actions:
            return None
        return discrete_int

    def _build_executed_action(self, proposal: ActionProposal) -> ExecutedAction:
        return ExecutedAction(
            agent_id=proposal.agent_id,
            action_type=proposal.action_type,
            params=proposal.params,
            executed_at_tick=proposal.proposed_at_tick,
            proposal_hash=hash_state(proposal),
        )

    def _build_receipt(
        self,
        executed: ExecutedAction,
        *,
        reward: float = 0.0,
        success: bool = True,
        outcome_code: str = OUTCOME_OK,
        info: dict[str, Any] | None = None,
    ) -> ActionReceipt:
        diagnostics: dict[str, Any] = {"reward": float(reward)}
        if info:
            diagnostics["info"] = dict(info)
        return ActionReceipt(
            executed_action_hash=hash_state(executed),
            outcome_code=outcome_code,
            success=success,
            energy_delta=0.0,
            events=(),
            diagnostics=diagnostics,
        )

    def _reject_proposal(
        self, proposal: ActionProposal, outcome_code: str = OUTCOME_ILLEGAL_ACTION
    ) -> tuple[ExecutedAction, ActionReceipt]:
        executed = self._build_executed_action(proposal)
        receipt = self._build_receipt(
            executed, reward=0.0, success=False, outcome_code=outcome_code, info={"rejected": True}
        )
        return executed, receipt

    # ------------------------------------------------------------------
    # WorldProtocol: step
    # ------------------------------------------------------------------

    def step(
        self,
        action: ExecutedAction,
        exogenous: ExogenousInput | None = None,
    ) -> TransitionRecord:
        if not self._initialized:
            raise RuntimeError(
                "OpenEnvWorldAdapter.step() called before reset(); call reset(seed) first."
            )

        state_before = self.observe()
        state_before_hash = hash_state(state_before)
        tick_before = self._tick

        executed_hash = hash_state(action)
        discrete = self._pending_actions.pop(executed_hash, None)
        proposal = self._pending_proposals.pop(executed_hash, None)
        if discrete is None:
            state_after = self.observe()
            state_after_hash = hash_state(state_after)
            state_delta = diff_state(state_before, state_after)
            receipt = self._build_receipt(
                action,
                reward=0.0,
                success=False,
                outcome_code=OUTCOME_ILLEGAL_ACTION,
                info={"reason": "action not validated or already consumed"},
            )
            return TransitionRecord(
                schema_version=PROTOCOL_SCHEMA_VERSION,
                producer_id=OPENENV_WORLD_ID,
                producer_version=OPENENV_WORLD_VERSION,
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
            step_result = self._client.step(discrete)
            if len(step_result) == 5:
                state, reward, terminated, truncated, info = step_result
            elif len(step_result) == 4:
                state, reward, terminated, info = step_result
                truncated = False
            else:
                raise RuntimeError(
                    f"OpenEnv client.step returned {len(step_result)} values; expected 5."
                )
        except Exception as exc:
            logger.exception("OpenEnv client.step raised: %s", exc)
            state_after = self.observe()
            state_after_hash = hash_state(state_after)
            state_delta = diff_state(state_before, state_after)
            receipt = self._build_receipt(
                action,
                reward=0.0,
                success=False,
                outcome_code=OUTCOME_UNKNOWN_FAILURE,
                info={"exception": str(exc)},
            )
            return TransitionRecord(
                schema_version=PROTOCOL_SCHEMA_VERSION,
                producer_id=OPENENV_WORLD_ID,
                producer_version=OPENENV_WORLD_VERSION,
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

        self._last_state = state if isinstance(state, dict) else {"observation": state}
        self._last_info = info if isinstance(info, dict) else {}
        self._tick += 1

        state_after = self.observe()
        state_after_hash = hash_state(state_after)
        state_delta = diff_state(state_before, state_after)

        receipt = self._build_receipt(
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
            producer_id=OPENENV_WORLD_ID,
            producer_version=OPENENV_WORLD_VERSION,
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
                "OpenEnvWorldAdapter.checkpoint() called before reset(); "
                "call reset(seed) first."
            )
        import copy
        import hashlib
        import pickle

        # Snapshot client state via deepcopy (best-effort).
        try:
            client_state_copy = copy.deepcopy(self._client.__dict__)
        except Exception:
            client_state_copy = {}
        opaque_payload = pickle.dumps({
            "client_state": client_state_copy,
            "last_state": self._last_state,
            "last_info": self._last_info,
            "tick": self._tick,
        })

        sv = self.observe()
        sv_hash = hash_state(sv)
        payload_hash = hashlib.sha256(opaque_payload).hexdigest()
        checksum = f"sha256:{sv_hash}:{payload_hash}"

        return Checkpoint(
            schema_version=PROTOCOL_SCHEMA_VERSION,
            world_id=OPENENV_WORLD_ID,
            world_version=OPENENV_WORLD_VERSION,
            tick=self._tick,
            state_view=sv,
            opaque_payload=opaque_payload,
            payload_codec=OPENENV_PAYLOAD_CODEC,
            capability_profile=self._cap,
            checksum=checksum,
        )

    def restore(self, checkpoint: Checkpoint) -> None:
        if checkpoint.payload_codec != OPENENV_PAYLOAD_CODEC:
            raise ValueError(
                f"unsupported payload codec: {checkpoint.payload_codec!r} "
                f"(expected {OPENENV_PAYLOAD_CODEC!r})"
            )
        import pickle

        payload = pickle.loads(checkpoint.opaque_payload)
        # Best-effort restore of client state.
        try:
            self._client.__dict__.update(payload.get("client_state", {}))
        except Exception:
            pass
        self._last_state = payload.get("last_state", {})
        self._last_info = payload.get("last_info", {})
        self._tick = payload.get("tick", 0)
        self._initialized = True


# ---------------------------------------------------------------------------
# InProcessOpenEnvClient: mock OpenEnv client for tests
# ---------------------------------------------------------------------------


class InProcessOpenEnvClient:
    """Mock OpenEnv client for conformance tests.

    Implements a simple counting state: observation = [tick, tick+1, 0, 0].
    Action 0 increments tick by 1; action 1 increments tick by 2. Reward
    is the new tick value. Termination when tick >= 10.
    """

    def __init__(self, max_tick: int = 10) -> None:
        self._tick = 0
        self._max_tick = max_tick
        self._state: dict[str, Any] = {"observation": [0.0, 0.0, 0.0, 0.0]}

    def action_space(self) -> tuple[int, ...]:
        return (0, 1)

    def reset(self, seed: int) -> dict[str, Any]:
        self._tick = 0
        self._state = {"observation": [0.0, 0.0, 0.0, 0.0]}
        return self._state

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        increment = 1 if action == 0 else 2
        self._tick += increment
        obs = [float(self._tick), float(self._tick + 1), 0.0, 0.0]
        self._state = {"observation": obs}
        reward = float(self._tick)
        terminated = self._tick >= self._max_tick
        truncated = False
        info = {"tick": self._tick}
        return self._state, reward, terminated, truncated, info

    def state(self) -> dict[str, Any]:
        return dict(self._state)
