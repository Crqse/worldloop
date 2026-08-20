"""OpenEnvServerWrapper: kernel world → OpenEnv-style server (A-07).

Wraps a kernel :class:`WorldProtocol` implementation as an in-process
OpenEnv-style server. Any OpenEnv client (or :class:`OpenEnvWorldAdapter`)
can then drive the kernel world through the standard
``reset`` / ``step`` / ``state`` / ``action_space`` interface.

This is the reverse direction of :class:`OpenEnvWorldAdapter`:
- :class:`OpenEnvWorldAdapter` wraps an OpenEnv client as a kernel world.
- :class:`OpenEnvServerWrapper` wraps a kernel world as an OpenEnv server.

Architecture (per main plan §12.4 E2 OpenEnv):
- The wrapper is in-process only. For HTTP / Docker / gRPC exposure, use
  OpenEnv upstream tooling — the kernel does not implement transport.
- The wrapper translates the OpenEnv 5-tuple ``step`` return
  ``(state, reward, terminated, truncated, info)`` to / from the kernel
  :class:`TransitionRecord` shape.
- ``state()`` returns the current :class:`StateView` serialized as a dict
  (observation + tick + entity ids).
- ``action_space()`` returns the discrete action ints extracted from
  :meth:`WorldProtocol.legal_actions` (only the ``discrete_action`` param
  is surfaced; non-discrete actions are not exposed via OpenEnv).

Single-agent convention:
- The wrapper assumes the underlying kernel world has a single primary
  agent (matching OpenEnv's single-agent semantics). Multi-agent kernel
  worlds should use :class:`PettingZooParallelAdapter` instead.
"""
from __future__ import annotations

import logging
from typing import Any

from worldloop_kernel.action import (
    ActionProposal,
    ExecutedAction,
    OUTCOME_OK,
)
from worldloop_kernel.canonical import hash_state
from worldloop_kernel.protocol import WorldProtocol

logger = logging.getLogger(__name__)

__all__ = ["OpenEnvServerWrapper"]


# ---------------------------------------------------------------------------
# OpenEnvServerWrapper
# ---------------------------------------------------------------------------


class OpenEnvServerWrapper:
    """Wrap a kernel :class:`WorldProtocol` as an OpenEnv-style server.

    The wrapper exposes the OpenEnv client interface
    (``reset`` / ``step`` / ``state`` / ``action_space``) backed by a
    kernel world. It is intended for in-process integration tests and
    local development; production deployment should use OpenEnv upstream
    HTTP/Docker tooling.
    """

    def __init__(
        self,
        world: WorldProtocol,
        *,
        agent_id: str = "agent_0",
        action_type: str = "step",
    ) -> None:
        """Construct the wrapper.

        Parameters
        ----------
        world:
            The kernel world to expose. MUST implement
            :class:`WorldProtocol` (``reset`` / ``observe`` /
            ``legal_actions`` / ``validate_action`` / ``step``).
        agent_id:
            The single agent id used in :meth:`ActionProposal` and
            :meth:`legal_actions` calls.
        action_type:
            The action_type string used in :class:`ActionProposal`.
            Defaults to ``"step"`` (matches OpenEnv convention).
        """
        self._world = world
        self._agent_id = agent_id
        self._action_type = action_type
        self._tick: int = 0
        self._initialized = False
        # Cache of the last executed action's hash → proposal, used so
        # that step() can re-use the proposal the client passed via
        # action int. The wrapper constructs the proposal internally
        # because the OpenEnv interface only accepts a discrete int.
        self._last_proposal: ActionProposal | None = None
        self._last_executed: ExecutedAction | None = None
        self._last_record: Any = None

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def action_space(self) -> tuple[int, ...]:
        """Return the discrete action ints the server accepts.

        Reads :meth:`WorldProtocol.legal_actions` and extracts the
        ``discrete_action`` param from each :class:`LegalAction`. Legal
        actions without a ``discrete_action`` param are skipped.
        """
        aspace = self._world.legal_actions(self._agent_id)
        actions: list[int] = []
        for legal in aspace.legal_actions:
            discrete = legal.params.get("discrete_action")
            if discrete is None:
                continue
            try:
                actions.append(int(discrete))
            except (TypeError, ValueError):
                continue
        return tuple(actions)

    def reset(self, seed: int) -> dict[str, Any]:
        """Reset the underlying kernel world and return the initial state.

        Parameters
        ----------
        seed:
            RNG seed for the world's ``reset``.
        """
        self._world.reset(seed=seed)
        self._tick = 0
        self._initialized = True
        self._last_proposal = None
        self._last_executed = None
        self._last_record = None
        return self.state()

    def state(self) -> dict[str, Any]:
        """Return the current world state as a dict (no advance)."""
        if not self._initialized:
            raise RuntimeError(
                "OpenEnvServerWrapper.state() called before reset(); "
                "call reset(seed) first."
            )
        sv = self._world.observe()
        # Extract observation from the single entity's attributes.
        obs: tuple[float, ...] = ()
        if sv.entities and sv.entities.ids:
            ids = sv.entities.ids
            try:
                idx = ids.index(self._agent_id)
            except ValueError:
                idx = 0
            attrs = sv.entities.columns.get("attributes", ())
            if idx < len(attrs):
                attr = attrs[idx] or {}
                obs_tuple = attr.get("observation", ())
                if isinstance(obs_tuple, (tuple, list)):
                    obs = tuple(float(x) for x in obs_tuple)
        return {
            "observation": list(obs),
            "tick": int(sv.meta.tick),
            "agent_id": self._agent_id,
        }

    def step(
        self, action: int
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Take a discrete action in the world.

        Parameters
        ----------
        action:
            Discrete action int (must be in :meth:`action_space`).

        Returns
        -------
        tuple
            OpenEnv 5-tuple ``(state, reward, terminated, truncated, info)``.
        """
        if not self._initialized:
            raise RuntimeError(
                "OpenEnvServerWrapper.step() called before reset(); "
                "call reset(seed) first."
            )

        legal = self.action_space()
        if action not in legal:
            # Out-of-space action: return zero reward, no advance.
            return self.state(), 0.0, False, False, {
                "rejected": True,
                "reason": f"action {action} not in legal space {legal}",
            }

        # Build a proposal and execute it.
        proposal = ActionProposal(
            agent_id=self._agent_id,
            action_type=self._action_type,
            params={"discrete_action": int(action)},
            proposed_at_tick=self._tick,
            proposer="openenv-server",
        )
        executed, _ = self._world.validate_action(proposal)
        record = self._world.step(executed)

        self._last_proposal = proposal
        self._last_executed = executed
        self._last_record = record
        self._tick += 1

        # Extract reward and termination from the receipt.
        receipt = record.receipts.get(self._agent_id)
        reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {"tick": self._tick}
        if receipt is not None:
            reward = float(receipt.diagnostics.get("reward", 0.0))
            env_info = receipt.diagnostics.get("info", {})
            if isinstance(env_info, dict):
                terminated = bool(env_info.get("termination", False))
                truncated = bool(env_info.get("truncation", False))
                info.update(env_info)
            info["outcome_code"] = receipt.outcome_code
            info["success"] = receipt.success

        return self.state(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Introspection (for tests / debugging)
    # ------------------------------------------------------------------

    @property
    def world(self) -> WorldProtocol:
        """The underlying kernel world (read-only)."""
        return self._world

    @property
    def tick(self) -> int:
        """Current tick (incremented after each successful step)."""
        return self._tick

    @property
    def last_record(self) -> Any:
        """The most recent :class:`TransitionRecord` (or ``None``)."""
        return self._last_record
