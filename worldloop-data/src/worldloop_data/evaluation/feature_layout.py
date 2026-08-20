"""Feature layout for M6 evaluation (audit F-02 / R3 fix).

Describes the block structure of feature matrices ``X`` used by baselines.
All baselines MUST use this layout instead of hardcoding column slices
like ``X[:, 1:8]`` or ``X[:, 8:12]``. Hardcoded slices break when the
scenario vocab has a different number of actions or agents (e.g.
``market_exchange_v0`` has 4 actions, ``market_exchange_v1`` has 5,
``emergency_resource_v0`` has 7). The audit F-02 finding showed that
``1:8`` in a 4-action scenario silently zeros out the agent block and
part of the parameter block, contaminating the no-action / shuffled-action
ablations.

Block order (when all present)::

    [tick | action | agent | parameter | state | exogenous]

``state`` and ``exogenous`` blocks are optional (``None``) in the
current implementation; they will be populated by R2
(``TrainingTransition`` / ``StateMaterializer``). When a block is
``None``, baselines treat it as absent and never slice into it.

This module is dependency-free (only stdlib ``dataclasses``) so it can
be imported by both ``data_loader.py`` and ``baselines.py`` without
introducing cycles.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureLayout:
    """Block layout of a feature matrix ``X``.

    All slices are half-open ``[start, end)`` along axis 1 (columns).
    A slice may be ``None`` if the corresponding block is not present
    in the current feature schema.

    Attributes
    ----------
    tick_slice:
        Single-column block holding the tick index. Always present in
        the current loader; kept as ``slice | None`` for forward
        compatibility with R2 layouts that may omit it.
    action_slice:
        Multi-column one-hot block encoding ``action_type``. Required.
    agent_slice:
        Multi-column one-hot block encoding ``agent_id``. Required.
    parameter_slice:
        Block holding action parameters (``target_node_idx``,
        ``target_agent_idx``, ``has_params``). Required.
    state_slice:
        Block holding materialized state features (field/entity/graph/
        registry/population). ``None`` until R2 lands.
    exogenous_slice:
        Block holding exogenous input features. ``None`` until R2 lands.
    """

    tick_slice: slice | None
    action_slice: slice
    agent_slice: slice
    parameter_slice: slice
    state_slice: slice | None = None
    exogenous_slice: slice | None = None

    # ------------------------------------------------------------------
    # Convenience derived dims
    # ------------------------------------------------------------------
    @property
    def n_actions(self) -> int:
        return self.action_slice.stop - self.action_slice.start

    @property
    def n_agents(self) -> int:
        return self.agent_slice.stop - self.agent_slice.start

    @property
    def n_parameters(self) -> int:
        return self.parameter_slice.stop - self.parameter_slice.start

    @property
    def n_state(self) -> int:
        return 0 if self.state_slice is None else self.state_slice.stop - self.state_slice.start

    @property
    def n_exogenous(self) -> int:
        return (
            0
            if self.exogenous_slice is None
            else self.exogenous_slice.stop - self.exogenous_slice.start
        )

    @property
    def feature_dim(self) -> int:
        """Total number of columns spanned by the layout."""
        candidates = [
            self.tick_slice.stop if self.tick_slice else 0,
            self.action_slice.stop,
            self.agent_slice.stop,
            self.parameter_slice.stop,
            self.state_slice.stop if self.state_slice else 0,
            self.exogenous_slice.stop if self.exogenous_slice else 0,
        ]
        return max(candidates)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_dims(
        cls,
        n_actions: int,
        n_agents: int,
        n_parameters: int = 3,
        n_state: int = 0,
        n_exogenous: int = 0,
        include_tick: bool = True,
    ) -> "FeatureLayout":
        """Build a layout from per-block dims.

        Block order: ``[tick?] + action + agent + parameter + state? + exogenous?``.

        Parameters
        ----------
        n_actions:
            Number of one-hot action columns. Must be ``>= 0``.
            ``0`` is allowed for R2 layouts that use the joint-action
            block in place of the legacy single-action one-hot — in
            that case the caller puts the joint-action dim into
            ``n_state`` or passes it via a custom construction.
            Legacy baselines expect ``n_actions >= 1``; R2 baselines
            use ``action_slice`` to point at the joint-action block
            (which may be allocated via ``n_actions`` in R2 callers).
        n_agents:
            Number of one-hot agent columns. Must be ``>= 0``.
            ``0`` is allowed for R2 layouts where the agent structure
            is encoded inside the joint-action block.
        n_parameters:
            Number of parameter columns. Defaults to ``3`` (target_node_idx,
            target_agent_idx, has_params) for backwards compatibility.
            ``0`` is allowed for R2 layouts.
        n_state:
            Number of materialized state columns. ``0`` means the state
            block is absent (``state_slice is None``). R2 will pass a
            positive value.
        n_exogenous:
            Number of exogenous input columns. ``0`` means absent.
        include_tick:
            Whether to reserve a leading column for the tick index.
            Defaults to ``True`` (matches the current ``DataLoader``
            layout).
        """
        if n_actions < 0:
            raise ValueError(f"n_actions must be >= 0, got {n_actions}")
        if n_agents < 0:
            raise ValueError(f"n_agents must be >= 0, got {n_agents}")
        if n_parameters < 0:
            raise ValueError(f"n_parameters must be >= 0, got {n_parameters}")
        if n_state < 0:
            raise ValueError(f"n_state must be >= 0, got {n_state}")
        if n_exogenous < 0:
            raise ValueError(f"n_exogenous must be >= 0, got {n_exogenous}")

        offset = 0
        tick_slice = slice(offset, offset + 1) if include_tick else None
        if include_tick:
            offset += 1
        action_slice = slice(offset, offset + n_actions)
        offset += n_actions
        agent_slice = slice(offset, offset + n_agents)
        offset += n_agents
        parameter_slice = slice(offset, offset + n_parameters)
        offset += n_parameters
        state_slice = slice(offset, offset + n_state) if n_state > 0 else None
        offset += n_state
        exogenous_slice = slice(offset, offset + n_exogenous) if n_exogenous > 0 else None
        return cls(
            tick_slice=tick_slice,
            action_slice=action_slice,
            agent_slice=agent_slice,
            parameter_slice=parameter_slice,
            state_slice=state_slice,
            exogenous_slice=exogenous_slice,
        )

    # ------------------------------------------------------------------
    # Invariance helpers (used by baselines and tests)
    # ------------------------------------------------------------------
    def blocks_except_action(self) -> list[slice]:
        """Return all non-action block slices for invariance checks.

        Used by ``NoActionBaseline`` / ``ShuffledActionBaseline`` to
        verify that only the action block is modified. Order: tick,
        agent, parameter, state, exogenous (skipping ``None``).
        """
        blocks: list[slice] = []
        if self.tick_slice is not None:
            blocks.append(self.tick_slice)
        blocks.append(self.agent_slice)
        blocks.append(self.parameter_slice)
        if self.state_slice is not None:
            blocks.append(self.state_slice)
        if self.exogenous_slice is not None:
            blocks.append(self.exogenous_slice)
        return blocks


# ----------------------------------------------------------------------
# Legacy default for emergency_resource_v0
# ----------------------------------------------------------------------
# Used as fallback when baselines are constructed without an explicit
# layout (e.g. in ad-hoc scripts that pre-date the audit fix). New code
# should pass an explicit ``FeatureLayout`` from the ``DataLoader``.
EMERGENCY_RESOURCE_V0_LAYOUT = FeatureLayout.from_dims(
    n_actions=7,
    n_agents=4,
    n_parameters=3,
    n_state=0,
    n_exogenous=0,
    include_tick=True,
)
