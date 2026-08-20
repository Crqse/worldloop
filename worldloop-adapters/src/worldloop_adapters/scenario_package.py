"""ExternalScenarioPackage — external envs as pipeline-ready scenario packages (B4).

Bridges external environment adapters (PettingZoo, Gymnasium, ...) into the
``worldloop_data.pipeline.run_pipeline`` scenario-package protocol WITHOUT
depending on ``worldloop-scenarios``. The pipeline consumes a package via
duck typing; the attributes it actually reads are (verified against
``worldloop_data/pipeline.py`` + ``exporter.py``):

- ``package.world_factory(seed) -> WorldProtocol`` (fresh world per call)
- ``package.world_parameters_hash`` (stable str)
- ``package.spec.scenario.scenario_id`` (producer-id fallback + episode meta)
- ``package.spec.to_dict()`` (exporter ``world_parameters/spec.json`` snapshot)

Honesty discipline (external envs, per B4 spec):
- Capabilities are reported as the adapter actually supports them; missing
  WST/graph/registry/population/event slots stay ``None`` with capability
  ``False`` — never zero-filled.
- ``terminations`` and ``truncations`` are kept distinct (surfaced in the
  per-step receipt diagnostics by the underlying adapter).
- Once an agent disappears from the env (terminated/truncated), it is no
  longer reported alive, so the rollout orchestrator stops proposing
  actions for it (see :class:`LifecycleAwareParallelWorld`).

Hard constraint: this module MUST NOT import ``current/worldloop/core/*``
and MUST NOT import ``worldloop_data`` (dependency direction: data → this
package is optional, never the reverse). External env packages (``mpe2``)
are imported lazily inside the factory so this module stays importable
without PettingZoo installed.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from worldloop_kernel.action import ExecutedAction, ExogenousInput
from worldloop_kernel.canonical import hash_state
from worldloop_kernel.joint import JointAction
from worldloop_kernel.state import EntityTable, StateView
from worldloop_kernel.transition import Checkpoint, TransitionRecord

from .pettingzoo.adapter import (
    PettingZooParallelAdapter,
    make_simple_spread_env,
    make_simple_tag_env,
)
from .pettingzoo.capability import make_pettingzoo_capability

__all__ = [
    "ExternalScenarioRef",
    "ExternalSpecView",
    "ExternalScenarioPackage",
    "SimpleSpreadConfig",
    "SimpleTagConfig",
    "LifecycleAwareParallelWorld",
    "hash_world_parameters",
    "make_simple_spread_package",
    "make_simple_tag_package",
]


# ---------------------------------------------------------------------------
# World-parameters hash
# ---------------------------------------------------------------------------


def hash_world_parameters(parameters: Mapping[str, Any]) -> str:
    """Stable SHA-256 over a JSON-canonicalized parameter mapping.

    Same parameters → same hash; any parameter change (agent count,
    max_cycles, ...) → different hash. Used for reproducibility checks
    and split leakage detection, mirroring
    ``ScenarioSpec.world_parameters_hash`` semantics.
    """
    canonical = json.dumps(dict(parameters), sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Spec shim — satisfies the pipeline/exporter spec protocol
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExternalScenarioRef:
    """Minimal scenario reference (``spec.scenario.*`` shape)."""

    scenario_id: str
    scenario_version: str = "0.1.0"


@dataclass(frozen=True)
class ExternalSpecView:
    """Read-only spec view for external packages.

    Provides the two members the pipeline/exporter actually touch:
    ``.scenario.scenario_id`` and ``.to_dict()``.
    """

    scenario: ExternalScenarioRef
    world_parameters: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": {
                "scenario_id": self.scenario.scenario_id,
                "scenario_version": self.scenario.scenario_version,
            },
            "world_parameters": dict(self.world_parameters),
            "metadata": dict(self.metadata),
            "source": "external",
        }


# ---------------------------------------------------------------------------
# ExternalScenarioPackage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExternalScenarioPackage:
    """Scenario package wrapping an external environment factory.

    Attributes
    ----------
    scenario_id:
        Stable scenario identifier (e.g., ``"external-pettingzoo-simple-spread-n3"``).
    world_factory:
        ``world_factory(seed: int) -> WorldProtocol``. MUST produce a fresh,
        deterministic world per call: two worlds from the same seed must
        yield identical transition records for identical action sequences.
    world_parameters_hash:
        Stable hash of the env construction parameters (see
        :func:`hash_world_parameters`).
    metadata:
        Free-form provenance (env package, versions, parameters, ...).
    """

    scenario_id: str
    world_factory: Callable[[int], Any]
    world_parameters_hash: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError("scenario_id must be non-empty")
        if not callable(self.world_factory):
            raise ValueError("world_factory must be callable")
        if not self.world_parameters_hash:
            raise ValueError("world_parameters_hash must be non-empty")

    @property
    def spec(self) -> ExternalSpecView:
        """Spec-shaped view consumed by pipeline/exporter duck typing."""
        return ExternalSpecView(
            scenario=ExternalScenarioRef(scenario_id=self.scenario_id),
            world_parameters=dict(self.metadata.get("world_parameters", {})),
            metadata={
                k: v for k, v in self.metadata.items() if k != "world_parameters"
            },
        )


# ---------------------------------------------------------------------------
# LifecycleAwareParallelWorld — adapter subclass for pipeline rollouts
# ---------------------------------------------------------------------------


class LifecycleAwareParallelWorld(PettingZooParallelAdapter):
    """PettingZoo adapter variant tuned for ``worldloop_data`` rollouts.

    Adds three pipeline-facing behaviors on top of
    :class:`PettingZooParallelAdapter` (no adapter semantics change):

    1. **Agent lifecycle**: ``observe()`` appends an ``alive`` column to the
       entity table — ``True`` only for agents still active in the env
       (``env.agents``). Landmarks and terminated/truncated agents are
       ``alive=False``, so the rollout's ``_pick_agent`` never proposes
       actions for disappeared agents (B4 honesty rule).
    2. **Replay of recorded actions**: ``step()`` accepts a raw
       :class:`ExecutedAction` reconstructed from a dataset record (no
       prior ``validate_action`` call) by re-deriving the discrete action
       from ``params["discrete_action"]``. Required by the Q3 bit-identical
       replay check.
    3. **Seed provenance**: every transition's provenance carries
       ``seed`` (Q4 provenance completeness).
    """

    def __init__(self, env: Any, **kwargs: Any) -> None:
        super().__init__(env, **kwargs)
        self._seed: int | None = None

    # -- lifecycle ------------------------------------------------------

    def reset(self, seed: int, parameters: Mapping[str, Any] | None = None) -> StateView:
        self._seed = int(seed)
        return super().reset(seed, parameters)

    def _active_agent_ids(self) -> set[str]:
        """Agents still alive per the env (post terminations/truncations)."""
        agents = getattr(self._env, "agents", None)
        if agents is None:
            unwrapped = getattr(self._env, "unwrapped", self._env)
            agents = getattr(unwrapped, "agents", None) or []
        return {str(a) for a in agents}

    def observe(self) -> StateView:
        sv = super().observe()
        entities = sv.entities
        if entities is None:
            return sv
        active = self._active_agent_ids()
        kinds = entities.columns.get("kind", ())
        alive = tuple(
            kind == "agent" and str(eid) in active
            for eid, kind in zip(entities.ids, kinds)
        )
        new_entities = EntityTable(
            schema_id=entities.schema_id,
            ids=entities.ids,
            columns={**dict(entities.columns), "alive": alive},
        )
        return dataclasses.replace(sv, entities=new_entities)

    # -- replay support + seed provenance --------------------------------

    def step(
        self,
        action: ExecutedAction,
        exogenous: ExogenousInput | None = None,
    ) -> TransitionRecord:
        executed_hash = hash_state(action)
        if executed_hash not in self._pending_actions:
            # Replay path (Q3): the action comes straight from a dataset
            # record without a validate_action round. Re-derive the
            # discrete action from params; out-of-range values fall
            # through to the parent's illegal-action handling.
            discrete = action.params.get("discrete_action")
            try:
                discrete_int = int(discrete)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                discrete_int = None
            if discrete_int is not None and discrete_int in self._legal_actions:
                self._pending_actions[executed_hash] = discrete_int
        record = super().step(action, exogenous=exogenous)
        if self._seed is not None and "seed" not in record.provenance:
            provenance = dict(record.provenance)
            provenance["seed"] = str(self._seed)
            record = dataclasses.replace(record, provenance=provenance)
        return record

    def step_joint(
        self,
        joint: JointAction,
        *,
        exogenous: ExogenousInput | None = None,
    ) -> TransitionRecord:
        # Same seed-provenance guarantee as the sequential step (Q4).
        # The joint replay path (cache miss → re-derive discrete actions
        # from executed params) lives in the base adapter already.
        record = super().step_joint(joint, exogenous=exogenous)
        if self._seed is not None and "seed" not in record.provenance:
            provenance = dict(record.provenance)
            provenance["seed"] = str(self._seed)
            record = dataclasses.replace(record, provenance=provenance)
        return record

    # -- restore ----------------------------------------------------------

    def restore(self, checkpoint: Checkpoint) -> None:
        super().restore(checkpoint)
        # Parallel wrappers cache their own ``agents`` list (set on
        # reset/step); after an exact restore of the unwrapped env we
        # re-sync it so the alive column reflects the restored state.
        unwrapped = getattr(self._env, "unwrapped", self._env)
        inner_agents = getattr(unwrapped, "agents", None)
        if unwrapped is not self._env and inner_agents is not None:
            try:
                self._env.agents = list(inner_agents)
            except Exception:  # pragma: no cover — read-only wrapper attr
                pass


# ---------------------------------------------------------------------------
# Simple Spread factory
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimpleSpreadConfig:
    """Construction parameters for the Simple Spread package.

    All parameters flow into ``world_parameters_hash`` — changing any of
    them yields a distinct package hash.
    """

    n_agents: int = 3
    max_cycles: int = 25
    local_ratio: float = 0.5

    def __post_init__(self) -> None:
        if self.n_agents < 1:
            raise ValueError("n_agents must be >= 1")
        if self.max_cycles < 1:
            raise ValueError("max_cycles must be >= 1")

    def to_parameters(self) -> dict[str, Any]:
        return {
            "env": "mpe2/simple_spread_v3",
            "api": "pettingzoo-parallel",
            "n_agents": self.n_agents,
            # Simple Spread ties landmark count to N — reported honestly,
            # not independently configurable.
            "n_landmarks": self.n_agents,
            "max_cycles": self.max_cycles,
            "local_ratio": self.local_ratio,
        }


def make_simple_spread_package(
    config: SimpleSpreadConfig | None = None,
) -> ExternalScenarioPackage:
    """Compile a Simple Spread :class:`ExternalScenarioPackage`.

    The returned package's ``world_factory(seed)`` builds a fresh
    ``mpe2`` Simple Spread Parallel env wrapped in
    :class:`LifecycleAwareParallelWorld`. Deterministic: two worlds from
    the same seed produce identical state hashes under identical action
    sequences (MPE is deterministic given seed + actions).

    ``mpe2`` is imported lazily inside the factory — constructing the
    package itself does not require PettingZoo.
    """
    cfg = config or SimpleSpreadConfig()
    parameters = cfg.to_parameters()
    wp_hash = hash_world_parameters(parameters)
    scenario_id = f"external-pettingzoo-simple-spread-n{cfg.n_agents}"

    def world_factory(seed: int) -> LifecycleAwareParallelWorld:
        env = make_simple_spread_env(
            n_agents=cfg.n_agents,
            n_landmarks=cfg.n_agents,
            max_cycles=cfg.max_cycles,
            local_ratio=cfg.local_ratio,
        )
        return LifecycleAwareParallelWorld(
            env,
            env_id=scenario_id,
            # Capability layered by the exact-restore allowlist (§10.4):
            # simple_spread_v3 is a verified family, so this resolves to
            # the full MPE profile — wired through the allowlist so the
            # claim and the verification stay coupled.
            capability=make_pettingzoo_capability("mpe2/simple_spread_v3"),
            run_id=f"{scenario_id}-seed{int(seed)}",
            config_hash=wp_hash,
        )

    return ExternalScenarioPackage(
        scenario_id=scenario_id,
        world_factory=world_factory,
        world_parameters_hash=wp_hash,
        metadata={
            "source": "pettingzoo.mpe2.simple_spread_v3",
            "world_parameters": parameters,
        },
    )


# ---------------------------------------------------------------------------
# Simple Tag factory (Phase 5 second external env)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimpleTagConfig:
    """Construction parameters for the Simple Tag package.

    All parameters flow into ``world_parameters_hash`` — changing any of
    them yields a distinct package hash.
    """

    num_good: int = 1
    num_adversaries: int = 2
    num_obstacles: int = 1
    max_cycles: int = 25

    def __post_init__(self) -> None:
        if self.num_good < 1:
            raise ValueError("num_good must be >= 1")
        if self.num_adversaries < 1:
            raise ValueError("num_adversaries must be >= 1")
        if self.num_obstacles < 0:
            raise ValueError("num_obstacles must be >= 0")
        if self.max_cycles < 1:
            raise ValueError("max_cycles must be >= 1")

    def to_parameters(self) -> dict[str, Any]:
        return {
            "env": "mpe2/simple_tag_v3",
            "api": "pettingzoo-parallel",
            "num_good": self.num_good,
            "num_adversaries": self.num_adversaries,
            "num_obstacles": self.num_obstacles,
            "max_cycles": self.max_cycles,
            "continuous_actions": False,
        }


def make_simple_tag_package(
    config: SimpleTagConfig | None = None,
) -> ExternalScenarioPackage:
    """Compile a Simple Tag :class:`ExternalScenarioPackage`.

    Predator-prey MPE env: adversaries chase good agents. All agents use
    the discrete 5-action space, so the adapter's action mapper applies
    unchanged. Deterministic given (seed, action sequence) — same
    contract as Simple Spread. ``mpe2`` is imported lazily inside the
    factory.
    """
    cfg = config or SimpleTagConfig()
    parameters = cfg.to_parameters()
    wp_hash = hash_world_parameters(parameters)
    scenario_id = (
        f"external-pettingzoo-simple-tag-g{cfg.num_good}"
        f"a{cfg.num_adversaries}"
    )

    def world_factory(seed: int) -> LifecycleAwareParallelWorld:
        env = make_simple_tag_env(
            num_good=cfg.num_good,
            num_adversaries=cfg.num_adversaries,
            num_obstacles=cfg.num_obstacles,
            max_cycles=cfg.max_cycles,
        )
        return LifecycleAwareParallelWorld(
            env,
            env_id=scenario_id,
            capability=make_pettingzoo_capability("mpe2/simple_tag_v3"),
            run_id=f"{scenario_id}-seed{int(seed)}",
            config_hash=wp_hash,
        )

    return ExternalScenarioPackage(
        scenario_id=scenario_id,
        world_factory=world_factory,
        world_parameters_hash=wp_hash,
        metadata={
            "source": "pettingzoo.mpe2.simple_tag_v3",
            "world_parameters": parameters,
        },
    )
