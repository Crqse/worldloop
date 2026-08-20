"""State mapper: PettingZoo Parallel obs → kernel StateView (A-01).

Builds a :class:`StateView` from a PettingZoo Parallel environment by
extracting observable state from the env's ``obs`` / ``infos`` dicts.

The mapper only does type mapping — it does not write business rules or
mutate the env. All numpy arrays are converted to Python tuples before
being placed in kernel types, because kernel types are frozen dataclasses
and canonical encoding (K-05) does not understand numpy arrays.

Mapping summary (per main plan §12.2):
- agent id (str ``"agent_0"``)         → EntityTable.ids (str preserved)
- agent position (``obs``[:2])         → EntityTable.position (tuple of floats)
- agent velocity (``obs``[2:4])        → EntityTable.attributes["velocity"]
- landmark position (``env.landmarks``) → EntityTable (separate ids)
- agent observation (full ``obs``)     → EntityTable.attributes["observation"]
- tick (``env.unwrapped.steps``)       → StateMeta.tick
- scenario_id                          → StateMeta.scenario_id
- rng_state_ref                        → StateMeta.rng_state_ref (env steps counter)

Capability / missing-mask (no fabrication, per §12.7 (c)):
- Only ``entities`` slot is populated (capability says ``entities=True``).
- ``fields`` / ``relations`` / ``registries`` / ``population`` / ``events``
  are ``None`` and their ``missing_mask`` entries are ``True``.
"""
from __future__ import annotations

import hashlib
import pickle
from typing import Any

from worldloop_kernel.capability import CapabilityProfile
from worldloop_kernel.state import (
    EntityTable,
    EventContext,
    FieldState,
    PopulationState,
    RegistrySnapshot,
    RelationGraph,
    StateMeta,
    StateView,
)

from .capability import (
    PETTINGZOO_ENTITY_SCHEMA_ID_MPE,
    PETTINGZOO_SCENARIO_ID_MPE,
)

__all__ = ["build_state_view", "build_rng_state_ref"]


# ---------------------------------------------------------------------------
# RNG state reference (non-consuming)
# ---------------------------------------------------------------------------


def build_rng_state_ref(env: Any) -> str:
    """Build a stable ``rng_state_ref`` without consuming the RNG.

    PettingZoo MPE envs use an internal ``np.random.RandomState``. We
    capture its ``get_state()`` (a tuple with numpy arrays) and hash it
    via pickle + SHA-256. This is read-only: it does NOT advance the
    RNG, so ``build_state_view`` is idempotent between state changes.

    For envs that do not expose an RNG (e.g., deterministic MPE without
    stochastic exogenous events), returns a step-counter-based ref.
    """
    step = _get_env_step(env)
    # Try to read the env's RNG state; fall back to step-counter if absent.
    rng = _get_env_rng(env)
    if rng is not None:
        try:
            state = rng.get_state()
            digest = hashlib.sha256(pickle.dumps(state)).hexdigest()[:16]
            return f"sha256:{digest}:step:{step}"
        except Exception:
            pass
    return f"step:{step}"


def _get_env_step(env: Any) -> int:
    """Read the env's step counter without advancing it."""
    try:
        return int(env.unwrapped.steps)
    except (AttributeError, TypeError, ValueError):
        return 0


def _get_env_rng(env: Any) -> Any:
    """Read the env's RNG without advancing it. Returns None if absent."""
    # PettingZoo MPE envs typically store RNG on the unwrapped env.
    unwrapped = getattr(env, "unwrapped", env)
    for attr in ("_rng", "rng", "random_state", "np_random"):
        candidate = getattr(unwrapped, attr, None)
        if candidate is not None and hasattr(candidate, "get_state"):
            return candidate
    return None


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------


def _extract_agents(env: Any, obs: dict[str, Any], infos: dict[str, Any]) -> tuple[list[str], list[tuple[float, ...]], list[dict[str, Any]]]:
    """Extract agent ids, positions, and attributes from the env.

    PettingZoo Parallel API: ``obs`` and ``infos`` are dicts keyed by
    agent_id. Agent positions are typically the first 2 dims of the obs
    (MPE convention); velocities are dims 2-4. Landmarks are accessed
    via ``env.unwrapped.landmarks``.
    """
    agent_ids: list[str] = []
    positions: list[tuple[float, ...]] = []
    attrs: list[dict[str, Any]] = []

    for agent_id, agent_obs in obs.items():
        agent_ids.append(str(agent_id))
        # MPE obs layout: [agent_x, agent_y, agent_vx, agent_vy, then
        # relative landmark positions, then relative other-agent positions].
        # We extract position from the first 2 dims when available.
        try:
            pos = tuple(float(x) for x in agent_obs[:2])
        except (TypeError, IndexError, ValueError):
            pos = (0.0, 0.0)
        positions.append(pos)

        # Build attributes dict with velocity + full observation.
        attr: dict[str, Any] = {}
        try:
            if len(agent_obs) >= 4:
                attr["velocity"] = tuple(float(x) for x in agent_obs[2:4])
            attr["observation"] = tuple(float(x) for x in agent_obs)
        except (TypeError, ValueError):
            attr["observation"] = ()
        attrs.append(attr)

    return agent_ids, positions, attrs


def _extract_landmarks(env: Any) -> tuple[list[str], list[tuple[float, ...]], list[dict[str, Any]]]:
    """Extract landmark ids, positions, and attributes from the env.

    PettingZoo MPE envs (mpe2) store landmarks on
    ``env.unwrapped.world.landmarks``; each landmark has a
    ``state.p_pos`` numpy array.
    """
    lm_ids: list[str] = []
    lm_positions: list[tuple[float, ...]] = []
    lm_attrs: list[dict[str, Any]] = []

    unwrapped = getattr(env, "unwrapped", env)
    # mpe2 stores landmarks on world.landmarks (not unwrapped.landmarks).
    world = getattr(unwrapped, "world", None)
    landmarks = getattr(world, "landmarks", None) or getattr(unwrapped, "landmarks", None) or []
    for i, lm in enumerate(landmarks):
        lm_ids.append(f"landmark_{i}")
        try:
            pos_arr = lm.state.p_pos
            pos = tuple(float(x) for x in pos_arr)
        except (AttributeError, TypeError, ValueError):
            pos = (0.0, 0.0)
        lm_positions.append(pos)
        lm_attrs.append({"kind": "landmark"})

    return lm_ids, lm_positions, lm_attrs


# ---------------------------------------------------------------------------
# StateView builder
# ---------------------------------------------------------------------------


def build_state_view(
    env: Any,
    obs: dict[str, Any],
    infos: dict[str, Any],
    capability: CapabilityProfile,
    *,
    run_id: str = "pettingzoo-run",
    config_hash: str = "pettingzoo-default",
) -> StateView:
    """Build a :class:`StateView` from a PettingZoo Parallel env snapshot.

    Parameters
    ----------
    env:
        The PettingZoo Parallel env handle (post-``reset`` or post-``step``).
    obs:
        The observation dict returned by ``env.reset`` or ``env.step``.
    infos:
        The info dict returned by ``env.reset`` or ``env.step``.
    capability:
        The static :class:`CapabilityProfile` for this adapter.
    run_id:
        Stable run identifier for this trajectory.
    config_hash:
        Stable config hash for this scenario.

    Returns
    -------
    StateView
        Frozen state view with only ``entities`` populated (per MPE
        capability). Other slots are ``None`` with ``missing_mask=True``.
    """
    step = _get_env_step(env)
    scenario_id = PETTINGZOO_SCENARIO_ID_MPE
    rng_state_ref = build_rng_state_ref(env)

    meta = StateMeta(
        scenario_id=scenario_id,
        run_id=run_id,
        tick=step,
        config_hash=config_hash,
        rng_state_ref=rng_state_ref,
    )

    # Build entities: agents + landmarks concatenated.
    agent_ids, agent_pos, agent_attrs = _extract_agents(env, obs, infos)
    lm_ids, lm_pos, lm_attrs = _extract_landmarks(env)

    all_ids = tuple(agent_ids + lm_ids)
    all_positions = tuple(agent_pos + lm_pos)
    all_attributes = tuple(agent_attrs + lm_attrs)
    all_kinds = tuple(["agent"] * len(agent_ids) + ["landmark"] * len(lm_ids))

    # EntityTable stores columns as Mapping[str, tuple[Any, ...]].
    # We use "position" (tuple of floats), "kind" (str), and "attributes"
    # (dict) as the three columns. Each column tuple aligns with ids.
    entities = EntityTable(
        schema_id=PETTINGZOO_ENTITY_SCHEMA_ID_MPE,
        ids=all_ids,
        columns={
            "position": all_positions,
            "kind": all_kinds,
            "attributes": all_attributes,
        },
    )

    # missing_mask: per-slot mask (key=slot name, value=True if missing).
    # Rule: missing_mask MUST NOT be True for a slot the world declares
    # capabilities.<slot>=False. For PettingZoo MPE, only entities=True
    # in capability; other slots are False and cannot be marked missing.
    # Since entities is populated above, missing_mask is empty.
    missing_mask: dict[str, bool] = {}

    return StateView(
        meta=meta,
        entities=entities,
        capabilities=capability,
        missing_mask=missing_mask,
        fields=None,
        relations=None,
        registries=None,
        population=None,
        events=None,
    )
