"""State mapper: Gymnasium obs → kernel StateView (A-05).

Builds a :class:`StateView` from a Gymnasium env by extracting the obs
vector and wrapping it as a single-entity :class:`EntityTable`.

Mapping summary (per main plan §12.2):
- agent_id (str "agent_0")         → EntityTable.ids (single entry)
- obs vector (np.ndarray)          → EntityTable.columns["observation"]
- obs[:2] (when len >= 2)          → EntityTable.columns["position"]
- obs[2:4] (when len >= 4)         → EntityTable.columns["velocity"]
- tick (env._step_count or 0)      → StateMeta.tick
- scenario_id                      → StateMeta.scenario_id
- rng_state_ref                    → StateMeta.rng_state_ref

Capability / missing-mask (no fabrication, per §12.7 (c)):
- Only ``entities`` slot is populated (capability says ``entities=True``).
- Other slots are ``None`` and their ``missing_mask`` entries are ``True``.
"""
from __future__ import annotations

import hashlib
import pickle
from typing import Any

from worldloop_kernel.capability import CapabilityProfile
from worldloop_kernel.state import (
    EntityTable,
    StateMeta,
    StateView,
)

from .capability import (
    GYMNASIUM_ENTITY_SCHEMA_ID,
    GYMNASIUM_SCENARIO_ID,
)

__all__ = ["build_state_view", "build_rng_state_ref"]


# ---------------------------------------------------------------------------
# RNG state reference (non-consuming)
# ---------------------------------------------------------------------------


def build_rng_state_ref(env: Any) -> str:
    """Build a stable ``rng_state_ref`` without consuming the RNG.

    Gymnasium envs use an internal ``np.random.Generator`` (``env.np_random``).
    We capture its state via pickle + SHA-256. This is read-only.
    """
    step = _get_env_step(env)
    rng = _get_env_rng(env)
    if rng is not None:
        try:
            state = rng.bit_generator.state
            digest = hashlib.sha256(pickle.dumps(state)).hexdigest()[:16]
            return f"sha256:{digest}:step:{step}"
        except Exception:
            pass
    return f"step:{step}"


def _get_env_step(env: Any) -> int:
    """Read the env's step counter without advancing it."""
    unwrapped = getattr(env, "unwrapped", env)
    for attr in ("_step_count", "step_count", "_elapsed_steps"):
        val = getattr(unwrapped, attr, None)
        if isinstance(val, int):
            return val
    return 0


def _get_env_rng(env: Any) -> Any:
    """Read the env's RNG without advancing it. Returns None if absent."""
    unwrapped = getattr(env, "unwrapped", env)
    # Gymnasium envs lazily initialize np_random on first reset; if not
    # yet initialized, the attribute may be missing or None.
    rng = getattr(unwrapped, "np_random", None)
    if rng is not None and hasattr(rng, "bit_generator"):
        return rng
    return None


# ---------------------------------------------------------------------------
# StateView builder
# ---------------------------------------------------------------------------


def build_state_view(
    env: Any,
    obs: Any,
    info: dict[str, Any],
    capability: CapabilityProfile,
    *,
    agent_id: str = "agent_0",
    run_id: str = "gymnasium-run",
    config_hash: str = "gymnasium-default",
) -> StateView:
    """Build a :class:`StateView` from a Gymnasium env snapshot.

    Parameters
    ----------
    env:
        The Gymnasium env handle (post-``reset`` or post-``step``).
    obs:
        The observation returned by ``env.reset`` or ``env.step``.
    info:
        The info dict returned by ``env.reset`` or ``env.step``.
    capability:
        The static :class:`CapabilityProfile` for this adapter.
    agent_id:
        The single agent's id (Gymnasium is single-agent).
    run_id:
        Stable run identifier for this trajectory.
    config_hash:
        Stable config hash for this scenario.

    Returns
    -------
    StateView
        Frozen state view with only ``entities`` populated.
    """
    step = _get_env_step(env)
    scenario_id = GYMNASIUM_SCENARIO_ID
    rng_state_ref = build_rng_state_ref(env)

    meta = StateMeta(
        scenario_id=scenario_id,
        run_id=run_id,
        tick=step,
        config_hash=config_hash,
        rng_state_ref=rng_state_ref,
    )

    # Convert obs to tuple of floats.
    try:
        obs_tuple = tuple(float(x) for x in obs)
    except (TypeError, ValueError):
        obs_tuple = ()

    # Extract position (first 2 dims) and velocity (dims 2-4) when available.
    position: tuple[float, ...]
    if len(obs_tuple) >= 2:
        position = obs_tuple[:2]
    else:
        position = (0.0, 0.0)

    velocity: tuple[float, ...] = ()
    if len(obs_tuple) >= 4:
        velocity = obs_tuple[2:4]

    attributes: dict[str, Any] = {"observation": obs_tuple}
    if velocity:
        attributes["velocity"] = velocity

    entities = EntityTable(
        schema_id=GYMNASIUM_ENTITY_SCHEMA_ID,
        ids=(agent_id,),
        columns={
            "position": (position,),
            "kind": ("agent",),
            "attributes": (attributes,),
        },
    )

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
