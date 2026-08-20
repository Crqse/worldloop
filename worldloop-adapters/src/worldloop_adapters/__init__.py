"""worldloop-adapters: WorldLoop v2 external environment adapters.

Package version: 0.1.3 (M2 Phase E complete + Phase 5 joint action mode).

This package wraps external RL ecosystem environments (PettingZoo
Parallel, Gymnasium, OpenEnv) as kernel :class:`WorldProtocol`
implementations. The kernel records and verifies transitions; the
adapter translates between the external environment's API and the
kernel's protocol.

Structure:
- :mod:`worldloop_adapters.pettingzoo` — PettingZoo Parallel adapter (A-01)
  + Phase 5 joint action mode (``validate_joint_action`` / ``step_joint``)
  + exact-restore verified allowlist (2 env families)
- :mod:`worldloop_adapters.gymnasium`  — Gymnasium adapter (A-05)
- :mod:`worldloop_adapters.openenv`    — OpenEnv adapter (A-06)

Hard constraint (per main plan §3.3):
- This package MAY import ``worldloop_kernel`` and external env packages
  (``pettingzoo``, ``gymnasium``, ``mpe2``, ``numpy``).
- This package MUST NOT import ``current.worldloop.core.*`` (v1 five-layer).
- This package MUST NOT call LLMs.
"""

__version__ = "0.1.3"

__all__ = ["__version__"]
